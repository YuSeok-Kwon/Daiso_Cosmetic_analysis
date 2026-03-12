"""
다이소 뷰티/위생 카테고리 통합 크롤러
- 제품 정보 (product_all.csv)
- 리뷰 (reviews_all.csv)
- 성분 (ingredients_all.csv)
- BigQuery 적재 지원
"""
import os
import sys
import time
import re

from dotenv import load_dotenv
# config 폴더에서 .env 로드
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', 'config', '.env'))
import pandas as pd
from collections import defaultdict
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from config import DAISO_BEAUTY_CATEGORIES
from modules.ocr_utils_split import extract_text_from_image_url_split, extract_text_bottom_up_3920
from modules.ingredient_parser import (
    normalize_ingredient_name,
    is_valid_ingredient,
    extract_from_text,
    extract_product_section,
    INGREDIENT_KEYWORDS
)
from utils import setup_logger, get_date_string, extract_rating

# BigQuery 모듈 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from bigquery.etl_loader import CrawlerETL
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False

# 로거 설정
logger = setup_logger('daiso_beauty_crawler', 'daiso_beauty_crawler.log')

# 기본 설정
BASE_URL = "https://www.daisomall.co.kr"
MAX_SCROLLS = 10
user_id_map = defaultdict(lambda: f"user_{len(user_id_map)+1:04d}")


def parse_count(text: str) -> int:
    """좋아요/공유 수 텍스트를 정수로 변환 (예: '1,234' → 1234, '1.2K' → 1200)"""
    if not text:
        return 0
    text = text.strip().replace(",", "")
    try:
        if text.upper().endswith("K"):
            return int(float(text[:-1]) * 1000)
        elif text.upper().endswith("M"):
            return int(float(text[:-1]) * 1000000)
        return int(text)
    except (ValueError, IndexError):
        return 0


def extract_ingredients_multi_source(driver, product_code: str, product_name: str) -> list:
    """
    다중 소스에서 성분 추출 및 교차 검증

    Returns:
        list of dicts with: product_id, name, ingredient
    """
    all_ingredients = {}  # {성분명: {confidence, sources[], reason}}

    # 소스 1: Picture alt 속성
    # 성분 헤더 키워드 (이것이 있으면 ALT를 신뢰)
    HEADER_KEYWORDS = ['전성분', '성분:', '모든성분', '화장품법', 'INGREDIENTS', 'Ingredients', '성분 :', '[성분명]', '성분명', '[전성분]']
    alt_has_header = False  # ALT에 "전성분" 같은 헤더가 있는지

    # 대표 성분명 (헤더 없이도 성분 섹션 감지용 - OCR과 동일)
    COMMON_INGREDIENTS_ALT = [
        '정제수', '글리세린', '부틸렌글라이콜', '프로판다이올', '나이아신아마이드',
        '히알루론산', '판테놀', '알란토인', '토코페롤', '카보머', '페녹시에탄올',
        '향료', '시트릭애씨드', '다이메티콘', '스쿠알란', '세틸알코올',
        '글리세레스', '헥산다이올', '소듐하이알루로네이트', '알지닌', '세린',
    ]

    try:
        # picture 태그 안의 img와 직접 img 태그 모두 찾기
        pictures = driver.find_elements(By.CSS_SELECTOR, "div.editor-content picture img, div.editor-content > img")

        for idx, img in enumerate(pictures):
            alt_text = img.get_attribute("alt") or ""

            # 긴 ALT 텍스트 처리 (100자 이상)
            if alt_text and len(alt_text) > 100:
                # 줄바꿈 제거 후 키워드 검색
                alt_text_flat = alt_text.replace('\n', ' ').replace('  ', ' ')
                logger.debug(f"ALT_{idx} 길이: {len(alt_text)}, 첫 100자: {alt_text_flat[:100]}...")

                # 헤더 키워드 확인
                has_header = any(kw in alt_text_flat for kw in HEADER_KEYWORDS)
                if has_header:
                    alt_has_header = True
                    logger.debug(f"ALT_{idx}에서 헤더 키워드 발견")

                # 대표 성분명 개수 확인 (헤더 없어도 성분 섹션 감지)
                common_ing_count = sum(1 for ing in COMMON_INGREDIENTS_ALT if ing in alt_text_flat)

                # 헤더가 있거나, 대표 성분명이 3개 이상이면 성분 섹션으로 판단
                if has_header or any(kw in alt_text_flat for kw in INGREDIENT_KEYWORDS) or common_ing_count >= 3:
                    if common_ing_count >= 3 and not has_header:
                        logger.info(f"ALT_{idx}에서 대표 성분명 {common_ing_count}개 감지 (헤더 없음) → 성분 추출 시도")

                    # 멀티 제품/옵션 텍스트에서 해당 제품 섹션만 추출
                    alt_text_section = extract_product_section(alt_text_flat, product_name)
                    alt_ingredients = extract_from_text(alt_text_section, source=f"ALT_{idx}")

                    for ing in alt_ingredients:
                        name = normalize_ingredient_name(ing['ingredient'])
                        is_valid, conf, reason = is_valid_ingredient(name)

                        if is_valid and conf >= 0.5:
                            if name not in all_ingredients:
                                all_ingredients[name] = {'confidence': conf, 'sources': [ing['source']], 'reason': reason}
                            else:
                                all_ingredients[name]['sources'].append(ing['source'])
                                all_ingredients[name]['confidence'] = min(1.0, all_ingredients[name]['confidence'] + 0.1)
                continue

            # 짧은 ALT 텍스트 처리 (100자 이하)
            # 헤더 키워드 확인 (전성분, 성분: 등)
            if any(kw in alt_text for kw in HEADER_KEYWORDS):
                alt_has_header = True

            if any(kw in alt_text for kw in INGREDIENT_KEYWORDS):
                # 멀티 제품/옵션 텍스트에서 해당 제품 섹션만 추출
                alt_text_section = extract_product_section(alt_text, product_name)
                alt_ingredients = extract_from_text(alt_text_section, source=f"ALT_{idx}")

                for ing in alt_ingredients:
                    name = normalize_ingredient_name(ing['ingredient'])
                    is_valid, conf, reason = is_valid_ingredient(name)

                    if is_valid and conf >= 0.5:
                        if name not in all_ingredients:
                            all_ingredients[name] = {'confidence': conf, 'sources': [ing['source']], 'reason': reason}
                        else:
                            all_ingredients[name]['sources'].append(ing['source'])
                            all_ingredients[name]['confidence'] = min(1.0, all_ingredients[name]['confidence'] + 0.1)

        alt_count = len([k for k in all_ingredients if any('ALT' in s for s in all_ingredients[k]['sources'])])
        logger.info(f"ALT에서 성분 발견: 총 {alt_count}개 (헤더 키워드: {'있음' if alt_has_header else '없음'})")

    except Exception as e:
        logger.debug(f"ALT 텍스트 추출 실패: {str(e)}")
        alt_count = 0

    # 소스 2: OCR
    # - 항상 OCR 실행 (ALT 여부와 무관하게 전체 제품 OCR 진행)
    # - ALT에서 성분 발견된 경우: 검증 모드
    # - ALT에서 성분 미발견: 보완 모드
    run_ocr = True  # 항상 OCR 실행
    if alt_count >= 6:
        ocr_mode = "검증"
        logger.info(f"ALT에서 성분 {alt_count}개 발견 → OCR 교차 검증")
    else:
        ocr_mode = "보완"
        logger.info(f"ALT 성분 {alt_count}개 → OCR 보완 모드")

    if run_ocr:
        try:
            # ALT에서 추출된 성분 목록 (검증용)
            alt_ingredients_set = set(all_ingredients.keys()) if ocr_mode == "검증" else set()

            # picture 태그 안의 img와 직접 img 태그 모두 찾기
            pictures = driver.find_elements(By.CSS_SELECTOR, "div.editor-content picture img, div.editor-content > img")
            ocr_only_count = 0
            verified_count = 0

            # 이미지 유형 판별: 긴 이미지 1장 vs 작은 이미지 여러장
            # picture 태그가 2개 이상이면 멀티 이미지로 판단
            is_multi_image = len(pictures) >= 2

            # 대표 성분명 키워드 (헤더 없이 성분명만 있는 경우 감지용)
            COMMON_INGREDIENTS = [
                '정제수', '글리세린', '부틸렌글라이콜', '프로판다이올', '나이아신아마이드',
                '히알루론산', '판테놀', '알란토인', '토코페롤', '카보머', '페녹시에탄올',
                '향료', '시트릭애씨드', '다이메티콘', '스쿠알란', '세틸알코올',
            ]

            found_ingredients_in_ocr = False

            if is_multi_image:
                # 멀티 이미지 케이스: 마지막 2개 이미지에서 전성분 찾기 (기존 로직 유지)
                logger.info(f"멀티 이미지 감지: {len(pictures)}개 이미지 → 마지막 2개에서 전성분 탐색")
                target_images = list(pictures[-2:])
                target_images.reverse()  # 마지막 이미지 먼저
                num_sections = 1  # 작은 이미지는 분할 불필요

                for idx, img in enumerate(target_images):
                    src = img.get_attribute("src")

                    if src:
                        logger.info(f"OCR 분석 중 ({ocr_mode} 모드): 이미지 {idx + 1}/{len(target_images)}")
                        sections = extract_text_from_image_url_split(src, num_sections=num_sections)

                        # 전체 텍스트에서 성분 키워드 존재 여부 확인
                        all_section_text = ' '.join([s.get('text', '') for s in sections or []])

                        # 헤더 키워드 또는 대표 성분명 3개 이상 포함 시 성분 섹션으로 판단
                        has_header = any(kw in all_section_text for kw in INGREDIENT_KEYWORDS)
                        common_ing_count = sum(1 for ing in COMMON_INGREDIENTS if ing in all_section_text)
                        has_ingredient_section = has_header or common_ing_count >= 3

                        if common_ing_count >= 3 and not has_header:
                            logger.info(f"성분명 {common_ing_count}개 감지 (헤더 없음) → 성분 섹션으로 판단")

                        if has_ingredient_section:
                            logger.info(f"이미지 {idx + 1}에서 전성분 키워드 발견!")
                            found_ingredients_in_ocr = True

                            for section_idx, section in enumerate(sections or []):
                                text = section.get('text', '')
                                text_section = extract_product_section(text, product_name)
                                ocr_ingredients = extract_from_text(text_section, source=f"OCR_{idx}_{section_idx}", force_mode=True)

                                for ing in ocr_ingredients:
                                    name = normalize_ingredient_name(ing['ingredient'])
                                    is_valid, conf, reason = is_valid_ingredient(name)
                                    conf *= 0.9

                                    if is_valid and conf >= 0.5:
                                        if name not in all_ingredients:
                                            all_ingredients[name] = {'confidence': conf, 'sources': [ing['source']], 'reason': reason}
                                            ocr_only_count += 1
                                        else:
                                            all_ingredients[name]['sources'].append(ing['source'])
                                            all_ingredients[name]['confidence'] = min(1.0, all_ingredients[name]['confidence'] + 0.15)
                                            verified_count += 1

                            break  # 성분 발견하면 더 이상 이미지 탐색 안 함

            else:
                # 싱글 이미지 케이스 (긴 이미지): Bottom-Up 3920px OCR
                logger.info(f"싱글/대형 이미지 감지: {len(pictures)}개 이미지 → Bottom-Up 3920px OCR 적용")

                # 마지막 이미지 사용 (주로 긴 이미지가 하나)
                last_img = pictures[-1] if pictures else None
                if last_img:
                    src = last_img.get_attribute("src")
                    if src:
                        logger.info(f"OCR 분석 중 ({ocr_mode} 모드): Bottom-Up 3920px")
                        sections = extract_text_bottom_up_3920(src)

                        # 전체 텍스트에서 성분 키워드 존재 여부 확인
                        all_section_text = ' '.join([s.get('text', '') for s in sections or []])

                        has_header = any(kw in all_section_text for kw in INGREDIENT_KEYWORDS)
                        common_ing_count = sum(1 for ing in COMMON_INGREDIENTS if ing in all_section_text)
                        has_ingredient_section = has_header or common_ing_count >= 3

                        if has_ingredient_section:
                            logger.info(f"Bottom-Up OCR에서 전성분 키워드 발견!")
                            found_ingredients_in_ocr = True

                            for section_idx, section in enumerate(sections or []):
                                text = section.get('text', '')
                                text_section = extract_product_section(text, product_name)
                                ocr_ingredients = extract_from_text(text_section, source=f"OCR_BU_{section_idx}", force_mode=True)

                                for ing in ocr_ingredients:
                                    name = normalize_ingredient_name(ing['ingredient'])
                                    is_valid, conf, reason = is_valid_ingredient(name)
                                    conf *= 0.9

                                    if is_valid and conf >= 0.5:
                                        if name not in all_ingredients:
                                            all_ingredients[name] = {'confidence': conf, 'sources': [ing['source']], 'reason': reason}
                                            ocr_only_count += 1
                                        else:
                                            all_ingredients[name]['sources'].append(ing['source'])
                                            all_ingredients[name]['confidence'] = min(1.0, all_ingredients[name]['confidence'] + 0.15)
                                            verified_count += 1

            if ocr_mode == "검증":
                logger.info(f"OCR 교차 검증: {verified_count}개 성분 확인, OCR에서만 발견: {ocr_only_count}개")
            else:
                logger.info(f"OCR에서 추가 성분: 총 {len([k for k in all_ingredients if any('OCR' in s for s in all_ingredients[k]['sources'])])}개")

        except Exception as e:
            logger.error(f"OCR 실패: {str(e)}")

    # 최종 필터링: 신뢰도 기준 정렬
    final_ingredients = []

    for name, info in all_ingredients.items():
        # 여러 소스에서 발견된 성분 우선
        multi_source_bonus = len(info['sources']) * 0.05
        final_conf = min(1.0, info['confidence'] + multi_source_bonus)

        # 신뢰도 50% 이상만 포함
        if final_conf >= 0.5:
            final_ingredients.append({
                'product_id': product_code,
                'name': product_name,
                'ingredient': name,
            })

    # 성분명 기준으로 정렬
    final_ingredients.sort(key=lambda x: x['ingredient'])

    logger.info(f"최종 성분: {len(final_ingredients)}개")

    return final_ingredients


def get_category_url(middle_code, small_code):
    """카테고리 URL 생성"""
    return f"{BASE_URL}/ds/exhCtgr/C208/CTGR_00014/{middle_code}/{small_code}"


def get_all_product_links(driver, category_url, category_name):
    """제품 링크 수집"""
    logger.info(f"[{category_name}] 제품 링크 수집 시작: {category_url}")
    driver.get(category_url)

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "product-list"))
    )

    # 페이지 스크롤
    for _ in range(MAX_SCROLLS):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

    # 제품 링크 수집
    items = driver.find_elements(By.CLASS_NAME, "prod-thumb__link")
    links = []

    for item in items:
        href = item.get_attribute("href")
        if href and "pdNo=" in href:
            links.append(href)
        else:
            html = item.get_attribute("outerHTML")
            match = re.search(r"pdNo=(\d+)", html)
            if match:
                pdno = match.group(1)
                full_url = f"{BASE_URL}/pd/pdr/SCR_PDR_0001?pdNo={pdno}&recmYn=N"
                links.append(full_url)

    dedup = list(dict.fromkeys(links))
    logger.info(f"[{category_name}] 총 {len(dedup)}개 제품 링크 수집 완료")
    return dedup


def extract_brand(driver, category_2=""):
    """브랜드 추출"""
    # 1. 브랜드 영역에서 추출 시도 (최우선)
    try:
        title_elem = driver.find_element(By.CSS_SELECTOR, "a.brand-area div.brand-area__detail div.detail-title")
        brand_text = title_elem.text.strip()
        if brand_text:
            return brand_text
    except:
        pass

    # 2. 브랜드 영역이 없는 경우
    # 2-1. 화장품이 아닌 카테고리는 "다이소"로 설정
    non_cosmetic_categories = ["메이크업 브러쉬", "메이크업 퍼프", "메이크업 소품", "퍼프브러시세척"]
    if category_2 in non_cosmetic_categories:
        return "다이소"

    # 2-2. 일반 제품은 제품명에서 첫 단어 추출
    try:
        product_title_elem = driver.find_element(By.CSS_SELECTOR, "h1.product-title")
        product_title = product_title_elem.text.strip()
        if product_title:
            # 첫 번째 띄어쓰기 전까지
            first_word = product_title.split()[0] if product_title.split() else ""
            if first_word:
                return first_word
    except:
        pass

    return ""


def _wait_for_page_load(driver, url_pdno: str) -> bool:
    """페이지 로딩 대기"""
    # 1단계: 기본 DOM 로딩 대기
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, "product-info-wrap"))
        )
        logger.debug("기본 DOM 로딩 완료")
    except Exception as e:
        logger.warning(f"product-info-wrap 로딩 타임아웃: {str(e)}")

    # 2단계: JavaScript 실행 완료 대기
    time.sleep(5)

    # 3단계: 스크롤로 Lazy Loading 트리거
    driver.execute_script("window.scrollTo(0, 300);")
    time.sleep(1)
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(1)

    # 4단계: 제품명이 실제로 로드될 때까지 대기
    max_retries = 5
    for retry in range(max_retries):
        try:
            name_element = WebDriverWait(driver, 3).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".info-area .product-title"))
            )
            name_text = name_element.text.strip()
            if name_text and len(name_text) > 0:
                logger.debug(f"제품명 로딩 완료 (시도 {retry + 1}/{max_retries})")
                return True
            else:
                logger.debug(f"제품명 로딩 대기 중 (시도 {retry + 1}/{max_retries})")
                time.sleep(2)
        except Exception as e:
            logger.debug(f"제품명 대기 중 예외 (시도 {retry + 1}/{max_retries}): {str(e)}")
            time.sleep(2)

    logger.warning(f"제품명 로딩 타임아웃 - pdNo: {url_pdno}")
    return False


def _extract_basic_info(driver, product: dict, url_pdno: str, category_2: str) -> str:
    """
    기본 제품 정보 추출 (브랜드, 제품명, 가격, 제조국 등)

    Returns:
        url_pdno: 업데이트된 제품 코드
    """
    # 리다이렉트 체크
    current_url = driver.current_url
    current_pdno_match = re.search(r"pdNo=([A-Z0-9]+)", current_url)

    if current_pdno_match:
        current_pdno = current_pdno_match.group(1)
        if current_pdno != url_pdno:
            logger.warning(f"URL 리다이렉트 감지! 요청: {url_pdno} → 실제: {current_pdno}")
            product["product_code"] = current_pdno
            url_pdno = current_pdno
    else:
        logger.warning(f"현재 URL에서 pdNo 추출 불가: {current_url}")

    # 브랜드
    product["brand"] = extract_brand(driver, category_2)

    # 제품명 추출
    name_selectors = ["h1.product-title", ".info-area h1", ".product-info-wrap h1"]
    for selector in name_selectors:
        try:
            element = driver.find_element(By.CSS_SELECTOR, selector)
            product["name"] = element.text.strip()
            if product["name"]:
                logger.info(f"제품명 추출 성공: {product['name'][:50]}")
                break
        except:
            continue

    # 옵션 정보 추가
    if product["name"]:
        try:
            option_element = driver.find_element(By.CSS_SELECTOR, ".product-option-text, .option-text, .selected-option")
            option_text = option_element.text.strip()
            if option_text and option_text not in product["name"]:
                product["name"] = f"{product['name']} ({option_text})"
        except:
            pass

    # 가격 추출
    try:
        price_element = driver.find_element(By.CSS_SELECTOR, ".prod-price--detail .price-value .value")
        product["price"] = price_element.text.strip().replace(",", "")
    except:
        try:
            price_element = driver.find_element(By.CSS_SELECTOR, ".inner-box .price-value .value")
            product["price"] = price_element.text.strip().replace(",", "")
        except:
            pass

    # 페이지 품번 확인 및 검증
    try:
        code_text = driver.find_element(By.CLASS_NAME, "code-text").text
        match = re.search(r"품번\s*(\d+)", code_text)
        if match:
            page_product_code = match.group(1)
            if page_product_code != url_pdno:
                logger.warning(f"품번 불일치! URL: {url_pdno}, 페이지: {page_product_code}")
                product["product_code"] = page_product_code
                url_pdno = page_product_code
    except:
        pass

    # 제조국
    try:
        product["country"] = driver.find_element(
            By.XPATH, "//th[contains(text(),'제조국')]/following-sibling::td"
        ).text.strip()
    except:
        pass

    # 좋아요/공유
    try:
        counts = driver.find_elements(By.CLASS_NAME, "btn__count")
        if len(counts) >= 2:
            product["likes"] = parse_count(counts[0].text)
            product["shares"] = parse_count(counts[1].text)
    except:
        pass

    return url_pdno


def _extract_reviews(driver, product_code: str, cutoff_date: str = None) -> list:
    """리뷰 크롤링

    Parameters
    ----------
    cutoff_date : 이 날짜 이하의 리뷰를 만나면 조기 종료 (YYYY-MM-DD 형식).
                  리뷰는 날짜 역순(최신→과거)으로 나열되므로
                  cutoff 이하를 만나면 이후 리뷰도 모두 이전 것이다.
    """
    reviews = []

    # 리뷰 영역 활성화를 위해 스크롤 후 최신순 정렬 클릭
    try:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.5);")
        time.sleep(2)

        # 클릭 전 첫 번째 리뷰 날짜 캡처 (리로드 감지용)
        old_first_date = None
        try:
            first_review = driver.find_elements(By.CLASS_NAME, "review-detail")
            logger.info(f"[정렬 전] 리뷰 요소 수: {len(first_review)}")
            if first_review:
                old_first_date = first_review[0].find_element(
                    By.CLASS_NAME, "cw-bar-list"
                ).text.split()[0]
                logger.info(f"[정렬 전] 첫 리뷰 날짜: {old_first_date}")
        except:
            pass

        sort_buttons = driver.find_elements(
            By.CSS_SELECTOR, "ul.filter-selection-group li button.el-button"
        )
        for btn in sort_buttons:
            if "최신순" in btn.text:
                # 이미 활성 상태면 클릭 불필요
                if "is-active" in (btn.get_attribute("class") or ""):
                    logger.info("리뷰 정렬: 최신순 이미 활성")
                    break

                driver.execute_script("arguments[0].click();", btn)

                # 1) 버튼 활성화 대기
                for _ in range(5):
                    time.sleep(1)
                    if "is-active" in (btn.get_attribute("class") or ""):
                        break

                # 2) 리뷰 목록 리로드 대기 (첫 리뷰 날짜 변경 감지)
                reloaded = False
                if old_first_date:
                    for wait_i in range(10):
                        time.sleep(1)
                        try:
                            new_reviews = driver.find_elements(By.CLASS_NAME, "review-detail")
                            if new_reviews:
                                new_first_date = new_reviews[0].find_element(
                                    By.CLASS_NAME, "cw-bar-list"
                                ).text.split()[0]
                                if new_first_date != old_first_date:
                                    logger.info(f"[정렬 후] 리로드 확인 ({wait_i+1}s): {old_first_date} → {new_first_date}")
                                    reloaded = True
                                    break
                        except:
                            pass
                    if not reloaded:
                        logger.warning(f"[정렬 후] 리로드 미감지 (10s 대기 완료), 리뷰 수: {len(driver.find_elements(By.CLASS_NAME, 'review-detail'))}")
                else:
                    logger.info("[정렬 전] 리뷰 없음 → 3초 대기")
                    time.sleep(3)

                logger.info("리뷰 정렬: 최신순 적용")
                break
    except Exception as e:
        logger.debug(f"최신순 정렬 클릭 실패 (무시): {e}")

    for page in range(1, 999):
        time.sleep(1)
        review_elements = driver.find_elements(By.CLASS_NAME, "review-detail")
        if page == 1:
            logger.info(f"[페이지1] 리뷰 요소 수: {len(review_elements)}")
            if review_elements:
                try:
                    first_date = review_elements[0].find_element(By.CLASS_NAME, "cw-bar-list").text.split()[0]
                    logger.info(f"[페이지1] 첫 리뷰 날짜: {first_date}")
                except:
                    logger.info("[페이지1] 첫 리뷰 날짜 추출 실패")

        hit_cutoff = False
        for ri, r in enumerate(review_elements):
            try:
                date_raw = r.find_element(By.CLASS_NAME, "cw-bar-list").text
                date = date_raw.split()[0]

                # 첫 페이지 처음 3개 리뷰 상세 로그
                if page == 1 and ri < 3:
                    date_conv = date.replace(".", "-")
                    logger.info(f"  [리뷰{ri}] raw='{date_raw}' → date='{date}' → conv='{date_conv}' cutoff='{cutoff_date}' hit={date_conv <= cutoff_date if cutoff_date else 'N/A'}")

                # 증분 크롤링: cutoff 이하 날짜 도달 시 조기 종료
                if cutoff_date and date.replace(".", "-") <= cutoff_date:
                    hit_cutoff = True
                    break

                user_raw = r.find_element(By.CLASS_NAME, "con-writer-id").text.strip()
                rating_raw = r.find_element(By.CLASS_NAME, "hiddenText").text.strip()
                text = r.find_element(By.CSS_SELECTOR, ".review-desc .cont").text.strip()
                image_count = len(r.find_elements(By.CSS_SELECTOR, ".swiper-wrapper img"))
                rating = extract_rating(rating_raw)
                user_id = user_id_map[user_raw]

                reviews.append({
                    "product_code": product_code,
                    "date": date,
                    "user_masked": user_raw,
                    "user": user_id,
                    "rating": rating,
                    "text": text,
                    "image_count": image_count,
                })
            except Exception as ex:
                if page == 1 and ri < 3:
                    logger.warning(f"  [리뷰{ri}] 추출 실패: {type(ex).__name__}: {ex}")
                continue

        if hit_cutoff:
            logger.info(f"cutoff 도달 ({cutoff_date}): {len(reviews)}개 신규 리뷰 수집 후 종료")
            break

        # 다음 페이지
        try:
            next_btn = driver.find_element(By.CLASS_NAME, "btn-next")
            next_class = next_btn.get_attribute("class") or ""
            next_style = next_btn.get_attribute("style") or ""

            if ("disabled" in next_class) or (not next_btn.is_enabled()) or ("pointer-events: none" in next_style):
                break

            driver.execute_script("arguments[0].click();", next_btn)
            time.sleep(1)
        except:
            break

    return reviews


def _extract_ingredients_and_certifications(driver, product: dict) -> list:
    """성분 정보 추출"""
    ingredients = []

    try:
        ingredients = extract_ingredients_multi_source(
            driver, product['product_code'], product['name']
        )
        logger.info(f"성분 추출 완료: {len(ingredients)}개")

    except Exception as e:
        logger.error(f"성분 추출 오류: {str(e)}")
        ingredients = []

    return ingredients


def crawl_product_detail(driver, url, category_home, category_1, category_2, crawl_reviews=True, crawl_ingredients=True, review_cutoff_date=None):
    """제품 상세 정보 크롤링 (리팩토링됨)"""
    # URL에서 pdNo 추출
    url_pdno_match = re.search(r"pdNo=([A-Z0-9]+)", url)
    if not url_pdno_match:
        logger.error(f"URL에서 pdNo 추출 실패: {url}")
        return None, [], []

    url_pdno = url_pdno_match.group(1)
    logger.info(f"제품 크롤링 시작 - pdNo: {url_pdno}")

    # 제품 정보 초기화
    product = {
        "product_code": url_pdno,
        "category_home": category_home,
        "category_1": category_1,
        "category_2": category_2,
        "brand": "",
        "name": "",
        "price": "",
        "country": "",
        "likes": 0,
        "shares": 0,
        "url": url,
    }

    # 1. 페이지 로드
    driver.get(url)
    _wait_for_page_load(driver, url_pdno)

    # 2. 기본 정보 추출
    url_pdno = _extract_basic_info(driver, product, url_pdno, category_2)

    # 3. 유효성 검증
    if not product["product_code"]:
        logger.error(f"제품 코드 없음 - 스킵")
        return None, [], []

    if not product["name"]:
        logger.error(f"제품명 추출 실패 - pdNo: {url_pdno}")

    # 4. 가격 검증 (5천원 초과 제외)
    try:
        if product["price"] and int(product["price"]) > 5000:
            logger.info(f"제외 (가격 초과): {product['name']} | {product['price']}원")
            return None, [], []
    except:
        pass

    logger.info(f"제품 정보: {product['product_code']} | {product['name'][:40]} | {product['price']}원")

    # 5. 리뷰 크롤링
    reviews = []
    if crawl_reviews:
        logger.info(f"리뷰 수집 시작" + (f" (cutoff: {review_cutoff_date})" if review_cutoff_date else ""))
        reviews = _extract_reviews(driver, product["product_code"], cutoff_date=review_cutoff_date)
        logger.info(f"리뷰 수집 완료: {len(reviews)}개")

    # 6. 성분 크롤링
    ingredients = []
    if crawl_ingredients:
        logger.info(f"성분 수집 시작")
        ingredients = _extract_ingredients_and_certifications(driver, product)

    return product, reviews, ingredients


def select_categories():
    """중분류/소분류 선택"""
    print("\n" + "="*60)
    print("다이소 뷰티/위생 카테고리 크롤러")
    print("="*60)

    # 중분류 선택
    print("\n[중분류 선택]")
    middle_categories = list(DAISO_BEAUTY_CATEGORIES.keys())
    for idx, cat in enumerate(middle_categories, 1):
        print(f"{idx}. {cat}")
    print("0. 전체")

    choice = input("\n선택 (번호 입력): ").strip()

    if choice == "0":
        selected_middle = middle_categories
    else:
        try:
            selected_middle = [middle_categories[int(choice) - 1]]
        except:
            print("잘못된 선택입니다.")
            return None

    # 소분류 선택
    selected_categories = []
    for middle in selected_middle:
        middle_code = DAISO_BEAUTY_CATEGORIES[middle]["중분류코드"]
        small_categories = DAISO_BEAUTY_CATEGORIES[middle]["소분류"]

        print(f"\n[{middle} - 소분류 선택]")
        small_list = list(small_categories.items())
        for idx, (code, name) in enumerate(small_list, 1):
            print(f"{idx}. {name}")
        print("0. 전체")

        choice = input("\n선택 (번호 입력, 여러 개는 쉼표로 구분): ").strip()

        if choice == "0":
            for code, name in small_list:
                selected_categories.append((middle, middle_code, code, name))
        else:
            try:
                indices = [int(x.strip()) for x in choice.split(",")]
                for idx in indices:
                    code, name = small_list[idx - 1]
                    selected_categories.append((middle, middle_code, code, name))
            except:
                print("잘못된 선택입니다.")
                continue

    return selected_categories


def select_crawl_targets():
    """크롤링 대상 선택"""
    print("\n[크롤링 대상 선택]")
    print("1. 제품 정보만")
    print("2. 제품 정보 + 리뷰")
    print("3. 제품 코드 + 성분만")
    print("4. 제품 코드 + 리뷰만")
    print("5. 전체 (제품 정보 + 리뷰 + 성분)")

    choice = input("\n선택 (번호 입력): ").strip()

    # (제품정보저장, 리뷰수집, 성분수집, 최소제품정보모드)
    targets = {
        "1": (True, False, False, False),   # 제품 정보만
        "2": (True, True, False, False),    # 제품 정보 + 리뷰
        "3": (False, False, True, True),    # 제품 코드 + 성분만
        "4": (False, True, False, True),    # 제품 코드 + 리뷰만 (NEW!)
        "5": (True, True, True, False),     # 전체
    }

    return targets.get(choice, (True, False, False, False))


def main():
    """메인 함수"""
    # 카테고리 선택
    categories = select_categories()
    if not categories:
        return

    # 크롤링 대상 선택
    crawl_products, crawl_reviews, crawl_ingredients, minimal_mode = select_crawl_targets()

    print(f"\n{'='*60}")
    print(f"선택된 카테고리: {len(categories)}개")
    if minimal_mode:
        if crawl_reviews and not crawl_ingredients:
            print(f"크롤링 대상: 제품 코드 + 리뷰")
        elif crawl_ingredients and not crawl_reviews:
            print(f"크롤링 대상: 제품 코드 + 성분")
        else:
            print(f"크롤링 대상: 최소 모드 (제품 코드만)")
    else:
        print(f"크롤링 대상: 제품={crawl_products}, 리뷰={crawl_reviews}, 성분={crawl_ingredients}")
    print(f"{'='*60}")

    confirm = input("\n시작하시겠습니까? (y/n): ").strip().lower()
    if confirm != 'y':
        print("취소되었습니다.")
        return

    # 크롤링 시작
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    all_products = []
    all_reviews = []
    all_ingredients = []

    # 중복 방지용: 이미 크롤링한 product_code 추적
    seen_product_codes = set()

    try:
        for middle, middle_code, small_code, small_name in categories:
            logger.info(f"{'='*60}")
            logger.info(f"카테고리: {middle} > {small_name}")
            logger.info(f"{'='*60}")

            # 카테고리 URL
            category_url = get_category_url(middle_code, small_code)

            # 제품 링크 수집
            links = get_all_product_links(driver, category_url, small_name)

            # 각 제품 크롤링
            for idx, link in enumerate(links, 1):
                try:
                    # URL에서 pdNo 미리 추출 (로깅용)
                    pdno_match = re.search(r"pdNo=([A-Z0-9]+)", link)
                    pdno_preview = pdno_match.group(1) if pdno_match else "알수없음"
                    logger.info(f"\n{'='*60}")
                    logger.info(f"[{idx}/{len(links)}] 제품 크롤링 - pdNo: {pdno_preview}")
                    logger.info(f"{'='*60}")

                    product, reviews, ingredients = crawl_product_detail(
                        driver, link,
                        category_home="뷰티/위생",
                        category_1=middle,
                        category_2=small_name,
                        crawl_reviews=crawl_reviews,
                        crawl_ingredients=crawl_ingredients
                    )

                    if product:
                        # 중복 체크
                        if product["product_code"] in seen_product_codes:
                            logger.warning(f"중복 제품 감지 - 스킵: product_code={product['product_code']}, 제품명={product['name'][:40]}")
                            logger.warning(f"   요청 pdNo: {pdno_preview} → 실제 product_code: {product['product_code']}")
                            continue

                        # 새로운 제품이면 추가
                        seen_product_codes.add(product["product_code"])

                        # minimal_mode일 때는 제품 정보 저장 안 함
                        if not minimal_mode:
                            all_products.append(product)
                        all_reviews.extend(reviews)
                        all_ingredients.extend(ingredients)

                        if minimal_mode:
                            if crawl_reviews and not crawl_ingredients:
                                logger.info(f"제품 코드 + 리뷰 크롤링 완료: [{product['name'][:40]}] | 리뷰: {len(reviews)}개")
                            elif crawl_ingredients and not crawl_reviews:
                                logger.info(f"제품 코드 + 성분 크롤링 완료: [{product['name'][:40]}] | 성분: {len(ingredients)}개")
                            else:
                                logger.info(f"제품 코드 크롤링 완료: [{product['name'][:40]}] | 리뷰: {len(reviews)}개 | 성분: {len(ingredients)}개")
                        else:
                            logger.info(f"제품 정보 + 리뷰 + 성분 크롤링 완료: [{product['name'][:40]}] | 리뷰: {len(reviews)}개 | 성분: {len(ingredients)}개")

                    time.sleep(1)

                except Exception as e:
                    logger.error(f"크롤링 실패: {link}")
                    logger.error(str(e))
                    continue

        # CSV 저장 - 하나의 파일로 통합
        date_str = get_date_string()

        # 카테고리 기반 파일명 생성
        # categories: [(middle, middle_code, small_code, small_name), ...]
        if categories:
            # 중분류 추출 (중복 제거)
            middle_names = list(dict.fromkeys([cat[0] for cat in categories]))
            # 소분류 추출 (중복 제거)
            small_names = list(dict.fromkeys([cat[3] for cat in categories]))

            # 파일명용 문자열 생성 (특수문자 제거)
            middle_str = '_'.join(middle_names)
            small_str = '_'.join(small_names)

            # 파일명에 사용할 수 없는 문자 제거/치환
            category_suffix = f"{middle_str}_{small_str}".replace('/', '_').replace(':', '_').replace(' ', '')
        else:
            category_suffix = "all"

        # data 디렉토리 생성
        os.makedirs('data', exist_ok=True)

        # 제품 정보 저장 (하나의 파일)
        if all_products and not minimal_mode:
            df_products = pd.DataFrame(all_products)
            product_file = f'data/products_{category_suffix}_{date_str}.csv'
            df_products.to_csv(product_file, index=False, encoding='utf-8-sig')
            logger.info(f"제품 정보 저장 완료: {product_file} ({len(df_products)}개)")
            print(f"\n제품 정보: {product_file} ({len(df_products)}개)")

        # 리뷰 저장 (하나의 파일)
        if all_reviews:
            df_reviews = pd.DataFrame(all_reviews)
            review_file = f'data/reviews_{category_suffix}_{date_str}.csv'
            df_reviews.to_csv(review_file, index=False, encoding='utf-8-sig')
            logger.info(f"리뷰 저장 완료: {review_file} ({len(df_reviews)}개)")
            print(f"리뷰: {review_file} ({len(df_reviews)}개)")

        # 성분 저장 (하나의 파일)
        if all_ingredients:
            df_ingredients = pd.DataFrame(all_ingredients)
            ingredient_file = f'data/ingredients_{category_suffix}_{date_str}.csv'
            df_ingredients.to_csv(ingredient_file, index=False, encoding='utf-8-sig')
            logger.info(f"성분 저장 완료: {ingredient_file} ({len(df_ingredients)}개)")
            print(f"성분: {ingredient_file} ({len(df_ingredients)}개)")

        # BigQuery 적재
        if BIGQUERY_AVAILABLE:
            print(f"\n{'='*60}")
            bq_confirm = input("BigQuery에 적재하시겠습니까? (y/n): ").strip().lower()
            if bq_confirm == 'y':
                try:
                    print("\nBigQuery 적재 시작...")
                    etl = CrawlerETL()

                    if all_products and not minimal_mode:
                        etl.load_products(product_file)
                        logger.info(f"BigQuery 제품 적재 완료")

                    if all_reviews:
                        etl.load_reviews(review_file)
                        logger.info(f"BigQuery 리뷰 적재 완료")

                    if all_ingredients:
                        etl.load_ingredients(ingredient_file)
                        logger.info(f"BigQuery 성분 적재 완료")

                    print("BigQuery 적재 완료!")
                except Exception as e:
                    logger.error(f"BigQuery 적재 실패: {str(e)}")
                    print(f"BigQuery 적재 실패: {str(e)}")

        print(f"\n{'='*60}")
        print("크롤링 완료")
        print(f"{'='*60}")

    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        # 에러 시에도 수집된 데이터 긴급 저장
        _save_intermediate(all_products, all_reviews, all_ingredients, prefix="emergency")

    finally:
        driver.quit()
        logger.info("브라우저 종료 완료")


def _save_intermediate(products, reviews, ingredients, prefix="intermediate"):
    """중간 저장: 수집된 데이터를 임시 CSV로 저장"""
    os.makedirs("data", exist_ok=True)
    date_str = get_date_string()
    if products:
        pd.DataFrame(products).to_csv(
            f"data/{prefix}_products_{date_str}.csv", index=False, encoding="utf-8-sig"
        )
    if reviews:
        pd.DataFrame(reviews).to_csv(
            f"data/{prefix}_reviews_{date_str}.csv", index=False, encoding="utf-8-sig"
        )
    if ingredients:
        pd.DataFrame(ingredients).to_csv(
            f"data/{prefix}_ingredients_{date_str}.csv", index=False, encoding="utf-8-sig"
        )


def run_all(crawl_reviews=True, crawl_ingredients=True, headless=True,
            categories_filter=None, history=None):
    """비대화형 모드: 전체/증분 카테고리 크롤링 (파이프라인용)

    전체 뷰티 카테고리를 자동으로 크롤링하고 CSV를 저장한다.
    main()의 대화형 input() 없이 실행 가능.

    Parameters
    ----------
    crawl_reviews : 리뷰 크롤링 여부
    crawl_ingredients : 성분 크롤링 여부
    headless : 헤드리스 모드 (True=브라우저 숨김)
    categories_filter : 카테고리 필터 dict (None이면 전체)
        예: {"스킨케어": ["all"], "메이크업": ["베이스메이크업", "립메이크업"]}
    history : CrawlHistory 인스턴스 (None이면 풀 크롤링)

    Returns
    -------
    (products_csv_path, reviews_csv_path, ingredients_csv_path)
    경로가 없으면 해당 위치에 None
    """
    # 카테고리 필터링 적용
    categories = []
    for middle_name, info in DAISO_BEAUTY_CATEGORIES.items():
        if categories_filter and middle_name not in categories_filter:
            continue
        allowed_sub = categories_filter.get(middle_name) if categories_filter else None
        middle_code = info["중분류코드"]
        for small_code, small_name in info["소분류"].items():
            if allowed_sub and "all" not in allowed_sub and small_name not in allowed_sub:
                continue
            categories.append((middle_name, middle_code, small_code, small_name))

    mode_label = "증분" if history else "풀"
    logger.info(f"run_all 시작 ({mode_label}): {len(categories)}개 카테고리, reviews={crawl_reviews}, ingredients={crawl_ingredients}")

    # 드라이버 설정
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    all_products = []
    all_reviews = []
    all_ingredients = []
    seen_product_codes = set()
    stats = {"new": 0, "updated": 0, "skipped": 0}
    crawled_count = 0  # 중간 저장 카운터

    try:
        for middle, middle_code, small_code, small_name in categories:
            logger.info(f"{'='*60}")
            logger.info(f"카테고리: {middle} > {small_name}")

            category_url = get_category_url(middle_code, small_code)
            links = get_all_product_links(driver, category_url, small_name)

            for idx, link in enumerate(links, 1):
                try:
                    pdno_match = re.search(r"pdNo=([A-Z0-9]+)", link)
                    product_code = pdno_match.group(1) if pdno_match else None
                    if not product_code:
                        logger.warning(f"pdNo 추출 실패: {link}")
                        continue

                    if product_code in seen_product_codes:
                        logger.debug(f"중복 스킵: {product_code}")
                        stats["skipped"] += 1
                        continue

                    seen_product_codes.add(product_code)

                    # 증분 모드: 기존 제품은 리뷰만 업데이트
                    if history and not history.is_new_product(product_code):
                        if not crawl_reviews:
                            stats["skipped"] += 1
                            continue

                        cutoff = history.get_last_review_date(product_code)
                        logger.info(f"[{idx}/{len(links)}] 기존 제품 리뷰 업데이트: {product_code} (cutoff: {cutoff})")

                        _, reviews, _ = crawl_product_detail(
                            driver, link,
                            category_home="뷰티/위생",
                            category_1=middle,
                            category_2=small_name,
                            crawl_reviews=True,
                            crawl_ingredients=False,
                            review_cutoff_date=cutoff,
                        )

                        all_reviews.extend(reviews)
                        if reviews:
                            max_date = max(r["date"] for r in reviews)
                            history.update_product(product_code, review_date=max_date)
                        stats["updated"] += 1

                    else:
                        # 신규 제품 또는 풀 모드 → 전체 크롤링
                        logger.info(f"[{idx}/{len(links)}] {'신규' if history else ''} 전체 크롤링: {product_code}")

                        product, reviews, ingredients = crawl_product_detail(
                            driver, link,
                            category_home="뷰티/위생",
                            category_1=middle,
                            category_2=small_name,
                            crawl_reviews=crawl_reviews,
                            crawl_ingredients=crawl_ingredients,
                        )

                        if product:
                            all_products.append(product)
                            all_reviews.extend(reviews)
                            all_ingredients.extend(ingredients)

                            logger.info(
                                f"완료: [{product['name'][:40]}] | "
                                f"리뷰: {len(reviews)}개 | 성분: {len(ingredients)}개"
                            )

                            if history:
                                max_date = max((r["date"] for r in reviews), default=None) if reviews else None
                                history.update_product(product_code, review_date=max_date)

                        stats["new"] += 1

                    crawled_count += 1

                    # 50개 제품마다 중간 저장 (에러 시 데이터 유실 방지)
                    if crawled_count % 50 == 0:
                        _save_intermediate(all_products, all_reviews, all_ingredients)
                        if history:
                            history.save()
                        logger.info(f"[중간저장] {crawled_count}개 제품 처리 완료")

                    time.sleep(1)

                except Exception as e:
                    logger.error(f"크롤링 실패: {link} — {e}")
                    continue

        logger.info(f"통계: 신규={stats['new']}, 업데이트={stats['updated']}, 스킵={stats['skipped']}")

        # CSV 저장
        date_str = get_date_string()
        os.makedirs("data", exist_ok=True)

        product_file = None
        review_file = None
        ingredient_file = None

        if all_products:
            product_file = f"data/products_all_{date_str}.csv"
            pd.DataFrame(all_products).to_csv(product_file, index=False, encoding="utf-8-sig")
            logger.info(f"제품 저장: {product_file} ({len(all_products)}개)")

        if all_reviews:
            review_file = f"data/reviews_all_{date_str}.csv"
            pd.DataFrame(all_reviews).to_csv(review_file, index=False, encoding="utf-8-sig")
            logger.info(f"리뷰 저장: {review_file} ({len(all_reviews)}개)")

        if all_ingredients:
            ingredient_file = f"data/ingredients_all_{date_str}.csv"
            pd.DataFrame(all_ingredients).to_csv(ingredient_file, index=False, encoding="utf-8-sig")
            logger.info(f"성분 저장: {ingredient_file} ({len(all_ingredients)}개)")

        logger.info("run_all 완료")
        return product_file, review_file, ingredient_file

    except Exception as e:
        logger.error(f"run_all 오류: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # 에러 시에도 수집된 데이터 긴급 저장
        _save_intermediate(all_products, all_reviews, all_ingredients, prefix="emergency")
        raise

    finally:
        # 이력은 에러 여부와 관계없이 반드시 저장
        if history:
            try:
                history.save()
                logger.info(f"크롤링 이력 저장 완료 (총 {history.product_count}개 제품)")
            except Exception as e:
                logger.error(f"이력 저장 실패: {e}")
        driver.quit()
        logger.info("브라우저 종료 완료")


def run_all_bq(headless=True, history=None, bq_dataset="daiso"):
    """BQ 기반 전체 제품 리뷰 증분 수집

    BQ products_core에서 전체 product_code를 조회하고,
    각 제품의 상세 페이지에 직접 접근하여 cutoff 이후 리뷰만 수집한다.
    성분/제품정보는 건너뛰고 리뷰만 수집하므로 효율적이다.

    Parameters
    ----------
    headless : 헤드리스 모드
    history : CrawlHistory 인스턴스 (cutoff 날짜 조회용, 필수)
    bq_dataset : BQ 데이터셋명

    Returns
    -------
    (None, reviews_csv_path, None)
    """
    if not history:
        logger.error("run_all_bq: history 인스턴스가 필요합니다 (cutoff 날짜 조회용)")
        return None, None, None

    # BQ에서 전체 제품 코드 조회
    bq_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '02_bigquery'))
    sys.path.insert(0, bq_dir)
    try:
        from bq_client import query_to_df
    except ImportError:
        logger.error("run_all_bq: bq_client 모듈을 불러올 수 없습니다")
        return None, None, None

    try:
        df_products = query_to_df(
            f"SELECT CAST(product_code AS STRING) AS product_code "
            f"FROM `{bq_dataset}.products_core`"
        )
    except Exception as e:
        logger.error(f"run_all_bq: BQ 제품 조회 실패 — {e}")
        return None, None, None

    product_codes = df_products["product_code"].tolist()
    logger.info(f"run_all_bq 시작: BQ에서 {len(product_codes)}개 제품 조회 완료")

    # 드라이버 설정
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )

    all_reviews = []
    stats = {"updated": 0, "skipped": 0, "error": 0}

    try:
        for idx, product_code in enumerate(product_codes, 1):
            try:
                cutoff = history.get_last_review_date(product_code)

                url = f"{BASE_URL}/pd/pdr/SCR_PDR_0001?pdNo={product_code}"
                logger.info(
                    f"[{idx}/{len(product_codes)}] BQ 리뷰 동기화: {product_code} "
                    f"(cutoff: {cutoff or '없음'})"
                )

                _, reviews, _ = crawl_product_detail(
                    driver, url,
                    category_home="",
                    category_1="",
                    category_2="",
                    crawl_reviews=True,
                    crawl_ingredients=False,
                    review_cutoff_date=cutoff,
                )

                if reviews:
                    all_reviews.extend(reviews)
                    max_date = max(r["date"] for r in reviews)
                    history.update_product(product_code, review_date=max_date)
                    stats["updated"] += 1
                    logger.info(f"  → 신규 리뷰 {len(reviews)}개 수집")
                else:
                    stats["skipped"] += 1

                # 100개 제품마다 중간 저장
                if idx % 100 == 0:
                    _save_intermediate([], all_reviews, [], prefix="bq_sync")
                    history.save()
                    logger.info(
                        f"[중간저장] {idx}/{len(product_codes)}개 처리 | "
                        f"리뷰 누적: {len(all_reviews)}개"
                    )

                time.sleep(1)

            except Exception as e:
                logger.error(f"BQ 리뷰 동기화 실패: {product_code} — {e}")
                stats["error"] += 1
                continue

        logger.info(
            f"run_all_bq 통계: 업데이트={stats['updated']}, "
            f"스킵={stats['skipped']}, 에러={stats['error']}"
        )

        # CSV 저장
        review_file = None
        if all_reviews:
            date_str = get_date_string()
            os.makedirs("data", exist_ok=True)
            review_file = f"data/reviews_bq_sync_{date_str}.csv"
            pd.DataFrame(all_reviews).to_csv(
                review_file, index=False, encoding="utf-8-sig"
            )
            logger.info(f"BQ 동기화 리뷰 저장: {review_file} ({len(all_reviews)}개)")

        return None, review_file, None

    except Exception as e:
        logger.error(f"run_all_bq 오류: {e}")
        import traceback
        logger.error(traceback.format_exc())
        _save_intermediate([], all_reviews, [], prefix="emergency_bq_sync")
        raise

    finally:
        if history:
            try:
                history.save()
                logger.info(f"BQ 동기화 이력 저장 완료 (총 {history.product_count}개 제품)")
            except Exception as e:
                logger.error(f"이력 저장 실패: {e}")
        driver.quit()
        logger.info("브라우저 종료 완료")


if __name__ == "__main__":
    main()
