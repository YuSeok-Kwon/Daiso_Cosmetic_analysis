"""
다이소 뷰티/위생 카테고리 통합 크롤러
- 제품 정보 (product_all.csv)
- 리뷰 (reviews_all.csv)
- 성분 (ingredients_all.csv)
"""
import os
import time
import re
import pandas as pd
from collections import defaultdict
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from config import DAISO_BEAUTY_CATEGORIES
from modules.ocr_utils_split import extract_text_from_image_url_split
from modules.halal_vegan_checker import check_halal_vegan_status
from modules.ingredient_parser import (
    normalize_ingredient_name,
    is_valid_ingredient,
    extract_from_text
)
from utils import setup_logger, get_date_string

# 로거 설정
logger = setup_logger('daiso_beauty_crawler', 'daiso_beauty_crawler.log')

# 기본 설정
BASE_URL = "https://www.daisomall.co.kr"
MAX_SCROLLS = 10
user_id_map = defaultdict(lambda: f"user_{len(user_id_map)+1:04d}")


def extract_ingredients_multi_source(driver, product_code: str, product_name: str) -> list:
    """
    다중 소스에서 성분 추출 및 교차 검증 + 할랄/비건 판정

    Returns:
        list of dicts with: product_id, name, ingredient, can_halal, can_vegan, not_halal
    """
    all_ingredients = {}  # {성분명: {confidence, sources[], reason}}

    # 소스 1: Picture alt 속성
    try:
        pictures = driver.find_elements(By.CSS_SELECTOR, "div.editor-content picture img")

        for idx, img in enumerate(pictures):
            alt_text = img.get_attribute("alt") or ""

            if any(kw in alt_text for kw in INGREDIENT_KEYWORDS):
                alt_ingredients = extract_from_text(alt_text, source=f"ALT_{idx}")

                for ing in alt_ingredients:
                    name = normalize_ingredient_name(ing['ingredient'])
                    is_valid, conf, reason = is_valid_ingredient(name)

                    if is_valid and conf >= 0.5:
                        if name not in all_ingredients:
                            all_ingredients[name] = {'confidence': conf, 'sources': [ing['source']], 'reason': reason}
                        else:
                            all_ingredients[name]['sources'].append(ing['source'])
                            all_ingredients[name]['confidence'] = min(1.0, all_ingredients[name]['confidence'] + 0.1)

        logger.info(f"ALT에서 성분 발견: 총 {len([k for k in all_ingredients if any('ALT' in s for s in all_ingredients[k]['sources'])])}개")

    except Exception as e:
        logger.debug(f"ALT 텍스트 추출 실패: {str(e)}")

    # 소스 2: OCR (성분이 적을 때만 - 10개 미만)
    if len(all_ingredients) < 10:
        try:
            pictures = driver.find_elements(By.CSS_SELECTOR, "div.editor-content picture img")

            for idx, img in enumerate(pictures[-3:]):  # 마지막 3개 이미지만
                src = img.get_attribute("src")

                if src:
                    logger.info(f"OCR 분석 중: 이미지 {idx + 1}")

                    sections = extract_text_from_image_url_split(src, num_sections=5)  # 5개 섹션으로 세분화

                    for section_idx, section in enumerate(sections or []):
                        text = section.get('text', '')

                        if any(kw in text for kw in INGREDIENT_KEYWORDS):
                            ocr_ingredients = extract_from_text(text, source=f"OCR_{idx}_{section_idx}")

                            for ing in ocr_ingredients:
                                name = normalize_ingredient_name(ing['ingredient'])
                                is_valid, conf, reason = is_valid_ingredient(name)

                                # OCR은 오류 가능성이 있으므로 신뢰도 페널티
                                conf *= 0.9

                                if is_valid and conf >= 0.5:
                                    if name not in all_ingredients:
                                        all_ingredients[name] = {'confidence': conf, 'sources': [ing['source']], 'reason': reason}
                                    else:
                                        all_ingredients[name]['sources'].append(ing['source'])
                                        all_ingredients[name]['confidence'] = min(1.0, all_ingredients[name]['confidence'] + 0.05)

            logger.info(f"OCR에서 추가 성분: 총 {len([k for k in all_ingredients if any('OCR' in s for s in all_ingredients[k]['sources'])])}개")

        except Exception as e:
            logger.error(f"OCR 실패: {str(e)}")

    # 최종 필터링: 신뢰도 기준 정렬 및 할랄/비건 판정
    final_ingredients = []

    for name, info in all_ingredients.items():
        # 여러 소스에서 발견된 성분 우선
        multi_source_bonus = len(info['sources']) * 0.05
        final_conf = min(1.0, info['confidence'] + multi_source_bonus)

        # 신뢰도 50% 이상만 포함
        if final_conf >= 0.5:
            # 할랄/비건 적합성 판정
            halal_vegan = check_halal_vegan_status(name)

            # 할랄 부적합 성분명 추출 (is_halal이 'No'인 경우)
            not_halal_ingredient = name if halal_vegan['is_halal'] == 'No' else ''

            final_ingredients.append({
                'product_id': product_code,
                'name': product_name,
                'ingredient': name,
                'can_halal': halal_vegan['is_halal'],
                'can_vegan': halal_vegan['is_vegan'],
                'not_halal': not_halal_ingredient
            })

    # 성분명 기준으로 정렬
    final_ingredients.sort(key=lambda x: x['ingredient'])

    logger.info(f"최종 성분: {len(final_ingredients)}개")

    # 할랄/비건 통계
    vegan_count = len([x for x in final_ingredients if x['can_vegan'] == 'Yes'])
    non_vegan_count = len([x for x in final_ingredients if x['can_vegan'] == 'No'])
    halal_questionable = len([x for x in final_ingredients if x['can_halal'] == 'Questionable'])
    haram_count = len([x for x in final_ingredients if x['can_halal'] == 'No'])

    logger.info(f"할랄/비건 분석:")
    logger.info(f"  - 비건 적합: {vegan_count}개 | 부적합: {non_vegan_count}개")
    logger.info(f"  - 할랄 의심: {halal_questionable}개 | 부적합: {haram_count}개")

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


def crawl_product_detail(driver, url, category_home, category_1, category_2, crawl_reviews=True, crawl_ingredients=True):
    """제품 상세 정보 크롤링"""
    # URL에서 pdNo를 먼저 추출 (primary key로 사용)
    url_pdno_match = re.search(r"pdNo=([A-Z0-9]+)", url)
    if not url_pdno_match:
        logger.error(f"URL에서 pdNo 추출 실패 - URL: {url}")
        return None, [], []

    url_pdno = url_pdno_match.group(1)
    logger.info(f"제품 크롤링 시작 - pdNo: {url_pdno}")

    product = {
        "product_code": url_pdno,  # URL pdNo를 기본값으로 사용
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
        "can_할랄인증": "Unknown",  # 할랄 인증 가능 여부
        "can_비건": "Unknown",  # 비건 인증 가능 여부
        "certifications": "",  # 제품 설명에 명시된 인증 정보
    }

    reviews = []
    ingredients = []

    # 페이지 로드
    driver.get(url)
    logger.debug("페이지 로딩 대기 시작")

    # 1단계: 기본 DOM 로딩 대기
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, "product-info-wrap"))
        )
        logger.debug("기본 DOM 로딩 완료")
    except Exception as e:
        logger.warning(f"product-info-wrap 로딩 타임아웃: {str(e)}")

    # 2단계: JavaScript 실행 완료 대기 (더 긴 시간)
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
            # info-area 내부의 product-title이 텍스트를 가질 때까지 대기
            name_element = WebDriverWait(driver, 3).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".info-area .product-title"))
            )
            name_text = name_element.text.strip()
            if name_text and len(name_text) > 0:
                logger.debug(f"제품명 로딩 완료 (시도 {retry + 1}/{max_retries})")
                break
            else:
                logger.debug(f"제품명 로딩 대기 중 (시도 {retry + 1}/{max_retries})")
                time.sleep(2)
        except Exception as e:
            logger.debug(f"제품명 대기 중 예외 (시도 {retry + 1}/{max_retries}): {str(e)}")
            time.sleep(2)
    else:
        logger.warning(f"제품명 로딩 타임아웃 - pdNo: {url_pdno}")

    # 실제 로드된 URL 확인 (리다이렉트 체크)
    current_url = driver.current_url
    current_pdno_match = re.search(r"pdNo=([A-Z0-9]+)", current_url)

    if current_pdno_match:
        current_pdno = current_pdno_match.group(1)
        if current_pdno != url_pdno:
            logger.warning(f"URL 리다이렉트 감지!")
            logger.warning(f"   요청 pdNo: {url_pdno} → 실제 pdNo: {current_pdno}")
            logger.warning(f"   이 제품은 중복 방지 로직에 의해 스킵될 수 있습니다")
            # 실제 로드된 pdNo로 업데이트
            product["product_code"] = current_pdno
            url_pdno = current_pdno
        else:
            logger.debug(f"URL 일치: {url_pdno}")
    else:
        logger.warning(f"현재 URL에서 pdNo 추출 불가: {current_url}")

    try:
        # 카테고리 breadcrumb
        crumbs = driver.find_elements(By.CSS_SELECTOR, ".el-breadcrumb__inner.is-link")
        texts = [c.text.strip() for c in crumbs if c.text.strip()]
    except:
        pass

    # 브랜드
    product["brand"] = extract_brand(driver, category_2)
    logger.debug(f"브랜드: {product['brand']}")

    # 제품명 추출 - h1.product-title 선택자 사용 (추천 제품 제외)
    # 핵심: h1 태그만 선택 (추천 제품은 div.product-title이므로 제외됨)
    name_selectors = [
        "h1.product-title",  # 메인 제품명 (h1 태그)
        ".info-area h1",  # 폴백 1
        ".product-info-wrap h1",  # 폴백 2
    ]

    for selector in name_selectors:
        try:
            element = driver.find_element(By.CSS_SELECTOR, selector)
            product["name"] = element.text.strip()
            if product["name"]:
                logger.info(f"제품명 추출 성공 ({selector}): {product['name'][:50]}")
                break
        except Exception as e:
            logger.debug(f"제품명 추출 실패 ({selector}): {str(e)}")
            continue

    # 3) 옵션 정보 추가 (있는 경우)
    if product["name"]:
        try:
            option_element = driver.find_element(By.CSS_SELECTOR, ".product-option-text, .option-text, .selected-option")
            option_text = option_element.text.strip()
            if option_text and option_text not in product["name"]:
                product["name"] = f"{product['name']} ({option_text})"
                logger.info(f"옵션 정보 추가: {option_text}")
        except:
            logger.debug("옵션 정보 없음")

    # 4) 최종 검증
    if not product["name"]:
        logger.error(f"제품명 추출 완전 실패 - pdNo: {url_pdno}")

        # 디버깅: HTML 구조 저장 (첫 번째 실패만)
        if not hasattr(crawl_product_detail, '_name_fail_saved'):
            try:
                os.makedirs('logs/name_fail', exist_ok=True)

                # HTML 저장
                html_file = f"logs/name_fail/fail_{url_pdno}.html"
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(driver.page_source)

                # 스크린샷 저장
                screenshot_file = f"logs/name_fail/fail_{url_pdno}.png"
                driver.save_screenshot(screenshot_file)

                # product-title 요소들 모두 출력
                all_titles = driver.find_elements(By.CLASS_NAME, "product-title")
                logger.info(f"페이지 내 product-title 요소 수: {len(all_titles)}")
                for idx, title in enumerate(all_titles):
                    logger.info(f"  [{idx}] {title.text[:50]}...")

                logger.info(f"디버깅 파일 저장: {html_file}")
                logger.info(f"스크린샷 저장: {screenshot_file}")
                crawl_product_detail._name_fail_saved = True
            except Exception as e:
                logger.error(f"디버깅 파일 저장 실패: {e}")

    # 가격 - 상세 페이지의 가격만 선택 (추천 제품 제외)
    try:
        # 1순위: .prod-price--detail 내부의 가격
        price_element = driver.find_element(By.CSS_SELECTOR, ".prod-price--detail .price-value .value")
        product["price"] = price_element.text.strip().replace(",", "")
    except:
        try:
            # 2순위: .inner-box 내부의 첫 번째 가격
            price_element = driver.find_element(By.CSS_SELECTOR, ".inner-box .price-value .value")
            product["price"] = price_element.text.strip().replace(",", "")
        except:
            pass

    # 페이지 품번 확인 (검증용)
    page_product_code = None
    try:
        code_text = driver.find_element(By.CLASS_NAME, "code-text").text
        match = re.search(r"품번\s*(\d+)", code_text)
        if match:
            page_product_code = match.group(1)
            logger.debug(f"페이지 품번: {page_product_code}")

            # URL pdNo와 비교
            if page_product_code != url_pdno:
                logger.warning(f"품번 불일치 감지!")
                logger.warning(f"   URL pdNo: {url_pdno}")
                logger.warning(f"   페이지 품번: {page_product_code}")
                logger.warning(f"   제품명: {product['name']}")

                # 디버깅: 첫 번째 불일치 케이스만 HTML 저장
                if not hasattr(crawl_product_detail, '_mismatch_saved'):
                    try:
                        os.makedirs('logs/mismatch', exist_ok=True)

                        # HTML 저장
                        html_file = f"logs/mismatch/mismatch_{url_pdno}_vs_{page_product_code}.html"
                        with open(html_file, 'w', encoding='utf-8') as f:
                            f.write(driver.page_source)

                        # 스크린샷 저장
                        screenshot_file = f"logs/mismatch/mismatch_{url_pdno}_vs_{page_product_code}.png"
                        driver.save_screenshot(screenshot_file)

                        logger.info(f"디버깅 파일 저장: {html_file}")
                        logger.info(f"스크린샷 저장: {screenshot_file}")
                        crawl_product_detail._mismatch_saved = True
                    except Exception as e:
                        logger.error(f"디버깅 파일 저장 실패: {e}")

                # 페이지 품번을 제품 코드로 사용 2
                logger.warning(f"   → 페이지 품번({page_product_code})을 제품 코드로 사용합니다")
                product["product_code"] = page_product_code
            else:
                logger.debug(f"품번 일치: {url_pdno}")
        else:
            logger.debug(f"code-text 있지만 품번 패턴 없음: {code_text}")
    except Exception as e:
        logger.debug(f"페이지 품번 추출 실패 (URL pdNo 사용): {str(e)}")

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

    # 제품 코드 검증
    if not product["product_code"]:
        logger.error(f"제품 코드 없음 - 스킵: {product['name']}")
        return None, [], []

    # 최종 제품 정보 로그
    logger.info(f"제품 정보 수집 완료:")
    logger.info(f"   - 제품 코드: {product['product_code']}")
    logger.info(f"   - 제품명: {product['name'][:50]}...")
    logger.info(f"   - 브랜드: {product['brand']}")
    logger.info(f"   - 가격: {product['price']}원")

    # 5천원 초과 제품 제외
    try:
        if product["price"] and int(product["price"]) > 5000:
            logger.info(f"제외 (가격 초과): {product['name']} | {product['price']}원")
            return None, [], []
    except:
        pass

    # 리뷰 크롤링
    if crawl_reviews:
        logger.info(f"리뷰 수집 시작: {product['name']}")
        for page in range(1, 999):
            time.sleep(1)
            review_elements = driver.find_elements(By.CLASS_NAME, "review-detail")
            logger.debug(f"{page}페이지 리뷰 수: {len(review_elements)}")

            for r in review_elements:
                try:
                    date = r.find_element(By.CLASS_NAME, "cw-bar-list").text.split()[0]
                    user_raw = r.find_element(By.CLASS_NAME, "con-writer-id").text.strip()
                    rating_raw = r.find_element(By.CLASS_NAME, "hiddenText").text.strip()
                    text = r.find_element(By.CSS_SELECTOR, ".review-desc .cont").text.strip()
                    image_count = len(r.find_elements(By.CSS_SELECTOR, ".swiper-wrapper img"))
                    rating = extract_rating(rating_raw)
                    user_id = user_id_map[user_raw]

                    reviews.append({
                        "product_code": product["product_code"],
                        "date": date,
                        "user_masked": user_raw,
                        "user": user_id,
                        "rating": rating,
                        "text": text,
                        "image_count": image_count,
                    })
                except:
                    continue

            # 다음 페이지
            try:
                next_btn = driver.find_element(By.CLASS_NAME, "btn-next")
                next_class = next_btn.get_attribute("class") or ""
                next_style = next_btn.get_attribute("style") or ""

                if ("disabled" in next_class) or (not next_btn.is_enabled()) or ("pointer-events: none" in next_style):
                    logger.debug("마지막 페이지 도달")
                    break

                driver.execute_script("arguments[0].click();", next_btn)
                time.sleep(1)
            except:
                logger.debug("다음 버튼 없음")
                break

    # 성분 크롤링
    if crawl_ingredients:
        logger.info(f"성분 수집 시작 (다중 소스): pdNo: {product['product_code']} | 제품명: {product['name'][:40]}")

        # 할랄/비건 인증 정보 추출
        certifications = []
        try:
            # 제품 설명에서 할랄/비건 키워드 검색
            editor_content = driver.find_element(By.CSS_SELECTOR, "div.editor-content")
            content_text = editor_content.text.lower()

            if '할랄' in content_text or 'halal' in content_text:
                certifications.append('할랄')
                logger.info("할랄 인증 제품 발견")

            if '비건' in content_text or 'vegan' in content_text:
                certifications.append('비건')
                logger.info("비건 인증 제품 발견")

            if certifications:
                product['certifications'] = ', '.join(certifications)
        except:
            pass

        try:
            # 새로운 다중 소스 성분 추출 함수 사용
            ingredients = extract_ingredients_multi_source(
                driver,
                product['product_code'],
                product['name']
            )

            logger.info(f"성분 추출 완료: {len(ingredients)}개 (다중 소스 검증)")

            # 할랄/비건 인증 가능 여부 판정
            if ingredients:
                # 비건 판정: 동물성 성분이 하나라도 있으면 No
                non_vegan_ingredients = [ing for ing in ingredients if ing.get('can_vegan') == 'No']
                if non_vegan_ingredients:
                    product['can_비건'] = 'No'
                    logger.info(f"비건 부적합: {len(non_vegan_ingredients)}개 동물성 성분 발견")
                elif any(ing.get('can_vegan') == 'Unknown' for ing in ingredients):
                    product['can_비건'] = 'Unknown'
                else:
                    product['can_비건'] = 'Yes'
                    logger.info("비건 인증 가능")

                # 할랄 판정: 부적합 성분이 하나라도 있으면 No
                haram_ingredients = [ing for ing in ingredients if ing.get('can_halal') == 'No']
                questionable_ingredients = [ing for ing in ingredients if ing.get('can_halal') == 'Questionable']

                if haram_ingredients:
                    product['can_할랄인증'] = 'No'
                    logger.info(f"할랄 부적합: {len(haram_ingredients)}개 부적합 성분 발견")
                elif questionable_ingredients:
                    product['can_할랄인증'] = 'Questionable'
                    logger.info(f"할랄 원료확인 필요: {len(questionable_ingredients)}개 의심 성분")
                elif any(ing.get('can_halal') == 'Unknown' for ing in ingredients):
                    product['can_할랄인증'] = 'Unknown'
                else:
                    product['can_할랄인증'] = 'Yes'
                    logger.info("할랄 인증 가능")

        except Exception as e:
            logger.error(f"다중 소스 성분 추출 오류: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

            # 폴백: 간단한 성분 추출 로직 (실패 시 빈 리스트)
            logger.warning("다중 소스 추출 실패 - 성분 정보 없음")
            ingredients = []

        # 성분 추출 실패 시 로그
        if not ingredients:
            logger.warning(f"성분 정보 추출 실패: {product['name']}")

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

        # data 디렉토리 생성
        os.makedirs('data', exist_ok=True)

        # 제품 정보 저장 (하나의 파일)
        if all_products and not minimal_mode:
            df_products = pd.DataFrame(all_products)
            product_file = f'data/products_{date_str}.csv'
            df_products.to_csv(product_file, index=False, encoding='utf-8-sig')
            logger.info(f"제품 정보 저장 완료: {product_file} ({len(df_products)}개)")
            print(f"\n제품 정보: {product_file} ({len(df_products)}개)")

        # 리뷰 저장 (하나의 파일)
        if all_reviews:
            df_reviews = pd.DataFrame(all_reviews)
            review_file = f'data/reviews_{date_str}.csv'
            df_reviews.to_csv(review_file, index=False, encoding='utf-8-sig')
            logger.info(f"리뷰 저장 완료: {review_file} ({len(df_reviews)}개)")
            print(f"리뷰: {review_file} ({len(df_reviews)}개)")

        # 성분 저장 (하나의 파일)
        if all_ingredients:
            df_ingredients = pd.DataFrame(all_ingredients)
            ingredient_file = f'data/ingredients_{date_str}.csv'
            df_ingredients.to_csv(ingredient_file, index=False, encoding='utf-8-sig')
            logger.info(f"성분 저장 완료: {ingredient_file} ({len(df_ingredients)}개)")
            print(f"성분: {ingredient_file} ({len(df_ingredients)}개)")

        print(f"\n{'='*60}")
        print("크롤링 완료")
        print(f"{'='*60}")

    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

    finally:
        driver.quit()
        logger.info("브라우저 종료 완료")


if __name__ == "__main__":
    main()
