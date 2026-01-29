"""
다이소몰 건강식품 크롤러
"""
from selenium.webdriver.common.by import By
from driver_setup import create_driver, quit_driver
from utils import *
from config import DAISO_CONFIG, DAISO_HEALTH_FOOD
from tqdm import tqdm
import traceback
import re

logger = setup_logger('daiso_health', 'daiso_health.log')

def extract_rating_from_text(text):
    """'별점 4.8점' 형식에서 숫자 추출"""
    match = re.search(r'별점\s*([\d.]+)점', text)
    if match:
        return float(match.group(1))
    return None

def extract_review_count(text):
    """'5,710 건 작성' 형식에서 숫자 추출"""
    match = re.search(r'([\d,]+)', text.strip())
    if match:
        return int(match.group(1).replace(',', ''))
    return 0

def build_health_category_url(large_code, middle_code, sub_code):
    """건강식품 카테고리 URL 생성

    구조: /ds/exhCtgr/{대분류코드}/{중분류코드}/{소분류코드}
    예: /ds/exhCtgr/CTGR_00022/CTGR_01020/CTGR_01024
    """
    base = DAISO_CONFIG['base_url']
    return f"{base}/ds/exhCtgr/{large_code}/{middle_code}/{sub_code}"

def crawl_health_category(driver, main_category, sub_category_name, large_code, middle_code, sub_code, max_products=50):
    """건강식품 카테고리별 상품 크롤링"""

    products = []
    url = build_health_category_url(large_code, middle_code, sub_code)
    full_category = f"{main_category}/{sub_category_name}"

    try:
        logger.info(f"카테고리 접속: {full_category} - {url}")
        driver.get(url)
        random_delay(3, 5)

        # 페이지 스크롤
        scroll_page(driver, scroll_pause=1, max_scrolls=10)

        # 상품 카드 찾기
        selectors = [
            '.product-card',
            '.swiper-slide',
            'div[class*="product-card"]',
        ]

        product_elements = []
        for selector in selectors:
            product_elements = driver.find_elements(By.CSS_SELECTOR, selector)
            if len(product_elements) > 0:
                logger.info(f"선택자 '{selector}' 사용, {len(product_elements)}개 발견")
                break

        if len(product_elements) == 0:
            logger.warning(f"상품을 찾을 수 없습니다: {full_category}")
            return products

        # 최대 개수 제한
        product_elements = product_elements[:max_products]

        for idx, elem in enumerate(tqdm(product_elements, desc=f"{full_category[:20]}")):
            try:
                driver.execute_script("arguments[0].scrollIntoView(true);", elem)
                random_delay(0.3, 0.7)

                product_data = {
                    'site': '다이소',
                    'main_category': main_category,
                    'sub_category': sub_category_name,
                    'crawled_at': get_timestamp(),
                    'brand': '다이소',
                }

                # 상품명
                try:
                    name_elem = elem.find_element(By.CSS_SELECTOR, '.product-title')
                    product_data['product_name'] = name_elem.text.strip()
                except:
                    product_data['product_name'] = ''

                # 가격
                try:
                    price_elem = elem.find_element(By.CSS_SELECTOR, '.price-value .value')
                    price_text = price_elem.text.strip()
                    product_data['price'] = int(price_text.replace(',', ''))
                except:
                    product_data['price'] = None

                # 원가, 할인율
                product_data['original_price'] = None
                product_data['discount_rate'] = ''

                # 평점
                try:
                    rating_elem = elem.find_element(By.CSS_SELECTOR, '.rating-star .hiddenText')
                    rating_text = rating_elem.text.strip()
                    product_data['rating'] = extract_rating_from_text(rating_text)
                except:
                    product_data['rating'] = None

                # 리뷰 수
                try:
                    review_elem = elem.find_element(By.CSS_SELECTOR, '.star-detail')
                    review_text = review_elem.text.strip()
                    product_data['review_count'] = extract_review_count(review_text)
                except:
                    product_data['review_count'] = 0

                # 이미지 URL
                try:
                    img_elem = elem.find_element(By.CSS_SELECTOR, 'img')
                    img_src = img_elem.get_attribute('src')
                    if not img_src:
                        srcset = img_elem.get_attribute('srcset')
                        if srcset:
                            img_src = srcset.split()[0]
                    product_data['image_url'] = img_src if img_src else ''
                except:
                    product_data['image_url'] = ''

                # 상품 URL
                try:
                    link_elem = elem.find_element(By.CSS_SELECTOR, '.detail-link')
                    href = link_elem.get_attribute('href')
                    if href and not href.startswith('http'):
                        href = DAISO_CONFIG['base_url'] + href
                    product_data['product_url'] = href if href else ''
                except:
                    product_data['product_url'] = ''

                # 데이터 추가
                if product_data['product_name']:
                    products.append(product_data)

            except Exception as e:
                logger.warning(f"상품 {idx+1} 추출 실패: {str(e)[:100]}")
                continue

        logger.info(f"카테고리 '{full_category}' 완료: {len(products)}개 수집")

    except Exception as e:
        logger.error(f"카테고리 크롤링 오류: {e}")
        logger.error(traceback.format_exc())

    return products

def select_categories():
    """사용자가 크롤링할 카테고리 선택"""

    print("\n" + "=" * 70)
    print("다이소 건강식품 카테고리 선택")
    print("=" * 70)

    health_data = DAISO_HEALTH_FOOD["건강식품"]
    sub_categories = health_data['소분류']

    print("\n💊 건강식품 소분류:")
    sub_list = list(sub_categories.items())
    for i, (code, name) in enumerate(sub_list, 1):
        print(f"  {i}. {name}")
    print(f"  0. 전체 크롤링 (9개 소분류)")

    choice = input(f"\n크롤링할 카테고리 번호 (0-{len(sub_list)}): ").strip()

    if choice == '0':
        return 'all'

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(sub_list):
            return sub_list[idx]
    except:
        pass

    print("❌ 잘못된 입력입니다. 전체 크롤링을 진행합니다.")
    return 'all'

def crawl_daiso_health():
    """다이소 건강식품 크롤링 메인 함수"""

    logger.info("=" * 60)
    logger.info("다이소 건강식품 크롤링 시작")
    logger.info("=" * 60)

    driver = None
    all_products = []

    try:
        # 카테고리 선택
        selected = select_categories()

        # 드라이버 생성
        logger.info("Chrome 드라이버 생성 중...")
        driver = create_driver()
        logger.info("드라이버 생성 완료")

        health_data = DAISO_HEALTH_FOOD["건강식품"]
        large_code = health_data['대분류코드']
        middle_code = health_data['중분류코드']
        max_products = DAISO_CONFIG['max_products_per_category']

        # 크롤링 실행
        if selected == 'all':
            # 전체 크롤링
            logger.info(f"\n{'='*20} 건강식품 전체 {'='*20}")
            for sub_code, sub_name in health_data['소분류'].items():
                products = crawl_health_category(
                    driver, "건강식품", sub_name, large_code, middle_code, sub_code, max_products
                )
                all_products.extend(products)
                random_delay(2, 3)
        else:
            # 특정 소분류만
            sub_code, sub_name = selected
            logger.info(f"\n{'='*20} 건강식품/{sub_name} {'='*20}")
            products = crawl_health_category(
                driver, "건강식품", sub_name, large_code, middle_code, sub_code, max_products
            )
            all_products.extend(products)

        # 데이터 저장
        if all_products:
            date_str = get_date_string()
            filename = f'daiso_health_{date_str}.csv'
            filepath = save_to_csv(all_products, filename)
            logger.info(f"\n✅ 데이터 저장 완료: {filepath}")
            logger.info(f"✅ 총 {len(all_products)}개 상품 수집")

            # 통계
            stats = {}
            for p in all_products:
                sub = p['sub_category']
                stats[sub] = stats.get(sub, 0) + 1

            logger.info("\n=== 카테고리별 수집 현황 ===")
            for cat, count in sorted(stats.items()):
                logger.info(f"  - {cat}: {count}개")

            print(f"\n✅ 크롤링 완료! 총 {len(all_products)}개 상품 수집")
            print(f"📁 파일: {filepath}")
        else:
            logger.warning("수집된 데이터가 없습니다.")
            print("\n⚠️  수집된 데이터가 없습니다.")

    except Exception as e:
        logger.error(f"크롤링 실패: {e}")
        logger.error(traceback.format_exc())
        print(f"\n❌ 크롤링 실패: {e}")

    finally:
        if driver:
            logger.info("브라우저 종료 중...")
            quit_driver(driver)
            logger.info("브라우저 종료 완료")

    return all_products

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("다이소몰 건강식품 크롤러")
    print("=" * 70)
    print("\n💊 건강식품 카테고리 (9개 소분류):")
    health_data = DAISO_HEALTH_FOOD["건강식품"]
    for code, name in health_data['소분류'].items():
        print(f"  • {name}")
    print("\n⚠️  주의: 교육/연구 목적입니다.")
    print("=" * 70)

    input("\n시작하려면 Enter를 누르세요...")

    crawl_daiso_health()
