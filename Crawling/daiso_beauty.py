"""
다이소몰 크롤러 
"""
from selenium.webdriver.common.by import By
from driver_setup import create_driver, quit_driver
from utils import *
from config import DAISO_CONFIG, DAISO_BEAUTY_CATEGORIES
from tqdm import tqdm
import traceback
import re

logger = setup_logger('daiso_complete', 'daiso_complete.log')

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

def build_category_url(middle_code, sub_code=None):
    """카테고리 URL 생성"""
    base = DAISO_CONFIG['base_url']
    large = DAISO_CONFIG['large_category']
    middle = DAISO_CONFIG['middle_category']

    if sub_code:
        # 소분류까지: /ds/exhCtgr/C208/CTGR_00014/CTGR_00057/CTGR_00366
        return f"{base}/ds/exhCtgr/{large}/{middle}/{middle_code}/{sub_code}"
    else:
        # 중분류까지: /ds/exhCtgr/C208/CTGR_00014/CTGR_00057
        return f"{base}/ds/exhCtgr/{large}/{middle}/{middle_code}"

def crawl_category(driver, main_category, sub_category_name, middle_code, sub_code, max_products=50):
    """카테고리별 상품 크롤링"""

    products = []
    url = build_category_url(middle_code, sub_code)
    full_category = f"{main_category}/{sub_category_name}" if sub_category_name else main_category

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
                    'sub_category': sub_category_name if sub_category_name else '',
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
    print("다이소 뷰티 카테고리 선택")
    print("=" * 70)

    print("\n대분류 카테고리:")
    main_categories = list(DAISO_BEAUTY_CATEGORIES.keys())
    for i, cat in enumerate(main_categories, 1):
        sub_count = len(DAISO_BEAUTY_CATEGORIES[cat]['소분류'])
        print(f"  {i}. {cat} ({sub_count}개 소분류)")
    print(f"  0. 전체 크롤링")

    choice = input("\n크롤링할 카테고리 번호 (0-5): ").strip()

    if choice == '0':
        return 'all', None

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(main_categories):
            selected_main = main_categories[idx]

            # 소분류 선택
            print(f"\n{selected_main} 소분류:")
            sub_categories = DAISO_BEAUTY_CATEGORIES[selected_main]['소분류']
            sub_list = list(sub_categories.items())

            for i, (code, name) in enumerate(sub_list, 1):
                print(f"  {i}. {name}")
            print(f"  0. {selected_main} 전체")

            sub_choice = input(f"\n크롤링할 소분류 번호 (0-{len(sub_list)}): ").strip()

            if sub_choice == '0':
                return selected_main, 'all'

            sub_idx = int(sub_choice) - 1
            if 0 <= sub_idx < len(sub_list):
                return selected_main, sub_list[sub_idx]

    except:
        pass

    print("잘못된 입력입니다. 전체 크롤링을 진행합니다.")
    return 'all', None

def crawl_daiso_beauty():
    """다이소 뷰티 크롤링 메인 함수"""

    logger.info("=" * 60)
    logger.info("다이소 뷰티 완전판 크롤링 시작")
    logger.info("=" * 60)

    driver = None
    all_products = []

    try:
        # 카테고리 선택
        main_cat, sub_cat = select_categories()

        # 드라이버 생성
        logger.info("Chrome 드라이버 생성 중")
        driver = create_driver()
        logger.info("드라이버 생성")

        max_products = DAISO_CONFIG['max_products_per_category']

        # 크롤링 실행
        if main_cat == 'all':
            # 전체 크롤링
            for main_name, main_data in DAISO_BEAUTY_CATEGORIES.items():
                logger.info(f"\n{'='*20} {main_name} {'='*20}")
                middle_code = main_data['중분류코드']

                for sub_code, sub_name in main_data['소분류'].items():
                    products = crawl_category(
                        driver, main_name, sub_name, middle_code, sub_code, max_products
                    )
                    all_products.extend(products)
                    random_delay(2, 3)

        elif sub_cat == 'all':
            # 특정 대분류의 전체 소분류
            main_data = DAISO_BEAUTY_CATEGORIES[main_cat]
            middle_code = main_data['중분류코드']

            logger.info(f"\n{'='*20} {main_cat} 전체 {'='*20}")
            for sub_code, sub_name in main_data['소분류'].items():
                products = crawl_category(
                    driver, main_cat, sub_name, middle_code, sub_code, max_products
                )
                all_products.extend(products)
                random_delay(2, 3)

        else:
            # 특정 소분류만
            main_data = DAISO_BEAUTY_CATEGORIES[main_cat]
            middle_code = main_data['중분류코드']
            sub_code, sub_name = sub_cat

            logger.info(f"\n{'='*20} {main_cat}/{sub_name} {'='*20}")
            products = crawl_category(
                driver, main_cat, sub_name, middle_code, sub_code, max_products
            )
            all_products.extend(products)

        # 데이터 저장
        if all_products:
            date_str = get_date_string()
            filename = f'daiso_beauty_complete_{date_str}.csv'
            filepath = save_to_csv(all_products, filename)
            logger.info(f"\n데이터 저장 완료: {filepath}")
            logger.info(f"총 {len(all_products)}개 상품 수집")

            # 통계
            stats = {}
            for p in all_products:
                main = p['main_category']
                sub = p['sub_category']
                key = f"{main}/{sub}" if sub else main
                stats[key] = stats.get(key, 0) + 1

            logger.info("\n=== 카테고리별 수집 현황 ===")
            for cat, count in sorted(stats.items()):
                logger.info(f"  - {cat}: {count}개")

            print(f"\n크롤링 완료! 총 {len(all_products)}개 상품 수집")
            print(f"파일: {filepath}")
        else:
            logger.warning("수집된 데이터가 없습니다.")
            print("\n 수집된 데이터가 없습니다.")

    except Exception as e:
        logger.error(f"크롤링 실패: {e}")
        logger.error(traceback.format_exc())
        print(f"\n크롤링 실패: {e}")

    finally:
        if driver:
            logger.info("브라우저 종료 중")
            quit_driver(driver)
            logger.info("브라우저 종료")

    return all_products

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("다이소몰 완전판 크롤러")
    print("=" * 70)
    print("\n전체 카테고리 구조:")
    for main, data in DAISO_BEAUTY_CATEGORIES.items():
        print(f"  • {main} ({len(data['소분류'])}개 소분류)")
    print("\n주의: 교육/연구 목적입니다.")
    print("=" * 70)

    input("\n시작하려면 Enter를 누르세요")

    crawl_daiso_beauty()
