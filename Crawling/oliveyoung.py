"""
올리브영 크롤러
"""
from selenium.webdriver.common.by import By
from driver_setup import create_driver, quit_driver
from utils import *
from config import OLIVEYOUNG_CONFIG, OLIVEYOUNG_CATEGORIES
from tqdm import tqdm
import traceback

logger = setup_logger('oliveyoung_complete', 'oliveyoung_complete.log')

def build_ranking_url(category_code, page_idx=1, rows_per_page=24):
    """베스트 랭킹 URL 생성"""
    base = OLIVEYOUNG_CONFIG['base_url']
    ranking_path = OLIVEYOUNG_CONFIG['ranking_url']
    disp_cat_no = OLIVEYOUNG_CONFIG['ranking_disp_cat_no']

    return (f"{base}{ranking_path}?"
            f"dispCatNo={disp_cat_no}"
            f"&fltDispCatNo={category_code}"
            f"&pageIdx={page_idx}"
            f"&rowsPerPage={rows_per_page}")

def crawl_oliveyoung_category(driver, category_name, category_code, max_products=100):
    """올리브영 카테고리별 베스트 상품 크롤링"""

    products = []
    rows_per_page = OLIVEYOUNG_CONFIG['rows_per_page']
    max_pages = (max_products + rows_per_page - 1) // rows_per_page  # 올림 계산

    try:
        for page_idx in range(1, max_pages + 1):
            url = build_ranking_url(category_code, page_idx, rows_per_page)

            logger.info(f"페이지 접속: {category_name} - 페이지 {page_idx}/{max_pages}")
            logger.info(f"URL: {url}")

            driver.get(url)
            random_delay(3, 5)

            # 페이지 스크롤
            scroll_page(driver, scroll_pause=1, max_scrolls=5)

            # 상품 요소 찾기
            selectors = [
                '.prd_info',
                '.prod-list li',
                '.cate_prd_list li',
                'li[class*="prod"]',
            ]

            product_elements = []
            for selector in selectors:
                product_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if len(product_elements) > 0:
                    if page_idx == 1:  # 첫 페이지에서만 로그
                        logger.info(f"선택자 '{selector}' 사용, {len(product_elements)}개 발견")
                    break

            if len(product_elements) == 0:
                logger.warning(f"페이지 {page_idx}에서 상품을 찾을 수 없습니다.")
                break

            # 상품 데이터 추출
            for idx, elem in enumerate(tqdm(product_elements, desc=f"{category_name[:15]} P{page_idx}")):
                try:
                    # 남은 수집 개수 확인
                    if len(products) >= max_products:
                        break

                    driver.execute_script("arguments[0].scrollIntoView(true);", elem)
                    random_delay(0.3, 0.7)

                    product_data = {
                        'site': '올리브영',
                        'category': category_name,
                        'crawled_at': get_timestamp(),
                    }

                    # 상품명
                    try:
                        name_elem = elem.find_element(By.CSS_SELECTOR, '.prd_name, .prod_name, p[class*="name"]')
                        product_data['product_name'] = name_elem.text.strip()
                    except:
                        product_data['product_name'] = ''

                    # 브랜드
                    try:
                        brand_elem = elem.find_element(By.CSS_SELECTOR, '.prd_brand, .prod_brand, span[class*="brand"]')
                        product_data['brand'] = brand_elem.text.strip()
                    except:
                        product_data['brand'] = ''

                    # 가격
                    try:
                        price_elem = elem.find_element(By.CSS_SELECTOR, '.prd_price, .prod_price, .price, span[class*="price"]')
                        price_text = price_elem.text
                        product_data['price'] = extract_price(price_text)
                    except:
                        product_data['price'] = None

                    # 원가
                    try:
                        original_elem = elem.find_element(By.CSS_SELECTOR, '.prd_price_org, .price_org, del')
                        product_data['original_price'] = extract_price(original_elem.text)
                    except:
                        product_data['original_price'] = None

                    # 할인율
                    try:
                        discount_elem = elem.find_element(By.CSS_SELECTOR, '.prd_discount, .discount_rate, span[class*="discount"]')
                        product_data['discount_rate'] = discount_elem.text.strip()
                    except:
                        product_data['discount_rate'] = ''

                    # 평점
                    try:
                        rating_elem = elem.find_element(By.CSS_SELECTOR, '.prd_rating, .rating, span[class*="rating"]')
                        rating_text = rating_elem.text.strip()
                        product_data['rating'] = extract_rating(rating_text)
                    except:
                        product_data['rating'] = None

                    # 리뷰 수
                    try:
                        review_elem = elem.find_element(By.CSS_SELECTOR, '.prd_review_count, .reviewCount, span[class*="review"]')
                        product_data['review_count'] = extract_review_count(review_elem.text)
                    except:
                        product_data['review_count'] = 0

                    # 이미지 URL
                    try:
                        img_elem = elem.find_element(By.CSS_SELECTOR, 'img')
                        product_data['image_url'] = img_elem.get_attribute('src')
                    except:
                        product_data['image_url'] = ''

                    # 상품 URL
                    try:
                        link_elem = elem.find_element(By.CSS_SELECTOR, 'a')
                        href = link_elem.get_attribute('href')
                        if href and not href.startswith('http'):
                            href = OLIVEYOUNG_CONFIG['base_url'] + href
                        product_data['product_url'] = href
                    except:
                        product_data['product_url'] = ''

                    # 데이터 추가
                    if product_data['product_name']:
                        products.append(product_data)

                except Exception as e:
                    logger.warning(f"상품 추출 실패: {str(e)[:100]}")
                    continue

            # 목표 개수 도달 시 중단
            if len(products) >= max_products:
                break

            # 페이지 간 대기
            random_delay(2, 3)

        logger.info(f"카테고리 '{category_name}' 완료: {len(products)}개 수집")

    except Exception as e:
        logger.error(f"카테고리 크롤링 오류: {e}")
        logger.error(traceback.format_exc())

    return products

def select_categories():
    """사용자가 크롤링할 카테고리 선택"""

    print("\n" + "=" * 70)
    print("올리브영 베스트 랭킹 카테고리 선택")
    print("=" * 70)

    # 카테고리를 그룹별로 분류
    beauty_cats = {k: v for k, v in OLIVEYOUNG_CATEGORIES.items() if v.startswith('10000010')}
    health_cats = {k: v for k, v in OLIVEYOUNG_CATEGORIES.items() if v.startswith('10000020')}
    life_cats = {k: v for k, v in OLIVEYOUNG_CATEGORIES.items() if v.startswith('10000030')}

    print("\n뷰티 카테고리:")
    for i, cat in enumerate(beauty_cats.keys(), 1):
        print(f"  {i:2d}. {cat}")

    print("\n헬스 카테고리:")
    for i, cat in enumerate(health_cats.keys(), len(beauty_cats) + 1):
        print(f"  {i:2d}. {cat}")

    print("\n라이프 카테고리:")
    for i, cat in enumerate(life_cats.keys(), len(beauty_cats) + len(health_cats) + 1):
        print(f"  {i:2d}. {cat}")

    print(f"\n  0. 전체 크롤링 ({len(OLIVEYOUNG_CATEGORIES)}개 카테고리)")

    choice = input(f"\n크롤링할 카테고리 번호 (0-{len(OLIVEYOUNG_CATEGORIES)}): ").strip()

    if choice == '0':
        return 'all'

    try:
        idx = int(choice) - 1
        all_cats = list(OLIVEYOUNG_CATEGORIES.keys())
        if 0 <= idx < len(all_cats):
            return all_cats[idx]
    except:
        pass

    print("잘못된 입력입니다. 전체 크롤링을 진행합니다.")
    return 'all'

def crawl_oliveyoung_best():
    """올리브영 베스트 랭킹 크롤링 메인 함수"""

    logger.info("=" * 60)
    logger.info("올리브영 베스트 랭킹 크롤링 시작")
    logger.info("=" * 60)

    driver = None
    all_products = []

    try:
        # 카테고리 선택
        selected = select_categories()

        # 드라이버 생성
        logger.info("Chrome 드라이버 생성 중")
        driver = create_driver()
        logger.info("드라이버 생성")

        max_products = OLIVEYOUNG_CONFIG['max_products_per_category']

        # 크롤링 실행
        if selected == 'all':
            # 전체 크롤링
            for category_name, category_code in OLIVEYOUNG_CATEGORIES.items():
                logger.info(f"\n{'='*20} {category_name} {'='*20}")
                products = crawl_oliveyoung_category(
                    driver, category_name, category_code, max_products
                )
                all_products.extend(products)
                random_delay(3, 5)
        else:
            # 특정 카테고리만
            category_code = OLIVEYOUNG_CATEGORIES[selected]
            logger.info(f"\n{'='*20} {selected} {'='*20}")
            products = crawl_oliveyoung_category(
                driver, selected, category_code, max_products
            )
            all_products.extend(products)

        # 데이터 저장
        if all_products:
            date_str = get_date_string()
            filename = f'oliveyoung_best_{date_str}.csv'
            filepath = save_to_csv(all_products, filename)
            logger.info(f"\n데이터 저장 완료: {filepath}")
            logger.info(f"총 {len(all_products)}개 상품 수집")

            # 통계
            stats = {}
            for p in all_products:
                cat = p['category']
                stats[cat] = stats.get(cat, 0) + 1

            logger.info("\n=== 카테고리별 수집 현황 ===")
            for cat, count in sorted(stats.items()):
                logger.info(f"  - {cat}: {count}개")

            print(f"\n크롤링 완료! 총 {len(all_products)}개 상품 수집")
            print(f"파일: {filepath}")
        else:
            logger.warning("수집된 데이터가 없습니다.")
            print("\n수집된 데이터가 없습니다.")

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
    print("올리브영 베스트 랭킹 크롤러")
    print("=" * 70)
    print(f"\n지원 카테고리: {len(OLIVEYOUNG_CATEGORIES)}개")
    print("  • 뷰티: 12개 (스킨케어, 메이크업, 마스크팩 등)")
    print("  • 헬스: 5개 (건강식품, 푸드, 구강용품 등)")
    print("  • 라이프: 3개 (홈리빙, 취미, 패션)")
    print("  • 주의: 교육/연구 목적입니다.")
    print("=" * 70)

    input("\n시작하려면 Enter를 누르세요")

    crawl_oliveyoung_best()
