"""
다이소몰 제품 상세정보에서 전성분 추출 크롤러
이미지 alt 속성에서 [전성분] 정보를 파싱합니다.
"""
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from driver_setup import create_driver, quit_driver
from utils import *
from config import DAISO_CONFIG
import traceback
import re
import pandas as pd

logger = setup_logger('daiso_ingredients', 'daiso_ingredients.log')


def clean_ingredient(ingredient_text):
    """
    성분 텍스트 정제

    Args:
        ingredient_text: 원본 성분 텍스트

    Returns:
        str: 정제된 성분명 (유효하지 않으면 빈 문자열)
    """
    if not ingredient_text:
        return ''

    # 특수문자 제거 (⌀, ×, 등)
    cleaned = re.sub(r'[⌀×][\d\w\s]*', '', ingredient_text)

    # 괄호 및 내용 제거
    cleaned = re.sub(r'\(.*?\)', '', cleaned)
    cleaned = re.sub(r'\[.*?\]', '', cleaned)

    # 숫자+단위 패턴 제거 (예: "10mm", "30ft")
    cleaned = re.sub(r'\d+\s*(mm|ft|ml|g|kg|%)', '', cleaned)

    # 불필요한 문구 제거
    exclude_patterns = [
        r'있는\s*경우',
        r'전문의',
        r'상처가\s*있는\s*부위',
        r'사용.*?할\s*것',
        r'보관.*?것',
        r'어린',
        r'눈에',
        r'곳에',
        r'본\s*상품',
        r'협력사',
        r'등록한',
        r'특성상',
        r'색상이',
        r'없습니다',
        r'출장샵',
        r'콜걸',
        r'언급',
        r'최강',
        r'가족관계',
        r'首位',
        r'百货',
        r'備',
        r'아이\s*:',
        r'메틸아이',
    ]

    for pattern in exclude_patterns:
        if re.search(pattern, cleaned):
            return ''

    # 공백 정리
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # 최소 길이 체크
    if len(cleaned) < 2:
        return ''

    # 너무 긴 경우 (100자 이상) - 오류일 가능성
    if len(cleaned) > 100:
        return ''

    # 숫자만 있는 경우 제외
    if cleaned.isdigit():
        return ''

    return cleaned


def extract_ingredients_from_alt(alt_text):
    """
    이미지 alt 텍스트에서 전성분 정보를 추출

    Args:
        alt_text: 이미지의 alt 속성 텍스트

    Returns:
        dict: {
            'ingredients': 전성분 리스트,
            'product_name': 제품명,
            'volume': 용량,
            'manufacturer': 제조업자,
            'raw_ingredients': 원본 전성분 텍스트
        }
    """
    result = {
        'ingredients': [],
        'product_name': '',
        'volume': '',
        'manufacturer': '',
        'raw_ingredients': ''
    }

    if not alt_text:
        return result

    # [전성분] 섹션 추출
    ingredients_pattern = r'\[전성분\](.*?)(?:\[|$)'
    ingredients_match = re.search(ingredients_pattern, alt_text, re.DOTALL)

    if ingredients_match:
        ingredients_section = ingredients_match.group(1).strip()
        result['raw_ingredients'] = ingredients_section

        # 제품 변형 찾기:
        # 패턴 1: "제품명:" (콜론 있음)
        # 패턴 2: "제품명 성분1, 성분2" (첫 단어가 제품명)

        # 콜론으로 구분되는 변형 찾기
        variant_pattern = r'([가-힣\s]+[가-힣]|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*:\s*'
        variants = list(re.finditer(variant_pattern, ingredients_section))

        if variants:
            # 변형이 있는 경우
            # 첫 번째 변형 전의 텍스트 처리 (콜론 없는 첫 제품)
            first_variant_pos = variants[0].start()
            if first_variant_pos > 0:
                pre_text = ingredients_section[:first_variant_pos].strip()
                # 첫 단어를 제품명으로 추출
                words = pre_text.split()
                if words:
                    first_product_name = words[0]
                    remaining_text = ' '.join(words[1:])

                    # 쉼표로 성분 분리
                    ingredients_list = [ing.strip() for ing in remaining_text.split(',')]

                    for ing in ingredients_list:
                        cleaned_ing = clean_ingredient(ing)
                        if cleaned_ing:
                            result['ingredients'].append({
                                'product_variant': first_product_name,
                                'ingredient': cleaned_ing
                            })

            # 콜론이 있는 변형들 처리
            for i, match in enumerate(variants):
                variant_name = match.group(1).strip()
                start_pos = match.end()
                end_pos = variants[i+1].start() if i+1 < len(variants) else len(ingredients_section)

                variant_ingredients_text = ingredients_section[start_pos:end_pos].strip()

                # 쉼표로 성분 분리
                ingredients_list = [ing.strip() for ing in variant_ingredients_text.split(',')]

                for ing in ingredients_list:
                    cleaned_ing = clean_ingredient(ing)
                    if cleaned_ing:
                        result['ingredients'].append({
                            'product_variant': variant_name,
                            'ingredient': cleaned_ing
                        })
        else:
            # 변형이 없는 경우 (단일 제품)
            # 첫 단어를 제품명으로 시도
            words = ingredients_section.split()
            if len(words) > 2:
                first_word = words[0]
                # 첫 단어가 한글이고 짧으면 제품명으로 간주
                if re.match(r'[가-힣]+$', first_word) and len(first_word) <= 10:
                    remaining_text = ' '.join(words[1:])
                    ingredients_list = [ing.strip() for ing in remaining_text.split(',')]

                    for ing in ingredients_list:
                        cleaned_ing = clean_ingredient(ing)
                        if cleaned_ing:
                            result['ingredients'].append({
                                'product_variant': first_word,
                                'ingredient': cleaned_ing
                            })
                else:
                    # 첫 단어가 제품명이 아니면 변형 없음
                    ingredients_list = [ing.strip() for ing in ingredients_section.split(',')]

                    for ing in ingredients_list:
                        cleaned_ing = clean_ingredient(ing)
                        if cleaned_ing:
                            result['ingredients'].append({
                                'product_variant': '',
                                'ingredient': cleaned_ing
                            })
            else:
                # 너무 짧은 경우
                ingredients_list = [ing.strip() for ing in ingredients_section.split(',')]

                for ing in ingredients_list:
                    cleaned_ing = clean_ingredient(ing)
                    if cleaned_ing:
                        result['ingredients'].append({
                            'product_variant': '',
                            'ingredient': cleaned_ing
                        })

    # [품명] 추출
    name_pattern = r'\[품명\]\s*(.*?)(?:\[|$)'
    name_match = re.search(name_pattern, alt_text)
    if name_match:
        result['product_name'] = name_match.group(1).strip()

    # [용량] 추출
    volume_pattern = r'\[용량\]\s*(.*?)(?:\[|$)'
    volume_match = re.search(volume_pattern, alt_text)
    if volume_match:
        result['volume'] = volume_match.group(1).strip()

    # [화장품제조업자 및 화장품책임판매업자] 추출
    manufacturer_pattern = r'\[화장품제조업자 및 화장품책임판매업자\]\s*(.*?)(?:\[|$)'
    manufacturer_match = re.search(manufacturer_pattern, alt_text)
    if manufacturer_match:
        result['manufacturer'] = manufacturer_match.group(1).strip()

    return result


def crawl_product_details(driver, product_url, product_id):
    """
    제품 상세 페이지에서 전성분 정보 크롤링

    Args:
        driver: Selenium WebDriver
        product_url: 제품 상세 페이지 URL
        product_id: 제품 고유 ID

    Returns:
        dict: 제품 상세 정보 및 전성분
    """
    product_data = {
        'product_id': product_id,
        'product_url': product_url,
        'crawled_at': get_timestamp(),
        'status': 'failed'
    }

    try:
        logger.info(f"제품 상세 페이지 접속: {product_url}")
        driver.get(product_url)
        random_delay(2, 3)

        # 페이지 로딩 대기
        wait = WebDriverWait(driver, 10)

        # 상세정보 영역 찾기
        detail_selectors = [
            '.editor-content',
            'div[class*="editor"]',
            'div[class*="detail"]',
            '.product-detail-content'
        ]

        detail_section = None
        for selector in detail_selectors:
            try:
                detail_section = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                logger.info(f"상세정보 영역 발견: {selector}")
                break
            except:
                continue

        if not detail_section:
            logger.warning("상세정보 영역을 찾을 수 없습니다.")
            return product_data

        # 이미지 요소들 찾기
        img_elements = detail_section.find_elements(By.TAG_NAME, 'img')
        logger.info(f"이미지 {len(img_elements)}개 발견")

        all_ingredients = []

        for idx, img in enumerate(img_elements):
            try:
                alt_text = img.get_attribute('alt')

                if alt_text and '[전성분]' in alt_text:
                    logger.info(f"전성분 정보 발견 (이미지 #{idx+1})")

                    # 전성분 추출
                    ingredients_data = extract_ingredients_from_alt(alt_text)

                    if ingredients_data['ingredients']:
                        product_data.update({
                            'product_name': ingredients_data['product_name'],
                            'volume': ingredients_data['volume'],
                            'manufacturer': ingredients_data['manufacturer'],
                            'raw_ingredients': ingredients_data['raw_ingredients'],
                            'status': 'success'
                        })

                        all_ingredients.extend(ingredients_data['ingredients'])
                        logger.info(f"전성분 {len(ingredients_data['ingredients'])}개 추출")

            except Exception as e:
                logger.warning(f"이미지 #{idx+1} 처리 실패: {str(e)}")
                continue

        if all_ingredients:
            product_data['ingredients'] = all_ingredients
            logger.info(f"총 {len(all_ingredients)}개 성분 추출 완료")
        else:
            logger.warning("전성분 정보를 찾을 수 없습니다.")

    except Exception as e:
        logger.error(f"제품 상세 크롤링 실패: {e}")
        logger.error(traceback.format_exc())

    return product_data


def process_products_csv(input_csv, max_products=None):
    """
    기존 크롤링 CSV에서 product_url을 읽어 전성분 추가 크롤링

    Args:
        input_csv: 입력 CSV 파일 경로
        max_products: 최대 처리 제품 수 (None이면 전체)

    Returns:
        list: 전성분이 추가된 제품 데이터 리스트
    """
    logger.info("=" * 60)
    logger.info("다이소 전성분 크롤링 시작")
    logger.info("=" * 60)

    driver = None
    results = []

    try:
        # CSV 읽기
        logger.info(f"CSV 파일 로드: {input_csv}")
        df = pd.read_csv(input_csv)
        logger.info(f"총 {len(df)}개 제품 발견")

        # product_url 컬럼 확인
        if 'product_url' not in df.columns:
            logger.error("CSV에 'product_url' 컬럼이 없습니다.")
            return results

        # URL이 있는 제품만 필터링
        df = df[df['product_url'].notna() & (df['product_url'] != '')]
        logger.info(f"URL이 있는 제품: {len(df)}개")

        if max_products:
            df = df.head(max_products)
            logger.info(f"최대 {max_products}개로 제한")

        # 드라이버 생성
        logger.info("Chrome 드라이버 생성 중")
        driver = create_driver()
        logger.info("드라이버 생성 완료")

        # 각 제품 처리
        for idx, row in df.iterrows():
            try:
                product_id = row.get('product_name', f'product_{idx}')
                product_url = row['product_url']

                logger.info(f"\n[{idx+1}/{len(df)}] {product_id}")

                # 상세 페이지 크롤링
                product_data = crawl_product_details(driver, product_url, product_id)

                # 기존 데이터 병합
                product_data.update({
                    'site': row.get('site', '다이소'),
                    'brand': row.get('brand', '다이소'),
                    'main_category': row.get('main_category', ''),
                    'sub_category': row.get('sub_category', ''),
                    'price': row.get('price'),
                    'rating': row.get('rating'),
                    'review_count': row.get('review_count'),
                    'image_url': row.get('image_url', '')
                })

                results.append(product_data)

                # 딜레이
                random_delay(2, 4)

            except Exception as e:
                logger.error(f"제품 처리 실패 [{idx+1}]: {str(e)}")
                continue

        logger.info(f"\n크롤링 완료: 총 {len(results)}개 제품 처리")

        # 결과 저장
        if results:
            # 전성분이 있는 제품만 필터링
            products_with_ingredients = [
                p for p in results if p.get('status') == 'success'
            ]

            logger.info(f"전성분 추출 성공: {len(products_with_ingredients)}개")
            logger.info(f"전성분 추출 실패: {len(results) - len(products_with_ingredients)}개")

            # CSV 저장 (제품 기본 정보)
            date_str = get_date_string()
            products_filename = f'daiso_products_with_ingredients_{date_str}.csv'

            products_df = pd.DataFrame([
                {
                    'product_id': p['product_id'],
                    'product_name': p.get('product_name', ''),
                    'volume': p.get('volume', ''),
                    'manufacturer': p.get('manufacturer', ''),
                    'site': p.get('site', ''),
                    'brand': p.get('brand', ''),
                    'main_category': p.get('main_category', ''),
                    'sub_category': p.get('sub_category', ''),
                    'price': p.get('price'),
                    'rating': p.get('rating'),
                    'review_count': p.get('review_count'),
                    'product_url': p.get('product_url', ''),
                    'image_url': p.get('image_url', ''),
                    'raw_ingredients': p.get('raw_ingredients', ''),
                    'status': p.get('status', 'failed'),
                    'crawled_at': p.get('crawled_at', '')
                }
                for p in results
            ])

            products_filepath = save_to_csv(products_df.to_dict('records'), products_filename)
            logger.info(f"\n제품 데이터 저장: {products_filepath}")

            # 전성분 상세 CSV 저장
            ingredients_records = []
            for p in products_with_ingredients:
                product_id = p['product_id']
                product_name = p.get('product_name', '')

                for ing_data in p.get('ingredients', []):
                    ingredients_records.append({
                        'product_id': product_id,
                        'product_name': product_name,
                        'product_variant': ing_data.get('product_variant', ''),
                        'ingredient': ing_data['ingredient'],
                        'crawled_at': p['crawled_at']
                    })

            if ingredients_records:
                ingredients_filename = f'daiso_ingredients_{date_str}.csv'
                ingredients_filepath = save_to_csv(ingredients_records, ingredients_filename)
                logger.info(f"전성분 데이터 저장: {ingredients_filepath}")
                logger.info(f"총 {len(ingredients_records)}개 성분 저장")

                # 통계
                print(f"\n" + "=" * 60)
                print(f"크롤링 완료!")
                print(f"=" * 60)
                print(f"처리 제품: {len(results)}개")
                print(f"전성분 추출 성공: {len(products_with_ingredients)}개")
                print(f"추출 성분 수: {len(ingredients_records)}개")
                print(f"\n저장 파일:")
                print(f"  - 제품 정보: {products_filepath}")
                print(f"  - 전성분 정보: {ingredients_filepath}")
                print("=" * 60)
            else:
                logger.warning("추출된 전성분이 없습니다.")

    except Exception as e:
        logger.error(f"크롤링 실패: {e}")
        logger.error(traceback.format_exc())
        print(f"\n크롤링 실패: {e}")

    finally:
        if driver:
            logger.info("브라우저 종료 중")
            quit_driver(driver)
            logger.info("브라우저 종료 완료")

    return results


def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("다이소몰 전성분 추출 크롤러")
    print("=" * 70)
    print("\n이 크롤러는 기존 제품 CSV에서 product_url을 읽어")
    print("각 제품 상세 페이지의 이미지 alt 속성에서 전성분을 추출합니다.")
    print("\n주의: 교육/연구 목적으로만 사용하세요.")
    print("=" * 70)

    # CSV 파일 경로 입력
    input_csv = input("\n입력 CSV 파일 경로: ").strip()

    if not input_csv:
        print("파일 경로가 입력되지 않았습니다.")
        return

    # 최대 제품 수 입력
    max_input = input("최대 크롤링 제품 수 (Enter=전체): ").strip()
    max_products = int(max_input) if max_input else None

    print(f"\n시작합니다...")
    process_products_csv(input_csv, max_products)


if __name__ == "__main__":
    main()
