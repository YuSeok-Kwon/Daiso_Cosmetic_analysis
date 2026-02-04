"""
리뷰는 있지만 제품 정보가 없는 제품 크롤링
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from daiso_beauty_crawler import crawl_product_detail
from utils import setup_logger, get_date_string
import time

# 로거 설정
logger = setup_logger('missing_products', 'missing_products.log')

# 누락된 제품 코드 목록 (리뷰는 있지만 제품 정보가 없음)
missing_codes = [
    31389, 44527, 69036, 90196, 1007503, 1010710, 1016426,
    1018260, 1023480, 1025659, 1035402, 1039480, 1039628, 1041093,
    1041403, 1041635, 1041672, 1041749, 1044556, 1045846, 1045847,
    1047194, 1047495, 1048570, 1051748, 1051750, 1053457, 1053482,
    1055169, 1056647, 1057229, 1057230, 1059486, 1060644, 1061696,
    1061701, 1061919, 1062703, 1062780, 1064412, 1064413, 1065368,
    1065374, 1066897, 1066898, 1066900, 1067401, 1067402, 1067403,
    1067525, 1067532, 1069014, 1070665, 1071645, 1072517, 1072519,
    1072806, 1073012, 1074271, 1075032, 1075034
]

print("=" * 100)
print("누락된 제품 정보 크롤링")
print("=" * 100)
print(f"\n대상 제품: {len(missing_codes)}개")
print("크롤링 대상: 제품 정보만")
print("=" * 100)

# 자동 시작
print("\n크롤링을 시작합니다")

# Chrome 옵션
chrome_options = Options()
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

# 드라이버 초기화
driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=chrome_options
)

all_products = []
failed_products = []

# 제품별 크롤링
for idx, product_code in enumerate(missing_codes, 1):
    url = f"https://www.daisomall.co.kr/pd/pdr/SCR_PDR_0001?pdNo={product_code}&recmYn=N"

    print(f"\n[{idx}/{len(missing_codes)}] 제품 {product_code}", end=" ")
    logger.info(f"제품 {product_code} 크롤링 시작")

    try:
        # 제품 정보만 크롤링 (리뷰, 성분 제외)
        product, reviews, ingredients = crawl_product_detail(
            driver, url,
            category_home="뷰티",  # 기본값
            category_1="",  # 페이지에서 추출
            category_2="",  # 페이지에서 추출
            crawl_reviews=False,  # 리뷰는 이미 있음
            crawl_ingredients=False  # 성분 제외
        )

        if product:
            all_products.append(product)
            print(f"성공")
            logger.info(f"  → 성공: 제품 정보")
        else:
            print(f"실패")
            logger.warning(f"  → 제품 정보 추출 실패")
            failed_products.append(product_code)

    except Exception as e:
        print(f"오류")
        logger.error(f"  → 오류: {str(e)}")
        failed_products.append(product_code)

    time.sleep(2)

driver.quit()

# 결과 저장
print("\n" + "=" * 100)
print("크롤링 완료 - 결과 저장")
print("=" * 100)

timestamp = get_date_string()

# 제품 데이터
if all_products:
    products_df = pd.DataFrame(all_products)
    products_path = f'/Users/yu_seok/Documents/workspace/nbCamp/Project/Why-pi/data/missing_products_{timestamp}.csv'
    products_df.to_csv(products_path, index=False, encoding='utf-8-sig')
    print(f"\n제품 데이터: {products_path}")
    print(f"   총 {len(products_df)}개 제품")
else:
    print("\n제품 데이터 없음")

# 실패 목록
if failed_products:
    print(f"\n실패한 제품 ({len(failed_products)}개):")
    print(f"   {failed_products[:10]}")
    if len(failed_products) > 10:
        print(f"외 {len(failed_products) - 10}개")

    # 실패 목록 저장
    failed_df = pd.DataFrame({'product_code': failed_products})
    failed_path = f'/Users/yu_seok/Documents/workspace/nbCamp/Project/Why-pi/data/failed_products_{timestamp}.csv'
    failed_df.to_csv(failed_path, index=False, encoding='utf-8-sig')
    print(f"   저장: {failed_path}")

# 통계
success_count = len(all_products)
total_count = len(missing_codes)
success_rate = success_count / total_count * 100 if total_count > 0 else 0

print(f"\n최종 통계:")
print(f"   - 전체: {total_count}개")
print(f"   - 성공: {success_count}개 ({success_rate:.1f}%)")
print(f"   - 실패: {len(failed_products)}개 ({100-success_rate:.1f}%)")

print("\n" + "=" * 100)
print("완료!")
print("=" * 100)
