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
missing_codes = [31386, 41301, 58830, 58831, 67851, 67853, 84456, 84989, 85092, 1026629, 1026630, 1026631,
 1035071, 1035082, 1035084, 1038226, 1038594, 1039941, 1040079, 1040471, 1040472, 1041677, 1042158, 1042248,
  1042979, 1043646, 1044964, 1046781, 1046782, 1046783, 1046784, 1047511, 1051351, 1051353, 1051730, 1051731,
   1053332, 1053333, 1053454, 1054347, 1056668, 1057314, 1057350, 1058460, 1059097, 1059203, 1059837, 1060063,
    1060212, 1060213, 1060216, 1060511, 1061878, 1061879, 1061880, 1061881, 1061914, 1061915, 1061916, 1061917,
     1062143, 1062155, 1062713, 1062782, 1062783, 1063609, 1065038, 1065039, 1065040, 1067023, 1067024, 1067851,
      1067853, 1068982, 1068983, 1070120, 1070122, 1070124, 1073704, 1073705]

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
