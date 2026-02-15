"""
제품 정보는 있지만 리뷰가 누락된 제품의 리뷰만 크롤링
- BigQuery 적재 지원
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from daiso_beauty_crawler import crawl_product_detail
from utils import setup_logger, get_date_string
import time

# BigQuery 모듈
try:
    from BigQuery.etl_loader import CrawlerETL
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False

# 로거 설정
logger = setup_logger('missing_reviews', 'missing_reviews.log')

# 리뷰만 필요한 제품 코드 목록 (제품 정보는 있지만 리뷰가 없는 829개)
missing_codes = [31386, 41301, 58830, 58831, 67851, 67853, 84456, 84989, 85092, 1026629, 1026630, 1026631,
 1035071, 1035082, 1035084, 1038226, 1038594, 1039941, 1040079, 1040471, 1040472, 1041677, 1042158, 1042248,
  1042979, 1043646, 1044964, 1046781, 1046782, 1046783, 1046784, 1047511, 1051351, 1051353, 1051730, 1051731,
   1053332, 1053333, 1053454, 1054347, 1056668, 1057314, 1057350, 1058460, 1059097, 1059203, 1059837, 1060063,
    1060212, 1060213, 1060216, 1060511, 1061878, 1061879, 1061880, 1061881, 1061914, 1061915, 1061916, 1061917,
     1062143, 1062155, 1062713, 1062782, 1062783, 1063609, 1065038, 1065039, 1065040, 1067023, 1067024, 1067851,
      1067853, 1068982, 1068983, 1070120, 1070122, 1070124, 1073704, 1073705]

print("=" * 100)
print("누락된 리뷰 크롤링")
print("=" * 100)
print(f"\n대상 제품: {len(missing_codes)}개")
print("크롤링 대상: 리뷰만")
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

all_reviews = []
failed_products = []

# 제품별 크롤링
for idx, product_code in enumerate(missing_codes, 1):
    url = f"https://www.daisomall.co.kr/pd/pdr/SCR_PDR_0001?pdNo={product_code}&recmYn=N"

    print(f"[{idx}/{len(missing_codes)}] 제품 {product_code}", end=" ")
    logger.info(f"제품 {product_code} 리뷰 크롤링 시작")

    try:
        # 리뷰만 크롤링
        product, reviews, ingredients = crawl_product_detail(
            driver, url,
            category_home="뷰티",
            category_1="",
            category_2="",
            crawl_reviews=True,   # 리뷰만 수집
            crawl_ingredients=False  # 성분 제외
        )

        if reviews:
            all_reviews.extend(reviews)
            print(f"{len(reviews)}개 리뷰")
            logger.info(f"  → {len(reviews)}개 리뷰 수집")
        else:
            print(f"리뷰 없음")
            logger.warning(f"  → 리뷰 없음")

    except Exception as e:
        print(f"X")
        logger.error(f"  → 오류: {str(e)}")
        failed_products.append(product_code)

    time.sleep(1.5)

driver.quit()

# 결과 저장
print("\n" + "=" * 100)
print("크롤링 완료 - 결과 저장")
print("=" * 100)

timestamp = get_date_string()

# 리뷰 데이터
if all_reviews:
    reviews_df = pd.DataFrame(all_reviews)

    # mancare_reviews.csv와 동일한 컬럼만 선택: product_code, date, user_masked, rating, text, image_count
    columns_to_keep = ['product_code', 'date', 'user_masked', 'rating', 'text', 'image_count']
    reviews_df = reviews_df[columns_to_keep]

    reviews_path = f'/Users/yu_seok/Documents/workspace/nbCamp/Project/Why-pi/data/missing_reviews_{timestamp}.csv'
    reviews_df.to_csv(reviews_path, index=False, encoding='utf-8-sig')
    print(f"\n 리뷰 데이터: {reviews_path}")
    print(f"   총 {len(reviews_df):,}개 리뷰")

    # 제품별 리뷰 통계
    review_counts = reviews_df.groupby('product_code').size().reset_index(name='review_count')
    print(f"\n 리뷰 수집 통계:")
    print(f"   - 리뷰 있음: {len(review_counts)}개 제품")
    print(f"   - 리뷰 없음: {len(missing_codes) - len(review_counts)}개 제품")
    print(f"   - 평균 리뷰 수: {review_counts['review_count'].mean():.1f}개")
else:
    print("\n  리뷰 데이터 없음")

# 실패 목록
if failed_products:
    print(f"\n 실패한 제품 ({len(failed_products)}개):")
    print(f"   {failed_products[:10]}")
    if len(failed_products) > 10:
        print(f"    외 {len(failed_products) - 10}개")

# 통계
success_count = len(missing_codes) - len(failed_products)
total_count = len(missing_codes)
success_rate = success_count / total_count * 100 if total_count > 0 else 0

print(f"\n최종 통계:")
print(f"   - 전체: {total_count}개")
print(f"   - 성공: {success_count}개 ({success_rate:.1f}%)")
print(f"   - 실패: {len(failed_products)}개 ({100-success_rate:.1f}%)")

# BigQuery 적재
if BIGQUERY_AVAILABLE and all_reviews:
    print("\n" + "=" * 100)
    bq_confirm = input("BigQuery에 적재하시겠습니까? (y/n): ").strip().lower()
    if bq_confirm == 'y':
        try:
            print("BigQuery 적재 시작...")
            etl = CrawlerETL()
            etl.load_reviews(reviews_path)
            print("BigQuery 적재 완료!")
        except Exception as e:
            print(f"BigQuery 적재 실패: {str(e)}")

print("\n" + "=" * 100)
print("완료!")
print("=" * 100)
