"""
Step A: 3단계 층화 샘플링으로 20k 리뷰 추출
- 1단계: 대분류(category_1) 비례 + 최소 보장
- 2단계: 소분류(category_2) 비례 + 최소 보장
- 3단계: sentiment 비율 맞춤 (30/30/40)
"""
import sys
import pandas as pd
from pathlib import Path

# Add RQ_absa to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from RQ_absa.sampling import load_and_sample_reviews
from RQ_absa.config import (
    REVIEWS_CSV_PATH,
    RAW_DATA_DIR,
    SAMPLING_CONFIG
)

# 상품 정보 경로
PRODUCTS_CSV_PATH = REVIEWS_CSV_PATH.parent / "products.csv"


def main():
    """샘플링 메인 함수"""
    print("="*60)
    print("STEP A: 3단계 층화 샘플링 (대분류/소분류/sentiment)")
    print("="*60)

    # 경로
    input_path = REVIEWS_CSV_PATH
    output_path = RAW_DATA_DIR / "sampled_reviews_20k.csv"

    # 입력 파일 확인
    if not input_path.exists():
        print(f"오류: 입력 파일을 찾을 수 없습니다: {input_path}")
        print("\nreviews.csv 파일이 올바른 위치에 있는지 확인하세요.")
        return

    # 리뷰 로드
    print("리뷰 데이터 로드 중...")
    reviews_df = pd.read_csv(input_path)
    print(f"리뷰 수: {len(reviews_df):,}개")

    # 상품 정보 조인
    if PRODUCTS_CSV_PATH.exists():
        print("\n상품 정보 조인 중...")
        products_df = pd.read_csv(PRODUCTS_CSV_PATH)
        print(f"상품 수: {len(products_df):,}개")

        # 필요한 컬럼만 선택하여 조인 (category_2 추가)
        products_cols = ['product_code', 'brand', 'category_1', 'category_2']
        reviews_df = reviews_df.merge(
            products_df[products_cols],
            on='product_code',
            how='left'
        )
        print(f"조인 후 리뷰 수: {len(reviews_df):,}개")

        # 제외 카테고리 필터링
        exclude_cats = SAMPLING_CONFIG.get('exclude_categories', [])
        if exclude_cats:
            before_filter = len(reviews_df)
            reviews_df = reviews_df[~reviews_df['category_1'].isin(exclude_cats)]
            print(f"제외 카테고리 {exclude_cats}: {before_filter:,}개 → {len(reviews_df):,}개")

        # 결측값 확인
        missing_brand = reviews_df['brand'].isna().sum()
        missing_cat1 = reviews_df['category_1'].isna().sum()
        missing_cat2 = reviews_df['category_2'].isna().sum()
        if missing_brand > 0 or missing_cat1 > 0 or missing_cat2 > 0:
            print(f"  - brand 결측: {missing_brand:,}개")
            print(f"  - category_1 결측: {missing_cat1:,}개")
            print(f"  - category_2 결측: {missing_cat2:,}개")
            reviews_df['brand'] = reviews_df['brand'].fillna('Unknown')
            reviews_df['category_1'] = reviews_df['category_1'].fillna('Unknown')
            reviews_df['category_2'] = reviews_df['category_2'].fillna('Unknown')

        # 계층화 컬럼 설정
        stratify_columns = ['category_1', 'category_2']
        print(f"\n계층화 컬럼: {stratify_columns}")
    else:
        print(f"\n경고: products.csv를 찾을 수 없습니다: {PRODUCTS_CSV_PATH}")
        print("product_code로만 계층화합니다.")
        stratify_columns = ['product_code']

    # 임시 파일로 저장 (조인된 데이터)
    temp_path = RAW_DATA_DIR / "temp_reviews_with_products.csv"
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    reviews_df.to_csv(temp_path, index=False, encoding='utf-8-sig')

    # 샘플링 실행 (자연 분포 기반 층화 샘플링)
    sampled_df = load_and_sample_reviews(
        input_path=temp_path,
        output_path=output_path,
        target_size=SAMPLING_CONFIG['target_size'],
        category_1_column=SAMPLING_CONFIG['category_1_column'],
        category_2_column=SAMPLING_CONFIG['category_2_column'],
        category_1_min_floor=SAMPLING_CONFIG['category_1_min_floor'],
        category_2_min_floor=SAMPLING_CONFIG['category_2_min_floor'],
        skip_cat2_categories=SAMPLING_CONFIG.get('skip_cat2_categories', []),
        target_sentiment_distribution=SAMPLING_CONFIG['sentiment_distribution'],
        random_state=SAMPLING_CONFIG['random_state']
    )

    # 임시 파일 삭제
    temp_path.unlink(missing_ok=True)

    # brand, category_1 컬럼도 최종 출력에서 제거 (필요시 주석 해제)
    # sampled_df = sampled_df.drop(columns=['brand', 'category_1'], errors='ignore')
    # sampled_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("\n" + "="*60)
    print("STEP A 완료")
    print("="*60)
    print(f"샘플링된 리뷰: {len(sampled_df):,}개")
    print(f"출력 파일: {output_path}")
    print("\n다음 단계: Step B 실행")
    print("  /opt/miniconda3/envs/py_study/bin/python scripts/step_b_labeling.py")
    print("="*60)


if __name__ == "__main__":
    main()
