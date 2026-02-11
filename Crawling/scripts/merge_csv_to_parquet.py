"""
CSV 컬럼을 Parquet 파일에 분배하여 추가하는 스크립트

- products_with_features.parquet에 6개 컬럼 추가 (제품 수준)
- reviews_with_features.parquet에 3개 컬럼 추가 (리뷰 수준)
"""

import pandas as pd
from pathlib import Path


def main():
    # 경로 설정
    base_path = Path(__file__).parent.parent
    csv_path = base_path / "data" / "csv" / "functional.csv"
    products_path = base_path / "data" / "products_with_features.parquet"
    reviews_path = base_path / "data" / "reviews_with_features.parquet"

    output_products_path = base_path / "data" / "products_with_features_v2.parquet"
    output_reviews_path = base_path / "data" / "reviews_with_features_v2.parquet"

    print("=" * 60)
    print("CSV → Parquet 컬럼 병합 스크립트")
    print("=" * 60)

    # 1. 파일 로드
    print("\n[1/4] 파일 로드 중...")
    df_csv = pd.read_csv(csv_path, low_memory=False)
    df_products = pd.read_parquet(products_path)
    df_reviews = pd.read_parquet(reviews_path)

    print(f"  - CSV: {len(df_csv):,}행, {len(df_csv.columns)}컬럼")
    print(f"  - Products: {len(df_products):,}행, {len(df_products.columns)}컬럼")
    print(f"  - Reviews: {len(df_reviews):,}행, {len(df_reviews.columns)}컬럼")

    # 2. Products 처리 (제품 수준 컬럼 추가)
    print("\n[2/4] Products 처리 중...")

    product_cols = ['product_code', 'functional', 'ENTP_NAME', 'ITEM_PH',
                    'whitening', 'wrinkle_reduction', 'sunscreen']

    # product_code별 첫 번째 값 사용
    df_product_cols = df_csv[product_cols].drop_duplicates(subset=['product_code'], keep='first')

    print(f"  - CSV에서 추출한 고유 product_code 수: {len(df_product_cols):,}")

    # Left join
    df_products_v2 = df_products.merge(
        df_product_cols,
        on='product_code',
        how='left'
    )

    matched_products = df_products_v2['functional'].notna().sum()
    print(f"  - 매칭된 제품 수: {matched_products:,} / {len(df_products):,}")

    # 3. Reviews 처리 (리뷰 수준 컬럼 추가)
    print("\n[3/4] Reviews 처리 중...")

    review_cols = ['order_id', 'description', 'event_type', 'promotion_yn']

    # order_id로 매칭
    df_review_cols = df_csv[review_cols].copy()

    # order_id 중복 제거 및 타입 변환 (float -> int)
    df_review_cols = df_review_cols.dropna(subset=['order_id'])
    df_review_cols['order_id'] = df_review_cols['order_id'].astype(int)
    df_review_cols_dedup = df_review_cols.drop_duplicates(subset=['order_id'], keep='first')
    print(f"  - CSV에서 추출한 고유 order_id 수: {len(df_review_cols_dedup):,}")

    # Left join (order_id 사용)
    join_keys = ['order_id']
    new_cols = ['description', 'event_type', 'promotion_yn']

    df_reviews_v2 = df_reviews.merge(
        df_review_cols_dedup[join_keys + new_cols],
        on=join_keys,
        how='left'
    )

    matched_reviews = df_reviews_v2['description'].notna().sum()
    print(f"  - 매칭된 리뷰 수: {matched_reviews:,} / {len(df_reviews):,}")

    # 4. 결과 저장
    print("\n[4/4] 결과 저장 중...")

    df_products_v2.to_parquet(output_products_path, index=False)
    df_reviews_v2.to_parquet(output_reviews_path, index=False)

    print(f"  - 저장 완료: {output_products_path.name}")
    print(f"  - 저장 완료: {output_reviews_path.name}")

    # 5. 검증 결과 출력
    print("\n" + "=" * 60)
    print("검증 결과")
    print("=" * 60)

    print(f"\nProducts:")
    print(f"  - 컬럼 수: {len(df_products.columns)} → {len(df_products_v2.columns)} (추가: {len(df_products_v2.columns) - len(df_products.columns)})")
    print(f"  - 행 수: {len(df_products):,} → {len(df_products_v2):,}")
    print(f"  - 새 컬럼: {', '.join(new_cols for new_cols in ['functional', 'ENTP_NAME', 'ITEM_PH', 'whitening', 'wrinkle_reduction', 'sunscreen'])}")

    print(f"\nReviews:")
    print(f"  - 컬럼 수: {len(df_reviews.columns)} → {len(df_reviews_v2.columns)} (추가: {len(df_reviews_v2.columns) - len(df_reviews.columns)})")
    print(f"  - 행 수: {len(df_reviews):,} → {len(df_reviews_v2):,}")
    print(f"  - 새 컬럼: {', '.join(new_cols)}")

    # 새 컬럼 결측치 확인
    print("\n새 컬럼 결측치 현황:")
    print("\n  Products:")
    for col in ['functional', 'ENTP_NAME', 'ITEM_PH', 'whitening', 'wrinkle_reduction', 'sunscreen']:
        null_pct = df_products_v2[col].isna().mean() * 100
        print(f"    - {col}: {null_pct:.1f}%")

    print("\n  Reviews:")
    for col in new_cols:
        null_pct = df_reviews_v2[col].isna().mean() * 100
        print(f"    - {col}: {null_pct:.1f}%")

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
