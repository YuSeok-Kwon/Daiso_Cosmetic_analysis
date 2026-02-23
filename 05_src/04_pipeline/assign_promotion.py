"""
프로모션 매칭 스크립트
=====================
reviews_core의 각 리뷰에 해당하는 promotion_id를 할당한다.

매칭 로직:
  1. 브랜드 식별: review → product_code → brand_id
  2. 프로모션 룩업: 동일 brand_id의 프로모션 중
     리뷰 작성일이 프로모션 기간(start_date ~ end_date) ±7일 이내
     - brand_id=95 (ALL)인 프로모션은 모든 브랜드에 매칭
  3. 복수 매칭 시: 프로모션 기간 중심일과 가장 가까운 프로모션 선택
  4. 미매칭: promotion_id = NULL (ERD Optional FK)

사용법:
  python assign_promotion.py                    # 기본 실행
  python assign_promotion.py --buffer 7         # 버퍼 일수 변경
  python assign_promotion.py --dry-run          # 저장 없이 결과만 출력
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path


# ── 설정 ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FINAL_DIR = PROJECT_ROOT / "02_processed_data" / "csv" / "final"
ALL_BRAND_ID = 0  # 전체 브랜드 대상 프로모션의 brand_id
DEFAULT_BUFFER_DAYS = 7


def load_data(final_dir: Path) -> tuple:
    """최종 CSV 파일 로드"""
    reviews = pd.read_csv(final_dir / "reviews_core.csv", parse_dates=["review_date"])
    products = pd.read_csv(final_dir / "products_core.csv")
    promotions = pd.read_csv(
        final_dir / "promotions.csv",
        parse_dates=["start_date", "end_date"],
    )
    return reviews, products, promotions


def assign_promotion(
    reviews: pd.DataFrame,
    products: pd.DataFrame,
    promotions: pd.DataFrame,
    buffer_days: int = DEFAULT_BUFFER_DAYS,
) -> pd.DataFrame:
    """
    리뷰에 promotion_id를 할당한다.

    Parameters
    ----------
    reviews : 리뷰 DataFrame (review_date, product_code 필수)
    products : 상품 DataFrame (product_code, brand_id 필수)
    promotions : 프로모션 DataFrame (promotion_id, brand_id, start_date, end_date 필수)
    buffer_days : 프로모션 기간 전후 버퍼 일수 (기본 7일)

    Returns
    -------
    reviews에 promotion_id 컬럼이 추가된 DataFrame
    """
    buffer = pd.Timedelta(days=buffer_days)

    # 1단계: 리뷰 → 브랜드 매핑
    brand_map = products.set_index("product_code")["brand_id"]
    reviews = reviews.copy()
    reviews["brand_id"] = reviews["product_code"].map(brand_map)

    # 2단계: 프로모션별 매칭 윈도우 계산
    promos = promotions.copy()
    promos["window_start"] = promos["start_date"] - buffer
    promos["window_end"] = promos["end_date"] + buffer
    promos["center_date"] = promos["start_date"] + (promos["end_date"] - promos["start_date"]) / 2

    # 3단계: 브랜드별 + 전체(ALL) 프로모션 매칭
    #   - 프로모션이 16건으로 적으므로 루프 기반이 효율적
    #   - 각 프로모션에 대해 매칭되는 리뷰를 벡터 연산으로 한번에 찾음
    matches = []

    for _, promo in promos.iterrows():
        # 날짜 조건: review_date가 윈도우 안에 있는지
        date_mask = (reviews["review_date"] >= promo["window_start"]) & (
            reviews["review_date"] <= promo["window_end"]
        )

        # 브랜드 조건: 동일 brand_id 또는 ALL(95)
        if promo["brand_id"] == ALL_BRAND_ID:
            brand_mask = pd.Series(True, index=reviews.index)
        else:
            brand_mask = reviews["brand_id"] == promo["brand_id"]

        matched_idx = reviews.index[date_mask & brand_mask]

        if len(matched_idx) > 0:
            # 프로모션 중심일과의 거리 계산 (복수 매칭 시 가까운 것 우선)
            distance = abs((reviews.loc[matched_idx, "review_date"] - promo["center_date"]).dt.days)
            # 브랜드 특정 프로모션(0)을 ALL 프로모션(1)보다 우선
            is_all = 1 if promo["brand_id"] == ALL_BRAND_ID else 0
            match_df = pd.DataFrame(
                {
                    "review_idx": matched_idx,
                    "promotion_id": promo["promotion_id"],
                    "distance": distance.values,
                    "is_all": is_all,
                }
            )
            matches.append(match_df)

    # 4단계: 복수 매칭 해소
    #   우선순위: ① 브랜드 특정 > ALL  ② 거리 가까운 것  ③ promotion_id 작은 것
    if matches:
        all_matches = pd.concat(matches, ignore_index=True)
        best = (
            all_matches.sort_values(["is_all", "distance", "promotion_id"])
            .drop_duplicates("review_idx", keep="first")
            .set_index("review_idx")["promotion_id"]
        )
        reviews["promotion_id"] = best.reindex(reviews.index).astype("Int64")
    else:
        reviews["promotion_id"] = pd.NA
    reviews["promotion_id"] = reviews["promotion_id"].astype("Int64")

    # 임시 컬럼 정리
    reviews.drop(columns=["brand_id"], inplace=True)

    return reviews


def print_summary(reviews: pd.DataFrame, promotions: pd.DataFrame):
    """매칭 결과 요약 출력"""
    total = len(reviews)
    matched = reviews["promotion_id"].notna().sum()
    unmatched = total - matched

    print("=" * 60)
    print("프로모션 매칭 결과")
    print("=" * 60)
    print(f"  전체 리뷰     : {total:>10,}건")
    print(f"  매칭 성공     : {matched:>10,}건 ({matched / total * 100:.2f}%)")
    print(f"  미매칭        : {unmatched:>10,}건 ({unmatched / total * 100:.2f}%)")

    print(f"\n  [프로모션별 매칭 건수]")
    promo_counts = reviews[reviews["promotion_id"].notna()]["promotion_id"].value_counts().sort_index()
    promo_desc = promotions.set_index("promotion_id")["description"]

    for pid, cnt in promo_counts.items():
        desc = promo_desc.get(pid, "?")
        print(f"    {pid:>3d}. {desc:35s} : {cnt:>6,}건")


def main():
    parser = argparse.ArgumentParser(description="리뷰 프로모션 매칭")
    parser.add_argument("--buffer", type=int, default=DEFAULT_BUFFER_DAYS, help="프로모션 기간 전후 버퍼 일수")
    parser.add_argument("--dry-run", action="store_true", help="저장 없이 결과만 출력")
    args = parser.parse_args()

    print(f"설정: buffer={args.buffer}일, dry_run={args.dry_run}")

    reviews, products, promotions = load_data(FINAL_DIR)
    print(f"로드: reviews {len(reviews):,}건, products {len(products):,}건, promotions {len(promotions):,}건")

    reviews = assign_promotion(reviews, products, promotions, buffer_days=args.buffer)

    print_summary(reviews, promotions)

    if not args.dry_run:
        out_path = FINAL_DIR / "reviews_core.csv"
        reviews.to_csv(out_path, index=False)
        print(f"\n  → {out_path} 저장 완료")
    else:
        print("\n  (dry-run: 저장하지 않음)")


if __name__ == "__main__":
    main()
