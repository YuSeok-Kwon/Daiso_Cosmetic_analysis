"""
운영 스냅샷 빌드 스크립트
  1) 번들 + 결과물을 버전 폴더로 고정
  2) Review-level / Aspect-level Long 테이블 분리
  3) 리뷰 메타(product, brand, category, rating, date) 조인
  4) 상품별 KPI 집계 (8개)
  5) SNAPSHOT.json (체크섬 + 메타 기록)

Usage:
  python build_snapshot.py
"""
import sys
import json
import hashlib
import shutil
import ast
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT.parent.parent / "02_processed_data" / "csv" / "final"

# 입력
BUNDLE_DIR = PROJECT_ROOT / "07_models" / "prod_bundle_stage3a_v1_20260225"
INFERENCE_CSV = PROJECT_ROOT / "04_outputs" / "inference" / "absa_results_stage3a_v2_full.csv"
SUMMARY_JSON = PROJECT_ROOT / "04_outputs" / "inference" / "absa_results_stage3a_v2_summary.json"

# 출력: 버전 폴더
VERSION = "stage3a_v2_20260225"
SNAPSHOT_DIR = PROJECT_ROOT / "04_outputs" / "snapshot" / VERSION
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_review_and_aspect_tables(df: pd.DataFrame):
    """Review-level + Aspect-level Long 테이블 분리."""
    # Aspect-level long
    rows = []
    for idx, row in df.iterrows():
        rid = row["review_id"]
        amb = row["is_ambiguous"]
        asp_list = ast.literal_eval(str(row["aspect_sentiments"]))
        for asp in asp_list:
            rows.append({
                "review_id": rid,
                "aspect": asp["aspect"],
                "aspect_sentiment": asp["sentiment"],
                "is_ambiguous": amb,
            })
    aspect_long = pd.DataFrame(rows)

    # Review-level: JSON 컬럼 제거, 미분류 여부 추가
    review_level = df.drop(columns=["aspect_sentiments"]).copy()
    # 미분류: aspect_labels가 '미분류'이거나, aspect_sentiments에 미분류만 있는 경우
    review_level["is_uncategorized"] = df["aspect_labels"].apply(
        lambda x: str(x).strip() == "미분류" if pd.notna(x) else True
    )

    return review_level, aspect_long


def join_meta(review_df: pd.DataFrame) -> pd.DataFrame:
    """리뷰 메타(product, brand, category, rating, date) 조인."""
    reviews_core = pd.read_csv(DATA_ROOT / "reviews_core.csv",
                               usecols=["review_id", "product_code", "rating", "review_date"])
    products_core = pd.read_csv(DATA_ROOT / "products_core.csv",
                                usecols=["product_code", "brand_id", "name", "price"])
    brands = pd.read_csv(DATA_ROOT / "brands.csv", encoding="cp949")
    categories = pd.read_csv(DATA_ROOT / "products_category.csv",
                             usecols=["product_code", "category_1", "category_2"])

    # review → product
    merged = review_df.merge(reviews_core, on="review_id", how="left")
    # product → brand, category
    merged = merged.merge(products_core, on="product_code", how="left")
    merged = merged.merge(brands, on="brand_id", how="left", suffixes=("", "_brand"))
    merged = merged.merge(categories, on="product_code", how="left")
    merged.rename(columns={"name": "product_name", "name_brand": "brand_name"}, inplace=True)

    return merged


def build_product_kpi(review_df: pd.DataFrame, aspect_long: pd.DataFrame) -> pd.DataFrame:
    """상품별 KPI 집계 (8개)."""
    # n_reviews
    n_reviews = review_df.groupby("product_code").size().reset_index(name="n_reviews")

    # overall pos/neg rate
    sent_agg = review_df.groupby("product_code")["sentiment"].value_counts(normalize=True).unstack(fill_value=0)
    for c in ["positive", "neutral", "negative"]:
        if c not in sent_agg.columns:
            sent_agg[c] = 0.0
    sent_agg = sent_agg.rename(columns={
        "positive": "overall_pos_rate",
        "neutral": "overall_neu_rate",
        "negative": "overall_neg_rate",
    }).reset_index()

    # aspect mention rate (전체 리뷰 중 해당 aspect 언급 비율)
    aspect_with_product = aspect_long.merge(
        review_df[["review_id", "product_code"]].drop_duplicates(), on="review_id", how="left"
    )
    # 미분류 제외
    aspect_real = aspect_with_product[aspect_with_product["aspect"] != "미분류"]

    # aspect별 언급 수
    asp_mention = aspect_real.groupby(["product_code", "aspect"]).size().reset_index(name="mention_count")
    asp_mention = asp_mention.merge(n_reviews, on="product_code", how="left")
    asp_mention["mention_rate"] = asp_mention["mention_count"] / asp_mention["n_reviews"]

    # aspect별 부정 수/비율
    asp_neg = aspect_real[aspect_real["aspect_sentiment"] == "negative"].groupby(
        ["product_code", "aspect"]
    ).size().reset_index(name="neg_count")
    asp_mention = asp_mention.merge(asp_neg, on=["product_code", "aspect"], how="left")
    asp_mention["neg_count"] = asp_mention["neg_count"].fillna(0).astype(int)
    asp_mention["neg_rate"] = asp_mention["neg_count"] / asp_mention["mention_count"]

    # 피벗: aspect별 mention_rate / neg_rate / neg_count
    mention_pivot = asp_mention.pivot_table(
        index="product_code", columns="aspect", values="mention_rate", fill_value=0
    ).add_prefix("mention_rate_")
    neg_rate_pivot = asp_mention.pivot_table(
        index="product_code", columns="aspect", values="neg_rate", fill_value=0
    ).add_prefix("neg_rate_")
    neg_count_pivot = asp_mention.pivot_table(
        index="product_code", columns="aspect", values="neg_count", fill_value=0
    ).add_prefix("neg_count_")

    # 재구매 언급률
    repurchase = aspect_real[aspect_real["aspect"] == "재구매"].groupby("product_code").size().reset_index(name="repurchase_mentions")
    repurchase = repurchase.merge(n_reviews, on="product_code", how="left")
    repurchase["repurchase_mention_rate"] = repurchase["repurchase_mentions"] / repurchase["n_reviews"]

    # ambiguous rate
    amb_rate = review_df.groupby("product_code")["is_ambiguous"].mean().reset_index()
    amb_rate.columns = ["product_code", "ambiguous_rate"]

    # uncategorized rate
    uncl_rate = review_df.groupby("product_code")["is_uncategorized"].mean().reset_index()
    uncl_rate.columns = ["product_code", "uncategorized_rate"]

    # 상품 메타
    products_core = pd.read_csv(DATA_ROOT / "products_core.csv",
                                usecols=["product_code", "brand_id", "name", "price"])
    brands = pd.read_csv(DATA_ROOT / "brands.csv", encoding="cp949")
    categories = pd.read_csv(DATA_ROOT / "products_category.csv",
                             usecols=["product_code", "category_1", "category_2"])

    # 조합
    kpi = n_reviews.copy()
    kpi = kpi.merge(products_core, on="product_code", how="left")
    kpi = kpi.merge(brands, on="brand_id", how="left", suffixes=("", "_brand"))
    kpi.rename(columns={"name": "product_name", "name_brand": "brand_name"}, inplace=True)
    kpi = kpi.merge(categories, on="product_code", how="left")
    kpi = kpi.merge(sent_agg[["product_code", "overall_pos_rate", "overall_neu_rate", "overall_neg_rate"]],
                    on="product_code", how="left")
    kpi = kpi.merge(mention_pivot, on="product_code", how="left")
    kpi = kpi.merge(neg_rate_pivot, on="product_code", how="left")
    kpi = kpi.merge(neg_count_pivot, on="product_code", how="left")
    kpi = kpi.merge(repurchase[["product_code", "repurchase_mention_rate"]], on="product_code", how="left")
    kpi = kpi.merge(amb_rate, on="product_code", how="left")
    kpi = kpi.merge(uncl_rate, on="product_code", how="left")

    kpi = kpi.fillna(0)
    kpi = kpi.sort_values("n_reviews", ascending=False)

    return kpi


def main():
    print("=" * 70)
    print(f"운영 스냅샷 빌드: {VERSION}")
    print("=" * 70)

    # 1) 원본 파일 체크섬
    print("\n[1/5] 원본 체크섬 계산...")
    checksums = {
        "inference_csv": md5(INFERENCE_CSV),
        "summary_json": md5(SUMMARY_JSON),
        "bundle_manifest": md5(BUNDLE_DIR / "MANIFEST.json"),
    }
    for k, v in checksums.items():
        print(f"  {k}: {v}")

    # 2) 원본 복사
    print("\n[2/5] 원본 → 스냅샷 폴더 복사...")
    shutil.copy2(INFERENCE_CSV, SNAPSHOT_DIR / "absa_results_full.csv")
    shutil.copy2(SUMMARY_JSON, SNAPSHOT_DIR / "summary.json")
    print(f"  → {SNAPSHOT_DIR}")

    # 3) 테이블 분리
    print("\n[3/5] Review-level + Aspect-level Long 분리...")
    df = pd.read_csv(INFERENCE_CSV)
    review_level, aspect_long = build_review_and_aspect_tables(df)

    # 메타 조인
    print("  리뷰 메타 조인 (product, brand, category, rating, date)...")
    review_level = join_meta(review_level)

    review_path = SNAPSHOT_DIR / "review_level.csv"
    aspect_path = SNAPSHOT_DIR / "aspect_long.csv"
    review_level.to_csv(review_path, index=False, encoding="utf-8-sig")
    aspect_long.to_csv(aspect_path, index=False, encoding="utf-8-sig")
    print(f"  review_level: {len(review_level):,}행, {len(review_level.columns)}컬럼")
    print(f"  aspect_long:  {len(aspect_long):,}행, {len(aspect_long.columns)}컬럼")
    print(f"  review_level 컬럼: {list(review_level.columns)}")

    # 4) 상품별 KPI
    print("\n[4/5] 상품별 KPI 집계...")
    kpi = build_product_kpi(review_level, aspect_long)
    kpi_path = SNAPSHOT_DIR / "product_kpi.csv"
    kpi.to_csv(kpi_path, index=False, encoding="utf-8-sig")
    print(f"  product_kpi: {len(kpi):,}개 상품, {len(kpi.columns)}컬럼")
    print(f"  컬럼: {list(kpi.columns)}")

    # Top 5 상품
    print("\n  --- 리뷰 수 Top 5 ---")
    for _, row in kpi.head(5).iterrows():
        print(f"  {row['product_name'][:20]:<22} "
              f"n={int(row['n_reviews']):>5} "
              f"pos={row['overall_pos_rate']:.1%} "
              f"neg={row['overall_neg_rate']:.1%} "
              f"재구매={row['repurchase_mention_rate']:.1%}")

    # 5) SNAPSHOT.json
    print("\n[5/5] SNAPSHOT.json 생성...")
    snapshot = {
        "version": VERSION,
        "bundle": str(BUNDLE_DIR.name),
        "created_at": "2026-02-25",
        "source_checksums": checksums,
        "output_checksums": {
            "review_level_csv": md5(review_path),
            "aspect_long_csv": md5(aspect_path),
            "product_kpi_csv": md5(kpi_path),
            "absa_results_full_csv": md5(SNAPSHOT_DIR / "absa_results_full.csv"),
            "summary_json": md5(SNAPSHOT_DIR / "summary.json"),
        },
        "stats": {
            "total_reviews": len(review_level),
            "total_aspects": len(aspect_long),
            "total_products": len(kpi),
            "review_level_columns": list(review_level.columns),
            "aspect_long_columns": list(aspect_long.columns),
            "product_kpi_columns": list(kpi.columns),
        },
        "files": {
            "absa_results_full.csv": "원본 추론 결과 (JSON 컬럼 포함)",
            "summary.json": "전체 통계 요약",
            "review_level.csv": "리뷰 단위 테이블 (메타 조인 완료, JSON 제거)",
            "aspect_long.csv": "Aspect-level Long 테이블 (aspect별 1행)",
            "product_kpi.csv": "상품별 KPI 집계 (8대 지표)",
            "SNAPSHOT.json": "스냅샷 메타데이터 + 체크섬",
        },
    }
    with open(SNAPSHOT_DIR / "SNAPSHOT.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}")
    print(f"스냅샷 빌드 완료: {SNAPSHOT_DIR}")
    print(f"{'=' * 70}")
    print(f"\n파일 목록:")
    for p in sorted(SNAPSHOT_DIR.iterdir()):
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  {p.name:<30} {size_mb:>8.1f} MB")


if __name__ == "__main__":
    main()
