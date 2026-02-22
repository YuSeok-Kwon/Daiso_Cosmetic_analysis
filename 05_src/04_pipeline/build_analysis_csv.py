#!/usr/bin/env python3
"""
분석용 통합 CSV 생성 스크립트

정규화된 final/ 테이블들을 JOIN하여 분석에 바로 사용할 수 있는
products_full.csv, reviews_full.csv, users_full.csv, promotions_full.csv, ingredients_full.csv를 생성한다.

사용법:
    python 05_src/04_pipeline/build_analysis_csv.py
"""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FINAL_DIR = PROJECT_ROOT / "02_processed_data" / "csv" / "final"
OUTPUT_DIR = PROJECT_ROOT / "02_processed_data" / "csv" / "analysis"


def build_products_full(final_dir: Path) -> pd.DataFrame:
    """제품 통합 테이블 (948행 × 23열)

    products_core + brand + manufacturer + category + stats + functional
    """
    products = pd.read_csv(final_dir / "products_core.csv")
    brand = pd.read_csv(final_dir / "brands.csv")
    mfr = pd.read_csv(final_dir / "manufacturer.csv")
    cat = pd.read_csv(final_dir / "products_category.csv")
    stats = pd.read_csv(final_dir / "products_stats.csv")
    func = pd.read_csv(final_dir / "functional.csv")
    pi = pd.read_csv(final_dir / "products_ingredients.csv")
    ing = pd.read_csv(final_dir / "ingredients_dic.csv")

    df = products.merge(brand, on="brand_id", how="left", suffixes=("", "_brand"))
    df = df.rename(columns={"name_brand": "brand_name"})
    df = df.merge(mfr, on="manufacturer_id", how="left")
    df = df.rename(columns={"ENTP_NAME": "manufacturer_name"})
    df = df.merge(cat, on="product_code", how="left")
    df = df.merge(stats, on="product_code", how="left")
    df = df.merge(func, on="product_code", how="left")

    # 제품별 알레르기 성분 수 집계
    product_allergic = (
        pi.merge(ing[["ingredient_id", "allergic"]], on="ingredient_id", how="left")
        .groupby("product_code")["allergic"].sum()
        .reset_index()
        .rename(columns={"allergic": "allergic_count"})
    )
    product_allergic["allergic_count"] = product_allergic["allergic_count"].astype(int)
    df = df.merge(product_allergic, on="product_code", how="left")
    df["allergic_count"] = df["allergic_count"].fillna(0).astype(int)

    # ID 컬럼 제거 (이미 이름으로 조인됨)
    df = df.drop(columns=["brand_id", "manufacturer_id"], errors="ignore")

    # 컬럼 순서 정리
    front = ["product_code", "name", "brand_name", "manufacturer_name",
             "price", "country", "category_1", "category_2"]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]

    return df


def build_reviews_full(final_dir: Path) -> pd.DataFrame:
    """리뷰 통합 테이블 (~32만행)

    reviews_core + reviews_text (유저/프로모션은 FK로만 참조)
    """
    reviews = pd.read_csv(final_dir / "reviews_core.csv")
    text = pd.read_csv(final_dir / "reviews_text.csv")

    df = reviews.merge(text, on="review_id", how="left")

    # 컬럼 순서 정리
    front = ["review_id", "product_code",
             "user_id", "rating", "review_date", "text", "review_length",
             "image_count", "is_reorder", "promotion_id"]
    df = df[front]

    return df


def build_promotions_full(final_dir: Path) -> pd.DataFrame:
    """프로모션 테이블

    promotions + brand 이름 조인
    """
    promos = pd.read_csv(final_dir / "promotions.csv")
    brand = pd.read_csv(final_dir / "brands.csv")

    df = promos.merge(brand, on="brand_id", how="left")
    df = df.rename(columns={"name": "brand_name", "description": "promotion_desc"})
    df = df.drop(columns=["brand_id"], errors="ignore")

    front = ["promotion_id", "promotion_desc", "event_type",
             "brand_name", "start_date", "end_date"]
    df = df[front]

    return df


def build_users_full(final_dir: Path) -> pd.DataFrame:
    """유저 통합 테이블

    users_profile + users_repurchase
    """
    users = pd.read_csv(final_dir / "users_profile.csv")
    users_rep = pd.read_csv(final_dir / "users_repurchase.csv")

    df = users.merge(users_rep, on="user_id", how="left")

    # 컬럼 순서 정리
    front = ["user_id", "user_total_reviews", "user_activity_level",
             "user_rating_tendency", "review_tenure",
             "reorder_user_category", "reorder_user_brand",
             "reorder_user_avg_rating"]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]

    return df


def build_ingredients_full(final_dir: Path) -> pd.DataFrame:
    """제품×성분 통합 테이블

    products_ingredients + ingredients_dic + 제품 기본정보(이름, 브랜드, 카테고리)
    """
    pi = pd.read_csv(final_dir / "products_ingredients.csv")
    ing = pd.read_csv(final_dir / "ingredients_dic.csv")
    products = pd.read_csv(final_dir / "products_core.csv")
    brand = pd.read_csv(final_dir / "brands.csv")
    cat = pd.read_csv(final_dir / "products_category.csv")

    df = pi.merge(ing, on="ingredient_id", how="left")
    df = df.merge(products[["product_code", "brand_id", "name", "price"]], on="product_code", how="left")
    df = df.merge(brand, on="brand_id", how="left", suffixes=("", "_brand"))
    df = df.rename(columns={"name": "product_name", "name_brand": "brand_name"})
    df = df.merge(cat, on="product_code", how="left")
    df = df.drop(columns=["brand_id"], errors="ignore")

    front = ["product_code", "product_name", "brand_name", "price",
             "category_1", "category_2",
             "ingredient_id", "ingredient_name",
             "ingredient_type", "allergic", "effect"]
    df = df[front]

    return df


def main():
    print("분석용 통합 CSV 생성")
    print(f"  소스: {FINAL_DIR}")
    print(f"  출력: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # products_full
    print("\n[1/5] products_full 생성 중...")
    pf = build_products_full(FINAL_DIR)
    pf_path = OUTPUT_DIR / "products_full.csv"
    pf.to_csv(pf_path, index=False, encoding="utf-8-sig")
    print(f"  → {pf_path.name}: {len(pf):,}행 × {len(pf.columns)}열")

    # reviews_full
    print("\n[2/5] reviews_full 생성 중...")
    rf = build_reviews_full(FINAL_DIR)
    rf_path = OUTPUT_DIR / "reviews_full.csv"
    rf.to_csv(rf_path, index=False, encoding="utf-8-sig")
    print(f"  → {rf_path.name}: {len(rf):,}행 × {len(rf.columns)}열")

    # users_full
    print("\n[3/5] users_full 생성 중...")
    uf = build_users_full(FINAL_DIR)
    uf_path = OUTPUT_DIR / "users_full.csv"
    uf.to_csv(uf_path, index=False, encoding="utf-8-sig")
    print(f"  → {uf_path.name}: {len(uf):,}행 × {len(uf.columns)}열")

    # promotions_full
    print("\n[4/5] promotions_full 생성 중...")
    pmf = build_promotions_full(FINAL_DIR)
    pmf_path = OUTPUT_DIR / "promotions_full.csv"
    pmf.to_csv(pmf_path, index=False, encoding="utf-8-sig")
    print(f"  → {pmf_path.name}: {len(pmf):,}행 × {len(pmf.columns)}열")

    # ingredients_full
    print("\n[5/5] ingredients_full 생성 중...")
    igf = build_ingredients_full(FINAL_DIR)
    igf_path = OUTPUT_DIR / "ingredients_full.csv"
    igf.to_csv(igf_path, index=False, encoding="utf-8-sig")
    print(f"  → {igf_path.name}: {len(igf):,}행 × {len(igf.columns)}열")

    print("\n완료!")


if __name__ == "__main__":
    main()
