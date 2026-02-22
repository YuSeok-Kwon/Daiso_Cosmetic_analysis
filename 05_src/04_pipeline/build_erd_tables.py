"""
Merged CSV → ERD 정규화 테이블 생성 스크립트
ERD 명세(07_docs/data/ERD.md)를 그대로 따름.

소스:
  - products_ingredients_merged.csv (PI)
  - products_reviews_merged.csv (PR)
  - final/functional.csv (기존: is_acne 보충)
  - final/ingredients_dic.csv (기존: application_role 보충)

출력: final/ 디렉토리 (13개 CSV)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── 경로 설정 ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_DIR = PROJECT_ROOT / "02_processed_data" / "csv"
PI_PATH = CSV_DIR / "products_ingredients_merged.csv"
PR_PATH = CSV_DIR / "products_reviews_merged.csv"
FINAL = CSV_DIR / "final"
FINAL.mkdir(exist_ok=True)

# 기존 보충 데이터
OLD_FUNC = FINAL / "functional.csv"

print("=" * 60)
print("ERD 정규화 테이블 생성 시작")
print("=" * 60)

# ── 1. 데이터 로드 ────────────────────────────────────────
print("\n[1/6] 데이터 로드...")
pi = pd.read_csv(PI_PATH)
pr = pd.read_csv(PR_PATH)
print(f"  PI: {pi.shape[0]:,}행 × {pi.shape[1]}컬럼")
print(f"  PR: {pr.shape[0]:,}행 × {pr.shape[1]}컬럼")

# PR 정리: null 리뷰 제거 + order_id 중복 제거
pr = pr.dropna(subset=["order_id", "user_id"])
pr = pr.drop_duplicates(subset="order_id", keep="first")
pr["order_id"] = pr["order_id"].astype(int)
pr["user_id"] = pr["user_id"].astype(int)
print(f"  PR 정리 후: {pr.shape[0]:,}행")

# PI 제품 기준 테이블
pi_prod = pi.drop_duplicates("product_code")

stats = {}  # 테이블별 행 수 기록

# ── 2. 차원 테이블 ────────────────────────────────────────
print("\n[2/6] 차원 테이블 추출...")

# 2-1. Brand
brands = pi_prod[["brand"]].drop_duplicates().dropna().reset_index(drop=True)
brands["brand_id"] = range(1, len(brands) + 1)
brands = brands.rename(columns={"brand": "name"})
brands = brands[["brand_id", "name"]]
brand_map = dict(zip(brands["name"], brands["brand_id"]))
stats["brands.csv"] = len(brands)

# 2-2. Manufacturer
manufacturers = pi_prod[["ENTP_NAME"]].drop_duplicates().dropna().reset_index(drop=True)
manufacturers["manufacturer_id"] = range(1, len(manufacturers) + 1)
manufacturers = manufacturers[["manufacturer_id", "ENTP_NAME"]]
mfr_map = dict(zip(manufacturers["ENTP_NAME"], manufacturers["manufacturer_id"]))
stats["manufacturer.csv"] = len(manufacturers)

# 2-3. Ingredients_dic

ing_unique = pi[["ingredient", "ingredient_type", "allergic", "effect"]].drop_duplicates("ingredient").dropna(subset=["ingredient"]).reset_index(drop=True)
ing_unique["ingredient_id"] = range(1, len(ing_unique) + 1)
ing_unique["ingredient_name"] = ing_unique["ingredient"]
ingredients_dic = ing_unique[["ingredient_id", "ingredient_name", "ingredient_type", "allergic", "effect"]]
ing_id_map = dict(zip(ingredients_dic["ingredient_name"], ingredients_dic["ingredient_id"]))
stats["ingredients_dic.csv"] = len(ingredients_dic)

# 2-4. Promotions
# PR에서 프로모션 추출 (description not null & event_type not null)
promo_reviews = pr[pr["promotion_yn"] == 1].copy()
promo_with_desc = promo_reviews.dropna(subset=["description", "event_type"])

# brand → brand_id 매핑
promo_with_desc = promo_with_desc.copy()
promo_with_desc["brand_id"] = promo_with_desc["brand"].map(brand_map)

# 고유 프로모션 추출: (description, brand_id, event_type)
promo_dim = (
    promo_with_desc.groupby(["description", "brand_id", "event_type"])
    .agg(start_date=("date", "min"), end_date=("date", "max"))
    .reset_index()
)
promo_dim["promotion_id"] = range(1, len(promo_dim) + 1)
promotions = promo_dim[["promotion_id", "start_date", "end_date", "description", "brand_id", "event_type"]]
promotions["brand_id"] = promotions["brand_id"].astype("Int64")
stats["promotions.csv"] = len(promotions)

# 프로모션 ID 매핑용 딕셔너리
promo_key_to_id = {}
for _, row in promotions.iterrows():
    key = (row["description"], row["brand_id"], row["event_type"])
    promo_key_to_id[key] = row["promotion_id"]

# ── 3. 상품 테이블 ────────────────────────────────────────
print("\n[3/6] 상품 테이블 추출...")

# 3-1. Products_core
products_core = pi_prod[["product_code", "ENTP_NAME", "brand", "name", "price", "country"]].copy()
products_core["manufacturer_id"] = products_core["ENTP_NAME"].map(mfr_map)
products_core["brand_id"] = products_core["brand"].map(brand_map)
products_core = products_core[["product_code", "manufacturer_id", "brand_id", "name", "price", "country"]]
stats["products_core.csv"] = len(products_core)

# 3-2. Products_category
products_category = pi_prod[["product_code", "category_1", "category_2"]].copy()
stats["products_category.csv"] = len(products_category)

# 3-3. Products_stats
products_stats = pi_prod[["product_code", "likes", "shares", "review_count",
                           "engagement_score", "cp_index", "review_density"]].copy()
stats["products_stats.csv"] = len(products_stats)

# 3-4. Functional (functional==1인 제품만)
func_prods = pi_prod[pi_prod["functional"] == 1].copy()

# ph_category 파생
def get_ph_category(ph_str):
    try:
        ph = float(ph_str)
    except (ValueError, TypeError):
        return np.nan
    if ph < 3:
        return np.nan
    elif ph < 4.5:
        return "산성"
    elif ph < 7:
        return "약산성"
    elif ph == 7:
        return "중성"
    else:
        return "알칼리성"

func_prods["ph_category"] = func_prods["ITEM_PH"].apply(get_ph_category)

# is_acne: 기존 functional에서 보충
old_func = pd.read_csv(OLD_FUNC)
acne_map = dict(zip(old_func["product_code"], old_func["is_acne"]))
func_prods["is_acne"] = func_prods["product_code"].map(acne_map).fillna(False).astype(bool)

# 컬럼 이름 ERD에 맞게
func_prods = func_prods.rename(columns={
    "whitening": "is_whitening",
    "wrinkle_reduction": "is_wrinkle_reduction",
    "sunscreen": "is_sunscreen",
})
func_prods["is_whitening"] = func_prods["is_whitening"].astype(bool)
func_prods["is_wrinkle_reduction"] = func_prods["is_wrinkle_reduction"].astype(bool)
func_prods["is_sunscreen"] = func_prods["is_sunscreen"].astype(bool)

functional = func_prods[["product_code", "ITEM_PH", "ph_category",
                          "is_whitening", "is_wrinkle_reduction", "is_sunscreen", "is_acne"]]
stats["functional.csv"] = len(functional)

# 3-5. Products_ingredients (N:M 매핑)
pi_mapping = pi[["product_code", "ingredient"]].dropna(subset=["ingredient"]).copy()
pi_mapping["ingredient_id"] = pi_mapping["ingredient"].map(ing_id_map)
products_ingredients = pi_mapping[["product_code", "ingredient_id"]].drop_duplicates()
stats["products_ingredients.csv"] = len(products_ingredients)

# ── 4. 유저 테이블 ────────────────────────────────────────
print("\n[4/6] 유저 테이블 추출...")

# 4-1. Users_profile
pr["date_dt"] = pd.to_datetime(pr["date"])
user_dates = pr.groupby("user_id")["date_dt"].agg(["min", "max"])
user_dates["review_tenure"] = (user_dates["max"] - user_dates["min"]).dt.days
tenure_map = user_dates["review_tenure"].to_dict()

user_cols = ["user_id", "user_total_reviews", "user_activity_level",
             "user_rating_tendency"]
users_profile = pr[user_cols].drop_duplicates("user_id").copy()
users_profile["review_tenure"] = users_profile["user_id"].map(tenure_map)
users_profile["user_total_reviews"] = users_profile["user_total_reviews"].astype("Int64")
users_profile = users_profile.sort_values("user_id").reset_index(drop=True)
stats["users_profile.csv"] = len(users_profile)

# 4-2. Users_repurchase (재구매 경험 유저만)
users_repurchase = (
    pr.groupby("user_id")
    .agg(
        reorder_user_category=("is_category_repurchase", "sum"),
        reorder_user_brand=("is_brand_repurchase", "sum"),
    )
    .reset_index()
)
users_repurchase["reorder_user_category"] = users_repurchase["reorder_user_category"].astype(int)
users_repurchase["reorder_user_brand"] = users_repurchase["reorder_user_brand"].astype(int)

# reorder_user_avg_rating 추가 (PR 원본에서 가져오기)
avg_rating = pr[["user_id", "user_avg_rating_reorder"]].drop_duplicates("user_id")
avg_rating = avg_rating.rename(columns={"user_avg_rating_reorder": "reorder_user_avg_rating"})
users_repurchase = users_repurchase.merge(avg_rating, on="user_id", how="left")

# 3개 컬럼 모두 0/NaN이면 제거
users_repurchase = users_repurchase[
    ~((users_repurchase["reorder_user_category"] == 0)
      & (users_repurchase["reorder_user_brand"] == 0)
      & (users_repurchase["reorder_user_avg_rating"].isna()))
].reset_index(drop=True)

users_repurchase = users_repurchase[["user_id", "reorder_user_category",
                                     "reorder_user_brand", "reorder_user_avg_rating"]]
stats["users_repurchase.csv"] = len(users_repurchase)

# ── 5. 리뷰 테이블 ────────────────────────────────────────
print("\n[5/6] 리뷰 테이블 추출...")

# 5-1. Reviews_core
reviews = pr.copy()
reviews["review_id"] = reviews["order_id"]

# promotion_id 매핑
reviews["_brand_id"] = reviews["brand"].map(brand_map)


def map_promo_id(row):
    if pd.isna(row.get("description")) or pd.isna(row.get("event_type")):
        return 0
    key = (row["description"], row["_brand_id"], row["event_type"])
    return promo_key_to_id.get(key, 0)


reviews["promotion_id"] = reviews.apply(map_promo_id, axis=1)

reviews_core = reviews[["review_id", "product_code", "user_id", "rating",
                         "date", "image_count", "is_reorder", "promotion_id"]].copy()
reviews_core = reviews_core.rename(columns={"date": "review_date"})
reviews_core["image_count"] = reviews_core["image_count"].fillna(0).astype(int)
reviews_core = reviews_core.sort_values("review_id").reset_index(drop=True)
stats["reviews_core.csv"] = len(reviews_core)

# 5-2. Reviews_text
reviews_text = reviews[["review_id", "text", "review_length"]].copy()
reviews_text["review_length"] = reviews_text["review_length"].fillna(0).astype(int)
reviews_text = reviews_text.sort_values("review_id").reset_index(drop=True)
stats["reviews_text.csv"] = len(reviews_text)

# ── 6. 저장 ───────────────────────────────────────────────
print("\n[6/6] CSV 저장...")

tables = {
    "brands.csv": brands,
    "manufacturer.csv": manufacturers,
    "ingredients_dic.csv": ingredients_dic,
    "promotions.csv": promotions,
    "products_core.csv": products_core,
    "products_category.csv": products_category,
    "products_stats.csv": products_stats,
    "functional.csv": functional,
    "products_ingredients.csv": products_ingredients,
    "users_profile.csv": users_profile,
    "users_repurchase.csv": users_repurchase,
    "reviews_core.csv": reviews_core,
    "reviews_text.csv": reviews_text,
}

for fname, df in tables.items():
    path = FINAL / fname
    df.to_csv(path, index=False)

# ── 검증 ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("생성 결과 요약")
print("=" * 60)
print(f"{'테이블':<28} {'행 수':>10}  PK 중복")
print("-" * 55)

pk_map = {
    "brands.csv": "brand_id",
    "manufacturer.csv": "manufacturer_id",
    "ingredients_dic.csv": "ingredient_id",
    "promotions.csv": "promotion_id",
    "products_core.csv": "product_code",
    "products_category.csv": "product_code",
    "products_stats.csv": "product_code",
    "functional.csv": "product_code",
    "products_ingredients.csv": ["product_code", "ingredient_id"],
    "users_profile.csv": "user_id",
    "users_repurchase.csv": "user_id",
    "reviews_core.csv": "review_id",
    "reviews_text.csv": "review_id",
}

all_ok = True
for fname, df in tables.items():
    pk = pk_map[fname]
    if isinstance(pk, list):
        dup_count = df.duplicated(subset=pk).sum()
    else:
        dup_count = df.duplicated(subset=pk).sum()
    status = "OK" if dup_count == 0 else f"중복 {dup_count}건!"
    if dup_count > 0:
        all_ok = False
    print(f"  {fname:<26} {len(df):>10,}  {status}")

print("-" * 55)
print(f"  총 테이블: {len(tables)}개")
print(f"  PK 검증: {'ALL PASS' if all_ok else 'FAIL - 중복 확인 필요'}")
print("=" * 60)
