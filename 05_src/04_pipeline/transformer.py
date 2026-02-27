"""
크롤러 raw → ERD 13개 테이블 변환 모듈

크롤러가 출력하는 3개 raw CSV(products, reviews, ingredients)를
ERD 스키마의 13개 테이블 DataFrame으로 변환한다.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional

try:
    from .derived_features import (
        compute_products_stats,
        compute_users_profile,
        compute_users_repurchase,
    )
except ImportError:
    from derived_features import (
        compute_products_stats,
        compute_users_profile,
        compute_users_repurchase,
    )

# assign_promotion 재사용
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "02_processed_data" / "csv" / "final" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
try:
    from assign_promotion import assign_promotion
    PROMOTION_AVAILABLE = True
except ImportError:
    PROMOTION_AVAILABLE = False


# ERD 테이블 순서 (FK 의존성 반영)
TABLE_ORDER = [
    "brands",
    "manufacturer",
    "ingredients_dic",
    "promotions",
    "products_core",
    "products_category",
    "products_stats",
    "products_ingredients",
    "functional",
    "users_profile",
    "users_repurchase",
    "reviews_core",
    "reviews_text",
]


class CrawlerToERDTransformer:
    """크롤러 raw 3개 DataFrame → ERD 13개 테이블로 변환"""

    def __init__(
        self,
        products_df: pd.DataFrame,
        reviews_df: pd.DataFrame,
        ingredients_df: pd.DataFrame,
        existing_data: Optional[Dict[str, pd.DataFrame]] = None,
        promotions_df: Optional[pd.DataFrame] = None,
    ):
        """
        Parameters
        ----------
        products_df : 크롤러 products raw
            (product_code, brand, name, price, country, category_1, category_2, likes, shares)
        reviews_df : 크롤러 reviews raw
            (product_code, date, user_masked, rating, text, image_count)
        ingredients_df : 크롤러 ingredients raw
            (product_id, ingredient)
        existing_data : 기존 final/ CSV 데이터 (ID 유지용)
            {"brands": df, "users_profile": df, ...}
        promotions_df : promotions.csv (수동 관리)
        """
        self.products_raw = products_df.copy()
        self.reviews_raw = reviews_df.copy()
        self.ingredients_raw = ingredients_df.copy()
        self.existing = existing_data or {}
        self.promotions_input = promotions_df

        # 변환 결과 저장
        self.tables: Dict[str, pd.DataFrame] = {}

    def transform_all(self) -> Dict[str, pd.DataFrame]:
        """전체 변환 실행. 13개 테이블 dict 반환."""
        self._transform_brand()
        self._transform_manufacturer()
        self._transform_ingredients_dic()
        self._transform_promotions()
        self._transform_products_core()
        self._transform_products_category()
        self._transform_products_ingredients()
        self._transform_functional()
        self._transform_reviews()         # reviews_core + reviews_text + users 할당
        self._transform_products_stats()  # 파생변수 (리뷰 데이터 필요)
        self._transform_users_profile()   # 파생변수
        self._transform_users_repurchase()

        return self.tables

    # ── 1. brands ────────────────────────────────────────────────

    def _transform_brand(self):
        """products_raw의 unique brand → brand_id 할당"""
        existing_brand = self.existing.get("brands", pd.DataFrame())

        # 기존 brand 매핑
        if not existing_brand.empty and "brand_id" in existing_brand.columns:
            brand_map = dict(zip(existing_brand["name"], existing_brand["brand_id"]))
            max_id = existing_brand["brand_id"].max()
        else:
            brand_map = {}
            max_id = 0

        # 크롤러에서 새 브랜드 추출
        col = "brand" if "brand" in self.products_raw.columns else None
        if col is None:
            self.tables["brands"] = existing_brand if not existing_brand.empty else pd.DataFrame(columns=["brand_id", "name"])
            return

        raw_brands = self.products_raw[col].dropna().unique()
        for b in raw_brands:
            if b not in brand_map:
                max_id += 1
                brand_map[b] = max_id

        # brand_map → DataFrame
        df = pd.DataFrame(list(brand_map.items()), columns=["name", "brand_id"])
        df = df[["brand_id", "name"]].sort_values("brand_id").reset_index(drop=True)
        self.tables["brands"] = df

        # products_raw에 brand_id 매핑 (이후 사용)
        self.products_raw["brand_id"] = self.products_raw[col].map(brand_map)

    # ── 2. manufacturer ────────────────────────────────────────

    def _transform_manufacturer(self):
        """크롤러에 제조사 정보 없음 → 기존 데이터 유지"""
        existing = self.existing.get("manufacturer", pd.DataFrame())
        if not existing.empty:
            self.tables["manufacturer"] = existing.copy()
        else:
            self.tables["manufacturer"] = pd.DataFrame(columns=["manufacturer_id", "ENTP_NAME"])

    # ── 3. ingredients_dic ──────────────────────────────────────

    def _transform_ingredients_dic(self):
        """ingredients_raw의 unique ingredient → ingredient_id 할당"""
        existing_dic = self.existing.get("ingredients_dic", pd.DataFrame())

        # 기존 매핑 (ingredient_name → ingredient_id)
        if not existing_dic.empty and "ingredient_id" in existing_dic.columns:
            name_col = "ingredient_name" if "ingredient_name" in existing_dic.columns else "name"
            ing_map = dict(zip(existing_dic[name_col], existing_dic["ingredient_id"]))
            max_id = existing_dic["ingredient_id"].max()
            # 기존 application_role, ingredient_type 보존
            existing_attrs = existing_dic.set_index(name_col)[
                [c for c in ["application_role", "ingredient_type"] if c in existing_dic.columns]
            ].to_dict("index")
        else:
            ing_map = {}
            max_id = 0
            existing_attrs = {}

        # 크롤러에서 새 성분
        ing_col = "ingredient"
        if ing_col not in self.ingredients_raw.columns:
            self.tables["ingredients_dic"] = existing_dic if not existing_dic.empty else pd.DataFrame(
                columns=["ingredient_id", "ingredient_name", "application_role", "ingredient_type"]
            )
            return

        raw_ings = self.ingredients_raw[ing_col].dropna().unique()
        for ing in raw_ings:
            if ing not in ing_map:
                max_id += 1
                ing_map[ing] = max_id

        # DataFrame 생성
        rows = []
        for name, iid in ing_map.items():
            attrs = existing_attrs.get(name, {})
            rows.append({
                "ingredient_id": iid,
                "ingredient_name": name,
                "application_role": attrs.get("application_role", None),
                "ingredient_type": attrs.get("ingredient_type", None),
            })

        df = pd.DataFrame(rows).sort_values("ingredient_id").reset_index(drop=True)
        self.tables["ingredients_dic"] = df

        # ingredients_raw에 ingredient_id 매핑
        self.ingredients_raw["ingredient_id"] = self.ingredients_raw[ing_col].map(ing_map)

    # ── 4. promotions ────────────────────────────────────────────

    def _transform_promotions(self):
        """수동 관리 테이블 — 입력 그대로 통과"""
        if self.promotions_input is not None and not self.promotions_input.empty:
            self.tables["promotions"] = self.promotions_input.copy()
        else:
            existing = self.existing.get("promotions", pd.DataFrame())
            if not existing.empty:
                self.tables["promotions"] = existing.copy()
            else:
                self.tables["promotions"] = pd.DataFrame(
                    columns=["promotion_id", "description", "brand_id", "event_type", "start_date", "end_date"]
                )

    # ── 5. products_core ────────────────────────────────────────

    def _transform_products_core(self):
        """product_code, brand_id, manufacturer_id, name, price, country"""
        df = self.products_raw.copy()
        df = df.drop_duplicates(subset=["product_code"])

        # manufacturer_id: 기존 데이터 참조
        existing_core = self.existing.get("products_core", pd.DataFrame())
        if not existing_core.empty and "manufacturer_id" in existing_core.columns:
            mfr_map = existing_core.set_index("product_code")["manufacturer_id"]
            df["manufacturer_id"] = df["product_code"].map(mfr_map)
        elif "manufacturer_id" not in df.columns:
            df["manufacturer_id"] = None

        # brand_id: 이미 _transform_brand()에서 매핑됨
        if "brand_id" not in df.columns:
            df["brand_id"] = None

        result = df[["product_code", "manufacturer_id", "brand_id", "name", "price", "country"]].copy()
        self.tables["products_core"] = result.reset_index(drop=True)

    # ── 6. products_category ────────────────────────────────────

    def _transform_products_category(self):
        """product_code, category_1, category_2"""
        df = self.products_raw.copy()
        df = df.drop_duplicates(subset=["product_code"])

        # 크롤러 컬럼명 매핑
        if "category_1" not in df.columns:
            for alt in ["category_home", "대분류"]:
                if alt in df.columns:
                    df["category_1"] = df[alt]
                    break
        if "category_2" not in df.columns:
            for alt in ["소분류"]:
                if alt in df.columns:
                    df["category_2"] = df[alt]
                    break

        if "category_1" in df.columns and "category_2" in df.columns:
            result = df[["product_code", "category_1", "category_2"]].copy()
        else:
            result = df[["product_code"]].copy()
            result["category_1"] = None
            result["category_2"] = None

        self.tables["products_category"] = result.reset_index(drop=True)

    # ── 7. products_ingredients ─────────────────────────────────

    def _transform_products_ingredients(self):
        """product_code + ingredient_id 매핑 (N:M)"""
        df = self.ingredients_raw.copy()

        # product_id → product_code
        if "product_id" in df.columns and "product_code" not in df.columns:
            df["product_code"] = df["product_id"]

        if "ingredient_id" not in df.columns:
            self.tables["products_ingredients"] = pd.DataFrame(columns=["product_code", "ingredient_id"])
            return

        result = df[["product_code", "ingredient_id"]].dropna(subset=["ingredient_id"]).copy()
        result["ingredient_id"] = result["ingredient_id"].astype(int)
        result = result.drop_duplicates().reset_index(drop=True)
        self.tables["products_ingredients"] = result

    # ── 8. functional ───────────────────────────────────────────

    def _transform_functional(self):
        """기존 functional.csv 유지 (MFDS API 결과, 크롤러에서 생성 불가)"""
        existing = self.existing.get("functional", pd.DataFrame())
        if not existing.empty:
            self.tables["functional"] = existing.copy()
        else:
            self.tables["functional"] = pd.DataFrame(
                columns=["product_code", "ITEM_PH", "ph_category",
                         "is_whitening", "is_wrinkle_reduction", "is_sunscreen", "is_acne"]
            )

    # ── 9. reviews_core + reviews_text ──────────────────────────

    def _transform_reviews(self):
        """리뷰 변환: user_id 할당, review_id 생성, is_reorder 계산, promotion_id 매칭"""
        df = self.reviews_raw.copy()

        if df.empty:
            self.tables["reviews_core"] = pd.DataFrame(
                columns=["review_id", "product_code", "user_id", "rating",
                         "review_date", "image_count", "is_reorder", "promotion_id"]
            )
            self.tables["reviews_text"] = pd.DataFrame(
                columns=["review_id", "text", "review_length"]
            )
            return

        # ── user_id 할당 (user_id_map.csv 매핑 테이블 활용) ──
        existing_profile = self.existing.get("users_profile", pd.DataFrame())
        existing_reviews = self.existing.get("reviews_core", pd.DataFrame())

        user_col = "user_masked" if "user_masked" in df.columns else "user"
        if user_col not in df.columns:
            df["user_id"] = range(1, len(df) + 1)
        else:
            # 기존 매핑 테이블 로드 (user_masked → user_id 영속 매핑)
            # 1) existing_data에 user_id_map이 있으면 사용 (BQ 모드)
            existing_map_df = self.existing.get("user_id_map", pd.DataFrame())
            existing_map = {}

            if not existing_map_df.empty and "user_masked" in existing_map_df.columns:
                existing_map = dict(zip(existing_map_df["user_masked"], existing_map_df["user_id"]))
            else:
                # 2) 로컬 CSV 폴백
                map_path = Path(self.final_dir) / "user_id_map.csv" if hasattr(self, 'final_dir') else None
                if map_path is None:
                    map_path = Path(__file__).resolve().parent.parent.parent / "02_processed_data" / "csv" / "final" / "user_id_map.csv"
                if map_path.exists():
                    map_df = pd.read_csv(map_path)
                    existing_map = dict(zip(map_df["user_masked"], map_df["user_id"]))

            # 기존 최대 user_id 결정
            max_uid = 0
            if existing_map:
                max_uid = max(existing_map.values())
            if not existing_profile.empty and "user_id" in existing_profile.columns:
                max_uid = max(max_uid, existing_profile["user_id"].max())
            if not existing_reviews.empty and "user_id" in existing_reviews.columns:
                max_uid = max(max_uid, existing_reviews["user_id"].max())

            # 기존 매핑 적용 + 신규 유저에게 새 ID 부여
            unique_users = df[user_col].dropna().unique()
            user_map = {}
            new_count = 0
            for u in unique_users:
                if u in existing_map:
                    user_map[u] = existing_map[u]
                else:
                    max_uid += 1
                    user_map[u] = max_uid
                    new_count += 1

            df["user_id"] = df[user_col].map(user_map)

            # 매핑 테이블 업데이트 — tables에 포함하여 BQ UPSERT 대상으로 전달
            updated_map = pd.DataFrame(
                [(k, v) for k, v in user_map.items()],
                columns=["user_masked", "user_id"]
            ).sort_values("user_id").reset_index(drop=True)
            self.tables["user_id_map"] = updated_map

            # 로컬 CSV 백업 (폴백용)
            if map_path and map_path.parent.exists():
                updated_map.to_csv(map_path, index=False)
            if new_count > 0:
                print(f"  user_id_map 업데이트: 기존 {len(existing_map)}명 + 신규 {new_count}명 = {len(user_map)}명")

        # ── review_id 생성 ──
        if not existing_reviews.empty and "review_id" in existing_reviews.columns:
            max_rid = existing_reviews["review_id"].max()
        else:
            max_rid = 0

        # 중복 제거 (product_code + user + date + text)
        dedup_cols = ["product_code"]
        if user_col in df.columns:
            dedup_cols.append(user_col)
        if "date" in df.columns:
            dedup_cols.append("date")
        if "text" in df.columns:
            dedup_cols.append("text")
        df = df.drop_duplicates(subset=dedup_cols)

        df["review_id"] = range(max_rid + 1, max_rid + 1 + len(df))

        # ── 날짜 변환 ──
        date_col = "date" if "date" in df.columns else "review_date"
        if date_col in df.columns:
            df["review_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
        else:
            df["review_date"] = None

        # ── is_reorder ──
        if "is_reorder" not in df.columns:
            if "text" in df.columns:
                df["is_reorder"] = df["text"].fillna("").str.startswith("재구매")
            else:
                df["is_reorder"] = False

        # ── image_count ──
        if "image_count" not in df.columns:
            df["image_count"] = 0

        # ── promotion_id 매칭 (NULL = 프로모션 없음, ERD Optional FK) ──
        df["promotion_id"] = pd.NA
        promotions = self.tables.get("promotions", pd.DataFrame())
        products_core = self.tables.get("products_core", pd.DataFrame())

        if (
            PROMOTION_AVAILABLE
            and not promotions.empty
            and not products_core.empty
            and "review_date" in df.columns
        ):
            # assign_promotion은 review_date가 datetime이어야 함
            temp = df.copy()
            temp["review_date"] = pd.to_datetime(temp["review_date"])
            try:
                temp = assign_promotion(temp, products_core, promotions)
                df["promotion_id"] = temp["promotion_id"].values
            except Exception:
                df["promotion_id"] = pd.NA
        df["promotion_id"] = df["promotion_id"].replace(0, pd.NA).astype("Int64")

        # ── reviews_core 출력 ──
        reviews_core = df[
            ["review_id", "product_code", "user_id", "rating",
             "review_date", "image_count", "is_reorder", "promotion_id"]
        ].copy()
        self.tables["reviews_core"] = reviews_core.reset_index(drop=True)

        # ── reviews_text 출력 ──
        text_col = "text" if "text" in df.columns else None
        if text_col:
            reviews_text = df[["review_id", text_col]].copy()
            reviews_text = reviews_text.rename(columns={text_col: "text"})
            reviews_text["review_length"] = reviews_text["text"].fillna("").str.len()
        else:
            reviews_text = df[["review_id"]].copy()
            reviews_text["text"] = ""
            reviews_text["review_length"] = 0

        self.tables["reviews_text"] = reviews_text.reset_index(drop=True)

    # ── 10. products_stats (파생변수) ───────────────────────────

    def _transform_products_stats(self):
        """파생변수 계산"""
        products_core = self.tables["products_core"]
        reviews_core = self.tables["reviews_core"]
        existing_stats = self.existing.get("products_stats", pd.DataFrame())

        # 크롤러 raw에서 likes, shares 가져오기
        raw_metrics = self.products_raw[["product_code"]].drop_duplicates().copy()
        for col in ["likes", "shares"]:
            if col in self.products_raw.columns:
                raw_metrics[col] = self.products_raw.drop_duplicates("product_code")[col].values

        # existing_stats에 raw metrics 병합 (크롤러 값 우선)
        if not existing_stats.empty:
            merged = existing_stats.copy()
            for col in ["likes", "shares"]:
                if col in raw_metrics.columns:
                    raw_map = raw_metrics.set_index("product_code")[col]
                    merged[col] = merged["product_code"].map(raw_map).combine_first(merged[col])
        else:
            merged = raw_metrics

        self.tables["products_stats"] = compute_products_stats(
            products_core, reviews_core, merged
        )

    # ── 11. users_profile (파생변수) ────────────────────────────

    def _transform_users_profile(self):
        """유저 프로필 파생변수"""
        reviews_core = self.tables["reviews_core"]
        if reviews_core.empty:
            self.tables["users_profile"] = pd.DataFrame(
                columns=["user_id", "user_total_reviews", "user_activity_level",
                         "user_avg_rating_reorder", "user_rating_tendency", "review_tenure"]
            )
            return
        self.tables["users_profile"] = compute_users_profile(reviews_core)

    # ── 12. users_repurchase (파생변수) ─────────────────────────

    def _transform_users_repurchase(self):
        """유저 재구매 파생변수"""
        reviews_core = self.tables["reviews_core"]
        products_core = self.tables["products_core"]
        products_category = self.tables["products_category"]

        if reviews_core.empty:
            self.tables["users_repurchase"] = pd.DataFrame(
                columns=["user_id", "reorder_user_category", "reorder_user_brand", "reorder_user_avg_rating"]
            )
            return
        self.tables["users_repurchase"] = compute_users_repurchase(
            reviews_core, products_core, products_category
        )


def load_existing_final(final_dir: str) -> Dict[str, pd.DataFrame]:
    """기존 02_processed_data/csv/final/ 디렉토리에서 CSV 로드 (레거시)"""
    final_path = Path(final_dir)
    data = {}

    for table_name in TABLE_ORDER:
        csv_path = final_path / f"{table_name}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                data[table_name] = df
            except Exception:
                pass

    return data


def load_existing_from_bq(dataset: str = "daiso") -> Dict[str, pd.DataFrame]:
    """BigQuery에서 ID 매핑에 필요한 기존 데이터 조회

    transformer가 신규 데이터만 변환할 때, 기존 ID 체계를 유지하기 위해
    BQ에서 최소한의 매핑 데이터만 조회한다.

    Returns
    -------
    dict[str, pd.DataFrame]
        TABLE_ORDER의 각 테이블에 대해 ID 매핑용 DataFrame 반환.
        추가 키: "user_id_map", "max_review_id"
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from bigquery.bq_client import query_to_df
    except ImportError:
        from bq_client import query_to_df

    data = {}

    # brands: name → brand_id 매핑
    try:
        data["brands"] = query_to_df(
            f"SELECT brand_id, name FROM `{dataset}.brands`"
        )
    except Exception:
        data["brands"] = pd.DataFrame(columns=["brand_id", "name"])

    # manufacturer: 전체 조회 (행 수 적음)
    try:
        data["manufacturer"] = query_to_df(
            f"SELECT manufacturer_id, ENTP_NAME FROM `{dataset}.manufacturer`"
        )
    except Exception:
        data["manufacturer"] = pd.DataFrame(columns=["manufacturer_id", "ENTP_NAME"])

    # ingredients_dic: ingredient_name → ingredient_id 매핑
    try:
        data["ingredients_dic"] = query_to_df(
            f"SELECT ingredient_id, ingredient_name, application_role, ingredient_type "
            f"FROM `{dataset}.ingredients_dic`"
        )
    except Exception:
        data["ingredients_dic"] = pd.DataFrame(
            columns=["ingredient_id", "ingredient_name", "application_role", "ingredient_type"]
        )

    # promotions: 전체 조회 (행 수 적음)
    try:
        data["promotions"] = query_to_df(
            f"SELECT * FROM `{dataset}.promotions`"
        )
    except Exception:
        data["promotions"] = pd.DataFrame(
            columns=["promotion_id", "description", "brand_id", "event_type", "start_date", "end_date"]
        )

    # products_core: product_code → manufacturer_id 매핑
    try:
        data["products_core"] = query_to_df(
            f"SELECT product_code, manufacturer_id, brand_id FROM `{dataset}.products_core`"
        )
    except Exception:
        data["products_core"] = pd.DataFrame(columns=["product_code", "manufacturer_id", "brand_id"])

    # products_stats: 기존 likes, shares 보존용
    try:
        data["products_stats"] = query_to_df(
            f"SELECT product_code, likes, shares FROM `{dataset}.products_stats`"
        )
    except Exception:
        data["products_stats"] = pd.DataFrame(columns=["product_code", "likes", "shares"])

    # functional: 전체 조회 (MFDS API 결과, 크롤러에서 생성 불가)
    try:
        data["functional"] = query_to_df(
            f"SELECT * FROM `{dataset}.functional`"
        )
    except Exception:
        data["functional"] = pd.DataFrame(
            columns=["product_code", "ITEM_PH", "ph_category",
                     "is_whitening", "is_wrinkle_reduction", "is_sunscreen", "is_acne"]
        )

    # users_profile: user_id 매핑 참조용
    try:
        data["users_profile"] = query_to_df(
            f"SELECT user_id FROM `{dataset}.users_profile`"
        )
    except Exception:
        data["users_profile"] = pd.DataFrame(columns=["user_id"])

    # reviews_core: max review_id + user_id 참조용
    try:
        data["reviews_core"] = query_to_df(
            f"SELECT review_id, user_id FROM `{dataset}.reviews_core`"
        )
    except Exception:
        data["reviews_core"] = pd.DataFrame(columns=["review_id", "user_id"])

    # user_id_map: user_masked → user_id 영속 매핑 (BQ에 없으면 로컬 폴백)
    try:
        data["user_id_map"] = query_to_df(
            f"SELECT user_masked, user_id FROM `{dataset}.user_id_map`"
        )
    except Exception:
        # BQ에 user_id_map 테이블이 없으면 로컬 CSV 폴백
        map_path = (
            Path(__file__).resolve().parent.parent.parent
            / "02_processed_data" / "csv" / "final" / "user_id_map.csv"
        )
        if map_path.exists():
            data["user_id_map"] = pd.read_csv(map_path)
        else:
            data["user_id_map"] = pd.DataFrame(columns=["user_masked", "user_id"])

    print(f"[BQ] 기존 데이터 로드 완료:")
    for key, df in data.items():
        if not df.empty:
            print(f"  {key}: {len(df):,}행")

    return data
