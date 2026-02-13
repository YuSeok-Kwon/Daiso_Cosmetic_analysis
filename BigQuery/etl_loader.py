"""
크롤러 데이터 → BigQuery ETL 모듈

크롤러에서 수집한 CSV 데이터를 BigQuery ERD 스키마에 맞게 변환하고 적재합니다.

사용법:
    from etl_loader import CrawlerETL

    etl = CrawlerETL()

    # 제품 데이터 적재
    etl.load_products("data/products_xxx.csv")

    # 리뷰 데이터 적재
    etl.load_reviews("data/reviews_xxx.csv")

    # 성분 데이터 적재
    etl.load_ingredients("data/ingredients_xxx.csv")
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple
try:
    from .bq_client import (
        get_client, query_to_df, insert_df, upsert_df,
        get_table_count, DEFAULT_DATASET
    )
except ImportError:
    from bq_client import (
        get_client, query_to_df, insert_df, upsert_df,
        get_table_count, DEFAULT_DATASET
    )


class CrawlerETL:
    """크롤러 데이터 → BigQuery 변환 및 적재"""

    def __init__(self, dataset: str = DEFAULT_DATASET):
        self.dataset = dataset
        self.client = get_client()

        # 마스터 테이블 캐시
        self._brands_cache: Optional[Dict[str, int]] = None
        self._categories_cache: Optional[Dict[str, int]] = None
        self._ingredients_cache: Optional[Dict[str, int]] = None
        self._users_cache: Optional[Dict[str, int]] = None

    # ========== 마스터 테이블 조회/캐시 ==========

    def get_brands_map(self, refresh: bool = False) -> Dict[str, int]:
        """브랜드명 → brand_id 매핑"""
        if self._brands_cache is None or refresh:
            df = query_to_df(f"SELECT brand_id, name FROM {self.dataset}.brands")
            self._brands_cache = dict(zip(df['name'], df['brand_id']))
        return self._brands_cache

    def get_categories_map(self, refresh: bool = False) -> Dict[str, int]:
        """카테고리명 → category_id 매핑 (category_2 기준)"""
        if self._categories_cache is None or refresh:
            df = query_to_df(f"SELECT category_id, category_2 FROM {self.dataset}.categories")
            self._categories_cache = dict(zip(df['category_2'], df['category_id']))
        return self._categories_cache

    def get_ingredients_map(self, refresh: bool = False) -> Dict[str, int]:
        """성분명 → ingredient_id 매핑"""
        if self._ingredients_cache is None or refresh:
            df = query_to_df(f"SELECT ingredient_id, name FROM {self.dataset}.ingredients_master")
            self._ingredients_cache = dict(zip(df['name'], df['ingredient_id']))
        return self._ingredients_cache

    def get_users_map(self, refresh: bool = False) -> Dict[str, int]:
        """user_masked → user_id 매핑"""
        if self._users_cache is None or refresh:
            df = query_to_df(f"SELECT user_id, user_masked FROM {self.dataset}.users")
            self._users_cache = dict(zip(df['user_masked'], df['user_id']))
        return self._users_cache

    def get_max_user_id(self) -> int:
        """users 테이블의 최대 user_id"""
        df = query_to_df(f"SELECT COALESCE(MAX(user_id), 0) as max_id FROM {self.dataset}.users")
        return int(df['max_id'].iloc[0])

    def get_max_order_id(self) -> int:
        """reviews 테이블의 최대 order_id"""
        df = query_to_df(f"SELECT COALESCE(MAX(order_id), 0) as max_id FROM {self.dataset}.reviews")
        return int(df['max_id'].iloc[0])

    # ========== 브랜드 자동 등록 ==========

    def register_new_brands(self, brand_names: list) -> Dict[str, int]:
        """새로운 브랜드를 brands 테이블에 등록"""
        brands_map = self.get_brands_map()
        new_brands = [b for b in brand_names if b and b not in brands_map]

        if not new_brands:
            return brands_map

        # 새 brand_id 할당
        max_id = max(brands_map.values()) if brands_map else 0
        new_data = []
        for i, name in enumerate(new_brands, start=1):
            new_id = max_id + i
            new_data.append({"brand_id": new_id, "name": name})
            brands_map[name] = new_id

        # BigQuery에 삽입
        df_new = pd.DataFrame(new_data)
        insert_df(df_new, "brands", self.dataset)
        print(f"새 브랜드 {len(new_brands)}개 등록: {new_brands}")

        self._brands_cache = brands_map
        return brands_map

    # ========== 사용자 자동 등록 ==========

    def register_new_users(self, user_masked_list: list) -> Dict[str, int]:
        """새로운 사용자를 users 테이블에 등록"""
        users_map = self.get_users_map()
        new_users = [u for u in user_masked_list if u and u not in users_map]

        if not new_users:
            return users_map

        max_id = self.get_max_user_id()
        new_data = []
        for i, masked in enumerate(new_users, start=1):
            new_id = max_id + i
            new_data.append({
                "user_id": new_id,
                "user_masked": masked,
                "activity_level": None,
                "rating_tendency": None
            })
            users_map[masked] = new_id

        df_new = pd.DataFrame(new_data)
        insert_df(df_new, "users", self.dataset)
        print(f"새 사용자 {len(new_users)}명 등록")

        self._users_cache = users_map
        return users_map

    # ========== 성분 자동 등록 ==========

    def register_new_ingredients(self, ingredient_names: list) -> Dict[str, int]:
        """새로운 성분을 ingredients_master 테이블에 등록"""
        ing_map = self.get_ingredients_map()
        new_ings = [i for i in ingredient_names if i and i not in ing_map]

        if not new_ings:
            return ing_map

        max_id = max(ing_map.values()) if ing_map else 0
        new_data = []
        for i, name in enumerate(new_ings, start=1):
            new_id = max_id + i
            new_data.append({
                "ingredient_id": new_id,
                "name": name,
                "ewg_grade": None,
                "is_caution": None
            })
            ing_map[name] = new_id

        df_new = pd.DataFrame(new_data)
        insert_df(df_new, "ingredients_master", self.dataset)
        print(f"새 성분 {len(new_ings)}개 등록")

        self._ingredients_cache = ing_map
        return ing_map

    # ========== 제품 데이터 적재 ==========

    def load_products(
        self,
        csv_path: str,
        category_id: Optional[int] = None
    ) -> Tuple[int, int, int]:
        """
        크롤러 products CSV → BigQuery 적재

        크롤러 컬럼: product_code, brand, name, price, country, likes, shares, url
        BigQuery 분리:
          - products: product_code, brand_id, category_id, name, price, country, created_at
          - product_metrics: product_code, likes, shares, review_count, ...
          - product_attributes: product_code, group, is_functional, ...

        Returns:
            (products 적재 수, metrics 적재 수, attributes 적재 수)
        """
        df = pd.read_csv(csv_path)
        print(f"제품 CSV 로드: {len(df)}행")

        # 브랜드 등록 및 매핑
        if 'brand' in df.columns:
            brands_map = self.register_new_brands(df['brand'].dropna().unique().tolist())
            df['brand_id'] = df['brand'].map(brands_map)
        else:
            df['brand_id'] = None

        # 카테고리 ID
        if category_id:
            df['category_id'] = category_id
        elif 'category_id' not in df.columns:
            df['category_id'] = None

        # 1. products 테이블
        df_products = df[['product_code', 'brand_id', 'category_id', 'name', 'price', 'country']].copy()
        df_products['created_at'] = datetime.now()
        df_products = df_products.drop_duplicates(subset=['product_code'])

        upsert_df(df_products, "products", dataset=self.dataset)
        print(f"  products: {len(df_products)}행 upsert")

        # 2. product_metrics 테이블
        metrics_cols = ['product_code']
        if 'likes' in df.columns:
            metrics_cols.append('likes')
        if 'shares' in df.columns:
            metrics_cols.append('shares')

        if len(metrics_cols) > 1:
            df_metrics = df[metrics_cols].drop_duplicates(subset=['product_code']).copy()
            df_metrics['last_updated'] = datetime.now().date()

            # 없는 컬럼 기본값
            for col in ['review_count', 'engagement_score', 'cp_index', 'is_god_sung_bi', 'review_density']:
                if col not in df_metrics.columns:
                    df_metrics[col] = None

            upsert_df(df_metrics, "product_metrics", dataset=self.dataset)
            print(f"  product_metrics: {len(df_metrics)}행 upsert")
        else:
            df_metrics = pd.DataFrame()

        # 3. product_attributes 테이블 (빈 프레임 - 나중에 채움)
        df_attrs = pd.DataFrame()

        return len(df_products), len(df_metrics), len(df_attrs)

    # ========== 리뷰 데이터 적재 ==========

    def load_reviews(self, csv_path: str) -> Tuple[int, int]:
        """
        크롤러 reviews CSV → BigQuery 적재

        크롤러 컬럼: product_code, date, user_masked, rating, text, image_count
        BigQuery:
          - reviews: order_id, product_code, user_id, write_date, rating, text, image_count, is_reorder
          - review_analysis: order_id, length, length_category, sentiment, ...

        Returns:
            (reviews 적재 수, review_analysis 적재 수)
        """
        df = pd.read_csv(csv_path)
        print(f"리뷰 CSV 로드: {len(df)}행")

        # 사용자 등록 및 매핑
        users_map = self.register_new_users(df['user_masked'].dropna().unique().tolist())
        df['user_id'] = df['user_masked'].map(users_map)

        # order_id 생성 (기존 최대값 + 1부터)
        max_order_id = self.get_max_order_id()

        # 중복 제거를 위해 (product_code, user_masked, date, text) 조합으로 체크
        df = df.drop_duplicates(subset=['product_code', 'user_masked', 'date', 'text'])
        df['order_id'] = range(max_order_id + 1, max_order_id + 1 + len(df))

        # 날짜 변환
        df['write_date'] = pd.to_datetime(df['date']).dt.date

        # is_reorder 처리 (없으면 False)
        if 'is_reorder' not in df.columns:
            df['is_reorder'] = False

        # 1. reviews 테이블
        df_reviews = df[['order_id', 'product_code', 'user_id', 'write_date',
                         'rating', 'text', 'image_count', 'is_reorder']].copy()

        insert_df(df_reviews, "reviews", dataset=self.dataset)
        print(f"  reviews: {len(df_reviews)}행 insert")

        # 2. review_analysis 테이블
        df_analysis = df[['order_id', 'text']].copy()
        df_analysis['length'] = df_analysis['text'].fillna('').str.len()
        df_analysis['length_category'] = pd.cut(
            df_analysis['length'],
            bins=[0, 20, 50, 100, 500, float('inf')],
            labels=['very_short', 'short', 'medium', 'long', 'very_long']
        )
        df_analysis = df_analysis[['order_id', 'length', 'length_category']]
        df_analysis['sentiment'] = None
        df_analysis['sentiment_score'] = None
        df_analysis['promo_type'] = None

        insert_df(df_analysis, "review_analysis", dataset=self.dataset)
        print(f"  review_analysis: {len(df_analysis)}행 insert")

        return len(df_reviews), len(df_analysis)

    # ========== 성분 데이터 적재 ==========

    def load_ingredients(self, csv_path: str) -> Tuple[int, int]:
        """
        크롤러 ingredients CSV → BigQuery 적재

        크롤러 컬럼: product_id, name, ingredient, can_halal, can_vegan
        BigQuery:
          - product_ingredients: product_code, ingredient_id, rank
          - product_attributes: can_halal, can_vegan 업데이트

        Returns:
            (product_ingredients 적재 수, attributes 업데이트 수)
        """
        df = pd.read_csv(csv_path)
        print(f"성분 CSV 로드: {len(df)}행")

        # product_id → product_code 이름 통일
        if 'product_id' in df.columns and 'product_code' not in df.columns:
            df['product_code'] = df['product_id']

        # 성분 등록 및 매핑
        ing_map = self.register_new_ingredients(df['ingredient'].dropna().unique().tolist())
        df['ingredient_id'] = df['ingredient'].map(ing_map)

        # rank 생성 (제품별 순서)
        df['rank'] = df.groupby('product_code').cumcount() + 1

        # 1. product_ingredients 테이블
        df_pi = df[['product_code', 'ingredient_id', 'rank']].copy()
        df_pi = df_pi.dropna(subset=['ingredient_id'])
        df_pi['ingredient_id'] = df_pi['ingredient_id'].astype(int)

        upsert_df(df_pi, "product_ingredients", dataset=self.dataset)
        print(f"  product_ingredients: {len(df_pi)}행 upsert")

        # 2. product_attributes 업데이트 (can_halal, can_vegan)
        if 'can_halal' in df.columns or 'can_vegan' in df.columns:
            df_attrs = df.groupby('product_code').agg({
                'can_halal': 'first',
                'can_vegan': 'first'
            }).reset_index()

            # 기존 product_attributes에 merge
            existing = query_to_df(f"SELECT * FROM {self.dataset}.product_attributes")
            if not existing.empty:
                merged = existing.merge(df_attrs, on='product_code', how='left', suffixes=('', '_new'))
                for col in ['can_halal', 'can_vegan']:
                    if f'{col}_new' in merged.columns:
                        merged[col] = merged[f'{col}_new'].combine_first(merged[col])
                        merged = merged.drop(columns=[f'{col}_new'])
                upsert_df(merged, "product_attributes", dataset=self.dataset)
                print(f"  product_attributes: {len(df_attrs)}행 업데이트")
            else:
                print("  product_attributes: 기존 데이터 없음, 스킵")
        else:
            df_attrs = pd.DataFrame()

        return len(df_pi), len(df_attrs)

    # ========== 전체 적재 ==========

    def load_all(
        self,
        products_csv: Optional[str] = None,
        reviews_csv: Optional[str] = None,
        ingredients_csv: Optional[str] = None,
        category_id: Optional[int] = None
    ) -> dict:
        """모든 데이터 일괄 적재"""
        results = {}

        if products_csv:
            results['products'] = self.load_products(products_csv, category_id)

        if reviews_csv:
            results['reviews'] = self.load_reviews(reviews_csv)

        if ingredients_csv:
            results['ingredients'] = self.load_ingredients(ingredients_csv)

        return results


# ========== 편의 함수 ==========

def load_crawler_data(
    products_csv: str = None,
    reviews_csv: str = None,
    ingredients_csv: str = None,
    category_id: int = None
) -> dict:
    """크롤러 데이터 일괄 적재 (간편 함수)"""
    etl = CrawlerETL()
    return etl.load_all(products_csv, reviews_csv, ingredients_csv, category_id)


if __name__ == "__main__":
    print("=== ETL Loader 테스트 ===")
    etl = CrawlerETL()

    print("\n브랜드 매핑 (샘플 5개):")
    brands = etl.get_brands_map()
    for name, bid in list(brands.items())[:5]:
        print(f"  {name}: {bid}")

    print(f"\n총 브랜드: {len(brands)}개")
    print(f"총 카테고리: {len(etl.get_categories_map())}개")
    print(f"총 성분: {len(etl.get_ingredients_map())}개")
    print(f"총 사용자: {len(etl.get_users_map())}명")
