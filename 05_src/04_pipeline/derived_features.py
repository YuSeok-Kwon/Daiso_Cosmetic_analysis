"""
파생변수 계산 모듈

products_stats, users_profile, users_repurchase 테이블의 파생변수를 계산한다.
"""
import pandas as pd
import numpy as np


def compute_products_stats(
    products_core: pd.DataFrame,
    reviews_core: pd.DataFrame,
    existing_stats: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    products_stats 테이블 계산

    Parameters
    ----------
    products_core : products_core DataFrame (product_code, price 필수)
    reviews_core : reviews_core DataFrame (product_code, review_date 필수)
    existing_stats : 기존 products_stats (likes, shares, risk_score 보존용)

    Returns
    -------
    products_stats DataFrame
    """
    # 기본 집계: 제품별 리뷰 수, 첫 리뷰일, 마지막 리뷰일
    review_agg = (
        reviews_core.groupby("product_code")
        .agg(
            review_count=("review_date", "count"),
            first_review=("review_date", "min"),
            last_review=("review_date", "max"),
        )
        .reset_index()
    )

    # products_core에서 product_code 목록
    stats = products_core[["product_code"]].drop_duplicates().copy()

    # 기존 stats에서 likes, shares, risk_score 가져오기
    if existing_stats is not None and not existing_stats.empty:
        keep_cols = ["product_code"]
        for col in ["likes", "shares", "risk_score"]:
            if col in existing_stats.columns:
                keep_cols.append(col)
        stats = stats.merge(existing_stats[keep_cols], on="product_code", how="left")

    # 없는 컬럼 기본값
    for col in ["likes", "shares", "risk_score"]:
        if col not in stats.columns:
            stats[col] = 0
    stats["likes"] = stats["likes"].fillna(0).astype(int)
    stats["shares"] = stats["shares"].fillna(0).astype(int)
    stats["risk_score"] = stats["risk_score"].fillna(0.0)

    # review_count 병합
    stats = stats.merge(review_agg, on="product_code", how="left")
    stats["review_count"] = stats["review_count"].fillna(0).astype(int)

    # engagement_score: 0.15 * likes + 0.30 * shares + 0.55 * review_count
    stats["engagement_score"] = (
        0.15 * stats["likes"] + 0.30 * stats["shares"] + 0.55 * stats["review_count"]
    ).round(2)

    # cp_index: (engagement_score / price) * 1000
    price_map = products_core.set_index("product_code")["price"]
    stats["price"] = stats["product_code"].map(price_map).fillna(0)
    stats["cp_index"] = np.where(
        stats["price"] > 0,
        (stats["engagement_score"] / stats["price"]) * 1000,
        0.0,
    ).round(4)

    # review_density: review_count / (마지막리뷰일 - 첫리뷰일).days (0일은 1일로)
    stats["first_review"] = pd.to_datetime(stats["first_review"])
    stats["last_review"] = pd.to_datetime(stats["last_review"])
    day_span = (stats["last_review"] - stats["first_review"]).dt.days.fillna(0).clip(lower=1)
    stats["review_density"] = (stats["review_count"] / day_span).round(4)
    stats.loc[stats["review_count"] == 0, "review_density"] = 0.0

    # 최종 컬럼 선택
    result = stats[
        [
            "product_code",
            "likes",
            "shares",
            "review_count",
            "engagement_score",
            "cp_index",
            "review_density",
            "risk_score",
        ]
    ].copy()

    return result


def compute_users_profile(
    reviews_core: pd.DataFrame,
) -> pd.DataFrame:
    """
    users_profile 테이블 계산

    Parameters
    ----------
    reviews_core : reviews_core DataFrame (user_id, rating, review_date, is_reorder 필수)

    Returns
    -------
    users_profile DataFrame
    """
    reviews = reviews_core.copy()
    reviews["review_date"] = pd.to_datetime(reviews["review_date"])

    # 기본 집계
    user_agg = (
        reviews.groupby("user_id")
        .agg(
            user_total_reviews=("review_date", "count"),
            first_review=("review_date", "min"),
            last_review=("review_date", "max"),
        )
        .reset_index()
    )

    # user_activity_level: Newbie(1) / Junior(2-5) / Regular(6-20) / VIP(21+)
    conditions = [
        user_agg["user_total_reviews"] == 1,
        user_agg["user_total_reviews"].between(2, 5),
        user_agg["user_total_reviews"].between(6, 20),
        user_agg["user_total_reviews"] >= 21,
    ]
    choices = ["Newbie", "Junior", "Regular", "VIP"]
    user_agg["user_activity_level"] = np.select(conditions, choices, default="Newbie")

    # user_avg_rating_reorder: 재구매 리뷰 평균 평점 (없으면 0)
    reorder_avg = (
        reviews[reviews["is_reorder"] == True]
        .groupby("user_id")["rating"]
        .mean()
        .reset_index()
        .rename(columns={"rating": "user_avg_rating_reorder"})
    )
    user_agg = user_agg.merge(reorder_avg, on="user_id", how="left")
    user_agg["user_avg_rating_reorder"] = user_agg["user_avg_rating_reorder"].fillna(0.0).round(2)

    # user_rating_tendency
    reorder_count = (
        reviews[reviews["is_reorder"] == True]
        .groupby("user_id")
        .size()
        .reset_index(name="reorder_count")
    )
    user_agg = user_agg.merge(reorder_count, on="user_id", how="left")
    user_agg["reorder_count"] = user_agg["reorder_count"].fillna(0).astype(int)

    conditions = [
        user_agg["reorder_count"] == 0,
        user_agg["user_avg_rating_reorder"] < 3.0,
        user_agg["user_avg_rating_reorder"].between(3.0, 4.0, inclusive="both"),
        user_agg["user_avg_rating_reorder"].between(4.0, 4.8, inclusive="left"),
        user_agg["user_avg_rating_reorder"] >= 4.8,
    ]
    choices = ["No Reorder", "Critical", "Balanced", "Mostly Positive", "Always Positive"]
    user_agg["user_rating_tendency"] = np.select(conditions, choices, default="No Reorder")

    # review_tenure: (마지막리뷰일 - 첫리뷰일).days
    user_agg["review_tenure"] = (user_agg["last_review"] - user_agg["first_review"]).dt.days.fillna(0).astype(int)

    result = user_agg[
        [
            "user_id",
            "user_total_reviews",
            "user_activity_level",
            "user_avg_rating_reorder",
            "user_rating_tendency",
            "review_tenure",
        ]
    ].copy()

    return result


def compute_users_repurchase(
    reviews_core: pd.DataFrame,
    products_core: pd.DataFrame,
    products_category: pd.DataFrame,
) -> pd.DataFrame:
    """
    users_repurchase 테이블 계산

    Parameters
    ----------
    reviews_core : reviews_core DataFrame (user_id, product_code 필수)
    products_core : products_core DataFrame (product_code, brand_id 필수)
    products_category : products_category DataFrame (product_code, category_1 필수)

    Returns
    -------
    users_repurchase DataFrame
    """
    reviews = reviews_core[["user_id", "product_code"]].copy()

    # 카테고리 매핑
    cat_map = products_category.set_index("product_code")["category_1"]
    reviews["category_1"] = reviews["product_code"].map(cat_map)

    # 브랜드 매핑
    brand_map = products_core.set_index("product_code")["brand_id"]
    reviews["brand_id"] = reviews["product_code"].map(brand_map)

    # user_category_repurchase: 동일 카테고리 2회 이상 구매한 카테고리의 총 리뷰 합
    user_cat = reviews.groupby(["user_id", "category_1"]).size().reset_index(name="cnt")
    user_cat_repurchase = (
        user_cat[user_cat["cnt"] >= 2]
        .groupby("user_id")["cnt"]
        .sum()
        .reset_index(name="user_category_repurchase")
    )

    # user_brand_repurchase: 동일 브랜드 2회 이상 구매한 브랜드의 총 리뷰 합
    user_brand = reviews.groupby(["user_id", "brand_id"]).size().reset_index(name="cnt")
    user_brand_repurchase = (
        user_brand[user_brand["cnt"] >= 2]
        .groupby("user_id")["cnt"]
        .sum()
        .reset_index(name="user_brand_repurchase")
    )

    # 전체 유저 기준으로 병합
    all_users = reviews[["user_id"]].drop_duplicates()
    result = all_users.merge(user_cat_repurchase, on="user_id", how="left")
    result = result.merge(user_brand_repurchase, on="user_id", how="left")
    result["user_category_repurchase"] = result["user_category_repurchase"].fillna(0).astype(int)
    result["user_brand_repurchase"] = result["user_brand_repurchase"].fillna(0).astype(int)

    return result
