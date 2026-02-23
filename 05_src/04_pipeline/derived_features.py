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

    # first_review_date 컬럼 추가
    stats["first_review_date"] = stats["first_review"].dt.strftime("%Y-%m-%d")
    stats.loc[stats["review_count"] == 0, "first_review_date"] = pd.NA

    # 최종 컬럼 선택
    result = stats[
        [
            "product_code",
            "likes",
            "shares",
            "review_count",
            "first_review_date",
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

    is_reorder=True인 리뷰만 대상으로, 동일 유저가 같은 category_2(소분류)
    또는 같은 brand_id에 대해 재구매 리뷰를 2건 이상 작성한 경우 해당 리뷰들을
    1로 표기한 뒤 유저별 합산한다.

    Parameters
    ----------
    reviews_core : reviews_core DataFrame (user_id, product_code, is_reorder, rating 필수)
    products_core : products_core DataFrame (product_code, brand_id 필수)
    products_category : products_category DataFrame (product_code, category_2 필수)

    Returns
    -------
    users_repurchase DataFrame
    """
    # is_reorder=True 리뷰만 대상
    reorder = reviews_core[reviews_core["is_reorder"] == True][
        ["user_id", "product_code", "rating"]
    ].copy()

    # 카테고리 매핑 (category_2 소분류 기준)
    cat_map = products_category.set_index("product_code")["category_2"]
    reorder["category_2"] = reorder["product_code"].map(cat_map)

    # 브랜드 매핑
    brand_map = products_core.set_index("product_code")["brand_id"]
    reorder["brand_id"] = reorder["product_code"].map(brand_map)

    # reorder_user_category: 동일 category_2에 재구매 리뷰 2건 이상 → 해당 리뷰들 합산
    user_cat = reorder.groupby(["user_id", "category_2"]).size().reset_index(name="cnt")
    user_cat_repurchase = (
        user_cat[user_cat["cnt"] >= 2]
        .groupby("user_id")["cnt"]
        .sum()
        .reset_index(name="reorder_user_category")
    )

    # reorder_user_brand: 동일 brand_id에 재구매 리뷰 2건 이상 → 해당 리뷰들 합산
    user_brand = reorder.groupby(["user_id", "brand_id"]).size().reset_index(name="cnt")
    user_brand_repurchase = (
        user_brand[user_brand["cnt"] >= 2]
        .groupby("user_id")["cnt"]
        .sum()
        .reset_index(name="reorder_user_brand")
    )

    # reorder_user_avg_rating: 재구매 리뷰 평균 평점
    reorder_avg = (
        reorder.groupby("user_id")["rating"]
        .mean()
        .round(2)
        .reset_index(name="reorder_user_avg_rating")
    )

    # 재구매 경험 유저만 포함
    reorder_users = reorder[["user_id"]].drop_duplicates()
    result = reorder_users.merge(user_cat_repurchase, on="user_id", how="left")
    result = result.merge(user_brand_repurchase, on="user_id", how="left")
    result = result.merge(reorder_avg, on="user_id", how="left")
    result["reorder_user_category"] = result["reorder_user_category"].fillna(0).astype(int)
    result["reorder_user_brand"] = result["reorder_user_brand"].fillna(0).astype(int)

    return result
