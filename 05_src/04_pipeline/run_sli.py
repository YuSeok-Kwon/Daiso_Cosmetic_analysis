#!/usr/bin/env python3
"""
SLI (Soft Landing Index) 연착륙 판별 파이프라인

BQ에서 리뷰 시계열 조회 → 4가지 방법론 실행 → 만장일치 투표 → sli_results BQ UPSERT

4가지 방법론:
  1. DTW 클러스터링 (tslearn)
  2. 생존분석 (lifelines Kaplan-Meier)
  3. 규칙 기반 (SLI v2 가중합 점수)
  4. ML (LightGBM OOF)

사용법:
    python 05_src/04_pipeline/run_sli.py
    python 05_src/04_pipeline/run_sli.py --min-reviews 30
    python 05_src/04_pipeline/run_sli.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "05_src"))
sys.path.insert(0, str(PROJECT_ROOT / "05_src" / "02_bigquery"))

try:
    import yaml
except ImportError:
    yaml = None


def load_config() -> dict:
    """config.yaml에서 SLI 설정 로드"""
    config_path = PROJECT_ROOT / "05_src" / "04_pipeline" / "config.yaml"
    if config_path.exists() and yaml is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("pipeline", {}).get("sli", {})
    return {"version": "v1", "min_votes": 3}


# ═══════════════════════════════════════════════════════════
# 데이터 준비
# ═══════════════════════════════════════════════════════════

def load_monthly_reviews(dataset: str = "daiso") -> pd.DataFrame:
    """BQ에서 월별 리뷰 수 시계열 조회"""
    from bq_client import query_to_df

    sql = f"""
        SELECT
            product_code,
            FORMAT_DATE('%Y-%m', review_date) AS month,
            COUNT(*) AS review_count,
            MIN(review_date) AS first_date,
            MAX(review_date) AS last_date
        FROM `{dataset}.reviews_core`
        WHERE review_date IS NOT NULL
        GROUP BY product_code, month
        ORDER BY product_code, month
    """
    return query_to_df(sql)


def load_product_info(dataset: str = "daiso") -> pd.DataFrame:
    """BQ에서 상품 정보 조회"""
    from bq_client import query_to_df

    sql = f"""
        SELECT
            pc.product_code,
            pc.name,
            b.name AS brand_name,
            pc.price,
            cat.category_1,
            cat.category_2
        FROM `{dataset}.products_core` pc
        LEFT JOIN `{dataset}.brands` b ON pc.brand_id = b.brand_id
        LEFT JOIN `{dataset}.products_category` cat ON pc.product_code = cat.product_code
    """
    return query_to_df(sql)


def prepare_timeseries(monthly_df: pd.DataFrame, min_total_reviews: int = 30) -> tuple:
    """시계열 데이터 준비: 상대 월 계산, 최소 리뷰 수 필터링

    Returns
    -------
    (monthly_valid, volume_df)
    """
    # 상품별 첫 리뷰 월 기준 상대 월 계산
    monthly_df["month_dt"] = pd.to_datetime(monthly_df["month"] + "-01")

    first_months = monthly_df.groupby("product_code")["month_dt"].min().reset_index()
    first_months.columns = ["product_code", "first_month"]
    monthly_df = monthly_df.merge(first_months, on="product_code")

    monthly_df["months_since_launch"] = (
        (monthly_df["month_dt"].dt.year - monthly_df["first_month"].dt.year) * 12
        + (monthly_df["month_dt"].dt.month - monthly_df["first_month"].dt.month)
    )

    # 총 리뷰 수 계산 및 필터링
    volume = monthly_df.groupby("product_code")["review_count"].sum().reset_index()
    volume.columns = ["product_code", "total_reviews"]
    volume = volume[volume["total_reviews"] >= min_total_reviews]

    monthly_valid = monthly_df[monthly_df["product_code"].isin(volume["product_code"])].copy()

    return monthly_valid, volume


# ═══════════════════════════════════════════════════════════
# Method 1: DTW 클러스터링
# ═══════════════════════════════════════════════════════════

def run_dtw_clustering(monthly_valid: pd.DataFrame, max_months: int = 12) -> pd.DataFrame:
    """DTW 기반 시계열 클러스터링으로 연착륙 판별"""
    try:
        from tslearn.clustering import TimeSeriesKMeans
        from tslearn.preprocessing import TimeSeriesScalerMeanVariance
        from tslearn.utils import to_time_series_dataset
    except ImportError:
        print("  [경고] tslearn 미설치 — DTW 건너뜀 (모두 False)")
        codes = monthly_valid["product_code"].unique()
        return pd.DataFrame({"product_code": codes, "is_soft_landing_dtw": False})

    # 12개월 시계열 행렬 생성
    pivot = monthly_valid[monthly_valid["months_since_launch"] < max_months].pivot_table(
        index="product_code", columns="months_since_launch",
        values="review_count", fill_value=0
    )

    # 열 정규화
    ts_data = to_time_series_dataset(pivot.values)
    scaler = TimeSeriesScalerMeanVariance()
    ts_scaled = scaler.fit_transform(ts_data)

    # 최적 k 선택 (silhouette score)
    from sklearn.metrics import silhouette_score

    best_k, best_score = 3, -1
    for k in range(3, 8):
        model = TimeSeriesKMeans(n_clusters=k, metric="dtw", max_iter=10, random_state=42)
        labels = model.fit_predict(ts_scaled)
        if len(set(labels)) > 1:
            score = silhouette_score(ts_scaled.reshape(len(ts_scaled), -1), labels)
            if score > best_score:
                best_k, best_score = k, score

    # 최종 클러스터링
    model = TimeSeriesKMeans(n_clusters=best_k, metric="dtw", max_iter=10, random_state=42)
    labels = model.fit_predict(ts_scaled)

    # 클러스터 라벨링: 후반부 유지율이 높은 클러스터 = 연착륙
    cluster_profiles = {}
    for k in range(best_k):
        mask = labels == k
        cluster_ts = pivot.values[mask]
        if len(cluster_ts) > 0:
            mean_ts = cluster_ts.mean(axis=0)
            first_half = mean_ts[:max_months // 2].mean() if len(mean_ts) >= max_months // 2 else 0
            second_half = mean_ts[max_months // 2:].mean() if len(mean_ts) >= max_months // 2 else 0
            ratio = second_half / (first_half + 1e-9)
            cluster_profiles[k] = {"ratio": ratio, "count": int(mask.sum())}

    # 후반부 유지율이 0.5 이상인 클러스터 = 연착륙
    sl_clusters = {k for k, v in cluster_profiles.items() if v["ratio"] >= 0.5}

    result = pd.DataFrame({
        "product_code": pivot.index,
        "is_soft_landing_dtw": [labels[i] in sl_clusters for i in range(len(labels))],
    })

    print(f"  DTW: k={best_k}, 연착륙 클러스터={sl_clusters}, 연착륙={result['is_soft_landing_dtw'].sum()}개")
    return result


# ═══════════════════════════════════════════════════════════
# Method 2: 생존분석
# ═══════════════════════════════════════════════════════════

def run_survival_analysis(monthly_valid: pd.DataFrame, max_months: int = 12) -> pd.DataFrame:
    """Kaplan-Meier 생존분석 기반 연착륙 판별"""
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        print("  [경고] lifelines 미설치 — 생존분석 건너뜀 (모두 False)")
        codes = monthly_valid["product_code"].unique()
        return pd.DataFrame({"product_code": codes, "is_soft_landing_surv": False})

    # 생존 데이터 구축: 피크 대비 30% 이하 연속 2개월 = 사망
    survival_data = []
    for pid, group in monthly_valid.groupby("product_code"):
        group = group.sort_values("months_since_launch")
        counts = group.set_index("months_since_launch")["review_count"]

        peak = counts.max()
        if peak == 0:
            survival_data.append({"product_code": pid, "duration": 1, "event": 1})
            continue

        threshold = peak * 0.3
        consecutive_below = 0
        death_month = None

        for m in range(len(counts)):
            if counts.iloc[m] <= threshold:
                consecutive_below += 1
                if consecutive_below >= 2:
                    death_month = counts.index[m]
                    break
            else:
                consecutive_below = 0

        if death_month is not None:
            survival_data.append({"product_code": pid, "duration": death_month, "event": 1})
        else:
            survival_data.append({"product_code": pid, "duration": counts.index.max() + 1, "event": 0})

    surv_df = pd.DataFrame(survival_data)

    # 상위 25% duration = 연착륙
    q75 = surv_df["duration"].quantile(0.75)
    surv_df["is_soft_landing_surv"] = (surv_df["duration"] > q75) | (surv_df["event"] == 0)

    print(f"  생존분석: threshold_q75={q75:.1f}, 연착륙={surv_df['is_soft_landing_surv'].sum()}개")
    return surv_df[["product_code", "is_soft_landing_surv"]]


# ═══════════════════════════════════════════════════════════
# Method 3: 규칙 기반 (SLI v2 가중합)
# ═══════════════════════════════════════════════════════════

def run_rule_based(monthly_valid: pd.DataFrame, volume_df: pd.DataFrame) -> pd.DataFrame:
    """SLI v2 가중합 점수 기반 연착륙 판별"""
    from sklearn.preprocessing import RobustScaler
    import statsmodels.api as sm

    codes = monthly_valid["product_code"].unique()

    # (1) 4구간 유지력
    def _retention_4phase(group):
        counts = group.set_index("months_since_launch")["review_count"]
        early = counts[(counts.index >= 0) & (counts.index <= 2)].median()
        mid = counts[(counts.index >= 3) & (counts.index <= 5)].median()
        late1 = counts[(counts.index >= 6) & (counts.index <= 8)].median()
        late2 = counts[(counts.index >= 9) & (counts.index <= 12)].median()
        return pd.Series({
            "early": early if pd.notna(early) else 0,
            "mid": mid if pd.notna(mid) else 0,
            "late1": late1 if pd.notna(late1) else 0,
            "late2": late2 if pd.notna(late2) else 0,
        })

    ret = monthly_valid.groupby("product_code").apply(_retention_4phase).reset_index()
    ret["mid_minus_early"] = np.log1p(ret["mid"]) - np.log1p(ret["early"])
    ret["late1_minus_mid"] = np.log1p(ret["late1"]) - np.log1p(ret["mid"])
    ret["late2_minus_late1"] = np.log1p(ret["late2"]) - np.log1p(ret["late1"])

    # (2) 방향성 기울기 (OLS)
    slope_rows = []
    for pid, group in monthly_valid.groupby("product_code"):
        g = group.sort_values("months_since_launch")
        if len(g) < 3:
            slope_rows.append({"product_code": pid, "slope_neg": 0.0})
            continue
        X = sm.add_constant(g["months_since_launch"].values)
        y = g["review_count"].values.astype(float)
        try:
            slope = sm.OLS(y, X).fit().params[1]
        except Exception:
            slope = 0.0
        slope_rows.append({"product_code": pid, "slope_neg": -slope if slope < 0 else 0.0})
    slope_df = pd.DataFrame(slope_rows)

    # (3) 후기 변동성 (CV)
    vol_rows = []
    late_data = monthly_valid[monthly_valid["months_since_launch"] >= 6]
    for pid, group in late_data.groupby("product_code"):
        if len(group) < 4:
            vol_rows.append({"product_code": pid, "late_vol_cv": 0.0})
            continue
        std = group["review_count"].std()
        mean = group["review_count"].mean()
        vol_rows.append({"product_code": pid, "late_vol_cv": std / (mean + 1e-9)})
    vol_df = pd.DataFrame(vol_rows)

    # (4) 규모 기여도
    vol_with_log = volume_df.copy()
    vol_with_log["log_total_reviews"] = np.log1p(vol_with_log["total_reviews"])

    # 병합
    sli = ret.merge(slope_df, on="product_code", how="left")
    sli = sli.merge(vol_df, on="product_code", how="left")
    sli = sli.merge(vol_with_log[["product_code", "log_total_reviews"]], on="product_code", how="left")
    sli = sli.fillna(0)

    # RobustScaler 표준화
    features = ["mid_minus_early", "late1_minus_mid", "late2_minus_late1",
                "log_total_reviews", "slope_neg", "late_vol_cv"]
    scaler = RobustScaler()
    scaled = scaler.fit_transform(sli[features])
    for i, f in enumerate(features):
        sli[f"{f}_z"] = scaled[:, i]

    # SLI v2 가중합
    sli["SLI_v2"] = (
        0.20 * sli["mid_minus_early_z"]
        + 0.25 * sli["late1_minus_mid_z"]
        + 0.35 * sli["late2_minus_late1_z"]
        + 0.15 * sli["log_total_reviews_z"]
        - 0.05 * sli["slope_neg_z"]
        - 0.05 * sli["late_vol_cv_z"]
    )

    sli["is_soft_landing_rule"] = sli["SLI_v2"] > 0

    print(f"  규칙기반: SLI_v2>0 연착륙={sli['is_soft_landing_rule'].sum()}개")
    return sli[["product_code", "is_soft_landing_rule"]]


# ═══════════════════════════════════════════════════════════
# Method 4: ML (LightGBM)
# ═══════════════════════════════════════════════════════════

def run_ml_classifier(
    monthly_valid: pd.DataFrame,
    volume_df: pd.DataFrame,
    pseudo_labels: pd.DataFrame,
) -> pd.DataFrame:
    """LightGBM OOF 기반 연착륙 판별

    Parameters
    ----------
    pseudo_labels : DTW+생존+규칙 3가지 투표 결과 (pseudo-label 생성용)
    """
    try:
        import lightgbm as lgb
        from sklearn.model_selection import StratifiedKFold
    except ImportError:
        print("  [경고] lightgbm 미설치 — ML 건너뜀 (모두 False)")
        codes = monthly_valid["product_code"].unique()
        return pd.DataFrame({"product_code": codes, "is_soft_landing_ml": False, "ml_prob": 0.0})

    # 피처 생성
    feature_rows = []
    for pid, group in monthly_valid.groupby("product_code"):
        g = group.sort_values("months_since_launch")
        counts = g["review_count"].values.astype(float)

        n_months = len(counts)
        early = counts[:3].mean() if n_months >= 3 else counts.mean()
        mid = counts[3:6].mean() if n_months >= 6 else 0
        late = counts[6:].mean() if n_months >= 6 else 0

        total = counts.sum()
        cv = counts.std() / (counts.mean() + 1e-9) if n_months > 1 else 0
        retention_mid = mid / (early + 1e-9)
        retention_late = late / (early + 1e-9)

        feature_rows.append({
            "product_code": pid,
            "ts_early_mean": early,
            "ts_mid_mean": mid,
            "ts_late_mean": late,
            "ts_total": total,
            "ts_cv": cv,
            "ts_retention_mid": retention_mid,
            "ts_retention_late": retention_late,
            "ts_n_months": n_months,
            "log_total": np.log1p(total),
        })

    features_df = pd.DataFrame(feature_rows)

    # pseudo-label 병합 (DTW+생존+규칙 majority vote)
    features_df = features_df.merge(pseudo_labels[["product_code", "pseudo_label"]], on="product_code")

    feature_cols = [c for c in features_df.columns if c not in ["product_code", "pseudo_label"]]
    X = features_df[feature_cols].values
    y = features_df["pseudo_label"].values

    # 5-Fold OOF
    oof_probs = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        model = lgb.LGBMClassifier(
            objective="binary", learning_rate=0.05, num_leaves=16,
            max_depth=5, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            is_unbalance=True, n_estimators=300, random_state=42,
            verbose=-1,
        )
        model.fit(
            X[train_idx], y[train_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        oof_probs[val_idx] = model.predict_proba(X[val_idx])[:, 1]

    features_df["ml_prob"] = oof_probs
    features_df["is_soft_landing_ml"] = oof_probs >= 0.5

    print(f"  ML(LightGBM): 연착륙={features_df['is_soft_landing_ml'].sum()}개, "
          f"평균 prob={oof_probs.mean():.3f}")
    return features_df[["product_code", "is_soft_landing_ml", "ml_prob"]]


# ═══════════════════════════════════════════════════════════
# 통합 투표
# ═══════════════════════════════════════════════════════════

def merge_votes(
    dtw_df: pd.DataFrame,
    surv_df: pd.DataFrame,
    rule_df: pd.DataFrame,
    ml_df: pd.DataFrame,
    min_votes: int = 3,
) -> pd.DataFrame:
    """4가지 방법론 결과를 통합하여 만장일치 투표"""
    result = dtw_df.merge(surv_df, on="product_code", how="outer")
    result = result.merge(rule_df, on="product_code", how="outer")
    result = result.merge(ml_df, on="product_code", how="outer")

    # 결측값 = False
    for col in ["is_soft_landing_dtw", "is_soft_landing_surv", "is_soft_landing_rule", "is_soft_landing_ml"]:
        result[col] = result[col].fillna(False).astype(bool)

    result["total_votes"] = (
        result["is_soft_landing_dtw"].astype(int)
        + result["is_soft_landing_surv"].astype(int)
        + result["is_soft_landing_rule"].astype(int)
        + result["is_soft_landing_ml"].astype(int)
    )

    result["final_soft_landing"] = result["total_votes"] >= min_votes

    # confidence: total_votes / 4
    result["confidence"] = result["total_votes"] / 4.0

    # ml_prob (없으면 0)
    if "ml_prob" not in result.columns:
        result["ml_prob"] = 0.0

    return result


# ═══════════════════════════════════════════════════════════
# BQ 적재
# ═══════════════════════════════════════════════════════════

def upload_to_bq(sli_df: pd.DataFrame, sli_version: str, dataset: str = "daiso") -> dict:
    """sli_results BQ UPSERT"""
    from bq_client import upsert_df

    now = datetime.now()

    upload_df = pd.DataFrame({
        "product_code": sli_df["product_code"].astype(int),
        "is_soft_landing_dtw": sli_df["is_soft_landing_dtw"].astype(bool),
        "is_soft_landing_surv": sli_df["is_soft_landing_surv"].astype(bool),
        "is_soft_landing_rule": sli_df["is_soft_landing_rule"].astype(bool),
        "is_soft_landing_ml": sli_df["is_soft_landing_ml"].astype(bool),
        "total_votes": sli_df["total_votes"].astype(int),
        "final_soft_landing": sli_df["final_soft_landing"].astype(bool),
        "confidence": sli_df["confidence"].astype(float),
        "ml_prob": sli_df["ml_prob"].astype(float),
        "sli_version": sli_version,
        "calculated_at": now,
    })

    print(f"  sli_results UPSERT: {len(upload_df):,}행")
    result = upsert_df(upload_df, "sli_results", dataset=dataset)
    return result


# ═══════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SLI 연착륙 판별 파이프라인")
    parser.add_argument("--min-reviews", type=int, default=30, help="최소 리뷰 수 (기본: 30)")
    parser.add_argument("--min-votes", type=int, default=None, help="최소 투표 수 (기본: config)")
    parser.add_argument("--dry-run", action="store_true", help="계산만 하고 BQ 적재 안 함")
    parser.add_argument("--dataset", type=str, default="daiso", help="BigQuery 데이터셋")
    parser.add_argument("--save-csv", type=str, default=None, help="로컬 CSV로도 저장 (경로)")
    args = parser.parse_args()

    config = load_config()
    sli_version = config.get("version", "v1")
    min_votes = args.min_votes or config.get("min_votes", 3)

    print("=" * 60)
    print("SLI 연착륙 판별 파이프라인")
    print(f"버전: {sli_version}, 최소 투표: {min_votes}")
    print(f"데이터셋: {args.dataset}")
    print("=" * 60)

    # Step 1: 데이터 로드
    print("\n[Step 1] 월별 리뷰 시계열 로드...")
    start = time.time()
    monthly_df = load_monthly_reviews(dataset=args.dataset)
    product_df = load_product_info(dataset=args.dataset)
    elapsed = time.time() - start
    print(f"  {len(monthly_df):,}행 (상품 {monthly_df['product_code'].nunique():,}개), {elapsed:.1f}초")

    # Step 2: 데이터 준비
    print(f"\n[Step 2] 시계열 준비 (최소 리뷰 {args.min_reviews}개)...")
    monthly_valid, volume_df = prepare_timeseries(monthly_df, min_total_reviews=args.min_reviews)
    n_products = monthly_valid["product_code"].nunique()
    print(f"  유효 상품: {n_products}개")

    if n_products == 0:
        print("\n분석 가능한 상품이 없습니다. 종료.")
        return {"status": "skip", "reason": "no valid products"}

    # Step 3: 4가지 방법론 실행
    print("\n[Step 3] 4가지 방법론 실행...")

    print("\n  [3-1] DTW 클러스터링")
    dtw_results = run_dtw_clustering(monthly_valid)

    print("\n  [3-2] 생존분석")
    surv_results = run_survival_analysis(monthly_valid)

    print("\n  [3-3] 규칙 기반 (SLI v2)")
    rule_results = run_rule_based(monthly_valid, volume_df)

    # ML용 pseudo-label (DTW+생존+규칙 majority vote)
    pseudo = dtw_results.merge(surv_results, on="product_code", how="outer")
    pseudo = pseudo.merge(rule_results, on="product_code", how="outer")
    for c in ["is_soft_landing_dtw", "is_soft_landing_surv", "is_soft_landing_rule"]:
        pseudo[c] = pseudo[c].fillna(False).astype(int)
    pseudo["pseudo_label"] = (
        (pseudo["is_soft_landing_dtw"] + pseudo["is_soft_landing_surv"] + pseudo["is_soft_landing_rule"]) >= 2
    ).astype(int)

    print("\n  [3-4] ML (LightGBM)")
    ml_results = run_ml_classifier(monthly_valid, volume_df, pseudo)

    # Step 4: 통합 투표
    print(f"\n[Step 4] 만장일치 투표 (min_votes={min_votes})...")
    sli_df = merge_votes(dtw_results, surv_results, rule_results, ml_results, min_votes=min_votes)

    # 상품 정보 병합
    sli_df = sli_df.merge(product_df, on="product_code", how="left")

    # 결과 요약
    sl_count = sli_df["final_soft_landing"].sum()
    total = len(sli_df)
    print(f"\n  결과: {sl_count}개 연착륙 / {total}개 전체 ({sl_count/total*100:.1f}%)")
    print(f"  투표 분포:")
    for v in range(5):
        cnt = (sli_df["total_votes"] == v).sum()
        if cnt > 0:
            print(f"    {v}표: {cnt}개")

    # Step 5: 저장
    if not args.dry_run:
        print(f"\n[Step 5] BigQuery UPSERT...")
        start = time.time()
        upload_to_bq(sli_df, sli_version, dataset=args.dataset)
        elapsed = time.time() - start
        print(f"  완료: {elapsed:.1f}초")

    if args.save_csv:
        csv_path = Path(args.save_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        sli_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  CSV 저장: {csv_path}")

    print(f"\n{'=' * 60}")
    print(f"SLI 파이프라인 완료: {sl_count}개 연착륙 제품")
    print(f"{'=' * 60}")

    return {
        "status": "success",
        "rows_affected": len(sli_df),
        "soft_landing_count": int(sl_count),
    }


if __name__ == "__main__":
    main()
