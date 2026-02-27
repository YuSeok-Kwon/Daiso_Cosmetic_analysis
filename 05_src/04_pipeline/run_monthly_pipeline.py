#!/usr/bin/env python3
"""
월간 전체 파이프라인 오케스트레이션

6단계를 순차적으로 실행하고, 각 단계의 결과를 pipeline_log에 기록한다.

  Step 1: 증분 크롤링 → raw CSV (실패 시 전체 중단)
  Step 2: 변환 + BQ UPSERT (실패 시 전체 중단)
  Step 3: ABSA 증분 추론 → review_absa + review_aspects (실패 시 다음으로)
  Step 4: SLI 연착륙 재계산 → sli_results (실패 시 다음으로)
  Step 5: 검색트렌드 수집 → search_trends (실패 시 다음으로)
  Step 6: 대시보드 생성 (실패 시 로그만 기록)

사용법:
    python 05_src/04_pipeline/run_monthly_pipeline.py
    python 05_src/04_pipeline/run_monthly_pipeline.py --skip-crawl   # 크롤링 건너뛰기
    python 05_src/04_pipeline/run_monthly_pipeline.py --steps 3,4    # 특정 단계만 실행
    python 05_src/04_pipeline/run_monthly_pipeline.py --dry-run      # 실행 안 하고 계획만 표시
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "05_src"))
sys.path.insert(0, str(PROJECT_ROOT / "05_src" / "02_bigquery"))
sys.path.insert(0, str(PIPELINE_DIR))

try:
    import yaml
except ImportError:
    yaml = None


def load_config() -> dict:
    """config.yaml 로드"""
    config_path = PIPELINE_DIR / "config.yaml"
    if config_path.exists() and yaml is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {"pipeline": {}}


def get_next_run_id(dataset: str = "daiso") -> int:
    """pipeline_log에서 다음 run_id 조회"""
    try:
        from bq_client import query_to_df
        result = query_to_df(
            f"SELECT COALESCE(MAX(run_id), 0) + 1 AS next_id FROM `{dataset}.pipeline_log`"
        )
        return int(result["next_id"].iloc[0])
    except Exception:
        return 1


def log_step(
    run_id: int,
    step_name: str,
    status: str,
    rows_affected: int = 0,
    duration_sec: float = 0.0,
    error_message: str = None,
    meta: dict = None,
    dataset: str = "daiso",
):
    """pipeline_log에 단계 결과 기록"""
    try:
        from bq_client import insert_df
        import json

        log_df = pd.DataFrame([{
            "run_id": run_id,
            "run_date": datetime.now(),
            "step_name": step_name,
            "status": status,
            "rows_affected": rows_affected,
            "duration_sec": round(duration_sec, 2),
            "error_message": error_message,
            "meta": json.dumps(meta, ensure_ascii=False) if meta else None,
        }])

        insert_df(log_df, "pipeline_log", dataset)
    except Exception as e:
        print(f"  [경고] pipeline_log 기록 실패: {e}")


# ═══════════════════════════════════════════════════════════
# Step 함수들
# ═══════════════════════════════════════════════════════════

def step_crawling(config: dict) -> dict:
    """Step 1: 증분 크롤링"""
    from run_pipeline import run_crawling
    products_csv, reviews_csv, ingredients_csv = run_crawling(config)

    return {
        "rows_affected": 0,
        "meta": {
            "products_csv": str(products_csv) if products_csv else None,
            "reviews_csv": str(reviews_csv) if reviews_csv else None,
            "ingredients_csv": str(ingredients_csv) if ingredients_csv else None,
        },
        "csv_paths": (products_csv, reviews_csv, ingredients_csv),
    }


def step_transform(config: dict, csv_paths: tuple = None) -> dict:
    """Step 2: 변환 + BQ UPSERT"""
    from run_pipeline import run_transform, run_upload_bq

    bq_cfg = config.get("pipeline", {}).get("storage", {}).get("bigquery", {})
    dataset = bq_cfg.get("dataset", "daiso")
    final_dir = str(PROJECT_ROOT / "02_processed_data" / "csv" / "final")

    if csv_paths:
        products_csv, reviews_csv, ingredients_csv = csv_paths
    else:
        # 가장 최근 크롤러 CSV 자동 탐색
        crawler_dir = PROJECT_ROOT / "05_src" / "01_crawling"
        products_csv = str(next(crawler_dir.glob("data/products_*.csv"), ""))
        reviews_csv = str(next(crawler_dir.glob("data/reviews_*.csv"), ""))
        ingredients_csv = str(next(crawler_dir.glob("data/ingredients_*.csv"), ""))

    tables = run_transform(
        products_csv, reviews_csv, ingredients_csv, final_dir,
        use_bq=True, bq_dataset=dataset,
    )

    bq_results = run_upload_bq(tables, config)

    total_rows = sum(len(df) for df in tables.values() if isinstance(df, pd.DataFrame))
    return {"rows_affected": total_rows, "meta": {"tables": list(tables.keys())}}


def step_absa(config: dict) -> dict:
    """Step 3: ABSA 증분 추론"""
    from run_absa_incremental import get_unanalyzed_reviews, run_inference, build_absa_tables, upload_to_bq

    absa_cfg = config.get("pipeline", {}).get("absa", {})
    if not absa_cfg.get("enabled", True):
        return {"rows_affected": 0, "meta": {"status": "disabled"}}

    dataset = config.get("pipeline", {}).get("storage", {}).get("bigquery", {}).get("dataset", "daiso")
    bundle_path = absa_cfg.get("bundle_path", "")
    version = absa_cfg.get("version", "stage3a_v2")

    reviews_df = get_unanalyzed_reviews(dataset=dataset)
    if reviews_df.empty:
        return {"rows_affected": 0, "meta": {"status": "no_new_reviews"}}

    results_df = run_inference(reviews_df, bundle_path)
    absa_df, aspects_df = build_absa_tables(reviews_df, results_df, version)
    upload_to_bq(absa_df, aspects_df, dataset=dataset)

    return {
        "rows_affected": len(absa_df),
        "meta": {"aspects_count": len(aspects_df), "version": version},
    }


def step_sli(config: dict) -> dict:
    """Step 4: SLI 연착륙 재계산"""
    from run_sli import (
        load_monthly_reviews, load_product_info, prepare_timeseries,
        run_dtw_clustering, run_survival_analysis, run_rule_based,
        run_ml_classifier, merge_votes, upload_to_bq,
    )

    sli_cfg = config.get("pipeline", {}).get("sli", {})
    if not sli_cfg.get("enabled", True):
        return {"rows_affected": 0, "meta": {"status": "disabled"}}

    dataset = config.get("pipeline", {}).get("storage", {}).get("bigquery", {}).get("dataset", "daiso")
    version = sli_cfg.get("version", "v1")
    min_votes = sli_cfg.get("min_votes", 3)

    monthly_df = load_monthly_reviews(dataset=dataset)
    monthly_valid, volume_df = prepare_timeseries(monthly_df)

    dtw_results = run_dtw_clustering(monthly_valid)
    surv_results = run_survival_analysis(monthly_valid)
    rule_results = run_rule_based(monthly_valid, volume_df)

    # ML pseudo-label
    pseudo = dtw_results.merge(surv_results, on="product_code", how="outer")
    pseudo = pseudo.merge(rule_results, on="product_code", how="outer")
    for c in ["is_soft_landing_dtw", "is_soft_landing_surv", "is_soft_landing_rule"]:
        pseudo[c] = pseudo[c].fillna(False).astype(int)
    pseudo["pseudo_label"] = (
        (pseudo["is_soft_landing_dtw"] + pseudo["is_soft_landing_surv"] + pseudo["is_soft_landing_rule"]) >= 2
    ).astype(int)

    ml_results = run_ml_classifier(monthly_valid, volume_df, pseudo)
    sli_df = merge_votes(dtw_results, surv_results, rule_results, ml_results, min_votes=min_votes)

    upload_to_bq(sli_df, version, dataset=dataset)

    sl_count = int(sli_df["final_soft_landing"].sum())
    return {
        "rows_affected": len(sli_df),
        "meta": {"soft_landing_count": sl_count, "version": version},
    }


def step_search_trend(config: dict) -> dict:
    """Step 5: 검색트렌드 수집"""
    st_cfg = config.get("pipeline", {}).get("search_trend", {})
    if not st_cfg.get("enabled", True):
        return {"rows_affected": 0, "meta": {"status": "disabled"}}

    # 검색트렌드 수집은 기존 스크립트(run_anchor_search_trend.py)를 활용
    # 연착륙 제품 목록을 BQ에서 조회하여 수집
    dataset = config.get("pipeline", {}).get("storage", {}).get("bigquery", {}).get("dataset", "daiso")

    try:
        from bq_client import query_to_df

        sl_products = query_to_df(f"""
            SELECT sr.product_code, pc.name, b.name AS brand_name
            FROM `{dataset}.sli_results` sr
            JOIN `{dataset}.products_core` pc ON sr.product_code = pc.product_code
            JOIN `{dataset}.brands` b ON pc.brand_id = b.brand_id
            WHERE sr.final_soft_landing = TRUE
        """)

        return {
            "rows_affected": len(sl_products),
            "meta": {"status": "product_list_loaded", "sl_count": len(sl_products)},
        }
    except Exception as e:
        return {"rows_affected": 0, "meta": {"status": "error", "message": str(e)}}


def step_dashboard(config: dict) -> dict:
    """Step 6: 대시보드 생성 (placeholder)"""
    dash_cfg = config.get("pipeline", {}).get("dashboard", {})
    if not dash_cfg.get("enabled", True):
        return {"rows_affected": 0, "meta": {"status": "disabled"}}

    # 대시보드 생성은 Phase 4에서 구현 예정
    return {"rows_affected": 0, "meta": {"status": "not_implemented"}}


# ═══════════════════════════════════════════════════════════
# 메인 오케스트레이션
# ═══════════════════════════════════════════════════════════

ALL_STEPS = [
    ("crawling", step_crawling, True),       # (이름, 함수, 핵심단계여부)
    ("transform", step_transform, True),
    ("absa", step_absa, False),
    ("sli", step_sli, False),
    ("search_trend", step_search_trend, False),
    ("dashboard", step_dashboard, False),
]


def main():
    parser = argparse.ArgumentParser(description="월간 파이프라인 오케스트레이션")
    parser.add_argument("--skip-crawl", action="store_true", help="크롤링+변환 건너뛰기")
    parser.add_argument("--steps", type=str, default=None,
                        help="실행할 단계 번호 (쉼표 구분, 예: 3,4,5)")
    parser.add_argument("--dry-run", action="store_true", help="실행 안 하고 계획만 표시")
    parser.add_argument("--dataset", type=str, default="daiso", help="BigQuery 데이터셋")
    args = parser.parse_args()

    config = load_config()
    dataset = args.dataset

    # 실행할 단계 결정
    if args.steps:
        step_indices = {int(s.strip()) - 1 for s in args.steps.split(",")}
        steps_to_run = [(i, name, func, critical) for i, (name, func, critical) in enumerate(ALL_STEPS)
                        if i in step_indices]
    elif args.skip_crawl:
        steps_to_run = [(i, name, func, critical) for i, (name, func, critical) in enumerate(ALL_STEPS)
                        if name not in ("crawling", "transform")]
    else:
        steps_to_run = [(i, name, func, critical) for i, (name, func, critical) in enumerate(ALL_STEPS)]

    print("=" * 60)
    print("월간 파이프라인 오케스트레이션")
    print(f"시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"데이터셋: {dataset}")
    print(f"실행 단계: {', '.join(name for _, name, _, _ in steps_to_run)}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY-RUN] 다음 단계를 실행할 예정:")
        for i, name, func, critical in steps_to_run:
            label = "[핵심]" if critical else "[분석]"
            print(f"  Step {i+1}: {name} {label}")
        return

    # run_id 할당
    run_id = get_next_run_id(dataset)
    print(f"\nRun ID: {run_id}")

    pipeline_start = time.time()
    csv_paths = None
    results = {}

    for step_idx, step_name, step_func, is_critical in steps_to_run:
        print(f"\n{'─' * 40}")
        print(f"[Step {step_idx + 1}] {step_name}")
        print(f"{'─' * 40}")

        start = time.time()
        try:
            # crawling 결과를 transform에 전달
            if step_name == "transform" and csv_paths:
                result = step_func(config, csv_paths=csv_paths)
            elif step_name == "crawling":
                result = step_func(config)
                csv_paths = result.get("csv_paths")
            else:
                result = step_func(config)

            elapsed = time.time() - start
            rows = result.get("rows_affected", 0)

            log_step(run_id, step_name, "success", rows, elapsed, meta=result.get("meta"), dataset=dataset)
            results[step_name] = {"status": "success", "rows": rows, "elapsed": elapsed}
            print(f"  완료: {rows:,}행, {elapsed:.1f}초")

        except Exception as e:
            elapsed = time.time() - start
            error_msg = str(e)

            log_step(run_id, step_name, "fail", 0, elapsed, error_message=error_msg, dataset=dataset)
            results[step_name] = {"status": "fail", "error": error_msg, "elapsed": elapsed}
            print(f"  실패: {error_msg}")

            if is_critical:
                print(f"\n[중단] 핵심 단계 '{step_name}' 실패 — 파이프라인 중단")
                break

            print(f"  → 다음 단계로 계속 진행")

    # 최종 요약
    total_elapsed = time.time() - pipeline_start
    print(f"\n{'=' * 60}")
    print(f"파이프라인 완료 (Run ID: {run_id}, 소요: {total_elapsed:.1f}초)")
    print(f"{'=' * 60}")

    for name, info in results.items():
        status_icon = "O" if info["status"] == "success" else "X"
        rows = info.get("rows", 0)
        elapsed = info.get("elapsed", 0)
        print(f"  [{status_icon}] {name:20s}: {rows:>10,}행 ({elapsed:.1f}초)")

    return results


if __name__ == "__main__":
    main()
