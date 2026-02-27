#!/usr/bin/env python3
"""
ABSA 증분 추론 파이프라인

BQ에서 미추론 리뷰를 조회 → KcELECTRA 추론 → review_absa + review_aspects BQ UPSERT

사용법:
    python 05_src/04_pipeline/run_absa_incremental.py
    python 05_src/04_pipeline/run_absa_incremental.py --limit 1000   # 테스트용
    python 05_src/04_pipeline/run_absa_incremental.py --dry-run      # 조회만, 적재 안 함
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
sys.path.insert(0, str(PROJECT_ROOT / "05_src"))
sys.path.insert(0, str(PROJECT_ROOT / "05_src" / "02_bigquery"))
sys.path.insert(0, str(PROJECT_ROOT / "06_analysis" / "03_ABSA"))

try:
    import yaml
except ImportError:
    yaml = None


def load_config() -> dict:
    """config.yaml에서 ABSA 설정 로드"""
    config_path = PROJECT_ROOT / "05_src" / "04_pipeline" / "config.yaml"
    if config_path.exists() and yaml is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("pipeline", {}).get("absa", {})
    return {
        "bundle_path": "06_analysis/03_ABSA/07_models/prod_bundle_stage3a_v1_20260225",
        "version": "stage3a_v2",
    }


def get_unanalyzed_reviews(dataset: str = "daiso", limit: int = None) -> pd.DataFrame:
    """BQ에서 미추론 리뷰 조회 (review_absa에 없는 review_id)"""
    from bq_client import query_to_df

    limit_clause = f"LIMIT {limit}" if limit else ""

    sql = f"""
        SELECT rc.review_id, rt.text
        FROM `{dataset}.reviews_core` rc
        JOIN `{dataset}.reviews_text` rt ON rc.review_id = rt.review_id
        LEFT JOIN `{dataset}.review_absa` ra ON rc.review_id = ra.review_id
        WHERE ra.review_id IS NULL
          AND rt.text IS NOT NULL
          AND TRIM(rt.text) != ''
        ORDER BY rc.review_id
        {limit_clause}
    """

    return query_to_df(sql)


def run_inference(reviews_df: pd.DataFrame, bundle_path: str, batch_size: int = 128) -> pd.DataFrame:
    """KcELECTRA 모델로 추론 실행"""
    from RQ_absa.s8_inference import ABSAInference

    bundle = Path(bundle_path)
    if not bundle.is_absolute():
        bundle = PROJECT_ROOT / bundle

    print(f"  모델 번들 로드: {bundle.name}")
    inferencer = ABSAInference.from_bundle(
        bundle_path=str(bundle),
        batch_size=batch_size,
    )

    print(f"  추론 실행 중 ({len(reviews_df):,}건)...")
    results_df = inferencer.infer_dataframe(reviews_df, text_column="text")

    return results_df


def build_absa_tables(reviews_df: pd.DataFrame, results_df: pd.DataFrame, absa_version: str) -> tuple:
    """추론 결과를 review_absa + review_aspects 테이블로 변환

    Returns
    -------
    (absa_df, aspects_df)
    """
    import ast

    now = datetime.now()

    # aspect_sentiments 파싱
    def parse_aspects(val):
        if pd.isna(val) or val == "[]":
            return []
        try:
            if isinstance(val, str):
                return ast.literal_eval(val)
            return val if isinstance(val, list) else []
        except (ValueError, SyntaxError):
            return []

    results_df["_parsed_aspects"] = results_df["aspect_sentiments"].apply(parse_aspects)

    # review_absa
    absa_df = pd.DataFrame({
        "review_id": reviews_df["review_id"].values,
        "sentiment": results_df["sentiment"].values,
        "sentiment_score": results_df["sentiment_score"].astype(float).values,
        "is_ambiguous": results_df["is_ambiguous"].astype(bool).values,
        "aspect_count": results_df["_parsed_aspects"].apply(len).values,
        "absa_version": absa_version,
        "inferred_at": now,
    })

    # review_aspects (1:N 정규화)
    aspects_rows = []
    for idx, aspects in enumerate(results_df["_parsed_aspects"]):
        rid = reviews_df.iloc[idx]["review_id"]
        for asp in aspects:
            aspects_rows.append({
                "review_id": int(rid),
                "aspect": asp.get("aspect", ""),
                "aspect_sentiment": asp.get("sentiment", ""),
                "aspect_confidence": float(asp.get("confidence", 0.0)),
            })

    aspects_df = pd.DataFrame(aspects_rows) if aspects_rows else pd.DataFrame(
        columns=["review_id", "aspect", "aspect_sentiment", "aspect_confidence"]
    )

    return absa_df, aspects_df


def upload_to_bq(absa_df: pd.DataFrame, aspects_df: pd.DataFrame, dataset: str = "daiso") -> dict:
    """review_absa + review_aspects BQ UPSERT"""
    from bq_client import upsert_df

    results = {}

    print(f"  review_absa UPSERT: {len(absa_df):,}행")
    r1 = upsert_df(absa_df, "review_absa", dataset=dataset)
    results["review_absa"] = r1

    if not aspects_df.empty:
        print(f"  review_aspects UPSERT: {len(aspects_df):,}행")
        r2 = upsert_df(aspects_df, "review_aspects", dataset=dataset)
        results["review_aspects"] = r2
    else:
        results["review_aspects"] = {"total_processed": 0, "status": "skip"}

    return results


def main():
    parser = argparse.ArgumentParser(description="ABSA 증분 추론 파이프라인")
    parser.add_argument("--limit", type=int, default=None, help="추론할 최대 리뷰 수 (테스트용)")
    parser.add_argument("--batch-size", type=int, default=128, help="추론 배치 크기")
    parser.add_argument("--dry-run", action="store_true", help="미추론 리뷰 수만 확인, 실행 안 함")
    parser.add_argument("--dataset", type=str, default="daiso", help="BigQuery 데이터셋")
    args = parser.parse_args()

    config = load_config()
    bundle_path = config.get("bundle_path", "")
    absa_version = config.get("version", "stage3a_v2")

    print("=" * 60)
    print("ABSA 증분 추론 파이프라인")
    print(f"모델 버전: {absa_version}")
    print(f"데이터셋: {args.dataset}")
    print("=" * 60)

    # Step 1: 미추론 리뷰 조회
    print("\n[Step 1] 미추론 리뷰 조회...")
    start = time.time()
    reviews_df = get_unanalyzed_reviews(dataset=args.dataset, limit=args.limit)
    elapsed = time.time() - start
    print(f"  미추론 리뷰: {len(reviews_df):,}건 ({elapsed:.1f}초)")

    if reviews_df.empty:
        print("\n추론할 리뷰가 없습니다. 종료.")
        return {"status": "skip", "reason": "no new reviews", "rows_affected": 0}

    if args.dry_run:
        print(f"\n[DRY-RUN] {len(reviews_df):,}건의 미추론 리뷰 발견. 실행 안 함.")
        return {"status": "dry_run", "rows_affected": len(reviews_df)}

    # Step 2: 추론 실행
    print(f"\n[Step 2] KcELECTRA 추론 ({absa_version})...")
    start = time.time()
    results_df = run_inference(reviews_df, bundle_path, batch_size=args.batch_size)
    elapsed = time.time() - start
    speed = len(reviews_df) / elapsed if elapsed > 0 else 0
    print(f"  추론 완료: {elapsed:.1f}초 ({speed:.1f} reviews/sec)")

    # Step 3: 테이블 변환
    print("\n[Step 3] review_absa + review_aspects 생성...")
    absa_df, aspects_df = build_absa_tables(reviews_df, results_df, absa_version)
    print(f"  review_absa: {len(absa_df):,}행")
    print(f"  review_aspects: {len(aspects_df):,}행 (평균 {len(aspects_df)/len(absa_df):.1f} aspects/review)")

    # 감성 분포 출력
    print(f"\n  감성 분포:")
    for sent, pct in absa_df["sentiment"].value_counts(normalize=True).items():
        print(f"    {sent}: {pct:.1%}")

    # Step 4: BQ UPSERT
    print(f"\n[Step 4] BigQuery UPSERT...")
    start = time.time()
    bq_results = upload_to_bq(absa_df, aspects_df, dataset=args.dataset)
    elapsed = time.time() - start
    print(f"  UPSERT 완료: {elapsed:.1f}초")

    # 완료 요약
    print(f"\n{'=' * 60}")
    print(f"ABSA 증분 추론 완료")
    print(f"  신규 리뷰: {len(absa_df):,}건")
    print(f"  Aspect 레코드: {len(aspects_df):,}건")
    print(f"{'=' * 60}")

    return {
        "status": "success",
        "rows_affected": len(absa_df),
        "aspects_count": len(aspects_df),
    }


if __name__ == "__main__":
    main()
