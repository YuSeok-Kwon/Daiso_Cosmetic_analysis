#!/usr/bin/env python3
"""
BigQuery 스키마 마이그레이션: ERD v2(13개) → ERD v3(18개)

신규 5개 테이블 생성 + 기존 CSV에서 초기 데이터 적재:
  - review_absa: ABSA 추론 결과 CSV → 리뷰 레벨 감성
  - review_aspects: ABSA 추론 결과 CSV → Aspect 레벨 감성 (1:N 정규화)
  - sli_results: SLI 통합 결과 CSV → 연착륙 판별 결과
  - search_trends: 빈 테이블 생성만 (추후 수집 시 적재)
  - pipeline_log: 빈 테이블 생성만

사용법:
  python migrate_v3.py                # 기본 실행
  python migrate_v3.py --dry-run      # SQL만 출력
  python migrate_v3.py --skip-load    # 테이블 생성만, 데이터 적재 건너뛰기
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from bigquery.bq_client import get_client, insert_df, upsert_df, DEFAULT_DATASET
except ImportError:
    from bq_client import get_client, insert_df, upsert_df, DEFAULT_DATASET

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 소스 데이터 경로
ABSA_CSV = PROJECT_ROOT / "02_outputs" / "ABSA" / "inference" / "absa_results_stage4b_rgx_full.csv"
SLI_CSV = PROJECT_ROOT / "02_outputs" / "Sli" / "sli_integrated_results.csv"

# schema_v3.sql 경로
SCHEMA_SQL = Path(__file__).resolve().parent / "schema_v3.sql"

# v3 신규 테이블
NEW_TABLES_V3 = ["review_absa", "review_aspects", "sli_results", "search_trends", "pipeline_log"]


def create_new_tables(client, dataset: str, dry_run: bool = False):
    """schema_v3.sql에서 신규 5개 테이블만 생성"""
    if not SCHEMA_SQL.exists():
        print(f"오류: {SCHEMA_SQL} 파일을 찾을 수 없습니다.")
        sys.exit(1)

    sql_content = SCHEMA_SQL.read_text(encoding="utf-8")
    statements = [s.strip() for s in sql_content.split(";") if s.strip() and not s.strip().startswith("--")]

    created = 0
    for stmt in statements:
        # 신규 테이블만 필터
        match = re.search(r"`daiso\.(\w+)`", stmt)
        if not match:
            continue
        table_name = match.group(1)
        if table_name not in NEW_TABLES_V3:
            continue

        if dry_run:
            print(f"  [DRY-RUN] CREATE {table_name}")
        else:
            client.query(stmt).result()
            print(f"  CREATE {table_name} 완료")
        created += 1

    return created


def load_absa_to_bq(client, dataset: str, dry_run: bool = False) -> dict:
    """ABSA 추론 CSV → review_absa + review_aspects 적재"""
    if not ABSA_CSV.exists():
        print(f"  [건너뜀] ABSA CSV 없음: {ABSA_CSV}")
        return {"review_absa": 0, "review_aspects": 0}

    print(f"  ABSA CSV 로드 중: {ABSA_CSV.name}")
    df = pd.read_csv(ABSA_CSV)
    print(f"  총 {len(df):,}행 로드됨")

    # ── review_absa 테이블 생성 ──
    now = datetime.now()

    # aspect_sentiments 파싱하여 aspect_count 계산
    def parse_aspect_sentiments(val):
        """aspect_sentiments 문자열을 파이썬 리스트로 파싱"""
        if pd.isna(val) or val == "[]":
            return []
        try:
            return ast.literal_eval(str(val))
        except (ValueError, SyntaxError):
            return []

    df["_parsed_aspects"] = df["aspect_sentiments"].apply(parse_aspect_sentiments)
    df["_aspect_count"] = df["_parsed_aspects"].apply(len)

    absa_df = pd.DataFrame({
        "review_id": df["review_id"].astype(int),
        "sentiment": df["sentiment"],
        "sentiment_score": df["sentiment_score"].astype(float),
        "is_ambiguous": df["is_ambiguous"].astype(bool),
        "aspect_count": df["_aspect_count"].astype(int),
        "absa_version": "stage4b_rgx",
        "inferred_at": now,
    })

    if dry_run:
        print(f"  [DRY-RUN] review_absa: {len(absa_df):,}행")
    else:
        insert_df(absa_df, "review_absa", dataset, if_exists="replace")
        print(f"  review_absa: {len(absa_df):,}행 INSERT 완료")

    # ── review_aspects 테이블 생성 (1:N 정규화) ──
    aspects_rows = []
    for _, row in df.iterrows():
        for asp in row["_parsed_aspects"]:
            aspects_rows.append({
                "review_id": int(row["review_id"]),
                "aspect": asp.get("aspect", ""),
                "aspect_sentiment": asp.get("sentiment", ""),
                "aspect_confidence": float(asp.get("confidence", 0.0)),
            })

    aspects_df = pd.DataFrame(aspects_rows)

    if dry_run:
        print(f"  [DRY-RUN] review_aspects: {len(aspects_df):,}행")
    else:
        if not aspects_df.empty:
            insert_df(aspects_df, "review_aspects", dataset, if_exists="replace")
            print(f"  review_aspects: {len(aspects_df):,}행 INSERT 완료")
        else:
            print(f"  review_aspects: 0행 (aspect 데이터 없음)")

    return {"review_absa": len(absa_df), "review_aspects": len(aspects_df)}


def load_sli_to_bq(client, dataset: str, dry_run: bool = False) -> dict:
    """SLI 통합 결과 CSV → sli_results 적재"""
    if not SLI_CSV.exists():
        print(f"  [건너뜀] SLI CSV 없음: {SLI_CSV}")
        return {"sli_results": 0}

    print(f"  SLI CSV 로드 중: {SLI_CSV.name}")
    df = pd.read_csv(SLI_CSV)
    print(f"  총 {len(df):,}행 로드됨")

    now = datetime.now()

    sli_df = pd.DataFrame({
        "product_code": df["product_code"].astype(int),
        "is_soft_landing_dtw": df["is_soft_landing_dtw"].astype(bool),
        "is_soft_landing_surv": df["is_soft_landing_surv"].astype(bool),
        "is_soft_landing_rule": df["is_soft_landing_rule"].astype(bool),
        "is_soft_landing_ml": df["is_soft_landing_ml"].astype(bool),
        "total_votes": df["total_votes"].astype(int),
        "final_soft_landing": df["final_soft_landing"].astype(bool),
        "confidence": None,  # 기존 CSV에는 텍스트("SL (만장일치)")이므로 NULL
        "ml_prob": df["ml_prob"].astype(float),
        "sli_version": "v1",
        "calculated_at": now,
    })

    # confidence를 total_votes 기반으로 계산 (0~1)
    sli_df["confidence"] = sli_df["total_votes"] / 4.0

    if dry_run:
        print(f"  [DRY-RUN] sli_results: {len(sli_df):,}행")
    else:
        insert_df(sli_df, "sli_results", dataset, if_exists="replace")
        print(f"  sli_results: {len(sli_df):,}행 INSERT 완료")

    return {"sli_results": len(sli_df)}


def verify(client, dataset: str, counts: dict):
    """적재 결과 검증"""
    project = client.project
    print("\n[검증] 적재 행 수 확인")
    all_ok = True

    for table_name, expected in counts.items():
        if expected == 0:
            continue
        try:
            sql = f"SELECT COUNT(*) as cnt FROM `{project}.{dataset}.{table_name}`"
            result = client.query(sql).result()
            bq_count = list(result)[0].cnt

            status = "OK" if bq_count == expected else "MISMATCH"
            if status == "MISMATCH":
                all_ok = False
            print(f"  {table_name:25s}: 예상={expected:>10,}  BQ={bq_count:>10,}  [{status}]")
        except Exception as e:
            print(f"  {table_name:25s}: 검증 실패 — {e}")
            all_ok = False

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="BigQuery 스키마 마이그레이션 v2 → v3")
    parser.add_argument("--dry-run", action="store_true", help="SQL만 출력, 실행하지 않음")
    parser.add_argument("--skip-load", action="store_true", help="테이블 생성만, 데이터 적재 건너뛰기")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="BigQuery 데이터셋명")
    args = parser.parse_args()

    print("=" * 60)
    print("BigQuery 스키마 마이그레이션: ERD v2 → v3")
    print(f"데이터셋: {args.dataset}")
    print(f"신규 테이블: {', '.join(NEW_TABLES_V3)}")
    print(f"모드: {'DRY-RUN' if args.dry_run else '실행'}")
    print("=" * 60)

    if not args.dry_run:
        confirm = input("\n마이그레이션을 시작하시겠습니까? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("취소되었습니다.")
            return

    client = get_client()

    # Step 1: 신규 테이블 CREATE
    print(f"\n[Step 1/3] 신규 5개 테이블 CREATE (schema_v3.sql)")
    created = create_new_tables(client, args.dataset, args.dry_run)
    print(f"  → {created}개 테이블 생성")

    if args.skip_load:
        print("\n데이터 적재 건너뜀 (--skip-load)")
        print("=" * 60)
        return

    # Step 2: 초기 데이터 적재
    counts = {}

    print(f"\n[Step 2/3] ABSA 추론 결과 → review_absa + review_aspects")
    absa_counts = load_absa_to_bq(client, args.dataset, args.dry_run)
    counts.update(absa_counts)

    print(f"\n[Step 2/3] SLI 결과 → sli_results")
    sli_counts = load_sli_to_bq(client, args.dataset, args.dry_run)
    counts.update(sli_counts)

    # Step 3: 검증
    if not args.dry_run:
        ok = verify(client, args.dataset, counts)
        print(f"\n{'=' * 60}")
        print(f"마이그레이션 {'완료' if ok else '완료 (일부 불일치)'}")
        sl_count = sli_counts.get("sli_results", 0)
        if sl_count > 0:
            print(f"  SLI: {sl_count}개 제품")
        absa_count = absa_counts.get("review_absa", 0)
        asp_count = absa_counts.get("review_aspects", 0)
        if absa_count > 0:
            print(f"  ABSA: {absa_count:,}개 리뷰, {asp_count:,}개 aspect")
        print(f"{'=' * 60}")
    else:
        print(f"\n{'=' * 60}")
        print("DRY-RUN 완료 — 실제 변경 없음")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
