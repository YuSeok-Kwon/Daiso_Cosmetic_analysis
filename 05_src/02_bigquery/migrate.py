#!/usr/bin/env python3
"""
BigQuery 스키마 마이그레이션: 구 스키마(11개) → ERD v2(13개)

전략: 기존 CSV final/ 데이터가 이미 정제되어 있으므로, 이를 직접 업로드.

실행 단계:
  1. 기존 11개 테이블 → _backup_{테이블명} 복사
  2. 기존 11개 테이블 DROP
  3. schema_v2.sql 실행 → 13개 테이블 CREATE
  4. 02_processed_data/csv/final/ 13개 CSV를 각 테이블에 INSERT
  5. 행 수 비교 검증

사용법:
  python migrate.py                # 기본 실행 (확인 프롬프트)
  python migrate.py --dry-run      # SQL만 출력
  python migrate.py --skip-backup  # 백업 건너뛰기
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from bigquery.bq_client import get_client, insert_df, DEFAULT_DATASET, TABLE_KEYS
except ImportError:
    from bq_client import get_client, insert_df, DEFAULT_DATASET, TABLE_KEYS

# 구 스키마 테이블명
LEGACY_TABLES = [
    "brands", "categories", "ingredients_master", "products",
    "product_attributes", "product_metrics", "product_ingredients",
    "users", "reviews", "review_analysis", "promotions",
]

# 신규 스키마 테이블명 (TABLE_KEYS에서 가져옴)
NEW_TABLES = list(TABLE_KEYS.keys())

# CSV final 디렉토리
FINAL_DIR = Path(__file__).resolve().parent.parent.parent / "02_processed_data" / "csv" / "final"

# schema_v2.sql 경로
SCHEMA_SQL = Path(__file__).resolve().parent / "schema_v2.sql"


def backup_tables(client, dataset: str, dry_run: bool = False):
    """기존 테이블을 _backup_ 접두어로 복사"""
    project = client.project
    existing = {t.table_id for t in client.list_tables(f"{project}.{dataset}")}

    for table in LEGACY_TABLES:
        if table not in existing:
            print(f"  [건너뜀] {table} — 존재하지 않음")
            continue

        backup_name = f"_backup_{table}"
        sql = f"""
        CREATE OR REPLACE TABLE `{project}.{dataset}.{backup_name}` AS
        SELECT * FROM `{project}.{dataset}.{table}`
        """

        if dry_run:
            print(f"  [DRY-RUN] {table} → {backup_name}")
        else:
            client.query(sql).result()
            print(f"  {table} → {backup_name} 백업 완료")


def drop_legacy_tables(client, dataset: str, dry_run: bool = False):
    """기존 테이블 DROP"""
    project = client.project
    existing = {t.table_id for t in client.list_tables(f"{project}.{dataset}")}

    for table in LEGACY_TABLES:
        if table not in existing:
            continue

        sql = f"DROP TABLE IF EXISTS `{project}.{dataset}.{table}`"

        if dry_run:
            print(f"  [DRY-RUN] DROP {table}")
        else:
            client.query(sql).result()
            print(f"  DROP {table} 완료")


def create_new_tables(client, dataset: str, dry_run: bool = False):
    """schema_v2.sql 실행하여 13개 테이블 생성"""
    if not SCHEMA_SQL.exists():
        print(f"오류: {SCHEMA_SQL} 파일을 찾을 수 없습니다.")
        sys.exit(1)

    sql_content = SCHEMA_SQL.read_text(encoding="utf-8")

    # 개별 문장으로 분리 (세미콜론 기준)
    statements = [s.strip() for s in sql_content.split(";") if s.strip() and not s.strip().startswith("--")]

    for stmt in statements:
        if dry_run:
            # 테이블명 추출
            import re
            match = re.search(r"`daiso\.(\w+)`", stmt)
            table_name = match.group(1) if match else "?"
            print(f"  [DRY-RUN] CREATE {table_name}")
        else:
            client.query(stmt).result()

    if not dry_run:
        print(f"  {len(statements)}개 테이블 생성 완료")


def load_csv_to_bq(client, dataset: str, dry_run: bool = False):
    """final/ CSV → BigQuery INSERT"""
    results = {}

    for table_name in NEW_TABLES:
        csv_path = FINAL_DIR / f"{table_name}.csv"

        if not csv_path.exists():
            print(f"  [건너뜀] {table_name}.csv — 파일 없음")
            results[table_name] = 0
            continue

        df = pd.read_csv(csv_path)

        if dry_run:
            print(f"  [DRY-RUN] INSERT {table_name}: {len(df)}행")
        else:
            insert_df(df, table_name, dataset, if_exists="replace")
            print(f"  {table_name}: {len(df)}행 INSERT 완료")

        results[table_name] = len(df)

    return results


def verify(client, dataset: str, csv_counts: dict):
    """행 수 비교 검증"""
    project = client.project
    print("\n[검증] CSV ↔ BigQuery 행 수 비교")
    all_ok = True

    for table_name, csv_count in csv_counts.items():
        if csv_count == 0:
            continue

        try:
            sql = f"SELECT COUNT(*) as cnt FROM `{project}.{dataset}.{table_name}`"
            result = client.query(sql).result()
            bq_count = list(result)[0].cnt

            status = "OK" if bq_count == csv_count else "MISMATCH"
            if status == "MISMATCH":
                all_ok = False
            print(f"  {table_name:25s}: CSV={csv_count:>10,}  BQ={bq_count:>10,}  [{status}]")
        except Exception as e:
            print(f"  {table_name:25s}: 검증 실패 — {e}")
            all_ok = False

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="BigQuery 스키마 마이그레이션")
    parser.add_argument("--dry-run", action="store_true", help="SQL만 출력, 실행하지 않음")
    parser.add_argument("--skip-backup", action="store_true", help="백업 단계 건너뛰기")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="BigQuery 데이터셋명")
    args = parser.parse_args()

    print("=" * 60)
    print("BigQuery 스키마 마이그레이션: 구 → ERD v2")
    print(f"데이터셋: {args.dataset}")
    print(f"CSV 소스: {FINAL_DIR}")
    print(f"모드: {'DRY-RUN' if args.dry_run else '실행'}")
    print("=" * 60)

    if not args.dry_run:
        confirm = input("\n마이그레이션을 시작하시겠습니까? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("취소되었습니다.")
            return

    client = get_client()

    # Step 1: 백업
    if not args.skip_backup:
        print(f"\n[Step 1/4] 기존 테이블 백업")
        backup_tables(client, args.dataset, args.dry_run)
    else:
        print(f"\n[Step 1/4] 백업 건너뜀")

    # Step 2: 기존 테이블 DROP
    print(f"\n[Step 2/4] 기존 테이블 DROP")
    drop_legacy_tables(client, args.dataset, args.dry_run)

    # Step 3: 새 테이블 CREATE
    print(f"\n[Step 3/4] 새 테이블 CREATE (schema_v2.sql)")
    create_new_tables(client, args.dataset, args.dry_run)

    # Step 4: CSV → BigQuery INSERT
    print(f"\n[Step 4/4] CSV → BigQuery INSERT")
    csv_counts = load_csv_to_bq(client, args.dataset, args.dry_run)

    # 검증
    if not args.dry_run:
        ok = verify(client, args.dataset, csv_counts)
        print(f"\n{'=' * 60}")
        print(f"마이그레이션 {'완료' if ok else '완료 (일부 불일치)'}")
        print(f"{'=' * 60}")
    else:
        print(f"\n{'=' * 60}")
        print("DRY-RUN 완료 — 실제 변경 없음")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
