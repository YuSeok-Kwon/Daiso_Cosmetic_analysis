"""
BigQuery 클라이언트 모듈

사용법:
    from bq_client import query_to_df, insert_df, upsert_df, list_tables

    # 조회
    df = query_to_df("SELECT * FROM daiso.products LIMIT 10")

    # 적재 (append)
    insert_df(df, "products")

    # 적재 (upsert - 중복시 업데이트)
    upsert_df(df, "products", key_columns=["product_code"])
"""
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
from pathlib import Path
from typing import List, Optional

# 설정 (config 폴더에서 키 파일 로드)
KEY_PATH = Path(__file__).parent.parent.parent / "config" / "daiso-analysis-4d05c813a295.json"
DEFAULT_DATASET = "daiso"

# 테이블별 Primary Key 매핑 (ERD v3 — 18개 테이블)
TABLE_KEYS = {
    # 기존 13개
    "brands": ["brand_id"],
    "manufacturer": ["manufacturer_id"],
    "ingredients_dic": ["ingredient_id"],
    "products_core": ["product_code"],
    "products_stats": ["product_code"],
    "products_category": ["product_code"],
    "products_ingredients": ["product_code", "ingredient_id"],
    "functional": ["product_code"],
    "promotions": ["promotion_id"],
    "reviews_core": ["review_id"],
    "reviews_text": ["review_id"],
    "user_id_map": ["user_masked"],
    "users_profile": ["user_id"],
    "users_repurchase": ["user_id"],

    # 신규 5개 (v3)
    "review_absa": ["review_id"],
    "review_aspects": ["review_id", "aspect"],
    "sli_results": ["product_code"],
    "search_trends": ["product_code", "period", "source"],
    "pipeline_log": ["run_id"],
}

# 구 스키마 매핑 (하위 호환)
TABLE_KEYS_LEGACY = {
    "brands": ["brand_id"],
    "categories": ["category_id"],
    "ingredients_master": ["ingredient_id"],
    "products": ["product_code"],
    "product_attributes": ["product_code"],
    "product_metrics": ["product_code"],
    "product_ingredients": ["product_code", "ingredient_id"],
    "users": ["user_id"],
    "reviews": ["order_id"],
    "review_analysis": ["order_id"],
}


def get_client() -> bigquery.Client:
    """BigQuery 클라이언트 반환"""
    credentials = service_account.Credentials.from_service_account_file(str(KEY_PATH))
    return bigquery.Client(credentials=credentials, project=credentials.project_id)


def query_to_df(sql: str) -> pd.DataFrame:
    """SQL 쿼리 실행 후 DataFrame 반환"""
    client = get_client()
    return client.query(sql).to_dataframe()


def list_tables(dataset: str = DEFAULT_DATASET) -> list:
    """데이터셋의 테이블 목록 반환"""
    client = get_client()
    tables = client.list_tables(f"{client.project}.{dataset}")
    return [t.table_id for t in tables]


def get_table_schema(table: str, dataset: str = DEFAULT_DATASET) -> pd.DataFrame:
    """테이블 스키마 정보 반환"""
    client = get_client()
    table_ref = client.get_table(f"{client.project}.{dataset}.{table}")
    schema_data = [
        {"컬럼명": field.name, "타입": field.field_type, "모드": field.mode}
        for field in table_ref.schema
    ]
    return pd.DataFrame(schema_data)


def preview_table(table: str, dataset: str = DEFAULT_DATASET, limit: int = 10) -> pd.DataFrame:
    """테이블 미리보기"""
    sql = f"SELECT * FROM `{dataset}.{table}` LIMIT {limit}"
    return query_to_df(sql)


def get_table_count(table: str, dataset: str = DEFAULT_DATASET) -> int:
    """테이블 행 수 반환"""
    sql = f"SELECT COUNT(*) as cnt FROM `{dataset}.{table}`"
    return int(query_to_df(sql)['cnt'].iloc[0])


def insert_df(
    df: pd.DataFrame,
    table: str,
    dataset: str = DEFAULT_DATASET,
    if_exists: str = "append"
) -> int:
    """
    DataFrame을 BigQuery 테이블에 적재

    Args:
        df: 적재할 DataFrame
        table: 테이블명
        dataset: 데이터셋명
        if_exists: "append" (추가) 또는 "replace" (전체 교체)

    Returns:
        적재된 행 수
    """
    client = get_client()
    table_id = f"{client.project}.{dataset}.{table}"

    job_config = bigquery.LoadJobConfig(
        write_disposition=(
            bigquery.WriteDisposition.WRITE_TRUNCATE
            if if_exists == "replace"
            else bigquery.WriteDisposition.WRITE_APPEND
        )
    )

    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()

    return len(df)


def upsert_df(
    df: pd.DataFrame,
    table: str,
    key_columns: Optional[List[str]] = None,
    dataset: str = DEFAULT_DATASET
) -> dict:
    """
    DataFrame을 BigQuery 테이블에 UPSERT (중복시 업데이트, 없으면 삽입)

    Args:
        df: 적재할 DataFrame
        table: 테이블명
        key_columns: PK 컬럼 리스트 (없으면 TABLE_KEYS에서 자동 조회)
        dataset: 데이터셋명

    Returns:
        {"inserted": n, "updated": m}
    """
    if df.empty:
        return {"inserted": 0, "updated": 0}

    client = get_client()
    project = client.project

    # PK 컬럼 결정
    if key_columns is None:
        key_columns = TABLE_KEYS.get(table)
        if key_columns is None:
            raise ValueError(f"테이블 '{table}'의 key_columns을 지정해주세요.")

    # 대상 테이블 스키마 조회 → NULL-only 컬럼에만 적용 (타입 추론 불가 방지)
    temp_table = f"{project}.{dataset}._temp_{table}"
    null_only_cols = set(col for col in df.columns if df[col].isna().all())

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
    )

    if null_only_cols:
        try:
            target_ref = f"{project}.{dataset}.{table}"
            target_schema = client.get_table(target_ref).schema
            # NULL-only 컬럼에 대해서만 대상 스키마 적용
            schema_fields = [f for f in target_schema if f.name in null_only_cols]
            if schema_fields:
                job_config.schema = schema_fields
        except Exception:
            pass

    job = client.load_table_from_dataframe(df, temp_table, job_config=job_config)
    job.result()

    # 대상 테이블 스키마로 CAST 매핑 구성 (타입 불일치 자동 해결)
    target_table = f"`{project}.{dataset}.{table}`"
    source_table = f"`{temp_table}`"

    # 스키마 타입 → SQL CAST 타입 매핑
    _BQ_TYPE_MAP = {
        "FLOAT": "FLOAT64", "INTEGER": "INT64", "BOOLEAN": "BOOL",
    }

    cast_map = {}  # col → "CAST(S.col AS TYPE)" 매핑
    target_col_set = None  # 대상 테이블에 존재하는 컬럼 집합
    try:
        target_ref = f"{project}.{dataset}.{table}"
        target_schema = {f.name: f.field_type for f in client.get_table(target_ref).schema}
        target_col_set = set(target_schema.keys())
        temp_schema = {f.name: f.field_type for f in client.get_table(temp_table).schema}
        for col in df.columns:
            if col in target_schema and col in temp_schema:
                if target_schema[col] != temp_schema[col]:
                    sql_type = _BQ_TYPE_MAP.get(target_schema[col], target_schema[col])
                    cast_map[col] = f"CAST(S.{col} AS {sql_type})"
    except Exception:
        pass

    # DataFrame 컬럼 중 대상 테이블에 존재하는 것만 사용
    if target_col_set is not None:
        merge_columns = [col for col in df.columns if col in target_col_set]
    else:
        merge_columns = list(df.columns)

    def _src(col):
        return cast_map.get(col, f"S.{col}")

    # JOIN 조건
    join_conditions = " AND ".join([f"T.{col} = {_src(col)}" for col in key_columns])

    # UPDATE SET 절 (PK 제외, 대상 테이블에 있는 컬럼만)
    update_columns = [col for col in merge_columns if col not in key_columns]
    if update_columns:
        update_set = ", ".join([f"T.{col} = {_src(col)}" for col in update_columns])
        update_clause = f"WHEN MATCHED THEN UPDATE SET {update_set}"
    else:
        update_clause = ""

    # INSERT 절 (대상 테이블에 있는 컬럼만)
    all_columns = ", ".join(merge_columns)
    insert_values = ", ".join([_src(col) for col in merge_columns])

    merge_sql = f"""
    MERGE {target_table} T
    USING {source_table} S
    ON {join_conditions}
    {update_clause}
    WHEN NOT MATCHED THEN
        INSERT ({all_columns})
        VALUES ({insert_values})
    """

    # MERGE 실행
    job = client.query(merge_sql)
    result = job.result()

    # 임시 테이블 삭제
    client.delete_table(temp_table, not_found_ok=True)

    # 결과 반환 (BigQuery는 MERGE 결과를 직접 반환하지 않음)
    return {"total_processed": len(df), "status": "success"}


def delete_by_keys(
    table: str,
    key_values: dict,
    dataset: str = DEFAULT_DATASET
) -> int:
    """
    특정 키 값으로 행 삭제

    Args:
        table: 테이블명
        key_values: {컬럼명: 값} 딕셔너리
        dataset: 데이터셋명

    Returns:
        삭제된 행 수 (추정)
    """
    client = get_client()

    conditions = " AND ".join([
        f"{col} = {repr(val)}" for col, val in key_values.items()
    ])

    sql = f"DELETE FROM `{dataset}.{table}` WHERE {conditions}"
    job = client.query(sql)
    job.result()

    return job.num_dml_affected_rows or 0


def truncate_table(table: str, dataset: str = DEFAULT_DATASET) -> None:
    """테이블 전체 삭제 (구조 유지)"""
    client = get_client()
    sql = f"TRUNCATE TABLE `{dataset}.{table}`"
    job = client.query(sql)
    job.result()
    print(f"테이블 {table} truncate 완료")


if __name__ == "__main__":
    print("=== BigQuery 클라이언트 테스트 ===")
    print(f"데이터셋: {DEFAULT_DATASET}")
    print(f"테이블 목록: {list_tables()}")

    for table, count in [(t, get_table_count(t)) for t in list_tables()]:
        print(f"  - {table}: {count:,}행")
