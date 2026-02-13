"""
BigQuery 연동 모듈

사용법:
    from BigQuery import query_to_df, insert_df, upsert_df, CrawlerETL

    # 쿼리 실행
    df = query_to_df("SELECT * FROM daiso.products LIMIT 10")

    # 데이터 적재
    insert_df(df, "products")

    # 크롤러 데이터 ETL
    etl = CrawlerETL()
    etl.load_products("data/products.csv")
"""
from .bq_client import (
    get_client,
    query_to_df,
    list_tables,
    get_table_schema,
    preview_table,
    get_table_count,
    insert_df,
    upsert_df,
    delete_by_keys,
    truncate_table,
    DEFAULT_DATASET,
    TABLE_KEYS,
)
from .etl_loader import CrawlerETL, load_crawler_data

__all__ = [
    # 클라이언트
    "get_client",
    "query_to_df",
    "list_tables",
    "get_table_schema",
    "preview_table",
    "get_table_count",
    "insert_df",
    "upsert_df",
    "delete_by_keys",
    "truncate_table",
    "DEFAULT_DATASET",
    "TABLE_KEYS",
    # ETL
    "CrawlerETL",
    "load_crawler_data",
]
