#!/usr/bin/env python3
"""
크롤러 → ERD 파이프라인 오케스트레이션

사용법:
    # 전체 실행 (크롤링 → 변환 → 로컬 저장)
    python 05_src/04_pipeline/run_pipeline.py

    # 변환+저장만 (기존 크롤러 CSV 재사용)
    python 05_src/04_pipeline/run_pipeline.py --skip-crawl \\
        --products data/products.csv --reviews data/reviews.csv --ingredients data/ingredients.csv

    # BigQuery 업로드도 포함
    python 05_src/04_pipeline/run_pipeline.py --upload-bq

    # 로컬만 (BQ 비활성)
    python 05_src/04_pipeline/run_pipeline.py --local-only
"""
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

try:
    import yaml
except ImportError:
    yaml = None
import json
import pandas as pd

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "05_src"))
sys.path.insert(0, str(PIPELINE_DIR))

from transformer import CrawlerToERDTransformer, load_existing_final, TABLE_ORDER
from storage import LocalStorage


def load_config() -> dict:
    """config.yaml 로드"""
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists() and yaml is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    # yaml 미설치 시 기본값 반환
    return {
        "pipeline": {
            "crawling": {"all_categories": True, "headless": True, "crawl_reviews": True, "crawl_ingredients": True},
            "storage": {
                "local": {"csv": True, "parquet": True, "base_dir": "02_processed_data"},
                "bigquery": {"enabled": False, "dataset": "daiso"},
            },
        }
    }


def run_crawling(config: dict) -> tuple:
    """크롤러 실행 → raw CSV 3개 경로 반환"""
    crawl_cfg = config.get("pipeline", {}).get("crawling", {})

    # 크롤러 경로 추가
    crawler_dir = PROJECT_ROOT / "05_src" / "01_crawling"
    sys.path.insert(0, str(crawler_dir))
    os.chdir(crawler_dir)

    from daiso_beauty_crawler import run_all

    products_path, reviews_path, ingredients_path = run_all(
        crawl_reviews=crawl_cfg.get("crawl_reviews", True),
        crawl_ingredients=crawl_cfg.get("crawl_ingredients", True),
        headless=crawl_cfg.get("headless", True),
    )

    # 경로를 절대경로로 변환
    os.chdir(PROJECT_ROOT)
    if products_path:
        products_path = str(crawler_dir / products_path)
    if reviews_path:
        reviews_path = str(crawler_dir / reviews_path)
    if ingredients_path:
        ingredients_path = str(crawler_dir / ingredients_path)

    return products_path, reviews_path, ingredients_path


def run_transform(
    products_csv: str,
    reviews_csv: str,
    ingredients_csv: str,
    final_dir: str,
) -> dict:
    """변환 실행 → 13개 테이블 dict 반환"""
    print("\n[변환] raw CSV 로드 중...")

    products_df = pd.read_csv(products_csv) if products_csv else pd.DataFrame()
    reviews_df = pd.read_csv(reviews_csv) if reviews_csv else pd.DataFrame()
    ingredients_df = pd.read_csv(ingredients_csv) if ingredients_csv else pd.DataFrame()

    print(f"  products: {len(products_df)}행, reviews: {len(reviews_df)}행, ingredients: {len(ingredients_df)}행")

    # 기존 final/ CSV 로드 (ID 유지용)
    existing_data = load_existing_final(final_dir)
    if existing_data:
        print(f"  기존 데이터 로드: {list(existing_data.keys())}")

    # 프로모션 로드
    promo_path = Path(final_dir) / "promotions.csv"
    promotions_df = None
    if promo_path.exists():
        promotions_df = pd.read_csv(promo_path, parse_dates=["start_date", "end_date"])
        print(f"  프로모션 로드: {len(promotions_df)}건")

    # 변환 실행
    print("\n[변환] ERD 13개 테이블 변환 중...")
    transformer = CrawlerToERDTransformer(
        products_df=products_df,
        reviews_df=reviews_df,
        ingredients_df=ingredients_df,
        existing_data=existing_data,
        promotions_df=promotions_df,
    )
    tables = transformer.transform_all()

    # 변환 결과 요약
    print("\n[변환 결과]")
    for name in TABLE_ORDER:
        df = tables.get(name)
        rows = len(df) if df is not None else 0
        print(f"  {name:25s}: {rows:>10,}행")

    return tables


def run_save(tables: dict, config: dict) -> dict:
    """로컬 저장"""
    storage_cfg = config.get("pipeline", {}).get("storage", {}).get("local", {})
    base_dir = str(PROJECT_ROOT / storage_cfg.get("base_dir", "02_processed_data"))

    storage = LocalStorage(base_dir=base_dir)
    results = storage.save_all(
        tables,
        save_csv=storage_cfg.get("csv", True),
        save_parquet=storage_cfg.get("parquet", True),
    )

    print("\n[저장 완료]")
    for name, info in results.items():
        paths = []
        if "csv" in info:
            paths.append("CSV")
        if "parquet" in info:
            paths.append("Parquet")
        print(f"  {name:25s}: {info['rows']:>10,}행 ({', '.join(paths)})")

    return results


def run_upload_bq(tables: dict, config: dict) -> dict:
    """BigQuery 업로드"""
    bq_cfg = config.get("pipeline", {}).get("storage", {}).get("bigquery", {})
    dataset = bq_cfg.get("dataset", "daiso")

    sys.path.insert(0, str(PROJECT_ROOT / "05_src" / "02_bigquery"))
    from etl_loader import CrawlerETLv2

    print(f"\n[BigQuery] {dataset} 데이터셋에 업로드 중...")
    etl = CrawlerETLv2(dataset=dataset)
    results = etl.upload_all(tables)

    print("\n[BigQuery 업로드 완료]")
    for name, info in results.items():
        print(f"  {name:25s}: {info.get('total_processed', 0)}행 ({info.get('status', '?')})")

    return results


def main():
    parser = argparse.ArgumentParser(description="크롤러 → ERD 파이프라인")
    parser.add_argument("--skip-crawl", action="store_true", help="크롤링 단계 건너뛰기")
    parser.add_argument("--products", type=str, help="기존 products CSV 경로")
    parser.add_argument("--reviews", type=str, help="기존 reviews CSV 경로")
    parser.add_argument("--ingredients", type=str, help="기존 ingredients CSV 경로")
    parser.add_argument("--upload-bq", action="store_true", help="BigQuery 업로드 포함")
    parser.add_argument("--local-only", action="store_true", help="로컬 저장만 (BQ 비활성)")
    parser.add_argument("--final-dir", type=str, default=None, help="기존 final/ CSV 디렉토리")
    args = parser.parse_args()

    config = load_config()
    start_time = datetime.now()

    print("=" * 60)
    print("크롤러 → ERD 파이프라인")
    print(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # final 디렉토리 결정
    final_dir = args.final_dir or str(PROJECT_ROOT / "02_processed_data" / "csv" / "final")

    # Step 1: 크롤링
    if not args.skip_crawl:
        print("\n[Step 1] 크롤링 실행...")
        products_csv, reviews_csv, ingredients_csv = run_crawling(config)
    else:
        products_csv = args.products
        reviews_csv = args.reviews
        ingredients_csv = args.ingredients
        print("\n[Step 1] 크롤링 건너뜀 (기존 CSV 사용)")

    if not any([products_csv, reviews_csv, ingredients_csv]):
        print("오류: 입력 CSV가 없습니다. --products, --reviews, --ingredients 옵션을 확인하세요.")
        sys.exit(1)

    # Step 2: 변환
    print("\n[Step 2] 변환 실행...")
    tables = run_transform(products_csv, reviews_csv, ingredients_csv, final_dir)

    # Step 3: 로컬 저장
    print("\n[Step 3] 로컬 저장...")
    run_save(tables, config)

    # Step 4: BigQuery 업로드 (선택)
    if args.upload_bq and not args.local_only:
        print("\n[Step 4] BigQuery 업로드...")
        run_upload_bq(tables, config)
    elif not args.local_only:
        bq_enabled = config.get("pipeline", {}).get("storage", {}).get("bigquery", {}).get("enabled", False)
        if bq_enabled:
            print("\n[Step 4] BigQuery 업로드 (config 설정)...")
            run_upload_bq(tables, config)
        else:
            print("\n[Step 4] BigQuery 업로드 건너뜀 (--upload-bq 미지정)")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'=' * 60}")
    print(f"파이프라인 완료 (소요 시간: {elapsed:.1f}초)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
