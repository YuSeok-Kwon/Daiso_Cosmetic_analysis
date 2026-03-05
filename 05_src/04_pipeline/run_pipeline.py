#!/usr/bin/env python3
"""
크롤러 → ERD 파이프라인 오케스트레이션

사용법:
    # 전체 실행 (증분 크롤링 → 변환 → 로컬 저장)
    python 05_src/04_pipeline/run_pipeline.py

    # 강제 풀 크롤링
    python 05_src/04_pipeline/run_pipeline.py --full

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

from transformer import CrawlerToERDTransformer, load_existing_final, load_existing_from_bq, TABLE_ORDER
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
            "crawling": {
                "mode": "incremental",
                "headless": True,
                "crawl_reviews": True,
                "crawl_ingredients": True,
                "history_file": "05_src/01_crawling/cache/crawl_history.json",
            },
            "storage": {
                "local": {"csv": True, "parquet": True, "base_dir": "02_processed_data"},
                "bigquery": {"enabled": False, "dataset": "daiso"},
            },
        }
    }


def _parse_active_categories(active_categories: dict | None) -> dict | None:
    """config.yaml의 active_categories를 파싱하여 categories_filter dict로 변환

    Parameters
    ----------
    active_categories : yaml에서 읽은 dict 또는 None
        예: {"스킨케어": ["all"], "메이크업": ["베이스메이크업"]}

    Returns
    -------
    dict | None
        {중분류명: [소분류명...]} 또는 None (전체 카테고리)
    """
    if not active_categories:
        return None

    # 크롤러의 카테고리 설정 로드 (검증용)
    crawler_dir = PROJECT_ROOT / "05_src" / "01_crawling"
    sys.path.insert(0, str(crawler_dir))
    from config import DAISO_BEAUTY_CATEGORIES

    result = {}
    for middle_name, sub_list in active_categories.items():
        if middle_name not in DAISO_BEAUTY_CATEGORIES:
            print(f"  [경고] config.yaml의 '{middle_name}'은 DAISO_BEAUTY_CATEGORIES에 없습니다. 무시합니다.")
            continue

        if not isinstance(sub_list, list):
            sub_list = [sub_list]

        # "all"이 아닌 개별 소분류 검증
        if "all" not in sub_list:
            valid_sub_names = set(DAISO_BEAUTY_CATEGORIES[middle_name]["소분류"].values())
            for sub in sub_list:
                if sub not in valid_sub_names:
                    print(f"  [경고] '{middle_name}' > '{sub}'는 유효한 소분류가 아닙니다. 무시합니다.")
            sub_list = [s for s in sub_list if s in valid_sub_names]

        if sub_list:
            result[middle_name] = sub_list

    return result if result else None


def _load_history(config: dict, crawler_dir) -> "CrawlHistory":
    """증분 크롤링용 이력 로드 (BQ → 로컬 CSV 폴백)"""
    from crawl_history import CrawlHistory

    crawl_cfg = config.get("pipeline", {}).get("crawling", {})
    history_file = crawl_cfg.get("history_file", "cache/crawl_history.json")
    history_path = str(PROJECT_ROOT / history_file)

    history = CrawlHistory(history_path)
    if history.product_count == 0:
        try:
            sys.path.insert(0, str(PROJECT_ROOT / "05_src" / "02_bigquery"))
            from bq_client import query_to_df
            bq_dataset = config.get("pipeline", {}).get("storage", {}).get("bigquery", {}).get("dataset", "daiso")
            bq_reviews = query_to_df(
                f"SELECT CAST(product_code AS STRING) AS product_code, "
                f"MAX(CAST(review_date AS STRING)) AS max_date "
                f"FROM `{bq_dataset}.reviews_core` GROUP BY product_code"
            )
            for _, row in bq_reviews.iterrows():
                history.update_product(row["product_code"], review_date=row["max_date"])
            history.save()
            print(f"  크롤링 이력 BQ 부트스트래핑: {len(bq_reviews)}개 제품")
        except Exception:
            reviews_core_path = str(PROJECT_ROOT / "02_processed_data" / "csv" / "final" / "reviews_core.csv")
            history = CrawlHistory.from_existing_csv(history_path, reviews_core_path)

    return history


def run_crawling(config: dict, force_full: bool = False) -> tuple:
    """크롤러 실행 → raw CSV 3개 경로 반환

    strategy에 따라 동작 방식이 달라진다:
    - scroll: 기존 카테고리 스크롤 방식 (기본)
    - bq_sync: BQ 전체 제품 리뷰 동기화만
    - hybrid: 카테고리 스크롤(신규 발견) + BQ 동기화(기존 리뷰 업데이트)
    """
    crawl_cfg = config.get("pipeline", {}).get("crawling", {})
    mode = crawl_cfg.get("mode", "incremental")
    strategy = crawl_cfg.get("strategy", "scroll")
    is_full = force_full or (mode == "full")
    bq_dataset = config.get("pipeline", {}).get("storage", {}).get("bigquery", {}).get("dataset", "daiso")

    # 크롤러 경로 추가
    crawler_dir = PROJECT_ROOT / "05_src" / "01_crawling"
    sys.path.insert(0, str(crawler_dir))
    os.chdir(crawler_dir)

    from daiso_beauty_crawler import run_all, run_all_bq
    from crawl_history import CrawlHistory

    # 카테고리 필터 파싱
    categories_filter = _parse_active_categories(crawl_cfg.get("active_categories"))
    if categories_filter:
        print(f"  카테고리 필터: {list(categories_filter.keys())}")

    # 이력 로드 (증분 모드일 때만)
    history = None
    if not is_full:
        history = _load_history(config, crawler_dir)
        print(f"  크롤링 모드: 증분 (이력: {history.product_count}개 제품)")
    else:
        print(f"  크롤링 모드: 풀 (전체 재크롤링)")

    # 풀 모드면 strategy 무시하고 기존 scroll 방식
    if is_full:
        strategy = "scroll"

    print(f"  크롤링 전략: {strategy}")

    products_path = None
    reviews_path = None
    ingredients_path = None

    # --- strategy 분기 ---
    if strategy == "scroll":
        products_path, reviews_path, ingredients_path = run_all(
            crawl_reviews=crawl_cfg.get("crawl_reviews", True),
            crawl_ingredients=crawl_cfg.get("crawl_ingredients", True),
            headless=crawl_cfg.get("headless", True),
            categories_filter=categories_filter,
            history=history,
        )

    elif strategy == "bq_sync":
        _, reviews_path, _ = run_all_bq(
            headless=crawl_cfg.get("headless", True),
            history=history,
            bq_dataset=bq_dataset,
        )

    elif strategy == "hybrid":
        # 1단계: 카테고리 스크롤 (신규 제품 발견 + 리뷰 수집)
        print("\n  [hybrid 1/2] 카테고리 스크롤 (신규 제품 발견)...")
        products_path, scroll_reviews_path, ingredients_path = run_all(
            crawl_reviews=crawl_cfg.get("crawl_reviews", True),
            crawl_ingredients=crawl_cfg.get("crawl_ingredients", True),
            headless=crawl_cfg.get("headless", True),
            categories_filter=categories_filter,
            history=history,
        )

        # 2단계: BQ 전체 제품 리뷰 동기화
        print("\n  [hybrid 2/2] BQ 전체 제품 리뷰 동기화...")
        _, bq_reviews_path, _ = run_all_bq(
            headless=crawl_cfg.get("headless", True),
            history=history,
            bq_dataset=bq_dataset,
        )

        # 리뷰 CSV 병합 (scroll + bq_sync)
        review_paths = [p for p in [scroll_reviews_path, bq_reviews_path] if p]
        if review_paths:
            dfs = [pd.read_csv(p) for p in review_paths]
            merged = pd.concat(dfs, ignore_index=True)
            from utils import get_date_string
            date_str = get_date_string()
            reviews_path = f"data/reviews_hybrid_{date_str}.csv"
            merged.to_csv(reviews_path, index=False, encoding="utf-8-sig")
            print(f"  hybrid 리뷰 병합: {len(merged)}개 → {reviews_path}")
        else:
            reviews_path = None

    else:
        print(f"  [경고] 알 수 없는 strategy '{strategy}', scroll로 폴백")
        products_path, reviews_path, ingredients_path = run_all(
            crawl_reviews=crawl_cfg.get("crawl_reviews", True),
            crawl_ingredients=crawl_cfg.get("crawl_ingredients", True),
            headless=crawl_cfg.get("headless", True),
            categories_filter=categories_filter,
            history=history,
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
    use_bq: bool = False,
    bq_dataset: str = "daiso",
) -> dict:
    """변환 실행 → 13개 테이블 dict 반환

    Parameters
    ----------
    use_bq : bool
        True면 BQ에서 기존 데이터를 조회하여 ID 유지.
        False면 기존 로컬 final/ CSV에서 로드 (레거시).
    """
    print("\n[변환] raw CSV 로드 중...")

    products_df = pd.read_csv(products_csv) if products_csv else pd.DataFrame()
    reviews_df = pd.read_csv(reviews_csv) if reviews_csv else pd.DataFrame()
    ingredients_df = pd.read_csv(ingredients_csv) if ingredients_csv else pd.DataFrame()

    print(f"  products: {len(products_df)}행, reviews: {len(reviews_df)}행, ingredients: {len(ingredients_df)}행")

    # 기존 데이터 로드 (ID 유지용) — BQ 또는 로컬
    if use_bq:
        print(f"  기존 데이터 소스: BigQuery ({bq_dataset})")
        existing_data = load_existing_from_bq(dataset=bq_dataset)
    else:
        existing_data = load_existing_final(final_dir)
        if existing_data:
            print(f"  기존 데이터 로드 (로컬): {list(existing_data.keys())}")

    # 프로모션 로드 — BQ → 로컬 CSV → 빈 DataFrame 순으로 폴백
    promotions_df = existing_data.get("promotions", None)
    if promotions_df is None or promotions_df.empty:
        promo_path = Path(final_dir) / "promotions.csv"
        if promo_path.exists():
            promotions_df = pd.read_csv(promo_path, parse_dates=["start_date", "end_date"])
            print(f"  프로모션 로드 (로컬): {len(promotions_df)}건")
        else:
            promotions_df = None
            print("  프로모션: BQ/로컬 모두 없음 → 프로모션 매칭 건너뜀")

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
    parser.add_argument("--full", action="store_true", help="강제 풀 크롤링 (증분 모드 무시)")
    parser.add_argument("--products", type=str, help="기존 products CSV 경로")
    parser.add_argument("--reviews", type=str, help="기존 reviews CSV 경로")
    parser.add_argument("--ingredients", type=str, help="기존 ingredients CSV 경로")
    parser.add_argument("--upload-bq", action="store_true", help="BigQuery 업로드 포함")
    parser.add_argument("--local-only", action="store_true", help="로컬 저장만 (BQ 비활성)")
    parser.add_argument("--final-dir", type=str, default=None, help="기존 final/ CSV 디렉토리")
    args = parser.parse_args()

    config = load_config()
    start_time = datetime.now()

    crawl_mode = "풀" if args.full else config.get("pipeline", {}).get("crawling", {}).get("mode", "incremental")

    print("=" * 60)
    print(f"크롤러 → ERD 파이프라인 (크롤링 모드: {crawl_mode})")
    print(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # final 디렉토리 결정
    final_dir = args.final_dir or str(PROJECT_ROOT / "02_processed_data" / "csv" / "final")

    # Step 1: 크롤링
    if not args.skip_crawl:
        print("\n[Step 1] 크롤링 실행...")
        products_csv, reviews_csv, ingredients_csv = run_crawling(config, force_full=args.full)
    else:
        products_csv = args.products
        reviews_csv = args.reviews
        ingredients_csv = args.ingredients
        print("\n[Step 1] 크롤링 건너뜀 (기존 CSV 사용)")

    if not any([products_csv, reviews_csv, ingredients_csv]):
        print("오류: 입력 CSV가 없습니다. --products, --reviews, --ingredients 옵션을 확인하세요.")
        sys.exit(1)

    # BQ 설정 확인
    bq_cfg = config.get("pipeline", {}).get("storage", {}).get("bigquery", {})
    bq_enabled = bq_cfg.get("enabled", False) and not args.local_only
    bq_dataset = bq_cfg.get("dataset", "daiso")
    local_cfg = config.get("pipeline", {}).get("storage", {}).get("local", {})
    local_enabled = local_cfg.get("csv", False) or local_cfg.get("parquet", False)

    # Step 2: 변환
    print("\n[Step 2] 변환 실행...")
    tables = run_transform(
        products_csv, reviews_csv, ingredients_csv, final_dir,
        use_bq=(bq_enabled or args.upload_bq),
        bq_dataset=bq_dataset,
    )

    # Step 3: 로컬 저장 (설정에 따라)
    if local_enabled or args.local_only:
        print("\n[Step 3] 로컬 저장...")
        run_save(tables, config)
    else:
        print("\n[Step 3] 로컬 저장 건너뜀 (BQ 전용 모드)")

    # Step 4: BigQuery UPSERT
    if args.upload_bq or bq_enabled:
        print("\n[Step 4] BigQuery UPSERT...")
        run_upload_bq(tables, config)
    else:
        print("\n[Step 4] BigQuery 업로드 건너뜀")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'=' * 60}")
    print(f"파이프라인 완료 (소요 시간: {elapsed:.1f}초)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
