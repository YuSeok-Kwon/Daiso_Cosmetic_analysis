"""네이버 검색 API 기반 검색량·언급량 분석 (블로그/쇼핑/뉴스)

DataLab 트렌드 API가 '상대 비율'만 제공하는 반면,
검색 API는 **실제 검색 결과 건수**를 반환합니다.

분석 내용:
  1) 브랜드별 블로그/쇼핑/뉴스 검색 건수 비교
  2) 키워드 변형별 건수 비교 (어떤 검색어가 많이 쓰이는지)
  3) 쇼핑 상세: 가격대, 최저가/최고가, 쇼핑몰 분포
  4) 블로그 상세: 최근 포스트 제목·링크 수집

사용법 (프로젝트 루트 Why-pi/ 에서 실행):
    # 브랜드 단위 전체 분석
    python 06_analysis/04_search_trend/run_search_volume.py --mode brand

    # 제품 단위
    python 06_analysis/04_search_trend/run_search_volume.py --mode product

    # 테스트 (상위 5개만)
    python 06_analysis/04_search_trend/run_search_volume.py --test

    # 쇼핑 상세 수집 포함
    python 06_analysis/04_search_trend/run_search_volume.py --mode brand --detail shop

    # 블로그 포스트 수집 포함
    python 06_analysis/04_search_trend/run_search_volume.py --mode brand --detail blog
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

# ── 패키지 경로 보정 (숫자 디렉토리명 → python -m 불가) ───
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[1]  # Why-pi/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

sys.path.insert(0, str(_THIS_DIR.parent))
from importlib import import_module as _im

_config = _im("04_search_trend.config")
_kb = _im("04_search_trend.keyword_builder")
_search_mod = _im("04_search_trend.naver_search_client")

DEFAULT_TOP_N = _config.DEFAULT_TOP_N
OUTPUT_DIR = _config.OUTPUT_DIR

build_brand_keyword_groups = _kb.build_brand_keyword_groups
build_product_keyword_groups = _kb.build_product_keyword_groups
get_keyword_mapping = _kb.get_keyword_mapping

NaverSearchClient = _search_mod.NaverSearchClient


# ── HTML 태그 제거 ─────────────────────────────────────────
_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(text: str) -> str:
    """검색 API 응답의 HTML 태그(<b> 등) 제거"""
    return _TAG_RE.sub("", text) if text else ""


# ── 1) 검색량 비교 ────────────────────────────────────────

def collect_volume(
    client: NaverSearchClient,
    keyword_groups: list[dict],
    endpoints: list[str],
) -> pd.DataFrame:
    """브랜드/제품별 블로그·쇼핑·뉴스 검색 건수 수집

    각 그룹의 대표 키워드(첫 번째 = "다이소 {브랜드}")로 총 건수 조회.
    """
    rows = []
    total_groups = len(keyword_groups)

    for i, group in enumerate(keyword_groups):
        name = group["groupName"]
        # 대표 키워드: 첫 번째 키워드 사용
        main_kw = group["keywords"][0] if group["keywords"] else name

        row = {"group_name": name, "main_keyword": main_kw}

        for ep in endpoints:
            total = client.search_total(ep, main_kw)
            row[f"{ep}_total"] = total

        rows.append(row)

        if (i + 1) % 10 == 0 or (i + 1) == total_groups:
            print(f"    {i + 1}/{total_groups} 완료")

    client.flush_cache()
    return pd.DataFrame(rows)


# ── 2) 키워드 변형별 비교 ─────────────────────────────────

def collect_keyword_variants(
    client: NaverSearchClient,
    keyword_groups: list[dict],
    endpoint: str = "blog",
) -> pd.DataFrame:
    """각 그룹 내 키워드 변형별 블로그 검색 건수 비교

    어떤 키워드 패턴이 가장 많이 사용되는지 파악.
    """
    rows = []
    total_groups = len(keyword_groups)

    for i, group in enumerate(keyword_groups):
        name = group["groupName"]
        for kw in group["keywords"]:
            total = client.search_total(endpoint, kw)
            rows.append({
                "group_name": name,
                "keyword": kw,
                f"{endpoint}_total": total,
            })

        if (i + 1) % 5 == 0 or (i + 1) == total_groups:
            print(f"    {i + 1}/{total_groups} 그룹 완료")

    client.flush_cache()
    return pd.DataFrame(rows)


# ── 3) 쇼핑 상세 수집 ─────────────────────────────────────

def collect_shop_details(
    client: NaverSearchClient,
    keyword_groups: list[dict],
    max_items_per_group: int = 100,
) -> pd.DataFrame:
    """그룹별 쇼핑 검색 결과 상세 수집 (가격, 쇼핑몰, 카테고리)"""
    rows = []
    total_groups = len(keyword_groups)

    for i, group in enumerate(keyword_groups):
        name = group["groupName"]
        main_kw = group["keywords"][0] if group["keywords"] else name

        items = client.search_all_pages("shop", main_kw, max_items=max_items_per_group)

        for item in items:
            rows.append({
                "group_name": name,
                "query": main_kw,
                "title": strip_html(item.get("title", "")),
                "lprice": int(item.get("lprice", 0) or 0),
                "hprice": int(item.get("hprice", 0) or 0),
                "mallName": item.get("mallName", ""),
                "maker": item.get("maker", ""),
                "brand": item.get("brand", ""),
                "category1": item.get("category1", ""),
                "category2": item.get("category2", ""),
                "category3": item.get("category3", ""),
                "category4": item.get("category4", ""),
                "link": item.get("link", ""),
            })

        if (i + 1) % 5 == 0 or (i + 1) == total_groups:
            print(f"    {i + 1}/{total_groups} 그룹 완료 (누적 {len(rows)}건)")

    client.flush_cache()
    return pd.DataFrame(rows)


# ── 4) 블로그 상세 수집 ────────────────────────────────────

def collect_blog_details(
    client: NaverSearchClient,
    keyword_groups: list[dict],
    max_items_per_group: int = 100,
    sort: str = "date",
) -> pd.DataFrame:
    """그룹별 블로그 포스트 상세 수집 (제목, 설명, 블로거, 날짜)"""
    rows = []
    total_groups = len(keyword_groups)

    for i, group in enumerate(keyword_groups):
        name = group["groupName"]
        main_kw = group["keywords"][0] if group["keywords"] else name

        items = client.search_all_pages("blog", main_kw, max_items=max_items_per_group, sort=sort)

        for item in items:
            rows.append({
                "group_name": name,
                "query": main_kw,
                "title": strip_html(item.get("title", "")),
                "description": strip_html(item.get("description", "")),
                "bloggername": item.get("bloggername", ""),
                "postdate": item.get("postdate", ""),
                "link": item.get("link", ""),
            })

        if (i + 1) % 5 == 0 or (i + 1) == total_groups:
            print(f"    {i + 1}/{total_groups} 그룹 완료 (누적 {len(rows)}건)")

    client.flush_cache()
    return pd.DataFrame(rows)


# ── 5) 쇼핑 가격 요약 ─────────────────────────────────────

def summarize_shop_prices(shop_df: pd.DataFrame) -> pd.DataFrame:
    """쇼핑 상세 → 그룹별 가격 요약 (최저가, 최고가, 평균, 중위수, 상품 수)"""
    if shop_df.empty:
        return pd.DataFrame()

    # lprice가 0인 건 제외 (가격 정보 없음)
    priced = shop_df[shop_df["lprice"] > 0].copy()
    if priced.empty:
        return pd.DataFrame()

    summary = priced.groupby("group_name").agg(
        product_count=("lprice", "count"),
        price_min=("lprice", "min"),
        price_max=("lprice", "max"),
        price_mean=("lprice", "mean"),
        price_median=("lprice", "median"),
    ).reset_index()

    summary["price_mean"] = summary["price_mean"].round(0).astype(int)
    summary["price_median"] = summary["price_median"].round(0).astype(int)

    # 쇼핑몰 분포 (상위 5개)
    mall_counts = (
        priced.groupby("group_name")["mallName"]
        .apply(lambda x: x.value_counts().head(5).to_dict())
        .reset_index()
        .rename(columns={"mallName": "top_malls"})
    )
    summary = summary.merge(mall_counts, on="group_name", how="left")

    return summary.sort_values("product_count", ascending=False)


# ── CSV/XLSX 저장 ──────────────────────────────────────────

def save_csv(df: pd.DataFrame, name: str, date_str: str) -> str:
    if df.empty:
        return ""
    path = OUTPUT_DIR / f"{name}_{date_str}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  저장: {path.name} ({len(df)}행)")
    return str(path)


def save_xlsx(
    dfs: dict[str, pd.DataFrame],
    mode: str,
    date_str: str,
    summary: dict,
) -> str:
    """시트별 분리 XLSX 저장"""
    path = OUTPUT_DIR / f"search_volume_{mode}_{date_str}.xlsx"

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in dfs.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        summary_df = pd.DataFrame([summary])
        summary_df.to_excel(writer, sheet_name="실행요약", index=False)

    print(f"  저장: {path.name}")
    return str(path)


# ── 메인 실행 ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="네이버 검색 API 기반 검색량 분석")
    parser.add_argument("--mode", choices=["brand", "product"], default="brand", help="분석 단위")
    parser.add_argument(
        "--detail",
        choices=["none", "blog", "shop", "both"],
        default="none",
        help="상세 수집 (blog: 블로그 포스트, shop: 쇼핑 상품, both: 둘 다)",
    )
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="연착륙 상품 수")
    parser.add_argument("--test", action="store_true", help="상위 5개만 테스트")
    parser.add_argument("--no-cache", action="store_true", help="캐시 무시")
    parser.add_argument("--max-items", type=int, default=100, help="상세 수집 시 그룹당 최대 건수")
    args = parser.parse_args()

    top_n = 5 if args.test else args.top_n
    date_str = datetime.now().strftime("%Y%m%d")
    endpoints = ["blog", "shop", "news"]

    print("=" * 60)
    print("네이버 검색량 분석 (블로그/쇼핑/뉴스)")
    print(f"  모드: {args.mode} | 상세: {args.detail}")
    print(f"  상품 수: {top_n}{'  (테스트)' if args.test else ''}")
    print("=" * 60)

    # 1. 키워드 그룹 생성
    print("\n[1] 키워드 그룹 생성")
    if args.mode == "brand":
        keyword_groups = build_brand_keyword_groups(top_n)
    else:
        keyword_groups = build_product_keyword_groups(top_n)

    print(f"  생성된 그룹: {len(keyword_groups)}개")
    for g in keyword_groups[:3]:
        print(f"    - {g['groupName']}: {g['keywords'][:2]}...")

    # 2. 클라이언트 초기화
    client = NaverSearchClient(use_cache=not args.no_cache)

    # 3. 검색량 비교 (대표 키워드 × 3개 엔드포인트)
    print(f"\n[2] 검색량 비교 (블로그/쇼핑/뉴스)")
    volume_df = collect_volume(client, keyword_groups, endpoints)

    if not volume_df.empty:
        # 블로그 기준 정렬
        volume_df = volume_df.sort_values("blog_total", ascending=False)
        print(f"\n  === 블로그 검색량 TOP 10 ===")
        for _, row in volume_df.head(10).iterrows():
            print(
                f"    {row['group_name']:12s} | "
                f"블로그 {row['blog_total']:>8,} | "
                f"쇼핑 {row['shop_total']:>8,} | "
                f"뉴스 {row['news_total']:>8,}"
            )

    # 4. 키워드 변형별 비교 (블로그 기준)
    print(f"\n[3] 키워드 변형별 블로그 검색량 비교")
    variant_df = collect_keyword_variants(client, keyword_groups, endpoint="blog")

    if not variant_df.empty:
        # 그룹별 최다 검색 키워드 출력
        print(f"\n  === 그룹별 최다 블로그 키워드 ===")
        for name, sub in variant_df.groupby("group_name"):
            best = sub.sort_values("blog_total", ascending=False).iloc[0]
            print(f"    {name:12s} → {best['keyword']} ({best['blog_total']:,}건)")

    # 5. 상세 수집 (옵션)
    shop_detail_df = pd.DataFrame()
    shop_summary_df = pd.DataFrame()
    blog_detail_df = pd.DataFrame()

    if args.detail in ("shop", "both"):
        print(f"\n[4-A] 쇼핑 상세 수집 (그룹당 최대 {args.max_items}건)")
        shop_detail_df = collect_shop_details(
            client, keyword_groups, max_items_per_group=args.max_items
        )
        shop_summary_df = summarize_shop_prices(shop_detail_df)

        if not shop_summary_df.empty:
            print(f"\n  === 쇼핑 가격 요약 ===")
            for _, row in shop_summary_df.head(10).iterrows():
                print(
                    f"    {row['group_name']:12s} | "
                    f"상품 {row['product_count']:>4}개 | "
                    f"최저 {row['price_min']:>8,}원 | "
                    f"평균 {row['price_mean']:>8,}원"
                )

    if args.detail in ("blog", "both"):
        print(f"\n[4-B] 블로그 상세 수집 (그룹당 최대 {args.max_items}건, 최신순)")
        blog_detail_df = collect_blog_details(
            client, keyword_groups, max_items_per_group=args.max_items, sort="date"
        )
        print(f"  수집 완료: {len(blog_detail_df)}건")

    # 6. 키워드 매핑
    mapping_df = get_keyword_mapping(top_n)

    # 7. 저장
    print("\n[저장]")
    save_csv(volume_df, f"search_volume_{args.mode}", date_str)
    save_csv(variant_df, f"search_volume_{args.mode}_variants", date_str)
    if not shop_detail_df.empty:
        save_csv(shop_detail_df, f"search_volume_{args.mode}_shop_detail", date_str)
    if not shop_summary_df.empty:
        save_csv(shop_summary_df, f"search_volume_{args.mode}_shop_summary", date_str)
    if not blog_detail_df.empty:
        save_csv(blog_detail_df, f"search_volume_{args.mode}_blog_detail", date_str)
    save_csv(mapping_df, "keyword_mapping", date_str)

    # XLSX
    try:
        sheets = {
            "검색량비교": volume_df,
            "키워드변형": variant_df,
            "키워드매핑": mapping_df,
        }
        if not shop_summary_df.empty:
            sheets["쇼핑가격요약"] = shop_summary_df
        if not shop_detail_df.empty:
            sheets["쇼핑상세"] = shop_detail_df
        if not blog_detail_df.empty:
            sheets["블로그상세"] = blog_detail_df

        summary = {
            "mode": args.mode,
            "detail": args.detail,
            "top_n": top_n,
            "keyword_groups": len(keyword_groups),
            "api_calls": client.api_call_count,
            "volume_rows": len(volume_df),
            "variant_rows": len(variant_df),
            "shop_detail_rows": len(shop_detail_df),
            "blog_detail_rows": len(blog_detail_df),
            "executed_at": datetime.now().isoformat(),
        }
        save_xlsx(sheets, args.mode, date_str, summary)
    except ImportError:
        print("  [참고] openpyxl 미설치 → XLSX 생성 생략 (pip install openpyxl)")

    # 8. 완료
    print("\n" + "=" * 60)
    client.print_key_stats()
    print(f"완료! API 호출: {client.api_call_count}회")
    print(f"출력 디렉토리: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
