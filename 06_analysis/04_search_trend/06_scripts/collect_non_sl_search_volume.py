"""Non-SL 제품 560개 네이버 검색량 수집

네이버 검색 API (블로그, 쇼핑, 뉴스)를 사용하여 Non-SL 제품의 실제 검색 결과 총 건수를 수집
시각화 없이 CSV 데이터만 생성
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

# 패키지 경로 보정
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]  # Why-pi/
_MODULE_DIR = _THIS_DIR.parent  # 04_search_trend/

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

import pandas as pd

# 상대 import 대신 직접 import
sys.path.insert(0, str(_MODULE_DIR))
from importlib import import_module as _im

_naver_client = _im("05_src.naver_search_client")
NaverSearchClient = _naver_client.NaverSearchClient

# 출력 디렉토리
OUTPUT_DIR = _PROJECT_ROOT / "02_outputs" / "Search_Trend"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_product_name(name: str) -> str:
    """제품명 전처리"""
    import re
    # 용량/단위 제거
    name = re.sub(r'\d+\s*(ml|g|매|ea|개입|입|P)\b', '', name, flags=re.IGNORECASE)
    # 괄호 내용 제거
    name = re.sub(r'\[.*?\]|\(.*?\)', '', name)
    # 연속 공백 정리
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def generate_keywords(product: dict, max_keywords: int = 5) -> list[str]:
    """제품별 검색 키워드 생성"""
    brand = product['brand_name']
    name_raw = product['name']
    name = clean_product_name(name_raw)

    keywords = []

    # 다이소 자체 브랜드인 경우
    if brand.lower() == '다이소':
        keywords.append(f"다이소 {name}")
        keywords.append(name)
        keywords.append(f"다이소 {name} 후기")
    else:
        # 일반 브랜드
        keywords.append(f"{brand} {name}")
        keywords.append(f"다이소 {brand} {name}")
        keywords.append(f"{brand} {name} 후기")
        keywords.append(f"{brand} {name} 리뷰")

    return keywords[:max_keywords]


def main():
    print("=" * 70)
    print("Non-SL 제품 560개 네이버 검색량 수집")
    print("=" * 70)

    # 1. Non-SL 제품 로드
    print("\n[1/4] Non-SL 제품 로드...")
    sli_path = _PROJECT_ROOT / "02_outputs" / "Sli" / "sli_integrated_results.csv"
    if not sli_path.exists():
        raise FileNotFoundError(f"SLI 파일 없음: {sli_path}")

    sli = pd.read_csv(sli_path)
    non_sl = sli[sli['final_soft_landing'] == False].copy()
    print(f"  Non-SL 제품: {len(non_sl)}개")

    # 2. 키워드 생성
    print("\n[2/4] 검색 키워드 생성...")
    non_sl['keywords'] = non_sl.apply(
        lambda row: generate_keywords({
            'product_code': row['product_code'],
            'name': row['name'],
            'brand_name': row['brand_name']
        }), axis=1
    )
    total_keywords = non_sl['keywords'].apply(len).sum()
    print(f"  총 키워드: {total_keywords}개 (평균 {total_keywords/len(non_sl):.1f}개/제품)")
    print(f"  예상 API 호출: 약 {total_keywords}회")

    # 3. 네이버 검색 API 호출
    print("\n[3/4] 네이버 검색 API 호출...")
    client = NaverSearchClient()

    results = []
    for idx, row in non_sl.iterrows():
        product_code = row['product_code']
        name = row['name']
        brand = row['brand_name']
        keywords = row['keywords']

        # 각 키워드별 블로그/쇼핑/뉴스 검색량
        for keyword in keywords:
            blog_total = client.search_total('blog', keyword)
            shop_total = client.search_total('shop', keyword)
            news_total = client.search_total('news', keyword)

            results.append({
                'product_code': product_code,
                'product_name': name,
                'brand_name': brand,
                'keyword': keyword,
                'blog_count': blog_total,
                'shop_count': shop_total,
                'news_count': news_total,
                'total_count': blog_total + shop_total + news_total,
                'collected_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

        if (idx + 1) % 50 == 0:
            print(f"  진행: {idx + 1}/{len(non_sl)} 제품 완료")

    client.flush_cache()
    print(f"\n  API 호출 완료: {client.api_call_count}회")
    if len(client.api_keys) > 1:
        client.print_key_stats()

    # 4. 결과 저장
    print("\n[4/4] 결과 저장...")
    date_str = datetime.now().strftime('%Y%m%d')

    # 상세 데이터
    df_detail = pd.DataFrame(results)
    detail_path = OUTPUT_DIR / f"search_volume_non_sl_detail_{date_str}.csv"
    df_detail.to_csv(detail_path, index=False, encoding='utf-8-sig')
    print(f"  상세 데이터: {detail_path.name}")
    print(f"    행 수: {len(df_detail)}")

    # 제품별 요약 (최대 검색량 키워드 기준)
    summary = df_detail.loc[df_detail.groupby('product_code')['total_count'].idxmax()].copy()
    summary = summary[['product_code', 'product_name', 'brand_name', 'keyword',
                       'blog_count', 'shop_count', 'news_count', 'total_count']]
    summary = summary.sort_values('total_count', ascending=False).reset_index(drop=True)

    summary_path = OUTPUT_DIR / f"search_volume_non_sl_summary_{date_str}.csv"
    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"  요약 데이터: {summary_path.name}")
    print(f"    행 수: {len(summary)}")

    # 통계
    print(f"\n{'=' * 70}")
    print("수집 완료")
    print(f"{'=' * 70}")
    print(f"\n검색량 통계:")
    print(f"  평균 total_count: {summary['total_count'].mean():.0f}")
    print(f"  중앙값: {summary['total_count'].median():.0f}")
    print(f"  최대: {summary['total_count'].max():,}")
    print(f"  최소: {summary['total_count'].min()}")

    print(f"\n검색량 상위 5개:")
    for i, row in summary.head(5).iterrows():
        print(f"  {i+1}. {row['brand_name']} - {row['product_name'][:40]}")
        print(f"     키워드: {row['keyword']}, 총 건수: {row['total_count']:,}")

    print(f"\n출력 디렉토리: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
