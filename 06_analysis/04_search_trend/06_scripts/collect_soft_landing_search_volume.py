"""연착륙 제품 146개 네이버 검색량 수집

네이버 검색 API (블로그, 쇼핑, 뉴스)를 사용하여 연착륙 제품의 실제 검색 결과 총 건수를 수집
"""
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# ── 패키지 경로 보정 ───
_THIS_DIR = Path(__file__).resolve().parent
_SEARCH_TREND_DIR = _THIS_DIR.parent  # 04_search_trend/
_ANALYSIS_DIR = _SEARCH_TREND_DIR.parent  # 06_analysis/
_PROJECT_ROOT = _ANALYSIS_DIR.parent  # Why-pi/

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SEARCH_TREND_DIR) not in sys.path:
    sys.path.insert(0, str(_SEARCH_TREND_DIR))

# 05_src가 패키지이므로 동적 import
import importlib
src_module = importlib.import_module("05_src.naver_search_client")
NaverSearchClient = src_module.NaverSearchClient

# 경로 설정
SLI_DIR = _PROJECT_ROOT / "02_outputs" / "Sli"
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
    print("연착륙 제품 네이버 검색량 수집")
    print("=" * 70)
    
    # 1. 연착륙 제품 로드
    print("\n[1/4] 연착륙 제품 로드...")
    sli_path = SLI_DIR / "sli_integrated_results.csv"
    sli = pd.read_csv(sli_path)
    soft_landing = sli[sli['final_soft_landing'] == True].copy()
    print(f"  연착륙 제품: {len(soft_landing)}개")
    
    # 2. 키워드 생성
    print("\n[2/4] 검색 키워드 생성...")
    soft_landing['keywords'] = soft_landing.apply(
        lambda row: generate_keywords({
            'product_code': row['product_code'],
            'name': row['name'],
            'brand_name': row['brand_name']
        }), axis=1
    )
    total_keywords = soft_landing['keywords'].apply(len).sum()
    print(f"  총 키워드: {total_keywords}개 (평균 {total_keywords/len(soft_landing):.1f}개/제품)")
    
    # 3. 네이버 검색 API 호출
    print("\n[3/4] 네이버 검색 API 호출...")
    client = NaverSearchClient()
    
    results = []
    for idx, row in soft_landing.iterrows():
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
        
        if (idx + 1) % 20 == 0:
            print(f"  진행: {idx + 1}/{len(soft_landing)} 제품 완료")
    
    client.flush_cache()
    print(f"\n  API 호출 완료: {client.api_call_count}회")
    if len(client.api_keys) > 1:
        client.print_key_stats()
    
    # 4. 결과 저장
    print("\n[4/4] 결과 저장...")
    date_str = datetime.now().strftime('%Y%m%d')
    
    # 상세 데이터
    df_detail = pd.DataFrame(results)
    detail_path = OUTPUT_DIR / f"search_volume_soft_landing_detail_{date_str}.csv"
    df_detail.to_csv(detail_path, index=False, encoding='utf-8-sig')
    print(f"  상세 데이터: {detail_path}")
    print(f"    행 수: {len(df_detail)}")
    
    # 제품별 요약 (최대 검색량 키워드 기준)
    summary = df_detail.loc[df_detail.groupby('product_code')['total_count'].idxmax()].copy()
    summary = summary[['product_code', 'product_name', 'brand_name', 'keyword',
                       'blog_count', 'shop_count', 'news_count', 'total_count']]
    summary = summary.sort_values('total_count', ascending=False).reset_index(drop=True)
    
    summary_path = OUTPUT_DIR / f"search_volume_soft_landing_summary_{date_str}.csv"
    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"  요약 데이터: {summary_path}")
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
        print(f"  {i+1}. {row['brand_name']} - {row['product_name'][:30]}")
        print(f"     키워드: {row['keyword']}, 총 건수: {row['total_count']:,}")


if __name__ == "__main__":
    main()
