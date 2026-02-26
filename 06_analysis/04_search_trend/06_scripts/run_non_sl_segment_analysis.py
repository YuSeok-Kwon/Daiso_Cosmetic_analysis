"""Non-SL 제품 560개 네이버 DataLab 검색 트렌드 수집

세그먼트별(성별, 연령, 기기) 검색 트렌드 데이터 수집
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
from dotenv import load_dotenv
import os

# 환경변수 먼저 로드
CONFIG_DIR = _PROJECT_ROOT / "config"
load_dotenv(CONFIG_DIR / ".env")

# 상대 import 대신 직접 import (숫자 디렉토리 호환)
sys.path.insert(0, str(_MODULE_DIR))
from importlib import import_module as _im

_config = _im("05_src.config")
_client_mod = _im("05_src.naver_trend_client")

AGE_GROUPS = _config.AGE_GROUPS
GENDER_MAP = _config.GENDER_MAP
DEVICE_MAP = _config.DEVICE_MAP

NaverTrendClient = _client_mod.NaverTrendClient

# 출력 디렉토리 (메인 outputs)
OUTPUT_DIR = _PROJECT_ROOT / "02_outputs" / "Search_Trend"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_product_name(name: str) -> str:
    """제품명 전처리"""
    import re
    name = re.sub(r'\d+\s*(ml|g|매|ea|개입|입|P)\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\[.*?\]|\(.*?\)', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def generate_keywords(product: dict, max_keywords: int = 10) -> list[str]:
    """제품별 검색 키워드 생성 (최대 10개)"""
    brand = product['brand_name']
    name_raw = product['name']
    name = clean_product_name(name_raw)

    keywords = []

    # 다이소 자체 브랜드
    if brand.lower() == '다이소':
        keywords.append(f"다이소 {name}")
        keywords.append(name)
        keywords.append(f"다이소 {name} 후기")
        keywords.append(f"다이소 {name} 리뷰")
    else:
        keywords.append(f"{brand} {name}")
        keywords.append(f"다이소 {brand} {name}")
        keywords.append(f"{brand} {name} 후기")
        keywords.append(f"{brand} {name} 리뷰")
        keywords.append(f"{brand} {name} 추천")

    # 중복 제거
    keywords = list(dict.fromkeys(keywords))
    return keywords[:max_keywords]


def build_non_sl_keyword_groups() -> list[dict]:
    """Non-SL 제품 560개 키워드 그룹 생성"""
    sli_path = _PROJECT_ROOT / "02_outputs" / "Sli" / "sli_integrated_results.csv"
    if not sli_path.exists():
        raise FileNotFoundError(f"SLI 파일 없음: {sli_path}")

    sli = pd.read_csv(sli_path)

    non_sl = sli[sli['final_soft_landing'] == False].copy()
    print(f"  Non-SL 제품: {len(non_sl)}개")

    keyword_groups = []
    for _, row in non_sl.iterrows():
        product = {
            'product_code': row['product_code'],
            'name': row['name'],
            'brand_name': row['brand_name']
        }
        keywords = generate_keywords(product)
        keyword_groups.append({
            'groupName': f"{row['brand_name']} - {clean_product_name(row['name'])}",
            'keywords': keywords
        })

    return keyword_groups


def parse_trend_results(
    results: list[dict], segment_type: str = "", segment_label: str = ""
) -> pd.DataFrame:
    """API 응답 리스트 → DataFrame 변환"""
    rows = []
    for resp in results:
        if not resp or "results" not in resp:
            continue
        for group in resp["results"]:
            group_name = group.get("title", "")
            for point in group.get("data", []):
                row = {
                    "keyword_group": group_name,
                    "period": point.get("period", ""),
                    "ratio": point.get("ratio", 0),
                }
                if segment_type:
                    row["segment_type"] = segment_type
                    row["segment_label"] = segment_label
                rows.append(row)

    return pd.DataFrame(rows)


def run_segment_analysis(
    client: NaverTrendClient,
    keyword_groups: list[dict],
    start_date: str,
    end_date: str,
    time_unit: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """전체 세그먼트 분석 실행

    Returns:
        (detail_df, summary_df)
    """
    all_dfs = []

    # 1. Base (전체)
    print("  [1/11] Base (전체)")
    results = client.search_trend_batch(
        keyword_groups, start_date, end_date, time_unit
    )
    df = parse_trend_results(results, "", "")
    all_dfs.append(df)

    # 2. 성별 (2개)
    for i, (label, code) in enumerate(GENDER_MAP.items(), 1):
        print(f"  [{1+i}/11] 성별 - {label}")
        results = client.search_trend_batch(
            keyword_groups, start_date, end_date, time_unit, gender=code
        )
        df = parse_trend_results(results, "gender", label)
        all_dfs.append(df)

    # 3. 연령대 (6개)
    for i, (label, codes) in enumerate(AGE_GROUPS.items(), 1):
        print(f"  [{3+i}/11] 연령 - {label}")
        results = client.search_trend_batch(
            keyword_groups, start_date, end_date, time_unit, ages=codes
        )
        df = parse_trend_results(results, "age", label)
        all_dfs.append(df)

    # 4. 기기 (2개)
    for i, (label, code) in enumerate(DEVICE_MAP.items(), 1):
        print(f"  [{9+i}/11] 기기 - {label}")
        results = client.search_trend_batch(
            keyword_groups, start_date, end_date, time_unit, device=code
        )
        df = parse_trend_results(results, "device", label)
        all_dfs.append(df)

    # 통합
    detail_df = pd.concat(all_dfs, ignore_index=True)

    # 요약 (제품별 평균)
    summary_df = detail_df.groupby(
        ['keyword_group', 'segment_type', 'segment_label']
    )['ratio'].mean().reset_index()
    summary_df.columns = ['keyword_group', 'segment_type', 'segment_label', 'avg_ratio']

    return detail_df, summary_df


def main():
    print("=" * 70)
    print("Non-SL 제품 560개 네이버 검색 트렌드 수집 (세그먼트별)")
    print("=" * 70)

    # 설정
    start_date = "2024-01-01"
    end_date = "2026-01-31"
    time_unit = "month"

    print(f"\n설정:")
    print(f"  기간: {start_date} ~ {end_date}")
    print(f"  단위: {time_unit}")
    print(f"  세그먼트: 11개 (base + gender 2 + age 6 + device 2)")

    # 1. 키워드 그룹 생성
    print("\n[1/4] 키워드 그룹 생성...")
    keyword_groups = build_non_sl_keyword_groups()
    print(f"  생성된 그룹: {len(keyword_groups)}개")
    total_keywords = sum(len(g['keywords']) for g in keyword_groups)
    print(f"  총 키워드: {total_keywords}개 (평균 {total_keywords/len(keyword_groups):.1f}개/제품)")

    # 예상 API 호출
    num_batches = -(-len(keyword_groups) // 5)  # 올림
    expected_calls = num_batches * 11
    print(f"  예상 API 호출: {expected_calls}회 ({num_batches}배치 × 11세그먼트)")

    # 2. 클라이언트 초기화
    client = NaverTrendClient(use_cache=True)

    # 3. 세그먼트 분석 실행
    print("\n[2/4] 세그먼트 분석 실행...")
    detail_df, summary_df = run_segment_analysis(
        client, keyword_groups, start_date, end_date, time_unit
    )

    print(f"\n  분석 완료:")
    print(f"    Detail: {len(detail_df)}행")
    print(f"    Summary: {len(summary_df)}행")
    print(f"    API 호출: {client.api_call_count}회")

    # 4. 키워드 매핑 생성
    print("\n[3/4] 키워드 매핑 생성...")
    mapping_rows = []
    for g in keyword_groups:
        for kw in g['keywords']:
            mapping_rows.append({
                'keyword_group': g['groupName'],
                'keyword': kw
            })
    mapping_df = pd.DataFrame(mapping_rows)
    print(f"  매핑: {len(mapping_df)}행")

    # 5. 저장
    print("\n[4/4] 결과 저장...")
    date_str = datetime.now().strftime('%Y%m%d')

    detail_path = OUTPUT_DIR / f"non_sl_segment_detail_{date_str}.csv"
    detail_df.to_csv(detail_path, index=False, encoding='utf-8-sig')
    print(f"  Detail: {detail_path.name}")

    summary_path = OUTPUT_DIR / f"non_sl_segment_summary_{date_str}.csv"
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"  Summary: {summary_path.name}")

    mapping_path = OUTPUT_DIR / f"non_sl_keyword_mapping_{date_str}.csv"
    mapping_df.to_csv(mapping_path, index=False, encoding='utf-8-sig')
    print(f"  Mapping: {mapping_path.name}")

    # 완료
    print(f"\n{'=' * 70}")
    print("수집 완료")
    print(f"{'=' * 70}")
    print(f"\nAPI 호출 통계:")
    print(f"  총 호출: {client.api_call_count}회")
    if len(client.api_keys) > 1:
        client.print_key_stats()
    print(f"\n출력 디렉토리: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
