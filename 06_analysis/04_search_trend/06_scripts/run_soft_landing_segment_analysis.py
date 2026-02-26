"""연착륙 제품 146개 네이버 검색트렌드 세그먼트 분석

성별/연령대/기기별 검색 트렌드를 포함한 전체 분석 수행

실행 방법 (Why-pi/ 루트에서):
    python 06_analysis/04_search_trend/06_scripts/run_soft_landing_segment_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

# 패키지 경로 보정
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]  # Why-pi/
_MODULE_DIR = _THIS_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

import pandas as pd
from dotenv import load_dotenv
import os

# 환경변수 먼저 로드 (config.py보다 먼저)
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
DEFAULT_START_DATE = _config.DEFAULT_START_DATE
DEFAULT_END_DATE = _config.DEFAULT_END_DATE
DEFAULT_TIME_UNIT = _config.DEFAULT_TIME_UNIT

NaverTrendClient = _client_mod.NaverTrendClient

# 출력 디렉토리 (메인 outputs)
OUTPUT_DIR = _PROJECT_ROOT / "02_outputs" / "Search_Trend"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 연착륙 제품 로드 ────────────────────────────────────
def load_soft_landing_products() -> pd.DataFrame:
    """연착륙 제품 146개 로드"""
    sli_path = _PROJECT_ROOT / "02_outputs" / "Sli" / "sli_integrated_results.csv"
    if not sli_path.exists():
        raise FileNotFoundError(f"SLI 파일 없음: {sli_path}")

    df = pd.read_csv(sli_path)
    sl_df = df[df['final_soft_landing'] == True].copy()

    print(f"연착륙 제품 수: {len(sl_df)}")
    return sl_df


# ── 키워드 그룹 생성 ────────────────────────────────────
def build_keyword_groups(df: pd.DataFrame) -> tuple[list[dict], dict]:
    """연착륙 제품별 키워드 그룹 생성

    Returns:
        (keyword_groups, groups_mapping)
    """
    import re

    groups = []
    used_names = set()

    for _, row in df.iterrows():
        brand = row["brand_name"]
        name = row["name"]
        product_code = row["product_code"]

        # 검색 키워드 추출
        kw = name
        kw = re.sub(r"\d+\s*ml\*?\d*개?입?", "", kw, flags=re.IGNORECASE)
        kw = re.sub(r"\d+\s*(ml|g|mg|l|매)\b", "", kw, flags=re.IGNORECASE)
        kw = re.sub(r"\[[^\]]*\]", "", kw)
        kw = re.sub(r"\([^)]*\)", "", kw)

        if brand and kw.strip().startswith(brand):
            kw = kw.strip()[len(brand):]
        kw = re.sub(r"\s+", " ", kw).strip()

        if not kw or brand == "ALL":
            continue

        # 키워드 리스트 생성
        keywords = []
        if brand == "다이소":
            keywords = [
                f"다이소 {kw}",
                kw,
                f"다이소 {kw} 추천",
                f"다이소 {kw} 후기",
                f"다이소 {kw} 리뷰",
            ]
        else:
            keywords = [
                f"{brand} {kw}",
                f"다이소 {brand} {kw}",
                f"다이소 {kw}",
                f"{brand} {kw} 추천",
                f"{brand} {kw} 리뷰",
            ]

        # 중복 제거
        keywords = list(dict.fromkeys(keywords))[:10]

        # groupName 충돌 해결
        words = kw.split()
        short_name = " ".join(words[:2]) if len(words) >= 2 else kw
        candidate = f"{brand} {short_name}"

        if candidate in used_names:
            candidate = f"{brand} {' '.join(words[:3])}" if len(words) >= 3 else candidate
        if candidate in used_names:
            candidate = f"{candidate} {str(product_code)[:6]}"

        used_names.add(candidate)

        groups.append({
            "groupName": candidate,
            "keywords": keywords,
            "product_code": product_code,
            "brand_name": brand,
            "product_name": name,
        })

    # 매핑 딕셔너리
    mapping = {
        g["groupName"]: {
            "product_code": g["product_code"],
            "brand_name": g["brand_name"],
            "product_name": g["product_name"]
        }
        for g in groups
    }

    return groups, mapping


# ── 응답 파싱 ───────────────────────────────────────────
def parse_trend_results(
    results: list[dict],
    groups_mapping: dict,
    segment_type: str = "",
    segment_label: str = ""
) -> pd.DataFrame:
    """API 응답 → DataFrame 변환 + 세그먼트 정보 포함"""
    rows = []
    for resp in results:
        if not resp or "results" not in resp:
            continue
        for group in resp["results"]:
            group_name = group.get("title", "")
            meta = groups_mapping.get(group_name, {})

            for point in group.get("data", []):
                row = {
                    "keyword_group": group_name,
                    "product_code": meta.get("product_code", ""),
                    "brand_name": meta.get("brand_name", ""),
                    "product_name": meta.get("product_name", ""),
                    "period": point.get("period", ""),
                    "ratio": point.get("ratio", 0),
                }
                if segment_type:
                    row["segment_type"] = segment_type
                    row["segment_label"] = segment_label
                rows.append(row)

    return pd.DataFrame(rows)


# ── 세그먼트별 수집 ─────────────────────────────────────
def run_segment_analysis(
    client: NaverTrendClient,
    keyword_groups: list[dict],
    groups_mapping: dict,
    start_date: str,
    end_date: str,
    time_unit: str,
) -> pd.DataFrame:
    """전체 세그먼트 분석 수행

    base(전체) + gender(남/여) + age(10대~60대) + device(PC/모바일)
    총 11개 조건
    """
    all_dfs = []

    # 네이버 API용 키워드 그룹만 추출
    api_groups = [
        {"groupName": g["groupName"], "keywords": g["keywords"]}
        for g in keyword_groups
    ]

    # 1. 기본(전체)
    print("\n[Base] 전체")
    results = client.search_trend_batch(api_groups, start_date, end_date, time_unit)
    df = parse_trend_results(results, groups_mapping)
    all_dfs.append(df)
    print(f"  수집: {len(df)}행")

    # 2. 성별
    print("\n[성별 비교]")
    for label, code in GENDER_MAP.items():
        print(f"  → {label} ({code})")
        results = client.search_trend_batch(
            api_groups, start_date, end_date, time_unit, gender=code
        )
        df = parse_trend_results(results, groups_mapping, "gender", label)
        all_dfs.append(df)
        print(f"    수집: {len(df)}행")

    # 3. 연령대
    print("\n[연령대 비교]")
    for label, codes in AGE_GROUPS.items():
        print(f"  → {label} (codes={codes})")
        results = client.search_trend_batch(
            api_groups, start_date, end_date, time_unit, ages=codes
        )
        df = parse_trend_results(results, groups_mapping, "age", label)
        all_dfs.append(df)
        print(f"    수집: {len(df)}행")

    # 4. 기기
    print("\n[기기 비교]")
    for label, code in DEVICE_MAP.items():
        print(f"  → {label} ({code})")
        results = client.search_trend_batch(
            api_groups, start_date, end_date, time_unit, device=code
        )
        df = parse_trend_results(results, groups_mapping, "device", label)
        all_dfs.append(df)
        print(f"    수집: {len(df)}행")

    # 통합
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"\n통합 데이터: {len(combined)}행")
        return combined
    return pd.DataFrame()


# ── 요약 통계 생성 ──────────────────────────────────────
def create_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    """제품별 × 세그먼트별 요약 통계"""
    if detail_df.empty:
        return pd.DataFrame()

    # segment_type / segment_label 처리
    if "segment_type" not in detail_df.columns:
        detail_df["segment_type"] = ""
        detail_df["segment_label"] = "전체"

    detail_df["segment_type"] = detail_df["segment_type"].fillna("")
    detail_df["segment_label"] = detail_df["segment_label"].fillna("전체")

    summary = detail_df.groupby([
        "product_code", "product_name", "brand_name",
        "segment_type", "segment_label"
    ]).agg(
        total_ratio=("ratio", "sum"),
        avg_ratio=("ratio", "mean"),
        max_ratio=("ratio", "max"),
        min_ratio=("ratio", "min"),
        data_points=("ratio", "count"),
    ).reset_index()

    return summary.sort_values(["segment_type", "total_ratio"], ascending=[True, False])


# ── 저장 ───────────────────────────────────────────────
def save_results(
    detail_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    date_str: str,
    api_calls: int,
):
    """CSV + XLSX 저장"""

    # 1. 상세 CSV
    detail_path = OUTPUT_DIR / f"soft_landing_segment_detail_{date_str}.csv"
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"저장: {detail_path.name} ({len(detail_df)}행)")

    # 2. 요약 CSV
    summary_path = OUTPUT_DIR / f"soft_landing_segment_summary_{date_str}.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"저장: {summary_path.name} ({len(summary_df)}행)")

    # 3. 키워드 매핑 CSV
    mapping_path = OUTPUT_DIR / f"soft_landing_keyword_mapping_{date_str}.csv"
    mapping_df.to_csv(mapping_path, index=False, encoding="utf-8-sig")
    print(f"저장: {mapping_path.name} ({len(mapping_df)}행)")

    # 4. XLSX (시트 분리)
    try:
        xlsx_path = OUTPUT_DIR / f"soft_landing_segment_analysis_{date_str}.xlsx"

        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            # 전체 데이터
            if "segment_type" in detail_df.columns:
                base = detail_df[detail_df["segment_type"] == ""]
            else:
                base = detail_df
            if not base.empty:
                base.to_excel(writer, sheet_name="기본_전체", index=False)

            # 세그먼트별 분리
            if "segment_type" in detail_df.columns:
                for seg_type in detail_df["segment_type"].unique():
                    if seg_type == "":
                        continue
                    sheet_name = {
                        "gender": "성별비교",
                        "age": "연령비교",
                        "device": "기기비교",
                    }.get(seg_type, seg_type)
                    sub = detail_df[detail_df["segment_type"] == seg_type]
                    sub.to_excel(writer, sheet_name=sheet_name, index=False)

            # 요약
            summary_df.to_excel(writer, sheet_name="요약_전체", index=False)

            # 키워드 매핑
            mapping_df.to_excel(writer, sheet_name="키워드매핑", index=False)

            # 실행 요약
            meta = pd.DataFrame([{
                "executed_at": datetime.now().isoformat(),
                "start_date": DEFAULT_START_DATE,
                "end_date": DEFAULT_END_DATE,
                "time_unit": DEFAULT_TIME_UNIT,
                "total_products": len(mapping_df),
                "total_rows": len(detail_df),
                "api_calls": api_calls,
            }])
            meta.to_excel(writer, sheet_name="실행정보", index=False)

        print(f"저장: {xlsx_path.name}")

    except ImportError:
        print("[참고] openpyxl 미설치 → XLSX 생성 생략")


# ── 메인 실행 ───────────────────────────────────────────
def main():
    date_str = datetime.now().strftime("%Y%m%d")

    print("=" * 60)
    print("연착륙 제품 146개 네이버 검색트렌드 세그먼트 분석")
    print(f"기간: {DEFAULT_START_DATE} ~ {DEFAULT_END_DATE}")
    print(f"단위: {DEFAULT_TIME_UNIT}")
    print("세그먼트: base + gender(2) + age(6) + device(2) = 11개")
    print("=" * 60)

    # 1. 연착륙 제품 로드
    print("\n[1] 연착륙 제품 로드")
    sl_df = load_soft_landing_products()

    # 2. 키워드 그룹 생성
    print("\n[2] 키워드 그룹 생성")
    keyword_groups, groups_mapping = build_keyword_groups(sl_df)
    print(f"생성된 그룹: {len(keyword_groups)}개")
    for g in keyword_groups[:3]:
        print(f"  - {g['groupName']}: {g['keywords'][:2]}...")

    # 배치 예상 계산
    total_batches = (len(keyword_groups) + 4) // 5  # 5개씩
    total_segments = 11  # base + gender(2) + age(6) + device(2)
    total_api_calls = total_batches * total_segments
    print(f"\n예상 API 호출: {total_batches}배치 × {total_segments}세그먼트 = {total_api_calls}회")

    # 3. API 클라이언트 초기화
    client = NaverTrendClient(use_cache=True)

    # 4. 세그먼트 분석 수행
    print("\n[3] 세그먼트 분석 수행")
    detail_df = run_segment_analysis(
        client, keyword_groups, groups_mapping,
        DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_TIME_UNIT
    )

    if detail_df.empty:
        print("[오류] 데이터 수집 실패")
        return

    # 5. 요약 통계
    print("\n[4] 요약 통계 생성")
    summary_df = create_summary(detail_df)
    print(f"요약 행 수: {len(summary_df)}")

    # 6. 키워드 매핑 데이터
    mapping_df = pd.DataFrame([
        {
            "product_code": g["product_code"],
            "brand_name": g["brand_name"],
            "product_name": g["product_name"],
            "keyword_group": g["groupName"],
            "keywords": " | ".join(g["keywords"]),
        }
        for g in keyword_groups
    ])

    # 7. 저장
    print("\n[5] 결과 저장")
    save_results(detail_df, summary_df, mapping_df, date_str, client.api_call_count)

    # 8. 주요 통계 출력
    print("\n[주요 통계]")
    print(f"총 제품 수: {len(keyword_groups)}")
    print(f"총 데이터 행: {len(detail_df):,}행")
    print(f"API 호출 수: {client.api_call_count}회")

    if not summary_df.empty:
        # 전체(base) 평균
        base_summary = summary_df[summary_df["segment_label"] == "전체"]
        if not base_summary.empty:
            print(f"평균 검색 ratio (전체): {base_summary['avg_ratio'].mean():.2f}")
            print(f"\n검색량 상위 5개 제품 (전체):")
            for i, row in base_summary.head(5).iterrows():
                print(f"  {i+1}. [{row['brand_name']}] {row['product_name'][:40]}... (ratio: {row['total_ratio']:.1f})")

    # 9. 완료
    print("\n" + "=" * 60)
    print(f"완료! 출력 디렉토리: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
