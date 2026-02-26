"""네이버 DataLab 검색어 트렌드 메인 실행 스크립트 (CLI)

사용법 (프로젝트 루트 Why-pi/ 에서 실행):
    # 브랜드 단위 + 전체 세분화
    python 06_analysis/04_search_trend/run_search_trend.py --mode brand --segment all

    # 제품 단위 + 성별만
    python 06_analysis/04_search_trend/run_search_trend.py --mode product --segment gender

    # 테스트 (상위 5개만)
    python 06_analysis/04_search_trend/run_search_trend.py --test
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# ── 패키지 경로 보정 (숫자 디렉토리명 → python -m 불가) ───
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[1]  # Why-pi/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

# 상대 import 대신 직접 import (숫자 디렉토리 호환)
sys.path.insert(0, str(_THIS_DIR.parent))
from importlib import import_module as _im

_config = _im("04_search_trend.config")
_kb = _im("04_search_trend.keyword_builder")
_client_mod = _im("04_search_trend.naver_trend_client")

AGE_GROUPS = _config.AGE_GROUPS
DEFAULT_START_DATE = _config.DEFAULT_START_DATE
DEFAULT_END_DATE = _config.DEFAULT_END_DATE
DEFAULT_TIME_UNIT = _config.DEFAULT_TIME_UNIT
DEFAULT_TOP_N = _config.DEFAULT_TOP_N
DEVICE_MAP = _config.DEVICE_MAP
GENDER_MAP = _config.GENDER_MAP
OUTPUT_DIR = _config.OUTPUT_DIR

build_brand_keyword_groups = _kb.build_brand_keyword_groups
build_all_brand_keyword_groups = _kb.build_all_brand_keyword_groups
build_product_keyword_groups = _kb.build_product_keyword_groups
get_keyword_mapping = _kb.get_keyword_mapping

NaverTrendClient = _client_mod.NaverTrendClient


# ── 응답 파싱 ───────────────────────────────────────────

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


# ── 세분화 호출 ─────────────────────────────────────────

def run_segment(
    client: NaverTrendClient,
    keyword_groups: list[dict],
    start_date: str,
    end_date: str,
    time_unit: str,
    segment: str,
) -> pd.DataFrame:
    """세분화(성별/연령/기기)별 호출 후 결합"""
    all_dfs = []

    if segment in ("gender", "all"):
        print("\n[성별 비교]")
        for label, code in GENDER_MAP.items():
            print(f"  → {label} ({code})")
            results = client.search_trend_batch(
                keyword_groups, start_date, end_date, time_unit, gender=code
            )
            df = parse_trend_results(results, "gender", label)
            all_dfs.append(df)

    if segment in ("age", "all"):
        print("\n[연령대 비교]")
        for label, codes in AGE_GROUPS.items():
            print(f"  → {label} (codes={codes})")
            results = client.search_trend_batch(
                keyword_groups, start_date, end_date, time_unit, ages=codes
            )
            df = parse_trend_results(results, "age", label)
            all_dfs.append(df)

    if segment in ("device", "all"):
        print("\n[기기 비교]")
        for label, code in DEVICE_MAP.items():
            print(f"  → {label} ({code})")
            results = client.search_trend_batch(
                keyword_groups, start_date, end_date, time_unit, device=code
            )
            df = parse_trend_results(results, "device", label)
            all_dfs.append(df)

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()


# ── 결승전: 배치별 1위 재비교 ────────────────────────────

def find_batch_winners(
    results: list[dict], keyword_groups: list[dict]
) -> list[dict]:
    """각 배치 응답에서 평균 ratio 1위 브랜드를 추출하고 원본 키워드 그룹 반환"""
    group_lookup = {g["groupName"]: g for g in keyword_groups}
    winners = []

    for resp in results:
        if not resp or "results" not in resp:
            continue
        best_name, best_avg = "", 0
        for group in resp["results"]:
            name = group.get("title", "")
            data = group.get("data", [])
            if not data:
                continue
            avg = sum(p.get("ratio", 0) for p in data) / len(data)
            if avg > best_avg:
                best_avg = avg
                best_name = name

        if best_name and best_name in group_lookup:
            winners.append(group_lookup[best_name])

    # 중복 제거 (같은 브랜드가 이미 있으면 스킵)
    seen = set()
    unique = []
    for w in winners:
        if w["groupName"] not in seen:
            seen.add(w["groupName"])
            unique.append(w)
    return unique


def _run_tournament(
    client: NaverTrendClient,
    winners: list[dict],
    start_date: str,
    end_date: str,
    time_unit: str,
    **api_kwargs,
) -> tuple[list[dict], list[dict]]:
    """5개 초과 시 토너먼트 방식으로 최종 5명 이하까지 축소

    Returns: (최종 winners, 모든 라운드의 API 응답)
    """
    current = winners
    all_results = []
    round_num = 1

    while len(current) > 5:
        print(f"    준결승 {round_num}: {len(current)}개 → {-(-len(current)//5)}배치")
        results = client.search_trend_batch(
            current, start_date, end_date, time_unit, **api_kwargs
        )
        all_results.extend(results)
        current = find_batch_winners(results, current)
        round_num += 1

    return current, all_results


def run_finals(
    client: NaverTrendClient,
    base_results: list[dict],
    keyword_groups: list[dict],
    start_date: str,
    end_date: str,
    time_unit: str,
    segment: str = "none",
) -> pd.DataFrame:
    """배치별 1위끼리 결승전 → 진짜 1위 결정

    6개 이상이면 토너먼트로 5개 이하까지 축소 후 결승.
    세분화가 있으면 세분화별로도 결승전을 수행한다.
    """
    winners = find_batch_winners(base_results, keyword_groups)

    if len(winners) <= 1:
        print("  결승전 불필요 (배치 1개 이하)")
        return pd.DataFrame()

    print(f"\n[결승전] 배치별 1위 {len(winners)}개 직접 비교")
    for w in winners:
        print(f"  - {w['groupName']}")

    all_dfs = []

    def _finals_call(groups, seg_type="", seg_label="", **api_kwargs):
        """토너먼트 + 최종 결승 실행"""
        finalists = groups
        if len(finalists) > 5:
            finalists, _ = _run_tournament(
                client, finalists, start_date, end_date, time_unit, **api_kwargs
            )
        resp = client.search_trend(finalists, start_date, end_date, time_unit, **api_kwargs)
        if resp:
            df = parse_trend_results([resp], seg_type, seg_label)
            df["round"] = "finals"
            all_dfs.append(df)

    # 기본 결승
    _finals_call(winners)

    # 세분화 결승
    if segment in ("gender", "all"):
        for label, code in GENDER_MAP.items():
            _finals_call(winners, "gender", label, gender=code)

    if segment in ("age", "all"):
        for label, codes in AGE_GROUPS.items():
            _finals_call(winners, "age", label, ages=codes)

    if segment in ("device", "all"):
        for label, code in DEVICE_MAP.items():
            _finals_call(winners, "device", label, device=code)

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()


# ── 앵커 비교: 세그먼트별 앵커로 전체 브랜드 정규화 ────

# 세그먼트 → 앵커 브랜드 매핑
ANCHOR_MAP = {
    "base": "손앤박",
    "gender_남성": "바세린",
    "gender_여성": "손앤박",
    "age_10대": "파티온",
    "age_20대": "파티온",
    "age_30대": "손앤박",
    "age_40대": "손앤박",
    "age_50대": "바세린",
    "age_60대": "바세린",
    "device_PC": "손앤박",
    "device_모바일": "손앤박",
}


def _anchor_batch_call(
    client: NaverTrendClient,
    anchor_group: dict,
    other_groups: list[dict],
    start_date: str,
    end_date: str,
    time_unit: str,
    **api_kwargs,
) -> list[dict]:
    """앵커를 매 배치에 포함하여 호출 (앵커 + 4개씩)"""
    batches = [other_groups[i : i + 4] for i in range(0, len(other_groups), 4)]
    results = []
    for idx, batch in enumerate(batches, 1):
        groups = [anchor_group] + batch
        resp = client.search_trend(
            groups, start_date, end_date, time_unit, **api_kwargs
        )
        if resp:
            results.append(resp)
    return results


def _normalize_with_anchor(results: list[dict], anchor_name: str) -> pd.DataFrame:
    """앵커 ratio 기준으로 정규화 (앵커=100)"""
    rows = []
    for resp in results:
        if not resp or "results" not in resp:
            continue

        # period별 앵커 ratio 수집
        anchor_by_period = {}
        for group in resp["results"]:
            if group.get("title") == anchor_name:
                for point in group.get("data", []):
                    anchor_by_period[point["period"]] = point.get("ratio", 0)
                break

        # 각 브랜드 정규화
        for group in resp["results"]:
            name = group.get("title", "")
            for point in group.get("data", []):
                period = point["period"]
                raw = point.get("ratio", 0)
                anchor_val = anchor_by_period.get(period, 0)
                normalized = (raw / anchor_val * 100) if anchor_val > 0 else 0
                rows.append({
                    "keyword_group": name,
                    "period": period,
                    "ratio_raw": raw,
                    "ratio_normalized": round(normalized, 2),
                    "anchor": anchor_name,
                })

    return pd.DataFrame(rows)


def run_anchor_comparison(
    client: NaverTrendClient,
    keyword_groups: list[dict],
    start_date: str,
    end_date: str,
    time_unit: str,
) -> pd.DataFrame:
    """세그먼트별 앵커를 사용하여 전체 브랜드 정규화 비교"""
    group_lookup = {g["groupName"]: g for g in keyword_groups}
    all_dfs = []

    # 실행할 조건 목록: (segment_key, api_kwargs, segment_type, segment_label)
    conditions = [
        ("base", {}, "", ""),
        ("gender_남성", {"gender": "m"}, "gender", "남성"),
        ("gender_여성", {"gender": "f"}, "gender", "여성"),
    ]
    for age_label, age_codes in AGE_GROUPS.items():
        conditions.append((f"age_{age_label}", {"ages": age_codes}, "age", age_label))
    conditions += [
        ("device_PC", {"device": "pc"}, "device", "PC"),
        ("device_모바일", {"device": "mo"}, "device", "모바일"),
    ]

    for seg_key, api_kwargs, seg_type, seg_label in conditions:
        anchor_name = ANCHOR_MAP[seg_key]
        anchor_group = group_lookup.get(anchor_name)
        if not anchor_group:
            print(f"  [경고] 앵커 '{anchor_name}' 그룹 없음 → 건너뜀")
            continue

        others = [g for g in keyword_groups if g["groupName"] != anchor_name]
        label = f"{seg_label}" if seg_label else "전체"
        print(f"  {label} (앵커: {anchor_name}) — {len(others)}그룹 → {-(-len(others)//4)}배치")

        results = _anchor_batch_call(
            client, anchor_group, others,
            start_date, end_date, time_unit, **api_kwargs,
        )
        df = _normalize_with_anchor(results, anchor_name)

        if not df.empty:
            if seg_type:
                df["segment_type"] = seg_type
                df["segment_label"] = seg_label
            all_dfs.append(df)

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()


# ── 제품 모드 앵커 비교: engagement 1위를 단일 앵커로 사용 ──

def run_product_anchor_comparison(
    client: NaverTrendClient,
    keyword_groups: list[dict],
    start_date: str,
    end_date: str,
    time_unit: str,
) -> pd.DataFrame:
    """제품 모드: engagement_score 1위(리스트 첫 번째)를 단일 앵커로 사용

    세그먼트별 동일 앵커를 사용하여 일관된 정규화 기준 제공.
    기존 _anchor_batch_call(), _normalize_with_anchor() 재사용.
    """
    if not keyword_groups:
        return pd.DataFrame()

    anchor_group = keyword_groups[0]
    anchor_name = anchor_group["groupName"]
    others = [g for g in keyword_groups[1:]]

    all_dfs = []

    # 실행할 조건 목록: (api_kwargs, segment_type, segment_label)
    conditions = [
        ({}, "", ""),
        ({"gender": "m"}, "gender", "남성"),
        ({"gender": "f"}, "gender", "여성"),
    ]
    for age_label, age_codes in AGE_GROUPS.items():
        conditions.append(({"ages": age_codes}, "age", age_label))
    conditions += [
        ({"device": "pc"}, "device", "PC"),
        ({"device": "mo"}, "device", "모바일"),
    ]

    for api_kwargs, seg_type, seg_label in conditions:
        label = seg_label if seg_label else "전체"
        print(f"  {label} (앵커: {anchor_name}) — {len(others)}그룹 → {-(-len(others)//4)}배치")

        results = _anchor_batch_call(
            client, anchor_group, others,
            start_date, end_date, time_unit, **api_kwargs,
        )
        df = _normalize_with_anchor(results, anchor_name)

        if not df.empty:
            if seg_type:
                df["segment_type"] = seg_type
                df["segment_label"] = seg_label
            all_dfs.append(df)

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()


# ── CSV 저장 ────────────────────────────────────────────

def save_csv(df: pd.DataFrame, name: str, date_str: str) -> str:
    """DataFrame → CSV 저장 후 경로 반환"""
    if df.empty:
        return ""
    path = OUTPUT_DIR / f"{name}_{date_str}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  저장: {path.name} ({len(df)}행)")
    return str(path)


# ── XLSX 저장 ───────────────────────────────────────────

def save_xlsx(
    base_df: pd.DataFrame,
    segment_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    mode: str,
    date_str: str,
    summary: dict,
    finals_df: pd.DataFrame = None,
    anchor_df: pd.DataFrame = None,
) -> str:
    """시트별 분리 XLSX 저장"""
    path = OUTPUT_DIR / f"search_trend_analysis_{mode}_{date_str}.xlsx"

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        if not base_df.empty:
            base_df.to_excel(writer, sheet_name="기본", index=False)

        if not segment_df.empty:
            for seg_type in segment_df["segment_type"].unique():
                sheet_name = {
                    "gender": "성별비교",
                    "age": "연령비교",
                    "device": "기기비교",
                }.get(seg_type, seg_type)
                sub = segment_df[segment_df["segment_type"] == seg_type]
                sub.to_excel(writer, sheet_name=sheet_name, index=False)

        if finals_df is not None and not finals_df.empty:
            finals_df.to_excel(writer, sheet_name="결승전", index=False)

        if anchor_df is not None and not anchor_df.empty:
            # 전체(base) 앵커 비교
            if "segment_type" in anchor_df.columns:
                base_anc = anchor_df[anchor_df["segment_type"].isna()]
            else:
                base_anc = anchor_df
            if not base_anc.empty:
                base_anc.to_excel(writer, sheet_name="앵커_전체", index=False)

            # 세분화별 앵커 비교
            if "segment_type" in anchor_df.columns:
                for seg_type in anchor_df["segment_type"].dropna().unique():
                    sheet_name = {
                        "gender": "앵커_성별",
                        "age": "앵커_연령",
                        "device": "앵커_기기",
                    }.get(seg_type, f"앵커_{seg_type}")
                    sub = anchor_df[anchor_df["segment_type"] == seg_type]
                    sub.to_excel(writer, sheet_name=sheet_name, index=False)

        if not mapping_df.empty:
            mapping_df.to_excel(writer, sheet_name="키워드매핑", index=False)

        summary_df = pd.DataFrame([summary])
        summary_df.to_excel(writer, sheet_name="실행요약", index=False)

    print(f"  저장: {path.name}")
    return str(path)


# ── 메인 실행 ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="네이버 DataLab 검색어 트렌드 분석")
    parser.add_argument("--mode", choices=["brand", "all-brand", "product"], default="brand", help="분석 단위")
    parser.add_argument(
        "--segment",
        choices=["none", "gender", "age", "device", "all"],
        default="none",
        help="세분화",
    )
    parser.add_argument("--start", default=DEFAULT_START_DATE, help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", default=DEFAULT_END_DATE, help="종료일 (YYYY-MM-DD)")
    parser.add_argument("--time-unit", choices=["date", "week", "month"], default=DEFAULT_TIME_UNIT)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="연착륙 상품 수")
    parser.add_argument("--test", action="store_true", help="상위 5개만 테스트")
    parser.add_argument("--no-cache", action="store_true", help="캐시 무시")
    args = parser.parse_args()

    top_n = 5 if args.test else args.top_n
    date_str = datetime.now().strftime("%Y%m%d")

    print("=" * 60)
    print("네이버 검색 트렌드 분석")
    print(f"  모드: {args.mode} | 세분화: {args.segment}")
    print(f"  기간: {args.start} ~ {args.end} | 단위: {args.time_unit}")
    print(f"  상품 수: {top_n}{'  (테스트)' if args.test else ''}")
    print("=" * 60)

    # 1. 키워드 그룹 생성
    print("\n[1] 키워드 그룹 생성")
    if args.mode == "all-brand":
        keyword_groups = build_all_brand_keyword_groups()
    elif args.mode == "brand":
        keyword_groups = build_brand_keyword_groups(top_n)
    else:
        keyword_groups = build_product_keyword_groups(top_n)

    print(f"  생성된 그룹: {len(keyword_groups)}개")
    for g in keyword_groups[:3]:
        print(f"    - {g['groupName']}: {g['keywords'][:2]}...")

    # 2. 클라이언트 초기화
    client = NaverTrendClient(use_cache=not args.no_cache)

    # 3. 기본 트렌드 조회
    print("\n[2] 기본 트렌드 조회")
    base_results = client.search_trend_batch(
        keyword_groups, args.start, args.end, args.time_unit
    )
    base_df = parse_trend_results(base_results)
    base_df["time_unit"] = args.time_unit
    print(f"  결과: {len(base_df)}행")

    # 4. 세분화 조회
    segment_df = pd.DataFrame()
    if args.segment != "none":
        print(f"\n[3] 세분화 조회: {args.segment}")
        segment_df = run_segment(
            client, keyword_groups, args.start, args.end, args.time_unit, args.segment
        )
        print(f"  결과: {len(segment_df)}행")

    # 5. 결승전: 배치별 1위 재비교
    finals_df = pd.DataFrame()
    if len(base_results) > 1:
        finals_df = run_finals(
            client, base_results, keyword_groups,
            args.start, args.end, args.time_unit, args.segment,
        )
        if not finals_df.empty:
            # 결승전 기본 라운드(세분화 없는 것)에서 진짜 1위 출력
            if "segment_type" in finals_df.columns:
                finals_base = finals_df[finals_df["segment_type"].isna()]
            else:
                finals_base = finals_df

            if not finals_base.empty:
                ranking = (
                    finals_base.groupby("keyword_group")["ratio"]
                    .mean()
                    .sort_values(ascending=False)
                )
                print("\n  === 결승전 결과 (평균 ratio) ===")
                for i, (name, val) in enumerate(ranking.items(), 1):
                    print(f"    {i}위: {name} ({val:.2f})")

    # 6. 앵커 비교: 전체 정규화 순위
    anchor_df = pd.DataFrame()
    if len(keyword_groups) > 5:
        if args.mode in ("brand", "all-brand"):
            print("\n[앵커 비교] 세그먼트별 앵커 기준 전체 브랜드 정규화")
            anchor_df = run_anchor_comparison(
                client, keyword_groups, args.start, args.end, args.time_unit
            )
        elif args.mode == "product":
            print("\n[앵커 비교] engagement 1위 기준 전체 제품 정규화")
            anchor_df = run_product_anchor_comparison(
                client, keyword_groups, args.start, args.end, args.time_unit
            )

        if not anchor_df.empty:
            # 전체(base) 정규화 랭킹 출력
            if "segment_type" in anchor_df.columns:
                base_anchor = anchor_df[anchor_df["segment_type"].isna()]
            else:
                base_anchor = anchor_df

            if not base_anchor.empty:
                ranking = (
                    base_anchor.groupby("keyword_group")["ratio_normalized"]
                    .mean()
                    .sort_values(ascending=False)
                )
                anchor_name = base_anchor["anchor"].iloc[0]
                print(f"\n  === 전체 정규화 순위 (앵커: {anchor_name}=100) ===")
                for i, (name, val) in enumerate(ranking.items(), 1):
                    print(f"    {i:2d}위: {name} ({val:.1f})")

    # 7. 키워드 매핑
    mapping_df = get_keyword_mapping(top_n)

    # 8. 저장
    print("\n[저장]")
    save_csv(base_df, f"search_trend_{args.mode}_base", date_str)
    if not segment_df.empty:
        save_csv(segment_df, f"search_trend_{args.mode}_segment", date_str)
    if not finals_df.empty:
        save_csv(finals_df, f"search_trend_{args.mode}_finals", date_str)
    if not anchor_df.empty:
        save_csv(anchor_df, f"search_trend_{args.mode}_anchor", date_str)
    save_csv(mapping_df, "keyword_mapping", date_str)

    # XLSX (openpyxl 필요)
    try:
        summary = {
            "mode": args.mode,
            "segment": args.segment,
            "start_date": args.start,
            "end_date": args.end,
            "time_unit": args.time_unit,
            "top_n": top_n,
            "keyword_groups": len(keyword_groups),
            "api_calls": client.api_call_count,
            "base_rows": len(base_df),
            "segment_rows": len(segment_df),
            "finals_rows": len(finals_df),
            "anchor_rows": len(anchor_df),
            "executed_at": datetime.now().isoformat(),
        }
        save_xlsx(base_df, segment_df, mapping_df, args.mode, date_str, summary,
                  finals_df=finals_df, anchor_df=anchor_df)
    except ImportError:
        print("  [참고] openpyxl 미설치 → XLSX 생성 생략 (pip install openpyxl)")

    # 7. 완료
    print("\n" + "=" * 60)
    client.print_key_stats()
    print(f"완료! API 호출: {client.api_call_count}회")
    print(f"출력 디렉토리: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
