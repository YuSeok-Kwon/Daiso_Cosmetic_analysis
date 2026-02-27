"""앵커 기반 SL vs Non-SL 검색트렌드 수집 스크립트

문제: 네이버 DataLab API의 ratio는 같은 배치(5개 그룹) 내 상대값이므로,
      SL/Non-SL을 따로 수집하면 배치 간 스케일이 달라 직접 비교 불가.

해결: 모든 배치에 동일한 앵커 제품을 포함시켜 앵커 대비 비율로 스케일 통일.
      - 앵커: 다이소 딥 클렌징 폼 (product_code=1035082)
      - 배치 구성: 앵커 1 + 제품 4 = 5그룹/배치

출력:
    - search_trend_anchor_raw_{date}.csv       — 원본 ratio (앵커 포함)
    - search_trend_anchor_normalized_{date}.csv — 앵커 대비 정규화 ratio
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

# 패키지 경로 보정
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]  # Why-pi/
_SRC_DIR = _THIS_DIR.parent / "05_src"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pandas as pd
import re
import os
from dotenv import load_dotenv
import hashlib
import json
import time
import requests
from typing import Optional

# 환경변수 로드
CONFIG_DIR = _PROJECT_ROOT / "config"
load_dotenv(CONFIG_DIR / ".env")

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID", "")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET", "")
NAVER_DATALAB_URL = "https://openapi.naver.com/v1/datalab/search"

# 복수 키 로드 (키 로테이션용)
NAVER_API_KEYS: list[dict[str, str]] = []
_key_idx = 1
while True:
    _suffix = "" if _key_idx == 1 else str(_key_idx)
    _cid = os.getenv(f"NAVER_CLIENT_ID{_suffix}", "")
    _csec = os.getenv(f"NAVER_CLIENT_SECRET{_suffix}", "")
    if not _cid or not _csec:
        break
    NAVER_API_KEYS.append({"client_id": _cid, "client_secret": _csec})
    _key_idx += 1

# 출력 디렉토리
OUTPUT_DIR = _PROJECT_ROOT / "02_outputs" / "Search_Trend"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 캐시 디렉토리
CACHE_DIR = _SRC_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 기본 설정
DEFAULT_START_DATE = "2024-01-01"
DEFAULT_END_DATE = "2026-01-31"
DEFAULT_TIME_UNIT = "month"

# ── 앵커 제품 설정 ──────────────────────────────────────
ANCHOR_PRODUCT_CODE = 1035082
ANCHOR_GROUP = {
    "groupName": "ANCHOR_딥클렌징폼",
    "keywords": [
        "다이소 딥 클렌징 폼",
        "다이소 딥클렌징폼",
        "딥 클렌징 폼",
        "다이소 딥 클렌징 폼 추천",
        "다이소 딥 클렌징 폼 후기",
        "다이소 딥 클렌징 폼 리뷰",
        "다이소 딥 클렌징 폼 가격",
    ],
}


# ── NaverTrendClient (키 로테이션 지원) ─────────────────
class NaverTrendClient:
    """네이버 DataLab 검색어 트렌드 API 클라이언트 (키 로테이션 지원)"""

    def __init__(self, use_cache: bool = True):
        if NAVER_API_KEYS:
            self.api_keys = NAVER_API_KEYS
        elif NAVER_CLIENT_ID and NAVER_CLIENT_SECRET:
            self.api_keys = [{"client_id": NAVER_CLIENT_ID, "client_secret": NAVER_CLIENT_SECRET}]
        else:
            raise ValueError("NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 이 설정되지 않았습니다.")

        self._current_key_idx = 0
        self._call_counts = [0] * len(self.api_keys)

        self.use_cache = use_cache
        self.cache_file = CACHE_DIR / "search_trend_anchor_cache.json"
        self.cache = self._load_cache() if use_cache else {}
        self._cache_writes = 0
        self.api_call_count = 0
        self.request_delay = 0.5

        if len(self.api_keys) > 1:
            print(f"  API 키 {len(self.api_keys)}개 로드 (라운드로빈 로테이션)")

    def _load_cache(self) -> dict:
        if self.cache_file.exists():
            with open(self.cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _cache_key(body: dict) -> str:
        raw = json.dumps(body, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _flush_if_needed(self):
        self._cache_writes += 1
        if self._cache_writes % 5 == 0:
            self._save_cache()

    def flush_cache(self):
        if self.use_cache and self.cache:
            self._save_cache()

    def _rotate_key(self):
        self._current_key_idx = (self._current_key_idx + 1) % len(self.api_keys)

    def _current_headers(self) -> dict:
        key = self.api_keys[self._current_key_idx]
        return {
            "X-Naver-Client-Id": key["client_id"],
            "X-Naver-Client-Secret": key["client_secret"],
            "Content-Type": "application/json",
        }

    def _request(self, body: dict, max_retries: int = 3) -> Optional[dict]:
        keys_exhausted = 0

        for attempt in range(max_retries):
            headers = self._current_headers()

            try:
                resp = requests.post(NAVER_DATALAB_URL, headers=headers, json=body, timeout=15)

                if resp.status_code == 200:
                    self.api_call_count += 1
                    self._call_counts[self._current_key_idx] += 1
                    self._rotate_key()
                    return resp.json()

                if resp.status_code == 429:
                    keys_exhausted += 1
                    if keys_exhausted < len(self.api_keys):
                        kid = self.api_keys[self._current_key_idx]["client_id"][:4]
                        print(f"  [429] 키 {kid}... 한도 초과 → 다음 키로 전환")
                        self._rotate_key()
                        continue
                    wait = 60
                    print(f"  [429] 모든 키 한도 초과 — {wait}초 대기 (attempt {attempt + 1})")
                    time.sleep(wait)
                    keys_exhausted = 0
                    continue

                if resp.status_code >= 500:
                    wait = 5
                    print(f"  [{resp.status_code}] 서버 에러 — {wait}초 대기 (attempt {attempt + 1})")
                    time.sleep(wait)
                    continue

                print(f"  [{resp.status_code}] 요청 실패: {resp.text[:200]}")
                return None

            except requests.exceptions.Timeout:
                print(f"  타임아웃 (attempt {attempt + 1})")
                time.sleep(2)
            except requests.exceptions.RequestException as e:
                print(f"  요청 에러: {e}")
                time.sleep(1)

        return None

    def search_trend(
        self,
        keyword_groups: list[dict],
        start_date: str,
        end_date: str,
        time_unit: str = "month",
    ) -> Optional[dict]:
        if len(keyword_groups) > 5:
            raise ValueError("keyword_groups는 최대 5개까지만 가능합니다.")

        body = {
            "startDate": start_date,
            "endDate": end_date,
            "timeUnit": time_unit,
            "keywordGroups": keyword_groups,
        }

        key = self._cache_key(body)
        if self.use_cache and key in self.cache:
            return self.cache[key]

        time.sleep(self.request_delay)
        result = self._request(body)

        if result and self.use_cache:
            self.cache[key] = result
            self._flush_if_needed()

        return result

    def print_key_stats(self):
        if len(self.api_keys) <= 1:
            return
        print("API 호출 통계:")
        for i, key in enumerate(self.api_keys):
            kid = key["client_id"][:4]
            print(f"  키 {i + 1} ({kid}...): {self._call_counts[i]}회")
        print(f"  총합: {self.api_call_count}회")


# ── 전체 제품 로드 ───────────────────────────────────────
def load_all_products() -> pd.DataFrame:
    """SLI 전체 제품 로드 (SL + Non-SL)"""
    sli_path = _PROJECT_ROOT / "02_outputs" / "Sli" / "sli_integrated_results.csv"
    df = pd.read_csv(sli_path)
    df["is_sl"] = df["final_soft_landing"].astype(bool)
    print(f"전체 제품 수: {len(df)} (SL: {df['is_sl'].sum()}, Non-SL: {(~df['is_sl']).sum()})")
    return df


# ── 키워드 추출 ──────────────────────────────────────────
def extract_search_keyword(product_name: str, brand_name: str) -> str:
    kw = product_name
    kw = re.sub(r"\d+\s*ml\*?\d*개?입?", "", kw, flags=re.IGNORECASE)
    kw = re.sub(r"\d+\s*(ml|g|mg|l|매)\b", "", kw, flags=re.IGNORECASE)
    kw = re.sub(r"\[[^\]]*\]", "", kw)
    kw = re.sub(r"\([^)]*\)", "", kw)
    if brand_name and kw.strip().startswith(brand_name):
        kw = kw.strip()[len(brand_name):]
    kw = re.sub(r"\s+", " ", kw).strip()
    return kw


# ── 키워드 그룹 생성 ─────────────────────────────────────
def build_keyword_groups(df: pd.DataFrame) -> list[dict]:
    """전체 제품의 키워드 그룹 생성 (앵커 제외)"""
    groups = []
    used_names = set()

    for _, row in df.iterrows():
        # 앵커 제품은 별도 처리
        if row["product_code"] == ANCHOR_PRODUCT_CODE:
            continue

        brand = row["brand_name"]
        name = row["name"]
        product_code = row["product_code"]

        kw = extract_search_keyword(name, brand)
        if not kw or brand == "ALL":
            continue

        if brand == "다이소":
            keywords = [
                f"다이소 {kw}",
                kw,
                f"다이소 {kw} 추천",
                f"다이소 {kw} 후기",
                f"다이소 {kw} 리뷰",
                f"다이소 {kw} 가격",
                f"다이소 {kw} 가성비",
            ]
        else:
            keywords = [
                f"{brand} {kw}",
                f"다이소 {brand} {kw}",
                f"다이소 {kw}",
                f"{brand} {kw} 추천",
                f"다이소 {brand} {kw} 후기",
                f"{brand} {kw} 리뷰",
                f"다이소 {brand}",
                f"{brand} {kw} 가격",
            ]

        keywords = list(dict.fromkeys(keywords))[:10]

        words = kw.split()
        short_name = " ".join(words[:2]) if len(words) >= 2 else kw
        candidate = f"{brand} {short_name}"

        if candidate in used_names:
            candidate = f"{brand} {' '.join(words[:3])}" if len(words) >= 3 else candidate
        if candidate in used_names:
            candidate = f"{candidate} {product_code}"

        used_names.add(candidate)

        groups.append({
            "groupName": candidate,
            "keywords": keywords,
            "product_code": product_code,
            "brand_name": brand,
            "product_name": name,
            "is_sl": bool(row["is_sl"]),
        })

    return groups


# ── 앵커 배치 생성 ────────────────────────────────────────
def build_anchor_batches(groups: list[dict]) -> list[list[dict]]:
    """앵커 1 + 제품 4 = 5그룹/배치로 분할

    각 배치의 첫 번째 그룹은 항상 앵커이므로,
    나머지 4자리에 제품을 순서대로 채움.
    """
    anchor_api_group = {
        "groupName": ANCHOR_GROUP["groupName"],
        "keywords": ANCHOR_GROUP["keywords"],
    }

    batches = []
    for i in range(0, len(groups), 4):
        chunk = groups[i:i + 4]
        batch = [anchor_api_group] + [
            {"groupName": g["groupName"], "keywords": g["keywords"]}
            for g in chunk
        ]
        batches.append(batch)

    return batches


# ── 배치 호출 ─────────────────────────────────────────────
def run_anchor_batches(
    client: NaverTrendClient,
    batches: list[list[dict]],
    start_date: str,
    end_date: str,
    time_unit: str,
) -> list[dict]:
    """모든 배치 호출 (각 배치: 앵커 + 제품4)"""
    results = []
    total = len(batches)

    print(f"  앵커 배치 호출: {total}배치 (앵커1 + 제품4)")

    for idx, batch in enumerate(batches, 1):
        resp = client.search_trend(batch, start_date, end_date, time_unit)
        if resp:
            results.append(resp)
            if idx % 20 == 0 or idx == total:
                print(f"    배치 {idx}/{total} 완료 (API 호출: {client.api_call_count})")
        else:
            print(f"    배치 {idx}/{total} 실패")

    client.flush_cache()
    return results


# ── 응답 파싱 (앵커 포함) ─────────────────────────────────
def parse_anchor_results(
    results: list[dict],
    groups_mapping: dict,
) -> pd.DataFrame:
    """API 응답 → DataFrame 변환 (앵커 ratio 포함)"""
    rows = []

    for resp in results:
        if not resp or "results" not in resp:
            continue

        # 앵커 ratio 추출 (이 배치에서의 앵커 시계열)
        anchor_data = {}
        product_data = []

        for group in resp["results"]:
            group_name = group.get("title", "")
            data_points = group.get("data", [])

            if group_name == ANCHOR_GROUP["groupName"]:
                for point in data_points:
                    anchor_data[point["period"]] = point["ratio"]
            else:
                product_data.append((group_name, data_points))

        # 앵커 ratio가 없으면 스킵
        if not anchor_data:
            continue

        # 앵커 자체 행도 추가
        for period, ratio in anchor_data.items():
            rows.append({
                "keyword_group": ANCHOR_GROUP["groupName"],
                "product_code": ANCHOR_PRODUCT_CODE,
                "brand_name": "다이소",
                "product_name": "딥 클렌징 폼 150 ml",
                "is_sl": True,
                "period": period,
                "raw_ratio": ratio,
                "anchor_ratio": ratio,
                "normalized_ratio": 1.0,  # 앵커 자체는 항상 1.0
            })

        # 제품별 정규화
        for group_name, data_points in product_data:
            meta = groups_mapping.get(group_name, {})

            for point in data_points:
                period = point["period"]
                raw_ratio = point["ratio"]
                anchor_r = anchor_data.get(period, 0)

                # 앵커 ratio가 0이면 정규화 불가
                if anchor_r > 0:
                    norm_ratio = raw_ratio / anchor_r
                else:
                    norm_ratio = None

                rows.append({
                    "keyword_group": group_name,
                    "product_code": meta.get("product_code", ""),
                    "brand_name": meta.get("brand_name", ""),
                    "product_name": meta.get("product_name", ""),
                    "is_sl": meta.get("is_sl", False),
                    "period": period,
                    "raw_ratio": raw_ratio,
                    "anchor_ratio": anchor_r,
                    "normalized_ratio": norm_ratio,
                })

    return pd.DataFrame(rows)


# ── 메인 실행 ─────────────────────────────────────────────
def main():
    date_str = datetime.now().strftime("%Y%m%d")

    print("=" * 60)
    print("앵커 기반 SL vs Non-SL 검색트렌드 수집")
    print(f"앵커: 다이소 딥 클렌징 폼 (product_code={ANCHOR_PRODUCT_CODE})")
    print(f"기간: {DEFAULT_START_DATE} ~ {DEFAULT_END_DATE}")
    print(f"단위: {DEFAULT_TIME_UNIT}")
    print("=" * 60)

    # 1. 전체 제품 로드
    print("\n[1] 전체 제품 로드")
    all_df = load_all_products()

    # 2. 키워드 그룹 생성
    print("\n[2] 키워드 그룹 생성 (앵커 제외)")
    keyword_groups = build_keyword_groups(all_df)
    print(f"생성된 그룹: {len(keyword_groups)}개")
    sl_count = sum(1 for g in keyword_groups if g["is_sl"])
    non_sl_count = len(keyword_groups) - sl_count
    print(f"  SL: {sl_count}개, Non-SL: {non_sl_count}개")

    # 그룹 메타데이터 매핑
    groups_mapping = {
        g["groupName"]: {
            "product_code": g["product_code"],
            "brand_name": g["brand_name"],
            "product_name": g["product_name"],
            "is_sl": g["is_sl"],
        }
        for g in keyword_groups
    }

    # 3. 앵커 배치 생성
    print("\n[3] 앵커 배치 생성")
    batches = build_anchor_batches(keyword_groups)
    print(f"배치 수: {len(batches)} (앵커1 + 제품4 × {len(batches)})")

    # 4. API 호출
    print("\n[4] 네이버 DataLab API 호출")
    client = NaverTrendClient(use_cache=True)

    try:
        results = run_anchor_batches(
            client, batches,
            DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_TIME_UNIT
        )
        print(f"\nAPI 호출 완료: {len(results)}개 배치, 실제 API 호출: {client.api_call_count}회")
        client.print_key_stats()
    except Exception as e:
        print(f"[오류] API 호출 실패: {e}")
        results = []

    # 5. 결과 파싱 및 저장
    print("\n[5] 결과 처리 및 저장")

    if not results:
        print("[안내] API 호출 결과가 없습니다.")
        return

    detail_df = parse_anchor_results(results, groups_mapping)

    if detail_df.empty:
        print("[경고] 유효한 데이터를 추출하지 못했습니다.")
        return

    # 앵커 중복 행 제거 (배치마다 앵커가 반복되므로 첫 번째만 유지)
    anchor_mask = detail_df["product_code"] == ANCHOR_PRODUCT_CODE
    anchor_df = detail_df[anchor_mask].drop_duplicates(subset=["period"], keep="first")
    product_df = detail_df[~anchor_mask]
    detail_df = pd.concat([anchor_df, product_df], ignore_index=True)

    # product_code를 int로 변환 (빈 문자열 제외)
    detail_df["product_code"] = pd.to_numeric(detail_df["product_code"], errors="coerce").astype("Int64")

    # 원본 raw 저장
    raw_path = OUTPUT_DIR / f"search_trend_anchor_raw_{date_str}.csv"
    detail_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"저장: {raw_path.name} ({len(detail_df)}행)")

    # 정규화 데이터만 별도 저장
    norm_df = detail_df[detail_df["normalized_ratio"].notna()].copy()
    norm_path = OUTPUT_DIR / f"search_trend_anchor_normalized_{date_str}.csv"
    norm_df.to_csv(norm_path, index=False, encoding="utf-8-sig")
    print(f"저장: {norm_path.name} ({len(norm_df)}행)")

    # 6. 검증
    print("\n[6] 데이터 검증")
    n_products = detail_df["product_code"].nunique()
    n_periods = detail_df["period"].nunique()
    print(f"제품 수: {n_products}")
    print(f"시점 수: {n_periods}")

    # 앵커 normalized_ratio 검증
    anchor_norm = detail_df[detail_df["product_code"] == ANCHOR_PRODUCT_CODE]["normalized_ratio"]
    print(f"앵커 normalized_ratio: min={anchor_norm.min()}, max={anchor_norm.max()} (모두 1.0이어야 함)")

    # SL vs Non-SL 통계
    sl_norm = norm_df[norm_df["is_sl"] == True]
    non_sl_norm = norm_df[norm_df["is_sl"] == False]
    print(f"\nSL 제품: {sl_norm['product_code'].nunique()}개, 행 수: {len(sl_norm)}")
    print(f"Non-SL 제품: {non_sl_norm['product_code'].nunique()}개, 행 수: {len(non_sl_norm)}")

    # 간단한 비교
    sl_mean = sl_norm.groupby("product_code")["normalized_ratio"].mean()
    non_sl_mean = non_sl_norm.groupby("product_code")["normalized_ratio"].mean()
    print(f"\n앵커 대비 평균 normalized_ratio:")
    print(f"  SL 평균: {sl_mean.mean():.3f}, 중앙값: {sl_mean.median():.3f}")
    print(f"  Non-SL 평균: {non_sl_mean.mean():.3f}, 중앙값: {non_sl_mean.median():.3f}")

    print("\n" + "=" * 60)
    print(f"완료! 출력 디렉토리: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
