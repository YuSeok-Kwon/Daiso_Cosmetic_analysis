"""비연착륙(Non-SL) 제품 560개 네이버 검색트렌드 ratio 수집 스크립트

SL vs Non-SL 시계열 비교분석을 위해 Non-SL 제품의 ratio 데이터를 수집한다.
기존 run_soft_landing_search_trend.py 와 동일한 로직/캐시를 사용한다.
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

# 출력 디렉토리
OUTPUT_DIR = _PROJECT_ROOT / "02_outputs" / "Search_Trend"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 캐시 디렉토리 (SL과 동일한 캐시 공유)
CACHE_DIR = _SRC_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 기본 날짜 설정 (SL과 동일)
DEFAULT_START_DATE = "2024-01-01"
DEFAULT_END_DATE = "2026-01-31"
DEFAULT_TIME_UNIT = "month"


# ── NaverTrendClient (SL 스크립트와 동일) ─────────────
class NaverTrendClient:
    """네이버 DataLab 검색어 트렌드 API 클라이언트"""

    def __init__(self, use_cache: bool = True):
        if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
            raise ValueError("NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 이 설정되지 않았습니다.")

        self.client_id = NAVER_CLIENT_ID
        self.client_secret = NAVER_CLIENT_SECRET
        self.use_cache = use_cache
        self.cache_file = CACHE_DIR / "search_trend_cache.json"
        self.cache = self._load_cache() if use_cache else {}
        self.api_call_count = 0
        self.request_delay = 0.5

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

    def flush_cache(self):
        if self.use_cache and self.cache:
            self._save_cache()

    def _request(self, body: dict, max_retries: int = 3) -> Optional[dict]:
        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
            "Content-Type": "application/json",
        }

        for attempt in range(max_retries):
            try:
                resp = requests.post(NAVER_DATALAB_URL, headers=headers, json=body, timeout=15)

                if resp.status_code == 200:
                    self.api_call_count += 1
                    return resp.json()

                if resp.status_code == 429:
                    wait = 60
                    print(f"  [429] API 한도 초과 — {wait}초 대기 (attempt {attempt + 1})")
                    time.sleep(wait)
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
            self._save_cache()

        return result

    def search_trend_batch(
        self,
        all_keyword_groups: list[dict],
        start_date: str,
        end_date: str,
        time_unit: str = "month",
    ) -> list[dict]:
        results = []
        total = len(all_keyword_groups)
        batches = [all_keyword_groups[i : i + 5] for i in range(0, total, 5)]

        print(f"  배치 호출: {total}그룹 → {len(batches)}배치")

        for idx, batch in enumerate(batches, 1):
            resp = self.search_trend(batch, start_date, end_date, time_unit)
            if resp:
                results.append(resp)
                if idx % 20 == 0 or idx == len(batches):
                    print(f"    배치 {idx}/{len(batches)} 완료 (API 호출: {self.api_call_count})")
            else:
                print(f"    배치 {idx}/{len(batches)} 실패")

        self.flush_cache()
        return results


# ── Non-SL 제품 로드 ──────────────────────────────────
def load_non_sl_products() -> pd.DataFrame:
    """비연착륙(Non-SL) 제품 560개 로드"""
    sli_path = _PROJECT_ROOT / "02_outputs" / "Sli" / "sli_integrated_results.csv"
    df = pd.read_csv(sli_path)
    non_sl_df = df[df['final_soft_landing'] == 0].copy()

    print(f"Non-SL 제품 수: {len(non_sl_df)}")
    return non_sl_df


# ── 키워드 추출 (SL과 동일) ─────────────────────────────
def extract_search_keyword(product_name: str, brand_name: str) -> str:
    kw = product_name

    # 용량/단위 제거
    kw = re.sub(r"\d+\s*ml\*?\d*개?입?", "", kw, flags=re.IGNORECASE)
    kw = re.sub(r"\d+\s*(ml|g|mg|l|매)\b", "", kw, flags=re.IGNORECASE)

    # 대괄호·소괄호 내용 제거
    kw = re.sub(r"\[[^\]]*\]", "", kw)
    kw = re.sub(r"\([^)]*\)", "", kw)

    # 브랜드명 제거
    if brand_name and kw.strip().startswith(brand_name):
        kw = kw.strip()[len(brand_name):]

    kw = re.sub(r"\s+", " ", kw).strip()
    return kw


# ── 키워드 그룹 생성 (SL과 동일 로직) ──────────────────
def build_keyword_groups(df: pd.DataFrame) -> list[dict]:
    """제품별 키워드 그룹 생성"""
    groups = []
    used_names = set()

    for _, row in df.iterrows():
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
        })

    return groups


# ── 응답 파싱 (SL과 동일) ─────────────────────────────
def parse_trend_results(results: list[dict], groups_mapping: dict) -> pd.DataFrame:
    rows = []
    for resp in results:
        if not resp or "results" not in resp:
            continue
        for group in resp["results"]:
            group_name = group.get("title", "")
            meta = groups_mapping.get(group_name, {})

            for point in group.get("data", []):
                rows.append({
                    "keyword_group": group_name,
                    "product_code": meta.get("product_code", ""),
                    "brand_name": meta.get("brand_name", ""),
                    "product_name": meta.get("product_name", ""),
                    "period": point.get("period", ""),
                    "ratio": point.get("ratio", 0),
                })

    return pd.DataFrame(rows)


# ── 요약 통계 생성 ──────────────────────────────────────
def create_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    summary = detail_df.groupby(["product_code", "product_name", "brand_name"]).agg(
        total_ratio=("ratio", "sum"),
        avg_ratio=("ratio", "mean"),
        max_ratio=("ratio", "max"),
        min_ratio=("ratio", "min"),
    ).reset_index()

    def classify_trend(group):
        if len(group) < 2:
            return "데이터 부족"
        first_half = group.head(len(group)//2)["ratio"].mean()
        second_half = group.tail(len(group)//2)["ratio"].mean()

        if second_half > first_half * 1.2:
            return "상승"
        elif second_half < first_half * 0.8:
            return "하락"
        else:
            return "안정"

    trends = detail_df.groupby("product_code").apply(classify_trend)
    summary = summary.merge(
        trends.reset_index().rename(columns={0: "trend_category"}),
        on="product_code",
        how="left"
    )

    return summary.sort_values("total_ratio", ascending=False)


# ── 메인 실행 ───────────────────────────────────────────
def main():
    date_str = datetime.now().strftime("%Y%m%d")

    print("=" * 60)
    print("Non-SL(비연착륙) 제품 네이버 검색트렌드 수집")
    print(f"기간: {DEFAULT_START_DATE} ~ {DEFAULT_END_DATE}")
    print(f"단위: {DEFAULT_TIME_UNIT}")
    print("=" * 60)

    # 1. Non-SL 제품 로드
    print("\n[1] Non-SL 제품 로드")
    non_sl_df = load_non_sl_products()

    # 2. 키워드 그룹 생성
    print("\n[2] 키워드 그룹 생성")
    keyword_groups = build_keyword_groups(non_sl_df)
    print(f"생성된 그룹: {len(keyword_groups)}개")
    for g in keyword_groups[:3]:
        print(f"  - {g['groupName']}: {g['keywords'][:2]}...")

    # 그룹 메타데이터 매핑
    groups_mapping = {
        g["groupName"]: {
            "product_code": g["product_code"],
            "brand_name": g["brand_name"],
            "product_name": g["product_name"]
        }
        for g in keyword_groups
    }

    # 3. API 호출
    print("\n[3] 네이버 DataLab API 호출")
    n_batches = (len(keyword_groups) + 4) // 5
    print(f"배치 수: {n_batches} (560개 제품 → 5개씩)")

    client = NaverTrendClient(use_cache=True)

    try:
        results = client.search_trend_batch(
            [{"groupName": g["groupName"], "keywords": g["keywords"]} for g in keyword_groups],
            DEFAULT_START_DATE,
            DEFAULT_END_DATE,
            DEFAULT_TIME_UNIT
        )
        print(f"API 호출 완료: {len(results)}개 배치, 실제 API 호출: {client.api_call_count}회")
    except Exception as e:
        print(f"[오류] API 호출 실패: {e}")
        results = []

    # 4. 결과 파싱 및 저장
    print("\n[4] 결과 처리 및 저장")

    if results:
        detail_df = parse_trend_results(results, groups_mapping)

        if not detail_df.empty:
            # 상세 데이터 저장
            detail_path = OUTPUT_DIR / f"search_trend_non_sl_product_{date_str}.csv"
            detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
            print(f"저장: {detail_path.name} ({len(detail_df)}행)")

            # 요약 통계 저장
            summary_df = create_summary(detail_df)
            summary_path = OUTPUT_DIR / f"search_trend_non_sl_summary_{date_str}.csv"
            summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
            print(f"저장: {summary_path.name} ({len(summary_df)}행)")

            # 주요 발견 출력
            print("\n[주요 발견]")
            print(f"총 제품 수: {len(summary_df)}")
            print(f"평균 검색 ratio: {summary_df['avg_ratio'].mean():.2f}")
            print(f"최대 검색 ratio: {summary_df['max_ratio'].max():.2f}")
            print(f"\n트렌드 분류:")
            for cat, cnt in summary_df['trend_category'].value_counts().items():
                print(f"  {cat}: {cnt}개 ({cnt/len(summary_df)*100:.1f}%)")
        else:
            print("[경고] API 응답에서 유효한 데이터를 추출하지 못했습니다.")
    else:
        print("[안내] API 호출 결과가 없습니다.")

    # 5. 완료
    print("\n" + "=" * 60)
    print(f"완료! 출력 디렉토리: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
