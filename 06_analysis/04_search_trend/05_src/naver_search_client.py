"""네이버 검색 API 클라이언트 (블로그 / 쇼핑 / 뉴스)

네이버 DataLab 트렌드 API가 '상대 비율(0~100)'만 제공하는 반면,
검색 API는 실제 검색 결과(총 건수, 개별 문서)를 반환합니다.

활용:
- 블로그: 브랜드/제품 언급량, 리뷰 텍스트 수집
- 쇼핑: 경쟁 상품 수, 가격대, 쇼핑몰 분포
- 뉴스: 미디어 노출량, 트렌드 키워드 탐지

API 문서: https://developers.naver.com/docs/serviceapi/search/blog/blog.md
"""

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from .config import NAVER_CLIENT_ID, NAVER_CLIENT_SECRET, NAVER_API_KEYS, CACHE_DIR


# ── 엔드포인트 ────────────────────────────────────────────
SEARCH_ENDPOINTS = {
    "blog": "https://openapi.naver.com/v1/search/blog.json",
    "shop": "https://openapi.naver.com/v1/search/shop.json",
    "news": "https://openapi.naver.com/v1/search/news.json",
}

# 정렬 옵션
SORT_OPTIONS = {
    "blog": ["sim", "date"],       # 정확도순, 날짜순
    "shop": ["sim", "date", "asc", "dsc"],  # 정확도, 날짜, 가격 오름/내림
    "news": ["sim", "date"],
}

# API 제한
MAX_DISPLAY = 100   # 한 번에 최대 100건
MAX_START = 1000    # start 최대 1000


class NaverSearchClient:
    """네이버 검색 API 클라이언트

    - 블로그/쇼핑/뉴스 3개 엔드포인트 지원
    - MD5 기반 JSON 캐시 (DataLab 클라이언트와 동일 패턴)
    - 429/5xx 자동 재시도
    - 페이지네이션 (최대 1000건)
    """

    def __init__(
        self,
        client_id: str = None,
        client_secret: str = None,
        api_keys: list[dict] = None,
        cache_name: str = "naver_search",
        use_cache: bool = True,
    ):
        # 키 로테이션: api_keys > NAVER_API_KEYS > 단일 키 폴백
        if api_keys:
            self.api_keys = api_keys
        elif NAVER_API_KEYS:
            self.api_keys = NAVER_API_KEYS
        else:
            cid = client_id or NAVER_CLIENT_ID
            csec = client_secret or NAVER_CLIENT_SECRET
            if not cid or not csec:
                raise ValueError(
                    "NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 이 설정되지 않았습니다.\n"
                    "config/.env 파일에 아래 값을 추가하세요:\n"
                    "  NAVER_CLIENT_ID=your_id\n"
                    "  NAVER_CLIENT_SECRET=your_secret\n"
                    "발급: https://developers.naver.com/apps/#/register"
                )
            self.api_keys = [{"client_id": cid, "client_secret": csec}]

        # 하위 호환: 첫 번째 키를 단일 키 속성으로 유지
        self.client_id = self.api_keys[0]["client_id"]
        self.client_secret = self.api_keys[0]["client_secret"]

        self._current_key_idx = 0
        self._call_counts = [0] * len(self.api_keys)

        self.use_cache = use_cache
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / f"{cache_name}_cache.json"
        self.cache: dict = self._load_cache() if use_cache else {}
        self._cache_writes = 0

        self.request_delay = 0.1  # 검색 API는 초당 10회 허용
        self.api_call_count = 0

        if len(self.api_keys) > 1:
            print(f"  API 키 {len(self.api_keys)}개 로드 (라운드로빈 로테이션)")

    # ── 캐시 ──────────────────────────────────────────────

    def _load_cache(self) -> dict:
        if self.cache_file.exists():
            with open(self.cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _cache_key(endpoint: str, params: dict) -> str:
        raw = json.dumps({"endpoint": endpoint, **params}, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _flush_if_needed(self):
        self._cache_writes += 1
        if self._cache_writes % 10 == 0:
            self._save_cache()

    def flush_cache(self):
        if self.use_cache and self.cache:
            self._save_cache()

    # ── API 호출 ──────────────────────────────────────────

    def _rotate_key(self):
        """다음 키로 라운드로빈 전환"""
        self._current_key_idx = (self._current_key_idx + 1) % len(self.api_keys)

    def _current_headers(self) -> dict:
        key = self.api_keys[self._current_key_idx]
        return {
            "X-Naver-Client-Id": key["client_id"],
            "X-Naver-Client-Secret": key["client_secret"],
        }

    def _request(self, url: str, params: dict, max_retries: int = 3) -> Optional[dict]:
        keys_exhausted = 0  # 연속 429로 소진된 키 수

        for attempt in range(max_retries):
            headers = self._current_headers()

            try:
                resp = requests.get(url, headers=headers, params=params, timeout=10)

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
                    # 모든 키 소진
                    wait = 60
                    print(f"  [429] 모든 키 한도 초과 — {wait}초 대기 (attempt {attempt + 1})")
                    time.sleep(wait)
                    keys_exhausted = 0
                    continue

                if resp.status_code >= 500:
                    wait = 3
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

    def print_key_stats(self):
        """키별 API 호출 통계 출력"""
        if len(self.api_keys) <= 1:
            return
        print("API 호출 통계:")
        for i, key in enumerate(self.api_keys):
            kid = key["client_id"][:4]
            print(f"  키 {i + 1} ({kid}...): {self._call_counts[i]}회")
        print(f"  총합: {self.api_call_count}회")

    # ── 공개 메서드 ───────────────────────────────────────

    def search(
        self,
        endpoint: str,
        query: str,
        display: int = 10,
        start: int = 1,
        sort: str = "sim",
    ) -> Optional[dict]:
        """단건 검색

        Parameters
        ----------
        endpoint : str  "blog" | "shop" | "news"
        query : str     검색어
        display : int   결과 수 (1~100)
        start : int     시작 위치 (1~1000)
        sort : str      정렬 ("sim": 정확도, "date": 날짜, "asc"/"dsc": 가격)

        Returns
        -------
        dict: {"lastBuildDate", "total", "start", "display", "items": [...]}
        """
        if endpoint not in SEARCH_ENDPOINTS:
            raise ValueError(f"지원하지 않는 엔드포인트: {endpoint}. 사용 가능: {list(SEARCH_ENDPOINTS.keys())}")

        url = SEARCH_ENDPOINTS[endpoint]
        params = {
            "query": query,
            "display": min(display, MAX_DISPLAY),
            "start": min(start, MAX_START),
            "sort": sort,
        }

        # 캐시
        key = self._cache_key(endpoint, params)
        if self.use_cache and key in self.cache:
            return self.cache[key]

        time.sleep(self.request_delay)
        result = self._request(url, params)

        if result and self.use_cache:
            self.cache[key] = result
            self._flush_if_needed()

        return result

    def search_total(self, endpoint: str, query: str, sort: str = "sim") -> int:
        """검색 결과 총 건수만 반환 (display=1로 빠르게 조회)"""
        result = self.search(endpoint, query, display=1, start=1, sort=sort)
        if result:
            return result.get("total", 0)
        return 0

    def search_all_pages(
        self,
        endpoint: str,
        query: str,
        max_items: int = 100,
        sort: str = "sim",
    ) -> list[dict]:
        """페이지네이션으로 여러 페이지 수집 (최대 1000건)

        Returns
        -------
        list[dict] : items 리스트 (중복 제거된 전체 결과)
        """
        max_items = min(max_items, MAX_START)  # API 한계
        all_items = []
        start = 1

        while start <= max_items:
            display = min(MAX_DISPLAY, max_items - start + 1)
            result = self.search(endpoint, query, display=display, start=start, sort=sort)

            if not result or "items" not in result:
                break

            items = result["items"]
            if not items:
                break

            all_items.extend(items)
            start += len(items)

            # 총 결과 수보다 더 가져올 것이 없으면 중단
            total = result.get("total", 0)
            if start > total:
                break

        return all_items

    def search_bulk_keywords(
        self,
        endpoint: str,
        keywords: list[str],
        sort: str = "sim",
    ) -> dict[str, int]:
        """여러 키워드의 검색 결과 총 건수를 일괄 조회

        Returns
        -------
        dict: {키워드: 총건수, ...}
        """
        totals = {}
        for i, kw in enumerate(keywords):
            total = self.search_total(endpoint, kw, sort=sort)
            totals[kw] = total
            if (i + 1) % 20 == 0:
                print(f"    {i + 1}/{len(keywords)} 완료")

        self.flush_cache()
        return totals
