"""네이버 DataLab 검색어 트렌드 API 클라이언트"""

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import requests

from .config import NAVER_CLIENT_ID, NAVER_CLIENT_SECRET, NAVER_API_KEYS, NAVER_DATALAB_URL, CACHE_DIR


class NaverTrendClient:
    """네이버 DataLab 검색어 트렌드 API 클라이언트

    - MD5 해시 기반 JSON 캐시
    - 429/5xx 자동 재시도
    - 5그룹 초과 시 자동 배치 분할
    """

    def __init__(
        self,
        client_id: str = None,
        client_secret: str = None,
        api_keys: list[dict] = None,
        cache_name: str = "search_trend",
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

        self.request_delay = 0.5  # API 호출 간격 (초)
        self.api_call_count = 0

        if len(self.api_keys) > 1:
            print(f"  API 키 {len(self.api_keys)}개 로드 (라운드로빈 로테이션)")

    # ── 캐시 ────────────────────────────────────────────

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
        """요청 body를 MD5 해시로 변환하여 캐시 키 생성"""
        raw = json.dumps(body, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _flush_if_needed(self):
        """5회마다 캐시 파일 저장"""
        self._cache_writes += 1
        if self._cache_writes % 5 == 0:
            self._save_cache()

    def flush_cache(self):
        """캐시를 디스크에 강제 저장"""
        if self.use_cache and self.cache:
            self._save_cache()

    # ── API 호출 ────────────────────────────────────────

    def _rotate_key(self):
        """다음 키로 라운드로빈 전환"""
        self._current_key_idx = (self._current_key_idx + 1) % len(self.api_keys)

    def _current_headers(self) -> dict:
        key = self.api_keys[self._current_key_idx]
        return {
            "X-Naver-Client-Id": key["client_id"],
            "X-Naver-Client-Secret": key["client_secret"],
            "Content-Type": "application/json",
        }

    def _request(self, body: dict, max_retries: int = 3) -> Optional[dict]:
        """단일 API 호출 (키 로테이션 + 재시도 포함)"""
        keys_exhausted = 0  # 연속 429로 소진된 키 수

        for attempt in range(max_retries):
            headers = self._current_headers()

            try:
                resp = requests.post(
                    NAVER_DATALAB_URL,
                    headers=headers,
                    json=body,
                    timeout=15,
                )

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
                    wait = 5
                    print(f"  [{resp.status_code}] 서버 에러 — {wait}초 대기 (attempt {attempt + 1})")
                    time.sleep(wait)
                    continue

                # 4xx (429 제외) — 재시도 불필요
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

    # ── 공개 메서드 ─────────────────────────────────────

    def search_trend(
        self,
        keyword_groups: list[dict],
        start_date: str,
        end_date: str,
        time_unit: str = "month",
        gender: str = "",
        ages: list[str] = None,
        device: str = "",
    ) -> Optional[dict]:
        """검색어 트렌드 조회 (최대 5그룹)

        Parameters
        ----------
        keyword_groups : list[dict]
            [{"groupName": "브랜드A", "keywords": ["키워드1", "키워드2"]}, ...]
        start_date : str  "2024-01-01"
        end_date : str    "2025-12-31"
        time_unit : str   "date" | "week" | "month"
        gender : str      "" | "m" | "f"
        ages : list[str]  ["3", "4"] 등 (네이버 연령코드)
        device : str      "" | "pc" | "mo"
        """
        if len(keyword_groups) > 5:
            raise ValueError("keyword_groups는 최대 5개까지만 가능합니다. search_trend_batch를 사용하세요.")

        body = {
            "startDate": start_date,
            "endDate": end_date,
            "timeUnit": time_unit,
            "keywordGroups": keyword_groups,
        }
        if gender:
            body["gender"] = gender
        if ages:
            body["ages"] = ages
        if device:
            body["device"] = device

        # 캐시 확인
        key = self._cache_key(body)
        if self.use_cache and key in self.cache:
            return self.cache[key]

        # API 호출
        time.sleep(self.request_delay)
        result = self._request(body)

        if result:
            if self.use_cache:
                self.cache[key] = result
                self._flush_if_needed()

        return result

    def search_trend_batch(
        self,
        all_keyword_groups: list[dict],
        start_date: str,
        end_date: str,
        time_unit: str = "month",
        gender: str = "",
        ages: list[str] = None,
        device: str = "",
    ) -> list[dict]:
        """5개씩 분할하여 배치 호출

        Returns
        -------
        list[dict] : 각 배치의 API 응답 리스트
        """
        results = []
        total = len(all_keyword_groups)
        batches = [all_keyword_groups[i : i + 5] for i in range(0, total, 5)]

        print(f"  배치 호출: {total}그룹 → {len(batches)}배치")

        for idx, batch in enumerate(batches, 1):
            resp = self.search_trend(
                keyword_groups=batch,
                start_date=start_date,
                end_date=end_date,
                time_unit=time_unit,
                gender=gender,
                ages=ages,
                device=device,
            )
            if resp:
                results.append(resp)
                print(f"    배치 {idx}/{len(batches)} 완료 ({len(batch)}그룹)")
            else:
                print(f"    배치 {idx}/{len(batches)} 실패")

        # 마지막에 남은 캐시 저장
        self.flush_cache()
        return results
