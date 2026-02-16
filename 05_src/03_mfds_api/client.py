"""MFDS API 베이스 클라이언트"""

import time
import json
import requests
from pathlib import Path
from typing import Optional

from .config import MFDS_API_KEY, CACHE_DIR


class MFDSBaseClient:
    """공공데이터포털 MFDS API 기본 클라이언트"""

    def __init__(self, api_key: str = None, cache_name: str = "default"):
        self.api_key = api_key or MFDS_API_KEY
        if not self.api_key:
            raise ValueError(
                "MFDS_API_KEY가 설정되지 않았습니다.\n"
                "config/.env 파일에 MFDS_API_KEY=your_key 를 추가하세요.\n"
                "발급: https://www.data.go.kr"
            )

        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / f"{cache_name}_cache.json"
        self.cache = self._load_cache()
        self.request_delay = 0.3  # API 호출 간격 (초)

    def _load_cache(self) -> dict:
        if self.cache_file.exists():
            with open(self.cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _request(self, url: str, params: dict, max_retries: int = 3) -> Optional[dict]:
        """API 호출 (재시도 포함)"""
        params["serviceKey"] = self.api_key
        params.setdefault("type", "json")

        for attempt in range(max_retries):
            try:
                resp = requests.get(url, params=params, timeout=15)
                resp.raise_for_status()

                data = resp.json()
                header = data.get("header", {})

                if header.get("resultCode") == "00":
                    return data.get("body", {})

                # 에러 코드 처리
                msg = header.get("resultMsg", "UNKNOWN")
                if header.get("resultCode") in ("01", "99"):
                    print(f"  API 에러: {msg} (attempt {attempt + 1})")
                    time.sleep(2)
                    continue

                return data.get("body", {})

            except requests.exceptions.Timeout:
                print(f"  타임아웃 (attempt {attempt + 1})")
                time.sleep(2)
            except requests.exceptions.RequestException as e:
                print(f"  요청 실패: {e}")
                time.sleep(1)
            except json.JSONDecodeError:
                print(f"  JSON 파싱 실패 (응답: {resp.text[:200]})")
                return None

        return None

    def fetch_all_pages(self, url: str, params: dict, num_of_rows: int = 100) -> list:
        """전체 페이지 순회하여 모든 데이터 수집"""
        params["numOfRows"] = num_of_rows
        params["pageNo"] = 1

        body = self._request(url, params)
        if not body:
            return []

        total = body.get("totalCount", 0)
        items = body.get("items", [])
        if isinstance(items, dict):
            items = items.get("item", [])
        if not isinstance(items, list):
            items = [items] if items else []

        all_items = list(items)
        total_pages = (total // num_of_rows) + (1 if total % num_of_rows else 0)

        print(f"  총 {total}건, {total_pages}페이지")

        for page in range(2, total_pages + 1):
            time.sleep(self.request_delay)
            params["pageNo"] = page

            body = self._request(url, params)
            if not body:
                continue

            items = body.get("items", [])
            if isinstance(items, dict):
                items = items.get("item", [])
            if not isinstance(items, list):
                items = [items] if items else []

            all_items.extend(items)

            if page % 10 == 0:
                print(f"  {page}/{total_pages} 페이지 완료 ({len(all_items)}건)")

        print(f"  수집 완료: {len(all_items)}건")
        return all_items
