"""
크롤링 이력 관리 모듈

증분 크롤링을 위해 제품별 마지막 크롤링 날짜와 리뷰 날짜를 JSON으로 관리한다.

사용법:
    history = CrawlHistory("cache/crawl_history.json")
    history.is_new_product("1056665")       # True/False
    history.get_last_review_date("1056665") # "2026-01-28" 또는 None
    history.update_product("1056665", review_date="2026-02-19")
    history.save()
"""
import json
import os
import tempfile
from datetime import date

import pandas as pd


def _normalize_date(date_str: str) -> str:
    """날짜 포맷을 YYYY-MM-DD로 통일 (크롤러: YYYY.MM.DD, CSV: YYYY-MM-DD)"""
    if not date_str:
        return ""
    return date_str.replace(".", "-")


class CrawlHistory:
    """제품별 크롤링 이력을 JSON으로 관리"""

    def __init__(self, history_path: str):
        self.history_path = history_path
        self._data = {"last_updated": "", "products": {}}
        self._load()

    def _load(self):
        """JSON 파일에서 이력 로드 (파싱 실패 시 백업 후 빈 dict로 시작)"""
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, ValueError) as e:
                # 손상된 파일 백업 후 빈 상태로 시작
                backup_path = self.history_path + ".corrupted"
                os.replace(self.history_path, backup_path)
                print(f"[경고] 이력 파일 파싱 실패 → 백업: {backup_path} ({e})")

    def is_new_product(self, product_code: str) -> bool:
        """이력에 없는 신규 제품이면 True"""
        return str(product_code) not in self._data["products"]

    def get_last_review_date(self, product_code: str) -> str | None:
        """마지막 리뷰 날짜 반환 (없으면 None)"""
        info = self._data["products"].get(str(product_code))
        if info:
            return info.get("last_review_date")
        return None

    def update_product(self, product_code: str, review_date: str = None):
        """제품 이력 갱신"""
        code = str(product_code)
        today = date.today().isoformat()

        if code not in self._data["products"]:
            self._data["products"][code] = {}

        self._data["products"][code]["last_crawled"] = today

        if review_date:
            normalized = _normalize_date(review_date)
            existing = self._data["products"][code].get("last_review_date", "")
            # 더 최신 날짜로만 갱신
            if not existing or normalized > existing:
                self._data["products"][code]["last_review_date"] = normalized

    def save(self):
        """임시파일 → rename 패턴으로 안전 저장"""
        self._data["last_updated"] = date.today().isoformat()

        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)

        # 같은 디렉토리에 임시파일 생성 후 rename (원자적 교체)
        dir_name = os.path.dirname(self.history_path)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.history_path)
        except Exception:
            # 실패 시 임시파일 정리
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    @property
    def product_count(self) -> int:
        return len(self._data["products"])

    @classmethod
    def from_existing_csv(cls, history_path: str, reviews_csv_path: str,
                          products_csv_path: str = None) -> "CrawlHistory":
        """첫 실행 bootstrapping: 기존 CSV에서 이력 초기화

        1) products CSV → 모든 제품을 이력에 등록 (리뷰 없는 제품도 "기존"으로 인식)
        2) reviews CSV → 제품별 마지막 리뷰 날짜 설정 (증분 cutoff 기준)

        - history_path에 파일이 이미 있으면 그대로 로드
        - CSV가 없으면 빈 이력 반환
        """
        history = cls(history_path)

        # 이미 이력 파일이 있으면 그대로 사용
        if history.product_count > 0:
            return history

        # 1) products CSV에서 제품 코드 등록
        if products_csv_path and os.path.exists(products_csv_path):
            df_products = pd.read_csv(products_csv_path, usecols=["product_code"])
            for code in df_products["product_code"].unique():
                history.update_product(str(code))

        # 2) reviews CSV에서 제품별 마지막 리뷰 날짜 설정
        if reviews_csv_path and os.path.exists(reviews_csv_path):
            # write_date / review_date 둘 다 지원
            df = pd.read_csv(reviews_csv_path, usecols=lambda c: c in ("product_code", "write_date", "review_date"))
            date_col = "review_date" if "review_date" in df.columns else "write_date"
            df[date_col] = df[date_col].astype(str).apply(_normalize_date)

            grouped = df.groupby("product_code")[date_col].max()

            for product_code, max_date in grouped.items():
                history.update_product(str(product_code), review_date=max_date)

        if history.product_count > 0:
            history.save()
        return history
