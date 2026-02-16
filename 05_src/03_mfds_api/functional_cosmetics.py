"""기능성화장품 보고품목정보 API

- API: http://apis.data.go.kr/1471000/FtnltCosmRptPrdlstInfoService/getRptPrdlstInq
- 제품명으로 기능성화장품 여부 및 효능(미백/주름개선/자외선차단) 조회
- products.parquet 기반으로 전체 제품 매칭
"""

import re
import time
import pandas as pd
from typing import Optional
from pathlib import Path

from .client import MFDSBaseClient
from .config import ENDPOINTS, EFFECT_FLAG_MAP, DATA_DIR


def clean_product_name(name: str) -> str:
    """제품명 정제 — API 검색을 위한 표준화"""
    if not name:
        return ""
    name = re.sub(r"\[.*?\]", "", name)
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"\s*by\s+\w+", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\d+\s*(ml|g|mg|L|oz|개|매|정)\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s*[xX]\s*", "", name)
    name = re.sub(r"[^\w\s가-힣a-zA-Z]", "", name)
    name = re.sub(r"\s+", "", name.strip())
    return name


def parse_target_flag(flag_name: Optional[str]) -> dict:
    """COSMETIC_TARGET_FLAG_NAME에서 효능 추출 (전체 11개 효능 지원)"""
    result = {col: False for _, (col, _) in EFFECT_FLAG_MAP.items()}
    if flag_name:
        for key, (col, _) in EFFECT_FLAG_MAP.items():
            if key in flag_name:
                result[col] = True
    return result


class FunctionalCosmeticsAPI(MFDSBaseClient):
    """기능성화장품 보고품목정보 조회 API"""

    def __init__(self, api_key: str = None):
        super().__init__(api_key=api_key, cache_name="functional_cosmetics")
        self.endpoint = ENDPOINTS["functional"]

    def search_by_name(self, product_name: str) -> list:
        """제품명으로 검색"""
        cache_key = f"search_{product_name}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        params = {"ITEM_NAME": product_name, "numOfRows": 50, "pageNo": 1}
        body = self._request(self.endpoint, params)

        items = []
        if body:
            total = body.get("totalCount", 0)
            raw = body.get("items", [])
            if isinstance(raw, dict):
                raw = raw.get("item", [])
            items = raw if isinstance(raw, list) else ([raw] if raw else [])

        self.cache[cache_key] = items
        return items

    def match_product(self, product_name: str, product_code: str = "") -> dict:
        """제품 1건에 대해 MFDS 매칭 수행"""
        search_key = clean_product_name(product_name)
        if not search_key:
            return {"product_code": product_code, "search_key": "", "mfds_matched": False}

        items = self.search_by_name(search_key)
        time.sleep(self.request_delay)

        if not items:
            return {
                "product_code": product_code,
                "search_key": search_key,
                "mfds_matched": False,
                "mfds_total_count": 0,
            }

        item = items[0]
        flag_name = item.get("COSMETIC_TARGET_FLAG_NAME", item.get("COSMETIC_TARGET_FLAG", ""))
        effects = parse_target_flag(flag_name)

        return {
            "product_code": product_code,
            "search_key": search_key,
            "mfds_matched": True,
            "mfds_total_count": len(items),
            "mfds_item_name": item.get("ITEM_NAME"),
            "mfds_entp_name": item.get("ENTP_NAME"),
            "mfds_report_seq": item.get("REPORT_SEQ"),
            "mfds_report_date": item.get("REPORT_DE"),
            "mfds_item_ph": item.get("ITEM_PH"),
            "mfds_target_flag": item.get("COSMETIC_TARGET_FLAG"),
            "mfds_target_flag_name": item.get("COSMETIC_TARGET_FLAG_NAME"),
            "mfds_std_code": item.get("STD_CODE"),
            "mfds_std_name": item.get("STD_NAME"),
            "mfds_ee_code": item.get("EE_CODE"),
            "mfds_ee_name": item.get("EE_NAME"),
            "mfds_spf": item.get("SPF"),
            "mfds_pa": item.get("PA"),
            "mfds_water_proofing_flag": item.get("WATER_PROOFING_FLAG"),
            "mfds_water_proofing_name": item.get("WATER_PROOFING_NAME"),
            "mfds_report_flag_code": item.get("REPORT_FLAG_CODE"),
            "mfds_report_flag_name": item.get("REPORT_FLAG_NAME"),
            "mfds_ethanol_over_yn": item.get("ETHANOL_OVER_YN"),
            **effects,
        }

    def match_all_products(self, products_path: str = None) -> pd.DataFrame:
        """products.parquet의 전체 제품을 MFDS API와 매칭"""
        if products_path is None:
            products_path = DATA_DIR / "parquet" / "products.parquet"

        products = pd.read_parquet(products_path)
        print(f"제품 수: {len(products)}건")

        # functional=1인 제품 우선 처리
        if "functional" in products.columns:
            func_products = products[products["functional"] == 1]
            other_products = products[products["functional"] != 1]
            print(f"  functional=1: {len(func_products)}건, functional=0: {len(other_products)}건")
        else:
            func_products = products
            other_products = pd.DataFrame()

        results = []

        # functional=1 제품 매칭
        for i, row in func_products.iterrows():
            name = row.get("name", "")
            code = row.get("product_code", "")
            result = self.match_product(name, str(code))
            results.append(result)

            if (len(results)) % 50 == 0:
                print(f"  진행: {len(results)}/{len(func_products)} ({len(results)/len(func_products)*100:.1f}%)")
                self._save_cache()

        self._save_cache()

        df = pd.DataFrame(results)
        matched = df["mfds_matched"].sum()
        total = len(df)
        print(f"\n매칭 결과: {matched}/{total} ({matched/total*100:.1f}%)")

        return df

    def save_results(self, df: pd.DataFrame, output_dir: str = None):
        """결과를 parquet + xlsx로 저장"""
        if output_dir is None:
            output_dir = DATA_DIR

        matched = df[df["mfds_matched"] == True]

        parquet_path = Path(output_dir) / "parquet" / "products_mfds.parquet"
        xlsx_path = Path(output_dir) / "csv" / "products_mfds.xlsx"

        matched.to_parquet(parquet_path, index=False)
        matched.to_excel(xlsx_path, index=False)

        print(f"저장: {parquet_path} ({len(matched)}건)")
        print(f"저장: {xlsx_path} ({len(matched)}건)")

        # 효능 통계
        for key, (col, label) in EFFECT_FLAG_MAP.items():
            if col in matched.columns:
                count = matched[col].sum()
                if count > 0:
                    print(f"  {label}: {count}건")


if __name__ == "__main__":
    api = FunctionalCosmeticsAPI()

    # 단건 테스트
    print("=== 단건 테스트 ===")
    result = api.match_product("해서린 스팟 케어 클리어 젤 10ml", "TEST001")
    for k, v in result.items():
        if v:
            print(f"  {k}: {v}")

    # 전체 매칭 (주석 해제하여 실행)
    # print("\n=== 전체 매칭 ===")
    # df = api.match_all_products()
    # api.save_results(df)
