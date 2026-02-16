"""화장품 원료성분정보 API

- API: https://apis.data.go.kr/1471000/CsmtcsIngdCpntInfoService01/getCsmtcsIngdCpntInfoService01
- 성분 한글명으로 영문명, CAS 번호 등 조회
- ingredients_master.csv에 name_eng, cas_no 컬럼 추가
"""

import time
import pandas as pd
from pathlib import Path

from .client import MFDSBaseClient
from .config import ENDPOINTS, DATA_DIR


class IngredientInfoAPI(MFDSBaseClient):
    """화장품 원료성분정보 조회 API"""

    def __init__(self, api_key: str = None):
        super().__init__(api_key=api_key, cache_name="ingredient_info")
        self.endpoint = ENDPOINTS["ingredient"]

    def search_by_name(self, ingredient_name: str) -> dict:
        """성분명으로 영문명/CAS 번호 조회"""
        cache_key = f"ingr_{ingredient_name}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        params = {"INGR_KOR_NAME": ingredient_name, "numOfRows": 5, "pageNo": 1}
        body = self._request(self.endpoint, params)

        result = {"name_eng": None, "cas_no": None}

        if body:
            items = body.get("items", [])
            if isinstance(items, dict):
                items = items.get("item", [])
            if isinstance(items, list) and items:
                item = items[0]
                result["name_eng"] = item.get("INGR_ENG_NAME")
                result["cas_no"] = item.get("CAS_NO")

        self.cache[cache_key] = result
        return result

    def enrich_ingredients(self, ingredients_path: str = None) -> pd.DataFrame:
        """ingredients_master.csv에 영문명/CAS 번호 추가"""
        if ingredients_path is None:
            ingredients_path = DATA_DIR / "csv" / "ingredients_master.csv"

        df = pd.read_csv(ingredients_path)
        print(f"성분 수: {len(df)}건")

        # 이미 영문명이 있는 건 스킵
        if "name_eng" in df.columns:
            missing = df["name_eng"].isna()
            targets = df[missing]
            print(f"  영문명 미매칭: {len(targets)}건 (기매칭: {(~missing).sum()}건)")
        else:
            df["name_eng"] = None
            df["cas_no"] = None
            targets = df

        for idx in targets.index:
            name = df.at[idx, "name"]
            result = self.search_by_name(name)
            time.sleep(self.request_delay)

            if result["name_eng"]:
                df.at[idx, "name_eng"] = result["name_eng"]
            if result["cas_no"]:
                df.at[idx, "cas_no"] = result["cas_no"]

            done = idx + 1
            if done % 100 == 0:
                print(f"  진행: {done}/{len(targets)}")
                self._save_cache()

        self._save_cache()

        eng_count = df["name_eng"].notna().sum()
        cas_count = df["cas_no"].notna().sum()
        print(f"\n결과: name_eng {eng_count}/{len(df)} ({eng_count/len(df)*100:.1f}%)")
        print(f"      cas_no  {cas_count}/{len(df)} ({cas_count/len(df)*100:.1f}%)")

        return df

    def save_results(self, df: pd.DataFrame, output_path: str = None):
        """결과 저장"""
        if output_path is None:
            output_path = DATA_DIR / "csv" / "ingredients_master.csv"

        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"저장: {output_path} ({len(df)}건)")


if __name__ == "__main__":
    api = IngredientInfoAPI()

    # 단건 테스트
    print("=== 단건 테스트 ===")
    result = api.search_by_name("토코페릴아세테이트")
    print(f"  영문명: {result['name_eng']}")
    print(f"  CAS No: {result['cas_no']}")

    # 전체 보강 (주석 해제하여 실행)
    # print("\n=== 전체 보강 ===")
    # df = api.enrich_ingredients()
    # api.save_results(df)
