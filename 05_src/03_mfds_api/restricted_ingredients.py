"""화장품 사용제한 원료정보 API

- API: https://apis.data.go.kr/1471000/CsmtcsUseRstrcInfoService/getCsmtcsUseRstrcInfoService
- 전체 데이터를 다운로드 후 로컬 매칭 (API 검색 기능이 비정상)
- ingredients_master.csv에 is_restricted, restriction_type 컬럼 추가
"""

import pandas as pd
from pathlib import Path

from .client import MFDSBaseClient
from .config import ENDPOINTS, DATA_DIR


class RestrictedIngredientsAPI(MFDSBaseClient):
    """화장품 사용제한 원료정보 API"""

    def __init__(self, api_key: str = None):
        super().__init__(api_key=api_key, cache_name="restricted_ingredients")
        self.endpoint = ENDPOINTS["restricted"]
        self._restricted_data = None

    def download_all(self) -> pd.DataFrame:
        """전체 사용제한 원료 데이터 다운로드"""
        cache_key = "all_restricted"
        if cache_key in self.cache and self.cache[cache_key]:
            print("캐시에서 사용제한 원료 로드")
            return pd.DataFrame(self.cache[cache_key])

        print("사용제한 원료 전체 다운로드 시작...")
        items = self.fetch_all_pages(self.endpoint, {}, num_of_rows=100)

        if items:
            self.cache[cache_key] = items
            self._save_cache()

        df = pd.DataFrame(items)
        print(f"다운로드 완료: {len(df)}건")

        if len(df) > 0 and "INGR_STD_NAME" in df.columns:
            unique_ingr = df["INGR_STD_NAME"].nunique()
            print(f"고유 원료: {unique_ingr}개")
            print(f"규제 타입: {df['REGULATE_TYPE'].value_counts().to_dict()}")

        return df

    def _build_restricted_lookup(self, restricted_df: pd.DataFrame) -> dict:
        """원료명 → 규제 타입 매핑 딕셔너리 생성"""
        lookup = {}
        for _, row in restricted_df.iterrows():
            name = str(row.get("INGR_STD_NAME", "")).strip()
            reg_type = str(row.get("REGULATE_TYPE", "")).strip()
            if name:
                # 더 엄격한 규제가 우선 (금지 > 한도/금지 > 한도)
                priority = {"금지": 3, "한도/금지": 2, "한도": 1}
                current = lookup.get(name, ("", 0))
                new_priority = priority.get(reg_type, 0)
                if new_priority > current[1]:
                    lookup[name] = (reg_type, new_priority)
        return {k: v[0] for k, v in lookup.items()}

    def match_ingredients(self, ingredients_path: str = None) -> pd.DataFrame:
        """ingredients_master.csv와 사용제한 원료 매칭"""
        if ingredients_path is None:
            ingredients_path = DATA_DIR / "csv" / "ingredients_master.csv"

        df = pd.read_csv(ingredients_path)
        print(f"성분 수: {len(df)}건")

        # 사용제한 데이터 다운로드
        restricted_df = self.download_all()
        if restricted_df.empty:
            print("사용제한 데이터가 없습니다.")
            return df

        lookup = self._build_restricted_lookup(restricted_df)
        print(f"사용제한 고유 원료 매핑: {len(lookup)}개")

        # 매칭
        df["is_restricted"] = False
        df["restriction_type"] = None

        matched = 0
        for idx, row in df.iterrows():
            name = str(row["name"]).strip()
            if name in lookup:
                df.at[idx, "is_restricted"] = True
                df.at[idx, "restriction_type"] = lookup[name]
                matched += 1

        print(f"\n매칭 결과: {matched}/{len(df)} ({matched/len(df)*100:.1f}%)")

        if matched > 0:
            types = df[df["is_restricted"]]["restriction_type"].value_counts()
            for t, c in types.items():
                print(f"  {t}: {c}건")

        # is_caution 업데이트 (is_restricted=True → is_caution=True)
        df.loc[df["is_restricted"] == True, "is_caution"] = True

        return df

    def save_results(self, df: pd.DataFrame, output_path: str = None):
        """결과 저장"""
        if output_path is None:
            output_path = DATA_DIR / "csv" / "ingredients_master.csv"

        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"저장: {output_path} ({len(df)}건)")


if __name__ == "__main__":
    api = RestrictedIngredientsAPI()

    # 전체 다운로드 테스트
    print("=== 사용제한 원료 다운로드 ===")
    restricted = api.download_all()
    print(restricted.head(3).to_string())

    # 매칭 (주석 해제하여 실행)
    # print("\n=== 성분 매칭 ===")
    # df = api.match_ingredients()
    # api.save_results(df)
