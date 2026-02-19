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


def parse_target_flag(
    flag_name: Optional[str],
    ee_name: Optional[str] = None,
    spf: Optional[str] = None,
    pa: Optional[str] = None,
    effect_yn1: Optional[str] = None,
    effect_yn2: Optional[str] = None,
    effect_yn3: Optional[str] = None,
) -> dict:
    """효능 추출 — EE_NAME > EFFECT_YN > FLAG_NAME > SPF/PA 순서

    우선순위:
    1. EE_NAME 텍스트에서 키워드 추출 (미백, 주름개선, 자외선차단 등)
    2. EFFECT_YN1/2/3 (flag=2 고시원료 제품의 효능 플래그)
       YN1=Y → 미백, YN2=Y → 주름개선, YN3=Y → 자외선차단
    3. FLAG_NAME (3호=자외선차단만 신뢰)
    4. SPF 또는 PA 값이 있으면 자외선차단 추가
    """
    result = {col: False for _, (col, _) in EFFECT_FLAG_MAP.items()}

    # EE_NAME 키워드 매핑
    _EE_KEYWORDS = {
        "미백": "is_whitening",
        "주름개선": "is_anti_wrinkle",
        "자외선": "is_sunscreen",
        "자외선차단": "is_sunscreen",
        "모발의 색상": "is_hair_color",
        "체모를 제거": "is_hair_removal",
        "여드름": "is_acne",
        "아토피": "is_atopic",
        "튼살": "is_stretch_marks",
        "탈모": "is_hair_loss",
        "눈가": "is_eye_wrinkle",
    }

    if ee_name and isinstance(ee_name, str) and ee_name.strip():
        # EE_NAME에서 모든 효능 키워드 탐색
        for keyword, col in _EE_KEYWORDS.items():
            if keyword in ee_name:
                result[col] = True

    elif flag_name:
        # EE_NAME이 없으면 FLAG_NAME으로 보수적 판단
        # 3호(자외선차단)만 FLAG로 신뢰 — 1호/2호는 ee_name 없이 불확실
        if "3호" in flag_name:
            result["is_sunscreen"] = True

    # EFFECT_YN1/2/3: flag=2(고시원료) 전용 효능 플래그
    if effect_yn1 == "Y":
        result["is_whitening"] = True
    if effect_yn2 == "Y":
        result["is_anti_wrinkle"] = True
    if effect_yn3 == "Y":
        result["is_sunscreen"] = True

    # SPF 또는 PA 값이 존재하면 자외선차단 기능 추가
    _has_spf = spf and isinstance(spf, str) and spf.strip() and spf.strip() != "0"
    _has_pa = pa and isinstance(pa, str) and pa.strip() and pa.strip() != "0"
    if _has_spf or _has_pa:
        result["is_sunscreen"] = True

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

        params = {"item_name": product_name, "numOfRows": 50, "pageNo": 1}
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
        ee_name = item.get("EE_NAME", "")
        spf = item.get("SPF", "")
        pa = item.get("PA", "")
        effects = parse_target_flag(
            flag_name, ee_name, spf=spf, pa=pa,
            effect_yn1=item.get("EFFECT_YN1"),
            effect_yn2=item.get("EFFECT_YN2"),
            effect_yn3=item.get("EFFECT_YN3"),
        )

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


    # ── ENTP_NAME 기반 제조사 단위 매칭 ──

    # MFDS에서 영문 브랜드를 한글로 등록하는 패턴
    _BRAND_ALIAS = {
        "vt": "브이티",
        "ahc": "에이에이치씨",
        "cnp": "씨엔피",
        "bb": "비비",
        "cc": "씨씨",
        "egf": "이지에프",
    }

    @staticmethod
    def _normalize_name(name: str) -> str:
        """매칭용 이름 정규화 (공백·특수문자 제거, 영문→한글 변환)"""
        if not name:
            return ""
        name = re.sub(r"\[.*?\]", "", name)
        name = re.sub(r"\(.*?\)", "", name)
        name = re.sub(r"\d+\s*(ml|g|mg|L|oz)\b", "", name, flags=re.IGNORECASE)
        name = re.sub(r"\d+\s*[*xX×]\s*\d+\s*(개입|매)?", "", name)
        name = re.sub(r"[^\w가-힣a-zA-Z]", "", name)
        name = name.lower().strip()
        # 영문 브랜드 약어 → 한글 변환
        for eng, kor in FunctionalCosmeticsAPI._BRAND_ALIAS.items():
            if name.startswith(eng):
                name = kor + name[len(eng):]
                break
        return name

    def _search_by_keyword(self, keyword: str, max_total: int = 500) -> list:
        """키워드로 MFDS 품목 검색 (item_name 파라미터, 페이징)

        max_total 초과 시 빈 리스트 반환 (너무 일반적인 검색어)
        """
        if not keyword:
            return []
        cache_key = f"kw_{keyword}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # 1페이지만 먼저 조회하여 총 건수 확인
        params = {"item_name": keyword, "numOfRows": 100, "pageNo": 1}
        body = self._request(self.endpoint, params)
        if not body:
            self.cache[cache_key] = []
            return []

        total = body.get("totalCount", 0)
        if total > max_total:
            # 너무 많은 결과 → 키워드가 너무 일반적
            self.cache[cache_key] = []
            return []

        # 1페이지 결과 파싱
        raw = body.get("items", [])
        if isinstance(raw, dict):
            raw = raw.get("item", [])
        items = raw if isinstance(raw, list) else ([raw] if raw else [])

        # 2페이지 이상 있으면 나머지 페이지도 수집
        if total > 100:
            remaining_params = {"item_name": keyword}
            all_items = self.fetch_all_pages(self.endpoint, remaining_params, num_of_rows=100)
            items = all_items

        self.cache[cache_key] = items
        self._save_cache()
        return items

    # 너무 일반적이어서 단독 검색 키워드로 부적합한 화장품 용어
    _GENERIC_TERMS = frozenset({
        "크림", "세럼", "로션", "토너", "에센스", "앰플", "미스트", "팩",
        "마스크", "선크림", "파운데이션", "쿠션", "컨실러", "립밤",
        "클렌징", "스킨", "에멀전", "밤", "젤", "오일", "워터", "폼",
        "파우더", "프라이머", "스틱", "패드", "패치", "시트", "키트",
        "S50", "S100", "SPF", "UV", "IU", "EGF", "PDRN",
    })

    @staticmethod
    def _extract_search_keys(product_name: str, brand: str = "") -> list:
        """제품명에서 API 검색용 키워드 추출 (공백 제거, 짧은 핵심 단어)

        MFDS 제품명은 공백 없이 연결 (예: "브이티리들샷300").

        전략 (우선순위):
        1. "by" 뒤의 원 브랜드명 (예: 셀퓨전씨)
        2. 2단어 조합 (공백 제거, 예: 에이지리뉴크림)
        3. 핵심 고유명사 단어 (일반 화장품 용어 제외, 4글자+)
        4. 전체 제품명 (공백 제거)
        """
        if not product_name:
            return []

        keys = []

        # "by" 브랜드 추출
        by_match = re.search(r"(?:\(?\s*by\s+|바이\s+)([\w가-힣]+)", product_name, re.IGNORECASE)
        by_brand = by_match.group(1) if by_match else ""

        # 전처리
        name = re.sub(r"\[.*?\]", "", product_name).strip()
        name = re.sub(r"\(.*?\)", "", name)
        name = re.sub(r"_\S+$", "", name)
        name = re.sub(r"\d+\s*(ml|g|mg|L|oz|매|개|정|ea)\b.*$", "", name, flags=re.IGNORECASE)
        name = re.sub(r"\d+\s*[*xX×]\s*\d+\s*(개입|매)?", "", name)
        name = name.strip()

        # 브랜드 제거
        core = name
        if brand:
            core = re.sub(re.escape(brand), "", core, flags=re.IGNORECASE).strip()
            brand_nospace = brand.replace(" ", "")
            if brand_nospace != brand:
                core = re.sub(re.escape(brand_nospace), "", core, flags=re.IGNORECASE).strip()

        words = [w for w in core.split() if w]

        # 단어 정제
        clean_words = []
        for w in words:
            w_clean = re.sub(r"[^\w가-힣a-zA-Z]", "", w)
            if w_clean:
                clean_words.append(w_clean)

        # 1. "by" 원 브랜드 (가장 효과적)
        if by_brand and len(by_brand) >= 2:
            keys.append(by_brand)

        # 2. 2단어 조합 (공백 제거) — 가장 정확한 검색
        for i in range(len(clean_words) - 1):
            combo = clean_words[i] + clean_words[i + 1]
            if len(combo) >= 5:
                keys.append(combo)

        # 3. 핵심 고유명사 단어 (일반 화장품 용어 제외, 4글자+)
        for w in clean_words:
            if w not in FunctionalCosmeticsAPI._GENERIC_TERMS and len(w) >= 4:
                keys.append(w)

        # 4. 전체 브랜드+제품명 (공백 제거)
        full_nospace = re.sub(r"[^\w가-힣a-zA-Z]", "", name)
        if full_nospace and len(full_nospace) >= 5:
            keys.append(full_nospace)

        # 중복 제거 (순서 유지)
        seen = set()
        unique = []
        for k in keys:
            if k and k not in seen and len(k) >= 3:
                seen.add(k)
                unique.append(k)
        return unique

    def _build_result(self, product_code: str, search_key: str, item: dict) -> dict:
        """API 응답 1건을 결과 dict로 변환"""
        flag_name = item.get(
            "COSMETIC_TARGET_FLAG_NAME", item.get("COSMETIC_TARGET_FLAG", "")
        )
        ee_name = item.get("EE_NAME", "")
        spf = item.get("SPF", "")
        pa = item.get("PA", "")
        effects = parse_target_flag(
            flag_name, ee_name, spf=spf, pa=pa,
            effect_yn1=item.get("EFFECT_YN1"),
            effect_yn2=item.get("EFFECT_YN2"),
            effect_yn3=item.get("EFFECT_YN3"),
        )

        return {
            "product_code": product_code,
            "search_key": search_key,
            "mfds_matched": True,
            "mfds_total_count": 1,
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

    def _match_product_to_api_items(
        self, product_name: str, product_code: str, api_items: list
    ) -> dict:
        """제품 1건을 API 품목 리스트와 이름 유사도 매칭"""
        if not api_items:
            return {
                "product_code": product_code,
                "search_key": product_name,
                "mfds_matched": False,
            }

        norm_product = self._normalize_name(product_name)
        if not norm_product:
            return {"product_code": product_code, "search_key": "", "mfds_matched": False}

        # 1단계: 정규화 완전 일치
        for item in api_items:
            norm_api = self._normalize_name(item.get("ITEM_NAME", ""))
            if norm_product == norm_api:
                return self._build_result(product_code, product_name, item)

        # 2단계: 포함 관계 (짧은 쪽이 긴 쪽에 포함, 최소 3글자 이상)
        best_item = None
        best_len = 0
        for item in api_items:
            norm_api = self._normalize_name(item.get("ITEM_NAME", ""))
            if not norm_api:
                continue
            if norm_product in norm_api or norm_api in norm_product:
                match_len = min(len(norm_product), len(norm_api))
                if match_len >= 3 and match_len > best_len:
                    best_len = match_len
                    best_item = item

        if best_item:
            return self._build_result(product_code, product_name, best_item)

        # 3단계: 핵심 부분 매칭 (앞 5글자 동일하면 후보, 공통 부분문자열 비교)
        for item in api_items:
            norm_api = self._normalize_name(item.get("ITEM_NAME", ""))
            if not norm_api or len(norm_api) < 5 or len(norm_product) < 5:
                continue
            # 앞 7글자 이상 동일하면 매칭 후보
            prefix_len = 0
            for a, b in zip(norm_product, norm_api):
                if a == b:
                    prefix_len += 1
                else:
                    break
            if prefix_len >= 7:
                overlap = prefix_len / max(len(norm_product), len(norm_api))
                if overlap > best_len:
                    best_len = overlap
                    best_item = item

        if best_item and best_len >= 0.4:
            return self._build_result(product_code, product_name, best_item)

        # 매칭 실패
        return {
            "product_code": product_code,
            "search_key": product_name,
            "mfds_matched": False,
        }

    def match_all_by_entp(self, products_path: str = None) -> pd.DataFrame:
        """브랜드 + 제품명 다중 검색 기반 전체 제품 매칭

        Phase 1: 브랜드명으로 API 조회 → 이름 매칭
        Phase 2: 미매칭 기능성 제품 다중 키워드 검색 (핵심 키워드, by 브랜드 등)
        결과에 ENTP_NAME 포함
        """
        if products_path is None:
            products_path = DATA_DIR / "parquet" / "products.parquet"

        products = pd.read_parquet(products_path)
        print(f"전체 제품 수: {len(products)}건")

        # ── Phase 1: 브랜드 기반 매칭 ──
        brands = products["brand"].dropna().unique()
        print(f"\n[Phase 1] 브랜드 기반 검색 ({len(brands)}개)")

        results = {}  # product_code → result dict
        brand_api_cache = {}  # brand → api_items

        for idx, brand in enumerate(sorted(brands), 1):
            brand_products = products[products["brand"] == brand]
            api_items = self._search_by_keyword(brand)
            brand_api_cache[brand] = api_items
            time.sleep(self.request_delay)

            matched = 0
            for _, row in brand_products.iterrows():
                name = row.get("name", "")
                code = str(row.get("product_code", ""))
                entp = row.get("ENTP_NAME", "")

                result = self._match_product_to_api_items(name, code, api_items)
                result["ENTP_NAME"] = entp
                results[code] = result

                if result.get("mfds_matched"):
                    matched += 1

            if idx % 20 == 0 or idx == len(brands):
                self._save_cache()
            print(f"  [{idx}/{len(brands)}] {brand}: API {len(api_items)}건, 매칭 {matched}/{len(brand_products)}")

        phase1_matched = sum(1 for r in results.values() if r.get("mfds_matched"))
        print(f"\nPhase 1 결과: {phase1_matched}/{len(results)} 매칭")

        # ── Phase 2: 미매칭 기능성 제품 다중 키워드 검색 ──
        func_col = "functional" if "functional" in products.columns else None
        unmatched_func = []
        for code, r in results.items():
            if r.get("mfds_matched"):
                continue
            if func_col:
                row = products[products["product_code"].astype(str) == code]
                if len(row) > 0 and row.iloc[0].get(func_col) == 1:
                    unmatched_func.append(code)
            else:
                unmatched_func.append(code)

        print(f"\n[Phase 2] 미매칭 기능성 {len(unmatched_func)}건 다중 키워드 검색")

        phase2_matched = 0
        for i, code in enumerate(unmatched_func, 1):
            row = products[products["product_code"].astype(str) == code].iloc[0]
            name = row.get("name", "")
            brand = row.get("brand", "")
            entp = row.get("ENTP_NAME", "")

            search_keys = self._extract_search_keys(name, brand)
            found = False

            for key in search_keys:
                if len(key) < 2:
                    continue
                api_items = self._search_by_keyword(key)
                time.sleep(self.request_delay)

                if api_items:
                    result = self._match_product_to_api_items(name, code, api_items)
                    result["ENTP_NAME"] = entp
                    if result.get("mfds_matched"):
                        results[code] = result
                        phase2_matched += 1
                        found = True
                        print(f"  ✓ [{brand}] {name[:30]} ← 키워드 '{key}'")
                        break

            if not found and i % 20 == 0:
                print(f"  진행: {i}/{len(unmatched_func)} (Phase 2 매칭: {phase2_matched}건)")

            if i % 30 == 0:
                self._save_cache()

        self._save_cache()
        print(f"Phase 2 결과: {phase2_matched}건 추가 매칭")

        # ── 수정본 효능 보정 ──
        df = pd.DataFrame(list(results.values()))
        df = self._apply_manual_fix(df)

        # ── 통계 출력 ──
        matched_total = df["mfds_matched"].sum()

        print("\n" + "=" * 60)
        print("매칭 결과 요약")
        print("=" * 60)
        print(f"전체 제품: {len(df)}건")
        print(f"매칭 성공: {matched_total}건 ({matched_total/len(df)*100:.1f}%)")
        print(f"  Phase 1 (브랜드): {phase1_matched}건")
        print(f"  Phase 2 (키워드): {phase2_matched}건")
        print(f"매칭 실패: {len(df) - matched_total}건")

        # 기능성 제품 매칭률
        if func_col and func_col in products.columns:
            func_codes = set(products[products[func_col] == 1]["product_code"].astype(str))
            func_matched = sum(1 for c, r in results.items() if c in func_codes and r.get("mfds_matched"))
            print(f"\n기능성 제품: {func_matched}/{len(func_codes)} ({func_matched/len(func_codes)*100:.1f}%)")

        # ENTP_NAME별 통계
        print("\n제조사(ENTP_NAME)별 매칭:")
        if "ENTP_NAME" in df.columns:
            for entp in sorted(df["ENTP_NAME"].unique()):
                sub = df[df["ENTP_NAME"] == entp]
                m = sub["mfds_matched"].sum()
                print(f"  {entp}: {m}/{len(sub)}")

        return df

    @staticmethod
    def _apply_manual_fix(df: pd.DataFrame) -> pd.DataFrame:
        """수정본(기능성 화장품 분류_수정본.xlsx)의 효능 정보로 보정

        API의 EFFECT_YN/EE_NAME이 없는 flag=2 제품 등에 대해
        수동 검수 결과를 반영한다.
        """
        fix_path = DATA_DIR / "기능성 화장품 분류_수정본 18.07.42.xlsx"
        if not fix_path.exists():
            print(f"  수정본 파일 없음: {fix_path}")
            return df

        fix = pd.read_excel(fix_path)
        fix["product_code"] = fix["product_code"].astype(str)
        df["product_code"] = df["product_code"].astype(str)

        # 수정본 효능 합산 (1차 or 2차 중 하나라도 1이면 True)
        fix["_fix_whitening"] = (fix["whitening"] == 1) | (fix["whitening2"] == 1)
        fix["_fix_wrinkle"] = (fix["wrinkle_reduction"] == 1) | (fix["wrinkle_reduction2"] == 1)
        fix["_fix_sunscreen"] = (fix["sunscreen"] == 1) | (fix["sunscreen2"] == 1)
        fix["_fix_acne"] = fix["acne"] == 1

        fix = fix.drop_duplicates(subset="product_code", keep="first")
        fix_map = fix.set_index("product_code")[
            ["_fix_whitening", "_fix_wrinkle", "_fix_sunscreen", "_fix_acne"]
        ].to_dict("index")

        updated = 0
        for idx, row in df.iterrows():
            code = row["product_code"]
            if code not in fix_map:
                continue
            f = fix_map[code]
            changed = False
            if f["_fix_whitening"] and not row.get("is_whitening"):
                df.at[idx, "is_whitening"] = True
                changed = True
            if f["_fix_wrinkle"] and not row.get("is_anti_wrinkle"):
                df.at[idx, "is_anti_wrinkle"] = True
                changed = True
            if f["_fix_sunscreen"] and not row.get("is_sunscreen"):
                df.at[idx, "is_sunscreen"] = True
                changed = True
            if f["_fix_acne"] and not row.get("is_acne"):
                df.at[idx, "is_acne"] = True
                changed = True
            if changed:
                updated += 1

        if updated:
            print(f"\n[수정본 보정] {updated}건 효능 추가 반영")

        return df


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
