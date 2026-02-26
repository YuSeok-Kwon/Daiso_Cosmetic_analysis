"""연착륙 상품 데이터 → 네이버 검색 키워드 그룹 변환"""

import re

import pandas as pd

from .config import FINAL_DIR, DEFAULT_TOP_N, PROJECT_ROOT


# ── 데이터 로드 ─────────────────────────────────────────

def load_top_products(
    top_n: int = DEFAULT_TOP_N,
    sort_by: str = "engagement_score",
) -> pd.DataFrame:
    """products_stats + products_core + brands 조인 → sort_by 상위 N종 반환

    Parameters
    ----------
    top_n : int     상위 N개 제품
    sort_by : str   정렬 기준 컬럼 (engagement_score, review_density 등)
    """
    stats = pd.read_csv(FINAL_DIR / "products_stats.csv")
    core = pd.read_csv(FINAL_DIR / "products_core.csv")
    brands = pd.read_csv(FINAL_DIR / "brands.csv", encoding="cp949")

    merged = (
        stats.merge(core[["product_code", "brand_id", "name", "price"]], on="product_code")
        .merge(brands.rename(columns={"name": "brand_name"}), on="brand_id")
    )
   
    top = merged.sort_values(sort_by, ascending=False).head(top_n).copy()
    top["search_keyword"] = top.apply(
        lambda r: extract_search_keyword(r["name"], r["brand_name"]), axis=1
    )
    return top.reset_index(drop=True)


# ── 키워드 추출 ─────────────────────────────────────────

def extract_search_keyword(product_name: str, brand_name: str) -> str:
    """제품명에서 검색용 핵심 키워드를 추출

    - 용량/단위 제거: 30ml, 200g, 2ml*6개입 등
    - 괄호 내용 제거
    - 브랜드명 중복 제거
    """
    kw = product_name

    # 용량/단위 제거: "30 ml", "200g", "28 g", "2ml*6개입", "60매" 등
    kw = re.sub(r"\d+\s*ml\*?\d*개?입?", "", kw, flags=re.IGNORECASE)
    kw = re.sub(r"\d+\s*(ml|g|mg|l|매)\b", "", kw, flags=re.IGNORECASE)

    # 대괄호 내용 제거: "[리뉴얼]", "[단독]" 등
    kw = re.sub(r"\[[^\]]*\]", "", kw)

    # 소괄호 내용 제거
    kw = re.sub(r"\([^)]*\)", "", kw)

    # 브랜드명 제거 (제품명 앞부분에 브랜드가 포함된 경우)
    if brand_name and kw.strip().startswith(brand_name):
        kw = kw.strip()[len(brand_name):]

    # 연속 공백 정리
    kw = re.sub(r"\s+", " ", kw).strip()
    return kw


# ── 브랜드 키워드 그룹 ──────────────────────────────────

# 포괄적 검색 키워드 템플릿 (최대 20개)
_BRAND_KW_TEMPLATES = [
    "다이소 {brand}",
    "{brand} 다이소",
    "다이소 {brand} 추천",
    "다이소 {brand} 후기",
    "다이소 {brand} 리뷰",
    "{brand} 다이소 추천",
    "{brand} 다이소 후기",
    "{brand} 다이소 리뷰",
    "다이소 {brand} 가성비",
    "다이소 {brand} 화장품",
    "다이소 {brand} 가격",
    "다이소 {brand} 인기",
    "다이소 {brand} 성분",
    "다이소 {brand} 효과",
    "{brand} 다이소 가격",
    "{brand} 다이소 가성비",
    "다이소 {brand} 뷰티",
    "{brand} 다이소 성분",
    "{brand} 다이소 효과",
]


# 일반명사와 겹쳐서 검색 노이즈가 심한 브랜드 → 검색 분석에서 완전 제외
_EXCLUDE_BRANDS = {"포인트"}


def build_brand_keyword_groups(
    top_n: int = DEFAULT_TOP_N,
    sort_by: str = "engagement_score",
) -> list[dict]:
    """브랜드별 포괄적 키워드 그룹 생성 (그룹당 최대 20개)

    각 그룹:
      groupName: 브랜드명
      keywords: ["다이소 {브랜드} 추천", "다이소 {브랜드} 후기", ...]
    """
    df = load_top_products(top_n, sort_by=sort_by)
    groups = []

    for brand_name, _ in df.groupby("brand_name"):
        if brand_name in ("ALL", "다이소") or brand_name in _EXCLUDE_BRANDS:
            continue

        keywords = [t.format(brand=brand_name) for t in _BRAND_KW_TEMPLATES]

        groups.append({
            "groupName": brand_name,
            "keywords": keywords[:20],
        })

    return groups


def build_all_brand_keyword_groups() -> list[dict]:
    """brands.parquet 전체 브랜드로 키워드 그룹 생성"""
    brands_path = PROJECT_ROOT / "02_processed_data" / "parquet" / "final_parquet" / "brands.parquet"
    brands = pd.read_parquet(brands_path)
    groups = []

    for _, row in brands.iterrows():
        brand_name = row["name"]
        if brand_name in ("ALL", "다이소") or brand_name in _EXCLUDE_BRANDS:
            continue

        keywords = [t.format(brand=brand_name) for t in _BRAND_KW_TEMPLATES]
        groups.append({
            "groupName": brand_name,
            "keywords": keywords[:20],
        })

    return groups


# ── 제품 키워드 그룹 ────────────────────────────────────

def _build_product_keywords(brand: str, kw: str, short_kw: str) -> list[str]:
    """일반 브랜드 제품: 최대 20개 키워드 생성 (중복 자동 제거)

    A. 풀네임 조합 (5개)
    B. 축약 조합 (4개)
    C. 의도 키워드 (6개)
    D. 속성 키워드 (5개)
    """
    seen = set()
    keywords = []

    def _add(text: str):
        text = re.sub(r"\s+", " ", text).strip()
        if text and text not in seen:
            seen.add(text)
            keywords.append(text)

    # A. 풀네임 조합
    _add(f"{brand} {kw}")
    _add(f"다이소 {brand} {kw}")
    _add(f"다이소 {kw}")
    _add(kw)
    _add(f"{kw} {brand}")

    # B. 축약 조합
    _add(f"{brand} {short_kw}")
    _add(f"다이소 {brand} {short_kw}")
    _add(f"다이소 {short_kw}")
    _add(f"{short_kw} {brand}")

    # C. 의도 키워드
    for intent in ("추천", "후기", "리뷰"):
        _add(f"{brand} {short_kw} {intent}")
        _add(f"다이소 {brand} {short_kw} {intent}")

    # D. 속성 키워드
    for attr in ("가격", "가성비", "효과", "성분", "사용법"):
        _add(f"{brand} {short_kw} {attr}")

    return keywords[:20]


def _build_daiso_product_keywords(kw: str, short_kw: str) -> list[str]:
    """다이소 자체 브랜드 제품: '다이소 {kw}' 패턴 위주 (최대 20개)"""
    seen = set()
    keywords = []

    def _add(text: str):
        text = re.sub(r"\s+", " ", text).strip()
        if text and text not in seen:
            seen.add(text)
            keywords.append(text)

    # 풀네임
    _add(f"다이소 {kw}")
    _add(kw)

    # 축약
    _add(f"다이소 {short_kw}")
    _add(short_kw)

    # 의도
    for intent in ("추천", "후기", "리뷰"):
        _add(f"다이소 {short_kw} {intent}")
        _add(f"{short_kw} {intent}")

    # 속성
    for attr in ("가격", "가성비", "효과", "성분", "사용법"):
        _add(f"다이소 {short_kw} {attr}")

    return keywords[:20]


def build_product_keyword_groups(
    top_n: int = DEFAULT_TOP_N,
    sort_by: str = "engagement_score",
) -> list[dict]:
    """제품별 키워드 그룹 생성 (그룹당 최대 20개)

    - 다이소 자체 브랜드 포함 (별도 키워드 템플릿)
    - groupName 충돌 해결: 2단어 → 3단어 → product_code 접미
    """
    df = load_top_products(top_n, sort_by=sort_by)
    groups = []
    used_names: set[str] = set()

    for _, row in df.iterrows():
        brand = row["brand_name"]
        kw = row["search_keyword"]
        product_code = row["product_code"]

        if not kw or brand == "ALL" or brand in _EXCLUDE_BRANDS:
            continue

        words = kw.split()
        short_kw = " ".join(words[:2]) if len(words) >= 2 else kw

        # 키워드 생성: 다이소 vs 일반 브랜드
        is_daiso = brand == "다이소"
        if is_daiso:
            keywords = _build_daiso_product_keywords(kw, short_kw)
        else:
            keywords = _build_product_keywords(brand, kw, short_kw)

        if not keywords:
            continue

        # groupName 충돌 해결: 2단어 → 3단어 → product_code 접미
        prefix = "다이소" if is_daiso else brand
        candidate = f"{prefix} {' '.join(words[:2])}" if len(words) >= 2 else f"{prefix} {kw}"
        if candidate in used_names:
            candidate = f"{prefix} {' '.join(words[:3])}" if len(words) >= 3 else candidate
        if candidate in used_names:
            candidate = f"{candidate} {product_code}"

        used_names.add(candidate)

        groups.append({
            "groupName": candidate,
            "keywords": keywords,
        })

    return groups


# ── 키워드 매핑 정보 ────────────────────────────────────

def get_keyword_mapping(
    top_n: int = DEFAULT_TOP_N,
    sort_by: str = "engagement_score",
) -> pd.DataFrame:
    """키워드 그룹 ↔ 원본 제품/브랜드 매핑 테이블"""
    df = load_top_products(top_n, sort_by=sort_by)
    mapping = df[["product_code", "brand_name", "name", "price", "engagement_score", "review_density", "search_keyword"]].copy()
    mapping = mapping.rename(columns={"name": "product_name"})
    return mapping
