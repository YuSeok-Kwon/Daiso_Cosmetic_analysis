"""
성분 추출 후처리 모듈
- OCR 결과에서 성분 리스트 정제
- 정규표현식 기반 불필요한 문구 제거
- 화장품 성분 사전과 매칭
"""

import re
import logging
from typing import List, Tuple
from difflib import get_close_matches

logger = logging.getLogger(__name__)


# 제거할 키워드/문구 패턴
REMOVE_PATTERNS = [
    r'전성분[:：]\s*',
    r'성분[:：]\s*',
    r'INGREDIENTS[:：]?\s*',
    r'Ingredients[:：]?\s*',
    r'화장품법에\s*따라\s*기재·?표시하여야\s*하는\s*모든\s*성분',
    r'화장품법에\s*따라\s*기재하여야\s*하는',
    r'※\s*',
    r'주의사항.*',
    r'사용방법.*',
    r'제조국.*',
    r'원산지.*',
    r'유통기한.*',
    r'보관방법.*',
]

# 성분이 아닌 문구 (완전 제거)
NOISE_PHRASES = [
    '기재·표시하여야하는',
    '자외선차단',
    '미백기능성',
    '주름개선',
    '피부보호',
    '식품의약품안전처',
    '본품',
    '적당량',
    '바른다',
    '두드려',
    '흡수',
]

# 화장품 성분 사전 (확장 가능)
INGREDIENT_DICTIONARY = {
    # 기본 성분
    '정제수', '물', '글리세린', '부틸렌글라이콜', '프로판다이올',

    # 방부제
    '페녹시에탄올', '메틸파라벤', '에틸파라벤', '프로필파라벤', '부틸파라벤',
    '소듐벤조에이트', '포타슘소르베이트',

    # UV 필터
    '에틸헥실메톡시신나메이트', '티타늄디옥사이드', '징크옥사이드',
    '옥토크릴렌', '아보벤존', '호모살레이트',

    # 증점제
    '카보머', '잔탄검', '히알루론산나트륨', '카라기난',

    # 색소
    '산화철', '황산화철', '적산화철', '흑산화철',
    '운모티타늄', '마이카',

    # 향료
    '향료',

    # 실리콘
    '다이메치콘', '사이클로펜타실록산', '사이클로헥사실록산',

    # 오일
    '스쿠알란', '호호바오일', '시어버터', '미네랄오일',

    # 식물추출물
    '알로에베라잎추출물', '녹차추출물', '병풀추출물', '감초추출물',
    '카모마일추출물', '라벤더추출물', '센텔라아시아티카추출물',

    # 항산화제
    '토코페롤', '아스코르브산', '부틸하이드록시톨루엔',

    # 유화제
    '폴리소르베이트60', '폴리소르베이트80', '레시틴',
    '스테아르산', '세틸알코올', '스테아릴알코올',
}


def remove_noise_patterns(text: str) -> str:
    """
    OCR 결과에서 불필요한 패턴 제거

    Args:
        text: 원본 텍스트

    Returns:
        정제된 텍스트
    """
    cleaned = text

    # 패턴 기반 제거
    for pattern in REMOVE_PATTERNS:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    # 노이즈 문구 제거
    for phrase in NOISE_PHRASES:
        cleaned = cleaned.replace(phrase, '')

    return cleaned.strip()


def split_ingredients(text: str) -> List[str]:
    """
    성분 텍스트를 개별 성분으로 분리

    전략:
    1. 콤마(,), 세미콜론(;), 파이프(|)로 1차 분리
    2. 공백 1개 이상으로 추가 분리
    3. 빈 문자열 제거

    Args:
        text: 성분 텍스트

    Returns:
        성분 리스트
    """
    # 1. 콤마, 세미콜론, 파이프로 분리
    # OCR 오류로 콤마가 세미콜론으로 인식되기도 함
    parts = re.split(r'[,;|]', text)

    # 2. 공백으로 추가 분리 (1개 이상)
    ingredients = []
    for part in parts:
        # 공백 1개 이상을 구분자로 사용
        # 단, 너무 많이 분리되지 않도록 조심
        sub_parts = part.split()
        ingredients.extend(sub_parts)

    # 3. 앞뒤 공백 제거 및 빈 문자열 제외
    ingredients = [ing.strip() for ing in ingredients if ing.strip()]

    return ingredients


def normalize_ingredient(ingredient: str) -> str:
    """
    성분명 정규화

    Args:
        ingredient: 원본 성분명

    Returns:
        정규화된 성분명
    """
    # 1. 앞뒤 공백 제거
    normalized = ingredient.strip()

    # 2. OCR 오류 수정 (기존 OCR_CORRECTIONS 활용)
    from daiso_beauty_crawler import OCR_CORRECTIONS
    for wrong, correct in OCR_CORRECTIONS.items():
        normalized = normalized.replace(wrong, correct)

    # 3. 괄호 내용 제거 (예: "글리세린(보습제)" → "글리세린")
    normalized = re.sub(r'\([^)]*\)', '', normalized)
    normalized = re.sub(r'\[[^\]]*\]', '', normalized)

    # 4. 특수문자 제거 (±, *, ★ 등)
    normalized = re.sub(r'[±*★☆※]', '', normalized)

    # 5. 숫자 제거 (성분명에는 보통 숫자 없음, 단 CI77891 같은 색소 코드는 유지)
    if not re.match(r'^CI\d+', normalized):
        # CI 코드가 아니면 끝의 숫자 제거
        normalized = re.sub(r'\d+$', '', normalized)

    # 6. 앞뒤 특수문자 제거
    normalized = re.sub(r'^[^\w가-힣]+|[^\w가-힣]+$', '', normalized, flags=re.UNICODE)

    # 7. 내부 공백 제거 (성분명은 보통 붙여쓰기)
    normalized = normalized.replace(' ', '')

    return normalized


def find_similar_ingredient(ingredient: str,
                            dictionary: set = INGREDIENT_DICTIONARY,
                            cutoff: float = 0.8) -> Tuple[str, float]:
    """
    사전에서 유사한 성분명 찾기 (오타 교정)

    Args:
        ingredient: 검색할 성분명
        dictionary: 성분 사전
        cutoff: 유사도 임계값 (0.0 ~ 1.0)

    Returns:
        (교정된 성분명, 유사도) 또는 (원본, 0.0)
    """
    # 정확히 일치하면 그대로 반환
    if ingredient in dictionary:
        return ingredient, 1.0

    # 유사한 성분 찾기
    matches = get_close_matches(ingredient, dictionary, n=1, cutoff=cutoff)

    if matches:
        corrected = matches[0]
        # 유사도 계산 (간단한 방식)
        similarity = len(set(ingredient) & set(corrected)) / len(set(ingredient) | set(corrected))

        logger.debug(f"오타 교정: '{ingredient}' → '{corrected}' (유사도: {similarity:.2f})")
        return corrected, similarity
    else:
        return ingredient, 0.0


def postprocess_ingredients(ocr_text: str,
                            use_dictionary: bool = True,
                            min_length: int = 2,
                            similarity_threshold: float = 0.8) -> List[dict]:
    """
    OCR 결과를 성분 리스트로 후처리

    전체 파이프라인:
    1. 불필요한 패턴 제거
    2. 성분 분리 (콤마, 공백)
    3. 개별 성분 정규화
    4. 사전 매칭 및 오타 교정
    5. 유효성 검증

    Args:
        ocr_text: OCR로 추출한 원본 텍스트
        use_dictionary: 사전 기반 오타 교정 사용 여부
        min_length: 최소 성분명 길이
        similarity_threshold: 사전 매칭 유사도 임계값

    Returns:
        List of dict: [
            {
                'original': '원본 성분명',
                'normalized': '정규화된 성분명',
                'corrected': '교정된 성분명',
                'similarity': 유사도,
                'is_valid': 유효 여부
            },
            ...
        ]
    """
    results = []

    # 1. 불필요한 패턴 제거
    cleaned_text = remove_noise_patterns(ocr_text)
    logger.info(f"패턴 제거 후 텍스트 길이: {len(cleaned_text)} (원본: {len(ocr_text)})")

    # 2. 성분 분리
    ingredient_list = split_ingredients(cleaned_text)
    logger.info(f"분리된 성분 수: {len(ingredient_list)}")

    # 3. 개별 성분 처리
    for original in ingredient_list:
        # 3-1. 정규화
        normalized = normalize_ingredient(original)

        # 길이 체크
        if len(normalized) < min_length:
            logger.debug(f"너무 짧은 성분명 제외: '{normalized}'")
            continue

        # 3-2. 사전 매칭 (오타 교정)
        corrected = normalized
        similarity = 0.0

        if use_dictionary:
            corrected, similarity = find_similar_ingredient(
                normalized,
                INGREDIENT_DICTIONARY,
                cutoff=similarity_threshold
            )

        # 3-3. 유효성 검증
        from daiso_beauty_crawler import is_valid_ingredient

        is_valid, confidence, reason = is_valid_ingredient(corrected, INGREDIENT_DICTIONARY)

        # 결과 저장
        results.append({
            'original': original,
            'normalized': normalized,
            'corrected': corrected,
            'similarity': similarity,
            'confidence': confidence,
            'is_valid': is_valid,
            'reason': reason
        })

        if normalized != corrected:
            logger.info(f"오타 교정: '{normalized}' → '{corrected}'")

    # 유효한 성분만 필터링
    valid_ingredients = [r for r in results if r['is_valid']]
    logger.info(f"유효한 성분: {len(valid_ingredients)} / {len(results)}")

    return valid_ingredients


# 테스트 코드
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 테스트 OCR 텍스트
    test_text = """
    전성분: 정제수, 글리세 린, 부틸렌글라이콜, 메칠파라벤,
    알로에베라잎추출물, 티타늄디옥사이드, 향료

    ※ 주의사항: 피부에 이상이 있을 경우 사용을 중지하세요.
    """

    print("성분 후처리 테스트")
    print("=" * 60)
    print(f"\n원본 텍스트:\n{test_text}\n")

    # 후처리
    results = postprocess_ingredients(test_text, use_dictionary=True)

    print(f"\n추출된 성분 ({len(results)}개):")
    print("-" * 60)

    for idx, ing in enumerate(results, 1):
        print(f"{idx}. {ing['corrected']}")
        if ing['normalized'] != ing['corrected']:
            print(f"   (오타 교정: {ing['normalized']} → {ing['corrected']})")
        print(f"   신뢰도: {ing['confidence']:.2f} | 이유: {ing['reason']}")
