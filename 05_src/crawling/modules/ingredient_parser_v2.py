"""
성분 파싱 고도화 모듈 v2
- 정규식 및 구분자 고도화
- 텍스트 정규화 고도화
- 언어 감지 후 한글만 수집
- 성분 영역만 선택적 추출
- 패턴 필터링 (기능성 문구 등)
"""

import re
import unicodedata
from typing import List, Tuple, Optional, Set
import logging

logger = logging.getLogger(__name__)

# OCR 오인식 교정 딕셔너리 (ingredient_parser.py에서 정의)
# 순환 import 방지를 위해 여기서 정의
OCR_CORRECTIONS_V2 = {
    # === 숫자 0 오인식 (아이 → 0) ===
    '폴리아0소부텐': '폴리아이소부텐',
    '하이드로제네이티드폴리아0소부텐': '하이드로제네이티드폴리아이소부텐',
    '아0소스테아릭': '아이소스테아릭',
    '아0소프로필': '아이소프로필',
    '아0소도데케인': '아이소도데케인',
    '트라0글리세라이드': '트라이글리세라이드',
    '다0메티콘': '다이메티콘',
    '0소스테아릴': '아이소스테아릴',
    '0소파라핀': '아이소파라핀',
    '0소노닐': '아이소노닐',
    '0소헥사데케인': '아이소헥사데케인',

    # === 숫자 1 오인식 (레이/리/이 → 1) ===
    '스테아러1이트': '스테아레이트',
    '솔비탄아이소스테아러1이트': '솔비탄아이소스테아레이트',
    '라우러1이트': '라우레이트',
    '팔미터1이트': '팔미테이트',
    '올러1이트': '올레이트',
    '미리스터1이트': '미리스테이트',
    '폴1글리세릴': '폴리글리세릴',
    '폴1아크릴': '폴리아크릴',
    '폴1소르베이트': '폴리소르베이트',
    '하이알루로네1트': '하이알루로네이트',
    '글리세릴트라1스테아레이트': '글리세릴트라이스테아레이트',
    '트라1에톡시실릴': '트라이에톡시실릴',

    # === 자음 혼동 (알 → 일) ===
    '하이일루로': '하이알루로',
    '하이일루론': '하이알루론',
    '하이일루로네이트': '하이알루로네이트',
    '하이일루로닉': '하이알루로닉',
    '소듐하이일루로네이트': '소듐하이알루로네이트',
    '하이드롤라이즈드소듐하이일루로네이트': '하이드롤라이즈드소듐하이알루로네이트',

    # === 기타 OCR 오류 (글리서 → 글리세) ===
    '글리서라이드': '글리세라이드',
    '트라이글리서라이드': '트라이글리세라이드',
    '스테아릭트라이글리서라이드': '스테아릭트라이글리세라이드',
    '카프릴릭/카프릭트라이글리서라이드': '카프릴릭/카프릭트라이글리세라이드',
    '카프릴릭/카프릭/미리스틱/스테아릭트라이글리서라이드': '카프릴릭/카프릭/미리스틱/스테아릭트라이글리세라이드',

    # === 유사 한글 오인식 ===
    '애씨드': '애시드', '에씨드': '애시드', '아씨드': '애시드',
    '에탄을': '에탄올', '메탄을': '메탄올', '프로판을': '프로판올',
    '글라이을': '글라이올', '다이을': '다이올',
}


# =============================================================================
# 1. 텍스트 정규화 고도화
# =============================================================================

class TextNormalizer:
    """텍스트 정규화 클래스"""

    # OCR 오인식 문자 매핑 (확장)
    OCR_CHAR_MAP = {
        # 숫자-문자 혼동
        '0': 'O', 'O': 'O',  # 영문 O와 숫자 0
        '1': 'I', 'l': 'I', 'I': 'I',  # 영문 I, l, 숫자 1
        # 유사 한글
        '애씨드': '애시드', '에씨드': '애시드', '아씨드': '애시드',
        '에탄을': '에탄올', '메탄을': '메탄올', '프로판을': '프로판올',
        '글라이을': '글라이올', '다이을': '다이올',
        # 특수문자 정규화
        '：': ':', '；': ';', '，': ',', '．': '.',
        '（': '(', '）': ')', '［': '[', '］': ']',
        '｛': '{', '｝': '}', '／': '/', '＼': '\\',

        # === 숫자 0 오인식 (아이 → 0) - 한글 문맥 ===
        '아0소': '아이소',
        '0소부텐': '아이소부텐',
        '폴리아0소부텐': '폴리아이소부텐',
        '다0메틸': '다이메틸',
        '다0메티콘': '다이메티콘',
        '트라0글리세라이드': '트라이글리세라이드',

        # === 숫자 1 오인식 (레이/리/이 → 1) ===
        '러1이트': '레이트',
        '터1이트': '테이트',
        '폴1글리세릴': '폴리글리세릴',
        '폴1아크릴': '폴리아크릴',

        # === 자음 혼동 (알 → 일) ===
        '하이일루로': '하이알루로',
        '하이일루론': '하이알루론',
        '하이일루로네이트': '하이알루로네이트',

        # === 글리서 → 글리세 ===
        '글리서라이드': '글리세라이드',
        '트라이글리서라이드': '트라이글리세라이드',
    }

    # 유니코드 정규화 대상 문자
    UNICODE_NORMALIZE_MAP = {
        '\u00a0': ' ',  # Non-breaking space
        '\u2002': ' ',  # En space
        '\u2003': ' ',  # Em space
        '\u2009': ' ',  # Thin space
        '\u200a': ' ',  # Hair space
        '\u200b': '',   # Zero-width space
        '\u200c': '',   # Zero-width non-joiner
        '\u200d': '',   # Zero-width joiner
        '\ufeff': '',   # BOM
        '\u3000': ' ',  # 전각 공백
    }

    @classmethod
    def normalize(cls, text: str) -> str:
        """텍스트 정규화"""
        if not text:
            return ""

        # 1. 유니코드 정규화 (NFC)
        text = unicodedata.normalize('NFC', text)

        # 2. 특수 유니코드 문자 정규화
        for old, new in cls.UNICODE_NORMALIZE_MAP.items():
            text = text.replace(old, new)

        # 3. OCR 오인식 문자 수정
        for old, new in cls.OCR_CHAR_MAP.items():
            if len(old) > 1:  # 문자열 치환
                text = text.replace(old, new)

        # 4. 연속 공백 제거
        text = re.sub(r'\s+', ' ', text)

        # 5. 양끝 공백 제거
        text = text.strip()

        return text

    @classmethod
    def normalize_ingredient_name(cls, name: str) -> str:
        """성분명 정규화"""
        if not name:
            return ""

        name = cls.normalize(name)

        # OCR 오인식 교정 적용
        for wrong, correct in OCR_CORRECTIONS_V2.items():
            name = name.replace(wrong, correct)

        # 괄호 안 함량 정보 보존하면서 정규화
        # 예: "병풀잎수(95%)" -> "병풀잎수(95%)"
        # 예: "소듐하이알루로네이트(50.1002ppm)" -> "소듐하이알루로네이트(50.1002ppm)"

        # 불필요한 공백 제거 (성분명 내부)
        name = re.sub(r'\s+', '', name)

        return name


# =============================================================================
# 2. 언어 감지 및 필터링
# =============================================================================

class LanguageDetector:
    """언어 감지 클래스"""

    # 한글 범위
    HANGUL_PATTERN = re.compile(r'[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]')

    # 영문 범위
    ENGLISH_PATTERN = re.compile(r'[a-zA-Z]')

    # 숫자 및 특수문자
    NUMBER_PATTERN = re.compile(r'[\d\.\-\+\,\%]')

    # 화학식 패턴 (C12-15, PEG-100 등)
    CHEMICAL_PATTERN = re.compile(r'^[A-Z]+\d+|^C\d+-\d+|^PEG-\d+|^PPG-\d+')

    @classmethod
    def get_language_ratio(cls, text: str) -> Tuple[float, float]:
        """한글/영문 비율 반환 (korean_ratio, english_ratio)"""
        if not text:
            return 0.0, 0.0

        hangul_count = len(cls.HANGUL_PATTERN.findall(text))
        english_count = len(cls.ENGLISH_PATTERN.findall(text))
        total = len(text.replace(' ', ''))

        if total == 0:
            return 0.0, 0.0

        return hangul_count / total, english_count / total

    @classmethod
    def is_korean(cls, text: str, threshold: float = 0.5) -> bool:
        """한글 텍스트인지 판단"""
        korean_ratio, english_ratio = cls.get_language_ratio(text)
        return korean_ratio >= threshold

    @classmethod
    def is_english(cls, text: str, threshold: float = 0.5) -> bool:
        """영문 텍스트인지 판단"""
        korean_ratio, english_ratio = cls.get_language_ratio(text)
        return english_ratio >= threshold

    @classmethod
    def is_chemical_formula(cls, text: str) -> bool:
        """화학식인지 판단 (C12-15알킬벤조에이트 등)"""
        return bool(cls.CHEMICAL_PATTERN.match(text))

    @classmethod
    def filter_korean_only(cls, ingredients: List[str]) -> List[str]:
        """한글 성분만 필터링 (화학식은 허용)"""
        result = []
        for ing in ingredients:
            # 화학식은 허용
            if cls.is_chemical_formula(ing):
                result.append(ing)
            # 한글이 포함된 경우만
            elif cls.HANGUL_PATTERN.search(ing):
                result.append(ing)
        return result


# =============================================================================
# 3. 구분자 및 성분 분리 고도화
# =============================================================================

class IngredientSplitter:
    """성분 분리 클래스"""

    # 성분 구분자 패턴 (우선순위 순)
    SEPARATORS = [
        r'\n+',           # 줄바꿈
        r'(?<=[가-힣])\s*,\s*(?=[가-힣])',  # 한글 사이 콤마
        r'(?<=[가-힣])\s+(?=[가-힣])',      # 한글 사이 공백 (2글자 이상)
        r'[,，]',         # 콤마 (전각 포함)
        r'[;；]',         # 세미콜론
        r'\|',            # 파이프
        r'·',             # 중점
    ]

    # 성분 내 허용되는 구분자 (분리하면 안됨)
    ALLOWED_INTERNAL_PATTERNS = [
        r'/[가-힣]+/',    # 슬래시로 구분된 한글 (약모밀꽃/잎/줄기수)
        r'\([^)]+\)',     # 괄호 내용 (95%, 50ppm 등)
        r'\[[^\]]+\]',    # 대괄호 내용
    ]

    # 함량 표기 패턴
    CONCENTRATION_PATTERN = re.compile(
        r'\(?\s*'
        r'(\d+\.?\d*)\s*'
        r'(ppm|ppb|%|mg|g|ml|L)?\s*'
        r'\)?'
    )

    @classmethod
    def split_ingredients(cls, text: str) -> List[str]:
        """성분 텍스트를 개별 성분으로 분리"""
        if not text:
            return []

        # 0. 슬래시(/)로 끝나는 줄바꿈 병합 (줄바꿈으로 분리된 성분명 병합)
        # 예: "카프릴릭/카프릭/미리스틱/\n스테아릭" → "카프릴릭/카프릭/미리스틱/스테아릭"
        text = re.sub(r'/\s*[\r\n]+\s*', '/', text)

        # 0-1. 하이픈(-)으로 끝나는 줄바꿈 + 숫자 병합
        # 예: "폴리글리세릴-\n10" → "폴리글리세릴-10"
        text = re.sub(r'-\s*[\r\n]+\s*(\d)', r'-\1', text)

        # 0-2. 한글 + 줄바꿈 + 한글 병합 (성분명이 줄바꿈으로 분리된 경우)
        # 예: "트라이글리서라\n이드" → "트라이글리서라이드"
        text = re.sub(r'([가-힣])\s*[\r\n]+\s*([가-힣])', r'\1\2', text)

        # 0-3. 나머지 줄바꿈을 먼저 콤마로 변환 (정규화 전)
        text = re.sub(r'[\r\n]+', ', ', text)

        # 1. 텍스트 정규화
        text = TextNormalizer.normalize(text)

        # 2. 키트 제품 분리 (여러 제품 성분표)
        kit_products = cls._split_kit_products(text)

        all_ingredients = []
        for product_text in kit_products:
            # 3. 구분자로 분리
            ingredients = cls._split_by_separators(product_text)

            # 4. 각 성분 정리
            for ing in ingredients:
                cleaned = cls._clean_ingredient(ing)
                if cleaned:
                    all_ingredients.append(cleaned)

        # 5. 중복 제거 (순서 유지)
        seen = set()
        unique = []
        for ing in all_ingredients:
            normalized = ing.lower().replace(' ', '')
            if normalized not in seen:
                seen.add(normalized)
                unique.append(ing)

        return unique

    @classmethod
    def _split_kit_products(cls, text: str) -> List[str]:
        """키트 제품의 개별 제품별 성분 분리"""
        # 제품 구분 패턴
        product_patterns = [
            r'([가-힣]+\s*:)',       # "클렌저:", "토너:" 등
            r'(\[[가-힣]+\])',       # "[클렌저]" 등
            r'(【[가-힣]+】)',       # "【클렌저】" 등
        ]

        # 패턴이 발견되면 분리
        for pattern in product_patterns:
            if re.search(pattern, text):
                parts = re.split(pattern, text)
                # 빈 문자열 제거
                parts = [p.strip() for p in parts if p.strip()]
                if len(parts) > 1:
                    return parts

        return [text]

    @classmethod
    def _split_by_separators(cls, text: str) -> List[str]:
        """구분자로 성분 분리"""
        # 1. 화학식의 콤마 보호 (1,2-헥산다이올 → 1@2-헥산다이올)
        # 숫자,숫자 패턴을 임시로 치환
        text = re.sub(r'(\d),(\d)', r'\1@COMMA@\2', text)

        # 2. 줄바꿈을 콤마로 변환
        text = re.sub(r'\n+', ',', text)

        # 3. 콤마로 분리
        parts = re.split(r'[,，;；]', text)

        result = []
        for part in parts:
            part = part.strip()
            if part:
                # 화학식 콤마 복원
                part = part.replace('@COMMA@', ',')
                # 공백으로 추가 분리가 필요한지 확인
                sub_parts = cls._split_by_space_if_needed(part)
                result.extend(sub_parts)

        return result

    @classmethod
    def _split_by_space_if_needed(cls, text: str) -> List[str]:
        """필요시 공백으로 추가 분리"""
        # 너무 긴 텍스트는 분리 시도
        if len(text) > 50:
            # 한글 사이의 큰 공백으로 분리
            parts = re.split(r'(?<=[가-힣])\s{2,}(?=[가-힣])', text)
            if len(parts) > 1:
                return [p.strip() for p in parts if p.strip()]

        return [text]

    @classmethod
    def _clean_ingredient(cls, text: str) -> str:
        """개별 성분 정리"""
        if not text:
            return ""

        # 양끝 공백/특수문자 제거
        text = text.strip(' \t\n\r·•-')

        # 빈 괄호 제거
        text = re.sub(r'\(\s*\)', '', text)
        text = re.sub(r'\[\s*\]', '', text)

        # 앞뒤 따옴표 제거
        text = text.strip('"\'""''')

        return text.strip()


# =============================================================================
# 4. 성분 영역 추출
# =============================================================================

class IngredientSectionExtractor:
    """성분 영역 추출 클래스"""

    # 성분 섹션 시작 키워드
    START_KEYWORDS = [
        '전성분', '성분:', '성분 :', '[전성분]', '(전성분)',
        '모든성분', '모든 성분', '화장품법에 따라',
        '성분명', '주성분', '성분은',
    ]

    # 성분 섹션 종료 키워드
    # 주의: 기능성 관련 단어(미백, 주름개선 등)는 포함하지 않음
    END_KEYWORDS = [
        '※', '주의사항', '주의 사항', '경고', '사용방법', '사용 방법',
        '보관방법', '보관 방법', '제조국', '용법', '용량',
        '본품', '적당량', '품질보증',
        '(Ingredient)', '[Ingredient]', 'Ingredients:',
        '(INCI)', 'INCI:', 'INCI Name',
        '가려움', '부어오름', '상처', '직사광선', '어린이',
        '피부에 이상', '사용을 중지',
    ]

    # 제외할 섹션 키워드
    EXCLUDE_SECTIONS = [
        '상품 특징', '상품특징', '제품 설명', '제품설명',
        '사용 방법', '사용방법', 'HOWTOUSE', 'HOW TO USE',
    ]

    @classmethod
    def extract_ingredient_section(cls, text: str) -> str:
        """전체 텍스트에서 성분 섹션만 추출"""
        if not text:
            return ""

        # 1. 텍스트 정규화
        text = TextNormalizer.normalize(text)

        # 2. 시작 위치 찾기
        start_pos = cls._find_start_position(text)
        if start_pos == -1:
            # 시작 키워드가 없으면 전체 텍스트 사용
            start_pos = 0

        # 3. 종료 위치 찾기
        end_pos = cls._find_end_position(text, start_pos)

        # 4. 섹션 추출
        section = text[start_pos:end_pos].strip()

        # 5. 시작 키워드 제거
        section = cls._remove_start_keywords(section)

        return section

    @classmethod
    def _find_start_position(cls, text: str) -> int:
        """성분 섹션 시작 위치 찾기"""
        min_pos = -1

        for keyword in cls.START_KEYWORDS:
            pos = text.find(keyword)
            if pos != -1:
                # 키워드 다음 위치
                keyword_end = pos + len(keyword)
                if min_pos == -1 or pos < min_pos:
                    min_pos = keyword_end

        return min_pos

    @classmethod
    def _find_end_position(cls, text: str, start_pos: int) -> int:
        """성분 섹션 종료 위치 찾기"""
        min_end = len(text)

        for keyword in cls.END_KEYWORDS:
            pos = text.find(keyword, start_pos)
            if pos != -1 and pos < min_end:
                # 괄호 안에 있는 키워드는 무시
                if not cls._is_inside_parentheses(text, pos):
                    min_end = pos

        return min_end

    @classmethod
    def _is_inside_parentheses(cls, text: str, pos: int) -> bool:
        """해당 위치가 괄호 안인지 확인"""
        # pos 이전의 여는 괄호와 닫는 괄호 수 비교
        before = text[:pos]
        open_count = before.count('(') + before.count('（')
        close_count = before.count(')') + before.count('）')
        return open_count > close_count

    @classmethod
    def _remove_start_keywords(cls, text: str) -> str:
        """시작 키워드 제거"""
        for keyword in cls.START_KEYWORDS:
            if text.startswith(keyword):
                text = text[len(keyword):].strip()
                # 콜론, 하이픈 등 제거
                text = text.lstrip(':：-–—').strip()

        return text


# =============================================================================
# 5. 패턴 필터링 (기능성 문구, 노이즈 등)
# =============================================================================

class IngredientFilter:
    """성분 필터링 클래스"""

    # 기능성 문구 패턴 (성분에서 분리해야 함)
    FUNCTIONAL_PATTERNS = [
        r'기능성\s*화장품',
        r'화장품법에?\s*따른?',
        r'심사를?\s*필함?',
        r'효능\s*성분',
        r'주름\s*개선',
        r'미백',
        r'자외선\s*차단',
        r'식품의약품\s*안전처?',
        r'식품의약품',
        r'고시\s*품목',
        r'기능성',
        r'주름개선',
        r'화장품',
    ]
    FUNCTIONAL_REGEX = re.compile('|'.join(FUNCTIONAL_PATTERNS))

    # 노이즈 패턴 (성분이 아님)
    NOISE_PATTERNS = [
        # 문장형
        r'.*(합니다|됩니다|입니다|해요|예요|있다|없다)$',
        r'.*(사용|바르|피부|효과|개선).*',

        # 용량/규격
        r'^\d+(\.\d+)?\s*(ml|g|mg|L|kg|cm|mm)$',
        r'^SPF\d+$', r'^PA\+*$',

        # 전화번호/순수 숫자
        r'^\d{2,4}-\d{3,4}-\d{4}$',
        r'^\d{5,}$',

        # 조사/짧은 한글 노이즈
        r'^(을|를|이|가|은|는|도|만|의|에|로|으로|와|과|하고)$',
        r'^(때|때의|및|또는|또한|그리고|하여|위해|통해|대한|관한)$',
        r'^(제품|주요|사양|모든|사용|개봉|전|후|타입)$',

        # 영문 노이즈
        r'^(POINT|STEP|TIP|NOTE|HOW|WHAT|WHY|KEY|BEST|NEW|SPECIAL)\d*$',
        r'^(Book|Marine|Complex|Extra|Super|Care|Solution|Recipe)$',
    ]
    NOISE_REGEX = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]

    # 최소/최대 길이
    MIN_LENGTH = 2
    MAX_LENGTH = 100

    # 영문 단독 성분 (허용)
    ALLOWED_ENGLISH = {
        'BHT', 'BHA', 'AHA', 'PHA', 'CI77891', 'CI77491', 'CI77492', 'CI77499',
        'Water', 'Aqua', 'Mica', 'Talc',
    }

    @classmethod
    def filter_ingredients(cls, ingredients: List[str]) -> List[str]:
        """성분 리스트 필터링"""
        result = []

        for ing in ingredients:
            # 기능성 문구 분리
            ing = cls._remove_functional_text(ing)

            # 유효성 검사
            if cls._is_valid_ingredient(ing):
                result.append(ing)

        return result

    @classmethod
    def _remove_functional_text(cls, text: str) -> str:
        """기능성 문구 제거"""
        # 1. 괄호 안 기능성 문구 제거 (예: "(식품의약품안전처 심사)")
        text = re.sub(r'\([^)]*(?:식품의약품|심사|고시|기능성)[^)]*\)', '', text)

        # 2. 기능성 문구 제거
        text = cls.FUNCTIONAL_REGEX.sub('', text)

        # 3. 빈 괄호 제거
        text = re.sub(r'\(\s*\)', '', text)

        # 4. 정리
        text = text.strip(' ,;:()')

        return text

    @classmethod
    def _is_valid_ingredient(cls, text: str) -> bool:
        """유효한 성분인지 판단"""
        if not text:
            return False

        # 길이 검사
        if len(text) < cls.MIN_LENGTH or len(text) > cls.MAX_LENGTH:
            return False

        # 노이즈 패턴 검사
        for pattern in cls.NOISE_REGEX:
            if pattern.match(text):
                return False

        # 한글이 없고 허용된 영문도 아닌 경우
        if not LanguageDetector.HANGUL_PATTERN.search(text):
            if text.upper() not in cls.ALLOWED_ENGLISH:
                # 화학식인 경우 허용
                if not LanguageDetector.is_chemical_formula(text):
                    return False

        return True


# =============================================================================
# 6. 중복 성분 병합
# =============================================================================

class IngredientDeduplicator:
    """중복 성분 제거 클래스"""

    # 동일 성분 변형 패턴
    EQUIVALENT_PATTERNS = [
        # 공백 유무
        (r'\s+', ''),
        # 하이픈 유무
        (r'-', ''),
        # 애시드/애씨드
        (r'애씨드', '애시드'),
        (r'에씨드', '애시드'),
    ]

    @classmethod
    def deduplicate(cls, ingredients: List[str]) -> List[str]:
        """중복 성분 제거"""
        seen = {}  # normalized -> original
        result = []

        for ing in ingredients:
            normalized = cls._normalize_for_comparison(ing)

            if normalized not in seen:
                seen[normalized] = ing
                result.append(ing)
            else:
                # 더 긴 버전 유지 (함량 정보 등 포함)
                if len(ing) > len(seen[normalized]):
                    # 기존 것 대체
                    idx = result.index(seen[normalized])
                    result[idx] = ing
                    seen[normalized] = ing

        return result

    @classmethod
    def _normalize_for_comparison(cls, text: str) -> str:
        """비교를 위한 정규화"""
        normalized = text.lower()

        for pattern, replacement in cls.EQUIVALENT_PATTERNS:
            normalized = re.sub(pattern, replacement, normalized)

        return normalized


# =============================================================================
# 7. 메인 파서 클래스
# =============================================================================

class IngredientParserV2:
    """고도화된 성분 파서 v2"""

    @classmethod
    def parse(cls, raw_text: str, korean_only: bool = True) -> List[str]:
        """
        원본 텍스트에서 성분 리스트 추출

        Args:
            raw_text: 원본 텍스트 (HTML에서 추출된 텍스트)
            korean_only: 한글 성분만 추출 여부

        Returns:
            성분 리스트
        """
        if not raw_text:
            return []

        logger.debug(f"원본 텍스트 길이: {len(raw_text)}")

        # 0. 줄바꿈으로 분리된 성분명 병합 (콤마 변환 전에!)
        # 0-1. 슬래시(/)로 끝나는 줄바꿈 병합
        # 예: "카프릴릭/카프릭/미리스틱/\n스테아릭" → "카프릴릭/카프릭/미리스틱/스테아릭"
        raw_text = re.sub(r'/\s*[\r\n]+\s*', '/', raw_text)

        # 0-2. 하이픈(-)으로 끝나는 줄바꿈 + 숫자 병합
        # 예: "폴리글리세릴-\n10" → "폴리글리세릴-10"
        raw_text = re.sub(r'-\s*[\r\n]+\s*(\d)', r'-\1', raw_text)

        # 0-3. 한글 + 줄바꿈 + 한글 병합 (성분명이 줄바꿈으로 분리된 경우)
        # 예: "트라이글리서라\n이드" → "트라이글리서라이드"
        raw_text = re.sub(r'([가-힣])\s*[\r\n]+\s*([가-힣])', r'\1\2', raw_text)

        # 0-4. 나머지 줄바꿈을 콤마로 변환 (정규화 전에!)
        raw_text = re.sub(r'[\r\n]+', ', ', raw_text)

        # 1. 텍스트 정규화
        normalized = TextNormalizer.normalize(raw_text)

        # 2. 성분 영역 추출
        section = IngredientSectionExtractor.extract_ingredient_section(normalized)
        logger.debug(f"성분 섹션 길이: {len(section)}")

        # 3. 성분 분리
        ingredients = IngredientSplitter.split_ingredients(section)
        logger.debug(f"분리된 성분 수: {len(ingredients)}")

        # 4. 필터링
        filtered = IngredientFilter.filter_ingredients(ingredients)
        logger.debug(f"필터링 후 성분 수: {len(filtered)}")

        # 5. 한글만 필터 (옵션)
        if korean_only:
            filtered = LanguageDetector.filter_korean_only(filtered)
            logger.debug(f"한글 필터 후 성분 수: {len(filtered)}")

        # 6. 중복 제거
        deduplicated = IngredientDeduplicator.deduplicate(filtered)
        logger.debug(f"중복 제거 후 성분 수: {len(deduplicated)}")

        # 7. 성분명 정규화
        result = [TextNormalizer.normalize_ingredient_name(ing) for ing in deduplicated]

        return result

    @classmethod
    def parse_with_concentration(cls, raw_text: str) -> List[Tuple[str, Optional[str]]]:
        """
        성분과 함량 정보를 함께 추출

        Returns:
            [(성분명, 함량), ...] 리스트. 함량이 없으면 None
        """
        ingredients = cls.parse(raw_text, korean_only=True)

        result = []
        for ing in ingredients:
            name, concentration = cls._extract_concentration(ing)
            result.append((name, concentration))

        return result

    @classmethod
    def _extract_concentration(cls, text: str) -> Tuple[str, Optional[str]]:
        """성분에서 함량 정보 추출"""
        # 괄호 안 함량 패턴
        pattern = r'\(([^)]*(?:ppm|ppb|%|mg|g|ml)[^)]*)\)'
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            concentration = match.group(1).strip()
            name = text[:match.start()].strip() + text[match.end():].strip()
            return name.strip(), concentration

        return text, None


# =============================================================================
# 8. 성분 사전 기반 매칭 및 오타 교정
# =============================================================================

class IngredientDictionary:
    """성분 사전 및 오타 교정 클래스"""

    # 화장품 성분 사전 (확장)
    KNOWN_INGREDIENTS = {
        # 기본 성분
        '정제수', '물', '글리세린', '부틸렌글라이콜', '프로판다이올',
        '1,2-헥산다이올', '펜틸렌글라이콜', '디프로필렌글라이콜',

        # 방부제
        '페녹시에탄올', '메틸파라벤', '에틸파라벤', '프로필파라벤', '부틸파라벤',
        '소듐벤조에이트', '포타슘소르베이트', '에틸헥실글리세린',
        '클로르페네신', '벤질알코올', '소르빅애시드',

        # UV 필터
        '에틸헥실메톡시신나메이트', '티타늄디옥사이드', '징크옥사이드',
        '옥토크릴렌', '아보벤존', '호모살레이트', '옥틸살리실레이트',

        # 보습제
        '히알루론산', '소듐히알루로네이트', '나이아신아마이드', '판테놀',
        '세라마이드', '콜레스테롤', '스쿠알란', '시어버터',

        # 계면활성제
        '폴리소르베이트60', '폴리소르베이트80', '레시틴',
        '소듐라우릴설페이트', '코코베타인', '데실글루코사이드',

        # 증점제
        '카보머', '잔탄검', '카라기난', '셀룰로오스검',
        '아크릴레이트/C10-30알킬아크릴레이트크로스폴리머',

        # 식물 추출물
        '알로에베라잎추출물', '녹차추출물', '병풀추출물', '감초추출물',
        '카모마일추출물', '라벤더추출물', '센텔라아시아티카추출물',
        '약모밀추출물', '어성초추출물', '티트리추출물',

        # 유화제/알코올
        '스테아르산', '세틸알코올', '스테아릴알코올', '세테아릴알코올',
        '글리세릴스테아레이트', '세테아릴글루코사이드',

        # 실리콘
        '다이메치콘', '사이클로펜타실록산', '사이클로헥사실록산',
        '페닐트라이메치콘', '아모다이메치콘',

        # pH조절
        '트로메타민', '시트릭애시드', '소듐하이드록사이드',
        '다이포타슘글리시리제이트', '알란토인',

        # 미백/주름개선
        '아스코르브산', '아스코빌글루코사이드', '아데노신', '레티놀',
        '트라넥사믹애시드', '알부틴',

        # 향료/기타
        '향료', '리날룰', '리모넨', 'BHT', 'BHA', 'EDTA',
        '토코페롤', '토코페릴아세테이트',
    }

    # 흔한 오타 매핑
    TYPO_CORRECTIONS = {
        '글리세 린': '글리세린',
        '메칠파라벤': '메틸파라벤',
        '에칠파라벤': '에틸파라벤',
        '트리메치콘': '트라이메치콘',
        '하이드록사이드': '하이드록사이드',
        '애씨드': '애시드',
        '에씨드': '애시드',
        '소디움': '소듐',
        '포타슘': '포타슘',
        '칼슘': '칼슘',
        '폴리소베이트': '폴리소르베이트',
    }

    @classmethod
    def correct_typo(cls, ingredient: str) -> str:
        """오타 교정"""
        corrected = ingredient
        for wrong, correct in cls.TYPO_CORRECTIONS.items():
            corrected = corrected.replace(wrong, correct)
        return corrected

    @classmethod
    def find_similar(cls, ingredient: str, cutoff: float = 0.75) -> Tuple[str, float]:
        """
        사전에서 유사한 성분 찾기

        Returns:
            (교정된 성분명, 유사도) 또는 (원본, 0.0)
        """
        from difflib import get_close_matches

        # 먼저 오타 교정 적용
        corrected = cls.correct_typo(ingredient)

        # 정확히 일치하면 반환
        if corrected in cls.KNOWN_INGREDIENTS:
            return corrected, 1.0

        # 공백 제거 버전으로 비교
        normalized = corrected.replace(' ', '')
        normalized_dict = {ing.replace(' ', ''): ing for ing in cls.KNOWN_INGREDIENTS}

        if normalized in normalized_dict:
            return normalized_dict[normalized], 1.0

        # 유사도 기반 매칭
        matches = get_close_matches(normalized, normalized_dict.keys(), n=1, cutoff=cutoff)

        if matches:
            matched = matches[0]
            # 간단한 유사도 계산
            similarity = len(set(normalized) & set(matched)) / len(set(normalized) | set(matched))
            return normalized_dict[matched], similarity

        return corrected, 0.0

    @classmethod
    def validate_and_correct(cls, ingredients: List[str]) -> List[Tuple[str, str, float]]:
        """
        성분 리스트 검증 및 교정

        Returns:
            [(원본, 교정됨, 유사도), ...]
        """
        results = []
        for ing in ingredients:
            corrected, similarity = cls.find_similar(ing)
            results.append((ing, corrected, similarity))
        return results


# =============================================================================
# 9. 통합 후처리 함수 (기존 postprocessor 대체)
# =============================================================================

def postprocess_ingredients_v2(raw_text: str,
                               korean_only: bool = True,
                               use_dictionary: bool = True,
                               min_length: int = 2) -> List[dict]:
    """
    OCR/HTML 텍스트를 성분 리스트로 후처리 (v2)

    기존 postprocess_ingredients 대체 함수

    Args:
        raw_text: 원본 텍스트
        korean_only: 한글 성분만 추출
        use_dictionary: 사전 기반 오타 교정 사용
        min_length: 최소 성분명 길이

    Returns:
        List of dict: [
            {
                'original': '원본 성분명',
                'normalized': '정규화된 성분명',
                'corrected': '교정된 성분명',
                'similarity': 유사도,
                'is_known': 사전에 있는 성분인지
            },
            ...
        ]
    """
    # 1. 파서로 성분 추출
    ingredients = IngredientParserV2.parse(raw_text, korean_only=korean_only)

    results = []
    for ing in ingredients:
        if len(ing) < min_length:
            continue

        result = {
            'original': ing,
            'normalized': ing,
            'corrected': ing,
            'similarity': 0.0,
            'is_known': False
        }

        # 2. 사전 기반 교정
        if use_dictionary:
            corrected, similarity = IngredientDictionary.find_similar(ing)
            result['corrected'] = corrected
            result['similarity'] = similarity
            result['is_known'] = similarity >= 0.75

        results.append(result)

    return results


def get_clean_ingredients(raw_text: str, korean_only: bool = True) -> List[str]:
    """
    간단한 성분 리스트 추출 (정리된 형태)

    Args:
        raw_text: 원본 텍스트
        korean_only: 한글만 추출

    Returns:
        정리된 성분 리스트
    """
    return IngredientParserV2.parse(raw_text, korean_only=korean_only)


# =============================================================================
# 테스트 함수
# =============================================================================

def test_parser():
    """파서 테스트"""
    test_cases = [
        # 일반 성분
        "정제수, 글리세린, 부틸렌글라이콜, 나이아신아마이드",

        # 함량 포함
        "병풀잎수(95%), 1,2-헥산다이올, 정제수, 병풀추출물(1%)",

        # 슬래시 포함
        "약모밀꽃/잎/줄기수(95%), 1-2-헥산다이올, 정제수, 약모밀추출물(1%)",

        # 기능성 문구 포함
        "다이포타슘글리시리제이트기능성화장품식품의약품미백, 나이아신아마이드",

        # 영문/한글 혼재
        "정제수, Water, 글리세린, Glycerin, 부틸렌글라이콜",

        # 줄바꿈
        "정제수\n글리세린\n부틸렌글라이콜",
    ]

    print("=" * 60)
    print("성분 파서 v2 테스트")
    print("=" * 60)

    for text in test_cases:
        print(f"\n입력: {text[:50]}...")
        result = IngredientParserV2.parse(text)
        print(f"결과 ({len(result)}개): {result}")

    # 후처리 함수 테스트
    print("\n" + "=" * 60)
    print("후처리 함수 테스트 (사전 매칭)")
    print("=" * 60)

    test_text = "정제수, 글리세 린, 메칠파라벤, 나이아신아마이드, 알로에베라잎추출물"
    results = postprocess_ingredients_v2(test_text)

    for r in results:
        status = "✓" if r['is_known'] else "?"
        correction = f" → {r['corrected']}" if r['original'] != r['corrected'] else ""
        print(f"  {status} {r['original']}{correction} (유사도: {r['similarity']:.2f})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_parser()
