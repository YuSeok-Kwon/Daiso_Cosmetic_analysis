"""
성분명 파싱 및 유효성 검증 모듈
"""
import re

# OCR 오인식 패턴 및 수정 맵
OCR_CORRECTIONS = {
    '메칠': '메틸',
    '에칠': '에틸',
    '부칠': '부틸',
    '프로필': '프로필',
    '다이 메치콘': '다이메티콘',
    '글리세 린': '글리세린',
    '페녹시 에탄올': '페녹시에탄올',
    '디옥사이 드': '디옥사이드',
    '파라 벤': '파라벤',
    '이트륨': '이트',
    '콜산': '콜',
}

# 화학 성분 패턴 (정규표현식)
INGREDIENT_PATTERNS = [
    r'.+이트$', r'.+올$', r'.+산$', r'.+염$',
    r'.+레이트$', r'.+라이드$', r'.+나이트$',
    r'.+민$', r'.+틴$', r'.+린$',
    r'.+옥사이드$', r'.+추출물$', r'.+오일$',
    r'.+왁스$', r'.+파라벤$', r'.+글라이콜$',
    r'.+다이올$', r'^CI\d+$',
]

# 노이즈 패턴 (성분이 아님)
NOISE_PATTERNS = [
    r'.*(사용|바르|피부|효과|개선|미백|주름|보습|수분|영양|진정|바릅니다|발라주세요).*',
    r'^SPF\d+$', r'^PA\+*$', r'^\d+ml$', r'^\d+g$',
    r'.*(합니다|됩니다|입니다|해요|돼요|예요|있다|없다|한다|된다)$',
    r'^(을|를|이|가|은|는|도|만|의|에|로|으로|와|과|하고)$',
]

# 성분 키워드 (실제 성분 리스트를 나타내는 키워드)
INGREDIENT_KEYWORDS = [
    "전성분", "화장품법", "모든 성분",
    "화장품법에 따라", "기재 표시", "기재·표시", "기재표시", "표시하여야", "표시 하여야",
    "하여야하는", "하여야 하는",
    "성분:",
    "INGREDIENTS", "Ingredients",
]

# 성분 섹션이 아닌 키워드
INGREDIENT_EXCLUDE_SECTIONS = [
    "상품 특징", "상품 설명", "제품 설명", "특징",
    "사용 방법", "주의 사항", "보관 방법",
    "HOWTOUSE", "상품특징", "제품특징"
]

# 성분 추출 중단 키워드
INGREDIENT_STOP_KEYWORDS = [
    '※', '주의사항', '경고', '사용방법', '보관방법', '제조국',
    '용법', '용량', '효능', '효과',
    '본품', '적당량', '취해', '골고루', '바른다', '식품의약품안전처'
]

# 성분이 아닌 불용어
INGREDIENT_STOPWORDS = [
    '기재하여야', '하는', '등', '하여야하는', '기재·표시',
    '자외선', '차단', '심사', '유무', '개선', '미백', '주름',
    '피부', '보호', '도움', '효능', '효과', '용법', '용량',
    '본품', '적당량', '바른다', '준다', '으로부터', '를',
    '한다', '합니다', '된다', '됩니다', '있다', '없다',
    '보호한다', '개선한다', '도움을준다', '미백에도움',
]

# 알려진 화장품 성분 데이터베이스
KNOWN_INGREDIENTS = {
    # 물/용매
    '정제수', '물', 'Water', 'Aqua',

    # 보습제
    '글리세린', 'Glycerin', '부틸렌글라이콜', 'ButyleneGlycol',
    '프로판다이올', 'Propanediol', '소르비톨', 'Sorbitol',
    '펜틸렌글라이콜', '1,2-헥산다이올', 'Dipropyleneglycol',
    '프로필렌글라이콜', 'PropyleneGlycol',

    # 유화제
    '폴리소르베이트60', '폴리소르베이트80', '레시틴', 'Lecithin',
    '스테아르산', '세틸알코올', '스테아릴알코올',

    # 증점제
    '카보머', 'Carbomer', '잔탄검', '히알루론산나트륨',
    '카라기난', '셀룰로오스검', '아크릴레이트코폴리머',

    # 방부제
    '페녹시에탄올', 'Phenoxyethanol', '메틸파라벤', '에틸파라벤',
    '프로필파라벤', '부틸파라벤', '소듐벤조에이트', '포타슘소르베이트',

    # 항산화제
    '토코페롤', 'Tocopherol', '아스코르브산', '부틸하이드록시톨루엔',
    'BHT', 'BHA',

    # UV 필터
    '에틸헥실메톡시신나메이트', '티타늄디옥사이드', '징크옥사이드',
    '옥토크릴렌', '아보벤존', '호모살레이트',

    # 색소
    '산화철', '황산화철', '적산화철', '흑산화철',
    'CI77891', 'CI77491', 'CI77492', 'CI77499',
    '운모티타늄', '마이카', 'Mica',

    # 향료
    '향료', 'Fragrance', 'Parfum',

    # 식물추출물
    '알로에베라잎추출물', '녹차추출물', '병풀추출물',
    '감초추출물', '카모마일추출물', '라벤더추출물',
    '센텔라아시아티카추출물',

    # 실리콘류
    '다이메치콘', 'Dimethicone', '사이클로펜타실록산',
    '사이클로헥사실록산', '다이메치콘올', '아모다이메치콘',

    # 오일/왁스
    '스쿠알란', 'Squalane', '호호바오일', '시어버터',
    '미네랄오일', 'MineralOil', '세레신', '마이크로크리스탈린왁스', '파라핀',
}


def normalize_ingredient_name(name: str) -> str:
    """
    성분명 정규화 및 OCR 오류 수정

    Args:
        name: 원본 성분명

    Returns:
        str: 정규화된 성분명
    """
    # 1. 공백 제거
    name = re.sub(r'\s+', '', name)

    # 2. OCR 오류 수정
    for wrong, correct in OCR_CORRECTIONS.items():
        name = name.replace(wrong, correct)

    # 3. 특수문자 정규화
    name = name.replace('±', '')
    name = name.replace('·', '')
    name = re.sub(r'[�]', '', name)  # 깨진 문자 제거

    # 4. 괄호 내용 제거
    name = re.sub(r'\([^)]*\)', '', name)
    name = re.sub(r'\[[^\]]*\]', '', name)

    # 5. 기타 특수문자 제거
    name = re.sub(r'[*★☆※]', '', name)

    # 6. 앞뒤 특수문자 제거
    name = re.sub(r'^[^\w가-힣]+|[^\w가-힣]+$', '', name, flags=re.UNICODE)

    # 7. OCR 오타 수정
    name = name.replace('에칠', '에틸')
    name = name.replace('애씨드', '애시드')
    name = name.replace('다이메i콘', '다이메티콘')
    name = name.replace('피이지', 'PEG')
    name = name.replace('비에이치티', 'BHT')

    return name.strip()


def is_valid_ingredient(text: str, known_db: set = KNOWN_INGREDIENTS) -> tuple:
    """
    성분명 유효성 검증

    Args:
        text: 검증할 텍스트
        known_db: 알려진 성분 데이터베이스

    Returns:
        tuple: (is_valid: bool, confidence: float, reason: str)
    """
    if not text or len(text.strip()) < 2:
        return False, 0.0, "too_short"

    text = text.strip()

    # 레벨 1: 알려진 성분 데이터베이스 매칭
    if text in known_db:
        return True, 1.0, "known_ingredient"

    # 레벨 2: 노이즈 패턴 매칭
    for pattern in NOISE_PATTERNS:
        if re.match(pattern, text):
            return False, 0.0, "noise"

    # 레벨 3: 불용어 체크
    if text in INGREDIENT_STOPWORDS:
        return False, 0.0, "stopword"

    for stopword in INGREDIENT_STOPWORDS:
        if stopword in text:
            return False, 0.0, "contains_stopword"

    # 레벨 4: 화학 성분 패턴 매칭
    for pattern in INGREDIENT_PATTERNS:
        if re.match(pattern, text):
            return True, 0.8, "pattern_match"

    # 레벨 5: 영문+숫자 조합
    if re.search(r'[A-Z0-9]', text) and len(text) >= 3:
        return True, 0.75, "chemical_formula"

    # 레벨 6: 영문 대문자 시작
    if re.match(r'^[A-Z][a-z]{2,}', text):
        return True, 0.6, "capitalized_word"

    # 레벨 7: 한글 화학 성분
    if re.search(r'[가-힣]{3,}', text):
        return True, 0.6, "korean_ingredient"

    return False, 0.3, "no_pattern_match"


def extract_from_text(text: str, source: str) -> list:
    """
    텍스트에서 성분 추출

    Args:
        text: OCR 또는 ALT 텍스트
        source: 출처 (HTML, ALT_0, OCR_0_1 등)

    Returns:
        list: [{'ingredient': str, 'source': str}, ...]
    """
    ingredients = []

    lines = text.split('\n')
    in_ingredients = False
    ingredient_text = []

    for idx, line in enumerate(lines):
        line = line.strip()

        # 제외 섹션
        if any(exclude in line for exclude in INGREDIENT_EXCLUDE_SECTIONS):
            in_ingredients = False
            continue

        # 성분 섹션 시작
        if any(kw in line for kw in INGREDIENT_KEYWORDS):
            in_ingredients = True

            # 키워드 제거
            for kw in INGREDIENT_KEYWORDS:
                line = line.replace(kw, '')

            line = line.strip()
            if line:
                ingredient_text.append(line)
            continue

        # 성분 섹션 종료
        if in_ingredients:
            if any(stop in line for stop in INGREDIENT_STOP_KEYWORDS):
                break

            if line:
                ingredient_text.append(line)

    # 파싱: 쉼표, 공백 모두 고려
    full_text = ' '.join(ingredient_text)

    # 1차: 쉼표로 분리
    parts = full_text.split(',')

    for part in parts:
        part = part.strip()

        if not part:
            continue

        # 2차: 공백으로 분리
        sub_parts = part.split()

        # 전체를 하나의 성분으로 시도
        if len(part) >= 2:
            ingredients.append({'ingredient': part, 'source': source})

        # 개별 단어도 성분일 수 있으므로 추가
        for sub in sub_parts:
            if len(sub) >= 2 and sub != part:
                ingredients.append({'ingredient': sub, 'source': source})

    return ingredients
