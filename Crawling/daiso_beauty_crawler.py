"""
다이소 뷰티/위생 카테고리 통합 크롤러
- 제품 정보 (product_all.csv)
- 리뷰 (reviews_all.csv)
- 성분 (ingredients_all.csv)
"""
import os
import time
import re
import pandas as pd
from collections import defaultdict
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from config import DAISO_BEAUTY_CATEGORIES
from modules.ocr_utils_split import extract_text_from_image_url_split
from utils import setup_logger, get_date_string
from modules.image_preprocessor import preprocess_for_korean_ocr

# 로거 설정
logger = setup_logger('daiso_beauty_crawler', 'daiso_beauty_crawler.log')

# 기본 설정
BASE_URL = "https://www.daisomall.co.kr"
MAX_SCROLLS = 10
user_id_map = defaultdict(lambda: f"user_{len(user_id_map)+1:04d}")

# ============================================================================
# 성분 분석 고도화: 데이터베이스 및 API 검증 시스템
# ============================================================================

# 할랄/비건 인증 정보
# 동물성 성분 데이터베이스 (비건/할랄 부적합)
ANIMAL_DERIVED_INGREDIENTS = {
    # 확실한 동물 유래 성분 (비건 불가)
    '라놀린', 'Lanolin', '양모지', '콜라겐', 'Collagen',
    '케라틴', 'Keratin', '밀크프로틴', 'MilkProtein', '실크프로틴', 'SilkProtein',
    '꿀', 'Honey', '밀랍', 'Beeswax', '프로폴리스', 'Propolis',
    '로열젤리', 'RoyalJelly', '비즈왁스',
    '카민', '코치닐', 'Carmine', '캐비어', 'Caviar', '펄', 'Pearl',
    '젤라틴', 'Gelatin', '카제인', 'Casein',
    '키토산', 'Chitosan', '콘드로이틴', 'Chondroitin',
    '엘라스틴', 'Elastin', '스쿠알렌', 'Squalene',  # 상어 유래
    '뮤스크', 'Musk', '앰버그리스', 'Ambergris',
    '타우린', 'Taurine', '알부민', 'Albumin',
}

# 할랄 부적합 성분 (알코올, 돼지 유래 등)
HARAM_INGREDIENTS = {
    # 알코올류 (할랄 금지)
    '에탄올', '알코올', 'Alcohol', 'Ethanol',
    '알코올변성', '변성알코올', 'AlcoholDenat', 'DenaturatedAlcohol',
    '이소프로필알코올', 'IsopropylAlcohol',
    'SD알코올', 'SDAlcohol',

    # 돼지 유래 성분 (할랄 절대 금지)
    '돼지콜라겐', 'PorkCollagen', '돼지젤라틴', 'PorkGelatin',
    '포신글리콜', 'Placenta',  # 돼지 태반

    # 주류 유래
    '와인추출물', 'WineExtract', '맥주효모', 'BeerYeast',
}

# 애매한 성분 (식물성/동물성 혼재 - 원료 확인 필요)
AMBIGUOUS_INGREDIENTS = {
    '글리세린', 'Glycerin', 'Glycerol',  # 대부분 식물성 (야자유, 코코넛)
    '스쿠알란', 'Squalane',  # 올리브 또는 상어 유래
    '레시틴', 'Lecithin',  # 대두 또는 계란 유래
    '스테아르산', 'StearicAcid',  # 식물 또는 동물 지방
    '세라마이드', 'Ceramide',  # 합성 또는 동물 유래
    '콜레스테롤', 'Cholesterol',  # 양모지 또는 합성
    '히알루론산', 'HyaluronicAcid',  # 닭벼슬 또는 발효 (현대는 대부분 발효)
    '히알루론산나트륨', 'SodiumHyaluronate',  # 발효 유래 (대부분 비건)
    '토코페롤', 'Tocopherol',  # 비타민E, 대부분 식물성
}

# 비건 인증 가능 성분 (식물성 확정)
VEGAN_SAFE_INGREDIENTS = {
    # 식물 추출물
    '알로에베라잎추출물', 'AloeVeraLeafExtract', '녹차추출물', 'GreenTeaExtract',
    '병풀추출물', 'CentellAAssiaticaExtract', '감초추출물', 'LicoriceExtract',
    '카모마일추출물', 'ChamomileExtract', '라벤더추출물', 'LavenderExtract',
    '센텔라아시아티카추출물', '감나무잎추출물', '카카오추출물',
    '포도추출물', '포도주추출물', '아사이팜열매추출물', '아사이베리추출물',
    '블루베리추출물', '로우스위트블루베리추출물', '라즈베리추출물',
    '딸기추출물', '서양산딸기추출물', '스트로베리추출물', '비치스트로베리추출물',
    '자작나무추출물', '자작나무싹추출물', '만주자작나무싹추출물',
    '개암추출물', '유럽개암싹추출물', '올리브나무싹추출물',
    '호두나무잎추출물', '초피나무열매추출물', '할미꽃추출물',
    '우스니아추출물', '커피콩추출물', '커피추출물',

    # 식물성 오일
    '호호바오일', 'JojobaOil', '시어버터', 'SheaButter',
    '코코넛오일', 'CoconutOil', '올리브오일', 'OliveOil',
    '아보카도오일', 'AvocadoOil', '해바라기씨오일', 'SunflowerSeedOil',
    '아르간오일', 'ArganOil', '아몬드오일', 'AlmondOil',
    '포도씨오일', 'GrapeSeedOil', '동백오일', 'CamelliaOil',
    '로즈힙오일', 'RosehipOil', '마카다미아오일', 'MacadamiaOil',

    # 합성 성분 (비동물성)
    '정제수', 'Water', 'Aqua',
    '부틸렌글라이콜', 'ButyleneGlycol',
    '디프로필렌글라이콜', 'DipropyleneGlycol',
    '프로판다이올', 'Propanediol',
    '페녹시에탄올', 'Phenoxyethanol',
    '소듐폴리아크릴레이트', 'SodiumPolyacrylate',
    '잔탄검', 'XanthanGum',
    '카보머', 'Carbomer',
    '티타늄디옥사이드', 'TitaniumDioxide',
    '징크옥사이드', 'ZincOxide',
    '나이아신아마이드', 'Niacinamide',
    '토코페롤', 'Tocopherol',  # 비타민E, 식물 유래
    '소듐하이알루로네이트', 'SodiumHyaluronate',  # 발효 유래
    '벤질글라이콜', 'BenzylGlycol',
    '에틸헥실글리세린', 'EthylhexylGlycerin',

    # 미네랄
    '마이카', 'Mica', '운모', '산화철', 'IronOxide',
}

# 알려진 화장품 성분 데이터베이스 (식약처 기준)
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

# OCR 오인식 패턴 및 수정 맵
OCR_CORRECTIONS = {
    # 자주 나오는 OCR 오류 패턴
    '메칠': '메틸',
    '에칠': '에틸',
    '부칠': '부틸',
    '프로필': '프로필',
    '다이 메치콘': '다이메치콘',
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
    "화장품법에 따라", "기재 표시", "표시하여야",
    "성분:",  # 추가: "성분:" 형식
    "INGREDIENTS", "Ingredients",  # 영어 키워드 추가
]

# 성분 섹션이 아닌 키워드 (성분이라는 단어가 포함되어도 제외해야 할 섹션)
INGREDIENT_EXCLUDE_SECTIONS = [
    "상품 특징", "상품 설명", "제품 설명", "특징",
    "사용 방법", "주의 사항", "보관 방법",
    "HOWTOUSE", "상품특징", "제품특징"
]

# 성분 추출 중단 키워드 (성분 이후 나오는 설명문)
INGREDIENT_STOP_KEYWORDS = [
    '※', '주의사항', '경고', '사용방법', '보관방법', '제조국',
    '용법', '용량', '효능', '효과',
    '본품', '적당량', '취해', '골고루', '바른다', '식품의약품안전처'
]

# 성분이 아닌 불용어 (단독으로 나오면 제외)
INGREDIENT_STOPWORDS = [
    '기재하여야', '하는', '등', '하여야하는', '기재·표시',
    '자외선', '차단', '심사', '유무', '개선', '미백', '주름',
    '피부', '보호', '도움', '효능', '효과', '용법', '용량',
    '본품', '적당량', '바른다', '준다', '으로부터', '를',
    '한다', '합니다', '된다', '됩니다', '있다', '없다',  # 어미 추가
    '보호한다', '개선한다', '도움을준다', '미백에도움',  # 조합 추가
]


def parse_count(text: str) -> int:
    """숫자 파싱"""
    try:
        return int(text.replace(",", "").replace("+", "").strip() or 0)
    except:
        return 0


def extract_rating(text: str):
    """평점 추출"""
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else None


def normalize_ingredient_name(name: str) -> str:
    """성분명 정규화 및 OCR 오류 수정"""
    # 1. 공백 제거
    name = name.replace(' ', '')

    # 2. OCR 오류 수정
    for wrong, correct in OCR_CORRECTIONS.items():
        name = name.replace(wrong, correct)

    # 3. 괄호 내용 제거 (농도 표시 등)
    name = re.sub(r'\([^)]*\)', '', name)
    name = re.sub(r'\[[^\]]*\]', '', name)

    # 4. 특수문자 제거 (±, *, 등)
    name = re.sub(r'[±*★☆※]', '', name)

    # 5. 앞뒤 특수문자 제거
    name = re.sub(r'^[^\w가-힣]+|[^\w가-힣]+$', '', name, flags=re.UNICODE)

    return name.strip()


def is_valid_ingredient(text: str, known_db: set = KNOWN_INGREDIENTS) -> tuple:
    """
    성분명 유효성 검증 (고도화 버전)

    Returns:
        (is_valid: bool, confidence: float, reason: str)
    """
    if not text or len(text.strip()) < 2:
        return False, 0.0, "too_short"

    text = text.strip()

    # 레벨 1: 알려진 성분 데이터베이스 매칭 (100% 확신)
    if text in known_db:
        return True, 1.0, "known_ingredient"

    # 레벨 2: 노이즈 패턴 매칭 (100% 제외)
    for pattern in NOISE_PATTERNS:
        if re.match(pattern, text):
            return False, 0.0, f"noise"

    # 레벨 3: 불용어 체크 (기존 로직 유지)
    if text in INGREDIENT_STOPWORDS:
        return False, 0.0, "stopword"

    for stopword in INGREDIENT_STOPWORDS:
        if stopword in text:
            return False, 0.0, "contains_stopword"

    # 레벨 4: 화학 성분 패턴 매칭 (80% 확신)
    for pattern in INGREDIENT_PATTERNS:
        if re.match(pattern, text):
            if len(text) >= 3:
                return True, 0.8, "pattern_match"

    # 레벨 5: 영문+숫자 조합 (화학식) (70% 확신)
    if re.search(r'[A-Z][a-z]*\d', text) and len(text) >= 4:
        return True, 0.7, "chemical_formula"

    # 레벨 6: 영문 대문자 시작 (60% 확신)
    if re.match(r'^[A-Z][a-z]{2,}', text):
        return True, 0.6, "capitalized_word"

    # 기타 (40% 미만은 제외)
    return False, 0.3, "no_pattern_match"


def clean_ingredient_text(text: str) -> str:
    """성분명 정제 (기존 함수 유지)"""
    # 앞뒤 공백 제거
    text = text.strip()

    # 괄호 제거 (예: "보호한다(SPF35" → "보호한다SPF35")
    text = re.sub(r'[\(\)\[\]]', '', text)

    # 앞뒤 특수문자 제거
    text = re.sub(r'^[^\w가-힣]+|[^\w가-힣]+$', '', text)

    # ± 기호 제거 (예: "±황색산화철" → "황색산화철")
    text = text.replace('±', '')

    # 연속된 공백 제거
    text = re.sub(r'\s+', '', text)

    return text


def extract_from_text(text: str, source: str) -> list:
    """텍스트에서 성분 추출 (다중 소스 지원)"""
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

        # 2차: 공백으로 분리 (단, 2단어 이상은 제외)
        if ' ' in part and part.count(' ') <= 2:
            sub_parts = part.split()
            for sub in sub_parts:
                if len(sub) >= 2:
                    ingredients.append({'ingredient': sub, 'source': source})
        else:
            if len(part) >= 2:
                ingredients.append({'ingredient': part, 'source': source})

    return ingredients


def check_halal_vegan_status(ingredient: str) -> dict:
    """
    성분의 할랄/비건 적합성 판정

    Returns:
        dict with: is_vegan, is_halal, warning
    """
    result = {
        'is_vegan': 'Unknown',
        'is_halal': 'Unknown',
        'warning': ''
    }

    # 비건 체크 (우선순위: 확정 > 애매 > 불가)
    if ingredient in VEGAN_SAFE_INGREDIENTS:
        result['is_vegan'] = 'Yes'
    elif ingredient in AMBIGUOUS_INGREDIENTS:
        # 글리세린, 히알루론산 등 - 현대 화장품은 대부분 식물성/발효
        result['is_vegan'] = 'Likely'  # 조건부 가능
        result['warning'] = '원료 확인 권장 (대부분 식물성/발효)'
    elif ingredient in ANIMAL_DERIVED_INGREDIENTS:
        result['is_vegan'] = 'No'
        result['warning'] = '동물성 유래 성분'

    # 할랄 체크
    if ingredient in HARAM_INGREDIENTS:
        result['is_halal'] = 'No'
        result['warning'] = result['warning'] + ' / 알코올 함유' if result['warning'] else '알코올 함유'
    elif ingredient in ANIMAL_DERIVED_INGREDIENTS and '돼지' not in ingredient:
        result['is_halal'] = 'Questionable'
        result['warning'] = result['warning'] + ' / 원료 확인 필요' if result['warning'] else '원료 확인 필요'
    elif ingredient in AMBIGUOUS_INGREDIENTS:
        # 애매한 성분은 할랄도 원료 확인 필요
        if not result['warning']:
            result['warning'] = '원료 확인 권장'

    return result


def extract_ingredients_multi_source(driver, product_code: str, product_name: str) -> list:
    """
    다중 소스에서 성분 추출 및 교차 검증 + 할랄/비건 판정

    Returns:
        list of dicts with: product_code, ingredient, confidence, sources, reason, is_vegan, is_halal, warning
    """
    all_ingredients = {}  # {성분명: {confidence, sources[], reason}}

    # 소스 1: HTML 텍스트
    try:
        editor_content = driver.find_element(By.CSS_SELECTOR, "div.editor-content")
        html_text = editor_content.text
        html_ingredients = extract_from_text(html_text, source="HTML")

        for ing in html_ingredients:
            name = normalize_ingredient_name(ing['ingredient'])
            is_valid, conf, reason = is_valid_ingredient(name)

            if is_valid and conf >= 0.5:  # 50% 이상만
                if name not in all_ingredients:
                    all_ingredients[name] = {'confidence': conf, 'sources': [ing['source']], 'reason': reason}
                else:
                    # 같은 성분이 여러 소스에서 발견되면 신뢰도 증가
                    all_ingredients[name]['sources'].append(ing['source'])
                    all_ingredients[name]['confidence'] = min(1.0, all_ingredients[name]['confidence'] + 0.1)

        logger.info(f"HTML에서 {len(html_ingredients)}개 성분 추출 → 유효: {len([k for k in all_ingredients if 'HTML' in all_ingredients[k]['sources']])}개")

    except Exception as e:
        logger.debug(f"HTML 텍스트 추출 실패: {str(e)}")

    # 소스 2: Picture alt 속성
    try:
        pictures = driver.find_elements(By.CSS_SELECTOR, "div.editor-content picture img")

        for idx, img in enumerate(pictures):
            alt_text = img.get_attribute("alt") or ""

            if any(kw in alt_text for kw in INGREDIENT_KEYWORDS):
                alt_ingredients = extract_from_text(alt_text, source=f"ALT_{idx}")

                for ing in alt_ingredients:
                    name = normalize_ingredient_name(ing['ingredient'])
                    is_valid, conf, reason = is_valid_ingredient(name)

                    if is_valid and conf >= 0.5:
                        if name not in all_ingredients:
                            all_ingredients[name] = {'confidence': conf, 'sources': [ing['source']], 'reason': reason}
                        else:
                            all_ingredients[name]['sources'].append(ing['source'])
                            all_ingredients[name]['confidence'] = min(1.0, all_ingredients[name]['confidence'] + 0.1)

        logger.info(f"ALT에서 추가 성분 발견: 총 {len([k for k in all_ingredients if any('ALT' in s for s in all_ingredients[k]['sources'])])}개")

    except Exception as e:
        logger.debug(f"ALT 텍스트 추출 실패: {str(e)}")

    # 소스 3: OCR (성분이 적을 때만)
    if len(all_ingredients) < 5:
        try:
            pictures = driver.find_elements(By.CSS_SELECTOR, "div.editor-content picture img")

            for idx, img in enumerate(pictures[-3:]):  # 마지막 3개 이미지만
                src = img.get_attribute("src")

                if src:
                    logger.info(f"OCR 분석 중: 이미지 {idx + 1}")

                    sections = extract_text_from_image_url_split(src, num_sections=5)  # 5개 섹션으로 세분화

                    for section_idx, section in enumerate(sections or []):
                        text = section.get('text', '')

                        if any(kw in text for kw in INGREDIENT_KEYWORDS):
                            ocr_ingredients = extract_from_text(text, source=f"OCR_{idx}_{section_idx}")

                            for ing in ocr_ingredients:
                                name = normalize_ingredient_name(ing['ingredient'])
                                is_valid, conf, reason = is_valid_ingredient(name)

                                # OCR은 오류 가능성이 있으므로 신뢰도 페널티
                                conf *= 0.9

                                if is_valid and conf >= 0.5:
                                    if name not in all_ingredients:
                                        all_ingredients[name] = {'confidence': conf, 'sources': [ing['source']], 'reason': reason}
                                    else:
                                        all_ingredients[name]['sources'].append(ing['source'])
                                        all_ingredients[name]['confidence'] = min(1.0, all_ingredients[name]['confidence'] + 0.05)

            logger.info(f"OCR에서 추가 성분: 총 {len([k for k in all_ingredients if any('OCR' in s for s in all_ingredients[k]['sources'])])}개")

        except Exception as e:
            logger.error(f"OCR 실패: {str(e)}")

    # 최종 필터링: 신뢰도 기준 정렬 및 할랄/비건 판정
    final_ingredients = []

    for name, info in all_ingredients.items():
        # 여러 소스에서 발견된 성분 우선
        multi_source_bonus = len(info['sources']) * 0.05
        final_conf = min(1.0, info['confidence'] + multi_source_bonus)

        # 신뢰도 50% 이상만 포함
        if final_conf >= 0.5:
            # 할랄/비건 적합성 판정
            halal_vegan = check_halal_vegan_status(name)

            final_ingredients.append({
                'product_code': product_code,
                'ingredient': name,
                'confidence': round(final_conf, 2),
                'sources': ','.join(info['sources']),
                'reason': info['reason'],
                'is_vegan': halal_vegan['is_vegan'],
                'is_halal': halal_vegan['is_halal'],
                'warning': halal_vegan['warning']
            })

    # 신뢰도 높은 순으로 정렬
    final_ingredients.sort(key=lambda x: x['confidence'], reverse=True)

    logger.info(f"최종 성분: {len(final_ingredients)}개 (신뢰도 50% 이상)")
    logger.info(f"  - 신뢰도 90% 이상: {len([x for x in final_ingredients if x['confidence'] >= 0.9])}개")
    logger.info(f"  - 신뢰도 70-90%: {len([x for x in final_ingredients if 0.7 <= x['confidence'] < 0.9])}개")
    logger.info(f"  - 신뢰도 50-70%: {len([x for x in final_ingredients if 0.5 <= x['confidence'] < 0.7])}개")

    # 할랄/비건 통계
    vegan_count = len([x for x in final_ingredients if x['is_vegan'] == 'Yes'])
    non_vegan_count = len([x for x in final_ingredients if x['is_vegan'] == 'No'])
    halal_questionable = len([x for x in final_ingredients if x['is_halal'] == 'Questionable'])
    haram_count = len([x for x in final_ingredients if x['is_halal'] == 'No'])

    logger.info(f"할랄/비건 분석:")
    logger.info(f"  - 비건 적합: {vegan_count}개 | 부적합: {non_vegan_count}개")
    logger.info(f"  - 할랄 의심: {halal_questionable}개 | 부적합: {haram_count}개")

    return final_ingredients


def get_category_url(middle_code, small_code):
    """카테고리 URL 생성"""
    return f"{BASE_URL}/ds/exhCtgr/C208/CTGR_00014/{middle_code}/{small_code}"


def get_all_product_links(driver, category_url, category_name):
    """제품 링크 수집"""
    logger.info(f"[{category_name}] 제품 링크 수집 시작: {category_url}")
    driver.get(category_url)

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "product-list"))
    )

    # 페이지 스크롤
    for _ in range(MAX_SCROLLS):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

    # 제품 링크 수집
    items = driver.find_elements(By.CLASS_NAME, "prod-thumb__link")
    links = []

    for item in items:
        href = item.get_attribute("href")
        if href and "pdNo=" in href:
            links.append(href)
        else:
            html = item.get_attribute("outerHTML")
            match = re.search(r"pdNo=(\d+)", html)
            if match:
                pdno = match.group(1)
                full_url = f"{BASE_URL}/pd/pdr/SCR_PDR_0001?pdNo={pdno}&recmYn=N"
                links.append(full_url)

    dedup = list(dict.fromkeys(links))
    logger.info(f"[{category_name}] 총 {len(dedup)}개 제품 링크 수집 완료")
    return dedup


def extract_brand(driver, category_2=""):
    """브랜드 추출"""
    # 1. 브랜드 영역에서 추출 시도 (최우선)
    try:
        title_elem = driver.find_element(By.CSS_SELECTOR, "a.brand-area div.brand-area__detail div.detail-title")
        brand_text = title_elem.text.strip()
        if brand_text:
            return brand_text
    except:
        pass

    # 2. 브랜드 영역이 없는 경우
    # 2-1. 화장품이 아닌 카테고리는 "다이소"로 설정
    non_cosmetic_categories = ["메이크업 브러쉬", "메이크업 퍼프", "메이크업 소품", "퍼프브러시세척"]
    if category_2 in non_cosmetic_categories:
        return "다이소"

    # 2-2. 일반 제품은 제품명에서 첫 단어 추출
    try:
        product_title_elem = driver.find_element(By.CSS_SELECTOR, "h1.product-title")
        product_title = product_title_elem.text.strip()
        if product_title:
            # 첫 번째 띄어쓰기 전까지
            first_word = product_title.split()[0] if product_title.split() else ""
            if first_word:
                return first_word
    except:
        pass

    return ""


def crawl_product_detail(driver, url, category_home, category_1, category_2, crawl_reviews=True, crawl_ingredients=True):
    """제품 상세 정보 크롤링"""
    # URL에서 pdNo를 먼저 추출 (primary key로 사용)
    url_pdno_match = re.search(r"pdNo=([A-Z0-9]+)", url)
    if not url_pdno_match:
        logger.error(f"URL에서 pdNo 추출 실패 - URL: {url}")
        return None, [], []

    url_pdno = url_pdno_match.group(1)
    logger.info(f"제품 크롤링 시작 - pdNo: {url_pdno}")

    product = {
        "product_code": url_pdno,  # URL pdNo를 기본값으로 사용
        "category_home": category_home,
        "category_1": category_1,
        "category_2": category_2,
        "brand": "",
        "name": "",
        "price": "",
        "country": "",
        "likes": 0,
        "shares": 0,
        "url": url,
        "can_할랄인증": "Unknown",  # 할랄 인증 가능 여부
        "can_비건": "Unknown",  # 비건 인증 가능 여부
        "certifications": "",  # 제품 설명에 명시된 인증 정보
    }

    reviews = []
    ingredients = []

    # 페이지 로드
    driver.get(url)
    logger.debug("페이지 로딩 대기 시작")

    # 1단계: 기본 DOM 로딩 대기
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, "product-info-wrap"))
        )
        logger.debug("기본 DOM 로딩 완료")
    except Exception as e:
        logger.warning(f"product-info-wrap 로딩 타임아웃: {str(e)}")

    # 2단계: JavaScript 실행 완료 대기 (더 긴 시간)
    time.sleep(5)

    # 3단계: 스크롤로 Lazy Loading 트리거
    driver.execute_script("window.scrollTo(0, 300);")
    time.sleep(1)
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(1)

    # 4단계: 제품명이 실제로 로드될 때까지 대기
    max_retries = 5
    for retry in range(max_retries):
        try:
            # info-area 내부의 product-title이 텍스트를 가질 때까지 대기
            name_element = WebDriverWait(driver, 3).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".info-area .product-title"))
            )
            name_text = name_element.text.strip()
            if name_text and len(name_text) > 0:
                logger.debug(f"제품명 로딩 완료 (시도 {retry + 1}/{max_retries})")
                break
            else:
                logger.debug(f"제품명 로딩 대기 중 (시도 {retry + 1}/{max_retries})")
                time.sleep(2)
        except Exception as e:
            logger.debug(f"제품명 대기 중 예외 (시도 {retry + 1}/{max_retries}): {str(e)}")
            time.sleep(2)
    else:
        logger.warning(f"제품명 로딩 타임아웃 - pdNo: {url_pdno}")

    # 실제 로드된 URL 확인 (리다이렉트 체크)
    current_url = driver.current_url
    current_pdno_match = re.search(r"pdNo=([A-Z0-9]+)", current_url)

    if current_pdno_match:
        current_pdno = current_pdno_match.group(1)
        if current_pdno != url_pdno:
            logger.warning(f"URL 리다이렉트 감지!")
            logger.warning(f"   요청 pdNo: {url_pdno} → 실제 pdNo: {current_pdno}")
            logger.warning(f"   이 제품은 중복 방지 로직에 의해 스킵될 수 있습니다")
            # 실제 로드된 pdNo로 업데이트
            product["product_code"] = current_pdno
            url_pdno = current_pdno
        else:
            logger.debug(f"URL 일치: {url_pdno}")
    else:
        logger.warning(f"현재 URL에서 pdNo 추출 불가: {current_url}")

    try:
        # 카테고리 breadcrumb
        crumbs = driver.find_elements(By.CSS_SELECTOR, ".el-breadcrumb__inner.is-link")
        texts = [c.text.strip() for c in crumbs if c.text.strip()]
    except:
        pass

    # 브랜드
    product["brand"] = extract_brand(driver, category_2)
    logger.debug(f"브랜드: {product['brand']}")

    # 제품명 추출 - h1.product-title 선택자 사용 (추천 제품 제외)
    # 핵심: h1 태그만 선택 (추천 제품은 div.product-title이므로 제외됨)
    name_selectors = [
        "h1.product-title",  # 메인 제품명 (h1 태그)
        ".info-area h1",  # 폴백 1
        ".product-info-wrap h1",  # 폴백 2
    ]

    for selector in name_selectors:
        try:
            element = driver.find_element(By.CSS_SELECTOR, selector)
            product["name"] = element.text.strip()
            if product["name"]:
                logger.info(f"제품명 추출 성공 ({selector}): {product['name'][:50]}")
                break
        except Exception as e:
            logger.debug(f"제품명 추출 실패 ({selector}): {str(e)}")
            continue

    # 3) 옵션 정보 추가 (있는 경우)
    if product["name"]:
        try:
            option_element = driver.find_element(By.CSS_SELECTOR, ".product-option-text, .option-text, .selected-option")
            option_text = option_element.text.strip()
            if option_text and option_text not in product["name"]:
                product["name"] = f"{product['name']} ({option_text})"
                logger.info(f"옵션 정보 추가: {option_text}")
        except:
            logger.debug("옵션 정보 없음")

    # 4) 최종 검증
    if not product["name"]:
        logger.error(f"제품명 추출 완전 실패 - pdNo: {url_pdno}")

        # 디버깅: HTML 구조 저장 (첫 번째 실패만)
        if not hasattr(crawl_product_detail, '_name_fail_saved'):
            try:
                os.makedirs('logs/name_fail', exist_ok=True)

                # HTML 저장
                html_file = f"logs/name_fail/fail_{url_pdno}.html"
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(driver.page_source)

                # 스크린샷 저장
                screenshot_file = f"logs/name_fail/fail_{url_pdno}.png"
                driver.save_screenshot(screenshot_file)

                # product-title 요소들 모두 출력
                all_titles = driver.find_elements(By.CLASS_NAME, "product-title")
                logger.info(f"페이지 내 product-title 요소 수: {len(all_titles)}")
                for idx, title in enumerate(all_titles):
                    logger.info(f"  [{idx}] {title.text[:50]}...")

                logger.info(f"디버깅 파일 저장: {html_file}")
                logger.info(f"스크린샷 저장: {screenshot_file}")
                crawl_product_detail._name_fail_saved = True
            except Exception as e:
                logger.error(f"디버깅 파일 저장 실패: {e}")

    # 가격 - 상세 페이지의 가격만 선택 (추천 제품 제외)
    try:
        # 1순위: .prod-price--detail 내부의 가격
        price_element = driver.find_element(By.CSS_SELECTOR, ".prod-price--detail .price-value .value")
        product["price"] = price_element.text.strip().replace(",", "")
    except:
        try:
            # 2순위: .inner-box 내부의 첫 번째 가격
            price_element = driver.find_element(By.CSS_SELECTOR, ".inner-box .price-value .value")
            product["price"] = price_element.text.strip().replace(",", "")
        except:
            pass

    # 페이지 품번 확인 (검증용)
    page_product_code = None
    try:
        code_text = driver.find_element(By.CLASS_NAME, "code-text").text
        match = re.search(r"품번\s*(\d+)", code_text)
        if match:
            page_product_code = match.group(1)
            logger.debug(f"페이지 품번: {page_product_code}")

            # URL pdNo와 비교
            if page_product_code != url_pdno:
                logger.warning(f"품번 불일치 감지!")
                logger.warning(f"   URL pdNo: {url_pdno}")
                logger.warning(f"   페이지 품번: {page_product_code}")
                logger.warning(f"   제품명: {product['name']}")

                # 디버깅: 첫 번째 불일치 케이스만 HTML 저장
                if not hasattr(crawl_product_detail, '_mismatch_saved'):
                    try:
                        os.makedirs('logs/mismatch', exist_ok=True)

                        # HTML 저장
                        html_file = f"logs/mismatch/mismatch_{url_pdno}_vs_{page_product_code}.html"
                        with open(html_file, 'w', encoding='utf-8') as f:
                            f.write(driver.page_source)

                        # 스크린샷 저장
                        screenshot_file = f"logs/mismatch/mismatch_{url_pdno}_vs_{page_product_code}.png"
                        driver.save_screenshot(screenshot_file)

                        logger.info(f"디버깅 파일 저장: {html_file}")
                        logger.info(f"스크린샷 저장: {screenshot_file}")
                        crawl_product_detail._mismatch_saved = True
                    except Exception as e:
                        logger.error(f"디버깅 파일 저장 실패: {e}")

                # 페이지 품번을 제품 코드로 사용 2
                logger.warning(f"   → 페이지 품번({page_product_code})을 제품 코드로 사용합니다")
                product["product_code"] = page_product_code
            else:
                logger.debug(f"품번 일치: {url_pdno}")
        else:
            logger.debug(f"code-text 있지만 품번 패턴 없음: {code_text}")
    except Exception as e:
        logger.debug(f"페이지 품번 추출 실패 (URL pdNo 사용): {str(e)}")

    # 제조국
    try:
        product["country"] = driver.find_element(
            By.XPATH, "//th[contains(text(),'제조국')]/following-sibling::td"
        ).text.strip()
    except:
        pass

    # 좋아요/공유
    try:
        counts = driver.find_elements(By.CLASS_NAME, "btn__count")
        if len(counts) >= 2:
            product["likes"] = parse_count(counts[0].text)
            product["shares"] = parse_count(counts[1].text)
    except:
        pass

    # 제품 코드 검증
    if not product["product_code"]:
        logger.error(f"제품 코드 없음 - 스킵: {product['name']}")
        return None, [], []

    # 최종 제품 정보 로그
    logger.info(f"제품 정보 수집 완료:")
    logger.info(f"   - 제품 코드: {product['product_code']}")
    logger.info(f"   - 제품명: {product['name'][:50]}...")
    logger.info(f"   - 브랜드: {product['brand']}")
    logger.info(f"   - 가격: {product['price']}원")

    # 5천원 초과 제품 제외
    try:
        if product["price"] and int(product["price"]) > 5000:
            logger.info(f"제외 (가격 초과): {product['name']} | {product['price']}원")
            return None, [], []
    except:
        pass

    # 리뷰 크롤링
    if crawl_reviews:
        logger.info(f"리뷰 수집 시작: {product['name']}")
        for page in range(1, 999):
            time.sleep(1)
            review_elements = driver.find_elements(By.CLASS_NAME, "review-detail")
            logger.debug(f"{page}페이지 리뷰 수: {len(review_elements)}")

            for r in review_elements:
                try:
                    date = r.find_element(By.CLASS_NAME, "cw-bar-list").text.split()[0]
                    user_raw = r.find_element(By.CLASS_NAME, "con-writer-id").text.strip()
                    rating_raw = r.find_element(By.CLASS_NAME, "hiddenText").text.strip()
                    text = r.find_element(By.CSS_SELECTOR, ".review-desc .cont").text.strip()
                    image_count = len(r.find_elements(By.CSS_SELECTOR, ".swiper-wrapper img"))
                    rating = extract_rating(rating_raw)
                    user_id = user_id_map[user_raw]

                    reviews.append({
                        "product_code": product["product_code"],
                        "date": date,
                        "user_masked": user_raw,
                        "user": user_id,
                        "rating": rating,
                        "text": text,
                        "image_count": image_count,
                    })
                except:
                    continue

            # 다음 페이지
            try:
                next_btn = driver.find_element(By.CLASS_NAME, "btn-next")
                next_class = next_btn.get_attribute("class") or ""
                next_style = next_btn.get_attribute("style") or ""

                if ("disabled" in next_class) or (not next_btn.is_enabled()) or ("pointer-events: none" in next_style):
                    logger.debug("마지막 페이지 도달")
                    break

                driver.execute_script("arguments[0].click();", next_btn)
                time.sleep(1)
            except:
                logger.debug("다음 버튼 없음")
                break

    # 성분 크롤링
    if crawl_ingredients:
        logger.info(f"성분 수집 시작 (다중 소스): pdNo: {product['product_code']} | 제품명: {product['name'][:40]}")

        # 할랄/비건 인증 정보 추출
        certifications = []
        try:
            # 제품 설명에서 할랄/비건 키워드 검색
            editor_content = driver.find_element(By.CSS_SELECTOR, "div.editor-content")
            content_text = editor_content.text.lower()

            if '할랄' in content_text or 'halal' in content_text:
                certifications.append('할랄')
                logger.info("할랄 인증 제품 발견")

            if '비건' in content_text or 'vegan' in content_text:
                certifications.append('비건')
                logger.info("비건 인증 제품 발견")

            if certifications:
                product['certifications'] = ', '.join(certifications)
        except:
            pass

        try:
            # 새로운 다중 소스 성분 추출 함수 사용
            ingredients = extract_ingredients_multi_source(
                driver,
                product['product_code'],
                product['name']
            )

            logger.info(f"성분 추출 완료: {len(ingredients)}개 (다중 소스 검증)")

            # 할랄/비건 인증 가능 여부 판정
            if ingredients:
                # 비건 판정: 동물성 성분이 하나라도 있으면 No
                non_vegan_ingredients = [ing for ing in ingredients if ing.get('is_vegan') == 'No']
                if non_vegan_ingredients:
                    product['can_비건'] = 'No'
                    logger.info(f"비건 부적합: {len(non_vegan_ingredients)}개 동물성 성분 발견")
                elif any(ing.get('is_vegan') == 'Unknown' for ing in ingredients):
                    product['can_비건'] = 'Unknown'
                else:
                    product['can_비건'] = 'Yes'
                    logger.info("비건 인증 가능")

                # 할랄 판정: 부적합 성분이 하나라도 있으면 No
                haram_ingredients = [ing for ing in ingredients if ing.get('is_halal') == 'No']
                questionable_ingredients = [ing for ing in ingredients if ing.get('is_halal') == 'Questionable']

                if haram_ingredients:
                    product['can_할랄인증'] = 'No'
                    logger.info(f"할랄 부적합: {len(haram_ingredients)}개 부적합 성분 발견")
                elif questionable_ingredients:
                    product['can_할랄인증'] = 'Questionable'
                    logger.info(f"할랄 원료확인 필요: {len(questionable_ingredients)}개 의심 성분")
                elif any(ing.get('is_halal') == 'Unknown' for ing in ingredients):
                    product['can_할랄인증'] = 'Unknown'
                else:
                    product['can_할랄인증'] = 'Yes'
                    logger.info("할랄 인증 가능")

        except Exception as e:
            logger.error(f"다중 소스 성분 추출 오류: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

            # 폴백: 간단한 성분 추출 로직 (실패 시 빈 리스트)
            logger.warning("다중 소스 추출 실패 - 성분 정보 없음")
            ingredients = []

        # 성분 추출 실패 시 로그
        if not ingredients:
            logger.warning(f"성분 정보 추출 실패: {product['name']}")

    return product, reviews, ingredients


def select_categories():
    """중분류/소분류 선택"""
    print("\n" + "="*60)
    print("다이소 뷰티/위생 카테고리 크롤러")
    print("="*60)

    # 중분류 선택
    print("\n[중분류 선택]")
    middle_categories = list(DAISO_BEAUTY_CATEGORIES.keys())
    for idx, cat in enumerate(middle_categories, 1):
        print(f"{idx}. {cat}")
    print("0. 전체")

    choice = input("\n선택 (번호 입력): ").strip()

    if choice == "0":
        selected_middle = middle_categories
    else:
        try:
            selected_middle = [middle_categories[int(choice) - 1]]
        except:
            print("잘못된 선택입니다.")
            return None

    # 소분류 선택
    selected_categories = []
    for middle in selected_middle:
        middle_code = DAISO_BEAUTY_CATEGORIES[middle]["중분류코드"]
        small_categories = DAISO_BEAUTY_CATEGORIES[middle]["소분류"]

        print(f"\n[{middle} - 소분류 선택]")
        small_list = list(small_categories.items())
        for idx, (code, name) in enumerate(small_list, 1):
            print(f"{idx}. {name}")
        print("0. 전체")

        choice = input("\n선택 (번호 입력, 여러 개는 쉼표로 구분): ").strip()

        if choice == "0":
            for code, name in small_list:
                selected_categories.append((middle, middle_code, code, name))
        else:
            try:
                indices = [int(x.strip()) for x in choice.split(",")]
                for idx in indices:
                    code, name = small_list[idx - 1]
                    selected_categories.append((middle, middle_code, code, name))
            except:
                print("잘못된 선택입니다.")
                continue

    return selected_categories


def select_crawl_targets():
    """크롤링 대상 선택"""
    print("\n[크롤링 대상 선택]")
    print("1. 제품 정보만")
    print("2. 제품 정보 + 리뷰")
    print("3. 제품 코드 + 성분만")
    print("4. 제품 코드 + 리뷰만")
    print("5. 전체 (제품 정보 + 리뷰 + 성분)")

    choice = input("\n선택 (번호 입력): ").strip()

    # (제품정보저장, 리뷰수집, 성분수집, 최소제품정보모드)
    targets = {
        "1": (True, False, False, False),   # 제품 정보만
        "2": (True, True, False, False),    # 제품 정보 + 리뷰
        "3": (False, False, True, True),    # 제품 코드 + 성분만
        "4": (False, True, False, True),    # 제품 코드 + 리뷰만 (NEW!)
        "5": (True, True, True, False),     # 전체
    }

    return targets.get(choice, (True, False, False, False))


def main():
    """메인 함수"""
    # 카테고리 선택
    categories = select_categories()
    if not categories:
        return

    # 크롤링 대상 선택
    crawl_products, crawl_reviews, crawl_ingredients, minimal_mode = select_crawl_targets()

    print(f"\n{'='*60}")
    print(f"선택된 카테고리: {len(categories)}개")
    if minimal_mode:
        if crawl_reviews and not crawl_ingredients:
            print(f"크롤링 대상: 제품 코드 + 리뷰")
        elif crawl_ingredients and not crawl_reviews:
            print(f"크롤링 대상: 제품 코드 + 성분")
        else:
            print(f"크롤링 대상: 최소 모드 (제품 코드만)")
    else:
        print(f"크롤링 대상: 제품={crawl_products}, 리뷰={crawl_reviews}, 성분={crawl_ingredients}")
    print(f"{'='*60}")

    confirm = input("\n시작하시겠습니까? (y/n): ").strip().lower()
    if confirm != 'y':
        print("취소되었습니다.")
        return

    # 크롤링 시작
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    all_products = []
    all_reviews = []
    all_ingredients = []

    # 중복 방지용: 이미 크롤링한 product_code 추적
    seen_product_codes = set()

    try:
        for middle, middle_code, small_code, small_name in categories:
            logger.info(f"{'='*60}")
            logger.info(f"카테고리: {middle} > {small_name}")
            logger.info(f"{'='*60}")

            # 카테고리 URL
            category_url = get_category_url(middle_code, small_code)

            # 제품 링크 수집
            links = get_all_product_links(driver, category_url, small_name)

            # 각 제품 크롤링
            for idx, link in enumerate(links, 1):
                try:
                    # URL에서 pdNo 미리 추출 (로깅용)
                    pdno_match = re.search(r"pdNo=([A-Z0-9]+)", link)
                    pdno_preview = pdno_match.group(1) if pdno_match else "알수없음"
                    logger.info(f"\n{'='*60}")
                    logger.info(f"[{idx}/{len(links)}] 제품 크롤링 - pdNo: {pdno_preview}")
                    logger.info(f"{'='*60}")

                    product, reviews, ingredients = crawl_product_detail(
                        driver, link,
                        category_home="뷰티/위생",
                        category_1=middle,
                        category_2=small_name,
                        crawl_reviews=crawl_reviews,
                        crawl_ingredients=crawl_ingredients
                    )

                    if product:
                        # 중복 체크
                        if product["product_code"] in seen_product_codes:
                            logger.warning(f"중복 제품 감지 - 스킵: product_code={product['product_code']}, 제품명={product['name'][:40]}")
                            logger.warning(f"   요청 pdNo: {pdno_preview} → 실제 product_code: {product['product_code']}")
                            continue

                        # 새로운 제품이면 추가
                        seen_product_codes.add(product["product_code"])

                        # minimal_mode일 때는 제품 정보 저장 안 함
                        if not minimal_mode:
                            all_products.append(product)
                        all_reviews.extend(reviews)
                        all_ingredients.extend(ingredients)

                        if minimal_mode:
                            if crawl_reviews and not crawl_ingredients:
                                logger.info(f"제품 코드 + 리뷰 크롤링 완료: [{product['name'][:40]}] | 리뷰: {len(reviews)}개")
                            elif crawl_ingredients and not crawl_reviews:
                                logger.info(f"제품 코드 + 성분 크롤링 완료: [{product['name'][:40]}] | 성분: {len(ingredients)}개")
                            else:
                                logger.info(f"제품 코드 크롤링 완료: [{product['name'][:40]}] | 리뷰: {len(reviews)}개 | 성분: {len(ingredients)}개")
                        else:
                            logger.info(f"제품 정보 + 리뷰 + 성분 크롤링 완료: [{product['name'][:40]}] | 리뷰: {len(reviews)}개 | 성분: {len(ingredients)}개")

                    time.sleep(1)

                except Exception as e:
                    logger.error(f"크롤링 실패: {link}")
                    logger.error(str(e))
                    continue

        # CSV 저장 - 대분류별로 분리
        date_str = get_date_string()

        # 선택된 카테고리에서 대분류별 소분류 정보 추출
        category_info = {}  # {대분류: set(소분류들)}
        for middle, middle_code, small_code, small_name in categories:
            if middle not in category_info:
                category_info[middle] = set()
            category_info[middle].add(small_name)

        # data 디렉토리 생성
        os.makedirs('data', exist_ok=True)

        if all_products and not minimal_mode:
            # 대분류별로 제품 그룹화
            df_products = pd.DataFrame(all_products)
            for category1, small_categories in category_info.items():
                category_products = df_products[df_products['category_1'] == category1]
                if len(category_products) > 0:
                    # 소분류가 1개면 소분류명, 여러 개면 'all'
                    sub_category = list(small_categories)[0] if len(small_categories) == 1 else 'all'
                    # 파일명에 사용할 수 없는 특수문자 제거
                    sub_category_safe = sub_category.replace('/', '_').replace(':', '_').replace('\\', '_')
                    product_file = f'data/product_{category1}_{sub_category_safe}_{date_str}.csv'
                    category_products.to_csv(product_file, index=False, encoding='utf-8-sig')
                    logger.info(f"제품 정보 저장 완료: {product_file} ({len(category_products)}개)")
                    print(f"\n제품 정보: {product_file} ({len(category_products)}개)")

        if all_reviews:
            # 리뷰는 대분류별로 분리 (product_code로 매핑)
            df_reviews = pd.DataFrame(all_reviews)
            if not minimal_mode and all_products:
                df_products = pd.DataFrame(all_products)
                product_category_map = dict(zip(df_products['product_code'], df_products['category_1']))

                df_reviews['category_1'] = df_reviews['product_code'].map(product_category_map)

                for category1, small_categories in category_info.items():
                    category_reviews = df_reviews[df_reviews['category_1'] == category1]
                    if len(category_reviews) > 0:
                        sub_category = list(small_categories)[0] if len(small_categories) == 1 else 'all'
                        # 파일명에 사용할 수 없는 특수문자 제거
                        sub_category_safe = sub_category.replace('/', '_').replace(':', '_').replace('\\', '_')
                        review_file = f'data/reviews_{category1}_{sub_category_safe}_{date_str}.csv'
                        category_reviews.drop('category_1', axis=1).to_csv(review_file, index=False, encoding='utf-8-sig')
                        logger.info(f"리뷰 저장 완료: {review_file} ({len(category_reviews)}개)")
                        print(f"리뷰: {review_file} ({len(category_reviews)}개)")
            else:
                # minimal_mode거나 제품 정보가 없으면 all로 저장
                review_file = f'data/reviews_all_{date_str}.csv'
                df_reviews.to_csv(review_file, index=False, encoding='utf-8-sig')
                logger.info(f"리뷰 저장 완료: {review_file} ({len(df_reviews)}개)")
                print(f"리뷰: {review_file} ({len(df_reviews)}개)")

        if all_ingredients:
            # 성분도 대분류별로 분리 (product_code로 매핑)
            df_ingredients = pd.DataFrame(all_ingredients)

            # minimal_mode에서는 제품 정보를 수집하지 않으므로,
            # 크롤링한 제품의 category_1을 따로 추적해야 함
            # 간단하게 처리: all_products가 있으면 매핑, 없으면 카테고리 정보에서 추론
            if not minimal_mode and all_products:
                df_products = pd.DataFrame(all_products)
                product_category_map = dict(zip(df_products['product_code'], df_products['category_1']))
            else:
                # minimal_mode: 크롤링한 순서대로 카테고리 매핑
                # (이 경우 product_code와 category를 직접 매핑하기 어려우므로 카테고리 정보 활용)
                product_category_map = {}

            df_ingredients['category_1'] = df_ingredients['product_code'].map(product_category_map)

            for category1, small_categories in category_info.items():
                category_ingredients = df_ingredients[df_ingredients['category_1'] == category1]
                if len(category_ingredients) > 0:
                    sub_category = list(small_categories)[0] if len(small_categories) == 1 else 'all'
                    # 파일명에 사용할 수 없는 특수문자 제거
                    sub_category_safe = sub_category.replace('/', '_').replace(':', '_').replace('\\', '_')
                    ingredient_file = f'data/ingredients_{category1}_{sub_category_safe}_{date_str}.csv'
                    category_ingredients.drop('category_1', axis=1).to_csv(ingredient_file, index=False, encoding='utf-8-sig')
                    logger.info(f"성분 저장 완료: {ingredient_file} ({len(category_ingredients)}개)")
                    print(f"성분: {ingredient_file} ({len(category_ingredients)}개)")

            # 카테고리 매핑이 안 된 성분이 있으면 별도 저장
            unmapped = df_ingredients[df_ingredients['category_1'].isna()]
            if len(unmapped) > 0:
                # minimal_mode이고 카테고리가 1개면 그 카테고리로 저장
                if len(category_info) == 1:
                    category1 = list(category_info.keys())[0]
                    small_categories = category_info[category1]
                    sub_category = list(small_categories)[0] if len(small_categories) == 1 else 'all'
                    # 파일명에 사용할 수 없는 특수문자 제거
                    sub_category_safe = sub_category.replace('/', '_').replace(':', '_').replace('\\', '_')
                    ingredient_file = f'data/ingredients_{category1}_{sub_category_safe}_{date_str}.csv'
                else:
                    ingredient_file = f'data/ingredients_unmapped_{date_str}.csv'
                unmapped.drop('category_1', axis=1, errors='ignore').to_csv(ingredient_file, index=False, encoding='utf-8-sig')
                logger.info(f"매핑 안 된 성분 저장 완료: {ingredient_file} ({len(unmapped)}개)")
                print(f"성분: {ingredient_file} ({len(unmapped)}개)")

        print(f"\n{'='*60}")
        print("크롤링 완료")
        print(f"{'='*60}")

    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

    finally:
        driver.quit()
        logger.info("브라우저 종료 완료")


if __name__ == "__main__":
    main()
