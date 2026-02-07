"""
할랄/비건 성분 검증 모듈 (로컬 데이터베이스)
"""

# 동물성 성분 데이터베이스 (비건/할랄 부적합)
ANIMAL_DERIVED_INGREDIENTS = {
    # 확실한 동물 유래 성분 (비건 불가)
    '라놀린', 'Lanolin', '양모지',
    '콜라겐', 'Collagen', '가수분해콜라겐', '마린콜라겐', '어류콜라겐',
    '케라틴', 'Keratin', '가수분해케라틴',
    '밀크프로틴', 'MilkProtein', '우유단백질',
    '실크프로틴', 'SilkProtein', '실크아미노산',
    '꿀', 'Honey', '벌꿀', '아카시아꿀',
    '밀랍', 'Beeswax', '비즈왁스', '밀납',
    '프로폴리스', 'Propolis',
    '로열젤리', 'RoyalJelly', '로얄젤리',
    '카민', '코치닐', 'Carmine', 'Cochineal',
    '캐비어', 'Caviar', '캐비어추출물',
    '펄', 'Pearl', '펄파우더', '진주가루',
    '젤라틴', 'Gelatin', '돼지젤라틴', '소젤라틴', '어류젤라틴',
    '카제인', 'Casein',
    '키토산', 'Chitosan',
    '콘드로이틴', 'Chondroitin',
    '엘라스틴', 'Elastin',
    '스쿠알렌', 'Squalene',  # 상어 유래 (식물성도 있음)
    '뮤스크', 'Musk', '사향',
    '앰버그리스', 'Ambergris',
    '타우린', 'Taurine',
    '알부민', 'Albumin',
    '레티놀', 'Retinol',  # 동물성 비타민A
    '오메가3', '피쉬오일', 'FishOil',
}

# 할랄 부적합 성분 (알코올, 돼지 유래 등)
HARAM_INGREDIENTS = {
    # 알코올류 (할랄 금지)
    '에탄올', 'Ethanol',
    '알코올', 'Alcohol',
    '에틸알코올', 'EthylAlcohol',
    '알코올변성', '변성알코올', 'AlcoholDenat', 'DenaturatedAlcohol',
    '이소프로필알코올', 'IsopropylAlcohol', 'IPA',
    'SD알코올', 'SDAlcohol', 'SD알코올40', 'SD알코올40-B',
    '벤질알코올', 'BenzylAlcohol',
    '페닐에틸알코올', 'Phenylethylalcohol',
    '메탄올', 'Methanol',
    '부탄올', 'Butanol',

    # 돼지 유래 성분 (할랄 절대 금지)
    '돼지콜라겐', 'PorkCollagen', '포크콜라겐',
    '돼지젤라틴', 'PorkGelatin', '포크젤라틴',
    '포신', 'Placenta', '돼지태반', '태반추출물',
    '돼지췌장추출물', '팬크레아틴',
    '돼지엘라스틴',
    '라드', 'Lard', '돼지기름',

    # 주류 유래
    '와인추출물', 'WineExtract',
    '맥주효모', 'BeerYeast', '비어효모',
    '사케추출물', '막걸리효모',
    '위스키추출물', '보드카추출물',

    # 기타 의심 성분
    '젤라틴',  # 출처 불명 시 돼지 가능성
}

# 애매한 성분 (식물성/동물성 혼재 - 원료 확인 필요)
# 이 성분들은 원료 출처에 따라 비건/할랄 여부가 달라질 수 있음
AMBIGUOUS_INGREDIENTS = {
    # === 원료 출처 확인 필요 (식물성/동물성 혼재) ===
    '글리세린', 'Glycerin', 'Glycerol',
    '부틸렌글라이콜', 'ButyleneGlycol', '1,3-부틸렌글라이콜',
    '세테아릴알코올', 'CetearylAlcohol',
    '에틸헥실글리세린', 'EthylhexylGlycerin',
    '토코페롤', 'Tocopherol', '비타민E',
    '펜틸렌글라이콜', 'PentyleneGlycol', '1,2-펜탄디올',
    '프로판다이올', 'Propanediol', '1,3-프로판다이올',
    '하이드로제네이티드레시틴', 'HydrogenatedLecithin', '수소첨가레시틴',
    '향료', 'Fragrance', 'Parfum',

    # 지방 유래 (식물성/동물성 혼재)
    '스테아르산', 'StearicAcid', '스테아릭애시드', '스테아릭산',
    '팔미트산', '팔미틱산', '팔미틱애시드', 'PalmiticAcid',
    '올레산', '올레익애시드', 'OleicAcid',
    '리놀레산', 'LinoleicAcid',
    '미리스트산', '미리스틱애시드', 'MyristicAcid',
    '라우르산', '라우릭애시드', 'LauricAcid',
    '카프릴산', '카프릴릭애시드',
    '세틸알코올', 'CetylAlcohol',  # 지방 알코올 (비취성)
    '스테아릴알코올', 'StearylAlcohol',
    '베헤닐알코올', 'BehenylAlcohol',

    # 스쿠알란/스쿠알렌
    '스쿠알란', 'Squalane',
    '스쿠알렌', 'Squalene',

    # 레시틴 계열
    '레시틴', 'Lecithin',

    # 세라마이드 계열
    '세라마이드', 'Ceramide',
    '세라마이드NP', '세라마이드AP', '세라마이드EOP',

    # 히알루론산 계열
    '히알루론산', 'HyaluronicAcid',
    '히알루론산나트륨', 'SodiumHyaluronate',
    '가수분해히알루론산', 'HydrolyzedHyaluronicAcid',
    '소듐하이알루로네이트', '하이알루로닉애시드',
    '하이드롤라이즈드하이알루로닉애시드',
    '소듐하이알루로네이트크로스폴리머',
    '소듐아세틸레이티드하이알루로네이트',

    # 콜레스테롤 계열
    '콜레스테롤', 'Cholesterol',

    # 비타민
    '레티놀', 'Retinol',
    '판테놀', 'Panthenol',

    # 아미노산 (발효/동물성)
    '알란토인', 'Allantoin',

    # 기타
    '구아닌', 'Guanine',
    '키틴', 'Chitin',
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
    '쇠비름추출물', '캐모마일꽃추출물', '체이스트트리추출물',

    # 식물성 오일
    '호호바오일', 'JojobaOil', '시어버터', 'SheaButter',
    '코코넛오일', 'CoconutOil', '올리브오일', 'OliveOil',
    '아보카도오일', 'AvocadoOil', '해바라기씨오일', 'SunflowerSeedOil',
    '아르간오일', 'ArganOil', '아몬드오일', 'AlmondOil',
    '포도씨오일', 'GrapeSeedOil', '동백오일', 'CamelliaOil',
    '로즈힙오일', 'RosehipOil', '마카다미아오일', 'MacadamiaOil',

    # 합성 성분 (비동물성 확정)
    '정제수', 'Water', 'Aqua',
    '디프로필렌글라이콜', 'DipropyleneGlycol',
    '1,2-헥산다이올', '2-헥산다이올',
    '페녹시에탄올', 'Phenoxyethanol',
    '소듐폴리아크릴레이트', 'SodiumPolyacrylate',
    '잔탄검', 'XanthanGum',
    '카보머', 'Carbomer',
    '티타늄디옥사이드', 'TitaniumDioxide',
    '징크옥사이드', 'ZincOxide',
    '나이아신아마이드', 'Niacinamide',
    '벤질글라이콜', 'BenzylGlycol',
    '아데노신', 'Adenosine',
    '트로메타민', 'Tromethamine',

    # 실리콘류
    '다이메티콘', 'Dimethicone',
    '사이클로헥사실록세인', 'Cyclohexasiloxane',
    '메틸트라이메티콘',
    '카프릴릴메티콘',

    # UV 필터
    '호모살레이트', 'Homosalate',
    '에틸헥실메톡시신나메이트',

    # 유화제/계면활성제
    '글리세릴스테아레이트',
    '글리세릴카프릴레이트',
    '세틸에틸헥사노에이트',
    '솔비탄아이소스테아레이트',
    '폴리글리세릴-30메틸글루코오스다이스테아레이트',

    # 증점제
    '아크릴레이트/C10-30',
    '알루미늄스테아레이트',

    # 산화철
    '마이카', 'Mica', '운모', '산화철', 'IronOxide',
    '황색산화철', '적색산화철', '흑색산화철',
    '알루미나', '알루미늄하이드록사이드',
    '실리카', 'Silica',
    '보론나이트라이드',

    # 기타 합성 성분
    '소듐클로라이드', 'SodiumChloride',
    '소듐시트레이트',
    '다이소듐이디티에이',
    '프로필렌카보네이트',
}


def check_halal_vegan_status(ingredient: str) -> dict:
    """
    성분의 할랄/비건 적합성 판정

    Args:
        ingredient: 성분명

    Returns:
        dict: {
            'is_vegan': 'Yes'|'No'|'Unknown',
            'is_halal': 'Yes'|'No'|'Questionable'|'Unknown',
            'warning': str
        }
    """
    result = {
        'is_vegan': 'Unknown',
        'is_halal': 'Unknown',
        'warning': ''
    }

    # 1단계: 할랄 부적합 성분 체크 (최우선)
    if ingredient in HARAM_INGREDIENTS:
        result['is_halal'] = 'No'
        result['warning'] = '할랄 부적합 (알코올 또는 돼지 유래)'
        # 알코올은 비건이지만 할랄 부적합
        if '알코올' in ingredient.lower() or 'alcohol' in ingredient.lower():
            result['is_vegan'] = 'Yes'
        elif '돼지' in ingredient or 'pork' in ingredient.lower():
            result['is_vegan'] = 'No'
        return result

    # 2단계: 확실한 동물성 성분 체크
    if ingredient in ANIMAL_DERIVED_INGREDIENTS:
        result['is_vegan'] = 'No'
        result['is_halal'] = 'Questionable'
        result['warning'] = '동물성 유래 성분'
        return result

    # 3단계: 비건 안전 성분 체크
    if ingredient in VEGAN_SAFE_INGREDIENTS:
        result['is_vegan'] = 'Yes'
        result['is_halal'] = 'Yes'
        return result

    # 4단계: 애매한 성분 (원료 출처에 따라 달라짐)
    if ingredient in AMBIGUOUS_INGREDIENTS:
        result['is_vegan'] = 'Unknown'
        result['is_halal'] = 'Unknown'
        result['warning'] = '원료 출처 확인 필요 (식물성/동물성 혼재 가능)'
        return result

    # 5단계: 알 수 없는 성분 - 기본적으로 합성/식물성으로 간주
    result['is_vegan'] = 'Yes'
    result['is_halal'] = 'Yes'
    result['warning'] = '합성/식물성 추정 (검증 필요)'

    return result
