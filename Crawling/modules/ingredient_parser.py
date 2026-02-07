"""
성분명 파싱 및 유효성 검증 모듈
"""
import re

# OCR 오인식 패턴 및 수정 맵 (실제 크롤링 데이터 기반)
OCR_CORRECTIONS = {
    # === 소듐(Sodium) 변형 ===
    '소톱': '소듐',
    '소득': '소듐',
    '소문': '소듐',
    '소둥': '소듐',
    '소듬': '소듐',
    '소등': '소듐',  # OCR에서 자주 발생

    # === 포타슘(Potassium) 변형 ===
    '포타습': '포타슘',

    # === 테트라(Tetra) 변형 ===
    '데트라': '테트라',

    # === 헥실(Hexyl)/헥산(Hexane) 변형 ===
    '핵실': '헥실',
    '핵사': '헥사',
    '핵산': '헥산',  # 1,2-헥산다이올 OCR 오류

    # === 펜타(Penta) 변형 ===
    '팬타': '펜타',

    # === 폴리쿼터늄 변형 ===
    '플리퀴터늄': '폴리쿼터늄',
    '플리쿼터늄': '폴리쿼터늄',

    # === 글라이콜/글리세릴 변형 ===
    '글라이골': '글라이콜',
    '글라이클': '글라이콜',
    '글리세렉': '글리세릴',
    '글리세킬': '글리세릴',

    # === 클로라이드 변형 ===
    '글로라이드': '클로라이드',

    # === 부틸렌 변형 ===
    '탤렌': '부틸렌',
    '털런': '부틸렌',

    # === 카프릴릴 변형 ===
    '카프질질': '카프릴릴',

    # === 라우릴/라우릭 변형 ===
    '라우리': '라우릭',
    '라우린': '라우릴',

    # === 설테인 변형 ===
    '설대인': '설테인',

    # === 에틸 변형 ===
    '에팅': '에틸',
    '에칠': '에틸',

    # === 캐스터 변형 ===
    '키스터': '캐스터',

    # === 시트릭 변형 ===
    '시트리': '시트릭',

    # === 엠이에이(MEA) 변형 ===
    '웨이에이': '엠이에이',

    # === 알지닌 변형 ===
    '알지난': '알지닌',

    # === 병풀잎 변형 ===
    '병물렉': '병풀잎',
    '병풀렉': '병풀잎',

    # === 비에이치티(BHT) 변형 ===
    '비에이치다': '비에이치티',

    # === 리모넨/리날룰 변형 ===
    '리모년': '리모넨',
    '리날문': '리날룰',
    '리날률': '리날룰',

    # === 쿠마린 변형 ===
    '구마린': '쿠마린',

    # === 올레핀설포네이트 변형 ===
    '올레민설뜨네이': '올레핀설포네이트',

    # === 에리스리틸 변형 ===
    '에리스리터': '에리스리틸',

    # === 팜(Palm) 변형 ===
    '이터드팔': '이티드팜',
    '이터드팜': '이티드팜',

    # === 애씨드/애시드 변형 (acid) ===
    '애씨드': '애시드',
    '에씨드': '애시드',
    '아씨드': '애시드',

    # === 공백 삽입 오인식 ===
    '다이 메치콘': '다이메티콘',
    '글리세 린': '글리세린',
    '페녹시 에탄올': '페녹시에탄올',
    '디옥사이 드': '디옥사이드',
    '파라 벤': '파라벤',

    # === 기타 오인식 ===
    '메칠': '메틸',
    '부칠': '부틸',
}

# 화학 성분 패턴 (정규표현식) - 컴파일하여 성능 최적화
_INGREDIENT_PATTERN_STRS = [
    r'.+이트$', r'.+올$', r'.+산$', r'.+염$',
    r'.+레이트$', r'.+라이드$', r'.+나이트$',
    r'.+민$', r'.+틴$', r'.+린$',
    r'.+옥사이드$', r'.+추출물$', r'.+오일$',
    r'.+왁스$', r'.+파라벤$', r'.+글라이콜$',
    r'.+다이올$', r'^CI\d+$',
    r'.+펩타이드-?\d+$',  # 펩타이드 계열 (헥사펩타이드-9 등)
    r'.+세스-?\d+$',  # 에톡실레이트 계열 (트라이데세스-10 등)
    r'^C\d+-\d+.+$',  # 탄화수소 계열 (C11-13아이소알케인 등)
]
INGREDIENT_PATTERNS = [re.compile(p) for p in _INGREDIENT_PATTERN_STRS]

# 영문 성분 키워드 (CamelCase 등 복합 패턴용)
ENGLISH_INGREDIENT_KEYWORDS = frozenset([
    'Hydrolyzed', 'Extract', 'Oil', 'Acid', 'Oxide', 'Glycol', 'Alcohol',
    'Sodium', 'Potassium', 'Calcium', 'Magnesium', 'Zinc', 'Titanium',
    'Dimethicone', 'Siloxane', 'Tocopherol', 'Retinol', 'Niacinamide',
    'Hyaluronate', 'Ceramide', 'Peptide', 'Collagen', 'Elastin',
    'Butter', 'Wax', 'Water', 'Aqua', 'Glycerin', 'Panthenol',
])

# 노이즈 패턴 (성분이 아님) - 컴파일하여 성능 최적화
_NOISE_PATTERN_STRS = [
    # 문장/서술형 패턴
    r'.*(사용|바르|피부|효과|개선|미백|주름|보습|수분|영양|진정|바릅니다|발라주세요).*',
    r'.*(합니다|됩니다|입니다|해요|돼요|예요|있다|없다|한다|된다)$',
    r'.*(하는|되는|있는|없는|같은|좋은|나쁜)$',
    r'.*(경우|때문|위해|통해|대해).*',

    # 용량/규격 패턴
    r'^SPF\d+$', r'^PA\+*$', r'^\d+ml$', r'^\d+g$', r'^\d+mg$',
    r'^\d+(\.\d+)?(ml|g|mg|cm|mm|L|kg)$',  # 숫자+단위
    r'^\d+(\.\d+)?cm$',  # 치수
    r'^\d+/\d+.*$',  # 분수형 (500/45 등)

    # 전화번호/숫자 패턴
    r'^\d{2,4}-\d{3,4}-\d{4}$',  # 전화번호
    r'^\d{5,}$',  # 5자리 이상 순수 숫자

    # 조사/접속사
    r'^(을|를|이|가|은|는|도|만|의|에|로|으로|와|과|하고)$',

    # 주의사항 관련 단어 (동사형 어미)
    r'.*(할|된|한|있을|없을|나타날|피해서|자제할|보관할|상담할)$',
    r'.*(증상|가려움|부어오름|상처|이상).*',
]
NOISE_PATTERNS = [re.compile(p) for p in _NOISE_PATTERN_STRS]

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
    '본품', '적당량', '취해', '골고루', '바른다', '식품의약품안전처',
    '공정거래위원회', '소비자', '보상', '품질보증', '고시품목',
    '가려움', '부어오름', '상처', '직사광선', '어린이',
    '전문의', '상담', '피해서', '자제', '드립니다',
]

# 성분이 아닌 불용어
INGREDIENT_STOPWORDS = [
    # 기존 불용어
    '기재하여야', '하는', '등', '하여야하는', '기재·표시',
    '자외선', '차단', '심사', '유무', '개선', '미백', '주름',
    '피부', '보호', '도움', '효능', '효과', '용법', '용량',
    '본품', '적당량', '바른다', '준다', '으로부터', '를',
    '한다', '합니다', '된다', '됩니다', '있다', '없다',
    '보호한다', '개선한다', '도움을준다', '미백에도움',

    # 주의사항 관련 단어
    '가려움증', '부어오름', '상처가', '이상이', '증상이나',
    '직사광선에', '직사광선을', '보관할', '상담할', '자제할',
    '경우에는', '어린이의', '의하여', '제품에', '피해서',
    '부작용이', '나타날', '다르게', '없도록', '보상해',

    # 기관/회사명
    '공정거래위원회', '식품의약안전처', '소비자', '전문의',
    '서울시', '새문안로', '씨보드', '에스테이트',
    'LG생활건강',

    # 제품 정보
    '품질보증기준', '고시품목별', '별도표기', '보상규정에',
    '화장품', '주의사항', '에어로졸', '가스가',

    # 브랜드/제품명
    'HYDRO', 'Schick', '쉐이브',

    # OCR 노이즈
    '드립니다', '드롤라이', '부츠쉬운', '에이프루악댕',

    # OCR 줄바꿈으로 인한 잘못된 분리 조각 (필터링 필요)
    '스페이트',  # ~포스페이트에서 분리된 조각
    '폴리메틸', '폴리실리콘', '폴리에터',  # 폴리~ 계열 앞부분만 분리
    '포타슘세틸포',  # 포타슘세틸포스페이트 앞부분
    '실세스퀴', '옥세인',  # 실세스퀴옥세인 분리
    '크로스폴리',  # 크로스폴리머 분리

    # 복합 성분에서 잘못 분리된 조각들 (단독으로는 무의미)
    '다이소', '듐이디티에이', '이디티에이',  # 다이소듐이디티에이 분리
    '소듐',  # 단독 소듐 (소듐하이알루로네이트 등에서 분리)
    '글리세라이드', '글리세릴',  # 트라이글리세라이드 등에서 분리
    '알루로네이트', '하이알루로네',  # ~하이알루로네이트에서 분리
    '소듐하이알루로네',  # 불완전 추출
    '크로스폴리머', '코폴리머',  # ~크로스폴리머 등에서 분리
    '오일', '폴리머',  # 단독 추출 (해바라기씨오일 등에서 분리)
    '헥사펩', '펩타이드',  # ~펩타이드에서 분리 (단독 제외)
    '히아루론산',  # 잘못된 표기 (하이알루로닉 변형)
    '트라이',  # 트라이~ 분리
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

    # 향료 알레르겐 성분 (EU 26종 전체)
    '리모넨', 'Limonene', '리날룰', 'Linalool',
    '시트로넬올', 'Citronellol', '게라니올', 'Geraniol',
    '시트랄', 'Citral', '유제놀', 'Eugenol', '쿠마린', 'Coumarin',
    '벤질알코올', 'BenzylAlcohol', '벤질벤조에이트', 'BenzylBenzoate',
    '신나밀알코올', 'CinnamylAlcohol', '알파이소메틸이오논', 'AlphaIsomethylIonone',
    '헥실신나밀알데하이드', 'HexylCinnamal', '부틸페닐메틸프로피오날', 'Lilial',
    '아밀신나밀알데하이드', 'AmylCinnamal', '아밀신나밀알코올', 'AmylCinnamylAlcohol',
    '아니스알코올', 'AniseAlcohol', '벤질신나메이트', 'BenzylCinnamate',
    '벤질살리실레이트', 'BenzylSalicylate', '신나밀알데하이드', 'Cinnamaldehyde',
    '파르네솔', 'Farnesol', '하이드록시시트로넬랄', 'Hydroxycitronellal',
    '이소유제놀', 'Isoeugenol', '메틸2옥티노에이트', 'Methyl2Octynoate',
    '오크모스추출물', 'EverniaPrunastriExtract',
    '트리모스추출물', 'EverniaFurfuraceaExtract',

    # 기능성 성분 - 미백
    '알란토인', 'Allantoin', '나이아신아마이드', 'Niacinamide',
    '아르부틴', 'Arbutin', '알파아르부틴', 'AlphaArbutin',
    '트라넥삼산', 'TranexamicAcid', '비타민C', 'VitaminC',
    '아스코르브산', 'AscorbicAcid', '에틸아스코르빌에테르', 'EthylAscorbylEther',
    '아스코르빌글루코사이드', 'AscorbylGlucoside',
    '마그네슘아스코르빌포스페이트', 'MagnesiumAscorbylPhosphate',
    '소듐아스코르빌포스페이트', 'SodiumAscorbylPhosphate',

    # 기능성 성분 - 주름개선
    '아데노신', 'Adenosine', '레티놀', 'Retinol',
    '레티닐팔미테이트', 'RetinylPalmitate', '레티날', 'Retinal',
    '바쿠치올', 'Bakuchiol',  # 펩타이드는 너무 일반적이므로 개별 펩타이드만 등록
    '아세틸헥사펩타이드', 'AcetylHexapeptide',
    '팔미토일펜타펩타이드', 'PalmitoylPentapeptide',
    '콜라겐', 'Collagen', '엘라스틴', 'Elastin',

    # 기능성 성분 - 보습/진정
    '판테놀', 'Panthenol', '덱스판테놀', 'Dexpanthenol',
    '베타인', 'Betaine', '우레아', 'Urea',
    '세라마이드', 'Ceramide', '세라마이드NP', 'CeramideNP',
    '세라마이드AP', 'CeramideAP', '세라마이드EOP', 'CeramideEOP',
    '스핑고신', 'Sphingosine', '피토스핑고신', 'Phytosphingosine',
    '마데카소사이드', 'Madecassoside', '아시아티코사이드', 'Asiaticoside',
    '비사보롤', 'Bisabolol', '아줄렌', 'Azulene',
    '카페인', 'Caffeine', '아르기닌', 'Arginine',

    # 기능성 성분 - 각질케어/모공
    '살리실산', 'SalicylicAcid', '베타하이드록시산', 'BHA',
    '글리콜산', 'GlycolicAcid', '락틱산', 'LacticAcid',
    '만델산', 'MandelicAcid', '아젤라산', 'AzelaicAcid',
    '피루브산', 'PyruvicAcid', 'AHA', '알파하이드록시산',
    '폴리하이드록시산', 'PHA', '글루코노락톤', 'Gluconolactone',
    '락토바이오닉산', 'LactobionicAcid',

    # 기타 자주 쓰이는 성분
    '징크옥사이드', 'ZincOxide', '티타늄디옥사이드', 'TitaniumDioxide',
    '나트륨하이알루로네이트', 'SodiumHyaluronate', '히알루론산', 'HyaluronicAcid',
    '콜로이달오트밀', 'ColloidalOatmeal', '알지닌', 'Alginin',
    '비오틴', 'Biotin', '티아민', 'Thiamine',
    '피리독신', 'Pyridoxine', '리보플라빈', 'Riboflavin',
    '니코틴산아마이드', 'Nicotinamide', '폴릭산', 'FolicAcid',

    # 히알루론산 유도체
    '소듐하이알루로네이트', 'SodiumHyaluronate',
    '하이드롤라이즈드하이알루로닉애씨드', 'HydrolyzedHyaluronicAcid',
    '하이드롤라이즈드하이알루로닉애시드', '하이드롤라이즈드소듐하이알루로네이트',
    '소듐하이알루로네이트크로스폴리머', '포타슘하이알루로네이트',
    '하이드록시프로필트라이모늄하이알루로네이트', '소듐아세틸레이티드하이알루로네이트',
    '하이알루로닉애씨드', '하이알루로닉애시드',

    # UV 필터 (자외선 차단제 성분)
    '비스에칠헥실옥시페놀메톡시페닐드리아진', 'BisEthylhexyloxyphenolMethoxyphenylTriazine',
    '에칠헥실트리아존', 'EthylhexylTriazone', '옥토크릴렌', 'Octocrylene',
    '호모살레이트', 'Homosalate', '에칠헥실살리실레이트', 'EthylhexylSalicylate',
    '부틸메톡시디벤조일메탄', 'ButylMethoxydibenzoylmethane',
    '에칠헥실메톡시신나메이트', 'EthylhexylMethoxycinnamate',
    '디에칠아미노하이드록시벤조일헥실벤조에이트', '디에틸아미노하이드록시벤조알헥실벤조에이트',
    '테레프탈릴리덴디캠퍼설포닉애시드', '캠퍼설포닉애시드', '테레프탈릴리덴디캠퍼설포닉애씨드',
    '메칠렌비스벤조트리아졸릴테트라메틸부틸페놀', 'MethyleneBisBenzotriazolylTetramethylbutylphenol',
    '드로메트리졸트리실록산', 'DrometrizoleTrisiloxane',

    # 폴리머/증점제 (자주 등장)
    '폴리메틸실세스퀴옥세인', '폴리프로필실세스퀴옥세인',
    '폴리실리콘-15', '아크릴레이트/C10-30알킬아크릴레이트크로스폴리머',
    '암모늄폴리아크릴로일다이메틸타우레이트', '폴리아크릴레이트크로스폴리머-6',
    '아크릴레이트코폴리머', '카보머', 'Carbomer',

    # 유화제/계면활성제
    '폴리글리세릴-3다이스테아레이트', '글리세릴스테아레이트', '글리세릴스테아레이트시트레이트',
    '포타슘세틸포스페이트', '세테아릴알코올', 'CetearylAlcohol',
    'C12-15알킬벤조에이트', '다이부틸아디페이트',

    # 꽃수/꽃물 (플로럴워터)
    '다마스크장미꽃수', 'DamaskRoseFlowerWater', '장미꽃수', 'RoseFlowerWater',
    '라벤더꽃수', 'LavenderFlowerWater', '캐모마일꽃수', 'ChamomileFlowerWater',
    '네롤리꽃수', 'NeroliFlowerWater', '오렌지꽃수', 'OrangeFlowerWater',

    # 식물 추출물 추가
    '바나나꽃추출물', '서양배추출물', '서양자두추출물', '멜론추출물',
    '서양송악잎/줄기추출물', '센텔라아시아티카추출물',

    # 기타 자주 등장 성분
    't-부틸알코올', 'tButylAlcohol', '메틸프로판다이올', 'Methylpropanediol',
    '에틸헥실글리세린', 'Ethylhexylglycerin', '하이드록시아세토페논', 'Hydroxyacetophenone',
    '트로메타민', 'Tromethamine', '팔미틱애시드', 'PalmiticAcid',
    '스테아릭애시드', 'StearicAcid', '폴리에터-1', 'Polyether1',

    # 세라마이드 유도체 (영문 약어 표기)
    '세라마이드엔피', 'CeramideNP', '세라마이드에이피', 'CeramideAP',
    '세라마이드이오피', 'CeramideEOP', '세라마이드엔에스', 'CeramideNS',
    '세라마이드에이에스', 'CeramideAS', '세라마이드엔지', 'CeramideNG',

    # 펩타이드 계열 (주름개선)
    '팔미토일트라이펩타이드-1', 'PalmitoylTripeptide1',
    '팔미토일트라이펩타이드-5', 'PalmitoylTripeptide5',
    '팔미토일테트라펩타이드-7', 'PalmitoylTetrapeptide7',
    '팔미토일펜타펩타이드-4', 'PalmitoylPentapeptide4',
    '팔미토일헥사펩타이드-12', 'PalmitoylHexapeptide12',
    '아세틸헥사펩타이드-8', 'AcetylHexapeptide8', '아세틸헥사펩타이드-3', 'AcetylHexapeptide3',
    '아세틸테트라펩타이드-2', 'AcetylTetrapeptide2',
    '카퍼트라이펩타이드-1', 'CopperTripeptide1',
    '헥사펩타이드-9', 'Hexapeptide9', '헥사펩타이드-11', 'Hexapeptide11',
    '노나펩타이드-1', 'Nonapeptide1', '트라이펩타이드-1', 'Tripeptide1',
    '다이펩타이드-2', 'Dipeptide2', '옥타펩타이드-3', 'Octapeptide3',

    # 아미노산
    '류신', 'Leucine', '라이신', 'Lysine', '아이소류신', 'Isoleucine',
    '페닐알라닌', 'Phenylalanine', '트레오닌', 'Threonine',
    '발린', 'Valine', '메티오닌', 'Methionine', '히스티딘', 'Histidine',
    '트립토판', 'Tryptophan', '프롤린', 'Proline', '글루타민', 'Glutamine',
    '아스파르트산', 'AsparticAcid', '글루탐산', 'GlutamicAcid',
    '시스테인', 'Cysteine', '타이로신', 'Tyrosine', '세린', 'Serine',
    '글리신', 'Glycine', '알라닌', 'Alanine', '아스파라긴', 'Asparagine',

    # 에톡실레이트 계열
    '트라이데세스-6', 'Trideceth6', '트라이데세스-10', 'Trideceth10',
    '라우레스-7', 'Laureth7', '라우레스-23', 'Laureth23',

    # 알케인/탄화수소
    'C11-13아이소알케인', 'C1113Isoalkane', 'C12-15알킬벤조에이트',
    '아이소도데칸', 'Isododecane', '아이소헥사데칸', 'Isohexadecane',

    # 피브이엠 계열 (필름형성)
    '피브이엠/엠에이코폴리머', 'PVMMAcopolymer',

    # 식물추출물
    '알로에베라잎추출물', '녹차추출물', '병풀추출물',
    '감초추출물', '카모마일추출물', '라벤더추출물',
    '센텔라아시아티카추출물',

    # 실리콘류
    '다이메치콘', 'Dimethicone', '사이클로펜타실록산',
    '사이클로헥사실록산', '다이메치콘올', '아모다이메치콘',

    # 오일/왁스/에스터
    '스쿠알란', 'Squalane', '호호바오일', '시어버터',
    '미네랄오일', 'MineralOil', '세레신', '마이크로크리스탈린왁스', '파라핀',
    '카프릴릭/카프릭트라이글리세라이드', 'CaprylicCapricTriglyceride',
    '트라이실록세인', 'Trisiloxane', '아이소프로필미리스테이트', 'IsopropylMyristate',

    # 가스/추진제
    '프로판', 'Propane', '부탄', 'Butane', '이소부탄', 'Isobutane',
    'LPG', '프로페인', '부테인', '이소부테인',
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


# === 한글 화학 성분명 접두사/접미사 (모듈 레벨 상수) ===
KOREAN_CHEMICAL_PREFIXES = frozenset([
    '다이', '트리', '테트라', '데트라',  # 테트라 OCR 변형
    '펜타', '헥사', '핵사', '핵실',  # 헥사 OCR 변형
    '옥타', '폴리',
    '하이드로', '소듐', '소톱', '소득', '소문',  # 소듐 OCR 변형
    '포타슘', '포타습',  # 포타슘 OCR 변형
    '칼슘', '마그네슘', '아연',
    '에틸', '메틸', '부틸', '프로필', '세틸', '스테아릴',
    '글리세릴', '카프릴', '라우릴', '라우린', '미리스틸', '올레일',
    '피이지', '피피지',  # PEG, PPG
    '코카미도', '코카마이드',  # Cocamido
    '플리퀴터늄', '폴리쿼터늄',  # Polyquaternium
    '팬타에리스리',  # Pentaerythrityl
])

KOREAN_CHEMICAL_SUFFIXES = frozenset([
    '글라이콜', '글라이골', '글리콜',  # glycol + OCR 변형
    '글리세린', '에탄올', '메탄올', '프로판올',
    '옥사이드', '하이드록사이드', '설페이트', '포스페이트',
    '아세테이트', '시트레이트', '락테이트', '글루코네이트',
    '애시드', '애씨드', '에씨드', '아씨드',  # acid + OCR 오인식 변형
    '알코올', '에스테르', '에테르', '아마이드',
    '실록산', '실록세인', '메티콘', '실리콘',
    '추출물', '오일', '버터', '왁스', '검',
    '파라벤', '벤조에이트', '소르베이트',
    '나트륨', '칼륨', '염', '산',
    '베타인', '글라이블',  # 추가 성분 패턴
    '코코에이트', '올레에이트',  # fatty acid esters
    '이디티에이', '에디티에이',  # EDTA OCR 변형
    '꽃수', '꽃물', '플로럴워터',  # 플로럴 워터 패턴
    '하이알루로네이트', '하이알루로닉',  # 히알루론산 패턴
    '크로스폴리머', '코폴리머',  # 폴리머 패턴
])

# 동사형 어미 예외 (화학 성분명에서 허용되는 어미)
CHEMICAL_ENDING_EXCEPTIONS = frozenset([
    '린', '올', '산', '염', '이트', '레이트', '라이드', '나이트',
    '민', '틴', '왁스', '벤', '콘', '검', '물',
    '에이',  # MEA, TEA, DEA 등 (모노에탄올아민)
    '글라이콜', '글라이골',  # glycol OCR 변형
])


def _check_fast_rejection(text: str) -> tuple:
    """
    빠른 거부 필터 (성능 최적화)

    Returns:
        tuple: (is_rejected: bool, reason: str) - 거부 시 (True, reason), 아니면 (False, "")
    """
    # 순수 숫자 또는 숫자+단위 거부
    if re.match(r'^[\d\.\-/]+$', text):
        return True, "pure_number"

    # 전화번호 패턴 거부
    if re.match(r'^\d{2,4}-\d{3,4}-\d{4}$', text):
        return True, "phone_number"

    # 치수/용량 패턴 거부 (예: 18.5cm, 5.3cm, 8.8g)
    if re.match(r'^\d+(\.\d+)?(cm|mm|g|mg|ml|L|kg)$', text):
        return True, "dimension"

    # 한글 동사형 어미로 끝나는 단어 거부
    if re.search(r'(할|된|한|는|을|를|가|이|에|로|의|과|와|도|만|서|라|다)$', text):
        # 화학 성분명 어미 예외 체크
        if not any(text.endswith(suffix) for suffix in CHEMICAL_ENDING_EXCEPTIONS):
            return True, "verb_ending"

    return False, ""


def _is_noise_or_stopword(text: str) -> tuple:
    """
    노이즈 패턴 및 불용어 체크

    Returns:
        tuple: (is_invalid: bool, reason: str)
    """
    # 노이즈 패턴 매칭
    for pattern in NOISE_PATTERNS:
        if pattern.match(text):  # 컴파일된 패턴 사용
            return True, "noise"

    # 완전 일치 불용어
    if text in INGREDIENT_STOPWORDS:
        return True, "stopword"

    # 불용어 포함 체크: 긴 텍스트(12자 이상)에만 적용
    # 짧은 성분명에서 오탐 방지
    if len(text) >= 12:
        for stopword in INGREDIENT_STOPWORDS:
            if len(stopword) >= 4 and stopword in text:
                return True, "contains_stopword"

    return False, ""


def _check_chemical_patterns(text: str) -> tuple:
    """
    화학 성분 패턴 매칭 (영문/한글)

    Returns:
        tuple: (is_valid: bool, confidence: float, reason: str) 또는 None
    """
    # 기본 화학 패턴 (INGREDIENT_PATTERNS - 컴파일된 패턴)
    for pattern in INGREDIENT_PATTERNS:
        if pattern.match(text):  # 컴파일된 패턴 사용
            return True, 0.9, "pattern_match"

    # CI 코드 (색소)
    if re.match(r'^CI\s*\d+$', text):
        return True, 0.95, "ci_colorant"

    # PEG/PPG 계열
    if re.match(r'^(PEG|PPG|폴리에틸렌글라이콜|폴리프로필렌글라이콜)[-\s]?\d+', text):
        return True, 0.9, "peg_ppg"

    # 에톡실레이트 계열 (라우레스-9, 세테스-20 등)
    if re.match(r'^(라우레스|세테스|올레스|스테아레스|세테아레스|폴리소르베이트|솔비탄)[-\s]?\d+', text):
        return True, 0.9, "ethoxylate"

    # 영문 화학명 패턴 (예: Glycerin, Tocopherol)
    if re.match(r'^[A-Z][a-z]{3,}(yl|ol|in|ate|ide|ene|one|ose)?$', text):
        return True, 0.8, "english_chemical"

    # 복합 성분명 (영문+숫자, 예: PEG-14M, CI77891)
    if re.match(r'^[A-Za-z]+[-]?\d+[A-Za-z]*$', text) and len(text) >= 4:
        if not re.match(r'^\d{5,}$', text):
            return True, 0.75, "alphanumeric_compound"

    return None


def _is_korean_chemical(text: str) -> bool:
    """
    한글 화학 성분명 패턴 체크 (접두사/접미사 기반)

    Returns:
        bool: 한글 화학 성분명 여부
    """
    has_prefix = any(text.startswith(p) for p in KOREAN_CHEMICAL_PREFIXES)
    has_suffix = any(text.endswith(s) for s in KOREAN_CHEMICAL_SUFFIXES)
    return has_prefix or has_suffix


def is_valid_ingredient(text: str, known_db: set = KNOWN_INGREDIENTS) -> tuple:
    """
    성분명 유효성 검증 (리팩토링 버전 - Early Return 패턴)

    Args:
        text: 검증할 텍스트
        known_db: 알려진 성분 데이터베이스

    Returns:
        tuple: (is_valid: bool, confidence: float, reason: str)
    """
    # 기본 검증
    if not text or len(text.strip()) < 2:
        return False, 0.0, "too_short"

    text = text.strip()

    # 1. 빠른 거부 필터
    is_rejected, reason = _check_fast_rejection(text)
    if is_rejected:
        return False, 0.0, reason

    # 2. 알려진 성분 DB 매칭 (최고 신뢰도)
    if text in known_db:
        return True, 1.0, "known_ingredient"

    # 3. 노이즈/불용어 체크
    is_invalid, reason = _is_noise_or_stopword(text)
    if is_invalid:
        return False, 0.0, reason

    # 4. 화학 성분 패턴 매칭
    pattern_result = _check_chemical_patterns(text)
    if pattern_result:
        return pattern_result

    # 5. 한글 화학 성분명 체크
    if _is_korean_chemical(text):
        return True, 0.85, "korean_chemical"

    # 6. 기본 거부 (모든 조건 불충족)
    # False일 때 confidence는 무조건 0.0으로 통일
    return False, 0.0, "no_pattern_match"


def extract_from_text(text: str, source: str) -> list:
    """
    텍스트에서 성분 추출

    Args:
        text: OCR 또는 ALT 텍스트
        source: 출처 (HTML, ALT_0, OCR_0_1 등)

    Returns:
        list: [{'ingredient': str, 'source': str}, ...]
    """
    import re
    ingredients = []

    # 실제 성분명 패턴 (성분 시작 검증용)
    REAL_INGREDIENT_PATTERNS = [
        '정제수', '티타늄', '글리세린', '부틸렌', '나이아신', '다이메티콘',
        '사이클로', '에칠헥실', '소듐', '토코페롤', '히알루론', '알란토인',
        '판테놀', '아데노신', '세라마이드', '콜라겐', '레티놀', '비타민',
    ]

    lines = text.split('\n')
    in_ingredients = False
    ingredient_text = []

    for idx, line in enumerate(lines):
        line = line.strip()

        # 제외 섹션
        if any(exclude in line for exclude in INGREDIENT_EXCLUDE_SECTIONS):
            in_ingredients = False
            continue

        # 성분 섹션 시작 (키워드 확인)
        if any(kw in line for kw in INGREDIENT_KEYWORDS):
            # 같은 줄 또는 다음 줄에 실제 성분명이 있는지 확인
            has_real_ingredient = any(ing in line for ing in REAL_INGREDIENT_PATTERNS)

            # 다음 3줄까지 확인 (성분이 키워드 다음 줄에 있을 수 있음)
            if not has_real_ingredient:
                for look_ahead in range(1, 4):
                    if idx + look_ahead < len(lines):
                        next_line = lines[idx + look_ahead].strip()
                        if any(ing in next_line for ing in REAL_INGREDIENT_PATTERNS):
                            has_real_ingredient = True
                            break

            if has_real_ingredient:
                in_ingredients = True

                # "법에 따라", "하여야하는" 등 패턴 뒤의 텍스트만 추출
                pattern = r'(?:법에\s*따라|하여야\s*하는|모든\s*성분|전성분|INGREDIENTS?)\s*(.+)'
                match = re.search(pattern, line, re.IGNORECASE)

                if match:
                    remaining = match.group(1).strip()
                    if remaining:
                        ingredient_text.append(remaining)
                else:
                    # 키워드 제거 후 추가
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

    seen = set()  # 중복 방지

    for part in parts:
        part = part.strip()

        if not part:
            continue

        # 전체를 하나의 성분으로 시도 (OCR 정규화 적용)
        part_valid = False
        if len(part) >= 2:
            normalized = normalize_ingredient_name(part)
            if normalized and len(normalized) >= 2:
                # 유효성 검증
                is_valid, confidence, reason = is_valid_ingredient(normalized)
                if is_valid and normalized not in seen:
                    ingredients.append({'ingredient': normalized, 'source': source})
                    seen.add(normalized)
                    part_valid = True

        # 핵심 개선: part 전체가 유효하면 sub_parts는 검사하지 않음 (과검출 방지)
        # sub_parts는 KNOWN_INGREDIENTS에 있을 때만 허용 (더 엄격한 기준)
        if not part_valid:
            sub_parts = part.split()
            for sub in sub_parts:
                if len(sub) >= 2:
                    normalized_sub = normalize_ingredient_name(sub)
                    if normalized_sub and len(normalized_sub) >= 2 and normalized_sub not in seen:
                        # sub_parts는 KNOWN_INGREDIENTS에 있을 때만 허용
                        if normalized_sub in KNOWN_INGREDIENTS:
                            ingredients.append({'ingredient': normalized_sub, 'source': source})
                            seen.add(normalized_sub)

    return ingredients
