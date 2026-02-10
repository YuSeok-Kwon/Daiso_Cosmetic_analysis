"""
성분명 파싱 및 유효성 검증 모듈
"""
import re
import logging

# 로거 설정
logger = logging.getLogger(__name__)

# v2 파서 import (고도화된 성분 파싱)
try:
    from modules.ingredient_parser_v2 import (
        IngredientParserV2,
        TextNormalizer,
        IngredientFilter,
        IngredientSectionExtractor,
        IngredientDictionary,
        postprocess_ingredients_v2,
        get_clean_ingredients,
    )
    V2_PARSER_AVAILABLE = True
    logger.info("ingredient_parser_v2 로드 성공")
except ImportError:
    V2_PARSER_AVAILABLE = False
    logger.warning("ingredient_parser_v2 로드 실패 - 기존 파서 사용")

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

    # === ~올/~롤 OCR 오인식 (ㄹ → ㄹ+ㅡ, ㄹ → ㄷ 등) ===
    '다이을': '다이올',  # 메틸프로판다이을 → 메틸프로판다이올
    '글라이을': '글라이올',
    '토코페돌': '토코페롤',
    '레티돌': '레티놀',
    '판테돌': '판테놀',
    '멘토': '멘톨',
    '에탄을': '에탄올',
    '메탄을': '메탄올',
    '프로판을': '프로판올',

    # === OCR 글자 오인식 추가 ===
    '둠둠': '소듐',  # 소듐 → 둠둠
    '니이아신': '나이아신',  # 나이아신 → 니이아신
    '소듐이크릴': '소듐아크릴',  # 아크릴 → 이크릴
    '에스어이치': '에스에이치',  # 에스에이치 → 에스어이치

    # === 폴리소르베이트 변형 (숫자 오인식) ===
    '폴리소르베0655': '폴리소르베이트65',
    '폴리소르베이65': '폴리소르베이트65',
    '폴리소르베0660': '폴리소르베이트60',
    '폴리소르베이60': '폴리소르베이트60',
    '폴리소르베0680': '폴리소르베이트80',
    '폴리소르베이80': '폴리소르베이트80',
    '폴리소르베0620': '폴리소르베이트20',
    '폴리소르베이20': '폴리소르베이트20',

    # === 스테아릭/스티아릭 변형 ===
    '스티아릭': '스테아릭',
    '스테아리': '스테아릭',

    # === 이소부탄 변형 ===
    '이소무탄': '이소부탄',
    '이소뮤탄': '이소부탄',

    # === 트리에탄올아민 변형 ===
    '트리에타놀아민': '트리에탄올아민',
    '트리에탄놀아민': '트리에탄올아민',

    # === 하이드록사이드 변형 ===
    '하이드록시이드': '하이드록사이드',
    '하이드록싸이드': '하이드록사이드',
    '드로사이드': '하이드록사이드',  # 앞부분 잘림

    # === 정제수 변형 ===
    '정쩌수': '정제수',
    '정쪄수': '정제수',

    # === "이" → "O"/"0" OCR 오인식 (영어 O/숫자 0으로 잘못 인식) ===
    # 접미사 패턴: ~에이 → ~에O, ~네이트 → ~네0트/~네O트
    '이디티에O': '이디티에이',  # 다이소듐이디티에이
    '이디티에0': '이디티에이',
    '루로네O트': '루로네이트',  # 하이알루로네이트
    '루로네0트': '루로네이트',
    '네O트': '네이트',  # 일반 ~네이트 접미사
    '네0트': '네이트',
    '레O트': '레이트',  # 스테아레이트 등
    '레0트': '레이트',
    '라O드': '라이드',  # 클로라이드 등
    '라0드': '라이드',
    '사O드': '사이드',  # 옥사이드 등
    '사0드': '사이드',

    # 접두사/중간 패턴: 아~ → O~/0~
    'O세틸': '아세틸',  # 아세틸레이티드
    '0세틸': '아세틸',
    'O크릴': '아크릴',  # 아크릴레이트
    '0크릴': '아크릴',
    'O마이드': '아마이드',  # 나이아신아마이드
    '0마이드': '아마이드',
    'O민산': '아민산',  # 아미노산
    '0민산': '아민산',

    # 기타 자주 발생하는 O/0 오인식
    '페녹시에탄O': '페녹시에탄올',  # 페녹시에탄올
    '페녹시에탄0': '페녹시에탄올',
    '글리세릴O': '글리세릴',  # 끝에 O가 붙는 경우 제거
    '글리세릴0': '글리세릴',
    '포스페0트': '포스페이트',  # 소듐아스코빌포스페이트
    '포스페O트': '포스페이트',
    '페0트': '페이트',  # 일반 ~페이트 접미사
    '페O트': '페이트',

    # === 폴리소르베이트 OCR 오인식 ===
    '솔베0l트': '소르베이트',  # 폴리솔베0l트 → 폴리소르베이트
    '솔베0|트': '소르베이트',
    '솔베이트': '소르베이트',  # 솔베 → 소르베
    '폴리솔베': '폴리소르베',

    # === 스테아레이트 OCR 오인식 ===
    '스터아레이트': '스테아레이트',  # 스터 → 스테
    '스타아레이트': '스테아레이트',
    '스태아레이트': '스테아레이트',

    # === 라우레이트 OCR 오인식 ===
    '리우레이트': '라우레이트',  # 리 → 라
    '리우릭': '라우릭',

    # === OCR 숫자-한글 오인식 (숫자가 한글 중간에 섞임) ===
    # 1 → 이/ㅣ 오인식
    '러1이트': '레이트',  # 타우러1이트 → 타우레이트
    '러1에이트': '레이트',
    '릴1이트': '릴레이트',
    '타우러1': '타우레',
    '글라0콜': '글라이콜',

    # 0 → 이/아 오인식 (한글 중간)
    '다0메틸': '다이메틸',
    '다0메칠': '다이메틸',
    '트라0아이소': '트라이아이소',
    '트라0이소': '트라이아이소',
    '폴리0크릴': '폴리아크릴',
    '하이0루로닉': '하이알루로닉',
    '하이일루로닉': '하이알루로닉',  # 이 → 알

    # === 추가 OCR 오인식 ===
    '소둠': '소듐',
    '케프': '켈프',  # 호스테일케프 → 호스테일켈프
    '이마이드': '아마이드',  # 옥틸아크릴이마이드 → 옥틸아크릴아마이드
    '아이소스스테아': '아이소스테아',  # 중복 'ㅅ' 제거
    '헥사노어이트': '헥사노에이트',  # 어 → 에

    # === 성분 시작 키워드가 붙은 경우 ===
    '전성분': '',  # 전성분변성알코올 → 변성알코올
    '[전성분]': '',
    '(전성분)': '',

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

    # === 영문 노이즈 패턴 (제품 설명, 마케팅 문구) ===
    r'^(POINT|STEP|TIP|NOTE|HOW|WHAT|WHY|KEY|BEST|NEW|SPECIAL)\d*$',
    r'^(Book|Marine|Complex|Extra|Super|Care|Soluti|Solution|Recipe)$',
    r'^(SUPER|EXTRA|PLUS|PRO|MAX|ULTRA)\d*$',
    r'^(Before|After|Texture|Ingredient|Recommend)$',
    r'^[A-Z]{2,6}\d*$',  # 순수 대문자 약어 (SPF50 제외, 위에서 처리)
]
NOISE_PATTERNS = [re.compile(p) for p in _NOISE_PATTERN_STRS]

# 성분 키워드 (실제 성분 리스트를 나타내는 키워드)
INGREDIENT_KEYWORDS = [
    "전성분", "화장품법", "모든 성분", "모든성분",  # 공백 유무 둘 다
    "화장품법에 따라", "기재 표시", "기재·표시", "기재표시", "표시하여야", "표시 하여야",
    "하여야하는", "하여야 하는",
    "성분:", "[성분명]", "성분명", "[전성분]",
    "성분은", "주성분",  # 추가: "성분은 정제수..." 형식 대응
    # 영문 키워드는 제외 - 영문 성분 리스트가 단어별로 분리되는 문제 발생
    # "INGREDIENTS", "Ingredients",
]

# 영문 성분 섹션 시작 키워드 (여기서 성분 추출 중단)
ENGLISH_INGREDIENT_KEYWORDS = [
    "(Ingredient)", "[Ingredient]", "Ingredient)", "Ingredients:",
    "(INCI)", "INCI:", "INCI Name",
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
    # 영문 성분 섹션 시작 (한글 성분 추출 후 중단)
    '(Ingredient)', '[Ingredient]', 'Ingredient)', 'Ingredients:',
    '(INCI)', 'INCI:', 'INCI Name',
]

# 메타데이터 키워드 (성분이 아닌 제품 정보)
METADATA_KEYWORDS = [
    '제조업자', '책임판매업자', '판매업자', '제조국', '원산지',
    '코스비전', '아모레퍼시픽', '엘지생활건강', 'LG생활건강',
    '대한민국', '한국', '중국', '일본', '프랑스', '미국',
    '기능성화장품', '해당사항', '심사필', '여부',
    '맞춤형', '수입원', '제조원', '유통기한', '개봉후',
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

    # === 영문 노이즈 (제품 설명, 마케팅 문구) ===
    'Book', 'Marine', 'Complex', 'Extra', 'Super', 'Care', 'Soluti', 'Solution',
    'Recipe', 'SUPER', 'EXTRA', 'PLUS', 'PRO', 'MAX', 'ULTRA', 'SUPER9',
    'Before', 'After', 'Texture', 'Ingredient', 'Recommend', 'POINT',
    'STEP', 'TIP', 'NOTE', 'HOW', 'WHAT', 'WHY', 'KEY', 'BEST', 'NEW', 'SPECIAL',
    'VPROVE', 'MEDIPEEL', 'Oriox', 'LipBalm', 'Essense', 'Essence',

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
    '카라기난', '셀룰로오스검', '아크릴레이트코폴리머', '아가',

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

    # 멘톨/쿨링 성분
    '멘톨', 'Menthol', '멘틸락테이트', 'MenthylLactate',
    '멘톡시프로판다이올', 'Menthoxypropanediol', '피씨멘톨', 'PCMenthol',
    '멘틸아세테이트', 'MenthylAcetate', '멘틸피씨에이', 'MenthylPCA',

    # 향료 성분 (자주 등장)
    '향료', 'Fragrance', 'Parfum', '라임향', '레몬향', '민트향',
    '시트러스향', '플로럴향', '우디향', '머스크향',

    # === Phase 1-1 추가: 누락 성분 확장 ===

    # 에톡실레이트 계열 (하이픈+숫자)
    '글리세레스-26', '글리세레스-7', '글리세레스-10',
    '폴리글리세릴-10미리스테이트', '폴리글리세릴-10라우레이트',
    '폴리글리세릴-10올레에이트', '폴리글리세릴-10스테아레이트',
    '폴리글리세릴-3다이아이소스테아레이트', '폴리글리세릴-6다이스테아레이트',
    '폴리글리세릴-2다이폴리하이드록시스테아레이트',

    # 곡물/식물 유래
    '옥수수전분', '쌀겨오일', '쌀전분', '감자전분', '타피오카전분',
    '밀전분', '고구마전분', '쌀겨추출물', '현미추출물',
    '쌀뜨물', '보리추출물', '귀리추출물', '메밀추출물',

    # 효소
    '파파인', '브로멜라인', '서브틸리신', '리파아제',
    '프로테아제', '아밀라아제', '셀룰라아제', '락타아제',
    '수퍼옥사이드디스뮤타아제', '카탈라아제',

    # 항산화/아미노산 유도체
    '글루타치온', '라이신에이치씨엘', '알지닌에이치씨엘',
    '히스티딘에이치씨엘', '시스테인에이치씨엘', '글루타민에이치씨엘',
    '라이신하이드로클로라이드', '알지닌하이드로클로라이드',
    '라이신염산염', '알지닌염산염', '시스테인염산염',

    # 솔비탄/솔비톨 계열
    '솔비탄모노스테아레이트', '솔비탄트라이스테아레이트',
    '솔비탄모노올레에이트', '솔비탄세스퀴올레에이트',
    '솔비탄모노라우레이트', '솔비탄모노팔미테이트',
    '솔비톨', 'Sorbitol',

    # 기타 누락 성분
    '카프릴릴글라이콜', '잔탄검', '합성왁스', '마이크로크리스탈린왁스',
    '세테아릴글루코사이드', '데실글루코사이드', '라우릴글루코사이드',
    '코코글루코사이드', '코코베타인', '라우라마이도프로필베타인',
    '코카마이도프로필베타인', '세틸피지', 'CetylPG',
    '하이드록시에틸셀룰로오스', '하이드록시프로필메틸셀룰로오스',
    '카복시메틸셀룰로오스', '에틸셀룰로오스', '메틸셀룰로오스',

    # 비타민 유도체
    '토코페릴아세테이트', '아스코르빌테트라이소팔미테이트',
    '레티닐아세테이트', '피리독신에이치씨엘', '리보플라빈포스페이트',
    '티아민에이치씨엘', '시아노코발라민', '엽산', '판토텐산',

    # 누락된 성분 추가 (OCR 크롤링 디버깅 기반)
    '아르지닌', 'Arginine',  # 알지닌의 다른 표기
    '하이드록시에틸우레아', 'HydroxyethylUrea',  # 보습제
    '다이메틸설폰', 'DimethylSulfone', 'MSM',  # 기능성 성분
    '소듐피씨에이', 'SodiumPCA',  # 보습제

    # 1065041 디버깅 추가
    '알진', 'Algin',  # 해조류 유래 점증제
    '규조토', 'DiatomaceousEarth',  # 흡착제
    '글루코오스', 'Glucose',  # 당류
    '에리스리톨', 'Erythritol',  # 보습제
    '비타민나무수', 'HippophaeRhamnoidesWater',  # 식물수
    '하이드롤라이즈드콜라겐', 'HydrolyzedCollagen',  # 펩타이드
    '엑사펩타이드-9', 'Hexapeptide9',  # 펩타이드
    '황색산화철', 'YellowIronOxide', 'CI77492',  # 색소
    '적색산화철', 'RedIronOxide', 'CI77491',
    '흑색산화철', 'BlackIronOxide', 'CI77499',
    '리놀레익애씨드', 'LinoleicAcid', '리놀레산',  # 지방산
    '에틸헥실글리세린', 'Ethylhexylglycerin',  # 방부보조제
    '폴리글리세릴-6카프릴레이트', 'Polyglyceryl6Caprylate',
    '하이드록시프로필트라이모늄하이알루로네이트',  # 양이온 히알루론산
    '소듐아세틸레이티드하이알루로네이트',  # 아세틸화 히알루론산
    '폴리글리세릴-10스테아레이트',

    # 기타 누락 성분 (1005698 등)
    '베타-글루칸', 'BetaGlucan', '베타글루칸',  # 보습/면역 성분
    '헥산디올', '헥산다이올',  # 1,2-헥산다이올 변형
    '라반딘오일', 'LavandinOil',  # 향료
    '연필향나무오일', 'JuniperusVirginianaOil',  # 향료

    # 기타 누락 성분 (1061921 등)
    '트레할로오스', 'Trehalose',  # 보습제/당류
    '헥실신남알', 'HexylCinnamal',  # 향료 성분
    '시트로넬롤', 'Citronellol',  # 향료 성분
    '리날룰', 'Linalool',  # 향료 성분
    '리모넨', 'Limonene',  # 향료 성분
    '피리독신', 'Pyridoxine',  # 비타민 B6
    '티아민에이치씨엘', 'ThiamineHCl',  # 비타민 B1
    '폴릭애시드', 'FolicAcid',  # 엽산
    '바이오틴', 'Biotin',  # 비타민 H
    '사이아노코발아민', 'Cyanocobalamin',  # 비타민 B12

    # 색소 (황색/적색/청색 번호)
    '황색4호', 'Yellow4', 'CI19140',
    '황색5호', 'Yellow5', 'CI15985',
    '황색201호', '황색202호', '황색203호',
    '적색2호', 'Red2', 'CI16185',
    '적색40호', 'Red40', 'CI16035',
    '적색102호', '적색201호', '적색202호',
    '청색1호', 'Blue1', 'CI42090',
    '청색2호', 'Blue2', 'CI73015',
}


def normalize_ingredient_name(name: str) -> str:
    """
    성분명 정규화 및 OCR 오류 수정

    Args:
        name: 원본 성분명

    Returns:
        str: 정규화된 성분명
    """
    # 0. 성분 앞에 붙은 키워드 제거 (전성분변성알코올 → 변성알코올, 주성분옥틸도데칸올 → 옥틸도데칸올)
    prefix_keywords = [
        '전성분', '[전성분]', '(전성분)', '성분:', '성분',
        '주성분', '주성분:', '원료:', '원료',
    ]
    for kw in prefix_keywords:
        if name.startswith(kw):
            name = name[len(kw):]

    # 0-1. 메타데이터 키워드가 성분 중간에 삽입된 경우 제거
    # 예: "카프릴릭/카프릭트라이글리세라이드화장품법에따라기재표시하여야하는모든성분" 같은 경우
    metadata_in_text = [
        '화장품법에따라', '화장품법에 따라', '기재표시하여야하는', '기재·표시하여야하는',
        '기재표시하여야 하는', '모든성분', '모든 성분', '하여야하는', '표시하여야',
        '전성분', '성분명',
    ]
    for meta in metadata_in_text:
        name = name.replace(meta, '')

    # 0-2. 영문 노이즈 접두사 제거 (Complex블래더랙추출물 → 블래더랙추출물)
    english_prefix_noise = [
        'Complex', 'Marine', 'Extra', 'Super', 'SUPER', 'Recipe', 'Aqua',
        'Special', 'Premium', 'Natural', 'Organic', 'Pure', 'Active',
        'Advanced', 'Essential', 'Ultimate', 'Intensive', 'MARINE',
    ]
    for prefix in english_prefix_noise:
        if name.startswith(prefix) and len(name) > len(prefix):
            # 접두사 뒤에 한글이 오는 경우만 제거
            rest = name[len(prefix):]
            if rest and re.match(r'^[가-힣]', rest):
                name = rest

    # 0-3. 한글 노이즈 접두사 제거 (슈퍼9콤플렉스아보카도 → 아보카도)
    korean_prefix_noise = [
        '슈퍼9콤플렉스', '슈퍼콤플렉스', '마린콤플렉스', '에피샷', '아쿠아콤플렉스',
        '프리미엄', '스페셜', '익스트라', '울트라', '인텐시브', '어드밴스드',
    ]
    for prefix in korean_prefix_noise:
        if name.startswith(prefix) and len(name) > len(prefix):
            rest = name[len(prefix):]
            if rest and len(rest) >= 3:  # 남은 부분이 3자 이상이면 제거
                name = rest

    # 1. 공백 제거
    name = re.sub(r'\s+', '', name)

    # 2. OCR 오류 수정
    for wrong, correct in OCR_CORRECTIONS.items():
        name = name.replace(wrong, correct)

    # 3. 특수문자 정규화
    name = name.replace('±', '')
    name = name.replace('·', '')
    name = re.sub(r'[�]', '', name)  # 깨진 문자 제거

    # 3.5 앞부분 OCR 잡음 제거 (숫자, 특수문자로 시작하는 경우)
    # 예: "10|하이드록사이드" → "하이드록사이드", "=10|드로사이드" → "드로사이드"
    # 주의: 1,2-헥산다이올 같은 화학 번호 접두사는 보존
    if not re.match(r'^\d,\d-', name):  # 화학 번호 패턴이 아닌 경우만 제거
        name = re.sub(r'^[0-9=|!@#$%^&*<>]+', '', name)

    # 4. 괄호 내용 제거 (Phase 1-2: 농도/함량 괄호 우선 처리)
    # 농도/함량 괄호 먼저 제거 (숫자+단위): 솔비톨(44.79%), 니코틴산아마이드(10ppm) 등
    # 지원 단위: %, ppb, ppm, mg/kg, µg/kg, mg/L, µg/L, g/L, w/w, w/v, v/v, mg, g, ml, L, kg
    concentration_pattern = r'\(\s*[\d,\.]+\s*(ppb|ppm|%|mg/kg|µg/kg|mg/L|µg/L|g/L|w/w|w/v|v/v|mg|g|ml|L|kg)?\s*\)'
    name = re.sub(concentration_pattern, '', name, flags=re.IGNORECASE)

    # 화학명 괄호는 보존 (C6-14올레핀, CI77891 등)
    def should_keep_paren(match):
        content = match.group(1)
        # CI + 숫자 패턴 (색소): CI77891, CI 77007 등
        if re.match(r'^CI\s*\d+$', content, re.IGNORECASE):
            return match.group(0)
        # 탄소 사슬 패턴: C6-14, C12-15 등 (+ 한글/영문 가능)
        if re.match(r'^C\d+[-]?\d*', content):
            return match.group(0)
        # 숫자-숫자 패턴: 6-14, 12-15 등
        if re.match(r'^\d+[-]\d+', content):
            return match.group(0)
        # 그 외는 제거 (농도, 설명 등)
        return ''

    name = re.sub(r'\(([^)]*)\)', should_keep_paren, name)
    name = re.sub(r'\[[^\]]*\]', '', name)

    # 5. 기타 특수문자 제거
    name = re.sub(r'[*★☆※]', '', name)

    # 6. 앞뒤 특수문자 제거 (괄호는 보존)
    name = re.sub(r'^[^\w가-힣()]+|[^\w가-힣()]+$', '', name, flags=re.UNICODE)

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

    # === Phase 2 추가: 확장 접미사 ===
    '전분',  # 곡물 유래 (옥수수전분, 쌀전분 등)
    '에이치씨엘', '하이드로클로라이드', '염산염',  # 아미노산+염 (라이신에이치씨엘 등)
    '황산염', '질산염', '탄산염',  # 기타 염류
    '아제', '라아제',  # 효소 (파파인, 리파아제 등)
    '글루코사이드',  # 당 유도체 (세테아릴글루코사이드 등)
    '셀룰로오스', '셀룰로스',  # 셀룰로오스 유도체
    '미리스테이트', '라우레이트', '팔미테이트', '스테아레이트',  # 지방산 에스터
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

    # 제품 설명 텍스트 거부 (POINT01, STEP1, PDRN5 등)
    if re.match(r'^(POINT|STEP|TIP|NOTE|PDRN)\d*$', text, re.IGNORECASE):
        return True, "product_description"

    # 2글자 한글 단어 중 성분이 아닌 것 거부
    # (세린, 멘톨, 알진, 류신, 발린, 향료, 아가 등은 유효한 성분)
    INVALID_2CHAR_KOREAN = {
        '고민', '어린', '그린', '여린', '마린', '계산', '끌올', '폴리', '에틸', '펜타', '헥산',
        '디엔', '피피', '피엠', '피비', '유전', '영양', '촉촉', '보습', '피부', '탄력',
        '주름', '미백', '기미', '잡티', '결점', '모공', '각질', '톤업', '윤기',
        # 추가: 잘못 파싱된 2글자 단어들
        '중산', '감염', '아가', '초유', '소듐', '트리', '오일', '부탄', '프로', '메틸',
    }
    if len(text) == 2 and re.match(r'^[가-힣]{2}$', text) and text in INVALID_2CHAR_KOREAN:
        return True, "invalid_2char"

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

    # === 영문 노이즈 필터링 ===
    # 순수 영문 단어 중 화학 성분이 아닌 것 거부
    if re.match(r'^[A-Za-z]+$', text):
        # 허용되는 영문 성분 패턴
        VALID_ENGLISH_INGREDIENTS = {
            'Water', 'Aqua', 'Glycerin', 'Niacinamide', 'Panthenol', 'Retinol',
            'Tocopherol', 'Collagen', 'Elastin', 'Ceramide', 'Squalane',
            'Allantoin', 'Adenosine', 'Caffeine', 'Menthol', 'Urea',
            'Dimethicone', 'Silicone', 'Paraben', 'BHT', 'BHA', 'PEG', 'PPG',
            'MEA', 'TEA', 'DEA', 'EDTA', 'CI', 'UV',
        }
        # 영문 성분 접미사 패턴
        ENGLISH_SUFFIX_PATTERNS = [
            r'.*(ol|ate|ide|ine|one|ene|ose|ase|ane|yl|ic|in)$',  # 화학 접미사
            r'^(Hydrogenated|Hydrolyzed|PEG|PPG|CI)\d*.*$',  # 접두사 패턴
        ]

        is_valid_english = text in VALID_ENGLISH_INGREDIENTS
        has_chemical_suffix = any(re.match(p, text, re.IGNORECASE) for p in ENGLISH_SUFFIX_PATTERNS)

        if not is_valid_english and not has_chemical_suffix:
            # 일반 영문 단어 (노이즈)
            if len(text) <= 10:  # 짧은 영문 단어는 대부분 노이즈
                return True, "english_noise"

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

    # === Phase 2 추가: 확장 패턴 ===

    # 에톡실레이트 확장 (하이픈+숫자 복합형): 글리세레스-26, 폴리글리세릴-10미리스테이트 등
    if re.match(r'^(글리세레스|폴리글리세릴|세테스|올레스|스테아레스|트라이데세스)[-]?\d+.*$', text):
        return True, 0.9, "ethoxylate_extended"

    # 효소 패턴: ~아제, ~라아제 (단, 동사 어미 제외)
    # 파파인, 브로멜라인, 리파아제, 프로테아제 등
    if re.match(r'^.{2,}(아제|라아제|인)$', text):
        # '~하는', '~되는' 등 동사형이 아닌지 확인
        if not re.search(r'(하는|되는|있는|없는)$', text):
            # '인'으로 끝나는 경우 더 엄격하게 검증 (효소명 패턴)
            if text.endswith('인') and len(text) >= 3:
                # 파파인, 브로멜라인, 트립신, 펩신 등 효소/단백질명
                if re.match(r'^[가-힣]{2,}인$', text):
                    return True, 0.85, "enzyme"
            elif text.endswith('아제') or text.endswith('라아제'):
                return True, 0.85, "enzyme"

    # 아미노산+염 패턴: ~에이치씨엘, ~하이드로클로라이드, ~염산염
    if re.match(r'.+(에이치씨엘|하이드로클로라이드|염산염|황산염|질산염)$', text):
        return True, 0.85, "amino_acid_salt"

    # 식물 유래 패턴: ~오일, ~전분, ~버터 (4자 이상)
    if len(text) >= 4:
        if re.match(r'^[가-힣]{2,}(오일|전분|버터)$', text):
            return True, 0.85, "plant_derived"

    # 글루코사이드 패턴: ~글루코사이드
    if text.endswith('글루코사이드') and len(text) >= 6:
        return True, 0.85, "glucoside"

    # 셀룰로오스 유도체: ~셀룰로오스, ~셀룰로스
    if re.match(r'^.+(셀룰로오스|셀룰로스)$', text) and len(text) >= 6:
        return True, 0.85, "cellulose_derivative"

    # === 영문 노이즈 블랙리스트 체크 (화학 패턴 매칭 전) ===
    ENGLISH_NOISE_WORDS = {
        'Book', 'Marine', 'Complex', 'Extra', 'Super', 'Care', 'Solution', 'Soluti',
        'Recipe', 'Before', 'After', 'Texture', 'Ingredient', 'Recommend', 'Point',
        'Step', 'Tip', 'Note', 'How', 'What', 'Why', 'Key', 'Best', 'New', 'Special',
        'Plus', 'Pro', 'Max', 'Ultra', 'Premium', 'Natural', 'Organic', 'Pure',
        'Active', 'Advanced', 'Essential', 'Ultimate', 'Intensive', 'Free',
        'SUPER', 'EXTRA', 'PLUS', 'PRO', 'MAX', 'ULTRA', 'MARINE', 'RECIPE',
        'VPROVE', 'MEDIPEEL', 'Oriox', 'LipBalm', 'Essense', 'Essence',
        # 영문+숫자 노이즈 (마케팅 문구)
        'SUPER9', 'SUPER5', 'PLUS9', 'EXTRA9', 'PRO5', 'STEP1', 'STEP2', 'STEP3',
        'POINT1', 'POINT2', 'POINT3', 'TIP1', 'TIP2',
    }
    if text in ENGLISH_NOISE_WORDS:
        return None  # 노이즈로 처리 (나중에 거부됨)

    # 영문 화학명 패턴 (예: Glycerin, Tocopherol)
    # 반드시 화학 접미사가 있어야 함
    if re.match(r'^[A-Z][a-z]{3,}(yl|ol|in|ate|ide|ene|one|ose|ine|ane|ene)$', text):
        return True, 0.8, "english_chemical"

    # 복합 성분명 (영문+숫자, 예: PEG-14M, CI77891)
    # SPF, PA 등 제품 규격은 제외
    if re.match(r'^[A-Za-z]+[-]?\d+[A-Za-z]*$', text) and len(text) >= 4:
        if not re.match(r'^\d{5,}$', text):
            # SPF, PA, SUPER, STEP, POINT 등 규격/마케팅 패턴 제외
            if not re.match(r'^(SPF|PA|UV|LED|SUPER|STEP|POINT|TIP|EXTRA|PLUS)\d+', text, re.IGNORECASE):
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

    # 1. 알려진 성분 DB 매칭 (최고 우선순위)
    # known_db에 있으면 다른 필터 무시하고 바로 승인
    if text in known_db:
        return True, 1.0, "known_ingredient"

    # 2. 빠른 거부 필터
    is_rejected, reason = _check_fast_rejection(text)
    if is_rejected:
        return False, 0.0, reason

    # 3. 화학 성분 패턴 매칭 (노이즈 체크보다 먼저 수행)
    # 화학 패턴에 매칭되면 불용어 체크를 건너뜀 (폴리글리세릴-6올레에이트 등)
    pattern_result = _check_chemical_patterns(text)
    if pattern_result:
        return pattern_result

    # 4. 한글 화학 성분명 체크 (접두사/접미사 기반)
    if _is_korean_chemical(text):
        return True, 0.85, "korean_chemical"

    # 5. 노이즈/불용어 체크 (화학 패턴에 매칭되지 않은 경우에만)
    is_invalid, reason = _is_noise_or_stopword(text)
    if is_invalid:
        return False, 0.0, reason

    # 6. 기본 거부 (모든 조건 불충족)
    # False일 때 confidence는 무조건 0.0으로 통일
    return False, 0.0, "no_pattern_match"


def extract_from_text(text: str, source: str, threshold: float = None, force_mode: bool = False, use_v2: bool = True) -> list:
    """
    텍스트에서 성분 추출

    Args:
        text: OCR 또는 ALT 텍스트
        source: 출처 (HTML, ALT_0, OCR_0_1 등)
        threshold: confidence threshold (None이면 source 기반 자동 결정)
        force_mode: True이면 헤더 검사 없이 전체 텍스트를 성분으로 처리
                    (호출자가 이미 성분 섹션으로 판단한 경우 사용)
        use_v2: v2 파서 사용 여부 (기본값: True)

    Returns:
        list: [{'ingredient': str, 'source': str}, ...]
    """
    import re
    import logging
    logger = logging.getLogger(__name__)
    ingredients = []

    # === v2 파서 사용 (고도화된 파싱) ===
    if use_v2 and V2_PARSER_AVAILABLE:
        try:
            v2_ingredients = IngredientParserV2.parse(text, korean_only=True)
            if v2_ingredients and len(v2_ingredients) >= 3:
                # v2 결과를 기존 형식으로 변환
                logger.debug(f"v2 파서 사용: {len(v2_ingredients)}개 성분 추출")
                return [{'ingredient': ing, 'source': f"{source}_v2"} for ing in v2_ingredients]
            else:
                logger.debug(f"v2 파서 결과 부족 ({len(v2_ingredients) if v2_ingredients else 0}개), 기존 파서로 폴백")
        except Exception as e:
            logger.warning(f"v2 파서 오류: {e}, 기존 파서로 폴백")

    # === 기존 파서 로직 ===

    # Phase 3: 소스별 threshold 적용
    if threshold is None:
        # ALT 텍스트나 CLOVA OCR은 신뢰도가 높으므로 낮은 threshold 적용
        if source.startswith('ALT') or 'CLOVA' in source.upper():
            threshold = 0.75
        else:
            threshold = 0.85

    # 실제 성분명 패턴 (성분 시작 검증용) - 확장
    REAL_INGREDIENT_PATTERNS = [
        '정제수', '티타늄', '글리세린', '부틸렌', '나이아신', '다이메티콘',
        '사이클로', '에칠헥실', '소듐', '토코페롤', '히알루론', '알란토인',
        '판테놀', '아데노신', '세라마이드', '콜라겐', '레티놀', '비타민',
        '프로판다이올', '카보머', '페녹시에탄올', '향료', '시트릭', '스쿠알란',
        '세틸알코올', '헥산다이올', '알지닌', '세린', '폴리소르베이트',
        '아이소도데케인', '헥실렌글라이콜', '리모넨', '리날룰',
    ]

    # === 구분자 유형 판단 ===
    # 쉼표가 거의 없으면 공백 구분자로 판단
    text_flat_check = text.replace('\n', ' ')
    # 화학 번호 쉼표 제외 (1,2-헥산다이올 등)
    comma_count_check = len(re.findall(r',(?!\d-)', text_flat_check))
    is_space_separated = comma_count_check < 5

    if is_space_separated:
        logger.debug(f"공백 구분자 모드 감지 (쉼표 {comma_count_check}개)")
        return _extract_space_separated(text, source, threshold, REAL_INGREDIENT_PATTERNS, logger)

    # === 기존 쉼표 구분자 로직 ===

    # === 헤더 없이 성분 감지 모드 ===
    # force_mode가 True이면 무조건 성분 모드
    # force_mode=False이면 키워드 기반으로만 동작 (ALT 텍스트 등에서 정확한 파싱 필요)
    detected_ingredient_count = sum(1 for ing in REAL_INGREDIENT_PATTERNS if ing in text_flat_check)

    # force_mode=True일 때만 대표 성분명 개수로 강제 추출
    # force_mode=False이면 키워드 기반으로만 추출 (예: "(전성분)" 이후만 파싱)
    force_ingredient_mode = force_mode
    if force_ingredient_mode:
        logger.debug(f"성분 강제 추출 모드: force_mode={force_mode}, 감지된 성분={detected_ingredient_count}개")

    # 전처리: "전성분" 키워드가 텍스트 중간에 있는 경우 처리
    # 예: "에틸헥실팔미테이트,\n글리세린,\n...\n전성분\n폴리글리세릴..."
    # 이 경우 키워드 앞의 성분들도 추출해야 함
    # 단, force_mode=True일 때만 적용 (ALT 텍스트 등에서 "비타민" 같은 제품명을 성분으로 오인 방지)
    if force_mode:
        text_flat = text.replace('\n', ' ').replace('  ', ' ')
        mid_keyword_patterns = ['전성분', '성분은', '[성분명]', '성분명]']

        for kw in mid_keyword_patterns:
            kw_pos = text_flat.find(kw)
            if kw_pos > 10:  # 키워드가 텍스트 시작이 아닌 중간에 있음
                before_keyword = text_flat[:kw_pos]
                # 키워드 앞에 실제 성분명이 있는지 확인
                if any(ing in before_keyword for ing in REAL_INGREDIENT_PATTERNS):
                    # 키워드를 쉼표로 대체하여 전체를 성분으로 처리
                    text = text_flat.replace(kw, ',')
                    break

    lines = text.split('\n')
    in_ingredients = False  # 성분 키워드를 만날 때까지 대기
    ingredient_text = []

    # force_ingredient_mode일 때: 줄 단위가 아닌 전체 텍스트에서 성분 추출
    # (2열 레이아웃에서 STOP 키워드가 성분 중간에 섞여 있어도 처리 가능)
    if force_ingredient_mode:
        # 전체 텍스트를 공백으로 연결하여 성분 텍스트로 사용
        # 줄바꿈으로 잘린 성분명이 전처리에서 합쳐질 수 있도록 함
        # 예: "나이아신아마\n이드," → "나이아신아마 이드," → "나이아신아마이드,"
        all_lines = []
        for line in lines:
            line_stripped = line.strip()
            if line_stripped:
                all_lines.append(line_stripped)
        # 공백으로 연결 (전처리에서 한글 사이 공백이 제거됨)
        if all_lines:
            ingredient_text.append(' '.join(all_lines))

        # 이미 처리했으므로 lines를 비움
        lines = []

    # 첫 줄에 실제 성분명이 있으면 바로 성분 섹션으로 판단 (Clova OCR 결과 대응)
    # 단, force_mode=True일 때만 (force_mode=False면 키워드 기반으로 동작)
    first_line = lines[0].strip() if lines else ''
    if force_mode and first_line and any(ing in first_line for ing in REAL_INGREDIENT_PATTERNS):
        in_ingredients = True
        ingredient_text.append(first_line)
        lines = lines[1:]  # 첫 줄은 이미 추가했으므로 제외

    for idx, line in enumerate(lines):
        line = line.strip()

        # 제외 섹션
        if any(exclude in line for exclude in INGREDIENT_EXCLUDE_SECTIONS):
            in_ingredients = False
            continue

        # 성분 섹션 시작 (키워드 확인)
        if any(kw in line for kw in INGREDIENT_KEYWORDS):
            # 정확한 키워드가 있으면 바로 성분 섹션으로 인식
            exact_keyword = '전성분' in line or '성분은' in line or '성분:' in line or '[성분명]' in line or '성분명]' in line

            # 같은 줄 또는 다음 줄에 실제 성분명이 있는지 확인
            has_real_ingredient = any(ing in line for ing in REAL_INGREDIENT_PATTERNS)

            # 다음 5줄까지 확인 (성분이 키워드 다음 줄에 있을 수 있음)
            if not has_real_ingredient:
                for look_ahead in range(1, 6):
                    if idx + look_ahead < len(lines):
                        next_line = lines[idx + look_ahead].strip()
                        if any(ing in next_line for ing in REAL_INGREDIENT_PATTERNS):
                            has_real_ingredient = True
                            break

            # '전성분' 정확 매칭이거나 실제 성분명이 발견된 경우
            if exact_keyword or has_real_ingredient:
                in_ingredients = True

                # 키워드가 중간에 있는지 확인 (키워드 앞에도 성분명이 있는 경우)
                # 예: "솔비톨,옥수수전분,...전성분...세라마이드엔피"
                # 단, force_mode=True일 때만 적용 (ALT 텍스트에서 "비타민" 같은 제품명을 성분으로 오인 방지)
                keyword_in_middle = False
                if force_mode:
                    for kw in ['전성분', '성분은', '성분:', '[성분명]', '성분명]']:
                        kw_pos = line.find(kw)
                        if kw_pos > 0:  # 키워드가 라인 시작이 아닌 중간에 있음
                            before_keyword = line[:kw_pos]
                            # 키워드 앞에 실제 성분명이 있는지 확인
                            if any(ing in before_keyword for ing in REAL_INGREDIENT_PATTERNS):
                                keyword_in_middle = True
                                # 키워드를 쉼표로 대체하여 전체 라인을 성분으로 처리
                                cleaned_line = line.replace(kw, ',').strip()
                                ingredient_text.append(cleaned_line)
                                logger.debug(f"키워드 중간 삽입 감지: '{kw}' → 전체 라인 처리")
                                break

                if not keyword_in_middle:
                    # 기존 로직: 키워드 뒤의 텍스트만 추출
                    pattern = r'(?:법에\s*따라|하여야\s*하는|모든\s*성분|전성분|성분은|성분:|\[성분명\]|성분명\]|INGREDIENTS?)\s*(.+)'
                    match = re.search(pattern, line, re.IGNORECASE)

                    if match:
                        remaining = match.group(1).strip()
                        # STOP 키워드가 있으면 그 이전까지만 사용
                        for stop in INGREDIENT_STOP_KEYWORDS:
                            stop_pos = remaining.find(stop)
                            if stop_pos > 0:
                                remaining = remaining[:stop_pos].strip()
                                break
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

    # 전처리 0: 줄바꿈으로 분리된 성분명 합치기
    # 예: "엑사펩타 이드-9" → "엑사펩타이드-9"
    # 한글 + 공백 + 한글(또는 하이픈+숫자)이면 공백 제거
    full_text = re.sub(r'([가-힣])\s+([가-힣])', r'\1\2', full_text)
    full_text = re.sub(r'([가-힣])\s+(이드-?\d+)', r'\1\2', full_text)  # ~펩타 이드-9
    full_text = re.sub(r'([가-힣])\s+(레이트)', r'\1\2', full_text)  # ~카프릴 레이트
    full_text = re.sub(r'(-\d+)\s+([가-힣])', r'\1\2', full_text)  # 폴리글리세릴-10 스테아레이트
    full_text = re.sub(r'([가-힣])\s+(-\d+)', r'\1\2', full_text)  # 폴리글리세릴 -10미리스테이트

    # 전처리 0-1: 영어/숫자 + 공백 + 한글 패턴 합치기
    # 예: "PEG-100 스테아레이트" → "PEG-100스테아레이트"
    # 예: "C12-14 파레스-12" → "C12-14파레스-12"
    full_text = re.sub(r'([A-Za-z0-9-]+)\s+([가-힣])', r'\1\2', full_text)
    # 예: "피이지- 100스테아레이트" → "피이지-100스테아레이트"
    full_text = re.sub(r'([가-힣])-\s+(\d)', r'\1-\2', full_text)

    # 전처리 1: 천 단위 구분자 쉼표 제거 (30,000 → 30000)
    # 주의: 1,2-헥산다이올 같은 화학명은 보존해야 함
    # 천 단위는 쉼표 뒤에 3자리 숫자가 오는 경우만 처리
    full_text = re.sub(r'(\d),(\d{3})(?!\d)', r'\1\2', full_text)

    # 전처리 2: ppm/ppb 농도 패턴 제거 (OCR 줄바꿈 오류 대응)
    # 주의: 쉼표는 제거하지 않음 (성분 구분자 보존)
    # 예: "(30000ppm)," → ","  (쉼표 보존)
    full_text = re.sub(r'\(\s*[\d\.]+\s*(ppm|ppb|PPM|PPB)?\s*\)', '', full_text)
    full_text = re.sub(r'\(\s*[\d\.]+\s*\)', '', full_text)  # 숫자만 있는 괄호도 제거
    full_text = re.sub(r'\bppm\b', '', full_text, flags=re.IGNORECASE)  # 단독 ppm 제거
    full_text = re.sub(r'\bppb\b', '', full_text, flags=re.IGNORECASE)  # 단독 ppb 제거
    # OCR 오류: 괄호 없이 숫자+ppb 붙은 경우 (소듐아스코빌포스페이트500 ppb → 소듐아스코빌포스페이트)
    full_text = re.sub(r'(\D)\d+\s*(ppm|ppb)', r'\1', full_text, flags=re.IGNORECASE)

    # 1차: 쉼표로 분리
    # 주의: 1,2-헥산다이올 같은 화학 번호의 쉼표는 보호
    # 패턴: 숫자,숫자- (예: 1,2- 2,3- 등)
    full_text = re.sub(r'(\d),(\d-)', r'\1@COMMA@\2', full_text)
    parts = full_text.split(',')
    # 보호된 쉼표 복원
    parts = [p.replace('@COMMA@', ',') for p in parts]

    seen = set()  # 중복 방지

    for part in parts:
        part = part.strip()

        if not part:
            continue

        # 핵심 수정: 공백이 있으면 여러 성분일 가능성 → sub_parts 먼저 처리
        # normalize_ingredient_name()이 공백을 제거하므로, 공백 분리를 먼저 수행
        if ' ' in part:
            sub_parts = part.split()
            for sub in sub_parts:
                if len(sub) >= 2:
                    normalized_sub = normalize_ingredient_name(sub)
                    if normalized_sub and len(normalized_sub) >= 2 and normalized_sub not in seen:
                        is_valid, confidence, reason = is_valid_ingredient(normalized_sub)
                        # Phase 3: 소스별 threshold 적용
                        if is_valid and confidence >= threshold:
                            ingredients.append({'ingredient': normalized_sub, 'source': source})
                            seen.add(normalized_sub)
        else:
            # 단일 단어인 경우 전체를 하나의 성분으로 시도
            if len(part) >= 2:
                normalized = normalize_ingredient_name(part)
                if normalized and len(normalized) >= 2:
                    is_valid, confidence, reason = is_valid_ingredient(normalized)
                    # Phase 3: 소스별 threshold 적용
                    if is_valid and confidence >= threshold and normalized not in seen:
                        ingredients.append({'ingredient': normalized, 'source': source})
                        seen.add(normalized)

    return ingredients


def _extract_space_separated(text: str, source: str, threshold: float,
                             real_patterns: list, logger) -> list:
    """
    공백으로 구분된 성분 텍스트 처리 (쉼표가 거의 없는 경우)

    Args:
        text: 원본 텍스트
        source: 출처
        threshold: confidence threshold
        real_patterns: 실제 성분명 패턴 리스트
        logger: 로거

    Returns:
        list: [{'ingredient': str, 'source': str}, ...]
    """
    import re

    ingredients = []
    seen = set()

    # 1. 농도 패턴 제거 (290ppm), (10ppm) 등
    text = re.sub(r'\(\s*\d+\s*(ppm|ppb|%)\s*\)', '', text, flags=re.IGNORECASE)

    # 2. 메타데이터 키워드가 있는 줄 제거 (성분 시작 전)
    lines = text.split('\n')

    # 성분 시작점 찾기
    ingredient_start_idx = None
    ingredient_keywords = ['화장품법에 따라', '전성분', '모든 성분']

    for i, line in enumerate(lines):
        if any(kw in line for kw in ingredient_keywords):
            ingredient_start_idx = i
            break

    # 성분 시작점 이전 줄 제거
    if ingredient_start_idx is not None:
        lines = lines[ingredient_start_idx:]

    # 성분 종료점 찾기
    stop_keywords = ['기능성화장품', '식품의약품', '심사필', '주의사항', '사용방법']
    ingredient_end_idx = len(lines)

    for i, line in enumerate(lines):
        if any(kw in line for kw in stop_keywords):
            ingredient_end_idx = i
            break

    lines = lines[:ingredient_end_idx]

    # 3. 메타데이터 줄 필터링
    filtered_lines = []
    for line in lines:
        line_stripped = line.strip()
        # 메타데이터 키워드가 있으면 건너뜀
        if any(kw in line_stripped for kw in METADATA_KEYWORDS):
            continue
        if line_stripped:
            filtered_lines.append(line_stripped)

    # 4. 줄 끝 하이픈 또는 열린 괄호로 잘린 성분 합치기
    # 예: '코코-카\n  프릴레이트' → '코코-카프릴레이트'
    # 예: '하이드로네이티드폴리(C6-\n  기재해야... 14올레핀)' → 합침
    merged_lines = []
    i = 0
    while i < len(filtered_lines):
        line = filtered_lines[i]

        # 열린 괄호가 있고 닫힌 괄호가 없으면 다음 줄과 합치기 (우선 처리)
        # 예: '하이드로네이티드폴리(C6-' + '기재해야... 14올레핀)'
        open_paren_count = line.count('(') - line.count(')')
        if open_paren_count > 0 and i + 1 < len(filtered_lines):
            next_line = filtered_lines[i + 1]
            # 다음 줄에서 닫힌 괄호 찾기
            if ')' in next_line:
                # 닫힌 괄호까지만 추출
                paren_pos = next_line.find(')')
                paren_content = next_line[:paren_pos + 1]
                remaining = next_line[paren_pos + 1:].strip()

                # 키워드 제거 후 합치기
                paren_content = re.sub(r'기재해야\s*하는\s*모든\s*성분\s*', '', paren_content)
                merged_lines.append(line + paren_content)

                # 나머지가 있으면 별도 줄로 추가
                if remaining:
                    merged_lines.append(remaining)
                i += 2
                continue

        # 하이픈으로 끝나면 다음 줄과 합치기
        if line.endswith('-') and i + 1 < len(filtered_lines):
            next_line = filtered_lines[i + 1]
            # 다음 줄 시작이 한글, 숫자, 영어면 합침
            if next_line and re.match(r'^[가-힣0-9A-Za-z]', next_line):
                merged_lines.append(line + next_line)
                i += 2
                continue

        merged_lines.append(line)
        i += 1

    # 5. 줄 끝 한글 + 다음 줄 접미사 합치기
    # 예: '다이포타슘포스페\n  이트' → '다이포타슘포스페이트'
    final_lines = []
    suffix_pattern = r'^(이트|에이트|레이트|라이드|올|린|드|릴|놀|롤|산|염|논|넨|닌|눌|넬)\b'

    i = 0
    while i < len(merged_lines):
        line = merged_lines[i]

        # 다음 줄이 접미사로 시작하면 합치기
        if i + 1 < len(merged_lines):
            next_line = merged_lines[i + 1]
            if re.match(suffix_pattern, next_line):
                # 현재 줄과 다음 줄 합치기
                final_lines.append(line + ' ' + next_line)
                i += 2
                continue

        final_lines.append(line)
        i += 1

    # 6. 텍스트 합치기
    text = ' '.join(final_lines)

    # 7. 성분 키워드 제거 (성분명 사이에 섞인 키워드)
    text = re.sub(r'화장품법에\s*따라', ' ', text)
    text = re.sub(r'기재해야\s*하는\s*모든\s*성분', ' ', text)
    text = re.sub(r'기재해야\s*하는', ' ', text)
    text = re.sub(r'모든\s*성분', ' ', text)

    # 8. 하이픈 뒤 공백 제거
    # 예: '코코-카 프릴레이트' → '코코-카프릴레이트'
    text = re.sub(r'-\s+([가-힣0-9A-Za-z])', r'-\1', text)

    # 9. 괄호 안의 공백 제거
    # 예: '(C6- 14올레핀)' → '(C6-14올레핀)'
    def remove_space_in_parens(m):
        return m.group(0).replace(' ', '')
    text = re.sub(r'\([^)]+\)', remove_space_in_parens, text)

    # 10. 여러 공백을 하나로
    text = re.sub(r'\s+', ' ', text).strip()

    logger.debug(f"공백구분자 전처리 결과: {text[:200]}...")

    # 11. 성분 시작점 찾기 (첫 번째 유효 성분)
    start_pos = len(text)
    for ing in real_patterns:
        pos = text.find(ing)
        if pos >= 0 and pos < start_pos:
            start_pos = pos

    if start_pos < len(text):
        text = text[start_pos:]

    # 12. 공백으로 분리
    tokens = text.split()

    # 13. 토큰 병합: 하이픈으로 끝나면 다음 토큰과 합치기
    # 예: '코코-카' + '프릴레이트/카프레이트' → '코코-카프릴레이트/카프레이트'
    merged_tokens = []

    # 불완전 접두사 패턴 (하이픈+한글로 끝나는 불완전한 성분명)
    # 예: 코코-카, 라우릴-글, 세틸-피 등
    incomplete_prefix_pattern = r'-[가-힣]{1,2}$'

    # 접미사 시작 패턴 (성분명 접미사로 시작하는 토큰)
    suffix_start_pattern = r'^(프릴|릴레이트|레이트|라이드|글루코|글라이콜|에이트|아마이드|올레|스테아|미리스|팔미)'

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # 하이픈으로 끝나고 다음 토큰이 있으면 합침
        if token.endswith('-') and i + 1 < len(tokens):
            next_token = tokens[i + 1]
            merged_tokens.append(token + next_token)
            i += 2
        # 불완전 접두사 패턴 (-한글1~2자로 끝남) + 다음 토큰이 접미사로 시작
        # 예: '코코-카' + '프릴레이트' → '코코-카프릴레이트'
        elif re.search(incomplete_prefix_pattern, token) and i + 1 < len(tokens):
            next_token = tokens[i + 1]
            if re.match(suffix_start_pattern, next_token):
                merged_tokens.append(token + next_token)
                i += 2
            else:
                merged_tokens.append(token)
                i += 1
        # 한글로 끝나고 다음 토큰이 접미사면 합치기
        elif re.search(r'[가-힣]$', token) and i + 1 < len(tokens):
            next_token = tokens[i + 1]
            if re.match(suffix_pattern, next_token):
                merged_tokens.append(token + next_token)
                i += 2
            else:
                merged_tokens.append(token)
                i += 1
        else:
            merged_tokens.append(token)
            i += 1

    # 14. 추가 알려진 성분 (유효성 검증에서 놓칠 수 있는 성분)
    extra_known = {
        '비니거', '구아이아줄렌', '아이소도데케인', '헥실렌글라이콜',
        '하이드로네이티드폴리', '벤조트라이아졸릴도데실p-크레솔',
        '코코-카프릴레이트/카프레이트', '카프릴릴/카프릴글루코사이드',
    }

    # 15. 각 토큰을 성분으로 검증
    for token in merged_tokens:
        normalized = normalize_ingredient_name(token)
        if normalized and len(normalized) >= 2 and normalized not in seen:
            # 추가 알려진 성분 체크
            if normalized in extra_known:
                ingredients.append({'ingredient': normalized, 'source': source})
                seen.add(normalized)
                continue

            is_valid, conf, reason = is_valid_ingredient(normalized)
            if is_valid and conf >= threshold:
                ingredients.append({'ingredient': normalized, 'source': source})
                seen.add(normalized)

    logger.debug(f"공백구분자 추출 결과: {len(ingredients)}개 성분")
    return ingredients


def extract_product_section(text: str, product_name: str) -> str:
    """
    멀티 제품/옵션이 포함된 텍스트에서 특정 제품의 성분 섹션만 추출

    두 가지 케이스 처리:
    1. 옵션 번호: [52 인더다크], [53 인더라이트] 등
    2. 제품명: 감자캡슐팩, 알로에캡슐팩, 화산송이캡슐팩 등

    Args:
        text: OCR/ALT에서 추출한 전체 텍스트
        product_name: 크롤링 중인 제품명

    Returns:
        해당 제품의 성분 섹션만 추출된 텍스트 (못 찾으면 원본 반환)
    """
    if not text or not product_name:
        return text

    # === 케이스 1: 옵션 번호 기반 ([52 인더다크] 등) ===
    option_match = re.search(r'\[(\d+)\s*([^\]]*)\]', product_name)
    if option_match:
        option_num = option_match.group(1)
        # [숫자 ...] 패턴으로 섹션 분리
        pattern = rf'\[{option_num}\s+[^\]]*\]\s*(.*?)(?=\[\d+\s+|\Z)'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            section = match.group(1).strip()
            if len(section) > 20:  # 충분한 내용이 있는 경우만
                logger.debug(f"옵션 [{option_num}] 섹션 추출: {len(section)}자")
                return section

    # === 케이스 2: 제품명 기반 (감자캡슐팩 등) ===
    # 제품명에서 핵심 키워드 추출 (브랜드, 용량 등 제외)
    keywords = _extract_product_keywords(product_name)

    if keywords:
        # 텍스트에서 해당 제품 섹션 찾기
        section = _find_product_section(text, keywords)
        if section and len(section) > 20:
            logger.debug(f"제품명 키워드 {keywords} 섹션 추출: {len(section)}자")
            return section

    # 못 찾으면 원본 반환
    return text


def _extract_product_keywords(product_name: str) -> list:
    """
    제품명에서 핵심 키워드 추출

    예: "화산송이 캡슐팩" → ["화산송이", "캡슐팩"]
        "본셉 젤 아이라이너 [01 젤블랙]" → ["젤", "아이라이너"]
    """
    # 옵션 부분 제거
    name = re.sub(r'\[\d+[^\]]*\]', '', product_name).strip()

    # 브랜드명 제거 (앞쪽 2-4글자 단어가 브랜드인 경우가 많음)
    # 흔한 브랜드: 다이소, 본셉, 머지, 프릴루드 딘토, 더봄, 베리썸 등
    brands = ['다이소', '본셉', '머지', '프릴루드', '딘토', '더봄', '베리썸',
              '파넬', '에뛰드', '플레이', '태그', '펠트', '밀크터치', '꽁래쉬',
              'VT', '메디필', '제이엠솔루션', '린제이', '채비공간', '비프루브']

    for brand in brands:
        name = re.sub(rf'^{re.escape(brand)}\s*', '', name, flags=re.IGNORECASE)

    # 용량/단위 제거
    name = re.sub(r'\d+\s*(ml|g|mg|매|개입|P)\b', '', name, flags=re.IGNORECASE)

    # 괄호 내용 제거
    name = re.sub(r'\([^)]*\)', '', name)

    # 공백 정규화
    name = ' '.join(name.split())

    # 핵심 단어 추출 (2글자 이상)
    words = [w for w in name.split() if len(w) >= 2]

    # 너무 일반적인 단어 제거
    generic = {'슬림', '프로', '수퍼', '에어', '픽싱', '롱', '듀얼', '트리플'}
    keywords = [w for w in words if w not in generic]

    return keywords[:3]  # 최대 3개 키워드


def _find_product_section(text: str, keywords: list) -> str:
    """
    텍스트에서 키워드와 매칭되는 제품 섹션 찾기

    예: 키워드 ["화산송이", "캡슐팩"]
        텍스트에서 "화산송이캡슐팩" 또는 "화산송이 캡슐팩" 찾기
    """
    if not keywords:
        return text

    # 공백 무시 패턴 생성
    # "화산송이캡슐팩" → "화\s*산\s*송\s*이\s*캡\s*슐\s*팩"
    def make_flexible_pattern(keyword):
        return r'\s*'.join(re.escape(c) for c in keyword)

    # 모든 키워드가 포함된 줄 찾기 (공백 무시)
    lines = text.split('\n')
    target_line_idx = None

    for i, line in enumerate(lines):
        line_normalized = line.replace(' ', '').lower()

        # 모든 키워드가 줄에 포함되어 있는지 확인
        all_found = all(
            kw.replace(' ', '').lower() in line_normalized
            for kw in keywords
        )

        if all_found:
            target_line_idx = i
            break

        # 키워드가 연속으로 나타나는지 확인 (예: "화산송이캡슐팩")
        combined = ''.join(keywords).replace(' ', '').lower()
        if combined in line_normalized:
            target_line_idx = i
            break

    if target_line_idx is None:
        return text

    # 해당 줄부터 다음 제품명까지 추출
    # 다음 제품명 패턴: 한글 2-6글자 + (캡슐팩|팩|라이너|섀도우|마스카라 등)
    product_name_pattern = r'^[가-힣]{2,10}(캡슐팩|팩|라이너|섀도우|마스카라|브로우|세럼)\s*$'

    section_lines = []
    for j in range(target_line_idx + 1, len(lines)):
        line = lines[j].strip()

        # 다음 제품명이 나오면 중단
        if re.match(product_name_pattern, line):
            break

        # 빈 줄이 연속 2개 이상이면 중단
        if not line:
            if section_lines and not section_lines[-1]:
                break

        section_lines.append(line)

    # 끝의 빈 줄 제거
    while section_lines and not section_lines[-1]:
        section_lines.pop()

    return '\n'.join(section_lines)
