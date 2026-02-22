"""
키워드 분석 모듈

리뷰 키워드의 빈도 계산, 그룹 비교, 카테고리별 필터링 등을 수행합니다.
"""

from collections import Counter
from typing import List, Dict, Tuple, Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re


# 키워드 카테고리 사전 (정규표현식 패턴)
KEYWORD_CATEGORIES = {
    '가성비': [
        r'가성비', r'싸[다고게요]', r'저렴[하한]?', r'가격', r'저가',
        r'착[하한]', r'가격대', r'합리', r'경제', r'부담[없안적]',
        r'비싸[다고]?', r'천원', r'가격대비', r'\d+원', r'행사', r'할인'
    ],
    '품질[긍정]': [
        r'품질\s?[이가]?\s?(?:좋|괜찮|최고|우수)', 
        r'효과\s?[가]?\s?(?:좋|괜찮|최고|있[다어요네])', 
        r'(?<!안\s)(?<!안\s)좋[다아고은아요네해]', 
        r'(?<!불)만족[스럽하해해요]', r'훌륭[하한해해요]', 
        r'퀄리티\s?[가]?\s?(?:좋|최고|우수)', r'우수[하한해해요]',
        r'괜찮[다아고게네요아]', r'최고', r'완벽[하한해해요]', r'추천[해하해요]', r'대박', r'신기[하한해해요]',
        r'놀랍[다고게게요]', r'기대\s?이상|기대보다\s?좋', 
        r'성능\s?[이가]?\s?(?:좋|괜찮|최고)', 
        r'쓸만[하한해해요]', r'생각보다\s?좋', r'가심비', r'다이소\s?최고|다이소만',
        r'인생템', r'정착[했해해요]', r'재구매[할해해요]', 
        r'광채\s?[가]?\s?좋|광채\s?나[요네]', r'탄력\s?[이]?\s?좋|탄력\s?있[다어요네]', 
        r'보습[력]?\s?(?:좋|최고)', r'진정[효과]?\s?좋|진정\s?[이]?\s?[되돼되요]'
    ],
    '품질[부정]': [
        r'트러블', r'자극', r'여드름', r'뒤집', r'알[레러]지', r'알레르기' 
        r'별로(?!\s?(?:안|인데\s?(?:좋|괜찮)))',
        r'따[갑끔]', r'쓰리[다고]', r'안\s?맞|맞지\s?않', r'부작용',
        r'피부[에]?\s?안\s?좋|피부\s?트러블',
        r'가렵', r'주의', r'발진', r'뾰루지', r'염증', 
        r'불편[하해](?!\s?지\s?(?:않|는\s?않))', 
        r'심[하한해](?!\s?지\s?않)',
        r'돈[이]?\s?아깝', r'재구매[는]?\s?안|재구매\s?없', r'최악', r'실망', 
        r'무겁[다고](?!\s?지\s?않)', r'무거워(?!\s?하지\s?않)', r'아쉬[워움]'
    ],
    '심미[긍정]': [
        r'디자인[이]?\s?(?:좋|예쁘|멋지|깔끔|세련)', 
        r'패키지[가]?\s?(?:좋|예쁘|깔끔|고급)', 
        r'용기[가]?\s?(?:예쁘|좋)', 
        r'(?<!안\s)예쁘[다고게네요해]', r'(?<!안\s)예뻐[요서며]', 
        r'(?<!안\s)이쁘[다고게네요해]', r'(?<!안\s)이뻐[요서며]',
        r'발색[이]?\s?(?:좋|예쁘|최고)', 
        r'색감[이]?\s?(?:좋|예쁘)', 
        r'색상[이]?\s?(?:예쁘|좋)|색\s?예쁘', 
        r'색깔[이]?\s?(?:예쁘|좋)',
        r'외관[이]?\s?(?:좋|예쁘|깔끔)', 
        r'모양[이]?\s?(?:예쁘|좋)',
        r'케이스[가]?\s?(?:예쁘|좋)', 
        r'포장[이]?\s?(?:좋|예쁘|깔끔)',
        r'귀엽[다고게네요해]|귀여워[요서]', r'세련[되된됐돼]|세련미', 
        r'깔끔[하한해해요]', r'심플[하한해해요]', 
        r'고급[스럽져스러워]|고급미|고급\s?느낌', r'럭셔리', 
        r'멋지[다고게네요해]|멋져[요서]', r'영롱[하한해해요]', 
        r'무드[가]?\s?좋', r'감성[적]', r'취향\s?저격'
    ],
    '심미[부정]': [
        r'디자인[이]?\s?별로|디자인\s?안\s?좋', r'패키지[가]?\s?별로|패키지\s?안\s?좋',
        r'안\s?예쁘', r'안\s?이쁘', r'못\s?생겼', r'촌스러워요',
        r'색상[이]\s?별로|색[이]\s?별로', r'색깔[이]\s?별로',
        r'저렴해\s?보[여이]|저렴하게\s?보[여이]', r'싸구려\s?같',
        r'품질\s?떨어져\s?보[여이]', r'조악', r'택배\s?상자.*허술|포장.*허술',
        r'포장[이]?\s?별로', r'용기[가]?\s?별로', r'케이스[가]?\s?별로',
        r'발색[이]?\s?안\s?좋|발색\s?별로', r'아쉬[워움]'
        r'외관[이]?\s?별로', r'보기\s?안\s?좋', r'피부.*지저분|얼굴.*지저분'
    ],
    '편의[긍정]': [
        r'편[하한해해요](?!\s?지\s?않)', r'간편[하한해해요](?!\s?지\s?않)', 
        r'사용감\s?[이가]?\s?(?:좋|괜찮)', 
        r'흡수\s?[가]?\s?(?:좋|빠르)|잘\s?흡수|흡수력\s?(?:좋|괜찮)', 
        r'발림성?\s?[이가]?\s?(?:좋|부드럽)|잘\s?발[라려]',
        r'쉽[다고게요네](?!.*(?:안|않|없|못))', r'빠르[다고게요네]', 
        r'(?<!불)편리[하한해해요]', r'손쉽[다게요네]', 
        r'부드럽[다고게요네]|부드러워[요서]', r'순[하한해해요]', 
        r'가볍[다고게요네]|가벼워[요서]',
        r'촉촉[하한해해요]|촉촉해[요서]', 
        r'여행[용]?\s?(?:좋|편)|휴대[하기]?\s?(?:좋|편)',
        r'텍스처\s?[가]?\s?좋', r'발리[다고]', 
        r'밀착\s?[이]?\s?좋|밀착력\s?좋', 
        r'산뜻[하한해해요]', r'데일리[로]?\s?좋'
    ],
    '물류[긍정]': [
        r'배송\s?빠르[다고게요]', r'빠른\s?배송', r'칼배송',
        r'재고\s?있[다어었]', r'품절\s?안', r'구하기\s?쉽[다더게]',
        r'매장[에]?\s?있[었다어]', r'오프라인[에]?\s?있[었다어]',
        r'입고\s?빠르[다더게요]', r'빠른\s?입고', r'재입고\s?빨[랐]',
        r'구매\s?쉽[다더]', r'사기\s?쉽[다더]', r'찾기\s?쉽[다더]',
        r'바로\s?배송', r'당일\s?배송', r'익일\s?배송',
        r'택배\s?빠르[다더게요]', r'빠른\s?택배',
        r'재고\s?많[았다]',  # "많이 뽑아주세요" 제외
        r'예약\s?가능', r'주문\s?가능'
    ],
    '편의[부정]': [
        r'(?<!안\s)불편[하해함](?!\s?지\s?(?:는|도)?\s?않)', 
        r'사용(?:하기)?\s?(?:어렵[다고게]|어려[워요운])(?!\s?지\s?않)', 
        r'쓰기\s?불편',
        r'발림성?\s?[이]?\s?(?:안\s?좋|별로)|잘\s?안\s?발[라려]',
        r'흡수\s?[가]?\s?(?:안\s?[되돼]|별로)',
        r'무겁[다고게]|무거워[요서](?!\s?하지\s?않)', 
        r'너무\s?끈적|매우\s?끈적|너무\s?번들|매우\s?번들', 
        r'뻑뻑(?:하[다고게]|해[요서])(?!\s?지\s?않)',
        r'피부.*(?:너무\s?건조|매우\s?건조)|제품.*(?:너무\s?건조|매우\s?건조)|건조[해하].*(?:불편|안\s?좋)', 
        r'당기[는다고]', r'땅기[는다고]',
        r'사용감\s?[이가]?\s?(?:별로|안\s?좋)',
        r'텍스처\s?[가]?\s?(?:별로|안\s?좋)',
        r'밀착\s?[이]?\s?(?:안\s?[되돼]|별로)',
        r'여행[용]?\s?불편|휴대[하기]?\s?불편',
        r'부드럽지\s?않', 
        r'딱딱(?:하[다고게]|해[요서]|함)(?!\s?지\s?않)', 
        r'까칠(?:하[다고게]|해[요서])(?!\s?지\s?않)',
        r'갈라[지짐]', 
        r'(?<!안\s)뭉[치쳐침]', 
        r'덜[어렁걱]', r'아쉬[워움]'
    ],
    '물류[부정]': [
        r'배송\s?(?:느리[다고게]|느려[요서]|늦[다고게어]|안\s?와|오래\s?걸리|최악|별로)', 
        r'재고\s?[가는]?\s?없', r'품절', r'매진', 
        r'입고\s?(?:늦[다고게어]|안\s?[되돼]|오래\s?걸리|느리[다고]|느려[요서])', 
        r'택배\s?(?:느리[다고게]|느려[요서]|늦[다고게어]|안\s?와|오래\s?걸리|최악|별로)',
        r'예약\s?(?:안\s?[되돼]|불가|막혔|어렵[다고게]|어려[워요운])', 
        r'구하기\s?(?:어렵[다고게]|어려[워요운]|힘들[다고게어요운])', 
        r'찾기\s?(?:어렵[다고게]|어려[워요운]|힘들[다고게어요운])',
        r'매장[에서]?\s?없', r'상품\s?[이]?\s?없', r'아쉬[워움]'
        r'온라인\s?(?:품절|재고\s?없)', r'오프라인\s?(?:품절|재고\s?없)'
    ],
    '희소성': [
        r'품절', r'재고[가는]?\s?없|재고\s?없', r'매진',
        r'구하기\s?어렵|구하기\s?힘들',
        r'구매[를]?\s?못', r'살\s?수\s?없|사기\s?어렵',
        r'없어[서]?\s?아쉽', r'제품[이]?\s?없', r'상품[이]?\s?없',
        r'한정', r'품절\s?대란', r'재입고', r'입고\s?언제',
        r'매장[에서]?\s?없|매장\s?없', r'오프라인[에서]?\s?없',
        r'구경\s?할\s?수\s?없', r'찾기\s?어렵|찾을\s?수\s?없',
        r'웨이팅', r'오픈런', r'귀한'
    ],
    '듀프': [
        r'듀프', r'대체재', r'비슷[하한해]', r'똑같[다아]', r'흡사',
        r'닮[았다아]', r'유사[하한]',
        r'차이[가]?\s?없',
        r'저렴이', r'비싼\s?거\s?필요\s?없|비싼\s?제품\s?필요\s?없',
        r'가성비템', r'짭', r'카피',
        # 맥락 기반 패턴 (브랜드/제품 대체 의미만)
        r'[가-힣]+\s?대신\s?[써쓰사용]',  # "XX 대신 써요"
        r'비교[했하면]\s?[때도]',  # "비교했을 때", "비교하면"
        r'[비슷똑]\s?같[은다]'  # "비슷 같은", "똑 같은" (일반 "같다"는 제외)
    ],

    '로드샵브랜드': [
        r'올리브영', r'올영',
        r'무신사\s?뷰티', r'컬리\s?뷰티',
        r'클리오', r'롬앤', r'페리페라', r'이니스프리', r'에뛰드', r'미샤',
        r'시코르', r'세포라', r'로드샵', r'백화점', r'면세점',
        r'카카오\s?선물', r'홈쇼핑', r'공홈', r'직구', r'해외\s?직구',
        r'닥터지', r'달바', r'맥', r'MAC', r'나스', r'헤라',
        r'어퓨', r'A\'pieu', r'토니모리', r'홀리카', r'더페이스샵', r'스킨푸드',
        r'라네즈', r'아이오페', r'에이프릴스킨', r'라운드랩', r'코스알엑스',
        r'비디비치', r'토르', r'독도', r'더마토리', r'피지오겔',
        r'닥터[^\s]{0,2}(?:[가-힣]|$)',  # 닥터지, 닥터자르트 등
        r'아누아', r'바이오더마', r'토리든', r'3ce',
        r'제로이드', r'이지듀', r'에스트라'
    ],
    '브랜드개념': [
        r'브랜드', r'명품', r'고가', r'럭셔리', r'저가', r'가성비템'
    ],
    '고가브랜드': [
        r'설화수', r'라\s?메르', r'랑콤',
        # 에스티로더 (에스티로더, 에스티 로더, 에스티)
        r'에스티\s?로더|(?<![가-힣])에스티(?![가-힣로])',
        r'시세이도', r'크리니크', r'클라랑스', r'키엘', r'디올',
        r'샤넬', r'톰\s?포드', r'겔랑', r'조\s?말론', r'이솝',
        r'라\s?프레리', r'시슬리', r'끌레드뽀\s?보떼', r'헬레나\s?루빈스타인',
        # 입생로랑 (입생로랑, 생로랑, 입생, YSL)
        r'입생\s?로랑|생로랑|(?<![가-힣])입생(?![가-힣로])|YSL',
        r'지방시', r'아르마니\s?뷰티', r'바비\s?브라운',
        r'딥티크', r'바이레도', r'메종\s?마르지엘라', r'르\s?라보',
        r'산타마리아\s?노벨라', r'크리드', r'킬리안', r'프레데릭\s?말'
    ],
    '피부타입': [
        r'건성', r'지성', r'복합성', r'민감성', r'수부지', r'속건조', r'홍조', r'좁쌀'
    ]

}

# 정규표현식 패턴용 카테고리 (하위 호환성 유지)
KEYWORD_CATEGORIES_LEGACY = {
    '가성비': ['가성비', '싸다', '저렴', '가격', '할인', '세일', '저가', '착하다'],
    '품질': ['품질', '효과', '좋다', '만족', '훌륭', '퀄리티', '우수', '괜찮다', '최고', '완벽', '추천'],
    '심미': ['디자인', '패키지', '용기', '예쁘다', '발색', '색감', '색상'],
    '편의': ['편하다', '간편', '사용감', '흡수', '발림성', '쉽다', '빠르다', '편리'],
    '물류': ['배송', '재고', '품절', '매진', '한정', '입고', '택배'],
    '품질불만': ['트러블', '자극', '여드름', '별로', '따갑다', '안맞다', '부작용'],
    '듀프': ['듀프', '대체', '비슷', '똑같', '흡사', '대신', '유사'],
    '브랜드': ['브랜드', '명품', '고가', '설화수', '라메르', '랑콤', '에스티로더', '올리브영', '올영']
}


def calculate_keyword_frequency(
    tokens_list: List[List[str]],
    top_n: int = 50
) -> pd.DataFrame:
    """
    토큰 리스트에서 키워드 빈도를 계산합니다.
    
    Parameters:
    -----------
    tokens_list : List[List[str]]
        토큰 리스트의 리스트
    top_n : int
        상위 N개 키워드
        
    Returns:
    --------
    pd.DataFrame
        키워드와 빈도가 포함된 데이터프레임
        
    Examples:
    ---------
    >>> tokens_list = [['좋다', '제품'], ['좋다', '가격']]
    >>> calculate_keyword_frequency(tokens_list, top_n=3)
       keyword  frequency
    0   좋다         2
    1   제품         1
    2   가격         1
    """
    # 모든 토큰을 하나의 리스트로 합치기
    all_tokens = []
    for tokens in tokens_list:
        all_tokens.extend(tokens)
    
    # 빈도 계산
    counter = Counter(all_tokens)
    
    # 상위 N개 추출
    most_common = counter.most_common(top_n)
    
    # 데이터프레임 생성
    df = pd.DataFrame(most_common, columns=['keyword', 'frequency'])
    
    return df


def compare_keyword_groups(
    group1_tokens: List[List[str]],
    group2_tokens: List[List[str]],
    top_n: int = 30,
    group1_name: str = 'Group1',
    group2_name: str = 'Group2'
) -> pd.DataFrame:
    """
    두 그룹 간 키워드 빈도를 비교합니다.
    
    Parameters:
    -----------
    group1_tokens : List[List[str]]
        그룹 1의 토큰 리스트
    group2_tokens : List[List[str]]
        그룹 2의 토큰 리스트
    top_n : int
        비교할 상위 키워드 수
    group1_name : str
        그룹 1의 이름
    group2_name : str
        그룹 2의 이름
        
    Returns:
    --------
    pd.DataFrame
        키워드별 그룹 빈도 비교 데이터프레임
    """
    # 각 그룹의 빈도 계산
    df1 = calculate_keyword_frequency(group1_tokens, top_n=top_n * 2)
    df2 = calculate_keyword_frequency(group2_tokens, top_n=top_n * 2)
    
    # 컬럼명 변경
    df1 = df1.rename(columns={'frequency': f'{group1_name}_freq'})
    df2 = df2.rename(columns={'frequency': f'{group2_name}_freq'})
    
    # 두 데이터프레임 병합
    df_merged = pd.merge(df1, df2, on='keyword', how='outer').fillna(0)
    
    # 전체 빈도 계산
    df_merged['total_freq'] = df_merged[f'{group1_name}_freq'] + df_merged[f'{group2_name}_freq']
    
    # 차이 계산
    df_merged['diff'] = df_merged[f'{group1_name}_freq'] - df_merged[f'{group2_name}_freq']
    df_merged['abs_diff'] = abs(df_merged['diff'])
    
    # 상위 N개 추출 (차이가 큰 순서대로)
    df_result = df_merged.nlargest(top_n, 'abs_diff')
    
    # 정렬 (그룹1 빈도가 높은 순)
    df_result = df_result.sort_values('diff', ascending=False)
    
    return df_result.reset_index(drop=True)


def filter_keywords_by_category(
    tokens: List[str],
    keyword_dict: Optional[Dict[str, List[str]]] = None
) -> Dict[str, int]:
    """
    토큰을 카테고리별로 필터링하여 빈도를 계산합니다.
    
    Parameters:
    -----------
    tokens : List[str]
        토큰 리스트
    keyword_dict : Optional[Dict[str, List[str]]]
        카테고리별 키워드 사전 (None이면 KEYWORD_CATEGORIES 사용)
        
    Returns:
    --------
    Dict[str, int]
        카테고리별 키워드 빈도
        
    Examples:
    ---------
    >>> tokens = ['가성비', '좋다', '디자인', '예쁘다']
    >>> filter_keywords_by_category(tokens)
    {'가성비': 1, '품질': 1, '심미': 2, ...}
    """
    if keyword_dict is None:
        keyword_dict = KEYWORD_CATEGORIES
    
    # 카테고리별 빈도 초기화
    category_counts = {category: 0 for category in keyword_dict.keys()}
    
    # 각 토큰을 카테고리와 매칭
    for token in tokens:
        for category, keywords in keyword_dict.items():
            if token in keywords:
                category_counts[category] += 1
    
    return category_counts


def calculate_category_frequency_for_reviews(
    tokens_list: List[List[str]],
    keyword_dict: Optional[Dict[str, List[str]]] = None
) -> pd.DataFrame:
    """
    여러 리뷰의 카테고리별 키워드 빈도를 계산합니다.
    
    Parameters:
    -----------
    tokens_list : List[List[str]]
        토큰 리스트의 리스트
    keyword_dict : Optional[Dict[str, List[str]]]
        카테고리별 키워드 사전
        
    Returns:
    --------
    pd.DataFrame
        카테고리별 총 빈도 데이터프레임
    """
    if keyword_dict is None:
        keyword_dict = KEYWORD_CATEGORIES
    
    # 전체 카테고리 빈도 누적
    total_counts = {category: 0 for category in keyword_dict.keys()}
    
    for tokens in tokens_list:
        category_counts = filter_keywords_by_category(tokens, keyword_dict)
        for category, count in category_counts.items():
            total_counts[category] += count
    
    # 데이터프레임 생성
    df = pd.DataFrame([
        {'category': category, 'frequency': count}
        for category, count in total_counts.items()
    ])
    
    # 빈도 내림차순 정렬
    df = df.sort_values('frequency', ascending=False).reset_index(drop=True)
    
    return df


def extract_tfidf_keywords(
    documents: List[str],
    top_n: int = 20,
    max_features: int = 1000
) -> pd.DataFrame:
    """
    TF-IDF를 사용하여 중요 키워드를 추출합니다.
    
    Parameters:
    -----------
    documents : List[str]
        문서 리스트 (각 문서는 공백으로 구분된 토큰 문자열)
    top_n : int
        추출할 상위 키워드 수
    max_features : int
        TF-IDF 벡터화 시 고려할 최대 특성 수
        
    Returns:
    --------
    pd.DataFrame
        키워드와 TF-IDF 점수가 포함된 데이터프레임
    """
    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        token_pattern=r'(?u)\b\w+\b'  # 한글 토큰 지원
    )
    
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # 평균 TF-IDF 점수 계산
    feature_names = vectorizer.get_feature_names_out()
    avg_tfidf = tfidf_matrix.mean(axis=0).A1
    
    # 키워드와 점수 매핑
    keyword_scores = list(zip(feature_names, avg_tfidf))
    
    # 점수 내림차순 정렬
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 상위 N개 추출
    top_keywords = keyword_scores[:top_n]
    
    # 데이터프레임 생성
    df = pd.DataFrame(top_keywords, columns=['keyword', 'tfidf_score'])
    
    return df


def find_keyword_cooccurrence(
    tokens_list: List[List[str]],
    target_keyword: str,
    top_n: int = 20
) -> pd.DataFrame:
    """
    특정 키워드와 함께 자주 등장하는 키워드를 찾습니다.
    
    Parameters:
    -----------
    tokens_list : List[List[str]]
        토큰 리스트의 리스트
    target_keyword : str
        대상 키워드
    top_n : int
        상위 N개 공출현 키워드
        
    Returns:
    --------
    pd.DataFrame
        공출현 키워드와 빈도
    """
    cooccur_keywords = []
    
    for tokens in tokens_list:
        if target_keyword in tokens:
            # 대상 키워드와 함께 등장한 다른 키워드들
            cooccur_keywords.extend([t for t in tokens if t != target_keyword])
    
    # 빈도 계산
    counter = Counter(cooccur_keywords)
    most_common = counter.most_common(top_n)
    
    # 데이터프레임 생성
    df = pd.DataFrame(most_common, columns=['cooccur_keyword', 'frequency'])
    
    return df


def calculate_category_ratio(
    tokens_list: List[List[str]],
    category_pairs: List[Tuple[str, str]],
    keyword_dict: Optional[Dict[str, List[str]]] = None
) -> Dict[str, float]:
    """
    두 카테고리 간 빈도 비율을 계산합니다.

    긍정/부정이 분리된 카테고리는 자동으로 합산합니다.
    예: '품질' → '품질[긍정]' + '품질[부정]'

    Parameters:
    -----------
    tokens_list : List[List[str]]
        토큰 리스트의 리스트
    category_pairs : List[Tuple[str, str]]
        비교할 카테고리 쌍 리스트 예: [('품질', '가성비')]
    keyword_dict : Optional[Dict[str, List[str]]]
        카테고리별 키워드 사전

    Returns:
    --------
    Dict[str, float]
        카테고리 쌍별 비율 (category1 / category2)
    """
    if keyword_dict is None:
        keyword_dict = KEYWORD_CATEGORIES

    # 카테고리별 빈도 계산
    df_freq = calculate_category_frequency_for_reviews(tokens_list, keyword_dict)
    freq_dict = dict(zip(df_freq['category'], df_freq['frequency']))

    # 긍정/부정 합산을 위한 헬퍼 함수
    def get_category_frequency(category_name):
        """카테고리 빈도 조회 (긍정/부정 자동 합산)"""
        # 직접 매칭되는 카테고리가 있으면 사용
        if category_name in freq_dict:
            return freq_dict[category_name]

        # 긍정/부정 합산
        positive = freq_dict.get(f'{category_name}[긍정]', 0)
        negative = freq_dict.get(f'{category_name}[부정]', 0)

        return positive + negative

    # 비율 계산
    ratios = {}
    for cat1, cat2 in category_pairs:
        freq1 = get_category_frequency(cat1)
        freq2 = get_category_frequency(cat2)

        if freq2 == 0:
            ratio = float('inf') if freq1 > 0 else 0
        else:
            ratio = freq1 / freq2

        ratios[f'{cat1}/{cat2}'] = ratio

    return ratios


def extract_reviews_with_keywords(
    df_reviews: pd.DataFrame,
    keywords: List[str],
    text_column: str = 'text'
) -> pd.DataFrame:
    """
    특정 키워드가 포함된 리뷰를 추출합니다.
    
    Parameters:
    -----------
    df_reviews : pd.DataFrame
        리뷰 데이터프레임
    keywords : List[str]
        검색할 키워드 리스트 (정규표현식 패턴 지원)
    text_column : str
        텍스트 컬럼 이름
        
    Returns:
    --------
    pd.DataFrame
        키워드가 포함된 리뷰 데이터프레임
    """
    # 각 패턴을 괄호로 묶어서 독립적인 정규표현식 그룹으로 만듦
    pattern = '|'.join(f'({kw})' for kw in keywords)
    
    # 키워드가 포함된 행 필터링 (정규표현식 사용)
    mask = df_reviews[text_column].str.contains(pattern, case=False, na=False, regex=True)
    
    return df_reviews[mask].copy()


def match_category_patterns_in_text(
    text: str,
    keyword_dict: Optional[Dict[str, List[str]]] = None
) -> Dict[str, int]:
    """
    텍스트에서 정규표현식 패턴을 사용하여 카테고리별 매칭 횟수를 계산합니다.
    
    Parameters:
    -----------
    text : str
        분석할 텍스트
    keyword_dict : Optional[Dict[str, List[str]]]
        카테고리별 정규표현식 패턴 사전
        
    Returns:
    --------
    Dict[str, int]
        카테고리별 매칭 횟수
    """
    if keyword_dict is None:
        keyword_dict = KEYWORD_CATEGORIES
    
    if not isinstance(text, str):
        return {category: 0 for category in keyword_dict.keys()}
    
    category_matches = {}
    for category, patterns in keyword_dict.items():
        match_count = 0
        for pattern in patterns:
            matches = re.findall(pattern, text)
            match_count += len(matches)
        category_matches[category] = match_count
    
    return category_matches


def calculate_category_frequency_regex(
    df_reviews: pd.DataFrame,
    text_column: str = 'text',
    keyword_dict: Optional[Dict[str, List[str]]] = None
) -> pd.DataFrame:
    """
    정규표현식 패턴을 사용하여 카테고리별 키워드 빈도를 계산합니다.
    
    Parameters:
    -----------
    df_reviews : pd.DataFrame
        리뷰 데이터프레임
    text_column : str
        텍스트 컬럼명
    keyword_dict : Optional[Dict[str, List[str]]]
        카테고리별 정규표현식 패턴 사전
        
    Returns:
    --------
    pd.DataFrame
        카테고리별 총 빈도 데이터프레임
    """
    if keyword_dict is None:
        keyword_dict = KEYWORD_CATEGORIES
    
    # 전체 카테고리 빈도 누적
    total_counts = {category: 0 for category in keyword_dict.keys()}
    
    for text in df_reviews[text_column]:
        category_counts = match_category_patterns_in_text(text, keyword_dict)
        for category, count in category_counts.items():
            total_counts[category] += count
    
    # 데이터프레임 생성
    df = pd.DataFrame([
        {'category': category, 'frequency': count}
        for category, count in total_counts.items()
    ])
    
    # 빈도 내림차순 정렬
    df = df.sort_values('frequency', ascending=False).reset_index(drop=True)
    
    return df


def has_category_pattern(
    text: str,
    category: str,
    keyword_dict: Optional[Dict[str, List[str]]] = None
) -> bool:
    """
    텍스트에 특정 카테고리의 패턴이 포함되어 있는지 확인합니다.
    
    Parameters:
    -----------
    text : str
        확인할 텍스트
    category : str
        카테고리 이름
    keyword_dict : Optional[Dict[str, List[str]]]
        카테고리별 정규표현식 패턴 사전
        
    Returns:
    --------
    bool
        패턴 포함 여부
    """
    if keyword_dict is None:
        keyword_dict = KEYWORD_CATEGORIES
    
    if not isinstance(text, str) or category not in keyword_dict:
        return False
    
    patterns = keyword_dict[category]
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    
    return False


def analyze_scarcity_pattern(
    df_reviews: pd.DataFrame,
    text_column: str = 'text',
    rating_column: str = 'rating',
    high_rating_threshold: int = 4,
    category: str = '희소성',
    keyword_dict: Optional[Dict[str, List[str]]] = None
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    희소성 패턴을 분석합니다.
    
    "제품은 좋은데 구하기 힘들다" 패턴: 희소성 키워드 + 높은 평점
    
    Parameters:
    -----------
    df_reviews : pd.DataFrame
        리뷰 데이터프레임
    text_column : str
        텍스트 컬럼명
    rating_column : str
        평점 컬럼명
    high_rating_threshold : int
        높은 평점 기준 (기본: 4점)
    category : str
        희소성 카테고리 이름 (기본: '희소성')
    keyword_dict : Optional[Dict[str, List[str]]]
        카테고리별 패턴 사전
        
    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, any]]
        (희소성+높은평점 리뷰 데이터프레임, 통계 딕셔너리)
    """
    if keyword_dict is None:
        keyword_dict = KEYWORD_CATEGORIES
    
    # 희소성 패턴 체크
    df_reviews['has_scarcity'] = df_reviews[text_column].apply(
        lambda x: has_category_pattern(x, category, keyword_dict)
    )
    
    # 통계 계산
    total_count = len(df_reviews)
    scarcity_count = df_reviews['has_scarcity'].sum()
    scarcity_ratio = scarcity_count / total_count if total_count > 0 else 0
    
    # 희소성 + 높은 평점
    df_good_but_hard = df_reviews[
        (df_reviews['has_scarcity']) & 
        (df_reviews[rating_column] >= high_rating_threshold)
    ].copy()
    
    good_but_hard_count = len(df_good_but_hard)
    good_but_hard_ratio = good_but_hard_count / total_count if total_count > 0 else 0
    high_rating_ratio_in_scarcity = (
        good_but_hard_count / scarcity_count if scarcity_count > 0 else 0
    )
    
    # 통계 딕셔너리
    stats = {
        'total_count': total_count,
        'scarcity_count': scarcity_count,
        'scarcity_ratio': scarcity_ratio,
        'good_but_hard_count': good_but_hard_count,
        'good_but_hard_ratio': good_but_hard_ratio,
        'high_rating_ratio_in_scarcity': high_rating_ratio_in_scarcity,
        'high_rating_threshold': high_rating_threshold
    }
    
    return df_good_but_hard, stats


def print_pattern_statistics(
    df_reviews: pd.DataFrame,
    category: str,
    text_column: str = 'text',
    keyword_dict: Optional[Dict[str, List[str]]] = None,
    top_n: Optional[int] = None
) -> Dict[str, int]:
    """
    특정 카테고리의 패턴별 매칭 빈도를 출력합니다.
    
    Parameters:
    -----------
    df_reviews : pd.DataFrame
        리뷰 데이터프레임
    category : str
        카테고리 이름
    text_column : str
        텍스트 컬럼명
    keyword_dict : Optional[Dict[str, List[str]]]
        카테고리별 패턴 사전
    top_n : Optional[int]
        상위 N개만 출력 (None이면 전체)
        
    Returns:
    --------
    Dict[str, int]
        패턴별 빈도 딕셔너리
    """
    if keyword_dict is None:
        keyword_dict = KEYWORD_CATEGORIES
    
    if category not in keyword_dict:
        print(f"카테고리 '{category}'를 찾을 수 없습니다.")
        return {}
    
    patterns = keyword_dict[category]
    pattern_counts = {}
    
    for pattern in patterns:
        count = df_reviews[text_column].apply(
            lambda x: bool(re.search(pattern, str(x)))
        ).sum()
        if count > 0:
            pattern_counts[pattern] = count
    
    # 정렬
    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
    
    # 출력
    print(f"\n[{category} 패턴별 매칭 빈도]")
    if top_n:
        sorted_patterns = sorted_patterns[:top_n]
    
    for pattern, count in sorted_patterns:
        print(f"  {pattern}: {count:,}개")
    
    return pattern_counts


def print_scarcity_samples(
    df_good_but_hard: pd.DataFrame,
    text_column: str = 'text',
    rating_column: str = 'rating',
    n_samples: int = 10,
    max_length: int = 150
):
    """
    희소성 패턴 샘플 리뷰를 출력합니다.
    
    Parameters:
    -----------
    df_good_but_hard : pd.DataFrame
        희소성 패턴 리뷰 데이터프레임
    text_column : str
        텍스트 컬럼명
    rating_column : str
        평점 컬럼명
    n_samples : int
        출력할 샘플 개수
    max_length : int
        텍스트 최대 길이
    """
    print("\n[샘플 리뷰]")
    for idx, (i, row) in enumerate(df_good_but_hard.head(n_samples).iterrows(), 1):
        text = str(row[text_column])[:max_length]
        rating = row[rating_column]
        print(f"{idx}. (평점 {rating}점) {text}...")


def print_scarcity_analysis(
    df_reviews: pd.DataFrame,
    text_column: str = 'text',
    rating_column: str = 'rating',
    high_rating_threshold: int = 4,
    category: str = '희소성',
    keyword_dict: Optional[Dict[str, List[str]]] = None,
    n_samples: int = 10,
    show_pattern_stats: bool = True
):
    """
    희소성 패턴 분석을 수행하고 결과를 출력합니다.
    
    Parameters:
    -----------
    df_reviews : pd.DataFrame
        리뷰 데이터프레임
    text_column : str
        텍스트 컬럼명
    rating_column : str
        평점 컬럼명
    high_rating_threshold : int
        높은 평점 기준
    category : str
        희소성 카테고리 이름
    keyword_dict : Optional[Dict[str, List[str]]]
        카테고리별 패턴 사전
    n_samples : int
        샘플 개수
    show_pattern_stats : bool
        패턴별 통계 출력 여부
    """
    # 분석 실행
    df_good_but_hard, stats = analyze_scarcity_pattern(
        df_reviews, text_column, rating_column, 
        high_rating_threshold, category, keyword_dict
    )
    
    # 통계 출력
    print("[희소성 패턴 통계]")
    print(f"희소성 패턴 포함 리뷰: {stats['scarcity_count']:,}개 ({stats['scarcity_ratio']*100:.1f}%)")
    
    print(f"\n[희소성 패턴 분석]")
    print(f"희소성 패턴 + 높은 평점({high_rating_threshold}-5점) 리뷰: {stats['good_but_hard_count']:,}개")
    print(f"전체 대비 비율: {stats['good_but_hard_ratio']*100:.1f}%")
    if stats['scarcity_count'] > 0:
        print(f"희소성 리뷰 중 높은 평점 비율: {stats['high_rating_ratio_in_scarcity']*100:.1f}%")
    
    # 샘플 출력
    print_scarcity_samples(df_good_but_hard, text_column, rating_column, n_samples)
    
    # 패턴별 통계
    if show_pattern_stats:
        print_pattern_statistics(df_reviews, category, text_column, keyword_dict)
    
    return df_good_but_hard, stats
