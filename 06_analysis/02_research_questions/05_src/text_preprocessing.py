"""
텍스트 전처리 모듈

다이소 리뷰 텍스트를 분석하기 위한 전처리 함수들을 제공합니다.
"""

import re
from typing import List, Optional
from konlpy.tag import Okt


# KoNLPy Okt 인스턴스 (전역 변수로 선언하여 재사용)
_okt = None


def get_okt():
    """Okt 인스턴스를 반환 (Lazy initialization)"""
    global _okt
    if _okt is None:
        _okt = Okt()
    return _okt


def extract_repurchase_flag(text: str) -> bool:
    """
    리뷰 텍스트 맨 앞에서 '재구매' 키워드를 감지합니다.
    
    Parameters:
    -----------
    text : str
        리뷰 텍스트
        
    Returns:
    --------
    bool
        재구매 여부 (True: 재구매, False: 일반)
        
    Examples:
    ---------
    >>> extract_repurchase_flag("재구매 너무 좋아요")
    True
    >>> extract_repurchase_flag("좋아요")
    False
    """
    if not isinstance(text, str):
        return False
    
    # 리뷰 맨 앞 10자 내에 '재구매' 키워드가 있는지 확인
    text_start = text[:10].strip()
    
    # 재구매 관련 키워드 패턴
    repurchase_patterns = [
        r'^재구매',
        r'^리구매',
        r'^re구매',
        r'^\[재구매\]',
        r'^\(재구매\)',
        r'^재.?구매'  # 재 구매, 재-구매 등
    ]
    
    for pattern in repurchase_patterns:
        if re.search(pattern, text_start, re.IGNORECASE):
            return True
    
    return False


def preprocess_text(
    text: str,
    extract_pos: List[str] = ['Noun', 'Adjective'],
    min_length: int = 2,
    remove_numbers: bool = True
) -> List[str]:
    """
    텍스트를 형태소 분석하여 토큰 리스트로 반환합니다.
    
    Parameters:
    -----------
    text : str
        분석할 텍스트
    extract_pos : List[str]
        추출할 품사 리스트 (Noun: 명사, Adjective: 형용사, Verb: 동사)
    min_length : int
        최소 토큰 길이
    remove_numbers : bool
        숫자 제거 여부
        
    Returns:
    --------
    List[str]
        토큰 리스트
        
    Examples:
    ---------
    >>> preprocess_text("가성비 좋은 제품입니다", extract_pos=['Noun', 'Adjective'])
    ['가성비', '좋다', '제품']
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    try:
        okt = get_okt()
        
        # 특수문자 제거 (단, 한글, 영문, 숫자, 공백만 유지)
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        
        # 형태소 분석 및 품사 태깅
        tokens_pos = okt.pos(text, norm=True, stem=True)
        
        # 지정된 품사만 추출
        tokens = []
        for word, pos in tokens_pos:
            if pos in extract_pos:
                # 길이 체크
                if len(word) < min_length:
                    continue
                
                # 숫자 제거
                if remove_numbers and word.isdigit():
                    continue
                
                tokens.append(word)
        
        return tokens
    
    except Exception as e:
        print(f"텍스트 전처리 중 오류 발생: {e}")
        return []


def remove_stopwords(
    tokens: List[str],
    stopwords_list: Optional[List[str]] = None
) -> List[str]:
    """
    토큰 리스트에서 불용어를 제거합니다.
    
    Parameters:
    -----------
    tokens : List[str]
        토큰 리스트
    stopwords_list : Optional[List[str]]
        불용어 리스트 (None이면 기본 불용어 사용)
        
    Returns:
    --------
    List[str]
        불용어가 제거된 토큰 리스트
        
    Examples:
    ---------
    >>> tokens = ['제품', '정말', '좋다', '것', '같다']
    >>> remove_stopwords(tokens)
    ['제품', '좋다']
    """
    if stopwords_list is None:
        stopwords_list = get_default_stopwords()
    
    return [token for token in tokens if token not in stopwords_list]


def get_default_stopwords() -> List[str]:
    """
    기본 불용어 리스트를 반환합니다.
    
    Returns:
    --------
    List[str]
        불용어 리스트
    """
    stopwords = [
        # 대명사
        '것', '수', '때', '곳', '등', '점',
        
        # 조사/접속사
        '이', '그', '저', '의', '가', '을', '를', '에', '와', '과', '도', '로', '으로',
        '은', '는', '이다', '있다', '하다', '되다', '같다',
        
        # 부사
        '정말', '너무', '진짜', '완전', '엄청', '되게', '아주', '좀', '막', '그냥',
        
        # 감탄사
        '아', '어', '오', '음', '흠',
        
        # 기타 의미 없는 단어
        '거', '걸', '게', '네', '요', '죠', '뭐', '좀', '더', '별로', '번', '개',
        '듯', '듯하다', '다', '든', '말', '및', '분', '씩', '안', '약', '어디',
        '여기', '오늘', '요즘', '월', '위', '이번', '일', '저', '전', '제', '중',
        '즈음', '지', '쪽', '채', '체', '후',
        
        # 한글자 단어 (대부분 의미 없음)
        '거', '게', '께', '네', '데', '든', '때', '뭐', '번', '뿐', '씩', '잘',
        '좀', '채', '캐', '테', '편', '히',
    ]
    
    return stopwords


def clean_tokens(tokens: List[str]) -> List[str]:
    """
    토큰 리스트를 정제합니다 (중복 제거, 공백 제거 등).
    
    Parameters:
    -----------
    tokens : List[str]
        토큰 리스트
        
    Returns:
    --------
    List[str]
        정제된 토큰 리스트
    """
    # 공백 제거 및 소문자 변환
    cleaned = []
    for token in tokens:
        token = token.strip()
        if token:  # 빈 문자열 제외
            cleaned.append(token)
    
    return cleaned


def tokenize_reviews(
    texts: List[str],
    extract_pos: List[str] = ['Noun', 'Adjective'],
    remove_stop: bool = True,
    verbose: bool = True
) -> List[List[str]]:
    """
    여러 리뷰 텍스트를 일괄 토큰화합니다.
    
    Parameters:
    -----------
    texts : List[str]
        리뷰 텍스트 리스트
    extract_pos : List[str]
        추출할 품사
    remove_stop : bool
        불용어 제거 여부
    verbose : bool
        진행상황 출력 여부
        
    Returns:
    --------
    List[List[str]]
        토큰화된 리뷰 리스트
    """
    results = []
    total = len(texts)
    
    for idx, text in enumerate(texts):
        if verbose and (idx + 1) % 1000 == 0:
            print(f"진행중: {idx + 1}/{total} ({(idx + 1)/total*100:.1f}%)")
        
        tokens = preprocess_text(text, extract_pos=extract_pos)
        
        if remove_stop:
            tokens = remove_stopwords(tokens)
        
        tokens = clean_tokens(tokens)
        results.append(tokens)
    
    if verbose:
        print(f"완료: {total}개 리뷰 토큰화")
    
    return results
