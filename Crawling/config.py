"""
크롤링 설정 파일
"""
import os

# 크롤링 설정
CRAWLING_CONFIG = {
    # 요청 간격 (초)
    'min_delay': 2,
    'max_delay': 4,

    # 페이지 로드 대기 시간
    'page_load_timeout': 30,
    'implicit_wait': 10,

    # 재시도 설정
    'max_retries': 3,
    'retry_delay': 5,

    # 헤드리스 모드 (False = 브라우저 창 보임)
    'headless': True,

    # 데이터 저장 경로
    'data_dir': 'data',
    'log_dir': 'logs',
}

# 다이소몰 설정
DAISO_CONFIG = {
    'base_url': 'https://www.daisomall.co.kr',
    'category_base': 'https://www.daisomall.co.kr/ds/exhCtgr/C208/CTGR_00014',  # 뷰티/위생 기본 URL
    'large_category': 'C208',  # 뷰티 대분류
    'middle_category': 'CTGR_00014',  # 뷰티 중분류
    'max_products_per_category': 50,
}

# 다이소 뷰티 전체 카테고리 구조
DAISO_BEAUTY_CATEGORIES = {
    "스킨케어": {
        "중분류코드": "CTGR_00057",
        "소분류": {
            "CTGR_00366": "기초스킨케어",
            "CTGR_00367": "립케어",
            "CTGR_00368": "팩/마스크",
            "CTGR_00369": "자외선차단제",
            "CTGR_00370": "클렌징/필링",
        }
    },
    "메이크업": {
        "중분류코드": "CTGR_00054",
        "소분류": {
            "CTGR_00346": "베이스메이크업",
            "CTGR_00347": "아이메이크업",
            "CTGR_00348": "립메이크업",
            "CTGR_00351": "치크/하이라이터",
            "CTGR_00353": "퍼프브러시세척",
            "CTGR_00354": "향수",
        }
    },
    "네일용품": {
        "중분류코드": "CTGR_00053",
        "소분류": {
            "CTGR_00341": "네일리무버",
            "CTGR_00342": "네일아트도구",
            "CTGR_00343": "네일패치/팁",
            "CTGR_00344": "매니큐어",
            "CTGR_00345": "네일케어용품",
        }
    },
    "맨케어": {
        "중분류코드": "CTGR_00055",
        "소분류": {
            "CTGR_00355": "남성향수",
            "CTGR_00356": "남성메이크업",
            "CTGR_00357": "클렌징/쉐이빙",
            "CTGR_00358": "남성스킨케어",
            "CTGR_00359": "남성용면도기",
        }
    },
    "미용소품": {
        "중분류코드": "CTGR_00056",
        "소분류": {
            "CTGR_00360": "미용가위",
            "CTGR_00361": "거울",
            "CTGR_00362": "미용소도구",
            "CTGR_00363": "화장소품수납",
            "CTGR_00364": "공병",
            "CTGR_00365": "세안소품",
        }
    },
    "헤어/바디": {
        "중분류코드": "CTGR_00058",
        "소분류": {
            "CTGR_00371": "염색/펌",
            "CTGR_00372": "빗/브러쉬",
            "CTGR_00373": "가발",
            "CTGR_00374": "헤어스타일링",
            "CTGR_00375": "트리트먼트/케어",
            "CTGR_00376": "헤어롤",
            "CTGR_00377": "샴푸",
            "CTGR_00378": "린스",
            "CTGR_00379": "바디보습",
            "CTGR_00380": "바디워시",
            "CTGR_00381": "입욕제",
            "CTGR_00382": "풋케어",
            "CTGR_00383": "핸드워시",
            "CTGR_00384": "데오케어",
            "CTGR_00385": "제모기",
            "CTGR_00386": "비누",
            "CTGR_00387": "핸드크림",
        }
    },
}

# 다이소 식품 > 건강식품 카테고리
DAISO_HEALTH_FOOD = {
    "건강식품": {
        "대분류": "식품",
        "대분류코드": "CTGR_00022",
        "중분류코드": "CTGR_01020",
        "소분류": {
            "CTGR_01024": "기초건강",
            "CTGR_01025": "눈건강",
            "CTGR_01026": "장/간건강",
            "CTGR_01027": "혈행/혈당/혈압",
            "CTGR_01028": "이너뷰티",
            "CTGR_01029": "다이어트",
            "CTGR_01030": "키즈",
            "CTGR_01031": "건강기능식품",
            "CTGR_01032": "일반식품",
        }
    }
}

# 올리브영 설정
OLIVEYOUNG_CONFIG = {
    'base_url': 'https://www.oliveyoung.co.kr',
    'ranking_url': '/store/main/getBestList.do',
    'ranking_disp_cat_no': '900000100100001',  # 베스트 랭킹 페이지 고정값
    'max_products_per_category': 100,
    'rows_per_page': 24,  # 페이지당 상품 수
}

# 올리브영 베스트 랭킹 카테고리
OLIVEYOUNG_CATEGORIES = {
    # 뷰티 (1000001XXXX)
    "스킨케어":      "10000010001",
    "메이크업":      "10000010002",
    "바디케어":      "10000010003",
    "헤어케어":      "10000010004",
    "향수/디퓨저":   "10000010005",
    "메이크업 툴":   "10000010006",
    "맨즈케어":      "10000010007",
    "더모 코스메틱": "10000010008",
    "마스크팩":      "10000010009",
    "클렌징":        "10000010010",
    "선케어":        "10000010011",
    "네일":          "10000010012",

    # 헬스 (1000002XXXX)
    "건강식품":      "10000020001",
    "푸드":          "10000020002",
    "구강용품":      "10000020003",
    "위생용품":      "10000020004",
    "헬스/건강용품": "10000020005",

    # 라이프 (1000003XXXX)
    "홈리빙/가전":   "10000030005",
    "취미/팬시":     "10000030006",
    "패션":          "10000030007",
}

# iHerb 설정
IHERB_CONFIG = {
    'base_url': 'https://kr.iherb.com',
    'categories': {
        '비타민C': '/c/vitamin-c',
        '프로바이오틱스': '/c/probiotics',
        '오메가3': '/c/omega-3-fish-oil',
        '종합비타민': '/c/multivitamins',
    },
    'bestseller_url': '/c/best-sellers',
    'max_products_per_category': 100,
}

# 저장할 데이터 필드
DATA_FIELDS = [
    'site',           # 사이트명
    'category',       # 카테고리
    'product_name',   # 상품명
    'brand',          # 브랜드
    'price',          # 가격
    'original_price', # 원가
    'discount_rate',  # 할인율
    'rating',         # 평점
    'review_count',   # 리뷰 수
    'image_url',      # 이미지 URL
    'product_url',    # 상품 URL
    'crawled_at',     # 수집 시간
]

# User-Agent 풀
USER_AGENTS = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
]
