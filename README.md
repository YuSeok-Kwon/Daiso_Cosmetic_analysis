# 다이소, '가성비'를 넘어 'K-뷰티 허브'로의 전환

## 프로젝트 개요

### 주제
다이소 뷰티의 지속 가능한 성장을 위해 **무엇이 성장을 돕고, 무엇이 리스크인가?** 를 분석하여 긍정적인 소비자 경험과 리스크 대응 전략을 수립한다.

### 목표
고객 리뷰 및 트렌드 데이터를 활용하여 수요를 예측하고, 브랜드 인지도와 수익성을 극대화하는 **데이터 기반의 유통 전략**을 수립한다.

---

## 분석 배경

### 1. 한국의 3고 경제난
고금리, 고물가, 고환율로 인한 소비 위축 상황에서 가성비 소비가 확대되고 있다.

### 2. 소비문화와 다이소의 역할
경제적 부담을 줄이면서도 품질을 포기하지 않으려는 소비자들이 다이소를 찾게 되었다. 특히 뷰티 카테고리에서 다이소는 저렴한 가격 대비 높은 품질로 주목받고 있다.

### 3. 아성다이소 현황
다이소는 단순한 생활용품 매장을 넘어 K-뷰티의 새로운 유통 채널로 부상하고 있다.

### 4. 핵심 타겟: 외국인 + Gen Z

| 구분 | 특징 |
|------|------|
| Gen Z | 스킨케어에 가장 큰 관심, '데일리케이션(Dailycation)' 추구 |
| 외국인 관광객 | K-뷰티 활성화, 기능성 화장품/비건/할랄 인증 제품에 관심 |
| MZ세대 영향력 | X세대까지 젊은이 문화를 수용하는 트렌드 확산 |

- 2025년 외국인 여행객 1,100만명 중 42%가 MZ세대
- "가장 한국인스러운 제품이 좋은 것"이라는 외국인 소비 심리
- SNS 검증 후 오프라인 구매 확인하는 패턴 (젊은 여행객 특징)

### 국가별 K-뷰티 선호도

| 국가/지역 | 선호 제품 | 특징 |
|-----------|-----------|------|
| 일본 | 진정 케어, 저자극 제품 | 센텔라/시카 성분 선호 |
| 미국 | 비타민C, 레티놀, AHA 토너 | 성분 투명성 중시, SNS 영향력 |
| 중국 | 프리미엄 안티에이징, 고급 세트 | 윤기 나는 제품 선호 |
| 호주 | 비건/클린뷰티 | 미니멀 스킨케어 루틴 지향 |
| 동남아 (싱가포르, 말레이시아) | 겔 타입, 자외선차단제 | 가볍고 빠른 흡수, 습도 대응 |

**스킨케어 트렌드 인사이트**
- 텍스처가 구매 결정의 핵심 요소
- 기후/피부타입에 따른 맞춤 선택 중요
- 여러 액티브 성분 동시 사용 지양

### 5. 스킨케어가 핵심이다
- 재구매율이 높은 카테고리: 팩/마스크, 기초 스킨케어, 클렌징
- 스킨케어는 다이소 뷰티의 핵심 성장 동력

---

## 분석 전략

### 메인 솔루션
MZ세대를 어떻게 타겟팅할 것인가?
- 히트작 분석: 무엇이 인기를 끌었는가
- 스테디셀러 분석: 무엇이 꾸준히 팔리는가
- 중심 문제 도출: 무엇이 성장을 방해하는가

### 서브 솔루션
리스크 대응 전략 수립
- 부정 리뷰 분석을 통한 개선점 도출
- 성분 기반 비건/할랄 인증 가능 제품 식별

---

## 데이터 수집 (Crawling)

다이소몰 뷰티 제품 데이터를 수집하는 크롤링 시스템

### 프로젝트 구조

```
Crawling/
├── daiso_beauty_crawler.py     # 메인 크롤러
├── config.py                   # 카테고리 설정
├── utils.py                    # 유틸리티 함수
│
├── modules/                    # 재사용 모듈
│   ├── clova_ocr.py           # 네이버 Clova OCR
│   ├── ingredient_detector.py  # 성분표 자동 감지
│   ├── ingredient_parser.py    # 성분 파싱 및 유효성 검증
│   ├── ingredient_postprocessor.py  # 성분 후처리
│   ├── halal_vegan_api.py     # 할랄/비건 분석
│   ├── halal_vegan_checker.py # 할랄/비건 체크
│   ├── certification_api.py   # 인증 API
│   ├── image_preprocessor.py  # 이미지 전처리
│   ├── ocr_utils_split.py     # OCR 유틸리티
│   └── driver_setup.py        # 웹드라이버 설정
│
├── scripts/                    # 유틸리티 스크립트
│   ├── crawl_missing_products.py  # 누락 제품 크롤링
│   └── crawl_missing_reviews.py   # 누락 리뷰 크롤링
│
├── tests/                      # 테스트 파일
│   └── test_ingredient_detection.py
│
├── logs/                       # 로그 파일
└── cache/                      # 캐시 (OCR 모델 등)
```

### 기술 스택

| 분류 | 기술 |
|------|------|
| 크롤링 | Selenium, WebDriver Manager |
| OCR | Naver Clova OCR, EasyOCR |
| 이미지 처리 | OpenCV, PIL |
| 데이터 처리 | Pandas |
| 환경 관리 | python-dotenv |

### 주요 기능

#### 1. 다중 소스 성분 추출
- HTML 텍스트 파싱
- 이미지 ALT 속성 추출
- OCR (Clova + EasyOCR) 교차 검증
- 신뢰도(confidence) 기반 필터링

#### 2. 성분 파싱 고도화
- OCR 오인식 자동 교정 (소톱 -> 소듐, 글라이골 -> 글라이콜 등)
- 화학 성분 패턴 매칭 (에톡실레이트, 효소, 아미노산염 등)
- 500+ 성분 데이터베이스 매칭
- 농도/함량 괄호 자동 제거 (%, ppm, mg/kg 등)

#### 3. 할랄/비건 분석
- 동물 유래 성분 감지
- 할랄 금지 성분 체크
- 제품별 인증 가능 여부 판정

#### 4. 성분표 자동 감지
- OpenCV 기반 성분표 영역 자동 인식
- 긴 이미지 하단 크롭 (Clova OCR 1960px 제한 대응)
- 멀티 이미지 vs 싱글 이미지 분기 처리

---

## 사용 방법

### 1. 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일에 API 키 입력
```

`.env` 파일:
```env
# Naver Clova OCR API
CLOVA_OCR_URL=https://...
CLOVA_OCR_SECRET=your-secret-key
```

### 2. 메인 크롤러 실행

```bash
python daiso_beauty_crawler.py
```

크롤링 옵션:
1. 제품 정보만
2. 제품 정보 + 리뷰
3. 제품 코드 + 성분만
4. 제품 코드 + 리뷰만
5. 전체 (제품 정보 + 리뷰 + 성분)

### 3. 누락 데이터 크롤링

```bash
# 제품 정보 누락
python scripts/crawl_missing_products.py

# 리뷰 누락
python scripts/crawl_missing_reviews.py
```

### 4. 성분 감지 테스트

```bash
python tests/test_ingredient_detection.py
```

---

## 출력 데이터

### 제품 정보
| 필드 | 설명 |
|------|------|
| product_code | 제품 코드 |
| name | 제품명 |
| price | 가격 |
| brand | 브랜드 |
| country | 원산지 |
| can_halal | 할랄 인증 가능 여부 |
| can_vegan | 비건 인증 가능 여부 |
| likes | 좋아요 수 |
| shares | 공유 수 |

### 리뷰
| 필드 | 설명 |
|------|------|
| product_code | 제품 코드 |
| date | 리뷰 작성일 |
| rating | 평점 |
| text | 리뷰 내용 |
| user_id | 사용자 ID |
| image_count | 첨부 이미지 수 |

### 성분
| 필드 | 설명 |
|------|------|
| product_code | 제품 코드 |
| ingredient | 성분명 |
| confidence | 신뢰도 (0.0 ~ 1.0) |
| source | 추출 소스 (ALT, OCR 등) |
| is_vegan | 비건 여부 |
| is_halal | 할랄 여부 |

---

## 모듈 설명

| 모듈 | 설명 |
|------|------|
| `clova_ocr.py` | 네이버 Clova OCR API 클라이언트 |
| `ingredient_detector.py` | OpenCV 기반 성분표 영역 자동 감지 |
| `ingredient_parser.py` | 성분 파싱 및 유효성 검증 (500+ 성분 DB) |
| `ingredient_postprocessor.py` | 정규표현식 + 사전 기반 성분 정제 |
| `halal_vegan_api.py` | 할랄/비건 성분 분석 |
| `halal_vegan_checker.py` | 할랄/비건 인증 가능 여부 판정 |
| `certification_api.py` | 외부 인증 API 연동 |
| `ocr_utils_split.py` | OCR 통합 유틸리티 (분할 OCR 지원) |

---

## 참고사항

- 크롤링 속도: 제품마다 상이 (리뷰 개수에 따라 편차)
- 로그 파일: `logs/` 폴더에 자동 저장
- 캐시: `cache/` 폴더 (EasyOCR 모델 등)
- 결과: `../data/` 폴더에 CSV/Parquet 저장

---

## 참고 자료

- [2025 외국인 관광객 통계 현황](https://www.lemonlab.pro/2025-%EC%99%B8%EA%B5%AD%EC%9D%B8-%EA%B4%80%EA%B4%91%EA%B0%9D-%ED%86%B5%EA%B3%84-%ED%98%84%ED%99%A9/)
- [K-Beauty 2026 글로벌 트렌드 및 외국인 선호도](https://kessence.kr/k-beauty-2026-global-trends-foreign-preferences/)
