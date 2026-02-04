# 다이소 뷰티 크롤러

다이소몰 뷰티 제품 크롤링 프로젝트

## 프로젝트 구조

```md
Crawling/
├── daiso_beauty_crawler.py # 메인 크롤러
├── config.py # 카테고리 설정
├── utils.py # 유틸리티 함수
│
├── modules/ # 재사용 모듈
│ ├── clova_ocr.py # 네이버 Clova OCR
│ ├── ingredient_detector.py # 성분표 자동 감지
│ ├── ingredient_postprocessor.py # 성분 후처리
│ ├── halal_vegan_api.py # 할랄/비건 분석
│ ├── image_preprocessor.py # 이미지 전처리
│ ├── ocr_utils_split.py # OCR 유틸리티
│ └── ...
│
├── scripts/ # 유틸리티 스크립트
│ ├── crawl_missing_products.py # 누락 제품 크롤링
│ └── crawl_missing_reviews.py # 누락 리뷰 크롤링
│
└── tests/ # 테스트 파일
├── test_ingredient_detection.py
└── test_results/
```

## 사용 방법

### 1. 메인 크롤러 실행

```bash
python daiso_beauty_crawler.py
```

**크롤링 옵션:**

1. 제품 정보만
2. 제품 정보 + 리뷰
3. 제품 코드 + 성분만
4. 제품 코드 + 리뷰만
5. 전체 (제품 정보 + 리뷰 + 성분)

### 2. 누락 데이터 크롤링

**제품 정보 누락 (리뷰는 있음):**

```bash
cd scripts
python crawl_missing_products.py
```

**리뷰 누락 (제품 정보는 있음):**

```bash
cd scripts
python crawl_missing_reviews.py
```

### 3. 성분 감지 테스트

```bash
cd tests
python test_ingredient_detection.py
```

## 환경 설정

`.env` 파일에 API 키 설정:

```env
# Naver Clova OCR API
CLOVA_OCR_URL=https://...
CLOVA_OCR_SECRET=your-secret-key
```

## 주요 기능

### 할랄/비건 분석

- 동물 유래 성분 감지
- 할랄 금지 성분 체크
- 제품별 인증 가능 여부 판정

### 성분 추출 (고도화)

- Clova OCR (95%+ 정확도)
- 성분표 자동 영역 감지
- 오타 자동 교정
- 화장품 성분 사전 매칭

### 다중 소스 검증

- HTML 텍스트
- 이미지 ALT 속성
- OCR (Clova + EasyOCR)
- 교차 검증으로 정확도 향상

## 출력 데이터

**제품 정보:**

- product_code, name, price, brand, country
- can*할랄인증, can*비건
- likes, shares

**리뷰:**

- product_code, date, rating, text
- user_id, image_count

**성분:**

- product_code, ingredient
- confidence, is_valid
- is_vegan, is_halal

## 모듈 설명

| 모듈                          | 설명                              |
| ----------------------------- | --------------------------------- |
| `clova_ocr.py`                | 네이버 Clova OCR API 클라이언트   |
| `ingredient_detector.py`      | OpenCV 기반 성분표 영역 자동 감지 |
| `ingredient_postprocessor.py` | 정규표현식 + 사전 기반 성분 정제  |
| `halal_vegan_api.py`          | 할랄/비건 성분 분석               |
| `ocr_utils_split.py`          | OCR 통합 유틸리티                 |

## 참고사항

- 크롤링 속도: 제품마다 상이(리뷰의 개수에 따라 큰 편차 보임)
- 로그 파일: `logs/` 폴더에 자동 저장
- 캐시: `cache/` 폴더 (EasyOCR 모델 등)
- 결과: `../data/` 폴더에 CSV/Parquet 저장
