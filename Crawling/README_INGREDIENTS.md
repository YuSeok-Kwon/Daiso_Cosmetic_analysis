# 다이소몰 전성분 추출 크롤러

이미지 alt 속성에서 전성분 정보를 추출하는 크롤러입니다.

## 📋 기능

- 제품 상세 페이지의 이미지 alt 텍스트 분석
- [전성분] 섹션 자동 추출
- 제품 변형별 성분 분리 (예: 우디가든, Tender Paris)
- 노이즈 제거 및 데이터 정제
- CSV 형식으로 저장 (제품 정보 + 전성분 상세)

## 🚀 사용 방법

### 1. 기존 제품 CSV에서 크롤링

```bash
cd /Users/yu_seok/Documents/workspace/nbCamp/Project/Why-pi/Crawling
python daiso_ingredients.py
```

입력:
- 기존 다이소 제품 CSV 파일 경로
- 최대 크롤링 제품 수 (선택사항)

출력:
- `daiso_products_with_ingredients_YYYYMMDD_HHMMSS.csv`: 제품 기본 정보
- `daiso_ingredients_YYYYMMDD_HHMMSS.csv`: 전성분 상세 정보

### 2. 추출 로직 테스트

```bash
python test_ingredients_extraction.py
```

실제 크롤링 없이 alt 텍스트 파싱 로직만 테스트합니다.

## 📊 출력 데이터 구조

### 제품 정보 CSV

| 컬럼           | 설명                   |
| -------------- | ---------------------- |
| product_id     | 제품 ID                |
| product_name   | 제품명                 |
| volume         | 용량                   |
| manufacturer   | 제조업자               |
| site           | 사이트 (다이소)        |
| brand          | 브랜드                 |
| main_category  | 대분류                 |
| sub_category   | 소분류                 |
| price          | 가격                   |
| rating         | 평점                   |
| review_count   | 리뷰 수                |
| product_url    | 제품 URL               |
| image_url      | 이미지 URL             |
| raw_ingredients| 원본 전성분 텍스트     |
| status         | 크롤링 상태 (success/failed) |
| crawled_at     | 크롤링 시간            |

### 전성분 상세 CSV

| 컬럼            | 설명                           |
| --------------- | ------------------------------ |
| product_id      | 제품 ID                        |
| product_name    | 제품명                         |
| product_variant | 제품 변형 (우디가든, Paris 등) |
| ingredient      | 성분명                         |
| crawled_at      | 크롤링 시간                    |

## 🔍 추출 예시

**입력 (이미지 alt 텍스트):**
```
[전성분] 우디가든 정제수, 프로필렌글라이콜 라벤더의 오일, 향료, 제라니올, 리모넨, ... Tender Paris: 에탄올, 정제수 프로필렌글라이콜라벤더오일, 향료 벤질알코올, ...
```

**출력:**
```csv
product_variant,ingredient
우디가든,정제수
우디가든,프로필렌글라이콜 라벤더의 오일
우디가든,향료
우디가든,제라니올
우디가든,리모넨
Tender Paris,에탄올
Tender Paris,정제수 프로필렌글라이콜라벤더오일
Tender Paris,향료 벤질알코올
...
```

## ⚙️ 추출 로직

1. **[전성분] 섹션 추출**: 정규표현식으로 전성분 영역 분리
2. **제품 변형 파싱**:
   - 콜론(`:`)으로 구분되는 변형 찾기 (예: "Tender Paris:")
   - 첫 단어가 제품명인 경우 처리 (예: "우디가든 성분1, 성분2")
3. **성분 분리**: 쉼표(`,`)로 분리
4. **노이즈 제거**:
   - 특수문자 제거 (⌀, ×)
   - 괄호 및 내용 제거
   - 숫자+단위 패턴 제거 (10mm, 30ft 등)
   - 불필요한 문구 제거 (사용법, 주의사항 등)
5. **검증**:
   - 최소 길이 2자 이상
   - 최대 길이 100자 이하
   - 숫자만 있는 경우 제외

## ⚠️ 주의사항

- 교육/연구 목적으로만 사용하세요
- 웹사이트의 robots.txt 및 이용약관을 준수하세요
- 크롤링 간격(딜레이)을 유지하세요 (2-4초)
- 서버에 과부하를 주지 마세요

## 📝 로그

크롤링 로그는 `daiso_ingredients.log` 파일에 저장됩니다.

## 🐛 문제 해결

### Q: 전성분이 추출되지 않아요
A:
- 이미지 alt 속성에 [전성분] 텍스트가 있는지 확인하세요
- `test_ingredients_extraction.py`로 파싱 로직을 테스트해보세요

### Q: 성분에 노이즈가 많아요
A:
- `clean_ingredient()` 함수의 `exclude_patterns`에 제거할 패턴을 추가하세요

### Q: 제품 변형이 제대로 분리되지 않아요
A:
- 제품명 뒤에 콜론(:)이 있는지 확인하세요
- 또는 첫 단어가 한글 제품명인지 확인하세요

## 📚 관련 파일

- `daiso_ingredients.py`: 메인 크롤러
- `test_ingredients_extraction.py`: 테스트 스크립트
- `driver_setup.py`: Selenium 드라이버 설정
- `utils.py`: 유틸리티 함수
- `config.py`: 설정 파일
