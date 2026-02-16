# MFDS 기능성화장품 API 데이터 수집 가이드

- **최종 수정일**: 2026-02-16
- **소스 코드**: `05_src/03_mfds_api/`

---

## 1. 개요

### 1.1 목적

식품의약품안전처(MFDS)의 공공데이터 API 3종을 활용하여:

- products.parquet의 제품들에 대해 기능성화장품 여부 검증
- 미백, 주름개선, 자외선차단 등 효능 정보 수집
- 성분 영문명, CAS 번호, 사용제한 정보 보강
- 기존 product_attributes.csv 데이터와 비교 검증

### 1.2 사용 API (3종)

| API | 용도 | 소스 코드 |
|-----|------|-----------|
| 기능성화장품 보고품목정보 | 제품 효능 조회 | `functional_cosmetics.py` |
| 화장품 원료성분정보 | 성분 영문명/CAS 번호 | `ingredient_info.py` |
| 화장품 사용제한 원료정보 | 금지/제한 성분 매칭 | `restricted_ingredients.py` |

### 1.3 실행 방법

```bash
# 환경 설정
# config/.env 에 MFDS_API_KEY=your_key 추가

# 프로젝트 루트에서 실행
cd Why-pi/

# 단건 테스트
python 05_src/03_mfds_api/run_all.py --test

# 전체 실행
python 05_src/03_mfds_api/run_all.py

# 개별 실행
python 05_src/03_mfds_api/run_all.py --target functional
python 05_src/03_mfds_api/run_all.py --target ingredient
python 05_src/03_mfds_api/run_all.py --target restricted
```

---

## 2. API 1: 기능성화장품 보고품목정보

### 2.1 API 정보

| 항목 | 내용 |
|------|------|
| API명 | 기능성화장품 보고품목정보 조회 서비스 |
| 제공기관 | 식품의약품안전처 |
| Endpoint | `http://apis.data.go.kr/1471000/FtnltCosmRptPrdlstInfoService/getRptPrdlstInq` |
| 신청 | [공공데이터포털](https://www.data.go.kr/data/15095680/openapi.do) |

### 2.2 요청 파라미터

| 파라미터명 | 필수 | 설명 | 예시 |
|-----------|------|------|------|
| serviceKey | Y | 인증키 | b90a9b72... |
| pageNo | N | 페이지 번호 | 1 |
| numOfRows | N | 한 페이지 결과 수 | 10 |
| type | N | 응답 형식 | json |
| ITEM_NAME | N | 제품명 (검색용) | 해서린스팟케어클리어젤 |
| ENTP_NAME | N | 업체명 | (주)와이-피코스메틱 |

### 2.3 제품명 매칭 방법

우리 데이터와 MFDS 데이터의 제품명 형식이 다릅니다:

| 구분 | 예시 |
|------|------|
| **우리 데이터** | 해서린 스팟 케어 클리어 젤 10 ml |
| **MFDS 데이터** | 해서린스팟케어클리어젤 |

**정제 규칙** (`functional_cosmetics.py > clean_product_name()`):

1. 대괄호/소괄호 내용 제거 (`[기획]`, `(리뉴얼)`)
2. "by 브랜드명" 패턴 제거
3. 용량 정보 제거 (`10ml`, `50g`)
4. 특수문자 제거, 모든 공백 제거

### 2.4 효능 판별

**주의: EFFECT_YN 필드 사용 금지** — API 응답의 `EFFECT_YN1~3`은 항상 'N'으로 반환됩니다.

`COSMETIC_TARGET_FLAG` 또는 `COSMETIC_TARGET_FLAG_NAME` 필드를 사용합니다.

#### 기능성화장품 전체 효능 목록 (화장품법 시행규칙 제10조)

| FLAG 값 | 효능 | 영문명 | 설명 |
|---------|------|--------|------|
| 제10조 제1항 제1호 | 미백 | whitening | 피부의 미백에 도움을 주는 제품 |
| 제10조 제1항 제2호 | 주름개선 | anti-wrinkle | 피부의 주름개선에 도움을 주는 제품 |
| 제10조 제1항 제3호 | 자외선차단 | sunscreen | 자외선으로부터 피부를 보호하는 제품 |
| 제10조 제1항 제4호 | 모발 색상 | hair_color | 모발의 색상 변화·제거·영양공급 |
| 제10조 제1항 제5호 | 체모 제거 | hair_removal | 체모를 제거하는 데 도움 |
| 제10조 제1항 제6호 | 여드름 완화 | acne | 여드름성 피부 완화 |
| 제10조 제1항 제7호 | 아토피 보습 | atopic | 아토피성 피부 건조함 완화 |
| 제10조 제1항 제8호 | 튼살 완화 | stretch_marks | 튼살 붉은선 완화 |
| 제10조 제1항 제9호 | 탈모 완화 | hair_loss | 탈모 증상 완화 |
| 제10조 제1항 제10호 | 눈가 개선 | eye_wrinkle | 눈 주위 일시적 개선 |
| 제10조 제1항 제11호 | 선탠 | tanning | 피부를 곱게 태워주는 기능 |

### 2.5 매칭 결과 (실제 데이터)

| 구분 | 제품 수 | 매칭 성공 | 매칭률 |
|------|---------|----------|--------|
| functional=1 | 266 | 259 | 97.4% |
| functional=0 | 688 | 0 | 0% |
| **전체** | **954** | **259** | **27.1%** |

> **참고:** `product_attributes.csv`(944건)에는 functional=1이 260건이지만, `products.parquet`(954건)에는 6개 제품이 추가로 포함되어 functional=1이 266건입니다. 매칭 코드는 `products.parquet` 기준으로 동작합니다.

#### 효능별 통계

| 효능 | 제품 수 |
|------|---------|
| 미백 (is_whitening) | 136건 |
| 주름개선 (is_anti_wrinkle) | 103건 |
| 자외선차단 (is_sunscreen) | 20건 |

> 4~11호 효능은 본 프로젝트 데이터에서 미확인

### 2.6 데이터 검증 — 오류 수정

기존 `product_attributes.csv`의 `is_functional=0`이지만 API에서 매칭된 제품 8건 → `is_functional=1`로 수정:

| product_code | 제품명 | 수정 내용 |
|-------------|--------|----------|
| Y1005~Y1007 | 해서린 듀오 커버 팩트 | 0 → 1 |
| Y1063~Y1064 | 리애 카밍 워터 팩 | 0 → 1 |
| Y1086~Y1087 | 더마틱스 울트라 | 0 → 1 |
| Y1161 | 클레어스 프레쉴리 쥬스드 비타민 드롭 | 0 → 1 |

---

## 3. API 2: 화장품 원료성분정보

### 3.1 API 정보

| 항목 | 내용 |
|------|------|
| API명 | 화장품 원료성분정보 |
| Endpoint | `https://apis.data.go.kr/1471000/CsmtcsIngdCpntInfoService01/getCsmtcsIngdCpntInfoService01` |
| 신청 | [공공데이터포털](https://www.data.go.kr/data/15111774/openapi.do) |
| 총 데이터 | 21,696건 |

### 3.2 요청 파라미터

| 파라미터명 | 필수 | 설명 |
|-----------|------|------|
| serviceKey | Y | 인증키 |
| INGR_KOR_NAME | N | 표준명 (검색용) |
| pageNo | N | 페이지 번호 |
| numOfRows | N | 한 페이지 결과 수 |
| type | N | 응답 형식 (json/xml) |

### 3.3 응답 컬럼

| 컬럼명 | 설명 | 예시 |
|--------|------|------|
| INGR_KOR_NAME | 표준명 | 토코페릴아세테이트 |
| INGR_ENG_NAME | 영문명 | Tocopheryl Acetate |
| CAS_NO | CAS 번호 | 58-95-7 |
| ORIGIN_MAJOR_KOR_NAME | 기원 및 정의 | - |
| INGR_SYNONYM | 이명 | - |

### 3.4 매칭 결과 (실제 데이터)

`ingredients_master.csv` (1,744건)에 다음 컬럼 추가:

| 추가 컬럼 | 매칭 건수 | 비율 |
|----------|----------|------|
| name_eng (영문명) | 1,222개 | 70.1% |
| cas_no (CAS 번호) | 229개 | 13.1% |

---

## 4. API 3: 화장품 사용제한 원료정보

### 4.1 API 정보

| 항목 | 내용 |
|------|------|
| API명 | 화장품 사용제한 원료정보 |
| Endpoint | `https://apis.data.go.kr/1471000/CsmtcsUseRstrcInfoService/getCsmtcsUseRstrcInfoService` |
| 신청 | [공공데이터포털](https://www.data.go.kr/data/15111772/openapi.do) |
| 총 데이터 | API 문서 기준 약 31,000건 (실제 다운로드 시 확인 필요) |

### 4.2 주의사항

- API 검색 기능이 정상 작동하지 않음 (전체 결과 반환)
- **전체 데이터 다운로드 후 로컬 매칭** 방식으로 처리 (`restricted_ingredients.py`)
- 제한사항(LIMIT_COND)은 다국어(영문/중문/한글) 혼용

### 4.3 규제 타입

| 타입 | 설명 |
|------|------|
| 금지 | 화장품에 사용 불가 |
| 한도 | 배합량 제한 있음 |
| 한도/금지 | 용도에 따라 금지 또는 제한 |

### 4.4 매칭 결과 (실제 데이터)

`ingredients_master.csv` (1,744건)에 다음 컬럼 추가:

| 추가 컬럼 | 매칭 건수 | 비율 |
|----------|----------|------|
| is_restricted | 173개 | 9.9% |
| restriction_type | 173개 | 9.9% |

**사용제한 타입 분포:**

| 타입 | 건수 |
|------|------|
| 한도 | 148개 |
| 한도/금지 | 20개 |
| 금지 | 5개 |

---

## 5. 심사품목 API (검토 결과: 불필요)

보고품목 API 외에 **심사품목 API**도 검토하였으나, 본 프로젝트에는 적합하지 않음.

| 항목 | 내용 |
|------|------|
| API명 | 기능성화장품 심사품목 정보 |
| Endpoint | `https://apis.data.go.kr/1471057/FtnltCosmSrngPrdlstInfoService04/getSrngPrdlstInq` |
| 총 데이터 | 약 20,080건 |

### 보고품목 vs 심사품목

| 구분 | 보고품목 API | 심사품목 API |
|------|-------------|-------------|
| 대상 | 기존 고시 원료 사용 제품 | 새로운 원료/배합 심사 제품 |
| 등록 방식 | 보고 (간소화) | 심사 (엄격) |
| **우리 브랜드** | **포함** | 미포함 |

테스트 결과 우리 브랜드(해서린, 드롭비, 메디필, CNP) 매칭 0건 → **불필요**

---

## 6. 소스 코드 구조

```
05_src/03_mfds_api/
├── __init__.py                  # 패키지 초기화
├── config.py                    # API 설정 (엔드포인트, 효능 매핑)
├── client.py                    # 베이스 클라이언트 (요청/캐시/페이징)
├── functional_cosmetics.py      # 기능성화장품 보고품목 API
├── ingredient_info.py           # 원료성분정보 API
├── restricted_ingredients.py    # 사용제한 원료정보 API
├── run_all.py                   # 전체/개별/테스트 실행 스크립트
└── cache/                       # API 응답 캐시 (자동 생성)
```

### 주요 클래스

| 클래스 | 파일 | 주요 메서드 |
|--------|------|-----------|
| `MFDSBaseClient` | client.py | `_request()`, `fetch_all_pages()` |
| `FunctionalCosmeticsAPI` | functional_cosmetics.py | `match_product()`, `match_all_products()`, `save_results()` |
| `IngredientInfoAPI` | ingredient_info.py | `search_by_name()`, `enrich_ingredients()`, `save_results()` |
| `RestrictedIngredientsAPI` | restricted_ingredients.py | `download_all()`, `match_ingredients()`, `save_results()` |

---

## 7. 최종 데이터 파일 요약

### 7.1 생성 파일

| 파일명 | 위치 | 용도 | 건수 |
|--------|------|------|------|
| products_mfds.parquet | 02_processed_data/parquet/ | 기능성화장품 매칭 결과 | 259건 |
| products_mfds.xlsx | 02_processed_data/csv/ | 위와 동일 (Excel) | 259건 |
| ingredients_master.csv | 02_processed_data/csv/ | 성분 마스터 (보강) | 1,744건 |

### 7.2 products_mfds 컬럼 (33개)

| 컬럼명 | 설명 | 예시 |
|--------|------|------|
| product_code | 원본 제품 코드 | 1056665 |
| search_key | API 검색에 사용한 정제된 제품명 | 해서린스팟케어클리어젤 |
| mfds_matched | API 매칭 여부 | True |
| mfds_total_count | API 검색 결과 수 | 1 |
| mfds_item_name | MFDS 등록 제품명 | 해서린스팟케어클리어젤 |
| mfds_entp_name | 제조/판매 업체명 | 주식회사디와이디 |
| mfds_report_seq | 보고 일련번호 | 2021014801 |
| mfds_report_date | 보고일자 | 20210430 |
| mfds_item_ph | 제품 pH | 5.3 |
| mfds_target_flag | 기능성 분류 코드 (1~11) | 1 |
| mfds_target_flag_name | 기능성 분류명 | 제10조 제1항 제1호 |
| mfds_spf | 자외선차단지수 SPF | 35 |
| mfds_pa | 자외선차단등급 PA | 3.0 |
| is_whitening | 미백 여부 | True/False |
| is_anti_wrinkle | 주름개선 여부 | True/False |
| is_sunscreen | 자외선차단 여부 | True/False |
| is_hair_color ~ is_tanning | 4~11호 효능 여부 | True/False |

### 7.3 ingredients_master.csv 최종 컬럼 (7개)

| 컬럼명 | 설명 | 출처 |
|--------|------|------|
| ingredient_id | 성분 ID | 기존 |
| name | 성분명 (한글) | 기존 |
| is_caution | 주의 성분 여부 | 사용제한 원료 API (is_restricted=True → True) |
| name_eng | 영문명 | 원료성분정보 API |
| cas_no | CAS 번호 | 원료성분정보 API |
| is_restricted | 사용제한 여부 | 사용제한 원료 API |
| restriction_type | 규제 타입 (금지/한도/한도·금지) | 사용제한 원료 API |

---

## 8. 한계점 및 개선 방안

### 8.1 매칭률 한계

- functional=1 제품 중 97.4% 매칭 (266개 중 259개, 미매칭 7건)
- functional=0 제품은 매칭 대상 아님
- 전체 기준 27.1% (954개 중 259개)

### 8.2 API 제한사항

- EFFECT_YN 필드 사용 불가 (항상 'N')
- 사용제한 원료 API 검색 기능 비정상 → 전체 다운로드 방식
- 제품명 완전 일치 검색만 지원

### 8.3 개선 방안

- 업체명(ENTP_NAME) 활용한 교차 검증
- 부분 문자열 매칭 적용
- 수동 매핑 테이블 구축

---

## 9. 참고 자료

- [공공데이터포털](https://www.data.go.kr)
- [기능성화장품 보고품목정보 API](https://www.data.go.kr/data/15095680/openapi.do)
- [기능성화장품 심사품목정보 API](https://www.data.go.kr/data/15056939/openapi.do)
- [화장품 원료성분정보 API](https://www.data.go.kr/data/15111774/openapi.do)
- [화장품 사용제한 원료정보 API](https://www.data.go.kr/data/15111772/openapi.do)
- [화장품법 시행규칙 제10조](https://www.law.go.kr)
