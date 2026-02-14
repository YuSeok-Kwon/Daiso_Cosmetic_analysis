# MFDS 기능성화장품 API 데이터 수집 가이드

## 1. 개요

식품의약품안전처(MFDS)의 기능성화장품 보고품목정보 API를 활용하여 제품의 기능성 여부 및 효능 정보(미백, 주름개선, 자외선차단)를 수집하는 과정을 문서화합니다.

### 1.1 목적

- products.parquet의 제품들에 대해 기능성화장품 여부 검증
- 미백, 주름개선, 자외선차단 등 효능 정보 수집
- 기존 product_attributes.csv 데이터와 비교 검증

### 1.2 사용 API

- **API명**: 기능성화장품 보고품목정보 조회 서비스
- **제공기관**: 식품의약품안전처
- **데이터포털**: https://www.data.go.kr

---

## 2. API 연결

### 2.1 API Endpoint

```
http://apis.data.go.kr/1471000/FtnltCosmRptPrdlstInfoService/getRptPrdlstInq
```

### 2.2 인증 방식

- **인증키 타입**: 공공데이터포털 발급 ServiceKey (Decoding)
- **전달 방식**: URL 쿼리 파라미터

### 2.3 요청 파라미터

| 파라미터명 | 필수 | 설명              | 예시                   |
| ---------- | ---- | ----------------- | ---------------------- |
| serviceKey | Y    | 인증키            | b90a9b72...            |
| pageNo     | N    | 페이지 번호       | 1                      |
| numOfRows  | N    | 한 페이지 결과 수 | 10                     |
| type       | N    | 응답 형식         | json                   |
| ITEM_NAME  | N    | 제품명 (검색용)   | 해서린스팟케어클리어젤 |
| ENTP_NAME  | N    | 업체명            | (주)와이-피코스메틱    |

### 2.4 요청 예시

```python
import requests
import urllib.parse

base_url = "http://apis.data.go.kr/1471000/FtnltCosmRptPrdlstInfoService/getRptPrdlstInq"
params = {
    'serviceKey': 'YOUR_API_KEY',
    'pageNo': 1,
    'numOfRows': 10,
    'type': 'json',
    'ITEM_NAME': '해서린스팟케어클리어젤'
}

response = requests.get(base_url, params=params)
data = response.json()
```

### 2.5 응답 구조

```json
{
  "header": {
    "resultCode": "00",
    "resultMsg": "NORMAL_CODE"
  },
  "body": {
    "pageNo": 1,
    "totalCount": 1,
    "numOfRows": 10,
    "items": [
      {
        "ITEM_NAME": "해서린스팟케어클리어젤",
        "ENTP_NAME": "(주)와이-피코스메틱",
        "REPORT_SEQ": "...",
        "COSMETIC_TARGET_FLAG": "제10조 제1항 제1호",
        "COSMETIC_TARGET_FLAG_NAME": "피부의 미백에 도움을 주는 제품",
        ...
      }
    ]
  }
}
```

---

## 3. 제품명 매칭 방법

### 3.1 문제점

우리 데이터와 MFDS 데이터의 제품명 형식이 다릅니다:

| 구분                  | 예시                             |
| --------------------- | -------------------------------- |
| **우리 데이터** | 해서린 스팟 케어 클리어 젤 10 ml |
| **MFDS 데이터** | 해서린스팟케어클리어젤           |

**주요 차이점:**

- 띄어쓰기 유무
- 용량 정보 (10ml, 50g 등) 포함 여부
- 대괄호/소괄호 내용 ([기획], (리뉴얼) 등)
- 특수문자 포함 여부

### 3.2 제품명 정제 함수

```python
import re

def clean_product_name(name):
    """제품명 정제 - API 검색을 위한 표준화"""
    if not name:
        return ""

    # 1. 대괄호 내용 제거 [기획], [세트] 등
    name = re.sub(r'\[.*?\]', '', name)

    # 2. 소괄호 내용 제거 (리뉴얼), (신제품) 등
    name = re.sub(r'\(.*?\)', '', name)

    # 3. "by 브랜드명" 패턴 제거
    name = re.sub(r'\s*by\s+\w+', '', name, flags=re.IGNORECASE)

    # 4. 용량 정보 제거 (숫자 + 단위)
    name = re.sub(r'\d+\s*(ml|g|mg|L|oz|개|매|정)\b', '', name, flags=re.IGNORECASE)

    # 5. X, x 문자 제거 (세트 표시용)
    name = re.sub(r'\s*[xX]\s*', '', name)

    # 6. 특수문자 제거 (한글, 영문, 숫자만 유지)
    name = re.sub(r'[^\w\s가-힣a-zA-Z]', '', name)

    # 7. 연속 공백 정리
    name = ' '.join(name.split())

    # 8. 모든 공백 제거 (MFDS는 공백 없음)
    name = re.sub(r'\s+', '', name.strip())

    return name
```

### 3.3 매칭 결과

| 구분           | 제품 수 | 매칭 성공 | 매칭률 |
| -------------- | ------- | --------- | ------ |
| functional=1   | 254     | 186       | 73.2%  |
| functional=0   | 700     | 0         | 0%     |
| **전체** | 954     | 186       | 19.5%  |

**매칭 실패 원인:**

- 브랜드명이 제품명에 다르게 포함된 경우
- 완전히 다른 제품명 사용
- API에 미등록된 제품

---

## 4. 효능 판별 방법

### 4.1 주의사항: EFFECT_YN 필드 사용 금지

API 응답의 `EFFECT_YN1`, `EFFECT_YN2`, `EFFECT_YN3` 필드는 **모두 'N'**으로 반환되어 사용할 수 없습니다.

```python
# 잘못된 방법
is_whitening = item.get('EFFECT_YN1') == 'Y'  # 항상 False
```

### 4.2 올바른 방법: COSMETIC_TARGET_FLAG 사용

`COSMETIC_TARGET_FLAG` 또는 `COSMETIC_TARGET_FLAG_NAME` 필드를 사용합니다.

#### 기능성화장품 전체 효능 목록 (화장품법 시행규칙 제10조)

| FLAG 값             | 효능        | 영문명        | 설명                                                                            |
| ------------------- | ----------- | ------------- | ------------------------------------------------------------------------------- |
| 제10조 제1항 제1호  | 미백        | whitening     | 피부의 미백에 도움을 주는 제품                                                  |
| 제10조 제1항 제2호  | 주름개선    | anti-wrinkle  | 피부의 주름개선에 도움을 주는 제품                                              |
| 제10조 제1항 제3호  | 자외선차단  | sunscreen     | 강한 햇볕을 방지하여 피부를 곱게 태워주거나 자외선으로부터 피부를 보호하는 제품 |
| 제10조 제1항 제4호  | 모발 색상   | hair_color    | 모발의 색상 변화·제거 또는 영양공급에 도움을 주는 제품                         |
| 제10조 제1항 제5호  | 체모 제거   | hair_removal  | 체모를 제거하는 데 도움을 주는 제품                                             |
| 제10조 제1항 제6호  | 여드름 완화 | acne          | 여드름성 피부를 완화하는 데 도움을 주는 제품                                    |
| 제10조 제1항 제7호  | 아토피 보습 | atopic        | 아토피성 피부로 인한 건조함 등을 완화하는 데 도움을 주는 제품                   |
| 제10조 제1항 제8호  | 튼살 완화   | stretch_marks | 피부의 튼살로 인한 붉은선을 완화하는 데 도움을 주는 제품                        |
| 제10조 제1항 제9호  | 탈모 완화   | hair_loss     | 탈모 증상의 완화에 도움을 주는 제품                                             |
| 제10조 제1항 제10호 | 눈가 개선   | eye_wrinkle   | 눈 주위 일시적 개선에 도움을 주는 제품                                          |
| 제10조 제1항 제11호 | 선탠        | tanning       | 피부를 곱게 태워주는 기능을 가진 제품                                           |

> **참고**: 본 프로젝트 데이터에서는 1~3호(미백, 주름개선, 자외선차단)만 확인됨

#### 현재 수집된 효능 (1~3호)

| FLAG 값            | 효능                    |
| ------------------ | ----------------------- |
| 제10조 제1항 제1호 | 미백 (whitening)        |
| 제10조 제1항 제2호 | 주름개선 (anti-wrinkle) |
| 제10조 제1항 제3호 | 자외선차단 (sunscreen)  |

### 4.3 효능 판별 함수

```python
def parse_target_flag(flag_name):
    """COSMETIC_TARGET_FLAG_NAME으로 효능 판단 (전체 11개 효능 지원)"""
    result = {
        'is_whitening': False,      # 1호: 미백
        'is_anti_wrinkle': False,   # 2호: 주름개선
        'is_sunscreen': False,      # 3호: 자외선차단
        'is_hair_color': False,     # 4호: 모발 색상
        'is_hair_removal': False,   # 5호: 체모 제거
        'is_acne': False,           # 6호: 여드름 완화
        'is_atopic': False,         # 7호: 아토피 보습
        'is_stretch_marks': False,  # 8호: 튼살 완화
        'is_hair_loss': False,      # 9호: 탈모 완화
        'is_eye_wrinkle': False,    # 10호: 눈가 개선
        'is_tanning': False,        # 11호: 선탠
    }

    if flag_name:
        if '1호' in flag_name:
            result['is_whitening'] = True
        if '2호' in flag_name:
            result['is_anti_wrinkle'] = True
        if '3호' in flag_name:
            result['is_sunscreen'] = True
        if '4호' in flag_name:
            result['is_hair_color'] = True
        if '5호' in flag_name:
            result['is_hair_removal'] = True
        if '6호' in flag_name:
            result['is_acne'] = True
        if '7호' in flag_name:
            result['is_atopic'] = True
        if '8호' in flag_name:
            result['is_stretch_marks'] = True
        if '9호' in flag_name:
            result['is_hair_loss'] = True
        if '10호' in flag_name:
            result['is_eye_wrinkle'] = True
        if '11호' in flag_name:
            result['is_tanning'] = True

    return result
```

**간단 버전 (1~3호만)**

```python
def parse_target_flag_simple(flag_name):
    """COSMETIC_TARGET_FLAG_NAME으로 효능 판단 (1~3호)"""
    is_whitening = '1호' in flag_name if flag_name else False
    is_anti_wrinkle = '2호' in flag_name if flag_name else False
    is_sunscreen = '3호' in flag_name if flag_name else False

    return is_whitening, is_anti_wrinkle, is_sunscreen
```

### 4.4 효능별 통계

| 효능       | 제품 수 |
| ---------- | ------- |
| 미백       | 94건    |
| 주름개선   | 73건    |
| 자외선차단 | 19건    |

---

## 5. 수집 데이터 컬럼 설명

### 5.1 products_mfds.parquet / products_mfds.xlsx

| 컬럼명                   | 설명                            | 예시                           |
| ------------------------ | ------------------------------- | ------------------------------ |
| product_code             | 원본 제품 코드                  | PRD001                         |
| search_key               | API 검색에 사용한 정제된 제품명 | 해서린스팟케어클리어젤         |
| mfds_matched             | API 매칭 여부                   | True                           |
| mfds_total_count         | API 검색 결과 수                | 1                              |
| mfds_item_name           | MFDS 등록 제품명                | 해서린스팟케어클리어젤         |
| mfds_entp_name           | 제조/판매 업체명                | (주)와이-피코스메틱            |
| mfds_report_seq          | 보고 일련번호                   | 20230001234                    |
| mfds_report_date         | 보고일자                        | 2023-01-15                     |
| mfds_item_ph             | 제품 pH                         | 5.5-6.5                        |
| mfds_target_flag         | 기능성 분류 코드                | 제10조 제1항 제1호             |
| mfds_target_flag_name    | 기능성 분류명                   | 피부의 미백에 도움을 주는 제품 |
| mfds_std_code            | 기준 및 시험방법 코드           | -                              |
| mfds_std_name            | 기준 및 시험방법 명칭           | -                              |
| mfds_ee_code             | 효력시험 코드                   | -                              |
| mfds_ee_name             | 효력시험 명칭                   | -                              |
| mfds_spf                 | 자외선차단지수 SPF              | SPF50+                         |
| mfds_pa                  | 자외선차단등급 PA               | PA++++                         |
| is_water_proofing        | 내수성 여부                     | True/False                     |
| mfds_water_proofing_name | 내수성 명칭                     | 내수성                         |
| mfds_report_flag_code    | 보고 구분 코드                  | -                              |
| mfds_report_flag_name    | 보고 구분 명칭                  | -                              |
| is_ethanol_over          | 에탄올 10% 초과 여부            | True/False                     |
| is_whitening             | 미백 기능성 여부                | True/False                     |
| is_anti_wrinkle          | 주름개선 기능성 여부            | True/False                     |
| is_sunscreen             | 자외선차단 기능성 여부          | True/False                     |

---

## 6. 데이터 검증 및 오류 수정

### 6.1 발견된 오류

기존 product_attributes.csv의 `is_functional=0`이지만 API에서 매칭된 제품 8건 발견:

| product_code | 제품명                               | 기존 값 | 수정 값 |
| ------------ | ------------------------------------ | ------- | ------- |
| Y1005        | 해서린 듀오 커버 팩트                | 0       | 1       |
| Y1006        | 해서린 듀오 커버 팩트                | 0       | 1       |
| Y1007        | 해서린 듀오 커버 팩트                | 0       | 1       |
| Y1063        | 리애 카밍 워터 팩                    | 0       | 1       |
| Y1064        | 리애 카밍 워터 팩                    | 0       | 1       |
| Y1086        | 더마틱스 울트라                      | 0       | 1       |
| Y1087        | 더마틱스 울트라                      | 0       | 1       |
| Y1161        | 클레어스 프레쉴리 쥬스드 비타민 드롭 | 0       | 1       |

### 6.2 수정된 파일

- `products.parquet`: functional 컬럼 0→1 수정
- `product_attributes.csv`: is_functional 컬럼 0→1 수정

---

## 7. 한계점 및 주의사항

### 7.1 매칭률 한계

- 전체 954개 제품 중 186개만 매칭 (19.5%)
- functional=1 제품 중 73.2%만 매칭
- 제품명 형식 차이로 인한 매칭 실패 다수

### 7.2 API 제한사항

- EFFECT_YN 필드 사용 불가 (항상 'N')
- 일부 API(심사품목 정보)는 500 오류 발생
- 제품명 검색 시 완전 일치만 지원

### 7.3 개선 방안

- 업체명(ENTP_NAME) 활용한 교차 검증
- 부분 문자열 매칭 적용
- 수동 매핑 테이블 구축

---

## 8. 심사품목 API (추가 검토)

### 8.1 개요

보고품목 API 외에 **심사품목 API**도 검토하였으나, 본 프로젝트에는 적합하지 않음.

### 8.2 API 정보

| 항목 | 내용 |
|------|------|
| API명 | 기능성화장품 심사품목 정보 |
| Endpoint | `https://apis.data.go.kr/1471057/FtnltCosmSrngPrdlstInfoService04/getSrngPrdlstInq` |
| 신청 페이지 | [공공데이터포털](https://www.data.go.kr/data/15056939/openapi.do) |
| 총 데이터 | 약 20,080건 |

### 8.3 보고품목 vs 심사품목 차이

| 구분 | 보고품목 API | 심사품목 API |
|------|-------------|-------------|
| 대상 | 기존 고시 원료 사용 제품 | 새로운 원료/배합 심사 제품 |
| 등록 방식 | 보고 (간소화) | 심사 (엄격) |
| 제품 특성 | 일반 기능성화장품 | 신규/특수 기능성화장품 |
| **우리 브랜드** | ✅ 포함 | ❌ 미포함 |

### 8.4 심사품목 API 테스트 결과

```
=== 심사품목 API 매칭 결과 ===
총 제품: 954건
매칭 성공: 0건 (0.0%)

브랜드별 검색:
- 해서린: 0건
- 드롭비: 0건
- 메디필: 0건
- CNP: 0건
```

### 8.5 심사품목 API 효능 분포

심사품목 API 전체 20,080건 분석 결과:

| 효능 | 건수 | 비율 |
|------|------|------|
| 미백 | ~6,000건 | 30% |
| 주름개선 | ~4,000건 | 20% |
| 자외선차단 | ~500건 | 2.5% |
| **여드름/탈모 등 (4~11호)** | **0건** | **0%** |

### 8.6 결론

- **심사품목 API는 본 프로젝트에 불필요**
- 우리 브랜드(해서린, 드롭비, 메디필 등)는 모두 **보고품목**으로 등록
- 4~11호 효능(여드름, 탈모 등)은 **두 API 모두 미제공**
- 기존 **보고품목 API(products_mfds)로 충분**

### 8.7 효능별 API 제공 현황

| 효능 | 보고품목 API | 심사품목 API |
|------|-------------|-------------|
| 1호 미백 | ✅ 94건 | ✅ (우리 제품 없음) |
| 2호 주름개선 | ✅ 73건 | ✅ (우리 제품 없음) |
| 3호 자외선차단 | ✅ 19건 | ✅ (우리 제품 없음) |
| 4~11호 (여드름, 탈모 등) | ❌ 없음 | ❌ 없음 |

---

## 9. 참고 자료

- [공공데이터포털](https://www.data.go.kr)
- [기능성화장품 보고품목정보 API](https://www.data.go.kr/data/15095680/openapi.do)
- [기능성화장품 심사품목정보 API](https://www.data.go.kr/data/15056939/openapi.do)
- [화장품법 시행규칙 제10조](https://www.law.go.kr)
