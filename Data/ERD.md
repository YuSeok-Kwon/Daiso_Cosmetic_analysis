# Why-pi 데이터 ERD

## Entity Relationship Diagram

```mermaid
erDiagram
    BRANDS {
        int brand_id PK "브랜드 ID"
        string name "브랜드명"
    }

    CATEGORIES {
        int category_id PK "카테고리 ID"
        string category_home "홈 카테고리"
        string category_1 "1차 카테고리"
        string category_2 "2차 카테고리"
    }

    INGREDIENTS_MASTER {
        int ingredient_id PK "성분 ID"
        string name "성분명"
        string ewg_grade "EWG 등급"
        bool is_caution "주의성분 여부"
    }

    PRODUCTS {
        int product_code PK "상품 코드"
        int brand_id FK "브랜드 ID"
        int category_id FK "카테고리 ID"
        string name "상품명"
        int price "정가"
        string country "원산지"
        datetime created_at "데이터생성일"
    }

    PRODUCT_ATTRIBUTES {
        int product_code PK_FK "상품 코드"
        string group "그룹"
        bool is_functional "기능성 여부"
        string price_tier "가격 등급"
        string price_position "가격 포지션"
        string item_ph "pH 농도"
        bool whitening "미백 기능"
        bool wrinkle_reduction "주름개선 기능"
        bool sunscreen "자외선차단 기능"
        bool can_vegan "비건 인증 가능"
        bool can_halal "할랄 인증 가능"
    }

    PRODUCT_METRICS {
        int product_code PK_FK "상품 코드"
        int likes "좋아요 수"
        int shares "공유 수"
        int review_count "리뷰 수"
        float engagement_score "참여 점수"
        float cp_index "가성비 지수"
        bool is_god_sung_bi "갓성비 여부"
        float review_density "리뷰 밀도"
        date last_updated "집계 기준일"
    }

    PRODUCT_INGREDIENTS {
        int product_code PK_FK "상품 코드"
        int ingredient_id PK_FK "성분 ID"
        int rank "전성분 순서"
    }

    USERS {
        int user_id PK "사용자 ID"
        string user_masked "마스킹된 이름"
        string activity_level "활동 레벨"
        string rating_tendency "평점 성향"
    }

    REVIEWS {
        int order_id PK "주문 ID"
        int product_code FK "상품 코드"
        int user_id FK "사용자 ID"
        date write_date "작성일"
        float rating "평점"
        string text "내용"
        int image_count "이미지 수"
        bool is_reorder "재구매 여부"
    }

    REVIEW_ANALYSIS {
        int order_id PK_FK "주문 ID"
        int length "글자 수"
        string length_category "길이 카테고리"
        string sentiment "감성분석 결과"
        float sentiment_score "감성 점수"
        string promo_type "프로모션 유형"
    }

    PROMOTIONS {
        int promotion_id PK "프로모션 ID"
        int brand_id FK "브랜드 ID"
        date start_date "시작일"
        date end_date "종료일"
        string description "설명"
        string event_type "이벤트 유형"
    }

    BRANDS ||--o{ PRODUCTS : "owns"
    CATEGORIES ||--o{ PRODUCTS : "categorizes"
    PRODUCTS ||--|| PRODUCT_ATTRIBUTES : "has details"
    PRODUCTS ||--|| PRODUCT_METRICS : "has stats"
    PRODUCTS ||--o{ REVIEWS : "receives"
    PRODUCTS ||--o{ PRODUCT_INGREDIENTS : "contains"
    INGREDIENTS_MASTER ||--o{ PRODUCT_INGREDIENTS : "included in"
    USERS ||--o{ REVIEWS : "writes"
    REVIEWS ||--|| REVIEW_ANALYSIS : "analyzed as"
    BRANDS ||--o{ PROMOTIONS : "runs"
```

## 테이블 요약

### 마스터 테이블

| 테이블             | 파일명                  | 행 수 | 컬럼 수 | 설명         |
| ------------------ | ----------------------- | ----- | ------- | ------------ |
| BRANDS             | brands.csv              | 94    | 2       | 브랜드 마스터 |
| CATEGORIES         | categories.csv          | 15    | 4       | 카테고리 마스터 |
| INGREDIENTS_MASTER | ingredients_master.csv  | 1,744 | 4       | 성분 마스터  |

### 제품 테이블

| 테이블              | 파일명                  | 행 수  | 컬럼 수 | 설명         |
| ------------------- | ----------------------- | ------ | ------- | ------------ |
| PRODUCTS            | products.csv            | 950    | 7       | 상품 기본 정보 |
| PRODUCT_ATTRIBUTES  | product_attributes.csv  | 944    | 11      | 상품 속성    |
| PRODUCT_METRICS     | product_metrics.csv     | 950    | 9       | 상품 지표    |
| PRODUCT_INGREDIENTS | product_ingredients.csv | 26,772 | 3       | 상품-성분 연결 |

### 사용자/리뷰 테이블

| 테이블          | 파일명             | 행 수   | 컬럼 수 | 설명       |
| --------------- | ------------------ | ------- | ------- | ---------- |
| USERS           | users.csv          | 25,536  | 4       | 사용자 정보 |
| REVIEWS         | reviews.csv        | 323,493 | 8       | 리뷰 기본 정보 |
| REVIEW_ANALYSIS | review_analysis.csv | 323,493 | 6       | 리뷰 분석 결과 |

### 프로모션 테이블

| 테이블     | 파일명         | 행 수 | 컬럼 수 | 설명         |
| ---------- | -------------- | ----- | ------- | ------------ |
| PROMOTIONS | promotions.csv | 118   | 6       | 프로모션 정보 |

## 테이블 관계

| 관계 | From              | To                  | 키            | 타입  | 설명                     |
| ---- | ----------------- | ------------------- | ------------- | ----- | ------------------------ |
| 1:N  | BRANDS            | PRODUCTS            | brand_id      | PK-FK | 브랜드별 여러 상품       |
| 1:N  | CATEGORIES        | PRODUCTS            | category_id   | PK-FK | 카테고리별 여러 상품     |
| 1:1  | PRODUCTS          | PRODUCT_ATTRIBUTES  | product_code  | PK-FK | 상품별 속성 정보         |
| 1:1  | PRODUCTS          | PRODUCT_METRICS     | product_code  | PK-FK | 상품별 지표 정보         |
| 1:N  | PRODUCTS          | REVIEWS             | product_code  | PK-FK | 상품별 여러 리뷰         |
| 1:N  | PRODUCTS          | PRODUCT_INGREDIENTS | product_code  | PK-FK | 상품별 여러 성분         |
| 1:N  | INGREDIENTS_MASTER | PRODUCT_INGREDIENTS | ingredient_id | PK-FK | 성분별 여러 상품         |
| 1:N  | USERS             | REVIEWS             | user_id       | PK-FK | 사용자별 여러 리뷰       |
| 1:1  | REVIEWS           | REVIEW_ANALYSIS     | order_id      | PK-FK | 리뷰별 분석 결과         |
| 1:N  | BRANDS            | PROMOTIONS          | brand_id      | PK-FK | 브랜드별 여러 프로모션   |

## 주요 키 정보

### Primary Keys (PK)

- **BRANDS**: `brand_id` (94 unique)
- **CATEGORIES**: `category_id` (15 unique)
- **INGREDIENTS_MASTER**: `ingredient_id` (1,744 unique)
- **PRODUCTS**: `product_code` (950 unique)
- **USERS**: `user_id` (25,536 unique)
- **REVIEWS**: `order_id` (323,493 unique)
- **PROMOTIONS**: `promotion_id` (118 unique)

### Foreign Keys (FK)

- **PRODUCTS.brand_id** → BRANDS.brand_id
- **PRODUCTS.category_id** → CATEGORIES.category_id
- **PRODUCT_ATTRIBUTES.product_code** → PRODUCTS.product_code
- **PRODUCT_METRICS.product_code** → PRODUCTS.product_code
- **PRODUCT_INGREDIENTS.product_code** → PRODUCTS.product_code
- **PRODUCT_INGREDIENTS.ingredient_id** → INGREDIENTS_MASTER.ingredient_id
- **REVIEWS.product_code** → PRODUCTS.product_code
- **REVIEWS.user_id** → USERS.user_id
- **REVIEW_ANALYSIS.order_id** → REVIEWS.order_id
- **PROMOTIONS.brand_id** → BRANDS.brand_id

## 파일 위치

```
/Data/csv/union/
├── brands.csv                 (마스터)
├── categories.csv             (마스터)
├── ingredients_master.csv     (마스터)
├── products.csv               (제품)
├── product_attributes.csv     (제품 속성)
├── product_metrics.csv        (제품 지표)
├── product_ingredients.csv    (제품-성분 연결)
├── users.csv                  (사용자)
├── reviews.csv                (리뷰)
├── review_analysis.csv        (리뷰 분석)
└── promotions.csv             (프로모션)
```

## 변경 이력

| 날짜       | 변경 내용                                                                 |
| ---------- | ------------------------------------------------------------------------- |
| 2026-02-12 | 11개 테이블로 정규화 (brands, categories, ingredients_master 마스터 추가) |
