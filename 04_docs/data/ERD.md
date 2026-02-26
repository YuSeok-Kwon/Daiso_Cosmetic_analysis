## 다이소 뷰티 프로젝트 최종 데이터 모델 명세

### 1. 리뷰 데이터 그룹

리뷰의 원천 정보와 텍스트 분석을 위한 구조입니다.

#### Reviews_core

리뷰의 기본 메타데이터와 재구매 여부를 관리합니다.

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `review_id` (PK) | INT | 리뷰 고유 식별자 |
| `product_code` (FK) | INT | 상품 코드 (Products_core 참조) |
| `user_id` (FK) | INT | 유저 고유 식별자 (Users_profile 참조) |
| `rating` | INT | 리뷰 평점 (1~5) |
| `review_date` | DATE | 리뷰 작성일 |
| `image_count` | INT | 리뷰에 첨부된 이미지 수 |
| `is_reorder` | BOOL | 재구매 리뷰 여부 ("재구매"로 시작하는 리뷰) |
| `promotion_id` (FK) | INT | 프로모션 ID (Promotions 참조, NULL=미매칭) |

#### Reviews_text

리뷰 전문과 텍스트 길이를 저장합니다.

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `review_id` (PK, FK) | INT | 리뷰 고유 식별자 (Reviews_core 참조) |
| `text` | TEXT | 리뷰 전문 텍스트 |
| `review_length` | INT | 리뷰 텍스트 글자 수 |

---

### 2. 상품 데이터 그룹

상품 마스터 정보와 성과 지표, 카테고리 분류를 포함합니다.

#### Products_core

상품 식별 정보입니다.

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `product_code` (PK) | INT | 상품 고유 코드 |
| `manufacturer_id` (FK) | INT | 제조사 ID (Manufacturer 참조) |
| `brand_id` (FK) | INT | 브랜드 ID (Brand 참조) |
| `name` | VARCHAR | 상품명 |
| `price` | INT | 판매 가격 (원) |
| `country` | VARCHAR | 제조 국가 |

#### Products_stats

상품 성과 지표 통합 테이블입니다.

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `product_code` (PK, FK) | INT | 상품 고유 코드 (Products_core 참조) |
| `likes` | INT | 좋아요 수 |
| `shares` | INT | 공유 수 |
| `review_count` | INT | 리뷰 수 |
| `first_review_date` | DATE | 첫 리뷰 작성일 (리뷰 없으면 NULL) |
| `engagement_score` | FLOAT | 인기도 점수 (0.15×likes + 0.30×shares + 0.55×review_count) |
| `cp_index` | FLOAT | 가성비 지표 ((engagement_score / price) × 1000) |
| `review_density` | FLOAT | 리뷰 밀도 (review_count / (마지막리뷰일 - 첫리뷰일)) |

#### Products_category

상품의 계층적 카테고리 정보입니다.

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `product_code` (PK, FK) | INT | 상품 고유 코드 (Products_core 참조) |
| `category_1` | VARCHAR | 1차 카테고리 (스킨케어, 메이크업 등) |
| `category_2` | VARCHAR | 2차 카테고리 (로션, 크림, 립스틱 등) |

#### Functional

기능성 화장품 인증 여부와 pH 범주를 관리합니다.

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `product_code` (PK, FK) | INT | 상품 고유 코드 (Products_core 참조) |
| `ITEM_PH` | VARCHAR | 제품 pH 값 (식약처 보고 기준) |
| **`ph_category`** | VARCHAR | pH 4단계 범주 (산성 3≤x<4.5, 약산성 4.5≤x<7, 중성 x=7, 알칼리성 x>7) |
| `is_whitening` | BOOL | 미백 기능성 인증 여부 |
| `is_wrinkle_reduction` | BOOL | 주름개선 기능성 인증 여부 |
| `is_sunscreen` | BOOL | 자외선차단 기능성 인증 여부 |
| `is_acne` | BOOL | 여드름성 피부 기능성 인증 여부 |

---

### 3. 성분 및 참조 그룹

OCR로 추출된 성분 사전과 브랜드/제조사 마스터 정보입니다.

#### Ingredients_dic

성분 사전입니다. **계열(`ingredient_type`)과 그룹(`application_role`)** 컬럼이 포함됩니다.

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `ingredient_id` (PK) | INT | 성분 고유 식별자 |
| `ingredient_name` | VARCHAR | 성분명 (한글) |
| **`ingredient_type`** | VARCHAR | 화학 계열 분류 (Polymer, Ester, Vitamin 등 33종) |
| `is_allergic` | BOOL | 알레르기 유발 성분 여부 (True/False) |
| `effect` | VARCHAR | 성분 효능 (Moisturizing, Anti-aging, Brightening 등) |

#### products_ingredients

상품과 성분 간의 다대다(N:M) 매핑 테이블입니다.

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `product_code` (PK, FK1) | INT | 상품 고유 코드 (Products_core 참조) |
| `ingredient_id` (PK, FK2) | INT | 성분 ID (Ingredients_dic 참조) |

#### Brand

브랜드 마스터 정보입니다.

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `brand_id` (PK) | INT | 브랜드 고유 식별자 |
| `name` | VARCHAR | 브랜드명 |

#### Manufacturer

제조사 마스터 정보입니다.

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `manufacturer_id` (PK) | INT | 제조사 고유 식별자 |
| `ENTP_NAME` | VARCHAR | 제조사명 (식약처 등록 기준) |

#### Promotions

프로모션 일정 및 유형(구매/리뷰 이벤트)을 관리합니다.

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `promotion_id` (PK) | INT | 프로모션 고유 식별자 |
| `start_date` | DATE | 프로모션 시작일 |
| `end_date` | DATE | 프로모션 종료일 |
| `description` | VARCHAR | 프로모션 설명 (다이소데이 뷰티 등) |
| `brand_id` (FK) | INT | 대상 브랜드 ID (Brand 참조) |
| `event_type` | VARCHAR | 이벤트 유형 (구매이벤트, 리뷰이벤트) |

---

### 4. 유저 및 분석 그룹 (Users & Loyalty)

고객 세그먼트와 재구매 패턴을 분석하는 테이블입니다.

#### User_id_map

마스킹된 유저 닉네임과 user_id 간의 1:1 매핑 테이블입니다. 증분 크롤링 시에도 동일 `user_masked`에 동일 `user_id`가 유지됩니다.

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `user_masked` (PK) | VARCHAR | 마스킹된 유저 닉네임 (예: `ths*****`) |
| `user_id` | INT | 유저 고유 식별자 |

#### Users_profile

유저의 활동 수준과 평점 성향을 분석합니다.

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `user_id` (PK) | INT | 유저 고유 식별자 |
| `user_total_reviews` | INT | 총 리뷰 작성 수 |
| `user_activity_level` | VARCHAR | 활동 수준 (Newbie, Junior, Regular, VIP) |
| `user_rating_tendency` | VARCHAR | 평점 성향 (Always Positive, Mostly Positive 등) |
| `review_tenure` | INT | 리뷰 활동 기간 (첫 리뷰 ~ 마지막 리뷰, 일 단위) |

#### Users_repurchase

재구매 경험이 있는 유저의 카테고리/브랜드 반복 구매 횟수와 재구매 평균 평점을 집계합니다. (재구매 이력이 없는 유저는 제외)

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `user_id` (PK) | INT | 유저 고유 식별자 (Users_profile 참조) |
| `reorder_user_category` | INT | 동일 카테고리 재구매 횟수 |
| `reorder_user_brand` | INT | 동일 브랜드 재구매 횟수 |
| `reorder_user_avg_rating` | FLOAT | 재구매 리뷰 평균 평점 |
