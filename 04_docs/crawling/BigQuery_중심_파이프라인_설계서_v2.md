# BigQuery 중심 월간 자동 파이프라인 설계서 v2.0

> **작성일:** 2026-03-03 (v1: 2026-02-27)
> **작성자:** 권유석
> **목적:** 로컬 CSV 제거 → BigQuery Single Source of Truth 전환 + ABSA/SLI 자동화 파이프라인 완성 + 월간 cron 스케줄링 활성화

---

## v2 변경사항 (v1 → v2)

| 항목                           | v1                    | v2                                                   |
| ------------------------------ | --------------------- | ---------------------------------------------------- |
| `auto_schedule.enabled`      | `false`             | **`true`** (cron 등록 완료)                  |
| `search_trend.enabled`       | `true`              | **`false`** (수동 실행 전환)                 |
| `scheduler.py` 대상 스크립트 | `run_pipeline.py`   | **`run_monthly_pipeline.py`**                |
| `scheduler.py` cron 플래그   | `--local-only`      | **(제거)** — run_monthly_pipeline에 없는 옵션 |
| SLI ML 모델                    | XGBoost (설계서 오기) | **LightGBM** (실제 구현)                       |
| Phase 3 검색트렌드             | ⬜ BQ 연결 미완       | **의도적 비활성화** (수동 전용)                |
| Phase 5 스케줄링               | ⬜ 미착수             | **cron 등록 완료**                             |
| 코드 수정 계획                 | 의사코드 (계획)       | **전량 구현 완료**                             |

---

## 1. 현행 아키텍처 (AS-IS)

### 1.1 데이터 흐름

```
[크롤러] → 3개 raw CSV (로컬)
    ↓
[transformer.py] → 13개 ERD 테이블 (로컬 final/ CSV)
    ↓
[CrawlerETLv2] → BigQuery UPSERT (선택, 현재 비활성)
    ↓
[ABSA 추론] → 별도 CSV (02_outputs/ABSA/inference/)
[SLI 계산] → 별도 CSV (02_outputs/Sli/) ← 노트북 수동 실행
[검색트렌드] → 별도 CSV (02_outputs/Search_Trend/) ← 스크립트 수동 실행
[대시보드] → 정적 PNG 이미지만 존재
```

### 1.2 문제점

| 문제            | 상세                                                                         |
| --------------- | ---------------------------------------------------------------------------- |
| 데이터 분산     | 13개 ERD + ABSA + SLI + 검색트렌드가 각각 다른 위치에 CSV로 존재             |
| 증분 병합 이슈  | transformer.py가 전체 raw를 기대하므로, 증분 크롤링 시 기존 32만건 누락 위험 |
| 수동 실행       | ABSA, SLI, 검색트렌드가 각각 독립 실행, 자동 연결 없음                       |
| SLI 노트북 의존 | SLI 계산이 주피터 노트북에만 존재, 스크립트화 안 됨                          |
| 대시보드 부재   | 인터랙티브 대시보드 없음, 정적 PNG만                                         |
| 로컬 의존성     | 모든 데이터가 로컬 CSV에 의존, 팀 공유 불가                                  |

---

## 2. 목표 아키텍처 (TO-BE)

### 2.1 핵심 원칙

- **BigQuery = Single Source of Truth**: 모든 데이터를 BQ에 적재, 로컬에는 임시 파일만
- **증분 UPSERT**: transformer가 신규 데이터만 변환 → BQ MERGE로 기존 데이터와 합침
- **파이프라인 연쇄**: 크롤링 → ABSA → SLI → 대시보드가 자동으로 연결
- **월 1회 자동 실행**: cron 스케줄러로 전체 파이프라인 트리거 (매월 1일 03:00)
- **검색트렌드 분리**: API 호출 제한·비용 고려하여 수동 실행으로 분리

### 2.2 전체 데이터 흐름

```
┌──────────────────────────────────────────────────────────────────┐
│                    월간 자동 파이프라인                           │
│                    cron: 0 3 1 * * (매월 1일 03시)               │
│                                                                  │
│  Step 1: 증분 크롤링                                             │
│  ┌─────────────┐    ┌───────────────┐    ┌──────────────────┐   │
│  │ Selenium     │───→│ transformer   │───→│ CrawlerETLv2     │   │
│  │ 크롤러       │    │ (신규만 변환)  │    │ UPSERT to BQ     │   │
│  └─────────────┘    └───────────────┘    └──────────────────┘   │
│       │                                          │               │
│       │ crawl_history                            ▼               │
│       │ (BQ 조회)                     ┌──────────────────┐      │
│       └──────────────────────────────→│ BigQuery         │      │
│                                       │ daiso 데이터셋    │      │
│  Step 2: 변환 + BQ UPSERT            │                  │      │
│  (Step 1에서 자동 처리)               │  13개 ERD 테이블  │      │
│                                       │  + 5개 분석 테이블│      │
│  Step 3: ABSA 증분 추론              │                  │      │
│  ┌─────────────┐                      │                  │      │
│  │ KcELECTRA   │←── 신규 review_id ──│                  │      │
│  │ 추론        │───→ UPSERT ────────→│                  │      │
│  └─────────────┘                      │                  │      │
│                                       │                  │      │
│  Step 4: SLI 연착륙 재계산            │                  │      │
│  ┌─────────────┐                      │                  │      │
│  │ DTW+생존    │←── 리뷰 시계열 ─────│                  │      │
│  │ +규칙+ML    │───→ UPSERT ────────→│                  │      │
│  │ (LightGBM)  │                      │                  │      │
│  └─────────────┘                      │                  │      │
│                                       │                  │      │
│  Step 5: 네이버 검색트렌드 [비활성]   │                  │      │
│  ┌─────────────┐                      │                  │      │
│  │ config에서   │  enabled: false      │                  │      │
│  │ 자동 SKIP    │  (수동 실행 전용)    │                  │      │
│  └─────────────┘                      │                  │      │
│                                       │                  │      │
│  Step 6: 대시보드 생성                │                  │      │
│  ┌─────────────┐                      │                  │      │
│  │ HTML/JS     │←── 집계 쿼리 ───────│                  │      │
│  │ 대시보드    │                      │                  │      │
│  └─────────────┘                      └──────────────────┘      │
│                                                                  │
│  * 각 Step 완료 시 pipeline_log 자동 기록                        │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                    수동 실행 파이프라인                           │
│                                                                  │
│  검색트렌드 수집 (네이버 DataLab + Search API)                    │
│  ┌─────────────┐                      ┌──────────────────┐      │
│  │ DataLab API │←── 연착륙 목록 ─────│ BigQuery         │      │
│  │ Search API  │───→ UPSERT ────────→│ search_trends    │      │
│  └─────────────┘                      └──────────────────┘      │
│                                                                  │
│  실행: python run_monthly_pipeline.py --steps 5                  │
│  (또는 기존 스크립트 직접 실행)                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. ERD v3 — 18개 테이블 설계

### 3.1 테이블 목록

기존 13개 테이블 + 신규 5개 테이블 = 총 18개

| #            | 테이블명                 | 유형           | 행 규모 (예상)     | 변경사항                    |
| ------------ | ------------------------ | -------------- | ------------------ | --------------------------- |
| 1            | brand                    | 마스터         | ~100               | 유지                        |
| 2            | manufacturer             | 마스터         | ~70                | 유지                        |
| 3            | ingredients_dic          | 마스터         | ~1,800             | 유지                        |
| 4            | promotions               | 마스터         | ~20                | 유지                        |
| 5            | products_core            | 제품           | ~950               | 유지                        |
| 6            | products_category        | 제품           | ~950               | 유지                        |
| 7            | products_stats           | 제품           | ~950               | 유지                        |
| 8            | products_ingredients     | 제품           | ~28,000            | 유지                        |
| 9            | functional               | 제품           | ~260               | 유지                        |
| 10           | users_profile            | 유저           | ~25,000            | 유지                        |
| 11           | users_repurchase         | 유저           | ~14,000            | 유지                        |
| 12           | reviews_core             | 리뷰           | ~323,000           | 유지                        |
| 13           | reviews_text             | 리뷰           | ~323,000           | **행 수 정합성 수정** |
| **14** | **review_absa**    | **ABSA** | **~323,000** | **신규**              |
| **15** | **review_aspects** | **ABSA** | **~500,000** | **신규**              |
| **16** | **sli_results**    | **SLI**  | **~950**     | **신규**              |
| **17** | **search_trends**  | **검색** | **~150,000** | **신규**              |
| **18** | **pipeline_log**   | **운영** | **~100/년**  | **신규**              |

### 3.2 ERD 다이어그램

```
┌─────────────┐       ┌──────────────┐       ┌─────────────────────┐
│   brand     │       │ manufacturer │       │  ingredients_dic    │
│─────────────│       │──────────────│       │─────────────────────│
│ brand_id PK │       │manufacturer_ │       │ ingredient_id PK    │
│ name        │       │  id PK       │       │ ingredient_name     │
└──────┬──────┘       │ ENTP_NAME    │       │ ingredient_type     │
       │              └──────┬───────┘       │ is_allergic         │
       │                     │               │ effect              │
       ▼                     ▼               └──────────┬──────────┘
┌──────────────────────────────────────┐                │
│              products_core           │                │
│──────────────────────────────────────│                │
│ product_code PK                      │                │
│ manufacturer_id FK → manufacturer    │                │
│ brand_id FK → brand                  │                │
│ name, price, country                 │                │
└──────────────────┬───────────────────┘                │
                   │                                    │
       ┌───────────┼───────────┬───────────┐            │
       ▼           ▼           ▼           ▼            │
┌────────────┐┌────────────┐┌────────────┐┌─────────────┴──┐
│ products_  ││ products_  ││ products_  ││ products_      │
│ category   ││ stats      ││ ingredients││ ingredients    │
│────────────││────────────││────────────││────────────────│
│product_code││product_code││product_code││ ingredient_id  │
│category_1  ││likes       ││ingredient_ ││  FK→ingredients│
│category_2  ││shares      ││  id FK     ││  _dic          │
│            ││review_count│└────────────┘└────────────────┘
│            ││engagement_ │
│            ││  score     │  ┌────────────┐
│            ││cp_index    │  │ functional │
└────────────┘│review_     │  │────────────│
              │  density   │  │product_code│
              └────────────┘  │ITEM_PH     │
                              │ph_category │
                              │is_whitening│
                              │...         │
                              └────────────┘

┌──────────────┐       ┌─────────────┐
│  promotions  │       │ user_id_map │
│──────────────│       │─────────────│
│promotion_id  │       │user_masked  │
│  PK          │       │  PK         │
│start_date    │       │user_id      │
│end_date      │       └──────┬──────┘
│description   │              │
│brand_id FK   │              ▼
│event_type    │       ┌─────────────────┐     ┌─────────────────┐
└──────┬───────┘       │  users_profile  │     │users_repurchase │
       │               │─────────────────│     │─────────────────│
       │               │user_id PK       │     │user_id PK FK    │
       │               │user_total_      │     │reorder_user_    │
       │               │  reviews        │     │  category       │
       │               │user_activity_   │     │reorder_user_    │
       │               │  level          │     │  brand          │
       │               │user_rating_     │     │reorder_user_    │
       │               │  tendency       │     │  avg_rating     │
       │               │review_tenure    │     └─────────────────┘
       │               └────────┬────────┘
       │                        │
       ▼                        ▼
┌───────────────────────────────────────┐
│              reviews_core             │
│───────────────────────────────────────│
│ review_id PK                          │
│ product_code FK → products_core       │
│ user_id FK → users_profile            │
│ rating (1~5)                          │
│ review_date                           │
│ image_count                           │
│ is_reorder                            │
│ promotion_id FK → promotions (NULL OK)│
└──────────┬──────────┬─────────────────┘
           │          │
           ▼          ▼
┌──────────────┐  ┌──────────────────────────┐
│ reviews_text │  │     review_absa [신규]    │
│──────────────│  │──────────────────────────│
│review_id PK  │  │ review_id PK FK          │
│  FK          │  │ sentiment                │
│text          │  │ sentiment_score (-1~1)   │
│review_length │  │ is_ambiguous             │
└──────────────┘  │ aspect_count             │
                  │ absa_version             │
                  │ inferred_at              │
                  └──────────┬───────────────┘
                             │
                             ▼
                  ┌──────────────────────────┐
                  │   review_aspects [신규]   │
                  │──────────────────────────│
                  │ review_id PK1 FK         │
                  │ aspect PK2               │
                  │ aspect_sentiment         │
                  │ aspect_confidence (0~1)  │
                  └──────────────────────────┘


┌───────────────────────────────┐
│      sli_results [신규]        │
│───────────────────────────────│
│ product_code PK FK             │
│ is_soft_landing_dtw            │
│ is_soft_landing_surv           │
│ is_soft_landing_rule           │
│ is_soft_landing_ml             │
│ total_votes                    │
│ final_soft_landing             │
│ confidence                     │
│ ml_prob                        │
│ sli_version                    │
│ calculated_at                  │
└───────────────────────────────┘


┌───────────────────────────────┐
│    search_trends [신규]        │
│───────────────────────────────│
│ product_code PK1 FK            │
│ period PK2 (YYYY-MM)           │
│ source PK3 (datalab/blog/shop) │
│ trend_value                    │
│ search_volume                  │
│ blog_count                     │
│ shop_count                     │
│ collected_at                   │
└───────────────────────────────┘


┌───────────────────────────────┐
│    pipeline_log [신규]         │
│───────────────────────────────│
│ run_id PK (AUTO)               │
│ run_date                       │
│ step_name                      │
│ status (success/fail/skip)     │
│ rows_affected                  │
│ duration_sec                   │
│ error_message                  │
│ meta (JSON)                    │
└───────────────────────────────┘
```

### 3.3 신규 테이블 상세 스키마

#### 3.3.1 review_absa — ABSA 리뷰 레벨 감성

리뷰 1건당 1행. reviews_core와 1:1 관계.

| 컬럼                   | 타입     | 설명                                      |
| ---------------------- | -------- | ----------------------------------------- |
| `review_id` (PK, FK) | INT64    | 리뷰 고유 식별자 (reviews_core 참조)      |
| `sentiment`          | STRING   | 전체 감성 (positive / neutral / negative) |
| `sentiment_score`    | FLOAT64  | 감성 점수 (-1.0 ~ 1.0, 신경망 출력)       |
| `is_ambiguous`       | BOOL     | 모호한 리뷰 여부                          |
| `aspect_count`       | INT64    | 언급된 aspect 수 (0~8, 0이면 미분류)      |
| `absa_version`       | STRING   | 모델 버전 ("stage3a_v2", "stage4" 등)     |
| `inferred_at`        | DATETIME | 추론 실행 시각                            |

**BigQuery DDL:**

```sql
CREATE TABLE IF NOT EXISTS `daiso.review_absa` (
  review_id       INT64 NOT NULL,
  sentiment       STRING,
  sentiment_score FLOAT64,
  is_ambiguous    BOOL,
  aspect_count    INT64,
  absa_version    STRING,
  inferred_at     DATETIME
);
```

#### 3.3.2 review_aspects — ABSA Aspect 레벨 감성

리뷰 1건당 0~8개 행. review_absa와 1:N 관계.

| 컬럼                    | 타입    | 설명                                                                                                 |
| ----------------------- | ------- | ---------------------------------------------------------------------------------------------------- |
| `review_id` (PK1, FK) | INT64   | 리뷰 고유 식별자 (review_absa 참조)                                                                  |
| `aspect` (PK2)        | STRING  | Aspect명 (8종: 배송/포장, 가격/가성비, 사용감/성능, 용량/휴대, 디자인, 재질/냄새, 재구매, 색상/발색) |
| `aspect_sentiment`    | STRING  | Aspect 감성 (positive / neutral / negative)                                                          |
| `aspect_confidence`   | FLOAT64 | 신뢰도 (0.0 ~ 1.0)                                                                                   |

**BigQuery DDL:**

```sql
CREATE TABLE IF NOT EXISTS `daiso.review_aspects` (
  review_id         INT64 NOT NULL,
  aspect            STRING NOT NULL,
  aspect_sentiment  STRING,
  aspect_confidence FLOAT64
);
```

**활용 예시 쿼리:**

```sql
-- Aspect별 감성 비율 (전체)
SELECT
  aspect,
  COUNTIF(aspect_sentiment = 'positive') / COUNT(*) AS pos_rate,
  COUNTIF(aspect_sentiment = 'negative') / COUNT(*) AS neg_rate,
  COUNT(*) AS mention_count
FROM `daiso.review_aspects`
GROUP BY aspect
ORDER BY mention_count DESC;

-- 특정 제품의 Aspect별 감성
SELECT
  ra.aspect,
  ra.aspect_sentiment,
  COUNT(*) AS cnt
FROM `daiso.review_aspects` ra
JOIN `daiso.reviews_core` rc ON ra.review_id = rc.review_id
WHERE rc.product_code = 1056665
GROUP BY ra.aspect, ra.aspect_sentiment;

-- 월별 부정 리뷰 추이 (사용감/성능)
SELECT
  FORMAT_DATE('%Y-%m', rc.review_date) AS month,
  COUNTIF(ra.aspect_sentiment = 'negative') AS neg_count,
  COUNT(*) AS total
FROM `daiso.review_aspects` ra
JOIN `daiso.reviews_core` rc ON ra.review_id = rc.review_id
WHERE ra.aspect = '사용감/성능'
GROUP BY month
ORDER BY month;
```

#### 3.3.3 sli_results — SLI 연착륙 판별 결과

제품 1건당 1행. products_core와 1:1 관계.

| 컬럼                      | 타입     | 설명                                    |
| ------------------------- | -------- | --------------------------------------- |
| `product_code` (PK, FK) | INT64    | 상품 고유 코드 (products_core 참조)     |
| `is_soft_landing_dtw`   | BOOL     | DTW 클러스터링 기반 연착륙 여부         |
| `is_soft_landing_surv`  | BOOL     | 생존분석(Kaplan-Meier) 기반 연착륙 여부 |
| `is_soft_landing_rule`  | BOOL     | SLI_v2 규칙 기반 연착륙 여부            |
| `is_soft_landing_ml`    | BOOL     | ML(LightGBM) 기반 연착륙 여부           |
| `total_votes`           | INT64    | 만장일치 투표 수 (0~4)                  |
| `final_soft_landing`    | BOOL     | 최종 연착륙 판별 (total_votes >= 3)     |
| `confidence`            | FLOAT64  | 판별 신뢰도                             |
| `ml_prob`               | FLOAT64  | LightGBM 모델 확률                      |
| `sli_version`           | STRING   | SLI 모델 버전 ("v1", "v2" 등)           |
| `calculated_at`         | DATETIME | 계산 시각                               |

**BigQuery DDL:**

```sql
CREATE TABLE IF NOT EXISTS `daiso.sli_results` (
  product_code          INT64 NOT NULL,
  is_soft_landing_dtw   BOOL,
  is_soft_landing_surv  BOOL,
  is_soft_landing_rule  BOOL,
  is_soft_landing_ml    BOOL,
  total_votes           INT64,
  final_soft_landing    BOOL,
  confidence            FLOAT64,
  ml_prob               FLOAT64,
  sli_version           STRING,
  calculated_at         DATETIME
);
```

#### 3.3.4 search_trends — 네이버 검색트렌드

제품 × 월 × 소스별 1행. 복합 PK.

| 컬럼                       | 타입     | 설명                                |
| -------------------------- | -------- | ----------------------------------- |
| `product_code` (PK1, FK) | INT64    | 상품 고유 코드                      |
| `period` (PK2)           | STRING   | 기간 (YYYY-MM 형식)                 |
| `source` (PK3)           | STRING   | 데이터 소스 (datalab / blog / shop) |
| `trend_value`            | FLOAT64  | DataLab 상대 검색량 (0~100)         |
| `search_volume`          | INT64    | 검색 API 결과 건수                  |
| `blog_count`             | INT64    | 블로그 검색 결과 수                 |
| `shop_count`             | INT64    | 쇼핑 검색 결과 수                   |
| `collected_at`           | DATETIME | 수집 시각                           |

**BigQuery DDL:**

```sql
CREATE TABLE IF NOT EXISTS `daiso.search_trends` (
  product_code  INT64 NOT NULL,
  period        STRING NOT NULL,
  source        STRING NOT NULL,
  trend_value   FLOAT64,
  search_volume INT64,
  blog_count    INT64,
  shop_count    INT64,
  collected_at  DATETIME
);
```

#### 3.3.5 pipeline_log — 파이프라인 실행 이력

실행 단계별 1행. 자동 증가 PK.

| 컬럼              | 타입     | 설명                                                                  |
| ----------------- | -------- | --------------------------------------------------------------------- |
| `run_id` (PK)   | INT64    | 실행 고유 ID (자동 증가)                                              |
| `run_date`      | DATETIME | 실행 시작 시각                                                        |
| `step_name`     | STRING   | 단계명 (crawling / transform / absa / sli / search_trend / dashboard) |
| `status`        | STRING   | 상태 (success / fail / skip)                                          |
| `rows_affected` | INT64    | 영향받은 행 수                                                        |
| `duration_sec`  | FLOAT64  | 소요 시간(초)                                                         |
| `error_message` | STRING   | 에러 메시지 (NULL = 정상)                                             |
| `meta`          | JSON     | 추가 메타데이터 (신규 제품 수, 신규 리뷰 수 등)                       |

**BigQuery DDL:**

```sql
CREATE TABLE IF NOT EXISTS `daiso.pipeline_log` (
  run_id         INT64 NOT NULL,
  run_date       DATETIME,
  step_name      STRING,
  status         STRING,
  rows_affected  INT64,
  duration_sec   FLOAT64,
  error_message  STRING,
  meta           JSON
);
```

---

## 4. 구현 완료 현황

> v1에서는 "코드 수정 계획 (의사코드)"이었으나, v2 기준으로 전량 구현이 완료되었습니다.

### 4.1 transformer.py — 증분 전용 전환

**변경 완료:** 전체 raw CSV 입력 → 신규 raw만 입력, BQ UPSERT가 병합 담당

| 수정 대상                 | AS-IS                                      | TO-BE (구현 완료)                                 |
| ------------------------- | ------------------------------------------ | ------------------------------------------------- |
| `load_existing_final()` | 로컬 final/ CSV 로드                       | BQ에서 기존 ID 매핑 조회 (`query_to_df`)        |
| `_transform_brand()`    | 로컬 brand.csv에서 max_id                  | `SELECT MAX(brand_id) FROM daiso.brand`         |
| `_transform_reviews()`  | 로컬 reviews_core.csv에서 max_rid          | `SELECT MAX(review_id) FROM daiso.reviews_core` |
| `user_id_map`           | 로컬 user_id_map.csv 읽기/쓰기             | BQ user_id_map 테이블 조회/UPSERT                 |
| 결과 저장                 | `LocalStorage.save_all()` → CSV/Parquet | `CrawlerETLv2.upload_all()` → BQ UPSERT만      |

### 4.2 run_absa_incremental.py — ABSA 증분 추론

BQ에서 미추론 리뷰 조회 → KcELECTRA(Stage 3A) 추론 → BQ UPSERT

- 미추론 리뷰: `LEFT JOIN review_absa WHERE ra.review_id IS NULL`
- 모델: `prod_bundle_stage3a_v1_20260225`
- 버전: `stage3a_v2`

### 4.3 run_sli.py (598줄) — SLI 스크립트화

노트북(`03_notebooks/06_SLI/6. SLI_통합분석_DTW_생존분석_ML.ipynb`)의 핵심 로직을 완전히 스크립트로 전환.

- 4가지 방법론: DTW 클러스터링 + 생존분석(Kaplan-Meier) + SLI_v2 규칙기반 + ML(LightGBM)
- 만장일치 투표: `min_votes=3` (config 설정 가능)
- 전체 재계산 방식 (증분이 아닌 매월 전체 리뷰 기반)

### 4.4 run_monthly_pipeline.py (365줄) — 오케스트레이션

6단계 순차 실행 + pipeline_log 자동 기록.

- 핵심 단계(crawling, transform) 실패 시 전체 중단
- 분석 단계(absa, sli, search_trend, dashboard) 실패 시 다음으로 계속
- CLI 옵션: `--skip-crawl`, `--steps 3,4`, `--dry-run`, `--dataset`

### 4.5 scheduler.py (121줄) — cron 스케줄러

- config.yaml의 cron 표현식을 읽어 crontab 등록/해제
- **v2 수정:** 대상 스크립트를 `run_monthly_pipeline.py`로 변경 (v1: `run_pipeline.py`)
- **v2 수정:** `--local-only` 플래그 제거 (run_monthly_pipeline에 없는 옵션)

### 4.6 config.yaml — 현재 설정 (v2)

```yaml
pipeline:
  auto_schedule:
    enabled: true           # ← v2: 활성화 (v1: false)
    cron: "0 3 1 * *"      # 매월 1일 03시

  crawling:
    mode: "incremental"
    headless: true
    crawl_reviews: true
    crawl_ingredients: true
    history_file: "05_src/01_crawling/cache/crawl_history.json"
    active_categories:
      스킨케어: [all]
      메이크업: [all]
      네일용품: [all]
      맨케어: [all]
      미용소품: [all]
      헤어/바디: [all]

  storage:
    local:
      csv: false             # 로컬 CSV 비활성화 (BQ 전환)
      parquet: false
      base_dir: "02_processed_data"
    bigquery:
      enabled: true
      dataset: "daiso"

  absa:
    enabled: true
    bundle_path: "06_analysis/03_ABSA/07_models/prod_bundle_stage3a_v1_20260225"
    version: "stage3a_v2"

  sli:
    enabled: true
    version: "v1"
    min_votes: 3

  search_trend:
    enabled: false            # ← v2: 비활성화 (수동 실행 전용)
    period_start: "2024-01-01"
    period_end: "auto"

  dashboard:
    enabled: true
    output_path: "02_outputs/dashboard/"
```

### 4.7 검색트렌드 BQ 연결 (부분 구현)

`step_search_trend()`는 BQ에서 연착륙 제품 목록 조회까지 구현. 실제 네이버 API 호출은 기존 스크립트 활용.

**수동 실행 방법:**

```bash
# 방법 1: 파이프라인에서 Step 5만 실행
python 05_src/04_pipeline/run_monthly_pipeline.py --steps 5

# 방법 2: 기존 스크립트 직접 실행
python 06_analysis/04_search_trend/06_scripts/run_soft_landing_search_trend.py
```

---

## 5. 월간 파이프라인 오케스트레이션

### 5.1 실행 방법

```bash
# 전체 실행 (Step 1~6)
python 05_src/04_pipeline/run_monthly_pipeline.py

# 크롤링 건너뛰기 (Step 3~6만)
python 05_src/04_pipeline/run_monthly_pipeline.py --skip-crawl

# 특정 단계만 실행
python 05_src/04_pipeline/run_monthly_pipeline.py --steps 3,4

# 실행 안 하고 계획만 표시
python 05_src/04_pipeline/run_monthly_pipeline.py --dry-run

# cron 스케줄러 관리
python 05_src/04_pipeline/scheduler.py --status   # 상태 확인
python 05_src/04_pipeline/scheduler.py --enable    # cron 등록
python 05_src/04_pipeline/scheduler.py --disable   # cron 해제
```

### 5.2 실행 순서 및 의존성

```
Step 1: 크롤링 (crawling) [핵심]
  └─ 선행 조건: 없음
  └─ 출력: 3개 raw DataFrame (메모리)
  └─ 실패 시: 전체 중단

Step 2: 변환 + BQ 적재 (transform) [핵심]
  └─ 선행 조건: Step 1 완료
  └─ 출력: BQ 13개 테이블 UPSERT
  └─ 실패 시: 전체 중단

Step 3: ABSA 증분 추론 (absa) [분석]
  └─ 선행 조건: Step 2 완료 (신규 리뷰가 BQ에 있어야)
  └─ 출력: BQ review_absa + review_aspects UPSERT
  └─ 실패 시: Step 4로 진행 (이전 ABSA 결과 사용)

Step 4: SLI 연착륙 재계산 (sli) [분석]
  └─ 선행 조건: Step 2 완료 (최신 리뷰 데이터)
  └─ 출력: BQ sli_results UPSERT
  └─ 실패 시: Step 5로 진행 (이전 SLI 결과 사용)

Step 5: 검색트렌드 수집 (search_trend) [비활성 — config에서 skip]
  └─ config.yaml: search_trend.enabled: false
  └─ 자동 실행 시 즉시 skip (rows_affected=0, status="disabled")
  └─ 수동 실행: python run_monthly_pipeline.py --steps 5

Step 6: 대시보드 생성 (dashboard) [분석]
  └─ 선행 조건: Step 2 이상 완료
  └─ 출력: HTML 대시보드 파일
  └─ 실패 시: 로그만 기록 (Phase 4 미착수)
```

### 5.3 예상 소요 시간

| 단계            | 풀 크롤링         | 증분 (월간)                |
| --------------- | ----------------- | -------------------------- |
| 크롤링          | 8~12시간          | 1~3시간                    |
| 변환 + BQ 적재  | 5~10분            | 2~5분                      |
| ABSA 추론 (GPU) | 63분 (32만건)     | 5~15분 (1~3만건)          |
| SLI 계산        | 10~20분           | 10~20분 (전체 재계산)      |
| 검색트렌드 API  | ~~30~60분~~      | **SKIP** (수동 전용) |
| 대시보드 생성   | 1~2분             | 1~2분                      |
| **합계**  | **~14시간** | **~3.5시간**         |

### 5.4 cron 스케줄 (v2 활성화)

```
# 현재 등록된 crontab
0 3 1 * * /opt/homebrew/opt/python@3.14/bin/python3.14 \
  /Users/yu_seok/.../05_src/04_pipeline/run_monthly_pipeline.py \
  # whypi-pipeline-auto
```

- **실행 주기:** 매월 1일 03:00
- **대상:** `run_monthly_pipeline.py` (6단계 오케스트레이션)
- **실제 실행:** Step 1~4, 6 (Step 5는 config에서 자동 skip)

---

## 6. BQ 테이블 PK 매핑 (bq_client.py)

기존 `TABLE_KEYS`에 신규 5개 테이블 추가 완료:

```python
TABLE_KEYS = {
    # 기존 13개
    "brand": ["brand_id"],
    "manufacturer": ["manufacturer_id"],
    "ingredients_dic": ["ingredient_id"],
    "products_core": ["product_code"],
    "products_stats": ["product_code"],
    "products_category": ["product_code"],
    "products_ingredients": ["product_code", "ingredient_id"],
    "functional": ["product_code"],
    "promotions": ["promotion_id"],
    "reviews_core": ["review_id"],
    "reviews_text": ["review_id"],
    "users_profile": ["user_id"],
    "users_repurchase": ["user_id"],

    # 신규 5개
    "review_absa": ["review_id"],
    "review_aspects": ["review_id", "aspect"],
    "sli_results": ["product_code"],
    "search_trends": ["product_code", "period", "source"],
    "pipeline_log": ["run_id"],
}
```

---

## 7. 마이그레이션 진행 현황

### 7.1 Phase 1 — BQ 스키마 확장 완료

1. `schema_v3.sql` 생성 (신규 5개 테이블 DDL 추가)
2. `bq_client.py`의 `TABLE_KEYS`에 신규 5개 테이블 추가 (ERD v3 — 18개)
3. `migrate_v3.py` 마이그레이션 스크립트 작성 (ABSA CSV → review_absa + review_aspects, SLI CSV → sli_results 초기 적재)
4. ⬜ BQ에서 DDL 실행하여 테이블 생성 (migrate_v3.py 실행 필요)
5. `config.yaml` 업데이트 (absa/sli/search_trend/dashboard 섹션 추가, bigquery.enabled=true)

### 7.2 Phase 2 — transformer BQ 전환 완료

1. `load_existing_from_bq()` 함수 구현 (transformer.py)
2. `transformer.py` user_id_map BQ 기반 전환 (existing_data["user_id_map"] 우선 사용)
3. `run_pipeline.py` BQ 직접 적재 모드 수정 (use_bq 플래그, 로컬/BQ 분기)
4. ⬜ 증분 크롤링 → BQ UPSERT 테스트
5. ⬜ `crawl_history.py` BQ 기반 전환 (현재 로컬 JSON 유지)

### 7.3 Phase 3 — 분석 파이프라인 통합 완료

1. `run_absa_incremental.py` 구현 (미추론 리뷰 조회 → KcELECTRA 추론 → BQ UPSERT)
2. `run_sli.py` SLI 스크립트화 (DTW + 생존분석 + 규칙기반 + **LightGBM** → 만장일치 투표)
3. `run_monthly_pipeline.py` 오케스트레이션 구현 (6단계 순차 실행 + pipeline_log 기록)
4. 검색트렌드: **의도적 비활성화** (`search_trend.enabled: false`) — API 호출 제한·비용 고려하여 수동 실행으로 분리

### 7.4 Phase 4 — 대시보드 개발 (미착수)

1. ⬜ BQ 집계 쿼리 작성
2. ⬜ 인터랙티브 HTML 대시보드 구현
3. ⬜ 자동 생성 테스트

### 7.5 Phase 5 — 검증 및 안정화 (진행 중)

1. ⬜ 전체 파이프라인 end-to-end 테스트
2. ⬜ 로컬 CSV 백업 후 삭제
3. **월간 스케줄링 설정** — cron 등록 완료 (`0 3 1 * *`, `scheduler.py --enable`)
4. **scheduler.py 수정** — 대상 스크립트 `run_monthly_pipeline.py`로 변경, `--local-only` 플래그 제거

---

## 8. 데이터 한계 및 주의사항

| 항목                                      | 현황                      | 대응                                             |
| ----------------------------------------- | ------------------------- | ------------------------------------------------ |
| reviews_core vs reviews_text 행 수 불일치 | 323K vs 436K              | BQ 전환 시 review_id FK 기준 정합성 검증 후 적재 |
| ABSA 미분류율 28.7%                       | 8개 aspect 모두 none      | aspect_count = 0으로 식별 가능, 대시보드에 표시  |
| user_masked 충돌                          | 마스킹된 닉네임 중복 가능 | user_id_map으로 영속 매핑 유지                   |
| OCR 오인식                                | 500+ 교정 패턴 적용 중    | 신규 성분 발견 시 교정 규칙 업데이트 필요        |
| 네이버 API 호출 제한                      | 일 25,000건               | 수동 실행으로 전환 (자동화 제외)                 |
| SLI 계산 시간                             | DTW 클러스터링이 O(n²)   | 제품 수 ~950이므로 현재 규모에서는 문제 없음     |
| cron 실행 환경                            | macOS 로컬 환경 의존      | 개인 PC 전원/네트워크 상태에 따라 실행 실패 가능 |

---

## 9. 최종 파일 구조 (v2)

```
05_src/04_pipeline/
├── run_monthly_pipeline.py    [365줄] 월간 오케스트레이션 (6단계 + pipeline_log)
├── run_pipeline.py            [수정]  BQ 직접 적재 (use_bq 플래그)
├── run_absa_incremental.py    [신규]  ABSA 증분 추론 (미추론 리뷰 → KcELECTRA → BQ)
├── run_sli.py                 [598줄] SLI 스크립트화 (DTW+생존+규칙+LightGBM → 투표)
├── scheduler.py               [121줄] cron 등록/해제/상태확인 → run_monthly_pipeline.py
├── transformer.py             [수정]  load_existing_from_bq() + user_id_map BQ 지원
├── storage.py                 유지
├── derived_features.py        유지
├── assign_promotion.py        유지
└── config.yaml                [수정]  auto_schedule=true, search_trend=false

05_src/02_bigquery/
├── bq_client.py               [수정]  TABLE_KEYS 5개 추가 (ERD v3 — 18개)
├── etl_loader.py              유지
├── schema_v2.sql              유지 (기존)
├── schema_v3.sql              [신규]  18개 테이블 DDL (기존 13 + 신규 5)
├── migrate.py                 유지 (v1→v2 마이그레이션)
└── migrate_v3.py              [신규]  v2→v3 마이그레이션 (ABSA+SLI 초기 적재)
```

---

## 10. 남은 작업 (TODO)

| 우선순위 | 항목                  | Phase | 설명                                    |
| -------- | --------------------- | ----- | --------------------------------------- |
| 높음     | migrate_v3.py 실행    | 1-4   | BQ DDL 실행하여 신규 5개 테이블 생성    |
| 높음     | E2E 테스트            | 5-1   | 전체 파이프라인 end-to-end 테스트       |
| 중간     | 증분 크롤링 테스트    | 2-4   | 증분 크롤링 → BQ UPSERT 실제 동작 검증 |
| 중간     | crawl_history BQ 전환 | 2-5   | 로컬 JSON → BQ 조회 방식 전환          |
| 낮음     | 대시보드 개발         | 4     | BQ 집계 쿼리 + 인터랙티브 HTML          |
| 낮음     | 로컬 CSV 정리         | 5-2   | BQ 안정화 확인 후 로컬 CSV 백업 및 삭제 |
