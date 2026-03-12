-- ERD v3 BigQuery DDL
-- 18개 테이블 (기존 13개 + 신규 5개)
-- v2 → v3 변경: review_absa, review_aspects, sli_results, search_trends, pipeline_log 추가

-- =============================================
-- 기존 13개 테이블 (FK 의존성 순서)
-- =============================================

-- 1. brands
CREATE TABLE IF NOT EXISTS `daiso.brands` (
  brand_id INT64 NOT NULL,
  name STRING
);

-- 2. manufacturer
CREATE TABLE IF NOT EXISTS `daiso.manufacturer` (
  manufacturer_id INT64 NOT NULL,
  ENTP_NAME STRING
);

-- 3. ingredients_dic
CREATE TABLE IF NOT EXISTS `daiso.ingredients_dic` (
  ingredient_id INT64 NOT NULL,
  ingredient_name STRING,
  application_role STRING,
  ingredient_type STRING
);

-- 4. promotions
CREATE TABLE IF NOT EXISTS `daiso.promotions` (
  promotion_id INT64 NOT NULL,
  description STRING,
  brand_id INT64,
  event_type STRING,
  start_date DATE,
  end_date DATE
);

-- 5. products_core
CREATE TABLE IF NOT EXISTS `daiso.products_core` (
  product_code INT64 NOT NULL,
  manufacturer_id INT64,
  brand_id INT64,
  name STRING,
  price INT64,
  country STRING
);

-- 6. products_category
CREATE TABLE IF NOT EXISTS `daiso.products_category` (
  product_code INT64 NOT NULL,
  category_1 STRING,
  category_2 STRING
);

-- 7. products_stats
CREATE TABLE IF NOT EXISTS `daiso.products_stats` (
  product_code INT64 NOT NULL,
  likes INT64,
  shares INT64,
  review_count INT64,
  first_review_date DATE,
  engagement_score FLOAT64,
  cp_index FLOAT64,
  review_density FLOAT64,
  risk_score FLOAT64
);

-- 8. products_ingredients
CREATE TABLE IF NOT EXISTS `daiso.products_ingredients` (
  product_code INT64 NOT NULL,
  ingredient_id INT64 NOT NULL
);

-- 9. functional
CREATE TABLE IF NOT EXISTS `daiso.functional` (
  product_code INT64 NOT NULL,
  ITEM_PH STRING,
  ph_category STRING,
  is_whitening BOOL,
  is_wrinkle_reduction BOOL,
  is_sunscreen BOOL,
  is_acne BOOL
);

-- 10. user_id_map
CREATE TABLE IF NOT EXISTS `daiso.user_id_map` (
  user_masked STRING NOT NULL,
  user_id INT64 NOT NULL
);

-- 11. users_profile
CREATE TABLE IF NOT EXISTS `daiso.users_profile` (
  user_id INT64 NOT NULL,
  user_total_reviews INT64,
  user_activity_level STRING,
  user_avg_rating_reorder FLOAT64,
  user_rating_tendency STRING,
  review_tenure INT64
);

-- 11. users_repurchase
CREATE TABLE IF NOT EXISTS `daiso.users_repurchase` (
  user_id INT64 NOT NULL,
  reorder_user_category INT64,
  reorder_user_brand INT64,
  reorder_user_avg_rating FLOAT64
);

-- 12. reviews_core
CREATE TABLE IF NOT EXISTS `daiso.reviews_core` (
  review_id INT64 NOT NULL,
  product_code INT64,
  user_id INT64,
  rating INT64,
  review_date DATE,
  image_count INT64,
  is_reorder BOOL,
  promotion_id INT64
);

-- 13. reviews_text
CREATE TABLE IF NOT EXISTS `daiso.reviews_text` (
  review_id INT64 NOT NULL,
  text STRING,
  review_length INT64
);

-- =============================================
-- 신규 5개 테이블 (v3 추가)
-- =============================================

-- 14. review_absa — ABSA 리뷰 레벨 감성
-- reviews_core와 1:1 관계
CREATE TABLE IF NOT EXISTS `daiso.review_absa` (
  review_id       INT64 NOT NULL,
  sentiment       STRING,
  sentiment_score FLOAT64,
  is_ambiguous    BOOL,
  aspect_count    INT64,
  absa_version    STRING,
  inferred_at     DATETIME
);

-- 15. review_aspects — ABSA Aspect 레벨 감성
-- review_absa와 1:N 관계 (리뷰당 0~8개 aspect)
CREATE TABLE IF NOT EXISTS `daiso.review_aspects` (
  review_id         INT64 NOT NULL,
  aspect            STRING NOT NULL,
  aspect_sentiment  STRING,
  aspect_confidence FLOAT64
);

-- 16. sli_results — SLI 연착륙 판별 결과
-- products_core와 1:1 관계
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

-- 17. search_trends — 네이버 검색트렌드
-- 복합 PK: product_code + period + source
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

-- 18. pipeline_log — 파이프라인 실행 이력
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
