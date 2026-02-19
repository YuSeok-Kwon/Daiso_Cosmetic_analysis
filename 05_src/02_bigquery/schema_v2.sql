-- ERD v2 BigQuery DDL
-- 13개 테이블 (FK 의존성 순서)

-- 1. brand
CREATE TABLE IF NOT EXISTS `daiso.brand` (
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

-- 10. users_profile
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
  user_category_repurchase INT64,
  user_brand_repurchase INT64
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
