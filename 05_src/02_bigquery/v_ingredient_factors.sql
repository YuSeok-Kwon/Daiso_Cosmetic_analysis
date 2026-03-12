-- =============================================
-- VIEW: 카테고리별 성분 +/- 요인 (골든 + 안티 통합)
-- =============================================
-- 기존 v_golden_ingredients를 확장하여
-- diff_pct > 0 (골든/+요인) + diff_pct < 0 (안티/-요인) 모두 포함
-- 용도: 축5 고도화 — 카테고리별 차등 성분 점수
-- 생성일: 2026-03-07
-- =============================================

CREATE OR REPLACE VIEW `daiso-analysis.daiso.v_ingredient_factors` AS
WITH sl_products AS (
  SELECT
    s.product_code,
    s.final_soft_landing,
    pc.category_2
  FROM `daiso-analysis.daiso.sli_results` s
  JOIN `daiso-analysis.daiso.products_category` pc USING (product_code)
),

category_counts AS (
  SELECT
    category_2,
    COUNTIF(final_soft_landing = TRUE)  AS n_sl,
    COUNTIF(final_soft_landing = FALSE) AS n_nsl
  FROM sl_products
  GROUP BY category_2
),

-- 성분별 SL/Non-SL 출현 제품 수
ingredient_freq AS (
  SELECT
    sp.category_2,
    pi.ingredient_id,
    COUNTIF(sp.final_soft_landing = TRUE)  AS sl_product_count,
    COUNTIF(sp.final_soft_landing = FALSE) AS nsl_product_count
  FROM `daiso-analysis.daiso.products_ingredients` pi
  JOIN sl_products sp USING (product_code)
  GROUP BY sp.category_2, pi.ingredient_id
),

-- 출현률 + diff 계산
ingredient_rates AS (
  SELECT
    f.category_2,
    f.ingredient_id,
    ig.ingredient_name,
    ig.ingredient_type,
    ig.effect,
    ig.is_allergic,
    f.sl_product_count,
    f.nsl_product_count,
    ROUND(f.sl_product_count  / cc.n_sl  * 100, 1)  AS sl_pct,
    ROUND(f.nsl_product_count / cc.n_nsl * 100, 1)  AS nsl_pct,
    ROUND(f.sl_product_count  / cc.n_sl  * 100, 1)
      - ROUND(f.nsl_product_count / cc.n_nsl * 100, 1) AS diff_pct,
    cc.n_sl,
    cc.n_nsl
  FROM ingredient_freq f
  JOIN category_counts cc USING (category_2)
  JOIN `daiso-analysis.daiso.ingredients_dic` ig USING (ingredient_id)
  WHERE (f.sl_product_count + f.nsl_product_count) >= 3  -- 최소 3개 제품에 등장
),

-- +요인 (SL 우위) 순위
positive_ranked AS (
  SELECT *,
    ROW_NUMBER() OVER (
      PARTITION BY category_2
      ORDER BY diff_pct DESC
    ) AS pos_rank
  FROM ingredient_rates
  WHERE diff_pct > 5   -- 5%p 이상 차이만
),

-- -요인 (Non-SL 우위) 순위
negative_ranked AS (
  SELECT *,
    ROW_NUMBER() OVER (
      PARTITION BY category_2
      ORDER BY diff_pct ASC
    ) AS neg_rank
  FROM ingredient_rates
  WHERE diff_pct < -5  -- -5%p 이상 차이만
)

-- 통합: +요인 Top 10 + -요인 Top 10
SELECT
  category_2,
  ingredient_id,
  ingredient_name,
  ingredient_type,
  effect,
  is_allergic,
  sl_product_count,
  nsl_product_count,
  sl_pct,
  nsl_pct,
  diff_pct,
  n_sl,
  n_nsl,
  'positive' AS factor_type,
  pos_rank AS factor_rank,
  CASE
    WHEN pos_rank <= 5  THEN 'Gold'
    WHEN pos_rank <= 10 THEN 'Silver'
    ELSE 'Bronze'
  END AS factor_tier
FROM positive_ranked
WHERE pos_rank <= 10

UNION ALL

SELECT
  category_2,
  ingredient_id,
  ingredient_name,
  ingredient_type,
  effect,
  is_allergic,
  sl_product_count,
  nsl_product_count,
  sl_pct,
  nsl_pct,
  diff_pct,
  n_sl,
  n_nsl,
  'negative' AS factor_type,
  neg_rank AS factor_rank,
  CASE
    WHEN neg_rank <= 5  THEN 'Red'
    WHEN neg_rank <= 10 THEN 'Orange'
    ELSE 'Yellow'
  END AS factor_tier
FROM negative_ranked
WHERE neg_rank <= 10

ORDER BY category_2, factor_type DESC, factor_rank;


-- =============================================
-- VIEW: 카테고리별 최적 성분 수 기준 (축4 고도화)
-- =============================================
-- 기존: 전 카테고리 동일 SL 중앙값 대비 비율
-- 고도화: 카테고리별 SL 제품의 성분 수 분포(Q1/중앙/Q3) 기반
--         최적 구간 안에 있으면 만점, 벗어날수록 감점
-- =============================================

CREATE OR REPLACE VIEW `daiso-analysis.daiso.v_category_ingredient_optimal` AS
WITH product_ingr_count AS (
  SELECT
    product_code,
    COUNT(*) AS ingr_cnt
  FROM `daiso-analysis.daiso.products_ingredients`
  GROUP BY product_code
),

sl_ingr AS (
  SELECT
    pc.category_2,
    s.final_soft_landing,
    ic.ingr_cnt
  FROM `daiso-analysis.daiso.sli_results` s
  JOIN `daiso-analysis.daiso.products_category` pc USING (product_code)
  JOIN product_ingr_count ic USING (product_code)
)

SELECT
  category_2,

  -- SL 제품 성분 수 분포
  COUNTIF(final_soft_landing = TRUE) AS sl_count,
  APPROX_QUANTILES(IF(final_soft_landing = TRUE, ingr_cnt, NULL), 100)[OFFSET(10)]  AS sl_p10,
  APPROX_QUANTILES(IF(final_soft_landing = TRUE, ingr_cnt, NULL), 100)[OFFSET(25)]  AS sl_q1,
  APPROX_QUANTILES(IF(final_soft_landing = TRUE, ingr_cnt, NULL), 100)[OFFSET(50)]  AS sl_median,
  APPROX_QUANTILES(IF(final_soft_landing = TRUE, ingr_cnt, NULL), 100)[OFFSET(75)]  AS sl_q3,
  APPROX_QUANTILES(IF(final_soft_landing = TRUE, ingr_cnt, NULL), 100)[OFFSET(90)]  AS sl_p90,
  ROUND(AVG(IF(final_soft_landing = TRUE, ingr_cnt, NULL)), 1) AS sl_mean,

  -- Non-SL 제품 성분 수 분포
  COUNTIF(final_soft_landing = FALSE) AS nsl_count,
  APPROX_QUANTILES(IF(final_soft_landing = FALSE, ingr_cnt, NULL), 100)[OFFSET(50)] AS nsl_median,
  ROUND(AVG(IF(final_soft_landing = FALSE, ingr_cnt, NULL)), 1) AS nsl_mean,

  -- 전체 분포
  APPROX_QUANTILES(ingr_cnt, 100)[OFFSET(50)] AS all_median

FROM sl_ingr
GROUP BY category_2
ORDER BY sl_count DESC;
