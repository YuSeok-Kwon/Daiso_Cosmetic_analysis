-- =============================================
-- PATCH: Division by Zero 방어 (오류코드 015CFBE6 해결)
-- =============================================
-- 적용 방법: BigQuery 콘솔에서 이 SQL을 순서대로 실행
-- 영향 뷰: v_score_distribution → v_sim_category_benchmark (의존)
--          v_ingredient_factors → v_golden_ingredients (의존)
-- 생성일: 2026-03-10
-- =============================================


-- =============================================
-- [1/2] v_score_distribution 패치
-- 변경: 축3 가격점수에 sl_price_median = 0 방어 추가
-- =============================================
CREATE OR REPLACE VIEW `daiso.v_score_distribution` AS
WITH
-- 각 제품별 골든/안티 성분 매칭 수 집계
product_factor_counts AS (
  SELECT
    pi.product_code,
    pc.category_2,
    COUNTIF(vf.factor_type = 'positive' AND vf.factor_tier = 'Gold')   AS gold_count,
    COUNTIF(vf.factor_type = 'positive' AND vf.factor_tier = 'Silver') AS silver_count,
    COUNTIF(vf.factor_type = 'negative' AND vf.factor_tier = 'Red')    AS red_count,
    COUNTIF(vf.factor_type = 'negative' AND vf.factor_tier = 'Orange') AS orange_count
  FROM `daiso.products_ingredients` pi
  JOIN `daiso.products_category` pc USING (product_code)
  LEFT JOIN `daiso.v_ingredient_factors` vf
    ON pc.category_2 = vf.category_2
    AND pi.ingredient_id = vf.ingredient_id
  GROUP BY pi.product_code, pc.category_2
),

product_scores AS (
  SELECT
    s.product_code,
    pc.category_2,
    p.price,

    -- [v2] 축1: 비선형 카테고리 점수 (5점 만점)
    CASE
      WHEN cat_rate.sl_rate >= 35 THEN 5
      WHEN cat_rate.sl_rate >= 20 THEN 3
      WHEN cat_rate.sl_rate >= 5  THEN 2
      ELSE 1
    END AS score_category,

    -- 축3: 비선형 가격 점수 (15점 만점)
    -- ★ PATCH: sl_price_median = 0 방어 추가
    CASE
      WHEN sl_p.sl_price_median IS NULL OR sl_p.sl_price_median = 0 THEN 5
      WHEN ABS(p.price - sl_p.sl_price_median) / sl_p.sl_price_median <= 0.0 THEN 15
      WHEN ABS(p.price - sl_p.sl_price_median) / sl_p.sl_price_median <= 0.2 THEN 12
      WHEN ABS(p.price - sl_p.sl_price_median) / sl_p.sl_price_median <= 0.4 THEN 8
      ELSE 4
    END AS score_price,

    -- [v3] 축4: 성분 수 점수 (15점 만점, 엄격 버전)
    CASE
      WHEN sl_i.sl_ingr_median IS NULL OR sl_i.sl_ingr_median = 0 THEN 7
      WHEN ic.ingr_cnt / sl_i.sl_ingr_median <= 0.6 THEN 15
      WHEN ic.ingr_cnt / sl_i.sl_ingr_median <= 0.7 THEN 14
      WHEN ic.ingr_cnt / sl_i.sl_ingr_median <= 0.8 THEN 13
      WHEN ic.ingr_cnt / sl_i.sl_ingr_median <= 1.0 THEN 12
      WHEN ic.ingr_cnt / sl_i.sl_ingr_median <= 1.2 THEN 8
      WHEN ic.ingr_cnt / sl_i.sl_ingr_median <= 1.5 THEN 4
      ELSE 1
    END AS score_ingredient,

    -- 축6: 기능성 점수 (5점)
    CASE WHEN f.product_code IS NOT NULL THEN 5 ELSE 0 END AS score_functional,

    -- 축5: 골든 성분 점수 (20점 만점)
    LEAST(COALESCE(pfc.gold_count, 0) * 2 + COALESCE(pfc.silver_count, 0) * 1, 20) AS score_golden,

    -- 축8: 안티 성분 감점 (-20점 최대)
    GREATEST(-COALESCE(pfc.red_count, 0) * 2 - COALESCE(pfc.orange_count, 0) * 1, -20) AS score_anti,

    -- 개별 카운트 (Tableau 참조용)
    COALESCE(pfc.gold_count, 0)   AS gold_count,
    COALESCE(pfc.silver_count, 0) AS silver_count,
    COALESCE(pfc.red_count, 0)    AS red_count,
    COALESCE(pfc.orange_count, 0) AS orange_count,

    s.final_soft_landing

  FROM `daiso.sli_results` s
  JOIN `daiso.products_core` p       USING (product_code)
  JOIN `daiso.products_category` pc  USING (product_code)

  -- 카테고리별 SL 비율 (축1)
  JOIN (
    SELECT pc2.category_2,
           ROUND(COUNTIF(s2.final_soft_landing = TRUE) / COUNT(*) * 100, 1) AS sl_rate
    FROM `daiso.sli_results` s2
    JOIN `daiso.products_category` pc2 USING (product_code)
    GROUP BY pc2.category_2
  ) cat_rate ON pc.category_2 = cat_rate.category_2

  -- SL 제품 가격 중앙값 (축3)
  LEFT JOIN (
    SELECT pc3.category_2,
           APPROX_QUANTILES(p3.price, 100)[OFFSET(50)] AS sl_price_median
    FROM `daiso.sli_results` s3
    JOIN `daiso.products_core` p3       USING (product_code)
    JOIN `daiso.products_category` pc3  USING (product_code)
    WHERE s3.final_soft_landing = TRUE
    GROUP BY pc3.category_2
  ) sl_p ON pc.category_2 = sl_p.category_2

  -- SL 제품 성분 수 중앙값 (축4)
  LEFT JOIN (
    SELECT pc4.category_2,
           APPROX_QUANTILES(ic2.cnt, 100)[OFFSET(50)] AS sl_ingr_median
    FROM `daiso.sli_results` s4
    JOIN `daiso.products_category` pc4 USING (product_code)
    JOIN (
      SELECT product_code, COUNT(*) AS cnt
      FROM `daiso.products_ingredients`
      GROUP BY product_code
    ) ic2 USING (product_code)
    WHERE s4.final_soft_landing = TRUE
    GROUP BY pc4.category_2
  ) sl_i ON pc.category_2 = sl_i.category_2

  -- 제품별 성분 수 (축4)
  LEFT JOIN (
    SELECT product_code, COUNT(*) AS ingr_cnt
    FROM `daiso.products_ingredients`
    GROUP BY product_code
  ) ic ON s.product_code = ic.product_code

  -- 기능성 등록 여부 (축6)
  LEFT JOIN `daiso.functional` f ON s.product_code = f.product_code

  -- 골든/안티 성분 카운트 (축5, 축8)
  LEFT JOIN product_factor_counts pfc
    ON s.product_code = pfc.product_code
    AND pc.category_2 = pfc.category_2
)

SELECT
  product_code,
  category_2,
  score_category,
  score_price,
  score_ingredient,
  score_functional,
  score_golden,
  score_anti,
  gold_count,
  silver_count,
  red_count,
  orange_count,
  GREATEST(
    score_category + score_price + score_ingredient + score_functional
    + score_golden + score_anti,
    0
  ) AS base_score,
  final_soft_landing
FROM product_scores
ORDER BY base_score DESC;


-- =============================================
-- [2/2] v_ingredient_factors 패치
-- 변경: n_sl, n_nsl = 0일 때 나눗셈 방어 추가
-- =============================================
CREATE OR REPLACE VIEW `daiso.v_ingredient_factors` AS
WITH
-- SL / Non-SL 제품별 성분 출현 빈도
ingredient_freq AS (
  SELECT
    pc.category_2,
    pi.ingredient_id,
    COUNTIF(s.final_soft_landing = TRUE)  AS sl_product_count,
    COUNTIF(s.final_soft_landing = FALSE) AS nsl_product_count
  FROM `daiso.products_ingredients` pi
  JOIN `daiso.products_category` pc USING (product_code)
  JOIN `daiso.sli_results` s        USING (product_code)
  GROUP BY pc.category_2, pi.ingredient_id
),

-- 카테고리별 SL / Non-SL 제품 수
category_counts AS (
  SELECT
    pc.category_2,
    COUNTIF(s.final_soft_landing = TRUE)  AS n_sl,
    COUNTIF(s.final_soft_landing = FALSE) AS n_nsl
  FROM `daiso.sli_results` s
  JOIN `daiso.products_category` pc USING (product_code)
  GROUP BY pc.category_2
),

-- 출현률 + diff 계산
-- ★ PATCH: n_sl, n_nsl = 0 방어 (SAFE_DIVIDE 적용)
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
    ROUND(SAFE_DIVIDE(f.sl_product_count,  cc.n_sl)  * 100, 1)  AS sl_pct,
    ROUND(SAFE_DIVIDE(f.nsl_product_count, cc.n_nsl) * 100, 1)  AS nsl_pct,
    ROUND(SAFE_DIVIDE(f.sl_product_count,  cc.n_sl)  * 100, 1)
      - ROUND(SAFE_DIVIDE(f.nsl_product_count, cc.n_nsl) * 100, 1) AS diff_pct,
    cc.n_sl,
    cc.n_nsl
  FROM ingredient_freq f
  JOIN category_counts cc USING (category_2)
  JOIN `daiso-analysis.daiso.ingredients_dic` ig USING (ingredient_id)
  WHERE (f.sl_product_count + f.nsl_product_count) >= 3
),

-- +요인 (SL 우위) 순위
positive_ranked AS (
  SELECT *,
    ROW_NUMBER() OVER (
      PARTITION BY category_2
      ORDER BY diff_pct DESC
    ) AS pos_rank
  FROM ingredient_rates
  WHERE diff_pct > 0
),

-- -요인 (Non-SL 우위) 순위
negative_ranked AS (
  SELECT *,
    ROW_NUMBER() OVER (
      PARTITION BY category_2
      ORDER BY diff_pct ASC
    ) AS neg_rank
  FROM ingredient_rates
  WHERE diff_pct < 0
)

-- 합치기: 카테고리별 상위 +요인 10개 + -요인 10개
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
  CASE
    WHEN pos_rank <= 3 THEN 'Gold'
    WHEN pos_rank <= 10 THEN 'Silver'
  END AS factor_tier,
  pos_rank AS factor_rank
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
  CASE
    WHEN neg_rank <= 3 THEN 'Red'
    WHEN neg_rank <= 10 THEN 'Orange'
  END AS factor_tier,
  neg_rank AS factor_rank
FROM negative_ranked
WHERE neg_rank <= 10

ORDER BY category_2, factor_type DESC, factor_rank;
