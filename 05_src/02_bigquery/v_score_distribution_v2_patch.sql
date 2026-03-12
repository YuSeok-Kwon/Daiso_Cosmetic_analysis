-- =============================================
-- PATCH: v_score_distribution에 축5(골든성분) + 축8(안티성분) 추가
-- =============================================
-- 변경 내용:
--   1. products_ingredients ↔ v_ingredient_factors JOIN으로
--      각 제품별 gold/silver/red/orange 성분 매칭 수 계산
--   2. score_golden (축5): MIN(gold*2 + silver*1, 20)
--   3. score_anti (축8): MAX(-red*2 - orange*1, -20)
--   4. base_score: 기존 40점 + 축5(+20) + 축8(-20) = 최대 60점
--
-- 적용 방법: 이 SQL을 BQ에서 실행하면 기존 뷰를 대체함
-- 생성일: 2026-03-08
-- =============================================

CREATE OR REPLACE VIEW `daiso.v_score_distribution` AS
WITH
-- 각 제품별 골든/안티 성분 매칭 수 집계
product_factor_counts AS (
  SELECT
    pi.product_code,
    pc.category_2,
    -- 골든 성분 (positive factor)
    COUNTIF(vf.factor_type = 'positive' AND vf.factor_tier = 'Gold')   AS gold_count,
    COUNTIF(vf.factor_type = 'positive' AND vf.factor_tier = 'Silver') AS silver_count,
    -- 안티 성분 (negative factor)
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
    CASE
      WHEN sl_p.sl_price_median IS NULL THEN 5
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

    -- ★ 축5: 골든 성분 점수 (20점 만점)
    LEAST(COALESCE(pfc.gold_count, 0) * 2 + COALESCE(pfc.silver_count, 0) * 1, 20) AS score_golden,

    -- ★ 축8: 안티 성분 감점 (-20점 최대)
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

  -- ★ 골든/안티 성분 카운트 (축5, 축8)
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
  score_golden,       -- ★ 신규: 축5
  score_anti,         -- ★ 신규: 축8
  gold_count,         -- ★ 신규: 개별 카운트
  silver_count,       -- ★ 신규
  red_count,          -- ★ 신규
  orange_count,       -- ★ 신규
  -- base_score: 축1+3+4+6+5+8 = 최대 60점 (최소 -20+4=음수 가능, 0 클리핑)
  GREATEST(
    score_category + score_price + score_ingredient + score_functional
    + score_golden + score_anti,
    0
  ) AS base_score,
  final_soft_landing
FROM product_scores
ORDER BY base_score DESC;
