-- =============================================
-- Tableau 입점 성공 확률 시뮬레이터 - BQ 참조 뷰 (v2)
-- 용도: Tableau 라이브 연결 → 스코어카드 + 시나리오 + 벤치마크
-- 생성일: 2026-03-05
-- 수정일: 2026-03-08 (v2 - 배점 구조 전면 개편: 100점 만점 체계)
-- =============================================
--
-- [v2 배점 변경 사항]
--   축1. 카테고리 적합성     30점 → 5점  (가산점화, 비선형 3단계)
--   축2. 브랜드 인지도       25점 → 25점 (변경 없음, 사용자 입력)
--   축3. 가격 포지셔닝       15점 → 15점 (변경 없음)
--   축4. 성분 심플리시티      10점 → 15점 (비선형, SL중앙값 대비 비율)
--   축5. 주 성분 적합성      15점 → 20점 (골든 성분 매칭률, 사용자 입력)
--   축6. 기능성 등록          5점 →  5점 (변경 없음)
--   축7. 성분 타입 적합성    10점 → 15점 (사용자 입력)
--   축8. 안티성분 감점         -  → -20점 (신규, Tableau에서 처리)
--
-- [합계] 축1~7 = 100점 만점, 축8 = 0~-20 감점
-- [등급] 카테고리별 백분위 기반 동적 cutoff (v_category_grade_cutoffs)
-- [base_score] 축1+축3+축4+축6 = 최대 40점 (축2/5/7은 사용자 입력)
-- =============================================


-- =============================================
-- VIEW 1: 카테고리별 연착륙 기준 통계
-- =============================================
-- 용도: 스코어카드 축1(카테고리) + 축3(가격) + 축4(성분 수)
-- Tableau: 파라미터로 선택한 category_2와 JOIN하여 기준값 참조
-- [v2] category_score_max30: 30/18/8/2 → 5/3/2/1
-- =============================================
CREATE OR REPLACE VIEW `daiso.v_category_scorecard_ref` AS
WITH category_sl AS (
  SELECT
    pc.category_1,
    pc.category_2,
    COUNT(*)                                                          AS total_products,
    COUNTIF(s.final_soft_landing = TRUE)                              AS sl_count,
    ROUND(COUNTIF(s.final_soft_landing = TRUE) / COUNT(*) * 100, 1)  AS sl_rate
  FROM `daiso.sli_results` s
  JOIN `daiso.products_category` pc USING (product_code)
  GROUP BY pc.category_1, pc.category_2
),

price_by_cat AS (
  SELECT
    pc.category_2,
    APPROX_QUANTILES(p.price, 100)[OFFSET(25)] AS sl_price_q1,
    APPROX_QUANTILES(p.price, 100)[OFFSET(50)] AS sl_price_median,
    APPROX_QUANTILES(p.price, 100)[OFFSET(75)] AS sl_price_q3,
    AVG(p.price)                                AS sl_price_mean,
    MIN(p.price)                                AS sl_price_min,
    MAX(p.price)                                AS sl_price_max
  FROM `daiso.sli_results` s
  JOIN `daiso.products_core` p      USING (product_code)
  JOIN `daiso.products_category` pc USING (product_code)
  WHERE s.final_soft_landing = TRUE
  GROUP BY pc.category_2
),

ingr_by_cat AS (
  SELECT
    pc.category_2,
    APPROX_QUANTILES(
      IF(s.final_soft_landing = TRUE, ingr_cnt, NULL), 100
    )[OFFSET(50)]  AS sl_ingredient_median,
    APPROX_QUANTILES(
      IF(s.final_soft_landing = FALSE, ingr_cnt, NULL), 100
    )[OFFSET(50)]  AS nonsl_ingredient_median,
    APPROX_QUANTILES(ingr_cnt, 100)[OFFSET(50)] AS all_ingredient_median
  FROM `daiso.sli_results` s
  JOIN `daiso.products_category` pc USING (product_code)
  JOIN (
    SELECT product_code, COUNT(*) AS ingr_cnt
    FROM `daiso.products_ingredients`
    GROUP BY product_code
  ) ic USING (product_code)
  GROUP BY pc.category_2
)

SELECT
  cs.category_1,
  cs.category_2,
  cs.total_products,
  cs.sl_count,
  cs.sl_rate,

  -- [v2] 축1 카테고리 점수 (5점 만점) - 비선형 3단계 (가산점 성격)
  -- 컬럼명 유지 (Tableau 참조 호환): category_score_max30
  -- 실제 값: 5/3/2/1
  CASE
    WHEN cs.sl_rate >= 35 THEN 5    -- 립케어(55.6%), 팩/마스크(40.6%), 클렌징(35.7%)
    WHEN cs.sl_rate >= 20 THEN 3    -- 기초스킨케어(27.5%), 베이스(23.6%), 아이(21.6%), 자외선(27.6%)
    WHEN cs.sl_rate >= 5  THEN 2    -- 치크/하이라이터(6.6%)
    ELSE 1                           -- 립메이크업(1.7%)
  END AS category_score_max30,

  -- 축3 가격 참조값
  pbc.sl_price_q1,
  pbc.sl_price_median,
  pbc.sl_price_q3,
  pbc.sl_price_mean,
  pbc.sl_price_min,
  pbc.sl_price_max,

  -- 축4 성분 수 참조값
  ibc.sl_ingredient_median,
  ibc.nonsl_ingredient_median,
  ibc.all_ingredient_median
FROM category_sl cs
LEFT JOIN price_by_cat pbc USING (category_2)
LEFT JOIN ingr_by_cat ibc  USING (category_2)
ORDER BY cs.sl_rate DESC;


-- =============================================
-- VIEW 2: 카테고리 x 가격대 시나리오 (제품 레벨)
-- =============================================
-- [v2] 변경 없음
-- =============================================
CREATE OR REPLACE VIEW `daiso.v_category_price_scenario` AS
SELECT
  pc.category_1,
  pc.category_2,
  CASE
    WHEN p.price <= 1000 THEN '1K'
    WHEN p.price <= 2000 THEN '2K'
    WHEN p.price <= 3000 THEN '3K'
    ELSE '5K'
  END                          AS price_tier,
  p.price,
  p.product_code,
  p.name                       AS product_name,
  b.name                       AS brand_name,
  s.final_soft_landing,
  s.ml_prob,
  s.confidence,
  ps.review_count,
  ps.engagement_score,
  ps.cp_index
FROM `daiso.sli_results` s
JOIN `daiso.products_core` p       USING (product_code)
JOIN `daiso.products_category` pc  USING (product_code)
JOIN `daiso.products_stats` ps     USING (product_code)
JOIN `daiso.brands` b              ON p.brand_id = b.brand_id;


-- =============================================
-- VIEW 3: 브랜드별 연착륙 실적 (벤치마크)
-- =============================================
-- [v2] 변경 없음
-- =============================================
CREATE OR REPLACE VIEW `daiso.v_brand_benchmark` AS
WITH brand_primary_cat AS (
  SELECT
    p.brand_id,
    pc.category_2,
    COUNT(*) AS cat_cnt,
    ROW_NUMBER() OVER (PARTITION BY p.brand_id ORDER BY COUNT(*) DESC) AS rn
  FROM `daiso.products_core` p
  JOIN `daiso.products_category` pc USING (product_code)
  GROUP BY p.brand_id, pc.category_2
)

SELECT
  b.brand_id,
  b.name                                                              AS brand_name,
  COUNT(*)                                                            AS total_products,
  COUNTIF(s.final_soft_landing = TRUE)                                AS sl_count,
  ROUND(COUNTIF(s.final_soft_landing = TRUE) / COUNT(*) * 100, 1)    AS sl_rate,
  ROUND(AVG(ps.engagement_score), 1)                                  AS avg_engagement,
  ROUND(AVG(ps.cp_index), 1)                                          AS avg_cp_index,
  ROUND(AVG(ps.review_count), 0)                                      AS avg_review_count,
  bpc.category_2                                                      AS primary_category
FROM `daiso.sli_results` s
JOIN `daiso.products_core` p      USING (product_code)
JOIN `daiso.products_stats` ps    USING (product_code)
JOIN `daiso.brands` b             ON p.brand_id = b.brand_id
LEFT JOIN brand_primary_cat bpc   ON b.brand_id = bpc.brand_id AND bpc.rn = 1
GROUP BY b.brand_id, b.name, bpc.category_2
ORDER BY total_products DESC;


-- =============================================
-- VIEW 4: 카테고리 x 가격대 집계 (히트맵)
-- =============================================
-- [v2] 변경 없음
-- =============================================
CREATE OR REPLACE VIEW `daiso.v_scenario_heatmap` AS
SELECT
  pc.category_1,
  pc.category_2,
  CASE
    WHEN p.price <= 1000 THEN '1K'
    WHEN p.price <= 2000 THEN '2K'
    WHEN p.price <= 3000 THEN '3K'
    ELSE '5K'
  END                                                                  AS price_tier,
  COUNT(*)                                                             AS total_products,
  COUNTIF(s.final_soft_landing = TRUE)                                 AS sl_count,
  ROUND(COUNTIF(s.final_soft_landing = TRUE) / COUNT(*) * 100, 1)     AS sl_rate,
  ROUND(AVG(s.ml_prob) * 100, 1)                                       AS avg_ml_prob_pct
FROM `daiso.sli_results` s
JOIN `daiso.products_core` p       USING (product_code)
JOIN `daiso.products_category` pc  USING (product_code)
GROUP BY pc.category_1, pc.category_2, price_tier
ORDER BY pc.category_1, pc.category_2, price_tier;


-- =============================================
-- VIEW 5: 카테고리별 골든 성분 (Golden Ingredients)
-- =============================================
-- [v2] 변경 없음
-- =============================================
CREATE OR REPLACE VIEW `daiso.v_golden_ingredients` AS
WITH sl_products AS (
  SELECT
    s.product_code,
    s.final_soft_landing,
    pc.category_2
  FROM `daiso.sli_results` s
  JOIN `daiso.products_category` pc USING (product_code)
),

category_counts AS (
  SELECT
    category_2,
    COUNTIF(final_soft_landing = TRUE)  AS n_sl,
    COUNTIF(final_soft_landing = FALSE) AS n_nsl
  FROM sl_products
  GROUP BY category_2
),

ingredient_freq AS (
  SELECT
    sp.category_2,
    pi.ingredient_id,
    COUNTIF(sp.final_soft_landing = TRUE)  AS sl_product_count,
    COUNTIF(sp.final_soft_landing = FALSE) AS nsl_product_count
  FROM `daiso.products_ingredients` pi
  JOIN sl_products sp USING (product_code)
  GROUP BY sp.category_2, pi.ingredient_id
),

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
      - ROUND(f.nsl_product_count / cc.n_nsl * 100, 1) AS diff_pct
  FROM ingredient_freq f
  JOIN category_counts cc USING (category_2)
  JOIN `daiso.ingredients_dic` ig USING (ingredient_id)
  WHERE f.sl_product_count >= 3
),

ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY category_2
      ORDER BY diff_pct DESC
    ) AS rank_in_category
  FROM ingredient_rates
  WHERE diff_pct > 0
)

SELECT
  category_2,
  rank_in_category,
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
  CASE
    WHEN rank_in_category <= 5  THEN 'Gold'
    WHEN rank_in_category <= 10 THEN 'Silver'
    ELSE 'Bronze'
  END AS golden_tier
FROM ranked
WHERE rank_in_category <= 15
ORDER BY category_2, rank_in_category;


-- =============================================
-- VIEW 6: 카테고리별 골든 성분 집계 요약
-- =============================================
-- [v2] 변경 없음
-- =============================================
CREATE OR REPLACE VIEW `daiso.v_golden_ingredients_summary` AS
SELECT
  category_2,
  STRING_AGG(
    IF(golden_tier = 'Gold', ingredient_name, NULL),
    ', ' ORDER BY rank_in_category
  ) AS gold_ingredients,
  STRING_AGG(
    IF(golden_tier = 'Silver', ingredient_name, NULL),
    ', ' ORDER BY rank_in_category
  ) AS silver_ingredients,
  COUNTIF(golden_tier = 'Gold')   AS gold_count,
  COUNTIF(golden_tier = 'Silver') AS silver_count,
  ANY_VALUE(effect HAVING MAX diff_pct) AS dominant_effect
FROM `daiso.v_golden_ingredients`
WHERE golden_tier IN ('Gold', 'Silver')
GROUP BY category_2
ORDER BY category_2;


-- =============================================
-- VIEW 7: 기존 제품 스코어 분포 (백분위 참조)
-- =============================================
-- [v2 변경]
--   축1: 30/18/8/2 → 5/3/2/1
--   축4: 10/7/4/1 → 15/10/6/2
--   base_score max: 60 → 40 (축2:25 + 축5:20 + 축7:15 = 60은 사용자 입력)
-- =============================================
CREATE OR REPLACE VIEW `daiso.v_score_distribution` AS
WITH product_scores AS (
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

    -- 축3: 비선형 가격 점수 (15점 만점, 변경 없음)
    CASE
      WHEN sl_p.sl_price_median IS NULL THEN 5
      WHEN ABS(p.price - sl_p.sl_price_median) / sl_p.sl_price_median <= 0.0 THEN 15
      WHEN ABS(p.price - sl_p.sl_price_median) / sl_p.sl_price_median <= 0.2 THEN 12
      WHEN ABS(p.price - sl_p.sl_price_median) / sl_p.sl_price_median <= 0.4 THEN 8
      ELSE 4
    END AS score_price,

    -- [v2] 축4: 비선형 성분 수 점수 (15점 만점)
    CASE
      WHEN sl_i.sl_ingr_median IS NULL OR sl_i.sl_ingr_median = 0 THEN 7
      WHEN ic.ingr_cnt / sl_i.sl_ingr_median <= 1.0 THEN 15
      WHEN ic.ingr_cnt / sl_i.sl_ingr_median <= 1.2 THEN 10
      WHEN ic.ingr_cnt / sl_i.sl_ingr_median <= 1.5 THEN 6
      ELSE 2
    END AS score_ingredient,

    -- 축6: 기능성 점수 (5점, 변경 없음)
    CASE WHEN f.product_code IS NOT NULL THEN 5 ELSE 0 END AS score_functional,

    s.final_soft_landing

  FROM `daiso.sli_results` s
  JOIN `daiso.products_core` p       USING (product_code)
  JOIN `daiso.products_category` pc  USING (product_code)

  JOIN (
    SELECT pc2.category_2,
           ROUND(COUNTIF(s2.final_soft_landing = TRUE) / COUNT(*) * 100, 1) AS sl_rate
    FROM `daiso.sli_results` s2
    JOIN `daiso.products_category` pc2 USING (product_code)
    GROUP BY pc2.category_2
  ) cat_rate ON pc.category_2 = cat_rate.category_2

  LEFT JOIN (
    SELECT pc3.category_2,
           APPROX_QUANTILES(p3.price, 100)[OFFSET(50)] AS sl_price_median
    FROM `daiso.sli_results` s3
    JOIN `daiso.products_core` p3       USING (product_code)
    JOIN `daiso.products_category` pc3  USING (product_code)
    WHERE s3.final_soft_landing = TRUE
    GROUP BY pc3.category_2
  ) sl_p ON pc.category_2 = sl_p.category_2

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

  LEFT JOIN (
    SELECT product_code, COUNT(*) AS ingr_cnt
    FROM `daiso.products_ingredients`
    GROUP BY product_code
  ) ic ON s.product_code = ic.product_code

  LEFT JOIN `daiso.functional` f ON s.product_code = f.product_code
)

SELECT
  product_code,
  category_2,
  score_category,
  score_price,
  score_ingredient,
  score_functional,
  -- [v2] 기본 점수 (축2/5/7 제외) → 최대 40점
  (score_category + score_price + score_ingredient + score_functional) AS base_score,
  final_soft_landing
FROM product_scores
ORDER BY base_score DESC;


-- =============================================
-- VIEW 8: 백분위 구간 테이블
-- =============================================
-- [v2] base_score 범위 변경에 따라 자동 반영 (수식 변경 없음)
-- =============================================
CREATE OR REPLACE VIEW `daiso.v_score_percentiles` AS
SELECT
  pct,
  score_value,
  '기본 점수 (브랜드/골든/성분타입 제외)' AS score_type
FROM UNNEST([10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95]) AS pct
JOIN (
  SELECT
    APPROX_QUANTILES(base_score, 100) AS quantiles
  FROM `daiso.v_score_distribution`
) q
ON TRUE
JOIN UNNEST(q.quantiles) AS score_value WITH OFFSET AS offset_val
ON offset_val = pct
ORDER BY pct;


-- =============================================
-- VIEW 9: 카테고리별 base_score 벤치마크 (BAN용)
-- =============================================
-- [v2] 등급 기준 변경: base_score max = 40이므로 절대 기준 불가
--      → 카테고리별 백분위 기반 동적 등급 (v_category_grade_cutoffs 참조)
--      → 여기서는 참고용 분포 통계만 제공
-- =============================================
CREATE OR REPLACE VIEW `daiso.v_sim_category_benchmark` AS
WITH cat_pcts AS (
  -- [v2] 카테고리별 백분위 컷오프 먼저 산출 (집계 중첩 방지)
  SELECT
    category_2,
    APPROX_QUANTILES(base_score, 100)[OFFSET(90)] AS p90,
    APPROX_QUANTILES(base_score, 100)[OFFSET(70)] AS p70,
    APPROX_QUANTILES(base_score, 100)[OFFSET(40)] AS p40
  FROM `daiso.v_score_distribution`
  GROUP BY category_2
),
cat_stats AS (
  SELECT
    sd.category_2,
    COUNT(*)                                   AS n_products,
    ROUND(AVG(sd.base_score), 1)               AS avg_base_score,
    ROUND(STDDEV(sd.base_score), 1)            AS std_base_score,
    APPROX_QUANTILES(sd.base_score, 100)[OFFSET(50)] AS median_base_score,
    APPROX_QUANTILES(sd.base_score, 100)[OFFSET(25)] AS q1_base_score,
    APPROX_QUANTILES(sd.base_score, 100)[OFFSET(75)] AS q3_base_score,
    MIN(sd.base_score)                         AS min_base_score,
    MAX(sd.base_score)                         AS max_base_score,

    -- [v2] 등급 카운트: 카테고리별 백분위 기준
    COUNTIF(sd.base_score >= cp.p90)                                  AS grade_A_cnt,
    COUNTIF(sd.base_score >= cp.p70 AND sd.base_score < cp.p90)       AS grade_B_cnt,
    COUNTIF(sd.base_score >= cp.p40 AND sd.base_score < cp.p70)       AS grade_C_cnt,
    COUNTIF(sd.base_score < cp.p40)                                   AS grade_D_cnt,

    -- SL 제품만의 base_score 통계
    ROUND(AVG(IF(sd.final_soft_landing = TRUE, sd.base_score, NULL)), 1)  AS sl_avg_base_score,
    COUNTIF(sd.final_soft_landing = TRUE)                                  AS sl_count,

    ROUND(COUNTIF(sd.final_soft_landing = TRUE) / COUNT(*) * 100, 1)      AS sl_rate_pct

  FROM `daiso.v_score_distribution` sd
  JOIN cat_pcts cp USING (category_2)
  GROUP BY sd.category_2
)

SELECT
  category_2,
  n_products,
  avg_base_score,
  std_base_score,
  median_base_score,
  q1_base_score,
  q3_base_score,
  min_base_score,
  max_base_score,

  grade_A_cnt,
  grade_B_cnt,
  grade_C_cnt,
  grade_D_cnt,

  sl_count,
  sl_avg_base_score,
  sl_rate_pct,

  ROUND(sl_avg_base_score - avg_base_score, 1)  AS sl_score_premium

FROM cat_stats
ORDER BY n_products DESC;


-- =============================================
-- VIEW 10: 연착륙 제품 축별 점수 상세 (유사 제품 매칭용)
-- =============================================
-- [v2] 정규화 분모 변경: category/30→/5, ingredient/10→/15
-- =============================================
CREATE OR REPLACE VIEW `daiso.v_sim_similar_sl_products` AS
SELECT
  sd.product_code,
  sd.category_2,
  p.name                       AS product_name,
  b.name                       AS brand_name,
  p.price,
  ps.review_count,
  ps.engagement_score,
  ps.cp_index,
  s.ml_prob,
  s.confidence,

  sd.score_category,
  sd.score_price,
  sd.score_ingredient,
  sd.score_functional,
  sd.base_score,

  -- [v2] 정규화 분모 변경
  ROUND(sd.score_category / 5.0, 3)    AS norm_category,    -- 5점 만점
  ROUND(sd.score_price / 15.0, 3)      AS norm_price,       -- 15점 만점 (변경 없음)
  ROUND(sd.score_ingredient / 15.0, 3) AS norm_ingredient,  -- 15점 만점
  ROUND(sd.score_functional / 5.0, 3)  AS norm_functional   --  5점 만점 (변경 없음)

FROM `daiso.v_score_distribution` sd
JOIN `daiso.sli_results` s      USING (product_code)
JOIN `daiso.products_core` p    USING (product_code)
JOIN `daiso.products_stats` ps  USING (product_code)
JOIN `daiso.brands` b           ON p.brand_id = b.brand_id

WHERE sd.final_soft_landing = TRUE

ORDER BY sd.category_2, sd.base_score DESC;
