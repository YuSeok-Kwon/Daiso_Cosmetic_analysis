-- =============================================
-- 입점 시뮬레이터 개선 - 추가 BQ 뷰 (전략 2)
-- 용도: 상단 BAN 카드 + 중단 우측 유사 연착륙 제품 Top-5
-- 생성일: 2026-03-06
-- 의존: v_score_distribution, v_category_scorecard_ref, sli_results
-- =============================================
--
-- [추가 컴포넌트]
--   상단 BAN 3개:
--     ① 카테고리 평균 base_score (내 점수 vs 카테고리 평균)
--     ② 카테고리 내 순위 ("N개 중 M위")
--     ③ 연착륙 확률 (해당 카테고리 SL 비율 기반 보정)
--
--   중단 우측:
--     유사 연착륙 제품 Top-5 (같은 카테고리 SL 제품 중 축별 점수 유사도 순)
-- =============================================


-- =============================================
-- VIEW 9: 카테고리별 base_score 벤치마크 (BAN용)
-- =============================================
-- 용도: Tableau BAN 카드에서 카테고리 평균, 순위 참조
-- 설계: 카테고리별 평균/중앙값/표준편차 + 등급별 분포
-- Tableau: 파라미터(p_category)로 필터 → BAN에 수치 표시
-- =============================================
CREATE OR REPLACE VIEW `daiso-analysis.daiso.v_sim_category_benchmark` AS
WITH cat_stats AS (
  SELECT
    category_2,
    COUNT(*)                                   AS n_products,
    ROUND(AVG(base_score), 1)                  AS avg_base_score,
    ROUND(STDDEV(base_score), 1)               AS std_base_score,
    APPROX_QUANTILES(base_score, 100)[OFFSET(50)] AS median_base_score,
    APPROX_QUANTILES(base_score, 100)[OFFSET(25)] AS q1_base_score,
    APPROX_QUANTILES(base_score, 100)[OFFSET(75)] AS q3_base_score,
    MIN(base_score)                            AS min_base_score,
    MAX(base_score)                            AS max_base_score,

    -- 등급별 제품 수
    COUNTIF(base_score >= 85)                  AS grade_A_cnt,
    COUNTIF(base_score >= 70 AND base_score < 85)  AS grade_B_cnt,
    COUNTIF(base_score >= 55 AND base_score < 70)  AS grade_C_cnt,
    COUNTIF(base_score < 55)                   AS grade_D_cnt,

    -- SL 제품만의 base_score 통계
    ROUND(AVG(IF(final_soft_landing = TRUE, base_score, NULL)), 1)  AS sl_avg_base_score,
    COUNTIF(final_soft_landing = TRUE)                               AS sl_count,

    -- 연착륙 확률 = SL비율 (이미 v_category_scorecard_ref에 있으나 여기서도 계산)
    ROUND(COUNTIF(final_soft_landing = TRUE) / COUNT(*) * 100, 1)   AS sl_rate_pct

  FROM `daiso-analysis.daiso.v_score_distribution`
  GROUP BY category_2
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

  -- BAN ③용: 사용자 점수에 따른 연착륙 확률 보정 기울기
  -- 로직: "카테고리 SL 평균 점수 이상이면 SL비율 × 1.5, 미만이면 × 0.7"
  -- → Tableau calculated field에서 처리하되, 필요 파라미터를 여기서 제공
  ROUND(sl_avg_base_score - avg_base_score, 1)  AS sl_score_premium

FROM cat_stats
ORDER BY n_products DESC;


-- =============================================
-- VIEW 10: 연착륙 제품 축별 점수 상세 (유사 제품 매칭용)
-- =============================================
-- 용도: 같은 카테고리의 SL 제품 축별 점수를 미리 계산
--       Tableau에서 사용자 입력 점수와의 거리(유클리디안) 산출 → Top-5
-- 설계: v_score_distribution 기반, SL 제품만 필터
--       + 제품명, 브랜드명, 가격, 리뷰 수 등 표시 정보 포함
-- =============================================
CREATE OR REPLACE VIEW `daiso-analysis.daiso.v_sim_similar_sl_products` AS
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

  -- 축별 점수 (Tableau에서 유사도 계산에 사용)
  sd.score_category,
  sd.score_price,
  sd.score_ingredient,
  sd.score_functional,
  sd.base_score,

  -- 축별 점수를 0~1로 정규화 (거리 계산 시 스케일 통일)
  ROUND(sd.score_category / 30.0, 3)   AS norm_category,    -- 30점 만점
  ROUND(sd.score_price / 15.0, 3)      AS norm_price,       -- 15점 만점
  ROUND(sd.score_ingredient / 10.0, 3) AS norm_ingredient,  -- 10점 만점
  ROUND(sd.score_functional / 5.0, 3)  AS norm_functional   --  5점 만점

FROM `daiso-analysis.daiso.v_score_distribution` sd
JOIN `daiso-analysis.daiso.sli_results` s      USING (product_code)
JOIN `daiso-analysis.daiso.products_core` p    USING (product_code)
JOIN `daiso-analysis.daiso.products_stats` ps  USING (product_code)
JOIN `daiso-analysis.daiso.brands` b           ON p.brand_id = b.brand_id

WHERE sd.final_soft_landing = TRUE   -- 연착륙 제품만

ORDER BY sd.category_2, sd.base_score DESC;
