-- =============================================
-- VIEW: 경쟁사 메트릭 비교 (런칭 시뮬레이터용)
-- =============================================
-- 용도: Tableau 런칭 시뮬레이터에서 동 카테고리 + 동 가격대 + 동 기능성
--       기준으로 경쟁사 6개 메트릭(engagement_score, cp_index, 성분 수,
--       review_count, likes, shares)을 바 차트로 비교
-- 기반: v_category_price_scenario (VIEW 2) 확장
-- 연결: 매개변수 p_category_2, p_price, p_is_functional 로 필터링
-- 정규화: 전체 평균 대비 세그먼트 비율(%) 계산용 글로벌 평균 포함
-- 생성일: 2026-03-12
-- 수정일: 2026-03-12 (전체 평균 컬럼 추가)
-- =============================================

CREATE OR REPLACE VIEW `daiso.v_competitor_metrics` AS
WITH ingredient_count AS (
  -- 제품별 전체 성분 수
  SELECT
    product_code,
    COUNT(*) AS ingredient_count
  FROM `daiso.products_ingredients`
  GROUP BY product_code
),

base AS (
  SELECT
    -- 기본 정보
    pc.category_1,
    pc.category_2,
    CASE
      WHEN p.price <= 1000 THEN '1K'
      WHEN p.price <= 2000 THEN '2K'
      WHEN p.price <= 3000 THEN '3K'
      ELSE '5K'
    END                                AS price_tier,
    p.price,
    p.product_code,
    p.name                             AS product_name,
    b.name                             AS brand_name,

    -- 기능성 여부 (4종 중 하나라도 해당)
    CASE
      WHEN f.is_whitening OR f.is_wrinkle_reduction OR f.is_sunscreen OR f.is_acne
      THEN TRUE
      ELSE FALSE
    END                                AS is_functional,

    -- 연착륙 정보
    s.final_soft_landing,
    s.ml_prob,
    s.confidence,

    -- ★ 6개 메트릭 (원본값)
    ps.engagement_score,
    ps.cp_index,
    COALESCE(ic.ingredient_count, 0)   AS ingredient_count,
    ps.review_count,
    ps.likes,
    ps.shares

  FROM `daiso.sli_results` s
  JOIN `daiso.products_core` p          USING (product_code)
  JOIN `daiso.products_category` pc     USING (product_code)
  JOIN `daiso.products_stats` ps        USING (product_code)
  JOIN `daiso.brands` b                 ON p.brand_id = b.brand_id
  LEFT JOIN `daiso.functional` f        ON p.product_code = f.product_code
  LEFT JOIN ingredient_count ic         ON p.product_code = ic.product_code
)

SELECT
  base.*,

  -- ★ 전체 다이소 뷰티 평균 (정규화 분모용)
  -- OVER()로 전체 행에 동일한 값 부여
  ROUND(AVG(engagement_score) OVER(), 2)    AS global_avg_engagement,
  ROUND(AVG(cp_index)         OVER(), 2)    AS global_avg_cp_index,
  ROUND(AVG(ingredient_count) OVER(), 2)    AS global_avg_ingredient,
  ROUND(AVG(review_count)     OVER(), 2)    AS global_avg_review,
  ROUND(AVG(likes)            OVER(), 2)    AS global_avg_likes,
  ROUND(AVG(shares)           OVER(), 2)    AS global_avg_shares

FROM base;
