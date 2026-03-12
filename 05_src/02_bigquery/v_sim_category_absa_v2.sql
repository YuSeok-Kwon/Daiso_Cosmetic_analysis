-- =============================================
-- VIEW: 카테고리 × 가격대 × 기능성별 ABSA Aspect 감성 (런칭 시뮬레이터용)
-- =============================================
-- 용도: Tableau 런칭 시뮬레이터에서 동 카테고리 + 동 가격대 + 동 기능성
--       기준으로 Aspect별 긍정/부정/중립 비율을 히트맵으로 시각화
-- 기반: v_sim_category_absa (VIEW) 확장
--   ※ 기존 뷰는 category_2 × aspect 수준 집계
--     → v2는 category_2 × price_tier × is_functional × aspect 수준 재집계
-- 연결: 매개변수 P_카테고리, p_price, p_is_functional 로 필터링
-- 생성일: 2026-03-12
-- =============================================

CREATE OR REPLACE VIEW `daiso.v_sim_category_absa_v2` AS
WITH product_info AS (
  -- 제품별 가격대 + 기능성 여부 사전 계산
  SELECT
    p.product_code,
    pc.category_2,
    CASE
      WHEN p.price <= 1000 THEN '1K'
      WHEN p.price <= 2000 THEN '2K'
      WHEN p.price <= 3000 THEN '3K'
      ELSE '5K'
    END AS price_tier,
    CASE
      WHEN f.is_whitening OR f.is_wrinkle_reduction OR f.is_sunscreen OR f.is_acne
      THEN TRUE
      ELSE FALSE
    END AS is_functional
  FROM `daiso.products_core` p
  JOIN `daiso.products_category` pc   USING (product_code)
  LEFT JOIN `daiso.functional` f      ON p.product_code = f.product_code
),

-- 그룹(카테고리 × 가격대 × 기능성)별 전체 리뷰 수 (언급률 분모)
group_total AS (
  SELECT
    pi.category_2,
    pi.price_tier,
    pi.is_functional,
    COUNT(DISTINCT rc.review_id) AS total_reviews
  FROM `daiso.reviews_core` rc
  JOIN product_info pi USING (product_code)
  GROUP BY pi.category_2, pi.price_tier, pi.is_functional
)

SELECT
  pi.category_2,
  pi.price_tier,
  pi.is_functional,

  ra.aspect,

  -- 감성 건수
  COUNTIF(ra.aspect_sentiment = 'positive') AS pos_count,
  COUNTIF(ra.aspect_sentiment = 'neutral')  AS neu_count,
  COUNTIF(ra.aspect_sentiment = 'negative') AS neg_count,
  COUNT(*)                                  AS mention_count,

  -- 감성 비율 (%)
  ROUND(SAFE_DIVIDE(COUNTIF(ra.aspect_sentiment = 'positive'), COUNT(*)) * 100, 1) AS pos_rate,
  ROUND(SAFE_DIVIDE(COUNTIF(ra.aspect_sentiment = 'neutral'),  COUNT(*)) * 100, 1) AS neu_rate,
  ROUND(SAFE_DIVIDE(COUNTIF(ra.aspect_sentiment = 'negative'), COUNT(*)) * 100, 1) AS neg_rate,

  -- 해당 그룹 총 리뷰 수 (언급률 계산용)
  MAX(gt.total_reviews) AS total_reviews,

  -- Aspect 언급률 (%)
  ROUND(SAFE_DIVIDE(COUNT(*), MAX(gt.total_reviews)) * 100, 1) AS mention_rate

FROM `daiso.review_aspects` ra
JOIN `daiso.reviews_core` rc     USING (review_id)
JOIN product_info pi             ON rc.product_code = pi.product_code
JOIN group_total gt              ON pi.category_2   = gt.category_2
                                AND pi.price_tier    = gt.price_tier
                                AND pi.is_functional = gt.is_functional

WHERE ra.aspect NOT IN ('미분류')

GROUP BY
  pi.category_2,
  pi.price_tier,
  pi.is_functional,
  ra.aspect

ORDER BY
  pi.category_2,
  pi.price_tier,
  pi.is_functional,
  mention_count DESC;
