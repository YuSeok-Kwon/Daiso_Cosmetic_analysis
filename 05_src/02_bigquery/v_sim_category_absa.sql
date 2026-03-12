-- =============================================
-- VIEW: 카테고리별 ABSA Aspect 감성 히트맵 (런칭 시뮬레이터용)
-- =============================================
-- 용도: Tableau 런칭 시뮬레이터에서 선택 카테고리의
--       Aspect별 긍정/부정/중립 비율을 히트맵으로 시각화
-- 연결: 매개변수 P_카테고리로 필터링
-- 생성일: 2026-03-10
-- =============================================

CREATE OR REPLACE VIEW `daiso.v_sim_category_absa` AS
SELECT
  pc.category_2,

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

  -- 해당 카테고리 총 리뷰 수 (언급률 계산용)
  MAX(cat_total.total_reviews) AS total_reviews,

  -- Aspect 언급률 (%)
  ROUND(SAFE_DIVIDE(COUNT(*), MAX(cat_total.total_reviews)) * 100, 1) AS mention_rate

FROM `daiso.review_aspects` ra
JOIN `daiso.reviews_core` rc    USING (review_id)
JOIN `daiso.products_category` pc USING (product_code)

-- 카테고리별 전체 리뷰 수 (언급률 분모)
JOIN (
  SELECT pc2.category_2, COUNT(DISTINCT rc2.review_id) AS total_reviews
  FROM `daiso.reviews_core` rc2
  JOIN `daiso.products_category` pc2 USING (product_code)
  GROUP BY pc2.category_2
) cat_total ON pc.category_2 = cat_total.category_2

WHERE ra.aspect NOT IN ('미분류')

GROUP BY pc.category_2, ra.aspect
ORDER BY pc.category_2, mention_count DESC;
