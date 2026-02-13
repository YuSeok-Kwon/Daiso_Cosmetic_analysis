"""
ABSA BigQuery 연동 모듈

리뷰 데이터 로드 및 분석 결과 저장을 위한 BigQuery 연동

사용법:
    from bq_connector import ABSABigQuery

    bq = ABSABigQuery()

    # BigQuery에서 리뷰 로드
    df = bq.load_reviews(limit=10000)

    # 분석 결과 업데이트
    bq.update_review_analysis(results_df)
"""
import sys
from pathlib import Path
import pandas as pd
from typing import Optional, List

# BigQuery 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from BigQuery.bq_client import (
        get_client, query_to_df, upsert_df, get_table_count, DEFAULT_DATASET
    )
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False
    print("Warning: BigQuery 모듈을 찾을 수 없습니다.")


class ABSABigQuery:
    """ABSA용 BigQuery 연동 클래스"""

    def __init__(self, dataset: str = None):
        if not BIGQUERY_AVAILABLE:
            raise ImportError("BigQuery 모듈이 설치되지 않았습니다.")

        self.dataset = dataset or DEFAULT_DATASET
        self.client = get_client()

    def load_reviews(
        self,
        limit: Optional[int] = None,
        product_codes: Optional[List[int]] = None,
        min_length: int = 10,
        exclude_analyzed: bool = False
    ) -> pd.DataFrame:
        """
        BigQuery에서 리뷰 데이터 로드

        Args:
            limit: 최대 로드 행 수
            product_codes: 특정 제품 코드만 로드
            min_length: 최소 텍스트 길이
            exclude_analyzed: 이미 분석된 리뷰 제외

        Returns:
            리뷰 DataFrame (order_id, product_code, text, rating 등)
        """
        conditions = [f"LENGTH(r.text) >= {min_length}"]

        if product_codes:
            codes_str = ", ".join(map(str, product_codes))
            conditions.append(f"r.product_code IN ({codes_str})")

        if exclude_analyzed:
            conditions.append("ra.sentiment IS NULL")

        where_clause = " AND ".join(conditions)
        limit_clause = f"LIMIT {limit}" if limit else ""

        sql = f"""
        SELECT
            r.order_id,
            r.product_code,
            r.user_id,
            r.write_date,
            r.rating,
            r.text,
            r.image_count,
            r.is_reorder,
            ra.sentiment,
            ra.sentiment_score
        FROM `{self.dataset}.reviews` r
        LEFT JOIN `{self.dataset}.review_analysis` ra
            ON r.order_id = ra.order_id
        WHERE {where_clause}
        ORDER BY r.order_id
        {limit_clause}
        """

        df = query_to_df(sql)
        print(f"리뷰 {len(df):,}개 로드 완료")
        return df

    def load_unanalyzed_reviews(self, limit: Optional[int] = None) -> pd.DataFrame:
        """분석되지 않은 리뷰만 로드"""
        return self.load_reviews(limit=limit, exclude_analyzed=True)

    def update_review_analysis(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        save_csv: Optional[str] = None
    ) -> int:
        """
        분석 결과를 review_analysis 테이블에 업데이트

        Args:
            df: 분석 결과 DataFrame (order_id 필수)
            columns: 업데이트할 컬럼 목록 (기본: sentiment, sentiment_score)
            save_csv: CSV 저장 경로 (옵션)

        Returns:
            업데이트된 행 수
        """
        if 'order_id' not in df.columns:
            raise ValueError("order_id 컬럼이 필요합니다.")

        # 기본 업데이트 컬럼
        if columns is None:
            columns = ['sentiment', 'sentiment_score']

        # 필요한 컬럼만 선택
        update_cols = ['order_id'] + [c for c in columns if c in df.columns]
        df_update = df[update_cols].copy()

        # 없는 컬럼은 기존 값 유지를 위해 제외
        df_update = df_update.dropna(subset=['order_id'])
        df_update['order_id'] = df_update['order_id'].astype(int)

        # CSV 저장 (옵션)
        if save_csv:
            csv_path = Path(save_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"CSV 저장: {csv_path} ({len(df):,}행)")

        # Upsert
        result = upsert_df(df_update, "review_analysis", dataset=self.dataset)
        print(f"review_analysis {len(df_update):,}행 업데이트 완료")

        return len(df_update)

    def update_sentiment_batch(
        self,
        results: List[dict]
    ) -> int:
        """
        배치로 감성 분석 결과 업데이트

        Args:
            results: [{"order_id": 1, "sentiment": "positive", "sentiment_score": 0.95}, ...]

        Returns:
            업데이트된 행 수
        """
        df = pd.DataFrame(results)
        return self.update_review_analysis(df)

    def get_analysis_stats(self) -> dict:
        """분석 현황 통계 조회"""
        sql = f"""
        SELECT
            COUNT(*) as total,
            COUNTIF(sentiment IS NOT NULL) as analyzed,
            COUNTIF(sentiment IS NULL) as unanalyzed,
            COUNTIF(sentiment = 'positive') as positive,
            COUNTIF(sentiment = 'negative') as negative,
            COUNTIF(sentiment = 'neutral') as neutral
        FROM `{self.dataset}.review_analysis`
        """
        df = query_to_df(sql)
        stats = df.iloc[0].to_dict()

        # 비율 계산
        total = stats['total']
        if total > 0:
            stats['analyzed_pct'] = stats['analyzed'] / total * 100
            stats['unanalyzed_pct'] = stats['unanalyzed'] / total * 100

        return stats

    def export_for_training(
        self,
        output_path: str,
        limit: Optional[int] = None
    ) -> str:
        """
        학습용 데이터 CSV 내보내기

        Args:
            output_path: 출력 CSV 경로
            limit: 최대 행 수

        Returns:
            저장된 파일 경로
        """
        df = self.load_reviews(limit=limit, min_length=10)

        # 학습에 필요한 컬럼만 선택
        train_cols = ['order_id', 'product_code', 'text', 'rating']
        df_train = df[train_cols].copy()

        df_train.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"학습 데이터 저장: {output_path} ({len(df_train):,}행)")

        return output_path


# ========== 편의 함수 ==========

def load_reviews_from_bq(limit: int = None, exclude_analyzed: bool = False) -> pd.DataFrame:
    """BigQuery에서 리뷰 로드 (간편 함수)"""
    bq = ABSABigQuery()
    return bq.load_reviews(limit=limit, exclude_analyzed=exclude_analyzed)


def save_analysis_to_bq(df: pd.DataFrame) -> int:
    """분석 결과를 BigQuery에 저장 (간편 함수)"""
    bq = ABSABigQuery()
    return bq.update_review_analysis(df)


if __name__ == "__main__":
    print("=== ABSA BigQuery 연동 테스트 ===")

    bq = ABSABigQuery()

    # 통계 조회
    stats = bq.get_analysis_stats()
    print(f"\n분석 현황:")
    print(f"  - 전체: {stats['total']:,}개")
    print(f"  - 분석 완료: {stats['analyzed']:,}개 ({stats.get('analyzed_pct', 0):.1f}%)")
    print(f"  - 미분석: {stats['unanalyzed']:,}개 ({stats.get('unanalyzed_pct', 0):.1f}%)")

    if stats['analyzed'] > 0:
        print(f"\n감성 분포:")
        print(f"  - Positive: {stats['positive']:,}개")
        print(f"  - Negative: {stats['negative']:,}개")
        print(f"  - Neutral: {stats['neutral']:,}개")

    # 샘플 로드
    print("\n샘플 리뷰 5개:")
    df_sample = bq.load_reviews(limit=5)
    for _, row in df_sample.iterrows():
        text_preview = row['text'][:50] + "..." if len(row['text']) > 50 else row['text']
        print(f"  [{row['order_id']}] {text_preview}")
