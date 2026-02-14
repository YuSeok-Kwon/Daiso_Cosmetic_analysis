"""
Batch labeling with ChatGPT for ABSA
"""
import json
import pandas as pd
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai_client import OpenAIClient


class ABSALabeler:
    """
    Batch labeler for ABSA using ChatGPT.
    Supports incremental saving and resumption.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        save_interval: int = 100
    ):
        """
        Args:
            model: OpenAI model to use
            save_interval: Save cache every N requests
        """
        self.model = model
        self.save_interval = save_interval
        self.client = OpenAIClient()

    def label_batch(
        self,
        input_path: Path,
        output_path: Path,
        resume: bool = True
    ) -> pd.DataFrame:
        """
        Label a batch of reviews.

        Args:
            input_path: Path to input CSV
            output_path: Path to output JSONL
            resume: Whether to resume from existing output

        Returns:
            Dataframe with labeled reviews
        """
        # Load input
        print(f"Loading reviews from: {input_path}")
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df):,} reviews")

        # Check required columns
        required_columns = ['text', 'product_code', 'rating']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Load existing results if resuming
        existing_results = {}
        if resume and output_path.exists():
            print(f"Resuming from: {output_path}")
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    result = json.loads(line)
                    # Use index as key (assuming first field is index)
                    if 'index' in result:
                        existing_results[result['index']] = result

            print(f"Found {len(existing_results):,} existing results")

        # Prepare output file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = 'a' if resume and output_path.exists() else 'w'

        # Label reviews
        print(f"\nStarting labeling with model: {self.model}")
        print(f"Save interval: {self.save_interval}")
        print(f"Estimated cost: ${len(df) * 0.0003:.2f} - ${len(df) * 0.0005:.2f}\n")

        labeled_count = 0
        skipped_count = 0
        error_count = 0

        with open(output_path, mode, encoding='utf-8') as f:
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Labeling"):
                # Skip if already labeled
                if idx in existing_results:
                    skipped_count += 1
                    continue

                try:
                    # Label review (name, category 정보 전달)
                    name = row.get('name', '') if 'name' in df.columns else ''
                    category_1 = row.get('category_1', '') if 'category_1' in df.columns else ''
                    category_2 = row.get('category_2', '') if 'category_2' in df.columns else ''
                    result = self.client.label_review(
                        review_text=row['text'],
                        product_code=row['product_code'],
                        rating=row['rating'],
                        name=name,
                        category_1=category_1,
                        category_2=category_2,
                        model=self.model
                    )

                    # Create output record
                    output_record = {
                        'index': idx,
                        'text': row['text'],
                        'product_code': row['product_code'],
                        'rating': row['rating'],
                        'sentiment': result.sentiment,
                        'sentiment_score': result.sentiment_score,
                        'aspect_labels': result.aspect_labels,
                        'evidence': result.evidence,
                        'summary': result.summary,
                        'model': result.model,
                        'tokens_input': result.tokens_input,
                        'tokens_output': result.tokens_output,
                        'cost': result.cost
                    }

                    # Copy other columns if present
                    for col in df.columns:
                        if col not in output_record:
                            output_record[col] = row[col]

                    # Write to file
                    f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                    f.flush()

                    labeled_count += 1

                    # Periodic status update
                    if labeled_count % self.save_interval == 0:
                        total_cost = self.client.get_total_cost()
                        total_requests = self.client.get_total_requests()
                        avg_cost = total_cost / total_requests if total_requests > 0 else 0
                        print(f"\n[Progress] Labeled: {labeled_count:,}, "
                              f"Skipped: {skipped_count:,}, "
                              f"Errors: {error_count:,}")
                        print(f"[Cost] Total: ${total_cost:.2f}, "
                              f"Avg: ${avg_cost:.4f}/request")
                        self.client.save_cache()

                except Exception as e:
                    print(f"\nError labeling review {idx}: {e}")
                    error_count += 1
                    continue

        # Final statistics
        total_cost = self.client.get_total_cost()
        total_requests = self.client.get_total_requests()

        print("\n" + "="*60)
        print("LABELING COMPLETE")
        print("="*60)
        print(f"Total labeled: {labeled_count:,}")
        print(f"Skipped (cached): {skipped_count:,}")
        print(f"Errors: {error_count:,}")
        print(f"Total cost: ${total_cost:.2f}")
        print(f"Total requests: {total_requests:,}")
        print(f"Average cost per request: ${total_cost/total_requests:.4f}")
        print(f"Output saved to: {output_path}")
        print("="*60)

        # Load and return results
        results = []
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                results.append(json.loads(line))

        results_df = pd.DataFrame(results)

        # Validate results
        self._validate_results(results_df)

        return results_df

    def _validate_results(self, df: pd.DataFrame):
        """Validate labeling results"""
        print("\n" + "="*60)
        print("LABELING VALIDATION")
        print("="*60)

        # Sentiment distribution
        print("\nSentiment distribution:")
        sentiment_dist = df['sentiment'].value_counts(normalize=True).sort_index()
        for sentiment, ratio in sentiment_dist.items():
            print(f"  {sentiment}: {ratio*100:.1f}%")

        # Sentiment score distribution
        print("\nSentiment score statistics:")
        print(df['sentiment_score'].describe())

        # Aspect distribution
        print("\nAspect frequency:")
        all_aspects = []
        for aspects in df['aspect_labels']:
            if isinstance(aspects, list):
                all_aspects.extend(aspects)

        aspect_counts = pd.Series(all_aspects).value_counts()
        for aspect, count in aspect_counts.items():
            print(f"  {aspect}: {count:,} ({count/len(df)*100:.1f}%)")

        # Aspects per review
        df['num_aspects'] = df['aspect_labels'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        print("\nAspects per review:")
        print(df['num_aspects'].describe())

        # Token statistics
        print("\nToken usage:")
        print(f"  Total input tokens: {df['tokens_input'].sum():,}")
        print(f"  Total output tokens: {df['tokens_output'].sum():,}")
        print(f"  Avg input tokens: {df['tokens_input'].mean():.1f}")
        print(f"  Avg output tokens: {df['tokens_output'].mean():.1f}")

        # Cost statistics
        print("\nCost statistics:")
        print(f"  Total cost: ${df['cost'].sum():.2f}")
        print(f"  Avg cost per review: ${df['cost'].mean():.4f}")

        # Sentiment-rating consistency
        print("\nSentiment-rating consistency:")
        rating_sentiment_map = {
            1: 'negative', 2: 'negative',
            3: 'neutral',
            4: 'positive', 5: 'positive'
        }
        df['expected_sentiment'] = df['rating'].map(rating_sentiment_map)
        consistency = (df['sentiment'] == df['expected_sentiment']).mean()
        print(f"  Consistency rate: {consistency*100:.1f}%")

        print("="*60)


def label_from_bigquery(
    output_path: Path,
    model: str = "gpt-4o-mini",
    limit: int = None,
    save_to_bq: bool = True,
    save_csv: str = None,
    resume: bool = True
) -> pd.DataFrame:
    """
    BigQuery에서 리뷰를 로드하여 라벨링 후 결과 저장

    Args:
        output_path: 결과 JSONL 저장 경로
        model: OpenAI 모델명
        limit: 최대 라벨링 리뷰 수
        save_to_bq: BigQuery에 결과 저장 여부
        save_csv: CSV 저장 경로 (옵션)
        resume: 이전 작업 이어서 진행 여부

    Returns:
        라벨링 결과 DataFrame
    """
    try:
        from bq_connector import ABSABigQuery
    except ImportError:
        print("Error: bq_connector 모듈을 찾을 수 없습니다.")
        return None

    # BigQuery 연결
    bq = ABSABigQuery()

    # 미분석 리뷰 로드
    print("BigQuery에서 미분석 리뷰 로드 중...")
    df = bq.load_unanalyzed_reviews(limit=limit)

    if len(df) == 0:
        print("라벨링할 리뷰가 없습니다.")
        return pd.DataFrame()

    print(f"총 {len(df):,}개 리뷰 로드 완료")

    # 임시 CSV 저장
    temp_csv = output_path.parent / f"_temp_reviews_{output_path.stem}.csv"
    df.to_csv(temp_csv, index=False, encoding='utf-8-sig')

    # 라벨링 실행
    labeler = ABSALabeler(model=model)
    results_df = labeler.label_batch(temp_csv, output_path, resume=resume)

    # 임시 파일 삭제
    temp_csv.unlink(missing_ok=True)

    # CSV 저장
    if save_csv and len(results_df) > 0:
        csv_path = Path(save_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\nCSV 저장: {csv_path} ({len(results_df):,}행)")

    # BigQuery 저장
    if save_to_bq and len(results_df) > 0:
        print("\nBigQuery에 결과 저장 중...")
        bq.update_review_analysis(results_df)
        print("저장 완료!")

    return results_df
