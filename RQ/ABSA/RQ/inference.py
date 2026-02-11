"""
Inference pipeline for ABSA model
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

from RQ_absa.model import MultiTaskABSAModel
from RQ_absa.config import ASPECT_LABELS, SENTIMENT_ID_TO_LABEL


class ABSAInference:
    """
    Inference pipeline for ABSA model.
    """

    def __init__(
        self,
        model: MultiTaskABSAModel,
        tokenizer,
        aspect_labels: List[str] = None,
        sentiment_labels: Dict[int, str] = None,
        device: str = None,
        max_length: int = 128,
        batch_size: int = 128,
        aspect_threshold: float = 0.5,
        ambiguous_sentiment_threshold: float = 0.6,
        ambiguous_aspect_range: tuple = (0.4, 0.6)
    ):
        """
        Args:
            model: Trained ABSA model
            tokenizer: Tokenizer
            aspect_labels: List of aspect label names
            sentiment_labels: Dict mapping sentiment IDs to labels
            device: Device to run inference on
            max_length: Maximum sequence length
            batch_size: Batch size for inference
            aspect_threshold: Threshold for aspect prediction
            ambiguous_sentiment_threshold: Threshold for ambiguous sentiment
            ambiguous_aspect_range: Range for ambiguous aspect probabilities
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.aspect_threshold = aspect_threshold
        self.ambiguous_sentiment_threshold = ambiguous_sentiment_threshold
        self.ambiguous_aspect_range = ambiguous_aspect_range

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        # Labels
        if aspect_labels is None:
            self.aspect_labels = ASPECT_LABELS
        else:
            self.aspect_labels = aspect_labels

        if sentiment_labels is None:
            self.sentiment_labels = SENTIMENT_ID_TO_LABEL
        else:
            self.sentiment_labels = sentiment_labels

        print(f"Inference initialized on device: {self.device}")

    def predict_batch(
        self,
        texts: List[str]
    ) -> Dict:
        """
        Predict sentiment and aspects for a batch of texts.

        Args:
            texts: List of review texts

        Returns:
            Dictionary with predictions
        """
        # Tokenize
        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)

            # Sentiment
            sentiment_probs = torch.softmax(outputs['sentiment_logits'], dim=-1)
            sentiment_preds = torch.argmax(sentiment_probs, dim=-1)
            sentiment_scores = self.model.get_sentiment_score(sentiment_probs)

            # Aspects
            aspect_probs = torch.sigmoid(outputs['aspect_logits'])
            aspect_preds = (aspect_probs >= self.aspect_threshold).long()

            # Confidence
            sentiment_confidence = torch.max(sentiment_probs, dim=-1)[0]

            return {
                'sentiment_preds': sentiment_preds.cpu().numpy(),
                'sentiment_probs': sentiment_probs.cpu().numpy(),
                'sentiment_scores': sentiment_scores.cpu().numpy(),
                'sentiment_confidence': sentiment_confidence.cpu().numpy(),
                'aspect_preds': aspect_preds.cpu().numpy(),
                'aspect_probs': aspect_probs.cpu().numpy()
            }

    def infer_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'text'
    ) -> pd.DataFrame:
        """
        Run inference on a dataframe.

        Args:
            df: Input dataframe
            text_column: Name of text column

        Returns:
            Dataframe with predictions
        """
        print(f"Running inference on {len(df):,} reviews...")

        # Extract texts
        texts = df[text_column].astype(str).tolist()

        # Predict in batches
        all_sentiment_preds = []
        all_sentiment_scores = []
        all_sentiment_confidence = []
        all_aspect_preds = []
        all_aspect_probs = []

        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in tqdm(range(num_batches), desc="Inference"):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]

            # Predict
            predictions = self.predict_batch(batch_texts)

            # Collect
            all_sentiment_preds.extend(predictions['sentiment_preds'])
            all_sentiment_scores.extend(predictions['sentiment_scores'])
            all_sentiment_confidence.extend(predictions['sentiment_confidence'])
            all_aspect_preds.extend(predictions['aspect_preds'])
            all_aspect_probs.extend(predictions['aspect_probs'])

        # Convert to arrays
        all_sentiment_preds = np.array(all_sentiment_preds)
        all_sentiment_scores = np.array(all_sentiment_scores)
        all_sentiment_confidence = np.array(all_sentiment_confidence)
        all_aspect_preds = np.array(all_aspect_preds)
        all_aspect_probs = np.array(all_aspect_probs)

        # Create output dataframe
        output_df = df.copy()

        # Sentiment
        output_df['sentiment'] = [self.sentiment_labels[pred] for pred in all_sentiment_preds]
        output_df['sentiment_score'] = all_sentiment_scores

        # Aspects
        aspect_labels_list = []
        for aspect_pred in all_aspect_preds:
            labels = [self.aspect_labels[i] for i, val in enumerate(aspect_pred) if val == 1]
            aspect_labels_list.append(labels)

        output_df['aspect_labels'] = aspect_labels_list

        # Evidence (placeholder for now)
        output_df['evidence'] = output_df[text_column].apply(
            lambda x: x[:100] + "..." if len(x) > 100 else x
        )

        # Summary (placeholder for now)
        output_df['summary'] = output_df.apply(
            lambda row: self._generate_summary(row['sentiment'], row['aspect_labels']),
            axis=1
        )

        # Identify ambiguous samples
        output_df['is_ambiguous'] = self._identify_ambiguous(
            all_sentiment_confidence,
            all_aspect_probs
        )

        print(f"\nInference complete!")
        print(f"Ambiguous samples: {output_df['is_ambiguous'].sum():,} "
              f"({output_df['is_ambiguous'].sum()/len(output_df)*100:.1f}%)")

        return output_df

    def _generate_summary(self, sentiment: str, aspect_labels: List[str]) -> str:
        """Generate summary (rule-based placeholder)"""
        if len(aspect_labels) == 0:
            return f"전반적으로 {sentiment}"

        aspects_str = ", ".join(aspect_labels[:3])  # Take first 3
        if len(aspect_labels) > 3:
            aspects_str += " 등"

        sentiment_kr = {
            'positive': '긍정적',
            'neutral': '중립적',
            'negative': '부정적'
        }.get(sentiment, sentiment)

        return f"{aspects_str}에 대해 {sentiment_kr}"

    def _identify_ambiguous(
        self,
        sentiment_confidence: np.ndarray,
        aspect_probs: np.ndarray
    ) -> np.ndarray:
        """Identify ambiguous samples"""
        # Low sentiment confidence
        low_sentiment_confidence = sentiment_confidence < self.ambiguous_sentiment_threshold

        # Aspect probabilities in ambiguous range
        aspect_in_range = (
            (aspect_probs >= self.ambiguous_aspect_range[0]) &
            (aspect_probs <= self.ambiguous_aspect_range[1])
        ).any(axis=1)

        # Combine conditions
        is_ambiguous = low_sentiment_confidence | aspect_in_range

        return is_ambiguous

    def save_results(
        self,
        df: pd.DataFrame,
        output_path: Path,
        save_ambiguous: bool = True,
        ambiguous_path: Path = None
    ):
        """
        Save inference results.

        Args:
            df: Dataframe with predictions
            output_path: Path to save full results
            save_ambiguous: Whether to save ambiguous samples separately
            ambiguous_path: Path to save ambiguous samples
        """
        # Save full results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Saved full results to: {output_path}")

        # Save ambiguous samples
        if save_ambiguous:
            ambiguous_df = df[df['is_ambiguous']].copy()
            if len(ambiguous_df) > 0:
                if ambiguous_path is None:
                    ambiguous_path = output_path.parent / f"{output_path.stem}_ambiguous.csv"

                ambiguous_df.to_csv(ambiguous_path, index=False, encoding='utf-8-sig')
                print(f"Saved ambiguous samples to: {ambiguous_path}")
                print(f"  Count: {len(ambiguous_df):,}")


def run_inference_on_reviews(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    model_name: str = "beomi/KcELECTRA-base",
    batch_size: int = 128,
    aspect_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Run inference on reviews CSV.

    Args:
        input_path: Path to input CSV
        output_path: Path to output CSV
        model_path: Path to model checkpoint
        model_name: Pretrained model name
        batch_size: Batch size
        aspect_threshold: Aspect threshold

    Returns:
        Dataframe with predictions
    """
    from transformers import AutoTokenizer
    from RQ_absa.model import load_model

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = load_model(
        checkpoint_path=model_path,
        model_name=model_name
    )

    print("\nLoading reviews...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} reviews")

    # Create inference pipeline
    inference = ABSAInference(
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        aspect_threshold=aspect_threshold
    )

    # Run inference
    results_df = inference.infer_dataframe(df)

    # Save results
    inference.save_results(results_df, output_path)

    # Print statistics
    print("\n" + "="*60)
    print("INFERENCE STATISTICS")
    print("="*60)
    print("\nSentiment distribution:")
    print(results_df['sentiment'].value_counts(normalize=True).sort_index())

    print("\nAspect frequency:")
    all_aspects = []
    for aspects in results_df['aspect_labels']:
        all_aspects.extend(aspects)
    aspect_counts = pd.Series(all_aspects).value_counts()
    for aspect, count in aspect_counts.items():
        print(f"  {aspect}: {count:,} ({count/len(results_df)*100:.1f}%)")

    print("\nAspects per review:")
    results_df['num_aspects'] = results_df['aspect_labels'].apply(len)
    print(results_df['num_aspects'].describe())

    print("="*60)

    return results_df
