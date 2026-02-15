"""
Dataset preparation for ABSA model training
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset


class ABSADataProcessor:
    """
    Process labeled data for ABSA model training.
    """

    def __init__(
        self,
        sentiment_labels: List[str] = None,
        aspect_labels: List[str] = None
    ):
        """
        Args:
            sentiment_labels: List of sentiment labels
            aspect_labels: List of aspect labels
        """
        if sentiment_labels is None:
            self.sentiment_labels = ["negative", "neutral", "positive"]
        else:
            self.sentiment_labels = sentiment_labels

        if aspect_labels is None:
            from RQ_absa.config import ASPECT_LABELS
            self.aspect_labels = ASPECT_LABELS
        else:
            self.aspect_labels = aspect_labels

        # Create label mappings
        self.sentiment_to_id = {label: idx for idx, label in enumerate(self.sentiment_labels)}
        self.id_to_sentiment = {idx: label for idx, label in enumerate(self.sentiment_labels)}

        self.aspect_to_id = {label: idx for idx, label in enumerate(self.aspect_labels)}
        self.id_to_aspect = {idx: label for idx, label in enumerate(self.aspect_labels)}

    def load_labeled_data(self, jsonl_path: Path) -> pd.DataFrame:
        """Load labeled data from JSONL"""
        print(f"Loading labeled data from: {jsonl_path}")

        data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        df = pd.DataFrame(data)
        print(f"Loaded {len(df):,} labeled reviews")

        return df

    def encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode sentiment and aspect labels.

        Args:
            df: Dataframe with 'sentiment' and 'aspect_labels' columns

        Returns:
            Dataframe with encoded labels
        """
        df = df.copy()

        # Encode sentiment (single label)
        df['sentiment_id'] = df['sentiment'].map(self.sentiment_to_id)

        # Check for missing sentiment mappings
        missing_sentiment = df['sentiment_id'].isna().sum()
        if missing_sentiment > 0:
            print(f"Warning: {missing_sentiment} reviews have unmapped sentiment labels")
            df = df.dropna(subset=['sentiment_id'])

        df['sentiment_id'] = df['sentiment_id'].astype(int)

        # Encode aspects (multi-label)
        aspect_vectors = []
        for aspects in df['aspect_labels']:
            # Create binary vector
            vector = [0] * len(self.aspect_labels)
            if isinstance(aspects, list):
                for aspect in aspects:
                    if aspect in self.aspect_to_id:
                        idx = self.aspect_to_id[aspect]
                        vector[idx] = 1
            aspect_vectors.append(vector)

        df['aspect_vector'] = aspect_vectors

        print(f"\nEncoded {len(df):,} reviews")
        print(f"Sentiment labels: {self.sentiment_labels}")
        print(f"Aspect labels: {len(self.aspect_labels)}")

        return df

    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test sets with stratification.

        Args:
            df: Input dataframe
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_state: Random seed

        Returns:
            (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(1 - train_ratio),
            stratify=df['sentiment_id'],
            random_state=random_state
        )

        # Second split: val vs test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_ratio_adjusted),
            stratify=temp_df['sentiment_id'],
            random_state=random_state
        )

        print("\n" + "="*60)
        print("DATA SPLIT")
        print("="*60)
        print(f"Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"Val:   {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
        print(f"Test:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
        print("="*60)

        # Validate sentiment distribution
        self._validate_split(train_df, val_df, test_df)

        return train_df, val_df, test_df

    def _validate_split(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Validate train/val/test split"""
        print("\nSentiment distribution:")
        print("\nTrain:")
        print(train_df['sentiment'].value_counts(normalize=True).sort_index())
        print("\nVal:")
        print(val_df['sentiment'].value_counts(normalize=True).sort_index())
        print("\nTest:")
        print(test_df['sentiment'].value_counts(normalize=True).sort_index())

    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: Path
    ):
        """Save train/val/test splits to CSV"""
        output_dir.mkdir(parents=True, exist_ok=True)

        train_path = output_dir / "train.csv"
        val_path = output_dir / "val.csv"
        test_path = output_dir / "test.csv"

        train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
        val_df.to_csv(val_path, index=False, encoding='utf-8-sig')
        test_df.to_csv(test_path, index=False, encoding='utf-8-sig')

        print(f"\nSaved splits to: {output_dir}")
        print(f"  Train: {train_path}")
        print(f"  Val:   {val_path}")
        print(f"  Test:  {test_path}")


class ABSADataset(Dataset):
    """
    PyTorch Dataset for ABSA model.
    """

    def __init__(
        self,
        texts: List[str],
        sentiment_labels: List[int],
        aspect_labels: List[List[int]],
        tokenizer,
        max_length: int = 128
    ):
        """
        Args:
            texts: List of review texts
            sentiment_labels: List of sentiment label IDs
            aspect_labels: List of aspect binary vectors
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.sentiment_labels = sentiment_labels
        self.aspect_labels = aspect_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        sentiment_label = self.sentiment_labels[idx]
        aspect_label = self.aspect_labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'sentiment_label': torch.tensor(sentiment_label, dtype=torch.long),
            'aspect_label': torch.tensor(aspect_label, dtype=torch.float)
        }


def create_datasets_from_csv(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    tokenizer,
    max_length: int = 128
) -> Tuple[ABSADataset, ABSADataset, ABSADataset]:
    """
    Create PyTorch datasets from CSV files.

    Args:
        train_path: Path to train CSV
        val_path: Path to val CSV
        test_path: Path to test CSV
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    # Load CSVs
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Parse aspect_vector (stored as string)
    for df in [train_df, val_df, test_df]:
        df['aspect_vector'] = df['aspect_vector'].apply(eval)

    # Create datasets
    train_dataset = ABSADataset(
        texts=train_df['text'].tolist(),
        sentiment_labels=train_df['sentiment_id'].tolist(),
        aspect_labels=train_df['aspect_vector'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    val_dataset = ABSADataset(
        texts=val_df['text'].tolist(),
        sentiment_labels=val_df['sentiment_id'].tolist(),
        aspect_labels=val_df['aspect_vector'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    test_dataset = ABSADataset(
        texts=test_df['text'].tolist(),
        sentiment_labels=test_df['sentiment_id'].tolist(),
        aspect_labels=test_df['aspect_vector'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    print(f"Created datasets:")
    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Val:   {len(val_dataset):,} samples")
    print(f"  Test:  {len(test_dataset):,} samples")

    return train_dataset, val_dataset, test_dataset
