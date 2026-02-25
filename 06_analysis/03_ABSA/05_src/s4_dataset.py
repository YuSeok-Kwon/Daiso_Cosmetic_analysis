"""
Dataset preparation for ABSA model training (Option A: aspect별 4-class 통합)

리뷰 단위로 그룹화하여 각 aspect별 sentiment를 예측하는 구조.
출력 라벨: [K] (각 값 0~3: none/positive/neutral/negative, K=len(ASPECT_LABELS))
출력 마스크: [K] (1=학습에 포함, 0=loss에서 제외)
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
    absa_analysis_ready.csv (행 단위: review_id, aspect, aspect_sentiment)를
    리뷰 단위로 그룹화하여 11-dim 4-class 라벨을 생성.
    """

    def __init__(
        self,
        sentiment_labels: List[str] = None,
        aspect_labels: List[str] = None,
        aspect_sentiment_labels: List[str] = None
    ):
        if sentiment_labels is None:
            from RQ_absa.s1_config import SENTIMENT_LABELS
            self.sentiment_labels = SENTIMENT_LABELS
        else:
            self.sentiment_labels = sentiment_labels

        if aspect_labels is None:
            from RQ_absa.s1_config import ASPECT_LABELS
            self.aspect_labels = ASPECT_LABELS
        else:
            self.aspect_labels = aspect_labels

        if aspect_sentiment_labels is None:
            from RQ_absa.s1_config import ASPECT_SENTIMENT_LABELS
            self.aspect_sentiment_labels = ASPECT_SENTIMENT_LABELS
        else:
            self.aspect_sentiment_labels = aspect_sentiment_labels

        # 매핑 생성 (config 기준 단일 소스)
        self.sentiment_to_id = {label: idx for idx, label in enumerate(self.sentiment_labels)}
        self.id_to_sentiment = {idx: label for idx, label in enumerate(self.sentiment_labels)}

        self.aspect_to_id = {label: idx for idx, label in enumerate(self.aspect_labels)}
        self.id_to_aspect = {idx: label for idx, label in enumerate(self.aspect_labels)}

        from RQ_absa.s1_config import ASPECT_SENTIMENT_TO_ID
        self.aspect_sentiment_to_id = ASPECT_SENTIMENT_TO_ID

    def load_and_group_csv(self, csv_path: Path) -> pd.DataFrame:
        """
        absa_analysis_ready.csv를 로드하고 리뷰 단위로 그룹화.

        입력 CSV 컬럼: review_id, text, aspect, aspect_sentiment, review_sentiment, ...
        출력 DataFrame: 리뷰당 1행, aspect_vector [11], aspect_mask_vector [11]

        충돌 처리: 동일 (review_id, aspect)에 다른 sentiment → mask=0, label=0
        """
        print(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df):,} rows (aspect-level)")

        # 충돌 탐지: (review_id, aspect)에 sentiment가 다른 경우
        conflict_keys = set()
        dup = df.groupby(["review_id", "aspect"])["aspect_sentiment"].nunique()
        for (rid, asp), cnt in dup.items():
            if cnt > 1:
                conflict_keys.add((rid, asp))
        if conflict_keys:
            print(f"  충돌 감지: {len(conflict_keys)}쌍 (mask=0 처리)")

        # 리뷰 단위로 그룹화
        grouped = df.groupby("review_id")
        reviews = []

        for review_id, group in grouped:
            first_row = group.iloc[0]

            # K-dim 초기화
            aspect_vector = [0] * len(self.aspect_labels)
            aspect_mask = [0] * len(self.aspect_labels)  # 기본: unknown

            # 리뷰 전체 sentiment (mixed는 제거)
            review_sent = str(first_row.get("review_sentiment", "neutral")).strip().lower()
            if review_sent not in self.sentiment_to_id:
                continue  # mixed 등 매핑 불가 → 해당 리뷰 제거
            sentiment_id = self.sentiment_to_id[review_sent]

            # 각 aspect별 sentiment 매핑
            for _, row in group.iterrows():
                aspect_name = row["aspect"]
                aspect_sent = str(row["aspect_sentiment"]).strip().lower()

                if aspect_name in self.aspect_to_id and aspect_sent in self.aspect_sentiment_to_id:
                    aspect_idx = self.aspect_to_id[aspect_name]

                    # 충돌인 경우 → mask=0, label=0
                    if (review_id, aspect_name) in conflict_keys:
                        aspect_vector[aspect_idx] = 0
                        aspect_mask[aspect_idx] = 0
                    else:
                        if aspect_name == "미분류":
                            sent_id = self.aspect_sentiment_to_id["neutral"]
                        else:
                            sent_id = self.aspect_sentiment_to_id[aspect_sent]
                        aspect_vector[aspect_idx] = sent_id
                        aspect_mask[aspect_idx] = 1

            reviews.append({
                "review_id": review_id,
                "text": first_row["text"],
                "sentiment_id": sentiment_id,
                "sentiment": review_sent,
                "aspect_vector": aspect_vector,
                "aspect_mask_vector": aspect_mask,
                "product_code": first_row.get("product_code"),
                "rating": first_row.get("rating"),
                "category_1": first_row.get("category_1"),
                "category_2": first_row.get("category_2"),
            })

        result_df = pd.DataFrame(reviews)
        print(f"Grouped into {len(result_df):,} reviews (from {len(df):,} rows)")

        # 라벨 분포 출력
        self._print_label_distribution(result_df)

        return result_df

    def _print_label_distribution(self, df: pd.DataFrame):
        """라벨 분포 출력"""
        print("\n--- Sentiment 분포 ---")
        print(df["sentiment"].value_counts().sort_index())

        print("\n--- Aspect-Sentiment 분포 ---")
        aspect_vectors = np.array(df["aspect_vector"].tolist())
        for i, aspect_name in enumerate(self.aspect_labels):
            counts = np.bincount(aspect_vectors[:, i], minlength=4)
            total_mentioned = counts[1:].sum()
            print(f"  {aspect_name}: none={counts[0]}, pos={counts[1]}, "
                  f"neu={counts[2]}, neg={counts[3]} (언급률: {total_mentioned/len(df)*100:.1f}%)")

    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Train/Val/Test 분할 (sentiment 기준 층화 샘플링)"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(1 - train_ratio),
            stratify=df["sentiment_id"],
            random_state=random_state
        )

        # Second split: val vs test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_ratio_adjusted),
            stratify=temp_df["sentiment_id"],
            random_state=random_state
        )

        print("\n" + "=" * 60)
        print("DATA SPLIT")
        print("=" * 60)
        print(f"Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"Val:   {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
        print(f"Test:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
        print("=" * 60)

        return train_df, val_df, test_df

    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: Path
    ):
        """Train/Val/Test 분할 저장"""
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            path = output_dir / f"{name}.csv"
            split_df.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"  Saved {name}: {path}")

    # --- 하위 호환용 메서드 (JSONL 기반 기존 파이프라인) ---

    def load_labeled_data(self, jsonl_path: Path) -> pd.DataFrame:
        """Load labeled data from JSONL (기존 호환)"""
        print(f"Loading labeled data from: {jsonl_path}")
        data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        df = pd.DataFrame(data)
        print(f"Loaded {len(df):,} labeled reviews")
        return df

    def encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        JSONL 기반 라벨 인코딩 (기존 호환).
        aspect_labels 컬럼이 리스트인 경우 → 이진 벡터.
        aspect_sentiment 컬럼이 있으면 → 4-class 벡터.
        """
        df = df.copy()

        # sentiment 인코딩
        df["sentiment_id"] = df["sentiment"].map(self.sentiment_to_id)
        missing = df["sentiment_id"].isna().sum()
        if missing > 0:
            print(f"Warning: {missing} reviews have unmapped sentiment labels")
            df = df.dropna(subset=["sentiment_id"])
        df["sentiment_id"] = df["sentiment_id"].astype(int)

        # aspect 인코딩 (4-class 벡터)
        aspect_vectors = []
        for _, row in df.iterrows():
            vector = [0] * len(self.aspect_labels)
            aspects = row.get("aspect_labels", [])
            if isinstance(aspects, list):
                for aspect in aspects:
                    if aspect in self.aspect_to_id:
                        vector[self.aspect_to_id[aspect]] = 1  # 존재만 표시 (호환용)
            aspect_vectors.append(vector)

        df["aspect_vector"] = aspect_vectors

        print(f"\nEncoded {len(df):,} reviews")
        return df


class ABSADataset(Dataset):
    """
    PyTorch Dataset for ABSA model (Option A).

    aspect_label: [K] LongTensor (각 값 0~3: none/pos/neu/neg)
    aspect_mask: [K] FloatTensor (1=학습 포함, 0=loss 제외)
    """

    def __init__(
        self,
        texts: List[str],
        sentiment_labels: List[int],
        aspect_labels: List[List[int]],
        tokenizer,
        max_length: int = 128,
        aspect_masks: List[List[int]] = None
    ):
        self.texts = texts
        self.sentiment_labels = sentiment_labels
        self.aspect_labels = aspect_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        # mask가 없으면 모든 셀 학습 (기존 호환)
        self.aspect_masks = aspect_masks

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        sentiment_label = self.sentiment_labels[idx]
        aspect_label = self.aspect_labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "sentiment_label": torch.tensor(sentiment_label, dtype=torch.long),
            "aspect_label": torch.tensor(aspect_label, dtype=torch.long),  # [11]
        }

        if self.aspect_masks is not None:
            item["aspect_mask"] = torch.tensor(self.aspect_masks[idx], dtype=torch.float)
        else:
            # mask 없으면 전부 1 (모든 셀 학습)
            item["aspect_mask"] = torch.ones(len(aspect_label), dtype=torch.float)

        return item


def create_datasets_from_golden(
    gold_csv_path: Path,
    tokenizer,
    max_length: int = 128,
    finetune_ratio: float = 0.60,
    tune_ratio: float = 0.20,
    test_ratio: float = 0.20,
    random_state: int = 42
) -> Tuple[ABSADataset, ABSADataset, ABSADataset]:
    """
    골든셋 CSV에서 finetune / tune / test 데이터셋 생성.

    역할 분리:
    - finetune: Stage 2 파인튜닝 학습용
    - tune: per-aspect none-threshold 튜닝용
    - test: 최종 검증용 (학습/튜닝에 절대 사용하지 않음)

    골든셋은 absa_analysis_ready.csv와 동일한 스키마.

    Args:
        gold_csv_path: golden_set.csv 경로
        tokenizer: HuggingFace tokenizer
        max_length: 최대 시퀀스 길이
        finetune_ratio: 파인튜닝용 비율
        tune_ratio: threshold 튜닝용 비율
        test_ratio: 최종 검증용 비율
        random_state: 랜덤 시드

    Returns:
        (finetune_dataset, tune_dataset, test_dataset)
    """
    assert abs(finetune_ratio + tune_ratio + test_ratio - 1.0) < 1e-6

    processor = ABSADataProcessor()
    df = processor.load_and_group_csv(gold_csv_path)

    print(f"\n골든셋: {len(df):,} 리뷰")

    # 3-way 분할 (sentiment 기준 층화)
    finetune_df, temp_df = train_test_split(
        df,
        test_size=(1 - finetune_ratio),
        stratify=df["sentiment_id"],
        random_state=random_state
    )

    tune_ratio_adj = tune_ratio / (tune_ratio + test_ratio)
    tune_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - tune_ratio_adj),
        stratify=temp_df["sentiment_id"],
        random_state=random_state
    )

    print(f"  finetune: {len(finetune_df):,} ({len(finetune_df)/len(df)*100:.1f}%)")
    print(f"  tune:     {len(tune_df):,} ({len(tune_df)/len(df)*100:.1f}%)")
    print(f"  test:     {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")

    datasets = []
    has_mask = "aspect_mask_vector" in df.columns
    for name, split_df in [("finetune", finetune_df), ("tune", tune_df), ("test", test_df)]:
        dataset = ABSADataset(
            texts=split_df["text"].tolist(),
            sentiment_labels=split_df["sentiment_id"].tolist(),
            aspect_labels=split_df["aspect_vector"].tolist(),
            tokenizer=tokenizer,
            max_length=max_length,
            aspect_masks=split_df["aspect_mask_vector"].tolist() if has_mask else None
        )
        datasets.append(dataset)

    return datasets[0], datasets[1], datasets[2]


def create_datasets_from_csv(
    csv_path: Path,
    tokenizer,
    max_length: int = 128,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[ABSADataset, ABSADataset, ABSADataset]:
    """
    absa_analysis_ready.csv에서 직접 데이터셋 생성.

    1) CSV 로드 → 리뷰 단위 그룹화
    2) Train/Val/Test 분할
    3) ABSADataset 생성

    Args:
        csv_path: absa_analysis_ready.csv 경로
        tokenizer: HuggingFace tokenizer
        max_length: 최대 시퀀스 길이
        train_ratio/val_ratio/test_ratio: 분할 비율
        random_state: 랜덤 시드

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    processor = ABSADataProcessor()
    df = processor.load_and_group_csv(csv_path)

    train_df, val_df, test_df = processor.split_data(
        df, train_ratio, val_ratio, test_ratio, random_state
    )

    has_mask = "aspect_mask_vector" in df.columns
    datasets = []
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dataset = ABSADataset(
            texts=split_df["text"].tolist(),
            sentiment_labels=split_df["sentiment_id"].tolist(),
            aspect_labels=split_df["aspect_vector"].tolist(),
            tokenizer=tokenizer,
            max_length=max_length,
            aspect_masks=split_df["aspect_mask_vector"].tolist() if has_mask else None
        )
        datasets.append(dataset)
        print(f"  {name}: {len(dataset):,} samples")

    return datasets[0], datasets[1], datasets[2]


def create_datasets_from_splits(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    tokenizer,
    max_length: int = 128
) -> Tuple[ABSADataset, ABSADataset, ABSADataset]:
    """
    사전 분할된 CSV에서 데이터셋 생성 (기존 호환).
    """
    datasets = []

    for name, path in [("Train", train_path), ("Val", val_path), ("Test", test_path)]:
        df = pd.read_csv(path)
        df["aspect_vector"] = df["aspect_vector"].apply(eval)

        dataset = ABSADataset(
            texts=df["text"].tolist(),
            sentiment_labels=df["sentiment_id"].tolist(),
            aspect_labels=df["aspect_vector"].tolist(),
            tokenizer=tokenizer,
            max_length=max_length
        )
        datasets.append(dataset)
        print(f"  {name}: {len(dataset):,} samples")

    return datasets[0], datasets[1], datasets[2]


def _load_wide_csv(wide_csv_path: Path) -> pd.DataFrame:
    """
    Wide format CSV 로드 및 파싱.

    Wide CSV 컬럼 구조:
    - 메타: review_id, product_code, user_id, rating, review_date, ...
    - label_<aspect>: 0=none, 1=positive, 2=neutral, 3=negative
    - mask_<aspect>: 1=학습 포함, 0=loss 제외

    Returns:
        DataFrame with 'text', 'sentiment_id', 'aspect_vector', 'aspect_mask_vector'
    """
    from RQ_absa.s1_config import ASPECT_LABELS, SENTIMENT_LABEL_TO_ID

    df = pd.read_csv(wide_csv_path)
    print(f"Loaded wide CSV: {len(df):,} reviews from {wide_csv_path}")

    # aspect 컬럼명 생성 (/ → _)
    aspect_cols = [a.replace('/', '_') for a in ASPECT_LABELS]

    # aspect_vector, aspect_mask_vector 생성
    aspect_vectors = []
    aspect_masks = []
    for _, row in df.iterrows():
        labels = [0 if pd.isna(row.get(f'label_{ac}', 0)) else int(row[f'label_{ac}']) for ac in aspect_cols]
        masks = [0 if pd.isna(row.get(f'mask_{ac}', 0)) else int(row[f'mask_{ac}']) for ac in aspect_cols]
        aspect_vectors.append(labels)
        aspect_masks.append(masks)

    df['aspect_vector'] = aspect_vectors
    df['aspect_mask_vector'] = aspect_masks

    # review_sentiment → sentiment_id
    sent_col = 'review_sentiment'
    if sent_col in df.columns:
        df['sentiment_id'] = df[sent_col].map(SENTIMENT_LABEL_TO_ID)
        invalid = df['sentiment_id'].isna().sum()
        if invalid > 0:
            print(f"  Warning: {invalid} reviews have unmapped sentiment, dropping")
            df = df.dropna(subset=['sentiment_id'])
        df['sentiment_id'] = df['sentiment_id'].astype(int)
    else:
        print(f"  Warning: '{sent_col}' column not found, defaulting to neutral(1)")
        df['sentiment_id'] = 1

    return df


def create_dataset_from_wide(
    wide_csv_path: Path,
    tokenizer,
    max_length: int = 128,
) -> ABSADataset:
    """
    Wide format CSV에서 단일 ABSADataset 생성 (mask 포함).
    분할은 외부에서 처리.
    """
    df = _load_wide_csv(wide_csv_path)

    # mask 통계
    masks_arr = np.array(df['aspect_mask_vector'].tolist())
    total_cells = masks_arr.size
    mask1_cells = masks_arr.sum()
    print(f"  mask=1: {int(mask1_cells)}/{total_cells} ({mask1_cells/total_cells*100:.1f}%)")

    dataset = ABSADataset(
        texts=df['text'].astype(str).tolist(),
        sentiment_labels=df['sentiment_id'].tolist(),
        aspect_labels=df['aspect_vector'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
        aspect_masks=df['aspect_mask_vector'].tolist()
    )
    print(f"  Dataset: {len(dataset):,} samples (with mask)")
    return dataset


def create_datasets_from_wide(
    wide_csv_path: Path,
    tokenizer,
    max_length: int = 128,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[ABSADataset, ABSADataset, ABSADataset]:
    """
    Wide format CSV에서 Train/Val/Test 데이터셋 생성 (mask 포함).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    df = _load_wide_csv(wide_csv_path)

    # 층화 분할 (sentiment 기준)
    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_ratio),
        stratify=df['sentiment_id'], random_state=random_state
    )
    val_adj = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_adj),
        stratify=temp_df['sentiment_id'], random_state=random_state
    )

    print(f"\nWide CSV split:")
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    datasets = []
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dataset = ABSADataset(
            texts=split_df['text'].astype(str).tolist(),
            sentiment_labels=split_df['sentiment_id'].tolist(),
            aspect_labels=split_df['aspect_vector'].tolist(),
            tokenizer=tokenizer,
            max_length=max_length,
            aspect_masks=split_df['aspect_mask_vector'].tolist()
        )
        datasets.append(dataset)

        masks = np.array(split_df['aspect_mask_vector'].tolist())
        print(f"  {name}: {len(dataset):,} samples, "
              f"mask=1: {int(masks.sum())}/{masks.size} ({masks.sum()/masks.size*100:.1f}%)")

    return datasets[0], datasets[1], datasets[2]
