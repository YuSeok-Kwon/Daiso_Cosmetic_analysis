# Aspect-Based Sentiment Analysis (ABSA)

Complete implementation of aspect-based sentiment analysis for 312k product reviews.

## Overview

This module provides:
- **Sentiment classification**: positive/neutral/negative
- **Aspect detection**: 9 fixed aspect categories (배송/포장, 품질/불량, etc.)
- **Sentiment scoring**: -1.0 to 1.0 scale
- **Evidence extraction**: Supporting text from reviews
- **Summary generation**: 1-sentence summary per review

## Quick Start

### 1. Install Dependencies

```bash
pip install torch>=2.0.0
pip install transformers>=4.36.0
pip install openai>=1.12.0
pip install accelerate>=0.25.0
pip install scikit-learn
pip install pandas numpy tqdm
```

### 2. Set OpenAI API Key

```bash
export OPENAI_API_KEY="your-key-here"
```

### 3. Run Pipeline

```bash
# Step A: Sample 20k reviews (5 minutes)
python scripts/absa/step_a_sampling.py

# Step B: Label with ChatGPT (2-4 hours, ~$3-5)
python scripts/absa/step_b_labeling.py

# Step C: Create train/val/test sets (2 minutes)
python scripts/absa/step_c_create_dataset.py

# Step D: Train model (2-3 hours, GPU required)
python scripts/absa/step_d_train.py

# Step E: Inference on 312k reviews (15-20 minutes, GPU)
python scripts/absa/step_e_inference.py
```

## Pipeline Steps

### Step A: Stratified Sampling

Samples 20k reviews from 312k with stratification by:
- Year
- Brand
- Category
- Rating group

Adjusts for rating imbalance (81.7% are 5-star reviews).

**Input**: `data/csv/reviews.csv`
**Output**: `data/absa/raw/sampled_reviews_20k.csv`

### Step B: ChatGPT Labeling

Labels 20k reviews using GPT-4o-mini with:
- Sentiment (positive/neutral/negative)
- Sentiment score (-1.0 to 1.0)
- Aspect labels (multi-label from 9 categories)
- Evidence (supporting text)
- Summary (1-sentence)

Features:
- Rate limiting (60 req/min, 90k tokens/min)
- Disk caching (prevents duplicate requests)
- Cost tracking
- Resumable (uses JSONL format)

**Input**: `data/absa/raw/sampled_reviews_20k.csv`
**Output**: `data/absa/raw/chatgpt_labels_20k.jsonl`
**Cost**: ~$3-5 for 20k reviews

### Step C: Dataset Creation

Creates train/val/test splits (70/15/15) with:
- Sentiment label encoding
- Aspect multi-label binary encoding
- Stratified splitting by sentiment

**Input**: `data/absa/raw/chatgpt_labels_20k.jsonl`
**Output**:
- `data/absa/processed/train.csv` (~14k)
- `data/absa/processed/val.csv` (~3k)
- `data/absa/processed/test.csv` (~3k)

### Step D: Model Training

Trains multi-task ABSA model:

**Architecture**:
- Base: KcELECTRA-base (110M params)
- Sentiment head: 3-class classifier
- Aspect head: 9-class multi-label classifier

**Loss**:
- Sentiment: CrossEntropyLoss
- Aspect: BCEWithLogitsLoss
- Combined: weighted sum

**Hyperparameters**:
- Batch size: 32
- Epochs: 10
- Learning rate: 2e-5
- Max length: 128 tokens

**Input**: train.csv, val.csv
**Output**: `models/absa/checkpoints/best_model.pt`
**Time**: 2-3 hours on GPU

### Step E: Full Inference

Runs trained model on all 312k reviews:
- Batch size: 128
- Aspect threshold: 0.5
- Identifies ambiguous samples (low confidence)

**Input**: `data/csv/reviews.csv` (312k)
**Output**:
- `data/absa/inference/reviews_with_absa.csv` (all reviews)
- `data/absa/inference/reviews_with_absa_ambiguous.csv` (ambiguous only)

**Time**: 15-20 minutes on GPU

## Fixed Aspect Labels (9 categories)

1. **배송/포장**: Shipping, packaging
2. **품질/불량**: Quality, defects
3. **가격/가성비**: Price, value
4. **사용감/성능**: Usability, performance
5. **사이즈/호환**: Size, compatibility
6. **디자인**: Design, appearance
7. **재질/냄새**: Material, smell
8. **CS/응대**: Customer service
9. **재구매**: Repurchase intent

## Output Schema

Final CSV columns:

| Column | Type | Description |
|--------|------|-------------|
| year | int | Review year (2024, 2025, 2026) |
| product_code | int | Product identifier |
| sentiment | str | positive/neutral/negative |
| sentiment_score | float | -1.0 to 1.0 |
| aspect_labels | list | ["배송/포장", "품질/불량", ...] |
| evidence | str | Supporting text from review |
| summary | str | 1-sentence summary |
| text | str | Original review text |
| rating | int | 1-5 stars |
| date | str | Review date |
| user_id | int | User identifier |
| order_id | int | Order identifier |
| is_ambiguous | bool | Whether sample is ambiguous |

## Evaluation

Evaluate model on test set:

```bash
python scripts/absa/evaluate_test.py
```

Metrics:
- **Sentiment**: Accuracy, Precision, Recall, F1 (macro/weighted/per-class)
- **Aspect**: Micro/Macro F1, Precision, Recall, Per-aspect F1

## Module Structure

```
RQ/absa/
├── __init__.py
├── config.py          # Configuration
├── sampling.py        # Stratified sampling
├── labeling.py        # ChatGPT labeling
├── dataset.py         # Dataset preparation
├── model.py           # Multi-task ABSA model
├── train.py           # Training pipeline
├── inference.py       # Inference pipeline
├── evaluation.py      # Evaluation metrics
└── README.md          # This file

Crawling/modules/
└── openai_client.py   # OpenAI API client

scripts/absa/
├── step_a_sampling.py
├── step_b_labeling.py
├── step_c_create_dataset.py
├── step_d_train.py
├── step_e_inference.py
└── evaluate_test.py

data/absa/
├── raw/               # Sampled and labeled data
├── processed/         # Train/val/test splits
├── inference/         # Final results
└── cache/             # API cache

models/absa/
└── checkpoints/       # Model checkpoints
```

## Cost Estimation

- ChatGPT labeling (20k): **$3-5**
- GPU training: **$1-2** (cloud) or free (local)
- GPU inference: **$0.50**
- **Total: ~$5-8**

## Time Estimation

- Step A (sampling): 5 minutes
- Step B (labeling): 2-4 hours
- Step C (dataset): 2 minutes
- Step D (training): 2-3 hours
- Step E (inference): 15-20 minutes
- **Total: ~5-8 hours**

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.36+
- OpenAI API key
- GPU with 16GB+ VRAM (for training)
- 8GB+ RAM

## Usage Examples

### Load Results

```python
import pandas as pd

# Load full results
df = pd.read_csv('data/absa/inference/reviews_with_absa.csv')

# Check sentiment distribution
print(df['sentiment'].value_counts())

# Filter positive reviews
positive_reviews = df[df['sentiment'] == 'positive']

# Find reviews about specific aspect
shipping_reviews = df[df['aspect_labels'].str.contains('배송/포장')]

# Load ambiguous samples
ambiguous_df = pd.read_csv('data/absa/inference/reviews_with_absa_ambiguous.csv')
```

### Use Model for Custom Inference

```python
from transformers import AutoTokenizer
from RQ.absa.model import load_model
from RQ.absa.inference import ABSAInference

# Load model
tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
model = load_model("models/absa/checkpoints/best_model.pt")

# Create inference pipeline
inference = ABSAInference(model=model, tokenizer=tokenizer)

# Predict
texts = ["배송이 빠르고 품질도 좋아요", "가격이 비싸고 불량이 있어요"]
predictions = inference.predict_batch(texts)

print(predictions['sentiment_preds'])  # [2, 0] (positive, negative)
print(predictions['aspect_preds'])     # Binary vectors for aspects
```

## Troubleshooting

### Out of Memory during Training

Reduce batch size in `config.py`:
```python
TRAIN_CONFIG['batch_size'] = 16  # or 8
```

### OpenAI Rate Limits

Adjust in `config.py`:
```python
OPENAI_CONFIG['rate_limit_rpm'] = 30  # Reduce to 30 req/min
```

### Slow Inference

Increase batch size if you have more GPU memory:
```python
INFERENCE_CONFIG['batch_size'] = 256  # or higher
```

## Citation

If you use this code, please cite:

```
Aspect-Based Sentiment Analysis for Korean Product Reviews
Implementation based on KcELECTRA and GPT-4o-mini
2026
```

## License

MIT License
