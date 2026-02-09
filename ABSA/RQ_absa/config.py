"""
Configuration for ABSA pipeline
"""
import os
from pathlib import Path

# Project root (Why-pi/ABSA)
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_ROOT / "raw"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
INFERENCE_DATA_DIR = DATA_ROOT / "inference"
CACHE_DIR = DATA_ROOT / "cache"

# Model paths
MODEL_ROOT = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODEL_ROOT / "checkpoints"

# Reviews data path (Why-pi project structure)
REVIEWS_CSV_PATH = PROJECT_ROOT.parent / "data" / "csv" / "reviews.csv"

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, INFERENCE_DATA_DIR,
                  CACHE_DIR, CHECKPOINT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Fixed aspect labels (9 categories)
ASPECT_LABELS = [
    "배송/포장",
    "품질/불량",
    "가격/가성비",
    "사용감/성능",
    "사이즈/호환",
    "디자인",
    "재질/냄새",
    "CS/응대",
    "재구매"
]

# Sentiment labels
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
SENTIMENT_LABEL_TO_ID = {label: idx for idx, label in enumerate(SENTIMENT_LABELS)}
SENTIMENT_ID_TO_LABEL = {idx: label for idx, label in enumerate(SENTIMENT_LABELS)}

# Sampling configuration (3단계 층화 샘플링)
SAMPLING_CONFIG = {
    "target_size": 20000,
    "category_1_column": "category_1",      # 대분류 (스킨케어/메이크업)
    "category_2_column": "category_2",      # 소분류
    "category_1_min_floor": 600,            # 대분류별 최소 보장
    "category_2_min_floor": 200,            # 소분류별 최소 보장
    "exclude_categories": [],                 # 제외할 대분류
    "skip_cat2_categories": ["맨케어"],       # 소분류 쿼터 배정 스킵 (sentiment만 적용)
    "sentiment_distribution": {
        "negative": 0.30,       # 1-2점 → 30%
        "neutral": 0.30,        # 3점 → 30%
        "positive": 0.40        # 4-5점 → 40%
    },
    "random_state": 42
}

# OpenAI configuration
OPENAI_CONFIG = {
    "model": "gpt-4o-mini",
    "temperature": 0.3,
    "max_tokens": 500,
    "rate_limit_rpm": 60,
    "rate_limit_tpm": 90000,
    "retry_max_attempts": 3,
    "retry_backoff_factor": 2.0
}

# Model training configuration
TRAIN_CONFIG = {
    "model_name": "beomi/KcELECTRA-base",  # or "klue/roberta-base"
    "max_length": 128,
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "dropout": 0.1,
    "sentiment_weight": 1.0,
    "aspect_weight": 1.0,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "save_steps": 500,
    "eval_steps": 500,
    "logging_steps": 100,
    "seed": 42,
    # Class imbalance handling
    "use_class_weight": True,       # Use balanced class weights
    "use_focal_loss": False,        # Use focal loss instead of CE (alternative)
    "focal_gamma": 2.0              # Focal loss gamma parameter
}

# Inference configuration
INFERENCE_CONFIG = {
    "batch_size": 128,
    "aspect_threshold": 0.5,
    "ambiguous_sentiment_threshold": 0.6,
    "ambiguous_aspect_range": (0.4, 0.6),
    "num_workers": 4
}

# Data split ratios
SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15
}

# Output schema
OUTPUT_COLUMNS = [
    "year",
    "product_code",
    "sentiment",
    "sentiment_score",
    "aspect_labels",
    "evidence",
    "summary",
    "text",
    "rating",
    "date",
    "user_id",
    "order_id",
    "is_ambiguous"
]

# Validation configuration
VALIDATION_CONFIG = {
    # Valid values
    "valid_sentiments": ["positive", "neutral", "negative"],
    "valid_aspects": ASPECT_LABELS,  # 9개 aspect 사용
    "score_range": (-1.0, 1.0),

    # Negative keywords for risk detection
    "negative_keywords": [
        '별로', '최악', '다시는', '환불', '불친절', '늦', '안좋', '안 좋',
        '실망', '후회', '짜증', '불량', '파손', '망', '싫', '아쉽', '거짓',
        '속았', '사기', '엉망', '쓰레기', '버림', '못씀', '안됨', '고장'
    ],

    # Contrast markers (대비 접속사)
    "contrast_markers": ['지만', '는데', '으나', '나', '만', '그러나', '하지만', '근데'],

    # Risk thresholds
    "long_text_threshold": 50,      # 50자 이상이면 long text
    "low_confidence_range": (0.3, 0.5),  # 낮은 신뢰도 범위

    # Rating-sentiment mapping (for mismatch detection)
    "rating_sentiment_map": {
        1: "negative", 2: "negative",
        3: "neutral",
        4: "positive", 5: "positive"
    },

    # Judge model configuration
    "judge_model": "gpt-4.1-mini",
    "judge_temperature": 0.2,
    "judge_max_tokens": 800,
}

# Validation data paths
VALIDATION_DATA_DIR = DATA_ROOT / "validation"
VALIDATION_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Validation output files
RULE_VALIDATION_OUTPUT = VALIDATION_DATA_DIR / "rule_validation.jsonl"
RISK_CASES_OUTPUT = VALIDATION_DATA_DIR / "risk_cases.jsonl"
JUDGE_RESULTS_OUTPUT = VALIDATION_DATA_DIR / "judge_results.jsonl"
VALIDATED_LABELS_OUTPUT = RAW_DATA_DIR / "chatgpt_labels_20k_validated.jsonl"

# Issue types for judge model
ISSUE_TYPES = [
    "wrong_sentiment",      # 감정 오분류
    "wrong_aspect",         # aspect 오분류
    "missing_aspect",       # aspect 누락
    "extra_aspect",         # 불필요한 aspect
    "wrong_evidence",       # 근거 불일치
    "ambiguous",            # 판단 불가
]
