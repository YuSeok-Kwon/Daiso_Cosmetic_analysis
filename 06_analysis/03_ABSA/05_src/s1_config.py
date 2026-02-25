"""
Configuration for ABSA pipeline
"""
import os
from pathlib import Path

# Project root (Why-pi/ABSA)
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths (번호 디렉토리 기준)
RAW_DATA_DIR = PROJECT_ROOT / "01_raw_data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "02_processed_data"
INFERENCE_DATA_DIR = PROJECT_ROOT / "04_outputs" / "inference"
CACHE_DIR = PROJECT_ROOT / "04_outputs" / "cache"
LOG_DIR = PROJECT_ROOT / "04_outputs" / "logs"

# Model paths
MODEL_ROOT = PROJECT_ROOT / "07_models"
CHECKPOINT_DIR = MODEL_ROOT / "checkpoints"

# Reviews data path (Why-pi project structure)
REVIEWS_CSV_PATH = PROJECT_ROOT.parent.parent / "Data" / "csv" / "reviews.csv"

# Ensure directories exist
for directory in [INFERENCE_DATA_DIR, CACHE_DIR, LOG_DIR, CHECKPOINT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Fixed aspect labels (8 categories, Stage 2: 미분류·CS/응대·품질/퀄리티 제거)
ASPECT_LABELS = [
    "배송/포장",
    "가격/가성비",
    "사용감/성능",
    "용량/휴대",
    "디자인",
    "재질/냄새",
    "재구매",
    "색상/발색",
]

# Sentiment labels
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
SENTIMENT_LABEL_TO_ID = {label: idx for idx, label in enumerate(SENTIMENT_LABELS)}
SENTIMENT_ID_TO_LABEL = {idx: label for idx, label in enumerate(SENTIMENT_LABELS)}

# Aspect-Sentiment labels (Option A: aspect별 4-class 통합)
# none=해당 aspect 미존재, positive/neutral/negative=해당 감성
# Wide CSV 인코딩과 일치: 0=none, 1=positive, 2=neutral, 3=negative
ASPECT_SENTIMENT_LABELS = ["none", "positive", "neutral", "negative"]
ASPECT_SENTIMENT_TO_ID = {label: idx for idx, label in enumerate(ASPECT_SENTIMENT_LABELS)}
ASPECT_SENTIMENT_ID_TO_LABEL = {idx: label for idx, label in enumerate(ASPECT_SENTIMENT_LABELS)}
NUM_ASPECT_SENTIMENT_CLASSES = len(ASPECT_SENTIMENT_LABELS)  # 4

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
    "ambiguous_sentiment_threshold": 0.6,
    "num_workers": 4
}

# Keyword gate configuration (post-processing: precision 보호)
# 모델이 non-none으로 예측했더라도 키워드 미포함 시 none으로 override
KEYWORD_GATE_CONFIG = {
    "디자인": [
        # Tier1: 구조물/용기류 (41.5% coverage, 1.5% FP)
        "용기", "뚜껑", "케이스", "패키지", "디자인", "스포이드", "펌프", "튜브",
        # Tier2: 외관 표현 (+10%p coverage, +2.2%p FP)
        "예쁘", "이쁘", "귀엽", "귀여", "깔끔", "고급", "앙증",
        # 미커버 리뷰에서 추가 발굴 (구조물/마감 관련)
        "팁", "입구", "마감",
    ],
    "가격/가성비": [
        "가격", "저렴", "가성비", "최저가", "비싸", "싸", "저가",
        "천원", "원짜리", "돈", "값", "가격대", "싸게", "만원",
    ],
    "재질/냄새": [
        "향", "냄새", "무향", "알코올", "역함", "역겨",
        "향기", "방향", "자극", "쏘", "독함", "불쾌", "악취", "에탄올",
    ],
    "용량/휴대": [
        "양", "용량", "사이즈", "작다", "크다", "작아", "작은",
        "ml", "g", "휴대", "여행", "오래", "적다", "많다",
        "넉넉", "부족", "모자", "들고", "파우치", "미니",
    ],
}

# 키워드 강제 ON: 텍스트에 키워드 포함 시 해당 aspect를 강제 활성화 (FN 보정)
KEYWORD_FORCE_ON_CONFIG = {
    "재구매": {
        "keywords": ["재구매"],
        "sentiment": 1,  # positive (ID 1)
    },
}

# Design rule-based override (Stage 3A: 디자인을 모델→규칙으로 전환)
# 모델 예측을 무시하고 키워드 매칭으로 디자인 감성을 판단
DESIGN_RULE_CONFIG = {
    "positive_keywords": [
        "예쁘", "이쁘", "귀엽", "귀여", "깔끔", "고급", "앙증", "세련",
    ],
    "negative_keywords": [
        "촌스럽", "투박", "칙칙", "유치", "조잡", "싸구려", "못생",
    ],
    "structure_keywords": [
        "디자인", "패키지", "용기", "뚜껑", "케이스",
    ],
    "structure_pos_modifiers": [
        "좋", "마음에 들", "괜찮",
    ],
    "structure_neg_modifiers": [
        "별로", "아쉽", "불편", "안 좋", "안좋",
    ],
}

# None-threshold tuning configuration
# 학습 완료 후 val set에서 aspect별 최적 threshold를 grid search
THRESHOLD_TUNING_CONFIG = {
    "search_range": (0.1, 0.95),   # threshold 탐색 범위
    "search_step": 0.05,           # grid search 간격
    "metric": "fbeta",             # F0.5 (precision 가중) — 과다 검출 억제
    "beta": 0.5,                   # precision 4배 가중
    "default_threshold": 0.5,      # 튜닝 전 기본값
    "polar_threshold": 0.55,       # neutral 복원 기준 (None이면 polar 미사용)
}

# Golden set split configuration (Stage 2: dev/test 분할)
# 70/30으로 test 분산↓ + dev 튜닝 안정↑
GOLDEN_SPLIT_CONFIG = {
    "dev_ratio": 0.70,
    "test_ratio": 0.30,
    "random_state": 42,
}

# Fine-tuning configuration (Stage 2: 골든셋 파인튜닝)
FINETUNE_CONFIG = {
    "learning_rate": 2e-6,        # Stage 1의 1/10
    "num_epochs": 5,
    "batch_size": 16,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "gold_finetune_ratio": 0.60,  # 골든셋 finetune/tune/test 분할
    "gold_tune_ratio": 0.20,
    "gold_test_ratio": 0.20,
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
    "valid_aspects": ASPECT_LABELS,  # 10개 aspect 사용
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
VALIDATION_DATA_DIR = PROCESSED_DATA_DIR / "validation"
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
