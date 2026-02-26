"""
Configuration for ABSA pipeline
"""
import os
from pathlib import Path

# Project root (Why-pi/ABSA)
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths (번호 디렉토리 기준)
RAW_DATA_DIR = PROJECT_ROOT / "00_raw_data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "01_outputs" / "data"
INFERENCE_DATA_DIR = PROJECT_ROOT / "01_outputs" / "inference"
CACHE_DIR = PROJECT_ROOT / "01_outputs" / "cache"
LOG_DIR = PROJECT_ROOT / "01_outputs" / "logs"

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
# Stage 4C: 정규표현식 기반으로 전환 (keyword_analysis.py 설계 철학 적용)
# - 활용형 제한: r'싸[다고게서니]' (substring "싸"의 "감싸" 오탐 차단)
# - negative lookahead: r'향(?!상|수[인를])' ("향상", "향수인" 제외)
# - 경계 처리: r'(?<![가-힣])양[이은도]' (복합어 속 "양" 제외)
# - 숫자+단위: r'\d+g\b' (영단어 속 "g" 제외)
KEYWORD_GATE_CONFIG = {
    "디자인": [
        # Tier1: 구조물/용기류 — 오탐 0%, 그대로 유지
        r"용기", r"뚜껑", r"케이스", r"패키지", r"디자인",
        r"스포이드", r"펌프", r"튜브",
        # Tier2: 외관 표현
        r"예쁘", r"이쁘", r"귀엽", r"귀여", r"앙증",
        # "깔끔" — 사용감 맥락 제외 (3.7% 오탐 → ~0%)
        r"깔끔(?!\s*(?:하게\s*)?(?:흡수|발[라림]|바[르름]|마무리|사용))",
        # "고급" — 사용감 맥락 제외 (2.4% 오탐 → ~0%)
        r"고급(?!\s*(?:스러운\s*)?(?:사용감|느낌|발림|텍스처))",
        # "팁" — 구조물 팁만 (41.5% 오탐 → ~0%)
        r"(?:실리콘|브러시|어플리케이터|스펀지|붓)\s*팁",
        r"팁[이은가]\s*(?:좋|예쁘|이쁘|귀엽|부드럽|딱딱|뻣뻣|얇|두꺼)",
        # 입구/마감 — 오탐 0%, 그대로 유지
        r"입구", r"마감",
    ],
    "가격/가성비": [
        # 고신뢰 키워드 (오탐 0%) — 그대로 유지
        r"가격", r"가성비", r"저렴", r"가격대", r"비싸",
        r"천원", r"원짜리", r"최저가", r"저가",
        r"돈[이을값]?", r"값[이어에]?",
        r"만원",
        # "싸" — 활용형만 매칭 (99.5% 오탐 → ~0%)
        r"싸[다고게서니요네]",
        r"싼\s*(?:거|것|편|가격|제품|데)",
        # "싸게" — "잽싸게" 제외 (12.4% 오탐 → ~0%)
        r"(?<!잽)싸게",
    ],
    "재질/냄새": [
        # "향" — "향상", "향수(시간)" 제외 (4.0% 오탐 → ~0%)
        r"향(?!상|수[인를이]?\s|후[에를])",
        # 고신뢰 키워드 (오탐 0%)
        r"냄새", r"무향", r"알코올", r"에탄올",
        r"역[함겨한하]",
        r"향기",
        r"불쾌", r"악취",
        r"독[함한하]",
        # 자극감 표현 — "쏘" 대신 활용형만 (97% 오탐 → 0%)
        r"쏘[는다아]",
        r"톡\s?쏘",
        r"자극[이적]?",
        r"따[끔갑가]",
        r"시리[다고]?",
        r"저릿",
        # "방향" — 방위 의미 제외, 향기 의미만
        r"방향[이가제]",
    ],
    "용량/휴대": [
        # "양" — 독립적 용량 맥락만 (91.7% 오탐 → ~5%)
        r"(?<![가-힣])양[이은도만]\s",
        r"(?<![가-힣])양\s*조절",
        r"(?<![가-힣])양[이은도]?\s*(?:적|많|넉넉|부족|충분|아쉽|괜찮)",
        # 고신뢰 키워드 (오탐 0%)
        r"용량", r"사이즈",
        r"작[다아은]",
        r"크[다고게]",
        r"휴대", r"여행",
        r"적[다고은]",
        r"많[다고은이]",
        r"넉넉", r"부족", r"모자[라르]",
        r"들고\s*(?:다니|나가|가[다요니])",
        r"파우치", r"미니",
        # "ml" — 단위 패턴만
        r"\d+\s*ml",
        # "g" — 숫자+g 단위만 (100% 오탐 → 0%)
        r"\d+\s*g(?![a-zA-Z])",
        # "오래" — 용량/지속 맥락만 (23.5% 오탐 → ~5%)
        r"오래\s*(?:쓸|쓰[다고니]|쓸\s*수|사용|가[다요니]|버티|갈\s*것|갈\s*듯)",
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
