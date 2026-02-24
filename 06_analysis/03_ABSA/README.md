# ABSA (Aspect-Based Sentiment Analysis)

다이소 뷰티 리뷰 데이터에 대한 속성 기반 감성 분석 프로젝트

## 프로젝트 개요

- **목적**: 고객 리뷰에서 제품 속성별 감성을 추출하여 연착륙 상품 분석 지원
- **데이터**: 20,000개 샘플링된 리뷰 (층화 샘플링)
- **방법**: GPT-4o Batch API 라벨링 → KoELECTRA 학습 예정

## 디렉토리 구조

```
03_ABSA/
├── 01_raw_data/              # 원본 데이터 (읽기 전용)
│   └── sampled_reviews_20k.csv
│
├── 02_processed_data/        # 처리된 데이터
│   ├── final/                # 최종 결과물
│   │   ├── absa_results_final.csv
│   │   └── absa_results_final.xlsx
│   ├── interim/              # 중간 파일 (팀별 라벨링)
│   └── validation/           # 검증 세트 (golden_set)
│
├── 03_notebooks/             # 분석 노트북
│
├── 04_outputs/               # 산출물
│   ├── cache/                # 캐시
│   ├── figures/
│   ├── inference/            # 추론 결과
│   ├── logs/                 # 로그 파일
│   └── reports/
│
├── 05_src/                   # Python 모듈 (import: RQ_absa)
│   ├── s1_config.py          # 설정 (라벨, 경로, 학습/추론 파라미터)
│   ├── s2_sampling.py        # 층화 샘플링
│   ├── s3_labeling.py        # GPT 라벨링
│   ├── s4_dataset.py         # CSV → Dataset (리뷰 그룹화, 골든셋 분할)
│   ├── s5_model.py           # MultiTaskABSAModel (KcELECTRA + 2 heads)
│   ├── s6_train.py           # 학습 루프 (2-stage: 약지도 → 골든셋 파인튜닝)
│   ├── s7_evaluation.py      # 평가 (4-class F1, threshold 튜닝)
│   └── s8_inference.py       # 배치 추론, streaming CSV
│
├── 06_scripts/               # 실행 스크립트
│   ├── batch_labeling.py     # Batch API 라벨링
│   ├── run_batch_pipeline.py # 배치 파이프라인
│   ├── evaluate_model.py     # 모델 평가 (GPT vs Golden Set)
│   ├── run_absa_bq.py        # BigQuery 연동
│   └── openai_client.py      # OpenAI API 클라이언트
│
├── 07_models/                # 모델 체크포인트
│   └── checkpoints/
│
├── 999_Temporary/            # 임시 파일 (주기적 정리)
│   ├── batch_inputs/         # Batch API jsonl 파일
│   └── cache/
│
├── .env                      # 환경 변수 (API 키 등, git 미포함)
├── RQ_absa -> 05_src         # 심볼릭 링크 (import 경로)
├── README.md
└── requirements.txt
```

## Aspect 카테고리 (11개)

| 카테고리 | 설명 |
|---------|------|
| 배송/포장 | 배송 속도, 포장 상태 |
| 품질/퀄리티 | 제품 퀄리티, 제조 결함 |
| 가격/가성비 | 가격 만족도, 가성비 평가 |
| 사용감/성능 | 발림성, 지속력, 커버력 |
| 용량/휴대 | 용량 만족도, 휴대성 |
| 디자인 | 외관, 패키지 디자인 |
| 재질/냄새 | 텍스처, 향 |
| CS/응대 | 고객 서비스 |
| 재구매 | 재구매 의사 |
| 색상/발색 | 발색력, 색상 만족도 |
| 미분류 | 분류 불가 |

## 라벨링 현황

| 단계 | 데이터 수 | 상태 |
|------|----------|------|
| GPT-4o Batch 라벨링 | 19,950개 | 완료 |
| 최종 결과 (aspect별 펼침) | 26,267행 | 완료 |
| 리뷰 단위 그룹화 | 18,016 리뷰 | 완료 (mixed 29건 제거) |

## 모델 아키텍처 (Option A)

**Aspect별 4-class 통합 예측:** 각 aspect에 대해 `none/negative/neutral/positive` 직접 분류

```
KcELECTRA Encoder (768-dim)
        │ [CLS]
   ┌────┴────┐
   ▼         ▼
Sentiment  Aspect-Sentiment
[B, 3]     [B, 11, 4]
```

| 라벨 ID | 의미 | 설명 |
|---------|------|------|
| 0 | none | 해당 aspect 미존재 |
| 1 | negative | 부정 |
| 2 | neutral | 중립 |
| 3 | positive | 긍정 |

**데이터 전처리 규칙:**
- `review_sentiment == "mixed"` → 리뷰 제거 (29건)
- `aspect == "미분류"` → 항상 `neutral(2)`로 강제

**불균형 처리:**
- CrossEntropyLoss + class weight (역빈도 가중치)
- 학습 후 val set에서 per-aspect none-threshold 자동 튜닝

## 사용법

```bash
# 환경 설정
conda activate py_study  # torch, transformers, sklearn 필요

# 03_ABSA/ 디렉토리에서 실행
cd 06_analysis/03_ABSA
export PYTHONPATH=$(pwd)

# 모델 학습 (Stage 1: 약지도 학습)
python -m RQ_absa.s6_train \
    --csv_path 02_processed_data/final/absa_analysis_ready.csv \
    --model_name beomi/KcELECTRA-base \
    --num_epochs 10 \
    --batch_size 32

# 추론 실행 (CSV 기반, streaming)
python -m RQ_absa.s8_inference \
    --input_path /path/to/reviews.csv \
    --output_path 04_outputs/inference/results.csv \
    --model_path 07_models/checkpoints/best_model.pt

# OpenAI 배치 라벨링
python 06_scripts/batch_labeling.py
```

## 주요 소스 파일

| 파일 | 역할 |
|------|------|
| `05_src/s1_config.py` | 라벨 정의, 경로, 학습/추론/threshold/파인튜닝 설정 |
| `05_src/s2_sampling.py` | 3단계 층화 샘플링 (대분류→소분류→감성) |
| `05_src/s3_labeling.py` | GPT-4o 라벨링 (단건/배치/BigQuery) |
| `05_src/s4_dataset.py` | CSV → 리뷰 그룹화 → Dataset + 골든셋 3-way 분할 |
| `05_src/s5_model.py` | MultiTaskABSAModel (KcELECTRA + 2 heads) |
| `05_src/s6_train.py` | 학습 루프 + 2-stage (약지도→골든셋 파인튜닝) |
| `05_src/s7_evaluation.py` | 4-class F1, detection F1, per-aspect none-threshold 튜닝 |
| `05_src/s8_inference.py` | streaming 배치 추론, DataFrame/BigQuery 추론 |

## 생성일

2025-02-16 (최종 수정: 2026-02-24)
