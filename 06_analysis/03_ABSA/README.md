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
│   ├── figures/
│   └── reports/
│
├── 05_src/                   # Python 모듈
│   ├── config.py             # 설정
│   ├── dataset.py            # 데이터셋 클래스
│   ├── model.py              # 모델 정의
│   ├── train.py              # 학습 로직
│   ├── inference.py          # 추론
│   ├── evaluation.py         # 평가
│   ├── labeling.py           # 라벨링
│   └── sampling.py           # 샘플링
│
├── 06_models/                # 모델 체크포인트
│   └── checkpoints/
│
├── 07_scripts/               # 실행 스크립트
│   ├── batch_labeling.py     # Batch API 라벨링
│   ├── run_batch_pipeline.py # 배치 파이프라인
│   ├── evaluate_model.py     # 모델 평가
│   └── ...
│
├── 999_Temporary/            # 임시 파일 (주기적 정리)
│   ├── batch_inputs/         # Batch API jsonl 파일
│   └── cache/
│
├── config/                   # 설정 파일
│   └── .env
│
├── docs/                     # 로컬 참조 문서
│   └── soft_hierarchy.txt
│
│   # 📌 주요 문서는 07_docs/ABSA/에 통합 관리
│   #    07_docs/ABSA/ABSA_파이프라인.md
│   #    07_docs/ABSA/TEAM_GUIDE.md
│
├── logs/                     # 로그 파일
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

# 패키지 경로 설정
export PYTHONPATH=$(pwd)

# 모델 학습
python -m RQ_absa.train \
    --csv_path 02_processed_data/interim/v4/absa_analysis_ready.csv \
    --model_name beomi/KcELECTRA-base \
    --num_epochs 10 \
    --batch_size 32

# 추론 실행 (CSV 기반)
python -m RQ_absa.inference \
    --input_path /path/to/reviews.csv \
    --output_path /path/to/results.csv \
    --model_path 06_models/checkpoints/best_model.pt

# OpenAI 배치 라벨링 (기존)
cp config/.env.example config/.env
python 07_scripts/batch_labeling.py
```

## 주요 소스 파일

| 파일 | 역할 |
|------|------|
| `05_src/config.py` | 라벨 정의, 학습/추론/threshold 설정 |
| `05_src/dataset.py` | CSV → 리뷰 단위 그룹화 → Dataset 생성 |
| `05_src/model.py` | MultiTaskABSAModel (KcELECTRA + 2 heads) |
| `05_src/train.py` | 학습 루프, threshold 자동 튜닝 |
| `05_src/evaluation.py` | 4-class F1, detection F1, per-aspect F1 |
| `05_src/inference.py` | 배치 추론, DataFrame 추론, BigQuery 연동 |

## 생성일

2025-02-16 (최종 수정: 2026-02-23)
