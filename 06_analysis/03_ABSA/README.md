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
| 최종 결과 | 28,981행 | 완료 (aspect별 펼침) |

## 사용법

```bash
# 환경 설정
cp config/.env.example config/.env
# OPENAI_API_KEY 설정

# 의존성 설치
pip install -r requirements.txt

# 배치 라벨링 실행
python 07_scripts/batch_labeling.py
```

## 생성일

2025-02-16
