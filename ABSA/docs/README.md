# Aspect-Based Sentiment Analysis (ABSA)

제품 리뷰 312,139개에 대한 감성 분석 + 측면(aspect) 추출 시스템

## 디렉토리 구조

```
ABSA/
├── README.md                    # 이 파일
├── ABSA_PY_STUDY_SETUP.md      # py_study 환경 설정 가이드 (한글)
├── ABSA_IMPLEMENTATION.md      # 전체 구현 가이드 (영문)
├── ABSA_QUICKSTART.md          # 빠른 시작 가이드 (영문)
├── requirements_absa.txt        # 의존성 패키지
├── openai_client.py            # OpenAI API 클라이언트
│
├── RQ_absa/                    # 핵심 모듈
│   ├── config.py               # 설정
│   ├── sampling.py             # 샘플링
│   ├── labeling.py             # ChatGPT 라벨링
│   ├── dataset.py              # 데이터셋 준비
│   ├── model.py                # 멀티태스크 모델
│   ├── train.py                # 학습 파이프라인
│   ├── inference.py            # 추론 파이프라인
│   └── evaluation.py           # 평가 메트릭
│
├── scripts/                    # 실행 스크립트
│   ├── step_a_sampling.py      # Step A: 샘플링
│   ├── step_b_labeling.py      # Step B: 라벨링
│   ├── step_c_create_dataset.py # Step C: 데이터셋
│   ├── step_d_train.py         # Step D: 학습
│   ├── step_e_inference.py     # Step E: 추론
│   ├── evaluate_test.py        # 테스트 평가
│   ├── check_setup.py          # 설정 체크
│   ├── run_with_py_study.sh    # py_study 환경 실행
│   └── run_all.sh              # 전체 파이프라인 실행
│
├── data/                       # 데이터 디렉토리
│   ├── raw/                    # 원본 (샘플링, 라벨링)
│   ├── processed/              # 전처리 (train/val/test)
│   ├── inference/              # 추론 결과
│   └── cache/                  # API 캐시
│
└── models/                     # 모델 디렉토리
    └── checkpoints/            # 체크포인트
```

## 빠른 시작

### 1. 설정 확인

```bash
cd /Users/yu_seok/Documents/workspace/nbCamp/Project/Why-pi/ABSA

# py_study 환경으로 설정 체크
/opt/miniconda3/envs/py_study/bin/python scripts/check_setup.py
```

### 2. 파이프라인 실행

#### 방법 1: 전체 자동 실행 (Step A-C, CPU 가능)

```bash
./scripts/run_all.sh
```

#### 방법 2: 단계별 실행

```bash
# Step A: 샘플링 (5분)
/opt/miniconda3/envs/py_study/bin/python scripts/step_a_sampling.py

# Step B: ChatGPT 라벨링 (2-4시간, ~$3-5)
export OPENAI_API_KEY='your-key'
/opt/miniconda3/envs/py_study/bin/python scripts/step_b_labeling.py

# Step C: 데이터셋 생성 (2분)
/opt/miniconda3/envs/py_study/bin/python scripts/step_c_create_dataset.py

# Step D: 모델 학습 (2-3시간, GPU 필요)
/opt/miniconda3/envs/py_study/bin/python scripts/step_d_train.py

# Step E: 전체 추론 (15-20분, GPU 필요)
/opt/miniconda3/envs/py_study/bin/python scripts/step_e_inference.py
```

#### 방법 3: Wrapper 스크립트 사용

```bash
cd scripts

./run_with_py_study.sh step_a_sampling.py
./run_with_py_study.sh step_b_labeling.py
./run_with_py_study.sh step_c_create_dataset.py
```

## 📊 파이프라인 단계

| 단계        | 시간    | 비용   | GPU | 입력               | 출력                     |
| ----------- | ------- | ------ | --- | ------------------ | ------------------------ |
| A. 샘플링   | 5분     | 무료   | X   | reviews.csv (312k) | sampled_reviews_20k.csv  |
| B. 라벨링   | 2-4시간 | $3-5   | X   | sampled 20k        | chatgpt_labels_20k.jsonl |
| C. 데이터셋 | 2분     | 무료   | X   | labels 20k         | train/val/test.csv       |
| D. 학습     | 2-3시간 | $1-2\* | O   | train/val          | best_model.pt            |
| E. 추론     | 15-20분 | $0.5\* | O   | reviews.csv (312k) | reviews_with_absa.csv    |

\*클라우드 GPU 기준, 로컬 GPU면 무료

## 9개 측면(Aspect) 카테고리

1. **배송/포장** - 배송 속도, 포장 상태
2. **품질/불량** - 제품 품질, 불량 여부
3. **가격/가성비** - 가격 대비 만족도
4. **사용감/성능** - 사용 편의성, 성능
5. **사이즈/호환** - 크기, 호환성
6. **디자인** - 외관, 디자인
7. **재질/냄새** - 재질, 냄새
8. **CS/응대** - 고객 서비스
9. **재구매** - 재구매 의향

## 출력 스키마

최종 CSV (`data/inference/reviews_with_absa.csv`):

| 컬럼            | 타입  | 설명          | 예시                          |
| --------------- | ----- | ------------- | ----------------------------- |
| sentiment       | str   | 감성 분류     | "positive"                    |
| sentiment_score | float | 감성 점수     | 0.85                          |
| aspect_labels   | list  | 측면 라벨     | ["배송/포장", "품질/불량"]    |
| evidence        | str   | 근거 문장     | "배송이 빠르고 품질도 좋아요" |
| summary         | str   | 요약          | "배송, 품질에 대해 긍정적"    |
| is_ambiguous    | bool  | 애매함 플래그 | False                         |
| (원본 컬럼들)   | ...   | ...           | ...                           |

## 환경 설정

### 필수 패키지

- Python 3.12+ (py_study 환경)
- torch 2.5.1
- transformers 5.1.0
- openai 2.17.0
- accelerate 1.12.0

### 설치된 환경

```bash
# py_study 환경 확인
/opt/miniconda3/envs/py_study/bin/pip list
```

## 사용 예시

### 결과 분석

```python
import pandas as pd
import ast

# 결과 로드
df = pd.read_csv('data/inference/reviews_with_absa.csv')

# aspect_labels 파싱 (문자열 → 리스트)
df['aspect_labels'] = df['aspect_labels'].apply(ast.literal_eval)

# 감성 분포
print(df['sentiment'].value_counts())

# 부정 리뷰 중 배송 관련
negative_shipping = df[
    (df['sentiment'] == 'negative') &
    (df['aspect_labels'].apply(lambda x: '배송/포장' in x))
]
print(f"배송 불만 리뷰: {len(negative_shipping):,}개")

# 높은 확신도 부정 리뷰
high_conf_neg = df[
    (df['sentiment'] == 'negative') &
    (df['sentiment_score'] < -0.5)
]
```

## 문제 해결

### GPU 메모리 부족

`RQ_absa/config.py` 수정:

```python
TRAIN_CONFIG['batch_size'] = 16  # 32 → 16
INFERENCE_CONFIG['batch_size'] = 64  # 128 → 64
```

### API Rate Limit

`RQ_absa/config.py` 수정:

```python
OPENAI_CONFIG['rate_limit_rpm'] = 30  # 60 → 30
```

### ModuleNotFoundError

```bash
/opt/miniconda3/envs/py_study/bin/pip install transformers openai accelerate
```

## 문서

- **한글 가이드**: `ABSA_PY_STUDY_SETUP.md`
- **전체 구현 가이드**: `ABSA_IMPLEMENTATION.md`
- **빠른 시작 가이드**: `ABSA_QUICKSTART.md`
- **모듈 문서**: `RQ_absa/README.md`

## 예상 비용 & 시간

| 항목                | 비용     | 시간         |
| ------------------- | -------- | ------------ |
| ChatGPT 라벨링      | $3-5     | 2-4시간      |
| GPU 학습 (클라우드) | $1-2     | 2-3시간      |
| GPU 추론 (클라우드) | $0.5     | 15-20분      |
| **총합**            | **$5-8** | **~5-8시간** |

## 체크리스트

- [ ] py_study 환경 활성화
- [ ] reviews.csv 경로 확인 (../data/csv/reviews.csv)
- [ ] OpenAI API 키 설정 (Step B용)
- [ ] GPU 환경 준비 (Step D, E용, 선택사항)
- [ ] Step A 실행 (샘플링)
- [ ] Step B 실행 (라벨링, API 키 필요)
- [ ] Step C 실행 (데이터셋)
- [ ] Step D 실행 (학습, GPU 권장)
- [ ] Step E 실행 (추론, GPU 권장)

## 시작하기

```bash
# 1. 디렉토리 이동
cd /Users/yu_seok/Documents/workspace/nbCamp/Project/Why-pi/ABSA

# 2. 설정 확인
/opt/miniconda3/envs/py_study/bin/python scripts/check_setup.py

# 3. Step A 실행
/opt/miniconda3/envs/py_study/bin/python scripts/step_a_sampling.py
```

## 지원

문제가 발생하면:

1. `scripts/check_setup.py` 실행
2. `ABSA_PY_STUDY_SETUP.md` 참고
3. `RQ_absa/config.py` 설정 확인

---

**Last Updated**: 2026-02-06
**Python Environment**: py_study (Python 3.12.8)
**Why-pi Project**: /Users/yu_seok/Documents/workspace/nbCamp/Project/Why-pi
