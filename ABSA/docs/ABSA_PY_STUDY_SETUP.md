# ABSA Pipeline - py_study 환경 설정 완료

## 설치 완료 상태

모든 필수 패키지가 `py_study` 환경에 설치되었습니다!

### 설치된 패키지

- **torch**: 2.5.1
- **transformers**: 5.1.0
- **openai**: 2.17.0
- **accelerate**: 1.12.0
- **scikit-learn**: 1.7.2
- **pandas**: 2.3.3
- **numpy**: 1.26.4
- **tqdm**: 4.67.1

### 데이터

- **reviews.csv**: 312,139개 리뷰 (year 컬럼은 자동 추출됨)

### 디렉토리

- 모든 ABSA 디렉토리 생성 완료

## 추가 설정 필요

### 1. OpenAI API 키 (Step B에만 필요)

```bash
export OPENAI_API_KEY='your-api-key-here'
```

### 2. GPU (Step D, E에 권장)

- Step A, B, C: CPU로 실행 가능
- Step D (학습): GPU 필요 또는 매우 느림
- Step E (추론): GPU 필요 또는 매우 느림

## 실행 방법

### 방법 1: Python 직접 실행 (권장)

```bash
# py_study 환경의 Python 사용
/opt/miniconda3/envs/py_study/bin/python scripts/absa/step_a_sampling.py
/opt/miniconda3/envs/py_study/bin/python scripts/absa/step_b_labeling.py
/opt/miniconda3/envs/py_study/bin/python scripts/absa/step_c_create_dataset.py
/opt/miniconda3/envs/py_study/bin/python scripts/absa/step_d_train.py
/opt/miniconda3/envs/py_study/bin/python scripts/absa/step_e_inference.py
```

### 방법 2: Wrapper 스크립트 사용

```bash
cd scripts/absa

# 단축 명령어
./run_with_py_study.sh step_a_sampling.py
./run_with_py_study.sh step_b_labeling.py
./run_with_py_study.sh step_c_create_dataset.py
./run_with_py_study.sh step_d_train.py
./run_with_py_study.sh step_e_inference.py
```

### 방법 3: Alias 설정 (편의성)

```bash
# ~/.zshrc 또는 ~/.bashrc에 추가
alias py_absa='/opt/miniconda3/envs/py_study/bin/python'

# 사용법
py_absa scripts/absa/step_a_sampling.py
```

## 전체 파이프라인 실행 순서

### Step A: 샘플링 (5분, CPU 가능)

```bash
/opt/miniconda3/envs/py_study/bin/python scripts/absa/step_a_sampling.py
```

- 입력: `data/csv/reviews.csv` (312,139개)
- 출력: `data/absa/raw/sampled_reviews_20k.csv` (20,000개)
- 계층화 샘플링으로 평점 불균형 조정

### Step B: ChatGPT 라벨링 (2-4시간, ~$3-5, CPU 가능)

```bash
# API 키 설정 필요
export OPENAI_API_KEY='your-key'

/opt/miniconda3/envs/py_study/bin/python scripts/absa/step_b_labeling.py
```

- 입력: `data/absa/raw/sampled_reviews_20k.csv`
- 출력: `data/absa/raw/chatgpt_labels_20k.jsonl`
- GPT-4o-mini로 감성 + 측면 라벨링
- 캐싱으로 중복 요청 방지
- 중단 후 재개 가능

### Step C: 데이터셋 생성 (2분, CPU 가능)

```bash
/opt/miniconda3/envs/py_study/bin/python scripts/absa/step_c_create_dataset.py
```

- 입력: `data/absa/raw/chatgpt_labels_20k.jsonl`
- 출력:
  - `data/absa/processed/train.csv` (~14,000개)
  - `data/absa/processed/val.csv` (~3,000개)
  - `data/absa/processed/test.csv` (~3,000개)
- Train/Val/Test 분할 (70/15/15)

### Step D: 모델 학습 (2-3시간, GPU 필요)

```bash
/opt/miniconda3/envs/py_study/bin/python scripts/absa/step_d_train.py
```

- 입력: train.csv, val.csv
- 출력: `models/absa/checkpoints/best_model.pt`
- KcELECTRA-base 멀티태스크 학습
- GPU 없으면 매우 느림 (권장하지 않음)

### Step E: 전체 추론 (15-20분, GPU 필요)

```bash
/opt/miniconda3/envs/py_study/bin/python scripts/absa/step_e_inference.py
```

- 입력: `data/csv/reviews.csv` (312,139개)
- 출력:
  - `data/absa/inference/reviews_with_absa.csv` (전체)
  - `data/absa/inference/reviews_with_absa_ambiguous.csv` (애매한 샘플)
- GPU 없으면 매우 느림

## 현재 바로 실행 가능한 단계

**CPU만으로 실행 가능 (Step A, B, C):**

```bash
# Step A 실행 (5분)
/opt/miniconda3/envs/py_study/bin/python scripts/absa/step_a_sampling.py

# Step B는 API 키 설정 후 실행
export OPENAI_API_KEY='your-key'
/opt/miniconda3/envs/py_study/bin/python scripts/absa/step_b_labeling.py

# Step C 실행 (2분)
/opt/miniconda3/envs/py_study/bin/python scripts/absa/step_c_create_dataset.py
```

이후 Step D, E는 GPU 환경에서 실행하거나, CPU로 긴 시간을 투자해야 합니다.

## 출력 스키마

최종 CSV (`reviews_with_absa.csv`) 컬럼:

| 컬럼명          | 타입  | 설명          | 예시                          |
| --------------- | ----- | ------------- | ----------------------------- |
| sentiment       | str   | 감성 분류     | "positive"                    |
| sentiment_score | float | 감성 점수     | 0.85                          |
| aspect_labels   | list  | 측면 라벨     | ["배송/포장", "품질/불량"]    |
| evidence        | str   | 근거 문장     | "배송이 빠르고 품질도 좋아요" |
| summary         | str   | 요약          | "배송, 품질에 대해 긍정적"    |
| is_ambiguous    | bool  | 애매함 플래그 | False                         |
| + 원본 컬럼들   | ...   | ...           | ...                           |

## 9개 측면 카테고리

1. **배송/포장** - 배송 속도, 포장 상태
2. **품질/불량** - 제품 품질, 불량 여부
3. **가격/가성비** - 가격 대비 만족도
4. **사용감/성능** - 사용 편의성, 성능
5. **사이즈/호환** - 크기, 호환성
6. **디자인** - 외관, 디자인
7. **재질/냄새** - 재질, 냄새
8. **CS/응대** - 고객 서비스
9. **재구매** - 재구매 의향

## 예상 비용

- Step B (ChatGPT): **$3-5**
- Step D (GPU 클라우드): **$1-2** (로컬 GPU면 무료)
- Step E (GPU 클라우드): **$0.50** (로컬 GPU면 무료)
- **총합: ~$5-8**

## 예상 시간

- Step A: 5분
- Step B: 2-4시간
- Step C: 2분
- Step D: 2-3시간 (GPU) / 수일 (CPU)
- Step E: 15-20분 (GPU) / 수시간 (CPU)
- **총합: ~5-8시간 (GPU 기준)**

## 문제 해결

### "ModuleNotFoundError" 발생 시

```bash
# py_study 환경 확인
/opt/miniconda3/envs/py_study/bin/pip list

# 패키지 재설치
/opt/miniconda3/envs/py_study/bin/pip install transformers openai accelerate
```

### API Rate Limit 초과 시

`RQ/absa/config.py` 수정:

```python
OPENAI_CONFIG['rate_limit_rpm'] = 30  # 60에서 30으로 감소
```

### GPU 메모리 부족 시

`RQ/absa/config.py` 수정:

```python
TRAIN_CONFIG['batch_size'] = 16  # 32에서 16으로 감소
INFERENCE_CONFIG['batch_size'] = 64  # 128에서 64로 감소
```

## 추가 문서

- **전체 가이드**: `ABSA_IMPLEMENTATION.md`
- **빠른 시작**: `ABSA_QUICKSTART.md`
- **모듈 문서**: `RQ/absa/README.md`

## 다음 단계

**지금 바로 시작 가능:**

```bash
# 1. 셋업 재확인
/opt/miniconda3/envs/py_study/bin/python scripts/absa/check_setup.py

# 2. Step A 실행 (5분, CPU)
/opt/miniconda3/envs/py_study/bin/python scripts/absa/step_a_sampling.py

# 3. Step B 실행 전 API 키 설정
export OPENAI_API_KEY='your-key'

# 4. Step B 실행 (2-4시간, CPU)
/opt/miniconda3/envs/py_study/bin/python scripts/absa/step_b_labeling.py
```

## 완료!

모든 준비가 끝났습니다. Step A부터 시작하세요!

```bash
/opt/miniconda3/envs/py_study/bin/python scripts/absa/step_a_sampling.py
```
