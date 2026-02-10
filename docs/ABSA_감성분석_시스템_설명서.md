# Aspect-Based Sentiment Analysis (ABSA)

제품 리뷰 312,139개에 대한 감성 분석 + 측면(aspect) 추출 시스템

## 디렉토리 구조

```markdown
ABSA/
├── requirements_absa.txt       # 의존성 패키지
├── openai_client.py            # OpenAI API 클라이언트
├── Soft hierarchy.txt          # Aspect 계층 구조 정의
│
├── RQ/                         # 핵심 모듈
│   ├── config.py               # 설정 (샘플링, 학습, 검증 설정)
│   ├── sampling.py             # 층화 샘플링
│   ├── labeling.py             # ChatGPT 라벨링
│   ├── dataset.py              # 데이터셋 준비
│   ├── model.py                # 멀티태스크 모델
│   ├── train.py                # 학습 파이프라인
│   ├── inference.py            # 추론 파이프라인
│   └── evaluation.py           # 평가 메트릭
│
├── scripts/                    # 실행 스크립트
│   ├── step_a_sampling.py      # Step A: 층화 샘플링
│   ├── step_b_labeling.py      # Step B1: GPT-4o-mini 라벨링
│   ├── step_b2_rule_validation.py   # Step B2: 규칙 기반 검증
│   ├── step_b3_risk_sampling.py     # Step B3: 위험 케이스 탐지
│   ├── step_b4_judge_review.py      # Step B4: Judge 모델 재검수
│   ├── step_b5_merge_results.py     # Step B5: 검증 결과 병합
│   ├── step_c_create_dataset.py     # Step C: 학습 데이터셋 생성
│   ├── step_d_train.py              # Step D: 모델 학습
│   ├── step_e_inference.py          # Step E: 전체 추론
│   ├── fix_aspects_pattern_based.py # 패턴 기반 aspect 보정
│   ├── auto_review_v2.py            # 자동 리뷰 분석
│   ├── check_setup.py               # 설정 체크
│   ├── run_with_py_study.sh         # py_study 환경 실행
│   └── run_all.sh                   # 전체 파이프라인 실행
│
├── docs/                       # 문서
│   ├── aspect_changes.md       # Aspect 변경 이력
│   ├── aspect_evaluation_guide.md  # Aspect 평가 가이드
│   └── 라벨링 기준.pdf         # 라벨링 기준 문서
│
├── data/                       # 데이터 디렉토리
│   ├── raw/                    # 원본 (샘플링, 라벨링)
│   ├── processed/              # 전처리 (train/val/test)
│   ├── inference/              # 추론 결과
│   ├── validation/             # 검증 중간 결과
│   └── cache/                  # API 캐시
│
└── models/                     # 모델 디렉토리
    └── checkpoints/            # 체크포인트
```

---

## 1. 전략적 샘플링 로직: 데이터 편향의 기술적 해결

원본 데이터의 극심한 긍정 편향(94.3%)을 해결하기 위해 **'자연 분포 유지'와 '인위적 균형' 사이의 절충안**을 채택했습니다.

### 대분류 및 소분류 쿼터 배정

- **비례 배분 + 최소 보장**: 단순 비례 배분 시 사라질 수 있는 소수 카테고리(맨케어 등)를 보호하기 위해 카테고리별 최소 600개의 샘플을 보장합니다.
- **스킨케어 중심 구조**: 데이터의 68.5%를 차지하는 스킨케어에 13,070개의 쿼터를 할당하여 도메인 특성을 유지합니다.

### 샘플링 단계

```
1단계: 대분류(category_1) 쿼터 배정 (비례 + 최소 보장)
    |
2단계: 소분류(category_2) 쿼터 배정 (비례 + 최소 보장)
    |
3단계: 소분류 단위 자연 추출 (sentiment 강제 X)
    |
4단계: 전체 2만 레벨에서 sentiment 균형 조정
```

### 감정 분포의 한계와 보정

- **Negative 데이터의 희소성**: 원본 데이터에서 1~2점(Negative) 비율이 1.3%에 불과하여, 전체 풀을 긁어모아도 목표치인 30%(6,000건)에 못 미치는 20.4%(4,085건)만 확보 가능합니다.
- **Loss Function을 통한 해결**: 부족한 Negative 데이터의 학습 효율을 높이기 위해 다음 두 가지 방식을 `config.py`에서 선택할 수 있도록 설계되었습니다:
- **Class Weight**: 각 클래스의 빈도 역수를 Loss에 곱해 소수 클래스의 오차를 더 크게 반영합니다.
- **Focal Loss**: 맞추기 쉬운 샘플(Positive)의 가중치는 낮추고, 맞추기 어려운 샘플에 집중하여 불균형을 해소합니다.

---

## 2. 3단계 검증 시스템: 라벨 품질 최적화

LLM(ChatGPT)의 생성 결과물을 맹신하지 않고, **규칙 → 위험 탐지 → 재검수**의 단계를 거칩니다.

### Step B2: 규칙 기반 필터링

- **데이터 무결성**: JSON 형식이 깨지거나, 필수 필드(`sentiment`, `evidence` 등)가 누락된 경우를 걸러냅니다.
- **도메인 제약**: 미리 정의된 9개의 `ASPECT_LABELS` 외의 값이 들어오거나, `sentiment_score`가 범위를 벗어나면 `invalid` 처리합니다.

#### 검증 항목

| 검증 항목                | 설명                                          | 오류 유형           |
| ------------------------ | --------------------------------------------- | ------------------- |
| **필수 필드**            | `sentiment`, `aspect_labels`, `evidence` 필수 | `missing_field`     |
| **sentiment 유효성**     | positive/neutral/negative 중 하나             | `invalid_sentiment` |
| **sentiment_score 범위** | -1.0 ~ 1.0 사이 값                            | `invalid_score`     |
| **aspect_labels 유효성** | 9개 정의된 aspect 중 선택                     | `invalid_aspect`    |
| **aspect 중복**          | 동일 aspect 중복 불가                         | `duplicate_aspect`  |
| **evidence 매칭**        | evidence가 원문에 포함되어야 함               | `evidence_mismatch` |
| **JSON 파싱**            | 유효한 JSON 형식                              | `invalid_json`      |

#### 유효한 Aspect 레이블 (9개)

```python
ASPECT_LABELS = [
    "배송/포장", "품질/불량", "가격/가성비", "사용감/성능",
    "사이즈/호환", "디자인", "재질/냄새", "CS/응대", "재구매"
]
```

### Step B3: 위험 케이스 샘플링

논리적으로 오류 가능성이 높은 케이스를 3단계 등급으로 분류합니다:

- **HIGH**: 텍스트에 '최악', '환불' 같은 부정 키워드가 있는데 감정이 `positive`로 찍힌 경우 등.
- **MEDIUM**: '하지만', '근데'와 같은 대비 접속사가 쓰였음에도 속성(Aspect)이 1개만 추출되어 대조적인 의견이 누락되었을 가능성이 높은 경우.

#### 위험 유형 예시

##### HIGH 위험 (반드시 검수 필요)

| 유형                   | 조건                   | 설명                    |
| ---------------------- | ---------------------- | ----------------------- |
| `no_aspect`            | aspect_labels = []     | aspect가 하나도 없음    |
| `all_neutral`          | neutral + aspect 없음  | 중립 + aspect 없음 조합 |
| `neg_keyword_positive` | 부정 키워드 + positive | 키워드와 감정 불일치    |

#### 부정 키워드 목록

```python
negative_keywords = [
    '별로', '최악', '다시는', '환불', '불친절', '늦', '안좋', '안 좋',
    '실망', '후회', '짜증', '불량', '파손', '망', '싫', '아쉽', '거짓',
    '속았', '사기', '엉망', '쓰레기', '버림', '못씀', '안됨', '고장'
]
```

#### 대비 접속사

```python
contrast_markers = ['지만', '는데', '으나', '나', '만', '그러나', '하지만', '근데']
```

### Step B4: Judge 모델(GPT-4.1-mini)의 역할

- 위험군으로 분류된 데이터를 고성능 모델이 다시 읽고 `ok`(유지), `fix`(수정), `uncertain`(사람 확인 필요) 중 하나로 판정합니다.
- **비용 효율성**: 2만 건 전체를 고성능 모델로 검수하는 대신, 위험군(약 15~25%)만 타겟팅하여 약 $6 내외의 저비용으로 고품질 라벨을 얻습니다.

#### 판정 결과

| 판정        | 설명                   | 처리 방식             |
| ----------- | ---------------------- | --------------------- |
| `ok`        | 원본 라벨 정확함       | 원본 유지             |
| `fix`       | 수정 필요, 수정안 제시 | 자동 수정 적용        |
| `uncertain` | 판단 불가              | 사람 검수 필요 표시   |
| `error`     | API 오류 등            | 재시도 또는 사람 검수 |

---

## 3. 데이터 파이프라인 요약 테이블

각 단계의 처리 결과와 상태는 다음과 같이 관리됩니다.

| 최종 상태 (`validation_status`) | 설명                                                  | 모델 학습 포함 여부 |
| ------------------------------- | ----------------------------------------------------- | ------------------- |
| **Verified**                    | Judge 모델이 원본 라벨이 정확하다고 판정한 경우       | 포함                |
| **Fixed**                       | Judge 모델이 수정한 라벨 (80% 가중치 신뢰)            | 포함                |
| **Unchecked**                   | 위험 케이스에 해당하지 않아 검수를 통과한 일반 데이터 | 포함                |
| **Needs_Human_Review**          | Judge 모델도 판단이 어렵다고 한 경우                  | 표시 후 포함        |
| **Removed**                     | 규칙 검수(B2)에서 탈락한 형식 오류 데이터             | 제외                |

---

## 4. ABSA 재분석 파이프라인

기존 Judge 검증 이후에도 품질 개선이 필요한 부분이 발견되어, **패턴 기반 재분석** 방식을 추가 적용합니다.

### Phase 1: ABSA 재분석

```
1단계: GPT-5 샘플 1,000개 → aspect-sentiment 패턴 추출
    |
2단계: 패턴을 24,670건 전체에 적용
    → 새 형식: (text, aspect, sentiment, score)
    → 의심 케이스 추출 (별점 ↔ sentiment 불일치)
```

#### 의심 케이스 정의

| 별점 | 예상 sentiment | 불일치 조건 |
|------|----------------|-------------|
| 1~2점 | negative | positive로 분류된 경우 |
| 4~5점 | positive | negative로 분류된 경우 |
| 3점 | neutral | - |

### Phase 2: Sentiment 검증

```
1단계: 의심 케이스 중 샘플 GPT-5 분석 → 검증 패턴 추출
    |
2단계: 패턴으로 의심 케이스 1차 수정
    |
3단계: 여전히 애매한 케이스만 GPT-4o로 최종 검증
```

#### 검증 전략

- **패턴 우선**: 비용 효율을 위해 LLM 직접 호출을 최소화
- **단계적 필터링**: 쉬운 케이스는 패턴으로, 어려운 케이스만 고성능 모델 사용
- **GPT-5 → GPT-4o**: 샘플 분석은 GPT-5, 최종 검증은 GPT-4o 활용

---

## 5. 전체 파이프라인 요약

```
[기존 파이프라인]
Step A: 층화 샘플링 (20,000건)
    ↓
Step B1: GPT-4o-mini 라벨링
    ↓
Step B2: 규칙 기반 검증
    ↓
Step B3: 위험 케이스 탐지
    ↓
Step B4: Judge 모델 재검수
    ↓
Step B5: 검증 결과 병합

[재분석 파이프라인]
Phase 1: GPT-5 패턴 추출 → 전체 적용 → 의심 케이스 추출
    ↓
Phase 2: 패턴 기반 1차 수정 → GPT-4o 최종 검증
    ↓
Step C: 학습 데이터셋 생성
    ↓
Step D: 모델 학습
    ↓
Step E: 전체 추론
```

---

이 시스템은 **데이터의 희소성(Negative 부족)** 과 **LLM의 생성 불안정성**이라는 두 가지 핵심 문제를 구조적으로 해결하도록 설계되었습니다.
