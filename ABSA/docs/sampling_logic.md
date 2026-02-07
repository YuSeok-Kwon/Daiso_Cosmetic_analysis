# 자연 분포 기반 층화 샘플링 로직

## 개요

ABSA 라벨링을 위한 20,000개 리뷰 샘플을 추출하는 자연 분포 기반 층화 샘플링 방법론입니다.

**핵심 원칙:**

- 복원추출(oversampling) 없음
- 소분류 단위에서는 sentiment 비율 강제 없이 자연 추출
- 전체 레벨에서 sentiment 균형 조정
- 부족한 클래스는 학습 시 class weight/focal loss로 보정

---

## 샘플링 단계

```
1단계: 대분류(category_1) 쿼터 배정 (비례 + 최소 보장)
    |
2단계: 소분류(category_2) 쿼터 배정 (비례 + 최소 보장)
    |
3단계: 소분류 단위 자연 추출 (sentiment 강제 X)
    |
4단계: 전체 2만 레벨에서 sentiment 균형 조정
```

---

## 1단계: 대분류 쿼터 배정

### 알고리즘

```
쿼터 = 최소 보장(600) + 비례 배분
```

### 결과

| 대분류   | 원본 개수 | 원본 비율 | 최소 보장 | 최종 쿼터  |
| -------- | --------- | --------- | --------- | ---------- |
| 스킨케어 | 215,185   | 68.5%     | 600       | 13,070     |
| 메이크업 | 92,863    | 29.6%     | 600       | 5,980      |
| 맨케어   | 6,055     | 1.9%      | 600       | 950        |
| **합계** | 314,103   | 100%      | 1,800     | **20,000** |

---

## 2단계: 소분류 쿼터 배정

### 스킨케어 내 소분류

| 소분류       | 원본 비율 | 최소 보장 | 최종 쿼터 |
| ------------ | --------- | --------- | --------- |
| 기초스킨케어 | 70.9%     | 200       | 8,753     |
| 팩/마스크    | 15.2%     | 200       | 2,029     |
| 클렌징/필링  | 5.6%      | 200       | 879       |
| 자외선차단제 | 5.5%      | 200       | 868       |
| 립케어       | 2.8%      | 200       | 541       |

### 메이크업 내 소분류

| 소분류          | 원본 비율 | 최소 보장 | 최종 쿼터 |
| --------------- | --------- | --------- | --------- |
| 립메이크업      | 30.1%     | 200       | 1,763     |
| 베이스메이크업  | 29.8%     | 200       | 1,742     |
| 아이메이크업    | 26.2%     | 200       | 1,557     |
| 치크/하이라이터 | 13.9%     | 200       | 918       |

### 맨케어

맨케어는 소분류 쿼터 배정을 **스킵**하고, 전체 950개에서 자연 추출합니다.

---

## 3단계: 소분류 단위 자연 추출

**핵심: sentiment 비율 강제 없음**

각 소분류에서 쿼터만큼 **자연스럽게 추출**합니다. 원본 데이터의 sentiment 분포가 그대로 반영됩니다.

### 1차 샘플 결과 (20,000개)

| Sentiment | 개수   | 비율  |
| --------- | ------ | ----- |
| positive  | 18,868 | 94.3% |
| neutral   | 843    | 4.2%  |
| negative  | 289    | 1.4%  |

원본 데이터가 positive 편향이므로 1차 샘플도 동일하게 편향됩니다.

---

## 4단계: 전체 레벨 sentiment 균형 조정

### 목표 분포

| Sentiment | 목표 비율 | 목표 개수 |
| --------- | --------- | --------- |
| negative  | 30%       | 6,000     |
| neutral   | 30%       | 6,000     |
| positive  | 40%       | 8,000     |

### 조정 방법

1. **부족한 sentiment 추가 확보**
   - 1차 샘플에서 제외된 잔여 풀에서 추가 샘플링
   - 복원추출 없이 가용 범위 내에서만

2. **초과한 sentiment 제거**
   - 목표 초과분은 제거하여 20,000개 맞춤
   - positive 우선 제거

### 최종 결과

| Sentiment | 최종 개수 | 최종 비율 | 목표 | 차이   |
| --------- | --------- | --------- | ---- | ------ |
| negative  | 4,085     | 20.4%     | 30%  | -9.6%p |
| neutral   | 6,000     | 30.0%     | 30%  | 0.0%p  |
| positive  | 9,915     | 49.6%     | 40%  | +9.6%p |

**negative 부족 원인:**

- 원본 데이터의 negative(1-2점) 비율이 약 1.3%
- 전체 잔여 풀에서 확보 가능한 negative가 4,085개뿐
- 복원추출 없이는 이 이상 확보 불가

---

## 클래스 불균형 보정 (학습 단계) - 구현 완료

negative가 목표에 미달하므로 학습 시 보정이 필요합니다. **이미 구현되어 있습니다.**

### 설정 (config.py)

```python
TRAIN_CONFIG = {
    ...
    "use_class_weight": True,       # Balanced class weights 사용
    "use_focal_loss": False,        # Focal Loss 사용 (대안)
    "focal_gamma": 2.0              # Focal Loss gamma
}
```

### 사용 방법 (train.py)

```python
from RQ_absa.train import create_model_with_class_weights

# 학습 데이터에서 자동으로 class weight 계산
model = create_model_with_class_weights(
    train_dataset=train_dataset,
    model_name="beomi/KcELECTRA-base",
    use_class_weight=True,      # Class weight 사용
    use_focal_loss=False,       # 또는 Focal Loss 사용
    focal_gamma=2.0
)
```

### Class Weight 계산 방식

```python
# Balanced weights: total / (num_classes * class_count)
weights = total / (num_classes * class_counts)

# 예: negative 4,085개, neutral 6,000개, positive 9,915개
# negative weight = 20,000 / (3 * 4,085) = 1.63
# neutral weight  = 20,000 / (3 * 6,000) = 1.11
# positive weight = 20,000 / (3 * 9,915) = 0.67
```

### Focal Loss (대안)

```python
# FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
# gamma=2: 쉬운 샘플의 영향 감소, 어려운 샘플에 집중
```

### 옵션 선택 가이드

| 옵션                    | 설명                 | 추천 상황                       |
| ----------------------- | -------------------- | ------------------------------- |
| `use_class_weight=True` | 클래스별 가중치 적용 | 일반적인 불균형 (기본값)        |
| `use_focal_loss=True`   | Focal Loss 사용      | 극심한 불균형, 어려운 샘플 강조 |

---

## 최종 샘플 분포

### 대분류별 분포

| 대분류   | 개수   | 비율  |
| -------- | ------ | ----- |
| 스킨케어 | 11,926 | 59.6% |
| 메이크업 | 7,310  | 36.5% |
| 맨케어   | 764    | 3.8%  |

### 평점 분포

| 평점 | 개수  | 비율  |
| ---- | ----- | ----- |
| 1점  | 2,084 | 10.4% |
| 2점  | 2,001 | 10.0% |
| 3점  | 6,000 | 30.0% |
| 4점  | 1,307 | 6.5%  |
| 5점  | 8,608 | 43.0% |

---

## 실행 방법

```bash
/opt/miniconda3/envs/py_study/bin/python scripts/step_a_sampling.py
```

### 출력 파일

- 경로: `data/raw/sampled_reviews_20k.csv`
- 컬럼: product_code, text, rating, date, user_id, order_id, brand, category_1, category_2

---

## 파일 구조

```
ABSA/
├── RQ_absa/
│   ├── config.py          # 샘플링 설정
│   └── sampling.py        # NaturalStratifiedSampler 클래스
├── scripts/
│   └── step_a_sampling.py # 샘플링 실행 스크립트
├── data/
│   └── raw/
│       └── sampled_reviews_20k.csv  # 출력 파일
└── docs/
    └── sampling_logic.md  # 이 문서
```

---

## 설계 근거

### 왜 복원추출을 하지 않는가?

1. **데이터 다양성 유지**: 동일 리뷰 중복 시 모델이 특정 패턴에 과적합
2. **평가 신뢰성**: 중복 샘플이 train/test에 섞이면 정확한 평가 불가
3. **자연스러운 분포**: 실제 데이터 분포를 반영한 학습

### 왜 전체 레벨에서 sentiment 조정하는가?

1. **소분류 보존**: 소분류 내 자연 분포 유지
2. **유연한 조정**: 가용 데이터 범위 내에서 최대한 균형화
3. **투명성**: 어떤 sentiment가 부족한지 명확히 파악

### 왜 class weight/focal loss를 사용하는가?

1. **데이터 한계 인정**: negative 샘플은 원본에서 1.3%뿐
2. **학습 효율**: 복원추출보다 loss 조정이 효과적
3. **모델 일반화**: 소수 클래스에 적절한 가중치 부여
