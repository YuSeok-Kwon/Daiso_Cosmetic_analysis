# ABSA Stage 2 — 골든셋 평가 결과

> **작성일:** 2026-02-25
> **모델:** KcELECTRA-base (Stage 2, Epoch 10)
> **평가 데이터:** golden_test.csv (266행, 8 aspects)
> **Threshold:** F0.5 튜닝 + polar_threshold=0.55

---

## 목차

1. [한 줄 요약](#1-한-줄-요약)
2. [평가 결과 요약](#2-평가-결과-요약)
3. [Detection 분석 — Precision이 0.65에 못 미치는 이유](#3-detection-분석)
4. [Aspect별 심층 진단](#4-aspect별-심층-진단)
5. [Sentiment 분석 — neutral의 딜레마](#5-sentiment-분석)
6. [근본 원인 정리](#6-근본-원인-정리)
7. [Stage 3 개선 방향](#7-stage-3-개선-방향)
8. [실행 우선순위 및 로드맵](#8-실행-우선순위-및-로드맵)

---

## 1. 한 줄 요약

골든셋 Test(266행)에서 Stage 2 모델을 평가한 결과, Detection Precision = **0.6079** (목표 0.65 미달). **디자인 Precision 0.17**이 전체를 끌어내리는 주범이며, **사용감/성능은 +23pp 과다검출**. Mentioned Sentiment Macro F1 = **0.43**. Stage 3에서 디자인 규칙 전환 + 사용감 threshold 조정 + polar 재튜닝으로 개선 예정.

---

## 2. 평가 결과 요약

### 2.0 Polar란?

Polar threshold는 ABSA 모델의 감성 분류 후처리 단계

```
모델의 기본 동작 (Polar OFF) : 
각 aspect에 대해 모델은 4개 확률을 출력 → P(none), P(positive), P(neutral), P(negative)

기본 흐름은:P(none) >= none_threshold → none (이 aspect 언급 안 됨)
나머지는 argmax(P(pos), P(neu), P(neg)) → 가장 높은 것 선택

문제: 모델이 neutral을 거의 예측하지 않았습니다. pos/neg 중 하나가 항상 neutral보다 높게 나와서, 실제로는 중립인 리뷰도 긍정이나 부정으로 분류됨.
```


```
Polar threshold가 하는 일 (Polar ON) :
none이 아닌 것으로 판단된 aspect에 대해 추가 관문을 둔다.

P(none) >= t_none  → none (언급 안 됨)
max(P(pos), P(neg)) < t_polar  → neutral (확신 부족 → 중립 처리)
else → argmax(pos, neg) 중 선택

즉, 긍정/부정 중 어느 쪽이든 확률이 polar threshold 미만이면 "확신이 부족하다"고 보고 neutral로 전환하는 것입니다.
```


  표에서 보이는 효과

- Detection 지표 동일: polar는 "언급 여부"가 아니라 "감성 방향"만 바꾸므로 detection에는 영향 없음
- Neutral Recall 개선: 0.2586 → 0.3103 (+5.2%p) — 실제 neutral인 것을 더 잘 잡음
- Pred Neutral 셀 수 증가: 94 → 155 — neutral 예측이 많아짐 (다소 과잉)
- Macro F1 소폭 하락: 0.4332 → 0.4185 — neutral을 과하게 만들어 다른 클래스 성능이 약간 떨어짐

  한마디로, **"모델이 pos/neg를 확신하지 못하면 neutral로 돌리는 안전장치"** 이다.

### 2.1 핵심 지표 (Polar OFF vs ON)

| 지표                                   | Polar OFF        | Polar ON         | 비고                                 |
| -------------------------------------- | ---------------- | ---------------- | ------------------------------------ |
| **Detection Precision (#1 KPI)** | **0.6079** | 0.6079           | 목표 0.65+,**-0.042 부족**     |
| Detection Recall                       | 0.6993           | 0.6993           | polar는 binary detection에 영향 없음 |
| Detection F1                           | 0.6504           | 0.6504           |                                      |
| Detection F0.5                         | 0.6242           | 0.6242           |                                      |
| Mentioned Sentiment Macro F1           | **0.4332** | 0.4185           | polar ON이 오히려 -0.015             |
| Neutral Recall (GT>0)                  | 0.2586           | **0.3103** | polar ON이 +5.2%p 개선               |
| Pred Neutral 셀 수                     | 94               | **155**    | polar ON이 +61셀 (과잉)              |

### 2.2 판정

- **Detection Precision:** 목표 미달. 디자인(0.17)이 전체를 끌어내리는 주범
- **Polar threshold:** neutral recall은 소폭 개선하나, 정답 pos/neg를 neutral로 전환하여 sentiment F1 하락. 순효과 미미
- **30만 건 적용 시 영향:** Precision 0.61은 "언급됐다고 판단한 것 중 39%가 오탐" → 카운트 기반 분석에서 왜곡 발생

---

## 3. Detection 분석

### 3.1 Aspect별 Detection Precision (핵심 진단표)

| Aspect           | Precision        | Recall           | F1     | F0.5   | GT 언급% | Pred 언급%      | ±pp            | 진단                   |
| ---------------- | ---------------- | ---------------- | ------ | ------ | -------- | --------------- | --------------- | ---------------------- |
| 재구매           | **0.8125** | 0.8125           | 0.8125 | 0.8125 | 6.0%     | 6.0%            | 0.0             | 최우수                 |
| 배송/포장        | **0.8049** | 0.6735           | 0.7333 | 0.7746 | 18.4%    | 15.4%           | -3.0            | 우수 (다소 보수적)     |
| 사용감/성능      | 0.6651           | **0.9329** | 0.7765 | 0.7056 | 57.1%    | **80.1%** | **+23.0** | 과다 검출              |
| 재질/냄새        | 0.6136           | 0.675            | 0.6429 | 0.625  | 15.0%    | 16.5%           | +1.5            | 양호                   |
| 가격/가성비      | 0.6053           | 0.451            | 0.5169 | 0.5665 | 19.2%    | 14.3%           | -4.9            | 과소 검출              |
| 색상/발색        | 0.589            | 0.7167           | 0.6466 | 0.6108 | 22.6%    | 27.5%           | +4.9            | 소폭 과다              |
| 용량/휴대        | 0.5526           | 0.525            | 0.5385 | 0.5469 | 15.0%    | 14.3%           | -0.8            | 분포 정상, 정밀도 부족 |
| **디자인** | **0.1739** | 0.2353           | 0.2    | 0.1835 | 12.8%    | **17.3%** | +4.5            | **붕괴**         |

### 3.2 Precision 가중 기여도 분석

전체 Detection Precision = 0.6079. 만약 디자인을 제외하면?

```
디자인 제외 시:
  GT 언급 셀:  439 - 34 = 405
  Pred 언급 셀: (전체 pred mentions) - 46 = 나머지
  디자인의 FP 기여: 46 × (1 - 0.1739) = 38건의 오탐

  디자인 제외 Precision (추정): ~0.66+  ← 목표 근접
```

**결론:** 디자인 1개 aspect가 전체 Precision을 0.05+ 깎고 있다.

---

## 4. Aspect별 심층 진단

### 4.1 디자인 — 구조적 붕괴 (최우선 개선 대상)

**Confusion Matrix (Polar ON):**

```
           pred_none  pred_pos  pred_neu  pred_neg
GT_none       194         0        37         1    ← none인데 37건을 neutral로 오탐
GT_pos         11         0         0         0    ← positive 11건 전부 놓침
GT_neu          2         0         2         0
GT_neg         13         0         6         0    ← negative 19건 중 13건 놓침
```

**핵심 문제:**

1. **GT_none → pred_neutral = 37건**: 모델이 "언급 안 됨"인데 neutral로 오탐 (FP의 주 원인)
2. **GT_pos 11건 → 전부 pred_none**: positive 디자인 표현("예쁘다", "귀엽다")을 전혀 못 잡음
3. **Polar ON 악화**: polar threshold가 모든 약한 확신을 neutral로 밀어서, GT_none → neutral 38건 발생
4. **학습 데이터 부족**: 디자인 positive/negative 실제 샘플 극소, negative sampling은 "confirmed none"만 보강

**Val에서도 동일:**

- Val Detection F1: Epoch 10 기준 0.094 (8 aspect 중 최저)
- Epoch 1~5까지 Detection F1 = 0.000 (전부 none 예측)

### 4.2 사용감/성능 — 과다 검출 (+23pp)

**분포 비교:**

```
GT:   112 none / 73 pos / 22 neu / 54 neg  (언급률 57.1%)
Pred:  52 none / 86 pos / 44 neu / 79 neg  (언급률 80.1%)
```

**Confusion Matrix (Polar ON):**

```
           pred_none  pred_pos  pred_neu  pred_neg
GT_none        42        30        17        23    ← none인데 70건을 언급으로 오탐!
GT_pos          3        49        16         5
GT_neu          2         6         9         5
GT_neg          5         1         2        46
```

**핵심 문제:**

1. **GT_none 112건 중 70건이 FP**: "좋아요", "괜찮아요" 같은 일반 감성을 사용감/성능으로 오분류
2. **Threshold 0.10 = 너무 공격적**: P(none)이 0.10만 넘으면 "언급됨"으로 판단
3. **모델의 bias**: 사용감/성능이 학습 데이터의 57%를 차지 → 기본적으로 "언급됐다"고 예측하는 경향

### 4.3 가격/가성비 — 과소 검출 (-4.9pp)

**Confusion Matrix (Polar ON):**

```
           pred_none  pred_pos  pred_neu  pred_neg
GT_none       200        12         2         1
GT_pos         17        21         0         0    ← positive 38건 중 17건 놓침 (45%)
GT_neu          3         1         0         0    ← neutral 4건 전부 놓침
GT_neg          8         0         0         1    ← negative 9건 중 8건 놓침 (89%)
```

**핵심 문제:**

1. **GT_pos 38건 중 17건 pred_none**: "가성비 좋다", "이 가격에 이 정도면" 같은 표현 미검출
2. **GT_neg 9건 중 8건 pred_none**: "비싸다", "가격 대비 별로" 놓침
3. **recall 0.451**: 실제 언급된 것 중 절반만 잡음

### 4.4 용량/휴대 — 감성 혼동

**Confusion Matrix (Polar ON):**

```
           pred_none  pred_pos  pred_neu  pred_neg
GT_none       209         1        10         6
GT_pos         12         2         6         1    ← positive 21건 중 12건 놓침
GT_neu          4         1         1         1
GT_neg          3         0         2         7
```

**핵심 문제:**

1. **GT_pos 놓침 (12/21)**: "양이 많다", "휴대하기 좋다"를 인식 못함
2. **GT_none → pred_neu/neg = 16건**: 과다 예측과 감성 오분류 동시 발생

### 4.5 양호 그룹: 재구매, 배송/포장

| Aspect    | F1(4cls) macro | 특징                                         |
| --------- | -------------- | -------------------------------------------- |
| 재구매    | 0.8044         | 분포·감성 모두 정확. "재구매" 키워드 명확   |
| 배송/포장 | 0.5506         | Detection 좋으나 positive(15건) 중 10건 놓침 |

---

## 5. Sentiment 분석

### 5.1 Mentioned Sentiment (GT>0 셀만)

| class      | precision | recall | f1    | support |
| ---------- | --------- | ------ | ----- | ------- |
| none(miss) | -         | -      | -     | 0       |
| positive   | 0.906     | 0.491  | 0.637 | 216     |
| neutral    | 0.286     | 0.310  | 0.298 | 58      |
| negative   | 0.850     | 0.655  | 0.740 | 165     |

**해석:**

- **positive recall 0.49**: 실제 긍정 216건 중 절반만 맞춤. 나머지는 neutral이나 none으로 분류
- **neutral F1 0.30**: 학습 데이터 내 neutral 극소(3.1%)로 인한 구조적 한계
- **negative**: 상대적으로 양호 (precision 0.85, recall 0.65)

### 5.2 Neutral 복원 실험 결과

```
GT neutral 58건의 운명 (Polar ON):
  → pred_none(놓침):      19건 (32.8%)
  → pred_positive:         9건 (15.5%)
  → pred_neutral(정답):   18건 (31.0%)  ← recall = 0.3103
  → pred_negative:        12건 (20.7%)
```

```
전체 neutral 분포:
  GT neutral:     58셀
  Pred neutral:  155셀  ← 2.67배 과다 예측
```

**Polar threshold의 부작용:**

- GT_neutral 복원: +3건 (15→18)
- GT_non-neutral을 neutral로 오전환: +61건 이상
- **비용 대비 효과가 낮음**: 3건 맞추기 위해 61건 이상을 오분류

### 5.3 Aspect별 Sentiment F1 (4-class macro)

| Aspect           | F1(4cls)        | F1(wgt)         | 주요 문제           |
| ---------------- | --------------- | --------------- | ------------------- |
| **디자인** | **0.235** | 0.750           | pos 전멸, neg 전멸  |
| 가격/가성비      | 0.417           | 0.819           | neg 거의 못 잡음    |
| 용량/휴대        | 0.419           | 0.820           | pos 대부분 놓침     |
| 사용감/성능      | 0.523           | 0.558           | none↔non-none 혼동 |
| 재질/냄새        | 0.527           | 0.861           | pos 거의 못 잡음    |
| 색상/발색        | 0.530           | 0.798           | 양호                |
| 배송/포장        | 0.551           | 0.893           | pos 거의 못 잡음    |
| **재구매** | **0.804** | **0.974** | 최우수              |

---

## 6. 근본 원인 정리

### 6.1 문제-원인 매핑

| # | 문제                           | 현상                         | 근본 원인                                                                   |
| - | ------------------------------ | ---------------------------- | --------------------------------------------------------------------------- |
| 1 | **디자인 붕괴**          | Precision 0.17, pos/neg 전멸 | 학습 데이터에 디자인 긍/부정 실제 샘플 극소 + 간접 표현("예쁘다") 미학습    |
| 2 | **사용감/성능 과다검출** | +23pp 과다, GT_none 70건 FP  | 학습 데이터의 57%가 사용감 → 모델 bias + threshold 0.10 너무 공격적        |
| 3 | **positive recall 부족** | 전체 pos recall 0.49         | 배송·가격·용량·재질 등에서 positive 표현 미검출                          |
| 4 | **neutral 구조적 한계**  | recall 0.31, 과다예측 2.7배  | 학습 neutral 1,356건(3.1%) → 모델이 neutral 직접 학습 불가, polar로도 한계 |
| 5 | **가격/가성비 과소검출** | -4.9pp, neg 8/9 놓침         | 가격 관련 간접 표현("이 가격에", "비싸") 미검출                             |

### 6.2 GPT-4o 라벨 vs 골든셋 라벨 Gap

Stage 2의 학습 데이터는 GPT-4o가 자동 라벨링한 17,132건이다. 골든셋은 사람이 직접 라벨링한 884건. 이 두 라벨 사이의 **체계적 차이(systematic gap)**가 모델 성능의 천장을 결정한다.

| Gap 유형                    | 예시                                               | 영향                        |
| --------------------------- | -------------------------------------------------- | --------------------------- |
| **간접 표현 미인식**  | "예쁘다" → GPT는 미분류, 사람은 디자인            | 디자인 pos 학습 데이터 부족 |
| **경계 모호**         | "좋다" → GPT는 사용감/성능, 사람은 none           | 사용감 과다 검출            |
| **neutral 기준 차이** | "그냥 그래" → GPT는 종종 positive, 사람은 neutral | neutral 학습 데이터 과소    |
| **복합 감성**         | "예쁜데 약해" → GPT는 하나만, 사람은 둘 다        | 동시 언급 패턴 미학습       |

---

## 7. Stage 3 개선 방향

### 방향 A: 골든셋 파인튜닝 (핵심)

**목적:** GPT-4o 라벨과 사람 라벨 사이의 gap을 모델 수준에서 보정

**방법:**

```
입력:  Stage 2 best_model.pt (8 aspect)
데이터: golden_dev.csv (618행)
lr:    2e-6 (Stage 2의 1/10)
Epoch: 3~5 (과적합 모니터링)
평가:  golden_test.csv (266행)
```

**기대 효과:**

- 사람 라벨의 패턴을 직접 학습 → "예쁘다"=디자인, "그냥 그래"=neutral 등
- Detection 전반의 미세 보정
- **리스크:** 618건은 소량이므로 과적합 가능. early stopping 필수

**세부 전략:**

- Class weight 재계산 (골든셋 분포 기준)
- Learning rate scheduler: cosine annealing with warm restarts
- Val loss가 2 epoch 연속 상승하면 중단
- 최적 epoch의 모델로 threshold 재튜닝

### 방향 B: 디자인 특별 처리

디자인은 파인튜닝만으로 해결이 어렵다 (골든셋 test에 34건뿐). 별도 전략 필요.

**Option B-1: 규칙 기반 전환 (가장 현실적)**

```
디자인 aspect를 모델 예측에서 제외하고, 키워드 규칙으로 대체:

Tier 1 (positive 키워드):
  예쁘, 이쁘, 귀엽, 귀여, 깔끔, 고급, 앙증, 세련

Tier 2 (negative 키워드):
  촌스럽, 투박, 칙칙, 유치, 조잡, 싸구려, 못생

Tier 3 (구조물 + 감성 조합):
  (디자인|패키지|용기|뚜껑|케이스) + (좋|마음에 들|별로|아쉽)

로직:
  if Tier1 매칭 → 디자인=positive
  elif Tier2 매칭 → 디자인=negative
  elif Tier3 매칭 → 문맥에서 감성 추출 (간단 규칙 or 모델)
  else → 디자인=none
```

**장점:** Precision 극적으로 개선 (오탐 38건 제거), 즉시 적용 가능
**단점:** recall 하락 가능 (간접 표현 누락), 유지보수 부담

**Option B-2: 추가 라벨링 + 재학습**

```
30만 리뷰에서 디자인 관련 리뷰 2,000~3,000건 추출
  → GPT-4o 라벨링 (디자인 특화 프롬프트)
  → 사람 검수 (200~300건)
  → Stage 2 데이터에 합쳐서 재학습
```

**장점:** 모델이 직접 디자인 패턴을 학습
**단점:** 시간·비용 필요 ($30~50 추가), 검수 인력

**Option B-3: B-1 + B-2 병행 (권장)**

```
Phase 1: 즉시 B-1 적용 → 30만 건 추론 시 디자인=규칙 기반
Phase 2: 추가 데이터 확보 후 B-2 반영 → 차기 모델에서 모델 기반 전환
```

### 방향 C: 사용감/성능 과다검출 억제

**C-1: Threshold 상향 조정**

현재 threshold 0.10 → **0.20~0.30으로 상향**

```
예상 효과:
  Threshold 0.10 → 0.25:
    - Pred 언급률 80.1% → ~65% (추정)
    - GT_none FP 70건 → ~40건 (추정)
    - Precision 향상, Recall 소폭 하락
```

golden_dev에서 grid search로 사용감/성능의 최적 threshold를 별도 튜닝.

**C-2: 사용감/성능 전용 Negative Rule**

```
일반 감성 표현만 있고 구체적 속성 키워드가 없는 리뷰는 사용감/성능=none으로 override:

구체적 키워드 (있으면 유지):
  발림, 흡수, 지속, 커버, 밀착, 촉촉, 건조, 끈적, 보습,
  세정, 거품, 자극, 트러블, 효과, 기능

일반 키워드 (이것만 있으면 none):
  좋다, 괜찮다, 별로, 그냥, 무난
```

### 방향 D: Positive Recall 개선

배송/포장, 가격/가성비, 용량/휴대, 재질/냄새에서 positive를 놓치는 문제.

**원인:** GPT-4o 라벨에서 positive 표현이 다른 aspect나 none으로 잘못 태깅된 경우가 많음

**해결:**

1. 골든셋 파인튜닝(방향 A)에서 자연스럽게 보정 기대
2. 추가로, 각 aspect의 positive 키워드를 post-hoc boosting:

```
if 모델이 none으로 예측했지만, 해당 aspect의 positive 키워드가 존재하면:
  → P(positive)를 0.1만큼 boost → 재판정
```

### 방향 E: Neutral 전략 재설계

**현행 문제:** polar_threshold=0.55로 neutral을 복원하면 오히려 정답을 깎음

**E-1: Polar threshold 상향 (보수적 neutral)**

```
현재: polar_threshold = 0.55 → neutral recall 0.31, pred_neutral 155건
제안: polar_threshold = 0.70 → 더 확신 없는 경우만 neutral → 과잉 예측 억제
```

**E-2: Aspect별 차별적 polar threshold**

```
사용감/성능: polar_threshold = 0.75 (과다검출 aspect → 보수적)
디자인:     polar_threshold = 없음 (규칙 기반으로 전환)
나머지:     polar_threshold = 0.60
```

**E-3: Neutral 포기 (실용적 대안)**

```
30만 건 분석에서 neutral은 실질적으로 활용도가 낮음.
binary로 전환: positive vs negative (none은 유지)
neutral → none으로 병합하면:
  - 모델 complexity 감소
  - Precision 향상 (neutral FP 제거)
  - 비즈니스 의사결정에 큰 영향 없음 (긍정/부정만 중요)
```

---

## 8. 실행 우선순위 및 로드맵

### 8.1 우선순위 매트릭스

| 순위        | 방향                                 | 난이도 | 기대 효과              | 소요 시간 |
| ----------- | ------------------------------------ | ------ | ---------------------- | --------- |
| **1** | **A. 골든셋 파인튜닝**         | 중     | 높음 (전반적 보정)     | 2~3시간   |
| **2** | **B-1. 디자인 규칙 전환**      | 낮     | 높음 (Precision +0.05) | 1시간     |
| **3** | **C-1. 사용감 threshold 상향** | 낮     | 중 (과다검출 억제)     | 30분      |
| **4** | **E-1. Polar threshold 상향**  | 낮     | 중 (neutral 과잉 억제) | 30분      |
| 5           | D. Positive boosting                 | 중     | 중                     | 2시간     |
| 6           | B-2. 디자인 추가 라벨링              | 높     | 높 (장기)              | 1~2일     |
| 7           | E-3. Neutral 제거                    | 낮     | 낮~중                  | 1시간     |

### 8.2 Phase별 실행 계획

```
Phase 3A: 즉시 실행 (코드 수정 + 재학습)
├── Step 1. 골든셋 파인튜닝 (lr=2e-6, 3~5 epoch)
├── Step 2. golden_dev에서 threshold 재튜닝
│           - 사용감/성능: 별도 grid search (0.15~0.40)
│           - polar_threshold: 0.60~0.75 grid search
├── Step 3. 디자인 규칙 기반 모듈 구현
│           - s8_inference.py에 design_rule_override() 추가
├── Step 4. golden_test 최종 평가
└── Step 5. 30만 건 추론

Phase 3B: 후속 (데이터 보강, 선택)
├── Step 6. 디자인·가격·용량 약점 aspect 리뷰 추출
├── Step 7. GPT-4o 추가 라벨링 (2,000~3,000건)
├── Step 8. 검수 후 학습 데이터 합산
└── Step 9. Stage 4 재학습 (필요 시)
```

### 8.3 Stage 3 목표

| 지표                         | Stage 2 현재 | Stage 3 목표        | 비고                                            |
| ---------------------------- | ------------ | ------------------- | ----------------------------------------------- |
| Detection Precision          | 0.6079       | **0.65+**     | 디자인 규칙 전환 + 사용감 threshold로 달성 가능 |
| Detection F1                 | 0.6504       | 0.63+               | Precision 올리면 Recall 소폭 하락 허용          |
| Detection F0.5               | 0.6242       | **0.65+**     | Precision 가중 지표도 목표 달성                 |
| Mentioned Sentiment Macro F1 | 0.4185       | **0.50+**     | 파인튜닝으로 개선 기대                          |
| 디자인 Precision             | 0.1739       | **0.50+**     | 규칙 기반 전환 시                               |
| 사용감 ±pp                  | +23.0        | **±10 이내** | threshold 상향으로                              |

### 8.4 판단 기준

Stage 3 완료 후 다음 기준으로 30만 건 적용 여부를 판단:

```
GO 조건 (모두 충족):
  Detection Precision ≥ 0.65
  Detection F1 ≥ 0.60
  사용감/성능 ±pp ≤ 15
  디자인 제외 7 aspect Precision 평균 ≥ 0.60

HOLD 조건 (하나라도 해당):
  Detection Precision < 0.60
  특정 aspect ±pp > 20
  → Phase 3B 데이터 보강 후 재시도
```

---

> **다음 문서:** [Stage 3A 후처리 튜닝 리포트](ABSA_Stage3A_후처리_튜닝_리포트.md)
