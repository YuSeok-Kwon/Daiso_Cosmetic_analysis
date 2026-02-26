# ABSA 파이프라인 문서

## 프로젝트 개요

**목표:** 다이소 뷰티 리뷰 데이터에서 Aspect-Based Sentiment Analysis를 수행하여 '연착륙 스킨케어' 제품 발굴

**데이터 규모:**
- 전체 리뷰: ~300,000개
- 샘플링 데이터: 20,000개
- Golden Set: 430개

---

## 파이프라인 흐름도

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ABSA 파이프라인                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [1] 층화 샘플링                                                     │
│       ↓                                                             │
│  [2] GPT-4o-mini 1차 라벨링 (20,000개)                               │
│       ↓                                                             │
│  [3] 사람 직접 검증 → 오류 패턴 파악                                  │
│       ↓                                                             │
│  [4] 프롬프트 재수정                                                 │
│       ↓                                                             │
│  [5] GPT-4o vs GPT-4o-mini 비교 (수정된 프롬프트)                     │
│       ↓                                                             │
│  [6] GPT-4o Batch API로 20,000개 라벨링 ✅ (현재 완료)                │
│       ↓                                                             │
│  [7] 사람 직접 검수 (샘플링)                                         │
│       ↓                                                             │
│      ┌──────────┴──────────┐                                        │
│      ↓                     ↓                                        │
│  [7-1] 이상 없음       [7-2] 이상 있음                               │
│      ↓                     ↓                                        │
│  [8] 모델 선별          [3]으로 돌아가기                             │
│      ↓                                                             │
│  [9] 전체 리뷰 적용 (300,000개)                                      │
│      ↓                                                             │
│  [10] 연착륙 상품 분석                                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 단계별 상세 설명

### 1단계: 층화 샘플링 전략

**목적:** 전체 300,000개 리뷰에서 대표성 있는 20,000개 추출

**샘플링 데이터 컬럼:**
- `product_code`: 제품 코드
- `name`: 제품명
- `category_2`: 카테고리
- `rating`: 평점 (1~5)
- `text`: 리뷰 텍스트
- `order_id`: 주문 ID

**카테고리 분포 (category_2):**
| 카테고리 | 건수 | 비율 |
|----------|------|------|
| 기초스킨케어 | 7,718 | 38.6% |
| 립메이크업 | 2,343 | 11.7% |
| 베이스메이크업 | 2,080 | 10.4% |
| 아이메이크업 | 1,992 | 10.0% |
| 팩/마스크 | 1,926 | 9.6% |
| 치크/하이라이터 | 1,079 | 5.4% |
| 자외선차단제 | 924 | 4.6% |
| 클렌징/필링 | 758 | 3.8% |
| 립케어 | 449 | 2.2% |
| 기타 (남성 등) | 731 | 3.7% |

**평점 분포:**
| 평점 | 건수 | 비율 |
|------|------|------|
| 1점 | 1,925 | 9.6% |
| 2점 | 1,769 | 8.8% |
| 3점 | 6,000 | 30.0% |
| 4점 | 1,502 | 7.5% |
| 5점 | 8,804 | 44.0% |

**결과:** `06_analysis/03_ABSA/01_raw_data/sampled_reviews_20k.csv`

---

### 2단계: GPT-4o-mini 1차 라벨링

**목적:** 빠르고 저렴하게 초기 라벨링 수행

**설정:**
- 모델: `gpt-4o-mini`
- 비용: ~$2.87 (20,000개 기준)
- 소요 시간: 약 2-3시간

**출력:**
- 전체 Sentiment (positive/neutral/negative)
- Aspect Labels (11개 카테고리)
- Confidence Score
- Evidence (근거 문장)

---

### 3단계: 사람 직접 검증 및 오류 패턴 파악

**Golden Set 구성:**
- Team 1: 148개
- Team 2: 137개
- Team 3: 145개
- **총 430개**

**주요 오류 패턴:**

| 혼동 패턴 | 빈도 | 원인 |
|----------|------|------|
| 색상/발색 → 사용감/성능 | 26회 (93%) | "발색" vs "발림" 혼동 |
| 배송/포장 → CS/응대 | 10회 | 배송 중 파손 vs 제품 결함 |
| 품질/퀄리티 → 사용감/성능 | 12회 | "퀄리티" 키워드 누락 |

---

### 4단계: 프롬프트 재수정

**개선 사항:**

1. **Aspect 혼동 방지 규칙 추가**
```
- 색상/발색: "발색, 색, 컬러, 톤" 키워드 → 무조건 색상/발색
- 사용감/성능: 발림성, 지속력, 커버력 → 사용감/성능
- "발색" ≠ "발림" 명시
```

2. **평점 기반 Sentiment 보조 판단**
```
- 1~2점: negative 가능성 높음
- 3점: neutral 가능성 높음
- 4~5점: positive 가능성 높음
```

3. **미분류 카테고리 추가**
```
- Confidence < 0.7이면 미분류로 처리
- 억지로 Aspect 추출하지 않음
```

4. **다이소 특화 규칙**
```
- 듀프 비교: "올리브영 XX랑 똑같다" → 가격/가성비 positive
- 품절 대란: "구하기 힘들어서 짜증 (5점)" → 재구매 positive
```

---

### 5단계: GPT-4o vs GPT-4o-mini 비교

**목적:** 수정된 프롬프트로 어떤 모델이 더 적합한지 평가

**평가 방법:**
- Golden Set (430개)으로 동일 조건 테스트
- 정확도 비교

**결과:**
| 모델 | Aspect 정확도 | Sentiment 정확도 | Both 정확도 |
|------|--------------|-----------------|-------------|
| GPT-4o-mini | 51.2% | 67.0% | 43.5% |
| GPT-4o | 67.6% | 72.1% | **58.3%** |

**비용 비교 (20,000개 기준, Batch API 50% 할인):**
| 모델 | 비용 | 정확도 |
|------|------|--------|
| GPT-4o-mini | $2.87 | 43.5% |
| GPT-4o | $95.70 | 58.3% |

**결론:**
- GPT-4o가 15%p 더 정확
- B2B 입점 제안 목적상 정확도가 중요 → **GPT-4o 채택**
- 정확한 라벨 1개당 추가 비용: ~44원 (사람 검수 비용 대비 저렴)

---

### 6단계: GPT-4o Batch API로 20,000개 라벨링 ✅

**현재 상태:** 완료

**설정:**
- 모델: `gpt-4o`
- 방식: Batch API (50% 비용 절감)
- 병렬 처리: 2개 API 키 사용

**비용:**
| 구분 | 리뷰 수 | 비용 |
|------|---------|------|
| 기존 키 | 13,150개 | $62.92 |
| 새 키 | 6,850개 | $32.78 |
| **총합** | **20,000개** | **$95.70** |

**출력 파일:**
- `06_analysis/03_ABSA/02_processed_data/interim/v2/absa_results_final.csv`

---

### 7단계: 사람 직접 검수

**방법:**
- 랜덤 샘플 100~200개 추출
- 오류율 확인
- 특히 "미분류" 케이스, 낮은 confidence 케이스 집중 검토

**판단 기준:**
| 오류율 | 조치 |
|--------|------|
| < 10% | ✅ 7-1로 진행 (모델링) |
| 10~20% | ⚠️ 오류 패턴 분석 후 판단 |
| > 20% | ❌ 7-2로 진행 (프롬프트 재수정) |

#### 7-1: 이상 없음 → 8단계로 진행

#### 7-2: 이상 있음 → 3단계로 돌아가기
- 오류 패턴 재분석
- 프롬프트 재수정
- 재라벨링

---

### 8단계: 모델 선별 및 아키텍처 설계

**선정 모델:** KcELECTRA-base (`beomi/KcELECTRA-base`)
- KoBERT 대비 4배 빠른 학습, 적은 데이터로도 우수한 성능
- 한국어 리뷰 텍스트에 최적화된 사전학습

#### 모델 아키텍처: Option A (Aspect별 4-class 통합)

**핵심 아이디어:** 기존의 "전체 감성 3-class + aspect 이진 탐지" 구조에서,
**각 aspect별로 감성을 직접 예측**하는 구조로 재설계.

```
예시: "발림성은 좋은데 향이 별로"
  → 사용감/성능 = positive
  → 재질/냄새 = negative
  → 나머지 = none (미존재)
```

**아키텍처:**
```
┌─────────────────────────────────────┐
│  KcELECTRA Encoder (768-dim)        │
│  beomi/KcELECTRA-base               │
└──────────────┬──────────────────────┘
               │ [CLS] pooling
       ┌───────┴───────┐
       ▼               ▼
┌──────────────┐ ┌──────────────────┐
│  Sentiment   │ │  Aspect-Sentiment│
│  Head (보조) │ │  Head (메인)     │
│  Linear→3    │ │  Linear→44       │
│  [B, 3]      │ │  reshape [B,11,4]│
└──────────────┘ └──────────────────┘
  neg/neu/pos     aspect별 none/pos/neu/neg
```

| 구성요소 | 설명 |
|----------|------|
| Encoder | KcELECTRA-base (768-dim hidden) |
| Sentiment Head | `Linear(768 → 3)` — 리뷰 전체 감성 (보조 태스크) |
| Aspect Head | `Linear(768 → 44)` → `reshape [B, 11, 4]` — aspect별 4-class |
| Loss | Masked CrossEntropyLoss (class weight 적용) + Focal Loss 옵션 |
| 출력 | aspect별 `none(0)/positive(1)/neutral(2)/negative(3)` |

**Aspect-Sentiment 라벨 체계:**
| ID | 라벨 | 의미 |
|----|------|------|
| 0 | none | 해당 aspect 미존재 |
| 1 | positive | 긍정 |
| 2 | neutral | 중립 |
| 3 | negative | 부정 |

> **라벨 매핑 변경 이력 (2026-02-24):**
> 기존 코드에서 `0=none, 1=negative, 2=neutral, 3=positive` 순서였으나,
> Wide CSV 인코딩(`SENT_MAP = {'positive': 1, 'neutral': 2, 'negative': 3}`)과 불일치하여
> `0=none, 1=positive, 2=neutral, 3=negative`으로 통일함.
> 모든 소스(config, dataset, model, evaluation, inference)에 일괄 반영 완료.

#### 데이터 포맷: Long vs Wide

학습 데이터는 **Long Format**과 **Wide Format** 두 가지로 관리한다.

**Long Format** (`absa.csv`, 42,177행)
- 1행 = 1개 (리뷰 × aspect) 쌍
- GPT 라벨링 원본 결과를 그대로 보존
- 컬럼: `review_id, product_code, ..., aspect, aspect_sentiment, confidence, reason, ...`

```
review_id | aspect     | aspect_sentiment | confidence
R001      | 사용감/성능 | positive         | 0.92
R001      | 재질/냄새   | negative         | 0.78
R002      | 가격/가성비 | positive         | 0.85
```

**Wide Format** (`absa_wide_train.csv`, 25,927행 — 실제 학습 데이터)
- 1행 = 1개 리뷰, 11개 aspect가 컬럼으로 펼쳐짐
- 각 aspect마다 `label_<aspect>` (0~3) + `mask_<aspect>` (0/1) + `ambig_<aspect>` (0/1) 3개 컬럼
- `absa_wide.csv` (25,959행)에서 골든셋 리뷰 제거 + ambig 컬럼 추가

```
review_id | text | label_사용감_성능 | mask_사용감_성능 | ambig_사용감_성능 | ...
R001      | ...  | 1 (positive)      | 1               | 0                 | ...
R002      | ...  | 0 (none)          | 0               | 1                 | ...
```

**Wide 인코딩 규칙:**
| 컬럼 접두사 | 값 | 의미 |
|-------------|-----|------|
| `label_*` = 0 | none | 해당 aspect 미존재 |
| `label_*` = 1 | positive | 긍정 |
| `label_*` = 2 | neutral | 중립 |
| `label_*` = 3 | negative | 부정 |
| `mask_*` = 1 | 확실 | loss 계산 포함 |
| `mask_*` = 0 | 불확실 | loss 계산 제외 |
| `ambig_*` = 1 | 모호 | 충돌/불확실 표시 |

**mask=0이 되는 경우:**
- 동일 리뷰에서 같은 aspect에 서로 다른 감성이 충돌 (예: positive + negative)
- NaN 또는 결측 라벨

**Long → Wide 변환:** `s4_dataset.py`의 `load_and_group_csv()` / `_load_wide_csv()` 함수

**골든셋 (Golden Set):**
- Long: `golden_set.csv` (2,955행)
- Wide: `golden_set_wide.csv` (1,256행)
- 2단계 파인튜닝(Stage 2)에서 사용

#### 데이터 전처리

**입력 데이터:**
- Stage 1 학습: `absa_wide_train.csv` (25,927 리뷰, 골든셋 제외 + ambig 컬럼 포함)
- Stage 2 파인튜닝: `golden_set_wide.csv` (1,256 리뷰)
- 참고: `absa_wide.csv` (25,959 리뷰)는 골든셋 포함 전체, `absa.csv`는 Long 원본

**Long Format 그룹화 프로세스 (`load_and_group_csv()`):**
```
CSV 행 단위 (review_id, aspect, aspect_sentiment)
       ↓
충돌 감지: 동일 (review_id, aspect)에 다른 sentiment → mask=0, label=0
       ↓
리뷰 단위 그룹화 → 11-dim label + 11-dim mask
       ↓
ABSADataset 생성 (texts, sentiment_labels, aspect_labels, aspect_masks)
```

**전처리 규칙:**
- `review_sentiment == "mixed"` → 해당 리뷰 제거
- `aspect == "미분류"` → 항상 `neutral(2)`로 강제 매핑
- 동일 (review_id, aspect) 충돌 → `label=0(none), mask=0` (loss 제외)
- 층화 샘플링 기반 Train/Val/Test 분할 (70/15/15)

| 분할 | 리뷰 수 | 비율 |
|------|---------|------|
| Train | ~18,171 | 70% |
| Val | ~3,894 | 15% |
| Test | ~3,894 | 15% |

#### 클래스 불균형 처리

**문제:** `none` 클래스가 대부분의 aspect에서 90% 이상 차지

| 전략 | 설명 |
|------|------|
| Masked CE Loss | `aspect_mask=1`인 셀만 loss 계산에 포함 (불확실한 셀 제외) |
| Class Weight | aspect 4-class에 대한 역빈도 가중치 자동 계산 (mask=1 셀 기준) |
| Focal Loss | 옵션: `focal_gamma=2.0` 으로 쉬운 샘플 가중치 감소 |
| Per-Aspect None-Threshold | 학습 후 **best_model.pt** 기준으로 aspect별 최적 threshold grid search |

**Masked CE Loss 구현:**
```python
# CE(reduction='none') → [B*11] → reshape [B, 11]
per_cell = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")
per_cell = per_cell.view(B, 11)

# mask=1인 셀만 합산
masked_loss = (per_cell * aspect_mask).sum() / aspect_mask.sum()
```

**None-Threshold 튜닝:**
```
1. 학습 완료 후 best_model.pt 로드
2. Val set에서 aspect별 grid search (mask=1인 셀만 대상)
3. 결과를 best_model.pt에 저장

각 aspect별로 P(none) >= threshold → none 예측
                  < threshold → argmax(pos/neu/neg) 예측

Grid search: threshold ∈ [0.1, 0.95], step=0.05
최적화 기준: aspect별 4-class macro F1
```

#### 평가 메트릭

| 메트릭 | 설명 |
|--------|------|
| Sentiment F1 (3-class macro) | 리뷰 전체 감성 분류 성능 |
| Aspect-Sentiment F1 (4-class macro) | aspect별 4-class 분류 성능 (메인 지표) |
| Aspect Detection F1 | none vs not-none 이진 분류 성능 |
| Per-Aspect F1 | 11개 aspect 각각의 F1 |

#### 추론 출력 형식

```json
{
    "sentiment": "positive",
    "sentiment_score": 0.85,
    "aspect_sentiments": [
        {"aspect": "사용감/성능", "sentiment": "positive", "confidence": 0.92},
        {"aspect": "재질/냄새", "sentiment": "negative", "confidence": 0.78}
    ]
}
```

---

### 9단계: 전체 리뷰 적용 (300,000개)

**방법:**
1. 8단계에서 선별된 모델로 학습 완료
2. 전체 300,000개 리뷰에 추론 적용
3. 결과 저장

**예상 소요 시간:**
| 모델 | 추론 시간 (300K) |
|------|-----------------|
| LSTM | ~2시간 |
| KoBERT | ~4시간 |
| KoELECTRA | ~2시간 |
| LightGBM | ~10분 |

**비용:** $0 (로컬 실행)

---

### 10단계: 연착륙 상품 분석

**분석 목표:**
> "6개월 이상 꾸준히 매출을 발생시키는 효자 상품" 발굴

**분석 기준:**

1. **스킨케어 카테고리 필터링**
   - 연착륙 제품의 83.3%가 스킨케어

2. **긍정 Aspect 기준**
   ```
   - 사용감/성능: positive 비율 높음
   - 재질/냄새: "자극 없음", "순함" 키워드
   - 재구매: positive 비율 높음
   ```

3. **부정 Aspect 회피**
   ```
   - 품질/퀄리티: negative 비율 낮음
   - 색상/발색: 기대 불일치 적음
   ```

**출력:**
- 연착륙 후보 상품 리스트
- 상품별 Aspect-Sentiment 분포
- B2B 입점 제안용 데이터

---

## 파일 구조

```
06_analysis/03_ABSA/
├── 01_raw_data/
│   ├── sampled_reviews_20k.csv              # 샘플링 데이터
│   └── sampled_reviews_part_*.csv           # 팀별 분할 데이터
├── 02_processed_data/
│   ├── interim/
│   │   ├── v1/                              # 팀별 3단계 라벨링 결과
│   │   │   ├── step1_team*_bulk_labels.csv
│   │   │   ├── step2_team*_reviewed_labels.csv
│   │   │   └── step3_team*_gold_set.csv
│   │   ├── v2/                              # GPT-4o Batch 결과
│   │   ├── v3/                              # 후처리 중간 버전
│   │   └── v4/
│   │       └── aspect_all_merged_v4.csv     # 최종 병합 (Long)
│   ├── final/                               # ★ 학습용 최종 데이터
│   │   ├── absa.csv                         # Long format 전체 (42,177행)
│   │   ├── absa_wide.csv                    # Wide format 전체 (25,959행)
│   │   ├── absa_wide_train.csv              # ★ Stage 1 학습 데이터 (25,927행, ambig 포함)
│   │   ├── absa_wide_invalid_rows.csv       # 변환 시 제외된 행
│   │   └── golden/                          # 골든셋
│   │       ├── golden_set.csv               # Long (2,955행)
│   │       ├── golden_set_wide.csv          # Wide (1,256행)
│   │       ├── golden_set_wide_full_eval.csv
│   │       └── golden_set_wide_partial_eval.csv
│   └── validation/                          # 검증 데이터
├── 03_notebooks/                            # 분석 노트북
├── 05_src/                                  # 소스 코드 (→ RQ_absa/ 하드링크)
│   ├── s1_config.py                         # 설정, 라벨 매핑, 하이퍼파라미터
│   ├── s2_sampling.py                       # 층화 샘플링
│   ├── s3_labeling.py                       # GPT 라벨링
│   ├── s4_dataset.py                        # Dataset, Long/Wide 로더
│   ├── s5_model.py                          # Multi-task 모델, Masked CE Loss
│   ├── s6_train.py                          # 학습 루프, 2단계 파인튜닝
│   ├── s7_evaluation.py                     # 평가, Threshold 튜닝
│   └── s8_inference.py                      # 추론 파이프라인
├── 06_models/                               # 모델 체크포인트
├── 07_scripts/                              # 실행 스크립트
└── logs/                                    # 로그
```

> **참고:** `05_src/`와 `RQ_absa/`(프로젝트 루트)는 하드링크로 연결되어 있어
> 한쪽 파일을 수정하면 다른 쪽에도 자동 반영됩니다.

---

## 비용 요약

| 단계 | 모델 | 데이터 수 | 비용 |
|------|------|----------|------|
| 2단계 | GPT-4o-mini | 20,000 | $2.87 |
| 5단계 | GPT-4o | 430 (Golden Set) | ~$0.50 |
| 6단계 | GPT-4o Batch | 20,000 | $95.70 |
| 9단계 | ML 모델 | 300,000 | $0 |
| **총합** | | | **~$99** |

**GPT-4o로 300,000개 직접 라벨링 시:** ~$1,400
**절감액:** ~$1,300 (93% 절감)

---

## 진행 상황

1. [x] 배치 작업 완료 (GPT-4o Batch API)
2. [x] 결과 파일 병합 → `absa.csv` (Long) + `absa_wide.csv` (Wide)
3. [x] 샘플 검수 및 EDA 완료
4. [x] 모델 아키텍처 설계 (Option A: aspect별 4-class)
5. [x] 소스 코드 구현 (s1_config ~ s8_inference)
6. [x] 데이터 전처리 규칙 확정 (mixed 제거, 미분류 neutral 강제)
7. [x] 라벨 매핑 통일 (Wide CSV 기준: 0=none, 1=pos, 2=neu, 3=neg)
8. [x] Masked CE Loss + aspect_mask 파이프라인 구현
9. [x] 학습 루프 버그 수정 (grad_accum, threshold 튜닝, checkpoint 메타)
10. [ ] KcELECTRA 모델 학습 실행
11. [ ] Val set에서 per-aspect none-threshold 튜닝
12. [ ] Test set 최종 평가
13. [ ] 전체 리뷰 추론 (300,000개)
14. [ ] 연착륙 상품 분석

---

## 코드 패치 이력

### v2 (2026-02-24) — 라벨 통일 + Masked CE + 학습 루프 수정

3건의 커밋으로 분리 반영.

#### Commit 1: `refactor` — 소스 파일 네이밍 표준화 + 라벨 매핑 통일

| 변경 | 내용 |
|------|------|
| 파일 리네이밍 | `config.py` → `s1_config.py`, `sampling.py` → `s2_sampling.py`, `labeling.py` → `s3_labeling.py` |
| 라벨 매핑 | `ASPECT_SENTIMENT_LABELS`를 `["none", "negative", "neutral", "positive"]` → `["none", "positive", "neutral", "negative"]`으로 변경 |
| 사유 | Wide CSV의 `SENT_MAP = {'positive': 1, 'neutral': 2, 'negative': 3}` 인코딩과 일치시킴 |

#### Commit 2: `feat` — Masked CE Loss + aspect_mask 데이터 파이프라인

| 파일 | 변경 내용 |
|------|----------|
| `s4_dataset.py` | `ABSADataset`에 `aspect_masks` 파라미터 추가, `__getitem__`에서 `aspect_mask` FloatTensor 반환 |
| | `load_and_group_csv()`: 동일 (review_id, aspect) 충돌 감지 → mask=0, label=0 처리 |
| | `_load_wide_csv()`: NaN 방어 (`pd.isna()` 체크) |
| | `create_dataset_from_wide()`, `create_datasets_from_wide()` 함수 추가 |
| `s5_model.py` | `forward()`에 `aspect_mask` 파라미터 추가 |
| | `_compute_masked_aspect_loss()`: mask=1인 셀만 loss 계산 (Focal Loss 호환) |
| | `compute_aspect_class_weights()`: mask=1 셀 기준 가중치 계산 |
| `s7_evaluation.py` | `evaluate()`, `_evaluate_aspect_sentiment()`, `_evaluate_aspect_detection()` 모두 mask 필터링 |
| | `tune_none_thresholds()`: mask=1 셀만 대상으로 threshold grid search |
| | `collect_predictions()`: batch에서 `aspect_mask` 수집 |
| `s8_inference.py` | `ASPECT_SENTIMENT_ID_TO_LABEL` import 자동 반영 (코드 변경 없음) |

#### Commit 3: `fix` — 학습 루프 버그 수정 및 개선

| 파일 | 변경 내용 |
|------|----------|
| `s6_train.py` | **P0 — grad_accum 중복 실행:** `did_update` 플래그 + `is_last_batch` 체크로 logging/eval/save가 실제 optimizer step 시에만 실행 |
| | **Threshold 튜닝 대상:** 마지막 모델 → `best_model.pt` 로드 후 튜닝, 결과를 best_model.pt에 저장 |
| | **AdamW 교체:** `transformers.AdamW` (deprecated) → `torch.optim.AdamW` |
| | **Checkpoint 메타데이터:** `save_checkpoint()`에 `label_meta` 추가 (`ASPECT_LABELS`, `ASPECT_SENTIMENT_TO_ID`) |
| | **load_checkpoint() 안전성:** `scheduler is not None` 가드, `.get()` 안전 접근, 라벨 매핑 불일치 경고 |
| | **class weight 계산:** `create_model_with_class_weights()`에서 불필요한 토크나이징 제거 → dataset 속성 직접 접근 |

---

## 소스 모듈 역할

| 모듈 | 역할 | 주요 함수/클래스 |
|------|------|-----------------|
| `s1_config.py` | 전역 설정 | 라벨 매핑, 경로, 하이퍼파라미터 |
| `s2_sampling.py` | 층화 샘플링 | 3단계 층화 샘플링 (대분류→소분류→감성) |
| `s3_labeling.py` | GPT 라벨링 | OpenAI API 호출, Batch API 처리 |
| `s4_dataset.py` | 데이터 로딩 | `ABSADataset`, `load_and_group_csv()`, `_load_wide_csv()`, `create_datasets_from_wide()` |
| `s5_model.py` | 모델 정의 | `MultiTaskABSAModel`, `FocalLoss`, `_compute_masked_aspect_loss()` |
| `s6_train.py` | 학습 루프 | `ABSATrainer.train()`, `.fine_tune()`, `_load_best_and_tune_thresholds()` |
| `s7_evaluation.py` | 평가 | `ABSAEvaluator.evaluate()`, `tune_none_thresholds()`, `evaluate_test_set()` |
| `s8_inference.py` | 추론 | `ABSAInference.predict_batch()`, 전체 리뷰 추론 |

---

*문서 작성일: 2026-02-14*
*최종 수정일: 2026-02-24*
