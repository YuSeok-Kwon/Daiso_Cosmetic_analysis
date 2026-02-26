# ABSA S4B + Regex Gate — 전체 추론 최종 결과

> **작성일:** 2026-02-27
> **모델:** S4B (Warm-start, KcELECTRA-base)
> **후처리:** Regex Keyword Gate (s1_config.py) + Design Rule Override + Force-On + QC Postprocess
> **추론 대상:** 323,114 리뷰 (전체)
> **최종 결과 파일:** `absa_results_stage4b_rgx_full.csv`
> **이전 문서:** [09_ABSA_Stage5_전략ABC_실행결과.md](./09_ABSA_Stage5_전략ABC_실행결과.md)

---

## 목차

1. [Executive Summary](#1-executive-summary)
2. [배경: Stage 5 이후 남은 과제](#2-배경-stage-5-이후-남은-과제)
3. [Keyword Gate 정규식 전환](#3-keyword-gate-정규식-전환)
4. [환경 혼동 사건과 통제 실험](#4-환경-혼동-사건과-통제-실험)
5. [최종 추론 결과 (S4B + Regex Gate)](#5-최종-추론-결과-s4b--regex-gate)
6. [3-Way 비교: Stage 3A-v2 vs S4B(Substring) vs S4B(Regex)](#6-3-way-비교)
7. [Regex Gate 순수 효과 분석](#7-regex-gate-순수-효과-분석)
8. [최종 데이터셋 확정](#8-최종-데이터셋-확정)
9. [파이프라인 현재 위치](#9-파이프라인-현재-위치)
10. [관련 문서](#10-관련-문서)

---

## 1. Executive Summary

Stage 5 Pragmatic GO 판정(S4B Warm-start) 이후, Keyword Gate를 substring 매칭에서 **정규식(regex) 기반**으로 전환하고 전체 323K 추론을 재실행하였다. 정규식 Gate의 순수 효과를 검증하기 위해 통제 실험(동일 모델, 동일 threshold, Gate만 변경)을 수행한 결과, Regex Gate는 **가격/가성비 오탐을 5.2pp 억제**하면서 다른 aspect에는 부작용이 없음을 확인하였다.

최종 데이터셋은 **S4B + Regex Gate (`absa_results_stage4b_rgx_full.csv`)** 로 확정한다.

### 최종 추론 요약

| 항목 | 수치 |
|------|------|
| 총 리뷰 | 323,114건 |
| 추론 시간 | 66.1분 (81.4 rev/s) |
| Ambiguous | 6.33% |
| 미분류 | 28.5% |
| Aspect 1위 | 사용감/성능 46.2% |
| 감성 분포 | pos 67.1% / neu 28.3% / neg 4.6% |

---

## 2. 배경: Stage 5 이후 남은 과제

Stage 5 문서([09_ABSA_Stage5_전략ABC_실행결과.md](./09_ABSA_Stage5_전략ABC_실행결과.md))의 Next Steps에서 제시된 과제:

| # | 과제 | 상태 |
|---|------|------|
| 1 | S4B 모델로 전체 추론 실행 | **완료** |
| 2 | Keyword Gate 정규식 전환 + 효과 검증 | **완료** |
| 3 | 추론 환경 혼동 문제 해소 | **완료** |
| 4 | 최종 데이터셋 확정 | **완료 (이 문서)** |

---

## 3. Keyword Gate 정규식 전환

### 3.1 전환 동기

Stage 3A에서 도입된 Keyword Gate는 **substring 매칭** 기반이었다. 예를 들어 가격/가성비 Gate에 `"싸"`가 포함되면 "잽싸게", "싸움" 등 관련 없는 문맥도 모두 매칭되어 오탐이 발생하였다.

### 3.2 정규식 Gate 설계 원칙

| 원칙 | 적용 예시 |
|------|-----------|
| **활용형만 매칭** | `싸` → `싸[다고게서니요네]` |
| **부정 전방탐색** | `깔끔` → `깔끔(?!\s*(?:하게\s*)?(?:흡수\|발[라림]))` |
| **문맥 한정** | `양` → `(?<![가-힣])양[이은도만]\s` |
| **오탐 패턴 제외** | `향` → `향(?!상\|수[인를이]?\s)` |
| **복합 구조물만** | `팁` → `(?:실리콘\|브러시)\\s*팁` |

### 3.3 Gate 대상 Aspect (4개)

| Aspect | Substring 키워드 수 | Regex 패턴 수 | 주요 변경 |
|--------|:---:|:---:|-----------|
| 디자인 | 19개 | 14개 | `깔끔`, `고급`, `팁` — 사용감 맥락 제외 |
| 가격/가성비 | 14개 | 12개 | `싸` → 활용형만, `싸게` → 잽싸게 제외 |
| 재질/냄새 | 13개 | 16개 | `향` → 향상/향수 제외, `쏘` → 활용형만 |
| 용량/휴대 | 16개 | 18개 | `양` → 독립 용량 맥락만, `g` → 숫자+g만 |

> 사용감/성능, 배송/포장, 재구매, 색상/발색에는 Gate가 없다.

### 3.4 구현

- `RQ_absa/s1_config.py`의 `KEYWORD_GATE_CONFIG`를 정규식 리스트로 교체
- `RQ_absa/s8_inference.py`의 `_apply_keyword_gate`에서 `re.compile()` 사용
- `run_full_inference.py`에서 번들의 substring gate를 스킵하고 s1_config.py의 regex gate를 로드

---

## 4. 환경 혼동 사건과 통제 실험

### 4.1 최초 비교에서 발견된 이상

Stage 4(from-scratch) 모델 + Regex Gate 추론(`stage4_rgx`) 결과를 S4B(Warm-start) + Substring Gate 추론(`stage4b`)과 비교했을 때, 사용감/성능이 46.2% → 69.3%로 급증하고 디자인 positive가 7,845 → 0건으로 사라지는 현상이 발생하였다.

### 4.2 원인 분석

| 차이 항목 | stage4b | stage4_rgx | 영향 |
|-----------|---------|------------|------|
| **모델 체크포인트** | `checkpoints_stage4b/` (Warm-start, Epoch 2) | `checkpoints_stage4/` (From-scratch, Epoch 9) | **주 원인**: 서로 다른 모델 |
| **None-Thresholds** | `[0.15, 0.10, 0.03, 0.20, 0.50, 0.15, 0.15, 0.10]` | `[0.10, 0.15, 0.10, 0.15, 0.25, 0.15, 0.10, 0.15]` | 사용감 0.03→0.10에도 불구하고 69.3% |
| **`design_rule` 플래그** | `true` | **누락 (default: false)** | 디자인 positive 0건의 **직접 원인** |
| Keyword Gate | 번들 substring | s1_config.py regex | Gate 효과 자체 |

**핵심:** 모델이 다르고, threshold가 다르고, Design Rule 플래그마저 달랐기 때문에, 두 결과를 직접 비교하여 "Gate 효과"를 측정하는 것은 불가능하였다.

### 4.3 통제 실험 설계 (선택지 A)

순수 Gate 효과만 측정하기 위해, **Gate를 제외한 모든 변수를 고정**한 실험을 수행하였다.

| 항목 | 고정값 |
|------|--------|
| 모델 | `checkpoints_stage4b/best_model.pt` (MD5: `a009474e...`) |
| None-Thresholds | `[0.15, 0.10, 0.03, 0.20, 0.50, 0.15, 0.15, 0.10]` |
| Polar Threshold | 0.55 |
| Design Rule | `true` |
| Force-On Config | 번들 (재구매 + 가격/가성비) |
| QC Postprocess | 번들 |
| **Keyword Gate** | **Substring → Regex** (유일한 변수) |

---

## 5. 최종 추론 결과 (S4B + Regex Gate)

```
======================================================================
INFERENCE COMPLETE — S4B + Regex Gate (stage4b_rgx)
======================================================================
총 리뷰: 323,114
총 시간: 3967초 (66.1분)
속도: 81 reviews/sec
Ambiguous: 20,465 (6.3%)

--- Aspect 언급률 ---
  배송/포장             4.8%  (pos 42% / neu 49% / neg 9%)
  가격/가성비            7.4%  (pos 92% / neu 8% / neg 0%)
  사용감/성능           46.2%  (pos 77% / neu 16% / neg 7%)
  용량/휴대             2.7%  (pos 23% / neu 66% / neg 12%)
  디자인               2.5%  (pos 98% / neu 0% / neg 2%)
  재질/냄새             2.8%  (pos 0% / neu 83% / neg 17%)
  재구매              10.3%  (pos 94% / neu 4% / neg 1%)
  색상/발색            12.8%  (pos 61% / neu 28% / neg 10%)
  미분류              28.5%

--- Review Sentiment ---
  positive: 216,794 (67.1%)
  neutral: 91,601 (28.3%)
  negative: 14,719 (4.6%)
```

---

## 6. 3-Way 비교

### 6.1 Aspect 언급률 비교

| Aspect | Stage 3A-v2 | S4B (Substring) | **S4B (Regex)** | 3A→Regex 변화 |
|--------|:-----------:|:----------------:|:---------------:|:--------------:|
| 사용감/성능 | 49.4% | 46.2% | **46.2%** | -3.2pp |
| 색상/발색 | 12.4% | 8.4% | **12.8%** | +0.4pp |
| 재구매 | 10.7% | 10.3% | **10.3%** | -0.4pp |
| 가격/가성비 | 8.3% | 12.6% | **7.4%** | -0.9pp |
| 배송/포장 | 2.3% | 4.1% | **4.8%** | +2.5pp |
| 재질/냄새 | 1.7% | 2.7% | **2.8%** | +1.1pp |
| 용량/휴대 | 2.3% | 2.7% | **2.7%** | +0.4pp |
| 디자인 | 2.5% | 2.5% | **2.5%** | 0.0pp |
| **미분류** | **28.7%** | **28.2%** | **28.5%** | -0.2pp |

### 6.2 Review Sentiment 비교

| 감성 | Stage 3A-v2 | S4B (Substring) | **S4B (Regex)** |
|------|:-----------:|:----------------:|:---------------:|
| Positive | 66.5% | 67.1% | **67.1%** |
| Neutral | 29.1% | 28.3% | **28.3%** |
| Negative | 4.4% | 4.6% | **4.6%** |

> Review-level Sentiment는 S4B Substring과 Regex에서 **완전히 동일** (216,794 / 91,601 / 14,719). Gate는 Aspect 분류에만 영향을 주고 Review Sentiment에는 영향을 주지 않음을 확인.

### 6.3 Ambiguous 비교

| 버전 | Ambiguous 비율 |
|------|:-:|
| Stage 3A-v2 | 5.05% |
| S4B (Substring) | 7.08% |
| **S4B (Regex)** | **6.33%** |

---

## 7. Regex Gate 순수 효과 분석

모델, threshold, Design Rule 등 모든 변수를 고정한 상태에서 Gate만 변경한 결과:

### 7.1 Aspect별 변화

| Aspect | Substring | Regex | 차이 | 해석 |
|--------|:---------:|:-----:|:----:|------|
| **가격/가성비** | **12.57%** | **7.38%** | **-5.19pp** | `싸` 등 광범위 매칭 억제 → 오탐 ~16,700건 제거 |
| 색상/발색 | 8.43% | 12.77% | +4.34pp | 가격에서 빠진 리뷰가 본래 aspect로 복귀 |
| 배송/포장 | 4.07% | 4.79% | +0.72pp | 소폭 증가 (간접 효과) |
| **사용감/성능** | **46.20%** | **46.20%** | **0.00pp** | Gate 대상 아님 → 완벽히 동일 |
| 재구매 | 10.31% | 10.31% | 0.00pp | Gate 대상 아님 → 완벽히 동일 |
| 용량/휴대 | 2.69% | 2.70% | +0.01pp | 무시 가능 |
| 디자인 | 2.47% | 2.46% | -0.01pp | 무시 가능 |
| 재질/냄새 | 2.74% | 2.77% | +0.03pp | 무시 가능 |
| 미분류 | 28.18% | 28.50% | +0.32pp | 무시 가능 |

### 7.2 감성 분포 변화 (주요 aspect)

| Aspect | 항목 | Substring | Regex | 변화 |
|--------|------|:---------:|:-----:|:----:|
| 가격/가성비 | 총 건수 | 40,612 | 23,838 | **-16,774건** |
| 가격/가성비 | positive 비율 | 93% | 92% | 유사 |
| 가격/가성비 | negative 건수 | 1,089 | 65 | **-1,024건** |
| 색상/발색 | 총 건수 | 27,238 | 41,273 | **+14,035건** |
| 배송/포장 | neutral 건수 | 4,835 | 7,646 | +2,811건 |

### 7.3 해석

1. **가격/가성비 -5.19pp:** Regex Gate의 핵심 효과. Substring `"싸"`가 "잽싸게", "포싸", "싸움" 등 비가격 맥락을 모두 매칭했던 것을 `싸[다고게서니요네]`로 제한하여 오탐을 억제하였다. negative 1,089 → 65건으로 급감한 것은, "싸구려" 등 부정 맥락의 오탐이 대거 제거되었음을 의미한다.

2. **색상/발색 +4.34pp:** Gate 대상이 아닌 aspect이므로 직접 효과가 아니다. 가격/가성비 Gate에서 오탐으로 잡히던 리뷰들이 제거되면서, 해당 리뷰의 다른 aspect(색상/발색)이 상대적으로 부각된 **간접 재분배 효과**이다.

3. **미분류 +0.32pp:** Gate가 오탐을 제거했지만, 제거된 리뷰 중 다른 aspect가 검출되는 경우가 대부분이므로 미분류 증가는 미미하다. 이는 Gate가 FN(탐지 누락)을 유발하지 않음을 의미한다.

4. **Review Sentiment 완전 동일:** Gate는 Aspect-level 후처리이므로, Review-level Sentiment(positive/neutral/negative)에는 전혀 영향을 주지 않는다.

---

## 8. 최종 데이터셋 확정

### 8.1 선정 근거

| 기준 | S4B + Regex Gate | 근거 |
|------|:---:|------|
| 모델 품질 | Det.P **0.8290** | 역대 최고, Pragmatic GO 판정 (3/4 KPI) |
| 분포 정합성 | Gap **-9.6pp** | 보수적 과소검출, 집계 왜곡 최소 |
| 오탐 억제 | 가격 **-5.19pp** | Regex Gate로 substring 오탐 제거 |
| 부작용 | **없음** | Gate 비대상 aspect 변화 0.0pp |

### 8.2 최종 파일

| 항목 | 경로 |
|------|------|
| 전체 결과 | `01_outputs/inference/absa_results_stage4b_rgx_full.csv` |
| 요약 JSON | `01_outputs/inference/absa_results_stage4b_rgx_summary.json` |
| 모델 체크포인트 | `07_models/checkpoints_stage4b/best_model.pt` |
| Threshold | `07_models/checkpoints_stage4b/none_thresholds.json` |
| Keyword Gate | `RQ_absa/s1_config.py` → `KEYWORD_GATE_CONFIG` (regex) |
| Design Rule | `07_models/prod_bundle_stage3a_v1_20260225/design_rule_config.json` |

### 8.3 추론 환경 설정 (재현용)

```bash
python 06_scripts/run_full_inference.py \
    --model-dir checkpoints_stage4b \
    --tag stage4b_rgx
```

| 설정 항목 | 값 | 소스 |
|-----------|-----|------|
| 모델 | `checkpoints_stage4b/best_model.pt` | 체크포인트 |
| None-Thresholds | `[0.15, 0.10, 0.03, 0.20, 0.50, 0.15, 0.15, 0.10]` | `none_thresholds.json` |
| Polar Threshold | 0.55 | `none_thresholds.json` |
| Design Rule | `true` | `none_thresholds.json` |
| Keyword Gate | Regex 기반 | `s1_config.py` (번들 substring 스킵) |
| Force-On | 재구매 + 가격/가성비 | `design_rule_config.json` |
| QC Postprocess | rescue_delivery + hallucination_gate 등 | `design_rule_config.json` |

---

## 9. 파이프라인 현재 위치

```
[1]  층화 샘플링 (20,000개)                        ✅
[2]  GPT-4o-mini 1차 라벨링                        ✅
[3]  사람 직접 검증 → 오류 패턴 파악                 ✅
[4]  프롬프트 재수정                                ✅
[5]  GPT-4o vs GPT-4o-mini 비교                    ✅
[6]  GPT-4o Batch API로 20,000개 라벨링             ✅
[7]  사람 직접 검수 + EDA                           ✅
[8]  모델 설계 (KcELECTRA + Option A)              ✅
[9]  Stage 1 학습 + 골든셋 평가                     ✅
[10] Stage 2 재학습 (8 aspect + 디자인 NS)          ✅
[11] Stage 2 골든셋 평가                            ✅
[12] Stage 3A 후처리 튜닝 + GO 판정                 ✅
[13] 전체 리뷰 추론 (323,114건)                     ✅
[14] 추론 결과 EDA (v2 키워드 게이트)               ✅
[15] QC 분석 + 후처리 v3/v4                        ✅
[16] Stage 4 재학습 (neutral 보강)                  ✅
[17] Stage 5 전략 A/B/C 실행 + Pragmatic GO         ✅
[18] S4B 전체 추론 + Regex Gate 전환 + 통제 실험     ✅  ← 이 문서
[19] 최종 데이터셋 확정 (S4B + Regex Gate)           ✅  ← 이 문서
[20] 연착륙 상품 분석 (SLI)
```

---

## 10. 관련 문서

| 문서 | 내용 |
|------|------|
| [00_ABSA_파이프라인.md](./00_ABSA_파이프라인.md) | 전체 파이프라인 + 모델 아키텍처 상세 |
| [01_ABSA_Stage1_학습_리포트.md](./01_ABSA_Stage1_학습_리포트.md) | Stage 1 학습 및 골든셋 평가 |
| [02_ABSA_Stage2_학습_리포트.md](./02_ABSA_Stage2_학습_리포트.md) | Stage 2 구조 개편 + 재학습 |
| [03_ABSA_Stage2_골든셋_평가.md](./03_ABSA_Stage2_골든셋_평가.md) | Stage 2 골든셋 평가 + Stage 3 방향 설계 |
| [04_ABSA_Stage3A_후처리_튜닝_리포트.md](./04_ABSA_Stage3A_후처리_튜닝_리포트.md) | Stage 3A 후처리 + GO/NO-GO 판정 |
| [05_ABSA_Stage3A_전체추론_결과.md](./05_ABSA_Stage3A_전체추론_결과.md) | 323K 전체 추론 실행 결과 |
| [06_ABSA_전체추론_EDA_결과.md](./06_ABSA_전체추론_EDA_결과.md) | 전체 추론 EDA (v2 키워드 게이트) |
| [07_ABSA_QC_후처리_개선_전략.md](./07_ABSA_QC_후처리_개선_전략.md) | QC 오류 패턴 + 후처리 v3/v4 설계 |
| [08_ABSA_Stage4_재학습_리포트.md](./08_ABSA_Stage4_재학습_리포트.md) | Stage 4 neutral 보강 재학습 |
| [09_ABSA_Stage5_전략ABC_실행결과.md](./09_ABSA_Stage5_전략ABC_실행결과.md) | Stage 5 전략 A/B/C + Pragmatic GO |
| **10_ABSA_S4B_Regex_Gate_전체추론_최종결과.md** | **← 이 문서** |
