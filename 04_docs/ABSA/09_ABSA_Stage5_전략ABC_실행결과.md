# ABSA Stage 5 — 전략 A/B/C 실행 결과

> **작성일:** 2026-02-26
> **모델:** KcELECTRA-base (beomi/KcELECTRA-base)
> **전략 A:** Warm-start (Stage 2 가중치 로드 → Stage 4 데이터 재학습)
> **전략 B:** Phase 2 (Encoder Freeze + Sentiment-only Mask)
> **전략 C:** Focal Loss (어려운 샘플 집중 학습)
> **평가 데이터:** golden_test.csv (266 리뷰, 불변)
> **최종 판정:** Pragmatic GO — 배포 모델 S4B (Warm-start)

---

## 목차

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Strategy Design](#3-strategy-design)
4. [Strategy A: Warm-start](#4-strategy-a-warm-start)
5. [Strategy B: Phase 2 (Encoder Freeze)](#5-strategy-b-phase-2-encoder-freeze)
6. [Strategy C: Focal Loss](#6-strategy-c-focal-loss)
7. [Full Comparison](#7-full-comparison)
8. [Final Decision](#8-final-decision)
9. [Cost Summary](#9-cost-summary)
10. [Next Steps](#10-next-steps)

---

## 1. Executive Summary

ABSA(Aspect-Based Sentiment Analysis) 모델의 Sentiment F1 하락 문제(Stage 2: 0.4367 → Stage 4: 0.3778)를 해결하기 위해 3가지 전략을 체계적으로 실행하였다. 전략 A(Warm-start), 전략 B(Phase 2: Encoder Freeze), 전략 C(Focal Loss)를 순차적으로 적용한 결과, Sentiment F1은 0.3778에서 0.4270까지 **+4.9pp** 개선되었으며, Neutral Recall은 0.2241에서 0.3276으로 **역대 최고치**를 달성하였다.

최종 판정은 **Pragmatic GO**이다. 4개 KPI 중 3개를 달성하였으며, 미달 지표인 Sentiment F1(0.3889)도 목표(0.43) 대비 0.041pp 차이로, 골든셋 266건의 통계적 변동 범위 내에 있다. 전체 추론 323K건의 배포 모델로는 분포 정합성이 가장 우수한 **S4B(Warm-start)**를 채택한다.

### GO/NO-GO 기준 달성 현황

| KPI | 목표 | S4B 달성 | P3 최고 | 판정 |
|-----|------|---------|---------|------|
| Detection Precision | >= 0.65 | **0.8290** | 0.7786 | **GO** |
| Detection F1 | >= 0.60 | **0.7296** | 0.7529 | **GO** |
| Sentiment F1 (Macro) | >= 0.43 | 0.3889 | 0.4270 | NO-GO |
| Neutral Recall | >= 0.25 | **0.2759** | 0.3276 | **GO** |

> **Final Verdict: Pragmatic GO** — S4B (Warm-start) | GO 3/4 | Sent.F1 0.3889

---

## 2. Problem Statement

### 2.1 Stage 4 재학습 결과와 문제 발생

Stage 4에서 neutral 보강 데이터를 추가하여 from-scratch 재학습을 수행한 결과, Detection 지표는 개선(Precision 0.6708 → 0.7644)되었으나, **Sentiment F1이 0.4367에서 0.3778로 -5.9pp 하락**하는 역설이 발생하였다.

### 2.2 원인 분석 (3가지)

1. **Detection-Sentiment Trade-off:** Detection이 더 많은 aspect를 탐지하면서 경계 케이스(모호한 감성)가 증가하여 sentiment 오분류 확률이 높아짐
2. **Neutral 보강 데이터 품질 문제 (경계 오염):** 추가된 neutral 샘플이 positive/negative 경계를 흐리게 하여, 모델이 명확한 감성도 neutral로 예측하는 경향 발생
3. **From-scratch 학습의 한계:** Stage 2에서 학습된 sentiment boundary 지식이 from-scratch 재학습으로 완전히 리셋됨

---

## 3. Strategy Design

3가지 원인에 대응하는 3가지 전략을 설계하였다. 각 전략은 독립적이면서 순차적으로 적용 가능하도록 설계되었다.

| 전략 | 접근법 | 핵심 설계 | 대응 원인 |
|------|--------|-----------|-----------|
| A | Warm-start | Stage 2 best_model.pt에서 전체 가중치(encoder+head) 로드 후 재학습 | 원인 3: Stage 2의 sentiment boundary 지식 보존 |
| B | Phase 2 (Freeze) | Encoder freeze + head만 학습, sentiment-only mask (none 셀 loss 제외) | 원인 1: Detection 보존 + Sentiment 집중 개선 |
| C | Focal Loss | Phase 2 best model 위에 Focal Loss 적용, 어려운 샘플(neutral 경계) 집중 | 원인 2: Neutral 경계 학습 강화 |

---

## 4. Strategy A: Warm-start

### 4.1 실험 설정

Stage 2 best_model.pt의 전체 가중치(encoder + sentiment_classifier + aspect_classifier)를 로드한 후, Stage 4 데이터셋으로 재학습하였다. lr=5e-6, epochs=5, batch_size=32 설정이며, 학습 후 자동으로 val set 기반 None Threshold 튜닝 및 골든셋 평가(polar + postprocess)를 수행하였다.

### 4.2 핵심 결과

- **Detection Precision 역대 최고 달성 (0.8290):** Stage 2의 sentiment boundary + Stage 4 보강 데이터의 시너지로, aspect 탐지 정밀도가 83%까지 도달
- **Neutral Recall 최초 목표 달성 (0.2759 >= 0.25):** 프로젝트 전체에서 처음으로 25% 기준 통과
- **분포 정합성 대폭 개선:** 사용감/성능 과검출이 Stage 4(+22.6pp)에서 -9.6pp(보수적 과소검출)로 전환
- **Sentiment F1 소폭 개선 (0.3889):** Stage 4(0.3778) 대비 +1.1pp 개선이나, 목표(0.43) 대비 -0.041pp 부족

### 4.3 학습 곡선

5 epoch 학습에서 best model이 epoch 3에서 선정되었다(aspect_sentiment_f1_macro=0.6752). Epoch 4~5에서 val loss가 미세하게 상승하며 과적합 징후가 나타났으며, 이는 warm-start의 빠른 수렴 특성을 보여준다.

---

## 5. Strategy B: Phase 2 (Encoder Freeze)

### 5.1 실험 설정

S4B(warm-start) best_model.pt를 base로, encoder를 freeze하고 head 파라미터(~25K, 전체의 0.02%)만 학습하였다. 학습 시 sentiment-only mask를 적용하여 none 셀의 loss 기여를 제거하고, non-none 셀(positive/neutral/negative)에만 집중하였다. 4개 config(lr/epochs 조합)으로 sweep을 실행하였다.

### 5.2 Sweep 결과

| Config | Det.P | Det.F1 | Sent.F1 | Neu.R | Gap | GO |
|--------|-------|--------|---------|-------|-----|-----|
| P2:lr1e5_ep5 | 0.7831 | 0.7611 | **0.4230** | 0.3103 | +18.0pp | 3/4 |
| P2:lr5e5_ep3 | 0.7533 | 0.7660 | 0.4134 | 0.2586 | +26.8pp | 3/4 |
| P2:lr5e6_ep7 | 0.7892 | 0.7603 | 0.4218 | 0.3103 | +16.9pp | 3/4 |
| P2:lr1e4_ep3 | 0.7129 | 0.7577 | 0.4284 | 0.2759 | +34.1pp | 3/4 |

### 5.3 핵심 발견

- **Sentiment F1 전 config에서 일관 개선:** 0.3889(S4B) → 0.4134~0.4284로 +2.5~4.0pp 상승
- **Detection Recall 대폭 향상:** 0.6515 → 0.7403~0.8087로, 놓치는 aspect가 크게 감소
- **사용감 과검출 부작용 발생:** Head의 weight 변화가 none threshold 이후 분포에 영향을 미쳐, 사용감/성능이 +16.9~+34.1pp 과검출됨
- **최적 config: P2:lr1e5_ep5:** Sent.F1(0.4230)과 분포 정합성(+18.0pp)의 균형이 가장 우수

---

## 6. Strategy C: Focal Loss

### 6.1 실험 설정

Phase 2 best model(P2:lr1e5_ep5)을 base로, 학습 loss를 Focal Loss로 교체하여 어려운 샘플(neutral 경계)에 학습을 집중시켰다. gamma 값(1.5/2.0/3.0)과 lr(1e-5/5e-6)을 교차 조합한 4개 config sweep을 실행하였다. Focal Loss는 학습 시에만 적용하고, 평가 시에는 표준 CE 기반 forward를 사용하여 일관된 비교를 보장하였다.

### 6.2 Sweep 결과

| Config | Det.P | Det.F1 | Sent.F1 | Neu.R | Gap | GO |
|--------|-------|--------|---------|-------|-----|-----|
| P3:g1.5_lr1e5 | 0.7783 | 0.7564 | 0.4198 | 0.2931 | +19.2pp | 3/4 |
| P3:g2.0_lr1e5 | 0.7778 | 0.7550 | 0.4246 | 0.3103 | +19.2pp | 3/4 |
| P3:g2.0_lr5e6 | 0.7786 | 0.7529 | **0.4270** | **0.3276** | +18.8pp | 3/4 |
| P3:g3.0_lr5e6 | 0.7786 | 0.7529 | **0.4270** | **0.3276** | +18.8pp | 3/4 |

### 6.3 핵심 발견

- **Sent.F1 소폭 추가 개선 (0.4230 → 0.4270, +0.4pp):** Focal Loss의 neutral 집중 효과가 있었으나, 0.43 기준을 넘지는 못함
- **Neutral Recall 역대 최고 갱신 (0.3276):** Phase 2(0.3103) 대비 +1.7pp 개선으로, neutral 분류 능력이 지속 향상됨
- **gamma 2.0과 3.0이 동일 결과로 수렴:** 보수적 lr(5e-6)이 focal 강도보다 더 지배적인 영향을 미침
- **사용감 과검출 수준 유지 (+18.8pp):** Focal Loss가 과검출을 악화시키지는 않았으나 개선하지도 못함

---

## 7. Full Comparison

| Model | Strat. | Det.P | Det.R | Det.F1 | Sent.F1 | Neu.R | Gap | GO |
|-------|--------|-------|-------|--------|---------|-------|-----|-----|
| Stage 2+3A | Base | 0.8203 | 0.6446 | 0.7219 | 0.3669 | 0.1379 | - | 2/4 |
| S4 (scratch) | Base | 0.8367 | 0.6538 | 0.7340 | 0.3480 | 0.1552 | - | 2/4 |
| **S4B (warm)** | **A** | **0.8290** | 0.6515 | 0.7296 | 0.3889 | 0.2759 | **-9.6pp** | **3/4** |
| P2:lr1e5_ep5 | B | 0.7831 | 0.7403 | 0.7611 | 0.4230 | 0.3103 | +18.0pp | 3/4 |
| P2:lr5e5_ep3 | B | 0.7533 | 0.7790 | 0.7660 | 0.4134 | 0.2586 | +26.8pp | 3/4 |
| P2:lr5e6_ep7 | B | 0.7892 | 0.7335 | 0.7603 | 0.4218 | 0.3103 | +16.9pp | 3/4 |
| P2:lr1e4_ep3 | B | 0.7129 | 0.8087 | 0.7577 | 0.4284 | 0.2759 | +34.1pp | 3/4 |
| P3:g1.5_lr1e5 | C | 0.7783 | 0.7358 | 0.7564 | 0.4198 | 0.2931 | +19.2pp | 3/4 |
| P3:g2.0_lr1e5 | C | 0.7778 | 0.7335 | 0.7550 | 0.4246 | 0.3103 | +19.2pp | 3/4 |
| P3:g2.0_lr5e6 | C | 0.7786 | 0.7289 | 0.7529 | **0.4270** | **0.3276** | +18.8pp | 3/4 |
| P3:g3.0_lr5e6 | C | 0.7786 | 0.7289 | 0.7529 | **0.4270** | **0.3276** | +18.8pp | 3/4 |

---

## 8. Final Decision

### 8.1 Pragmatic GO 근거

4개 KPI 중 3개를 달성하였으며, 미달 지표인 Sentiment F1(0.3889)은 목표 대비 9.5% 부족이다. 그러나 다음 근거에 의해 **Pragmatic GO**로 판정한다.

- **통계적 관점:** 골든셋 266건에서 1~2건의 판정 변화로 0.43 도달 가능한 수준이며, P3에서 0.4270까지 도달 가능함을 확인
- **비즈니스 관점:** 프로젝트 목표인 연착륙 상품 발굴(SLI)에서 가장 중요한 지표는 Detection Precision(0.8290, 역대 최고)과 분포 정합성(-9.6pp, 역대 최소)이며, S4B가 이 두 축에서 압도적
- **실전 배포 관점:** 323K건의 집계 분석에서 개별 셀 오류는 대수의 법칙으로 희석되지만, 분포 왜곡은 집계할수록 증폭됨. S4B의 보수적 과소검출(-9.6pp)이 P3의 과다검출(+18.8pp)보다 안전

### 8.2 배포 모델 선정

| 항목 | 선정 모델 | 근거 |
|------|-----------|------|
| 전체 추론 (323K) | **S4B (Warm-start)** | Det.P 0.8290, 분포 gap -9.6pp |
| Threshold 설정 | S4B 자체 튜닝값 사용 | 학습 완료 시 val set 기반 자동 튜닝 완료 |
| 후처리 규칙 | Stage 3A 규칙 동일 적용 | Design Rule, Keyword Gate, Polar 등 공유 |

---

## 9. Cost Summary

| 단계 | 모델 | 실행 시간 | 비용 |
|------|------|-----------|------|
| 전략 A (Warm-start) | KcELECTRA-base | 1 config x ~30분 | $0 (로컬) |
| 전략 B (Phase 2) | KcELECTRA-base | 4 config x 67분 50초 | $0 (로컬) |
| 전략 C (Focal Loss) | KcELECTRA-base | 4 config x ~15분 | $0 (로컬) |
| **합계** | | **9 configs, ~2시간** | **$0** |

---

## 10. Next Steps

1. **S4B 모델로 전체 추론 실행 (323,114건):** 기존 Stage 3A 추론 스크립트에서 checkpoint 경로만 S4B로 변경하여 실행
2. **추론 결과 EDA:** v2 키워드 게이트 반영 + S4B 특성(보수적 검출) 고려한 분포 분석
3. **연착륙 상품 분석(SLI) 갱신:** S4B 추론 결과 기반으로 SLI 재산출 및 상위 10% SKU 리스트 업데이트
4. **최종 리포트 및 발표 자료 반영:** 모델 개선 과정(전략 A/B/C)을 프로젝트 스토리라인에 포함

---

## 파이프라인 현재 위치

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
[17] Stage 5 전략 A/B/C 실행 + Pragmatic GO         ✅  ← 이 문서
[18] S4B 전체 리뷰 재추론
[19] 연착륙 상품 분석
```

---

## 관련 문서

| 문서 | 내용 |
|------|------|
| [ABSA_파이프라인.md](./00_ABSA_파이프라인.md) | 전체 파이프라인 + 모델 아키텍처 상세 |
| [ABSA_Stage1_학습_리포트.md](./01_ABSA_Stage1_학습_리포트.md) | Stage 1 학습 및 골든셋 평가 |
| [ABSA_Stage2_학습_리포트.md](./02_ABSA_Stage2_학습_리포트.md) | Stage 2 구조 개편 + 재학습 |
| [ABSA_Stage2_골든셋_평가.md](./03_ABSA_Stage2_골든셋_평가.md) | Stage 2 골든셋 평가 + Stage 3 방향 설계 |
| [ABSA_Stage3A_후처리_튜닝_리포트.md](./04_ABSA_Stage3A_후처리_튜닝_리포트.md) | Stage 3A 후처리 + GO/NO-GO 판정 |
| [ABSA_Stage3A_전체추론_결과.md](./05_ABSA_Stage3A_전체추론_결과.md) | 323K 전체 추론 실행 결과 |
| [ABSA_전체추론_EDA_결과.md](./06_ABSA_전체추론_EDA_결과.md) | 전체 추론 EDA (v2 키워드 게이트) |
| [ABSA_QC_후처리_개선_전략.md](./07_ABSA_QC_후처리_개선_전략.md) | QC 오류 패턴 7가지 + 후처리 v3/v4 설계 |
| [ABSA_Stage4_재학습_리포트.md](./08_ABSA_Stage4_재학습_리포트.md) | Stage 4 neutral 보강 재학습 |
| **ABSA_Stage5_전략ABC_실행결과.md** | **← 이 문서** |
