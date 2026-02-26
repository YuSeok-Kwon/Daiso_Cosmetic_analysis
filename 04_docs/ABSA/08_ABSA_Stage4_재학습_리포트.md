# ABSA Stage 4 — Neutral 보강 재학습 리포트

> **작성일:** 2026-02-25
> **모델:** KcELECTRA-base (beomi/KcELECTRA-base) — from-scratch 재학습
> **학습 데이터:** absa_wide_train_stage4.csv (13,749 리뷰, neutral 보강 + QC 교정)
> **검증 데이터:** absa_wide_val_stage4.csv (2,563 리뷰)
> **평가 데이터:** golden_test.csv (266 리뷰, 불변)
> **Threshold 메트릭:** F0.5 (precision 가중)

---

## 목차

1. [한 줄 요약](#1-한-줄-요약)
2. [왜 Stage 4인가 — 재학습 배경](#2-왜-stage-4인가--재학습-배경)
3. [Stage 3A에서 발견된 한계](#3-stage-3a에서-발견된-한계)
4. [QC 분석 결과 — 구조적 문제 확정](#4-qc-분석-결과--구조적-문제-확정)
5. [보충 데이터 설계](#5-보충-데이터-설계)
6. [데이터 병합 상세](#6-데이터-병합-상세)
7. [학습 설정](#7-학습-설정)
8. [Stage 2 → Stage 4 데이터 비교](#8-stage-2--stage-4-데이터-비교)
9. [GO 기준](#9-go-기준)
10. [파이프라인 현재 위치](#10-파이프라인-현재-위치)

---

## 1. 한 줄 요약

Stage 3A 후처리 개선(v3/v4)과 QC 분석에서 **neutral 클래스의 구조적 부족**이 핵심 병목으로 확정됨. neutral 증강(750건) + QC 교정(443건) + 골든셋 활용(877건) 총 2,070건의 보충 데이터를 기존 학습 셋과 병합하여 **13,749건**으로 from-scratch 재학습 수행. neutral class weight 8.06 → 6.82로 감소하여 모델이 neutral을 더 적극적으로 예측할 수 있는 환경 확보.

---

## 2. 왜 Stage 4인가 — 재학습 배경

### 2.1 파이프라인 Stage 정의

```
Stage 1: GPT-4o 자동 라벨 17,132개 → 11 aspect 학습 → 골든셋 평가 → 구조적 문제 발견
Stage 2: 구조 개편(8 aspect) + 디자인 NS + F0.5 재학습
Stage 3A: 후처리만 변경 (디자인 규칙, threshold 보정, polar 조정) → GO → 전체 추론
  └─ v2: 키워드 게이트 확장 + Force-ON 추가 → 전체 추론 v2
  └─ v3/v4: QC 기반 후처리 규칙 확장 (배송 구출, 환각 제거 등)
Stage 4: neutral 보강 데이터 병합 → from-scratch 재학습                    ← 이 문서
```

### 2.2 왜 후처리(Stage 3)가 아니라 재학습(Stage 4)인가

Stage 3A-v3/v4 후처리 개선으로 해결할 수 있는 문제(배송 누락, 환각 FP 등)는 규칙으로 대응했다. 그러나 **neutral 분류 실패**는 후처리로 해결할 수 없는 **모델 내부의 학습 불균형** 문제였다:

| 접근 | 해결 가능 | 해결 불가 |
|------|-----------|-----------|
| 후처리 (Stage 3) | 배송 누락, 가격 FN, 환각 FP, 디자인 규칙 | neutral 미학습, aspect별 감성 혼동 |
| 재학습 (Stage 4) | — | **neutral 분류, 감성 정확도 전반** |

---

## 3. Stage 3A에서 발견된 한계

### 3.1 neutral의 구조적 부족

Stage 2 학습 데이터에서 neutral은 전체 mask=1 셀의 **2.5%**(1,356건)에 불과했다:

| 클래스 | 학습 셀 수 | 비율 | class weight |
|--------|-----------|------|-------------|
| none | 37,022 | 70.2% | 0.36 |
| positive | 8,150 | 15.5% | 1.62 |
| **neutral** | **1,356** | **2.5%** | **8.06** |
| negative | 6,218 | 11.8% | 2.12 |

class weight 8.06은 보상을 시도하지만, **절대 샘플 수가 너무 적어** 모델이 neutral 패턴을 일반화하기 어려웠다.

### 3.2 골든셋 성능에서 확인된 neutral 약점

**Stage 3A golden_test 평가** (후처리 적용 후):

| 지표 | 수치 | 비고 |
|------|------|------|
| Detection Precision | **0.6708** | GO 기준 달성 |
| Detection F1 | **0.6429** | GO 기준 달성 |
| Mentioned Sentiment F1 | **0.4367** | 개선 필요 |
| **Neutral Recall** | **~0.17** | 근본 한계 |

→ 전체 Detection 성능은 GO였으나, neutral recall이 극히 낮아 감성 분류 정밀도에 한계.

### 3.3 전체 추론 EDA에서 확인된 분포 왜곡

323K 전체 추론 결과:

| 감성 | 비율 |
|------|------|
| 긍정 | **66.5%** |
| 중립 | 29.1% (대부분 review-level) |
| 부정 | 4.4% |

→ aspect-level에서 neutral 예측이 극히 적어 "좋거나 나쁘거나" 이분법적 출력 경향.

---

## 4. QC 분석 결과 — 구조적 문제 확정

### 4.1 QC 오류 패턴 7가지

Stage 3A-v2 전체추론에 대한 200건 QC 결과 ([ABSA_QC_후처리_개선_전략.md](./07_ABSA_QC_후처리_개선_전략.md) 참조):

| # | 오류 패턴 | 건수 | 우선순위 | 대응 |
|---|----------|------|---------|------|
| 1 | 미분류 배송 누락 | 78건 | P0 | **후처리** (v3에서 해결) |
| 2 | 가격/가성비 FN | 37건 | P0 | **후처리** (Force-ON) |
| 3 | 배송 보조 언급 | 22건 | P1 | **후처리** (패턴 매칭) |
| 4 | Aspect 환각 | 18건 | P1 | **후처리** (환각 gate) |
| 5 | 모호 판정 부족 | 8건 | P1 | **후처리** (접속사 플래그) |
| 6 | 미분류 2차 키워드 | 15건 | P2 | 보류 (FP 리스크) |
| 7 | 접속사 감성 보정 | 12건 | P2 | 보류 |

→ 1~5번은 후처리(v3/v4)로 대응. **neutral 부족은 7가지에 포함되지 않는 근본 원인** — 후처리 불가, 재학습 필수.

### 4.2 3가지 전략 비교

| 전략 | 비용 | 효과 | 리스크 |
|------|------|------|--------|
| Strategy 1: 후처리 규칙 추가 | 1-2일, $0 | P0-P1 해결 | neutral 미해결 |
| Strategy 2: neutral 보강 재학습 | 3-5일, $0 | **근본 해결** | 모델 불안정 가능 |
| Strategy 3: 골든셋 파인튜닝 | 2-3일, $0 | 제한적 | 오버피팅 리스크 |

→ **Strategy 1 + 2 병행** 채택: 후처리(v3/v4)로 즉시 해결 가능한 것은 처리하고, 나머지는 재학습으로 해결.

---

## 5. 보충 데이터 설계

### 5.1 데이터 소스 구성 (2,070건)

| 소스 | 건수 | 목적 |
|------|------|------|
| **golden** | 877건 | 사람이 직접 검수한 고품질 라벨 학습 활용 |
| **qc** | 181건 | QC에서 발견된 오류 교정 + v4 보강분 |
| **aug (neutral 증강)** | 750건 | aspect별 neutral 샘플 타겟 추가 |
| **qc_v4 (추가 교정)** | 262건 | 재구매 문맥, 배송 neutral, 디자인 neutral 등 |

### 5.2 aug 750건 상세 — aspect별 neutral 타겟 증강

| 소스 | 건수 | 타겟 aspect | 타겟 감성 |
|------|------|------------|----------|
| aug_가격_가성비_neu | 100건 | 가격/가성비 | neutral |
| aug_배송_포장_neu | 100건 | 배송/포장 | neutral |
| aug_사용감_성능_neu | — | *(기존 충분)* | — |
| aug_용량_휴대_neu | 100건 | 용량/휴대 | neutral |
| aug_디자인_neu | 100건 | 디자인 | neutral |
| aug_재질_냄새_neu | 100건 | 재질/냄새 | neutral |
| aug_재구매_neu | 100건 | 재구매 | neutral |
| aug_재구매_neg | 150건 | 재구매 | negative |

### 5.3 aug 비타겟 mask 문제와 해결

**문제:** aug 750건은 v3 추론 결과에서 추출. 8개 aspect 전부에 모델 예측 라벨이 있으나, **사람이 확인한 것은 타겟 aspect뿐**. 비타겟 라벨은 모델 자가 예측값.

**해결 (Option A 채택 — 보수적):**
- 타겟 aspect → mask=1 유지 (학습에 포함)
- 비타겟 7개 aspect → mask=0 처리 (loss에서 제외)

```
처리 전: 행당 mask=1 = 8.0개 (5,984셀)
처리 후: 행당 mask=1 = 1.0개 (748셀)
```

→ 검증되지 않은 라벨로 인한 노이즈 유입 완전 차단.

---

## 6. 데이터 병합 상세

### 6.1 병합 절차

```
[1] 기존 데이터 로드
    train_aug: 11,992건 / val: 2,570건

[2] golden_test 겹침 제거
    보충 2,070건 중 golden_test 266건 제거 → 1,804건
    (평가 벤치마크 오염 방지)

[3] aug 비타겟 mask=0 처리
    748건 × 7 aspect = 5,236셀 mask=0 전환

[4] train/val 겹침 제거 (보충 라벨 우선)
    train에서 47건 제거 / val에서 7건 제거

[5] 병합
    train: 11,945 + 1,804 = 13,749건
    val: 2,563건
```

### 6.2 무결성 검증

| 검증 항목 | 결과 |
|-----------|------|
| golden_test ∩ train | **0건** |
| golden_test ∩ val | **0건** |
| train ∩ val | **0건** |
| aug 행당 mask=1 | **모두 1개** |

### 6.3 neutral 증가량

| Aspect | Stage 2 | Stage 4 | 증가 |
|--------|---------|---------|------|
| 배송/포장 | 28 | 119 | **+91** |
| 가격/가성비 | 81 | 164 | +83 |
| 사용감/성능 | 747 | 869 | +122 |
| 용량/휴대 | 74 | 157 | +83 |
| **디자인** | **8** | **105** | **+97** |
| 재질/냄새 | 119 | 245 | +126 |
| 재구매 | 15 | 65 | +50 |
| 색상/발색 | 284 | 314 | +30 |

→ 디자인(+97), 재질/냄새(+126), 배송(+91)이 가장 큰 폭으로 증가.

---

## 7. 학습 설정

### 7.1 하이퍼파라미터 (Stage 2와 동일)

| 항목 | 값 |
|------|-----|
| 모델 | beomi/KcELECTRA-base (768-dim) |
| max_length | 128 |
| batch_size | 32 |
| num_epochs | 10 |
| learning_rate | 2e-5 |
| warmup_ratio | 0.1 |
| weight_decay | 0.01 |
| use_class_weight | True |
| threshold_metric | F0.5 (beta=0.5) |
| 디바이스 | MPS (Apple Silicon) |

### 7.2 학습 방식

- **From-scratch**: Stage 2 체크포인트가 아닌 KcELECTRA 사전학습 가중치에서 시작
- **이유**: 데이터 분포가 크게 변경됨 (neutral +682건), warm-start 시 기존 편향 잔류 리스크
- **디자인 NS**: 기존 train_aug에 이미 적용된 상태, 추가 적용 불필요 (class weight가 보정)

### 7.3 Class Weight 변화

| 클래스 | Stage 2 | Stage 4 | 변화 |
|--------|---------|---------|------|
| none | 0.36 | 0.36 | 유지 |
| positive | 1.62 | 1.50 | 소폭 감소 |
| **neutral** | **8.06** | **6.82** | **-15.4%** |
| negative | 2.12 | 2.63 | 소폭 증가 |

→ neutral weight 감소 = **neutral 샘플이 충분해져서 극단적 보상이 불필요**해진 것.

### 7.4 파일 구조

```
06_analysis/03_ABSA/
├── 02_processed_data/final/absa/stage4/
│   ├── absa_wide_train_stage4.csv   (13,749건)
│   ├── absa_wide_val_stage4.csv     (2,563건)
│   └── merge_report.txt
├── 06_scripts/
│   ├── merge_supplement_data.py     (병합 스크립트)
│   └── train_stage4.py              (학습 스크립트)
└── 07_models/
    ├── checkpoints/                  (Stage 2 — 보존)
    └── checkpoints_stage4/           (Stage 4 — 신규)
        ├── best_model.pt
        ├── none_thresholds.json
        └── training_history.json
```

---

## 8. Stage 2 → Stage 4 데이터 비교

| 항목 | Stage 2 | Stage 4 | 변화 |
|------|---------|---------|------|
| Train 건수 | 11,992 | 13,749 | +14.7% |
| Val 건수 | 2,570 | 2,563 | -0.3% |
| 총 neutral (train) | ~1,356 | ~2,038 | **+50.3%** |
| 디자인 neutral | 8 | 105 | **+1,212%** |
| 재구매 negative | 105 | 255+ | +143% |
| mask=1 셀 비율 | 43.6% | 39.1% | -4.5pp (aug mask=0 영향) |
| neutral class weight | 8.06 | 6.82 | -15.4% |
| golden 활용 | 평가 전용 | **학습에 일부 활용** | 877건 |
| QC 교정 | 없음 | **443건 포함** | 오류 패턴 보정 |

---

## 9. GO 기준

Stage 2 성능을 유지하면서 neutral 개선이 핵심 목표:

| 지표 | Stage 2 | Stage 4 목표 | 판정 기준 |
|------|---------|-------------|----------|
| Detection Precision | 0.6708 | **≥ 0.65** | #1 KPI 유지 |
| Detection F1 | 0.6429 | **≥ 0.60** | GO 최소 기준 |
| Mentioned Sentiment F1 | 0.4367 | **≥ 0.43** | neutral 증가 효과 기대 |
| Neutral Recall | ~0.17 | **≥ 0.25** | 구조적 개선 핵심 지표 |

---

## 10. 파이프라인 현재 위치

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
[16] Stage 4 재학습 (neutral 보강)                  🔄  ← 현재
[17] Stage 4 골든셋 평가
[18] (GO 시) 전체 리뷰 재추론
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
| **ABSA_Stage4_재학습_리포트.md** | **← 이 문서** |

---

**다음 단계:** 학습 완료 후 golden_test 평가 → GO/NO-GO 판정
