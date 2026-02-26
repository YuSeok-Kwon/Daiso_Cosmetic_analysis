# ABSA QC 후처리 개선 전략

> Stage 3A-v2 전체추론(323K) QC에서 식별된 반복 오류 패턴 7가지에 대한 후처리 개선 전략

---

## 1. QC 결과 요약

### 1.1 전체 오류 현황

| 항목 | 수치 |
|------|------|
| QC 샘플 수 | 200건 |
| 오류 식별 건수 | 190건 (복수 패턴 포함) |
| 고유 오류 패턴 | 7가지 |

### 1.2 오류 패턴 분류

| # | 오류 패턴 | 건수 | 비율 | 우선순위 | 설명 |
|---|----------|------|------|---------|------|
| 1 | 미분류 배송 누락 | 78건 | 41% | **P0** | 배송 관련 키워드가 있는데 모든 aspect가 none → 미분류 |
| 2 | 가격/가성비 FN | 37건 | 19% | **P0** | 가성비 키워드가 있는데 가격 aspect가 none |
| 3 | 배송 보조 언급 누락 | 22건 | 12% | P1 | "배송도 빠르고" 패턴에서 배송 미검출 |
| 4 | Aspect 환각 (FP) | 18건 | 9% | P1 | 키워드 없이 배송/재구매/색상 활성화 |
| 5 | 모호 판정 부족 | 8건 | 4% | P1 | 별점-감성 불일치인데 is_ambiguous=False |
| 6 | 미분류 2차 키워드 | 15건 | 8% | P2 | 미분류 중 다른 aspect 키워드 포함 |
| 7 | 접속사 감성 보정 | 12건 | 6% | P2 | "좋은데 배송이 느려요" → 배송 부정 미반영 |

---

## 2. 전략 비교

### Strategy 1: 후처리 규칙 추가 (채택)

| 항목 | 내용 |
|------|------|
| 방법 | Python 후처리 코드에 규칙 7개 추가 |
| 소요 시간 | 1-2일 |
| 비용 | $0 |
| 재학습 | 불필요 |
| 효과 | 190건 중 ~94% 해결 예상 |
| 리스크 | 과도한 규칙 → 다른 리뷰에서 FP 증가 가능 |

### Strategy 2: 학습 데이터 보강 + 재학습

| 항목 | 내용 |
|------|------|
| 방법 | QC 오류 패턴 기반 학습 데이터 추가 → Stage 4 학습 |
| 소요 시간 | 1-2주 |
| 비용 | GPU + 라벨링 비용 |
| 재학습 | 필요 |
| 효과 | 근본적 해결, 일반화 성능 향상 |
| 리스크 | 시간 소요, 기존 성능 regression 가능 |

### Strategy 3: 하이브리드 (후처리 즉시 + 재학습 병행)

| 항목 | 내용 |
|------|------|
| 방법 | Strategy 1 즉시 적용 후, Strategy 2를 중기 과제로 진행 |
| 소요 시간 | 즉시(1-2일) + 중기(1-2주) |
| 효과 | 즉각 효과 + 장기 안정성 |

### 채택 근거

- 프로젝트 일정상 즉시 적용 가능한 Strategy 1 우선 채택
- P2 규칙은 `enabled=False`로 보수적 비활성화, 검증 후 활성화 가능

---

## 3. 규칙별 상세 명세

### 3.1 [P0] 가격/가성비 키워드 Force-On

**문제:** "가성비 좋아요", "싸고 좋아요" 등 가격 관련 리뷰에서 가격/가성비 aspect가 none

**해결:** 기존 `KEYWORD_FORCE_ON_CONFIG`에 가격/가성비 추가

```python
"가격/가성비": {
    "keywords": [
        "가성비", "저렴", "가격", "싸다", "싸고", "싸서", "싼", "저가",
        "돈값", "가격대비", "최저가", "천원", "원짜리", "값", "가격대",
        "싸게", "만원", "혜자", "쏠쏠", "착하",  # 구어체 추가
    ],
    "sentiment": 1,  # positive (기본)
    "negative_keywords": ["비싸", "비싼", "비쌌"],  # → negative(3)
    "negative_sentiment": 3,
}
```

**v2 대비 개선:**
- gate와 force_on 키워드 통일 (gate에만 있던 10개 추가)
- 구어체 표현 "혜자", "쏠쏠", "착하" 추가
- "비싸"류 키워드 → negative 감성 분기 (하드코딩 positive 문제 해결)

**예상 효과:** ~70건 해결

---

### 3.2 [P0] 미분류 배송 구출 (rescue_unclassified)

**문제:** 배송 관련 키워드("배송", "택배", "도착" 등)가 있는데 모든 aspect가 none → 미분류

**해결:** 모든 aspect가 none인 리뷰에서 배송 키워드 매칭 시 배송/포장 aspect 활성화

```python
배송 키워드: ["배송", "택배", "도착", "발송", "출고", "수령", "배달"]
활성화 감성: 별점 기반 (4-5점 → positive, 3점 → neutral, 1-2점 → negative)
           별점 없으면 positive (다이소 리뷰 기본 긍정 경향)
```

**예상 효과:** 78건 해결

---

### 3.3 [P1] 배송 보조 언급 (auxiliary_mention)

**문제:** "배송도 빠르고 제품도 좋아요" → 사용감/성능만 검출, 배송은 미검출

**해결:** 정규식 `배송도|택배도` 패턴 감지 시 배송/포장 추가 활성화

```python
패턴: r"(배송|택배)도\s*(빠르|빨리|좋|괜찮|느리|늦)"
활성화: 배송/포장 aspect
감성: "빠르/빨리/좋/괜찮" → positive, "느리/늦" → negative
```

**예상 효과:** 22건 해결

---

### 3.4 [P1] Aspect 환각 제거 (hallucination_gate)

**문제:** 리뷰 텍스트에 관련 키워드가 전혀 없는데 배송/재구매/색상 aspect가 활성화

**해결:** 기존 `KEYWORD_GATE`가 커버하지 않는 3개 aspect에 키워드 검증 추가

```python
추가 게이트 대상:
  - 배송/포장: ["배송", "택배", "도착", "발송", "출고", "배달", "수령", "포장"]
  - 재구매: ["재구매", "재주문", "또 사", "또사", "다시 사", "다시사", "재방문"]
  - 색상/발색: ["발색", "색", "컬러", "색감", "색상", "립", "틴트", "섀도"]
```

**주의:** 사용감/성능은 커버리지 49.4%로 **제외** — 키워드 리스트가 불완전하면 Recall 하락 위험

**예상 효과:** 18건 해결

---

### 3.5 [P1] 모호 판정 강화 (ambiguous_enhancement)

**문제:** 별점 1점인데 "좋아요" → positive로 분류, is_ambiguous=False

**해결:** 기존 ambiguous 판정에 추가 조건:
- 별점-감성 불일치 (1-2점 + positive 또는 4-5점 + negative)
- 접속사("지만", "는데", "그러나") 포함 + 다수 aspect 활성화

**예상 효과:** 8건 해결

---

### 3.6 [P2] 미분류 2차 키워드 구출 (secondary_rescue) — 비활성화

**문제:** 미분류 리뷰 중 사용감/재질 등 다른 aspect 키워드가 포함된 경우

**해결:** 미분류 리뷰에서 2차 키워드 매칭으로 추가 aspect 활성화

**비활성화 사유:** 사용감/성능의 키워드 리스트가 불완전하여 FP 위험. 키워드 리스트 정교화 후 활성화

**예상 효과:** (활성화 시) 15건 해결

---

### 3.7 [P2] 접속사 감성 보정 (contrast_correction) — 비활성화

**문제:** "제품은 좋은데 배송이 느려요" → 배송 긍정으로 분류

**해결:** 접속사("지만", "는데", "근데") 뒤의 aspect 감성을 반전

**비활성화 사유:** 접속사 뒤가 항상 반전은 아님("비싸긴 한데 그래도 좋아요"). 충분한 테스트 후 활성화

**예상 효과:** (활성화 시) 12건 해결

---

## 4. 후처리 파이프라인 순서

### 기존 (Stage 3A-v2)

```
predict_batch → threshold → design_rule → keyword_gate → keyword_force_on
→ extract_aspect_sentiments → ambiguous → output
```

### 확장 후 (Stage 3A-v3)

```
[Phase 1: 기존]
  predict_batch → threshold → design_rule → keyword_gate → keyword_force_on

[Phase 2: QC 규칙 — 신규]
  → rescue_unclassified (P0)
  → auxiliary_mention (P1)
  → hallucination_gate (P1)

[Phase 3: 출력]
  → extract_aspect_sentiments
  → ambiguous (기존 + P1 강화)
  → secondary_rescue (P2, 비활성화)
  → contrast_correction (P2, 비활성화)
  → output
```

---

## 5. 예상 효과

### 5.1 우선순위별 해결 건수

| 범위 | 해결 건수 | 해결 비율 |
|------|----------|----------|
| P0만 | 115건 | 57% |
| P0 + P1 | 163건 | 81% |
| P0 + P1 + P2 (전체) | 190건 | 94% |

### 5.2 분포 변화 예상

| 지표 | 변경 전 (v2) | 변경 후 (v3) 예상 |
|------|-------------|-----------------|
| 미분류 비율 | 28.7% | ~25% (배송 구출 효과) |
| 배송/포장 언급률 | 15.2% | ~17% |
| 가격/가성비 언급률 | 21.3% | ~23% |

---

## 6. 검증 계획

### 6.1 단위 테스트
- 각 규칙별 5건씩 수동 검증 텍스트로 동작 확인
- 예상 입출력 매칭 확인

### 6.2 Dry-run
- 기존 v2 결과 CSV에 `reapply_qc_postprocess.py --dry-run` 실행
- 변경 건수, 변경 유형별 통계 확인
- 기대: P0 규칙으로 ~115건 변경

### 6.3 분포 비교
- aspect 언급률 변경 전후 비교
- 미분류 비율 변화 확인
- 감성 분포 변화 확인

### 6.4 QC 샘플 재검토
- 변경된 리뷰 100건 샘플링 → 수동 정확도 확인
- FP 증가 여부 모니터링

---

## 7. 설정 독립 제어

각 규칙은 `QC_POSTPROCESS_CONFIG`에서 `enabled` 플래그로 독립 제어:

```python
QC_POSTPROCESS_CONFIG = {
    "rescue_delivery": {"enabled": True, "priority": "P0"},
    "price_force_on": {"enabled": True, "priority": "P0"},  # KEYWORD_FORCE_ON으로 구현
    "auxiliary_delivery": {"enabled": True, "priority": "P1"},
    "hallucination_gate": {"enabled": True, "priority": "P1"},
    "ambiguous_enhancement": {"enabled": True, "priority": "P1"},
    "secondary_rescue": {"enabled": False, "priority": "P2"},
    "contrast_correction": {"enabled": False, "priority": "P2"},
}
```

---

## 8. 파일 변경 목록

| 파일 | 변경 내용 |
|------|----------|
| `05_src/s1_config.py` | KEYWORD_FORCE_ON_CONFIG 확장 + QC_POSTPROCESS_CONFIG 추가 |
| `05_src/s8_inference.py` | 6개 후처리 메서드 추가 + infer_dataframe 파이프라인 확장 |
| `06_scripts/reapply_qc_postprocess.py` | 기존 결과에 후처리 재적용 스크립트 (신규) |
| `07_models/prod_bundle/.../design_rule_config.json` | 가격/가성비 force_on + qc_postprocess_config 추가 |

---

## 9. 후속 — Stage 4 재학습으로의 연결

이 문서의 후처리 개선(v3/v4)은 **규칙으로 해결 가능한 P0-P1 패턴**을 대응한다. 그러나 QC 과정에서 동시에 확인된 **neutral 클래스 구조적 부족**은 후처리로 해결 불가하며, 모델 재학습이 필요하다.

이에 따라:
1. 후처리 규칙(v3/v4)은 `s1_config.py`에 반영 완료 → 추론 시 자동 적용
2. QC에서 교정된 라벨(443건) + neutral 증강(750건) + 골든셋(877건) = **2,070건 보충 데이터** 구성
3. 기존 학습 데이터와 병합하여 **Stage 4 from-scratch 재학습** 수행

→ 상세: [ABSA_Stage4_재학습_리포트.md](./08_ABSA_Stage4_재학습_리포트.md)
