# Stage 4B Keyword Gate 오탐 분석 - 액션 아이템

**분석 완료일:** 2025-02-26  
**분석자:** Claude  
**우선순위:** High  

---

## 1. 즉시 실행 액션 (2주 내)

### 액션 1.1: 키워드 제거 (s1_config.py)

**상태:** 🔴 Not Started

**작업:**
```python
# 파일: 06_analysis/03_ABSA/05_src/s1_config.py

# 수정 전
KEYWORD_GATE_CONFIG = {
    "가격/가성비": ["가격", "저렴", "가성비", "최저가", "비싸", "싸", "저가", "천원", "원짜리", "돈", "값", "가격대", "싸게", "만원"],
    "용량/휴대": ["양", "용량", "사이즈", "작다", "크다", "작아", "작은", "ml", "g", "휴대", "여행", "오래", "적다", "많다", "넉넉", "부족", "모자", "들고", "파우치", "미니"],
}

# 수정 후
KEYWORD_GATE_CONFIG = {
    "가격/가성비": ["가격", "저렴", "가성비", "최저가", "비싸", "저가", "천원", "원짜리", "돈", "값", "가격대", "싸게", "만원"],
    # 제거: "싸" (99.5% 오탐률)
    
    "용량/휴대": ["용량", "사이즈", "작다", "크다", "작아", "작은", "ml", "휴대", "여행", "적다", "많다", "넉넉", "부족", "모자", "들고", "파우치", "미니"],
    # 제거: "양" (91.7% 오탐률), "g" (100% 오탐률)
}
```

**기대 효과:**
- 가격/가성비: 2,534건 오탐 → 95건 (-96.3%)
- 용량/휴대: 3,287건 오탐 → 52건 (-98.4%)
- **전체 정확도: 92.6% → 96.5% (+3.9%p)**

**검증 방법:**
```bash
# 테스트 실행
python 06_analysis/03_ABSA/05_src/inference_pipeline.py --version=stage4c-v1 --test-mode

# 골든셋 재평가 (100건 이상)
python 06_analysis/03_ABSA/05_src/golden_set_eval.py --version=stage4c-v1
```

**담당:** [이름 입력]  
**예상 완료일:** 2025-03-12

---

### 액션 1.2: 후처리 정규식 필터 추가 (s3a_postprocessing.py)

**상태:** 🔴 Not Started

**작업:**
```python
# 파일: 06_analysis/03_ABSA/05_src/s3a_postprocessing.py

# 디자인 Aspect 필터
DESIGN_FALSEPOSITIVE_PATTERNS = [
    {
        "keyword": "팁",
        "patterns": [
            r"팁\s*알려",      # "팁 알려드릴게"
            r"팁이",           # "팁이 좋아요"
            r"붓\s*팁",        # "붓 팁이"
        ],
        "action": "none"
    },
    {
        "keyword": "깔끔",
        "patterns": [
            r"깔끔하게\s*(?:발림|사용|흡수|마무리)",  # "깔끔하게 흡수돼요"
            r"깔끔한\s*(?:사용감|발림감)",
        ],
        "action": "none"
    },
    {
        "keyword": "고급",
        "patterns": [
            r"고급스러운\s*(?:발림|사용감|느낌|텍스처)",  # "고급스러운 느낌"
        ],
        "action": "none"
    },
]

# 재질/냄새 Aspect 필터
SCENT_FALSEPOSITIVE_PATTERNS = [
    {
        "keyword": "향",
        "patterns": [
            r"향상|향상시",    # "향상된"
            r"향수",          # "향수를" (시간)
            r"고급진\s*향",    # "고급진 향"
        ],
        "action": "none"
    },
]

# 용량/휴대 Aspect 필터
VOLUME_FALSEPOSITIVE_PATTERNS = [
    {
        "keyword": "오래",
        "patterns": [
            r"오래\s*(?:쓸|버티|지속|가지고|사용)",  # "오래 쓸 것 같아요"
            r"오래\s*(?:보관|들고)",                # "오래 들고 다니기"
        ],
        "action": "none"
    },
]
```

**기대 효과:**
- 디자인: 83건 → 50건 오탐 (-39.8%)
- 재질/냄새: 304건 → 250건 오탐 (-17.8%)
- 용량/휴대: 52건 → 30건 오탐 (-42.3%)
- **전체 정확도: 96.5% → 96.6% (+0.1%p)**

**검증 방법:**
```bash
# 50건 샘플 수동 검증
python scripts/sample_validation.py --aspect=디자인 --filter-version=v1 --size=50
python scripts/sample_validation.py --aspect=재질/냄새 --filter-version=v1 --size=50
python scripts/sample_validation.py --aspect=용량/휴대 --filter-version=v1 --size=50
```

**담당:** [이름 입력]  
**예상 완료일:** 2025-03-19

---

## 2. 중기 실행 액션 (3주~1개월)

### 액션 2.1: Aspect별 신뢰도 가중치 도입

**상태:** 🟡 Planning

**작업:**
```python
# 파일: 06_analysis/03_ABSA/05_src/confidence_weighting.py

ASPECT_RELIABILITY_WEIGHT = {
    "디자인": 0.99,          # 고신뢰
    "가격/가성비": 0.95,    # 중신뢰 (after Phase 1)
    "재질/냄새": 0.97,      # 고신뢰
    "사용감/성능": 0.92,    # 중신뢰
    "배송/포장": 0.85,      # 저신뢰
    "용량/휴대": 0.96,      # Phase 1 이후 개선 (after Phase 1)
    "재구매": 0.88,         # 저신뢰
    "색상/발색": 0.90,      # 중신뢰
}

# 후처리: confidence < 0.7 + gate aspect인 경우
def apply_confidence_weight(aspect, sentiment, confidence):
    is_gate = aspect in ["디자인", "가격/가성비", "재질/냄새", "용량/휴대"]
    
    if is_gate and confidence < 0.7:
        # 추가 검증 스텝
        if not keyword_gate_verified(aspect, text):
            return None  # 제거
    
    return sentiment
```

**기대 효과:**
- 신뢰도 기반 이중 검증으로 추가 1~2%p 개선 가능
- **최종 정확도: 96.6% → 97%+ 목표**

**담당:** [이름 입력]  
**예상 완료일:** 2025-04-09

---

### 액션 2.2: 버전 관리 정리

**상태:** 🟡 Planning

**작업:**

| 버전 | 변경사항 | 정확도 예상 | 배포 시기 |
|------|---------|----------|--------|
| Stage 4B | 현재 상태 (baseline) | 92.6% | ✅ 완료 |
| **Stage 4C-v1** | 키워드 제거 (싸, 양, g) | 96.5% | 2025-03-12 |
| **Stage 4C-v2** | 후처리 필터 추가 | 96.6% | 2025-03-19 |
| **Stage 4D** | 신뢰도 가중치 | 97%+ | 2025-04-09 |

**담당:** [이름 입력]

---

## 3. 모니터링 및 QA

### 액션 3.1: 배포 후 모니터링

**대상:**
- 각 Phase별 배포 후 1주일간 모니터링
- 골든셋 성능 추적

**체크리스트:**
- [ ] Stage 4C-v1 배포 후 골든셋 성능 유지 확인
- [ ] Stage 4C-v2 배포 후 false positive 감소 확인
- [ ] 프로덕션 환경에서 실시간 aspect 분포 체크
- [ ] 사용자 피드백 수집 및 반영

---

### 액션 3.2: 주간 리포트

**일정:** 매주 금요일 10:00  
**담당:** [이름 입력]

**내용:**
1. Phase별 진행 상황
2. 테스트 결과 (정확도, 오탐률)
3. 이슈 및 차단 요소
4. 다음주 계획

---

## 4. 문서화

### 액션 4.1: 코드 주석 및 문서화

**작업:**
- [ ] s1_config.py에 키워드 근거 주석 추가
- [ ] s3a_postprocessing.py에 정규식 설명 추가
- [ ] README.md에 Gate aspect 설명 추가
- [ ] ABSA 파이프라인 문서 업데이트

**담당:** [이름 입력]  
**예상 완료일:** 2025-03-26

---

## 5. 위험 요소 및 대응

| 위험 | 영향도 | 발생 확률 | 대응 방안 |
|------|--------|--------|--------|
| 키워드 제거로 인한 recall 감소 | 중 | 중 | 다른 키워드(저렴, 용량)로 보강 |
| 정규식 패턴 오류 | 중 | 낮 | 충분한 테스트 케이스 확보 |
| 모니터링 부족 | 중 | 중 | 주간 리포트 의무화 |
| 버전 호환성 이슈 | 낮 | 낮 | 별도 테스트 환경 구축 |

---

## 6. 성공 기준

| 항목 | 현재 | 목표 | 달성 기준 |
|------|------|-----|--------|
| 전체 정확도 | 92.6% | 96.6%+ | ✅ |
| 용량/휴대 정확도 | 72.6% | 95%+ | 🎯 우선 |
| 오탐률 | 7.4% | 0.5% 이하 | ✅ |
| 성능 저하 | N/A | 0.5%p 이상 감소 없음 | 필수 |

---

## 7. 참고 자료

- 📄 상세 분석: `/07_docs/ABSA/ABSA_Stage4B_Keyword_Gate_오탐분석.md`
- 📊 데이터: `/06_analysis/03_ABSA/04_outputs/absa_results_stage4b_full.csv`
- 💻 코드: `/06_analysis/03_ABSA/05_src/`

---

**최종 검토:** [이름 입력]  
**승인 날짜:** YYYY-MM-DD

