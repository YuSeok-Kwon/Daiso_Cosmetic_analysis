# ABSA 팀 분석 가이드

### 분석 대상 Aspect (10개)

| Aspect      | 설명                   |
| ----------- | ---------------------- |
| 배송/포장   | 배송 속도, 포장 상태   |
| 품질/불량   | 제품 품질, 불량 여부   |
| 가격/가성비 | 가격 적정성, 가성비    |
| 사용감/성능 | 사용 경험, 제품 성능   |
| 디자인      | 외관, 디자인           |
| 재질/냄새   | 소재 품질, 냄새        |
| CS/응대     | 고객 서비스, 문의 응대 |
| 재구매      | 재구매 의사            |
| 색상/발색   | 색상, 발색력           |
| 용량/휴대   | 용량, 휴대성           |

### Sentiment 분류

- **positive**: 긍정적
- **neutral**: 중립적
- **negative**: 부정적

---

## 팀원별 담당 데이터

| 팀원  | 파일명       | 리뷰 범위         | 건수    |
| ----- | ------------ | ----------------- | ------- |
| 팀원1 | `team_1.csv` | 1 ~ 3,334번       | 3,334건 |
| 팀원2 | `team_2.csv` | 3,335 ~ 6,668번   | 3,334건 |
| 팀원3 | `team_3.csv` | 6,669 ~ 10,001번  | 3,333건 |
| 팀원4 | `team_4.csv` | 10,002 ~ 13,334번 | 3,333건 |
| 팀원5 | `team_5.csv` | 13,335 ~ 16,667번 | 3,333건 |
| 팀원6 | `team_6.csv` | 16,668 ~ 20,000번 | 3,333건 |

---

## 분석 3단계 전략

| 단계     | 모델        | 목적                      | 예상 비용 (1인) |
| -------- | ----------- | ------------------------- | --------------- |
| 1단계    | GPT-4o-mini | 전체 대량 라벨링          | ~$0.80          |
| 2단계    | GPT-4o      | 불확실/충돌 케이스 재판정 | ~$1.50          |
| 3단계    | GPT-4o      | 골드셋 생성 (100건)       | ~$0.50          |
| **합계** |             |                           | **~$2.80**      |

---

## 환경 설정

### 1. 필수 패키지 설치

```bash
pip install openai pandas tqdm
```

### 2. OpenAI API 키 설정

**Windows (CMD)**

```cmd
set OPENAI_API_KEY=sk-your-api-key-here
```

**Windows (PowerShell)**

```powershell
$env:OPENAI_API_KEY="sk-your-api-key-here"
```

**Mac / Linux**

```bash
export OPENAI_API_KEY=sk-your-api-key-here
```

## 실행 방법

### 프로젝트 폴더로 이동

ABSA 폴더를 **바탕화면**에 저장 후 이동합니다.

**Windows**

```cmd
cd %USERPROFILE%\Desktop\ABSA
```

**Mac / Linux**

```bash
cd ~/Desktop/ABSA
```

### 전체 3단계 실행 (권장)

```bash
# 본인 팀 번호로 실행
python scripts/absa_3step_analysis.py --team 1
python scripts/absa_3step_analysis.py --team 2
python scripts/absa_3step_analysis.py --team 3
python scripts/absa_3step_analysis.py --team 4
python scripts/absa_3step_analysis.py --team 5
python scripts/absa_3step_analysis.py --team 6
```

### 특정 단계만 실행

```bash
# 1단계만 실행
python scripts/absa_3step_analysis.py --team 1 --step 1

# 2단계만 실행 (1단계 완료 후)
python scripts/absa_3step_analysis.py --team 1 --step 2

# 3단계만 실행 (2단계 완료 후)
python scripts/absa_3step_analysis.py --team 1 --step 3
```

### 골드셋 크기 조정

```bash
# 골드셋 50건으로 줄이기
python scripts/absa_3step_analysis.py --team 1 --gold-size 50
```

---

## 출력 파일

실행 완료 후 `data/processed/` 폴더에 결과가 저장됩니다:

```
data/processed/
├── step1_team1_bulk_labels.csv      # 1단계 결과
├── step2_team1_reviewed_labels.csv  # 2단계 결과 (최종 라벨)
└── step3_team1_gold_set.csv         # 3단계 골드셋
```

### 출력 컬럼 설명

| 컬럼           | 설명                             |
| -------------- | -------------------------------- |
| original_index | 원본 데이터 인덱스               |
| text           | 리뷰 텍스트                      |
| rating         | 별점 (1-5)                       |
| aspect         | 추출된 Aspect                    |
| sentiment      | 감성 (positive/neutral/negative) |
| confidence     | 판단 확신도 (0.0~1.0)            |
| reason         | 판단 근거                        |
| model          | 사용된 모델 (gpt-4o-mini/gpt-4o) |

---

## 예상 소요 시간

| 단계     | 예상 시간   |
| -------- | ----------- |
| 1단계    | 30~40분     |
| 2단계    | 10~20분     |
| 3단계    | 5~10분      |
| **합계** | **45~70분** |

> 네트워크 상태와 API 응답 속도에 따라 달라질 수 있습니다

---

## 문제 해결

### 1. "OPENAI_API_KEY 환경변수가 필요합니다" 오류

API 키가 설정되지 않았습니다. 위의 "OpenAI API 키 설정" 섹션을 참고하세요.

### 2. "team_X.csv 파일이 없습니다" 오류

분할 파일이 없습니다. 다음 명령으로 분할 먼저 실행:

```bash
python scripts/split_data.py
```

### 3. 중간에 중단된 경우

- 1단계 완료 후 중단: `--step 2`부터 재시작
- 2단계 완료 후 중단: `--step 3`부터 재시작

```bash
# 2단계부터 재시작
python scripts/absa_3step_analysis.py --team 1 --step 2
```

### 4. Rate Limit 오류

API 호출 제한에 걸린 경우입니다. 잠시 기다린 후 `--step` 옵션으로 중단된 단계부터 재시작하세요.

### 5. 인코딩 오류 (Windows)

PowerShell에서 한글이 깨지는 경우:

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

---

## 완료 후 제출

분석 완료 후 다음 파일들을 슬랙에 공유해주세요:

1. `step2_teamX_reviewed_labels.csv` - 최종 라벨 결과
2. `step3_teamX_gold_set.csv` - 골드셋

---

## 폴더 구조

```
ABSA/
├── scripts/
│   ├── split_data.py              # 데이터 분할 스크립트
│   └── absa_3step_analysis.py     # 3단계 분석 스크립트
│
├── data/
│   ├── raw/
│   │   ├── sampled_reviews_20k.csv  # 원본 데이터
│   │   └── split/
│   │       ├── team_1.csv
│   │       ├── team_2.csv
│   │       ├── team_3.csv
│   │       ├── team_4.csv
│   │       ├── team_5.csv
│   │       └── team_6.csv
│   │
│   └── processed/                  # 결과 저장 위치
│
└── TEAM_GUIDE.md                   # 이 가이드
```

---

## 문의

문제가 있으면 저(유석)에게 연락해주세요!
