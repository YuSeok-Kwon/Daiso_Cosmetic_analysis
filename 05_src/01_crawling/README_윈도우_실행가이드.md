# 다이소 뷰티 크롤러 — 윈도우 실행 가이드

## 1. 사전 준비

### 1.1 필수 프로그램 설치

| 프로그램        | 버전      | 다운로드                          |
| --------------- | --------- | --------------------------------- |
| Python          | 3.10 이상 | https://www.python.org/downloads/ |
| Chrome 브라우저 | 최신      | https://www.google.com/chrome/    |
| Git (선택)      | 최신      | https://git-scm.com/download/win  |

> Python 설치 시 **"Add Python to PATH"** 체크 필수!

### 1.2 Chrome 버전 확인

크롤러가 `undetected-chromedriver`를 사용하므로 Chrome이 설치되어 있어야 합니다.
드라이버는 자동으로 다운로드됩니다.

```
chrome://settings/help  ← 브라우저에서 버전 확인
```

---

## 2. 폴더 구조

### 2.1 전체 구조

```
daiso_crawler/                          ← 루트 폴더 (이름 자유)
├── 05_src/
│   ├── 01_crawling/
│   │   ├── run_incremental.py          ← ★ 팀원용 실행 파일 (증분 크롤링)
│   │   ├── daiso_beauty_crawler.py     ← 크롤링 엔진 (직접 실행 X)
│   │   ├── config.py                   ← 설정
│   │   ├── crawl_history.py            ← 증분 크롤링 이력
│   │   ├── utils.py                    ← 유틸리티
│   │   ├── requirements.txt            ← 의존성
│   │   └── modules/
│   │       ├── __init__.py
│   │       ├── driver_setup.py         ← Selenium 드라이버
│   │       ├── clova_ocr.py            ← Clova OCR API
│   │       ├── ocr_utils_split.py      ← 이미지 분할 OCR
│   │       ├── image_preprocessor.py   ← 이미지 전처리
│   │       ├── ingredient_detector.py  ← 성분표 감지
│   │       ├── ingredient_parser.py    ← 성분 파싱
│   │       ├── ingredient_parser_v2.py
│   │       └── ingredient_postprocessor.py
│   └── 02_bigquery/                    ← BQ 적재 모듈 (선택)
│       ├── __init__.py
│       ├── bq_client.py               ← BQ 클라이언트
│       └── etl_loader.py              ← CSV → BQ 변환/적재
└── config/
    ├── .env                            ← API 키 (팀장에게 전달받기)
    └── daiso-analysis-xxxxx.json       ← GCP 서비스 계정 키 (BQ용, 선택)
```

### 2.2 config 폴더 위치 (중요!)

`config/` 폴더는 반드시 **`05_src/`와 같은 레벨(루트)** 에 있어야 합니다.

크롤러가 `.env` 파일을 찾는 경로:

```
daiso_beauty_crawler.py 위치:  daiso_crawler/05_src/01_crawling/
.env 탐색 경로:               daiso_crawler/05_src/01_crawling/../../config/.env
                              → daiso_crawler/config/.env
```

즉, 코드가 자기 위치(`01_crawling/`)에서 **두 단계 위로 올라간 뒤** `config/.env`를 찾습니다.

```
daiso_crawler/              ← 루트
├── config/                 ← 05_src와 같은 레벨 (필수!)
│   └── .env
└── 05_src/
    └── 01_crawling/
        └── daiso_beauty_crawler.py
```

> **잘못된 예시 (동작 안 함):**
>
> ```
> ✗ 01_crawling/config/.env          ← 크롤링 폴더 안에 넣으면 안 됨
> ✗ 05_src/config/.env               ← 한 단계만 위로 올려도 안 됨
> ✗ daiso_crawler/.env               ← config 폴더 없이 루트에 놓아도 안 됨
> ```

### 2.3 .env 파일

직접 만들 필요 없습니다.

`.env` 내용 (참고):

```
CLOVA_OCR_URL=https://...        # Naver Clova OCR API URL
CLOVA_OCR_SECRET=...              # Naver Clova OCR Secret
```

> Clova OCR 키가 없어도 **제품 정보 + 리뷰 크롤링은 정상 동작**합니다.
> 성분 추출만 건너뛰고 나머지는 진행됩니다.

---

## 3. 설치 순서

### 3.1 폴더 배치

압축 파일을 아무 위치에 풀면 됩니다. 예시:

```
C:\Users\팀원\Desktop\daiso_crawler\
├── 05_src\01_crawling\...
└── config\.env
```

> `05_src`와 `config`가 같은 폴더 안에 나란히 있으면 OK!

### 3.2 가상환경 생성 (권장)

```powershell
# PowerShell 또는 CMD에서 실행
cd C:\Users\팀원\Desktop\daiso_crawler

# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (PowerShell)
.\venv\Scripts\Activate.ps1

# 가상환경 활성화 (CMD)
.\venv\Scripts\activate.bat
```

> PowerShell에서 실행 정책 오류 발생 시:
>
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 3.3 패키지 설치

```powershell
pip install -r 05_src\01_crawling\requirements.txt
```

> EasyOCR 설치 시 PyTorch가 함께 설치됩니다 (1~2GB, 시간 소요).

### 3.4 BigQuery 패키지 설치 (BQ 적재 시에만)

크롤링 후 BigQuery에도 적재하려면 추가 설치가 필요합니다:

```powershell
pip install google-cloud-bigquery google-auth
```

> BQ 적재가 필요 없으면 이 단계는 건너뛰세요. CSV는 항상 저장됩니다.

---

## 4. 환경 설정

### 4.1 .env 파일

팀장에게 전달받은 `.env` 파일을 `config/` 폴더에 넣으세요.
폴더 위치는 **2.2절** 참고.

### 4.2 GCP 서비스 계정 키 (BigQuery 적재 시에만)

BigQuery에 적재하려면 GCP 서비스 계정 키 JSON 파일이 필요합니다.
팀장에게 전달받은 `daiso-analysis-xxxxx.json` 파일을 `config/` 폴더에 넣으세요.

```
config/
├── .env                            ← Clova OCR 키
└── daiso-analysis-xxxxx.json       ← GCP 서비스 계정 키
```

> BQ 키가 없으면 크롤링 완료 후 BQ 적재 프롬프트가 나타나지 않고, CSV만 저장됩니다.

### 4.3 성분 추출 없이 실행 (OCR 키 없는 경우)

Clova OCR 키가 없어도 **제품 정보 + 리뷰 크롤링은 정상 동작**합니다.
성분 추출만 건너뛰고 나머지는 진행됩니다.

---

## 5. 실행

### 5.1 사전 준비: 기존 CSV 배치

증분 크롤링은 **기존 CSV**를 기반으로 "어떤 제품이 있고, 어디까지 수집했는지" 이력을 생성합니다.
팀장에게 전달받은 CSV 파일 2개를 `data/` 폴더에 넣으세요.

```
05_src\01_crawling\data\
├── products_core.csv          ← 기존 제품 목록 (81KB)
└── reviews_core.csv           ← 기존 리뷰 목록 (14MB)
```

| 파일 | 역할 | 없으면? |
|------|------|---------|
| `products_core.csv` | 기존 제품을 "이미 수집됨"으로 인식 | 리뷰 없는 제품이 신규로 오판 → 불필요한 재크롤링 |
| `reviews_core.csv` | 제품별 마지막 리뷰 날짜 → 증분 cutoff 기준 | 기존 제품 리뷰를 처음부터 다시 수집 |

> 둘 다 없으면 자동으로 풀 크롤링(처음부터 전부)으로 전환됩니다.

### 5.2 증분 크롤링 실행

```powershell
cd C:\Users\팀원\Desktop\daiso_crawler\05_src\01_crawling
python run_incremental.py
```

**옵션:**

```powershell
# 성분 크롤링 생략 (OCR 키 없을 때)
python run_incremental.py --no-ingredients

# 브라우저 창 보이게 (디버깅용)
python run_incremental.py --visible

# 풀 크롤링 (증분 무시, 처음부터 전부)
python run_incremental.py --full
```

### 5.3 증분 크롤링 동작 원리

```
첫 실행 시:
   data/products_core.csv → 기존 제품 코드 등록
   data/reviews_core.csv  → 제품별 마지막 리뷰 날짜 설정
   → cache/crawl_history.json 생성 (부트스트랩)

크롤링 시:
   기존 제품 → 마지막 리뷰 날짜 이후의 새 리뷰만 수집 (제품/성분 스킵)
   신규 제품 → 전체 크롤링 (제품 정보 + 리뷰 + 성분)

2회차 이후:
   cache/crawl_history.json 이 이미 있으므로 바로 증분 동작
```

### 5.4 출력 파일

실행 완료 후 `05_src/01_crawling/data/` 폴더에 생성:

```
data/
├── products_YYYYMMDD.csv       ← 제품 정보
├── reviews_YYYYMMDD.csv        ← 리뷰 (신규분)
└── ingredients_YYYYMMDD.csv    ← 성분 (OCR 키 필요)
```

### 5.4 BigQuery 적재

CSV 저장이 완료되면 BigQuery 적재 여부를 묻는 프롬프트가 나타납니다:

```
BigQuery에 적재하시겠습니까? (y/n): y
```

**적재 조건:**

- `google-cloud-bigquery` 패키지가 설치되어 있어야 함 (3.4절)
- `config/daiso-analysis-xxxxx.json` GCP 키 파일이 있어야 함 (4.2절)
- 두 조건 중 하나라도 없으면 프롬프트 자체가 나타나지 않고 CSV만 저장

**적재 흐름:**

```
크롤링 완료
   │
   ├─ 1) CSV 저장 (항상 자동 실행)
   │     data/products_YYYYMMDD.csv
   │     data/reviews_YYYYMMDD.csv
   │     data/ingredients_YYYYMMDD.csv
   │
   └─ 2) BigQuery 적재 (조건 충족 시)
         "BigQuery에 적재하시겠습니까? (y/n)"
         ├─ y → 제품/리뷰/성분 순서대로 BQ에 적재
         └─ n → 스킵 (CSV만 유지)
```

> BQ 적재 없이 CSV만 받아서 팀장에게 전달해도 됩니다.

---

## 6. 자주 발생하는 문제

### 6.1 "chromedriver를 찾을 수 없습니다"

Chrome 브라우저가 설치되어 있는지 확인하세요.
`undetected-chromedriver`가 자동으로 맞는 버전을 다운로드합니다.

### 6.2 "ModuleNotFoundError: No module named 'xxx'"

가상환경이 활성화되었는지 확인:

```powershell
# 프롬프트에 (venv) 표시 확인
(venv) C:\Users\팀원\...>

# 안 보이면 다시 활성화
.\venv\Scripts\activate.bat
```

### 6.3 EasyOCR 설치 실패 (torch 관련)

```powershell
# torch를 먼저 설치 (CPU 버전)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install easyocr
```

### 6.4 "SSL: CERTIFICATE_VERIFY_FAILED"

네트워크 프록시 문제일 수 있습니다:

```powershell
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

### 6.5 PowerShell 스크립트 실행 차단

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---
