# Research Questions 분석

다이소 뷰티 리뷰 데이터를 기반으로 2가지 연구 질문(RQ)에 대한 분석을 수행한다.

---

## 연구 질문

### RQ1. 재구매를 만드는 핵심 속성은 무엇인가?

- **가설:** 재구매 고객은 일반 고객 대비 특정 제품 속성(품질, 디자인, 편의성)을 더 높게 평가한다
- **방법:** 재구매/일반 그룹 간 키워드 빈도 비교, 카테고리별 감성 분석
- **주요 결과:**
  - 재구매 비율 34.1%, 재구매 그룹 평균 평점 4.84점
  - 재구매 그룹은 심미/편의 키워드를 더 많이 언급
  - "제품은 좋은데 구하기 힘들다"는 희소성 패턴 2.2% 존재
- **시사점:** 다이소는 "싸서 한번 사는 시장"이 아니라 "품질이 괜찮아서 계속 쓰는 시장"

### RQ2. 다이소 뷰티는 저가 듀프(Dupe) 시장인가?

- **가설:** 소비자는 다이소 뷰티 제품을 고가 브랜드의 대체재(듀프)로 인식한다
- **방법:** 고가 브랜드 언급률, 듀프 키워드 분석, 브랜드 vs 제품력 중요도 비교
- **주요 결과:**
  - 듀프 키워드 언급률 0.68%, 제품력/브랜드 비율 6.11
  - 브랜드보다 제품력(효과, 품질)을 중시
  - 유명 브랜드가 아니어도 품질만 좋으면 성공 가능
- **시사점:** 다이소는 "브랜드 충성" 시장이 아닌 "제품 단위 충성" 시장

---

## 디렉토리 구조

```
02_research_questions/
├── 03_notebooks/              # 분석 노트북
│   ├── RQ1_재구매_핵심속성_분석.ipynb
│   └── RQ2_저가_듀프_시장_분석.ipynb
├── 04_outputs/                # 분석 결과물
│   ├── RQ1/
│   │   ├── category_comparison.csv
│   │   ├── insights_RQ1.txt
│   │   └── keyword_comparison.csv
│   └── RQ2/
│       ├── brand_ratings.csv
│       ├── dupe_reviews.csv
│       ├── insights_RQ2.txt
│       ├── luxury_brand_mentions.csv
│       └── market_indicators.csv
├── 05_src/                    # 재사용 모듈
│   ├── keyword_analysis.py    # 키워드 빈도/카테고리 분석
│   ├── text_preprocessing.py  # 텍스트 전처리, 토큰화
│   └── visualization.py       # 워드클라우드, 차트 생성
├── README.md
└── requirements.txt
```

---

## 실행 방법

```bash
# 의존성 설치
pip install -r requirements.txt

# 노트북 실행 (Jupyter Lab/Notebook)
jupyter lab 03_notebooks/
```

**데이터 요구사항:** `06_analysis/data/` 경로에 `reviews.parquet`, `products.parquet` 파일이 필요하다.

---

## 기술 스택

| 분류 | 라이브러리 |
|------|-----------|
| 데이터 처리 | pandas, numpy, pyarrow |
| 시각화 | matplotlib, seaborn, wordcloud |
| 텍스트 분석 | konlpy, scikit-learn |
