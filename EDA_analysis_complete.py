import json

# 완전한 EDA 노트북 구조 생성
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

# 셀 추가 함수
def add_markdown_cell(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text
    })

def add_code_cell(code):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code
    })

# 노트북 내용 구성
add_markdown_cell("# 다이소 화장품 데이터 EDA\n\n## 분석 목차\n1. 데이터 로딩 및 전처리\n2. 상품(Product) 관점 분석\n3. 가격 관점 분석\n4. 리뷰 텍스트 분석\n5. 시간/프로모션 분석\n6. 사용자 분석\n7. 종합 분석")

# 섹션 1: 데이터 로딩
add_markdown_cell("## 1. 데이터 로딩 및 전처리")

add_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 시각화 스타일
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("라이브러리 로딩 완료")""")

add_code_cell("""# 데이터 로딩
data_dir = '/Users/yu_seok/Documents/workspace/nbCamp/Project/Why-pi/data'

products_df = pd.read_parquet(f'{data_dir}/products.parquet')
reviews_df = pd.read_parquet(f'{data_dir}/reviews.parquet')
promo_df = pd.read_csv(f'{data_dir}/promotion.csv')

print(f"제품 데이터: {len(products_df):,}개")
print(f"리뷰 데이터: {len(reviews_df):,}개")
print(f"프로모션 데이터: {len(promo_df):,}개")

print("\\n제품 데이터 컬럼:", products_df.columns.tolist())
print("리뷰 데이터 컬럼:", reviews_df.columns.tolist())

# 기본 통계
print("\\n=== 기본 통계 ===")
print(f"브랜드 수: {products_df['brand'].nunique()}개")
print(f"대분류 카테고리: {products_df['category_1'].nunique()}개")
print(f"소분류 카테고리: {products_df['category_2'].nunique()}개")
print(f"리뷰 작성 사용자: {reviews_df['user'].nunique()}명")""")

# 파생변수 생성
add_markdown_cell("### 파생변수 생성")

add_code_cell("""# 제품 데이터 파생변수

# A-1. Engagement Score
w1, w2, w3 = 0.15, 0.30, 0.55
products_df['engagement_score'] = (
    w1 * products_df['likes'] + 
    w2 * products_df['shares'] + 
    w3 * products_df['review_count']
)

# A-2. Price Position
category_price_stats = products_df.groupby('category_2')['price'].agg(['min', 'max']).reset_index()
products_df = products_df.merge(category_price_stats, on='category_2', how='left', suffixes=('', '_cat'))
products_df['price_range'] = products_df['max'] - products_df['min']
products_df['price_position'] = np.where(
    products_df['price_range'] > 0,
    (products_df['price'] - products_df['min']) / products_df['price_range'],
    0.5
)

def classify_price_position(position):
    if position <= 0.33:
        return 'Low'
    elif position <= 0.67:
        return 'Mid'
    else:
        return 'High'

products_df['price_tier'] = products_df['price_position'].apply(classify_price_position)

# A-3. CP Index (가성비 지표)
products_df['cp_index'] = products_df['engagement_score'] / (products_df['price'] + 1)

# 가격대 분류
products_df['price_range_group'] = pd.cut(products_df['price'], 
                                          bins=[0, 3000, 5000, 10000, np.inf],
                                          labels=['~3천원', '3~5천원', '5천~1만원', '1만원~'])

print("제품 파생변수 생성 완료")
print(f"Engagement Score 범위: {products_df['engagement_score'].min():.2f} ~ {products_df['engagement_score'].max():.2f}")""")

add_code_cell("""# 리뷰 데이터 파생변수

# 날짜 파싱
reviews_df['date'] = pd.to_datetime(reviews_df['date'])
reviews_df['year'] = reviews_df['date'].dt.year
reviews_df['month'] = reviews_df['date'].dt.month
reviews_df['day_of_week'] = reviews_df['date'].dt.dayofweek
reviews_df['day_name'] = reviews_df['date'].dt.day_name()

# 계절 분류
def get_season(month):
    if month in [3, 4, 5]:
        return '봄'
    elif month in [6, 7, 8]:
        return '여름'
    elif month in [9, 10, 11]:
        return '가을'
    else:
        return '겨울'

reviews_df['season'] = reviews_df['month'].apply(get_season)

# C-0. 재구매 플래그
reviews_df['is_reorder'] = reviews_df['text'].fillna('').str.strip().str.startswith('재구매')

# 리뷰 길이 및 품질 지표
reviews_df['review_length'] = reviews_df['text'].fillna('').str.len()
reviews_df['has_image'] = reviews_df['image_count'] > 0
reviews_df['is_quality_review'] = (reviews_df['review_length'] >= 50) & (reviews_df['image_count'] > 0)

print("리뷰 파생변수 생성 완료")
print(f"재구매 리뷰: {reviews_df['is_reorder'].sum():,}개 ({reviews_df['is_reorder'].mean()*100:.1f}%)")""")

add_code_cell("""# 프로모션 데이터 전처리
promo_df['date'] = pd.to_datetime(promo_df['date'])

# 프로모션 타입 카테고리화
def categorize_promo_type(event_type, description):
    if '신상' in description or '신제품' in description or '뉴' in description.lower():
        return '신상품'
    elif '세일' in description or '할인' in description or 'SALE' in description.upper():
        return '할인'
    elif event_type == '구매이벤트':
        return '구매이벤트'
    elif event_type == '리뷰이벤트':
        return '리뷰이벤트'
    else:
        return '기타'

promo_df['promo_type_category'] = promo_df.apply(
    lambda row: categorize_promo_type(row['event_type'], row['description']), 
    axis=1
)

# 리뷰와 프로모션 매칭 (±7일)
def check_promo_period(review_date):
    for _, promo in promo_df.iterrows():
        if abs((review_date - promo['date']).days) <= 7:
            return True
    return False

# 샘플링해서 확인 (전체 적용은 시간이 오래 걸림)
sample_size = min(10000, len(reviews_df))
reviews_sample = reviews_df.sample(sample_size, random_state=42)
reviews_sample['is_during_promo'] = reviews_sample['date'].apply(check_promo_period)

print(f"\\n프로모션 기간 리뷰 비율 (샘플): {reviews_sample['is_during_promo'].mean()*100:.1f}%")
print(f"프로모션 타입: {promo_df['promo_type_category'].value_counts().to_dict()}")""")

# 섹션 2: 상품 관점 분석
add_markdown_cell("## 2. 상품(Product) 관점 분석")
add_markdown_cell("### 2-1. 카테고리별 상품 수")

add_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 대분류
category1_counts = products_df['category_1'].value_counts()
axes[0].barh(range(len(category1_counts)), category1_counts.values)
axes[0].set_yticks(range(len(category1_counts)))
axes[0].set_yticklabels(category1_counts.index)
axes[0].set_xlabel('상품 수')
axes[0].set_title('대분류 카테고리별 상품 수', fontsize=14, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

for i, v in enumerate(category1_counts.values):
    axes[0].text(v + 1, i, str(v), va='center')

# 소분류 (상위 15개)
category2_counts = products_df['category_2'].value_counts().head(15)
axes[1].barh(range(len(category2_counts)), category2_counts.values, color='coral')
axes[1].set_yticks(range(len(category2_counts)))
axes[1].set_yticklabels(category2_counts.index)
axes[1].set_xlabel('상품 수')
axes[1].set_title('소분류 카테고리별 상품 수 (상위 15개)', fontsize=14, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

for i, v in enumerate(category2_counts.values):
    axes[1].text(v + 0.5, i, str(v), va='center')

plt.tight_layout()
plt.show()""")

print("EDA 노트북 스크립트 생성 중...")

# 파일 저장
with open('/Users/yu_seok/Documents/workspace/nbCamp/Project/Why-pi/EDA_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print("노트북 파일 생성 완료!")
