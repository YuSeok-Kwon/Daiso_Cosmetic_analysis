"""
성분 데이터 분석 스크립트
- products.csv에 없는 제품 탐색
- 이상치 개수 분석
"""
import pandas as pd
import numpy as np

# 파일 로드
ingredients_path = "/Users/yu_seok/Documents/workspace/nbCamp/Project/Why-pi/data/ingredients_스킨케어_립케어:클렌징:선크림20260209.csv"
products_path = "/Users/yu_seok/Documents/workspace/nbCamp/Project/Why-pi/data/csv/products.csv"

print("=" * 70)
print("성분 데이터 분석")
print("=" * 70)

# 데이터 로드
ingredients_df = pd.read_csv(ingredients_path, encoding='utf-8-sig')
products_df = pd.read_csv(products_path)

print(f"\n📊 기본 정보")
print(f"  - ingredients.csv: {len(ingredients_df)} 행")
print(f"  - products.csv: {len(products_df)} 행")

# 컬럼 확인
print(f"\n📋 ingredients.csv 컬럼: {list(ingredients_df.columns)}")
print(f"📋 products.csv 컬럼: {list(products_df.columns)}")

# 고유 제품 ID 추출
ingredient_product_ids = set(ingredients_df['product_id'].unique())
product_ids = set(products_df['product_code'].unique())

print(f"\n📦 제품 수")
print(f"  - ingredients.csv 고유 제품: {len(ingredient_product_ids)}개")
print(f"  - products.csv 고유 제품: {len(product_ids)}개")

# products에 없는 제품 찾기
missing_in_products = ingredient_product_ids - product_ids
print(f"\n⚠️ products.csv에 없는 제품: {len(missing_in_products)}개")

if missing_in_products:
    print("\n  누락된 제품 ID 목록:")
    for pid in sorted(missing_in_products):
        # 해당 제품의 이름 찾기
        name = ingredients_df[ingredients_df['product_id'] == pid]['name'].iloc[0]
        count = len(ingredients_df[ingredients_df['product_id'] == pid])
        print(f"    - {pid}: {name} ({count}개 성분)")

# ingredients에 없는 products 찾기
missing_in_ingredients = product_ids - ingredient_product_ids
print(f"\n⚠️ ingredients.csv에 없는 products: {len(missing_in_ingredients)}개")

# 제품별 성분 개수 분석
print("\n" + "=" * 70)
print("성분 개수 분석")
print("=" * 70)

ingredient_counts = ingredients_df.groupby('product_id').size()

print(f"\n📊 제품별 성분 개수 통계")
print(f"  - 평균: {ingredient_counts.mean():.1f}개")
print(f"  - 중앙값: {ingredient_counts.median():.1f}개")
print(f"  - 최소: {ingredient_counts.min()}개")
print(f"  - 최대: {ingredient_counts.max()}개")
print(f"  - 표준편차: {ingredient_counts.std():.1f}")

# IQR 기반 이상치 탐지
Q1 = ingredient_counts.quantile(0.25)
Q3 = ingredient_counts.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\n📏 IQR 기반 이상치 기준")
print(f"  - Q1: {Q1:.1f}")
print(f"  - Q3: {Q3:.1f}")
print(f"  - IQR: {IQR:.1f}")
print(f"  - 하한: {lower_bound:.1f}")
print(f"  - 상한: {upper_bound:.1f}")

# 이상치 제품 찾기
outliers_low = ingredient_counts[ingredient_counts < lower_bound]
outliers_high = ingredient_counts[ingredient_counts > upper_bound]

print(f"\n🔍 이상치 제품")
print(f"  - 성분 개수 너무 적음 (<{lower_bound:.0f}): {len(outliers_low)}개")
print(f"  - 성분 개수 너무 많음 (>{upper_bound:.0f}): {len(outliers_high)}개")

# 성분 개수 분포
print(f"\n📊 성분 개수 분포")
bins = [0, 5, 10, 20, 30, 40, 50, 60, 100, 200]
for i in range(len(bins) - 1):
    count = len(ingredient_counts[(ingredient_counts > bins[i]) & (ingredient_counts <= bins[i+1])])
    if count > 0:
        print(f"  - {bins[i]+1}~{bins[i+1]}개: {count}개 제품")

# 성분 개수가 너무 적은 제품 (10개 미만)
print(f"\n⚠️ 성분 개수 10개 미만 제품 (데이터 품질 의심)")
low_ingredient_products = ingredient_counts[ingredient_counts < 10].sort_values()
for pid, count in low_ingredient_products.items():
    name = ingredients_df[ingredients_df['product_id'] == pid]['name'].iloc[0]
    print(f"  - {pid}: {name} ({count}개)")

# 성분 개수가 너무 많은 제품 (60개 초과)
print(f"\n⚠️ 성분 개수 60개 초과 제품")
high_ingredient_products = ingredient_counts[ingredient_counts > 60].sort_values(ascending=False)
for pid, count in high_ingredient_products.items():
    name = ingredients_df[ingredients_df['product_id'] == pid]['name'].iloc[0]
    print(f"  - {pid}: {name} ({count}개)")

# 중복 성분 체크
print("\n" + "=" * 70)
print("중복 성분 분석")
print("=" * 70)

duplicates = ingredients_df.groupby(['product_id', 'ingredient']).size()
duplicates = duplicates[duplicates > 1]

print(f"\n🔄 중복 성분 (동일 제품에 같은 성분이 2번 이상)")
print(f"  - 중복 케이스: {len(duplicates)}개")

if len(duplicates) > 0:
    print("\n  중복 상세:")
    for (pid, ing), count in duplicates.head(20).items():
        name = ingredients_df[ingredients_df['product_id'] == pid]['name'].iloc[0]
        print(f"    - {pid} ({name[:20]}...): '{ing}' x{count}")

# 결측치 체크
print("\n" + "=" * 70)
print("결측치 분석")
print("=" * 70)

for col in ingredients_df.columns:
    null_count = ingredients_df[col].isnull().sum()
    if null_count > 0:
        print(f"  - {col}: {null_count}개 ({null_count/len(ingredients_df)*100:.1f}%)")

# 빈 문자열 체크
empty_ingredients = ingredients_df[ingredients_df['ingredient'].str.strip() == '']
print(f"\n  - 빈 성분명: {len(empty_ingredients)}개")

print("\n" + "=" * 70)
print("분석 완료")
print("=" * 70)
