"""
카테고리(중분류)별 성분 Type & Effect 분석
- SL vs Non-SL 제품의 성분 계열(Type)과 효능(Effect) 차이를 category_2별로 비교
- 시각화: 10_07 (Type by category), 10_08 (Effect by category)
"""
import sys
sys.path.insert(0, '/Users/yu_seok/Documents/문서 - Mac/workspace/01_현재진행/01_nbCamp/nb_Project/Why-pi/05_src/02_bigquery')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from bq_client import query_to_df

# 한글 폰트
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

FIG_DIR = '/Users/yu_seok/Documents/문서 - Mac/workspace/01_현재진행/01_nbCamp/nb_Project/Why-pi/02_outputs/01_figures/SLI'

# ── 1. 데이터 로드 ──
print("=== BigQuery에서 데이터 로드 ===")

sql = """
SELECT
    pi.product_code,
    pcat.category_2,
    id.ingredient_type,
    id.effect,
    sr.final_soft_landing
FROM daiso.products_ingredients pi
JOIN daiso.ingredients_dic id ON pi.ingredient_id = id.ingredient_id
JOIN daiso.sli_results sr ON pi.product_code = sr.product_code
JOIN daiso.products_category pcat ON pi.product_code = pcat.product_code
WHERE id.ingredient_type IS NOT NULL
  AND id.effect IS NOT NULL
"""
df = query_to_df(sql)
df['is_sl'] = df['final_soft_landing'].astype(bool)
print(f"총 {len(df):,}행, 제품 {df.product_code.nunique()}개")
print(f"카테고리(중분류): {sorted(df.category_2.unique())}")

# ── 2. 카테고리별 기본 통계 ──
cat_stats = df.groupby('category_2').agg(
    n_products=('product_code', 'nunique'),
    n_sl=('product_code', lambda x: x[df.loc[x.index, 'is_sl']].nunique()),
).reset_index()
cat_stats['n_nsl'] = cat_stats['n_products'] - cat_stats['n_sl']
# 최소 제품 수 필터 (SL/Non-SL 각각 5개 이상)
cat_stats = cat_stats[(cat_stats['n_sl'] >= 5) & (cat_stats['n_nsl'] >= 5)]
valid_cats = sorted(cat_stats['category_2'].tolist())
print(f"\n분석 대상 카테고리 ({len(valid_cats)}개): {valid_cats}")
print(cat_stats.to_string(index=False))

df = df[df['category_2'].isin(valid_cats)]


# ── 3. 카테고리별 Type 분석 ──
def calc_type_diff(group_df):
    """카테고리 내 SL vs Non-SL의 ingredient_type 출현율 차이"""
    sl = group_df[group_df['is_sl']]
    nsl = group_df[~group_df['is_sl']]
    n_sl = sl['product_code'].nunique()
    n_nsl = nsl['product_code'].nunique()

    if n_sl == 0 or n_nsl == 0:
        return pd.DataFrame()

    sl_rate = sl.groupby('ingredient_type')['product_code'].nunique() / n_sl
    nsl_rate = nsl.groupby('ingredient_type')['product_code'].nunique() / n_nsl

    comp = pd.DataFrame({'sl_rate': sl_rate, 'nsl_rate': nsl_rate}).fillna(0)
    comp['diff'] = comp['sl_rate'] - comp['nsl_rate']
    comp['lift'] = (comp['sl_rate'] / comp['nsl_rate'].replace(0, np.nan)).round(2)
    return comp.sort_values('diff', ascending=False)


def calc_effect_diff(group_df):
    """카테고리 내 SL vs Non-SL의 effect 출현율 차이"""
    sl = group_df[group_df['is_sl']]
    nsl = group_df[~group_df['is_sl']]
    n_sl = sl['product_code'].nunique()
    n_nsl = nsl['product_code'].nunique()

    if n_sl == 0 or n_nsl == 0:
        return pd.DataFrame()

    sl_rate = sl.groupby('effect')['product_code'].nunique() / n_sl
    nsl_rate = nsl.groupby('effect')['product_code'].nunique() / n_nsl

    comp = pd.DataFrame({'sl_rate': sl_rate, 'nsl_rate': nsl_rate}).fillna(0)
    comp['diff'] = comp['sl_rate'] - comp['nsl_rate']
    comp['lift'] = (comp['sl_rate'] / comp['nsl_rate'].replace(0, np.nan)).round(2)
    return comp.sort_values('diff', ascending=False)


# ── 4. 시각화: Type by Category (10_07) ──
print("\n=== Type by Category 시각화 ===")

n_cats = len(valid_cats)
fig, axes = plt.subplots(n_cats, 1, figsize=(14, 5 * n_cats))
if n_cats == 1:
    axes = [axes]

for ax, cat in zip(axes, valid_cats):
    cat_df = df[df['category_2'] == cat]
    comp = calc_type_diff(cat_df)

    # Top/Bottom 각 7개
    top = comp.head(7)
    bottom = comp.tail(7)
    show = pd.concat([top, bottom]).drop_duplicates()
    show = show.sort_values('diff', ascending=True)

    n_sl = cat_df[cat_df['is_sl']]['product_code'].nunique()
    n_nsl = cat_df[~cat_df['is_sl']]['product_code'].nunique()

    colors = ['#E74C3C' if v > 0 else '#3498DB' for v in show['diff']]
    bars = ax.barh(range(len(show)), show['diff'] * 100, color=colors, height=0.7)
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels(show.index, fontsize=10)
    ax.set_xlabel('출현율 차이 (%p)', fontsize=11)
    ax.set_title(f'{cat}  (SL {n_sl}개 / Non-SL {n_nsl}개)', fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.8)

    # 값 라벨
    for bar, val in zip(bars, show['diff'] * 100):
        x = bar.get_width()
        ha = 'left' if x >= 0 else 'right'
        offset = 0.3 if x >= 0 else -0.3
        ax.text(x + offset, bar.get_y() + bar.get_height()/2,
                f'{val:+.1f}%p', va='center', ha=ha, fontsize=9)

fig.suptitle('카테고리(중분류)별 성분 계열(Type) — SL vs Non-SL 출현율 차이',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/10_07_ingredient_type_by_category.png', dpi=150, bbox_inches='tight')
print(f"저장: 10_07_ingredient_type_by_category.png")
plt.close()


# ── 5. 시각화: Effect by Category (10_08) ──
print("\n=== Effect by Category 시각화 ===")

fig, axes = plt.subplots(n_cats, 1, figsize=(14, 5 * n_cats))
if n_cats == 1:
    axes = [axes]

for ax, cat in zip(axes, valid_cats):
    cat_df = df[df['category_2'] == cat]
    comp = calc_effect_diff(cat_df)

    top = comp.head(7)
    bottom = comp.tail(7)
    show = pd.concat([top, bottom]).drop_duplicates()
    show = show.sort_values('diff', ascending=True)

    n_sl = cat_df[cat_df['is_sl']]['product_code'].nunique()
    n_nsl = cat_df[~cat_df['is_sl']]['product_code'].nunique()

    colors = ['#E74C3C' if v > 0 else '#3498DB' for v in show['diff']]
    bars = ax.barh(range(len(show)), show['diff'] * 100, color=colors, height=0.7)
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels(show.index, fontsize=10)
    ax.set_xlabel('출현율 차이 (%p)', fontsize=11)
    ax.set_title(f'{cat}  (SL {n_sl}개 / Non-SL {n_nsl}개)', fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.8)

    for bar, val in zip(bars, show['diff'] * 100):
        x = bar.get_width()
        ha = 'left' if x >= 0 else 'right'
        offset = 0.3 if x >= 0 else -0.3
        ax.text(x + offset, bar.get_y() + bar.get_height()/2,
                f'{val:+.1f}%p', va='center', ha=ha, fontsize=9)

fig.suptitle('카테고리(중분류)별 성분 효능(Effect) — SL vs Non-SL 출현율 차이',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/10_08_ingredient_effect_by_category.png', dpi=150, bbox_inches='tight')
print(f"저장: 10_08_ingredient_effect_by_category.png")
plt.close()


# ── 6. 요약 테이블 출력 ──
print("\n" + "="*80)
print("카테고리별 성분 Type Top 3 (SL 특화) / Bottom 3 (Non-SL 특화)")
print("="*80)

for cat in valid_cats:
    cat_df = df[df['category_2'] == cat]
    comp = calc_type_diff(cat_df)
    n_sl = cat_df[cat_df['is_sl']]['product_code'].nunique()
    n_nsl = cat_df[~cat_df['is_sl']]['product_code'].nunique()

    print(f"\n▶ {cat} (SL {n_sl} / Non-SL {n_nsl})")
    top3 = comp.head(3)
    bot3 = comp.tail(3)
    for idx, row in top3.iterrows():
        print(f"  ✅ SL 특화: {idx:20s} +{row['diff']*100:.1f}%p (Lift {row['lift']})")
    for idx, row in bot3.iloc[::-1].iterrows():
        print(f"  ❌ Non-SL:  {idx:20s} {row['diff']*100:.1f}%p (Lift {row['lift']})")

print("\n" + "="*80)
print("카테고리별 성분 Effect Top 3 (SL 특화) / Bottom 3 (Non-SL 특화)")
print("="*80)

for cat in valid_cats:
    cat_df = df[df['category_2'] == cat]
    comp = calc_effect_diff(cat_df)
    n_sl = cat_df[cat_df['is_sl']]['product_code'].nunique()
    n_nsl = cat_df[~cat_df['is_sl']]['product_code'].nunique()

    print(f"\n▶ {cat} (SL {n_sl} / Non-SL {n_nsl})")
    top3 = comp.head(3)
    bot3 = comp.tail(3)
    for idx, row in top3.iterrows():
        print(f"  ✅ SL 특화: {idx:20s} +{row['diff']*100:.1f}%p (Lift {row['lift']})")
    for idx, row in bot3.iloc[::-1].iterrows():
        print(f"  ❌ Non-SL:  {idx:20s} {row['diff']*100:.1f}%p (Lift {row['lift']})")

print("\n완료!")
