"""연착륙 제품 매출 기여도 분석

매출 = 리뷰 수 × 가격
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SLI_DIR = PROJECT_ROOT / "02_outputs" / "Sli"
DATA_DIR = PROJECT_ROOT / "02_outputs" / "00_data" / "csv"
OUTPUT_DIR = PROJECT_ROOT / "02_outputs" / "01_figures" / "SLI"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("연착륙 제품 매출 기여도 분석")
print("=" * 80)

# 1. SLI 결과 로드
print("\n[1/5] 데이터 로드...")
sli = pd.read_csv(SLI_DIR / "sli_integrated_results.csv", encoding='utf-8-sig')
# final_soft_landing을 불리언으로 변환
sli['final_soft_landing'] = sli['final_soft_landing'].astype(bool)
print(f"  SLI 데이터: {len(sli)}개 제품")
print(f"  연착륙 제품: {sli['final_soft_landing'].sum()}개")

# 2. 리뷰 수 데이터 로드 (여러 파일 시도)
review_count = None
try_files = [
    DATA_DIR / "final" / "products_stats.csv",
    DATA_DIR / "analysis" / "products_full.csv",
    PROJECT_ROOT / "02_outputs" / "ABSA" / "snapshot_stage4b_rgx_20260227" / "product_kpi.csv"
]

for file_path in try_files:
    if file_path.exists():
        df = pd.read_csv(file_path)
        if 'product_code' in df.columns and 'review_count' in df.columns:
            review_count = df[['product_code', 'review_count']].copy()
            print(f"  리뷰 수 데이터: {file_path.name}")
            break

if review_count is None:
    print("  ⚠️  리뷰 수 데이터를 찾을 수 없습니다!")
    exit(1)

# 3. 데이터 병합
print("\n[2/5] 데이터 병합...")
df = sli.merge(review_count, on='product_code', how='left')
print(f"  병합 후: {len(df)}개 제품")
print(f"  리뷰 수 결측치: {df['review_count'].isna().sum()}개")

# 결측치 처리 (0으로)
df['review_count'] = df['review_count'].fillna(0)

# 4. 매출 계산
print("\n[3/5] 매출 계산...")
df['revenue'] = df['review_count'] * df['price']
df['soft_landing_group'] = df['final_soft_landing'].map({True: '연착륙(SL)', False: '비연착륙(Non-SL)'})

# 5. 통계 계산
print("\n[4/5] 통계 분석...")

# 전체 통계
total_revenue = df['revenue'].sum()
total_products = len(df)

# 그룹별 통계
stats = df.groupby('soft_landing_group', dropna=False).agg({
    'product_code': 'count',
    'revenue': 'sum',
    'review_count': 'sum',
    'price': 'mean'
}).round(0)

stats.columns = ['제품 수', '총 매출', '총 리뷰 수', '평균 가격']
stats['매출 비중 (%)'] = (stats['총 매출'] / total_revenue * 100).round(1)
stats['제품 비중 (%)'] = (stats['제품 수'] / total_products * 100).round(1)

print("\n" + "=" * 80)
print("📊 연착륙 제품 매출 기여도 분석 결과")
print("=" * 80)
print(f"\n전체 제품 수: {total_products:,}개")
print(f"전체 매출 (추정): {total_revenue:,.0f}원")
if len(stats) > 0:
    print(f"\n{stats.to_string()}")
else:
    print("\n⚠️  그룹별 통계 생성 실패")

# 상위 통계
sl_products = df[df['final_soft_landing'] == True]
non_sl_products = df[df['final_soft_landing'] == False]

print("\n" + "-" * 80)
print("연착륙(SL) 제품 상세")
print("-" * 80)
print(f"제품 수: {len(sl_products):,}개 ({len(sl_products)/total_products*100:.1f}%)")
print(f"총 매출: {sl_products['revenue'].sum():,.0f}원 ({sl_products['revenue'].sum()/total_revenue*100:.1f}%)")
print(f"평균 매출: {sl_products['revenue'].mean():,.0f}원/제품")
print(f"중앙값 매출: {sl_products['revenue'].median():,.0f}원/제품")
print(f"총 리뷰 수: {sl_products['review_count'].sum():,.0f}개")
print(f"평균 리뷰 수: {sl_products['review_count'].mean():.1f}개/제품")
print(f"평균 가격: {sl_products['price'].mean():,.0f}원")

print("\n비연착륙(Non-SL) 제품 상세")
print("-" * 80)
print(f"제품 수: {len(non_sl_products):,}개 ({len(non_sl_products)/total_products*100:.1f}%)")
print(f"총 매출: {non_sl_products['revenue'].sum():,.0f}원 ({non_sl_products['revenue'].sum()/total_revenue*100:.1f}%)")
print(f"평균 매출: {non_sl_products['revenue'].mean():,.0f}원/제품")
print(f"중앙값 매출: {non_sl_products['revenue'].median():,.0f}원/제품")
print(f"총 리뷰 수: {non_sl_products['review_count'].sum():,.0f}개")
print(f"평균 리뷰 수: {non_sl_products['review_count'].mean():.1f}개/제품")
print(f"평균 가격: {non_sl_products['price'].mean():,.0f}원")

# 6. 시각화
print("\n[5/5] 시각화 생성...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('연착륙 제품 매출 기여도 분석', fontsize=16, fontweight='bold')

# (1) 매출 비중 파이 차트
ax1 = axes[0, 0]
revenue_data = stats['총 매출']
colors = ['#FF6B6B', '#4ECDC4']
wedges, texts, autotexts = ax1.pie(
    revenue_data,
    labels=revenue_data.index,
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    textprops={'fontsize': 12}
)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(14)
ax1.set_title('총 매출 비중', fontsize=14, fontweight='bold', pad=20)

# (2) 제품 수 vs 매출 비교
ax2 = axes[0, 1]
x = range(len(stats))
width = 0.35
bars1 = ax2.bar([i - width/2 for i in x], stats['제품 비중 (%)'], width,
                label='제품 수 비중', color='#95E1D3', alpha=0.8)
bars2 = ax2.bar([i + width/2 for i in x], stats['매출 비중 (%)'], width,
                label='매출 비중', color='#F38181', alpha=0.8)
ax2.set_xlabel('그룹', fontsize=12)
ax2.set_ylabel('비중 (%)', fontsize=12)
ax2.set_title('제품 수 vs 매출 비중 비교', fontsize=14, fontweight='bold', pad=20)
ax2.set_xticks(x)
ax2.set_xticklabels(stats.index)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 바 위에 값 표시
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)

# (3) 매출 분포 히스토그램 (로그 스케일)
ax3 = axes[1, 0]
df_plot = df[df['revenue'] > 0].copy()  # 0 제외
sl_revenue = df_plot[df_plot['soft_landing_group'] == '연착륙(SL)']['revenue']
non_sl_revenue = df_plot[df_plot['soft_landing_group'] == '비연착륙(Non-SL)']['revenue']

ax3.hist([sl_revenue, non_sl_revenue], bins=30, label=['연착륙(SL)', '비연착륙(Non-SL)'],
         color=colors, alpha=0.7)
ax3.set_xscale('log')
ax3.set_xlabel('매출 (원, 로그 스케일)', fontsize=12)
ax3.set_ylabel('제품 수', fontsize=12)
ax3.set_title('매출 분포 (히스토그램)', fontsize=14, fontweight='bold', pad=20)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# (4) 상위 20개 제품 매출
ax4 = axes[1, 1]
top20 = df.nlargest(20, 'revenue')[['name', 'revenue', 'soft_landing_group']].copy()
top20['name_short'] = top20['name'].str[:30]
colors_map = {'연착륙(SL)': '#FF6B6B', '비연착륙(Non-SL)': '#4ECDC4'}
bar_colors = top20['soft_landing_group'].map(colors_map)

bars = ax4.barh(range(len(top20)), top20['revenue'], color=bar_colors, alpha=0.8)
ax4.set_yticks(range(len(top20)))
ax4.set_yticklabels(top20['name_short'], fontsize=9)
ax4.set_xlabel('매출 (원)', fontsize=12)
ax4.set_title('매출 상위 20개 제품', fontsize=14, fontweight='bold', pad=20)
ax4.invert_yaxis()
ax4.grid(axis='x', alpha=0.3)

# 범례 추가
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FF6B6B', label='연착륙(SL)', alpha=0.8),
    Patch(facecolor='#4ECDC4', label='비연착륙(Non-SL)', alpha=0.8)
]
ax4.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
output_path = OUTPUT_DIR / "soft_landing_revenue_analysis.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"  시각화 저장: {output_path}")

# 7. 결과 CSV 저장
result_summary = pd.DataFrame({
    '구분': ['전체', '연착륙(SL)', '비연착륙(Non-SL)'],
    '제품 수': [total_products, len(sl_products), len(non_sl_products)],
    '총 매출': [total_revenue, sl_products['revenue'].sum(), non_sl_products['revenue'].sum()],
    '평균 매출': [df['revenue'].mean(), sl_products['revenue'].mean(), non_sl_products['revenue'].mean()],
    '총 리뷰 수': [df['review_count'].sum(), sl_products['review_count'].sum(), non_sl_products['review_count'].sum()],
    '평균 리뷰 수': [df['review_count'].mean(), sl_products['review_count'].mean(), non_sl_products['review_count'].mean()],
    '평균 가격': [df['price'].mean(), sl_products['price'].mean(), non_sl_products['price'].mean()]
})

result_path = SLI_DIR / "soft_landing_revenue_summary.csv"
result_summary.to_csv(result_path, index=False, encoding='utf-8-sig')
print(f"\n결과 저장: {result_path}")

print("\n" + "=" * 80)
print("분석 완료!")
print("=" * 80)
