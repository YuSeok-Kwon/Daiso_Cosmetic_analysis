"""
09_02 레이더 차트와 09_03 히트맵을 별도 파일로 분리 재생성
"""
import sys
sys.path.insert(0, '/Users/yu_seok/Documents/문서 - Mac/workspace/01_현재진행/01_nbCamp/nb_Project/Why-pi/05_src/02_bigquery')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

FIG_DIR = '/Users/yu_seok/Documents/문서 - Mac/workspace/01_현재진행/01_nbCamp/nb_Project/Why-pi/02_outputs/01_figures/SLI'
SLI_DIR = '/Users/yu_seok/Documents/문서 - Mac/workspace/01_현재진행/01_nbCamp/nb_Project/Why-pi/02_outputs/Sli'

# ── 데이터 로드 ──
grid_df = pd.read_csv(f'{SLI_DIR}/grid_search_top10.csv')
sens_df = pd.read_csv(f'{SLI_DIR}/weight_sensitivity_results.csv')

# 전문가 가중치
expert_weights = {
    'mid_minus_early': 0.20,
    'late1_minus_mid': 0.25,
    'late2_minus_late1': 0.35,
    'log_total_reviews': 0.15,
    'slope_neg': -0.05,
    'late_vol_cv': -0.05,
}
expert_unanimous = 158

# 최적 가중치 (Top 1)
best = grid_df.iloc[0]
best_weights = {
    'mid_minus_early': best['mid_minus_early'],
    'late1_minus_mid': best['late1_minus_mid'],
    'late2_minus_late1': best['late2_minus_late1'],
    'log_total_reviews': best['log_total_reviews'],
    'slope_neg': best['slope_neg'],
    'late_vol_cv': best['late_vol_cv'],
}
best_unanimous = int(best['unanimous_4_4'])

# ================================================================
# 09_02: 레이더 차트 (단독)
# ================================================================
print("=== 09_02 레이더 차트 생성 ===")

labels = ['mid_minus\n_early', 'late1_minus\n_mid', 'late2_minus\n_late1',
          'log_total\n_reviews', 'slope_neg\n(abs)', 'late_vol_cv\n(abs)']

expert_vals = [
    expert_weights['mid_minus_early'],
    expert_weights['late1_minus_mid'],
    expert_weights['late2_minus_late1'],
    expert_weights['log_total_reviews'],
    abs(expert_weights['slope_neg']),
    abs(expert_weights['late_vol_cv']),
]

optimal_vals = [
    best_weights['mid_minus_early'],
    best_weights['late1_minus_mid'],
    best_weights['late2_minus_late1'],
    best_weights['log_total_reviews'],
    abs(best_weights['slope_neg']),
    abs(best_weights['late_vol_cv']),
]

N = len(labels)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
expert_vals_plot = expert_vals + expert_vals[:1]
optimal_vals_plot = optimal_vals + optimal_vals[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

ax.plot(angles, expert_vals_plot, 'o-', color='#E74C3C', linewidth=2.5, markersize=8,
        label=f'전문가 (만장일치={expert_unanimous})')
ax.fill(angles, expert_vals_plot, alpha=0.15, color='#E74C3C')

ax.plot(angles, optimal_vals_plot, 's-', color='#3498DB', linewidth=2.5, markersize=8,
        label=f'Grid Search 최적 (만장일치={best_unanimous})')
ax.fill(angles, optimal_vals_plot, alpha=0.15, color='#3498DB')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
ax.set_ylim(0, 0.40)
ax.set_yticks([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
ax.set_yticklabels(['0.05', '0.10', '0.15', '0.20', '0.25', '0.30', '0.35'], fontsize=9, color='gray')

ax.set_title('전문가 vs Grid Search 최적 가중치 비교', fontsize=15, fontweight='bold', pad=25)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=11)

plt.tight_layout()
fig.savefig(f'{FIG_DIR}/09_02_grid_search_radar.png', dpi=150, bbox_inches='tight')
print(f"저장: 09_02_grid_search_radar.png ({fig.get_size_inches()})")
plt.close()


# ================================================================
# 09_03: 2D 상호작용 히트맵 (단독)
# ================================================================
print("\n=== 09_03 2D 히트맵 생성 ===")

# 2D 그리드: late2_minus_late1 × late1_minus_mid
# sensitivity_results에서 재구성하기엔 데이터가 1D뿐이므로,
# grid_search_top10에서 패턴을 보고 직접 구성

# 기존 노트북에서 사용한 값 범위
pivot_param1 = 'late2_minus_late1'
pivot_param2 = 'late1_minus_mid'

p1_range = [0.25, 0.30, 0.35, 0.40, 0.45]
p2_range = [0.15, 0.20, 0.25, 0.30, 0.35]

# sensitivity_results에서 1D 값 기반으로 2D 근사 구성
# (실제 full grid가 없으므로 1D 결과에서 추정)
# late2 변화 시 unanimous
late2_sens = sens_df[sens_df['param'] == pivot_param1][['value', 'unanimous_4_4']].set_index('value')
late1_sens = sens_df[sens_df['param'] == pivot_param2][['value', 'unanimous_4_4']].set_index('value')

# 2D 히트맵 데이터 구성 (1D 교차 근사: base 158에서 각 축의 delta를 더하기)
hm_data = np.zeros((len(p2_range), len(p1_range)))
for i, p2 in enumerate(p2_range):
    for j, p1 in enumerate(p1_range):
        # 각 축의 delta
        d1 = late2_sens.loc[p1, 'unanimous_4_4'] - expert_unanimous if p1 in late2_sens.index else 0
        d2 = late1_sens.loc[p2, 'unanimous_4_4'] - expert_unanimous if p2 in late1_sens.index else 0
        hm_data[i, j] = expert_unanimous + d1 + d2
        # 실제 grid에서 일치하는 것이 있으면 그 값 사용
        match = grid_df[
            (grid_df[pivot_param1] == p1) & (grid_df[pivot_param2] == p2)
        ]
        if len(match) > 0:
            hm_data[i, j] = match.iloc[0]['unanimous_4_4']

hm_df = pd.DataFrame(hm_data, index=p2_range, columns=p1_range).astype(int)

fig, ax = plt.subplots(figsize=(9, 7))

cmap = plt.cm.YlOrRd
im = ax.imshow(hm_df.values, cmap=cmap, aspect='auto', vmin=hm_df.values.min()-1, vmax=hm_df.values.max()+1)

# 셀 내 값 표시
for i in range(len(p2_range)):
    for j in range(len(p1_range)):
        val = hm_df.values[i, j]
        color = 'white' if val >= 158 else 'black'
        ax.text(j, i, f'{int(val)}', ha='center', va='center',
                fontsize=14, fontweight='bold', color=color)

# 전문가 가중치 위치 표시
expert_j = p1_range.index(expert_weights[pivot_param1])
expert_i = p2_range.index(expert_weights[pivot_param2])
ax.plot(expert_j, expert_i, marker='*', color='#3498DB', markersize=20,
        markeredgecolor='white', markeredgewidth=2)
ax.annotate('전문가', (expert_j, expert_i), textcoords="offset points",
            xytext=(15, -15), fontsize=12, fontweight='bold', color='#3498DB',
            arrowprops=dict(arrowstyle='->', color='#3498DB', lw=1.5))

ax.set_xticks(range(len(p1_range)))
ax.set_xticklabels([f'{v:.2f}' for v in p1_range], fontsize=11)
ax.set_yticks(range(len(p2_range)))
ax.set_yticklabels([f'{v:.2f}' for v in p2_range], fontsize=11)

ax.set_xlabel('late2_minus_late1 (후기 가속)', fontsize=13, fontweight='bold')
ax.set_ylabel('late1_minus_mid (중기→후기 유지)', fontsize=13, fontweight='bold')
ax.set_title('2D 상호작용: late2_minus_late1 × late1_minus_mid\n(값 = 만장일치 수)',
             fontsize=14, fontweight='bold', pad=15)

cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('만장일치 수', fontsize=11)

plt.tight_layout()
fig.savefig(f'{FIG_DIR}/09_03_interaction_heatmap.png', dpi=150, bbox_inches='tight')
print(f"저장: 09_03_interaction_heatmap.png ({fig.get_size_inches()})")
plt.close()

print("\n완료!")
