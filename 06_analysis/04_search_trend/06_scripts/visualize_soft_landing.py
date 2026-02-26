"""
네이버 검색 트렌드 시각화 - 연착륙 제품 146개

SLI 연착륙 제품 146개의 세그먼트별 검색 트렌드를 시각화한다.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

# 한글 폰트
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# ── 경로 설정 ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "02_outputs" / "Search_Trend"
CHART_DIR = PROJECT_ROOT / "02_outputs" / "01_figures" / "search_trend" / "soft_landing"
CHART_DIR.mkdir(parents=True, exist_ok=True)

# 색상 팔레트
COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6",
    "#1ABC9C", "#E67E22", "#34495E", "#E91E63", "#00BCD4",
]

# ── 데이터 로드 ────────────────────────────────────────
print("데이터 로드 중...")
detail_df = pd.read_csv(DATA_DIR / "soft_landing_segment_detail_20260227.csv")
detail_df["period"] = pd.to_datetime(detail_df["period"])

summary_df = pd.read_csv(DATA_DIR / "soft_landing_segment_summary_20260227.csv")

print(f"상세 데이터 행 수: {len(detail_df):,}")
print(f"기간: {detail_df['period'].min().strftime('%Y-%m')} ~ {detail_df['period'].max().strftime('%Y-%m')}")
print(f"제품 수: {detail_df['keyword_group'].nunique()}")
print(f"세그먼트별 요약: {len(summary_df):,}행\n")


# ── 유틸리티 함수 ──────────────────────────────────────
def plot_bar_ranking(
    df: pd.DataFrame,
    col_name: str,
    col_value: str,
    title: str,
    filename: str = None,
    top_n: int = 20,
    figsize: tuple = (14, 8),
    xlabel: str = "정규화 점수",
):
    """바 차트 랭킹 시각화"""
    data = df.head(top_n).copy()

    fig, ax = plt.subplots(figsize=figsize)
    colors_bar = ["#E74C3C" if i == 0 else "#3498DB" for i in range(len(data))]
    ax.barh(range(len(data) - 1, -1, -1), data[col_value], color=colors_bar, edgecolor="white")

    ax.set_yticks(range(len(data) - 1, -1, -1))
    ax.set_yticklabels(data[col_name], fontsize=10)

    for i, (_, row) in enumerate(data.iterrows()):
        ax.text(row[col_value] + (max(data[col_value]) * 0.01), len(data) - 1 - i, f"{row[col_value]:.1f}",
                va="center", fontsize=9, fontweight="bold")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--", axis="x")
    plt.tight_layout()

    if filename:
        path = CHART_DIR / filename
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  저장: {filename}")

    plt.close(fig)


def plot_segment_comparison_bar(
    summary_df: pd.DataFrame,
    seg_type: str,
    products: list,
    title: str,
    filename: str = None,
    figsize: tuple = (16, 6),
):
    """세그먼트별 비교 바 차트"""
    sub = summary_df[
        (summary_df["segment_type"] == seg_type) &
        (summary_df["product_name"].isin(products))
    ].copy()

    if sub.empty:
        print(f"  [경고] {seg_type} 데이터 없음")
        return

    pivot = sub.pivot_table(
        index="product_name", columns="segment_label",
        values="avg_ratio", aggfunc="first"
    ).reindex(products).dropna(how="all")

    fig, ax = plt.subplots(figsize=figsize)
    pivot.plot(kind="bar", ax=ax, color=COLORS[:pivot.shape[1]], edgecolor="white", width=0.8)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("평균 ratio", fontsize=11)
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    ax.legend(title=seg_type, fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    plt.tight_layout()

    if filename:
        path = CHART_DIR / filename
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  저장: {filename}")

    plt.close(fig)


def plot_timeseries(
    df: pd.DataFrame,
    products: list[str],
    title: str,
    filename: str = None,
    figsize: tuple = (14, 6),
    highlight: str = None,
):
    """제품별 시계열 차트 생성"""
    fig, ax = plt.subplots(figsize=figsize)

    for i, product in enumerate(products):
        sub = df[df["keyword_group"] == product].sort_values("period")
        if sub.empty:
            continue

        color = COLORS[i % len(COLORS)]
        lw = 2.5 if product == highlight else 1.8
        alpha = 1.0 if product == highlight else 0.8
        zorder = 10 if product == highlight else 5

        ax.plot(
            sub["period"], sub["ratio"],
            label=product, color=color,
            linewidth=lw, alpha=alpha, zorder=zorder,
            marker="o", markersize=3,
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("검색 트렌드 (ratio)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha="right")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(
        df["period"].min() - pd.Timedelta(days=15),
        df["period"].max() + pd.Timedelta(days=15),
    )
    plt.tight_layout()

    if filename:
        path = CHART_DIR / filename
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  저장: {filename}")

    plt.close(fig)


# ── 1. 앵커 정규화 TOP 20 바차트 ────────────────────────
print("1. 앵커 정규화 TOP 20 바차트 생성 중...")
base_summary = summary_df[summary_df["segment_label"] == "전체"].copy()
base_ranking = base_summary.sort_values("avg_ratio", ascending=False).reset_index(drop=True)
base_ranking.index = range(1, len(base_ranking) + 1)
base_ranking.index.name = "순위"

print(f"  전체 평균 ratio TOP 5:")
for _, row in base_ranking.head(5).iterrows():
    print(f"    {row.name}위: {row['product_name']} ({row['avg_ratio']:.2f})")

plot_bar_ranking(
    base_ranking,
    "product_name",
    "avg_ratio",
    "연착륙 제품 146개 - 전체 평균 검색량 TOP 20",
    filename="01_anchor_top20_bar.png",
    xlabel="평균 ratio",
)


# ── 2. 결승전 결과 비교 ────────────────────────────────
print("\n2. 결승전 결과 비교 시각화 중...")
segment_winners = []

# 전체
overall = base_ranking.iloc[0]
segment_winners.append({"세그먼트": "전체", "제품": overall["product_name"], "평균 ratio": overall["avg_ratio"]})

# 성별
for label in ["남성", "여성"]:
    sub = summary_df[(summary_df["segment_type"] == "gender") & (summary_df["segment_label"] == label)]
    if not sub.empty:
        winner = sub.sort_values("avg_ratio", ascending=False).iloc[0]
        segment_winners.append({"세그먼트": f"성별-{label}", "제품": winner["product_name"], "평균 ratio": winner["avg_ratio"]})

# 연령대
for label in ["10대", "20대", "30대", "40대", "50대", "60대"]:
    sub = summary_df[(summary_df["segment_type"] == "age") & (summary_df["segment_label"] == label)]
    if not sub.empty:
        winner = sub.sort_values("avg_ratio", ascending=False).iloc[0]
        segment_winners.append({"세그먼트": f"연령-{label}", "제품": winner["product_name"], "평균 ratio": winner["avg_ratio"]})

# 기기
for label in ["PC", "모바일"]:
    sub = summary_df[(summary_df["segment_type"] == "device") & (summary_df["segment_label"] == label)]
    if not sub.empty:
        winner = sub.sort_values("avg_ratio", ascending=False).iloc[0]
        segment_winners.append({"세그먼트": f"기기-{label}", "제품": winner["product_name"], "평균 ratio": winner["avg_ratio"]})

winners_df = pd.DataFrame(segment_winners)

fig, ax = plt.subplots(figsize=(14, 7))
colors = ["#E74C3C" if i == 0 else "#3498DB" for i in range(len(winners_df))]
ax.barh(range(len(winners_df) - 1, -1, -1), winners_df["평균 ratio"], color=colors, edgecolor="white")

ax.set_yticks(range(len(winners_df) - 1, -1, -1))
ax.set_yticklabels(winners_df["세그먼트"], fontsize=10)

for i, (_, row) in enumerate(winners_df.iterrows()):
    product_short = row["제품"][:30] + "..." if len(row["제품"]) > 30 else row["제품"]
    ax.text(row["평균 ratio"] + 0.5, len(winners_df) - 1 - i, f"{product_short} ({row['평균 ratio']:.1f})",
            va="center", fontsize=8, fontweight="bold")

ax.set_title("세그먼트별 결승전 1위 제품 비교", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("평균 ratio", fontsize=11)
ax.grid(True, alpha=0.3, linestyle="--", axis="x")
plt.tight_layout()

fig.savefig(CHART_DIR / "02_finals_winners_comparison.png", dpi=150, bbox_inches="tight", facecolor="white")
print("  저장: 02_finals_winners_comparison.png")
plt.close(fig)


# ── 3. 성별 비교 ───────────────────────────────────────
print("\n3. 성별 비교 시각화 중...")
male_top5 = summary_df[
    (summary_df["segment_type"] == "gender") & (summary_df["segment_label"] == "남성")
].sort_values("avg_ratio", ascending=False).head(5)["product_name"].tolist()

female_top5 = summary_df[
    (summary_df["segment_type"] == "gender") & (summary_df["segment_label"] == "여성")
].sort_values("avg_ratio", ascending=False).head(5)["product_name"].tolist()

gender_products = list(set(male_top5 + female_top5))

plot_segment_comparison_bar(
    summary_df, "gender", gender_products,
    "성별 비교 — TOP 5 제품",
    filename="03_gender_comparison.png",
)


# ── 4. 연령대별 비교 ───────────────────────────────────
print("\n4. 연령대별 비교 시각화 중...")
age_products = set()
for age in ["10대", "20대", "30대", "40대", "50대", "60대"]:
    sub = summary_df[
        (summary_df["segment_type"] == "age") & (summary_df["segment_label"] == age)
    ].sort_values("avg_ratio", ascending=False).head(5)
    age_products.update(sub["product_name"].tolist())

age_products = list(age_products)

plot_segment_comparison_bar(
    summary_df, "age", age_products,
    "연령대별 비교 — TOP 5 제품",
    filename="04_age_comparison.png",
    figsize=(18, 8),
)


# ── 5. 기기별 비교 ─────────────────────────────────────
print("\n5. 기기별 비교 시각화 중...")
pc_top5 = summary_df[
    (summary_df["segment_type"] == "device") & (summary_df["segment_label"] == "PC")
].sort_values("avg_ratio", ascending=False).head(5)["product_name"].tolist()

mobile_top5 = summary_df[
    (summary_df["segment_type"] == "device") & (summary_df["segment_label"] == "모바일")
].sort_values("avg_ratio", ascending=False).head(5)["product_name"].tolist()

device_products = list(set(pc_top5 + mobile_top5))

plot_segment_comparison_bar(
    summary_df, "device", device_products,
    "기기별 비교 — TOP 5 제품",
    filename="05_device_comparison.png",
)


# ── 6. 브랜드별 포트폴리오 ─────────────────────────────
print("\n6. 브랜드별 포트폴리오 시각화 중...")
top20_brands = base_ranking.head(20)["brand_name"].value_counts().reset_index()
top20_brands.columns = ["브랜드", "제품수"]

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(range(len(top20_brands)), top20_brands["제품수"],
      color=COLORS[:len(top20_brands)] * 3, edgecolor="white")

ax.set_xticks(range(len(top20_brands)))
ax.set_xticklabels(top20_brands["브랜드"], rotation=45, ha="right", fontsize=10)

for i, (_, row) in enumerate(top20_brands.iterrows()):
    ax.text(i, row["제품수"] + 0.1, str(row["제품수"]),
            ha="center", fontsize=10, fontweight="bold")

ax.set_title("브랜드별 TOP 20 진입 제품 수", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("제품 수", fontsize=11)
ax.grid(True, alpha=0.3, linestyle="--", axis="y")
plt.tight_layout()

fig.savefig(CHART_DIR / "06_brand_portfolio.png", dpi=150, bbox_inches="tight", facecolor="white")
print("  저장: 06_brand_portfolio.png")
plt.close(fig)


# ── 7. 시계열 분석 ─────────────────────────────────────
print("\n7. 시계열 분석 시각화 중...")
base_detail = detail_df[detail_df["segment_label"] == "전체"].copy()
top10_products = base_ranking.head(10)["product_name"].tolist()

plot_timeseries(
    base_detail, top10_products,
    "연착륙 제품 TOP 10 - 월별 검색 트렌드",
    filename="07_top10_timeseries.png",
    highlight=top10_products[0] if top10_products else None,
)


# ── 8. 트렌드 패턴 분포 ────────────────────────────────
print("\n8. 트렌드 패턴 분포 시각화 중...")

def classify_trend(product_name, df):
    sub = df[(df["keyword_group"] == product_name) & (df["segment_label"] == "전체")].sort_values("period")
    if len(sub) < 6:
        return "데이터부족"

    first_3 = sub.head(3)["ratio"].mean()
    last_3 = sub.tail(3)["ratio"].mean()

    if last_3 > first_3 * 1.2:
        return "상승"
    elif last_3 < first_3 * 0.8:
        return "하락"
    else:
        return "안정"

all_products = base_ranking["product_name"].tolist()
trend_patterns = {p: classify_trend(p, detail_df) for p in all_products}
trend_df = pd.DataFrame(list(trend_patterns.items()), columns=["제품", "트렌드"])

trend_counts = trend_df["트렌드"].value_counts().reset_index()
trend_counts.columns = ["트렌드", "제품수"]

fig, ax = plt.subplots(figsize=(10, 6))
colors_trend = {"상승": "#2ECC71", "안정": "#3498DB", "하락": "#E74C3C", "데이터부족": "#95A5A6"}
bar_colors = [colors_trend.get(t, "#95A5A6") for t in trend_counts["트렌드"]]

ax.bar(trend_counts["트렌드"], trend_counts["제품수"], color=bar_colors, edgecolor="white")

for i, (_, row) in enumerate(trend_counts.iterrows()):
    ax.text(i, row["제품수"] + 1, str(row["제품수"]),
            ha="center", fontsize=11, fontweight="bold")

ax.set_title("연착륙 제품 트렌드 패턴 분포", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("제품 수", fontsize=11)
ax.set_xlabel("트렌드", fontsize=11)
ax.grid(True, alpha=0.3, linestyle="--", axis="y")
plt.tight_layout()

fig.savefig(CHART_DIR / "08_trend_pattern_distribution.png", dpi=150, bbox_inches="tight", facecolor="white")
print("  저장: 08_trend_pattern_distribution.png")
plt.close(fig)

print(f"\n=== 완료 ===")
print(f"차트 저장 완료: {CHART_DIR}")
print(f"생성된 차트: {len(list(CHART_DIR.glob('*.png')))}개")
for p in sorted(CHART_DIR.glob("*.png")):
    print(f"  {p.name}")
