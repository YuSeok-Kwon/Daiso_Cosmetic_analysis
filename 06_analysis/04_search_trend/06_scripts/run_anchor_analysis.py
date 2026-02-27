"""앵커 기반 SL vs Non-SL 시계열 비교분석 + 시각화

입력: search_trend_anchor_normalized_{date}.csv
출력:
    - 02_outputs/01_figures/SLI/08_01 ~ 08_09 시각화
    - 분석 결과 CSV
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# 한글 폰트 설정
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# 경로
OUTPUT_DIR = _PROJECT_ROOT / "02_outputs" / "Search_Trend"
FIG_DIR = _PROJECT_ROOT / "02_outputs" / "01_figures" / "SLI"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 색상 팔레트
SL_COLOR = "#4A90D9"
NON_SL_COLOR = "#E74C3C"
ANCHOR_COLOR = "#2ECC71"

ANCHOR_PRODUCT_CODE = 1035082


def load_data(date_str: str = None) -> pd.DataFrame:
    """앵커 정규화 데이터 로드"""
    if date_str:
        path = OUTPUT_DIR / f"search_trend_anchor_normalized_{date_str}.csv"
    else:
        files = sorted(OUTPUT_DIR.glob("search_trend_anchor_normalized_*.csv"))
        if not files:
            raise FileNotFoundError("search_trend_anchor_normalized_*.csv 파일을 찾을 수 없습니다.")
        path = files[-1]

    df = pd.read_csv(path)
    df["period"] = pd.to_datetime(df["period"])
    df["is_sl"] = df["is_sl"].astype(bool)
    print(f"로드: {path.name} ({len(df)}행, {df['product_code'].nunique()}개 제품)")
    return df


def filter_valid_products(df: pd.DataFrame, min_periods: int = 10) -> pd.DataFrame:
    """충분한 시점 데이터가 있는 제품만 필터링"""
    counts = df.groupby("product_code")["period"].nunique()
    valid = counts[counts >= min_periods].index
    filtered = df[df["product_code"].isin(valid)].copy()
    print(f"유효 제품 (≥{min_periods}시점): {len(valid)}개 / 전체 {df['product_code'].nunique()}개")
    return filtered


def compute_product_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """제품별 시계열 지표 산출"""
    metrics = []

    for pcode, grp in df.groupby("product_code"):
        grp = grp.sort_values("period")
        ratios = grp["normalized_ratio"].values
        is_sl = grp["is_sl"].iloc[0]
        brand = grp["brand_name"].iloc[0]
        name = grp["product_name"].iloc[0]

        n = len(ratios)
        if n < 4:
            continue

        # 1. max 대비 정규화 (형태 비교용)
        max_r = ratios.max()
        if max_r > 0:
            shape_norm = ratios / max_r
        else:
            continue

        # 2. 트렌드 분류 (전반/후반 비교)
        half = n // 2
        first_half_mean = ratios[:half].mean()
        second_half_mean = ratios[half:].mean()

        if first_half_mean > 0:
            change_rate = (second_half_mean - first_half_mean) / first_half_mean
        else:
            change_rate = 0

        if change_rate > 0.2:
            trend = "상승"
        elif change_rate < -0.2:
            trend = "하락"
        else:
            trend = "안정"

        # 3. 피크 후 잔존율 (최근 3개월 평균 / 최대값)
        recent_3 = ratios[-3:].mean() if n >= 3 else ratios[-1]
        retention = recent_3 / max_r

        # 4. CV (변동계수)
        mean_r = ratios.mean()
        if mean_r > 0:
            cv = ratios.std() / mean_r
        else:
            cv = np.nan

        # 5. 앵커 대비 평균 normalized_ratio
        avg_norm = ratios.mean()
        median_norm = np.median(ratios)

        metrics.append({
            "product_code": pcode,
            "brand_name": brand,
            "product_name": name,
            "is_sl": is_sl,
            "n_periods": n,
            "avg_normalized_ratio": avg_norm,
            "median_normalized_ratio": median_norm,
            "max_normalized_ratio": max_r,
            "change_rate": change_rate,
            "trend": trend,
            "retention": retention,
            "cv": cv,
        })

    return pd.DataFrame(metrics)


def stat_test(sl_vals, non_sl_vals, label: str) -> dict:
    """Mann-Whitney U 검정"""
    sl_clean = sl_vals.dropna()
    non_sl_clean = non_sl_vals.dropna()
    if len(sl_clean) < 5 or len(non_sl_clean) < 5:
        return {"label": label, "U": np.nan, "p": np.nan, "r": np.nan}

    u, p = stats.mannwhitneyu(sl_clean, non_sl_clean, alternative="two-sided")
    n = len(sl_clean) + len(non_sl_clean)
    r = abs((u - len(sl_clean) * len(non_sl_clean) / 2) / (len(sl_clean) * len(non_sl_clean) / 2))
    r = min(r, 1.0)
    return {"label": label, "U": u, "p": p, "r": r}


# ── 시각화 함수들 ──────────────────────────────────────────


def plot_08_01_timeseries_normalized(df: pd.DataFrame, metrics_df: pd.DataFrame):
    """08_01: 제품별 max 대비 정규화 시계열 비교 (방법 C)"""
    fig, ax = plt.subplots(figsize=(14, 7))

    # 제품별 max 대비 정규화
    for is_sl, color, label in [(True, SL_COLOR, "SL"), (False, NON_SL_COLOR, "Non-SL")]:
        sub = df[df["is_sl"] == is_sl].copy()
        monthly = []
        for pcode, grp in sub.groupby("product_code"):
            grp = grp.sort_values("period")
            max_r = grp["normalized_ratio"].max()
            if max_r > 0:
                grp = grp.copy()
                grp["shape_norm"] = grp["normalized_ratio"] / max_r
                monthly.append(grp[["period", "shape_norm"]])

        if monthly:
            all_monthly = pd.concat(monthly)
            agg = all_monthly.groupby("period")["shape_norm"].agg(["mean", "sem"])
            ax.plot(agg.index, agg["mean"], color=color, linewidth=2.5, label=f"{label} (n={len(monthly)})")
            ax.fill_between(agg.index, agg["mean"] - agg["sem"], agg["mean"] + agg["sem"],
                            alpha=0.15, color=color)

    ax.axvline(pd.Timestamp("2024-09-01"), color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(pd.Timestamp("2024-09-01"), ax.get_ylim()[1] * 0.95, " 2024-09",
            fontsize=9, color="gray", va="top")

    ax.set_xlabel("기간", fontsize=12)
    ax.set_ylabel("정규화 검색지수 (제품별 max=1)", fontsize=12)
    ax.set_title("SL vs Non-SL 검색트렌드 비교 (제품별 max 대비 정규화)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_01_timeseries_normalized_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  08_01 저장 완료")


def plot_08_02_trend_pattern(metrics_df: pd.DataFrame):
    """08_02: 트렌드 패턴 분류 비교"""
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ["상승", "안정", "하락"]
    sl_pcts = []
    non_sl_pcts = []

    sl_total = len(metrics_df[metrics_df["is_sl"]])
    non_sl_total = len(metrics_df[~metrics_df["is_sl"]])

    for cat in categories:
        sl_count = len(metrics_df[(metrics_df["is_sl"]) & (metrics_df["trend"] == cat)])
        non_sl_count = len(metrics_df[(~metrics_df["is_sl"]) & (metrics_df["trend"] == cat)])
        sl_pcts.append(sl_count / sl_total * 100 if sl_total > 0 else 0)
        non_sl_pcts.append(non_sl_count / non_sl_total * 100 if non_sl_total > 0 else 0)

    x = np.arange(len(categories))
    w = 0.35

    bars1 = ax.bar(x - w / 2, sl_pcts, w, label=f"SL (n={sl_total})", color=SL_COLOR, alpha=0.85)
    bars2 = ax.bar(x + w / 2, non_sl_pcts, w, label=f"Non-SL (n={non_sl_total})", color=NON_SL_COLOR, alpha=0.85)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel("비율 (%)", fontsize=12)
    ax.set_title("검색 트렌드 패턴 분류 (SL vs Non-SL)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_02_trend_pattern_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  08_02 저장 완료")


def plot_08_03_retention_distribution(metrics_df: pd.DataFrame):
    """08_03: 잔존율 분포 비교"""
    fig, ax = plt.subplots(figsize=(12, 6))

    sl_ret = metrics_df[metrics_df["is_sl"]]["retention"].dropna()
    non_sl_ret = metrics_df[~metrics_df["is_sl"]]["retention"].dropna()

    bins = np.linspace(0, 1, 30)
    ax.hist(sl_ret, bins=bins, alpha=0.6, color=SL_COLOR, label=f"SL (중앙값={sl_ret.median():.3f})", density=True)
    ax.hist(non_sl_ret, bins=bins, alpha=0.6, color=NON_SL_COLOR, label=f"Non-SL (중앙값={non_sl_ret.median():.3f})", density=True)

    ax.axvline(sl_ret.median(), color=SL_COLOR, linestyle="--", linewidth=1.5)
    ax.axvline(non_sl_ret.median(), color=NON_SL_COLOR, linestyle="--", linewidth=1.5)

    # 30~50% 구간 강조
    ax.axvspan(0.3, 0.5, alpha=0.1, color=ANCHOR_COLOR, label="30~50% 구간")

    ax.set_xlabel("잔존율 (최근3개월 / 최대값)", fontsize=12)
    ax.set_ylabel("밀도", fontsize=12)
    ax.set_title("피크 후 잔존율 분포 (SL vs Non-SL)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_03_retention_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  08_03 저장 완료")


def plot_08_04_cv_boxplot(metrics_df: pd.DataFrame):
    """08_04: CV 박스플롯"""
    fig, ax = plt.subplots(figsize=(8, 7))

    sl_cv = metrics_df[metrics_df["is_sl"]]["cv"].dropna()
    non_sl_cv = metrics_df[~metrics_df["is_sl"]]["cv"].dropna()

    bp = ax.boxplot([sl_cv, non_sl_cv], labels=["SL", "Non-SL"],
                    patch_artist=True, widths=0.5, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black"))

    bp["boxes"][0].set_facecolor(SL_COLOR)
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor(NON_SL_COLOR)
    bp["boxes"][1].set_alpha(0.6)

    # 개별 점 오버레이
    for i, (data, color) in enumerate([(sl_cv, SL_COLOR), (non_sl_cv, NON_SL_COLOR)]):
        x = np.random.normal(i + 1, 0.04, size=len(data))
        ax.scatter(x, data, alpha=0.3, s=15, color=color, zorder=2)

    u_test = stat_test(sl_cv, non_sl_cv, "CV")
    sig = "***" if u_test["p"] < 0.001 else "**" if u_test["p"] < 0.01 else "*" if u_test["p"] < 0.05 else "ns"
    ax.text(1.5, ax.get_ylim()[1] * 0.95,
            f"p={u_test['p']:.4f} {sig}\nr={u_test['r']:.3f}",
            ha="center", fontsize=10, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    ax.set_ylabel("CV (변동계수)", fontsize=12)
    ax.set_title("검색트렌드 변동성 비교 (CV)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_04_cv_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  08_04 저장 완료")


def plot_08_05_retention_bins(metrics_df: pd.DataFrame):
    """08_05: 잔존율 구간별 비율"""
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    labels = ["~10%", "10~20%", "20~30%", "30~50%", "50~70%", "70~100%"]

    sl_ret = metrics_df[metrics_df["is_sl"]]["retention"].dropna()
    non_sl_ret = metrics_df[~metrics_df["is_sl"]]["retention"].dropna()

    sl_hist = pd.cut(sl_ret, bins=bins, labels=labels).value_counts(normalize=True).reindex(labels, fill_value=0) * 100
    non_sl_hist = pd.cut(non_sl_ret, bins=bins, labels=labels).value_counts(normalize=True).reindex(labels, fill_value=0) * 100

    x = np.arange(len(labels))
    w = 0.35

    bars1 = ax.bar(x - w / 2, sl_hist.values, w, label=f"SL (n={len(sl_ret)})", color=SL_COLOR, alpha=0.85)
    bars2 = ax.bar(x + w / 2, non_sl_hist.values, w, label=f"Non-SL (n={len(non_sl_ret)})", color=NON_SL_COLOR, alpha=0.85)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 1:
                ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel("잔존율 구간", fontsize=12)
    ax.set_ylabel("비율 (%)", fontsize=12)
    ax.set_title("잔존율 구간별 SL/Non-SL 분포", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_05_retention_bins.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  08_05 저장 완료")


def plot_08_06_category_scatter(df: pd.DataFrame, metrics_df: pd.DataFrame):
    """08_06: 카테고리별 잔존율/CV 산점도"""
    # SLI에서 카테고리 정보 가져오기
    sli_path = _PROJECT_ROOT / "02_outputs" / "Sli" / "sli_integrated_results.csv"
    sli_df = pd.read_csv(sli_path)[["product_code", "category_2"]]
    merged = metrics_df.merge(sli_df, on="product_code", how="left")

    # 주요 카테고리만 (category_2 사용)
    top_cats = merged["category_2"].value_counts().head(7).index.tolist()
    merged = merged[merged["category_2"].isin(top_cats)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for metric, ax, title in [
        ("retention", axes[0], "잔존율"),
        ("cv", axes[1], "CV (변동계수)"),
    ]:
        cat_data = merged.groupby(["category_2", "is_sl"])[metric].median().unstack(fill_value=0)

        for cat in cat_data.index:
            sl_val = cat_data.loc[cat, True] if True in cat_data.columns else 0
            non_sl_val = cat_data.loc[cat, False] if False in cat_data.columns else 0
            ax.scatter(non_sl_val, sl_val, s=120, zorder=3)
            ax.annotate(cat, (non_sl_val, sl_val), fontsize=9,
                        xytext=(5, 5), textcoords="offset points")

        lim_max = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim_max], [0, lim_max], "k--", alpha=0.3, linewidth=1)
        ax.set_xlabel(f"Non-SL 중앙값 {title}", fontsize=11)
        ax.set_ylabel(f"SL 중앙값 {title}", fontsize=11)
        ax.set_title(f"카테고리별 {title} 비교", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_06_category_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  08_06 저장 완료")


def plot_08_07_three_metrics_dashboard(metrics_df: pd.DataFrame):
    """08_07: 3대 핵심 지표 대시보드"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sl_m = metrics_df[metrics_df["is_sl"]]
    non_sl_m = metrics_df[~metrics_df["is_sl"]]

    # (a) 변화율
    ax = axes[0]
    sl_vals = sl_m["change_rate"].dropna()
    non_sl_vals = non_sl_m["change_rate"].dropna()
    bp = ax.boxplot([sl_vals, non_sl_vals], labels=["SL", "Non-SL"],
                    patch_artist=True, widths=0.5, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black"))
    bp["boxes"][0].set_facecolor(SL_COLOR)
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor(NON_SL_COLOR)
    bp["boxes"][1].set_alpha(0.6)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("변화율 (후반/전반 - 1)")
    ax.set_title("(a) 변화율", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # (b) 잔존율
    ax = axes[1]
    sl_vals = sl_m["retention"].dropna()
    non_sl_vals = non_sl_m["retention"].dropna()
    bp = ax.boxplot([sl_vals, non_sl_vals], labels=["SL", "Non-SL"],
                    patch_artist=True, widths=0.5, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black"))
    bp["boxes"][0].set_facecolor(SL_COLOR)
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor(NON_SL_COLOR)
    bp["boxes"][1].set_alpha(0.6)
    ax.set_ylabel("잔존율")
    ax.set_title("(b) 잔존율", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # (c) CV
    ax = axes[2]
    sl_vals = sl_m["cv"].dropna()
    non_sl_vals = non_sl_m["cv"].dropna()
    bp = ax.boxplot([sl_vals, non_sl_vals], labels=["SL", "Non-SL"],
                    patch_artist=True, widths=0.5, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black"))
    bp["boxes"][0].set_facecolor(SL_COLOR)
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor(NON_SL_COLOR)
    bp["boxes"][1].set_alpha(0.6)
    ax.set_ylabel("CV (변동계수)")
    ax.set_title("(c) CV (변동성)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("SL vs Non-SL 3대 핵심 지표 대시보드", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_07_three_metrics_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  08_07 저장 완료")


def plot_08_08_example_products(df: pd.DataFrame, metrics_df: pd.DataFrame):
    """08_08: 대표 제품 사례 비교"""
    # SL 안정 대표, Non-SL 하락 대표 선정
    sl_stable = metrics_df[(metrics_df["is_sl"]) & (metrics_df["trend"] == "안정")].sort_values("cv")
    non_sl_decline = metrics_df[(~metrics_df["is_sl"]) & (metrics_df["trend"] == "하락")].sort_values("cv", ascending=False)

    if sl_stable.empty or non_sl_decline.empty:
        print("  08_08 대표 제품 부족 → 스킵")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # SL 안정 대표
    sl_ex = sl_stable.iloc[0]
    sl_data = df[df["product_code"] == sl_ex["product_code"]].sort_values("period")
    ax = axes[0]
    ax.plot(sl_data["period"], sl_data["normalized_ratio"], color=SL_COLOR, linewidth=2, marker="o", markersize=4)
    ax.set_title(f"SL \"안정\" 대표:\n{sl_ex['brand_name']} {sl_ex['product_name'][:25]}...",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("앵커 대비 ratio")
    ax.set_xlabel("기간")
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, f"잔존율={sl_ex['retention']:.2f}\nCV={sl_ex['cv']:.2f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # Non-SL 하락 대표
    non_sl_ex = non_sl_decline.iloc[0]
    non_sl_data = df[df["product_code"] == non_sl_ex["product_code"]].sort_values("period")
    ax = axes[1]
    ax.plot(non_sl_data["period"], non_sl_data["normalized_ratio"], color=NON_SL_COLOR, linewidth=2, marker="o", markersize=4)
    ax.set_title(f"Non-SL \"하락\" 대표:\n{non_sl_ex['brand_name']} {non_sl_ex['product_name'][:25]}...",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("앵커 대비 ratio")
    ax.set_xlabel("기간")
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, f"잔존율={non_sl_ex['retention']:.2f}\nCV={non_sl_ex['cv']:.2f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("대표 제품 검색트렌드 사례 비교", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_08_example_products.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  08_08 저장 완료")


def plot_08_09_anchor_absolute(df: pd.DataFrame):
    """08_09: 앵커 대비 절대 ratio 시계열 비교 (방법 A) — 신규"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # (a) 월별 평균 normalized_ratio 시계열
    ax = axes[0]
    for is_sl, color, label in [(True, SL_COLOR, "SL"), (False, NON_SL_COLOR, "Non-SL")]:
        sub = df[(df["is_sl"] == is_sl) & (df["product_code"] != ANCHOR_PRODUCT_CODE)]
        monthly = sub.groupby("period")["normalized_ratio"].agg(["mean", "median", "sem"])
        ax.plot(monthly.index, monthly["median"], color=color, linewidth=2.5,
                label=f"{label} 중앙값 (n={sub['product_code'].nunique()})")
        ax.fill_between(monthly.index,
                        monthly["median"] - monthly["sem"],
                        monthly["median"] + monthly["sem"],
                        alpha=0.15, color=color)

    # 앵커 기준선
    ax.axhline(1.0, color=ANCHOR_COLOR, linestyle="--", linewidth=1.5, alpha=0.7, label="앵커 (=1.0)")

    ax.set_xlabel("기간", fontsize=12)
    ax.set_ylabel("앵커 대비 ratio (딥클렌징폼=1.0)", fontsize=12)
    ax.set_title("(a) 앵커 대비 절대 검색지수 비교", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # (b) 제품별 평균 normalized_ratio 분포
    ax = axes[1]
    product_means = df[df["product_code"] != ANCHOR_PRODUCT_CODE].groupby(
        ["product_code", "is_sl"])["normalized_ratio"].mean().reset_index()

    sl_means = product_means[product_means["is_sl"]]["normalized_ratio"]
    non_sl_means = product_means[~product_means["is_sl"]]["normalized_ratio"]

    # 로그 스케일 히스토그램
    bins = np.logspace(-2, 2, 40)
    ax.hist(sl_means, bins=bins, alpha=0.6, color=SL_COLOR,
            label=f"SL (중앙값={sl_means.median():.2f})", density=True)
    ax.hist(non_sl_means, bins=bins, alpha=0.6, color=NON_SL_COLOR,
            label=f"Non-SL (중앙값={non_sl_means.median():.2f})", density=True)
    ax.axvline(1.0, color=ANCHOR_COLOR, linestyle="--", linewidth=1.5, alpha=0.7, label="앵커 (=1.0)")
    ax.set_xscale("log")
    ax.set_xlabel("앵커 대비 평균 ratio (로그 스케일)", fontsize=12)
    ax.set_ylabel("밀도", fontsize=12)
    ax.set_title("(b) 앵커 대비 평균 ratio 분포", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle("방법 A: 앵커(딥클렌징폼) 대비 절대 검색지수 비교", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_09_anchor_absolute_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  08_09 저장 완료")


# ── 메인 ──────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("앵커 기반 SL vs Non-SL 시계열 비교분석")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n[1] 데이터 로드")
    df = load_data()

    # 2. 유효 제품 필터링
    print("\n[2] 유효 제품 필터링")
    valid_df = filter_valid_products(df, min_periods=10)

    # 3. 제품별 지표 산출
    print("\n[3] 제품별 지표 산출")
    metrics_df = compute_product_metrics(valid_df)
    sl_m = metrics_df[metrics_df["is_sl"]]
    non_sl_m = metrics_df[~metrics_df["is_sl"]]
    print(f"지표 산출: SL {len(sl_m)}개, Non-SL {len(non_sl_m)}개")

    # 4. 통계 검정
    print("\n[4] 통계 검정 (Mann-Whitney U)")
    tests = [
        stat_test(sl_m["change_rate"], non_sl_m["change_rate"], "변화율"),
        stat_test(sl_m["retention"], non_sl_m["retention"], "잔존율"),
        stat_test(sl_m["cv"], non_sl_m["cv"], "CV"),
        stat_test(sl_m["avg_normalized_ratio"], non_sl_m["avg_normalized_ratio"], "앵커대비ratio"),
    ]
    for t in tests:
        sig = "***" if t["p"] < 0.001 else "**" if t["p"] < 0.01 else "*" if t["p"] < 0.05 else "ns"
        print(f"  {t['label']}: U={t['U']:.0f}, p={t['p']:.6f} {sig}, r={t['r']:.3f}")

    # 5. 종합 비교표 출력
    print("\n[5] 종합 비교")
    print(f"{'지표':<20} {'SL 중앙값':>12} {'Non-SL 중앙값':>14} {'배율':>8}")
    print("-" * 56)
    for label, sl_col, non_sl_col in [
        ("변화율", sl_m["change_rate"], non_sl_m["change_rate"]),
        ("잔존율", sl_m["retention"], non_sl_m["retention"]),
        ("CV", sl_m["cv"], non_sl_m["cv"]),
        ("앵커대비ratio", sl_m["avg_normalized_ratio"], non_sl_m["avg_normalized_ratio"]),
    ]:
        sl_med = sl_col.median()
        non_sl_med = non_sl_col.median()
        ratio = sl_med / non_sl_med if non_sl_med != 0 else float("inf")
        print(f"{label:<20} {sl_med:>12.3f} {non_sl_med:>14.3f} {ratio:>8.2f}x")

    # 트렌드 패턴 비율
    print("\n트렌드 패턴 분류:")
    for trend in ["상승", "안정", "하락"]:
        sl_pct = len(sl_m[sl_m["trend"] == trend]) / len(sl_m) * 100
        non_sl_pct = len(non_sl_m[non_sl_m["trend"] == trend]) / len(non_sl_m) * 100
        print(f"  {trend}: SL {sl_pct:.1f}% vs Non-SL {non_sl_pct:.1f}%")

    # 잔존율 구간별
    print("\n잔존율 구간별:")
    bins_labels = [(0, 0.1, "~10%"), (0.1, 0.3, "10~30%"), (0.3, 0.5, "30~50%"), (0.5, 0.7, "50~70%"), (0.7, 1.0, "70~100%")]
    for lo, hi, label in bins_labels:
        sl_pct = len(sl_m[(sl_m["retention"] >= lo) & (sl_m["retention"] < hi)]) / len(sl_m) * 100
        non_sl_pct = len(non_sl_m[(non_sl_m["retention"] >= lo) & (non_sl_m["retention"] < hi)]) / len(non_sl_m) * 100
        print(f"  {label}: SL {sl_pct:.1f}% vs Non-SL {non_sl_pct:.1f}%")

    # 카테고리별 비교
    sli_path = _PROJECT_ROOT / "02_outputs" / "Sli" / "sli_integrated_results.csv"
    sli_df = pd.read_csv(sli_path)[["product_code", "category_2"]]
    merged = metrics_df.merge(sli_df, on="product_code", how="left")

    print("\n카테고리별 비교 (잔존율 중앙값):")
    top_cats = merged["category_2"].value_counts().head(7).index
    for cat in top_cats:
        cat_sl = merged[(merged["category_2"] == cat) & (merged["is_sl"])]
        cat_non_sl = merged[(merged["category_2"] == cat) & (~merged["is_sl"])]
        if len(cat_sl) >= 3 and len(cat_non_sl) >= 3:
            print(f"  {cat}: SL {cat_sl['retention'].median():.3f} ({len(cat_sl)}개) vs Non-SL {cat_non_sl['retention'].median():.3f} ({len(cat_non_sl)}개)")

    # 6. 시각화
    print("\n[6] 시각화 생성")
    plot_08_01_timeseries_normalized(valid_df, metrics_df)
    plot_08_02_trend_pattern(metrics_df)
    plot_08_03_retention_distribution(metrics_df)
    plot_08_04_cv_boxplot(metrics_df)
    plot_08_05_retention_bins(metrics_df)
    plot_08_06_category_scatter(valid_df, metrics_df)
    plot_08_07_three_metrics_dashboard(metrics_df)
    plot_08_08_example_products(valid_df, metrics_df)
    plot_08_09_anchor_absolute(valid_df)

    # 7. 분석 결과 CSV 저장
    date_str = datetime.now().strftime("%Y%m%d")
    metrics_path = OUTPUT_DIR / f"search_trend_anchor_metrics_{date_str}.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"\n지표 CSV 저장: {metrics_path.name}")

    # 정규화 시계열 테이블 (문서용)
    print("\n[7] 정규화 시계열 테이블 (문서용)")
    for is_sl, label in [(True, "SL"), (False, "Non-SL")]:
        sub = valid_df[(valid_df["is_sl"] == is_sl) & (valid_df["product_code"] != ANCHOR_PRODUCT_CODE)]
        # 제품별 max 대비 정규화
        shape_rows = []
        for pcode, grp in sub.groupby("product_code"):
            grp = grp.sort_values("period")
            max_r = grp["normalized_ratio"].max()
            if max_r > 0:
                grp = grp.copy()
                grp["shape_norm"] = grp["normalized_ratio"] / max_r
                shape_rows.append(grp[["period", "shape_norm"]])
        if shape_rows:
            all_shape = pd.concat(shape_rows)
            monthly = all_shape.groupby("period")["shape_norm"].mean()
            print(f"\n{label} 정규화 시계열 (제품별 max=1):")
            for period, val in monthly.items():
                print(f"  {period.strftime('%Y-%m')}: {val:.3f}")

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
