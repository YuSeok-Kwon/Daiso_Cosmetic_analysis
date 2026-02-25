"""
골든셋 Dev/Test 분할 스크립트

입력: golden_set_wide_full_eval.csv
출력: golden/golden_dev.csv, golden/golden_test.csv

층화 기준: review_sentiment × 디자인 언급 여부 (복합 층화)
  - 희소 aspect(디자인 등)가 test에 몰리거나 빠지는 것을 방지
  - 복합 그룹이 너무 작으면 review_sentiment만으로 fallback
원본 파일 보존
"""
import sys
from pathlib import Path

ABSA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ABSA_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split

from RQ_absa.s1_config import GOLDEN_SPLIT_CONFIG, PROCESSED_DATA_DIR


def main():
    cfg = GOLDEN_SPLIT_CONFIG

    # ── 입력 ──
    golden_dir = PROCESSED_DATA_DIR / "final" / "golden"
    input_path = golden_dir / "golden_set_wide_full_eval.csv"
    assert input_path.exists(), f"골든셋 파일 없음: {input_path}"

    df = pd.read_csv(input_path)
    print(f"골든셋 로드: {len(df):,}행 from {input_path}")
    print(f"  review_sentiment 분포:\n{df['review_sentiment'].value_counts().to_string()}\n")

    # ── 복합 층화 키 생성 ──
    # review_sentiment × 디자인 언급 여부 (label_디자인 > 0)
    if "label_디자인" in df.columns:
        df["_design_mentioned"] = (df["label_디자인"] > 0).astype(int)
        df["_stratify_key"] = (
            df["review_sentiment"].astype(str) + "_d" + df["_design_mentioned"].astype(str)
        )
        min_group = df["_stratify_key"].value_counts().min()
        if min_group >= 2:
            stratify_col = "_stratify_key"
            print(f"복합 층화: review_sentiment × 디자인 (최소 그룹: {min_group})")
        else:
            stratify_col = "review_sentiment"
            print(f"WARNING: 복합 그룹 최소 {min_group}행 → review_sentiment만 사용")
    else:
        stratify_col = "review_sentiment"
        print("label_디자인 컬럼 없음 → review_sentiment만 사용")

    # ── 층화 분할 ──
    dev_df, test_df = train_test_split(
        df,
        test_size=cfg["test_ratio"],
        stratify=df[stratify_col],
        random_state=cfg["random_state"],
    )

    # 임시 컬럼 제거
    for col in ["_design_mentioned", "_stratify_key"]:
        if col in dev_df.columns:
            dev_df = dev_df.drop(columns=[col])
            test_df = test_df.drop(columns=[col])

    print(f"\n분할 결과 (dev {cfg['dev_ratio']*100:.0f}% / test {cfg['test_ratio']*100:.0f}%):")
    print(f"  Dev:  {len(dev_df):,}행 ({len(dev_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df):,}행 ({len(test_df)/len(df)*100:.1f}%)")

    # ── 저장 ──
    dev_path = golden_dir / "golden_dev.csv"
    test_path = golden_dir / "golden_test.csv"

    dev_df.to_csv(dev_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    print(f"\n저장 완료:")
    print(f"  Dev:  {dev_path}")
    print(f"  Test: {test_path}")

    # ── 검증: sentiment + 디자인 분포 ──
    print(f"\nDev sentiment 분포:\n{dev_df['review_sentiment'].value_counts().to_string()}")
    print(f"\nTest sentiment 분포:\n{test_df['review_sentiment'].value_counts().to_string()}")

    if "label_디자인" in dev_df.columns:
        dev_design = (dev_df["label_디자인"] > 0).sum()
        test_design = (test_df["label_디자인"] > 0).sum()
        print(f"\n디자인 언급: Dev {dev_design}건, Test {test_design}건")


if __name__ == "__main__":
    main()
