"""
골든셋 Dev/Test 분할 스크립트

입력: golden_set_wide_full_eval.csv
출력: golden/golden_dev.csv, golden/golden_test.csv

층화 기준: review_sentiment
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

    # ── 층화 분할 ──
    dev_df, test_df = train_test_split(
        df,
        test_size=cfg["test_ratio"],
        stratify=df["review_sentiment"],
        random_state=cfg["random_state"],
    )

    print(f"분할 결과:")
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

    # ── 검증 ──
    print(f"\nDev sentiment 분포:\n{dev_df['review_sentiment'].value_counts().to_string()}")
    print(f"\nTest sentiment 분포:\n{test_df['review_sentiment'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
