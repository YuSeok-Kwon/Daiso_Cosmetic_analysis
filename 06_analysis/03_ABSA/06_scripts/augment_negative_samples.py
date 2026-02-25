"""
약점 Aspect Negative Sampling 보강 스크립트

문제: 디자인 등 mask=0(미라벨)인 샘플이 많아 모델이 학습 부족
해결: mask=0인 리뷰 중 해당 aspect 키워드가 미포함된 것을 "confirmed none"으로 전환

핵심 원칙:
  (A) train split에만 적용 — val/test는 자연 분포 유지 (일반화 검증)
  (B) 키워드 0개인 텍스트만 변환 — 1개라도 있으면 mask=0 유지
  (C) per-review 변환 cap — 한 리뷰에서 최대 MAX_CONVERSIONS_PER_REVIEW개 aspect만

입력: absa_wide_train.csv (전체 ~17k)
출력:
  - absa_wide_train_augmented.csv (train split + negative sampling)
  - absa_wide_val.csv (val split, 원본 그대로)
  - absa_wide_test.csv (test split, 원본 그대로)
"""
import sys
from pathlib import Path

ABSA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ABSA_ROOT))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from RQ_absa.s1_config import PROCESSED_DATA_DIR, SPLIT_RATIOS

# ── 보강 대상 aspect별 키워드 ──
# 키워드가 포함된 리뷰는 "확실한 none"이 아니므로 변환 제외
ASPECT_KEYWORDS = {
    "디자인": [
        "디자인", "예쁘", "이쁘", "귀엽", "깔끔", "패키지", "외관",
        "모양", "생김", "패킹", "포장", "포잠", "고급스러", "심플",
        "용기", "뚜껑", "케이스", "스포이드", "펌프", "튜브",
        "앙증", "팁", "입구", "마감",
    ],
}

MAX_SAMPLES_PER_ASPECT = 2500       # aspect당 보강 상한
MAX_CONVERSIONS_PER_REVIEW = 1      # 한 리뷰에서 최대 변환 aspect 수


def has_keyword(text: str, keywords: list) -> bool:
    """텍스트에 키워드 중 하나라도 포함되면 True"""
    text_lower = str(text).lower()
    return any(kw.lower() in text_lower for kw in keywords)


def drop_removed_aspect_columns(df: pd.DataFrame) -> tuple:
    """미분류·CS/응대·품질/퀄리티 컬럼 제거 (Stage 2에서 모델 학습 제외)"""
    drop_cols = [
        c for c in df.columns
        if "미분류" in c or "CS_응대" in c or "품질_퀄리티" in c
    ]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df, drop_cols


def main():
    # ── 입력 ──
    data_dir = PROCESSED_DATA_DIR / "final" / "absa"
    input_path = data_dir / "absa_wide_train.csv"
    assert input_path.exists(), f"학습 데이터 없음: {input_path}"

    df = pd.read_csv(input_path)
    print(f"원본 데이터 로드: {len(df):,}행")

    # ── Step 1: 미분류/CS/품질 컬럼 먼저 제거 ──
    df, dropped = drop_removed_aspect_columns(df)
    if dropped:
        print(f"제거된 컬럼 ({len(dropped)}개): {dropped[:6]}...")

    # ── Step 2: Train/Val/Test 분할 (sentiment 기준 층화) ──
    train_ratio = SPLIT_RATIOS["train"]
    val_ratio = SPLIT_RATIOS["val"]
    test_ratio = SPLIT_RATIOS["test"]

    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_ratio),
        stratify=df["review_sentiment"],
        random_state=42,
    )
    val_adj = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_adj),
        stratify=temp_df["review_sentiment"],
        random_state=42,
    )

    print(f"\n분할 결과:")
    print(f"  Train: {len(train_df):,}행 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df):,}행 ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df):,}행 ({len(test_df)/len(df)*100:.1f}%)")

    # ── Step 3: Train에만 Negative sampling 적용 ──
    print("\n" + "=" * 60)
    print("NEGATIVE SAMPLING (train split only)")
    print("=" * 60)

    augmented = train_df.copy()
    total_augmented = 0
    # per-review 변환 횟수 추적
    review_conversion_count = pd.Series(0, index=augmented.index)

    for aspect_col, keywords in ASPECT_KEYWORDS.items():
        mask_col = f"mask_{aspect_col}"
        label_col = f"label_{aspect_col}"

        if mask_col not in augmented.columns:
            print(f"  [{aspect_col}] 컬럼 없음: {mask_col}, 스킵")
            continue

        # 보강 전 통계
        mask0 = (augmented[mask_col] == 0).sum()
        mask1 = (augmented[mask_col] == 1).sum()
        print(f"\n[{aspect_col}] 보강 전: mask=0 {mask0:,} / mask=1 {mask1:,}")

        # mask=0이고 per-review cap 미도달인 행만 대상
        candidates_mask = (
            (augmented[mask_col] == 0) &
            (review_conversion_count < MAX_CONVERSIONS_PER_REVIEW)
        )
        candidates = augmented[candidates_mask].copy()
        print(f"  mask=0 후보 (cap 미도달): {len(candidates):,}행")

        # 키워드 미포함 필터 (키워드 1개라도 있으면 변환 안 함)
        no_keyword_mask = ~candidates["text"].apply(lambda t: has_keyword(t, keywords))
        candidates = candidates[no_keyword_mask]
        print(f"  키워드 0개 (확실한 none): {len(candidates):,}행")

        # 짧은 리뷰 우선 (해당 aspect 언급 가능성 낮음)
        candidates["_text_len"] = candidates["text"].astype(str).str.len()
        candidates = candidates.sort_values("_text_len")

        # 상한 적용
        n_sample = min(len(candidates), MAX_SAMPLES_PER_ASPECT)
        selected_indices = candidates.head(n_sample).index

        # mask=0 → mask=1, label=0 (confirmed none)
        augmented.loc[selected_indices, mask_col] = 1
        augmented.loc[selected_indices, label_col] = 0
        review_conversion_count.loc[selected_indices] += 1

        # 보강 후 통계
        new_mask1 = (augmented[mask_col] == 1).sum()
        new_label0_mask1 = (
            (augmented[mask_col] == 1) & (augmented[label_col] == 0)
        ).sum()
        print(f"  보강: {n_sample:,}건 (mask=0 → mask=1, label=0)")
        print(f"  보강 후: mask=1 {new_mask1:,} (label=0: {new_label0_mask1:,})")
        total_augmented += n_sample

    # 임시 컬럼 제거
    if "_text_len" in augmented.columns:
        augmented = augmented.drop(columns=["_text_len"])

    # ── Step 4: 저장 (3개 파일) ──
    train_path = data_dir / "absa_wide_train_augmented.csv"
    val_path = data_dir / "absa_wide_val.csv"
    test_path = data_dir / "absa_wide_test.csv"

    augmented.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    print(f"\n" + "=" * 60)
    print(f"저장 완료:")
    print(f"  Train (augmented): {train_path} ({len(augmented):,}행, 보강 셀: {total_augmented:,})")
    print(f"  Val   (원본):      {val_path} ({len(val_df):,}행)")
    print(f"  Test  (원본):      {test_path} ({len(test_df):,}행)")
    print(f"  per-review 변환 cap: {MAX_CONVERSIONS_PER_REVIEW}")
    print("=" * 60)


if __name__ == "__main__":
    main()
