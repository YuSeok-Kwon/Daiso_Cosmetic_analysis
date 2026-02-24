"""
약점 Aspect Negative Sampling 보강 스크립트

문제: 디자인/품질/CS 등 mask=0(미라벨)인 샘플이 많아 모델이 학습 부족
해결: mask=0인 리뷰 중 해당 aspect 키워드가 미포함된 것을 "confirmed none"으로 전환

입력: absa_wide_train.csv
출력: absa_wide_train_augmented.csv (원본 보존)
"""
import sys
from pathlib import Path

ABSA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ABSA_ROOT))

import pandas as pd
import numpy as np

from RQ_absa.s1_config import PROCESSED_DATA_DIR

# 보강 대상 aspect별 키워드 (이 키워드가 포함된 리뷰는 "확실한 none"이 아니므로 제외)
ASPECT_KEYWORDS = {
    "디자인": [
        "디자인", "예쁘", "이쁘", "귀엽", "깔끔", "패키지", "외관",
        "모양", "생김", "패킹", "포장", "포잠", "고급스러", "심플",
    ],
}

# aspect당 보강 상한
MAX_SAMPLES_PER_ASPECT = 2500


def has_keyword(text: str, keywords: list) -> bool:
    """텍스트에 키워드 중 하나라도 포함되면 True"""
    text_lower = str(text).lower()
    return any(kw.lower() in text_lower for kw in keywords)


def main():
    # ── 입력 ──
    data_dir = PROCESSED_DATA_DIR / "final" / "absa"
    input_path = data_dir / "absa_wide_train.csv"
    assert input_path.exists(), f"학습 데이터 없음: {input_path}"

    df = pd.read_csv(input_path)
    print(f"원본 데이터 로드: {len(df):,}행")

    # ── 보강 전 통계 ──
    print("\n보강 전 mask 통계:")
    for aspect_col, keywords in ASPECT_KEYWORDS.items():
        mask_col = f"mask_{aspect_col}"
        label_col = f"label_{aspect_col}"
        mask0 = (df[mask_col] == 0).sum()
        mask1 = (df[mask_col] == 1).sum()
        print(f"  {aspect_col}: mask=0 {mask0:,} / mask=1 {mask1:,}")

    # ── Negative sampling ──
    augmented = df.copy()
    total_augmented = 0

    for aspect_col, keywords in ASPECT_KEYWORDS.items():
        mask_col = f"mask_{aspect_col}"
        label_col = f"label_{aspect_col}"

        # mask=0인 행만 대상
        candidates = augmented[augmented[mask_col] == 0].copy()
        print(f"\n[{aspect_col}] mask=0 후보: {len(candidates):,}행")

        # 키워드 미포함 필터
        no_keyword_mask = ~candidates["text"].apply(lambda t: has_keyword(t, keywords))
        candidates = candidates[no_keyword_mask]
        print(f"  키워드 미포함: {len(candidates):,}행")

        # 짧은 리뷰 우선 (해당 aspect 언급 가능성 낮음)
        candidates["_text_len"] = candidates["text"].astype(str).str.len()
        candidates = candidates.sort_values("_text_len")

        # 상한 적용
        n_sample = min(len(candidates), MAX_SAMPLES_PER_ASPECT)
        selected_indices = candidates.head(n_sample).index

        # mask=0 → mask=1, label=0 (confirmed none)
        augmented.loc[selected_indices, mask_col] = 1
        augmented.loc[selected_indices, label_col] = 0

        print(f"  보강: {n_sample:,}행 (mask=0 → mask=1, label=0)")
        total_augmented += n_sample

    # _text_len 임시 컬럼 제거
    if "_text_len" in augmented.columns:
        augmented = augmented.drop(columns=["_text_len"])

    # ── 미분류·CS/응대·품질/퀄리티 컬럼 제거 (Stage 2에서 모델 학습 제외) ──
    drop_cols = [c for c in augmented.columns if "미분류" in c or "CS_응대" in c or "품질_퀄리티" in c]
    if drop_cols:
        augmented = augmented.drop(columns=drop_cols)
        print(f"\n제거된 컬럼: {drop_cols}")

    # ── 저장 ──
    output_path = data_dir / "absa_wide_train_augmented.csv"
    augmented.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n보강 완료:")
    print(f"  원본: {len(df):,}행")
    print(f"  보강 후: {len(augmented):,}행 (행 수 동일, mask 변경)")
    print(f"  총 보강 셀: {total_augmented:,}개")
    print(f"  저장: {output_path}")

    # ── 보강 후 통계 ──
    print("\n보강 후 mask 통계:")
    for aspect_col in ASPECT_KEYWORDS:
        mask_col = f"mask_{aspect_col}"
        label_col = f"label_{aspect_col}"
        mask1 = (augmented[mask_col] == 1).sum()
        label0_mask1 = ((augmented[mask_col] == 1) & (augmented[label_col] == 0)).sum()
        print(f"  {aspect_col}: mask=1 {mask1:,} (그 중 label=0: {label0_mask1:,})")


if __name__ == "__main__":
    main()
