"""
Stage 4 데이터 병합 스크립트

기존 학습 데이터(train_aug 11,992건 + val 2,570건)에
보충 데이터(final_training_supplement_v5.csv, 2,070건)를 병합.

처리 로직:
  1. golden_test(266건) review_id를 보충에서 제거 (평가 벤치마크 보존)
  2. aug 750건 비타겟 aspect mask=0 처리 (타겟만 mask=1 유지)
  3. 기존 train에서 보충과 겹치는 건 제거 (보충 라벨 우선)
  4. val에서 보충과 겹치는 건 제거
  5. 병합 → stage4/ 디렉토리에 저장
  6. 분포 리포트 출력

출력:
  01_outputs/data/training/stage4/
  ├── absa_wide_train_stage4.csv
  ├── absa_wide_val_stage4.csv
  └── merge_report.txt
"""
import sys
import re
from pathlib import Path
from datetime import datetime

ABSA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ABSA_ROOT))

import numpy as np
import pandas as pd

from RQ_absa.s1_config import ASPECT_LABELS, PROCESSED_DATA_DIR


# ── 경로 설정 ──
DATA_DIR = PROCESSED_DATA_DIR / "training"
GOLDEN_DIR = PROCESSED_DATA_DIR / "golden"
STAGE4_DIR = DATA_DIR / "stage4"

TRAIN_AUG_PATH = DATA_DIR / "absa_wide_train_augmented.csv"
VAL_PATH = DATA_DIR / "absa_wide_val.csv"
SUPPLEMENT_PATH = GOLDEN_DIR / "v2" / "final_training_supplement_v5.csv"
GOLDEN_TEST_PATH = GOLDEN_DIR / "v1" / "golden_test.csv"

# aspect 컬럼명 (/ → _)
ASPECT_COLS = [a.replace("/", "_") for a in ASPECT_LABELS]
LABEL_COLS = [f"label_{ac}" for ac in ASPECT_COLS]
MASK_COLS = [f"mask_{ac}" for ac in ASPECT_COLS]
AMBIG_COLS = [f"ambig_{ac}" for ac in ASPECT_COLS]

# aug source에서 타겟 aspect 추출을 위한 매핑
SOURCE_ASPECT_MAP = {
    "배송_포장": "배송_포장",
    "가격_가성비": "가격_가성비",
    "사용감_성능": "사용감_성능",
    "용량_휴대": "용량_휴대",
    "디자인": "디자인",
    "재질_냄새": "재질_냄새",
    "재구매": "재구매",
    "색상_발색": "색상_발색",
}


def extract_target_aspect(source: str) -> str | None:
    """aug source명에서 타겟 aspect 추출.

    예: 'aug_가격_가성비_neu' → '가격_가성비'
         'aug_재구매_neg' → '재구매'
    """
    if not isinstance(source, str) or not source.startswith("aug_"):
        return None

    # 'aug_' 제거 후 뒤의 _neu/_neg/_pos 제거
    body = source[4:]  # 'aug_' 제거
    body = re.sub(r'_(neu|neg|pos)$', '', body)

    if body in SOURCE_ASPECT_MAP:
        return SOURCE_ASPECT_MAP[body]
    return None


def fix_aug_masks(supp: pd.DataFrame) -> pd.DataFrame:
    """aug 750건 비타겟 aspect mask=0 처리.

    타겟 aspect만 mask=1 유지, 나머지 7개 → mask=0.
    """
    aug_mask = supp["source"].str.startswith("aug_", na=False)
    aug_rows = supp[aug_mask].copy()
    non_aug_rows = supp[~aug_mask].copy()

    if len(aug_rows) == 0:
        return supp

    # 처리 전 mask=1 셀 수
    before_mask1 = aug_rows[MASK_COLS].sum().sum()

    fixed_count = 0
    for idx in aug_rows.index:
        source = aug_rows.loc[idx, "source"]
        target_aspect = extract_target_aspect(source)
        if target_aspect is None:
            print(f"  경고: aug source '{source}'에서 타겟 aspect를 추출할 수 없음 (스킵)")
            continue

        target_mask_col = f"mask_{target_aspect}"

        # 모든 mask=0으로 설정
        for mc in MASK_COLS:
            aug_rows.loc[idx, mc] = 0
        # 타겟 aspect만 mask=1 복원
        aug_rows.loc[idx, target_mask_col] = 1
        fixed_count += 1

    # 처리 후 mask=1 셀 수
    after_mask1 = aug_rows[MASK_COLS].sum().sum()

    print(f"\n[aug mask 처리] {fixed_count}/{len(aug_rows)}건 처리")
    print(f"  mask=1 셀: {int(before_mask1)} → {int(after_mask1)} (감소: {int(before_mask1 - after_mask1)})")
    print(f"  행당 mask=1 평균: {before_mask1/len(aug_rows):.1f} → {after_mask1/len(aug_rows):.1f}")

    # 재결합
    result = pd.concat([non_aug_rows, aug_rows], ignore_index=True)
    return result


def normalize_columns(df: pd.DataFrame, has_extra_cols: bool = False) -> pd.DataFrame:
    """_load_wide_csv()가 요구하는 컬럼으로 정규화.

    필수: text, review_sentiment, review_id, rating, product_code, category_1, category_2
    aspect: label_*, mask_*, ambig_* (8개 × 3 = 24컬럼)
    """
    required_meta = ["text", "review_sentiment", "review_id", "rating",
                     "product_code", "category_1", "category_2"]

    # 기존 train/val에 있는 추가 메타 컬럼
    optional_meta = ["user_id", "review_date", "is_reorder", "name",
                     "price", "brand_id", "review_sentiment_score"]

    # aspect 컬럼
    aspect_all = LABEL_COLS + MASK_COLS + AMBIG_COLS

    # 보충 데이터에만 있는 컬럼 (저장은 하되 학습에 미사용)
    extra_cols = ["source", "sentiment_id", "_qc_severity",
                  "_qc_issue_type", "_qc_recommendation"]

    # 사용할 컬럼 결정
    cols = []
    for c in required_meta:
        if c in df.columns:
            cols.append(c)
    for c in optional_meta:
        if c in df.columns:
            cols.append(c)
    for c in aspect_all:
        if c in df.columns:
            cols.append(c)
    if has_extra_cols:
        for c in extra_cols:
            if c in df.columns:
                cols.append(c)

    # 없는 ambig 컬럼은 0으로 채움
    for c in AMBIG_COLS:
        if c not in df.columns:
            df[c] = 0
            if c not in cols:
                cols.append(c)

    return df[cols]


def compute_aspect_distribution(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """aspect별 none/pos/neu/neg 분포 계산."""
    rows = []
    for ac, label_col, mask_col in zip(ASPECT_COLS, LABEL_COLS, MASK_COLS):
        if label_col not in df.columns:
            continue
        labels = df[label_col].values
        masks = df[mask_col].values if mask_col in df.columns else np.ones(len(df))

        total = int(masks.sum())
        active = labels[masks.astype(bool)]
        n_none = int((active == 0).sum())
        n_pos = int((active == 1).sum())
        n_neu = int((active == 2).sum())
        n_neg = int((active == 3).sum())

        rows.append({
            "aspect": ac,
            "mask=1": total,
            "none": n_none,
            "pos": n_pos,
            "neu": n_neu,
            "neg": n_neg,
            "mentioned": n_pos + n_neu + n_neg,
            "mention_%": round((n_pos + n_neu + n_neg) / total * 100, 1) if total > 0 else 0,
        })
    result = pd.DataFrame(rows)
    result.index.name = name
    return result


def main():
    print("=" * 78)
    print("  ABSA Stage 4 — 데이터 병합")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)

    # ── 1. 데이터 로드 ──
    print("\n[1] 데이터 로드")
    train_aug = pd.read_csv(TRAIN_AUG_PATH)
    val = pd.read_csv(VAL_PATH)
    supp = pd.read_csv(SUPPLEMENT_PATH)
    golden_test = pd.read_csv(GOLDEN_TEST_PATH)

    print(f"  train_aug:   {len(train_aug):,}건")
    print(f"  val:         {len(val):,}건")
    print(f"  supplement:  {len(supp):,}건")
    print(f"  golden_test: {len(golden_test):,}건")

    # ── 2. golden_test review_id를 보충에서 제거 ──
    print("\n[2] golden_test 겹침 제거")
    golden_test_ids = set(golden_test["review_id"].unique())
    overlap_golden = supp["review_id"].isin(golden_test_ids)
    n_golden_overlap = overlap_golden.sum()
    print(f"  보충 ∩ golden_test: {n_golden_overlap}건 → 보충에서 제거")
    supp = supp[~overlap_golden].copy()
    print(f"  보충 데이터 잔여: {len(supp):,}건")

    # ── 3. aug 비타겟 mask=0 처리 ──
    print("\n[3] aug 비타겟 aspect mask=0 처리")
    supp = fix_aug_masks(supp)

    # ── 4. train/val 겹침 제거 ──
    print("\n[4] train/val 겹침 제거 (보충 라벨 우선)")
    supp_ids = set(supp["review_id"].unique())

    train_overlap = train_aug["review_id"].isin(supp_ids)
    n_train_overlap = train_overlap.sum()
    print(f"  train_aug ∩ supplement: {n_train_overlap}건 → train에서 제거")
    train_clean = train_aug[~train_overlap].copy()

    val_overlap = val["review_id"].isin(supp_ids)
    n_val_overlap = val_overlap.sum()
    print(f"  val ∩ supplement: {n_val_overlap}건 → val에서 제거")
    val_clean = val[~val_overlap].copy()

    print(f"  train 정제: {len(train_aug):,} → {len(train_clean):,}")
    print(f"  val 정제:   {len(val):,} → {len(val_clean):,}")

    # ── 5. 컬럼 정규화 + 병합 ──
    print("\n[5] 컬럼 정규화 + 병합")

    # train 컬럼에 맞춰 보충 정규화
    # train에는 source 등 추가 컬럼이 없으므로, 공통 컬럼만 사용
    # 보충에 없는 train 전용 컬럼은 NaN으로
    train_cols = set(train_clean.columns)
    supp_cols = set(supp.columns)
    print(f"  train 전용 컬럼: {sorted(train_cols - supp_cols)}")
    print(f"  supplement 전용 컬럼: {sorted(supp_cols - train_cols)}")

    # 공통 컬럼 기준 병합 (train이 가진 컬럼 기준)
    target_cols = list(train_clean.columns)

    # 보충 데이터에서 train 컬럼만 추출 (없으면 NaN)
    supp_aligned = pd.DataFrame()
    for col in target_cols:
        if col in supp.columns:
            supp_aligned[col] = supp[col].values
        else:
            supp_aligned[col] = np.nan

    # 병합
    train_merged = pd.concat([train_clean, supp_aligned], ignore_index=True)
    print(f"  병합 완료: {len(train_clean):,} + {len(supp_aligned):,} = {len(train_merged):,}건")

    # ── 6. 무결성 검증 ──
    print("\n[6] 무결성 검증")

    # golden_test와 겹침 0건 확인
    merged_ids = set(train_merged["review_id"].unique())
    val_clean_ids = set(val_clean["review_id"].unique())

    golden_in_train = merged_ids & golden_test_ids
    golden_in_val = val_clean_ids & golden_test_ids
    print(f"  golden_test ∩ train: {len(golden_in_train)}건 {'✅' if len(golden_in_train) == 0 else '❌'}")
    print(f"  golden_test ∩ val:   {len(golden_in_val)}건 {'✅' if len(golden_in_val) == 0 else '❌'}")

    # train-val 겹침 확인
    train_val_overlap = merged_ids & val_clean_ids
    print(f"  train ∩ val:         {len(train_val_overlap)}건 {'✅' if len(train_val_overlap) == 0 else '⚠️'}")

    # aug mask 검증: aug 건의 mask=1 셀이 정확히 1개인지
    if "source" in supp.columns:
        # 원본 보충에서 aug 확인 (golden 제거 후)
        supp_aug = supp[supp["source"].str.startswith("aug_", na=False)]
        if len(supp_aug) > 0:
            aug_mask_sums = supp_aug[MASK_COLS].sum(axis=1)
            all_one = (aug_mask_sums == 1).all()
            print(f"  aug 행당 mask=1: 모두 1개 {'✅' if all_one else '❌'}")
            if not all_one:
                print(f"    mask=1 분포: {aug_mask_sums.value_counts().to_dict()}")

    # ── 7. 저장 ──
    print("\n[7] 저장")
    STAGE4_DIR.mkdir(parents=True, exist_ok=True)

    train_out = STAGE4_DIR / "absa_wide_train_stage4.csv"
    val_out = STAGE4_DIR / "absa_wide_val_stage4.csv"

    train_merged.to_csv(train_out, index=False)
    val_clean.to_csv(val_out, index=False)
    print(f"  Train: {train_out} ({len(train_merged):,}건)")
    print(f"  Val:   {val_out} ({len(val_clean):,}건)")

    # ── 8. 분포 리포트 ──
    print("\n[8] 분포 리포트")

    report_lines = []

    def p(s=""):
        report_lines.append(s)
        print(s)

    p("=" * 78)
    p("  Stage 4 데이터 병합 리포트")
    p(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p("=" * 78)

    p(f"\n■ 데이터 규모")
    p(f"  기존 train_aug:     {len(train_aug):,}건")
    p(f"  기존 val:           {len(val):,}건")
    p(f"  보충 원본:          {len(pd.read_csv(SUPPLEMENT_PATH)):,}건")
    p(f"  golden_test 제거:   -{n_golden_overlap}건")
    p(f"  train 겹침 제거:    -{n_train_overlap}건 (train에서)")
    p(f"  val 겹침 제거:      -{n_val_overlap}건 (val에서)")
    p(f"  ───")
    p(f"  최종 train (Stage4): {len(train_merged):,}건")
    p(f"  최종 val (Stage4):   {len(val_clean):,}건")
    p(f"  golden_test:         {len(golden_test):,}건 (불변)")

    # 보충 소스별 건수
    p(f"\n■ 보충 데이터 소스별 (golden_test 제거 후)")
    for src, cnt in supp["source"].value_counts().items():
        p(f"  {src}: {cnt}건")

    # 기존 vs Stage4 neutral 비교
    p(f"\n■ Neutral 분포 비교 (train)")
    for ac, lc in zip(ASPECT_COLS, LABEL_COLS):
        if lc not in train_aug.columns or lc not in train_merged.columns:
            continue
        old_neu = int((train_aug[lc] == 2).sum())
        new_neu = int((train_merged[lc] == 2).sum())
        diff = new_neu - old_neu
        p(f"  {ac:12s}: {old_neu:5d} → {new_neu:5d} (+{diff})")

    # Stage4 train aspect 분포
    p(f"\n■ Stage4 Train aspect 분포")
    dist = compute_aspect_distribution(train_merged, "Stage4 Train")
    p(dist.to_string(index=False))

    # Stage4 val aspect 분포
    p(f"\n■ Stage4 Val aspect 분포")
    dist_val = compute_aspect_distribution(val_clean, "Stage4 Val")
    p(dist_val.to_string(index=False))

    p("\n" + "=" * 78)

    # 리포트 저장
    report_path = STAGE4_DIR / "merge_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n리포트 저장: {report_path}")

    print("\n병합 완료!")


if __name__ == "__main__":
    main()
