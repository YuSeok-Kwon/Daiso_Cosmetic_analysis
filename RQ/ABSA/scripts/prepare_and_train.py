"""
4단계: Gold set 통합 → 전처리 → 분할 → 모델 학습 준비
"""

import pandas as pd
from pathlib import Path
import re
from sklearn.model_selection import train_test_split
import argparse


def remove_repurchase_tag(text):
    """재구매 태그 제거"""
    if pd.isna(text):
        return text

    patterns = [
        r'^재구매\s*[\|｜:：]?\s*',   # "재구매 | ", "재구매: " 등
        r'^\[재구매\]\s*',            # "[재구매] "
        r'^【재구매】\s*',            # "【재구매】"
        r'^재구매\s+',                # "재구매 " (공백만)
    ]

    result = str(text)
    for pattern in patterns:
        result = re.sub(pattern, '', result)

    return result.strip()


def load_and_merge_gold_sets(data_dir: Path) -> pd.DataFrame:
    """6개 팀의 gold set 통합"""
    gold_sets = []

    for i in range(1, 7):
        path = data_dir / f"step3_team{i}_gold_set.csv"
        if path.exists():
            df = pd.read_csv(path)
            df['team'] = i
            gold_sets.append(df)
            print(f"  팀{i}: {len(df)}건 로드")
        else:
            print(f"  팀{i}: 파일 없음 ({path})")

    if not gold_sets:
        raise FileNotFoundError("Gold set 파일을 찾을 수 없습니다.")

    merged = pd.concat(gold_sets, ignore_index=True)
    return merged


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """전처리: 재구매 태그 제거 등"""
    df = df.copy()

    # 텍스트 컬럼 확인
    text_col = 'text' if 'text' in df.columns else 'review_text'

    # 재구매 태그 제거
    df['text_cleaned'] = df[text_col].apply(remove_repurchase_tag)

    # 빈 텍스트 제거
    before_count = len(df)
    df = df[df['text_cleaned'].str.len() > 0]
    after_count = len(df)

    if before_count > after_count:
        print(f"  빈 텍스트 제거: {before_count - after_count}건")

    return df


def split_data(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Train/Valid/Test 분할 (product_code 기준)"""

    # product_code 컬럼 확인
    id_col = None
    for col in ['product_code', 'product_id', 'original_index']:
        if col in df.columns:
            id_col = col
            break

    if id_col and df[id_col].nunique() > 10:
        # product 기준 분할
        unique_ids = df[id_col].unique()

        train_ids, temp_ids = train_test_split(
            unique_ids, test_size=(1-train_ratio), random_state=seed
        )
        val_ids, test_ids = train_test_split(
            temp_ids, test_size=0.5, random_state=seed
        )

        train_df = df[df[id_col].isin(train_ids)]
        val_df = df[df[id_col].isin(val_ids)]
        test_df = df[df[id_col].isin(test_ids)]

        print(f"  분할 기준: {id_col}")
    else:
        # 랜덤 분할
        train_df, temp_df = train_test_split(df, test_size=(1-train_ratio), random_state=seed)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)
        print(f"  분할 기준: 랜덤")

    return train_df, val_df, test_df


def analyze_data(df: pd.DataFrame, name: str):
    """데이터 분포 분석"""
    print(f"\n  [{name}] 총 {len(df)}건")

    # Aspect 분포
    if 'aspect' in df.columns:
        print(f"    Aspect 분포:")
        for aspect, count in df['aspect'].value_counts().head(5).items():
            print(f"      - {aspect}: {count}건")

    # Sentiment 분포
    if 'sentiment' in df.columns:
        print(f"    Sentiment 분포:")
        for sent, count in df['sentiment'].value_counts().items():
            print(f"      - {sent}: {count}건")


def main():
    parser = argparse.ArgumentParser(description='Gold set 통합 및 학습 데이터 준비')
    parser.add_argument('--version', type=str, default='v2', help='버전 태그 (기본: v2)')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Train 비율')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    args = parser.parse_args()

    # 경로 설정
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / "data" / "processed"

    print("=" * 60)
    print("4단계: Gold set 통합 및 학습 데이터 준비")
    print("=" * 60)

    # 1. Gold set 통합
    print("\n[1/4] Gold set 통합")
    merged_df = load_and_merge_gold_sets(processed_dir)
    print(f"  → 총 {len(merged_df)}건 통합 완료")

    # 2. 전처리
    print("\n[2/4] 전처리 (재구매 태그 제거)")
    processed_df = preprocess_data(merged_df)
    print(f"  → 전처리 후 {len(processed_df)}건")

    # 3. 데이터 분할
    print("\n[3/4] Train/Valid/Test 분할")
    train_df, val_df, test_df = split_data(
        processed_df,
        train_ratio=args.train_ratio,
        seed=args.seed
    )

    analyze_data(train_df, "Train")
    analyze_data(val_df, "Valid")
    analyze_data(test_df, "Test")

    # 4. 저장
    print("\n[4/4] 파일 저장")

    version = args.version

    # 통합 파일
    merged_path = processed_dir / f"gold_set_merged_{version}.csv"
    processed_df.to_csv(merged_path, index=False, encoding='utf-8-sig')
    print(f"  → {merged_path.name}")

    # 분할 파일
    train_path = processed_dir / f"train_{version}.csv"
    val_path = processed_dir / f"valid_{version}.csv"
    test_path = processed_dir / f"test_{version}.csv"

    train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
    val_df.to_csv(val_path, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_path, index=False, encoding='utf-8-sig')

    print(f"  → {train_path.name} ({len(train_df)}건)")
    print(f"  → {val_path.name} ({len(val_df)}건)")
    print(f"  → {test_path.name} ({len(test_df)}건)")

    print("\n" + "=" * 60)
    print("완료! 다음 단계:")
    print("=" * 60)
    print(f"""
1. 학습 데이터 확인:
   - {train_path}
   - {val_path}
   - {test_path}

2. 모델 학습 실행 (예정):
   python -m RQ.train --train_file {train_path} --valid_file {val_path}
    """)


if __name__ == "__main__":
    main()
