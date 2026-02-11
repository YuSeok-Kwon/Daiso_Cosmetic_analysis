"""
카테고리별 성분 CSV 파일을 하나로 합치는 스크립트
"""

import pandas as pd
from pathlib import Path

# 프로젝트 루트 경로
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "csv" / "Category"

# 합칠 파일 목록
FILES = [
    "ingredients_맨케어.csv",
    "ingredients_메이크업_립메이크업_20260209.csv",
    "ingredients_메이크업_베이스메이크업_치크_하이라이터_20260210.csv",
    "ingredients_메이크업_아이메이크업_20260209.csv",
    "ingredients_스킨케어_기초스킨케어_20260210.csv",
    "ingredients_스킨케어_립케어_클렌징_필링_선크림20260209.csv",
    "ingredients_스킨케어_팩_마스크20260208.csv",
]

def main():
    dfs = []
    for f in FILES:
        path = DATA_DIR / f
        if path.exists():
            df = pd.read_csv(path)
            dfs.append(df)
            print(f"  ✓ {f}: {len(df):,}행")
        else:
            print(f"  ✗ {f}: 파일 없음")

    if not dfs:
        print("합칠 파일이 없습니다.")
        return

    # 병합 (중복 제거)
    merged = pd.concat(dfs, ignore_index=True)
    before_count = len(merged)
    merged = merged.drop_duplicates(
        subset=["product_id", "name", "ingredient"],
        keep="first"
    )
    after_count = len(merged)
    dup_count = before_count - after_count

    # 저장
    output_path = DATA_DIR / "ingredients_통합.csv"
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n결과: {output_path}")
    print(f"  - 총 행 수: {after_count:,}")
    print(f"  - 중복 제거: {dup_count:,}건")

if __name__ == "__main__":
    main()
