#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
missing_ingredients.txt의 성분을 ingredients_통합.csv에 추가하는 스크립트
"""

import pandas as pd
import re
from pathlib import Path

BASE = Path(__file__).parent.parent
MISSING_PATH = BASE / "missing_ingredients.txt"
INGREDIENTS_PATH = BASE / "data" / "csv" / "ingredients_통합.csv"
PRODUCTS_PATH = BASE / "data" / "products.parquet"


def parse_missing_ingredients(path: Path) -> dict[int, list[str]]:
    """missing_ingredients.txt 파싱: product_id -> [ingredient, ...]"""
    text = path.read_text(encoding="utf-8")
    result = {}

    current_id = None
    for line in text.splitlines():
        line = line.rstrip()
        # 제품 ID (숫자로 시작하는 줄)
        m = re.match(r"^(\d+)\s*$", line)
        if m:
            current_id = int(m.group(1))
            result[current_id] = []
            continue
        # 성분 (• 로 시작)
        if current_id is not None and ("•" in line or "\t•\t" in line):
            ing = line.replace("\t•\t", "").replace("•", "").strip()
            if ing:
                result[current_id].append(ing)

    return result


def main():
    print("=" * 60)
    print("missing_ingredients.txt → ingredients_통합.csv 병합")
    print("=" * 60)

    # 1. 파싱
    print("\n[1/4] missing_ingredients.txt 파싱 중...")
    parsed = parse_missing_ingredients(MISSING_PATH)
    print(f"  - 추출된 제품: {len(parsed)}개")

    # 2. 제품명 조회
    print("\n[2/4] 제품명 조회 중...")
    products = pd.read_parquet(PRODUCTS_PATH)
    product_map = dict(zip(products["product_code"], products["name"]))

    missing_names = [pid for pid in parsed if pid not in product_map]
    if missing_names:
        print(f"  - 제품명 없음: {missing_names[:5]}{'...' if len(missing_names) > 5 else ''}")
        for pid in missing_names:
            product_map[pid] = f"(제품 {pid})"

    # 3. 기존 ingredients 로드
    print("\n[3/4] 기존 ingredients_통합.csv 로드 중...")
    df = pd.read_csv(INGREDIENTS_PATH)
    existing_ids = set(df["product_id"].unique())
    print(f"  - 기존 행: {len(df):,}")
    print(f"  - 기존 제품 수: {len(existing_ids):,}")

    # 4. 추가할 행 생성 (아직 없는 제품만)
    new_rows = []
    added_products = 0
    skipped_existing = 0

    for product_id, ingredients in parsed.items():
        if product_id in existing_ids:
            skipped_existing += 1
            continue
        name = product_map.get(product_id, f"(제품 {product_id})")
        for ing in ingredients:
            new_rows.append({
                "product_id": product_id,
                "name": name,
                "ingredient": ing.strip()
            })
        added_products += 1

    if not new_rows:
        print("\n[4/4] 추가할 성분이 없습니다.")
        if skipped_existing:
            print(f"  - 이미 성분이 있는 제품({skipped_existing}개)은 건너뜀")
        return

    new_df = pd.DataFrame(new_rows)
    df_merged = pd.concat([df, new_df], ignore_index=True)
    df_merged = df_merged.drop_duplicates(subset=["product_id", "name", "ingredient"], keep="first")
    df_merged = df_merged.sort_values(["product_id", "ingredient"]).reset_index(drop=True)

    df_merged.to_csv(INGREDIENTS_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[4/4] 저장 완료")
    print(f"  - 추가된 제품: {added_products}개")
    print(f"  - 추가된 행: {len(new_rows):,}")
    print(f"  - 최종 행: {len(df_merged):,}")
    print(f"  - 최종 제품 수: {df_merged['product_id'].nunique():,}")


if __name__ == "__main__":
    main()
