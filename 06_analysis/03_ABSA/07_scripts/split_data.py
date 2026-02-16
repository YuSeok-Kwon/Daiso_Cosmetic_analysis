"""
데이터 분할 스크립트
29,534건 리뷰를 6명 팀원용으로 분할
"""

import pandas as pd
from pathlib import Path

# 경로 설정 (Windows/Mac 호환)
project_root = Path(__file__).parent.parent
raw_dir = project_root / "data" / "raw"
split_dir = raw_dir / "split"
split_dir.mkdir(parents=True, exist_ok=True)

# 원본 데이터 로드
source_file = raw_dir / "sampled_reviews_20k.csv"
df = pd.read_csv(source_file)

print(f"총 리뷰 수: {len(df)}건")
print(f"컬럼: {list(df.columns)}")

# 6등분
n_members = 6
chunk_size = len(df) // n_members
remainder = len(df) % n_members

# 분할
start = 0
for i in range(n_members):
    # 나머지 분배 (앞쪽 팀원에게 1건씩 추가)
    end = start + chunk_size + (1 if i < remainder else 0)

    chunk = df.iloc[start:end].copy()
    chunk['original_index'] = range(start, end)  # 원본 인덱스 보존

    output_file = split_dir / f"team_{i+1}.csv"
    chunk.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"팀원{i+1}: {start+1} ~ {end}번 ({len(chunk)}건) → {output_file.name}")

    start = end

print(f"\n분할 완료! 저장 위치: {split_dir}")
