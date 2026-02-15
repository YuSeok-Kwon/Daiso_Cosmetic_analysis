#!/usr/bin/env python3
"""
temp 디렉토리 정리 스크립트
CSV 파일들을 내용에 따라 분류하고, 문서/스크립트를 정리합니다.
"""

import os
import shutil
from pathlib import Path

# 기준 디렉토리
BASE_DIR = Path(__file__).parent

# 생성할 폴더 구조
FOLDERS = {
    'data/sales': '화장품 매출 데이터',
    'data/analysis': '분석 결과 CSV',
    'docs': '분석 문서 (MD 파일)',
    'scripts': 'Python 스크립트',
}

# 파일 분류 규칙
FILE_RULES = {
    # CSV 파일 분류
    '화장품_매출_2024.csv': 'data/sales',
    '화장품_매출_2025_1_3분기.csv': 'data/sales',
    'Hub_Spoke_상권추천.csv': 'data/analysis',

    # MD 파일 분류
    '다이소뷰티_화장품시장_분석보고서.md': 'docs',
    '통합분석_리포트_2024_2025.md': 'docs',
    '화장품_매출_분석_2024.md': 'docs',
    '화장품_매출_분석_2025_1_3분기.md': 'docs',

    # Python 스크립트 분류
    'analysis_2024_2025.py': 'scripts',
    'download_sales_api.py': 'scripts',
}

def create_folders():
    """필요한 폴더 생성"""
    print("=" * 50)
    print("1. 폴더 생성")
    print("=" * 50)

    for folder, desc in FOLDERS.items():
        folder_path = BASE_DIR / folder
        if not folder_path.exists():
            folder_path.mkdir(parents=True)
            print(f"  [생성] {folder}/ - {desc}")
        else:
            print(f"  [존재] {folder}/ - {desc}")

def move_files():
    """파일 이동"""
    print("\n" + "=" * 50)
    print("2. 파일 이동")
    print("=" * 50)

    moved = []
    skipped = []
    not_found = []

    for filename, dest_folder in FILE_RULES.items():
        src = BASE_DIR / filename
        dest = BASE_DIR / dest_folder / filename

        if src.exists():
            if not dest.exists():
                shutil.move(str(src), str(dest))
                moved.append((filename, dest_folder))
                print(f"  [이동] {filename} → {dest_folder}/")
            else:
                skipped.append((filename, "이미 존재"))
                print(f"  [건너뜀] {filename} - 대상 위치에 이미 존재")
        else:
            # 이미 이동되었는지 확인
            if dest.exists():
                skipped.append((filename, "이미 이동됨"))
                print(f"  [건너뜀] {filename} - 이미 {dest_folder}/에 있음")
            else:
                not_found.append(filename)
                print(f"  [없음] {filename}")

    return moved, skipped, not_found

def print_summary(moved, skipped, not_found):
    """결과 요약"""
    print("\n" + "=" * 50)
    print("3. 정리 결과")
    print("=" * 50)

    print(f"\n  이동 완료: {len(moved)}개")
    print(f"  건너뜀: {len(skipped)}개")
    print(f"  찾을 수 없음: {len(not_found)}개")

    print("\n" + "=" * 50)
    print("최종 디렉토리 구조")
    print("=" * 50)
    print("""
temp/
├── data/
│   ├── sales/           # 화장품 매출 원본 데이터
│   │   ├── 화장품_매출_2024.csv
│   │   └── 화장품_매출_2025_1_3분기.csv
│   └── analysis/        # 분석 결과 CSV
│       └── Hub_Spoke_상권추천.csv
├── docs/                # 분석 문서
│   ├── 다이소뷰티_화장품시장_분석보고서.md
│   ├── 통합분석_리포트_2024_2025.md
│   ├── 화장품_매출_분석_2024.md
│   └── 화장품_매출_분석_2025_1_3분기.md
├── scripts/             # Python 스크립트
│   ├── analysis_2024_2025.py
│   └── download_sales_api.py
├── FOREIGNER/           # 외국인 데이터 (기존 유지)
│   ├── data/
│   ├── foreigner_analysis.py
│   └── run_daiso_analysis.py
└── organize_temp.py     # 본 정리 스크립트 (실행 후 삭제 가능)
""")

def main():
    print("\n" + "=" * 50)
    print("temp 디렉토리 정리 시작")
    print("=" * 50)
    print(f"작업 디렉토리: {BASE_DIR}")

    create_folders()
    moved, skipped, not_found = move_files()
    print_summary(moved, skipped, not_found)

    print("\n정리 완료!")

if __name__ == '__main__':
    main()
