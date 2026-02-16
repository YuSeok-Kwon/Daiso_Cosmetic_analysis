#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ingredients_통합.csv에 성분명 정리 규칙 적용
- docs/ingredients/ 성분명 정리 규칙.pdf, OCR 파일 교정 규칙 정리.docx 기반
- 1) 삭제 2) 수정 3) 분리 규칙 적용
"""

import pandas as pd
import re
from pathlib import Path

BASE = Path(__file__).parent.parent
INPUT_PATH = BASE / "data" / "csv" / "ingredients_통합.csv"
OUTPUT_PATH = BASE / "data" / "csv" / "ingredients_통합.csv"


def load_rules():
    """규칙 데이터 로드"""
    # 1) 삭제 대상
    delete_targets = [
        '계산', '고민', '고민인', '그린', '국내산', '다이소입장브랜드', 'Panax',
        '대한민', '독자적인', '신강남추출물', '씨엠에스랩이올', '엘론추출물',
        '반짝눈물요정프로필렌글라이콜', '성분추출물', '전성분 없음', '전성분변성알코올',
        '국민더추출물', '금사연둥지추출물', '노티드랙추출물',
        '다이메청색본', '다이바닐피아파틴', '메디아티놀레알리파리아이리폴리메이트',
        '6868추출물', 'DSM사순도99%순수비타민', 'KOREA산다이올',
        '나무열매오일', '나무잎추출물', '나프탈레이트', '미리스테이트', '미리스틸',
        '동수서일수수꽃다리조추출물', '듐둠이이루루네이트',
        'Came', 'Caracol', 'Complex', 'Derm', 'Derma', 'Ingredient', 'Skin', 'Skincare',
    ]
    # PDF 기반 삭제 목록 추가
    delete_targets += [
        '네스탈[스테이트', '네임늄[리스테이트', '글리세릴카', '글리세릴폴리',
        '다이소에서만나는메마른피부를위한수분보습', '랩팩토리멀티밤진정스틱드오일',
        '듐눔파이트', '듐듐로라이드', '듐듐테테아로일글루타메이트',
        '반짝반짝작은별프로필렌글라이콜', '부위인', '발효물인', '성분인',
        '베이트', '벤조에이트', '레이트', '로네이트', '로닉애시드', '로아세틱애시드',
        '비니거', '비로메민', '소듐아', '소듐하', '타타인', '턱라인', '테아레이트',
        '테이트', '트리', '팩미인', '펜타', '포네이트', '포타슘', '포타슘하',
        '폴리', '폴리C', '폴리글리', '폴리글리세', '폴리머', '폴리메틸',
        '피부고민', '피부염', '해서린', '핵심적인', '헥사머', '화이트', '효과적인',
    ]

    # 2) 수정 규칙 (오타 → 표준명)
    corrections = {
        # 다이/소듐/듐 오타
        '다0프로필렌글라이콜': '다이프로필렌글라이콜',
        '다이포타슘글리리제이트': '다이포타슘글리시리제이트',
        '다이포타슘글리시리제o트': '다이포타슘글리시리제이트',
        '둠듐하이알루로네이트': '소듐하이알루로네이트',
        '듐하이알루로네이트': '소듐하이알루로네이트',
        '솔비탄아이소스테o레이트': '솔비탄아이소스테아레이트',
        '소듐아크릴레이트/소듐아크릴로일다O메틸타우레이트코폴리머': '소듐아크릴레이트/소듐아크릴로일다이메틸타우레이트코폴리머',
        # 실리콘/다이메티콘
        '다이메티콘/비닐다O메티콘크로스폴리머': '다이메티콘/비닐다이메티콘크로스폴리머',
        '다이메티콘/비닐다이메티콘크스폴리머': '다이메티콘/비닐다이메티콘크로스폴리머',
        '이소프로필팔미터이트': '이소프로필팔미테이트',
        # 펜타에리스리틸
        '펜타에리스리틸테트라-다0-+-부틸하이드록시하이드로신나메이트': '펜타에리스리틸테트라-다이-t-부틸하이드록시하이드로신나메이트',
        '펜타에리스리틸테트라-다이-T-부틸하이드록시하이드로신나메이트': '펜타에리스리틸테트라-다이-t-부틸하이드록시하이드로신나메이트',
        '부담하이드록시하이드로신나메이트': '다이-t-부틸하이드록시하이드로신나메이트',
        # 피토스테릴
        '피토스테릴0소스테아레이트': '피토스테릴아이소스테아레이트',
        # VT 오타 (데이터에서 확인)
        'PDRN5': 'PDRN',
        '계산': None,  # 삭제
    }

    # None은 삭제 처리
    corrections = {k: v for k, v in corrections.items() if v is not None}

    # 3) 분리 규칙 (복합 → [A, B])
    split_rules = {
        '나이아신아마이드다이프로필렌글라이콜': ['나이아신아마이드', '다이프로필렌글라이콜'],
        '다이소듐이디티에이부틸렌글라이콜': ['다이소듐이디티에이', '부틸렌글라이콜'],
        '글리세릴카프릴레이트에틸헥실글리세린': ['글리세릴카프릴레이트', '에틸헥실글리세린'],
        '다이메티콘이소프로필팔미터이트': ['다이메티콘', '아이소프로필팔미테이트'],
        '판테놀부틸렌글라이콜': ['판테놀', '부틸렌글라이콜'],
        '소듐하이알루로네이트다이메틸실란올': ['소듐하이알루로네이트', '다이메틸실란올'],
        '티타늄디옥사이드헥실라우레이트': ['티타늄디옥사이드', '헥실라우레이트'],
        '마그네슘/포타슘/실리콘/플루오라이드/하이드록사이드/옥사이드': ['마그네슘', '포타슘', '실리콘', '플루오라이드', '하이드록사이드', '옥사이드'],
    }

    return delete_targets, corrections, split_rules


def main():
    print("=" * 60)
    print("성분명 정리 규칙 적용")
    print("=" * 60)

    if not INPUT_PATH.exists():
        print(f"오류: {INPUT_PATH} 파일이 없습니다.")
        return

    df = pd.read_csv(INPUT_PATH)
    print(f"\n[입력] {len(df):,}행, {df['product_id'].nunique():,}개 제품")

    delete_targets, corrections, split_rules = load_rules()

    # 1) 삭제
    delete_mask = df['ingredient'].isin(delete_targets)
    deleted = delete_mask.sum()
    df = df[~delete_mask].copy()
    print(f"[삭제] {deleted:,}행 제거")

    # 2) 수정 (여러 패스 - corrections 적용)
    corr_count = 0
    for old, new in corrections.items():
        mask = df['ingredient'] == old
        if mask.any():
            df.loc[mask, 'ingredient'] = new
            corr_count += mask.sum()
    print(f"[수정] {corr_count:,}건 교정")

    # 3) 마케팅 문구 제거 (정규식)
    for pattern in [
        r'^기재해야하는모든성분', r'^기재표시하여야하는', r'^기재\.표시', r'^기재',
        r'^표시하여야하는모든성분', r'^표시', r'제조국대한민국',
        r'화장품법에따라', r'식품의약안전처', r'사용할때의주의사항',
    ]:
        df['ingredient'] = df['ingredient'].str.replace(pattern, '', regex=True)

    # 4) 분리
    new_rows = []
    drop_idx = []
    for idx, row in df.iterrows():
        ing = row['ingredient']
        if ing in split_rules:
            drop_idx.append(idx)
            for new_ing in split_rules[ing]:
                r = row.copy()
                r['ingredient'] = new_ing.strip()
                if r['ingredient']:
                    new_rows.append(r)
    if drop_idx:
        df = df.drop(drop_idx)
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        print(f"[분리] {len(drop_idx)}행 → {len(new_rows)}행으로 분리")

    # 5) 정리
    df['ingredient'] = df['ingredient'].str.strip()
    df = df[df['ingredient'] != ''].copy()
    df = df.drop_duplicates(subset=['product_id', 'name', 'ingredient'], keep='first')
    df = df.sort_values(['product_id', 'ingredient']).reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"\n[저장] {OUTPUT_PATH}")
    print(f"[결과] {len(df):,}행, {df['ingredient'].nunique():,}개 고유 성분")


if __name__ == "__main__":
    main()
