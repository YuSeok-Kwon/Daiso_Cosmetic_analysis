"""
약점 Aspect 데이터 추출 스크립트 (GPT-4o 라벨링 후보)

소스: reviews_text.csv (30만+ 리뷰)
대상 aspect: CS/응대, 디자인, 품질/퀄리티
출력: aspect별 키워드 매치 리뷰 CSV (GPT-4o 라벨링 대상)
"""
import sys
from pathlib import Path

ABSA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ABSA_ROOT))

import pandas as pd

from RQ_absa.s1_config import PROCESSED_DATA_DIR

# Why-pi 프로젝트 루트의 reviews_text.csv
REVIEWS_TEXT_PATH = ABSA_ROOT.parent.parent / "02_outputs" / "data" / "csv" / "final" / "reviews_text.csv"

# 기존 학습 데이터 (중복 제거용)
EXISTING_TRAIN_PATH = PROCESSED_DATA_DIR / "training" / "absa_wide_train.csv"

# Aspect별 키워드 & 상한
ASPECT_CONFIGS = {
    "CS_응대": {
        "keywords": [
            "교환", "환불", "반품", "고객센터", "불친절", "응대", "답변",
            "문의", "상담", "cs", "CS", "서비스", "처리", "안내",
            "전화", "연락", "대응", "클레임",
        ],
        "max_samples": 2000,
    },
    "디자인_긍정": {
        "keywords": [
            "예쁘", "이쁘", "귀엽", "디자인", "패키지", "깔끔",
            "고급스러", "심플", "외관", "색감",
        ],
        "max_samples": 1500,
    },
    "품질_혼동": {
        "keywords": [
            "불량", "깨짐", "깨져", "분리", "새서", "새어",
            "터짐", "파손", "하자", "결함", "찢어", "갈라",
        ],
        "max_samples": 1500,
    },
}


def has_any_keyword(text: str, keywords: list) -> bool:
    text_lower = str(text).lower()
    return any(kw.lower() in text_lower for kw in keywords)


def main():
    # ── 리뷰 로드 ──
    assert REVIEWS_TEXT_PATH.exists(), f"리뷰 파일 없음: {REVIEWS_TEXT_PATH}"
    print(f"리뷰 로드: {REVIEWS_TEXT_PATH}")
    reviews_df = pd.read_csv(REVIEWS_TEXT_PATH)
    print(f"  총 리뷰: {len(reviews_df):,}행")

    # ── 기존 학습 데이터 review_id 제거 ──
    existing_ids = set()
    if EXISTING_TRAIN_PATH.exists():
        train_df = pd.read_csv(EXISTING_TRAIN_PATH, usecols=["review_id"])
        existing_ids = set(train_df["review_id"].astype(str).tolist())
        print(f"  기존 학습 데이터 review_id: {len(existing_ids):,}개")

    reviews_df["review_id_str"] = reviews_df["review_id"].astype(str)
    new_reviews = reviews_df[~reviews_df["review_id_str"].isin(existing_ids)].copy()
    print(f"  기존 학습 데이터 제외 후: {len(new_reviews):,}행")

    # ── Aspect별 추출 ──
    output_dir = PROCESSED_DATA_DIR / "weak_aspect_candidates"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_extracted = 0

    for aspect_name, cfg in ASPECT_CONFIGS.items():
        keywords = cfg["keywords"]
        max_samples = cfg["max_samples"]

        # 키워드 매치
        matched = new_reviews[new_reviews["text"].apply(
            lambda t: has_any_keyword(t, keywords)
        )].copy()
        print(f"\n[{aspect_name}]")
        print(f"  키워드 매치: {len(matched):,}행")

        # 상한 적용 (랜덤 샘플)
        if len(matched) > max_samples:
            matched = matched.sample(n=max_samples, random_state=42)
            print(f"  샘플링: {max_samples:,}행")

        # 저장
        output_path = output_dir / f"{aspect_name}_candidates.csv"
        matched[["review_id", "text", "review_length"]].to_csv(
            output_path, index=False, encoding="utf-8-sig"
        )
        print(f"  저장: {output_path}")
        total_extracted += len(matched)

    print(f"\n총 추출: {total_extracted:,}행")
    print(f"출력 디렉토리: {output_dir}")


if __name__ == "__main__":
    main()
