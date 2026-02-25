"""
Stage 3A 운영 번들로 전체 리뷰 추론 (43만 건)
- 번들에서 모델+threshold+design rule 로드
- 10K 청크 단위 streaming CSV 저장
- 중간 진행률 + ETA 출력
"""
import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from RQ_absa.s1_config import ASPECT_LABELS
from RQ_absa.s8_inference import ABSAInference

PROJECT_ROOT = Path(__file__).parent.parent
BUNDLE_PATH = PROJECT_ROOT / "07_models" / "prod_bundle_stage3a_v1_20260225"
REVIEWS_PATH = PROJECT_ROOT.parent.parent / "02_processed_data" / "csv" / "final" / "reviews_text.csv"
OUTPUT_PATH = PROJECT_ROOT / "04_outputs" / "inference" / "absa_results_stage3a_full.csv"
SUMMARY_PATH = PROJECT_ROOT / "04_outputs" / "inference" / "absa_results_stage3a_summary.json"

CHUNK_SIZE = 10_000


def main():
    print("=" * 70)
    print("FULL INFERENCE — Stage 3A 운영 번들")
    print("=" * 70)

    # 총 행 수 사전 확인
    total_lines = sum(1 for _ in open(REVIEWS_PATH)) - 1
    print(f"입력: {REVIEWS_PATH}")
    print(f"총 리뷰: {total_lines:,}건")
    print(f"출력: {OUTPUT_PATH}")
    print(f"청크 크기: {CHUNK_SIZE:,}\n")

    # 번들 로드
    inference = ABSAInference.from_bundle(str(BUNDLE_PATH))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    total_ambiguous = 0
    header_written = False
    t_start = time.time()
    chunk_idx = 0

    # aspect 집계용
    aspect_counts = {a: {"mentioned": 0, "positive": 0, "neutral": 0, "negative": 0} for a in ASPECT_LABELS}
    aspect_counts["미분류"] = {"mentioned": 0, "positive": 0, "neutral": 0, "negative": 0}
    sent_counts = {"positive": 0, "neutral": 0, "negative": 0}

    print(f"Streaming inference 시작...\n")

    for chunk_df in pd.read_csv(REVIEWS_PATH, chunksize=CHUNK_SIZE):
        chunk_idx += 1
        t_chunk = time.time()

        chunk_result = inference.infer_dataframe(chunk_df, text_column="text")

        # CSV append
        chunk_result.to_csv(
            OUTPUT_PATH,
            mode="a" if header_written else "w",
            header=not header_written,
            index=False,
            encoding="utf-8-sig",
        )
        header_written = True

        total_rows += len(chunk_result)
        total_ambiguous += chunk_result["is_ambiguous"].sum()

        # aspect 집계
        for _, row in chunk_result.iterrows():
            asp_list = row["aspect_sentiments"]
            if isinstance(asp_list, str):
                asp_list = eval(asp_list)
            for asp in asp_list:
                name = asp["aspect"]
                if name in aspect_counts:
                    aspect_counts[name]["mentioned"] += 1
                    sent = asp["sentiment"]
                    if sent in aspect_counts[name]:
                        aspect_counts[name][sent] += 1
            # review sentiment
            sent = row["sentiment"]
            if sent in sent_counts:
                sent_counts[sent] += 1

        elapsed = time.time() - t_start
        chunk_elapsed = time.time() - t_chunk
        speed = total_rows / elapsed
        remaining = (total_lines - total_rows) / speed if speed > 0 else 0
        pct = total_rows / total_lines * 100

        print(f"  [{chunk_idx}] {total_rows:>7,}/{total_lines:,} "
              f"({pct:5.1f}%) | "
              f"chunk {chunk_elapsed:.0f}s | "
              f"총 {elapsed:.0f}s | "
              f"ETA {remaining:.0f}s | "
              f"{speed:.0f} rev/s")

    elapsed_total = time.time() - t_start

    # 최종 통계
    print(f"\n{'=' * 70}")
    print(f"INFERENCE COMPLETE")
    print(f"{'=' * 70}")
    print(f"총 리뷰: {total_rows:,}")
    print(f"총 시간: {elapsed_total:.0f}초 ({elapsed_total/60:.1f}분)")
    print(f"속도: {total_rows/elapsed_total:.0f} reviews/sec")
    print(f"Ambiguous: {total_ambiguous:,} ({total_ambiguous/total_rows*100:.1f}%)")

    print(f"\n--- Aspect 언급률 ---")
    aspects = list(ASPECT_LABELS)
    for asp in aspects:
        s = aspect_counts[asp]
        m = s["mentioned"]
        pct = m / total_rows * 100
        if m > 0:
            pos = s["positive"] / m * 100
            neu = s["neutral"] / m * 100
            neg = s["negative"] / m * 100
        else:
            pos = neu = neg = 0
        print(f"  {asp:<14} {pct:>6.1f}%  (pos {pos:.0f}% / neu {neu:.0f}% / neg {neg:.0f}%)")
    mc = aspect_counts["미분류"]["mentioned"]
    print(f"  {'미분류':<14} {mc/total_rows*100:>6.1f}%")

    print(f"\n--- Review Sentiment ---")
    for sent in ["positive", "neutral", "negative"]:
        print(f"  {sent}: {sent_counts[sent]:,} ({sent_counts[sent]/total_rows*100:.1f}%)")

    # JSON 요약 저장
    summary = {
        "total_reviews": total_rows,
        "elapsed_sec": round(elapsed_total, 1),
        "speed_rev_per_sec": round(total_rows / elapsed_total, 1),
        "ambiguous_count": int(total_ambiguous),
        "ambiguous_pct": round(total_ambiguous / total_rows * 100, 2),
        "bundle": "prod_bundle_stage3a_v1_20260225",
        "aspect_mention_rate": {
            a: round(aspect_counts[a]["mentioned"] / total_rows * 100, 2) for a in aspects
        },
        "aspect_sentiment_dist": {
            a: {
                "positive": aspect_counts[a]["positive"],
                "neutral": aspect_counts[a]["neutral"],
                "negative": aspect_counts[a]["negative"],
            } for a in aspects
        },
        "unclassified_rate": round(mc / total_rows * 100, 2),
        "review_sentiment_dist": {k: v for k, v in sent_counts.items()},
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n요약 저장: {SUMMARY_PATH}")
    print(f"결과 저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
