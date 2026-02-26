"""
전체 리뷰 추론 (323K건)
- 번들 또는 체크포인트 디렉토리에서 모델+threshold+design rule 로드
- 10K 청크 단위 streaming CSV 저장
- 중간 진행률 + ETA 출력

사용법:
  # Stage 3A 번들 (기본)
  python run_full_inference.py

  # S4B 체크포인트
  python run_full_inference.py --model-dir checkpoints_stage4b --tag stage4b
"""
import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from RQ_absa.s1_config import ASPECT_LABELS
from RQ_absa.s8_inference import ABSAInference

PROJECT_ROOT = Path(__file__).parent.parent
BUNDLE_PATH = PROJECT_ROOT / "07_models" / "prod_bundle_stage3a_v1_20260225"
REVIEWS_PATH = PROJECT_ROOT / "01_outputs" / "data" / "source" / "reviews_text_no_reorder.csv"

CHUNK_SIZE = 10_000


def _load_from_checkpoint_dir(model_dir: Path, batch_size: int = 128) -> ABSAInference:
    """체크포인트 디렉토리에서 모델+threshold+design rule 직접 로드.

    번들이 아닌 일반 체크포인트 디렉토리(예: checkpoints_stage4b/)에서
    모델을 로드하고, design_rule_config는 운영 번들에서 가져온다.
    """
    from transformers import AutoTokenizer
    from RQ_absa.s5_model import load_model
    import RQ_absa.s1_config as cfg

    model_path = model_dir / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")

    print(f"=== 체크포인트 직접 로드: {model_dir.name} ===")

    # (1) 모델 + 토크나이저
    model_name = "beomi/KcELECTRA-base"
    print(f"  모델 로드: {model_path.name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = load_model(checkpoint_path=model_path, model_name=model_name)

    # (2) Threshold 로드
    none_thresholds = None
    polar_threshold = None
    use_design_rule = False

    threshold_path = model_dir / "none_thresholds.json"
    if threshold_path.exists():
        with open(threshold_path, "r", encoding="utf-8") as f:
            tdata = json.load(f)
        none_thresholds = np.array(tdata["thresholds"])
        polar_threshold = tdata.get("polar_threshold")
        use_design_rule = tdata.get("design_rule", False)
        print(f"  Thresholds 로드: {threshold_path.name}")
        print(f"    polar_threshold={polar_threshold}, design_rule={use_design_rule}")

    # (3) Design Rule Config — 운영 번들에서 가져오기
    drule_path = BUNDLE_PATH / "design_rule_config.json"
    if drule_path.exists():
        with open(drule_path, "r", encoding="utf-8") as f:
            drule_data = json.load(f)
        if "design_rule_config" in drule_data:
            cfg.DESIGN_RULE_CONFIG = drule_data["design_rule_config"]
            print("  Design Rule Config: 운영 번들에서 로드")
        # keyword_gate_config: 번들(substring 기반) 대신 s1_config.py(정규식 기반) 사용
        # Stage 4C에서 정규식으로 전환했으므로 번들 로드 스킵
        if "keyword_gate_config" in drule_data:
            print("  Keyword Gate Config: s1_config.py 정규식 Gate 유지 (번들 스킵)")
        if "keyword_force_on_config" in drule_data:
            cfg.KEYWORD_FORCE_ON_CONFIG = drule_data["keyword_force_on_config"]
            print("  Keyword Force-On Config: 운영 번들에서 로드")
        if "qc_postprocess_config" in drule_data:
            cfg.QC_POSTPROCESS_CONFIG = drule_data["qc_postprocess_config"]
            print("  QC Postprocess Config: 운영 번들에서 로드")

    print("=== 체크포인트 로드 완료 ===\n")

    return ABSAInference(
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        none_thresholds=none_thresholds,
        polar_threshold=polar_threshold,
        use_design_rule=use_design_rule,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="ABSA 전체 리뷰 추론")
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="체크포인트 디렉토리 (07_models/ 하위). 미지정 시 운영 번들 사용",
    )
    parser.add_argument(
        "--tag", type=str, default="stage3a_v4",
        help="출력 파일 태그 (기본: stage3a_v4)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 출력 경로 결정
    output_path = PROJECT_ROOT / "01_outputs" / "inference" / f"absa_results_{args.tag}_full.csv"
    summary_path = PROJECT_ROOT / "01_outputs" / "inference" / f"absa_results_{args.tag}_summary.json"

    # 모델 로드
    if args.model_dir:
        model_dir = PROJECT_ROOT / "07_models" / args.model_dir
        if not model_dir.exists():
            model_dir = Path(args.model_dir)  # 절대 경로 fallback
        print("=" * 70)
        print(f"FULL INFERENCE — {model_dir.name}")
        print("=" * 70)
        inference = _load_from_checkpoint_dir(model_dir)
    else:
        print("=" * 70)
        print("FULL INFERENCE — Stage 3A 운영 번들")
        print("=" * 70)
        inference = ABSAInference.from_bundle(str(BUNDLE_PATH))

    # 총 행 수 사전 확인
    total_lines = sum(1 for _ in open(REVIEWS_PATH)) - 1
    print(f"입력: {REVIEWS_PATH}")
    print(f"총 리뷰: {total_lines:,}건")
    print(f"출력: {output_path}")
    print(f"청크 크기: {CHUNK_SIZE:,}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)

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
            output_path,
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
        "bundle": args.tag,
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
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n요약 저장: {summary_path}")
    print(f"결과 저장: {output_path}")


if __name__ == "__main__":
    main()
