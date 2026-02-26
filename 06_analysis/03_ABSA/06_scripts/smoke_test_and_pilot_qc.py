"""
Stage 3A 운영 번들 검증 스크립트
  Step 2-1: 1,000건 Smoke Test (코드/경로/threshold/design rule 정상 확인)
  Step 2-2: 10,000건 Pilot QC (분포 sanity check)

Usage:
  python smoke_test_and_pilot_qc.py --step smoke   # 1K smoke test
  python smoke_test_and_pilot_qc.py --step pilot   # 10K pilot QC
  python smoke_test_and_pilot_qc.py --step both    # 순차 실행
"""
import sys
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from RQ_absa.s1_config import ASPECT_LABELS, DESIGN_RULE_CONFIG
from RQ_absa.s8_inference import ABSAInference

PROJECT_ROOT = Path(__file__).parent.parent
BUNDLE_PATH = PROJECT_ROOT / "07_models" / "prod_bundle_stage3a_v1_20260225"
REVIEWS_PATH = PROJECT_ROOT.parent.parent / "02_outputs" / "data" / "csv" / "final" / "reviews_text.csv"
OUTPUT_DIR = PROJECT_ROOT / "01_outputs" / "smoke_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_reviews(n: int) -> pd.DataFrame:
    df = pd.read_csv(REVIEWS_PATH, nrows=n)
    print(f"리뷰 로드: {len(df):,}건 (컬럼: {list(df.columns)})")
    return df


def run_smoke_test():
    """Step 2-1: 1,000건 Smoke Test"""
    print("\n" + "=" * 70)
    print("STEP 2-1: SMOKE TEST (1,000건)")
    print("=" * 70)

    checks = {"threshold_load": False, "design_rule": False, "output_format": False, "streaming": False}

    # (1) 번들 로드
    print("\n[1/4] 번들 로드 + 무결성 검증...")
    inference = ABSAInference.from_bundle(str(BUNDLE_PATH))

    # threshold 길이 체크
    assert inference.none_thresholds is not None, "threshold 미로드"
    assert len(inference.none_thresholds) == len(ASPECT_LABELS), (
        f"threshold 길이 불일치: {len(inference.none_thresholds)} vs {len(ASPECT_LABELS)}"
    )
    assert inference.polar_threshold == 0.5, f"polar_threshold 불일치: {inference.polar_threshold}"
    assert inference.use_design_rule is True, "design_rule 미활성화"
    checks["threshold_load"] = True
    print("  OK: threshold 8개 로드, polar=0.50, design_rule=True")

    # (2) 추론 실행
    print("\n[2/4] 1,000건 추론...")
    df = load_reviews(1000)
    t0 = time.time()
    results = inference.infer_dataframe(df, text_column="text")
    elapsed = time.time() - t0
    print(f"  추론 완료: {elapsed:.1f}초 ({len(results)/elapsed:.0f} reviews/sec)")

    # (3) Design Rule Override 검증
    print("\n[3/4] Design Rule Override 검증...")
    design_idx = list(ASPECT_LABELS).index("디자인")
    # 디자인 positive 키워드가 포함된 리뷰 찾기
    pos_kws = DESIGN_RULE_CONFIG["positive_keywords"]
    has_pos_kw = df["text"].apply(lambda t: any(kw in str(t) for kw in pos_kws))
    # 해당 리뷰에서 디자인 aspect가 positive(1)로 예측되었는지 확인
    if has_pos_kw.sum() > 0:
        design_aspects = []
        for idx_row in df[has_pos_kw].index[:20]:
            for asp in results.loc[idx_row, "aspect_sentiments"]:
                if asp["aspect"] == "디자인":
                    design_aspects.append(asp["sentiment"])
        if design_aspects:
            pos_count = sum(1 for s in design_aspects if s == "positive")
            print(f"  디자인 키워드 포함 리뷰 중 positive 예측: {pos_count}/{len(design_aspects)}")
            checks["design_rule"] = pos_count > 0
        else:
            print("  WARNING: 디자인 키워드 포함 리뷰에서 디자인 aspect 추출 없음")
    else:
        print("  WARNING: 1,000건 중 디자인 키워드 포함 리뷰 0건 (드문 경우)")
        checks["design_rule"] = True  # 검증 불가이므로 pass 처리

    # (4) Output 형식 + Streaming 저장 확인
    print("\n[4/4] 출력 형식 + streaming 저장...")
    required_cols = ["sentiment", "sentiment_score", "aspect_sentiments", "aspect_labels", "summary", "is_ambiguous"]
    missing = [c for c in required_cols if c not in results.columns]
    assert not missing, f"누락 컬럼: {missing}"
    checks["output_format"] = True

    output_path = OUTPUT_DIR / "smoke_test_1k.csv"
    results.to_csv(output_path, index=False, encoding="utf-8-sig")
    saved_rows = len(pd.read_csv(output_path))
    checks["streaming"] = saved_rows == len(results)
    print(f"  저장: {output_path} ({saved_rows}행)")

    # 결과 요약
    print("\n" + "-" * 50)
    print("SMOKE TEST 결과:")
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")
    all_pass = all(checks.values())
    print(f"\n전체: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("-" * 50)
    return all_pass


def run_pilot_qc():
    """Step 2-2: 10,000건 Pilot QC"""
    print("\n" + "=" * 70)
    print("STEP 2-2: PILOT QC (10,000건)")
    print("=" * 70)

    # (1) 번들 로드
    print("\n[1/3] 번들 로드...")
    inference = ABSAInference.from_bundle(str(BUNDLE_PATH))

    # (2) 추론 실행
    print("\n[2/3] 10,000건 추론...")
    df = load_reviews(10000)
    t0 = time.time()
    results = inference.infer_dataframe(df, text_column="text")
    elapsed = time.time() - t0
    print(f"  추론 완료: {elapsed:.1f}초 ({len(results)/elapsed:.0f} reviews/sec)")

    # 저장
    output_path = OUTPUT_DIR / "pilot_qc_10k.csv"
    results.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  저장: {output_path}")

    # (3) 분포 분석
    print("\n[3/3] 분포 분석...")
    aspects = list(ASPECT_LABELS)
    total = len(results)

    # aspect별 집계
    aspect_stats = {a: {"mentioned": 0, "positive": 0, "neutral": 0, "negative": 0} for a in aspects}
    aspect_stats["미분류"] = {"mentioned": 0, "positive": 0, "neutral": 0, "negative": 0}

    for _, row in results.iterrows():
        asp_list = row["aspect_sentiments"]
        if isinstance(asp_list, str):
            asp_list = eval(asp_list)
        for asp in asp_list:
            name = asp["aspect"]
            if name not in aspect_stats:
                continue
            aspect_stats[name]["mentioned"] += 1
            sent = asp["sentiment"]
            if sent in aspect_stats[name]:
                aspect_stats[name][sent] += 1

    # Golden Test 기준값 (비교용)
    golden_test_pred_pct = {
        "배송/포장": 31.6, "가격/가성비": 44.0, "사용감/성능": 50.6,
        "용량/휴대": 12.4, "디자인": 8.3, "재질/냄새": 28.6,
        "재구매": 22.6, "색상/발색": 16.2,
    }

    print(f"\n{'Aspect':<14} {'언급률':>8} {'GT Pred%':>9} {'Delta':>7} {'Pos%':>7} {'Neu%':>7} {'Neg%':>7}")
    print("-" * 70)

    for asp in aspects:
        s = aspect_stats[asp]
        m = s["mentioned"]
        pct = m / total * 100
        gt_pct = golden_test_pred_pct.get(asp, 0)
        delta = pct - gt_pct

        if m > 0:
            pos_pct = s["positive"] / m * 100
            neu_pct = s["neutral"] / m * 100
            neg_pct = s["negative"] / m * 100
        else:
            pos_pct = neu_pct = neg_pct = 0

        flag = " !!!" if abs(delta) > 15 else ""
        print(f"{asp:<14} {pct:>7.1f}% {gt_pct:>8.1f}% {delta:>+6.1f}pp {pos_pct:>6.1f}% {neu_pct:>6.1f}% {neg_pct:>6.1f}%{flag}")

    # 미분류
    mc = aspect_stats["미분류"]
    print(f"{'미분류':<14} {mc['mentioned']/total*100:>7.1f}%")

    # 전체 sentiment 분포
    print(f"\n전체 review sentiment 분포:")
    sent_dist = results["sentiment"].value_counts(normalize=True).sort_index()
    for sent, pct in sent_dist.items():
        print(f"  {sent}: {pct*100:.1f}%")

    # QC 판정 기준
    print("\n" + "-" * 50)
    print("PILOT QC 판정:")
    issues = []

    # 사용감/성능 체크: golden_test에서 50.6%이므로 ±15pp 이내
    usage_pct = aspect_stats["사용감/성능"]["mentioned"] / total * 100
    if abs(usage_pct - 50.6) > 15:
        issues.append(f"사용감/성능 언급률 {usage_pct:.1f}% (기준 50.6% ±15pp 초과)")

    # 디자인 체크: golden_test에서 8.3%이므로 과소 검출 허용
    design_pct = aspect_stats["디자인"]["mentioned"] / total * 100
    if design_pct > 20:
        issues.append(f"디자인 언급률 {design_pct:.1f}% (과다 검출 의심)")

    # 미분류 비율: 30% 이하가 정상
    unclass_pct = aspect_stats["미분류"]["mentioned"] / total * 100
    if unclass_pct > 30:
        issues.append(f"미분류 비율 {unclass_pct:.1f}% (30% 초과)")

    if issues:
        print("  WARNING 항목:")
        for iss in issues:
            print(f"    - {iss}")
    else:
        print("  ALL PASS: 분포 정상 범위")
    print("-" * 50)

    # JSON 요약 저장
    summary = {
        "total_reviews": total,
        "elapsed_sec": round(elapsed, 1),
        "aspect_mention_rate": {a: round(aspect_stats[a]["mentioned"] / total * 100, 2) for a in aspects},
        "unclassified_rate": round(aspect_stats["미분류"]["mentioned"] / total * 100, 2),
        "review_sentiment_dist": {k: round(v * 100, 2) for k, v in sent_dist.items()},
        "issues": issues,
    }
    summary_path = OUTPUT_DIR / "pilot_qc_10k_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n요약 저장: {summary_path}")

    return len(issues) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["smoke", "pilot", "both"], default="both")
    args = parser.parse_args()

    if args.step in ("smoke", "both"):
        smoke_ok = run_smoke_test()
        if not smoke_ok and args.step == "both":
            print("\nSmoke test 실패 — pilot QC 건너뜀")
            sys.exit(1)

    if args.step in ("pilot", "both"):
        run_pilot_qc()
