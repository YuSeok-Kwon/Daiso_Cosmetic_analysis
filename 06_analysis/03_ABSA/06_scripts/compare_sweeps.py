"""
ABSA Stage 5 — Sweep 결과 비교

전략 A(warm-start) + 전략 B(Phase 2) + 전략 C(Focal Loss) 실험 결과를 한 눈에 비교.
각 실험의 golden_eval_metrics*.json을 읽어 핵심 지표 테이블 출력.

사용법:
  python compare_sweeps.py
  python compare_sweeps.py --include-baselines
"""
import json
import argparse
from pathlib import Path

ABSA_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = ABSA_ROOT / "07_models"

# 전략 A: Warm-start sweep 태그
WARMSTART_TAGS = [
    "ws_lr5e6_ep5",
    "ws_lr2e6_ep7",
    "ws_lr1e5_ep3",
    "ws_lr5e6_ep3",
]

# 전략 B: Phase 2 sweep 태그
PHASE2_TAGS = [
    "p2_lr1e5_ep5",
    "p2_lr5e5_ep3",
    "p2_lr5e6_ep7",
    "p2_lr1e4_ep3",
]

# 전략 C: Phase 3 (Focal Loss) sweep 태그
PHASE3_TAGS = [
    "p3_g1.5_lr1e5_ep5",
    "p3_g2.0_lr1e5_ep5",
    "p3_g2.0_lr5e6_ep7",
    "p3_g3.0_lr5e6_ep5",
]

# GO 기준
GO_CRITERIA = {
    "det_precision": 0.65,
    "det_f1": 0.60,
    "mentioned_sentiment_f1": 0.43,
    "neutral_recall": 0.25,
}

# 베이스라인 (Golden Test 266건, full postprocess 기준)
BASELINES = {
    "Stage2+3A": {
        "det_precision": 0.8203,
        "det_recall": 0.6446,
        "det_f1": 0.7219,
        "mentioned_sentiment_f1": 0.3669,
        "neutral_recall": 0.1379,
        "config": "lr=2e-5, ep=10, scratch+3A후처리",
    },
    "S4(scratch)+3A": {
        "det_precision": 0.8367,
        "det_recall": 0.6538,
        "det_f1": 0.7340,
        "mentioned_sentiment_f1": 0.3480,
        "neutral_recall": 0.1552,
        "config": "lr=2e-5, ep=10, scratch+3A후처리",
    },
    "S4B(warm)+3A": {
        "det_precision": 0.8290,
        "det_recall": 0.6515,
        "det_f1": 0.7296,
        "mentioned_sentiment_f1": 0.3889,
        "neutral_recall": 0.2759,
        "config": "lr=5e-6, ep=5, warm-start+polar=0.55",
    },
}


def load_metrics(tag: str) -> dict | None:
    """실험 태그에서 golden_eval_metrics*.json 로드."""
    ckpt_dir = MODEL_ROOT / tag
    if not ckpt_dir.exists():
        return None

    candidates = sorted(ckpt_dir.glob("golden_eval_metrics*.json"), reverse=True)
    if not candidates:
        return None

    with open(candidates[0]) as f:
        return json.load(f)


def extract_key_metrics(data: dict) -> dict:
    """JSON에서 핵심 비교 지표 추출."""
    metrics = {}

    det = data.get("detection", {})
    metrics["det_precision"] = det.get("precision")
    metrics["det_recall"] = det.get("recall")
    metrics["det_f1"] = det.get("f1")
    metrics["det_f05"] = det.get("f0.5") or det.get("f05")

    sent = data.get("mentioned_sentiment", {})
    metrics["mentioned_sentiment_f1"] = sent.get("macro_f1")

    # neutral recall — neutral_analysis.recall
    neutral_info = data.get("neutral_analysis", {})
    metrics["neutral_recall"] = neutral_info.get("recall")

    # 사용감/성능 분포 — distribution (list)
    dist = data.get("distribution", [])
    if isinstance(dist, list):
        for asp in dist:
            if asp.get("aspect") == "사용감/성능":
                metrics["usage_pred_pct"] = asp.get("pred_pct")
                metrics["usage_gt_pct"] = asp.get("gt_pct")
                metrics["usage_diff_pp"] = asp.get("diff_pp")
                break

    return metrics


def fmt(val, fmt_str=".4f", go_val=None) -> str:
    if val is None:
        return "  -   "
    s = f"{val:{fmt_str}}"
    if go_val is not None:
        marker = "O" if val >= go_val else "X"
        s = f"{s} {marker}"
    return s


def main():
    parser = argparse.ArgumentParser(description="Sweep 결과 비교")
    parser.add_argument("--include-baselines", action="store_true")
    args = parser.parse_args()

    results = {}

    # 베이스라인
    if args.include_baselines:
        for name, bl in BASELINES.items():
            results[name] = bl

    # Warm-start sweep
    ws_found = 0
    for tag in WARMSTART_TAGS:
        data = load_metrics(tag)
        if data is None:
            continue
        ws_found += 1
        metrics = extract_key_metrics(data)
        tag_parts = tag.replace("ws_", "").split("_")
        lr_part = [p for p in tag_parts if p.startswith("lr")]
        ep_part = [p for p in tag_parts if p.startswith("ep")]
        metrics["config"] = (
            f"lr={lr_part[0][2:] if lr_part else '?'}, "
            f"ep={ep_part[0][2:] if ep_part else '?'}, warm-start"
        )
        results[f"WS:{tag.replace('ws_', '')}"] = metrics

    # Phase 2 sweep
    p2_found = 0
    for tag in PHASE2_TAGS:
        data = load_metrics(tag)
        if data is None:
            continue
        p2_found += 1
        metrics = extract_key_metrics(data)
        tag_parts = tag.replace("p2_", "").split("_")
        lr_part = [p for p in tag_parts if p.startswith("lr")]
        ep_part = [p for p in tag_parts if p.startswith("ep")]
        metrics["config"] = (
            f"lr={lr_part[0][2:] if lr_part else '?'}, "
            f"ep={ep_part[0][2:] if ep_part else '?'}, phase2(freeze)"
        )
        results[f"P2:{tag.replace('p2_', '')}"] = metrics

    # Phase 3 (Focal Loss) sweep
    p3_found = 0
    for tag in PHASE3_TAGS:
        data = load_metrics(tag)
        if data is None:
            continue
        p3_found += 1
        metrics = extract_key_metrics(data)
        # 태그 파싱: p3_g2.0_lr1e5_ep5 → gamma=2.0, lr=1e5, ep=5
        tag_clean = tag.replace("p3_", "")
        parts = tag_clean.split("_")
        gamma_part = [p for p in parts if p.startswith("g")]
        lr_part = [p for p in parts if p.startswith("lr")]
        ep_part = [p for p in parts if p.startswith("ep")]
        metrics["config"] = (
            f"g={gamma_part[0][1:] if gamma_part else '?'}, "
            f"lr={lr_part[0][2:] if lr_part else '?'}, "
            f"ep={ep_part[0][2:] if ep_part else '?'}, focal"
        )
        results[f"P3:{tag_clean}"] = metrics

    if not results:
        print("결과가 없습니다. 먼저 run_sweep.sh 또는 run_sweep_phase2.sh를 실행하세요.")
        return

    print(f"\n  [로드] Warm-start: {ws_found}/{len(WARMSTART_TAGS)}, "
          f"Phase 2: {p2_found}/{len(PHASE2_TAGS)}, "
          f"Phase 3: {p3_found}/{len(PHASE3_TAGS)}")

    # ── 테이블 출력 ──
    names = list(results.keys())
    col_width = max(len(n) for n in names) + 2

    header_metrics = [
        ("Det.P", "det_precision", GO_CRITERIA["det_precision"]),
        ("Det.R", "det_recall", None),
        ("Det.F1", "det_f1", GO_CRITERIA["det_f1"]),
        ("Sent.F1", "mentioned_sentiment_f1", GO_CRITERIA["mentioned_sentiment_f1"]),
        ("Neu.R", "neutral_recall", GO_CRITERIA["neutral_recall"]),
    ]

    print()
    print("=" * 90)
    print("  ABSA Stage 5 — Sweep 결과 비교 (Warm-start + Phase 2 + Phase 3 Focal)")
    print("=" * 90)
    print()

    header = f"{'실험':<{col_width}}"
    for label, _, _ in header_metrics:
        header += f" {label:>12}"
    header += f"  {'GO':>5}"
    header += f"  {'설정'}"
    print(header)
    print("-" * (len(header) + 30))

    criteria_row = f"{'[GO 기준]':<{col_width}}"
    for label, key, go_val in header_metrics:
        if go_val is not None:
            criteria_row += f" {f'>={go_val:.2f}':>12}"
        else:
            criteria_row += f" {'':>12}"
    print(criteria_row)
    print("-" * (len(header) + 30))

    # 그룹별 구분선
    prev_group = None
    for name in names:
        # 그룹 구분
        if name.startswith("WS:") and prev_group != "WS":
            if prev_group is not None:
                print("-" * (len(header) + 30))
            prev_group = "WS"
        elif name.startswith("P2:") and prev_group != "P2":
            print("-" * (len(header) + 30))
            prev_group = "P2"
        elif name.startswith("P3:") and prev_group != "P3":
            print("-" * (len(header) + 30))
            prev_group = "P3"

        m = results[name]
        row = f"{name:<{col_width}}"

        go_count = 0
        go_total = 0
        for label, key, go_val in header_metrics:
            val = m.get(key)
            row += f" {fmt(val, '.4f', go_val):>12}"
            if go_val is not None:
                go_total += 1
                if val is not None and val >= go_val:
                    go_count += 1

        row += f"  {go_count}/{go_total}"
        row += f"  {m.get('config', '')}"
        print(row)

    print()

    # ── 사용감/성능 분포 ──
    has_usage = any(results[n].get("usage_gt_pct") for n in names)
    if has_usage:
        print("사용감/성능 분포 비교:")
        print(f"{'실험':<{col_width}} {'GT%':>8} {'Pred%':>8} {'±pp':>8}")
        print("-" * (col_width + 30))
        for name in names:
            m = results[name]
            gt = m.get("usage_gt_pct")
            pred = m.get("usage_pred_pct")
            diff = m.get("usage_diff_pp")
            gt_s = f"{gt:.1f}" if gt is not None else "-"
            pred_s = f"{pred:.1f}" if pred is not None else "-"
            diff_s = f"{diff:+.1f}" if diff is not None else "-"
            print(f"{name:<{col_width}} {gt_s:>8} {pred_s:>8} {diff_s:>8}")
        print()

    print("O = GO 기준 충족, X = 미달")
    print()


if __name__ == "__main__":
    main()
