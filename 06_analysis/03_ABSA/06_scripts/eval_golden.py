"""
골든셋 성능 리포트 (Stage 2: 8 aspects, dev/test 분리)

출력 지표:
  (1) Aspect Detection: Precision(#1 KPI) / Recall / F1 / F0.5
      + aspect별 Precision / Recall / F0.5
  (2) 분포 Sanity: aspect별 GT 언급률 vs Pred 언급률 (±pp 차이)
  (3) Mentioned Sentiment F1 (GT>0 셀만) + neutral 복원 분석
  (4) Aspect별 4-class Confusion Matrix (약점 분석)

인자:
  --split dev|test|full (기본: test)
  --polar  polar_threshold 적용 여부
"""
import sys
import json
from pathlib import Path

ABSA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ABSA_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import (
    f1_score, precision_score, recall_score, fbeta_score,
    accuracy_score, classification_report, confusion_matrix
)

from RQ_absa.s1_config import (
    TRAIN_CONFIG, ASPECT_LABELS, ASPECT_SENTIMENT_LABELS,
    SENTIMENT_LABELS, CHECKPOINT_DIR, PROCESSED_DATA_DIR,
)
from RQ_absa.s4_dataset import create_dataset_from_wide
from RQ_absa.s5_model import MultiTaskABSAModel
from RQ_absa.s7_evaluation import (
    apply_none_thresholds, apply_thresholds_with_polar, collect_predictions,
    apply_design_rule_override,
)

NUM_ASPECTS = len(ASPECT_LABELS)


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_checkpoint_and_thresholds(device):
    """best_model.pt 로드 + none_thresholds + polar_threshold 추출"""
    ckpt_path = CHECKPOINT_DIR / "best_model.pt"
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    model = MultiTaskABSAModel(model_name=TRAIN_CONFIG["model_name"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded model from: {ckpt_path}")
    if "epoch" in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")

    # threshold 로드 (JSON 우선: Stage 3A 튜닝 반영)
    polar_threshold = None
    thresholds = None
    thresholds_path = CHECKPOINT_DIR / "none_thresholds.json"
    if thresholds_path.exists():
        with open(thresholds_path) as f:
            data = json.load(f)
        thresholds = np.array(data["thresholds"])
        polar_threshold = data.get("polar_threshold")
        stage = data.get("stage", "unknown")
        print(f"Loaded none_thresholds from JSON (stage={stage}): {thresholds.round(2).tolist()}")
        if polar_threshold is not None:
            print(f"Loaded polar_threshold: {polar_threshold}")
    else:
        # JSON 없으면 checkpoint fallback
        thresholds = checkpoint.get("none_thresholds")
        if thresholds is not None:
            thresholds = np.array(thresholds)
            print(f"Loaded none_thresholds from checkpoint: {thresholds.round(2).tolist()}")

    return model, thresholds, polar_threshold


def run_inference(model, golden_loader, device, thresholds, polar_threshold=None):
    """골든셋 추론 → predictions 수집"""
    results = collect_predictions(golden_loader, model, device)

    if polar_threshold is not None and thresholds is not None:
        aspect_preds = apply_thresholds_with_polar(
            results["aspect_probs"], thresholds, polar_threshold
        )
        print(f"Using polar_threshold={polar_threshold}")
    elif thresholds is not None:
        aspect_preds = apply_none_thresholds(results["aspect_probs"], thresholds)
    else:
        aspect_preds = np.argmax(results["aspect_probs"], axis=-1)

    return {
        "sentiment_preds": results["sentiment_preds"],
        "sentiment_labels": results["sentiment_labels"],
        "aspect_preds": aspect_preds,
        "aspect_probs": results["aspect_probs"],
        "aspect_labels": results["aspect_labels"],
        "aspect_masks": results.get("aspect_masks"),
    }


def _compute_fbeta(labels, preds, beta=0.5):
    """F-beta score (binary) 계산"""
    p = precision_score(labels, preds, zero_division=0)
    r = recall_score(labels, preds, zero_division=0)
    denom = (beta**2 * p + r)
    if denom == 0:
        return 0.0
    return (1 + beta**2) * p * r / denom


# ─────────────────────────────────────────────
# (1) Aspect Detection: P / R / F1 / F0.5
# ─────────────────────────────────────────────
def eval_aspect_detection(preds, labels, masks=None):
    """none(0) vs non-none(1~3) 이진 분류"""
    bin_p = (preds > 0).astype(int)
    bin_l = (labels > 0).astype(int)
    results = {}

    # 전체 (mask 적용)
    if masks is not None:
        m_flat = masks.flatten().astype(bool)
        fp = bin_p.flatten()[m_flat]
        fl = bin_l.flatten()[m_flat]
    else:
        fp = bin_p.flatten()
        fl = bin_l.flatten()

    prec = precision_score(fl, fp, zero_division=0)
    rec = recall_score(fl, fp, zero_division=0)
    results["전체"] = {
        "precision": prec,
        "recall": rec,
        "f1": f1_score(fl, fp, zero_division=0),
        "f0.5": _compute_fbeta(fl, fp, beta=0.5),
        "accuracy": accuracy_score(fl, fp),
    }

    # aspect별 Precision / Recall / F1 / F0.5
    per_aspect = {}
    for i, name in enumerate(ASPECT_LABELS):
        bp = bin_p[:, i]
        bl = bin_l[:, i]
        if masks is not None:
            m = masks[:, i].astype(bool)
            bp = bp[m]
            bl = bl[m]
        ap = precision_score(bl, bp, zero_division=0)
        ar = recall_score(bl, bp, zero_division=0)
        per_aspect[name] = {
            "precision": ap,
            "recall": ar,
            "f1": f1_score(bl, bp, zero_division=0),
            "f0.5": _compute_fbeta(bl, bp, beta=0.5),
        }
    results["per_aspect"] = per_aspect
    return results


# ─────────────────────────────────────────────
# (2) 분포 Sanity: GT vs Pred 언급률 + ±pp 차이
# ─────────────────────────────────────────────
def eval_distribution_sanity(preds, labels, masks=None):
    """aspect 언급률(non-none) 및 pos/neu/neg 비율 비교"""
    rows = []
    for i, name in enumerate(ASPECT_LABELS):
        p = preds[:, i]
        l = labels[:, i]
        if masks is not None:
            m = masks[:, i].astype(bool)
            p = p[m]
            l = l[m]
        n = len(l)
        gt_mentioned = int((l > 0).sum())
        pred_mentioned = int((p > 0).sum())
        gt_pct = gt_mentioned / n * 100 if n else 0
        pred_pct = pred_mentioned / n * 100 if n else 0

        gt_dist = {v: int((l == v).sum()) for v in [0, 1, 2, 3]}
        pred_dist = {v: int((p == v).sum()) for v in [0, 1, 2, 3]}

        rows.append({
            "aspect": name,
            "n": n,
            "gt_mention": gt_mentioned,
            "gt_pct": round(gt_pct, 1),
            "pred_mention": pred_mentioned,
            "pred_pct": round(pred_pct, 1),
            "diff_pp": round(pred_pct - gt_pct, 1),
            "gt_none": gt_dist[0], "gt_pos": gt_dist[1],
            "gt_neu": gt_dist[2], "gt_neg": gt_dist[3],
            "pred_none": pred_dist[0], "pred_pos": pred_dist[1],
            "pred_neu": pred_dist[2], "pred_neg": pred_dist[3],
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# (3) Mentioned Sentiment F1 + neutral 분석
# ─────────────────────────────────────────────
def eval_mentioned_sentiment(preds, labels, masks=None):
    """GT label > 0 (pos/neu/neg)인 셀만 대상"""
    results = {}

    p_all = preds
    l_all = labels
    if masks is not None:
        m_all = masks
    else:
        m_all = np.ones_like(l_all)

    # GT > 0 AND mask=1
    mentioned = (l_all > 0) & (m_all.astype(bool))
    fp = p_all[mentioned]
    fl = l_all[mentioned]

    report = classification_report(
        fl, fp,
        labels=[0, 1, 2, 3],
        target_names=["none(miss)", "positive", "neutral", "negative"],
        zero_division=0,
        output_dict=True,
    )

    results["전체"] = {
        "n_cells": int(mentioned.sum()),
        "macro_f1": f1_score(fl, fp, average="macro", zero_division=0),
        "weighted_f1": f1_score(fl, fp, average="weighted", zero_division=0),
        "report": report,
    }

    # neutral 복원 분석 — GT=neutral인 셀에서 pred 분포
    gt_neu_mask = (fl == 2)
    n_gt_neu = int(gt_neu_mask.sum())
    if n_gt_neu > 0:
        pred_on_gt_neu = fp[gt_neu_mask]
        results["neutral_analysis"] = {
            "gt_neutral_total": n_gt_neu,
            "pred_none": int((pred_on_gt_neu == 0).sum()),
            "pred_positive": int((pred_on_gt_neu == 1).sum()),
            "pred_neutral": int((pred_on_gt_neu == 2).sum()),
            "pred_negative": int((pred_on_gt_neu == 3).sum()),
            "recall": int((pred_on_gt_neu == 2).sum()) / n_gt_neu,
        }
    else:
        results["neutral_analysis"] = {"gt_neutral_total": 0}

    # 전체 pred에서 neutral 예측 수 (polar 효과 확인)
    total_pred_neu = int((preds == 2).sum())
    total_gt_neu = int((labels == 2).sum())
    results["neutral_overall"] = {
        "gt_neutral_cells": total_gt_neu,
        "pred_neutral_cells": total_pred_neu,
    }

    # aspect별 mentioned sentiment F1
    per_aspect = {}
    for i, name in enumerate(ASPECT_LABELS):
        col_p = preds[:, i]
        col_l = labels[:, i]
        if masks is not None:
            col_m = masks[:, i].astype(bool)
        else:
            col_m = np.ones(len(col_l), dtype=bool)
        sel = (col_l > 0) & col_m
        if sel.sum() == 0:
            per_aspect[name] = {"n": 0, "f1": 0.0}
        else:
            asp_f1 = f1_score(col_l[sel], col_p[sel], average="macro", zero_division=0)
            per_aspect[name] = {"n": int(sel.sum()), "f1": asp_f1}
    results["per_aspect"] = per_aspect
    return results


# ─────────────────────────────────────────────
# (4) Aspect별 4-class Confusion Matrix
# ─────────────────────────────────────────────
def eval_per_aspect_detail(preds, labels, masks=None):
    """각 aspect별 4-class confusion matrix + F1"""
    details = {}
    for i, name in enumerate(ASPECT_LABELS):
        p = preds[:, i]
        l = labels[:, i]
        if masks is not None:
            m = masks[:, i].astype(bool)
            p = p[m]
            l = l[m]

        details[name] = {
            "n": len(l),
            "f1_macro": f1_score(l, p, average="macro", zero_division=0),
            "f1_weighted": f1_score(l, p, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(l, p, labels=[0, 1, 2, 3]),
        }
    return details


# ─────────────────────────────────────────────
# 리포트 출력
# ─────────────────────────────────────────────
def print_report(det_results, sent_results, aspect_details, dist_df,
                 split_name="test", polar_used=False, output_path=None):
    """전체 리포트 출력 + 파일 저장"""
    lines = []

    def p(s=""):
        lines.append(s)
        print(s)

    p("=" * 78)
    p(f"  ABSA Stage 2 — 골든셋 성능 리포트 ({split_name})")
    p(f"  {NUM_ASPECTS} aspects | polar={'ON' if polar_used else 'OFF'}")
    p("=" * 78)

    # ── (1) Detection: P / R / F1 / F0.5 ──
    p("\n" + "─" * 78)
    p("  (1) Aspect Detection — Precision (#1 KPI) / Recall / F1 / F0.5")
    p("─" * 78)

    d = det_results["전체"]
    p(f"\n  ★ Precision: {d['precision']:.4f}    ← #1 KPI (목표: 0.65+)")
    p(f"    Recall:    {d['recall']:.4f}")
    p(f"    F1:        {d['f1']:.4f}")
    p(f"    F0.5:      {d['f0.5']:.4f}    ← precision 가중 종합")
    p(f"    Accuracy:  {d['accuracy']:.4f}")

    pa = det_results["per_aspect"]
    p(f"\n  {'Aspect':12s}  Precision   Recall     F1      F0.5")
    p(f"  {'─'*12}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}")
    for name in ASPECT_LABELS:
        v = pa[name]
        prec_marker = " ◀" if v["precision"] < 0.5 else ""
        p(f"  {name:12s}  {v['precision']:7.4f}   {v['recall']:7.4f}   {v['f1']:7.4f}   {v['f0.5']:7.4f}{prec_marker}")

    # ── (2) 분포 Sanity ──
    p("\n" + "─" * 78)
    p("  (2) 분포 Sanity — GT 언급률 vs Pred 언급률 (±pp)")
    p("─" * 78)

    p(f"\n  {'Aspect':12s}  GT언급%  Pred언급%   ±pp    | GT(n/p/u/ng)         Pred(n/p/u/ng)")
    p(f"  {'─'*12}  {'─'*7}  {'─'*8}  {'─'*6}  | {'─'*20}  {'─'*20}")
    for _, row in dist_df.iterrows():
        diff = row["diff_pp"]
        sign = "+" if diff >= 0 else ""
        marker = " ⚠" if abs(diff) > 10 else ""
        gt_str = f"{row['gt_none']:4d}/{row['gt_pos']:3d}/{row['gt_neu']:3d}/{row['gt_neg']:3d}"
        pred_str = f"{row['pred_none']:4d}/{row['pred_pos']:3d}/{row['pred_neu']:3d}/{row['pred_neg']:3d}"
        p(f"  {row['aspect']:12s}  {row['gt_pct']:5.1f}%   {row['pred_pct']:6.1f}%  {sign}{diff:5.1f}pp | {gt_str}  {pred_str}{marker}")

    # ── (3) Mentioned Sentiment + neutral 분석 ──
    p("\n" + "─" * 78)
    p("  (3) Mentioned Sentiment F1 (GT label > 0) + neutral 복원 분석")
    p("─" * 78)

    s = sent_results["전체"]
    p(f"\n  대상: {s['n_cells']}셀 (GT에서 실제 언급된 aspect)")
    p(f"  Macro F1:    {s['macro_f1']:.4f}")
    p(f"  Weighted F1: {s['weighted_f1']:.4f}")

    rpt = s["report"]
    p(f"\n  {'class':12s}  precision  recall  f1-score  support")
    p(f"  {'─'*12}  {'─'*9}  {'─'*6}  {'─'*8}  {'─'*7}")
    for cls in ["none(miss)", "positive", "neutral", "negative"]:
        if cls in rpt:
            r = rpt[cls]
            p(f"  {cls:12s}  {r['precision']:7.4f}   {r['recall']:.4f}  {r['f1-score']:8.4f}  {int(r['support']):7d}")

    # neutral 복원 분석
    neu = sent_results["neutral_analysis"]
    p(f"\n  ── neutral 복원 분석 {'(polar ON)' if polar_used else '(polar OFF)'} ──")
    if neu["gt_neutral_total"] > 0:
        p(f"  GT neutral 셀: {neu['gt_neutral_total']}건")
        p(f"    → pred none(놓침):     {neu['pred_none']:3d}건")
        p(f"    → pred positive:       {neu['pred_positive']:3d}건")
        p(f"    → pred neutral(정답):  {neu['pred_neutral']:3d}건  ← recall={neu['recall']:.4f}")
        p(f"    → pred negative:       {neu['pred_negative']:3d}건")
    else:
        p(f"  GT neutral 셀: 0건 (해당 split에 neutral 없음)")

    n_ov = sent_results["neutral_overall"]
    p(f"\n  전체 neutral 분포 (8 aspects × {dist_df['n'].iloc[0] if len(dist_df) > 0 else '?'}행):")
    p(f"    GT neutral 셀:   {n_ov['gt_neutral_cells']:5d}")
    p(f"    Pred neutral 셀: {n_ov['pred_neutral_cells']:5d}")

    # aspect별 mentioned sentiment
    spa = sent_results["per_aspect"]
    p(f"\n  Aspect별 Mentioned Sentiment F1:")
    for name in ASPECT_LABELS:
        v = spa[name]
        marker = " ⚠" if v["n"] > 0 and v["f1"] < 0.4 else ""
        p(f"    {name:12s}: F1={v['f1']:.4f} (n={v['n']}){marker}")

    # ── (4) Confusion Matrix (약점) ──
    p("\n" + "─" * 78)
    p("  (4) Aspect별 Confusion Matrix (약점 분석)")
    p("─" * 78)

    sorted_aspects = sorted(aspect_details.items(), key=lambda x: x[1]["f1_macro"])
    p(f"\n  {'Aspect':12s}  F1(4cls)  F1(wgt)  n")
    p(f"  {'─'*12}  {'─'*8}  {'─'*7}  {'─'*5}")
    for name, d in sorted_aspects:
        marker = " ⚠" if d["f1_macro"] < 0.5 else ""
        p(f"  {name:12s}  {d['f1_macro']:.4f}    {d['f1_weighted']:.4f}   {d['n']}{marker}")

    # 약점 + 디자인은 항상 표시
    weak_aspects = {name for name, d in aspect_details.items() if d["f1_macro"] < 0.5}
    weak_aspects.add("디자인")

    labels_4cls = ["none", "pos", "neu", "neg"]
    for name in ASPECT_LABELS:
        if name not in weak_aspects:
            continue
        d = aspect_details[name]
        cm = d["confusion_matrix"]
        p(f"\n  [{name}] F1(4cls)={d['f1_macro']:.4f}")
        p(f"  {'':8s} pred_none  pred_pos  pred_neu  pred_neg")
        for r, row_label in enumerate(labels_4cls):
            vals = "  ".join(f"{cm[r, c]:8d}" for c in range(4))
            p(f"  {('GT_'+row_label):8s} {vals}")

    # ── Summary ──
    p("\n" + "=" * 78)
    p("  SUMMARY")
    p("=" * 78)
    d_all = det_results["전체"]
    s_all = sent_results["전체"]
    p(f"  ★ Detection Precision (#1 KPI):    {d_all['precision']:.4f}")
    p(f"    Detection Recall:                {d_all['recall']:.4f}")
    p(f"    Detection F1:                    {d_all['f1']:.4f}")
    p(f"    Detection F0.5:                  {d_all['f0.5']:.4f}")
    p(f"    Mentioned Sentiment Macro F1:    {s_all['macro_f1']:.4f}")
    p(f"    Neutral recall (GT>0 셀):        {neu.get('recall', 0):.4f}")

    weak_list = [name for name, d in sorted_aspects if d["f1_macro"] < 0.5]
    if weak_list:
        p(f"    약점 Aspect (F1 < 0.5):          {', '.join(weak_list)}")

    # precision 기준 약점
    low_prec = [name for name in ASPECT_LABELS if pa[name]["precision"] < 0.5]
    if low_prec:
        p(f"    과다검출 Aspect (Prec < 0.5):    {', '.join(low_prec)}")
    p("=" * 78)

    if output_path:
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
        print(f"\n리포트 저장: {output_path}")


def save_metrics_json(det_results, sent_results, aspect_details, dist_df, output_path):
    """JSON으로 수치 저장"""
    data = {
        "detection": {},
        "mentioned_sentiment": {},
        "neutral_analysis": sent_results.get("neutral_analysis", {}),
        "neutral_overall": sent_results.get("neutral_overall", {}),
        "per_aspect_f1": {},
        "distribution": dist_df.to_dict(orient="records"),
    }

    d = det_results["전체"]
    data["detection"] = {k: round(v, 4) for k, v in d.items()}

    pa = det_results["per_aspect"]
    data["detection"]["per_aspect"] = {
        k: {kk: round(vv, 4) for kk, vv in v.items()}
        for k, v in pa.items()
    }

    s = sent_results["전체"]
    data["mentioned_sentiment"] = {
        "n_cells": s["n_cells"],
        "macro_f1": round(s["macro_f1"], 4),
        "weighted_f1": round(s["weighted_f1"], 4),
    }
    spa = sent_results["per_aspect"]
    data["mentioned_sentiment"]["per_aspect"] = {
        k: {"n": v["n"], "f1": round(v["f1"], 4)} for k, v in spa.items()
    }

    # neutral recall
    neu = sent_results.get("neutral_analysis", {})
    if "recall" in neu:
        data["neutral_analysis"]["recall"] = round(neu["recall"], 4)

    for name, d in aspect_details.items():
        data["per_aspect_f1"][name] = {
            "f1_macro": round(d["f1_macro"], 4),
            "f1_weighted": round(d["f1_weighted"], 4),
            "n": d["n"],
            "confusion_matrix": d["confusion_matrix"].tolist(),
        }

    with open(output_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"메트릭 JSON 저장: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="골든셋 평가 (Stage 2)")
    parser.add_argument("--split", default="test", choices=["dev", "test", "full"],
                        help="평가 대상: dev, test, full")
    parser.add_argument("--polar", action="store_true",
                        help="polar_threshold 적용 여부")
    parser.add_argument("--design-rule", action="store_true",
                        help="디자인 aspect를 규칙 기반으로 override")
    parser.add_argument("--polar-value", type=float, default=None,
                        help="polar_threshold 직접 지정 (예: 0.70)")
    parser.add_argument("--usage-threshold", type=float, default=None,
                        help="사용감/성능 threshold 직접 지정 (예: 0.25)")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # 모델 + threshold 로드
    model, thresholds, polar_threshold = load_checkpoint_and_thresholds(device)

    if not args.polar:
        polar_threshold = None

    # polar_threshold 직접 지정
    if args.polar_value is not None:
        polar_threshold = args.polar_value
        print(f"Polar threshold override: {polar_threshold}")

    # 사용감/성능 threshold 직접 지정
    if args.usage_threshold is not None and thresholds is not None:
        usage_idx = ASPECT_LABELS.index("사용감/성능")
        old_val = thresholds[usage_idx]
        thresholds[usage_idx] = args.usage_threshold
        print(f"사용감/성능 threshold override: {old_val:.2f} → {args.usage_threshold:.2f}")

    # 골든셋 로드 (dev/test/full)
    golden_dir = PROCESSED_DATA_DIR / "final" / "golden"
    if args.split == "dev":
        golden_csv = golden_dir / "golden_dev.csv"
    elif args.split == "test":
        golden_csv = golden_dir / "golden_test.csv"
    else:
        golden_csv = golden_dir / "golden_set_wide_full_eval.csv"

    assert golden_csv.exists(), f"골든셋 파일 없음: {golden_csv}"
    print(f"골든셋: {golden_csv}")

    cfg = TRAIN_CONFIG
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    golden_dataset = create_dataset_from_wide(golden_csv, tokenizer, cfg["max_length"])
    golden_loader = DataLoader(golden_dataset, batch_size=cfg["batch_size"],
                               shuffle=False, num_workers=0)

    # 추론
    results = run_inference(model, golden_loader, device, thresholds, polar_threshold)
    preds = results["aspect_preds"]
    labels = results["aspect_labels"]
    masks = results["aspect_masks"]

    # 디자인 규칙 기반 override
    if args.design_rule:
        golden_df = pd.read_csv(golden_csv)
        texts = golden_df["text"].astype(str).tolist()
        print(f"\n디자인 규칙 기반 override 적용 (n={len(texts)})...")
        preds = apply_design_rule_override(texts, preds)

    # 평가
    det_results = eval_aspect_detection(preds, labels, masks)
    dist_df = eval_distribution_sanity(preds, labels, masks)
    sent_results = eval_mentioned_sentiment(preds, labels, masks)
    aspect_details = eval_per_aspect_detail(preds, labels, masks)

    # 리포트 출력 + 저장
    suffix = f"_{args.split}"
    if args.polar or args.polar_value is not None:
        suffix += "_polar"
    if args.design_rule:
        suffix += "_drule"
    if args.usage_threshold is not None:
        suffix += f"_ut{args.usage_threshold:.2f}"
    report_path = CHECKPOINT_DIR / f"golden_eval_report{suffix}.txt"
    json_path = CHECKPOINT_DIR / f"golden_eval_metrics{suffix}.json"

    print_report(det_results, sent_results, aspect_details, dist_df,
                 args.split, polar_used=(polar_threshold is not None),
                 output_path=report_path)
    save_metrics_json(det_results, sent_results, aspect_details, dist_df, json_path)


if __name__ == "__main__":
    main()
