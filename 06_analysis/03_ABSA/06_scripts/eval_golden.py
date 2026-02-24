"""
골든셋 성능 리포트 (Stage 2: 10 aspects, dev/test 분리)

출력 지표:
  (1) Aspect Detection F1 (none vs non-none) — 10 aspects
  (2) Aspect Sentiment F1 (pos/neu/neg) — GT label>0 "언급된 셀만"
  (3) Aspect별 F1 + Confusion Matrix (특히 디자인, CS/응대)
  (4) 예측 분포 sanity: aspect 언급률, pos/neu/neg 비율

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
    f1_score, precision_score, recall_score,
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
)

# Stage 2: 미분류 없이 10개 aspect만 사용
NUM_ASPECTS = len(ASPECT_LABELS)


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_checkpoint_and_thresholds(device):
    """best_model.pt 로드 + none_thresholds + polar_threshold 추출"""
    from RQ_absa.s5_model import MultiTaskABSAModel

    ckpt_path = CHECKPOINT_DIR / "best_model.pt"
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    # 모델 생성 (class weight 없이) + strict=False로 로드
    model = MultiTaskABSAModel(model_name=TRAIN_CONFIG["model_name"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded model from: {ckpt_path}")
    if "epoch" in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if "val_metrics" in checkpoint:
        print(f"  Val metrics: {checkpoint['val_metrics']}")

    # threshold 로드
    polar_threshold = None
    thresholds = checkpoint.get("none_thresholds")
    if thresholds is not None:
        thresholds = np.array(thresholds)
        print(f"Loaded none_thresholds from checkpoint: {thresholds.round(2).tolist()}")
    else:
        thresholds_path = CHECKPOINT_DIR / "none_thresholds.json"
        if thresholds_path.exists():
            with open(thresholds_path) as f:
                data = json.load(f)
            thresholds = np.array(data["thresholds"])
            polar_threshold = data.get("polar_threshold")
            print(f"Loaded none_thresholds from JSON: {thresholds.round(2).tolist()}")
            if polar_threshold is not None:
                print(f"Loaded polar_threshold: {polar_threshold}")

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


# ─────────────────────────────────────────────
# (1) Aspect Detection F1
# ─────────────────────────────────────────────
def eval_aspect_detection(preds, labels, masks=None):
    """none(0) vs non-none(1~3) 이진 분류 — 10 aspects"""
    bin_p = (preds > 0).astype(int)
    bin_l = (labels > 0).astype(int)

    idx_list = list(range(NUM_ASPECTS))
    names = ASPECT_LABELS

    results = {}

    if masks is not None:
        m_sel = masks[:, idx_list].flatten().astype(bool)
        fp = bin_p[:, idx_list].flatten()[m_sel]
        fl = bin_l[:, idx_list].flatten()[m_sel]
    else:
        fp = bin_p[:, idx_list].flatten()
        fl = bin_l[:, idx_list].flatten()

    det = {
        "precision": precision_score(fl, fp, zero_division=0),
        "recall": recall_score(fl, fp, zero_division=0),
        "f1": f1_score(fl, fp, zero_division=0),
        "accuracy": accuracy_score(fl, fp),
    }
    results["전체"] = det

    # aspect별 detection F1
    per_aspect = {}
    for i, name in zip(idx_list, names):
        bp = bin_p[:, i]
        bl = bin_l[:, i]
        if masks is not None:
            m = masks[:, i].astype(bool)
            bp = bp[m]
            bl = bl[m]
        per_aspect[name] = f1_score(bl, bp, zero_division=0)
    results["전체_per_aspect"] = per_aspect

    return results


# ─────────────────────────────────────────────
# (2) Aspect Sentiment F1 — "언급된 셀만" (GT label > 0)
# ─────────────────────────────────────────────
def eval_mentioned_sentiment(preds, labels, masks=None):
    """GT label > 0 (pos/neu/neg)인 셀만 대상 — sentiment만 평가
    이때 pred가 none(0)이면 오분류로 처리 (그대로 포함)
    10 aspects 전용"""
    results = {}
    idx_list = list(range(NUM_ASPECTS))
    names = ASPECT_LABELS

    p_sel = preds[:, idx_list]
    l_sel = labels[:, idx_list]

    if masks is not None:
        m_sel = masks[:, idx_list]
    else:
        m_sel = np.ones_like(l_sel)

    # GT > 0 AND mask=1
    mentioned = (l_sel > 0) & (m_sel.astype(bool))
    fp = p_sel[mentioned]
    fl = l_sel[mentioned]

    macro_f1 = f1_score(fl, fp, average="macro", zero_division=0)
    weighted_f1 = f1_score(fl, fp, average="weighted", zero_division=0)

    per_class = {}
    for cls_id, cls_name in [(1, "positive"), (2, "neutral"), (3, "negative")]:
        cls_mask = fl == cls_id
        if cls_mask.sum() > 0:
            cls_f1 = f1_score(fl == cls_id, fp == cls_id, zero_division=0)
            per_class[cls_name] = cls_f1
        else:
            per_class[cls_name] = 0.0

    report = classification_report(
        fl, fp,
        labels=[1, 2, 3],
        target_names=["positive", "neutral", "negative"],
        zero_division=0,
        output_dict=True
    )

    results["전체"] = {
        "n_cells": int(mentioned.sum()),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "report": report,
    }

    # aspect별 mentioned sentiment F1
    per_aspect = {}
    for i, name in zip(idx_list, names):
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
    results["전체_per_aspect"] = per_aspect

    return results


# ─────────────────────────────────────────────
# (3) Aspect별 F1 + Confusion Matrix
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

        f1_4cls = f1_score(l, p, average="macro", zero_division=0)
        f1_weighted = f1_score(l, p, average="weighted", zero_division=0)
        cm = confusion_matrix(l, p, labels=[0, 1, 2, 3])

        details[name] = {
            "n": len(l),
            "f1_macro": f1_4cls,
            "f1_weighted": f1_weighted,
            "confusion_matrix": cm,
        }
    return details


# ─────────────────────────────────────────────
# (4) 예측 분포 sanity check
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

        gt_dist = {v: int((l == v).sum()) for v in [0, 1, 2, 3]}
        pred_dist = {v: int((p == v).sum()) for v in [0, 1, 2, 3]}

        rows.append({
            "aspect": name,
            "n": n,
            "gt_mention": gt_mentioned,
            "gt_mention%": round(gt_mentioned / n * 100, 1) if n else 0,
            "pred_mention": pred_mentioned,
            "pred_mention%": round(pred_mentioned / n * 100, 1) if n else 0,
            "gt_none": gt_dist[0], "gt_pos": gt_dist[1], "gt_neu": gt_dist[2], "gt_neg": gt_dist[3],
            "pred_none": pred_dist[0], "pred_pos": pred_dist[1], "pred_neu": pred_dist[2], "pred_neg": pred_dist[3],
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 리포트 출력
# ─────────────────────────────────────────────
def print_report(det_results, sent_results, aspect_details, dist_df, split_name="test", output_path=None):
    """전체 리포트 출력 + 파일 저장"""
    lines = []

    def p(s=""):
        lines.append(s)
        print(s)

    p("=" * 72)
    p(f"  ABSA Stage 2 — 골든셋 성능 리포트 ({split_name})")
    p(f"  {NUM_ASPECTS} aspects (미분류 제거)")
    p("=" * 72)

    # ── (1) Aspect Detection ──
    p("\n" + "─" * 72)
    p("  (1) Aspect Detection F1 (none vs non-none)")
    p("─" * 72)

    d = det_results["전체"]
    p(f"\n  Precision: {d['precision']:.4f}")
    p(f"  Recall:    {d['recall']:.4f}")
    p(f"  F1:        {d['f1']:.4f}")
    p(f"  Accuracy:  {d['accuracy']:.4f}")

    pa = det_results["전체_per_aspect"]
    p(f"\n  Aspect별 Detection F1:")
    for name, v in pa.items():
        marker = " ⚠" if v < 0.3 else ""
        p(f"    {name:12s}: {v:.4f}{marker}")

    # ── (2) Mentioned Sentiment ──
    p("\n" + "─" * 72)
    p("  (2) Aspect Sentiment F1 — '언급된 셀만' (GT label > 0)")
    p("─" * 72)

    s = sent_results["전체"]
    p(f"\n  {s['n_cells']}셀")
    p(f"  Macro F1:    {s['macro_f1']:.4f}")
    p(f"  Weighted F1: {s['weighted_f1']:.4f}")

    rpt = s["report"]
    p(f"\n  {'class':12s}  precision  recall  f1-score  support")
    for cls in ["positive", "neutral", "negative"]:
        r = rpt[cls]
        p(f"  {cls:12s}  {r['precision']:.4f}     {r['recall']:.4f}  {r['f1-score']:.4f}    {int(r['support'])}")

    pa = sent_results["전체_per_aspect"]
    p(f"\n  Aspect별 Mentioned Sentiment F1:")
    for name, v in pa.items():
        marker = " ⚠" if v["n"] > 0 and v["f1"] < 0.4 else ""
        p(f"    {name:12s}: F1={v['f1']:.4f} (n={v['n']}){marker}")

    # ── (3) Confusion Matrix ──
    p("\n" + "─" * 72)
    p("  (3) Aspect별 Confusion Matrix (약점 분석)")
    p("─" * 72)

    p(f"\n  {'Aspect':12s}  F1(4cls)  F1(wgt)  n")
    sorted_aspects = sorted(aspect_details.items(), key=lambda x: x[1]["f1_macro"])
    for name, d in sorted_aspects:
        marker = " ⚠" if d["f1_macro"] < 0.5 else ""
        p(f"  {name:12s}  {d['f1_macro']:.4f}    {d['f1_weighted']:.4f}   {d['n']}{marker}")

    weak_aspects = set()
    for name, d in aspect_details.items():
        if d["f1_macro"] < 0.5:
            weak_aspects.add(name)
    weak_aspects.add("디자인")

    labels_4cls = ["none", "pos", "neu", "neg"]
    for name in ASPECT_LABELS:
        if name not in weak_aspects:
            continue
        if name not in aspect_details:
            continue
        d = aspect_details[name]
        cm = d["confusion_matrix"]
        p(f"\n  [{name}] F1(4cls)={d['f1_macro']:.4f}")
        p(f"  {'':8s} pred_none  pred_pos  pred_neu  pred_neg")
        for r, row_label in enumerate(labels_4cls):
            vals = "  ".join(f"{cm[r, c]:8d}" for c in range(4))
            p(f"  {('GT_'+row_label):8s} {vals}")

    # ── (4) 분포 Sanity ──
    p("\n" + "─" * 72)
    p("  (4) 예측 분포 Sanity Check")
    p("─" * 72)

    p(f"\n  {'Aspect':12s}  GT언급%  Pred언급%  | GT(n/p/u/ng)          Pred(n/p/u/ng)")
    for _, row in dist_df.iterrows():
        gt_str = f"{row['gt_none']:4d}/{row['gt_pos']:3d}/{row['gt_neu']:3d}/{row['gt_neg']:3d}"
        pred_str = f"{row['pred_none']:4d}/{row['pred_pos']:3d}/{row['pred_neu']:3d}/{row['pred_neg']:3d}"
        p(f"  {row['aspect']:12s}  {row['gt_mention%']:5.1f}%   {row['pred_mention%']:5.1f}%    | {gt_str}  {pred_str}")

    # ── Summary ──
    p("\n" + "─" * 72)
    p("  Summary")
    p("─" * 72)
    d_all = det_results["전체"]
    s_all = sent_results["전체"]
    p(f"  Detection Precision:                {d_all['precision']:.4f}")
    p(f"  Detection F1:                       {d_all['f1']:.4f}")
    p(f"  Mentioned Sentiment Macro F1:       {s_all['macro_f1']:.4f}")

    weak_list = [name for name, d in sorted_aspects if d["f1_macro"] < 0.5]
    if weak_list:
        p(f"  약점 Aspect (F1 < 0.5):             {', '.join(weak_list)}")
    p("=" * 72)

    if output_path:
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
        print(f"\n리포트 저장: {output_path}")


def save_metrics_json(det_results, sent_results, aspect_details, dist_df, output_path):
    """JSON으로 수치 저장"""
    data = {
        "detection": {},
        "mentioned_sentiment": {},
        "per_aspect_f1": {},
        "distribution": dist_df.to_dict(orient="records"),
    }

    d = det_results["전체"]
    data["detection"] = {k: round(v, 4) for k, v in d.items()}
    pa = det_results["전체_per_aspect"]
    data["detection"]["per_aspect"] = {k: round(v, 4) for k, v in pa.items()}

    s = sent_results["전체"]
    data["mentioned_sentiment"] = {
        "n_cells": s["n_cells"],
        "macro_f1": round(s["macro_f1"], 4),
        "weighted_f1": round(s["weighted_f1"], 4),
    }
    spa = sent_results["전체_per_aspect"]
    data["mentioned_sentiment"]["per_aspect"] = {
        k: {"n": v["n"], "f1": round(v["f1"], 4)} for k, v in spa.items()
    }

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
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # 모델 + threshold 로드
    model, thresholds, polar_threshold = load_checkpoint_and_thresholds(device)

    if not args.polar:
        polar_threshold = None

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
    golden_loader = DataLoader(golden_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    # 추론
    results = run_inference(model, golden_loader, device, thresholds, polar_threshold)
    preds = results["aspect_preds"]
    labels = results["aspect_labels"]
    masks = results["aspect_masks"]

    # 평가
    det_results = eval_aspect_detection(preds, labels, masks)
    sent_results = eval_mentioned_sentiment(preds, labels, masks)
    aspect_details = eval_per_aspect_detail(preds, labels, masks)
    dist_df = eval_distribution_sanity(preds, labels, masks)

    # 리포트 출력 + 저장
    suffix = f"_{args.split}"
    if args.polar:
        suffix += "_polar"
    report_path = CHECKPOINT_DIR / f"golden_eval_report{suffix}.txt"
    json_path = CHECKPOINT_DIR / f"golden_eval_metrics{suffix}.json"
    print_report(det_results, sent_results, aspect_details, dist_df, args.split, report_path)
    save_metrics_json(det_results, sent_results, aspect_details, dist_df, json_path)


if __name__ == "__main__":
    main()
