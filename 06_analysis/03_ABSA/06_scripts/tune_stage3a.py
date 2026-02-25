"""
Stage 3A — golden_dev에서 후처리/threshold 튜닝

실행 순서:
  1. 디자인 규칙 기반 override 효과 측정
  2. 사용감/성능 threshold grid search (0.15~0.40)
  3. polar_threshold grid search (0.55~0.80)
  4. 최적 threshold 저장 → none_thresholds_stage3a.json

사용법:
  python tune_stage3a.py
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
from sklearn.metrics import f1_score, precision_score, recall_score

from RQ_absa.s1_config import (
    TRAIN_CONFIG, ASPECT_LABELS, CHECKPOINT_DIR, PROCESSED_DATA_DIR,
)
from RQ_absa.s4_dataset import create_dataset_from_wide
from RQ_absa.s5_model import MultiTaskABSAModel
from RQ_absa.s7_evaluation import (
    apply_none_thresholds, apply_thresholds_with_polar,
    collect_predictions, apply_design_rule_override,
)

NUM_ASPECTS = len(ASPECT_LABELS)
USAGE_IDX = ASPECT_LABELS.index("사용감/성능")


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def compute_fbeta(labels, preds, beta=0.5):
    p = precision_score(labels, preds, zero_division=0)
    r = recall_score(labels, preds, zero_division=0)
    d = beta**2 * p + r
    return (1 + beta**2) * p * r / d if d > 0 else 0.0


def compute_detection_metrics(preds, labels, masks=None):
    """Detection precision/recall/f1/f0.5 계산"""
    bp = (preds > 0).astype(int)
    bl = (labels > 0).astype(int)
    if masks is not None:
        m = masks.flatten().astype(bool)
        fp = bp.flatten()[m]
        fl = bl.flatten()[m]
    else:
        fp = bp.flatten()
        fl = bl.flatten()
    prec = precision_score(fl, fp, zero_division=0)
    rec = recall_score(fl, fp, zero_division=0)
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1_score(fl, fp, zero_division=0),
        "f0.5": compute_fbeta(fl, fp, beta=0.5),
    }


def compute_mentioned_f1(preds, labels, masks=None):
    """GT>0 셀만 대상 sentiment macro F1"""
    m_all = masks if masks is not None else np.ones_like(labels)
    mentioned = (labels > 0) & (m_all.astype(bool))
    if mentioned.sum() == 0:
        return 0.0
    return f1_score(labels[mentioned], preds[mentioned], average="macro", zero_division=0)


def compute_usage_detection(preds, labels, masks=None):
    """사용감/성능 aspect의 detection 메트릭"""
    bp = (preds[:, USAGE_IDX] > 0).astype(int)
    bl = (labels[:, USAGE_IDX] > 0).astype(int)
    if masks is not None:
        m = masks[:, USAGE_IDX].astype(bool)
        bp = bp[m]
        bl = bl[m]
    n = len(bl)
    pred_mention = int(bp.sum())
    gt_mention = int(bl.sum())
    prec = precision_score(bl, bp, zero_division=0)
    rec = recall_score(bl, bp, zero_division=0)
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1_score(bl, bp, zero_division=0),
        "pred_pct": pred_mention / n * 100 if n else 0,
        "gt_pct": gt_mention / n * 100 if n else 0,
        "diff_pp": (pred_mention - gt_mention) / n * 100 if n else 0,
    }


def main():
    device = get_device()
    print(f"Device: {device}")

    # ── 모델 + threshold 로드 ──
    ckpt_path = CHECKPOINT_DIR / "best_model.pt"
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = MultiTaskABSAModel(model_name=TRAIN_CONFIG["model_name"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    thresholds_path = CHECKPOINT_DIR / "none_thresholds.json"
    with open(thresholds_path) as f:
        th_data = json.load(f)
    base_thresholds = np.array(th_data["thresholds"])
    print(f"Base thresholds: {base_thresholds.round(2).tolist()}")

    # ── golden_dev 로드 ──
    golden_csv = PROCESSED_DATA_DIR / "final" / "golden" / "golden_dev.csv"
    print(f"Golden dev: {golden_csv}")
    golden_df = pd.read_csv(golden_csv)
    texts = golden_df["text"].astype(str).tolist()
    print(f"  Rows: {len(golden_df)}")

    cfg = TRAIN_CONFIG
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    golden_dataset = create_dataset_from_wide(golden_csv, tokenizer, cfg["max_length"])
    golden_loader = DataLoader(golden_dataset, batch_size=cfg["batch_size"],
                               shuffle=False, num_workers=0)

    # ── 추론 (raw probs 수집) ──
    results = collect_predictions(golden_loader, model, device)
    aspect_probs = results["aspect_probs"]
    aspect_labels = results["aspect_labels"]
    aspect_masks = results.get("aspect_masks")

    # ════════════════════════════════════════════
    # Phase 1: 디자인 규칙 override 효과 측정
    # ════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Phase 1: 디자인 규칙 override 효과")
    print("=" * 70)

    # Baseline: 현재 threshold + polar=0.55
    preds_baseline = apply_thresholds_with_polar(
        aspect_probs, base_thresholds, polar_threshold=0.55
    )
    det_baseline = compute_detection_metrics(preds_baseline, aspect_labels, aspect_masks)
    print(f"\n  [Baseline] polar=0.55")
    print(f"    Precision={det_baseline['precision']:.4f}  Recall={det_baseline['recall']:.4f}  "
          f"F1={det_baseline['f1']:.4f}  F0.5={det_baseline['f0.5']:.4f}")

    # + Design rule
    preds_drule = apply_design_rule_override(texts, preds_baseline)
    det_drule = compute_detection_metrics(preds_drule, aspect_labels, aspect_masks)
    print(f"\n  [+Design rule]")
    print(f"    Precision={det_drule['precision']:.4f}  Recall={det_drule['recall']:.4f}  "
          f"F1={det_drule['f1']:.4f}  F0.5={det_drule['f0.5']:.4f}")
    print(f"    Precision 변화: {det_baseline['precision']:.4f} → {det_drule['precision']:.4f} "
          f"({det_drule['precision'] - det_baseline['precision']:+.4f})")

    # ════════════════════════════════════════════
    # Phase 2: 사용감/성능 threshold grid search
    # ════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Phase 2: 사용감/성능 threshold grid search")
    print("=" * 70)

    usage_candidates = np.arange(0.10, 0.45, 0.05)
    best_usage_t = base_thresholds[USAGE_IDX]
    best_usage_score = -1.0
    usage_results = []

    print(f"\n  {'threshold':>10s}  {'Prec':>7s}  {'Recall':>7s}  {'F1':>7s}  {'F0.5':>7s}  "
          f"{'사용감P':>7s}  {'사용감±pp':>8s}")
    print(f"  {'─'*10}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}")

    for t in usage_candidates:
        th = base_thresholds.copy()
        th[USAGE_IDX] = t
        preds = apply_thresholds_with_polar(aspect_probs, th, polar_threshold=0.55)
        preds = apply_design_rule_override(texts, preds)

        det = compute_detection_metrics(preds, aspect_labels, aspect_masks)
        usage_det = compute_usage_detection(preds, aspect_labels, aspect_masks)

        # F0.5로 최적화 (precision 가중)
        score = det["f0.5"]
        if score > best_usage_score:
            best_usage_score = score
            best_usage_t = t

        usage_results.append({"threshold": t, **det, "usage": usage_det})
        print(f"  {t:>10.2f}  {det['precision']:>7.4f}  {det['recall']:>7.4f}  "
              f"{det['f1']:>7.4f}  {det['f0.5']:>7.4f}  "
              f"{usage_det['precision']:>7.4f}  {usage_det['diff_pp']:>+7.1f}pp")

    print(f"\n  ★ Best 사용감/성능 threshold: {best_usage_t:.2f} (F0.5={best_usage_score:.4f})")

    # ════════════════════════════════════════════
    # Phase 3: polar_threshold grid search
    # ════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Phase 3: polar_threshold grid search")
    print("=" * 70)

    # 사용감/성능 threshold를 best로 고정
    opt_thresholds = base_thresholds.copy()
    opt_thresholds[USAGE_IDX] = best_usage_t

    polar_candidates = np.arange(0.50, 0.85, 0.05)
    best_polar = 0.55
    best_polar_score = -1.0
    polar_results = []

    # polar OFF (baseline comparison)
    preds_no_polar = apply_none_thresholds(aspect_probs, opt_thresholds)
    preds_no_polar = apply_design_rule_override(texts, preds_no_polar)
    det_no_polar = compute_detection_metrics(preds_no_polar, aspect_labels, aspect_masks)
    mf1_no_polar = compute_mentioned_f1(preds_no_polar, aspect_labels, aspect_masks)

    print(f"\n  [polar OFF] Prec={det_no_polar['precision']:.4f}  F1={det_no_polar['f1']:.4f}  "
          f"MentionedF1={mf1_no_polar:.4f}")

    print(f"\n  {'polar':>7s}  {'Prec':>7s}  {'Recall':>7s}  {'F1':>7s}  {'F0.5':>7s}  "
          f"{'MentF1':>7s}  {'NeuRecall':>9s}")
    print(f"  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*9}")

    for pt in polar_candidates:
        preds = apply_thresholds_with_polar(aspect_probs, opt_thresholds, polar_threshold=pt)
        preds = apply_design_rule_override(texts, preds)

        det = compute_detection_metrics(preds, aspect_labels, aspect_masks)
        mf1 = compute_mentioned_f1(preds, aspect_labels, aspect_masks)

        # neutral recall 계산
        m_all = aspect_masks if aspect_masks is not None else np.ones_like(aspect_labels)
        mentioned = (aspect_labels > 0) & (m_all.astype(bool))
        fl = aspect_labels[mentioned]
        fp = preds[mentioned]
        gt_neu_mask = (fl == 2)
        neu_recall = 0.0
        if gt_neu_mask.sum() > 0:
            neu_recall = (fp[gt_neu_mask] == 2).sum() / gt_neu_mask.sum()

        # Mentioned F1 최적화 (neutral 복원의 핵심 목표)
        score = mf1
        if score > best_polar_score:
            best_polar_score = score
            best_polar = pt

        polar_results.append({"polar": pt, **det, "mentioned_f1": mf1, "neu_recall": neu_recall})
        print(f"  {pt:>7.2f}  {det['precision']:>7.4f}  {det['recall']:>7.4f}  "
              f"{det['f1']:>7.4f}  {det['f0.5']:>7.4f}  "
              f"{mf1:>7.4f}  {neu_recall:>9.4f}")

    print(f"\n  ★ Best polar_threshold: {best_polar:.2f} (MentionedF1={best_polar_score:.4f})")

    # ════════════════════════════════════════════
    # 최종 조합 평가 (golden_dev)
    # ════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("최종 조합 평가 (golden_dev)")
    print("=" * 70)

    final_thresholds = opt_thresholds.copy()
    preds_final = apply_thresholds_with_polar(
        aspect_probs, final_thresholds, polar_threshold=best_polar
    )
    preds_final = apply_design_rule_override(texts, preds_final)

    det_final = compute_detection_metrics(preds_final, aspect_labels, aspect_masks)
    mf1_final = compute_mentioned_f1(preds_final, aspect_labels, aspect_masks)

    print(f"\n  사용감/성능 threshold: {best_usage_t:.2f} (기존: {base_thresholds[USAGE_IDX]:.2f})")
    print(f"  polar_threshold:      {best_polar:.2f} (기존: 0.55)")
    print(f"  디자인:               규칙 기반 override")
    print(f"\n  Detection Precision:      {det_final['precision']:.4f}")
    print(f"  Detection Recall:         {det_final['recall']:.4f}")
    print(f"  Detection F1:             {det_final['f1']:.4f}")
    print(f"  Detection F0.5:           {det_final['f0.5']:.4f}")
    print(f"  Mentioned Sentiment F1:   {mf1_final:.4f}")

    # Aspect별 분포 sanity
    print(f"\n  Aspect별 Detection:")
    for i, name in enumerate(ASPECT_LABELS):
        bp = (preds_final[:, i] > 0).astype(int)
        bl = (aspect_labels[:, i] > 0).astype(int)
        if aspect_masks is not None:
            m = aspect_masks[:, i].astype(bool)
            bp = bp[m]
            bl = bl[m]
        n = len(bl)
        gt_pct = bl.sum() / n * 100 if n else 0
        pred_pct = bp.sum() / n * 100 if n else 0
        prec = precision_score(bl, bp, zero_division=0)
        rec = recall_score(bl, bp, zero_division=0)
        print(f"    {name:12s}  P={prec:.3f}  R={rec:.3f}  "
              f"GT={gt_pct:5.1f}%  Pred={pred_pct:5.1f}%  ±{pred_pct-gt_pct:+.1f}pp")

    # ════════════════════════════════════════════
    # 저장
    # ════════════════════════════════════════════
    output = {
        "thresholds": final_thresholds.tolist(),
        "per_aspect_results": [
            {"aspect": name, "threshold": round(final_thresholds[i], 3)}
            for i, name in enumerate(ASPECT_LABELS)
        ],
        "polar_threshold": best_polar,
        "design_rule": True,
        "tuning_source": "golden_dev",
        "stage": "3a",
        "detection_dev": {k: round(v, 4) for k, v in det_final.items()},
        "mentioned_f1_dev": round(mf1_final, 4),
    }

    out_path = CHECKPOINT_DIR / "none_thresholds_stage3a.json"
    with open(out_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out_path}")

    # 기존 파일도 업데이트 (백업 후)
    import shutil
    backup_path = CHECKPOINT_DIR / "none_thresholds_stage2_backup.json"
    if thresholds_path.exists() and not backup_path.exists():
        shutil.copy2(thresholds_path, backup_path)
        print(f"Stage 2 threshold 백업: {backup_path}")

    with open(thresholds_path, "w") as f:
        json.dump({
            "thresholds": final_thresholds.tolist(),
            "per_aspect_results": output["per_aspect_results"],
            "polar_threshold": best_polar,
            "design_rule": True,
            "default_f1": th_data.get("default_f1"),
            "tuned_f1": th_data.get("tuned_f1"),
        }, f, ensure_ascii=False, indent=2)
    print(f"none_thresholds.json 업데이트 완료")

    print("\n" + "=" * 70)
    print("다음 단계: golden_test 평가")
    print(f"  python eval_golden.py --split test --polar --design-rule")
    print("=" * 70)


if __name__ == "__main__":
    main()
