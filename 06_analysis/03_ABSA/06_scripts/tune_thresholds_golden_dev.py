"""
Golden Dev 618건 기반 Threshold 재튜닝 — Stage 4B Warm-start 모델용

3단계 튜닝:
  Step 1: Per-aspect none-threshold sweep (F0.5 최적화, 후처리 없이)
  Step 2: Polar threshold sweep (후처리 포함, mentioned sentiment F1 최적화)
  Step 3: 최종 검증 (full postprocess + 모든 지표 출력)

사용법:
  python tune_thresholds_golden_dev.py --checkpoint-dir checkpoints_stage4b
  python tune_thresholds_golden_dev.py --checkpoint-dir checkpoints_stage4b --run-test
"""
import sys
import json
import argparse
from pathlib import Path

ABSA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ABSA_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score

from RQ_absa.s1_config import (
    TRAIN_CONFIG, ASPECT_LABELS, CHECKPOINT_DIR, PROCESSED_DATA_DIR, MODEL_ROOT,
)
from RQ_absa.s4_dataset import create_dataset_from_wide
from RQ_absa.s5_model import MultiTaskABSAModel
from RQ_absa.s7_evaluation import (
    apply_none_thresholds, apply_thresholds_with_polar, collect_predictions,
    apply_full_postprocess,
)

NUM_ASPECTS = len(ASPECT_LABELS)


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _compute_fbeta(labels, preds, beta=0.5):
    p = precision_score(labels, preds, zero_division=0)
    r = recall_score(labels, preds, zero_division=0)
    denom = (beta**2 * p + r)
    if denom == 0:
        return 0.0
    return (1 + beta**2) * p * r / denom


def load_model(ckpt_dir, device):
    """best_model.pt 로드"""
    ckpt_path = ckpt_dir / "best_model.pt"
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = MultiTaskABSAModel(model_name=TRAIN_CONFIG["model_name"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    print(f"모델 로드: {ckpt_path}")
    if "epoch" in checkpoint:
        print(f"  Best epoch: {checkpoint['epoch']}")
    return model


def run_inference_raw(model, loader, device):
    """추론 → raw probs + labels (threshold 미적용)"""
    results = collect_predictions(loader, model, device)
    return results


# ────────────────────────────────────────────────
# Step 1: Per-aspect none-threshold sweep
# ────────────────────────────────────────────────
def sweep_aspect_thresholds(aspect_probs, aspect_labels, aspect_masks=None,
                            grid_min=0.03, grid_max=0.60, grid_step=0.01):
    """각 aspect별로 독립적으로 none-threshold를 sweep, F0.5 최적화"""
    candidates = np.arange(grid_min, grid_max + grid_step / 2, grid_step)
    best_thresholds = np.full(NUM_ASPECTS, 0.5)
    per_aspect_results = []

    print("\n" + "=" * 78)
    print("  Step 1: Per-aspect None-Threshold Sweep (F0.5, 후처리 없이)")
    print("=" * 78)
    print(f"  Grid: {grid_min:.2f} ~ {grid_max:.2f}, step={grid_step:.2f}")
    print(f"  {'Aspect':12s}  {'Best_t':>6s}  {'F0.5':>7s}  {'Prec':>7s}  {'Rec':>7s}  {'Det_F1':>7s}")
    print(f"  {'─'*12}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")

    for j in range(NUM_ASPECTS):
        name = ASPECT_LABELS[j]

        # mask=1인 샘플만
        if aspect_masks is not None:
            m = aspect_masks[:, j].astype(bool)
            true_labels = aspect_labels[:, j][m]
            probs_j = aspect_probs[:, j][m]
        else:
            true_labels = aspect_labels[:, j]
            probs_j = aspect_probs[:, j]

        best_score = -1.0
        best_t = 0.5
        best_detail = {}

        for t in candidates:
            p_none = probs_j[:, 0]
            is_none = p_none >= t
            non_none_probs = probs_j[:, 1:]
            best_non_none = np.argmax(non_none_probs, axis=-1) + 1
            preds_j = np.where(is_none, 0, best_non_none)

            binary_preds = (preds_j > 0).astype(int)
            binary_labels = (true_labels > 0).astype(int)

            f05 = _compute_fbeta(binary_labels, binary_preds, beta=0.5)
            prec = precision_score(binary_labels, binary_preds, zero_division=0)
            rec = recall_score(binary_labels, binary_preds, zero_division=0)
            det_f1 = f1_score(binary_labels, binary_preds, zero_division=0)

            if f05 > best_score:
                best_score = f05
                best_t = t
                best_detail = {"f0.5": f05, "prec": prec, "rec": rec, "det_f1": det_f1}

        best_thresholds[j] = best_t
        per_aspect_results.append({
            "aspect": name, "threshold": round(best_t, 3), **{k: round(v, 4) for k, v in best_detail.items()},
        })
        print(f"  {name:12s}  {best_t:6.2f}  {best_detail['f0.5']:7.4f}  "
              f"{best_detail['prec']:7.4f}  {best_detail['rec']:7.4f}  {best_detail['det_f1']:7.4f}")

    print("=" * 78)
    return best_thresholds, per_aspect_results


# ────────────────────────────────────────────────
# Step 2: Polar threshold sweep (with postprocess)
# ────────────────────────────────────────────────
def sweep_polar_threshold(aspect_probs, aspect_labels, aspect_masks,
                          texts, ratings, aspect_thresholds,
                          polar_min=0.30, polar_max=0.80, polar_step=0.05):
    """Polar threshold sweep — mentioned sentiment macro F1 + neutral recall 최적화"""
    candidates = np.arange(polar_min, polar_max + polar_step / 2, polar_step)

    print("\n" + "=" * 78)
    print("  Step 2: Polar Threshold Sweep (후처리 포함, Mentioned Sentiment F1)")
    print("=" * 78)
    print(f"  Grid: {polar_min:.2f} ~ {polar_max:.2f}, step={polar_step:.2f}")
    print(f"  {'polar':>6s}  {'Sent_F1':>8s}  {'Neu_Rec':>8s}  {'Det_P':>8s}  {'Det_F1':>8s}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    results = []
    for polar_t in candidates:
        # threshold 적용
        preds = apply_thresholds_with_polar(aspect_probs, aspect_thresholds, polar_t)
        # 후처리
        preds = apply_full_postprocess(texts, preds, ratings, aspect_probs=aspect_probs, use_design_rule=True)

        # detection metrics (전체)
        bin_p = (preds > 0).astype(int)
        bin_l = (aspect_labels > 0).astype(int)
        if aspect_masks is not None:
            m_flat = aspect_masks.flatten().astype(bool)
            fp = bin_p.flatten()[m_flat]
            fl = bin_l.flatten()[m_flat]
        else:
            fp = bin_p.flatten()
            fl = bin_l.flatten()
        det_prec = precision_score(fl, fp, zero_division=0)
        det_f1 = f1_score(fl, fp, zero_division=0)

        # mentioned sentiment F1 (GT > 0 셀)
        if aspect_masks is not None:
            mentioned = (aspect_labels > 0) & (aspect_masks.astype(bool))
        else:
            mentioned = (aspect_labels > 0)
        fp_sent = preds[mentioned]
        fl_sent = aspect_labels[mentioned]
        sent_f1 = f1_score(fl_sent, fp_sent, average="macro", zero_division=0)

        # neutral recall (GT=neutral 셀에서 pred=neutral 비율)
        gt_neu_mask = (fl_sent == 2)
        n_gt_neu = gt_neu_mask.sum()
        if n_gt_neu > 0:
            pred_on_gt_neu = fp_sent[gt_neu_mask]
            neu_rec = (pred_on_gt_neu == 2).sum() / n_gt_neu
        else:
            neu_rec = 0.0

        results.append({
            "polar": round(polar_t, 2), "sent_f1": round(sent_f1, 4),
            "neu_rec": round(float(neu_rec), 4), "det_prec": round(det_prec, 4),
            "det_f1": round(det_f1, 4),
        })
        print(f"  {polar_t:6.2f}  {sent_f1:8.4f}  {float(neu_rec):8.4f}  {det_prec:8.4f}  {det_f1:8.4f}")

    # best by sent_f1
    best = max(results, key=lambda x: x["sent_f1"])
    print(f"\n  ★ Best by Sent F1: polar={best['polar']:.2f}  "
          f"SentF1={best['sent_f1']:.4f}  NeuRec={best['neu_rec']:.4f}  "
          f"DetP={best['det_prec']:.4f}")
    print("=" * 78)

    return best["polar"], results


# ────────────────────────────────────────────────
# Step 3: 최종 검증
# ────────────────────────────────────────────────
def final_validation(aspect_probs, aspect_labels, aspect_masks,
                     texts, ratings, aspect_thresholds, polar_threshold):
    """최종 검증: 모든 GO/NO-GO 지표 계산"""
    preds = apply_thresholds_with_polar(aspect_probs, aspect_thresholds, polar_threshold)
    preds = apply_full_postprocess(texts, preds, ratings, aspect_probs=aspect_probs, use_design_rule=True)

    # Detection
    bin_p = (preds > 0).astype(int)
    bin_l = (aspect_labels > 0).astype(int)
    if aspect_masks is not None:
        m_flat = aspect_masks.flatten().astype(bool)
        fp = bin_p.flatten()[m_flat]
        fl = bin_l.flatten()[m_flat]
    else:
        fp = bin_p.flatten()
        fl = bin_l.flatten()
    det_prec = precision_score(fl, fp, zero_division=0)
    det_rec = recall_score(fl, fp, zero_division=0)
    det_f1 = f1_score(fl, fp, zero_division=0)
    det_f05 = _compute_fbeta(fl, fp, beta=0.5)

    # Mentioned sentiment F1
    if aspect_masks is not None:
        mentioned = (aspect_labels > 0) & (aspect_masks.astype(bool))
    else:
        mentioned = (aspect_labels > 0)
    fp_sent = preds[mentioned]
    fl_sent = aspect_labels[mentioned]
    sent_macro_f1 = f1_score(fl_sent, fp_sent, average="macro", zero_division=0)
    sent_weighted_f1 = f1_score(fl_sent, fp_sent, average="weighted", zero_division=0)

    # Neutral recall
    gt_neu_mask = (fl_sent == 2)
    n_gt_neu = gt_neu_mask.sum()
    if n_gt_neu > 0:
        pred_on_gt_neu = fp_sent[gt_neu_mask]
        neu_rec = (pred_on_gt_neu == 2).sum() / n_gt_neu
    else:
        neu_rec = 0.0

    # Per-aspect detection
    per_aspect_det = {}
    for j, name in enumerate(ASPECT_LABELS):
        bp = bin_p[:, j]
        bl = bin_l[:, j]
        if aspect_masks is not None:
            m = aspect_masks[:, j].astype(bool)
            bp = bp[m]
            bl = bl[m]
        per_aspect_det[name] = {
            "precision": round(precision_score(bl, bp, zero_division=0), 4),
            "recall": round(recall_score(bl, bp, zero_division=0), 4),
            "f1": round(f1_score(bl, bp, zero_division=0), 4),
            "f0.5": round(_compute_fbeta(bl, bp, beta=0.5), 4),
        }

    print("\n" + "=" * 78)
    print("  Step 3: 최종 검증 (Full Postprocess)")
    print("=" * 78)
    print(f"  Aspect Thresholds: {aspect_thresholds.round(2).tolist()}")
    print(f"  Polar Threshold:   {polar_threshold}")
    print()
    print(f"  ★ Detection Precision:         {det_prec:.4f}  {'GO' if det_prec >= 0.65 else 'NO-GO'}  (목표 ≥0.65)")
    print(f"    Detection Recall:            {det_rec:.4f}")
    print(f"    Detection F1:                {det_f1:.4f}  {'GO' if det_f1 >= 0.60 else 'NO-GO'}  (목표 ≥0.60)")
    print(f"    Detection F0.5:              {det_f05:.4f}")
    print(f"    Mentioned Sentiment F1:      {sent_macro_f1:.4f}  {'GO' if sent_macro_f1 >= 0.43 else 'NO-GO'}  (목표 ≥0.43)")
    print(f"    Neutral Recall:              {float(neu_rec):.4f}  {'GO' if neu_rec >= 0.25 else 'NO-GO'}  (목표 ≥0.25)")

    print(f"\n  Per-aspect Detection:")
    print(f"  {'Aspect':12s}  {'Prec':>7s}  {'Rec':>7s}  {'F1':>7s}  {'F0.5':>7s}")
    print(f"  {'─'*12}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
    for name in ASPECT_LABELS:
        v = per_aspect_det[name]
        print(f"  {name:12s}  {v['precision']:7.4f}  {v['recall']:7.4f}  {v['f1']:7.4f}  {v['f0.5']:7.4f}")
    print("=" * 78)

    return {
        "detection": {
            "precision": round(det_prec, 4), "recall": round(det_rec, 4),
            "f1": round(det_f1, 4), "f0.5": round(det_f05, 4),
            "per_aspect": per_aspect_det,
        },
        "mentioned_sentiment": {
            "macro_f1": round(sent_macro_f1, 4), "weighted_f1": round(sent_weighted_f1, 4),
        },
        "neutral_recall": round(float(neu_rec), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Golden Dev Threshold 재튜닝")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="체크포인트 디렉토리 (기본: checkpoints_stage4b)")
    parser.add_argument("--run-test", action="store_true",
                        help="튜닝 후 Golden Test 평가 자동 실행")
    parser.add_argument("--grid-step", type=float, default=0.01,
                        help="Aspect threshold grid search step (기본: 0.01)")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # 체크포인트 디렉토리
    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
    else:
        ckpt_dir = MODEL_ROOT / "checkpoints_stage4b"
    print(f"Checkpoint dir: {ckpt_dir}")

    # 모델 로드
    model = load_model(ckpt_dir, device)

    # Golden Dev 로드
    golden_dir = PROCESSED_DATA_DIR / "golden" / "v1"
    golden_dev_csv = golden_dir / "golden_dev.csv"
    assert golden_dev_csv.exists(), f"Golden Dev 없음: {golden_dev_csv}"
    print(f"Golden Dev: {golden_dev_csv}")

    cfg = TRAIN_CONFIG
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    dev_dataset = create_dataset_from_wide(golden_dev_csv, tokenizer, cfg["max_length"])
    dev_loader = DataLoader(dev_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    # 추론 (1회만)
    print("\n추론 중...")
    results = run_inference_raw(model, dev_loader, device)
    aspect_probs = results["aspect_probs"]   # [N, 8, 4]
    aspect_labels = results["aspect_labels"]  # [N, 8]
    aspect_masks = results.get("aspect_masks")  # [N, 8]

    # 텍스트 + 별점 로드 (후처리용)
    golden_df = pd.read_csv(golden_dev_csv)
    texts = golden_df["text"].astype(str).tolist()
    ratings = golden_df["rating"].values if "rating" in golden_df.columns else None

    print(f"Golden Dev: {len(texts)}건")

    # ── Step 1: Per-aspect threshold sweep ──
    best_thresholds, per_aspect_results = sweep_aspect_thresholds(
        aspect_probs, aspect_labels, aspect_masks,
        grid_min=0.03, grid_max=0.60, grid_step=args.grid_step,
    )

    # ── Step 2: Polar threshold sweep ──
    best_polar, polar_results = sweep_polar_threshold(
        aspect_probs, aspect_labels, aspect_masks,
        texts, ratings, best_thresholds,
        polar_min=0.30, polar_max=0.80, polar_step=0.05,
    )

    # ── Step 3: 최종 검증 ──
    validation = final_validation(
        aspect_probs, aspect_labels, aspect_masks,
        texts, ratings, best_thresholds, best_polar,
    )

    # ── Threshold 저장 ──
    threshold_data = {
        "thresholds": best_thresholds.tolist(),
        "per_aspect_results": per_aspect_results,
        "polar_threshold": best_polar,
        "stage": "4b",
        "tuning_source": "golden_dev",
        "tuning_metric": "F0.5 (detection)",
        "tuning_grid_step": args.grid_step,
        "validation_on_dev": validation,
    }

    # 기존 threshold 백업
    existing_path = ckpt_dir / "none_thresholds.json"
    backup_path = ckpt_dir / "none_thresholds_stage3a_restored.json"
    if existing_path.exists() and not backup_path.exists():
        import shutil
        shutil.copy2(existing_path, backup_path)
        print(f"\n기존 threshold 백업: {backup_path.name}")

    # 새 threshold 저장
    with open(existing_path, "w") as f:
        json.dump(threshold_data, f, ensure_ascii=False, indent=2)
    print(f"새 threshold 저장: {existing_path}")

    # 튜닝 상세 로그도 별도 저장
    log_data = {
        "aspect_thresholds": best_thresholds.tolist(),
        "polar_threshold": best_polar,
        "per_aspect_sweep": per_aspect_results,
        "polar_sweep": polar_results,
        "validation": validation,
    }
    log_path = ckpt_dir / "threshold_tuning_golden_dev_log.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    print(f"튜닝 로그 저장: {log_path}")

    # ── Stage 3A-v2와 비교 ──
    stage3a_thresholds = [0.15, 0.1, 0.03, 0.2, 0.5, 0.15, 0.15, 0.1]
    print("\n" + "=" * 78)
    print("  Threshold 비교: Stage 3A-v2 vs Stage 4B 재튜닝")
    print("=" * 78)
    print(f"  {'Aspect':12s}  {'3A-v2':>6s}  {'4B new':>6s}  {'변화':>10s}")
    print(f"  {'─'*12}  {'─'*6}  {'─'*6}  {'─'*10}")
    for j, name in enumerate(ASPECT_LABELS):
        old = stage3a_thresholds[j]
        new = best_thresholds[j]
        diff = new - old
        arrow = "→" if abs(diff) < 0.005 else ("↑" if diff > 0 else "↓")
        print(f"  {name:12s}  {old:6.2f}  {new:6.2f}  {arrow} {diff:+.2f}")
    print(f"  {'polar':12s}  {0.50:6.2f}  {best_polar:6.2f}")
    print("=" * 78)

    # ── Golden Test 평가 자동 실행 (선택) ──
    if args.run_test:
        print("\n" + "=" * 78)
        print("  Golden Test 최종 평가 (Stage 4B + 재튜닝 threshold)")
        print("=" * 78)

        import subprocess
        eval_script = ABSA_ROOT / "06_scripts" / "eval_golden.py"
        python = sys.executable
        cmd = [
            python, str(eval_script),
            "--split", "test",
            "--polar",
            "--postprocess",
            "--checkpoint-dir", str(ckpt_dir),
        ]
        print(f"  실행: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print(f"\n  Golden Test 평가 실패 (returncode={result.returncode})")
        else:
            print("\n  Golden Test 평가 완료!")


if __name__ == "__main__":
    main()
