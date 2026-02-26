"""
ABSA Stage 5 Phase 2 — Encoder Freeze + Sentiment-Only Fine-tuning

전략 B: Detection과 Sentiment 학습 분리
  - Phase 1: Stage 4 from-scratch 모델의 Detection 능력 활용 (이미 완료)
  - Phase 2: encoder를 freeze하고, head만 fine-tuning
    → loss를 non-none 셀(mask=1 AND label != 0)에만 집중
    → Detection 능력 보존 + Sentiment 분류 정확도 집중 개선

핵심 차이 (vs train_stage5_warmstart.py):
  1. encoder 전체 freeze (gradient 계산 안 함 → 메모리/속도 절약)
  2. optimizer: sentiment_classifier + aspect_classifier만 포함
  3. 학습 mask: aspect_mask * (aspect_labels != 0) — none 셀 loss 제외
  4. 평가 mask: 원본 유지 (Detection 성능도 함께 측정)
  5. 소스 모델: Stage 4 best_model.pt (Detection 우수)

sweep 사용법:
  python train_stage5_phase2.py --lr 1e-5 --epochs 5 --tag p2_lr1e5_ep5
  python train_stage5_phase2.py --lr 5e-5 --epochs 3 --tag p2_lr5e5_ep3
"""
import sys
import math
import subprocess
import argparse
from pathlib import Path

ABSA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ABSA_ROOT))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import json

from RQ_absa.s1_config import (
    TRAIN_CONFIG,
    PROCESSED_DATA_DIR,
    MODEL_ROOT,
    THRESHOLD_TUNING_CONFIG,
    ASPECT_LABELS,
    ASPECT_SENTIMENT_LABELS,
)
from RQ_absa.s4_dataset import create_dataset_from_wide
from RQ_absa.s5_model import (
    MultiTaskABSAModel,
    compute_class_weights,
    compute_aspect_class_weights,
)
from RQ_absa.s6_train import create_model_with_class_weights
from RQ_absa.s7_evaluation import (
    ABSAEvaluator,
    collect_predictions,
    tune_none_thresholds,
    apply_none_thresholds,
)


# Stage 4 데이터 + 체크포인트
STAGE4_DATA_DIR = PROCESSED_DATA_DIR / "training" / "stage4"
STAGE4_CHECKPOINT_DIR = MODEL_ROOT / "checkpoints_stage4"


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_stage4_model(model, device):
    """Stage 4 best_model.pt 전체 로드."""
    ckpt_path = STAGE4_CHECKPOINT_DIR / "best_model.pt"
    assert ckpt_path.exists(), f"Stage 4 체크포인트 없음: {ckpt_path}"

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    loaded = len(state_dict) - len(unexpected)
    print(f"Stage 4 모델 로드: {loaded}개 파라미터")
    if missing:
        print(f"  missing: {missing}")
    if unexpected:
        print(f"  unexpected: {unexpected}")
    if "epoch" in ckpt:
        print(f"  Source: epoch {ckpt['epoch']}, best_val={ckpt.get('best_val_metric', 'N/A')}")

    return ckpt


def freeze_encoder(model):
    """Encoder 파라미터를 freeze하고 head만 학습 가능하게 설정."""
    frozen_count = 0
    trainable_count = 0

    # encoder freeze
    for param in model.encoder.parameters():
        param.requires_grad = False
        frozen_count += param.numel()

    # dropout은 학습 모드에서만 동작하므로 별도 처리 불필요

    # head는 학습 가능
    for name in ["sentiment_classifier", "aspect_classifier"]:
        module = getattr(model, name)
        for param in module.parameters():
            param.requires_grad = True
            trainable_count += param.numel()

    total = frozen_count + trainable_count
    print(f"\nEncoder freeze 완료:")
    print(f"  Frozen:    {frozen_count:>12,} ({frozen_count/total*100:.1f}%)")
    print(f"  Trainable: {trainable_count:>12,} ({trainable_count/total*100:.1f}%)")
    print(f"  Total:     {total:>12,}")

    return trainable_count


def compute_sentiment_only_weights(train_dataset):
    """non-none 셀만 대상으로 3-class(pos/neu/neg) 가중치 계산.

    모델은 여전히 4-class 출력이므로, none weight=0으로 설정하여
    none 클래스가 loss에 기여하지 않도록 함.
    """
    aspect_labels = torch.tensor(train_dataset.aspect_labels, dtype=torch.long)
    aspect_masks = (
        torch.tensor(train_dataset.aspect_masks, dtype=torch.float)
        if train_dataset.aspect_masks is not None
        else torch.ones_like(aspect_labels, dtype=torch.float)
    )

    # mask=1 AND label != 0 인 셀만 추출
    mask_flat = aspect_masks.view(-1).bool()
    labels_flat = aspect_labels.view(-1)
    active = mask_flat & (labels_flat != 0)  # non-none + masked
    active_labels = labels_flat[active]

    # 4-class 분포 (none은 0건)
    counts = torch.bincount(active_labels, minlength=4).float()
    total = counts.sum()

    # none weight = 0 (학습에서 제외)
    weights = torch.zeros(4)
    for i in range(1, 4):  # pos, neu, neg
        if counts[i] > 0:
            weights[i] = total / (3.0 * counts[i])  # 3-class balanced

    print("\nSentiment-Only class weights (non-none 셀 기준):")
    for i, (name, count, w) in enumerate(zip(ASPECT_SENTIMENT_LABELS, counts, weights)):
        marker = "(제외)" if i == 0 else ""
        print(f"  {name}: {int(count.item())} samples, weight={w.item():.4f} {marker}")

    return weights


class Phase2Trainer:
    """Phase 2 전용 Trainer.

    ABSATrainer와의 차이:
    1. optimizer에 head 파라미터만 등록
    2. 학습 시 sentiment_mask 사용 (none 셀 제외)
    3. 평가 시 원본 mask 사용 (Detection 성능 측정)
    """

    def __init__(
        self,
        model: MultiTaskABSAModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        sentiment_weight: float = 0.5,
        aspect_weight: float = 1.0,
        device: str = None,
        checkpoint_dir: Path = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm
        self.sentiment_weight = sentiment_weight
        self.aspect_weight = aspect_weight
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device(device or get_device())

        self.model.to(self.device)

        # Head 파라미터만 optimizer에 등록
        head_params = []
        for name in ["sentiment_classifier", "aspect_classifier"]:
            head_params.extend(getattr(model, name).parameters())

        self.optimizer = AdamW(
            head_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.evaluator = ABSAEvaluator()
        self.best_val_metric = 0.0
        self.current_epoch = 0
        self.global_step = 0
        self.training_history = []
        self.none_thresholds = None

    def train(self, num_epochs: int, logging_steps: int = 50, eval_steps: int = 0):
        total_steps = len(self.train_loader) * num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        print(f"\n{'='*60}")
        print(f"PHASE 2 TRAINING START")
        print(f"{'='*60}")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {num_epochs}")
        print(f"  LR: {self.learning_rate}")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Loss: sentiment*{self.sentiment_weight} + aspect*{self.aspect_weight}")
        print(f"  Aspect mask: non-none only (label != 0)")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # ── Train (sentiment-only mask) ──
            train_metrics = self._train_epoch(logging_steps)

            # ── Evaluate (원본 mask — Detection 포함) ──
            val_metrics = self._evaluate(self.val_loader)

            self.training_history.append({
                "epoch": epoch + 1,
                "train": train_metrics,
                "val": val_metrics,
            })

            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} "
                  f"(sent={train_metrics['sentiment_loss']:.4f}, "
                  f"asp={train_metrics['aspect_loss']:.4f})")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Aspect-Sent F1: {val_metrics['aspect_sentiment_f1_macro']:.4f}")
            print(f"  Val Detection F1: {val_metrics['aspect_detection_f1']:.4f}")

            val_metric = val_metrics["aspect_sentiment_f1_macro"]
            if val_metric > self.best_val_metric:
                self.best_val_metric = val_metric
                self._save_checkpoint(is_best=True, val_metrics=val_metrics)
                print(f"  ** New best! Aspect-Sentiment F1: {val_metric:.4f}")

        print(f"\n{'='*60}")
        print(f"PHASE 2 COMPLETE — Best F1: {self.best_val_metric:.4f}")
        print(f"{'='*60}")

        # Threshold 튜닝
        self._load_best_and_tune_thresholds()

        # History 저장
        if self.checkpoint_dir:
            with open(self.checkpoint_dir / "training_history.json", "w") as f:
                json.dump(self.training_history, f, indent=2)

    def _train_epoch(self, logging_steps: int):
        self.model.train()
        # encoder는 freeze 상태이므로 eval 모드로 유지 (dropout/BN 비활성화)
        self.model.encoder.eval()

        total_loss = 0.0
        total_sent_loss = 0.0
        total_asp_loss = 0.0
        n_batches = 0
        n_sentiment_cells = 0  # 실제 학습에 사용된 non-none 셀 수

        progress = tqdm(self.train_loader, desc=f"Phase2 Epoch {self.current_epoch+1}")

        for batch in progress:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            sentiment_labels = batch["sentiment_label"].to(self.device)
            aspect_labels = batch["aspect_label"].to(self.device)      # [B, K]
            aspect_mask = batch["aspect_mask"].to(self.device)         # [B, K]

            # ── 핵심: sentiment-only mask ──
            # mask=1 AND label!=0 인 셀만 loss에 포함
            sentiment_mask = aspect_mask * (aspect_labels != 0).float()

            # Forward (sentiment_mask로 none 셀 제외)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sentiment_labels=sentiment_labels,
                aspect_labels=aspect_labels,
                aspect_mask=sentiment_mask,  # non-none only
            )

            sent_loss = outputs["sentiment_loss"]
            asp_loss = outputs["aspect_loss"]
            loss = self.sentiment_weight * sent_loss + self.aspect_weight * asp_loss

            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.max_grad_norm,
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.global_step += 1

            total_loss += loss.item()
            total_sent_loss += sent_loss.item()
            total_asp_loss += asp_loss.item()
            n_batches += 1
            n_sentiment_cells += sentiment_mask.sum().item()

            progress.set_postfix({
                "loss": total_loss / n_batches,
                "asp": total_asp_loss / n_batches,
                "lr": self.scheduler.get_last_lr()[0],
            })

            if logging_steps > 0 and self.global_step % logging_steps == 0:
                avg_loss = total_loss / n_batches
                avg_cells = n_sentiment_cells / n_batches
                print(f"\n  [Step {self.global_step}] loss={avg_loss:.4f}, "
                      f"avg non-none cells/batch={avg_cells:.1f}")

        return {
            "loss": total_loss / n_batches,
            "sentiment_loss": total_sent_loss / n_batches,
            "aspect_loss": total_asp_loss / n_batches,
            "total_sentiment_cells": int(n_sentiment_cells),
        }

    def _evaluate(self, data_loader: DataLoader):
        """원본 mask로 평가 (Detection + Sentiment 모두 측정)."""
        self.model.eval()

        all_sent_preds, all_sent_labels = [], []
        all_asp_probs, all_asp_labels, all_asp_masks = [], [], []
        total_loss = 0.0
        total_sent_loss = 0.0
        total_asp_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                sentiment_labels = batch["sentiment_label"].to(self.device)
                aspect_labels = batch["aspect_label"].to(self.device)
                aspect_mask = batch["aspect_mask"].to(self.device)

                # 평가는 원본 mask 사용
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    sentiment_labels=sentiment_labels,
                    aspect_labels=aspect_labels,
                    aspect_mask=aspect_mask,
                )

                sent_loss = outputs["sentiment_loss"]
                asp_loss = outputs["aspect_loss"]
                loss = self.sentiment_weight * sent_loss + self.aspect_weight * asp_loss

                total_loss += loss.item()
                total_sent_loss += sent_loss.item()
                total_asp_loss += asp_loss.item()
                n_batches += 1

                sent_preds = torch.argmax(outputs["sentiment_logits"], dim=-1)
                asp_probs = torch.softmax(outputs["aspect_logits"], dim=-1)

                all_sent_preds.extend(sent_preds.cpu().numpy())
                all_sent_labels.extend(sentiment_labels.cpu().numpy())
                all_asp_probs.extend(asp_probs.cpu().numpy())
                all_asp_labels.extend(aspect_labels.cpu().numpy())
                all_asp_masks.extend(aspect_mask.cpu().numpy())

        all_asp_probs = np.array(all_asp_probs)
        all_asp_labels = np.array(all_asp_labels)
        all_asp_masks = np.array(all_asp_masks)

        if self.none_thresholds is not None:
            all_asp_preds = apply_none_thresholds(all_asp_probs, self.none_thresholds)
        else:
            all_asp_preds = np.argmax(all_asp_probs, axis=-1)

        metrics = self.evaluator.evaluate(
            sentiment_preds=np.array(all_sent_preds),
            sentiment_labels=np.array(all_sent_labels),
            aspect_preds=all_asp_preds,
            aspect_labels=all_asp_labels,
            aspect_masks=all_asp_masks,
        )
        metrics["loss"] = total_loss / n_batches
        metrics["sentiment_loss"] = total_sent_loss / n_batches
        metrics["aspect_loss"] = total_asp_loss / n_batches

        return metrics

    def _save_checkpoint(self, is_best=False, val_metrics=None):
        if not self.checkpoint_dir:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        from RQ_absa.s1_config import ASPECT_LABELS, ASPECT_SENTIMENT_TO_ID

        ckpt = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_metric": self.best_val_metric,
            "val_metrics": val_metrics,
            "label_meta": {
                "aspect_labels": ASPECT_LABELS,
                "aspect_sentiment_to_id": ASPECT_SENTIMENT_TO_ID,
            },
            "phase2_config": {
                "source": str(self.source_checkpoint) if hasattr(self, 'source_checkpoint') else "unknown",
                "encoder_frozen": True,
                "sentiment_only_mask": True,
                "learning_rate": self.learning_rate,
            },
        }

        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
        torch.save(ckpt, path)
        print(f"  Saved {'best model' if is_best else 'checkpoint'}: {path}")

    def _load_best_and_tune_thresholds(self):
        if not self.checkpoint_dir:
            return

        best_path = self.checkpoint_dir / "best_model.pt"
        if not best_path.exists():
            return

        ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        print(f"\nBest model 로드 완료: {best_path}")

        # Threshold 튜닝
        print("Threshold tuning (val set)...")
        results = collect_predictions(self.val_loader, self.model, self.device)

        tuning_result = tune_none_thresholds(
            aspect_probs=results["aspect_probs"],
            aspect_labels=results["aspect_labels"],
            aspect_masks=results.get("aspect_masks"),
            search_range=THRESHOLD_TUNING_CONFIG["search_range"],
            search_step=THRESHOLD_TUNING_CONFIG["search_step"],
            metric=THRESHOLD_TUNING_CONFIG["metric"],
            beta=THRESHOLD_TUNING_CONFIG.get("beta", 0.5),
        )

        self.none_thresholds = tuning_result["thresholds"]

        # JSON 저장
        threshold_data = {
            "thresholds": self.none_thresholds.tolist(),
            "per_aspect_results": tuning_result["per_aspect_results"],
            "default_f1": tuning_result["default_f1"],
            "tuned_f1": tuning_result["tuned_f1"],
            "polar_threshold": THRESHOLD_TUNING_CONFIG.get("polar_threshold"),
        }
        threshold_path = self.checkpoint_dir / "none_thresholds.json"
        with open(threshold_path, "w", encoding="utf-8") as f:
            json.dump(threshold_data, f, indent=2, ensure_ascii=False)
        print(f"Thresholds 저장: {threshold_path}")

        # best_model.pt에도 threshold 포함
        ckpt["none_thresholds"] = self.none_thresholds.tolist()
        torch.save(ckpt, best_path)
        print(f"best_model.pt 업데이트 완료")


def main():
    parser = argparse.ArgumentParser(description="ABSA Stage 5 Phase 2 (Encoder Freeze)")
    parser.add_argument("--lr", type=float, default=1e-5, help="학습률 (기본: 1e-5)")
    parser.add_argument("--epochs", type=int, default=5, help="에포크 수 (기본: 5)")
    parser.add_argument("--tag", type=str, default="p2_default", help="실험 태그")
    parser.add_argument(
        "--source-checkpoint", type=str, default=None,
        help="Phase 1 소스 체크포인트 (기본: Stage 4 best_model.pt)",
    )
    parser.add_argument(
        "--sentiment-weight", type=float, default=0.5,
        help="sentiment loss 가중치 (기본: 0.5, head만 학습이므로 줄임)",
    )
    parser.add_argument("--skip-golden-eval", action="store_true")
    args = parser.parse_args()

    cfg = TRAIN_CONFIG.copy()
    device = get_device()
    checkpoint_dir = MODEL_ROOT / args.tag

    print("=" * 60)
    print("ABSA Stage 5 Phase 2 — Encoder Freeze + Sentiment Fine-tuning")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Tag: {args.tag}")
    print(f"  LR: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Sentiment weight: {args.sentiment_weight}")
    print(f"  Checkpoint: {checkpoint_dir}")
    print()

    # ── 토크나이저 ──
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    # ── 데이터 ──
    train_path = STAGE4_DATA_DIR / "absa_wide_train_stage4.csv"
    val_path = STAGE4_DATA_DIR / "absa_wide_val_stage4.csv"
    assert train_path.exists() and val_path.exists(), "Stage 4 데이터 없음"

    print(f"데이터 로드: {train_path.name} / {val_path.name}")
    train_dataset = create_dataset_from_wide(train_path, tokenizer, cfg["max_length"])
    val_dataset = create_dataset_from_wide(val_path, tokenizer, cfg["max_length"])

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=0, pin_memory=False,
    )
    print(f"  Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

    # ── 모델 생성 + Stage 4 로드 ──
    print("\n모델 생성 중...")

    # Sentiment-only class weights (non-none 셀 기준)
    sentiment_only_weights = compute_sentiment_only_weights(train_dataset)

    # Sentiment class weights (review-level, 기존과 동일)
    sentiment_labels = torch.tensor(train_dataset.sentiment_labels)
    sentiment_class_weights = compute_class_weights(sentiment_labels, 3)

    model = MultiTaskABSAModel(
        model_name=cfg["model_name"],
        dropout=cfg["dropout"],
        sentiment_class_weights=sentiment_class_weights,
        aspect_class_weights=sentiment_only_weights,
        use_focal_loss=cfg["use_focal_loss"],
        focal_gamma=cfg["focal_gamma"],
    )

    # Stage 4 모델 로드
    source_path = args.source_checkpoint or str(STAGE4_CHECKPOINT_DIR / "best_model.pt")
    print(f"\n소스 모델 로드: {source_path}")
    ckpt = torch.load(source_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  missing: {missing}")
    if unexpected:
        print(f"  unexpected: {unexpected}")

    # Encoder freeze
    freeze_encoder(model)

    # ── Trainer ──
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = Phase2Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        max_grad_norm=cfg["max_grad_norm"],
        sentiment_weight=args.sentiment_weight,
        aspect_weight=cfg["aspect_weight"],
        device=device,
        checkpoint_dir=checkpoint_dir,
    )
    trainer.source_checkpoint = source_path

    # ── 학습 ──
    trainer.train(
        num_epochs=args.epochs,
        logging_steps=cfg["logging_steps"],
    )

    print(f"\n학습 완료!")
    print(f"  Best model: {checkpoint_dir / 'best_model.pt'}")
    print(f"  Mode: Phase 2 (encoder freeze, sentiment-only mask)")
    print(f"  LR: {args.lr}, Epochs: {args.epochs}")

    # ── 골든셋 평가 ──
    if not args.skip_golden_eval:
        print(f"\n{'='*60}")
        print(f"골든셋 평가 ({args.tag})")
        print(f"{'='*60}")

        eval_script = ABSA_ROOT / "06_scripts" / "eval_golden.py"
        cmd = [
            sys.executable, str(eval_script),
            "--split", "test",
            "--polar",
            "--postprocess",
            "--checkpoint-dir", str(checkpoint_dir),
        ]
        print(f"  실행: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print(f"골든셋 평가 실패 (code={result.returncode})")
        else:
            print(f"골든셋 평가 완료! ({args.tag})")


if __name__ == "__main__":
    main()
