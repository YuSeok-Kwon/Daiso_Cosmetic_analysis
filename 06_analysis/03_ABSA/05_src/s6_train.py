"""
Training pipeline for ABSA model (Option A: aspect별 4-class 통합)
"""
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional
import json

from RQ_absa.s5_model import MultiTaskABSAModel, compute_class_weights, compute_aspect_class_weights, get_best_device
from RQ_absa.s7_evaluation import (
    ABSAEvaluator,
    collect_predictions,
    tune_none_thresholds,
    apply_none_thresholds,
)


class ABSATrainer:
    """
    Trainer for multi-task ABSA model (Option A).
    Aspect 예측이 4-class (none/neg/neu/pos)로 변경됨.
    """

    def __init__(
        self,
        model: MultiTaskABSAModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        sentiment_weight: float = 1.0,
        aspect_weight: float = 1.0,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        device: str = None,
        checkpoint_dir: Path = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.sentiment_weight = sentiment_weight
        self.aspect_weight = aspect_weight
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir

        if device is None:
            self.device = torch.device(get_best_device())
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = None
        self.evaluator = ABSAEvaluator()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = 0.0
        self.training_history = []
        self.none_thresholds = None  # 학습 후 튜닝된 threshold

        print(f"Trainer initialized on device: {self.device}")

    def train(
        self,
        num_epochs: int,
        gradient_accumulation_steps: int = 1,
        logging_steps: int = 100,
        eval_steps: int = 500,
        save_steps: int = 500
    ):
        # leftover batch 포함한 정확한 step 수 계산
        steps_per_epoch = math.ceil(len(self.train_loader) / gradient_accumulation_steps)
        total_steps = steps_per_epoch * num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        print("\n" + "=" * 60)
        print("TRAINING START")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Total steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")
        print(f"Gradient accumulation: {gradient_accumulation_steps}")
        print("=" * 60 + "\n")

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            train_metrics = self._train_epoch(
                gradient_accumulation_steps=gradient_accumulation_steps,
                logging_steps=logging_steps,
                eval_steps=eval_steps,
                save_steps=save_steps
            )

            val_metrics = self.evaluate(self.val_loader)

            epoch_metrics = {
                "epoch": epoch + 1,
                "train": train_metrics,
                "val": val_metrics
            }
            self.training_history.append(epoch_metrics)

            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Sentiment Acc: {val_metrics['sentiment_accuracy']:.4f}")
            print(f"  Val Sentiment F1: {val_metrics['sentiment_f1_macro']:.4f}")
            print(f"  Val Aspect F1 (macro): {val_metrics['aspect_sentiment_f1_macro']:.4f}")
            print(f"  Val Aspect Detection F1: {val_metrics['aspect_detection_f1']:.4f}")

            # Best model 기준: aspect_sentiment_f1_macro (메인 태스크)
            val_metric = val_metrics["aspect_sentiment_f1_macro"]
            if val_metric > self.best_val_metric:
                self.best_val_metric = val_metric
                self.save_checkpoint(is_best=True, val_metrics=val_metrics)
                print(f"  New best model! Aspect-Sentiment F1: {val_metric:.4f}")

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Best validation Aspect-Sentiment F1: {self.best_val_metric:.4f}")
        print("=" * 60)

        # Best model 로드 후 threshold 튜닝
        self._load_best_and_tune_thresholds()

        if self.checkpoint_dir:
            history_path = self.checkpoint_dir / "training_history.json"
            with open(history_path, "w") as f:
                json.dump(self.training_history, f, indent=2)
            print(f"Training history saved to: {history_path}")

    def _train_epoch(
        self,
        gradient_accumulation_steps: int,
        logging_steps: int,
        eval_steps: int,
        save_steps: int
    ) -> Dict:
        self.model.train()

        total_loss = 0.0
        total_sentiment_loss = 0.0
        total_aspect_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc="Training")

        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            sentiment_labels = batch["sentiment_label"].to(self.device)
            aspect_labels = batch["aspect_label"].to(self.device)  # [B, 11] LongTensor
            aspect_mask = batch["aspect_mask"].to(self.device)     # [B, 11] FloatTensor

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sentiment_labels=sentiment_labels,
                aspect_labels=aspect_labels,
                aspect_mask=aspect_mask
            )

            sentiment_loss = outputs["sentiment_loss"]
            aspect_loss = outputs["aspect_loss"]
            loss = (
                self.sentiment_weight * sentiment_loss
                + self.aspect_weight * aspect_loss
            )

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()

            # Optimizer step: accumulation 완료 또는 마지막 배치
            did_update = False
            is_accum_boundary = (step + 1) % gradient_accumulation_steps == 0
            is_last_batch = (step + 1) == len(self.train_loader)

            if is_accum_boundary or is_last_batch:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                did_update = True

            total_loss += loss.item() * gradient_accumulation_steps
            total_sentiment_loss += sentiment_loss.item()
            total_aspect_loss += aspect_loss.item()
            num_batches += 1

            progress_bar.set_postfix({
                "loss": total_loss / num_batches,
                "sent": total_sentiment_loss / num_batches,
                "asp": total_aspect_loss / num_batches,
                "lr": self.scheduler.get_last_lr()[0]
            })

            # logging/eval/save는 optimizer step 직후에만 실행
            if did_update and self.global_step > 0:
                if logging_steps > 0 and self.global_step % logging_steps == 0:
                    avg_loss = total_loss / num_batches
                    print(f"\n[Step {self.global_step}] Loss: {avg_loss:.4f}")

                if eval_steps > 0 and self.global_step % eval_steps == 0:
                    val_metrics = self.evaluate(self.val_loader)
                    print(f"\n[Step {self.global_step}] Val Metrics:")
                    print(f"  Sentiment Acc: {val_metrics['sentiment_accuracy']:.4f}")
                    print(f"  Sentiment F1: {val_metrics['sentiment_f1_macro']:.4f}")
                    print(f"  Aspect-Sentiment F1: {val_metrics['aspect_sentiment_f1_macro']:.4f}")
                    self.model.train()

                if save_steps > 0 and self.global_step % save_steps == 0:
                    self.save_checkpoint(is_best=False)

        return {
            "loss": total_loss / num_batches,
            "sentiment_loss": total_sentiment_loss / num_batches,
            "aspect_loss": total_aspect_loss / num_batches
        }

    def evaluate(self, data_loader: DataLoader) -> Dict:
        self.model.eval()

        all_sentiment_preds = []
        all_sentiment_labels = []
        all_aspect_probs = []
        all_aspect_labels = []
        all_aspect_masks = []

        total_loss = 0.0
        total_sentiment_loss = 0.0
        total_aspect_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                sentiment_labels = batch["sentiment_label"].to(self.device)
                aspect_labels = batch["aspect_label"].to(self.device)
                aspect_mask = batch["aspect_mask"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    sentiment_labels=sentiment_labels,
                    aspect_labels=aspect_labels,
                    aspect_mask=aspect_mask
                )

                sentiment_loss = outputs["sentiment_loss"]
                aspect_loss = outputs["aspect_loss"]
                loss = (
                    self.sentiment_weight * sentiment_loss
                    + self.aspect_weight * aspect_loss
                )

                total_loss += loss.item()
                total_sentiment_loss += sentiment_loss.item()
                total_aspect_loss += aspect_loss.item()
                num_batches += 1

                sentiment_preds = torch.argmax(outputs["sentiment_logits"], dim=-1)
                aspect_probs = torch.softmax(outputs["aspect_logits"], dim=-1)

                all_sentiment_preds.extend(sentiment_preds.cpu().numpy())
                all_sentiment_labels.extend(sentiment_labels.cpu().numpy())
                all_aspect_probs.extend(aspect_probs.cpu().numpy())
                all_aspect_labels.extend(aspect_labels.cpu().numpy())
                all_aspect_masks.extend(aspect_mask.cpu().numpy())

        all_sentiment_preds = np.array(all_sentiment_preds)
        all_sentiment_labels = np.array(all_sentiment_labels)
        all_aspect_probs = np.array(all_aspect_probs)
        all_aspect_labels = np.array(all_aspect_labels)
        all_aspect_masks = np.array(all_aspect_masks)

        # Threshold가 튜닝되어 있으면 적용, 아니면 argmax
        if self.none_thresholds is not None:
            all_aspect_preds = apply_none_thresholds(all_aspect_probs, self.none_thresholds)
        else:
            all_aspect_preds = np.argmax(all_aspect_probs, axis=-1)

        metrics = self.evaluator.evaluate(
            sentiment_preds=all_sentiment_preds,
            sentiment_labels=all_sentiment_labels,
            aspect_preds=all_aspect_preds,
            aspect_labels=all_aspect_labels,
            aspect_masks=all_aspect_masks
        )

        metrics["loss"] = total_loss / num_batches
        metrics["sentiment_loss"] = total_sentiment_loss / num_batches
        metrics["aspect_loss"] = total_aspect_loss / num_batches

        return metrics

    def tune_thresholds(
        self,
        search_range: tuple = (0.1, 0.95),
        search_step: float = 0.05,
        metric: str = "f1",
        beta: float = 0.5,
        polar_threshold: float = None,
    ):
        """
        Val set에서 aspect별 최적 none-threshold를 grid search.
        결과를 self.none_thresholds에 저장하고, 체크포인트에도 포함.
        polar_threshold가 주어지면 json에 함께 저장 (inference에서 자동 로드).
        """
        print("\nCollecting val predictions for threshold tuning...")
        results = collect_predictions(self.val_loader, self.model, self.device)

        tuning_result = tune_none_thresholds(
            aspect_probs=results["aspect_probs"],
            aspect_labels=results["aspect_labels"],
            aspect_masks=results.get("aspect_masks"),
            search_range=search_range,
            search_step=search_step,
            metric=metric,
            beta=beta,
        )

        self.none_thresholds = tuning_result["thresholds"]

        # 체크포인트에 threshold 추가 저장
        if self.checkpoint_dir:
            threshold_path = self.checkpoint_dir / "none_thresholds.json"
            threshold_data = {
                "thresholds": self.none_thresholds.tolist(),
                "per_aspect_results": tuning_result["per_aspect_results"],
                "default_f1": tuning_result["default_f1"],
                "tuned_f1": tuning_result["tuned_f1"],
            }
            # polar_threshold가 있으면 함께 저장 → inference에서 자동 로드
            if polar_threshold is not None:
                threshold_data["polar_threshold"] = polar_threshold
            with open(threshold_path, "w", encoding="utf-8") as f:
                json.dump(threshold_data, f, indent=2, ensure_ascii=False)
            print(f"Saved thresholds to: {threshold_path}")
            if polar_threshold is not None:
                print(f"  polar_threshold={polar_threshold} 포함")

    def _load_best_and_tune_thresholds(self):
        """
        Best model을 로드한 뒤 THRESHOLD_TUNING_CONFIG 기준으로 threshold 튜닝.
        튜닝된 threshold를 best_model.pt에도 포함시킴.
        """
        from RQ_absa.s1_config import THRESHOLD_TUNING_CONFIG

        tune_kwargs = {
            "search_range": THRESHOLD_TUNING_CONFIG["search_range"],
            "search_step": THRESHOLD_TUNING_CONFIG["search_step"],
            "metric": THRESHOLD_TUNING_CONFIG["metric"],
            "beta": THRESHOLD_TUNING_CONFIG.get("beta", 0.5),
            "polar_threshold": THRESHOLD_TUNING_CONFIG.get("polar_threshold"),
        }

        if self.checkpoint_dir:
            best_path = self.checkpoint_dir / "best_model.pt"
            if best_path.exists():
                ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(ckpt["model_state_dict"])
                print(f"Loaded best model for threshold tuning: {best_path}")

                self.tune_thresholds(**tune_kwargs)

                # threshold를 best_model.pt에도 저장
                if self.none_thresholds is not None:
                    ckpt["none_thresholds"] = self.none_thresholds.tolist()
                    torch.save(ckpt, best_path)
                    print(f"Updated best_model.pt with none_thresholds")
                return

        # best_model.pt가 없으면 현재 모델로 튜닝
        self.tune_thresholds(**tune_kwargs)

    def fine_tune(
        self,
        gold_finetune_loader: DataLoader,
        gold_tune_loader: DataLoader,
        num_epochs: int = 5,
        learning_rate: float = 2e-6,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
    ):
        """
        Stage 2: 골든셋 파인튜닝.

        Stage 1 best model에서 이어서 낮은 LR로 학습하여 바이어스를 교정한다.
        학습 후 gold_tune_loader로 none-threshold를 재튜닝한다.

        Args:
            gold_finetune_loader: 골든셋 finetune 분할 DataLoader
            gold_tune_loader: 골든셋 tune 분할 DataLoader (threshold 튜닝용)
            num_epochs: 파인튜닝 epoch 수
            learning_rate: Stage 1보다 낮은 LR (기본 2e-6)
            warmup_ratio: warmup 비율
            weight_decay: weight decay
        """
        print("\n" + "=" * 60)
        print("STAGE 2: GOLDEN SET FINE-TUNING")
        print("=" * 60)
        print(f"  Fine-tune samples: {len(gold_finetune_loader.dataset):,}")
        print(f"  Tune samples: {len(gold_tune_loader.dataset):,}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Epochs: {num_epochs}")
        print("=" * 60 + "\n")

        # optimizer 재초기화 (낮은 LR)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        total_steps = len(gold_finetune_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # train/val loader를 골든셋으로 교체
        original_train_loader = self.train_loader
        original_val_loader = self.val_loader
        self.train_loader = gold_finetune_loader
        self.val_loader = gold_tune_loader

        best_finetune_metric = 0.0

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            print(f"\n[Fine-tune] Epoch {epoch + 1}/{num_epochs}")

            # gradient_accumulation_steps=1, mid-epoch eval 비활성화
            train_metrics = self._train_epoch(
                gradient_accumulation_steps=1,
                logging_steps=50,
                eval_steps=0,
                save_steps=0
            )

            val_metrics = self.evaluate(gold_tune_loader)

            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Tune Loss: {val_metrics['loss']:.4f}")
            print(f"  Tune Sentiment F1: {val_metrics['sentiment_f1_macro']:.4f}")
            print(f"  Tune Aspect F1: {val_metrics['aspect_sentiment_f1_macro']:.4f}")

            val_metric = val_metrics["aspect_sentiment_f1_macro"]
            if val_metric > best_finetune_metric:
                best_finetune_metric = val_metric
                self.best_val_metric = val_metric
                self.save_checkpoint(is_best=True, val_metrics=val_metrics)
                print(f"  New best fine-tuned model! Aspect-Sentiment F1: {val_metric:.4f}")

        print("\n" + "=" * 60)
        print("FINE-TUNING COMPLETE")
        print("=" * 60)
        print(f"Best fine-tune Aspect-Sentiment F1: {best_finetune_metric:.4f}")
        print("=" * 60)

        # Best model 로드 후 threshold 재튜닝
        self._load_best_and_tune_thresholds()

        # 원래 loader 복원
        self.train_loader = original_train_loader
        self.val_loader = original_val_loader

    def save_checkpoint(self, is_best: bool = False, val_metrics: Optional[Dict] = None):
        if self.checkpoint_dir is None:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        from RQ_absa.s1_config import ASPECT_LABELS, ASPECT_SENTIMENT_TO_ID

        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_metric": self.best_val_metric,
            "val_metrics": val_metrics,
            # 라벨 매핑 메타 (추론 시 해석 보장)
            "label_meta": {
                "aspect_labels": ASPECT_LABELS,
                "aspect_sentiment_to_id": ASPECT_SENTIMENT_TO_ID,
            },
        }

        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, path)
            print(f"Saved best model to: {path}")
        else:
            path = self.checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
            torch.save(checkpoint, path)
            print(f"Saved checkpoint to: {path}")

    def load_checkpoint(self, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_metric = checkpoint.get("best_val_metric", 0.0)

        print(f"Loaded checkpoint from: {checkpoint_path}")
        print(f"  Epoch: {self.current_epoch}")
        print(f"  Global step: {self.global_step}")
        print(f"  Best val metric: {self.best_val_metric:.4f}")

        # 라벨 매핑 검증
        if "label_meta" in checkpoint:
            from RQ_absa.s1_config import ASPECT_SENTIMENT_TO_ID
            saved_map = checkpoint["label_meta"].get("aspect_sentiment_to_id", {})
            if saved_map != ASPECT_SENTIMENT_TO_ID:
                print(f"  WARNING: 체크포인트 라벨 매핑이 현재 config와 다름!")
                print(f"    Checkpoint: {saved_map}")
                print(f"    Config:     {ASPECT_SENTIMENT_TO_ID}")


def create_model_with_class_weights(
    train_dataset,
    model_name: str = "beomi/KcELECTRA-base",
    num_sentiment_labels: int = 3,
    num_aspect_labels: int = None,
    num_aspect_sentiment_classes: int = 4,
    dropout: float = 0.1,
    use_class_weight: bool = True,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0
) -> MultiTaskABSAModel:
    """
    학습 데이터에서 class weights를 계산하여 모델 생성.
    sentiment + aspect 둘 다 가중치 적용.
    """
    from RQ_absa.s1_config import ASPECT_LABELS as _AL
    if num_aspect_labels is None:
        num_aspect_labels = len(_AL)

    sentiment_class_weights = None
    aspect_class_weights = None

    if use_class_weight or use_focal_loss:
        # Dataset 속성에서 직접 접근 (토크나이징 회피)
        from RQ_absa.s1_config import SENTIMENT_LABELS, ASPECT_SENTIMENT_LABELS

        sentiment_tensor = torch.tensor(train_dataset.sentiment_labels)
        aspect_tensor = torch.tensor(train_dataset.aspect_labels, dtype=torch.long)
        mask_tensor = (
            torch.tensor(train_dataset.aspect_masks, dtype=torch.float)
            if train_dataset.aspect_masks is not None
            else None
        )

        sentiment_class_weights = compute_class_weights(sentiment_tensor, num_sentiment_labels)

        print("\nSentiment class distribution:")
        class_counts = torch.bincount(sentiment_tensor, minlength=num_sentiment_labels)
        for name, count, weight in zip(SENTIMENT_LABELS, class_counts, sentiment_class_weights):
            print(f"  {name}: {count.item()} samples, weight={weight.item():.4f}")

        # Aspect class weights (4-class: none/pos/neu/neg, mask=1만 사용)
        aspect_class_weights = compute_aspect_class_weights(
            aspect_tensor, num_aspect_sentiment_classes, aspect_masks=mask_tensor
        )

        print("\nAspect-Sentiment class distribution (mask=1 only):")
        if mask_tensor is not None:
            mask_flat = mask_tensor.view(-1).bool()
            flat_aspects = aspect_tensor.view(-1)[mask_flat]
        else:
            flat_aspects = aspect_tensor.view(-1)
        aspect_counts = torch.bincount(flat_aspects, minlength=num_aspect_sentiment_classes)
        for name, count, weight in zip(ASPECT_SENTIMENT_LABELS, aspect_counts, aspect_class_weights):
            print(f"  {name}: {count.item()} samples, weight={weight.item():.4f}")

    model = MultiTaskABSAModel(
        model_name=model_name,
        num_sentiment_labels=num_sentiment_labels,
        num_aspect_labels=num_aspect_labels,
        num_aspect_sentiment_classes=num_aspect_sentiment_classes,
        dropout=dropout,
        sentiment_class_weights=sentiment_class_weights,
        aspect_class_weights=aspect_class_weights,
        use_focal_loss=use_focal_loss,
        focal_gamma=focal_gamma
    )

    return model
