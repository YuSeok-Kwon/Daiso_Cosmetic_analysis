"""
Training pipeline for ABSA model with class weight / focal loss support
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional
import json

from RQ_absa.model import MultiTaskABSAModel, compute_class_weights
from RQ_absa.evaluation import ABSAEvaluator


class ABSATrainer:
    """
    Trainer for multi-task ABSA model.
    Supports class weight and focal loss for imbalanced data.
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
        """
        Args:
            model: ABSA model
            train_loader: Training data loader
            val_loader: Validation data loader
            sentiment_weight: Weight for sentiment loss
            aspect_weight: Weight for aspect loss
            learning_rate: Learning rate
            warmup_ratio: Warmup ratio
            weight_decay: Weight decay
            max_grad_norm: Max gradient norm for clipping
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
        """
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

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Scheduler (will be initialized in train())
        self.scheduler = None

        # Evaluator
        self.evaluator = ABSAEvaluator()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = 0.0
        self.training_history = []

        print(f"Trainer initialized on device: {self.device}")

    def train(
        self,
        num_epochs: int,
        gradient_accumulation_steps: int = 1,
        logging_steps: int = 100,
        eval_steps: int = 500,
        save_steps: int = 500
    ):
        """
        Train the model.

        Args:
            num_epochs: Number of epochs
            gradient_accumulation_steps: Gradient accumulation steps
            logging_steps: Logging frequency
            eval_steps: Evaluation frequency
            save_steps: Checkpoint saving frequency
        """
        # Initialize scheduler
        total_steps = len(self.train_loader) * num_epochs // gradient_accumulation_steps
        warmup_steps = int(total_steps * self.warmup_ratio)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        print("\n" + "="*60)
        print("TRAINING START")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Total steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")
        print(f"Gradient accumulation: {gradient_accumulation_steps}")
        print("="*60 + "\n")

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_metrics = self._train_epoch(
                gradient_accumulation_steps=gradient_accumulation_steps,
                logging_steps=logging_steps,
                eval_steps=eval_steps,
                save_steps=save_steps
            )

            # Validate
            val_metrics = self.evaluate(self.val_loader)

            # Log epoch metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                'train': train_metrics,
                'val': val_metrics
            }
            self.training_history.append(epoch_metrics)

            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Sentiment Acc: {val_metrics['sentiment_accuracy']:.4f}")
            print(f"  Val Sentiment F1: {val_metrics['sentiment_f1_macro']:.4f}")
            print(f"  Val Aspect F1 (micro): {val_metrics['aspect_f1_micro']:.4f}")
            print(f"  Val Aspect F1 (macro): {val_metrics['aspect_f1_macro']:.4f}")

            # Save best model
            val_metric = val_metrics['sentiment_f1_macro']  # Use sentiment F1 as main metric
            if val_metric > self.best_val_metric:
                self.best_val_metric = val_metric
                self.save_checkpoint(
                    is_best=True,
                    val_metrics=val_metrics
                )
                print(f"  New best model! Val F1: {val_metric:.4f}")

        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Best validation F1: {self.best_val_metric:.4f}")
        print("="*60)

        # Save training history
        if self.checkpoint_dir:
            history_path = self.checkpoint_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            print(f"Training history saved to: {history_path}")

    def _train_epoch(
        self,
        gradient_accumulation_steps: int,
        logging_steps: int,
        eval_steps: int,
        save_steps: int
    ) -> Dict:
        """Train for one epoch"""
        self.model.train()

        total_loss = 0.0
        total_sentiment_loss = 0.0
        total_aspect_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc="Training")

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            sentiment_labels = batch['sentiment_label'].to(self.device)
            aspect_labels = batch['aspect_label'].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sentiment_labels=sentiment_labels,
                aspect_labels=aspect_labels
            )

            # Combined loss
            sentiment_loss = outputs['sentiment_loss']
            aspect_loss = outputs['aspect_loss']
            loss = (
                self.sentiment_weight * sentiment_loss +
                self.aspect_weight * aspect_loss
            )

            # Backward pass
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()

            # Update weights
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Accumulate metrics
            total_loss += loss.item() * gradient_accumulation_steps
            total_sentiment_loss += sentiment_loss.item()
            total_aspect_loss += aspect_loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss / num_batches,
                'sent_loss': total_sentiment_loss / num_batches,
                'asp_loss': total_aspect_loss / num_batches,
                'lr': self.scheduler.get_last_lr()[0]
            })

            # Logging
            if self.global_step % logging_steps == 0:
                avg_loss = total_loss / num_batches
                print(f"\n[Step {self.global_step}] Loss: {avg_loss:.4f}")

            # Evaluation
            if eval_steps > 0 and self.global_step % eval_steps == 0:
                val_metrics = self.evaluate(self.val_loader)
                print(f"\n[Step {self.global_step}] Val Metrics:")
                print(f"  Sentiment Acc: {val_metrics['sentiment_accuracy']:.4f}")
                print(f"  Sentiment F1: {val_metrics['sentiment_f1_macro']:.4f}")
                print(f"  Aspect F1 (micro): {val_metrics['aspect_f1_micro']:.4f}")
                self.model.train()

            # Save checkpoint
            if save_steps > 0 and self.global_step % save_steps == 0:
                self.save_checkpoint(is_best=False)

        # Epoch metrics
        return {
            'loss': total_loss / num_batches,
            'sentiment_loss': total_sentiment_loss / num_batches,
            'aspect_loss': total_aspect_loss / num_batches
        }

    def evaluate(self, data_loader: DataLoader) -> Dict:
        """
        Evaluate the model.

        Args:
            data_loader: Data loader

        Returns:
            Dictionary of metrics
        """
        self.model.eval()

        all_sentiment_preds = []
        all_sentiment_labels = []
        all_aspect_preds = []
        all_aspect_labels = []

        total_loss = 0.0
        total_sentiment_loss = 0.0
        total_aspect_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                sentiment_labels = batch['sentiment_label'].to(self.device)
                aspect_labels = batch['aspect_label'].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    sentiment_labels=sentiment_labels,
                    aspect_labels=aspect_labels
                )

                # Loss
                sentiment_loss = outputs['sentiment_loss']
                aspect_loss = outputs['aspect_loss']
                loss = (
                    self.sentiment_weight * sentiment_loss +
                    self.aspect_weight * aspect_loss
                )

                total_loss += loss.item()
                total_sentiment_loss += sentiment_loss.item()
                total_aspect_loss += aspect_loss.item()
                num_batches += 1

                # Predictions
                sentiment_preds = torch.argmax(outputs['sentiment_logits'], dim=-1)
                aspect_preds = (torch.sigmoid(outputs['aspect_logits']) >= 0.5).long()

                # Collect
                all_sentiment_preds.extend(sentiment_preds.cpu().numpy())
                all_sentiment_labels.extend(sentiment_labels.cpu().numpy())
                all_aspect_preds.extend(aspect_preds.cpu().numpy())
                all_aspect_labels.extend(aspect_labels.cpu().numpy())

        # Calculate metrics
        all_sentiment_preds = np.array(all_sentiment_preds)
        all_sentiment_labels = np.array(all_sentiment_labels)
        all_aspect_preds = np.array(all_aspect_preds)
        all_aspect_labels = np.array(all_aspect_labels)

        metrics = self.evaluator.evaluate(
            sentiment_preds=all_sentiment_preds,
            sentiment_labels=all_sentiment_labels,
            aspect_preds=all_aspect_preds,
            aspect_labels=all_aspect_labels
        )

        # Add loss metrics
        metrics['loss'] = total_loss / num_batches
        metrics['sentiment_loss'] = total_sentiment_loss / num_batches
        metrics['aspect_loss'] = total_aspect_loss / num_batches

        return metrics

    def save_checkpoint(
        self,
        is_best: bool = False,
        val_metrics: Optional[Dict] = None
    ):
        """Save model checkpoint"""
        if self.checkpoint_dir is None:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_metric': self.best_val_metric,
            'val_metrics': val_metrics
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
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_metric = checkpoint['best_val_metric']

        print(f"Loaded checkpoint from: {checkpoint_path}")
        print(f"  Epoch: {self.current_epoch}")
        print(f"  Global step: {self.global_step}")
        print(f"  Best val metric: {self.best_val_metric:.4f}")


def create_model_with_class_weights(
    train_dataset,
    model_name: str = "beomi/KcELECTRA-base",
    num_sentiment_labels: int = 3,
    num_aspect_labels: int = 9,
    dropout: float = 0.1,
    use_class_weight: bool = True,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0
) -> MultiTaskABSAModel:
    """
    Create model with class weights computed from training data.

    Args:
        train_dataset: Training dataset with sentiment_label
        model_name: Pretrained model name
        num_sentiment_labels: Number of sentiment labels
        num_aspect_labels: Number of aspect labels
        dropout: Dropout rate
        use_class_weight: Whether to use class weights
        use_focal_loss: Whether to use focal loss
        focal_gamma: Focal loss gamma

    Returns:
        MultiTaskABSAModel with class weights
    """
    sentiment_class_weights = None

    if use_class_weight or use_focal_loss:
        # Collect all sentiment labels from training data
        all_labels = []
        for i in range(len(train_dataset)):
            item = train_dataset[i]
            all_labels.append(item['sentiment_label'])

        labels_tensor = torch.tensor(all_labels)

        # Compute class weights
        sentiment_class_weights = compute_class_weights(labels_tensor, num_sentiment_labels)

        print("\nSentiment class distribution:")
        class_counts = torch.bincount(labels_tensor, minlength=num_sentiment_labels)
        class_names = ['negative', 'neutral', 'positive']
        for i, (name, count, weight) in enumerate(zip(class_names, class_counts, sentiment_class_weights)):
            print(f"  {name}: {count.item()} samples, weight={weight.item():.4f}")

    model = MultiTaskABSAModel(
        model_name=model_name,
        num_sentiment_labels=num_sentiment_labels,
        num_aspect_labels=num_aspect_labels,
        dropout=dropout,
        sentiment_class_weights=sentiment_class_weights,
        use_focal_loss=use_focal_loss,
        focal_gamma=focal_gamma
    )

    return model
