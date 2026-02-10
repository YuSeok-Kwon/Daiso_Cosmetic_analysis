"""
Evaluation metrics for ABSA model
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from typing import Dict, List
import pandas as pd


class ABSAEvaluator:
    """
    Evaluator for ABSA model.
    """

    def __init__(
        self,
        sentiment_labels: List[str] = None,
        aspect_labels: List[str] = None
    ):
        """
        Args:
            sentiment_labels: List of sentiment label names
            aspect_labels: List of aspect label names
        """
        if sentiment_labels is None:
            self.sentiment_labels = ["negative", "neutral", "positive"]
        else:
            self.sentiment_labels = sentiment_labels

        if aspect_labels is None:
            from RQ_absa.config import ASPECT_LABELS
            self.aspect_labels = ASPECT_LABELS
        else:
            self.aspect_labels = aspect_labels

    def evaluate(
        self,
        sentiment_preds: np.ndarray,
        sentiment_labels: np.ndarray,
        aspect_preds: np.ndarray,
        aspect_labels: np.ndarray
    ) -> Dict:
        """
        Evaluate sentiment and aspect predictions.

        Args:
            sentiment_preds: Sentiment predictions [n_samples]
            sentiment_labels: Sentiment ground truth [n_samples]
            aspect_preds: Aspect predictions [n_samples, n_aspects]
            aspect_labels: Aspect ground truth [n_samples, n_aspects]

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Sentiment metrics
        sentiment_metrics = self._evaluate_sentiment(sentiment_preds, sentiment_labels)
        metrics.update(sentiment_metrics)

        # Aspect metrics
        aspect_metrics = self._evaluate_aspects(aspect_preds, aspect_labels)
        metrics.update(aspect_metrics)

        return metrics

    def _evaluate_sentiment(
        self,
        preds: np.ndarray,
        labels: np.ndarray
    ) -> Dict:
        """Evaluate sentiment classification"""
        metrics = {}

        # Accuracy
        metrics['sentiment_accuracy'] = accuracy_score(labels, preds)

        # Precision, Recall, F1 (macro)
        metrics['sentiment_precision_macro'] = precision_score(
            labels, preds, average='macro', zero_division=0
        )
        metrics['sentiment_recall_macro'] = recall_score(
            labels, preds, average='macro', zero_division=0
        )
        metrics['sentiment_f1_macro'] = f1_score(
            labels, preds, average='macro', zero_division=0
        )

        # Precision, Recall, F1 (weighted)
        metrics['sentiment_precision_weighted'] = precision_score(
            labels, preds, average='weighted', zero_division=0
        )
        metrics['sentiment_recall_weighted'] = recall_score(
            labels, preds, average='weighted', zero_division=0
        )
        metrics['sentiment_f1_weighted'] = f1_score(
            labels, preds, average='weighted', zero_division=0
        )

        # Per-class F1
        per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
        for i, label in enumerate(self.sentiment_labels):
            if i < len(per_class_f1):
                metrics[f'sentiment_f1_{label}'] = per_class_f1[i]

        return metrics

    def _evaluate_aspects(
        self,
        preds: np.ndarray,
        labels: np.ndarray
    ) -> Dict:
        """Evaluate aspect multi-label classification"""
        metrics = {}

        # Micro-average (across all aspect-sample pairs)
        metrics['aspect_precision_micro'] = precision_score(
            labels, preds, average='micro', zero_division=0
        )
        metrics['aspect_recall_micro'] = recall_score(
            labels, preds, average='micro', zero_division=0
        )
        metrics['aspect_f1_micro'] = f1_score(
            labels, preds, average='micro', zero_division=0
        )

        # Macro-average (average across aspects)
        metrics['aspect_precision_macro'] = precision_score(
            labels, preds, average='macro', zero_division=0
        )
        metrics['aspect_recall_macro'] = recall_score(
            labels, preds, average='macro', zero_division=0
        )
        metrics['aspect_f1_macro'] = f1_score(
            labels, preds, average='macro', zero_division=0
        )

        # Per-aspect metrics
        per_aspect_f1 = f1_score(labels, preds, average=None, zero_division=0)
        for i, aspect in enumerate(self.aspect_labels):
            if i < len(per_aspect_f1):
                metrics[f'aspect_f1_{aspect}'] = per_aspect_f1[i]

        return metrics

    def print_report(
        self,
        sentiment_preds: np.ndarray,
        sentiment_labels: np.ndarray,
        aspect_preds: np.ndarray,
        aspect_labels: np.ndarray
    ):
        """Print detailed evaluation report"""
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)

        # Sentiment report
        print("\n### SENTIMENT CLASSIFICATION ###\n")
        print(classification_report(
            sentiment_labels,
            sentiment_preds,
            target_names=self.sentiment_labels,
            zero_division=0
        ))

        # Confusion matrix
        print("Confusion Matrix:")
        cm = confusion_matrix(sentiment_labels, sentiment_preds)
        cm_df = pd.DataFrame(
            cm,
            index=[f"True {l}" for l in self.sentiment_labels],
            columns=[f"Pred {l}" for l in self.sentiment_labels]
        )
        print(cm_df)

        # Aspect report
        print("\n### ASPECT DETECTION ###\n")

        # Overall metrics
        aspect_metrics = self._evaluate_aspects(aspect_preds, aspect_labels)
        print(f"Micro-average:")
        print(f"  Precision: {aspect_metrics['aspect_precision_micro']:.4f}")
        print(f"  Recall:    {aspect_metrics['aspect_recall_micro']:.4f}")
        print(f"  F1:        {aspect_metrics['aspect_f1_micro']:.4f}")

        print(f"\nMacro-average:")
        print(f"  Precision: {aspect_metrics['aspect_precision_macro']:.4f}")
        print(f"  Recall:    {aspect_metrics['aspect_recall_macro']:.4f}")
        print(f"  F1:        {aspect_metrics['aspect_f1_macro']:.4f}")

        # Per-aspect metrics
        print(f"\nPer-aspect F1 scores:")
        for aspect in self.aspect_labels:
            key = f'aspect_f1_{aspect}'
            if key in aspect_metrics:
                f1 = aspect_metrics[key]
                print(f"  {aspect}: {f1:.4f}")

        # Aspect frequency
        print(f"\nAspect frequency (ground truth):")
        aspect_counts = aspect_labels.sum(axis=0)
        for i, aspect in enumerate(self.aspect_labels):
            if i < len(aspect_counts):
                count = int(aspect_counts[i])
                ratio = count / len(labels)
                print(f"  {aspect}: {count} ({ratio*100:.1f}%)")

        print("="*60)


def evaluate_test_set(
    test_loader,
    model,
    device,
    evaluator: ABSAEvaluator = None
) -> Dict:
    """
    Evaluate model on test set.

    Args:
        test_loader: Test data loader
        model: ABSA model
        device: Device
        evaluator: Evaluator instance

    Returns:
        Dictionary of metrics
    """
    if evaluator is None:
        evaluator = ABSAEvaluator()

    model.eval()

    all_sentiment_preds = []
    all_sentiment_labels = []
    all_aspect_preds = []
    all_aspect_labels = []

    import torch
    from tqdm import tqdm

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment_labels = batch['sentiment_label'].to(device)
            aspect_labels = batch['aspect_label'].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Predictions
            sentiment_preds = torch.argmax(outputs['sentiment_logits'], dim=-1)
            aspect_preds = (torch.sigmoid(outputs['aspect_logits']) >= 0.5).long()

            # Collect
            all_sentiment_preds.extend(sentiment_preds.cpu().numpy())
            all_sentiment_labels.extend(sentiment_labels.cpu().numpy())
            all_aspect_preds.extend(aspect_preds.cpu().numpy())
            all_aspect_labels.extend(aspect_labels.cpu().numpy())

    # Convert to arrays
    all_sentiment_preds = np.array(all_sentiment_preds)
    all_sentiment_labels = np.array(all_sentiment_labels)
    all_aspect_preds = np.array(all_aspect_preds)
    all_aspect_labels = np.array(all_aspect_labels)

    # Evaluate
    metrics = evaluator.evaluate(
        sentiment_preds=all_sentiment_preds,
        sentiment_labels=all_sentiment_labels,
        aspect_preds=all_aspect_preds,
        aspect_labels=all_aspect_labels
    )

    # Print report
    evaluator.print_report(
        sentiment_preds=all_sentiment_preds,
        sentiment_labels=all_sentiment_labels,
        aspect_preds=all_aspect_preds,
        aspect_labels=all_aspect_labels
    )

    return metrics
