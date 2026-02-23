"""
Evaluation metrics for ABSA model (Option A: aspect별 4-class 통합)

평가 항목:
1. Sentiment 분류 (3-class): accuracy, F1 등
2. Aspect-Sentiment 분류 (4-class per aspect): macro F1
3. Aspect Detection (none vs not-none): 이진 분류 메트릭
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
    Evaluator for ABSA model (Option A).
    """

    def __init__(
        self,
        sentiment_labels: List[str] = None,
        aspect_labels: List[str] = None,
        aspect_sentiment_labels: List[str] = None
    ):
        if sentiment_labels is None:
            self.sentiment_labels = ["negative", "neutral", "positive"]
        else:
            self.sentiment_labels = sentiment_labels

        if aspect_labels is None:
            from RQ_absa.config import ASPECT_LABELS
            self.aspect_labels = ASPECT_LABELS
        else:
            self.aspect_labels = aspect_labels

        if aspect_sentiment_labels is None:
            self.aspect_sentiment_labels = ["none", "negative", "neutral", "positive"]
        else:
            self.aspect_sentiment_labels = aspect_sentiment_labels

    def evaluate(
        self,
        sentiment_preds: np.ndarray,
        sentiment_labels: np.ndarray,
        aspect_preds: np.ndarray,
        aspect_labels: np.ndarray
    ) -> Dict:
        """
        Args:
            sentiment_preds: [N]
            sentiment_labels: [N]
            aspect_preds: [N, 11] (각 값 0~3)
            aspect_labels: [N, 11] (각 값 0~3)
        """
        metrics = {}

        # 1. Sentiment 메트릭
        sentiment_metrics = self._evaluate_sentiment(sentiment_preds, sentiment_labels)
        metrics.update(sentiment_metrics)

        # 2. Aspect-Sentiment 4-class 메트릭
        aspect_sent_metrics = self._evaluate_aspect_sentiment(aspect_preds, aspect_labels)
        metrics.update(aspect_sent_metrics)

        # 3. Aspect Detection (none vs not-none) 이진 메트릭
        detection_metrics = self._evaluate_aspect_detection(aspect_preds, aspect_labels)
        metrics.update(detection_metrics)

        return metrics

    def _evaluate_sentiment(self, preds: np.ndarray, labels: np.ndarray) -> Dict:
        metrics = {}

        metrics["sentiment_accuracy"] = accuracy_score(labels, preds)
        metrics["sentiment_precision_macro"] = precision_score(
            labels, preds, average="macro", zero_division=0
        )
        metrics["sentiment_recall_macro"] = recall_score(
            labels, preds, average="macro", zero_division=0
        )
        metrics["sentiment_f1_macro"] = f1_score(
            labels, preds, average="macro", zero_division=0
        )
        metrics["sentiment_f1_weighted"] = f1_score(
            labels, preds, average="weighted", zero_division=0
        )

        per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
        for i, label in enumerate(self.sentiment_labels):
            if i < len(per_class_f1):
                metrics[f"sentiment_f1_{label}"] = per_class_f1[i]

        return metrics

    def _evaluate_aspect_sentiment(
        self, preds: np.ndarray, labels: np.ndarray
    ) -> Dict:
        """
        Aspect-Sentiment 4-class 분류 평가.
        각 aspect별 + 전체 macro F1.
        """
        metrics = {}

        # 전체 aspect-sentiment (flatten하여 4-class 분류로 평가)
        flat_preds = preds.flatten()
        flat_labels = labels.flatten()

        metrics["aspect_sentiment_accuracy"] = accuracy_score(flat_labels, flat_preds)
        metrics["aspect_sentiment_f1_macro"] = f1_score(
            flat_labels, flat_preds, average="macro", zero_division=0
        )
        metrics["aspect_sentiment_f1_weighted"] = f1_score(
            flat_labels, flat_preds, average="weighted", zero_division=0
        )

        # aspect별 4-class F1 (각 aspect 컬럼을 독립적으로 평가)
        per_aspect_f1 = []
        for i, aspect_name in enumerate(self.aspect_labels):
            asp_preds = preds[:, i]
            asp_labels = labels[:, i]
            f1 = f1_score(asp_labels, asp_preds, average="macro", zero_division=0)
            metrics[f"aspect_f1_{aspect_name}"] = f1
            per_aspect_f1.append(f1)

        metrics["aspect_sentiment_f1_per_aspect_avg"] = np.mean(per_aspect_f1)

        return metrics

    def _evaluate_aspect_detection(
        self, preds: np.ndarray, labels: np.ndarray
    ) -> Dict:
        """
        Aspect Detection: none(0) vs not-none(1~3) 이진 분류 평가.
        """
        metrics = {}

        # 이진화: 0 → 0, 1~3 → 1
        binary_preds = (preds > 0).astype(int)
        binary_labels = (labels > 0).astype(int)

        # 전체 (flatten)
        flat_preds = binary_preds.flatten()
        flat_labels = binary_labels.flatten()

        metrics["aspect_detection_accuracy"] = accuracy_score(flat_labels, flat_preds)
        metrics["aspect_detection_precision"] = precision_score(
            flat_labels, flat_preds, zero_division=0
        )
        metrics["aspect_detection_recall"] = recall_score(
            flat_labels, flat_preds, zero_division=0
        )
        metrics["aspect_detection_f1"] = f1_score(
            flat_labels, flat_preds, zero_division=0
        )

        # aspect별 detection F1
        for i, aspect_name in enumerate(self.aspect_labels):
            asp_preds = binary_preds[:, i]
            asp_labels = binary_labels[:, i]
            f1 = f1_score(asp_labels, asp_preds, zero_division=0)
            metrics[f"aspect_detection_f1_{aspect_name}"] = f1

        return metrics

    def print_report(
        self,
        sentiment_preds: np.ndarray,
        sentiment_labels: np.ndarray,
        aspect_preds: np.ndarray,
        aspect_labels: np.ndarray
    ):
        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)

        # --- Sentiment ---
        print("\n### SENTIMENT CLASSIFICATION ###\n")
        print(classification_report(
            sentiment_labels,
            sentiment_preds,
            target_names=self.sentiment_labels,
            zero_division=0
        ))

        print("Confusion Matrix:")
        cm = confusion_matrix(sentiment_labels, sentiment_preds)
        cm_df = pd.DataFrame(
            cm,
            index=[f"True {l}" for l in self.sentiment_labels],
            columns=[f"Pred {l}" for l in self.sentiment_labels]
        )
        print(cm_df)

        # --- Aspect-Sentiment (4-class) ---
        print("\n### ASPECT-SENTIMENT CLASSIFICATION (4-class) ###\n")
        metrics = self._evaluate_aspect_sentiment(aspect_preds, aspect_labels)
        print(f"Overall Accuracy: {metrics['aspect_sentiment_accuracy']:.4f}")
        print(f"Overall Macro F1: {metrics['aspect_sentiment_f1_macro']:.4f}")
        print(f"Overall Weighted F1: {metrics['aspect_sentiment_f1_weighted']:.4f}")

        print(f"\nPer-aspect Macro F1:")
        for aspect_name in self.aspect_labels:
            key = f"aspect_f1_{aspect_name}"
            if key in metrics:
                print(f"  {aspect_name}: {metrics[key]:.4f}")

        # --- Aspect Detection (binary) ---
        print("\n### ASPECT DETECTION (none vs not-none) ###\n")
        det_metrics = self._evaluate_aspect_detection(aspect_preds, aspect_labels)
        print(f"Detection Accuracy:  {det_metrics['aspect_detection_accuracy']:.4f}")
        print(f"Detection Precision: {det_metrics['aspect_detection_precision']:.4f}")
        print(f"Detection Recall:    {det_metrics['aspect_detection_recall']:.4f}")
        print(f"Detection F1:        {det_metrics['aspect_detection_f1']:.4f}")

        print(f"\nPer-aspect Detection F1:")
        for aspect_name in self.aspect_labels:
            key = f"aspect_detection_f1_{aspect_name}"
            if key in det_metrics:
                print(f"  {aspect_name}: {det_metrics[key]:.4f}")

        # --- Aspect 빈도 ---
        print(f"\nAspect frequency (ground truth):")
        binary_labels = (aspect_labels > 0).astype(int)
        aspect_counts = binary_labels.sum(axis=0)
        for i, aspect_name in enumerate(self.aspect_labels):
            if i < len(aspect_counts):
                count = int(aspect_counts[i])
                ratio = count / len(aspect_labels)
                print(f"  {aspect_name}: {count} ({ratio * 100:.1f}%)")

        print("=" * 60)


def apply_none_thresholds(
    aspect_probs: np.ndarray,
    none_thresholds: np.ndarray
) -> np.ndarray:
    """
    Per-aspect none-threshold를 적용하여 예측값 생성.

    로직: P(none) >= threshold → none(0) 예측
          P(none) < threshold → non-none 중 최대 확률 클래스 예측

    Args:
        aspect_probs: [N, 11, 4] softmax 확률
        none_thresholds: [11] aspect별 none-threshold

    Returns:
        aspect_preds: [N, 11] (각 값 0~3)
    """
    N, num_aspects, num_classes = aspect_probs.shape
    preds = np.zeros((N, num_aspects), dtype=int)

    for j in range(num_aspects):
        p_none = aspect_probs[:, j, 0]  # P(none) for aspect j
        threshold = none_thresholds[j]

        # P(none) >= threshold → none(0)
        is_none = p_none >= threshold

        # P(none) < threshold → non-none 클래스 중 argmax (인덱스 1,2,3)
        non_none_probs = aspect_probs[:, j, 1:]  # [N, 3]
        best_non_none = np.argmax(non_none_probs, axis=-1) + 1  # offset by 1

        preds[:, j] = np.where(is_none, 0, best_non_none)

    return preds


def tune_none_thresholds(
    aspect_probs: np.ndarray,
    aspect_labels: np.ndarray,
    aspect_labels_list: List[str] = None,
    search_range: tuple = (0.1, 0.95),
    search_step: float = 0.05,
    metric: str = "f1"
) -> dict:
    """
    Validation set에서 aspect별 최적 none-threshold를 grid search.

    각 aspect에 대해:
    - threshold 범위를 순회하며
    - apply_none_thresholds → F1 계산
    - 최고 F1의 threshold 선택

    Args:
        aspect_probs: [N, 11, 4] softmax 확률
        aspect_labels: [N, 11] 정답 (각 값 0~3)
        aspect_labels_list: aspect 이름 리스트
        search_range: (min, max) threshold 범위
        search_step: grid search 간격
        metric: 최적화 대상 ("f1" or "detection_f1")

    Returns:
        {
            "thresholds": np.ndarray [11],
            "per_aspect_results": [{aspect, threshold, f1, detection_f1}, ...],
            "default_f1": float,  # threshold=0.5일 때 전체 F1
            "tuned_f1": float     # 튜닝된 threshold 적용 시 전체 F1
        }
    """
    if aspect_labels_list is None:
        from RQ_absa.config import ASPECT_LABELS
        aspect_labels_list = ASPECT_LABELS

    num_aspects = aspect_probs.shape[1]
    candidates = np.arange(search_range[0], search_range[1] + search_step / 2, search_step)

    best_thresholds = np.full(num_aspects, 0.5)  # default
    per_aspect_results = []

    print("\n" + "=" * 60)
    print("NONE-THRESHOLD TUNING (per-aspect)")
    print("=" * 60)

    for j in range(num_aspects):
        aspect_name = aspect_labels_list[j] if j < len(aspect_labels_list) else f"aspect_{j}"
        true_labels = aspect_labels[:, j]

        best_score = -1.0
        best_t = 0.5
        best_detail = {}

        for t in candidates:
            # 이 aspect만 threshold 적용
            p_none = aspect_probs[:, j, 0]
            is_none = p_none >= t
            non_none_probs = aspect_probs[:, j, 1:]
            best_non_none = np.argmax(non_none_probs, axis=-1) + 1

            preds_j = np.where(is_none, 0, best_non_none)

            # 4-class F1
            f1_4class = f1_score(true_labels, preds_j, average="macro", zero_division=0)

            # Detection F1 (none vs not-none)
            binary_preds = (preds_j > 0).astype(int)
            binary_labels = (true_labels > 0).astype(int)
            det_f1 = f1_score(binary_labels, binary_preds, zero_division=0)

            score = f1_4class if metric == "f1" else det_f1

            if score > best_score:
                best_score = score
                best_t = t
                best_detail = {"f1": f1_4class, "detection_f1": det_f1}

        best_thresholds[j] = best_t
        per_aspect_results.append({
            "aspect": aspect_name,
            "threshold": round(best_t, 3),
            **{k: round(v, 4) for k, v in best_detail.items()}
        })
        print(f"  {aspect_name}: threshold={best_t:.2f}  "
              f"F1={best_detail['f1']:.4f}  Det.F1={best_detail['detection_f1']:.4f}")

    # 전체 비교: default(0.5) vs tuned
    default_thresholds = np.full(num_aspects, 0.5)
    default_preds = apply_none_thresholds(aspect_probs, default_thresholds)
    tuned_preds = apply_none_thresholds(aspect_probs, best_thresholds)

    flat_labels = aspect_labels.flatten()
    default_f1 = f1_score(flat_labels, default_preds.flatten(), average="macro", zero_division=0)
    tuned_f1 = f1_score(flat_labels, tuned_preds.flatten(), average="macro", zero_division=0)

    improvement = tuned_f1 - default_f1
    print(f"\nOverall Macro F1: default={default_f1:.4f} → tuned={tuned_f1:.4f} "
          f"(+{improvement:.4f})")
    print("=" * 60)

    return {
        "thresholds": best_thresholds,
        "per_aspect_results": per_aspect_results,
        "default_f1": default_f1,
        "tuned_f1": tuned_f1,
    }


def collect_predictions(
    data_loader,
    model,
    device,
) -> dict:
    """
    DataLoader로부터 모델 예측값과 정답을 수집.
    threshold 튜닝과 평가에서 공통으로 사용.

    Returns:
        {
            "sentiment_preds": [N],
            "sentiment_labels": [N],
            "aspect_probs": [N, 11, 4],
            "aspect_labels": [N, 11],
        }
    """
    import torch
    from tqdm import tqdm

    model.eval()

    all_sentiment_preds = []
    all_sentiment_labels = []
    all_aspect_probs = []
    all_aspect_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Collecting predictions"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sentiment_labels_batch = batch["sentiment_label"].to(device)
            aspect_labels_batch = batch["aspect_label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            sentiment_preds = torch.argmax(outputs["sentiment_logits"], dim=-1)
            aspect_probs = torch.softmax(outputs["aspect_logits"], dim=-1)  # [B, 11, 4]

            all_sentiment_preds.extend(sentiment_preds.cpu().numpy())
            all_sentiment_labels.extend(sentiment_labels_batch.cpu().numpy())
            all_aspect_probs.extend(aspect_probs.cpu().numpy())
            all_aspect_labels.extend(aspect_labels_batch.cpu().numpy())

    return {
        "sentiment_preds": np.array(all_sentiment_preds),
        "sentiment_labels": np.array(all_sentiment_labels),
        "aspect_probs": np.array(all_aspect_probs),
        "aspect_labels": np.array(all_aspect_labels),
    }


def evaluate_test_set(
    test_loader,
    model,
    device,
    evaluator: ABSAEvaluator = None,
    none_thresholds: np.ndarray = None
) -> Dict:
    """
    테스트셋 평가.

    Args:
        none_thresholds: [11] per-aspect none-threshold.
            None이면 기본 argmax 사용.
    """
    if evaluator is None:
        evaluator = ABSAEvaluator()

    results = collect_predictions(test_loader, model, device)

    # Aspect predictions: threshold 적용 여부
    if none_thresholds is not None:
        aspect_preds = apply_none_thresholds(results["aspect_probs"], none_thresholds)
        print(f"Using tuned none-thresholds for evaluation")
    else:
        aspect_preds = np.argmax(results["aspect_probs"], axis=-1)

    metrics = evaluator.evaluate(
        sentiment_preds=results["sentiment_preds"],
        sentiment_labels=results["sentiment_labels"],
        aspect_preds=aspect_preds,
        aspect_labels=results["aspect_labels"]
    )

    evaluator.print_report(
        sentiment_preds=results["sentiment_preds"],
        sentiment_labels=results["sentiment_labels"],
        aspect_preds=aspect_preds,
        aspect_labels=results["aspect_labels"]
    )

    return metrics
