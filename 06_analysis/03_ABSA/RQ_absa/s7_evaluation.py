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
            from RQ_absa.s1_config import SENTIMENT_LABELS
            self.sentiment_labels = SENTIMENT_LABELS
        else:
            self.sentiment_labels = sentiment_labels

        if aspect_labels is None:
            from RQ_absa.s1_config import ASPECT_LABELS
            self.aspect_labels = ASPECT_LABELS
        else:
            self.aspect_labels = aspect_labels

        if aspect_sentiment_labels is None:
            from RQ_absa.s1_config import ASPECT_SENTIMENT_LABELS
            self.aspect_sentiment_labels = ASPECT_SENTIMENT_LABELS
        else:
            self.aspect_sentiment_labels = aspect_sentiment_labels

    def evaluate(
        self,
        sentiment_preds: np.ndarray,
        sentiment_labels: np.ndarray,
        aspect_preds: np.ndarray,
        aspect_labels: np.ndarray,
        aspect_masks: np.ndarray = None
    ) -> Dict:
        """
        Args:
            sentiment_preds: [N]
            sentiment_labels: [N]
            aspect_preds: [N, 11] (각 값 0~3)
            aspect_labels: [N, 11] (각 값 0~3)
            aspect_masks: [N, 11] (1=평가 포함, 0=제외, optional)
        """
        metrics = {}

        # 1. Sentiment 메트릭
        sentiment_metrics = self._evaluate_sentiment(sentiment_preds, sentiment_labels)
        metrics.update(sentiment_metrics)

        # 2. Aspect-Sentiment 4-class 메트릭
        aspect_sent_metrics = self._evaluate_aspect_sentiment(
            aspect_preds, aspect_labels, aspect_masks
        )
        metrics.update(aspect_sent_metrics)

        # 3. Aspect Detection (none vs not-none) 이진 메트릭
        detection_metrics = self._evaluate_aspect_detection(
            aspect_preds, aspect_labels, aspect_masks
        )
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
        self, preds: np.ndarray, labels: np.ndarray, masks: np.ndarray = None
    ) -> Dict:
        """
        Aspect-Sentiment 4-class 분류 평가.
        masks가 제공되면 mask=1인 셀만 평가.
        """
        metrics = {}

        if masks is not None:
            # mask=1인 셀만 평가
            mask_flat = masks.flatten().astype(bool)
            flat_preds = preds.flatten()[mask_flat]
            flat_labels = labels.flatten()[mask_flat]
        else:
            flat_preds = preds.flatten()
            flat_labels = labels.flatten()

        metrics["aspect_sentiment_accuracy"] = accuracy_score(flat_labels, flat_preds)
        metrics["aspect_sentiment_f1_macro"] = f1_score(
            flat_labels, flat_preds, average="macro", zero_division=0
        )
        metrics["aspect_sentiment_f1_weighted"] = f1_score(
            flat_labels, flat_preds, average="weighted", zero_division=0
        )

        # aspect별 4-class F1
        per_aspect_f1 = []
        for i, aspect_name in enumerate(self.aspect_labels):
            if masks is not None:
                asp_mask = masks[:, i].astype(bool)
                asp_preds = preds[:, i][asp_mask]
                asp_labels = labels[:, i][asp_mask]
            else:
                asp_preds = preds[:, i]
                asp_labels = labels[:, i]

            if len(asp_labels) == 0:
                f1 = 0.0
            else:
                f1 = f1_score(asp_labels, asp_preds, average="macro", zero_division=0)
            metrics[f"aspect_f1_{aspect_name}"] = f1
            per_aspect_f1.append(f1)

        metrics["aspect_sentiment_f1_per_aspect_avg"] = np.mean(per_aspect_f1)

        return metrics

    def _evaluate_aspect_detection(
        self, preds: np.ndarray, labels: np.ndarray, masks: np.ndarray = None
    ) -> Dict:
        """
        Aspect Detection: none(0) vs not-none(1~3) 이진 분류 평가.
        masks가 제공되면 mask=1인 셀만 평가.
        """
        metrics = {}

        # 이진화: 0 → 0, 1~3 → 1
        binary_preds = (preds > 0).astype(int)
        binary_labels = (labels > 0).astype(int)

        if masks is not None:
            mask_flat = masks.flatten().astype(bool)
            flat_preds = binary_preds.flatten()[mask_flat]
            flat_labels = binary_labels.flatten()[mask_flat]
        else:
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
            if masks is not None:
                asp_mask = masks[:, i].astype(bool)
                asp_preds = binary_preds[:, i][asp_mask]
                asp_labels = binary_labels[:, i][asp_mask]
            else:
                asp_preds = binary_preds[:, i]
                asp_labels = binary_labels[:, i]

            if len(asp_labels) == 0:
                f1 = 0.0
            else:
                f1 = f1_score(asp_labels, asp_preds, zero_division=0)
            metrics[f"aspect_detection_f1_{aspect_name}"] = f1

        return metrics

    def print_report(
        self,
        sentiment_preds: np.ndarray,
        sentiment_labels: np.ndarray,
        aspect_preds: np.ndarray,
        aspect_labels: np.ndarray,
        aspect_masks: np.ndarray = None
    ):
        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        if aspect_masks is not None:
            mask_total = aspect_masks.sum()
            total_cells = aspect_masks.size
            print(f"(mask=1 셀만 평가: {int(mask_total)}/{total_cells})")
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
        metrics = self._evaluate_aspect_sentiment(aspect_preds, aspect_labels, aspect_masks)
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
        det_metrics = self._evaluate_aspect_detection(aspect_preds, aspect_labels, aspect_masks)
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
        print(f"\nAspect frequency (ground truth, mask=1 only):")
        for i, aspect_name in enumerate(self.aspect_labels):
            if aspect_masks is not None:
                asp_mask = aspect_masks[:, i].astype(bool)
                asp_labels = aspect_labels[:, i][asp_mask]
            else:
                asp_labels = aspect_labels[:, i]

            count = int((asp_labels > 0).sum())
            total = len(asp_labels)
            ratio = count / total if total > 0 else 0
            print(f"  {aspect_name}: {count}/{total} ({ratio * 100:.1f}%)")

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
    aspect_masks: np.ndarray = None,
    aspect_labels_list: List[str] = None,
    search_range: tuple = (0.1, 0.95),
    search_step: float = 0.05,
    metric: str = "f1",
    beta: float = 0.5,
) -> dict:
    """
    Validation set에서 aspect별 최적 none-threshold를 grid search.
    aspect_masks가 제공되면 mask=1인 셀만 사용하여 튜닝.

    Args:
        aspect_probs: [N, K, 4] softmax 확률
        aspect_labels: [N, K] 정답 (각 값 0~3)
        aspect_masks: [N, K] (1=사용, 0=제외, optional)
        aspect_labels_list: aspect 이름 리스트
        search_range: (min, max) threshold 범위
        search_step: grid search 간격
        metric: 최적화 대상 ("f1", "detection_f1", "fbeta", "f0.5")
        beta: F-beta에서 beta 값 (precision 가중: beta < 1)
    """
    if aspect_labels_list is None:
        from RQ_absa.s1_config import ASPECT_LABELS
        aspect_labels_list = ASPECT_LABELS

    num_aspects = aspect_probs.shape[1]
    candidates = np.arange(search_range[0], search_range[1] + search_step / 2, search_step)

    best_thresholds = np.full(num_aspects, 0.5)  # default
    per_aspect_results = []

    print("\n" + "=" * 60)
    print("NONE-THRESHOLD TUNING (per-aspect)")
    if aspect_masks is not None:
        print(f"(mask=1 셀만 사용: {int(aspect_masks.sum())}/{aspect_masks.size})")
    print("=" * 60)

    for j in range(num_aspects):
        aspect_name = aspect_labels_list[j] if j < len(aspect_labels_list) else f"aspect_{j}"

        # mask=1인 샘플만 필터링
        if aspect_masks is not None:
            asp_mask = aspect_masks[:, j].astype(bool)
            true_labels = aspect_labels[:, j][asp_mask]
            probs_j = aspect_probs[:, j][asp_mask]  # [M, 4]
        else:
            true_labels = aspect_labels[:, j]
            probs_j = aspect_probs[:, j]  # [N, 4]

        if len(true_labels) == 0:
            per_aspect_results.append({
                "aspect": aspect_name, "threshold": 0.5, "f1": 0.0, "detection_f1": 0.0
            })
            print(f"  {aspect_name}: no mask=1 samples, skipping")
            continue

        best_score = -1.0
        best_t = 0.5
        best_detail = {}

        for t in candidates:
            p_none = probs_j[:, 0]
            is_none = p_none >= t
            non_none_probs = probs_j[:, 1:]
            best_non_none = np.argmax(non_none_probs, axis=-1) + 1

            preds_j = np.where(is_none, 0, best_non_none)

            f1_4class = f1_score(true_labels, preds_j, average="macro", zero_division=0)

            binary_preds = (preds_j > 0).astype(int)
            binary_labels = (true_labels > 0).astype(int)
            det_f1 = f1_score(binary_labels, binary_preds, zero_division=0)

            if metric in ("fbeta", "f0.5"):
                det_prec = precision_score(binary_labels, binary_preds, zero_division=0)
                det_rec = recall_score(binary_labels, binary_preds, zero_division=0)
                denom = (beta**2 * det_prec + det_rec)
                fbeta_val = (1 + beta**2) * det_prec * det_rec / denom if denom > 0 else 0.0
                score = fbeta_val
            elif metric == "detection_f1":
                score = det_f1
            else:  # "f1"
                score = f1_4class

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

    # 전체 비교: default(0.5) vs tuned (mask=1만)
    default_thresholds = np.full(num_aspects, 0.5)
    default_preds = apply_none_thresholds(aspect_probs, default_thresholds)
    tuned_preds = apply_none_thresholds(aspect_probs, best_thresholds)

    if aspect_masks is not None:
        mask_flat = aspect_masks.flatten().astype(bool)
        flat_labels = aspect_labels.flatten()[mask_flat]
        default_f1 = f1_score(flat_labels, default_preds.flatten()[mask_flat],
                              average="macro", zero_division=0)
        tuned_f1 = f1_score(flat_labels, tuned_preds.flatten()[mask_flat],
                            average="macro", zero_division=0)
    else:
        flat_labels = aspect_labels.flatten()
        default_f1 = f1_score(flat_labels, default_preds.flatten(),
                              average="macro", zero_division=0)
        tuned_f1 = f1_score(flat_labels, tuned_preds.flatten(),
                            average="macro", zero_division=0)

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
            "aspect_masks": [N, 11],
        }
    """
    import torch
    from tqdm import tqdm

    model.eval()

    all_sentiment_preds = []
    all_sentiment_labels = []
    all_aspect_probs = []
    all_aspect_labels = []
    all_aspect_masks = []

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

            if "aspect_mask" in batch:
                all_aspect_masks.extend(batch["aspect_mask"].cpu().numpy())

    result = {
        "sentiment_preds": np.array(all_sentiment_preds),
        "sentiment_labels": np.array(all_sentiment_labels),
        "aspect_probs": np.array(all_aspect_probs),
        "aspect_labels": np.array(all_aspect_labels),
    }

    if all_aspect_masks:
        result["aspect_masks"] = np.array(all_aspect_masks)

    return result


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

    aspect_masks = results.get("aspect_masks")

    metrics = evaluator.evaluate(
        sentiment_preds=results["sentiment_preds"],
        sentiment_labels=results["sentiment_labels"],
        aspect_preds=aspect_preds,
        aspect_labels=results["aspect_labels"],
        aspect_masks=aspect_masks
    )

    evaluator.print_report(
        sentiment_preds=results["sentiment_preds"],
        sentiment_labels=results["sentiment_labels"],
        aspect_preds=aspect_preds,
        aspect_labels=results["aspect_labels"],
        aspect_masks=aspect_masks
    )

    return metrics


def apply_thresholds_with_polar(
    aspect_probs: np.ndarray,
    none_thresholds: np.ndarray,
    polar_threshold: float = 0.6,
) -> np.ndarray:
    """
    3단계 threshold 적용:
    (1) P(none) >= t_none → 0 (none)
    (2) max(P(pos), P(neg)) < t_polar → 2 (neutral)
    (3) else → argmax(pos, neg) 중 선택 (1 또는 3)

    Args:
        aspect_probs: [N, K, 4] softmax 확률
        none_thresholds: [K] aspect별 none-threshold
        polar_threshold: polar 확신 기준 (pos/neg 최대값이 이 미만이면 neutral)

    Returns:
        aspect_preds: [N, K] (각 값 0~3)
    """
    N, num_aspects, _ = aspect_probs.shape
    preds = np.zeros((N, num_aspects), dtype=int)

    for j in range(num_aspects):
        p_none = aspect_probs[:, j, 0]
        p_pos = aspect_probs[:, j, 1]
        p_neg = aspect_probs[:, j, 3]
        threshold = none_thresholds[j]

        # Step 1: none 판단
        is_none = p_none >= threshold

        # Step 2: non-none 중에서 polar 확신 부족 → neutral
        max_polar = np.maximum(p_pos, p_neg)
        is_weak_polar = max_polar < polar_threshold

        # Step 3: pos vs neg 중 argmax
        polar_pred = np.where(p_pos >= p_neg, 1, 3)  # 1=positive, 3=negative

        # 조합: none > neutral(weak polar) > polar pred
        preds[:, j] = np.where(
            is_none, 0,
            np.where(is_weak_polar, 2, polar_pred)
        )

    return preds


def apply_design_rule_override(texts, aspect_preds, aspect_labels_list=None):
    """디자인 aspect를 키워드 규칙으로 override (모델 예측 무시).

    규칙 우선순위:
      Tier1: positive 키워드 → positive(1)
      Tier2: negative 키워드 → negative(3)
      Tier3: 구조물 키워드 + 감성 수식어 → positive(1) 또는 negative(3)
      매칭 없음 → none(0)
    """
    from RQ_absa.s1_config import DESIGN_RULE_CONFIG, ASPECT_LABELS

    if aspect_labels_list is None:
        aspect_labels_list = list(ASPECT_LABELS)

    if "디자인" not in aspect_labels_list:
        return aspect_preds

    design_idx = aspect_labels_list.index("디자인")
    cfg = DESIGN_RULE_CONFIG
    preds = aspect_preds.copy()
    stats = {"positive": 0, "negative": 0, "none": 0}

    for i, text in enumerate(texts):
        t = str(text)
        assigned = False

        # Tier 1: positive 키워드
        if any(kw in t for kw in cfg["positive_keywords"]):
            preds[i, design_idx] = 1
            stats["positive"] += 1
            assigned = True

        # Tier 2: negative 키워드
        if not assigned and any(kw in t for kw in cfg["negative_keywords"]):
            preds[i, design_idx] = 3
            stats["negative"] += 1
            assigned = True

        # Tier 3: 구조물 + 수식어
        if not assigned:
            has_struct = any(kw in t for kw in cfg["structure_keywords"])
            if has_struct:
                if any(m in t for m in cfg["structure_pos_modifiers"]):
                    preds[i, design_idx] = 1
                    stats["positive"] += 1
                    assigned = True
                elif any(m in t for m in cfg["structure_neg_modifiers"]):
                    preds[i, design_idx] = 3
                    stats["negative"] += 1
                    assigned = True

        # 매칭 없음 → none
        if not assigned:
            preds[i, design_idx] = 0
            stats["none"] += 1

    total = len(texts)
    mentioned = stats["positive"] + stats["negative"]
    print(f"  Design rule override: pos={stats['positive']}, "
          f"neg={stats['negative']}, none={stats['none']} "
          f"(언급률 {mentioned/total*100:.1f}%)")
    return preds


def tune_polar_threshold(
    aspect_probs: np.ndarray,
    aspect_labels: np.ndarray,
    none_thresholds: np.ndarray,
    aspect_masks: np.ndarray = None,
    search_range: tuple = (0.3, 0.8),
    search_step: float = 0.05,
) -> dict:
    """
    Golden dev set에서 polar_threshold를 grid search.
    '언급된 셀'(GT label > 0)의 sentiment macro F1을 최적화.

    Args:
        aspect_probs: [N, K, 4]
        aspect_labels: [N, K]
        none_thresholds: [K]
        aspect_masks: [N, K] (optional)
        search_range: (min, max)
        search_step: grid search 간격

    Returns:
        {"polar_threshold": float, "best_f1": float, "results": list}
    """
    candidates = np.arange(search_range[0], search_range[1] + search_step / 2, search_step)

    best_t = 0.6
    best_f1 = -1.0
    results = []

    print("\n" + "=" * 60)
    print("POLAR THRESHOLD TUNING")
    print("=" * 60)

    for t in candidates:
        preds = apply_thresholds_with_polar(aspect_probs, none_thresholds, polar_threshold=t)

        # 언급된 셀만 (GT label > 0)
        mentioned = aspect_labels > 0
        if aspect_masks is not None:
            mentioned = mentioned & (aspect_masks.astype(bool))

        fp = preds[mentioned]
        fl = aspect_labels[mentioned]

        macro_f1 = f1_score(fl, fp, average="macro", zero_division=0)
        results.append({"polar_threshold": round(t, 2), "mentioned_macro_f1": round(macro_f1, 4)})

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_t = t

        print(f"  polar_threshold={t:.2f}  mentioned_macro_F1={macro_f1:.4f}")

    print(f"\n  Best: polar_threshold={best_t:.2f}  F1={best_f1:.4f}")
    print("=" * 60)

    return {"polar_threshold": round(best_t, 2), "best_f1": round(best_f1, 4), "results": results}


def apply_full_postprocess(
    texts, aspect_preds, ratings=None, *, aspect_probs=None, use_design_rule=True
):
    """전체 후처리 체인: design_rule + keyword_gate(regex) + force_on.

    eval_golden.py, reapply_qc_postprocess.py 등에서 독립 함수로 호출.
    s8_inference.py의 인스턴스 메서드와 동일 로직.

    Args:
        texts: [N] 리뷰 텍스트 리스트
        aspect_preds: [N, K] aspect 예측 (0=none, 1=pos, 2=neu, 3=neg)
        ratings: [N] 별점 (현재 미사용, 확장용)
        aspect_probs: [N, K, 4] softmax 확률 (현재 미사용, 확장용)
        use_design_rule: 디자인 규칙 override 적용 여부

    Returns:
        aspect_preds: [N, K] 후처리 적용 후
    """
    import re
    from RQ_absa.s1_config import (
        ASPECT_LABELS, KEYWORD_GATE_CONFIG, KEYWORD_FORCE_ON_CONFIG,
    )

    preds = aspect_preds.copy()
    aspect_labels_list = list(ASPECT_LABELS)

    # (1) Design Rule Override
    if use_design_rule:
        preds = apply_design_rule_override(texts, preds, aspect_labels_list)

    # (2) Keyword Gate — 정규식 기반 (Stage 4C)
    if KEYWORD_GATE_CONFIG:
        gate_count = 0
        compiled_gates = {}
        for aspect_name, patterns in KEYWORD_GATE_CONFIG.items():
            if aspect_name not in aspect_labels_list:
                continue
            j = aspect_labels_list.index(aspect_name)
            compiled_gates[aspect_name] = (
                j,
                [re.compile(p, re.IGNORECASE) for p in patterns],
            )

        for aspect_name, (j, compiled_patterns) in compiled_gates.items():
            for i, text in enumerate(texts):
                if preds[i, j] == 0:
                    continue
                text_str = str(text)
                if not any(p.search(text_str) for p in compiled_patterns):
                    preds[i, j] = 0
                    gate_count += 1

        if gate_count > 0:
            print(f"  Keyword gate (regex): {gate_count}건 override (non-none → none)")

    # (3) Keyword Force-On (FN 보정)
    if KEYWORD_FORCE_ON_CONFIG:
        force_count = 0
        for aspect_name, config in KEYWORD_FORCE_ON_CONFIG.items():
            if aspect_name not in aspect_labels_list:
                continue
            j = aspect_labels_list.index(aspect_name)
            keywords = config.get("keywords", [])
            sentiment = config.get("sentiment", 1)
            neg_keywords = config.get("negative_keywords", [])
            neg_sentiment = config.get("negative_sentiment", 3)

            for i, text in enumerate(texts):
                if preds[i, j] != 0:
                    continue
                text_str = str(text)
                if neg_keywords and any(kw in text_str for kw in neg_keywords):
                    preds[i, j] = neg_sentiment
                    force_count += 1
                elif any(kw in text_str for kw in keywords):
                    preds[i, j] = sentiment
                    force_count += 1

        if force_count > 0:
            print(f"  Keyword force-on: {force_count}건 활성화")

    return preds
