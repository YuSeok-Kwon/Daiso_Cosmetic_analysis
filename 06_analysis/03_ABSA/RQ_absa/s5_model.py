"""
Multi-task ABSA model (Option A: aspect별 4-class 통합)

출력 구조:
- sentiment_logits: [B, 3] (리뷰 전체 감성, 보조 태스크)
- aspect_logits: [B, N, 4] (aspect별 none/positive/neutral/negative)

라벨 매핑 (s1_config 기준):
  0=none, 1=positive, 2=neutral, 3=negative
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Optional

from RQ_absa.s1_config import ASPECT_LABELS


def get_best_device() -> str:
    """CUDA > MPS > CPU 순으로 최적 디바이스 반환."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)

        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_weight = alpha_t * focal_weight

        focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class MultiTaskABSAModel(nn.Module):
    """
    Multi-task model for ABSA (Option A):
    - Sentiment classification: [B, 3] (보조 태스크)
    - Aspect-Sentiment classification: [B, 11, 4] (메인 태스크)
      각 aspect별 none(0)/positive(1)/neutral(2)/negative(3)
    """

    def __init__(
        self,
        model_name: str = "beomi/KcELECTRA-base",
        num_sentiment_labels: int = 3,
        num_aspect_labels: int = len(ASPECT_LABELS),
        num_aspect_sentiment_classes: int = 4,
        dropout: float = 0.1,
        sentiment_class_weights: Optional[torch.Tensor] = None,
        aspect_class_weights: Optional[torch.Tensor] = None,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0
    ):
        super().__init__()

        self.model_name = model_name
        self.num_sentiment_labels = num_sentiment_labels
        self.num_aspect_labels = num_aspect_labels
        self.num_aspect_sentiment_classes = num_aspect_sentiment_classes
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma

        # Masked CE loss에서 직접 사용하기 위해 class weights 저장
        self._aspect_class_weights = aspect_class_weights

        # Pretrained encoder
        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        self.hidden_size = config.hidden_size

        self.dropout = nn.Dropout(dropout)

        # Sentiment head (보조 태스크): [B, hidden] → [B, 3]
        self.sentiment_classifier = nn.Linear(self.hidden_size, num_sentiment_labels)

        # Aspect-Sentiment head (메인 태스크): [B, hidden] → [B, 11*4=44] → reshape [B, 11, 4]
        self.aspect_classifier = nn.Linear(
            self.hidden_size, num_aspect_labels * num_aspect_sentiment_classes
        )

        # Sentiment loss
        if use_focal_loss:
            self.sentiment_loss_fn = FocalLoss(
                alpha=sentiment_class_weights, gamma=focal_gamma
            )
            print(f"Sentiment: Focal Loss (gamma={focal_gamma})")
        else:
            self.sentiment_loss_fn = nn.CrossEntropyLoss(weight=sentiment_class_weights)
            if sentiment_class_weights is not None:
                print(f"Sentiment: CE with class weights {sentiment_class_weights.tolist()}")
            else:
                print("Sentiment: CE without class weights")

        # Aspect loss: CrossEntropyLoss (4-class per aspect)
        if use_focal_loss:
            self.aspect_loss_fn = FocalLoss(
                alpha=aspect_class_weights, gamma=focal_gamma
            )
            print(f"Aspect: Focal Loss (gamma={focal_gamma})")
        else:
            self.aspect_loss_fn = nn.CrossEntropyLoss(weight=aspect_class_weights)
            if aspect_class_weights is not None:
                print(f"Aspect: CE with class weights {aspect_class_weights.tolist()}")
            else:
                print("Aspect: CE without class weights")

    def forward(
        self,
        input_ids,
        attention_mask,
        sentiment_labels=None,
        aspect_labels=None,
        aspect_mask=None
    ):
        """
        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]
            sentiment_labels: [B] (optional)
            aspect_labels: [B, 11] LongTensor, 각 값 0~3 (optional)
            aspect_mask: [B, 11] FloatTensor, 1=학습 포함, 0=loss 제외 (optional)

        Returns:
            dict with:
                sentiment_logits: [B, 3]
                aspect_logits: [B, 11, 4]
                (+ losses if labels provided)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # [CLS] representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        # Sentiment logits: [B, 3]
        sentiment_logits = self.sentiment_classifier(pooled_output)

        # Aspect logits: [B, 44] → [B, 11, 4]
        aspect_logits_flat = self.aspect_classifier(pooled_output)
        aspect_logits = aspect_logits_flat.view(
            -1, self.num_aspect_labels, self.num_aspect_sentiment_classes
        )

        output = {
            "sentiment_logits": sentiment_logits,
            "aspect_logits": aspect_logits,
        }

        if sentiment_labels is not None and aspect_labels is not None:
            # Sentiment loss
            sentiment_loss = self.sentiment_loss_fn(sentiment_logits, sentiment_labels)

            # Aspect loss: masked CE
            if aspect_mask is not None:
                aspect_loss = self._compute_masked_aspect_loss(
                    aspect_logits, aspect_labels, aspect_mask
                )
            else:
                # 기존 방식 (모든 셀 학습, 하위 호환)
                aspect_loss = self.aspect_loss_fn(
                    aspect_logits.view(-1, self.num_aspect_sentiment_classes),
                    aspect_labels.view(-1)
                )

            output["sentiment_loss"] = sentiment_loss
            output["aspect_loss"] = aspect_loss

        return output

    def _compute_masked_aspect_loss(self, aspect_logits, aspect_labels, aspect_mask):
        """
        Masked CE loss: mask=1인 셀만 loss에 포함.

        aspect_logits: [B, 11, 4]
        aspect_labels: [B, 11]
        aspect_mask: [B, 11] (1=포함, 0=제외)
        """
        B = aspect_logits.size(0)

        # Class weights
        weight = None
        if self._aspect_class_weights is not None:
            weight = self._aspect_class_weights.to(aspect_logits.device)

        if self.use_focal_loss:
            # Focal loss with reduction='none'
            ce_loss = F.cross_entropy(
                aspect_logits.view(-1, self.num_aspect_sentiment_classes),
                aspect_labels.view(-1),
                reduction="none"
            )  # [B*11]
            pt = torch.exp(-ce_loss)
            focal_weight = (1 - pt) ** self.focal_gamma
            if weight is not None:
                alpha_t = weight[aspect_labels.view(-1)]
                focal_weight = alpha_t * focal_weight
            per_cell = focal_weight * ce_loss
        else:
            per_cell = F.cross_entropy(
                aspect_logits.view(-1, self.num_aspect_sentiment_classes),
                aspect_labels.view(-1),
                weight=weight,
                reduction="none"
            )  # [B*11]

        per_cell = per_cell.view(B, self.num_aspect_labels)  # [B, 11]
        mask_f = aspect_mask.float()
        masked_loss = (per_cell * mask_f).sum()
        mask_count = mask_f.sum().clamp(min=1.0)

        return masked_loss / mask_count

    def predict(self, input_ids, attention_mask):
        """
        추론 모드.

        Returns:
            dict with:
                sentiment_preds: [B]
                sentiment_probs: [B, 3]
                aspect_preds: [B, 11] (각 값 0~3)
                aspect_probs: [B, 11, 4]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)

            # Sentiment
            sentiment_probs = torch.softmax(outputs["sentiment_logits"], dim=-1)
            sentiment_preds = torch.argmax(sentiment_probs, dim=-1)

            # Aspect: softmax over 4 classes per aspect → argmax
            aspect_probs = torch.softmax(outputs["aspect_logits"], dim=-1)  # [B, 11, 4]
            aspect_preds = torch.argmax(aspect_probs, dim=-1)  # [B, 11]

            return {
                "sentiment_preds": sentiment_preds,
                "sentiment_probs": sentiment_probs,
                "aspect_preds": aspect_preds,
                "aspect_probs": aspect_probs,
            }

    def get_sentiment_score(self, sentiment_probs):
        """
        Sentiment score = P(positive) - P(negative)
        """
        neg_prob = sentiment_probs[:, 0]
        pos_prob = sentiment_probs[:, 2]
        return pos_prob - neg_prob


def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Balanced class weights: total / (num_classes * class_count)
    """
    class_counts = torch.bincount(labels, minlength=num_classes).float()
    total = class_counts.sum()
    weights = total / (num_classes * class_counts)
    weights[class_counts == 0] = 0.0
    return weights


def compute_aspect_class_weights(
    aspect_labels: torch.Tensor,
    num_classes: int = 4,
    aspect_masks: torch.Tensor = None
) -> torch.Tensor:
    """
    Aspect-level class weights 계산.
    mask가 제공되면 mask=1인 셀만 사용하여 가중치 계산.
    """
    if aspect_masks is not None:
        # mask=1인 셀만 사용
        mask_flat = aspect_masks.view(-1).bool()
        flat_labels = aspect_labels.view(-1)[mask_flat]
    else:
        flat_labels = aspect_labels.view(-1)
    return compute_class_weights(flat_labels, num_classes)


def load_model(
    checkpoint_path: str,
    model_name: str = "beomi/KcELECTRA-base",
    num_sentiment_labels: int = 3,
    num_aspect_labels: int = None,
    num_aspect_sentiment_classes: int = 4,
    device: str = None,
):
    """모델 체크포인트 로드.

    num_aspect_labels 결정 우선순위:
    1) 명시적으로 전달된 값
    2) checkpoint의 label_meta.aspect_labels 길이
    3) 현재 s1_config.ASPECT_LABELS 길이 (fallback)
    """
    if device is None:
        device = get_best_device()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # num_aspect_labels 자동 결정
    if num_aspect_labels is None:
        label_meta = checkpoint.get("label_meta", {})
        ckpt_aspects = label_meta.get("aspect_labels")
        if ckpt_aspects is not None:
            num_aspect_labels = len(ckpt_aspects)
            if num_aspect_labels != len(ASPECT_LABELS):
                print(f"  NOTE: checkpoint aspects({num_aspect_labels}) != "
                      f"config aspects({len(ASPECT_LABELS)}), "
                      f"using checkpoint value")
        else:
            num_aspect_labels = len(ASPECT_LABELS)

    model = MultiTaskABSAModel(
        model_name=model_name,
        num_sentiment_labels=num_sentiment_labels,
        num_aspect_labels=num_aspect_labels,
        num_aspect_sentiment_classes=num_aspect_sentiment_classes
    )

    result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if result.unexpected_keys:
        print(f"  무시된 키(loss weight 등): {result.unexpected_keys}")
    model.to(device)
    model.eval()

    # checkpoint의 aspect_labels를 모델에 부착 (source of truth)
    label_meta = checkpoint.get("label_meta", {})
    ckpt_aspects = label_meta.get("aspect_labels")
    model.aspect_labels = ckpt_aspects if ckpt_aspects else list(ASPECT_LABELS)

    print(f"Loaded model from: {checkpoint_path}")
    print(f"  num_aspect_labels: {num_aspect_labels}, aspects: {model.aspect_labels}")
    if "epoch" in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if "val_metrics" in checkpoint:
        print(f"  Val metrics: {checkpoint['val_metrics']}")

    return model
