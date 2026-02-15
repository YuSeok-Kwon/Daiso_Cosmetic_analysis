"""
Multi-task ABSA model with class weight / focal loss support
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Class weights [num_classes]. None for uniform weights.
            gamma: Focusing parameter. gamma=0 is standard CE loss.
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits [batch_size, num_classes]
            targets: Labels [batch_size]

        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t = softmax probability of correct class

        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_weight = alpha_t * focal_weight

        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiTaskABSAModel(nn.Module):
    """
    Multi-task model for ABSA:
    - Sentiment classification (3-class)
    - Aspect detection (9-class multi-label)

    Supports class weight and focal loss for imbalanced data.
    """

    def __init__(
        self,
        model_name: str = "beomi/KcELECTRA-base",
        num_sentiment_labels: int = 3,
        num_aspect_labels: int = 9,
        dropout: float = 0.1,
        sentiment_class_weights: Optional[torch.Tensor] = None,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0
    ):
        """
        Args:
            model_name: Pretrained model name
            num_sentiment_labels: Number of sentiment labels
            num_aspect_labels: Number of aspect labels
            dropout: Dropout rate
            sentiment_class_weights: Class weights for sentiment [num_sentiment_labels]
            use_focal_loss: Whether to use focal loss instead of CE
            focal_gamma: Gamma parameter for focal loss
        """
        super().__init__()

        self.model_name = model_name
        self.num_sentiment_labels = num_sentiment_labels
        self.num_aspect_labels = num_aspect_labels
        self.use_focal_loss = use_focal_loss

        # Load pretrained model
        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        self.hidden_size = config.hidden_size

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Sentiment classification head
        self.sentiment_classifier = nn.Linear(self.hidden_size, num_sentiment_labels)

        # Aspect detection head
        self.aspect_classifier = nn.Linear(self.hidden_size, num_aspect_labels)

        # Loss functions
        if use_focal_loss:
            self.sentiment_loss_fn = FocalLoss(
                alpha=sentiment_class_weights,
                gamma=focal_gamma
            )
            print(f"Using Focal Loss (gamma={focal_gamma})")
        else:
            self.sentiment_loss_fn = nn.CrossEntropyLoss(
                weight=sentiment_class_weights
            )
            if sentiment_class_weights is not None:
                print(f"Using CrossEntropyLoss with class weights: {sentiment_class_weights.tolist()}")
            else:
                print("Using CrossEntropyLoss without class weights")

        self.aspect_loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids,
        attention_mask,
        sentiment_labels=None,
        aspect_labels=None
    ):
        """
        Forward pass.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            sentiment_labels: Sentiment labels [batch_size] (optional)
            aspect_labels: Aspect labels [batch_size, num_aspects] (optional)

        Returns:
            Dictionary with logits and optionally loss
        """
        # Encode
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Get [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)

        # Sentiment logits
        sentiment_logits = self.sentiment_classifier(pooled_output)  # [batch_size, 3]

        # Aspect logits
        aspect_logits = self.aspect_classifier(pooled_output)  # [batch_size, 9]

        # Prepare output
        output = {
            'sentiment_logits': sentiment_logits,
            'aspect_logits': aspect_logits
        }

        # Calculate loss if labels are provided
        if sentiment_labels is not None and aspect_labels is not None:
            # Sentiment loss (with class weight or focal loss)
            sentiment_loss = self.sentiment_loss_fn(sentiment_logits, sentiment_labels)

            # Aspect loss (BCEWithLogits for multi-label)
            aspect_loss = self.aspect_loss_fn(aspect_logits, aspect_labels)

            output['sentiment_loss'] = sentiment_loss
            output['aspect_loss'] = aspect_loss

        return output

    def predict(
        self,
        input_ids,
        attention_mask,
        aspect_threshold: float = 0.5
    ):
        """
        Make predictions.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            aspect_threshold: Threshold for aspect prediction

        Returns:
            Dictionary with predictions
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)

            # Sentiment predictions
            sentiment_probs = torch.softmax(outputs['sentiment_logits'], dim=-1)
            sentiment_preds = torch.argmax(sentiment_probs, dim=-1)

            # Aspect predictions
            aspect_probs = torch.sigmoid(outputs['aspect_logits'])
            aspect_preds = (aspect_probs >= aspect_threshold).long()

            return {
                'sentiment_preds': sentiment_preds,
                'sentiment_probs': sentiment_probs,
                'aspect_preds': aspect_preds,
                'aspect_probs': aspect_probs
            }

    def get_sentiment_score(self, sentiment_probs):
        """
        Calculate sentiment score from probabilities.
        Score = P(positive) - P(negative)

        Args:
            sentiment_probs: Sentiment probabilities [batch_size, 3]

        Returns:
            Sentiment scores [batch_size]
        """
        # Assuming order: [negative, neutral, positive]
        neg_prob = sentiment_probs[:, 0]
        pos_prob = sentiment_probs[:, 2]
        scores = pos_prob - neg_prob
        return scores


def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Compute balanced class weights from labels.

    Args:
        labels: Label tensor [num_samples]
        num_classes: Number of classes

    Returns:
        Class weights [num_classes]
    """
    class_counts = torch.bincount(labels, minlength=num_classes).float()
    total = class_counts.sum()

    # Balanced weights: total / (num_classes * class_count)
    weights = total / (num_classes * class_counts)

    # Handle zero counts
    weights[class_counts == 0] = 0.0

    return weights


def load_model(
    checkpoint_path: str,
    model_name: str = "beomi/KcELECTRA-base",
    num_sentiment_labels: int = 3,
    num_aspect_labels: int = 9,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        model_name: Pretrained model name
        num_sentiment_labels: Number of sentiment labels
        num_aspect_labels: Number of aspect labels
        device: Device to load model on

    Returns:
        Loaded model
    """
    model = MultiTaskABSAModel(
        model_name=model_name,
        num_sentiment_labels=num_sentiment_labels,
        num_aspect_labels=num_aspect_labels
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded model from: {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'val_metrics' in checkpoint:
        print(f"Val metrics: {checkpoint['val_metrics']}")

    return model
