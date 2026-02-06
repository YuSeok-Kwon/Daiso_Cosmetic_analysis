"""
Evaluate model on test set
"""
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer
from RQ.absa.model import load_model
from RQ.absa.dataset import create_datasets_from_csv
from RQ.absa.evaluation import evaluate_test_set, ABSAEvaluator
from RQ.absa.config import (
    PROCESSED_DATA_DIR,
    CHECKPOINT_DIR,
    TRAIN_CONFIG,
    ASPECT_LABELS,
    SENTIMENT_LABELS
)


def main():
    """Main function for evaluation"""
    print("="*60)
    print("EVALUATE MODEL ON TEST SET")
    print("="*60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Paths
    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"
    model_path = CHECKPOINT_DIR / "best_model.pt"

    # Check if inputs exist
    for path in [train_path, val_path, test_path]:
        if not path.exists():
            print(f"Error: Input file not found: {path}")
            return

    if not model_path.exists():
        print(f"Error: Model checkpoint not found: {model_path}")
        return

    # Load tokenizer and datasets
    print("\nLoading tokenizer and datasets...")
    tokenizer = AutoTokenizer.from_pretrained(TRAIN_CONFIG['model_name'])

    _, _, test_dataset = create_datasets_from_csv(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        tokenizer=tokenizer,
        max_length=TRAIN_CONFIG['max_length']
    )

    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )

    # Load model
    print("\nLoading model...")
    model = load_model(
        checkpoint_path=model_path,
        model_name=TRAIN_CONFIG['model_name'],
        num_sentiment_labels=len(SENTIMENT_LABELS),
        num_aspect_labels=len(ASPECT_LABELS),
        device=device
    )

    # Evaluate
    print("\nEvaluating...")
    evaluator = ABSAEvaluator(
        sentiment_labels=SENTIMENT_LABELS,
        aspect_labels=ASPECT_LABELS
    )

    metrics = evaluate_test_set(
        test_loader=test_loader,
        model=model,
        device=device,
        evaluator=evaluator
    )

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
