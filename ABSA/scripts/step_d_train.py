"""
Step D: Train ABSA model
"""
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer
from RQ.absa.model import MultiTaskABSAModel
from RQ.absa.dataset import create_datasets_from_csv
from RQ.absa.train import ABSATrainer
from RQ.absa.config import (
    PROCESSED_DATA_DIR,
    CHECKPOINT_DIR,
    TRAIN_CONFIG,
    ASPECT_LABELS,
    SENTIMENT_LABELS
)


def main():
    """Main function for training"""
    print("="*60)
    print("STEP D: TRAIN ABSA MODEL")
    print("="*60)

    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cpu":
        print("WARNING: Training on CPU will be very slow. Consider using a GPU.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting.")
            return

    # Paths
    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    # Check if inputs exist
    for path in [train_path, val_path, test_path]:
        if not path.exists():
            print(f"Error: Input file not found: {path}")
            print("\nPlease run step_c_create_dataset.py first.")
            return

    print("\nLoading tokenizer and creating datasets...")
    tokenizer = AutoTokenizer.from_pretrained(TRAIN_CONFIG['model_name'])

    train_dataset, val_dataset, test_dataset = create_datasets_from_csv(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        tokenizer=tokenizer,
        max_length=TRAIN_CONFIG['max_length']
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=True,
        num_workers=0,  # Set to 0 for compatibility
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    print("\nInitializing model...")
    model = MultiTaskABSAModel(
        model_name=TRAIN_CONFIG['model_name'],
        num_sentiment_labels=len(SENTIMENT_LABELS),
        num_aspect_labels=len(ASPECT_LABELS),
        dropout=TRAIN_CONFIG['dropout']
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize trainer
    trainer = ABSATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        sentiment_weight=TRAIN_CONFIG['sentiment_weight'],
        aspect_weight=TRAIN_CONFIG['aspect_weight'],
        learning_rate=TRAIN_CONFIG['learning_rate'],
        warmup_ratio=TRAIN_CONFIG['warmup_ratio'],
        weight_decay=TRAIN_CONFIG['weight_decay'],
        max_grad_norm=TRAIN_CONFIG['max_grad_norm'],
        device=device,
        checkpoint_dir=CHECKPOINT_DIR
    )

    # Train
    trainer.train(
        num_epochs=TRAIN_CONFIG['num_epochs'],
        gradient_accumulation_steps=TRAIN_CONFIG['gradient_accumulation_steps'],
        logging_steps=TRAIN_CONFIG['logging_steps'],
        eval_steps=TRAIN_CONFIG['eval_steps'],
        save_steps=TRAIN_CONFIG['save_steps']
    )

    print("\n" + "="*60)
    print("STEP D COMPLETE")
    print("="*60)
    print(f"Best model saved to: {CHECKPOINT_DIR / 'best_model.pt'}")
    print("\nNext step: Run step_e_inference.py")
    print("  python scripts/absa/step_e_inference.py")
    print("="*60)


if __name__ == "__main__":
    main()
