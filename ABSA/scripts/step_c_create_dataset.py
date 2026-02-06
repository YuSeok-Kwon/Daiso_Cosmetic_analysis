"""
Step C: Create train/val/test datasets
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from RQ.absa.dataset import ABSADataProcessor
from RQ.absa.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SPLIT_RATIOS,
    ASPECT_LABELS,
    SENTIMENT_LABELS
)


def main():
    """Main function for creating datasets"""
    print("="*60)
    print("STEP C: CREATE DATASETS")
    print("="*60)

    # Paths
    input_path = RAW_DATA_DIR / "chatgpt_labels_20k.jsonl"

    # Check if input exists
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        print("\nPlease run step_b_labeling.py first.")
        return

    # Initialize processor
    processor = ABSADataProcessor(
        sentiment_labels=SENTIMENT_LABELS,
        aspect_labels=ASPECT_LABELS
    )

    # Load and encode
    df = processor.load_labeled_data(input_path)
    df = processor.encode_labels(df)

    # Split
    train_df, val_df, test_df = processor.split_data(
        df,
        train_ratio=SPLIT_RATIOS['train'],
        val_ratio=SPLIT_RATIOS['val'],
        test_ratio=SPLIT_RATIOS['test'],
        random_state=42
    )

    # Save
    processor.save_splits(train_df, val_df, test_df, PROCESSED_DATA_DIR)

    print("\n" + "="*60)
    print("STEP C COMPLETE")
    print("="*60)
    print(f"Train: {len(train_df):,} samples")
    print(f"Val:   {len(val_df):,} samples")
    print(f"Test:  {len(test_df):,} samples")
    print(f"Output directory: {PROCESSED_DATA_DIR}")
    print("\nNext step: Run step_d_train.py")
    print("  python scripts/absa/step_d_train.py")
    print("="*60)


if __name__ == "__main__":
    main()
