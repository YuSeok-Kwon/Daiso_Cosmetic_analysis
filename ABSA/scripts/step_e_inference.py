"""
Step E: Run inference on full dataset
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from RQ.absa.inference import run_inference_on_reviews
from RQ.absa.config import (
    DATA_ROOT,
    INFERENCE_DATA_DIR,
    CHECKPOINT_DIR,
    TRAIN_CONFIG,
    INFERENCE_CONFIG
)


def main():
    """Main function for inference"""
    print("="*60)
    print("STEP E: INFERENCE ON FULL DATASET")
    print("="*60)

    # Paths
    input_path = DATA_ROOT / "csv" / "reviews.csv"
    output_path = INFERENCE_DATA_DIR / "reviews_with_absa.csv"
    model_path = CHECKPOINT_DIR / "best_model.pt"

    # Check if inputs exist
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        print("\nPlease ensure you have the reviews.csv file.")
        return

    if not model_path.exists():
        print(f"Error: Model checkpoint not found: {model_path}")
        print("\nPlease run step_d_train.py first.")
        return

    # Run inference
    results_df = run_inference_on_reviews(
        input_path=input_path,
        output_path=output_path,
        model_path=model_path,
        model_name=TRAIN_CONFIG['model_name'],
        batch_size=INFERENCE_CONFIG['batch_size'],
        aspect_threshold=INFERENCE_CONFIG['aspect_threshold']
    )

    print("\n" + "="*60)
    print("STEP E COMPLETE")
    print("="*60)
    print(f"Processed {len(results_df):,} reviews")
    print(f"Output: {output_path}")
    print(f"Ambiguous samples: {output_path.parent / (output_path.stem + '_ambiguous.csv')}")
    print("\n" + "="*60)
    print("ABSA PIPELINE COMPLETE!")
    print("="*60)
    print("\nYou can now use the results for analysis.")
    print(f"\nMain output: {output_path}")
    print("\nColumns:")
    print("  - sentiment: positive/neutral/negative")
    print("  - sentiment_score: -1.0 to 1.0")
    print("  - aspect_labels: list of detected aspects")
    print("  - evidence: supporting text")
    print("  - summary: 1-sentence summary")
    print("  - is_ambiguous: whether the sample is ambiguous")
    print("="*60)


if __name__ == "__main__":
    main()
