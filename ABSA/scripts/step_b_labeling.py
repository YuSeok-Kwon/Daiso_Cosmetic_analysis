"""
Step B: ChatGPT로 20k 리뷰 라벨링
"""
import sys
import os
from pathlib import Path

# Add RQ_absa to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from RQ_absa.labeling import ABSALabeler
from RQ_absa.config import (
    RAW_DATA_DIR,
    OPENAI_CONFIG
)


def main():
    """Main function for labeling"""
    print("="*60)
    print("STEP B: CHATGPT LABELING")
    print("="*60)

    # Check API key
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("\nPlease set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-key-here'")
        return

    # Paths
    input_path = RAW_DATA_DIR / "sampled_reviews_20k.csv"
    output_path = RAW_DATA_DIR / "chatgpt_labels_20k.jsonl"

    # Check if input exists
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        print("\nPlease run step_a_sampling.py first.")
        return

    # Initialize labeler
    labeler = ABSALabeler(
        model=OPENAI_CONFIG['model'],
        save_interval=100
    )

    # Label
    results_df = labeler.label_batch(
        input_path=input_path,
        output_path=output_path,
        resume=True
    )

    print("\n" + "="*60)
    print("STEP B COMPLETE")
    print("="*60)
    print(f"Labeled {len(results_df):,} reviews")
    print(f"Output: {output_path}")
    print("\nNext step: Run step_c_create_dataset.py")
    print("  python scripts/absa/step_c_create_dataset.py")
    print("="*60)


if __name__ == "__main__":
    main()
