"""
Check ABSA setup and dependencies
"""
import sys
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")

    dependencies = {
        'torch': '2.0.0',
        'transformers': '4.36.0',
        'openai': '1.12.0',
        'sklearn': '1.2.0',
        'pandas': '1.5.0',
        'numpy': '1.23.0',
        'tqdm': '4.65.0'
    }

    missing = []
    outdated = []

    for package, min_version in dependencies.items():
        try:
            if package == 'sklearn':
                import sklearn
                pkg = sklearn
                pkg_name = 'scikit-learn'
            else:
                pkg = __import__(package)
                pkg_name = package

            version = getattr(pkg, '__version__', 'unknown')
            print(f"  {pkg_name}: {version}")

            if version != 'unknown' and version < min_version:
                outdated.append((pkg_name, version, min_version))

        except ImportError:
            print(f"  {package}: NOT FOUND")
            missing.append(package)

    if missing:
        print("\nMissing packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with:")
        print("  pip install -r requirements_absa.txt")
        return False

    if outdated:
        print("\nOutdated packages:")
        for pkg, current, required in outdated:
            print(f"  - {pkg}: {current} (required: >={required})")
        print("\nUpgrade with:")
        print("  pip install --upgrade -r requirements_absa.txt")

    return True


def check_gpu():
    """Check GPU availability"""
    print("\nChecking GPU...")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"  GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("  GPU not available")
            print("  WARNING: Training will be very slow on CPU")
            return False
    except ImportError:
        print("  Cannot check (PyTorch not installed)")
        return False


def check_openai_key():
    """Check OpenAI API key"""
    print("\nChecking OpenAI API key...")

    import os
    if 'OPENAI_API_KEY' in os.environ:
        key = os.environ['OPENAI_API_KEY']
        masked = key[:8] + '...' + key[-4:]
        print(f"  API key found: {masked}")
        return True
    else:
        print("  API key not found")
        print("  Set with: export OPENAI_API_KEY='your-key-here'")
        return False


def check_data():
    """Check if data files exist"""
    print("\nChecking data files...")

    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "csv" / "reviews.csv"

    if data_path.exists():
        import pandas as pd
        df = pd.read_csv(data_path)
        print(f"  reviews.csv found: {len(df):,} reviews")

        # Check required columns
        required_cols = ['text', 'product_code', 'rating']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"  ERROR: Missing required columns: {missing_cols}")
            return False

        # Check for year or date column
        if 'year' not in df.columns and 'date' not in df.columns:
            print(f"  ERROR: Need 'year' or 'date' column for stratification")
            return False

        if 'year' not in df.columns and 'date' in df.columns:
            print(f"  'year' column missing but 'date' found (will auto-extract)")

        print(f"  All required columns present")
        return True
    else:
        print(f"  reviews.csv not found at: {data_path}")
        print("  Please ensure your data is in the correct location")
        return False


def check_directories():
    """Check if required directories exist"""
    print("\nChecking directories...")

    project_root = Path(__file__).parent.parent.parent

    dirs = {
        'data/absa/raw': 'Raw data (sampled and labeled)',
        'data/absa/processed': 'Processed datasets',
        'data/absa/inference': 'Inference results',
        'data/absa/cache': 'API cache',
        'models/absa/checkpoints': 'Model checkpoints',
        'scripts/absa': 'Scripts'
    }

    all_exist = True
    for dir_path, description in dirs.items():
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"  {dir_path}: OK")
        else:
            print(f"  {dir_path}: MISSING (will be created)")
            full_path.mkdir(parents=True, exist_ok=True)
            all_exist = False

    return all_exist


def main():
    """Main setup check"""
    print("="*60)
    print("ABSA SETUP CHECK")
    print("="*60)

    results = {
        'dependencies': check_dependencies(),
        'gpu': check_gpu(),
        'openai_key': check_openai_key(),
        'data': check_data(),
        'directories': check_directories()
    }

    print("\n" + "="*60)
    print("SETUP SUMMARY")
    print("="*60)

    for check, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")

    all_passed = all(results.values())

    print("\n" + "="*60)
    if all_passed:
        print("✓ All checks passed!")
        print("\nYou can now run the pipeline:")
        print("  python scripts/absa/step_a_sampling.py")
    else:
        print("✗ Some checks failed")
        print("\nPlease fix the issues above before running the pipeline.")

        if not results['dependencies']:
            print("\n1. Install dependencies:")
            print("   pip install -r requirements_absa.txt")

        if not results['openai_key']:
            print("\n2. Set OpenAI API key:")
            print("   export OPENAI_API_KEY='your-key-here'")

        if not results['data']:
            print("\n3. Ensure reviews.csv is in data/csv/")

        if not results['gpu']:
            print("\nNote: GPU is recommended but not required for Steps A-C.")
            print("      GPU is required for Steps D-E (training and inference).")

    print("="*60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
