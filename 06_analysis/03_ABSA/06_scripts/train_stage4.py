"""
ABSA Stage 4 학습 스크립트

Stage 4: neutral 보충 데이터 병합 후 from-scratch 재학습
- 데이터: stage4/absa_wide_train_stage4.csv + stage4/absa_wide_val_stage4.csv
- 체크포인트: 07_models/checkpoints_stage4/
- 학습 후: threshold tuning (F0.5) + golden_test 자동 평가

Stage 2(train_stage1.py)와의 차이:
  1. 데이터 경로: final/absa/stage4/
  2. 체크포인트 경로: checkpoints_stage4/
  3. 학습 완료 후 골든셋 평가 자동 실행
"""
import sys
import subprocess
import argparse
from pathlib import Path

ABSA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ABSA_ROOT))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from RQ_absa.s1_config import (
    TRAIN_CONFIG,
    PROCESSED_DATA_DIR,
    MODEL_ROOT,
)
from RQ_absa.s4_dataset import create_dataset_from_wide
from RQ_absa.s6_train import ABSATrainer, create_model_with_class_weights


# Stage 4 전용 경로
STAGE4_DATA_DIR = PROCESSED_DATA_DIR / "training" / "stage4"
STAGE4_CHECKPOINT_DIR = MODEL_ROOT / "checkpoints_stage4"


def get_device() -> str:
    """MPS > CUDA > CPU 순으로 디바이스 결정"""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="ABSA Stage 4 학습")
    parser.add_argument(
        "--skip-golden-eval", action="store_true",
        help="학습 완료 후 골든셋 평가 건너뛰기",
    )
    args = parser.parse_args()

    # ── 설정 ──
    cfg = TRAIN_CONFIG
    device = get_device()
    print(f"Device: {device}")
    print(f"Config: {cfg}")

    # ── 토크나이저 ──
    print(f"\n토크나이저 로드: {cfg['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    # ── 데이터 로드 ──
    train_path = STAGE4_DATA_DIR / "absa_wide_train_stage4.csv"
    val_path = STAGE4_DATA_DIR / "absa_wide_val_stage4.csv"

    assert train_path.exists(), f"Stage 4 train 데이터 없음: {train_path}"
    assert val_path.exists(), f"Stage 4 val 데이터 없음: {val_path}"

    print(f"\n[Stage 4 데이터 로드]")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    train_dataset = create_dataset_from_wide(train_path, tokenizer, cfg["max_length"])
    val_dataset = create_dataset_from_wide(val_path, tokenizer, cfg["max_length"])

    # ── DataLoader ──
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print(f"\nDataLoader: train={len(train_loader)} batches, val={len(val_loader)} batches")

    # ── 모델 생성 (class weights 자동 계산) ──
    print("\n모델 생성 중...")
    model = create_model_with_class_weights(
        train_dataset=train_dataset,
        model_name=cfg["model_name"],
        dropout=cfg["dropout"],
        use_class_weight=cfg["use_class_weight"],
        use_focal_loss=cfg["use_focal_loss"],
        focal_gamma=cfg["focal_gamma"],
    )

    # ── Trainer 초기화 ──
    STAGE4_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    trainer = ABSATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        sentiment_weight=cfg["sentiment_weight"],
        aspect_weight=cfg["aspect_weight"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        max_grad_norm=cfg["max_grad_norm"],
        device=device,
        checkpoint_dir=STAGE4_CHECKPOINT_DIR,
    )

    # ── 학습 시작 ──
    trainer.train(
        num_epochs=cfg["num_epochs"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        logging_steps=cfg["logging_steps"],
        eval_steps=cfg["eval_steps"],
        save_steps=cfg["save_steps"],
    )

    print("\n학습 완료!")
    print(f"  Best model: {STAGE4_CHECKPOINT_DIR / 'best_model.pt'}")
    print(f"  Thresholds: {STAGE4_CHECKPOINT_DIR / 'none_thresholds.json'}")
    print(f"  History: {STAGE4_CHECKPOINT_DIR / 'training_history.json'}")
    print(f"  Mode: scratch (KcELECTRA base, Stage 4 neutral 보강 데이터)")

    # ── 골든셋 평가 자동 실행 ──
    if not args.skip_golden_eval:
        print("\n" + "=" * 60)
        print("골든셋 평가 자동 실행 (Stage 4 체크포인트)")
        print("=" * 60)

        eval_script = ABSA_ROOT / "06_scripts" / "eval_golden.py"
        python = sys.executable

        cmd = [
            python, str(eval_script),
            "--split", "test",
            "--polar",
            "--postprocess",
            "--checkpoint-dir", str(STAGE4_CHECKPOINT_DIR),
        ]
        print(f"  실행: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print(f"\n⚠️ 골든셋 평가 실패 (returncode={result.returncode})")
            print("  수동 실행:")
            print(f"  {python} {eval_script} --split test --polar --postprocess \\")
            print(f"    --checkpoint-dir {STAGE4_CHECKPOINT_DIR}")
        else:
            print("\n골든셋 평가 완료!")


if __name__ == "__main__":
    main()
