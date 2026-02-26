"""
ABSA Stage 4B 학습 스크립트 — Warm-start from Stage 2

전략 A: Stage 2 best_model.pt에서 warm-start하여
Stage 4 보강 데이터로 fine-tuning, 학습 후 Stage 3A-v2 후처리 파이프라인 적용.

Stage 4 from-scratch와의 차이:
  1. Stage 2 best_model.pt의 encoder + head를 로드 (warm-start)
  2. lr=5e-6 (Stage 2의 1/4)
  3. 5 epoch만 fine-tuning (과적합 방지)
  4. 체크포인트: checkpoints_stage4b/
  5. threshold tuning 후 Stage 3A-v2 threshold 복원 옵션
"""
import sys
import shutil
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


# 경로
STAGE4_DATA_DIR = PROCESSED_DATA_DIR / "training" / "stage4"
STAGE2_CHECKPOINT_DIR = MODEL_ROOT / "checkpoints"
STAGE4B_CHECKPOINT_DIR = MODEL_ROOT / "checkpoints_stage4b"

# Warm-start 하이퍼파라미터
WARMSTART_CONFIG = {
    "learning_rate": 5e-6,   # Stage 2 (2e-5)의 1/4
    "num_epochs": 5,         # 짧게 (과적합 방지)
    "warmup_ratio": 0.1,
}


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_stage2_weights(model, stage2_ckpt_path: Path, device: str):
    """Stage 2 best_model.pt에서 모델 가중치를 로드 (warm-start)"""
    print(f"\n[Warm-start] Stage 2 체크포인트 로드: {stage2_ckpt_path}")

    ckpt = torch.load(stage2_ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # 현재 모델의 state_dict 키와 비교
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    matched = model_keys & ckpt_keys
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    print(f"  매칭 파라미터: {len(matched)}/{len(model_keys)}")
    if missing:
        print(f"  누락 (random init): {len(missing)} — {list(missing)[:5]}")
    if unexpected:
        print(f"  무시 (checkpoint only): {len(unexpected)} — {list(unexpected)[:5]}")

    # strict=False로 로드 (일부 키 불일치 허용)
    model.load_state_dict(state_dict, strict=False)
    print(f"  ✓ Stage 2 가중치 로드 완료")

    # 에포크 정보 출력
    if "epoch" in ckpt:
        print(f"  Stage 2 best epoch: {ckpt['epoch']}")

    return model


def main():
    parser = argparse.ArgumentParser(description="ABSA Stage 4B 학습 (Warm-start)")
    parser.add_argument(
        "--skip-golden-eval", action="store_true",
        help="학습 완료 후 골든셋 평가 건너뛰기",
    )
    parser.add_argument(
        "--epochs", type=int, default=WARMSTART_CONFIG["num_epochs"],
        help=f"학습 에포크 수 (기본: {WARMSTART_CONFIG['num_epochs']})",
    )
    parser.add_argument(
        "--lr", type=float, default=WARMSTART_CONFIG["learning_rate"],
        help=f"학습률 (기본: {WARMSTART_CONFIG['learning_rate']})",
    )
    parser.add_argument(
        "--restore-stage3a-thresholds", action="store_true",
        help="학습 후 threshold tuning 대신 Stage 3A-v2 threshold 복원",
    )
    args = parser.parse_args()

    # ── 설정 ──
    cfg = TRAIN_CONFIG.copy()
    cfg["learning_rate"] = args.lr
    cfg["num_epochs"] = args.epochs
    cfg["warmup_ratio"] = WARMSTART_CONFIG["warmup_ratio"]

    device = get_device()
    print(f"Device: {device}")
    print(f"Mode: Warm-start from Stage 2")
    print(f"  lr: {cfg['learning_rate']}")
    print(f"  epochs: {cfg['num_epochs']}")
    print(f"  warmup_ratio: {cfg['warmup_ratio']}")

    # ── 토크나이저 ──
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    # ── 데이터 로드 ──
    train_path = STAGE4_DATA_DIR / "absa_wide_train_stage4.csv"
    val_path = STAGE4_DATA_DIR / "absa_wide_val_stage4.csv"

    assert train_path.exists(), f"Stage 4 train 데이터 없음: {train_path}"
    assert val_path.exists(), f"Stage 4 val 데이터 없음: {val_path}"

    print(f"\n[데이터 로드]")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    train_dataset = create_dataset_from_wide(train_path, tokenizer, cfg["max_length"])
    val_dataset = create_dataset_from_wide(val_path, tokenizer, cfg["max_length"])

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=0, pin_memory=False,
    )

    print(f"Epochs: {cfg['num_epochs']}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # ── 모델 생성 (class weights 자동 계산) ──
    model = create_model_with_class_weights(
        train_dataset=train_dataset,
        model_name=cfg["model_name"],
        dropout=cfg["dropout"],
        use_class_weight=cfg["use_class_weight"],
        use_focal_loss=cfg["use_focal_loss"],
        focal_gamma=cfg["focal_gamma"],
    )

    # ── Stage 2 가중치 로드 (warm-start) ──
    stage2_ckpt = STAGE2_CHECKPOINT_DIR / "best_model.pt"
    assert stage2_ckpt.exists(), f"Stage 2 체크포인트 없음: {stage2_ckpt}"
    model = load_stage2_weights(model, stage2_ckpt, device)

    # ── Trainer 초기화 ──
    STAGE4B_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

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
        checkpoint_dir=STAGE4B_CHECKPOINT_DIR,
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
    print(f"  Best model: {STAGE4B_CHECKPOINT_DIR / 'best_model.pt'}")
    print(f"  Thresholds: {STAGE4B_CHECKPOINT_DIR / 'none_thresholds.json'}")
    print(f"  Mode: warm-start (Stage 2 → Stage 4B, lr={cfg['learning_rate']})")

    # ── Stage 3A-v2 threshold 복원 (선택) ──
    if args.restore_stage3a_thresholds:
        stage3a_thresholds = STAGE2_CHECKPOINT_DIR / "none_thresholds_stage3a.json"
        target_thresholds = STAGE4B_CHECKPOINT_DIR / "none_thresholds.json"

        if stage3a_thresholds.exists():
            shutil.copy2(stage3a_thresholds, target_thresholds)
            print(f"\n  ✓ Stage 3A-v2 threshold 복원: {stage3a_thresholds.name}")
        else:
            print(f"\n  ⚠ Stage 3A threshold 없음: {stage3a_thresholds}")

    # ── 골든셋 평가 자동 실행 ──
    if not args.skip_golden_eval:
        print("\n" + "=" * 60)
        print("골든셋 평가 자동 실행 (Stage 4B 체크포인트)")
        print("=" * 60)

        eval_script = ABSA_ROOT / "06_scripts" / "eval_golden.py"
        python = sys.executable

        cmd = [
            python, str(eval_script),
            "--split", "test",
            "--polar",
            "--postprocess",
            "--checkpoint-dir", str(STAGE4B_CHECKPOINT_DIR),
        ]
        print(f"  실행: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print(f"\n⚠️ 골든셋 평가 실패 (returncode={result.returncode})")
            print("  수동 실행:")
            print(f"  {python} {eval_script} --split test --polar --postprocess \\")
            print(f"    --checkpoint-dir {STAGE4B_CHECKPOINT_DIR}")
        else:
            print("\n골든셋 평가 완료!")


if __name__ == "__main__":
    main()
