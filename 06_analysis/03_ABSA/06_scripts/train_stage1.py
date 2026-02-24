"""
ABSA Stage 2 학습 스크립트

Wide CSV (absa_wide_train_augmented.csv) → KcELECTRA 멀티태스크 모델 학습
- Sentiment (3-class) + Aspect-Sentiment (10 aspects × 4-class)
- 미분류 제거, negative sampling 보강 데이터 사용
- Threshold tuning: F0.5 (precision 가중)
- 학습 완료 후 best_model.pt + none_thresholds 자동 저장
"""
import sys
from pathlib import Path

# ABSA 프로젝트 루트를 sys.path에 추가 (RQ_absa import 경로)
ABSA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ABSA_ROOT))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from RQ_absa.s1_config import (
    TRAIN_CONFIG,
    SPLIT_RATIOS,
    THRESHOLD_TUNING_CONFIG,
    PROCESSED_DATA_DIR,
    CHECKPOINT_DIR,
)
from RQ_absa.s4_dataset import create_datasets_from_wide
from RQ_absa.s6_train import ABSATrainer, create_model_with_class_weights


def get_device() -> str:
    """MPS > CUDA > CPU 순으로 디바이스 결정"""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    # ── 설정 ──
    cfg = TRAIN_CONFIG
    device = get_device()
    print(f"Device: {device}")
    print(f"Config: {cfg}")

    # ── 데이터 경로 (augmented 데이터 우선, 없으면 원본) ──
    wide_csv_path = PROCESSED_DATA_DIR / "final" / "absa" / "absa_wide_train_augmented.csv"
    if not wide_csv_path.exists():
        wide_csv_path = PROCESSED_DATA_DIR / "final" / "absa" / "absa_wide_train.csv"
    assert wide_csv_path.exists(), f"학습 데이터 없음: {wide_csv_path}"
    print(f"\n학습 데이터: {wide_csv_path}")

    # ── 토크나이저 ──
    print(f"\n토크나이저 로드: {cfg['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    # ── 데이터셋 생성 (Wide CSV → Train/Val/Test) ──
    print("\n데이터셋 생성 중...")
    train_dataset, val_dataset, test_dataset = create_datasets_from_wide(
        wide_csv_path=wide_csv_path,
        tokenizer=tokenizer,
        max_length=cfg["max_length"],
        train_ratio=SPLIT_RATIOS["train"],
        val_ratio=SPLIT_RATIOS["val"],
        test_ratio=SPLIT_RATIOS["test"],
        random_state=cfg["seed"],
    )

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
        checkpoint_dir=CHECKPOINT_DIR,
    )

    # ── 학습 시작 ──
    trainer.train(
        num_epochs=cfg["num_epochs"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        logging_steps=cfg["logging_steps"],
        eval_steps=cfg["eval_steps"],
        save_steps=cfg["save_steps"],
    )

    # ── Threshold 재튜닝 (F0.5 metric) ──
    th_cfg = THRESHOLD_TUNING_CONFIG
    print(f"\nThreshold 튜닝: metric={th_cfg['metric']}, beta={th_cfg.get('beta', 0.5)}")
    trainer.tune_thresholds(
        search_range=th_cfg["search_range"],
        search_step=th_cfg["search_step"],
        metric=th_cfg["metric"],
        beta=th_cfg.get("beta", 0.5),
    )

    print("\n학습 완료!")
    print(f"  Best model: {CHECKPOINT_DIR / 'best_model.pt'}")
    print(f"  Thresholds: {CHECKPOINT_DIR / 'none_thresholds.json'}")
    print(f"  History: {CHECKPOINT_DIR / 'training_history.json'}")


if __name__ == "__main__":
    main()
