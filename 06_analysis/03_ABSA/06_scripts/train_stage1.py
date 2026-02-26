"""
ABSA Stage 2 학습 스크립트

Wide CSV → KcELECTRA 멀티태스크 모델 학습
- Sentiment (3-class) + Aspect-Sentiment (8 aspects × 4-class)
- 미분류·CS/응대·품질/퀄리티 제거, negative sampling 보강 데이터 사용
- Threshold tuning: F0.5 (precision 가중, trainer.train() 내부에서 자동 실행)
- 학습 완료 후 best_model.pt + none_thresholds 자동 저장

데이터 로드:
  (1) augment_negative_samples.py가 생성한 pre-split 파일 우선 사용
      → absa_wide_train_augmented.csv + absa_wide_val.csv
  (2) pre-split 없으면 단일 CSV에서 자동 분할 (fallback)

옵션:
  --warm-start <path>   Stage1 체크포인트에서 encoder만 로드 (head는 랜덤 init)
"""
import sys
import argparse
from pathlib import Path

ABSA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ABSA_ROOT))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from RQ_absa.s1_config import (
    TRAIN_CONFIG,
    SPLIT_RATIOS,
    PROCESSED_DATA_DIR,
    CHECKPOINT_DIR,
)
from RQ_absa.s4_dataset import create_dataset_from_wide, create_datasets_from_wide
from RQ_absa.s6_train import ABSATrainer, create_model_with_class_weights


def get_device() -> str:
    """MPS > CUDA > CPU 순으로 디바이스 결정"""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_encoder_from_checkpoint(model, checkpoint_path: str, device: str):
    """Stage1 체크포인트에서 encoder 가중치만 로드 (warm-start).

    head(sentiment_classifier, aspect_classifier)는 shape이 다를 수 있으므로 스킵.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]

    # encoder.* 키만 필터
    encoder_keys = {k: v for k, v in state_dict.items() if k.startswith("encoder.")}
    skipped_keys = [k for k in state_dict if not k.startswith("encoder.")]

    missing, unexpected = model.load_state_dict(encoder_keys, strict=False)
    print(f"Warm-start: encoder {len(encoder_keys)}개 파라미터 로드")
    print(f"  스킵 (head): {skipped_keys}")
    print(f"  missing (새 head): {[k for k in missing if not k.startswith('encoder.')]}")

    if "epoch" in ckpt:
        print(f"  Source checkpoint: epoch {ckpt['epoch']}")
    label_meta = ckpt.get("label_meta", {})
    src_aspects = label_meta.get("aspect_labels")
    if src_aspects:
        print(f"  Source aspects ({len(src_aspects)}): {src_aspects}")


def main():
    parser = argparse.ArgumentParser(description="ABSA Stage 2 학습")
    parser.add_argument(
        "--warm-start", type=str, default=None,
        help="Stage1 체크포인트 경로 (encoder만 로드, head는 랜덤 init)",
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
    data_dir = PROCESSED_DATA_DIR / "training"
    train_path = data_dir / "absa_wide_train_augmented.csv"
    val_path = data_dir / "absa_wide_val.csv"

    if train_path.exists() and val_path.exists():
        # Pre-split 파일 사용 (augment가 train에만 적용된 상태)
        print(f"\n[Pre-split 모드] augmented train + clean val 로드")
        print(f"  Train: {train_path}")
        print(f"  Val:   {val_path}")
        train_dataset = create_dataset_from_wide(train_path, tokenizer, cfg["max_length"])
        val_dataset = create_dataset_from_wide(val_path, tokenizer, cfg["max_length"])
    else:
        # Fallback: 단일 CSV에서 자동 분할
        print(f"\n[Fallback 모드] 단일 CSV에서 자동 분할")
        wide_csv_path = data_dir / "absa_wide_train.csv"
        assert wide_csv_path.exists(), f"학습 데이터 없음: {wide_csv_path}"
        print(f"  데이터: {wide_csv_path}")
        train_dataset, val_dataset, _ = create_datasets_from_wide(
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

    # ── Warm-start: Stage1 encoder 로드 ──
    if args.warm_start:
        print(f"\n[Warm-start] Stage1 encoder 로드: {args.warm_start}")
        load_encoder_from_checkpoint(model, args.warm_start, device)

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
    # trainer.train()이 끝나면 _load_best_and_tune_thresholds()가 자동 실행됨
    # (THRESHOLD_TUNING_CONFIG 기준 F0.5 metric + polar_threshold 저장)
    trainer.train(
        num_epochs=cfg["num_epochs"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        logging_steps=cfg["logging_steps"],
        eval_steps=cfg["eval_steps"],
        save_steps=cfg["save_steps"],
    )

    print("\n학습 완료!")
    print(f"  Best model: {CHECKPOINT_DIR / 'best_model.pt'}")
    print(f"  Thresholds: {CHECKPOINT_DIR / 'none_thresholds.json'}")
    print(f"  History: {CHECKPOINT_DIR / 'training_history.json'}")
    if args.warm_start:
        print(f"  Mode: warm-start from {args.warm_start}")
    else:
        print(f"  Mode: scratch (KcELECTRA base)")


if __name__ == "__main__":
    main()
