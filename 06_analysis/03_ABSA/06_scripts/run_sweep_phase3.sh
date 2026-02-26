#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ABSA Stage 5 Phase 3 — Focal Loss Sweep (전략 C)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Phase 2 best model (P2:lr1e5_ep5) 위에
# Focal Loss를 적용하여 head만 추가 fine-tuning.
#
# gamma 값에 따른 효과:
#   gamma=1.0: 약한 focal (CE 대비 미세한 차이)
#   gamma=1.5: 중간 focal (권장 시작점)
#   gamma=2.0: 표준 focal (대부분의 논문에서 사용)
#   gamma=3.0: 강한 focal (매우 어려운 샘플만 집중)
#
# head 파라미터: ~25K → config당 ~3-5분, 총 ~15분
#
# 사용법:
#   cd Why-pi/06_analysis/03_ABSA/06_scripts
#   chmod +x run_sweep_phase3.sh
#   ./run_sweep_phase3.sh
#
# 결과 확인:
#   python compare_sweeps.py --include-baselines
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ABSA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_stage5_phase3_focal.py"
PYTHON="/opt/miniconda3/envs/py_study/bin/python"

# 소스 모델: Phase 2 best (P2:lr1e5_ep5)
SOURCE_CKPT="${ABSA_ROOT}/07_models/p2_lr1e5_ep5/best_model.pt"

# 소스 모델 존재 확인
if [ ! -f "$SOURCE_CKPT" ]; then
    echo "[ERROR] Phase 2 소스 모델이 없습니다: ${SOURCE_CKPT}"
    echo "먼저 run_sweep_phase2.sh를 실행하세요."
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ABSA Stage 5 Phase 3 — Focal Loss Sweep (전략 C)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "소스 모델: ${SOURCE_CKPT}"
echo ""
echo "실험 조합 (encoder frozen, head only, Focal Loss):"
echo "  [1/4] gamma=1.5, lr=1e-5, ep=5  (중간 focal, P2와 동일 lr)"
echo "  [2/4] gamma=2.0, lr=1e-5, ep=5  (표준 focal)"
echo "  [3/4] gamma=2.0, lr=5e-6, ep=7  (표준 focal, 보수적 lr)"
echo "  [4/4] gamma=3.0, lr=5e-6, ep=5  (강한 focal, 보수적 lr)"
echo ""

START_TIME=$(date +%s)

# ── Config 1: 중간 focal + P2 동일 lr ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [1/4] gamma=1.5, lr=1e-5, ep=5"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON "$TRAIN_SCRIPT" \
    --gamma 1.5 --lr 1e-5 --epochs 5 \
    --tag p3_g1.5_lr1e5_ep5 \
    --source-checkpoint "$SOURCE_CKPT"
echo ""

# ── Config 2: 표준 focal ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [2/4] gamma=2.0, lr=1e-5, ep=5"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON "$TRAIN_SCRIPT" \
    --gamma 2.0 --lr 1e-5 --epochs 5 \
    --tag p3_g2.0_lr1e5_ep5 \
    --source-checkpoint "$SOURCE_CKPT"
echo ""

# ── Config 3: 표준 focal + 보수적 lr ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [3/4] gamma=2.0, lr=5e-6, ep=7"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON "$TRAIN_SCRIPT" \
    --gamma 2.0 --lr 5e-6 --epochs 7 \
    --tag p3_g2.0_lr5e6_ep7 \
    --source-checkpoint "$SOURCE_CKPT"
echo ""

# ── Config 4: 강한 focal + 보수적 lr ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [4/4] gamma=3.0, lr=5e-6, ep=5"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON "$TRAIN_SCRIPT" \
    --gamma 3.0 --lr 5e-6 --epochs 5 \
    --tag p3_g3.0_lr5e6_ep5 \
    --source-checkpoint "$SOURCE_CKPT"
echo ""

# ── 완료 ──
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Phase 3 Focal Loss sweep 완료! (${MINUTES}분 ${SECONDS}초)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "결과 비교:"
echo "  python ${SCRIPT_DIR}/compare_sweeps.py --include-baselines"
