#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ABSA Stage 5 Phase 2 — Encoder Freeze Sweep
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Phase 2: encoder freeze 상태에서 head만 fine-tuning.
# head만 학습하므로 Stage 5(full warm-start)보다 lr을 높게 설정.
#
# head 파라미터: ~25K (전체 110M의 0.02%)
# → 학습 속도가 매우 빠름 (config당 ~2-3분 예상, 총 ~10분)
#
# 사용법:
#   cd Why-pi/06_analysis/03_ABSA/06_scripts
#   chmod +x run_sweep_phase2.sh
#   ./run_sweep_phase2.sh
#
# 결과 확인:
#   python compare_sweeps.py --include-baselines
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ABSA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_stage5_phase2.py"
PYTHON="/opt/miniconda3/envs/py_study/bin/python"

# 소스 모델: Stage 4B warm-start (현재 best)
SOURCE_CKPT="${ABSA_ROOT}/07_models/checkpoints_stage4b/best_model.pt"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ABSA Stage 5 Phase 2 — Sweep 시작"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "소스 모델: ${SOURCE_CKPT}"
echo ""
echo "실험 조합 (encoder frozen, head only):"
echo "  [1/4] lr=1e-5, epochs=5  (기본 권장)"
echo "  [2/4] lr=5e-5, epochs=3  (공격적 — head만이므로 가능)"
echo "  [3/4] lr=5e-6, epochs=7  (보수적)"
echo "  [4/4] lr=1e-4, epochs=3  (매우 공격적)"
echo ""

START_TIME=$(date +%s)

# ── Config 1: 기본 권장 ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [1/4] lr=1e-5, epochs=5"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON "$TRAIN_SCRIPT" --lr 1e-5 --epochs 5 --tag p2_lr1e5_ep5 --source-checkpoint "$SOURCE_CKPT"
echo ""

# ── Config 2: 공격적 ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [2/4] lr=5e-5, epochs=3"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON "$TRAIN_SCRIPT" --lr 5e-5 --epochs 3 --tag p2_lr5e5_ep3 --source-checkpoint "$SOURCE_CKPT"
echo ""

# ── Config 3: 보수적 ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [3/4] lr=5e-6, epochs=7"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON "$TRAIN_SCRIPT" --lr 5e-6 --epochs 7 --tag p2_lr5e6_ep7 --source-checkpoint "$SOURCE_CKPT"
echo ""

# ── Config 4: 매우 공격적 ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [4/4] lr=1e-4, epochs=3"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON "$TRAIN_SCRIPT" --lr 1e-4 --epochs 3 --tag p2_lr1e4_ep3 --source-checkpoint "$SOURCE_CKPT"
echo ""

# ── 완료 ──
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Phase 2 sweep 완료! (${MINUTES}분 ${SECONDS}초)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "결과 비교:"
echo "  python ${SCRIPT_DIR}/compare_sweeps.py --include-baselines"
