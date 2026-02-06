#!/bin/bash
# ABSA 파이프라인 전체 실행 스크립트 (py_study 환경)

PYTHON="/opt/miniconda3/envs/py_study/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================="
echo "ABSA 파이프라인 실행"
echo "========================================="
echo "Python: $PYTHON"
echo "작업 디렉토리: $(dirname $SCRIPT_DIR)"
echo ""

# Step A: 샘플링
echo "Step A: 샘플링 시작..."
$PYTHON "$SCRIPT_DIR/step_a_sampling.py"
if [ $? -ne 0 ]; then
    echo "Step A 실패!"
    exit 1
fi
echo ""

# OpenAI API 키 확인
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY가 설정되지 않았습니다."
    echo "Step B를 실행하려면 다음을 실행하세요:"
    echo "  export OPENAI_API_KEY='your-key'"
    echo ""
    read -p "계속하시겠습니까? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# Step B: 라벨링
echo "Step B: ChatGPT 라벨링 시작 (2-4시간 소요)..."
$PYTHON "$SCRIPT_DIR/step_b_labeling.py"
if [ $? -ne 0 ]; then
    echo "Step B 실패!"
    exit 1
fi
echo ""

# Step C: 데이터셋 생성
echo "Step C: 데이터셋 생성 시작..."
$PYTHON "$SCRIPT_DIR/step_c_create_dataset.py"
if [ $? -ne 0 ]; then
    echo "Step C 실패!"
    exit 1
fi
echo ""

echo "========================================="
echo "Step A-C 완료!"
echo "========================================="
echo ""
echo "다음 단계:"
echo "  Step D (학습): GPU 필요"
echo "  $PYTHON scripts/step_d_train.py"
echo ""
echo "  Step E (추론): GPU 필요"
echo "  $PYTHON scripts/step_e_inference.py"
echo "========================================="
