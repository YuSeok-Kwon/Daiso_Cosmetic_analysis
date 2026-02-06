#!/bin/bash
# Wrapper script to run ABSA scripts with py_study environment

PYTHON_ENV="/opt/miniconda3/envs/py_study/bin/python"

# Check if script argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: ./run_with_py_study.sh <script_name>"
    echo ""
    echo "Examples:"
    echo "  ./run_with_py_study.sh check_setup.py"
    echo "  ./run_with_py_study.sh step_a_sampling.py"
    echo "  ./run_with_py_study.sh step_b_labeling.py"
    echo "  ./run_with_py_study.sh step_c_create_dataset.py"
    echo "  ./run_with_py_study.sh step_d_train.py"
    echo "  ./run_with_py_study.sh step_e_inference.py"
    echo "  ./run_with_py_study.sh evaluate_test.py"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the Python script with py_study environment
$PYTHON_ENV "$SCRIPT_DIR/$1" "${@:2}"
