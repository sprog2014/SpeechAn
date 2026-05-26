#!/bin/bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CT2_USE_EXPERIMENTAL_PACKED_GEMM=1
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT"

export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT:$PROJECT_ROOT/src
python3.12 src/worker.py "$@"
