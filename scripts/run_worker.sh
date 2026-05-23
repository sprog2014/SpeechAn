#!/bin/bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CT2_USE_EXPERIMENTAL_PACKED_GEMM=1
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/src
python3 src/worker.py "$@"
