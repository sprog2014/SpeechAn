#!/bin/bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CT2_USE_EXPERIMENTAL_PACKED_GEMM=1
# Автоматическое определение корневой папки проекта
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."

# Попытка активировать venv если он есть
if [ -d "venv" ]; then
    source venv/bin/activate
fi

python3 src/dispatcher.py
