#!/bin/bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CT2_USE_EXPERIMENTAL_PACKED_GEMM=1
cd /app
python src/dispatcher.py