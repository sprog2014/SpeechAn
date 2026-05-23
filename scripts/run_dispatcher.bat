@echo off
set OMP_NUM_THREADS=8
set MKL_NUM_THREADS=8
set CT2_USE_EXPERIMENTAL_PACKED_GEMM=1

pushd "%~dp0.."
set PROJECT_ROOT=%CD%

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment "venv" not found.
    echo Please create it with: py -m venv venv
    pause
    exit /b 1
)

set PYTHONPATH=%PYTHONPATH%;%PROJECT_ROOT%\src
py src/dispatcher.py

popd
