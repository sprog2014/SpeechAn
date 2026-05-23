@echo off

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
py -m streamlit run dashboard/dashboard.py --server.port 8501

popd
