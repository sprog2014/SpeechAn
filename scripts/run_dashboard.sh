#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT"

export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT:$PROJECT_ROOT/src
python3.12 -m streamlit run dashboard/dashboard.py --server.port 80
