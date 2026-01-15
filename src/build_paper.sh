#!/bin/bash
# Activates the virtual environment and runs the pipeline

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$DIR/.."

source "$ROOT_DIR/.venv/bin/activate"
python3 "$ROOT_DIR/src/research_pipeline.py"
