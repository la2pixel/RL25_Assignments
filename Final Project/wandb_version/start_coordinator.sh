üy#!/usr/bin/env bash
# Start the round-based training coordinator: ensure conda env and deps, then run coordinator with training.yaml.
# Usage: ./wandb_version/start_coordinator.sh [optional args, e.g. --max_rounds 2 --poll_interval 15]
set -e

CONDA_ENV="${CONDA_ENV:-hockey}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="$SCRIPT_DIR/config/training.yaml"

if ! command -v conda &>/dev/null; then
  echo "Conda not found. Install Miniconda or Anaconda first." >&2
  exit 1
fi

if ! conda env list | grep -q "^${CONDA_ENV} "; then
  echo "Creating conda env: $CONDA_ENV"
  conda create -n "$CONDA_ENV" python=3.11 -y
fi

echo "Installing dependencies in env: $CONDA_ENV"
conda run -n "$CONDA_ENV" pip install -q -r "$SCRIPT_DIR/requirements.txt" 2>/dev/null || true
conda run -n "$CONDA_ENV" pip install -q "git+https://github.com/martius-lab/hockey-env.git" 2>/dev/null || true

cd "$PROJECT_ROOT"
echo "Running coordinator (config: $CONFIG)"
exec conda run -n "$CONDA_ENV" python "$SCRIPT_DIR/coordinator.py" --config "$CONFIG" "$@"
