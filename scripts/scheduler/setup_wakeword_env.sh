#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_DIR="${ROOT_DIR}/.venv-wakeword"
REQ_FILE="${ROOT_DIR}/scripts/scheduler/requirements-wakeword-train.txt"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but not found."
  exit 1
fi

if ! command -v espeak-ng >/dev/null 2>&1; then
  echo "espeak-ng is required for phoneme generation."
  echo "Install with: sudo apt-get update && sudo apt-get install -y espeak-ng"
  exit 1
fi

if [ ! -d "${ENV_DIR}" ]; then
  python3 -m venv "${ENV_DIR}"
fi

source "${ENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

# Force CPU wheels to avoid pulling large CUDA dependencies on small devices.
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
python -m pip install -r "${REQ_FILE}"

echo
echo "Environment ready at: ${ENV_DIR}"
echo "Activate with:"
echo "  source ${ENV_DIR}/bin/activate"
