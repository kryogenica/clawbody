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
python -m pip install --upgrade pip wheel

# Select PyTorch/ONNX Runtime build:
#   - If an NVIDIA GPU is present, install CUDA wheels matching the runtime
#     reported by the driver/toolkit (does NOT change the system CUDA install).
#   - Otherwise fall back to CPU wheels.
# Override the detected CUDA tag with WAKEWORD_CUDA_TAG (e.g. cu121, cu124, cu126).
detect_cuda_tag() {
  local ver=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    ver="$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')"
  fi
  if [ -z "${ver}" ] && command -v nvcc >/dev/null 2>&1; then
    ver="$(nvcc --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')"
  fi
  case "${ver}" in
    12.6*|12.7*|12.8*|12.9*) echo "cu126" ;;
    12.4*|12.5*)             echo "cu124" ;;
    12.1*|12.2*|12.3*)       echo "cu121" ;;
    12.0*)                   echo "cu121" ;;
    "")                      echo "" ;;
    *)                       echo "cu126" ;;
  esac
}

CUDA_TAG="${WAKEWORD_CUDA_TAG:-$(detect_cuda_tag)}"

if [ -n "${CUDA_TAG}" ]; then
  echo "Detected NVIDIA GPU; installing CUDA (${CUDA_TAG}) PyTorch wheels."
  # Remove any CPU build left from a prior run; pip otherwise keeps torch 2.x+cpu
  # as "already satisfied" even when pointed at the CUDA index.
  python -m pip uninstall -y torch torchaudio >/dev/null 2>&1 || true
  python -m pip install --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" torch torchaudio
  # GPU ONNX Runtime for melspectrogram/embedding feature extraction.
  # Pin to a CUDA 12 build: onnxruntime-gpu >=1.23 links against CUDA 13
  # (needs libcudart.so.13), while the torch CUDA 12.x wheels ship
  # libcudart.so.12. 1.22.0 is the last CUDA 12 release and loads fine using
  # the CUDA libs torch already pulls in (no system CUDA changes needed).
  # Override with WAKEWORD_ORT_GPU_VERSION if your driver uses CUDA 13.
  ORT_GPU_VERSION="${WAKEWORD_ORT_GPU_VERSION:-1.22.0}"
  python -m pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
  # --no-deps: deps (numpy, protobuf, …) are already installed from requirements;
  # reinstalling them can bump numpy past numba's upper bound (<2.5).
  python -m pip install --no-deps "onnxruntime-gpu==${ORT_GPU_VERSION}"
else
  echo "No NVIDIA GPU detected; installing CPU PyTorch wheels."
  python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
fi

python -m pip install -r "${REQ_FILE}"

echo
echo "Environment ready at: ${ENV_DIR}"
echo "Activate with:"
echo "  source ${ENV_DIR}/bin/activate"
