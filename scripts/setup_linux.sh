#!/usr/bin/env bash
# Run from any directory: bash /path/to/timesfm-local-api/scripts/setup_linux.sh cuda
set -euo pipefail

mode="${1:-mock}"
case "$mode" in
  mock|cuda) ;;
  *) echo "Usage: bash scripts/setup_linux.sh [mock|cuda]" >&2; exit 2 ;;
esac

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_dir"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This launcher is for Linux. See README for other systems." >&2
  exit 1
fi
if [[ "$mode" == "cuda" && "$(uname -m)" != "x86_64" ]]; then
  echo "This CUDA preset requires x86_64 Linux." >&2
  exit 1
fi

if [[ -n "${TF_SETUP_PYTHON:-}" ]]; then
  setup_python="$TF_SETUP_PYTHON"
elif command -v python3.12 >/dev/null 2>&1; then
  setup_python="python3.12"
elif command -v python3.11 >/dev/null 2>&1; then
  setup_python="python3.11"
else
  setup_python="python3"
fi
if ! command -v "$setup_python" >/dev/null 2>&1; then
  echo "Install Python 3.11 or 3.12 first, then rerun this launcher." >&2
  exit 1
fi
"$setup_python" -c 'import sys; assert (3,11) <= sys.version_info[:2] <= (3,12), "Use Python 3.11 or 3.12 (TF_SETUP_PYTHON can select it)"'

if [[ "$mode" == "cuda" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is missing. Configure your NVIDIA driver before CUDA setup." >&2
    exit 1
  fi
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
fi

if [[ ! -d .venv ]]; then
  "$setup_python" -m venv .venv || {
    echo "Virtual environment creation failed. Install your distro's Python venv package." >&2
    exit 1
  }
fi
venv_python="$project_dir/.venv/bin/python"
if [[ ! -x "$venv_python" ]]; then
  echo "Existing .venv is not a usable Linux virtual environment. Use a fresh project folder." >&2
  exit 1
fi
"$venv_python" -c 'import sys; assert (3,11) <= sys.version_info[:2] <= (3,12), "Existing venv must use Python 3.11 or 3.12"'
"$venv_python" -m pip install --upgrade pip

if [[ "$mode" == "cuda" ]]; then
  # Official older CUDA 11.8 build; avoid relying on a CPU-only default wheel.
  "$venv_python" -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu118
  "$venv_python" -m pip install -e '.[model]'
  "$venv_python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA. Check your NVIDIA driver; setup did not start the server.")
device = torch.cuda.current_device()
print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(device))
print("VRAM GiB:", round(torch.cuda.get_device_properties(device).total_memory / 1024**3, 2))
print("Compute capability:", torch.cuda.get_device_capability(device))
print("CUDA allocation check:", torch.ones(1, device="cuda").cpu().item())
PY
else
  "$venv_python" -m pip install -e .
fi

if [[ ! -e .env ]]; then
  # Preserve any existing user configuration. New GPU setup uses smaller profiles.
  "$venv_python" - <<'PY'
from pathlib import Path
text = Path('.env.example').read_text()
text = text.replace('TF_MAX_CONTEXT=512', 'TF_MAX_CONTEXT=256')
text = text.replace('TF_MAX_HORIZON=64', 'TF_MAX_HORIZON=16')
text = text.replace('TF_QUEUE_CAPACITY=32', 'TF_QUEUE_CAPACITY=4')
Path('.env').write_text(text)
PY
fi

echo "Setup complete. Existing .env settings were preserved."
if [[ "$mode" == "cuda" ]]; then
  echo "Read the model license linked in README and configure TF_LICENSE_ACCEPTED for permitted use."
fi
echo "Start: bash scripts/start_linux.sh $mode"
