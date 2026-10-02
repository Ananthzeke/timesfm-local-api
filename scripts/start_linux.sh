#!/usr/bin/env bash
set -euo pipefail
mode="${1:-cuda}"
case "$mode" in
  mock|cuda) ;;
  *) echo "Usage: bash scripts/start_linux.sh [mock|cuda]" >&2; exit 2 ;;
esac
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_dir"
if [[ -z "${HF_HOME:-}" && -d "$project_dir/.cache/huggingface" ]]; then
  export HF_HOME="$project_dir/.cache/huggingface"
fi
venv_python="$project_dir/.venv/bin/python"
if [[ ! -x "$venv_python" ]]; then
  echo "Run bash scripts/setup_linux.sh $mode first." >&2
  exit 1
fi
if [[ "$mode" == "mock" ]]; then
  export TF_BACKEND=mock
else
  export TF_BACKEND=timesfm3 TF_DEVICE=cuda
  "$venv_python" - <<'PY'
from app.config import Settings
if not Settings().license_accepted:
    raise SystemExit("Read the TimesFM model license linked in README; set TF_LICENSE_ACCEPTED=true only for permitted use.")
PY
fi
echo "Starting on http://127.0.0.1:8000 — interactive docs at /docs; Ctrl+C stops the server."
exec "$venv_python" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1 \
  --limit-concurrency 64 --timeout-keep-alive 5
