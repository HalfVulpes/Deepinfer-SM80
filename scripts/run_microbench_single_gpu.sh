#!/usr/bin/env bash
# Run the three isolated SM80 microbenchmarks on ONE GPU and archive the output.
#
#   scripts/run_microbench_single_gpu.sh [--python /path/to/python] [--out-dir DIR]
#
# The microbenchmarks need the patched vLLM package importable (they import
# vllm.models.deepseek_v4_1.*, vllm.v1.attention.ops.fp8_sm80, ...), but they do
# NOT load the model or any checkpoint, so a single SM80 card is enough.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
OUT_ROOT="${REPO_ROOT}/benchmarks/results"

while [ $# -gt 0 ]; do
  case "$1" in
    --python)  PYTHON="${2:?--python needs a path}"; shift 2 ;;
    --out-dir) OUT_ROOT="${2:?--out-dir needs a path}"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done

if ! "$PYTHON" -c 'import torch, vllm' 2>/dev/null; then
  echo "error: $PYTHON cannot import torch+vllm; pass --python of the patched env" >&2
  exit 2
fi
if ! "$PYTHON" -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
  echo "error: no CUDA device visible to $PYTHON" >&2
  exit 3
fi

GPU_NAME="$("$PYTHON" - <<'PY'
import re
import torch
name = torch.cuda.get_device_name(0)
print(re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_"))
PY
)"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${OUT_ROOT}/${STAMP}-${GPU_NAME}-gpu0"
mkdir -p "$OUT_DIR"

echo "==> python : $PYTHON"
echo "==> gpu    : $("$PYTHON" -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "==> out    : $OUT_DIR"

{
  echo "date:      $(date -Is)"
  echo "host:      $(hostname)"
  echo "gpu:       $("$PYTHON" -c 'import torch; print(torch.cuda.get_device_name(0))')"
  echo "capability:$("$PYTHON" -c 'import torch; p=torch.cuda.get_device_properties(0); print(f"{p.major}.{p.minor}")')"
  echo "sms:       $("$PYTHON" -c 'import torch; print(torch.cuda.get_device_properties(0).multi_processor_count)')"
  echo "torch:     $("$PYTHON" -c 'import torch; print(torch.__version__)')"
  echo "vllm:      $("$PYTHON" -c 'import vllm; print(vllm.__version__)')"
  echo "git:       $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
} > "$OUT_DIR/env.txt"

status=0
for script in microbench_sm80_kernels.py microbench_sm80_part2.py microbench_sm80_part3.py; do
  name="${script%.py}"
  echo
  echo "==> $script"
  if "$PYTHON" "$REPO_ROOT/benchmarks/$script" 2>&1 | tee "$OUT_DIR/${name}.log"; then
    echo "    ok -> $OUT_DIR/${name}.log"
  else
    echo "    FAILED -> $OUT_DIR/${name}.log" >&2
    status=1
  fi
done

echo
if [ "$status" -eq 0 ]; then
  echo "all microbenchmarks finished; archive: $OUT_DIR"
else
  echo "one or more microbenchmarks failed; see $OUT_DIR" >&2
fi
exit "$status"
