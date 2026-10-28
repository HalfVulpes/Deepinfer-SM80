#!/usr/bin/env bash
# Apply the DeepInfer-SM80 patches (in order) onto the exact vLLM base commit.
#
# Usage:
#   scripts/apply_patches.sh --vllm-dir /path/to/vllm [--dry-run] [--allow-other-base]
#
# The two patches are stacked and both are required:
#   1. patches/00-vllm-pr56201-sm80-base.patch   SM80 base enablement
#      (Sparse MLA / software FP8 / DSpark-on-PP), rebased onto vLLM PR #56201
#      commit 79a7108d9aea27ddab99ce1779290d300b17fc23
#   2. patches/vllm-sm80-deepinfer.patch         DeepInfer-SM80 kernels +
#      YOCO / PP share / 1M-context engine work
#
# Patch 2 is a diff against the tree produced by patch 1, so it cannot be
# `git apply --check`-ed against the bare base tree. --dry-run therefore applies
# both patches into a throwaway copy of the git index (working tree untouched),
# which is also how CI can validate that the stack still applies.
#
# Exit codes:
#   0  both patches applied (or --dry-run: the stack applies cleanly)
#   2  HEAD is not the documented base commit
#   3  target tree has local changes
#   4  a patch does not apply cleanly
#   5  post-apply sanity check failed
set -euo pipefail

BASE_COMMIT="79a7108d9aea27ddab99ce1779290d300b17fc23"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_DIR=""
DRY_RUN=0
ALLOW_OTHER_BASE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --vllm-dir)          VLLM_DIR="${2:?--vllm-dir needs a path}"; shift 2 ;;
    --dry-run)           DRY_RUN=1; shift ;;
    --allow-other-base)  ALLOW_OTHER_BASE=1; shift ;;
    -h|--help)           sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done

if [ -z "$VLLM_DIR" ]; then
  if [ -e "$REPO_ROOT/vllm/.git" ]; then
    VLLM_DIR="$REPO_ROOT/vllm"
  else
    echo "error: --vllm-dir is required (no ./vllm clone next to this repo)" >&2
    exit 64
  fi
fi
VLLM_DIR="$(cd "$VLLM_DIR" && pwd)"

PATCHES=(
  "$REPO_ROOT/patches/00-vllm-pr56201-sm80-base.patch"
  "$REPO_ROOT/patches/vllm-sm80-deepinfer.patch"
)
for p in "${PATCHES[@]}"; do
  [ -s "$p" ] || { echo "error: missing patch $p" >&2; exit 64; }
done

HEAD_SHA="$(git -C "$VLLM_DIR" rev-parse HEAD)"
echo "==> vLLM tree : $VLLM_DIR"
echo "==> HEAD       : $HEAD_SHA"
for p in "${PATCHES[@]}"; do
  echo "==> patch      : $(basename "$p")"
done

if [ "$HEAD_SHA" != "$BASE_COMMIT" ]; then
  echo
  echo "error: HEAD is not the documented base commit." >&2
  echo "       expected $BASE_COMMIT (vLLM PR #56201, v0.28.1rc0 base)" >&2
  echo "       upstream PR #56741 renamed vllm/models/deepseek_v4_1/ ->" >&2
  echo "       vllm/models/deepseek_v41/, so newer trees will not apply." >&2
  if git -C "$VLLM_DIR" cat-file -e "$BASE_COMMIT^{commit}" 2>/dev/null; then
    echo "       the expected commit exists in this clone; do:" >&2
    echo "         git -C '$VLLM_DIR' checkout $BASE_COMMIT" >&2
  fi
  if [ "$ALLOW_OTHER_BASE" -eq 0 ]; then
    echo "       pass --allow-other-base to try anyway (the patches may still apply)" >&2
    exit 2
  fi
  echo "       continuing because --allow-other-base was passed" >&2
fi

if [ -n "$(git -C "$VLLM_DIR" status --porcelain)" ]; then
  echo "error: $VLLM_DIR has local changes; commit/stash them first" >&2
  git -C "$VLLM_DIR" status --short >&2
  exit 3
fi

apply_fail() {
  local patch="$1"
  echo "error: $(basename "$patch") does not apply cleanly to $HEAD_SHA" >&2
  echo "hint: inspect the rejected files with:" >&2
  echo "        git -C '$VLLM_DIR' apply --check --verbose '$patch'" >&2
  echo "hint: or attempt a 3-way merge:" >&2
  echo "        git -C '$VLLM_DIR' apply --3way '$patch'" >&2
  exit 4
}

if [ "$DRY_RUN" -eq 1 ]; then
  # Validate the stacked patches against a private copy of the index.
  TMP_INDEX="$(mktemp)"
  trap 'rm -f "$TMP_INDEX"' EXIT
  cp "$(git -C "$VLLM_DIR" rev-parse --git-path index)" "$TMP_INDEX"
  export GIT_INDEX_FILE="$TMP_INDEX"
  for p in "${PATCHES[@]}"; do
    echo
    echo "==> checking $(basename "$p")"
    git -C "$VLLM_DIR" apply --cached "$p" || apply_fail "$p"
    echo "    applies cleanly"
  done
  echo
  echo "dry run OK: the two-patch stack applies cleanly to $HEAD_SHA"
  exit 0
fi

for p in "${PATCHES[@]}"; do
  echo
  echo "==> checking $(basename "$p")"
  git -C "$VLLM_DIR" apply --check "$p" || apply_fail "$p"
  git -C "$VLLM_DIR" apply "$p"
  echo "    applied"
done

echo
echo "==> post-apply sanity check"
missing=0
for f in \
  vllm/models/deepseek_v4_1/ampere/ampere_sparse.py \
  vllm/models/deepseek_v4_1/pp_share.py \
  vllm/models/deepseek_v4_1/common/ops/cache_utils.py \
  vllm/models/deepseek_v4_1/common/engram.py \
  vllm/v1/attention/ops/fp8_sm80.py \
  vllm/v1/attention/ops/rocm_aiter_mla_sparse.py \
  vllm/model_executor/kernels/mhc/tilelang.py \
  vllm/model_executor/kernels/mhc/warmup.py
do
  if [ -f "$VLLM_DIR/$f" ]; then
    echo "    ok   $f"
  else
    echo "    MISS $f" >&2
    missing=1
  fi
done
if [ "$missing" -ne 0 ]; then
  echo "error: patches applied but expected modules are missing (partial stack?)" >&2
  exit 5
fi

echo
echo "==> build with:"
echo "    export VLLM_SM80_ONLY=1        # skip Hopper/Blackwell extension targets"
echo "    export TORCH_CUDA_ARCH_LIST='8.0 8.6'"
echo "    pip install -e . --no-build-isolation"
exit 0
