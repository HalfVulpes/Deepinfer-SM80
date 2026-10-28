#!/usr/bin/env python3
"""Check that the published drop-in modules match a given vLLM tree.

``vllm_sm80_deepinfer/`` is meant to be copied/bind-mounted over the vLLM
package. Mismatch between the two is the usual cause of "patch applied but the
model still imports the old module" reports (see issue #2), so this script
compares every drop-in file byte-for-byte.

Usage:
    python scripts/check_dropin_sync.py --vllm-dir /path/to/vllm

Exit codes:
    0  all drop-in files identical
    1  at least one file differs or is missing
"""

from __future__ import annotations

import argparse
import filecmp
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DROPIN_ROOT = REPO_ROOT / "vllm_sm80_deepinfer"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vllm-dir", required=True, help="vLLM checkout to compare against"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="only report differing / missing files"
    )
    args = parser.parse_args()

    vllm_dir = Path(args.vllm_dir).resolve()
    target_root = vllm_dir / "vllm"
    if not target_root.is_dir():
        print(f"error: {target_root} is not a directory", file=sys.stderr)
        return 1
    if not DROPIN_ROOT.is_dir():
        print(f"error: {DROPIN_ROOT} is missing", file=sys.stderr)
        return 1

    same: list[str] = []
    diff: list[str] = []
    missing: list[str] = []

    for src in sorted(DROPIN_ROOT.rglob("*")):
        if not src.is_file() or "__pycache__" in src.parts:
            continue
        rel = src.relative_to(DROPIN_ROOT)  # keeps the leading "vllm/"
        dst = vllm_dir / rel
        if not dst.is_file():
            missing.append(str(rel))
        elif filecmp.cmp(src, dst, shallow=False):
            same.append(str(rel))
        else:
            diff.append(str(rel))

    for rel in same:
        if not args.quiet:
            print(f"  SAME    {rel}")
    for rel in diff:
        print(f"  DIFF    {rel}")
    for rel in missing:
        print(f"  MISSING {rel}")

    print()
    print(f"{len(same)} same, {len(diff)} diff, {len(missing)} missing")
    if diff or missing:
        print()
        print("hint: re-apply patches/vllm-sm80-deepinfer.patch or re-copy the drop-ins")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
