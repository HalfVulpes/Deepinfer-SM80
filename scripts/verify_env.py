#!/usr/bin/env python3
"""Print the GPU / toolchain facts that DeepInfer-SM80 depends on.

Run this on the *compute node* that will host the engine, with the vLLM
environment activated:

    python scripts/verify_env.py

Every number printed here maps to a tuning assumption in
``patches/vllm-sm80-deepinfer.patch``: SM count drives the split-K heuristics,
``capability major == 8`` selects ``TRITON_MLA_SPARSE_DSV41``, and the shared
memory per SM caps the kernel tile sizes. Reporting the real values makes
"74 SMs" style discrepancies visible instead of silent.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys

# (min shared memory per SM in KiB, max shared memory per block in KiB)
SMEM_BY_ARCH = {
    (8, 0): (164, 163),
    (8, 6): (100, 99),
    (8, 9): (100, 99),
    (9, 0): (228, 227),
    (10, 0): (228, 227),
    (12, 0): (228, 227),
}


def run(cmd: list[str]) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, check=False
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []

    section("host")
    print(f"hostname        : {platform.node()}")
    print(f"kernel          : {platform.release()}")
    print(f"python          : {platform.python_version()} ({sys.executable})")
    print(f"cwd             : {os.getcwd()}")

    try:
        import torch
    except ImportError:
        print()
        print("error: torch is not importable; activate the vLLM environment first")
        return 2

    section("torch")
    print(f"torch           : {torch.__version__}")
    print(f"torch cuda      : {torch.version.cuda}")
    print(f"cuda available  : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print()
        print("error: torch sees no CUDA device (login node? wrong --gres?)")
        return 3

    section("toolchain")
    print(f"nvcc            : {shutil.which('nvcc') or '<not on PATH>'}")
    nvcc = run(["nvcc", "--version"])
    if nvcc:
        for line in nvcc.splitlines():
            if "release" in line:
                print(f"nvcc version    : {line.strip()}")
    for mod in ("triton", "tilelang", "flashinfer"):
        try:
            m = __import__(mod)
            print(f"{mod:<15} : {getattr(m, '__version__', '<no __version__>')}")
        except ImportError:
            print(f"{mod:<15} : <not installed>")

    try:
        import vllm
    except ImportError:
        print()
        print("error: vllm is not importable; build/install the patched tree first")
        return 2

    print(f"vllm            : {vllm.__version__}")
    print(f"vllm path       : {vllm.__file__}")

    section("environment")
    for name in (
        "TORCH_CUDA_ARCH_LIST",
        "VLLM_PP_LAYER_PARTITION",
        "CUDA_HOME",
    ):
        print(f"{name:<24}= {os.environ.get(name, '<unset>')}")

    section("devices")
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        cap = (props.major, props.minor)
        smem = SMEM_BY_ARCH.get(cap)
        print(f"gpu {idx}: {props.name}")
        print(f"  capability          : {props.major}.{props.minor}")
        print(f"  SMs                 : {props.multi_processor_count}")
        print(f"  total memory        : {props.total_memory / 2**30:.1f} GiB")
        if smem is not None:
            print(f"  smem / SM           : {smem[0]} KiB")
            print(f"  smem / block (max)  : {smem[1]} KiB")
        if props.major != 8:
            failures.append(
                f"gpu {idx} ({props.name}) is SM{props.major}{props.minor}, not Ampere: "
                "the TRITON_MLA_SPARSE_DSV41 backend only claims major == 8"
            )

    section("assessment")
    sm_counts = {
        torch.cuda.get_device_properties(i).multi_processor_count
        for i in range(torch.cuda.device_count())
    }
    if 108 in sm_counts:
        warnings.append(
            "SM count 108 detected (A100/A800-80GB SXM4). The patch comments and "
            "benchmark headers still say '74 SMs'; the split heuristics read the "
            "real count at runtime, but wave-aligned tunings were measured on 74."
        )
    for msg in warnings:
        print(f"warn: {msg}")
    for msg in failures:
        print(f"FAIL: {msg}")
    if not warnings and not failures:
        print("ok: Ampere device with the SM count the tunings were measured on")
    print()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
