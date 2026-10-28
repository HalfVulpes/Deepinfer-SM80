# Reproducing DeepInfer-SM80 on NVIDIA Ampere

This document is the "install it and see it work" path that issues
[#1](https://github.com/HalfVulpes/Deepinfer-SM80/issues/1) and
[#2](https://github.com/HalfVulpes/Deepinfer-SM80/issues/2) were missing.
It covers only the isolated kernels and the vLLM integration; serving the full
`DeepSeek-V4.1-Flash` checkpoint additionally needs ~475 GiB of weights and a
6- or 8-GPU Ampere node.

## 0. What you need

| Item | Requirement |
| :--- | :--- |
| GPU | Ampere: A100 / A800 (SM80) or RTX 3090 / A6000 / A40 / A10 / A30 (SM86) |
| Driver | any driver that supports CUDA 13 (verified with 595.71.05 / CUDA 13.2) |
| Toolkit | CUDA 13.0 (`nvcc` >= 13.0) |
| Python | 3.12 |
| Compiler | GCC >= 13 |
| Disk | ~8 GiB for vLLM + ~2 GiB build artifacts |

## 1. Get the exact vLLM base

The patches are written against commit
[`79a7108d9a`](https://github.com/vllm-project/vllm/commit/79a7108d9aea27ddab99ce1779290d300b17fc23)
of vLLM PR [#56201](https://github.com/vllm-project/vllm/pull/56201)
(`[Model] Support DeepSeek-V4.1-Flash`, `v0.28.1rc0` base). The PR was closed
without merging, so the commit is not on any branch and a plain `git clone`
does not contain it. Fetch the PR ref instead (blobless keeps it to a few
seconds):

```bash
git init vllm && cd vllm
git remote add origin https://github.com/vllm-project/vllm.git
git fetch --filter=blob:none --no-tags origin refs/pull/56201/head
git checkout 79a7108d9aea27ddab99ce1779290d300b17fc23
```

### Why not the latest `vllm/main`

1. **Package rename.** Upstream PR
   [#56741](https://github.com/vllm-project/vllm/pull/56741) (commit
   [`cf1584f373`](https://github.com/vllm-project/vllm/commit/cf1584f373))
   renamed `vllm/models/deepseek_v4_1/` to `vllm/models/deepseek_v41/`, so the
   patches no longer apply.
2. **No SM80 backend upstream.** `vllm/main` only ships the Blackwell (`SM100`
   FlashInfer / CuteDSL) and AMD ROCm (`gfx942` / `gfx950`) backends for
   `DeepSeek-V4.1-Flash`. The Ampere path comes from the community fork
   [`haosdent/vllm`](https://github.com/haosdent/vllm/tree/dsv4-flash-a100)
   (`dsv4-flash-a100` / `deepseek-v4-a100-support`, commits `62195e9..c3046d1`),
   rebased onto `79a7108d9a` and extended with V4.1 SM80 Sparse MLA,
   software FP8 and DSpark-on-PP enablement. That layer is
   `patches/00-vllm-pr56201-sm80-base.patch`.

## 2. Apply both patches

```bash
# dry run first: verifies the base commit, dirty tree, and patch application
bash scripts/apply_patches.sh --vllm-dir /path/to/vllm --dry-run

# then apply for real
bash scripts/apply_patches.sh --vllm-dir /path/to/vllm
```

The script refuses to patch a non-base tree unless `--allow-other-base` is
given, and afterwards asserts that the key modules exist, including
`vllm/models/deepseek_v4_1/pp_share.py` (the file missing in issue #2).

If you prefer doing it by hand:

```bash
git -C /path/to/vllm apply patches/00-vllm-pr56201-sm80-base.patch
git -C /path/to/vllm apply patches/vllm-sm80-deepinfer.patch
```

**Existing SM80 tree.** If you already run the SM80 base branch (`11cf4a67f9`
on top of `79a7108d9a`), apply only `patches/vllm-sm80-deepinfer.patch`, or copy
the 30 drop-in modules over the installed package:

```bash
cp -r vllm_sm80_deepinfer/vllm/* /path/to/site-packages/vllm/
```

## 3. Build for Ampere only

Restrict the CUDA architectures so nothing is compiled for `sm_90a` /
`sm_100a`:

```bash
export TORCH_CUDA_ARCH_LIST="8.0 8.6"  # A100/A800 and 3090/A6000
pip install -r requirements/build/cuda.txt
pip install -e . --no-build-isolation
```

The Hopper/Blackwell-only extensions (`_vllm_fa3_C`, `_flashmla_C`,
`_flashmla_extension_C`, `_flashkda_C`, `_deep_gemm_C`, `_qutlass_C`,
`fmha_sm100`, `tml_fa4`) are declared `optional=True` in vLLM's `setup.py` and
are skipped when the arch list has no matching target. The Rust frontend is
optional too unless `VLLM_REQUIRE_RUST_FRONTEND=1` is set.

`docker/Dockerfile.sm80` performs steps 1–3 in a CUDA 13.0 / Python 3.12 /
GCC 13 image:

```bash
docker build -f docker/Dockerfile.sm80 -t deepinfer-sm80:cu130 .
docker run --rm --gpus all deepinfer-sm80:cu130
```

## 4. Verify the environment

```bash
python scripts/verify_env.py
```

It prints the device name, compute capability, **SM count**, shared memory per
SM, and the torch/triton/tilelang/flashinfer/vLLM versions, and fails if any GPU
is not Ampere (only `major == 8` selects `TRITON_MLA_SPARSE_DSV41`).

It also warns when the SM count is **108** (A100/A800-80GB SXM4). The patch
comments and benchmark headers still say *74 SMs*; the split heuristics read the
real count at runtime, but the wave-aligned tunings were measured on 74.

## 5. Check the drop-ins match the tree

```bash
python scripts/check_dropin_sync.py --vllm-dir /path/to/vllm
```

This byte-compares `vllm_sm80_deepinfer/vllm/**` against the built tree. A
`DIFF`/`MISSING` line means the Python files you import are not the ones in this
repository — the exact failure mode behind issue #2.

## 6. Run the kernel microbenchmarks (1 GPU is enough)

```bash
bash scripts/run_microbench_single_gpu.sh --python "$(which python)"
```

Archives `env.txt` plus one log per script under
`benchmarks/results/<timestamp>-<gpu>-gpu0/`. The scripts do not load any
checkpoint; they build synthetic tensors and compare against reference outputs.

Expected shape of the output (numbers are hardware-dependent):

| Suite | What is measured |
| :--- | :--- |
| `microbench_sm80_kernels.py` | fused mHC, sparse MLA decode/prefill, KV gather |
| `microbench_sm80_part2.py` | TileLang mHC pre/post kernels |
| `microbench_sm80_part3.py` | TopK+SWA index combiner vs the torch reference |

## 7. Known pitfalls

1. **`ModuleNotFoundError: vllm.models.deepseek_v4_1.pp_share`** — you applied
   the patches before commit `8895510`, or only one of the two patches. Re-pull
   and apply both.
2. **`git apply` fails on `vllm/models/deepseek_v41/...`** — you are on a tree
   newer than `79a7108d9a`; upstream PR #56741 renamed the package.
   `fatal: reference is not a tree: 79a7108d9a...` means the PR ref was not
   fetched (see step 1).
3. **`nvcc` fails in `_vllm_fa3_C` / `_deep_gemm_C`** — `TORCH_CUDA_ARCH_LIST`
   was not set, so the build targeted the Hopper/Blackwell arches too.
4. **First request stalls ~1 s** — Triton compiles `_dequantize_and_gather_k_kernel`
   lazily at inference time (the `jit_monitor` warning). Keep an eye on
   `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` / warmup coverage.
5. **`--gpu-memory-utilization 0.92` yields less KV than expected** — CUDA graph
   memory profiling makes it equivalent to ~0.89 (PP0) / ~0.855 (PP1); raise it
   per the startup log if you need the advertised KV pool.
6. **`SymmMemCommunicator: Device capability 8.0 not supported`** — expected on
   Ampere; all-reduce falls back to `CUSTOM`/`PYNCCL`, and FlashInfer AllReduce
   is disabled at world_size=2.

## 8. Reporting a reproduction

When filing an issue, include the output of `scripts/verify_env.py`, the
`env.txt` written by `run_microbench_single_gpu.sh`, and the three benchmark
logs. That is enough to separate "kernel changed" from "environment differs".

## 9. Verification status

Measured on this checkout (independent of the repository authors):

| Check | Command | Result |
| :--- | :--- | :--- |
| Patch stack applies to the documented base | `bash scripts/apply_patches.sh --dry-run` | both patches clean on `79a7108d9a` |
| `pp_share.py` present after patching (issue #2) | `bash scripts/apply_patches.sh --vllm-dir /tmp/vllm-base-verify` | `ok vllm/models/deepseek_v4_1/pp_share.py` |
| Patched tree == published drop-ins | `python scripts/check_dropin_sync.py --vllm-dir /tmp/vllm-base-verify` | `30 same, 0 diff, 0 missing` |
| Live deployment == published drop-ins | `python scripts/check_dropin_sync.py --vllm-dir <production tree>` | `30 same, 0 diff, 0 missing` |
| `verify_env.py` failure mode off-GPU | run on a login node | exits `3` with "no CUDA device" |

The patch chain and the drop-in directory are therefore consistent: applying
both patches to the exact base reproduces every published Python module
byte-for-byte, which is the property issues #1 and #2 were really asking for.
Kernel-level benchmarks still have to run on real hardware; see
`benchmarks/results/` for the archived runs.

### 9.1 Independent measurement — A800-SXM4-80GB

First reported run on 108-SM Ampere silicon (`sms: 108` in the archived
`env.txt`; the docs and benchmark headers in this repository say 74 SMs).
Archive: `benchmarks/results/20261027-a800-80gb-108sm/`.

| Kernel | This run | README claim |
| :--- | :--- | :--- |
| mHC fused post+pre (`T=12`) | 506.3 → 210.3 µs (2.41×) | 357.7 → 123.8 µs (2.89×) |
| Sparse MLA decode `B=1..24` | 1.15–1.24×, err ≤ 4.9e-04 | 2.08–2.14× |
| `combine_topk_swa` (`T=512..4096`) | 26.4–30.0×, bit-exact | 22.17× |
| `dequantize_and_gather_k_cache` (1K–12K) | 1.59–8.92× per call, bit-exact | 4.02× @16K, 7.64× @48K |

All three scripts ran on one A800-80GB. `microbench_sm80_part3.py` stopped at
`TEST 2` with `TypeError: unsupported format string passed to NoneType` because
the default split count was not part of the sweep for `B >= 16`; this is fixed.

### 9.2 Independent measurement — A100-SXM4-80GB

Built with `docker/Dockerfile.sm80` and run with
`scripts/run_microbench_single_gpu.sh` on one A100-SXM4-80GB (108 SMs):
`verify_env.py` passes, drop-ins `30 same, 0 diff, 0 missing`, all three
microbenchmarks complete, every bit-exactness check passes. Archive and
comparison: `benchmarks/results/20261028-a100-sxm4-80gb-108sm/`.
