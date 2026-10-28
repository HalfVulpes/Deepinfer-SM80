# DeepInfer-SM80

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-SM80_%7C_SM86-76B900.svg)](https://developer.nvidia.com/cuda-gpus)
[![Target GPUs](https://img.shields.io/badge/GPUs-A100_%7C_A800_%7C_RTX_3090_%7C_RTX_A6000-success.svg)](#supported-hardware)
[![vLLM](https://img.shields.io/badge/Engine-vLLM_V1-orange.svg)](https://github.com/vllm-project/vllm)

**DeepSeek-V4.1 kernels and 1M-context engine optimizations for NVIDIA Ampere (`SM80` / `SM86`) on vLLM V1.**

DeepInfer-SM80 ports the core ideas of DeepSeek's [`DeepInfer`](https://github.com/deepseek-ai/DeepInfer), which targets Blackwell (`tcgen05`, `TMEM`, CuteDSL), to Ampere GPUs such as the A100, A800, RTX 3090 and RTX A6000. It ships as two vLLM patches plus the equivalent drop-in Python/Triton/TileLang modules.

## Results at a glance

Measured on NVIDIA GA100 64GB (`SM80`, 74 SMs) nodes serving `DeepSeek-V4.1-Flash` with `DSpark K=5` speculative decoding. Kernel speedups are lower on 108-SM A100/A800 parts; full tables, charts and independent runs are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

**6 GPUs, `TP2 × PP3`: baseline vLLM vs. DeepInfer-SM80**

| Metric | Baseline vLLM | DeepInfer-SM80 | Change |
| :--- | :---: | :---: | :---: |
| 16K cold prefill | 3,349 tok/s | **4,037 tok/s** | **+20.5%** |
| 64K cold prefill | 3,627 tok/s | **4,507 tok/s** | **+24.2%** |
| 64K time-to-first-token | 13.2 s | **10.7 s** | **−18.9%** |
| 1-stream decode (code) | 151.8 tok/s | **160.8 tok/s** | **+5.9%** |
| GPU-resident KV capacity | 4.65M tokens | 4.65M tokens | no VRAM overhead |

**8 GPUs, `TP4 × PP2`**: 13.3M GPU-resident KV tokens (12.69 full 1M-token contexts), 153–180 tok/s single-stream code decode, 620 tok/s aggregate at 12 streams, ~3,460 tok/s cold prefill.

**Kernel microbenchmarks** (isolated, verified against the baseline kernels):

| Kernel | Speedup | Accuracy |
| :--- | :---: | :---: |
| `_combine_topk_swa_indices_kernel` (`T=1536`) | **22.17×** | bit-exact |
| `dequantize_and_gather_k_cache` (16K / 48K context) | **4.02× / 7.64×** | bit-exact |
| `mhc_fused_post_pre_delayed` (`T=12`) | **2.89×** | < 1.9e-3 |
| `_sparse_attn_decode_kernel` (`B=6..24`) | **2.08–2.14×** | bit-exact |
| `_sparse_attn_prefill_kernel` (`T=1536`) | **1.17×** | < 4.8e-3 |

## Quick start

The patches target vLLM commit [`79a7108d9a`](https://github.com/vllm-project/vllm/commit/79a7108d9aea27ddab99ce1779290d300b17fc23) (PR [#56201](https://github.com/vllm-project/vllm/pull/56201)) and must be applied in order.

```bash
# 1. Check out the exact vLLM base (only reachable through the PR ref)
git init vllm && cd vllm
git remote add origin https://github.com/vllm-project/vllm.git
git fetch --filter=blob:none origin refs/pull/56201/head
git checkout 79a7108d9aea27ddab99ce1779290d300b17fc23
cd ..

# 2. Apply both patches (validates the base commit and the result)
bash Deepinfer-SM80/scripts/apply_patches.sh --vllm-dir vllm

# 3. Build for Ampere only, then check the environment
cd vllm && TORCH_CUDA_ARCH_LIST="8.0 8.6" pip install -e . --no-build-isolation && cd ..
python Deepinfer-SM80/scripts/verify_env.py

# 4. Run the kernel microbenchmarks on one GPU
bash Deepinfer-SM80/scripts/run_microbench_single_gpu.sh
```

Already running an SM80 vLLM tree? Copy the 30 drop-in modules from [`vllm_sm80_deepinfer/`](vllm_sm80_deepinfer/) over your `vllm` package and confirm with `scripts/check_dropin_sync.py`. See [docs/REPRODUCTION.md](docs/REPRODUCTION.md) for the full procedure, the Docker recipe and known pitfalls.

## Supported hardware

| Compute capability | GPUs | Notes |
| :---: | :--- | :--- |
| `SM80` | A100 40GB / 80GB, GA100 64GB, A800 40GB / 80GB | 164 KB shared memory per SM |
| `SM86` | RTX 3090 / 3090 Ti, RTX A6000, A40, A10 | 100 KB shared memory per SM; all kernels use ≤ 48 KB per CTA |

## Documentation

| Document | Contents |
| :--- | :--- |
| [REPRODUCTION.md](docs/REPRODUCTION.md) | Base commit and fork lineage, patching, building, verification, pitfalls |
| [BENCHMARKS.md](docs/BENCHMARKS.md) | End-to-end serving, kernel microbenchmarks, multi-GPU topology comparison |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Kernel-by-kernel design, forward-pass diagram, repository layout |
| [01 YOCO prefill compaction](docs/01_YOCO_PREFILL_COMPACTION_SM80.md) | Decoder SWA bounded compaction on `TP2 × PP3` |
| [02 DSpark MTP](docs/02_DSPARK_MTP_VARLEN_TOPK_SAMPLING_SM80.md) | Variable-length verification, confidence head, top-k drafting |
| [03 DeepSelect attention](docs/03_DEEPSELECT_ATTENTION_KV_KERNELS_SM80.md) | Sparse MLA, indexer and KV compression kernels |
| [04 mHC / Engram / MoE](docs/04_MHC_ENGRAM_MOE_ZOS_SM80.md) | Lazy mHC fusion, Engram prefetch, Marlin MoE, zero-overhead scheduling |
| [05 Topology & 1M context](docs/05_TOPOLOGY_1M_CONTEXT_ANALYSIS_SM80.md) | `TP2 × PP3` vs `PP6` vs `TP4 × PP2` capacity and latency analysis |

## Credits

Built on [DeepInfer](https://github.com/deepseek-ai/DeepInfer) (DeepSeek), [vLLM](https://github.com/vllm-project/vllm) and the [`haosdent/vllm`](https://github.com/haosdent/vllm/tree/dsv4-flash-a100) SM80 enablement, [TileLang](https://github.com/tile-ai/tilelang), [Triton](https://github.com/triton-lang/triton), [FlashInfer](https://github.com/flashinfer-ai/flashinfer) and [Marlin](https://github.com/IST-DASLab/marlin). To cite this work, use [CITATION.cff](CITATION.cff).

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
