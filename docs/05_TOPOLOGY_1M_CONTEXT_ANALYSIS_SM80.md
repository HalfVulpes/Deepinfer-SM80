# Pillar 5: 6-GPU Topology Tradeoff Analysis (`TP2×PP3` vs. `PP6` vs. `TP2+TP4`) for Single-User Experience & Full 1M Context Streams

## 1. Current Configuration Expected Context Capacity (`baseline-sm80` & `1020-mem3`)

On your live 6× GA100 64GB (`SM80`, 74 SMs, PCIe without P2P) deployment running **`TP2 × PP3` (`partition = 14, 14, 12`) + DSpark `K=5`**:

| Storage Tier | `1019e` (`104 GiB` RAM Tier) | **`1020-mem3` (Current Live, `24 GiB` RAM Tier)** | Exact Token / Stream Capacity & Latency |
| :--- | :--- | :--- | :--- |
| **Tier 1: GPU HBM (`fp8_ds_mla` KV Pool)** | **`4,650,231` tokens (`4.43 × 1M`)** | **`4,756,478` tokens (`4.536 × 1M`)** | **4 concurrent full `1,048,576`-token (1M) streams** active simultaneously in GPU HBM with **zero preemption** (`+562,174` tokens of safety margin above `4 × 1M`). *(Note: Without DSpark `K=5`, `13/13/14` holds **`5,445,683` tokens = `5.19 × 1M` streams**).* |
| **Tier 2: Pinned Host RAM (`/dev/shm`)** | `104 GiB` (`~17 × 1M` streams) | **`24 GiB` (`~4 × 1M` streams)** | Stores `~4` additional evicted/inactive 1M sessions in pinned RAM (`~5.90 GiB` per 1M stream). Restores a full 1M stream to GPU in **`~2.7s`** (`11.07s` for `4×1M`). |
| **Tier 3: NVMe SSD (`/mnt/nvme/kv-ssd`)** | `512 GiB` (`~85 × 1M` streams) | **`512 GiB` (`~85 × 1M` streams)** | Durable across container restarts. Restores `4×1M` (`34.69 GB`) in **`56.5 – 59.9s`** (`~14s` per 1M stream, vs `550s` cold recompute). |

### Why 1M Tokens Takes Only `~5.90 GiB` Cluster-Wide (`~1.40 GiB/GPU` on the Bottleneck Stage)
In `DeepSeek-V4.1-Flash-1017`, global KV is **only** produced at **4 layers (`kv_source_layer_ids = [2, 8, 14, 20]`)**, and uses **1-head MQA (`num_key_value_heads = 1`)**:
- Each compressed KV state is **`584 bytes`** (`512 B` FP8 + `16 B` UE8M0 scale + `64 B` RoPE) and each Indexer K state is **`132 bytes`** (`128 B` FP8 + `4 B` FP32 scale) = **`716 bytes/state`**.
- **Layers 2, 8, 14 (`compress_ratio = 2`)**: 1 state per 2 tokens $\implies \mathbf{358\text{ B/token}}$ per producer layer (`375.4 MiB` per 1M tokens).
- **Layer 20 (`compress_ratio = 1`)**: 1 state per 1 token $\implies \mathbf{716\text{ B/token}}$ (`750.8 MiB` per 1M tokens).
- **All 40 Backbone + 3 DSpark SWA Layers (`sliding_window = 128`)**: Only maintain a 128-token ring buffer (`+32` lookahead for DSpark), plus the SM80 node's `cache_geometry.py` retains `256` SWA tokens every `8,192` tokens (`3.125%` density $\approx 18.25\text{ B/token/layer}$) for prefix caching.
- **Crucial MQA Rule**: Because `num_key_value_heads = 1`, **Tensor Parallelism (`TP=2` or `TP=4`) CANNOT shard the KV cache**—every GPU in a TP group must hold a **full replica** of its PP stage's `compress_kv_cache` and `indexer.k_cache`!

---

## 2. Head-to-Head Comparison: `TP2×PP3` vs. `PP6` vs. `TP2+TP4` on 6× GA100 64GB

To simultaneously **maximize single-user experience (`C1` decode tok/s and 1M TTFT)** and **increase full 1M context streams**, we evaluate all three 6-GPU topologies using exact hardware math and empirical measurements from SM80 Node (`WORK-STATE.md` & `video-research/pp6-OPTIMIZATION_PLAN.md`):

| Metric | **`TP2 × PP3` (`14,14,12` Current)** | **`PP6` (`TP1 × PP6`, `7,7,7,7,7,5`)** | **`TP2 + TP4` (2-GPU `TP2` + 4-GPU `TP4`)** | **`TP2 × PP3` (`v25` Optimized)** |
| :--- | :--- | :--- | :--- | :--- |
| **GPU Allocation (6 Cards)** | 3 stages × 2 GPUs (`2+2+2`) | 6 stages × 1 GPU (`1+1+1+1+1+1`) | Stage 0: 2 GPUs (`TP2`, `L0–13`)<br>Stage 1: 4 GPUs (`TP4`, `L14–39+MTP`) | 3 stages × 2 GPUs (`2+2+2`) |
| **Bottleneck Stage Weights/GPU** | **PP0: `50.35 GiB`** (`PP1: 49.49`, `PP2: 47.84`) | **Rank 0: `50.17 GiB`** (`Rank 2: 48.5`, `Rank 5: 45.4`) | **Stage 0 (`TP2`): `50.17 GiB`**<br>Stage 1 (`TP4`): `47.47 GiB` | **Balanced: `~48.2 GiB/GPU`** *(see Section 3)* |
| **KV Replication per Producer** | `2×` (replicated across 2 TP ranks) | **`1×`** (`L20` shadowed on Ranks 3,4,5) | `2×` on Stage 0 (`L2, L8`)<br>**`4×` on Stage 1 (`L14, L20`!)** | `2×` (replicated across 2 TP ranks) |
| **GPU-Resident KV Tokens (`1M` Streams)** | **`4.76M` tok (`4.53 × 1M`)** *(4 full 1M streams)* | **`4.94M – 5.8M` tok (`4.72 – 5.5 × 1M`)** *(4–5 full 1M streams)* | **`~4.5M – 4.7M` tok (`~4.4 × 1M`)** *(Clamped by Stage 0 & 4× `L20` KV!)* | **`~6.3M – 6.6M` tok (`6.0 – 6.3 × 1M`)** *(**6 full 1M streams** in GPU HBM!)* |
| **C1 Non-Speculative Decode** | **`58.0 – 58.4 tok/s`** *(Fastest)* | `43.4 – 46.4 tok/s` *(`-23%` slower)* | `~55.3 – 56.5 tok/s` *(TP4 PCIe AllReduce penalty)* | **`~65 – 70 tok/s`** |
| **C1 Code Decode (DSpark `K=5`)** | **`141.5 – 151.8 tok/s`** | `65.5 – 68.5 tok/s` *(`-54%` slower!)* | `~125 – 135 tok/s` | **`165.0 – 180.0 tok/s`** |
| **C1 Prose Decode (DSpark `K=5`, `T=1`)** | `57.5 – 72.7 tok/s` | `~44 – 52 tok/s` | `~54 – 68 tok/s` | **`90.0 – 112.0 tok/s`** *(Pillar 2 `MTPVarlen` + Top-128)* |
| **112K / 1M Cold Prefill TTFT** | `36.6s` (112K) / `549.9s` (1M) | `27.6s` (112K) / `~460s` (1M) | `~37.5s` (112K) / `~560s` (1M) | **`~19s` (112K) / `~255s` (1M)** *(Pillar 1 YOCO + Pillar 3/4)* |
| **Verdict** | **Current Baseline** (4×1M streams, fast C1 code) | **Rejected for Single-User UX** (`C1` decode drops by **>2×** under DSpark!) | **Suboptimal on PCIe No-P2P** (`TP4` adds PCIe AllReduce latency & `4×` KV replication) | **🏆 Winner (`TP2 × PP3` + `v25` Optimizations)** |

---

### 2.1 Why `PP6` (`TP1 × PP6`) Hurts Single-User Experience Severely
1. **Only 1/6th of GPU Memory Bandwidth Is Active at Any Instant During C1 Decode**:
   - In pure Pipeline Parallelism (`PP6`, `TP=1`), a single user's token batch (`1` token, or `6` tokens under DSpark) executes sequentially on **1 GPU at a time** (`GPU0 -> GPU1 -> GPU2 -> GPU3 -> GPU4 -> GPU5`).
   - At any millisecond during single-user decode, **5 out of 6 GPUs (83.3% of your cluster's HBM2 bandwidth and SMs) are sitting completely idle**!
   - By contrast, under `TP2 × PP3`, **2 GPUs work in parallel at every stage**, doubling the effective HBM bandwidth for streaming dense MXFP8 weights (`wq_a`, `wq_b`, `wo_b`, `shared_experts`), `lm_head`, and active routed experts, and doubling the SMs available for Sparse MLA and the Lightning Indexer.
2. **DSpark `K=5` Drafting Runs Unsharded on 1 GPU (`Rank 5`)**:
   - Under `PP6`, all 3 DSpark layers (`L40–L42`, `9.72 GiB`), the `129,280`-vocab `lm_head`, and the 5-step `markov_head` execute on a **single GPU (`TP=1`)** instead of being sharded across `TP=2` (`prepare_dspark_sharded.py`), nearly doubling DSpark drafting overhead.
   - This is why empirical `PP6 + DSpark K=5` measurements (`video-research/pp6-OPTIMIZATION_PLAN.md:69`) topped out at **`65.5–68.5 tok/s`** on C1 decode, whereas `TP2 × PP3 + DSpark K=5` on SM80 Node achieves **`141.5–151.8 tok/s`** (**`2.15×` faster single-user decode!**).

---

### 2.2 Why `TP2 + TP4` (Heterogeneous `2 + 4 = 6` GPUs) Loses to `TP2 × PP3` on PCIe (No-P2P)
On a 6-GPU node (`2 + 4 = 6`), `TP2 + TP4` assigns **2 GPUs (`TP=2`) to Stage 0 (`L0–13`)** and **4 GPUs (`TP=4`) to Stage 1 (`L14–39 + DSpark`)** (note: running two independent model replicas on 2 GPUs + 4 GPUs is impossible because the `288.7 GiB` weights do not fit in `128 GiB` or `256 GiB` without catastrophic PCIe CPU weight offloading, as proven in `efficiency-flex-gpu-20261017`).

Even as a 2-stage heterogeneous pipeline (`Stage 0 = TP2`, `Stage 1 = TP4`), `TP2 + TP4` is strictly worse than `TP2 × PP3` on GA100 PCIe (No-P2P) for **three hardware reasons**:
1. **PCIe Host-Staged AllReduce Overhead on `TP4` Slows Down Single-User Decode**:
   - With `NCCL_P2P_DISABLE=1`, a `TP=2` AllReduce is a direct 2-rank exchange, whereas `TP=4` requires a 4-rank Ring AllReduce across PCIe (`2×` per transformer layer = **58 AllReduces on Stage 1**).
   - In the SM80 node's matched topology trials (`WORK-STATE.md:249, 349`), increasing `TP=2` to `TP=4` **reduced** single-stream C1 decode speed from **`58.40 tok/s` (`TP2`) down to `55.52 tok/s` (`TP4`)** because decode MoE only activates `6` small experts per token (`M=1..6`), making `TP4` collective latency larger than the weight-read savings!
2. **Marlin MXFP4 Intermediate Dimension Misalignment on `TP4` (`2304 / 4 = 576`)**:
   - `moe_intermediate_size = 2304`.
   - At **`TP=2`**: $2304 / 2 = \mathbf{1152 = 9 \times 128}$ — an **exact multiple of 128** (Marlin's native optimal N-tile width, zero padding!).
   - At **`TP=4`**: $2304 / 4 = \mathbf{576 = 4.5 \times 128}$ — not divisible by 128! Stock Marlin pads `576 -> 640` (`+11.1%` wasted FLOPs/VRAM), and even with `VLLM_SM80_DSV41_MARLIN_ALIGN64=1` (`64`-wide tiles), 64-wide tiles have lower Tensor Core L2 reuse than 128-wide tiles on `TP2`.
3. **Zero Gain in 1M Context Capacity (`4×` KV Replication on Stage 1)**:
   - Because `num_key_value_heads = 1`, the `TP=4` stage must replicate Layer 14 (`358 B/tok`) and Layer 20 (`716 B/tok`) KV caches across **all 4 GPUs** (`4 × 1.074 KB/tok = 4.30 KB/tok` of cluster VRAM per token on Stage 1, vs `2 × 1.074 = 2.15 KB/tok` on `TP2`!), while Stage 0 (`TP2`, `L0–13`, `50.17 GiB/GPU` weights) still clamps total KV capacity at `~4.5 × 1M` streams!

---

## 3. How to Unlock `6 × 1M` Full Context Streams AND `165–180 tok/s` Single-User UX on `TP2 × PP3`

Why is current `TP2 × PP3` (`14, 14, 12`) capped at `4.53 × 1M` (`4,756,478` tokens), and how do we expand it to **`6.0+ × 1M` (`~6.3M–6.6M` tokens)** on the exact same 6 GPUs?

### 3.1 The Hidden Stage Imbalance in Current `14, 14, 12`
Look at the current per-GPU weight footprint across the 3 stages of `TP2 × PP3` (`14, 14, 12`):
- **PP0 (`GPUs 0–1`, `L0–13` + `embed_tokens` + `vision`)**: **`50.35 GiB/GPU`** $\implies$ leaves **`6.67 GiB/GPU`** for KV cache (**THE BOTTLENECK**).
- **PP1 (`GPUs 2–3`, `L14–27`)**: **`49.49 GiB/GPU`** $\implies$ has **`+0.86 GiB/GPU` stranded VRAM**.
- **PP2 (`GPUs 4–5`, `L28–39` + `lm_head` + `DSpark L40–42`)**: **`47.84 GiB/GPU`** $\implies$ has **`+2.51 GiB/GPU` stranded VRAM**!

Because vLLM sizes the KV block pool to the **minimum** across all PP stages, **PP0 (`50.35 GiB/GPU`) clamps the entire 6-GPU cluster**, leaving `~6.7 GiB` of aggregate VRAM stranded on PP1 and PP2!

### 3.2 Four Concrete Steps to Reach `6.0+ × 1M` Streams (`~6.4M` Tokens) on `TP2 × PP3`

1. **Eliminate `wo_a` BF16 Dequantization (`+0.50 GiB/GPU` Saved on PP0/PP1, `+0.55 GiB/GPU` on PP2)**:
   - Currently (`Patch 0008`), `self_attn.wo_a` (`[8, 1024, 4096]` MXFP8 = `32 MiB/layer`, `16 MiB/GPU` at TP2) is dequantized to BF16 (`32 MiB/GPU/layer`) at load time, adding **`+0.45–0.55 GiB/GPU`** of static weight memory per stage (`+3.0 GiB` across the 6 GPUs).
   - Dequantizing `wo_a` on the fly inside our 74-SM Split-K `sm80_recipe_thin` / Marlin W8A16 grouped GEMM keeps `wo_a` in native MXFP8 (`1B`), immediately freeing **`~0.50 GiB/GPU`** on every stage!

2. **In-Place mHC Residual Aliasing (`sm80_mhc.py`) + Lazy `HCLazyPostData` + Direct `cuMemAlloc` `profile_run` (`+0.95 GiB/GPU` Freed for KV Pool)**:
   - During startup `profile_run` (`max_num_batched_tokens = 1536`), allocating new `[1536, 4, 5120]` BF16 tensors (`60 MiB` each) across layers plus temporary KV blocks inside PyTorch's caching allocator inflates the non-KV peak activation reservation by **`~1.1 GiB/GPU`**.
   - Enabling `sm80_mhc.py` (in-place `hc_commit_`) + `DeepInfer`'s `HCLazyPostData` and using `cuMemAlloc` (`DeepInfer/csrc/cuda_utils.cpp`) for temporary profiling buffers frees **`~0.95 GiB/GPU`** directly into the usable KV cache pool!

3. **Deduplicate Drafter `embed_tokens` or Balance Partition to `13, 14, 13` (`+0.70 GiB/GPU` Bottleneck Relief)**:
   - Moving 1 layer from PP0 (`L13`) to PP2 (`partition = 13, 14, 13`) reduces PP0 weights by **`-3.44 GiB/GPU`** (`50.35 -> 46.91 GiB/GPU`) while raising PP2 to `51.28 GiB/GPU` (or **`49.55 GiB/GPU`** after keeping `wo_a` in MXFP8 and sharing/quantizing drafter `embed_tokens`!).
   - This reduces the maximum weight footprint across **all three stages** from `50.35 GiB/GPU` down to **`48.99 GiB/GPU`** (`+1.36 GiB/GPU` net reduction on the worst stage!).

4. **Drop Consumer-Layer SWA Checkpoint Retention (`L21–L39`) via YOCO Bounded Replay (`+0.35 GiB/GPU` KV Density Gain)**:
   - As `DeepInfer` (`cache_engine.py:221-307`) demonstrates, with YOCO `KEEP_TOKENS = 140..256`, prefix checkpoints only need to store `compress_kv_cache` (`L2, 8, 14, 20`) and Encoder SWA (`L0–19`)—consumer SWA rings (`L20–39`) for any prefix cache hit are regenerated in **`< 15 ms`** by replaying just the last `140` tokens through `L20–39`!

### 3.3 Resulting KV Pool & Stream Capacity on `TP2 × PP3`
Combining Steps 1–4 increases the bottleneck stage's available KV pool from **`6.67 GiB/GPU` to `~9.15 GiB/GPU`**, increasing GPU-resident KV capacity on the same 6× GA100 64GB cards from:
$$\mathbf{4,756,478\text{ tokens } (4.53 \times 1\text{M})} \;\longrightarrow\; \mathbf{\sim 6,520,000\text{ tokens } (6.22 \times 1\text{M full context streams!})}$$
while simultaneously delivering **`~2.0×` faster 1M cold prefill (`~255s`)** and **`165–180 tok/s` (code) / `90–112 tok/s` (prose)** single-user decode throughput!
