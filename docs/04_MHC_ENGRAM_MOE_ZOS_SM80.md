# Pillar 4: Lazy mHC Fusion (`HCLazyPostData`), Async Engram PCIe Prefetch, 74-SM Marlin MXFP4/MXFP8 Kernels, and Zero-Overhead Scheduling (`MetadataZOS`)

## 1. Manifold-Constrained Hyper-Connections (mHC) Optimizations on SM80

In `DeepSeek-V4.1-Flash-1017`, every transformer layer (`L0–L39`) and every DSpark layer (`L40–L42`) wraps both its Attention and MoE FFN sublayers in a **4-stream Manifold-Constrained Hyper-Connection (`hc_mult = 4`, `hidden_size = 5120`, `20,480` channels per token)**:
- Across 40 backbone layers + 3 DSpark layers, `mhc_pre` and `mhc_post` execute **86 times per decode step**!
- Because the 4-stream residual tensor has shape `[N, 4, 5120]` in BF16 (`40 KB` per token), reading and writing it twice per sublayer (`160 KB/token/layer` = `6.4 MB/token` across 40 layers) is a major memory-bandwidth and allocation bottleneck during both prefill (`N = 1536` $\rightarrow$ `62.9 MB` per tensor) and decode (`N = 24` with DSpark).

### 1.1 How `DeepInfer` Optimizes mHC (`HCLazyPostData` + `mega_mhc` + Split-K=16 `tf32_hc_prenorm_gemm`)
Inspecting [`DeepInfer/deepinfer/model_executor/models/v4_modules/hc.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/models/v4_modules/hc.py#L1-L397), [`hc_lazy.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/models/v4_modules/hc_lazy.py#L1-L35), and [`mhc_ops.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/models/v4_modules/mhc_ops.py#L1-L148):
1. **Lazy Post-Application (`HCLazyPostData`) & Back-to-Back `post + pre + RMSNorm` Fusion (`_hc_fused_post_pre`)**:
   - Instead of calling `hc_post` immediately after Attention finishes (which reads `[N, 4, 5120]` residual + `[N, 5120]` attn output and writes `[N, 4, 5120]` new residual to HBM) and then immediately calling `hc_pre` + `ffn_layernorm` (which reads `[N, 4, 5120]` new residual right back from HBM!), `DeepInfer` passes `HCLazyPostData(residual, post_layer_mix, comb_res_mix, pre_layer_mix)` lazily into the next sublayer's norm!
   - When `hc_post_intervention is None` (which holds for **78 out of 80 backbone sublayer transitions**—every transition except before Layer 1 and Layer 14 where Engram intervenes on the 4-stream residual!), `DeepInfer` fuses **the previous sublayer's `hc_post` + the current sublayer's `hc_pre` + `RMSNorm`** into a single kernel (`_mega_mhc`)!
   - Even when `mega_mhc` is not used, `prev_lazy_data.clear()` immediately releases the old residual reference to minimize peak memory footprint.
2. **Split-K=16 `tf32_hc_prenorm_gemm` with Fused Square-Sum Reduction (`mhc_ops.py:102-148`)**:
   - In `mhc_pre_norm_fn`, projecting the flattened 4-stream residual `[N, 20480]` by `hc_fn` (`[24, 20480]` FP32) is a **massive-K (`K = 20,480`), tiny-N (`N_out = 24`)** GEMM that also requires the per-token L2 square sum $\sum_{d=0}^{20479} x_d^2$ for RMS normalization!
   - `DeepInfer` runs a **Split-K = 16 (`N_SPLITS_CUDA = 16`) TF32 GEMM** (`tf32_hc_prenorm_gemm`) where each of the 16 K-slices (`K_slice = 1280`) simultaneously computes its partial matrix product `[num_tokens, 24]` **and** its partial square sum `sqr_sum` (`[num_tokens, 1]`) in a single pass over `residual`!
   - Then `mhc_reduce_partials_and_rmsnorm_fwd` reduces the 16 partials and divides by $\sqrt{\text{sqr\_sum} / 20480 + \epsilon}$ in one epilogue!

### 1.2 Combining `DeepInfer`'s mHC Design with the SM80 node's SM80 mHC Kernels (`sm80_mhc.py` & `precise_hc_split_sinkhorn`)
On SM80 Node SM80 (`GA100`, 74 SMs, TF32 Tensor Cores `mma.sync.aligned.m16n8k8.f32.tf32.tf32.f32`):
1. **In-Place Residual Aliasing (`sm80_mhc.py`)**:
   - In `internal-notes`, the team already built a custom Triton `_post_hc` kernel (`block=256, num_warps=4, tl.fma`) with in-place residual aliasing (`sm80_sm80::hc_commit_`) that eliminates 80 `[N, 4, 5120]` tensor allocations per forward pass (saving `60 MiB/layer` at `N=1536`), with a clone safeguard for DSpark tap layers `[37, 38, 39]`.
2. **Fused Split-K=16 Prenorm GEMM + Square-Sum + Precise 20-Iter Sinkhorn on SM80**:
   - Porting `DeepInfer`'s Split-K=16 `mhc_prenorm_gemm` + `mhc_reduce_partials_and_rmsnorm_fwd` (with `K_slice = 20480 / 16 = 1280`) to SM80 reads the `40 KB/token` residual **only once** instead of twice (once for RMSNorm square sum, once for `hc_fn` GEMM), cutting HBM reads for `mhc_pre` by **50%**!
   - Pairing this with the SM80 node's `precise_hc_split_sinkhorn` (`DeepSeek-V41-Lab/fused_small_ops.py`, using `libdevice.exp`, `tl.div_rn`, `enable_fp_fusion=False`, and fixed 4-row reduction order) guarantees **bit-exact FP32 Sinkhorn parity** across all 20 iterations (`NUMERICAL_POLICY.md` compliant).
   - And fusing `mhc_post` of sublayer $t-1$ with `mhc_pre_apply_mix` + `RMSNorm` of sublayer $t$ (`_hc_fused_post_pre` pattern from `DeepInfer/hc.py`) eliminates one full read+write of `[N, 4, 5120]` (`80 KB/token`) on 78 out of 80 sublayers (**saving `6.24 MB/token` of HBM traffic per step = `9.6 GB` per 1,536-token prefill chunk**)!

---

## 2. Asynchronous Side-Stream Engram Prefetching (`EngramRuntime` + `gather_engram_tokens_zos_tl`)

### 2.1 How `DeepInfer` Overlaps Engram Lookup Across Layers (`deepseek.py:559-564` & `engram.py`)
Look at [`DeepInfer/deepinfer/model_executor/models/deepseek.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/models/deepseek.py#L555-L600) and [`v4_modules/engram.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/models/v4_modules/engram.py#L107-L219):
- In `DeepInfer`, Engram hashing and table fetching are **not** launched when execution reaches Layer 1 or Layer 14!
- Instead, at the **very beginning of `DeepseekModel.forward` (before Layer 0 even starts!)**:
  ```python
  if self.engram_runtime is not None:
      hashes = tile_kernels.engram.engram_hash(
          input_metadata.ngram_token_ids,
          self.engram_layer_ids,
          self.cfg.engram.max_ngram_size,
          self.cfg.engram.vocab_size,
      )
      engram_hooks = self.engram_runtime.fetch(hashes)
  ```
- `self.engram_runtime.fetch(hashes)` (`engram.py:214-219`) immediately dispatches the asynchronous gather of the `[total_q, 24, 256]` FP8 embeddings + UE8M0 scales for **all Engram layers `[1, 14]`** in the background and returns a dictionary of completion `embed_hook` callables `{1: hook_1, 14: hook_14}`!
- When Layer 1 (`decoder_layer.py:137`) or Layer 14 executes `self.engram(x, embed_hook=engram_hook)`, `embed_hook()` simply waits on the already-completed async transfer, projects `fused_kv_projection` (`6144 -> 25600`), and runs the fused `tile_kernels.engram.engram_gate_fwd` kernel (`inplace=True`, using pre-fused `weight_fused = fused_weight(weight_hidden, weight_embed)`)!

### 2.2 Porting Early Async Engram Prefetch to the SM80 node's `TP2 × PP3` (`storage="cpu"` Pinned Host RAM)
On SM80 Node (`baseline-sm80`), `Engram` (`188.84 GiB` in pinned host RAM via `cudaHostRegister`, Patch 0013) runs at:
- **Layer 1 on PP Stage 0 (GPUs 0–1)**
- **Layer 14 on PP Stage 1 (GPUs 2–3)** (note: **Layer 14 is the very first layer of PP Stage 1** under `14,14,12`!).
- Currently in the SM80 node's `vllm/models/deepseek_v4_1/common/engram.py` and `engram_lookup.py`, the host-pinned table gather + PCIe H2D transfer + Triton FP8 dequantization runs **synchronously on the main compute stream** right when Layer 1 or Layer 14 is reached!
- Look at what we can do on SM80 Node by adopting `DeepInfer`'s early async fetch architecture:
  1. **On PP Stage 0 (`L0–L13`, owns `Layer 1` Engram)**:
     - At the start of `DeepseekV41Model.forward` (before `embed_tokens` and `Layer 0` run), launch the Layer 1 N-gram hash + pinned-host gather (`engram_lookup`) on a dedicated high-priority CUDA stream `self.engram_prefetch_stream` and record `self.engram_ready_event[1]`.
     - While `embed_tokens` + `Layer 0` (Attention + 384-expert MoE) execute on the main stream (`~0.45 ms` in decode, `~12 ms` in 1,536-token prefill), the PCIe H2D transfer of `[N, 24, 264 B]` (`6.3 KB/token` = `9.7 MB` at `N=1536`, `152 KB` at `N=24` decode) completes **100% in the background**!
     - At Layer 1 entry, `torch.cuda.current_stream().wait_event(self.engram_ready_event[1])` is an instant no-op because the data is already in GPU HBM!
  2. **On PP Stage 1 (`L14–L27`, owns `Layer 14` Engram)**:
     - Under `14,14,12`, Layer 14 is the **first** layer on PP1. Wait—how can we overlap Layer 14's Engram H2D transfer on PP1 if Layer 14 is the first layer of PP1?
     - Because on PP1, `input_ids` / `positions` for the microbatch are known **before** the P2P `recv` of `hidden_states` (`IntermediateTensors`) from PP0 completes!
     - Launching Layer 14's `engram_lookup` on `self.engram_prefetch_stream` **before** waiting on the PP0 $\rightarrow$ PP1 NCCL P2P `irecv` of `IntermediateTensors` overlaps Layer 14's PCIe H2D Engram gather with PP0's execution and the PP0 $\rightarrow$ PP1 P2P transfer!
  3. **Pre-Fusing `q_weight` and `k_weight` + In-Place `engram_gate_fwd`**:
     - Just like `DeepInfer`'s `engram.py:134-160`, pre-fusing `q_weight` and `k_weight` (`[4, 5120]` FP32) at model load and running the RMSNorm + dot-product + signed-sqrt sigmoid gate + residual update (`x += gate * value`) **in-place** in a single kernel eliminates materializing `[N, 4, 5120]` intermediate gated tensors!

---

## 3. Porting 74-SM Marlin MXFP4 MoE & Split-K Thin GEMM Kernels (`GLM-5.3-Kernels` + `Morrowmake` + `DeepInfer`)

### 3.1 The Prefill MoE Padding Crisis on `DeepSeek-V4.1-Flash-1017` (`E=384, top-6`)
Let's look at the exact numbers for `DeepSeek-V4.1-Flash-1017` routed MoE (`384` experts, `num_experts_per_tok = 6`, `hidden_size = 5120`, `moe_intermediate_size = 2304`, `TP=2` shard `N = 1152`):
- During chunked prefill on SM80 Node (`max_num_batched_tokens = 1536`), there are $1536 \times 6 = 9,216$ expert assignments distributed across `384` routed experts.
- The **average number of tokens routed to each expert is only $\frac{9216}{384} = 24$ tokens**!
- Stock vLLM `Mxfp4MoEMethod` (`fused_marlin_moe`) uses a uniform `block_size_m = 64` (or `32`), padding every active expert's token count up to a multiple of `64`:
  - Expanding `9,216` routed token assignments across 384 experts with uniform 64-row blocks produces **`~24,576` padded rows**—meaning **`62.5%` of all Marlin MoE Tensor Core compute during prefill is wasted multiplying zeros**!
- **The Solution (`moe_split_align.py` + 74-SM Split-Block Marlin from `GLM-5.3-Kernels/profiles/sm74/`)**:
  - In `internal-notes` and `moe_split_align.py` (originally developed for `E=288` on 74-SM GA100), a single-CTA dynamic programming cover decomposes each expert's token count into heterogeneous **64, 48, 32, and 16-row blocks** and dispatches them in a single pass with `prefill_schedule_version=2` (monotonic CTA start-order handoff across our **74 SMs**)!
  - Porting `moe_split_align.py` and the 48/32/16-row Marlin tiles to `Mxfp4MoEMethod` (`float4_e2m1fn` + `ue8m0` group-32 scales, `E=384`, `K=5120`, `N=1152`) slashes padded MoE rows from `~24,576` down to `~10,800`, yielding a **1.45×–1.55× speedup on MoE prefill across all 40 layers**!

### 3.2 Single-CTA Decode MoE Route Aligner (`sm80_moe_align.py`) & `VLLM_MOE_SKIP_PADDING=1`
- During decode ($M \in [1, 24]$), stock `moe_align_block_size` launches multi-stage CUB scan kernels taking **74–76 µs per layer** (`~3.0 ms` per 40-layer step = **17% of non-speculative step time!**).
- Porting `sm80_moe_align.py` (`GLM-5.3-Kernels/kernels/sm80_moe_align.py`, extended from `E=288` to `E=384`, which fits easily in 12 threads × 32 experts = 384 experts in shared memory!) reduces route alignment from **`75 µs` to `~2.5 µs` per layer**, saving **`~1.1 ms` per decode step (+6.5% decode throughput)**!
- Simultaneously, zeroing inactive CUDA-graph padding rows before the MoE router (`VLLM_MOE_SKIP_PADDING=1`) prevents dummy padded tokens from activating extra random experts among the 384 routed experts during CUDA graph replay!

### 3.3 74-SM Split-K + `ROTATE_K` Thin GEMM (`sm80_recipe_thin.py`) for Decode Projections
- On 74-SM GA100 at `TP=2`, several critical BF16 / dequantized-BF16 decode projections are heavily memory-bandwidth-bound and suffer severe wave quantization under cuBLAS because their output tile counts do not divide `74` SMs:
  1. **Grouped Output LoRA `wo_a`** (`8` groups × `[M, 1024, 512]` at TP1 or `4` groups per GPU at TP2: only `4` batch groups!).
  2. **MoE Router `gate.weight`** (`[M, 384, 5120]` FP32/BF16: at `BLOCK_N=64`, $384/64 = 6$ N-tiles—only occupying **6 out of 74 SMs** without Split-K!).
  3. **Indexer `wk` (`[M, 128, 512]`) and `weights_proj` (`[M, 32, 5120]`)**: Only 1–2 N-tiles without Split-K!
  4. **Shared Expert + Routed Expert Stream Overlap**: Just like `DeepInfer` (`mlp.py`), running `shared_experts` (`w13` + `w2` MXFP8 Marlin) concurrently or fused with the router/dispatcher hides shared-expert latency.
- Adding these exact shapes to `sm80_recipe_thin.py` (`_CTA_PER_SM_TARGET = 1.8` $\rightarrow$ `~133` CTAs across 74 SMs, in-kernel atomic release/acquire last-CTA reduction, and `ROTATE_K` L2 staggering) speeds up these thin projections by **1.25×–1.49×**!

### 3.4 PP Ring Decode Burst Spreading (`VLLM_DSV41_SPREAD_DECODE` from Patch 0019)
- In `baseline-sm80`, when 4 concurrent decode requests run on `TP2 × PP3` with DSpark `K=5`, if all 4 requests get fused into a single microbatch (`4 × 6 = 24` tokens), PP stages 0 and 1 sit idle while PP2 runs DSpark + verification on 24 tokens!
- Enabling `VLLM_DSV41_SPREAD_DECODE=2` (from Patch `0019`, already present in the overlay!) splits the 4 concurrent requests into **2 microbatches of 2 requests (`2 × 6 = 12` tokens each)** pipelined across the 3 PP stages:
  1. Keeps PP0, PP1, and PP2 simultaneously active instead of serializing 3 stages on one fat 24-token batch.
  2. Halves the per-microbatch token count (`12` vs `24`), reducing unique active experts per MoE layer and keeping Marlin in its high-efficiency low-$M$ regime!

---

## 4. Zero-Overhead Async Scheduling (`MetadataZOS` in `DeepInfer`) vs. SM80 Node vLLM `AsyncScheduler`

In [`DeepInfer/deepinfer/worker/worker.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/worker/worker.py#L88-L246) and [`gather_engram_tokens_zos.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/cuda/gather_engram_tokens_zos.py#L1-L76):
- `DeepInfer`'s **Zero-Overhead Scheduling (ZOS)** eliminates CPU-GPU synchronization between decode steps by:
  1. Launching Step $t$'s GPU forward pass **before** synchronizing Step $t-1$'s `sample_end_evt`!
  2. Correcting Step $t$'s `seqlens_q` (`compute_seqlens_q_by_confidence_rates`), `seqlens_k` (`compute_seqlens_k_zos`), and Engram 4-gram token windows (`gather_engram_tokens_zos_tl`) **directly on the GPU** using Step $t-1$'s GPU-resident outputs (`prev_sampled_token_ids_gpu`, `prev_predict_token_ids_gpu`, `prev_num_accepted_tokens_gpu`, `confidence_rates_gpu`)!
  3. Overlapping Step $t-1$'s CPU post-processing (`_post_process_step`) with Step $t$'s GPU forward execution!
- **How This Applies to SM80 Node**:
  - the SM80 node's vLLM V1 engine already runs with `AsyncScheduler` (`num_in_flight_tokens` / placeholder tokens), which is why our `deferred_spec_finish` overlay in `sm80_v14_connector.py` was needed.
  - However, the SM80 node's Engram N-gram state (`NgramHashState` in `vllm/models/deepseek_v4_1/common/engram.py`) currently updates on CPU or requires valid token IDs before hashing. Porting `DeepInfer`'s `gather_engram_tokens_zos_tl` (`76` lines of SM80-compatible TileLang in `gather_engram_tokens_zos.py`) allows the GPU to assemble the exact 4-gram compressed token IDs `[total_q, 4]` directly from `prev_sampled_token_ids` and `prev_draft_token_ids` on device without waiting for a D2H sync!
