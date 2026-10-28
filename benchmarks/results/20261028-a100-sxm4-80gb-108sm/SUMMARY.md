# A100-SXM4-80GB (108 SM), 2026-10-28

Run while reviewing PR #3, inside the image built from `docker/Dockerfile.sm80`
(CUDA 13.0, torch 2.13.0+cu130, Triton 3.7.1, TileLang 0.1.12, vLLM
`79a7108d9a` + both patches, `TORCH_CUDA_ARCH_LIST="8.0 8.6"`) with
`scripts/run_microbench_single_gpu.sh`. `verify_env.py` exited 0 and
`check_dropin_sync.py` reported `30 same, 0 diff, 0 missing`.

| Kernel | This run | A800 run (108 SM) | README (74 SM) |
| :--- | :--- | :--- | :--- |
| mHC fused post+pre, `T=12` | 192.1 → 148.4 µs (1.29×) | 2.41× | 2.89× |
| Sparse MLA decode, `B=1..24` | 1.19–1.25×, err ≤ 9.8e-04 | 1.15–1.24× | 2.08–2.14× |
| `combine_topk_swa` (`T=512..4096`) | 25.6–28.7×, bit-exact | 26.4–30.0× | 22.17× |
| `dequantize_and_gather_k_cache` (1K–12K) | 1.45–8.80× per call, bit-exact | 1.59–8.92× | 4.02× @16K, 7.64× @48K |
| Sparse MLA prefill consumer, `T=1536` | 2244 → 2215 µs (1.01×), bit-exact | — | 1.17× |

Every bit-exactness check passes. The mHC and sparse-MLA speedups are well
below the README figures on both 108-SM parts; the tunings were measured on a
74-SM GA100, and the `part3` split sweep shows the default split count leaves
1.29× (`B=16`) and 1.38× (`B=24`) on the table at 108 SMs.

`microbench_sm80_part3.py` TEST 2 / TEST 3, which crashed on the A800 run, now
complete.
