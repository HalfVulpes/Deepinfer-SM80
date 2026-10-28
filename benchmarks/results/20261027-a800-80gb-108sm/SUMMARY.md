# Independent run — A800-SXM4-80GB, 2026-10-27

First reported run on **108-SM** Ampere silicon (the repository's docs and
benchmark headers say 74 SMs; see the `sms:` line in `env.txt`).

```
host:       gpu4
gpu:        NVIDIA A800-SXM4-80GB
capability: 8.0
sms:        108
torch:      2.13.0+cu130
vllm:       0.1.dev1+g79a7108d9.d20261025
git:        eaf7c6a (this branch)
```

Raw logs: `microbench_sm80_kernels.log`, `microbench_sm80_part2.log`,
`microbench_sm80_part3.log` in this directory.

## mHC fused post+pre (83 boundaries / step)

| `T` | baseline (µs) | fused (µs) | speedup | err(res) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 371.28 | 182.22 | 2.04× | 0 |
| 6 | 176.23 | 119.36 | 1.48× | 0 |
| 12 | 506.33 | 210.32 | 2.41× | 0 |
| 16 | 190.32 | 130.72 | 1.46× | 0 |
| 32 | 218.05 | 167.21 | 1.30× | 0 |
| 512 | 221.13 | 172.35 | 1.28× | 0 |
| 1536 | 237.70 | 207.91 | 1.14× | 0 |
| 4096 | 576.44 | 481.67 | 1.20× | 0 |

The README reports 2.89× at `T=12`; we measure 2.41×, but the absolute
baseline also differs (357.7 µs there vs 506.3 µs here), so this looks like
clock/SM-count differences rather than a regression.

## Sparse MLA decode (41 launches / step)

| `B` | baseline (µs) | best (µs) | speedup | err |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 84.40 | 68.25 (`w=4`) | 1.24× | 0 |
| 6 | 85.75 | 69.58 (`w=8`) | 1.23× | 0 |
| 12 | 134.19 | 112.64 (`w=8`) | 1.19× | 4.9e-04 |
| 24 | 252.36 | 213.35 (`w=8`) | 1.18× | 4.9e-04 |

The README's 2.08–2.14× uses a different baseline split configuration; at the
default splits the gain here is 1.15–1.24×.

## `combine_topk_swa_indices` (30× / prefill chunk)

| `T` | torch (12 ops, µs) | triton (1 op, µs) | speedup | bit-exact |
| ---: | ---: | ---: | ---: | :---: |
| 512 | 1037.88 | 39.26 | 26.43× | yes |
| 1536 | 1079.71 | 39.80 | 27.13× | yes |
| 4096 | 1197.87 | 39.88 | 30.04× | yes |

README claims 22.17× (1083.72 → 48.88 µs); the independent run is faster on
both sides and stays bit-exact.

## `dequantize_and_gather_k_cache`

| compressed tokens | baseline (µs) | SM80-LUT (µs) | per-call | 38-layer → 8-source | bit-exact |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 1024 | 29.12 | 18.32 | 1.59× | 1.11 ms → 0.15 ms (7.55×) | yes |
| 4096 | 104.94 | 18.26 | 5.75× | 3.99 ms → 0.15 ms (27.30×) | yes |
| 12288 | 304.82 | 34.18 | 8.92× | 11.58 ms → 0.27 ms (42.36×) | yes |

README reports 4.02× @16K and 7.64× @48K; the per-call numbers scale with the
context length used, so these are consistent in trend.

## Known failure in `microbench_sm80_part3.py`

`TEST 2: Decode Split & Warp Tuning` crashes at `B=16`:

```
TypeError: unsupported format string passed to NoneType.__format__
  at benchmarks/microbench_sm80_part3.py:174
```

`default_t` is only assigned when the sweep hits `(num_splits == default_s and
n_warps == 8)`; for `B >= 16` that combination is not in the search space, so
`default_t` stays `None` and the f-string formatting fails. `TEST 1`
(bit-exactness) passes for all three sizes.
