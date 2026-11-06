# Changelog

Notable changes to DeepInfer-SM80. There are no tagged releases yet, so entries are
dated and name the commits.

## Unreleased

### Fixed

- **Some layers could use another index source's top-k** (`e2b585f`).
  `amd/rocm.py` memoised the ragged top-k indices on the sliding-window metadata
  that vLLM shares between the layers of a KV-cache group, keyed by the shared
  buffer's address instead of the layer's index source. Layers that follow
  different index sources could then use an earlier source's top-512 on contexts
  longer than about 512 tokens (ratio-1 layers) or 1K tokens (ratio-2 layers).
  Found by code review and reproduced by the CPU regression test; a before/after
  A/B on 8× GA100 64GB showed no measurable change in long-context copy fidelity
  or speed. The fix restores DeepSeek's reference behaviour. Every commit from
  `2a828fc` to `e32bbfe` has the old code; see
  [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md).
- The fix's comments in `amd/rocm.py` now fit vLLM's 88-column lint limit.

### Added

- `tests/test_topk_source_reuse.py`: a GPU-free regression test for the per-layer
  top-k rule, run by the new `tests` workflow on CPU-only PyTorch.
- `docs/KNOWN_ISSUES.md` and this changelog.

### Changed

- `docs/ARCHITECTURE.md` now describes how layers reuse their index source's top-k
  (the old section described a forward-context cache that the code does not have)
  and gives the right compression ratios in the forward-pass diagram.
- `docs/BENCHMARKS.md` notes that its end-to-end numbers predate the fix and
  that the A/B showed no measurable change.
- The microbenchmark suite descriptions in `ARCHITECTURE.md` and `REPRODUCTION.md`
  match what the scripts run; `REPRODUCTION.md` lists the unit tests.

## 2026-10-28

- `e32bbfe` Slimmer README; details moved to `docs/`.
- `f41fd64` Build-step corrections found while testing PR #3 on an A100.
- `3fc77ce` PR #3 by Louie0806: reproducible SM80 build and verify scripts,
  `docker/Dockerfile.sm80`, `docs/REPRODUCTION.md`, independent A800-80GB
  microbenchmarks and the GPU-free `patch-stack` CI.

## 2026-10-24

- `8895510` Add the missing `pp_share.py` and complete the `TP2 × PP3` /
  `TP4 × PP2`, DSpark and 1M-context indexer modules (issue #2).

## 2026-10-22

- `e375de2` Link the vLLM base (PR #56201, `79a7108d9a`) and the `haosdent/vllm` SM80
  fork; add `patches/00-vllm-pr56201-sm80-base.patch` (issue #1).
- `a11809d` 8-GPU `TP4 × PP2` decode and prefill scaling benchmarks and charts.

## 2026-10-20

- `2a828fc` Initial release: kernels, vLLM patch, benchmarks and 1M-context
  architecture docs.
