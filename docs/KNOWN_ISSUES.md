# Known issues

## Fixed: some layers could use another index source's top-k beyond ~512 tokens

| | |
| :--- | :--- |
| **Affected code** | every commit from the initial release `2a828fc` (2026-10-20) through `e32bbfe` (2026-10-28): `patches/vllm-sm80-deepinfer.patch` and the drop-in `vllm/models/deepseek_v4_1/amd/rocm.py` |
| **Fixed in** | `e2b585f` |
| **Found by** | code review against DeepSeek's reference; reproduced by the CPU regression test `tests/test_topk_source_reuse.py` |
| **Measured effect** | none measurable: in a before/after A/B, long-context copy fidelity and speed could not be told apart (below) |
| **Why update** | the fix restores DeepSeek's reference behaviour |

### What the old code did

DeepSeek-V4.1 computes a top-512 selection of compressed positions only on its
index-source layers (2, 8, 14, 20, 24, 28, 32 and 36). Every other compressed layer
attends over the selection of the most recent index source below it, as DeepSeek's
reference does (`shared_attn.topk_idxs` in `inference/model.py`). In vLLM the
index sources write that selection into one shared `topk_indices_buffer`.

The DeepInfer-SM80 version of `amd/rocm.py`, whose attention class SM80/SM86 runs
through `ampere/ampere_sparse.py`, memoised the ragged indices derived from that
buffer (`_cached_decode_topk`, `_cached_prefill_ragged`) on the sliding-window
attention metadata. vLLM builds one metadata object per KV-cache group and deals
layers into groups round-robin, so one object is shared by layers that follow
different index sources. The memo was keyed by the buffer's address and the batch
layout, not by the index source, so a layer could reuse indices that another layer
of its group had derived for an earlier source.

With 4 sliding-window groups, for example, the code hands layers 9–11, 13, 15–17,
19, 25–27, 29–31, 33–35 and 37–39 (20 of the 38 compressed layers) an earlier
source's selection in prefill, and some or all of them in decode. On a single
pipeline stage every group count from 2 to 19 does this; only 1 group, or 20 and
more, avoid it in every layout.

The selections differ only once a layer has more than 512 compressed positions:
beyond 512 tokens on the ratio-1 layers (20–39) and beyond about 1K tokens on the
ratio-2 layers (2–19). Below that every index source selects every position, so
shorter contexts take the same path with or without the memo.

### The fix

`e2b585f` removes the memo. Every compressed layer derives its indices from the
shared buffer when it runs, as base vLLM does, so it always reads the selection of
its most recent index source. `tests/test_topk_source_reuse.py` checks this on CPU
for KV-cache group counts between 1 and 40, and fails on every earlier commit.

### Before/after A/B

`DeepSeek-V4.1-Flash-1017` (official weights) on one node with 8× GA100 64GB
(`SM80`), `TP4 × PP2`, `DSpark K=5`. Before: our deployed build of these patches,
with the memo. After: the same build with the memo switched off, which runs the
same kernels on the same inputs as `e2b585f`. The copy probes ask the model to
copy rare items out of long contexts, either listed just before the question
(near) or scattered through the context and recalled by index (far), and compare
the answers with a reference deployment of the same model.

| | Before (memo) | After (no memo) |
| :--- | :---: | :---: |
| Copy probe, cold, 2K–128K-token contexts, near and far | 870/870 exact¹ | 900/900 exact |
| Copy probe, warm repeat (prefix-cache hits) | 570/570 exact | 570/570 exact |
| Multi-turn copy, thinking off / on | 600/600, 360/360 | 600/600, 360/360 |
| Code decode, 1 / 4 streams (tok/s) | 159.8 / 345.2 | 159.7 / 352.1 |
| Prose decode, 1 / 4 streams (tok/s) | 87.2 / 192.1 | 80.6 / 190.5 |
| Cold prefill, 18K–40K-token prompts (tok/s) | 3,399–3,544 | 3,517–3,630 |

¹ One trial misread the task, as one trial did on the reference; its items are not
counted.

Both runs copied at the reference's level, and the teacher-forced probabilities of
the copied tokens showed the same pattern in both. The speed differences go both
ways: prose decode at one stream was 8% slower without the memo, prefill was 2–4%
faster, and the other rows were within 2%. These are single quick runs, so we read
them as no measurable change. We could not tell the two builds apart, but the memo
is still wrong against the reference, so please update.

### Does your install have the memo?

Look for it in the `rocm.py` your vLLM imports:

```bash
ROCM_PY="$(python -c 'import os, vllm; print(os.path.dirname(vllm.__file__))')/models/deepseek_v4_1/amd/rocm.py"
grep -n "_cached_decode_topk\|_cached_prefill_ragged" "$ROCM_PY"   # any output: the old code
```

Or run the regression test against it from this repository's root (needs torch,
no GPU):

```bash
DEEPINFER_ROCM_PY="$ROCM_PY" python -m unittest discover -s tests -v
```

### Updating

1. Re-apply both patches from `e2b585f` or later to a clean `79a7108d9a` tree, or copy
   `vllm_sm80_deepinfer/vllm/models/deepseek_v4_1/amd/rocm.py` over your tree.
2. Confirm with `python scripts/check_dropin_sync.py --vllm-dir /path/to/vllm`.
3. Restart the server.

### Performance

The A/B above showed no measurable change in speed, so the end-to-end numbers in
[BENCHMARKS.md](BENCHMARKS.md), which were measured before the fix, stand.²

² Without the memo every compressed layer derives its own indices, a handful of
small kernels per layer per decode step and per prefill chunk. We had estimated
this at about 0.2 ms per decode step with CUDA graphs.
