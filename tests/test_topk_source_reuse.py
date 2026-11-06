"""GPU-free regression test: each layer attends over its own index source's top-k.

DeepSeek-V4.1 computes a top-k selection only on the index-source layers. Every
other compressed layer attends over the selection of the most recent index source
below it (DeepSeek's reference reuses ``shared_attn.topk_idxs``; vLLM keeps it in
the shared ``topk_indices_buffer``). vLLM builds one attention-metadata object per
KV-cache group and deals layers into groups round-robin, so layers that follow
different index sources can share one metadata object. Before e2b585f the drop-in
``amd/rocm.py`` memoised the derived indices on that shared object, so such layers
could attend over an earlier source's selection on long contexts.

The test runs the real ``forward_mqa``, ``_forward_prefill`` and ``_forward_decode``
of the drop-in module on CPU. The vLLM and Triton imports are stubbed, and the
Triton launchers and attention kernels are replaced by torch stand-ins that record
which selection each layer would attend over. Every compressed layer must get the
selection of its own most recent index source, whatever the number of KV-cache
groups.

    python -m unittest discover -s tests -v

Set ``DEEPINFER_ROCM_PY`` to test another copy of ``rocm.py``, for example the
``vllm/models/deepseek_v4_1/amd/rocm.py`` of an installed vLLM.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
ROCM_PY = Path(
    os.environ.get(
        "DEEPINFER_ROCM_PY",
        REPO_ROOT / "vllm_sm80_deepinfer/vllm/models/deepseek_v4_1/amd/rocm.py",
    )
)

# DeepSeek-V4.1-Flash: 40 backbone layers; compress ratio 0 (sliding window only)
# on layers 0-1, 2 on layers 2-19 and 1 on layers 20-39.
NUM_LAYERS = 40
COMPRESS_RATIOS = [0, 0] + [2] * 18 + [1] * 20
KV_SOURCE_LAYERS = (2, 8, 14, 20)
INDEX_SOURCE_LAYERS = (2, 8, 14, 20, 24, 28, 32, 36)

# The old memo was safe with 1 group and with 20 or more; 2-19 groups were not.
GROUP_COUNTS = (1, 2, 3, 4, 5, 8, 15, NUM_LAYERS)

# A small batch: 2 decode requests with 2 tokens each (as with speculative
# decoding), then 2 prefill requests with 3 and 5 new tokens.
DECODE_QUERY_LENS = (2, 2)
PREFILL_QUERY_LENS = (3, 5)
PREFILL_SEQ_LENS = (40, 50)
MAX_TOKENS = sum(DECODE_QUERY_LENS) + sum(PREFILL_QUERY_LENS)
TOP_K = 16  # selection width; 512 in the real model
NUM_COMPRESSED_POSITIONS = 64
NUM_HEADS = 2
HEAD_DIM = 8


def _most_recent(sources: tuple[int, ...], layer_id: int) -> int:
    return max(s for s in sources if s <= layer_id)


def _not_called(*args, **kwargs):
    raise AssertionError("this stub must not be called by the test")


class _AttentionBase:
    """Stands in for ``vllm.models.deepseek_v4_1.attention.DeepseekV4Attention``."""

    PREFILL_CHUNK_SIZE = 4  # same as the real class

    def _compressed_kv_cache(self) -> torch.Tensor:
        return self.test_compressed_kv_cache


def _stub_modules() -> dict[str, types.ModuleType]:
    """Just enough of vLLM and Triton for ``rocm.py`` to import on CPU."""

    def stub_classes(*names: str) -> dict[str, type]:
        return {name: type(name, (), {}) for name in names}

    members = {
        "vllm.distributed": {
            "get_tensor_model_parallel_world_size": lambda: 1,
            "tensor_model_parallel_all_reduce": _not_called,
        },
        "vllm.forward_context": {"get_forward_context": _not_called},
        "vllm.logger": {"init_logger": logging.getLogger},
        "vllm.models.deepseek_v4_1.attention": {"DeepseekV4Attention": _AttentionBase},
        "vllm.models.deepseek_v4_1.common.ops": {
            "dequantize_and_gather_k_cache": _not_called
        },
        "vllm.models.deepseek_v4_1.sparse_mla": stub_classes(
            "DeepseekV4FlashMLAMetadata",
            "DeepseekV4SparseMLABackend",
            "DeepseekV4SparseMLAMetadataBuilder",
            "DeepseekV41SparseSWAMetadataBuilder",
        ),
        "vllm.platforms": {"current_platform": None},
        "vllm.platforms.rocm": {"_ON_GFX950": False},
        "vllm.triton_utils": {
            "tl": types.SimpleNamespace(constexpr=int),
            "triton": types.SimpleNamespace(
                jit=lambda fn: fn, cdiv=_not_called, next_power_of_2=_not_called
            ),
        },
        "vllm.v1.attention.backend": stub_classes("CommonAttentionMetadata"),
        "vllm.v1.attention.backends.mla.sparse_swa": stub_classes(
            "DeepseekSparseSWABackend", "DeepseekSparseSWAMetadata"
        ),
        "vllm.v1.attention.ops.rocm_aiter_mla_sparse": dict.fromkeys(
            (
                "build_ragged_indices_from_dense",
                "rocm_inv_rope_einsum",
                "rocm_sparse_attn_decode",
                "rocm_sparse_attn_prefill",
            ),
            _not_called,
        ),
        "vllm.v1.worker.workspace": {"current_workspace_manager": _not_called},
    }
    modules: dict[str, types.ModuleType] = {}
    for name, attrs in members.items():
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            parent = ".".join(parts[:i])
            modules.setdefault(parent, types.ModuleType(parent))
        modules[name].__dict__.update(attrs)
    return modules


def _load_rocm() -> types.ModuleType:
    name = "deepinfer_sm80_rocm_under_test"
    spec = importlib.util.spec_from_file_location(name, ROCM_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # No __pycache__ inside vllm_sm80_deepinfer/, which users copy over vLLM.
    with mock.patch.object(sys, "dont_write_bytecode", True):
        with mock.patch.dict(sys.modules, {**_stub_modules(), name: module}):
            spec.loader.exec_module(module)
    return module


# Torch stand-ins for the Triton launchers and kernels. They only need to be pure
# functions of their inputs. Like the real kernels they write fresh tensors, so a
# later write to the shared buffer cannot change what an earlier layer recorded.


def _decode_topk_indices(
    topk_indices, token_to_req_indices, block_table, block_size, is_valid_token
):
    selection = topk_indices.reshape(topk_indices.shape[0], -1).clone()
    lens = (selection >= 0).sum(dim=1).to(torch.int32)
    indptr = torch.zeros(selection.shape[0] + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(lens, dim=0)
    return selection, indptr, lens


def _combine_topk_swa_indices(
    topk_indices,
    query_start_loc,
    seq_lens,
    gather_lens,
    window_size,
    compress_ratio,
    topk,
    M,
    N,
):
    selection = topk_indices.reshape(topk_indices.shape[0], -1)[:, :topk].clone()
    lens = torch.full((selection.shape[0],), selection.shape[1], dtype=torch.int32)
    return selection, lens


def _build_ragged_indices_from_dense(indices, lengths, num_rows=-1):
    indptr = torch.zeros(indices.shape[0] + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(lengths, dim=0)
    return indices.reshape(-1).clone(), indptr


class _Workspace:
    def get_simultaneous(self, *specs):
        # Only the shape's ends matter: [chunk, 1, head_dim] stands in for the
        # [chunk, M, head_dim] gather workspace.
        return [
            torch.zeros(shape[0], 1, shape[2], dtype=dtype) for shape, dtype in specs
        ]


class _Recorder:
    """What each layer hands to the sparse attention kernels."""

    def __init__(self) -> None:
        self.layer_id: int | None = None
        self.decode: dict[int, torch.Tensor | None] = {}
        self.prefill: dict[int, list[torch.Tensor]] = {}

    def sparse_attn_decode(self, **kwargs) -> None:
        self.decode[self.layer_id] = kwargs["topk_ragged_indices"]

    def sparse_attn_prefill(self, **kwargs) -> None:
        self.prefill.setdefault(self.layer_id, []).append(kwargs["ragged_indices"])


def _selections() -> dict[int, torch.Tensor]:
    """The top-k each index source writes into the shared buffer, one row per token."""
    out = {}
    for source in INDEX_SOURCE_LAYERS:
        generator = torch.Generator().manual_seed(1000 + source)
        out[source] = torch.randint(
            0,
            NUM_COMPRESSED_POSITIONS,
            (MAX_TOKENS, TOP_K),
            generator=generator,
            dtype=torch.int32,
        )
    return out


def _swa_metadata(decode_lens, prefill_lens, group: int) -> types.SimpleNamespace:
    """Batch layout as one KV-cache group's SWA metadata sees it."""
    query_lens = list(decode_lens) + list(prefill_lens)
    num_reqs = len(query_lens)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
    query_start_loc[1:] = torch.cumsum(torch.tensor(query_lens), dim=0)
    seq_lens = torch.tensor(PREFILL_SEQ_LENS[: len(prefill_lens)], dtype=torch.int32)
    return types.SimpleNamespace(
        num_decodes=len(decode_lens),
        num_decode_tokens=sum(decode_lens),
        num_prefills=len(prefill_lens),
        num_prefill_tokens=sum(prefill_lens),
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.clone(),
        token_to_req_indices=torch.repeat_interleave(
            torch.arange(len(decode_lens), dtype=torch.int32),
            torch.tensor(decode_lens, dtype=torch.long),
        ),
        is_valid_token=torch.ones(sum(decode_lens), dtype=torch.bool),
        prefill_seq_lens=seq_lens,
        prefill_gather_lens=seq_lens.clone(),
        # each KV-cache group has its own block table
        block_table=torch.arange(2 * num_reqs, dtype=torch.int32).view(num_reqs, 2)
        + 1000 * group,
        block_size=32,
        decode_swa_indices=None,
        decode_swa_lens=None,
        decode_swa_ragged_indices=None,
        decode_swa_ragged_indptr=None,
    )


def _make_layer(rocm, layer_id: int, buffer: torch.Tensor):
    layer = object.__new__(rocm.DeepseekV41ROCMAiterMLAAttention)
    ratio = COMPRESS_RATIOS[layer_id]
    layer.compress_ratio = ratio
    # Only index sources own an indexer; it writes the shared buffer (see below).
    layer.indexer = object() if layer_id in INDEX_SOURCE_LAYERS else None
    layer.topk_indices_buffer = buffer
    layer.swa_cache_layer = types.SimpleNamespace(
        prefix=f"layers.{layer_id}.swa_cache", kv_cache=torch.zeros(1)
    )
    layer.compressed_cache_prefix = (
        f"layers.{_most_recent(KV_SOURCE_LAYERS, layer_id)}" if ratio else None
    )
    layer.test_compressed_kv_cache = torch.zeros(1)
    layer.attn_sink = torch.zeros(NUM_HEADS)
    layer.scale = HEAD_DIM**-0.5
    layer.head_dim = HEAD_DIM
    layer.nope_head_dim = HEAD_DIM - 2
    layer.rope_head_dim = 2
    layer.kv_cache_dtype = "fp8_ds_mla"
    layer._has_kv_transfer = False
    layer.window_size = 4
    layer.max_model_len = 128
    layer.max_num_batched_tokens = MAX_TOKENS
    return layer


def _run_forward(rocm, num_groups: int, decode_lens, prefill_lens) -> _Recorder:
    """One forward step over all layers; layer i uses SWA metadata group i % n."""
    num_tokens = sum(decode_lens) + sum(prefill_lens)
    num_reqs = len(decode_lens) + len(prefill_lens)
    selections = _selections()
    buffer = torch.full((MAX_TOKENS, TOP_K), -1, dtype=torch.int32)

    groups = [_swa_metadata(decode_lens, prefill_lens, g) for g in range(num_groups)]
    attn_metadata = {
        f"layers.{layer_id}.swa_cache": groups[layer_id % num_groups]
        for layer_id in range(NUM_LAYERS)
    }
    for source in KV_SOURCE_LAYERS:  # compressed-KV metadata of each kv source
        attn_metadata[f"layers.{source}"] = types.SimpleNamespace(
            block_table=torch.arange(2 * num_reqs, dtype=torch.int32).view(num_reqs, 2)
            + 100 * source,
            block_size=64,
        )
    context = types.SimpleNamespace(attn_metadata=attn_metadata)

    recorder = _Recorder()
    with mock.patch.multiple(
        rocm,
        get_forward_context=lambda: context,
        compute_global_topk_ragged_indices_and_indptr=_decode_topk_indices,
        combine_topk_swa_indices=_combine_topk_swa_indices,
        build_ragged_indices_from_dense=_build_ragged_indices_from_dense,
        rocm_sparse_attn_decode=recorder.sparse_attn_decode,
        rocm_sparse_attn_prefill=recorder.sparse_attn_prefill,
        dequantize_and_gather_k_cache=lambda *args, **kwargs: None,
        current_workspace_manager=_Workspace,
        current_platform=types.SimpleNamespace(is_fp8_fnuz=lambda: False),
    ):
        for layer_id in range(NUM_LAYERS):
            layer = _make_layer(rocm, layer_id, buffer)
            if layer.indexer is not None:
                buffer.copy_(selections[layer_id])  # the indexer's write
            recorder.layer_id = layer_id
            q = torch.zeros(num_tokens, NUM_HEADS, HEAD_DIM)
            positions = torch.zeros(num_tokens, dtype=torch.long)
            layer.forward_mqa(q, None, positions, torch.zeros_like(q))
    return recorder


def _name_source(got: torch.Tensor, expected: dict[int, torch.Tensor]) -> str:
    for source, rows in expected.items():
        if torch.equal(got, rows):
            return f"the selection of index source {source}"
    return "a selection no index source wrote"


class TopkSourceReuseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rocm = _load_rocm()

    def _check(self, decode_lens: tuple[int, ...], prefill_lens: tuple[int, ...]):
        num_decode_tokens = sum(decode_lens)
        num_tokens = num_decode_tokens + sum(prefill_lens)
        selections = _selections()
        decode_expected = {
            s: rows[:num_decode_tokens] for s, rows in selections.items()
        }
        prefill_expected = {
            s: rows[num_decode_tokens:num_tokens].reshape(-1)
            for s, rows in selections.items()
        }
        for num_groups in GROUP_COUNTS:
            with self.subTest(kv_cache_groups=num_groups):
                recorder = _run_forward(
                    self.rocm, num_groups, decode_lens, prefill_lens
                )
                wrong = []
                for layer_id in range(NUM_LAYERS):
                    if COMPRESS_RATIOS[layer_id] == 0:
                        continue  # sliding-window-only layers read no top-k
                    source = _most_recent(INDEX_SOURCE_LAYERS, layer_id)
                    if decode_lens:
                        got = recorder.decode[layer_id]
                        if not torch.equal(got, decode_expected[source]):
                            wrong.append(
                                f"decode: layer {layer_id} attended over "
                                f"{_name_source(got, decode_expected)}, not {source}"
                            )
                    if prefill_lens:
                        (got,) = recorder.prefill[layer_id]  # one prefill chunk
                        if not torch.equal(got, prefill_expected[source]):
                            wrong.append(
                                f"prefill: layer {layer_id} attended over "
                                f"{_name_source(got, prefill_expected)}, not {source}"
                            )
                if wrong:
                    self.fail("\n" + "\n".join(wrong))

    def test_mixed_batch(self) -> None:
        self._check(DECODE_QUERY_LENS, PREFILL_QUERY_LENS)

    def test_decode_only_batch(self) -> None:
        self._check(DECODE_QUERY_LENS, ())

    def test_prefill_only_batch(self) -> None:
        self._check((), PREFILL_QUERY_LENS)


if __name__ == "__main__":
    unittest.main()
