# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline parallelism across DeepSeek V4.1's shared caches.

V4.1 shares state across layers: every layer with ``compress_ratio > 0`` reads
the compressed KV cache of the nearest ``kv_source_layer_ids`` entry below
it, non-owning index sources read that source's indexer K cache, the
``candidate_source_layer_id`` indexer publishes candidate blocks for every
later indexer, and each index source publishes top-k indices for the layers
up to the next one. Upstream only supports this when a whole sharing group
sits on one PP rank -- impossible here, because layer 20's group is the
last 20 layers of the model.

This module makes any partition legal by *replicating* the shared state on
the ranks that need it:

* A downstream rank that hosts consumers of source ``S`` (but not ``S``)
  registers a :class:`CompressedKVReplica` under ``S``'s exact layer name
  (and a :class:`DeepseekV4IndexerCache` under ``S``'s indexer K cache name).
  They declare the same KV cache spec as the source, so the KV cache
  manager gives them the same block tables and slot mappings, and the
  consumer layers resolve their sources through the forward context exactly
  as they do on the source rank.
* Every step, the rows the source wrote for the current tokens (584 bytes
  per compressed state, 132 bytes per indexer key) are gathered by slot and
  shipped downstream inside the pipeline's ``IntermediateTensors``, along
  with the candidate-block and top-k rows of the batch when a boundary
  splits their consumers. Each receiving rank scatters the rows into its
  replicas before running its layers and re-gathers them for the next hop,
  so the data crosses as many boundaries as the sharing group does.

Per 4096-token prefill chunk this is ~2.4 MB of compressed rows, ~0.5 MB
of index keys, 32 MB of candidate blocks and 8 MB of top-k indices per hop.
"""

from dataclasses import dataclass, field

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.distributed.utils import get_pp_indices
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.models.deepseek_v4_1 import attention as _attn
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheSpec

logger = init_logger(__name__)

# fp8_ds_mla compressed state: 576 bytes of values (448 e4m3 NoPE + 128 bf16
# RoPE) at ``page + slot_in_block * 576`` and 8 UE8M0 scale bytes at
# ``page + block_size * 576 + slot_in_block * 8``.
# tl.constexpr so the @triton.jit kernels below may read them as globals.
KV_VALUE_BYTES = tl.constexpr(576)
KV_SCALE_BYTES = tl.constexpr(8)
KV_ROW_BYTES = tl.constexpr(576 + 8)
# fp8 indexer key: 128 e4m3 values then one fp32 scale, segregated per block.
IK_VALUE_BYTES = tl.constexpr(128)
IK_SCALE_BYTES = tl.constexpr(4)
IK_ROW_BYTES = tl.constexpr(128 + 4)


@triton.jit
def _kv_rows_kernel(
    cache,
    slots,
    positions,
    rows,
    cache_stride,
    CACHE_BLOCK: tl.constexpr,
    RATIO: tl.constexpr,
    GATHER: tl.constexpr,
):
    """Copy this token's compressed state between the cache and ``rows[t]``.

    Mirrors ``_rope_quant_insert_kernel``'s write rule: only a slot >= 0 at a
    group-boundary position holds a state for this step.
    """
    t = tl.program_id(0)
    slot = tl.load(slots + t)
    if slot < 0:
        return
    position = tl.load(positions + t)
    if (position + 1) % RATIO != 0:
        return
    page = cache + (slot // CACHE_BLOCK).to(tl.int64) * cache_stride
    values = page + (slot % CACHE_BLOCK) * KV_VALUE_BYTES
    scales = page + CACHE_BLOCK * KV_VALUE_BYTES + (slot % CACHE_BLOCK) * KV_SCALE_BYTES
    row = rows + t.to(tl.int64) * KV_ROW_BYTES
    d = tl.arange(0, 1024)
    dv = d < KV_VALUE_BYTES
    s = tl.arange(0, 8)
    if GATHER:
        tl.store(row + d, tl.load(values + d, dv, other=0), dv)
        tl.store(row + KV_VALUE_BYTES + s, tl.load(scales + s))
    else:
        tl.store(values + d, tl.load(row + d, dv, other=0), dv)
        tl.store(scales + s, tl.load(row + KV_VALUE_BYTES + s))


@triton.jit
def _ik_rows_kernel(
    cache,
    slots,
    positions,
    rows,
    cache_stride,
    CACHE_BLOCK: tl.constexpr,
    RATIO: tl.constexpr,
    GATHER: tl.constexpr,
):
    """Same as ``_kv_rows_kernel`` for the fp8 indexer K cache
    (``_indexer_k_norm_rope_quant_store_kernel``'s non-tiled layout)."""
    t = tl.program_id(0)
    slot = tl.load(slots + t)
    if slot < 0:
        return
    position = tl.load(positions + t)
    if (position + 1) % RATIO != 0:
        return
    page = cache + (slot // CACHE_BLOCK).to(tl.int64) * cache_stride
    values = page + (slot % CACHE_BLOCK) * IK_VALUE_BYTES
    scales = page + CACHE_BLOCK * IK_VALUE_BYTES + (slot % CACHE_BLOCK) * IK_SCALE_BYTES
    row = rows + t.to(tl.int64) * IK_ROW_BYTES
    d = tl.arange(0, IK_VALUE_BYTES)
    s = tl.arange(0, IK_SCALE_BYTES)
    if GATHER:
        tl.store(row + d, tl.load(values + d))
        tl.store(row + IK_VALUE_BYTES + s, tl.load(scales + s))
    else:
        tl.store(values + d, tl.load(row + d))
        tl.store(scales + s, tl.load(row + IK_VALUE_BYTES + s))


def _copy_rows(kernel, cache: torch.Tensor, slots: torch.Tensor,
               positions: torch.Tensor, rows: torch.Tensor, ratio: int,
               gather: bool) -> None:
    num_tokens = positions.shape[0]
    if num_tokens == 0 or cache.numel() == 0:
        return
    # Bound caches are [num_blocks, block_size, bytes]; address them as bytes.
    assert cache.dtype == torch.uint8, cache.dtype
    kernel[(num_tokens,)](
        cache,
        slots,
        positions,
        rows,
        cache.stride(0),
        CACHE_BLOCK=cache.shape[1],
        RATIO=ratio,
        GATHER=gather,
        num_warps=1,
    )


class CompressedKVReplica(nn.Module, AttentionLayerBase):
    """A kv-source attention layer's compressed KV cache on a rank without
    the layer. Registered under the source's own layer name."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        compress_ratio: int,
        head_dim: int,
        backend_cls: type[AttentionBackend],
        use_fp8_ds_mla_layout: bool,
    ) -> None:
        super().__init__()
        cache_config = vllm_config.cache_config
        self.prefix = prefix
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.backend_cls = backend_cls
        self.kv_cache_dtype, self.kv_cache_torch_dtype = (
            _attn._resolve_dsv4_kv_cache_dtype(
                use_fp8_ds_mla_layout, cache_config.cache_dtype, cache_config
            )
        )
        if self.kv_cache_dtype != "fp8_ds_mla":
            raise NotImplementedError(
                "PP replication of DeepSeek V4.1 compressed caches is implemented "
                f"for the fp8_ds_mla layout only (got {self.kv_cache_dtype})."
            )
        self.kv_cache = torch.tensor([])
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        self.kv_cache = kv_cache.squeeze(1)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        # Byte-identical to DeepseekV4Attention.get_kv_cache_spec for a kv
        # source: the KV cache manager asserts equality across ranks.
        return _attn.MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8,
            tokens_per_state=self.compress_ratio,
            cache_dtype_str=self.kv_cache_dtype,
            alignment=576,
            model_version="deepseek_v4",
            kv_quant_mode=_attn.get_kv_quant_mode(self.kv_cache_dtype),
            state_content_bytes=584,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.backend_cls

    def forward(self): ...


@dataclass
class _Boundary:
    """What crosses the hand-off after a rank."""

    kv_sources: list[int] = field(default_factory=list)  # ship compressed rows
    ik_sources: list[int] = field(default_factory=list)  # ship indexer K rows
    topk: bool = False
    cand: bool = False

    def empty(self) -> bool:
        return not (self.kv_sources or self.ik_sources or self.topk or self.cand)


class PPSharePlan:
    """Per-rank plan: which replicas to host, what to receive, what to send."""

    def __init__(self, vllm_config: VllmConfig) -> None:
        config = vllm_config.model_config.hf_config
        pp = get_pp_group()
        self.pp_rank, self.pp_size = pp.rank_in_group, pp.world_size
        num_layers = config.num_hidden_layers
        self.ranges = [get_pp_indices(num_layers, r, self.pp_size) for r in range(self.pp_size)]
        self.start, self.end = self.ranges[self.pp_rank]
        ratios = list(getattr(config, "compress_ratios", []) or [])
        self.ratio = {i: int(ratios[i]) if i < len(ratios) else 0 for i in range(num_layers)}
        kv_sources = sorted(getattr(config, "kv_source_layer_ids", None) or ())
        index_sources = sorted(getattr(config, "index_source_layer_ids", None) or ())
        cand = getattr(config, "candidate_source_layer_id", -1)
        self.candidate_topk_blocks = getattr(config, "candidate_topk_blocks", 0)
        self.index_topk = config.index_topk
        self.index_head_dim = config.index_head_dim
        self.head_dim = config.head_dim

        def group_end(sources, s):
            later = [x for x in sources if x > s]
            return later[0] if later else num_layers

        # Consumers of kv source S: compressed layers in (S, next source).
        self.kv_consumers = {
            s: [i for i in range(s + 1, group_end(kv_sources, s)) if self.ratio[i] > 0]
            for s in kv_sources
        }
        # Non-owning index sources read S's indexer K cache.
        self.ik_consumers = {
            s: [i for i in index_sources if s < i < group_end(kv_sources, s)]
            for s in kv_sources
        }
        # Layers using index source I's top-k: (I, next index source).
        self.topk_consumers = {
            i: [j for j in range(i + 1, group_end(index_sources, i)) if self.ratio[j] > 0]
            for i in index_sources
        }
        self.cand_source = cand
        self.cand_consumers = [i for i in index_sources if i > cand] if cand >= 0 else []

        self.boundaries: list[_Boundary] = []
        for r in range(self.pp_size - 1):
            last = self.ranges[r][1] - 1
            b = _Boundary()
            for s in kv_sources:
                if s <= last and any(c > last for c in self.kv_consumers[s]):
                    b.kv_sources.append(s)
                if s <= last and any(c > last for c in self.ik_consumers[s]):
                    b.ik_sources.append(s)
            b.topk = any(
                i <= last and any(c > last for c in self.topk_consumers[i])
                for i in index_sources
            )
            b.cand = (
                cand >= 0 and cand <= last and any(c > last for c in self.cand_consumers)
            )
            self.boundaries.append(b)
        self.recv_plan = self.boundaries[self.pp_rank - 1] if self.pp_rank > 0 else _Boundary()
        self.send_plan = (
            self.boundaries[self.pp_rank] if self.pp_rank < self.pp_size - 1 else _Boundary()
        )
        local = range(self.start, self.end)
        self.replica_kv = [
            s for s in kv_sources
            if s not in local and any(c in local for c in self.kv_consumers[s])
        ]
        self.replica_ik = [
            s for s in kv_sources
            if s not in local and any(c in local for c in self.ik_consumers[s])
        ]
        self.replicas: dict[int, CompressedKVReplica] = {}
        self.ik_replicas: dict[int, _attn.DeepseekV4IndexerCache] = {}
        self._layers_prefix = ""
        self._buffers: dict[str, torch.Tensor] = {}
        logger.info(
            "DSv4.1 PP share plan rank %d layers [%d, %d): replicas kv=%s ik=%s; "
            "recv kv=%s ik=%s topk=%s cand=%s; send kv=%s ik=%s topk=%s cand=%s",
            self.pp_rank, self.start, self.end, self.replica_kv, self.replica_ik,
            self.recv_plan.kv_sources, self.recv_plan.ik_sources, self.recv_plan.topk, self.recv_plan.cand,
            self.send_plan.kv_sources, self.send_plan.ik_sources, self.send_plan.topk, self.send_plan.cand,
        )

    @property
    def active(self) -> bool:
        return bool(self.replica_kv or self.replica_ik or not self.recv_plan.empty()
                    or not self.send_plan.empty())

    # ---- construction --------------------------------------------------
    def create_replicas(
        self,
        vllm_config: VllmConfig,
        layers_prefix: str,
        attn_cls: type,
    ) -> None:
        """Register the stand-in caches BEFORE the consumer layers are built."""
        self._layers_prefix = layers_prefix
        for s in self.replica_kv:
            self.replicas[s] = CompressedKVReplica(
                vllm_config,
                prefix=f"{layers_prefix}.{s}.attn",
                compress_ratio=self.ratio[s],
                head_dim=self.head_dim,
                backend_cls=attn_cls.backend_cls,
                use_fp8_ds_mla_layout=attn_cls.use_fp8_ds_mla_layout,
            )
        for s in self.replica_ik:
            use_fp4 = _attn.dsa_indexer_uses_fp4(vllm_config)
            if use_fp4:
                raise NotImplementedError(
                    "PP replication of the indexer K cache supports the fp8 "
                    "layout only."
                )
            self.ik_replicas[s] = _attn.DeepseekV4IndexerCache(
                head_dim=_attn._indexer_k_cache_head_dim(self.index_head_dim, False),
                dtype=torch.uint8,
                prefix=f"{layers_prefix}.{s}.attn.indexer.k_cache",
                cache_config=vllm_config.cache_config,
                compress_ratio=self.ratio[s],
            )

    def alloc_buffers(self, max_tokens: int, device: torch.device) -> None:
        """Persistent send-side staging so captured graphs see stable addresses."""
        for s in self.send_plan.kv_sources:
            self._buffers[f"pp_kv{s}"] = torch.zeros(
                (max_tokens, KV_ROW_BYTES.value), dtype=torch.uint8, device=device
            )
        for s in self.send_plan.ik_sources:
            self._buffers[f"pp_ik{s}"] = torch.zeros(
                (max_tokens, IK_ROW_BYTES.value), dtype=torch.uint8, device=device
            )

    def recv_tensors(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        """Entries for ``make_empty_intermediate_tensors`` (the incoming layout)."""
        out: dict[str, torch.Tensor] = {}
        for s in self.recv_plan.kv_sources:
            out[f"pp_kv{s}"] = torch.zeros((batch_size, KV_ROW_BYTES.value), dtype=torch.uint8, device=device)
        for s in self.recv_plan.ik_sources:
            out[f"pp_ik{s}"] = torch.zeros((batch_size, IK_ROW_BYTES.value), dtype=torch.uint8, device=device)
        if self.recv_plan.cand:
            out["pp_cand"] = torch.zeros((batch_size, self.candidate_topk_blocks), dtype=torch.int32, device=device)
        if self.recv_plan.topk:
            out["pp_topk"] = torch.zeros((batch_size, self.index_topk), dtype=torch.int32, device=device)
        return out

    # ---- per step --------------------------------------------------------
    def _static(self, name: str):
        # The forward context is keyed the same way as static_forward_context;
        # the replicas / source layers registered themselves there at init.
        return self._static_ctx[name]

    def bind_static_context(self, static_forward_context: dict) -> None:
        self._static_ctx = static_forward_context

    def receive(
        self,
        tensors: dict[str, torch.Tensor],
        positions: torch.Tensor,
        topk_indices_buffer: torch.Tensor | None,
        candidate_block_buffer: torch.Tensor | None,
    ) -> None:
        """Scatter the shipped rows into this rank's replicas/buffers."""
        num_tokens = positions.shape[0]
        attn_metadata = get_forward_context().attn_metadata
        if self.recv_plan.cand and candidate_block_buffer is not None:
            candidate_block_buffer[:num_tokens].copy_(tensors["pp_cand"][:num_tokens])
        if self.recv_plan.topk and topk_indices_buffer is not None:
            topk_indices_buffer[:num_tokens].copy_(tensors["pp_topk"][:num_tokens])
        if not isinstance(attn_metadata, dict):
            return  # profile / dummy run: caches unbound, nothing to place
        for s in self.recv_plan.kv_sources:
            name = f"{self._layers_prefix}.{s}.attn"
            md = attn_metadata.get(name)
            if md is None:
                continue
            _copy_rows(_kv_rows_kernel, self._static(name).kv_cache, md.slot_mapping,
                       positions, tensors[f"pp_kv{s}"], self.ratio[s], gather=False)
        for s in self.recv_plan.ik_sources:
            name = f"{self._layers_prefix}.{s}.attn.indexer.k_cache"
            md = attn_metadata.get(name)
            if md is None:
                continue
            _copy_rows(_ik_rows_kernel, self._static(name).kv_cache, md.slot_mapping,
                       positions, tensors[f"pp_ik{s}"], self.ratio[s], gather=False)

    def send(
        self,
        positions: torch.Tensor,
        topk_indices_buffer: torch.Tensor | None,
        candidate_block_buffer: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Gather this step's rows from local caches (source or replica)."""
        num_tokens = positions.shape[0]
        out: dict[str, torch.Tensor] = {}
        attn_metadata = get_forward_context().attn_metadata
        have_md = isinstance(attn_metadata, dict)
        for s in self.send_plan.kv_sources:
            name = f"{self._layers_prefix}.{s}.attn"
            rows = self._buffers[f"pp_kv{s}"][:num_tokens]
            md = attn_metadata.get(name) if have_md else None
            if md is not None:
                _copy_rows(_kv_rows_kernel, self._static(name).kv_cache, md.slot_mapping,
                           positions, rows, self.ratio[s], gather=True)
            out[f"pp_kv{s}"] = rows
        for s in self.send_plan.ik_sources:
            name = f"{self._layers_prefix}.{s}.attn.indexer.k_cache"
            rows = self._buffers[f"pp_ik{s}"][:num_tokens]
            md = attn_metadata.get(name) if have_md else None
            if md is not None:
                _copy_rows(_ik_rows_kernel, self._static(name).kv_cache, md.slot_mapping,
                           positions, rows, self.ratio[s], gather=True)
            out[f"pp_ik{s}"] = rows
        if self.send_plan.cand:
            assert candidate_block_buffer is not None
            out["pp_cand"] = candidate_block_buffer[:num_tokens]
        if self.send_plan.topk:
            assert topk_indices_buffer is not None
            out["pp_topk"] = topk_indices_buffer[:num_tokens]
        return out
