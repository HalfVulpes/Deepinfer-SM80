# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram: n-gram hash lookups gated into the hyper-connection stream.

Port of the reference ``inference/engram.py`` + ``Engram`` /
``ParallelEngramEmbedding`` from ``inference/model.py`` (DeepSeek V4.1
checkpoint layout). Engram modules live on the backbone layers listed in
``engram_layer_ids`` only.

Two pieces of cross-forward state are needed because vLLM streams tokens
chunk-by-chunk while an n-gram at position ``p`` needs the token ids at
``p-1..p-3``:

- ``token_map``: token id -> compressed vocab id, built once from the
  model's tokenizer at init (deterministic; asserted against
  ``engram_compressed_vocab_size``).
- ``hash_cache``: one int32 slot per KV slot of the first local layer's
  sliding-window cache, holding the compressed id (or DEAD) of the token
  last written to that slot. Slots are stable per (request, position) —
  the block table pins a position to a physical slot, prefix-cache hits
  reuse both the physical blocks and the identical token ids, and
  spec-decode rollbacks rewrite the same slots — so lookbacks read back
  exactly what the owning request wrote. Lookback depth (3) is far inside
  the sliding window (128), so window eviction never frees a block a
  live lookback still needs.

  Slots are not part of the KV cache, so KV loaded from another instance
  (P/D, offload connectors) leaves them unwritten. The runner therefore
  passes ``lookback_token_ids``, the ids just before each request's chunk
  start, which take precedence over the slots. The V2 runner reads them
  from its device-resident token history and needs no slot cache; the V1
  runner's CPU token table holds placeholders for generated tokens under
  async scheduling, so it passes prompt positions only and keeps the slot
  cache for the rest.
"""

import concurrent.futures
import functools
import json
import os
import re
import struct
import time
import weakref

import numpy as np
import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
from vllm.v1.attention.ops.fp8_sm80 import _decode_fp8_f32

logger = init_logger(__name__)

# Cache value for tokens that take no part in an n-gram (image spans).
DEAD_ID = -1


def _is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for n < 2**32 (avoids a sympy import)."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    """The smallest prime above `start` that has not been handed out yet."""
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Map every token id onto a smaller id space where tokens that normalize
    alike collapse together.

    N-grams are hashed over these compressed ids, so " The", "the" and "THE"
    all hash the same way. The compressed size matters beyond bounds checking:
    every hash multiplier is derived from it.
    """
    from tokenizers import Regex, normalizers

    # A private-use char, so a token that is exactly one space survives
    # Strip() instead of collapsing to the empty string and merging with
    # unrelated tokens.
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    # The raw Rust tokenizer, matching what training decodes with
    # (no clean_up_tokenization_spaces).
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # A partial UTF-8 byte token: nothing to normalize, so key it
            # by its raw form.
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text

        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, compressed_vocab_size: int
) -> torch.Tensor:
    """One multiplier per (layer, lookback), from a per-layer RNG so layers
    hash differently. Kept odd and bounded so `token_id * multiplier` cannot
    overflow int64.
    """
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0,
            high=multiplier_bound,
            size=(max_ngram_size,),
            dtype=np.int64,
        )
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


class EngramLayout:
    """Bucket layout of the n-gram hash tables.

    A position is hashed as `max_ngram_size - 1` n-grams (2-gram .. max), each
    split over `n_heads` heads. Every (n-gram size, head) pair owns its own
    prime-sized bucket range in the layer's table; the primes are drawn in
    order and never reused, which keeps the ranges disjoint.
    """

    def __init__(self, config) -> None:
        self.layer_ids: tuple[int, ...] = tuple(config.engram_layer_ids)
        self.num_embeddings: tuple[int, ...] = tuple(config.engram_num_embeddings)
        self.max_ngram_size: int = config.engram_max_ngram_size
        self.n_heads: int = config.engram_n_heads
        self.head_dim: int = config.engram_head_dim
        self.compressed_vocab_size: int = config.engram_compressed_vocab_size
        self.pad_token_id: int = config.engram_pad_token_id
        assert len(self.layer_ids) == len(self.num_embeddings)

        primes = []
        seen: set[int] = set()
        for _ in self.layer_ids:
            per_ngram = []
            for _ in range(self.max_ngram_size - 1):
                sizes, current = [], config.engram_vocab_size - 1
                for _ in range(self.n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        self.primes: tuple[tuple[tuple[int, ...], ...], ...] = tuple(primes)
        self.n_hash_cols = (self.max_ngram_size - 1) * self.n_heads
        flat = [[p for per_ngram in layer for p in per_ngram] for layer in primes]
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
        self.offsets = torch.tensor(np.array(offsets))  # [n_layers, n_hash_cols]

    @classmethod
    def from_config(cls, config) -> "EngramLayout | None":
        if not getattr(config, "engram_layer_ids", None):
            return None
        return cls(config)


@triton.jit(do_not_specialize=["num_tokens"])
def _write_hash_cache_kernel(
    input_ids,
    token_map,
    dead_mask,
    slot_mapping,
    cache,
    num_tokens,
    input_stride,
    mask_stride,
    slot_stride,
    BLOCK_SIZE: tl.constexpr,
    dead_id,
):
    token_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    slot = tl.load(
        slot_mapping + token_idx * slot_stride, token_idx < num_tokens, other=-1
    ).to(tl.int64)
    valid = (token_idx < num_tokens) & (slot >= 0)
    token = tl.load(input_ids + token_idx * input_stride, valid, other=0)
    value = tl.load(token_map + token, valid, other=0)
    dead = tl.load(dead_mask + token_idx * mask_stride, valid, other=False)
    value = tl.where(dead, dead_id, value)
    tl.store(cache + slot, value, valid)


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "num_slots",
        "num_query_rows",
        "num_table_rows",
        "max_blocks",
    ]
)
def _hash_ids_kernel(
    input_ids,
    token_map,
    dead_mask,
    positions,
    block_table,
    query_start_loc,
    multipliers,
    primes,
    offsets,
    cache,
    lookback_token_ids,
    lookback_dead_mask,
    output,
    num_tokens,
    num_slots,
    pad_id,
    input_stride,
    mask_stride,
    position_stride,
    table_stride,
    table_col_stride,
    query_stride,
    num_query_rows,
    num_table_rows,
    max_blocks,
    cache_block_size,
    MAX_NGRAM: tl.constexpr,
    num_heads,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    dead_id,
    lookback_depth,
    lookback_row_stride,
    lookback_col_stride,
    lookback_mask_row_stride,
    lookback_mask_col_stride,
):
    token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    layer = tl.program_id(1)
    num_layers = tl.num_programs(1)
    valid = token < num_tokens
    # Upper bound in query_start_loc[1:], including repeated padding boundaries.
    lo = tl.full((BLOCK_T,), 0, tl.int32)
    hi = tl.full((BLOCK_T,), num_query_rows, tl.int32)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        end = tl.load(
            query_start_loc + (mid + 1) * query_stride,
            lo < hi,
            other=0,
        )
        right = token >= end
        active = lo < hi
        lo = tl.where(active & right, mid + 1, lo)
        hi = tl.where(active & ~right, mid, hi)
    req = tl.minimum(lo, num_query_rows - 1).to(tl.int64)
    chunk_idx = tl.load(query_start_loc + req * query_stride)
    chunk_idx = tl.minimum(chunk_idx, num_tokens - 1).to(tl.int64)
    chunk_start = tl.load(positions + chunk_idx * position_stride)
    position = tl.load(positions + token * position_stride, valid, other=0).to(tl.int64)
    head = tl.arange(0, BLOCK_H)
    blocked = tl.full((BLOCK_T,), False, tl.int1)
    rolling = tl.full((BLOCK_T,), 0, tl.int64)
    for shift in tl.static_range(MAX_NGRAM):
        lookback = position - shift
        in_batch = lookback >= chunk_start
        batch_idx = tl.maximum(token - shift, 0)
        batch_token = tl.load(
            input_ids + batch_idx * input_stride, valid & in_batch, other=0
        )
        batch_source = tl.load(token_map + batch_token, valid & in_batch, other=0)
        batch_dead = tl.load(
            dead_mask + batch_idx * mask_stride, valid & in_batch, other=False
        )
        batch_source = tl.where(batch_dead, dead_id, batch_source)

        col = chunk_start - 1 - lookback
        in_window = valid & ~in_batch & (col >= 0) & (col < lookback_depth)
        col = tl.minimum(tl.maximum(col, 0), lookback_depth - 1)
        window_token = tl.load(
            lookback_token_ids + req * lookback_row_stride + col * lookback_col_stride,
            in_window,
            other=-1,
        )
        known = in_window & (window_token >= 0)
        window_source = tl.load(token_map + window_token, known, other=0)
        window_dead = tl.load(
            lookback_dead_mask
            + req * lookback_mask_row_stride
            + col * lookback_mask_col_stride,
            known,
            other=False,
        )
        window_source = tl.where(window_dead, dead_id, window_source)

        if cache is not None:
            clamped = tl.minimum(
                tl.maximum(lookback, 0), max_blocks * cache_block_size - 1
            )
            block_row = tl.minimum(req, num_table_rows - 1)
            needs_cache = valid & ~in_batch & ~known
            block = tl.load(
                block_table
                + block_row * table_stride
                + (clamped // cache_block_size) * table_col_stride,
                needs_cache,
                other=0,
            ).to(tl.int64)
            slot = tl.minimum(
                tl.maximum(block * cache_block_size + clamped % cache_block_size, 0),
                num_slots - 1,
            )
            fallback = tl.load(cache + slot, needs_cache, other=0)
        else:
            fallback = tl.full((BLOCK_T,), pad_id, tl.int32)
        source = tl.where(
            in_batch, batch_source, tl.where(known, window_source, fallback)
        ).to(tl.int64)
        blocked |= (lookback < 0) | (source == dead_id)
        value = tl.where(blocked, pad_id, source)
        multiplier = tl.load(multipliers + layer * MAX_NGRAM + shift)
        rolling ^= value * multiplier
        if shift > 0:
            col = (shift - 1) * num_heads + head
            param_offset = layer * (MAX_NGRAM - 1) * num_heads + col
            prime = tl.load(primes + param_offset, head < num_heads, other=1)
            offset = tl.load(offsets + param_offset, head < num_heads, other=0)
            hashed = rolling[:, None] % prime[None, :] + offset[None, :]
            out_offset = (token.to(tl.int64) * num_layers + layer)[:, None] * (
                (MAX_NGRAM - 1) * num_heads
            ) + col[None, :]
            tl.store(output + out_offset, hashed, valid[:, None] & (head < num_heads))


class NgramHashState(nn.Module):
    """Maps each position to the hash ids of the n-grams ending there.

    Stateless on the V2 runner, which supplies every lookback token id. On
    the V1 runner it also keeps `hash_cache`, the slot-keyed rolling store
    of compressed ids (see module docstring), for generated tokens.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        layout: EngramLayout,
        swa_cache_module: nn.Module,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.swa_cache_module = swa_cache_module
        self.block_size: int = swa_cache_module.block_size
        self.lookback_depth: int = layout.max_ngram_size - 1
        self.use_slot_cache: bool = not vllm_config.use_v2_model_runner
        self._cache: torch.Tensor | None = None
        self._kv_cache_ref: weakref.ReferenceType[torch.Tensor] | None = None

        model_config = vllm_config.model_config
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_config.tokenizer,
            trust_remote_code=model_config.trust_remote_code,
            revision=model_config.revision,
        )
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        if vocab_size != layout.compressed_vocab_size:
            raise ValueError(
                f"Compressed vocab size mismatch: built {vocab_size} from the "
                f"tokenizer, config expects {layout.compressed_vocab_size}; "
                "every hash multiplier derives from it, so the engram tables "
                "would be silently rehashed."
            )
        self.pad_id = token_map[layout.pad_token_id]
        multipliers = compute_hash_multipliers(
            layout.layer_ids, layout.max_ngram_size, vocab_size
        )
        self.register_buffer(
            "token_map", torch.tensor(token_map, dtype=torch.int32), persistent=False
        )
        self.register_buffer("primes", torch.tensor(layout.primes), persistent=False)
        self.register_buffer("offsets", layout.offsets, persistent=False)
        self.register_buffer("multipliers", multipliers, persistent=False)
        logger.info(
            "Built engram token map (%d -> %d ids) for layers %s",
            len(token_map),
            vocab_size,
            layout.layer_ids,
        )

    def ensure_cache(self) -> bool:
        """Lazily size the slot-keyed cache from the bound SWA KV cache.

        Returns False while the KV cache is unbound (profile run); the caller
        skips engram hashing then. Without the slot cache only that check
        remains.
        """
        kv_cache = self.swa_cache_module.kv_cache
        if kv_cache.numel() == 0:
            self._cache = None
            self._kv_cache_ref = None
            return False
        if not self.use_slot_cache:
            return True
        if self._kv_cache_ref is not None and self._kv_cache_ref() is kv_cache:
            return True
        # Graph memory profiling binds a temporary, smaller KV cache first.
        # Rebinding must discard its hash history without retaining KV storage.
        self._cache = torch.zeros(
            kv_cache.shape[0] * self.block_size,
            dtype=torch.int32,
            device=kv_cache.device,
        )
        self._kv_cache_ref = weakref.ref(kv_cache)
        return True

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        dead_mask: torch.Tensor,
        lookback_token_ids: torch.Tensor,
        lookback_dead_mask: torch.Tensor,
        slot_mapping: torch.Tensor | None,
        block_table: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute [tokens, layers, hash columns] int32 n-gram hashes.

        History comes from the current chunk, then the runner's lookback
        window, then the optional V1 slot cache. V2 needs only one launch.
        """
        cache = self._cache if self.use_slot_cache else None
        num_tokens = input_ids.shape[0]
        num_layers, max_ngram = self.multipliers.shape
        num_heads = self.primes.shape[-1]
        output = input_ids.new_empty(
            (num_tokens, num_layers, (max_ngram - 1) * num_heads), dtype=torch.int32
        )
        if num_tokens == 0:
            return output
        if self.use_slot_cache:
            assert cache is not None and slot_mapping is not None
            assert block_table is not None
            # Finish writes before other thread blocks read fallback history.
            _write_hash_cache_kernel[(triton.cdiv(num_tokens, 256),)](
                input_ids,
                self.token_map,
                dead_mask,
                slot_mapping,
                cache,
                num_tokens,
                input_ids.stride(0),
                dead_mask.stride(0),
                slot_mapping.stride(0),
                256,
                DEAD_ID,
            )
        _hash_ids_kernel[(triton.cdiv(num_tokens, 32), num_layers)](
            input_ids,
            self.token_map,
            dead_mask,
            positions,
            block_table,
            query_start_loc,
            self.multipliers,
            self.primes,
            self.offsets,
            cache,
            lookback_token_ids,
            lookback_dead_mask,
            output,
            num_tokens,
            cache.shape[0] if cache is not None else 0,
            self.pad_id,
            input_stride=input_ids.stride(0),
            mask_stride=dead_mask.stride(0),
            position_stride=positions.stride(0),
            table_stride=block_table.stride(0) if block_table is not None else 0,
            table_col_stride=block_table.stride(1) if block_table is not None else 0,
            query_stride=query_start_loc.stride(0),
            num_query_rows=query_start_loc.numel() - 1,
            num_table_rows=block_table.shape[0] if block_table is not None else 0,
            max_blocks=block_table.shape[1] if block_table is not None else 0,
            cache_block_size=self.block_size,
            MAX_NGRAM=max_ngram,
            num_heads=num_heads,
            BLOCK_T=32,
            BLOCK_H=triton.next_power_of_2(num_heads),
            dead_id=DEAD_ID,
            lookback_depth=lookback_token_ids.shape[1],
            lookback_row_stride=lookback_token_ids.stride(0),
            lookback_col_stride=lookback_token_ids.stride(1),
            lookback_mask_row_stride=lookback_dead_mask.stride(0),
            lookback_mask_col_stride=lookback_dead_mask.stride(1),
            num_warps=4,
        )
        return output


def _engram_head_shard_weight_loader(
    param: torch.nn.Parameter, loaded_weight: torch.Tensor
) -> None:
    """Load this rank's complete head buckets. ue8m0 scales arrive as
    float8_e8m0fnu; keep the raw bytes (the param stores uint8)."""
    part_rows = param.shape[0]
    if loaded_weight.dtype == torch.float8_e8m0fnu:
        loaded_weight = loaded_weight.view(torch.uint8)
    shard = loaded_weight.narrow(0, param.engram_vocab_start, part_rows)
    assert shard.shape == param.shape, (
        f"engram shard {tuple(shard.shape)} does not fit param {tuple(param.shape)}"
    )
    param.data.copy_(shard)


@triton.jit
def _engram_lookup_kernel(
    weight,
    scales,
    ids,
    out,
    vocab_start,
    vocab_end,
    num_rows,
    ids_stride_t,
    ids_stride_h,
    HEAD_START: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    TOTAL_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GRID: tl.constexpr,
):
    """Gather fp8 rows, apply their ue8m0 block scales, write bf16.

    Only this rank's heads are read; padded heads write zeros for all-gather.
    `weight`/`scales` may address pinned host memory through UVA. `weight`
    is the e4m3 table viewed as uint8: Triton has no fp8 convert below SM89,
    so the bytes are decoded by ``_decode_fp8_f32`` (hardware convert where
    it exists, ALU decode on SM8x).
    """
    cols = tl.arange(0, DIM)
    scale_cols = cols // QUANT_BLOCK
    for base in tl.range(tl.program_id(0) * BLOCK_R, num_rows, GRID * BLOCK_R):
        rows = base + tl.arange(0, BLOCK_R)
        valid = rows < num_rows
        head = HEAD_START + rows % LOCAL_HEADS
        token = (rows // LOCAL_HEADS).to(tl.int64)
        index = tl.load(
            ids + token * ids_stride_t + head * ids_stride_h,
            mask=valid & (head < TOTAL_HEADS),
            other=-1,
        ).to(tl.int64)
        owned = valid & (head < TOTAL_HEADS)
        owned &= (index >= vocab_start) & (index < vocab_end)
        local = tl.where(owned, index - vocab_start, 0)
        values_u8 = tl.load(
            weight + local[:, None] * DIM + cols[None, :],
            mask=owned[:, None],
            other=0,
        )
        values = _decode_fp8_f32(values_u8, False)
        scale = tl.load(
            scales + local[:, None] * (DIM // QUANT_BLOCK) + scale_cols[None, :],
            mask=owned[:, None],
            other=0,
        )
        # ue8m0 is a power of two, so its byte *is* the fp32 exponent field.
        scale = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        tl.store(
            out + rows[:, None] * DIM + cols[None, :],
            (values * scale).to(tl.bfloat16),
            mask=valid[:, None],
        )


_ENGRAM_EMBED_KEY_RE = re.compile(r"(layers\.\d+\.engram)\b")


def _hf_engram_prefix(prefix: str) -> str:
    """``model.layers.1.engram`` (vLLM prefix) -> ``layers.1.engram`` (checkpoint)."""
    m = _ENGRAM_EMBED_KEY_RE.search(prefix)
    if m is None:
        raise ValueError(f"cannot derive the checkpoint engram key from {prefix!r}")
    return m.group(1)


_ENGRAM_PREAD_C = r"""
#include <pthread.h>
#include <stdint.h>
#include <unistd.h>

typedef struct {
    int fd_w, fd_s;
    int64_t base_w, base_s;
    const int64_t *rows;
    int64_t lo, hi;
    uint8_t *vals, *scales;
    int dim, sdim;
    int64_t short_reads;
} job_t;

static void *worker(void *arg) {
    job_t *j = (job_t *)arg;
    for (int64_t i = j->lo; i < j->hi; i++) {
        int64_t r = j->rows[i];
        ssize_t a = pread(j->fd_w, j->vals + i * j->dim, j->dim, j->base_w + r * j->dim);
        ssize_t b = pread(j->fd_s, j->scales + i * j->sdim, j->sdim, j->base_s + r * j->sdim);
        if (a != j->dim || b != j->sdim) j->short_reads++;
    }
    return NULL;
}

/* Gather rows[0..n) of a (values, scales) table with `threads` pthreads.
   Returns the number of short/failed reads (0 = success). */
int64_t engram_gather(int fd_w, int fd_s, int64_t base_w, int64_t base_s,
                      const int64_t *rows, int64_t n, uint8_t *vals,
                      uint8_t *scales, int dim, int sdim, int threads) {
    if (n <= 0) return 0;
    if (threads < 1) threads = 1;
    if (threads > n) threads = (int)n;
    pthread_t tid[256];
    job_t jobs[256];
    if (threads > 256) threads = 256;
    int64_t per = (n + threads - 1) / threads;
    int started = 0;
    for (int t = 0; t < threads; t++) {
        int64_t lo = t * per, hi = lo + per > n ? n : lo + per;
        if (lo >= hi) break;
        jobs[t] = (job_t){fd_w, fd_s, base_w, base_s, rows, lo, hi, vals, scales, dim, sdim, 0};
        if (pthread_create(&tid[t], NULL, worker, &jobs[t]) != 0) {
            worker(&jobs[t]);
            continue;
        }
        started = t + 1;
    }
    int64_t bad = 0;
    for (int t = 0; t < started; t++) {
        pthread_join(tid[t], NULL);
        bad += jobs[t].short_reads;
    }
    return bad;
}
"""


@functools.cache
def _engram_pread_lib():
    """Compile (once per host, cached on disk) and load the C gather helper.

    The Python ``preadv`` loop caps at ~125k syscalls/s regardless of thread
    count (per-call interpreter overhead under the GIL), which made a
    4096-token prefill chunk cost 1.6 s on the ranks owning an Engram layer
    and capped prefill at ~2.5k tok/s. The C loop issues the same reads
    from pthreads with the GIL released.
    """
    import ctypes
    import hashlib
    import subprocess
    import tempfile

    cache_dir = os.path.join(
        os.environ.get("VLLM_CACHE_ROOT", os.path.expanduser("~/.cache/vllm")),
        "engram_pread",
    )
    os.makedirs(cache_dir, exist_ok=True)
    digest = hashlib.sha256(_ENGRAM_PREAD_C.encode()).hexdigest()[:16]
    so_path = os.path.join(cache_dir, f"engram_pread_{digest}.so")
    if not os.path.exists(so_path):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "engram_pread.c")
            with open(src, "w") as f:
                f.write(_ENGRAM_PREAD_C)
            out = os.path.join(tmp, "engram_pread.so")
            subprocess.run(
                ["gcc", "-O2", "-shared", "-fPIC", "-pthread", src, "-o", out],
                check=True,
                capture_output=True,
            )
            os.replace(out, so_path)
    lib = ctypes.CDLL(so_path)
    lib.engram_gather.restype = ctypes.c_int64
    lib.engram_gather.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    return lib


class _DiskEngramTable:
    """Rows of one engram table gathered straight from the safetensors shard.

    ``storage="disk"``: nothing of the ~98 GiB table is ever loaded. A lookup
    takes the batch's hash ids to the host, deduplicates them, reads each
    unique row (256 e4m3 bytes + 8 UE8M0 scale bytes) with ``pread`` from a
    thread pool -- NVMe queue depth is what makes this fast -- into pinned
    staging buffers, copies them to the device and dequantizes there with the
    same Triton kernel the resident tables use (the staged rows are simply a
    tiny table addressed by the inverse-unique index).

    Costs: one device sync per lookup (the ids must reach the host) and NVMe
    random-read latency, both once per step per engram layer. Meant for the
    eager model-state hook that runs before the forward, never inside a
    captured CUDA graph.
    """

    HEADER_PREFIX = 8  # safetensors: u64 header length, then the JSON header

    def __init__(
        self,
        model_path: str,
        hf_prefix: str,
        num_embeddings: int,
        dim: int,
        block_size: int,
        vocab_start: int,
        vocab_end: int,
        num_threads: int,
        device: torch.device,
        staging_rows: int,
    ) -> None:
        if not os.path.isdir(model_path):
            raise ValueError(
                "engram storage='disk' needs a local checkpoint directory "
                f"(got {model_path!r}); download the checkpoint first."
            )
        self.dim = dim
        self.scale_dim = dim // block_size
        self.vocab_start = vocab_start
        self.vocab_end = vocab_end
        self.num_threads = max(1, num_threads)
        self.device = device
        self.weight_key = f"{hf_prefix}.embed.weight"
        self.scale_key = f"{hf_prefix}.embed.scale"
        (self.weight_path, self.weight_base) = self._locate(
            model_path, self.weight_key, (num_embeddings, dim), "F8_E4M3"
        )
        (self.scale_path, self.scale_base) = self._locate(
            model_path, self.scale_key, (num_embeddings, self.scale_dim), "F8_E8M0"
        )
        self._fds: dict[str, int] = {}
        # Pinned host + device staging for at most ``staging_rows`` unique
        # rows per lookup (batch tokens x local hash columns). ~26 MiB each
        # at 4096 x 24 rows.
        self.capacity = max(1, staging_rows)
        # Explicit device: model init runs under a ``torch.device("cuda")``
        # context, under which a bare ``torch.empty`` is not a CPU tensor.
        self.host_vals = torch.empty(
            (self.capacity, dim), dtype=torch.uint8, device="cpu", pin_memory=True
        )
        self.host_scales = torch.empty(
            (self.capacity, self.scale_dim),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        self.dev_vals = torch.empty((self.capacity, dim), dtype=torch.uint8, device=device)
        self.dev_scales = torch.empty(
            (self.capacity, self.scale_dim), dtype=torch.uint8, device=device
        )
        self._np_vals = self.host_vals.numpy()
        self._np_scales = self.host_scales.numpy()
        self._pool = concurrent.futures.ThreadPoolExecutor(self.num_threads)
        try:
            self._lib = _engram_pread_lib()
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "engram disk gather: C helper unavailable (%s); using the slower "
                "Python thread pool", e
            )
            self._lib = None
        logger.info(
            "Engram table %s served from disk (%s, %.1f GiB never loaded); "
            "%d reader threads, staging %d rows",
            hf_prefix,
            os.path.basename(self.weight_path),
            num_embeddings * (dim + self.scale_dim) / 1024**3,
            self.num_threads,
            self.capacity,
        )

    @staticmethod
    def _locate(
        model_path: str, key: str, shape: tuple[int, int], dtype: str
    ) -> tuple[str, int]:
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        shard = None
        if os.path.exists(index_path):
            with open(index_path) as f:
                shard = json.load(f)["weight_map"].get(key)
        candidates = (
            [os.path.join(model_path, shard)]
            if shard is not None
            else sorted(
                os.path.join(model_path, f)
                for f in os.listdir(model_path)
                if f.endswith(".safetensors")
            )
        )
        for path in candidates:
            with open(path, "rb") as f:
                (header_len,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(header_len))
            meta = header.get(key)
            if meta is None:
                continue
            if tuple(meta["shape"]) != tuple(shape) or meta["dtype"] != dtype:
                raise ValueError(
                    f"{key} in {path} is {meta['dtype']} {meta['shape']}, expected "
                    f"{dtype} {list(shape)}"
                )
            start, end = meta["data_offsets"]
            assert end - start == shape[0] * shape[1], (key, start, end, shape)
            return path, _DiskEngramTable.HEADER_PREFIX + header_len + start
        raise ValueError(f"{key} not found in the safetensors shards under {model_path}")

    def _fd(self, path: str) -> int:
        fd = self._fds.get(path)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            self._fds[path] = fd
        return fd

    def _read_rows(self, rows: np.ndarray) -> None:
        """Fill host staging rows [0, len(rows)) with table rows ``rows``."""
        n = len(rows)
        if n == 0:
            return
        wfd, sfd = self._fd(self.weight_path), self._fd(self.scale_path)
        lib = self._lib
        if lib is not None:
            rows64 = np.ascontiguousarray(rows, dtype=np.int64)
            bad = lib.engram_gather(
                wfd, sfd, self.weight_base, self.scale_base,
                rows64.ctypes.data, n,
                self._np_vals.ctypes.data, self._np_scales.ctypes.data,
                self.dim, self.scale_dim, self.num_threads,
            )
            if bad:
                raise RuntimeError(f"engram disk gather: {bad} short reads")
            return
        vals, scales = self._np_vals, self._np_scales
        dim, sdim = self.dim, self.scale_dim
        wbase, sbase = self.weight_base, self.scale_base

        def work(lo: int, hi: int) -> None:
            for i in range(lo, hi):
                r = int(rows[i])
                os.preadv(wfd, [memoryview(vals[i])], wbase + r * dim)
                os.preadv(sfd, [memoryview(scales[i])], sbase + r * sdim)

        step = max(64, -(-n // self.num_threads))
        futures = [
            self._pool.submit(work, lo, min(lo + step, n)) for lo in range(0, n, step)
        ]
        for fut in futures:
            fut.result()

    def gather(
        self, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """indices: [T, local_heads] global hash ids -> (dev_vals, dev_scales,
        inverse ids [T, local_heads] int32 (-1 = not owned), num_unique)."""
        ids = indices.to("cpu", torch.int64).numpy()  # one sync per lookup
        local = ids - self.vocab_start
        valid = (local >= 0) & (local < self.vocab_end - self.vocab_start)
        uniq, inv = np.unique(local[valid], return_inverse=True)
        n = len(uniq)
        if n > self.capacity:
            raise RuntimeError(
                f"engram disk staging too small: {n} unique rows > {self.capacity}"
            )
        self._read_rows(uniq)
        inv_ids = np.full(ids.shape, -1, dtype=np.int32)
        inv_ids[valid] = inv.astype(np.int32)
        if n > 0:
            self.dev_vals[:n].copy_(self.host_vals[:n], non_blocking=True)
            self.dev_scales[:n].copy_(self.host_scales[:n], non_blocking=True)
        inv_dev = torch.from_numpy(inv_ids).to(self.device, non_blocking=True)
        return self.dev_vals, self.dev_scales, inv_dev, n


def _pinned_exact(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """A CPU tensor pinned at its exact size (cudaHostRegister on a plain
    allocation), bypassing the caching host allocator's power-of-two rounding.
    Registered memory is page-locked and UVA-addressable like cudaHostAlloc
    memory (``is_pinned()`` is true, ``get_cuda_view_from_cpu_tensor`` works).
    Never unregistered: the tables live for the whole process."""
    t = torch.empty(shape, dtype=dtype, device="cpu")
    nbytes = t.numel() * t.element_size()
    if nbytes:
        err = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), nbytes, 0)
        if int(err) != 0:
            raise RuntimeError(
                f"cudaHostRegister of {nbytes / 1024**3:.1f} GiB failed with {err}"
            )
        assert t.is_pinned()
    return t


class ParallelEngramEmbedding(nn.Module):
    """The n-gram hash table, sharded by complete hash heads over TP ranks.
    Rows stay fp8 and are dequantized with ue8m0 per-32 scales on lookup.

    With `cpu_offload` the shard lives in pinned host memory and is read over
    UVA instead of HBM; the TP sharding is unchanged either way.
    """

    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        head_sizes: tuple[int, ...],
        block_size: int = 32,
        cpu_offload: bool = False,
        storage: str | None = None,
        hf_prefix: str | None = None,
        disk_threads: int = 32,
        staging_rows: int | None = None,
    ):
        super().__init__()
        if storage is None:
            storage = "cpu" if cpu_offload else "gpu"
        assert storage in ("cpu", "gpu", "disk"), storage
        cpu_offload = storage == "cpu"
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        assert head_sizes and all(size > 0 for size in head_sizes)
        assert sum(head_sizes) <= num_embeddings
        if cpu_offload and not is_uva_available():
            raise RuntimeError("Engram CPU offload requires UVA support")
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.block_size = block_size
        self.n_hash_cols = len(head_sizes)
        self.part_n_hash_cols = triton.cdiv(self.n_hash_cols, tp_size)
        self.head_start = tp_rank * self.part_n_hash_cols
        head_end = self.head_start + self.part_n_hash_cols
        self.vocab_start_idx = sum(head_sizes[: self.head_start])
        self.vocab_end_idx = sum(head_sizes[:head_end])
        self.part_num_embeddings = self.vocab_end_idx - self.vocab_start_idx
        self.tp_size = tp_size
        self.cpu_offload = cpu_offload
        self.storage = storage
        self._views: tuple[torch.Tensor, torch.Tensor] | None = None
        self._view_src: tuple[int, int] | None = None
        self._num_sms = torch.cuda.get_device_properties(
            torch.accelerator.current_device_index()
        ).multi_processor_count

        if storage == "disk":
            # No parameters at all: the loader never sees these tensors (they
            # are registered as skipped by name), the "weights not
            # initialized" check has nothing to miss, and rows are gathered
            # from the shard per step (see _DiskEngramTable).
            assert hf_prefix is not None
            from vllm.model_executor.model_loader.weight_utils import (
                register_skip_weight_names,
            )

            register_skip_weight_names(
                [f"{hf_prefix}.embed.weight", f"{hf_prefix}.embed.scale"]
            )
            self.weight = None
            self.weight_scale_inv = None
            vllm_config = get_current_vllm_config()
            self.disk = _DiskEngramTable(
                vllm_config.model_config.model,
                hf_prefix,
                num_embeddings,
                dim,
                block_size,
                self.vocab_start_idx,
                self.vocab_end_idx,
                disk_threads,
                torch.device(torch.accelerator.current_device_index()),
                staging_rows
                if staging_rows is not None
                else vllm_config.scheduler_config.max_num_batched_tokens
                * self.part_n_hash_cols,
            )
            return
        self.disk = None

        # cpu mode: the generic loader would materialise the whole 98 GiB
        # table in host RAM (safe_open().get_tensor copies) before copy_()
        # into the pinned parameter -- ~200 GiB per table at peak, which is
        # what OOM-killed the workers on a 278 GB host. Skip the tensors in
        # the loader and fill the pinned buffers straight from the shard in
        # chunks after loading (fill_pinned_from_checkpoint).
        self._pinned_fill: tuple[str, str] | None = None
        if cpu_offload and hf_prefix is not None:
            from vllm.model_executor.model_loader.weight_utils import (
                register_skip_weight_names,
            )

            register_skip_weight_names(
                [f"{hf_prefix}.embed.weight", f"{hf_prefix}.embed.scale"]
            )
            self._pinned_fill = (get_current_vllm_config().model_config.model, hf_prefix)

        # Explicit device: model init runs under a `torch.device("cuda")`
        # context, which would otherwise put the shard in HBM. In cpu mode
        # the buffers are pinned with cudaHostRegister at their exact size:
        # torch's pinned allocator rounds every block up to a power of two,
        # which turns a 98.3 GB table into a 128 GiB pinned block (the two
        # tables would need 264 GiB of host memory before anything else).
        if cpu_offload:
            self.weight = nn.Parameter(
                _pinned_exact((self.part_num_embeddings, dim), torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                _pinned_exact((self.part_num_embeddings, dim // block_size), torch.uint8),
                requires_grad=False,
            )
        else:
            self.weight = nn.Parameter(
                torch.empty(self.part_num_embeddings, dim, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(
                    self.part_num_embeddings, dim // block_size, dtype=torch.uint8
                ),
                requires_grad=False,
            )
        for param in (self.weight, self.weight_scale_inv):
            set_weight_attrs(
                param,
                {
                    "weight_loader": _engram_head_shard_weight_loader,
                    "engram_vocab_start": self.vocab_start_idx,
                },
            )
        if cpu_offload:
            logger.info(
                "Engram table offloaded to pinned host memory: %d rows x %d, "
                "%.2f GiB per rank",
                self.part_num_embeddings,
                dim,
                self.part_num_embeddings * (dim + dim // block_size) / 1024**3,
            )

    def needs_checkpoint_fill(self) -> bool:
        return self._pinned_fill is not None

    def fill_pinned_from_checkpoint(self, threads: int = 8) -> None:
        """Read this rank's rows of the two table tensors from the safetensors
        shard straight into the pinned parameters (chunked ``preadv``, no
        intermediate copy). The page cache the reads leave behind is
        evictable, so the only resident cost is the parameters themselves."""
        assert self._pinned_fill is not None
        model_path, hf_prefix = self._pinned_fill
        t0 = time.perf_counter()
        for key, param, width, dtype in (
            (f"{hf_prefix}.embed.weight", self.weight, self.dim, "F8_E4M3"),
            (
                f"{hf_prefix}.embed.scale",
                self.weight_scale_inv,
                self.dim // self.block_size,
                "F8_E8M0",
            ),
        ):
            path, base = _DiskEngramTable._locate(
                model_path, key, (self.num_embeddings, width), dtype
            )
            dst = param.data.view(torch.uint8).reshape(-1).numpy()
            nbytes = dst.nbytes
            assert nbytes == self.part_num_embeddings * width
            start = base + self.vocab_start_idx * width
            step = 256 << 20
            fd = os.open(path, os.O_RDONLY)
            try:

                def read_chunk(off: int) -> None:
                    end = min(off + step, nbytes)
                    view = memoryview(dst)[off:end]
                    pos = 0
                    while pos < end - off:
                        pos += os.preadv(fd, [view[pos:]], start + off + pos)

                with concurrent.futures.ThreadPoolExecutor(threads) as pool:
                    list(pool.map(read_chunk, range(0, nbytes, step)))
            finally:
                os.close(fd)
        self._pinned_fill = None
        logger.info(
            "Engram table %s: %.1f GiB read into pinned host memory in %.0f s",
            hf_prefix,
            (self.weight.numel() + self.weight_scale_inv.numel()) / 1024**3,
            time.perf_counter() - t0,
        )

    def _storage(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Parameters when resident, else cached UVA views of the pinned shard.

        Rebuilt if anything swaps `.data`, so a stale device pointer cannot
        survive silently.
        """
        # The lookup kernel takes the e4m3 table as bytes (no fp8 convert
        # below SM89); a uint8 view shares storage, so nothing is copied.
        if not self.cpu_offload:
            return self.weight.data.view(torch.uint8), self.weight_scale_inv.data
        src = (self.weight.data_ptr(), self.weight_scale_inv.data_ptr())
        if self._view_src != src:
            self._views = (
                get_accelerator_view_from_cpu_tensor(
                    self.weight.data.view(torch.uint8)
                ),
                get_accelerator_view_from_cpu_tensor(self.weight_scale_inv.data),
            )
            self._view_src = src
        assert self._views is not None
        return self._views

    def lookup(
        self, indices: torch.Tensor, out: torch.Tensor, background: bool = False
    ) -> None:
        """Look up local heads of [T, heads] into [T, local_heads, dim] bf16.

        `background` limits the grid to leave SMs for concurrent work.
        """
        rows = indices.shape[0] * self.part_n_hash_cols
        if not rows:
            return
        if self.disk is not None:
            self._lookup_from_disk(indices, out, background)
            return
        weight, scales = self._storage()
        # The table dwarfs TLB reach, so a persistent grid near the SM count
        # beats one program per row; halve it to leave SMs for the main stream.
        tiles = triton.cdiv(rows, 16)
        grid = min(tiles, self._num_sms // 2 if background else self._num_sms)
        _engram_lookup_kernel[(grid,)](
            weight,
            scales,
            indices,
            out,
            self.vocab_start_idx,
            self.vocab_end_idx,
            rows,
            indices.stride(0),
            indices.stride(1),
            HEAD_START=self.head_start,
            LOCAL_HEADS=self.part_n_hash_cols,
            TOTAL_HEADS=self.n_hash_cols,
            DIM=self.dim,
            QUANT_BLOCK=self.block_size,
            BLOCK_R=16,
            GRID=grid,
        )

    def _lookup_from_disk(
        self, indices: torch.Tensor, out: torch.Tensor, background: bool
    ) -> None:
        """``storage="disk"``: stage the batch's unique rows from the shard,
        then run the ordinary lookup kernel over that staged mini-table with
        the inverse-unique ids (padded/unowned heads map to -1 -> zeros)."""
        assert self.disk is not None
        head_end = min(self.head_start + self.part_n_hash_cols, self.n_hash_cols)
        local = indices[:, self.head_start : head_end]
        vals, scales, inv_ids, num_unique = self.disk.gather(local)
        if inv_ids.shape[1] != self.part_n_hash_cols:
            padded = inv_ids.new_full((inv_ids.shape[0], self.part_n_hash_cols), -1)
            padded[:, : inv_ids.shape[1]] = inv_ids
            inv_ids = padded
        rows = indices.shape[0] * self.part_n_hash_cols
        tiles = triton.cdiv(rows, 16)
        grid = min(tiles, self._num_sms // 2 if background else self._num_sms)
        _engram_lookup_kernel[(grid,)](
            vals,
            scales,
            inv_ids,
            out,
            0,
            num_unique,
            rows,
            inv_ids.stride(0),
            inv_ids.stride(1),
            HEAD_START=0,
            LOCAL_HEADS=self.part_n_hash_cols,
            TOTAL_HEADS=self.part_n_hash_cols,
            DIM=self.dim,
            QUANT_BLOCK=self.block_size,
            BLOCK_R=16,
            GRID=grid,
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: [num_tokens, n_hash_cols] -> [num_tokens, n_hash_cols, dim]
        bf16, gathered from all TP shards."""
        out = torch.empty(
            (indices.shape[0], self.part_n_hash_cols, self.dim),
            dtype=torch.bfloat16,
            device=indices.device,
        )
        self.lookup(indices, out)
        if self.tp_size > 1:
            out = tensor_model_parallel_all_gather(out, dim=1)
            out = out[:, : self.n_hash_cols]
        return out


@triton.jit(do_not_specialize=["num_kv_tokens"])
def _fused_engram_post_wkv_kernel(
    hidden_states,
    kv,
    q_weight,
    k_weight,
    token_mask,
    output,
    num_kv_tokens,
    hidden_stride_t,
    hidden_stride_h,
    hidden_stride_d,
    kv_stride_t,
    kv_stride_d,
    q_stride_h,
    q_stride_d,
    k_stride_h,
    k_stride_d,
    mask_stride,
    output_stride_t,
    output_stride_h,
    output_stride_d,
    eps,
    clamp_value,
    DIM: tl.constexpr,
    HC_MULT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    program_idx = tl.program_id(0)
    token_idx = program_idx // HC_MULT
    hc_idx = program_idx % HC_MULT
    token_idx = token_idx.to(tl.int64)
    source_idx = token_idx
    source_valid = source_idx < num_kv_tokens

    dim_offsets = tl.arange(0, BLOCK_SIZE)
    dim_valid = dim_offsets < DIM
    hidden = tl.load(
        hidden_states
        + token_idx * hidden_stride_t
        + hc_idx * hidden_stride_h
        + dim_offsets * hidden_stride_d,
        mask=dim_valid,
        other=0.0,
    ).to(tl.float32)
    key = tl.load(
        kv + source_idx * kv_stride_t + (hc_idx * DIM + dim_offsets) * kv_stride_d,
        mask=source_valid & dim_valid,
        other=0.0,
    ).to(tl.float32)
    q = tl.load(
        q_weight + hc_idx * q_stride_h + dim_offsets * q_stride_d,
        mask=dim_valid,
        other=0.0,
    ).to(tl.float32)
    k = tl.load(
        k_weight + hc_idx * k_stride_h + dim_offsets * k_stride_d,
        mask=dim_valid,
        other=0.0,
    ).to(tl.float32)

    hidden_rms = tl.rsqrt(tl.sum(hidden * hidden, axis=0) / DIM + eps)
    key_rms = tl.rsqrt(tl.sum(key * key, axis=0) / DIM + eps)
    dot = tl.sum(hidden * q * k * key, axis=0)
    dot *= hidden_rms * key_rms * tl.rsqrt(DIM * 1.0)
    gate_input = tl.sqrt(tl.maximum(tl.abs(dot), clamp_value))
    gate_input = tl.where(dot < 0.0, -gate_input, gate_input)
    gate = tl.sigmoid(gate_input)
    if HAS_MASK:
        active = tl.load(
            token_mask + source_idx * mask_stride,
            mask=source_valid,
            other=0,
        )
        gate = tl.where(active, gate, 0.0)

    value = tl.load(
        kv + source_idx * kv_stride_t + (HC_MULT * DIM + dim_offsets) * kv_stride_d,
        mask=source_valid & dim_valid,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output
        + token_idx * output_stride_t
        + hc_idx * output_stride_h
        + dim_offsets * output_stride_d,
        hidden + gate * value,
        mask=dim_valid,
    )


@triton.jit(do_not_specialize=["num_tokens", "token_start", "num_elements"])
def _engram_sp_rows_kernel(
    gathered,
    output,
    num_tokens,
    token_start,
    num_elements,
    LOCAL_WIDTH: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tokens = token_start + offsets // WIDTH
    cols = offsets % WIDTH
    source = (cols // LOCAL_WIDTH * num_tokens + tokens) * LOCAL_WIDTH
    source += cols % LOCAL_WIDTH
    values = tl.load(
        gathered + source, (offsets < num_elements) & (tokens < num_tokens), other=0
    )
    tl.store(output + offsets, values, offsets < num_elements)


class Engram(nn.Module):
    """Writes an n-gram lookup into the residual stream, gated by how well it
    matches that stream.

    The hash ids fetch `n_hash_cols` rows; `wkv` turns them into one key per
    hc copy plus a shared value. The gate is a normalized dot product of the
    stream against the key, signed-sqrt'ed before the sigmoid (matching the
    training kernel).
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None,
        layout: EngramLayout,
        layer_hash_index: int,
        use_sequence_parallel: bool,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layer_hash_index = layer_hash_index
        self.dim = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        self.clamp_value = 1e-6
        self.use_sequence_parallel = use_sequence_parallel

        # Named ``embed_tokens`` so the checkpoint's ``engram.embed.weight``
        # survives the mapper's ``embed.weight`` -> ``embed_tokens.weight``
        # suffix rule.
        engram_config = get_current_vllm_config().engram_config
        self.embed_tokens = ParallelEngramEmbedding(
            layout.num_embeddings[layer_hash_index],
            layout.head_dim,
            tuple(size for order in layout.primes[layer_hash_index] for size in order),
            storage=engram_config.resolved_storage() if engram_config else "cpu",
            hf_prefix=_hf_engram_prefix(prefix),
            disk_threads=engram_config.disk_threads if engram_config else 32,
        )
        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        self.wkv = ReplicatedLinear(
            n_hash_cols * layout.head_dim,
            self.dim * (self.hc_mult + 1),
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wkv",
        )
        self.q_weight = nn.Parameter(
            torch.empty(self.hc_mult, self.dim, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.k_weight = nn.Parameter(
            torch.empty(self.hc_mult, self.dim, dtype=torch.bfloat16),
            requires_grad=False,
        )

        max_tokens = get_current_vllm_config().scheduler_config.max_num_batched_tokens
        # Keep lookup results alive across breakable graph segments.
        self.staged_rows = torch.empty(
            max_tokens,
            self.embed_tokens.part_n_hash_cols,
            layout.head_dim,
            dtype=torch.bfloat16,
        )

    @property
    def rows_staged_by_runner(self) -> bool:
        """``storage="disk"``: rows are gathered by the model-state hook before
        the forward (host round trip; illegal inside graph capture), so the
        forward must not call :meth:`prepare_embeddings` itself."""
        return self.embed_tokens.storage == "disk"

    def prepare_embeddings(self, hash_ids: torch.Tensor) -> None:
        """Gather this layer's rows on the main stream before decoder layers."""
        self.embed_tokens.lookup(hash_ids, self.staged_rows[: hash_ids.shape[0]])

    def embed(self, hash_ids: torch.Tensor) -> torch.Tensor:
        """Gather heads, returning only local tokens when SP is enabled."""
        rows = self.staged_rows[: hash_ids.shape[0]]
        if self.embed_tokens.tp_size == 1:
            return rows
        if self.use_sequence_parallel:
            tp_size = self.embed_tokens.tp_size
            num_tokens, local_heads, dim = rows.shape
            gathered = tensor_model_parallel_all_gather(rows, dim=0)
            chunk = (num_tokens + tp_size - 1) // tp_size
            rows = rows.new_empty((chunk, self.embed_tokens.n_hash_cols, dim))
            _engram_sp_rows_kernel[(triton.cdiv(rows.numel(), 1024),)](
                gathered,
                rows,
                num_tokens,
                get_tensor_model_parallel_rank() * chunk,
                rows.numel(),
                local_heads * dim,
                self.embed_tokens.n_hash_cols * dim,
                BLOCK_SIZE=1024,
            )
            return rows
        rows = tensor_model_parallel_all_gather(rows, dim=1)
        return rows[:, : self.embed_tokens.n_hash_cols]

    def forward(
        self,
        hidden_states: torch.Tensor,
        hash_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """hidden_states: [T, hc_mult, dim]; hash_ids: [T, n_hash_cols] (all
        tokens, pre sequence-parallel shard); token_mask: [T], False shuts
        the gate so those positions pass through untouched."""
        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        num_kv_tokens = hash_ids.shape[0]
        assert token_mask is None or token_mask.shape == (num_kv_tokens,)
        if self.use_sequence_parallel:
            tp_size = get_tensor_model_parallel_world_size()
            tp_rank = get_tensor_model_parallel_rank()
            shard_size = (num_kv_tokens + tp_size - 1) // tp_size
            assert hidden_states.shape[0] == shard_size
            start = min(tp_rank * shard_size, num_kv_tokens)
            num_kv_tokens = min(shard_size, num_kv_tokens - start)
            if token_mask is not None:
                token_mask = token_mask[start : start + num_kv_tokens]

        num_tokens, hc_mult, dim = hidden_states.shape
        assert hc_mult == self.hc_mult and dim == self.dim
        assert kv.ndim == 2 and kv.shape[1] == (hc_mult + 1) * dim
        output = torch.empty_like(hidden_states)
        if num_tokens == 0:
            return output

        block_size = triton.next_power_of_2(dim)
        num_warps = 8 if block_size >= 2048 else 4
        mask = token_mask if token_mask is not None else hidden_states
        _fused_engram_post_wkv_kernel[(num_tokens * hc_mult,)](
            hidden_states,
            kv,
            self.q_weight,
            self.k_weight,
            mask,
            output,
            num_kv_tokens,
            hidden_states.stride(0),
            hidden_states.stride(1),
            hidden_states.stride(2),
            kv.stride(0),
            kv.stride(1),
            self.q_weight.stride(0),
            self.q_weight.stride(1),
            self.k_weight.stride(0),
            self.k_weight.stride(1),
            token_mask.stride(0) if token_mask is not None else 0,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            self.eps,
            self.clamp_value,
            DIM=dim,
            HC_MULT=hc_mult,
            BLOCK_SIZE=block_size,
            HAS_MASK=token_mask is not None,
            num_warps=num_warps,
        )
        return output
