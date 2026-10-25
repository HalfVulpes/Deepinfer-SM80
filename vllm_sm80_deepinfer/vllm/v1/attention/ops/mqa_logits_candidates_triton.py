# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate-only indexer logits for DeepSeek-V4.1's two-level selection.

V4.1 index layers after the candidate source (24/28/32/36 in V4.1-Flash)
only ever keep scores inside the ``candidate_topk_blocks`` blocks that layer
20 published; ``apply_candidate_mask`` sets every other column to ``-inf``
before the row top-k. On the sm_80 Triton path that meant scoring the whole
context (``[rows, max_seq_len]``) and then discarding all but
``topk_blocks * block_size`` columns per row.

These kernels score only the candidate positions, producing a compact
``[rows, topk_blocks * block_size]`` logits matrix, and map the row top-k back
to request-local positions. The selected set is identical to
``full logits -> apply_candidate_mask -> top-k`` (the masked columns could
never win), so this is a pure work reduction: 8x fewer scored columns at
128k context, 64x at 1M, on half of the index layers.

Both the prefill (gathered contiguous K) and the paged decode cache layouts
are handled by one kernel. Q is pre-decoded to bf16 once per row (it is tiny);
K bytes are decoded in-kernel through the same bf16 LUT the paged decode
kernel uses. The row top-k runs vLLM's CUDA ``top_k_per_row_decode`` on the
compact matrix and a small Triton kernel maps the columns back to positions.
"""

import os

import torch

import vllm._custom_ops as ops
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.mqa_logits_triton import (
    _decode_e4m3fn_bf16_lut,
    _get_e4m3fn_bf16_lut,
)

# Candidate blocks scored per CTA; the tile is TILE * block_size positions
# (128 at V4.1's block_size=8, matching the prefill kernel's BLOCK_N).
_TILE = 16
_NUM_WARPS = 4
_NUM_STAGES = 2

_PAD_SENTINEL = torch.iinfo(torch.int32).max
_PREFILL_ROW_STEP = 1024
# Unpaged (prefill) launches address K as one block of this size.
_UNPAGED_KV_BLOCK = 1 << 30
_ZERO_BT: dict[torch.device, torch.Tensor] = {}


def _zero_block_table(device: torch.device) -> torch.Tensor:
    t = _ZERO_BT.get(device)
    if t is None:
        t = torch.zeros((1, 1), dtype=torch.int32, device=device)
        _ZERO_BT[device] = t
    return t


def candidate_logits_enabled() -> bool:
    """Env kill switch for A/B validation (VLLM_DSV41_CAND_LOGITS=0)."""
    return os.environ.get("VLLM_DSV41_CAND_LOGITS", "1") != "0"


def use_candidate_logits(
    candidate_blocks: torch.Tensor | None,
    candidate_write: bool,
    candidate_block_size: int,
    width: int,
) -> bool:
    """Whether the compact candidate path replaces full scoring + mask.

    Only consumers of the candidate set qualify (the source layer must score
    everything to pick the candidates). Below ``topk_blocks * block_size``
    columns every block is a candidate and the mask is a no-op, so the full
    kernel is the cheaper one there.
    """
    if candidate_blocks is None or candidate_write:
        return False
    if not candidate_logits_enabled():
        return False
    return width > candidate_blocks.shape[1] * candidate_block_size


@triton.jit
def _candidate_logits_kernel(
    q_ptr,
    k_ptr,
    k_scale_ptr,
    weights_ptr,
    fp8_lut_ptr,
    cand_ptr,
    start_ptr,
    end_ptr,
    block_tables_ptr,
    logits_ptr,
    stride_q_m,
    stride_q_h,
    stride_q_d,
    stride_k_blk,
    stride_k_n,
    stride_k_d,
    stride_ks_blk,
    stride_ks_n,
    stride_w_m,
    stride_w_h,
    stride_c_m,
    stride_c_k,
    stride_bt_b,
    stride_bt_k,
    stride_l_m,
    stride_l_n,
    num_cand,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    CAND_BLOCK: tl.constexpr,
    KV_BLOCK: tl.constexpr,
    NEXT_N: tl.constexpr,
    HAS_START: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    m = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1)

    # Column n of this tile is offset n % CAND_BLOCK inside candidate slot
    # t*TILE + n // CAND_BLOCK of row m.
    offs_n = tl.arange(0, BLOCK_N)
    slot = t * TILE + offs_n // CAND_BLOCK
    within = offs_n % CAND_BLOCK
    cand = tl.load(
        cand_ptr + m * stride_c_m + slot * stride_c_k,
        mask=slot < num_cand,
        other=-1,
    ).to(tl.int64)
    start = tl.load(start_ptr + m) if HAS_START else 0
    end = tl.load(end_ptr + m)
    pos = start + cand * CAND_BLOCK + within
    # Same bounds as apply_candidate_mask: a real candidate slot and inside
    # the row's causal range [start, end). Padded slots carry -1.
    valid = (cand >= 0) & (pos < end)
    pos = tl.where(valid, pos, 0)

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    mask_h = offs_h < num_heads
    mask_d = offs_d < head_dim

    q = tl.load(
        q_ptr + m * stride_q_m + offs_h[:, None] * stride_q_h + offs_d[None, :] * stride_q_d,
        mask=mask_h[:, None] & mask_d[None, :],
        other=0.0,
    )

    # One code path for both layouts: the prefill wrapper passes a single
    # "block" (KV_BLOCK larger than any N, a one-entry zero block table with
    # zero strides), so blk == 0 and in_blk == pos. A constexpr if here
    # trips Triton 3.7's RemoveLayoutConversions pass on sm_80.
    blk = tl.load(
        block_tables_ptr + (m // NEXT_N) * stride_bt_b + (pos // KV_BLOCK) * stride_bt_k,
        mask=valid,
        other=0,
    ).to(tl.int64)
    in_blk = pos % KV_BLOCK
    k_row = blk * stride_k_blk + in_blk * stride_k_n
    s_row = blk * stride_ks_blk + in_blk * stride_ks_n

    k_byte = tl.load(
        k_ptr + k_row[:, None] + offs_d[None, :] * stride_k_d,
        mask=valid[:, None] & mask_d[None, :],
        other=0,
    )
    k_scale = tl.load(k_scale_ptr + s_row, mask=valid, other=0.0)
    k = _decode_e4m3fn_bf16_lut(k_byte, fp8_lut_ptr)
    s = tl.dot(q, tl.trans(k)) * k_scale[None, :]

    w = tl.load(
        weights_ptr + m * stride_w_m + offs_h * stride_w_h,
        mask=mask_h,
        other=0.0,
    )
    s = tl.where(s > 0, s, 0.0) * w[:, None]
    out = tl.sum(s, axis=0)
    out = tl.where(valid, out, float("-inf"))

    col = t * BLOCK_N + offs_n
    tl.store(
        logits_ptr + m * stride_l_m + col * stride_l_n,
        out,
        mask=col < num_cand * CAND_BLOCK,
    )


def _launch(
    q_bf16: torch.Tensor,
    k_byte: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    cand: torch.Tensor,
    start: torch.Tensor | None,
    end: torch.Tensor,
    block_tables: torch.Tensor | None,
    next_n: int,
    block_size: int,
    kv_block_size: int,
    paged: bool,
) -> torch.Tensor:
    rows, num_heads, head_dim = q_bf16.shape
    num_cand = cand.shape[1]
    width = num_cand * block_size
    logits = torch.empty((rows, width), dtype=torch.float32, device=q_bf16.device)
    if rows == 0:
        return logits
    fp8_lut = _get_e4m3fn_bf16_lut(q_bf16.device)
    if paged:
        k_strides = (k_byte.stride(0), k_byte.stride(1), k_byte.stride(2))
        s_strides = (k_scale.stride(0), k_scale.stride(1))
        bt_strides = (block_tables.stride(0), block_tables.stride(1))
    else:
        k_strides = (0, k_byte.stride(0), k_byte.stride(1))
        s_strides = (0, k_scale.stride(0))
        bt_strides = (0, 0)
        block_tables = _zero_block_table(q_bf16.device)
        kv_block_size = _UNPAGED_KV_BLOCK
    block_n = _TILE * block_size
    grid = (rows, triton.cdiv(num_cand, _TILE))
    _candidate_logits_kernel[grid](
        q_bf16,
        k_byte,
        k_scale,
        weights,
        fp8_lut,
        cand,
        end if start is None else start,
        end,
        block_tables,
        logits,
        q_bf16.stride(0),
        q_bf16.stride(1),
        q_bf16.stride(2),
        *k_strides,
        *s_strides,
        weights.stride(0),
        weights.stride(1),
        cand.stride(0),
        cand.stride(1),
        *bt_strides,
        logits.stride(0),
        logits.stride(1),
        num_cand,
        num_heads=num_heads,
        head_dim=head_dim,
        CAND_BLOCK=block_size,
        KV_BLOCK=kv_block_size,
        NEXT_N=next_n,
        HAS_START=start is not None,
        BLOCK_H=max(16, triton.next_power_of_2(num_heads)),
        BLOCK_D=triton.next_power_of_2(head_dim),
        TILE=_TILE,
        BLOCK_N=block_n,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return logits


@triton.jit
def _map_topk_kernel(
    idx_ptr,
    logits_ptr,
    cand_ptr,
    out_ptr,
    stride_i_m,
    stride_i_k,
    stride_l_m,
    stride_l_n,
    stride_c_m,
    stride_c_k,
    stride_o_m,
    stride_o_k,
    pad_value,
    CAND_BLOCK: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    m = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < TOPK
    col = tl.load(idx_ptr + m * stride_i_m + offs * stride_i_k, mask=mask, other=-1)
    ok = col >= 0
    col = tl.where(ok, col, 0)
    # top_k_per_row_decode fills a row that has fewer finite columns than
    # TOPK with -inf picks (it never emits -1). Those map to candidate
    # positions past the row's causal bound, so drop them here: a selected
    # column is only valid if its score is finite.
    score = tl.load(
        logits_ptr + m * stride_l_m + col * stride_l_n,
        mask=mask & ok,
        other=float("-inf"),
    )
    ok = ok & (score > float("-inf"))
    block = tl.load(
        cand_ptr + m * stride_c_m + (col // CAND_BLOCK) * stride_c_k,
        mask=mask & ok,
        other=-1,
    )
    ok = ok & (block >= 0)
    rel = block * CAND_BLOCK + col % CAND_BLOCK
    tl.store(
        out_ptr + m * stride_o_m + offs * stride_o_k,
        tl.where(ok, rel, pad_value),
        mask=mask,
    )


def _topk_from_candidate_logits(
    logits: torch.Tensor,
    cand: torch.Tensor,
    block_size: int,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    sort: bool,
) -> None:
    """Row top-k over the compact logits, emitted as request-local positions.

    ``top_k_per_row_decode`` (the kernel the full decode path already trusts
    on sm_80) selects the columns; the map kernel turns compact column
    ``c`` into position ``cand[row, c // block] * block + c % block``. With
    ``sort`` the row is emitted in ascending position order with the ``-1``
    pads at the tail, matching ``_top_k_per_row_prefill_torch``; decode keeps
    the selection kernel's own order like the full path does.
    """
    rows, width = logits.shape
    assert topk_tokens <= width
    cols = torch.empty((rows, topk_tokens), dtype=torch.int32, device=logits.device)
    row_width = torch.full((rows,), width, dtype=torch.int32, device=logits.device)
    ops.top_k_per_row_decode(
        logits,
        1,
        row_width,
        cols,
        rows,
        logits.stride(0),
        logits.stride(1),
        topk_tokens,
    )
    out = cols if sort else topk_indices
    _map_topk_kernel[(rows,)](
        cols,
        logits,
        cand,
        out,
        cols.stride(0),
        cols.stride(1),
        logits.stride(0),
        logits.stride(1),
        cand.stride(0),
        cand.stride(1),
        out.stride(0),
        out.stride(1),
        _PAD_SENTINEL if sort else -1,
        CAND_BLOCK=block_size,
        TOPK=topk_tokens,
        BLOCK_K=triton.next_power_of_2(topk_tokens),
    )
    if sort:
        out, _ = out.sort(dim=-1)
        topk_indices[:] = torch.where(out == _PAD_SENTINEL, out.new_full((), -1), out)


def candidate_topk_prefill(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    candidate_blocks: torch.Tensor,
    block_size: int,
    topk_indices: torch.Tensor,
    topk_tokens: int,
) -> None:
    """Prefill: score candidates against the gathered K and write the top-k.

    Args:
        q:             [M, H, D] fp8_e4m3fn
        kv:            (k_fp8 [N, D], k_scales [N]) as in fp8_mqa_logits_triton
        weights:       [M, H] float32
        cu_seqlen_ks/ke: [M] int32 packed-column bounds per row
        candidate_blocks: [M, K] int32 request-local block ids, -1 padded
        topk_indices:  [M, topk_tokens] int32 output (relative to ks)
    """
    k_fp8, k_scales = kv
    q_bf16 = q.to(torch.bfloat16)
    k_byte = k_fp8.view(torch.uint8)
    k_scales = k_scales.reshape(-1)
    # Row-chunk so the compact transient stays at
    # _PREFILL_ROW_STEP * topk_blocks * block_size * 4 B (64 MiB at V4.1).
    for r0 in range(0, q.shape[0], _PREFILL_ROW_STEP):
        r1 = min(r0 + _PREFILL_ROW_STEP, q.shape[0])
        logits = _launch(
            q_bf16[r0:r1],
            k_byte,
            k_scales,
            weights[r0:r1],
            candidate_blocks[r0:r1],
            cu_seqlen_ks[r0:r1],
            cu_seqlen_ke[r0:r1],
            None,
            1,
            block_size,
            1,
            False,
        )
        _topk_from_candidate_logits(
            logits,
            candidate_blocks[r0:r1],
            block_size,
            topk_indices[r0:r1],
            topk_tokens,
            sort=True,
        )
        del logits


def candidate_topk_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    candidate_blocks: torch.Tensor,
    block_size: int,
    topk_indices: torch.Tensor,
    topk_tokens: int,
) -> None:
    """Decode: score candidates straight out of the paged indexer cache.

    Args:
        q:          [B, next_n, H, D] fp8_e4m3fn
        kv_cache:   [num_blocks, kv_block_size, 1, D+4] uint8 (FP8 + fp32 scale)
        weights:    [B*next_n, H] float32
        seq_lens:   per-token effective context lengths: (B,) / (B, 1) when
                    every row of a request shares one bound, or (B, next_n)
                    with [b, j] = L_b - next_n + j + 1 (native spec decode)
        block_tables: [B, max_blocks] int32
        candidate_blocks: [B*next_n, K] int32, -1 padded
        topk_indices: [B*next_n, topk_tokens] int32 output
    """
    B, next_n, num_heads, head_dim = q.shape
    rows = B * next_n
    num_blocks, kv_block_size, one, d_plus_4 = kv_cache.shape
    assert one == 1 and d_plus_4 == head_dim + 4
    kv_flat = kv_cache.view(num_blocks, -1)
    k_end = kv_block_size * head_dim
    kv_byte = kv_flat[:, :k_end].as_strided(
        (num_blocks, kv_block_size, head_dim),
        (kv_flat.stride(0), head_dim, 1),
    )
    kv_scale = kv_flat[:, k_end:].view(torch.float32)

    vis = seq_lens.reshape(-1)
    if vis.numel() == rows:
        row_end = vis
    else:
        # One bound per request: row j of request b sees L_b - next_n + j + 1
        # positions (the paged kernel's `k_offset <= q_offset`).
        offs = torch.arange(next_n, device=vis.device, dtype=vis.dtype)
        row_end = (vis[:B, None] - next_n + 1 + offs[None, :]).clamp_(min=0)
        row_end = row_end.reshape(-1)
    row_end = row_end.to(torch.int32)

    logits = _launch(
        q.to(torch.bfloat16).reshape(rows, num_heads, head_dim),
        kv_byte,
        kv_scale,
        weights,
        candidate_blocks,
        None,
        row_end,
        block_tables,
        next_n,
        block_size,
        kv_block_size,
        True,
    )
    _topk_from_candidate_logits(
        logits, candidate_blocks, block_size, topk_indices, topk_tokens, sort=False
    )


def warmup_candidate_logits_triton(
    num_heads: int,
    head_dim: int,
    block_size: int,
    kv_block_sizes: list[int],
    device: torch.device,
) -> None:
    """Compile both specializations before any cudagraph capture."""
    rows, num_cand = 2, 64
    q = torch.zeros((rows, num_heads, head_dim), dtype=torch.bfloat16, device=device)
    w = torch.zeros((rows, num_heads), dtype=torch.float32, device=device)
    cand = torch.arange(num_cand, dtype=torch.int32, device=device)
    cand = cand[None, :].repeat(rows, 1)
    n = num_cand * block_size
    k = torch.zeros((n, head_dim), dtype=torch.uint8, device=device)
    ks = torch.zeros(rows, dtype=torch.int32, device=device)
    ke = torch.full((rows,), n, dtype=torch.int32, device=device)
    _launch(q, k, ks.new_zeros(n, dtype=torch.float32), w, cand, ks, ke,
            None, 1, block_size, 1, False)
    for kv_bs in kv_block_sizes:
        nb = triton.cdiv(n, kv_bs)
        kv_byte = torch.zeros((nb, kv_bs, head_dim), dtype=torch.uint8, device=device)
        kv_scale = torch.zeros((nb, kv_bs), dtype=torch.float32, device=device)
        bt = torch.arange(nb, dtype=torch.int32, device=device)[None, :].repeat(rows, 1)
        _launch(q, kv_byte, kv_scale, w, cand, None, ke, bt, 1, block_size,
                kv_bs, True)
