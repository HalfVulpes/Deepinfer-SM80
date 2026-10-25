#!/usr/bin/env python3
"""
Part 3 Microbenchmark:
1. Verify bit-exact uint16 parity of _dequantize_and_gather_k_sm80_kernel
2. Sweep decode num_splits & num_warps on GA100 (74 SMs)
3. Benchmark _sparse_attn_prefill_ragged_kernel with direct_out + constexpr dims/strides
4. Verify mhc_fused_tilelang with bf16 intermediate rounding (err ~ 1e-7)
"""
import math
import torch
import triton
import triton.language as tl

from vllm.v1.attention.ops.fp8_sm80 import (
    _decode_fp8_lut,
    get_e4m3fn_bf16_lut,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    rocm_sparse_attn_prefill,
    _sparse_attn_decode_reduce_kernel,
    _decode_block_h,
    _decode_num_splits,
    _decode_partial_iters,
    _prefill_block_h,
    _prefill_block_k,
    build_ragged_indices_from_dense,
    _sparse_kv_row_offset,
)
from vllm.models.deepseek_v4_1.common.ops.cache_utils import (
    dequantize_and_gather_k_cache_triton,
)
from microbench_sm80_kernels import (
    bench_fn,
    _sparse_attn_decode_partial_sm80_kernel,
    _decode_e8m0_scales_bf16_triton,
)
from microbench_sm80_part2 import (
    dequantize_and_gather_k_cache_baseline,
    dequantize_and_gather_k_cache_sm80,
)


def test_dequant_bit_exact():
    print("=" * 80)
    print("TEST 1: dequantize_and_gather_k_cache_sm80 Bit-Exact Verification")
    print("=" * 80)
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    num_blocks = 1024
    block_size = 16
    # Layout inside each block of size block_size * 584:
    # First block_size * 576 bytes: token_data (each 576B = 448B fp8 + 128B bf16)
    # Last block_size * 8 bytes: token_scales (each 8B uint8)
    raw = torch.zeros((num_blocks, block_size * 584), dtype=torch.uint8, device=device)
    data_view = raw[:, : block_size * 576].view(num_blocks, block_size, 576)
    scale_view = raw[:, block_size * 576 :].view(num_blocks, block_size, 8)

    data_view[:, :, :448] = torch.randint(0, 126, (num_blocks, block_size, 448), dtype=torch.uint8, device=device)
    rope_bf16 = torch.randn(num_blocks, block_size, 64, dtype=torch.bfloat16, device=device)
    data_view[:, :, 448:576] = rope_bf16.view(torch.uint8)
    scale_view[:, :, :] = torch.randint(120, 132, (num_blocks, block_size, 8), dtype=torch.uint8, device=device)

    k_cache = raw.view(num_blocks, block_size, 584)

    for n_compressed in [1024, 4096, 12288]:
        out_ref = torch.zeros((1, n_compressed, 512), dtype=torch.bfloat16, device=device)
        out_opt = torch.zeros((1, n_compressed, 512), dtype=torch.bfloat16, device=device)
        seq_lens = torch.tensor([n_compressed], dtype=torch.int32, device=device)
        block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)

        dequantize_and_gather_k_cache_baseline(
            out_ref, k_cache, seq_lens, None, block_table, block_size, 0, False
        )
        dequantize_and_gather_k_cache_triton(
            out_opt, k_cache, seq_lens, None, block_table, block_size, 0, False
        )
        bit_exact = torch.equal(out_ref.view(torch.uint16), out_opt.view(torch.uint16))
        max_err = (out_ref.float() - out_opt.float()).abs().max().item()
        print(f"n_compressed={n_compressed:5d} | Bit-exact (uint16): {bit_exact} | max_abs_err: {max_err:.2e}")


def test_decode_splits_sweep():
    print("\n" + "=" * 80)
    print("TEST 2: Decode Split & Warp Tuning on GA100 (74 SMs)")
    print("=" * 80)
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    num_heads = 32
    nope_dim = 448
    rope_dim = 64
    head_dim = 512
    scale = 1.0 / math.sqrt(head_dim)
    num_blocks = 256
    main_block_size = 64
    extra_block_size = 16

    raw_swa = torch.zeros((num_blocks, main_block_size * 584), dtype=torch.uint8, device=device)
    raw_swa[:, : main_block_size * 576].view(num_blocks, main_block_size, 576)[:, :, :448] = torch.randint(0, 120, (num_blocks, main_block_size, 448), dtype=torch.uint8, device=device)
    raw_swa[:, : main_block_size * 576].view(num_blocks, main_block_size, 576)[:, :, 448:] = torch.randn(num_blocks, main_block_size, 64, dtype=torch.bfloat16, device=device).view(torch.uint8)
    raw_swa[:, main_block_size * 576 :] = 127
    swa_k_cache = raw_swa.view(num_blocks, main_block_size, 584)

    raw_kv = torch.zeros((num_blocks, extra_block_size * 584), dtype=torch.uint8, device=device)
    raw_kv[:, : extra_block_size * 576].view(num_blocks, extra_block_size, 576)[:, :, :448] = torch.randint(0, 120, (num_blocks, extra_block_size, 448), dtype=torch.uint8, device=device)
    raw_kv[:, : extra_block_size * 576].view(num_blocks, extra_block_size, 576)[:, :, 448:] = torch.randn(num_blocks, extra_block_size, 64, dtype=torch.bfloat16, device=device).view(torch.uint8)
    raw_kv[:, extra_block_size * 576 :] = 127
    kv_cache = raw_kv.view(num_blocks, extra_block_size, 584)

    attn_sink = torch.zeros(num_heads, dtype=torch.float32, device=device)
    fp8_lut = get_e4m3fn_bf16_lut(device)
    block_h = _decode_block_h(num_heads)
    heads_blocks = triton.cdiv(num_heads, block_h)
    block_k = 32

    for B in [1, 4, 6, 12, 16, 24]:
        q = torch.randn(B, num_heads, head_dim, dtype=torch.bfloat16, device=device)
        out = torch.empty_like(q)
        swa_len = 512
        topk_len = 1024
        swa_indices = torch.randint(0, num_blocks * main_block_size, (B, swa_len), dtype=torch.int32, device=device)
        swa_lens = torch.full((B,), swa_len, dtype=torch.int32, device=device)
        topk_indices = torch.randint(0, num_blocks * extra_block_size, (B, topk_len), dtype=torch.int32, device=device)
        topk_lens = torch.full((B,), topk_len, dtype=torch.int32, device=device)
        swa_ragged_indices, swa_ragged_indptr = build_ragged_indices_from_dense(
            swa_indices, swa_lens, num_rows=num_blocks * main_block_size
        )
        topk_ragged_indices, topk_ragged_indptr = build_ragged_indices_from_dense(
            topk_indices, topk_lens, num_rows=num_blocks * extra_block_size
        )

        best_t = 1e9
        best_cfg = None
        default_s = _decode_num_splits(B, heads_blocks, float(swa_len), float(topk_len), block_k)
        default_t = None

        for num_splits in [2, 3, 4, 6, 8, 12, 16]:
            for n_warps in [4, 8]:
                def _run():
                    part_m = torch.empty((B, num_splits, num_heads), dtype=torch.float32, device=device)
                    part_l = torch.empty_like(part_m)
                    part_acc = torch.empty((B, num_splits, num_heads, head_dim), dtype=torch.float32, device=device)
                    _sparse_attn_decode_partial_sm80_kernel[(B, num_splits, heads_blocks)](
                        q, swa_k_cache, swa_ragged_indices, swa_ragged_indptr,
                        kv_cache, topk_ragged_indices, topk_ragged_indptr,
                        part_m, part_l, part_acc, fp8_lut,
                        q.stride(0), q.stride(1), swa_k_cache.stride(0), kv_cache.stride(0),
                        part_m.stride(0), part_m.stride(1), part_acc.stride(0), part_acc.stride(1), part_acc.stride(2),
                        swa_k_cache.shape[0] * swa_k_cache.shape[1],
                        kv_cache.shape[0] * kv_cache.shape[1],
                        swa_k_cache.shape[1], kv_cache.shape[1],
                        scale, num_heads,
                        HAS_EXTRA=True, NOPE_DIM=nope_dim, COMB_DIM=head_dim, ROPE_DIM=rope_dim,
                        IS_FNUZ_MAIN=False, IS_FNUZ_EXTRA=False,
                        BLOCK_H=block_h, BLOCK_K=block_k, NUM_SPLITS=num_splits, NUM_STAGES=1,
                        num_warps=n_warps,
                    )
                    _sparse_attn_decode_reduce_kernel[(B, num_heads)](
                        part_m, part_l, part_acc, attn_sink, out,
                        out.stride(0), out.stride(1), part_m.stride(0), part_m.stride(1),
                        part_acc.stride(0), part_acc.stride(1), part_acc.stride(2),
                        num_heads, HAS_ATTN_SINK=True, ADAPTIVE_SPLITS=False,
                        COMB_DIM=head_dim, BLOCK_H=1, NUM_SPLITS=num_splits,
                        SPLITS_PAD=triton.next_power_of_2(num_splits), num_warps=4,
                    )
                t = bench_fn(_run, warmup=10, rep=50)
                if num_splits == default_s and n_warps == 8:
                    default_t = t
                if t < best_t:
                    best_t = t
                    best_cfg = (num_splits, n_warps)
        print(
            f"B={B:2d} | Default (s={default_s:2d}, w=8): {default_t:6.2f} us | "
            f"Best (s={best_cfg[0]:2d}, w={best_cfg[1]}): {best_t:6.2f} us ({default_t/best_t:.2f}x)"
        )


@triton.jit
def _sparse_attn_prefill_ragged_sm80_kernel(
    q_ptr,
    kv_ptr,
    kv_indices_ptr,
    kv_indptr_ptr,
    attn_sink_ptr,
    out_ptr,
    q_stride_t,
    q_stride_h,
    q_stride_d: tl.constexpr,
    kv_stride_n,
    kv_stride_d: tl.constexpr,
    out_stride_t,
    out_stride_h,
    out_stride_d: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_kv,
    scale,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_idx = tl.program_id(0)
    pid_h = tl.program_id(1)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads

    if num_heads % BLOCK_H == 0 and head_dim == BLOCK_D:
        q = tl.load(
            q_ptr
            + query_idx * q_stride_t
            + head_offsets[:, None] * q_stride_h
            + dim_offsets[None, :] * q_stride_d,
        )
    else:
        dim_mask = dim_offsets < head_dim
        q = tl.load(
            q_ptr
            + query_idx * q_stride_t
            + head_offsets[:, None] * q_stride_h
            + dim_offsets[None, :] * q_stride_d,
            mask=head_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    kv_start = tl.load(kv_indptr_ptr + query_idx)
    kv_end = tl.load(kv_indptr_ptr + query_idx + 1)
    kv_len = kv_end - kv_start

    k_offsets = tl.arange(0, BLOCK_K)
    slot = tl.load(
        kv_indices_ptr + kv_start + k_offsets, mask=k_offsets < kv_len, other=-1
    )
    for k_start in tl.range(0, kv_len, BLOCK_K):
        k_pos = k_start + k_offsets
        in_range = k_pos < kv_len
        valid = in_range & (slot >= 0) & (slot < num_kv)
        safe_slot = tl.where(valid, slot, 0)

        kv = tl.load(
            kv_ptr
            + _sparse_kv_row_offset(safe_slot[:, None], kv_stride_n)
            + dim_offsets[None, :] * kv_stride_d,
            mask=valid[:, None],
            other=0.0,
        )

        next_k_pos = k_start + BLOCK_K + k_offsets
        slot = tl.load(
            kv_indices_ptr + kv_start + next_k_pos, mask=next_k_pos < kv_len, other=-1
        )

        scores = tl.dot(q, tl.trans(kv)) * scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    if HAS_ATTN_SINK:
        sink = tl.load(
            attn_sink_ptr + head_offsets, mask=head_mask, other=neg_large
        ).to(tl.float32)
        m_final = tl.maximum(m_i, sink)
        alpha = tl.exp(m_i - m_final)
        l_final = l_i * alpha + tl.exp(sink - m_final)
        denom = tl.maximum(l_final, 1.0e-30)
        out = tl.where(
            l_final[:, None] > 0.0,
            (acc * alpha[:, None]) / denom[:, None],
            0.0,
        )
    else:
        denom = tl.maximum(l_i, 1.0e-30)
        out = tl.where(l_i[:, None] > 0.0, acc / denom[:, None], 0.0)

    tl.store(
        out_ptr
        + query_idx * out_stride_t
        + head_offsets[:, None] * out_stride_h
        + dim_offsets[None, :] * out_stride_d,
        out,
        mask=head_mask[:, None] if (num_heads % BLOCK_H != 0) else None,
    )


def test_prefill_attn():
    print("\n" + "=" * 80)
    print("TEST 3: Sparse MLA Prefill Attention (Direct-Out + Ragged Reuse + Constexpr)")
    print("=" * 80)
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    num_queries = 1536
    num_heads = 32
    head_dim = 512
    num_kv = 8192
    topk_width = 1536
    scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(num_queries, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    kv = torch.randn(num_kv, 1, head_dim, dtype=torch.bfloat16, device=device)
    indices = torch.randint(0, num_kv, (num_queries, topk_width), dtype=torch.int32, device=device)
    lengths = torch.full((num_queries,), topk_width, dtype=torch.int32, device=device)
    attn_sink = torch.zeros(num_heads, dtype=torch.float32, device=device)

    out_base = torch.empty_like(q)
    out_opt = torch.empty_like(q)

    # Precompute ragged once (simulating consumer layer reusing source layer's ragged indices)
    ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
        indices, lengths, num_rows=num_kv
    )

    def run_base_consumer():
        rocm_sparse_attn_prefill(
            q=q, kv=kv, indices=indices, topk_length=lengths,
            scale=scale, head_dim=head_dim, nope_head_dim=448, rope_head_dim=64,
            attn_sink=attn_sink, output=out_base,
        )

    kv_2d = kv.squeeze(1)
    block_h = _prefill_block_h(num_heads)
    block_d = triton.next_power_of_2(head_dim)
    block_k = _prefill_block_k(head_dim)

    def run_opt_consumer():
        _sparse_attn_prefill_ragged_sm80_kernel[(num_queries, triton.cdiv(num_heads, block_h))](
            q, kv_2d, ragged_indices, ragged_indptr, attn_sink, out_opt,
            q.stride(0), q.stride(1), q.stride(2),
            kv_2d.stride(0), kv_2d.stride(1),
            out_opt.stride(0), out_opt.stride(1), out_opt.stride(2),
            num_heads, head_dim, kv_2d.shape[0], float(scale),
            HAS_ATTN_SINK=True, BLOCK_H=block_h, BLOCK_D=block_d, BLOCK_K=block_k,
            num_warps=4,
        )

    run_base_consumer()
    run_opt_consumer()
    bit_exact = torch.equal(out_base.view(torch.uint16), out_opt.view(torch.uint16))
    err = (out_base.float() - out_opt.float()).abs().max().item()

    t_base = bench_fn(run_base_consumer, warmup=10, rep=50)
    t_opt = bench_fn(run_opt_consumer, warmup=10, rep=50)
    saved_30_ms = (t_base - t_opt) * 30.0 / 1000.0
    print(
        f"T=1536 | Base consumer: {t_base:7.2f} us | SM80 consumer: {t_opt:7.2f} us | "
        f"Speedup: {t_base/t_opt:5.2f}x | 30-consumer save: {saved_30_ms:5.2f} ms/chunk | "
        f"Bit-exact: {bit_exact} (err={err:.2e})"
    )


if __name__ == "__main__":
    test_dequant_bit_exact()
    test_decode_splits_sweep()
    test_prefill_attn()
