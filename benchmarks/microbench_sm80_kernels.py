#!/usr/bin/env python3
"""
Microbenchmark & Numerical Parity Suite for SM80 DeepInfer Kernel Optimizations
Target Hardware: NVIDIA GA100 64GB (SM80, 74 SMs, 32 MiB L2)
"""
import time
import math
import torch
import triton
import triton.language as tl

from vllm.platforms import current_platform
from vllm.model_executor.kernels.mhc.tilelang import (
    mhc_post_tilelang,
    mhc_pre_delayed_tilelang,
    _tilelang_hc_prenorm_gemm,
    _SMALL_FMA_CONFIG,
)
from vllm.model_executor.kernels.mhc.tilelang_kernels import (
    mhc_fused_tilelang,
    mhc_post_sqrsum_tilelang,
)
from vllm.model_executor.kernels.mhc.warmup import MHC_PRE_NORM_KERNEL
from vllm.v1.attention.ops.fp8_sm80 import (
    _decode_fp8_lut,
    get_e4m3fn_bf16_lut,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    rocm_sparse_attn_decode,
    rocm_sparse_attn_prefill,
    _sparse_attn_decode_reduce_kernel,
    _decode_block_h,
    _decode_num_splits,
    _prefill_block_h,
    _prefill_block_k,
    build_ragged_indices_from_dense,
)
from vllm.models.deepseek_v4_1.amd.rocm import (
    combine_topk_swa_indices as patched_combine_topk_swa_indices,
    compute_global_topk_ragged_indices_and_indptr,
)


def bench_fn(fn, warmup=25, rep=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) * 1000.0) / rep  # microseconds


# ============================================================================
# 1. Fused mHC Post + Pre Delayed (SM80)
# ============================================================================
def mhc_fused_post_pre_delayed_tilelang(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, hc_mult, hidden_size = residual.shape
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * (hc_mult + 2)
    hc_hidden_size = hc_mult * hidden_size

    x_flat = x.view(num_tokens, hidden_size)
    post_layer_mix_flat = post_layer_mix.view(num_tokens, hc_mult)
    comb_res_mix_flat = comb_res_mix.view(num_tokens, hc_mult, hc_mult)

    use_small_fma = num_tokens <= 16
    if use_small_fma:
        tile_n, n_splits, fma_n_thr = _SMALL_FMA_CONFIG
        if (
            hc_mult3 % tile_n
            or hidden_size % n_splits
            or (hidden_size // n_splits) % fma_n_thr
        ):
            tile_n = 2 if num_tokens < 8 else 3
            n_splits = 8 if (num_tokens < 8 and hidden_size <= 4096) else 4
            fma_n_thr = 256
    else:
        n_splits = 1

    gemm_out_mul = torch.empty(
        n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=residual.device
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )
    residual_cur = torch.empty_like(residual)
    post_mix_cur = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix_cur = torch.empty(
        num_tokens, hc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input_cur = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )
    next_pre_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )

    if use_small_fma:
        mhc_fused_tilelang(
            comb_res_mix_flat,
            residual,
            post_layer_mix_flat,
            x_flat,
            fn.view(hc_mult3, hc_mult, hidden_size),
            gemm_out_mul,
            gemm_out_sqrsum,
            residual_cur,
            hc_mult,
            hidden_size,
            hc_mult3,
            fma_n_thr,
            256,
            tile_n,
            n_splits,
        )
    else:
        residual_cur_2d = residual_cur.view(num_tokens, hc_hidden_size)
        mhc_post_sqrsum_tilelang(
            comb_res_mix_flat,
            residual,
            post_layer_mix_flat,
            x_flat,
            residual_cur,
            gemm_out_sqrsum,
            hc_mult,
            hidden_size,
        )
        _tilelang_hc_prenorm_gemm(
            residual_cur_2d,
            fn,
            gemm_out_mul,
            gemm_out_sqrsum,
            hidden_size,
            hc_mult,
            sqrsum_ready=True,
        )

    MHC_PRE_NORM_KERNEL(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_cur,
        post_mix_cur,
        comb_mix_cur,
        layer_input_cur,
        norm_weight,
        pre_mix if pre_mix is not None else post_mix_cur,
        next_pre_mix,
        hidden_size=hidden_size,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        norm_eps=norm_eps,
        hc_mult=hc_mult,
        use_pre_mix_in=pre_mix is not None,
        save_pre_mix=True,
        rms_numel=hc_hidden_size,
    )
    return (
        residual_cur,
        post_mix_cur.unsqueeze(-1),
        comb_mix_cur.view(num_tokens, hc_mult, hc_mult),
        layer_input_cur,
        next_pre_mix,
    )


def test_mhc_optimizations():
    print("=" * 80)
    print("BENCHMARK 1: mHC Boundary Pair (83x per step in Target + DSpark)")
    print("=" * 80)
    device = torch.device("cuda:0")
    torch.manual_seed(42)
    hc_mult = 4
    hidden_size = 5120
    mix_size = hc_mult * (hc_mult + 2)  # 24
    input_size = hc_mult * hidden_size  # 20480

    fn = torch.randn(mix_size, input_size, dtype=torch.float32, device=device) * 0.02
    hc_scale = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
    hc_base = torch.zeros(mix_size, dtype=torch.float32, device=device)
    norm_weight = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)

    for T in [1, 6, 12, 16, 32, 512, 1536, 4096]:
        torch.cuda.empty_cache()
        x = torch.randn(T, hidden_size, dtype=torch.bfloat16, device=device)
        residual = torch.randn(T, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
        post_mix = torch.sigmoid(torch.randn(T, hc_mult, 1, dtype=torch.float32, device=device))
        res_mix = torch.softmax(torch.randn(T, hc_mult, hc_mult, dtype=torch.float32, device=device), dim=-1)
        pre_mix = torch.sigmoid(torch.randn(T, hc_mult, dtype=torch.float32, device=device))

        def run_baseline():
            res = mhc_post_tilelang(x, residual, post_mix, res_mix)
            pm, rm, li, npm = mhc_pre_delayed_tilelang(
                res, fn, hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 1.0, 4,
                pre_mix=pre_mix, norm_weight=norm_weight, norm_eps=1e-6
            )
            return res, pm, rm, li, npm

        def run_fused():
            return mhc_fused_post_pre_delayed_tilelang(
                x, residual, post_mix, res_mix, fn, hc_scale, hc_base,
                1e-6, 1e-6, 1e-6, 1.0, 4,
                pre_mix=pre_mix, norm_weight=norm_weight, norm_eps=1e-6
            )

        res_b, pm_b, rm_b, li_b, npm_b = run_baseline()
        res_f, pm_f, rm_f, li_f, npm_f = run_fused()

        err_res = (res_b.float() - res_f.float()).abs().max().item()
        err_li = (li_b.float() - li_f.float()).abs().max().item()
        err_npm = (npm_b - npm_f).abs().max().item()

        t_base = bench_fn(run_baseline, warmup=20, rep=100)
        t_fused = bench_fn(run_fused, warmup=20, rep=100)
        speedup = t_base / t_fused
        saved_83_ms = (t_base - t_fused) * 83.0 / 1000.0
        print(
            f"T={T:4d} | Base: {t_base:7.2f} us | Fused: {t_fused:7.2f} us | "
            f"Speedup: {speedup:5.2f}x | 83-boundary save: {saved_83_ms:6.2f} ms | "
            f"err(res={err_res:.1e}, li={err_li:.1e}, npm={err_npm:.1e})"
        )


# ============================================================================
# 2. Sparse MLA Decode Partial Kernel (SM80 Unified 512-dim + Bitshift Scales)
# ============================================================================
@triton.jit
def _sparse_attn_decode_partial_base_kernel(
    q_ptr,
    main_cache_ptr,
    main_indices_ptr,
    main_indptr_ptr,
    extra_cache_ptr,
    extra_indices_ptr,
    extra_indptr_ptr,
    part_m_ptr,
    part_l_ptr,
    part_acc_ptr,
    fp8_lut_ptr,
    q_stride0,
    q_stride1,
    main_cache_stride0,
    extra_cache_stride0,
    pm_stride0,
    pm_stride_s,
    pa_stride0,
    pa_stride_s,
    pa_stride_h,
    main_num_rows,
    extra_num_rows,
    main_block_size: tl.constexpr,
    extra_block_size: tl.constexpr,
    scale,
    num_heads: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    IS_FNUZ_MAIN: tl.constexpr,
    IS_FNUZ_EXTRA: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """Unpatched 2-tile (448-NoPE + 64-RoPE + exp2 scale) baseline kernel for comparison."""
    query_idx = tl.program_id(0)
    split_id = tl.program_id(1)
    pid_h = tl.program_id(2)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    nope_offsets = tl.arange(0, NOPE_BLOCK)
    nope_mask = nope_offsets < NOPE_DIM
    rope_offsets = tl.arange(0, ROPE_DIM)

    q_row_ptr = q_ptr + query_idx * q_stride0 + head_offsets[:, None] * q_stride1
    q_nope = tl.load(
        q_row_ptr + nope_offsets[None, :],
        mask=head_mask[:, None] & nope_mask[None, :],
        other=0.0,
    )
    q_rope = tl.load(
        q_row_ptr + NOPE_DIM + rope_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_nope = tl.zeros((BLOCK_H, NOPE_BLOCK), dtype=tl.float32)
    acc_rope = tl.zeros((BLOCK_H, ROPE_DIM), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)
    zero_nope = tl.zeros((BLOCK_K, NOPE_BLOCK), dtype=tl.bfloat16)
    zero_rope = tl.zeros((BLOCK_K, ROPE_DIM), dtype=tl.bfloat16)

    main_start = tl.load(main_indptr_ptr + query_idx)
    main_end = tl.load(main_indptr_ptr + query_idx + 1)
    main_len = main_end - main_start
    main_chunk = (main_len + NUM_SPLITS - 1) // NUM_SPLITS
    main_lo = split_id * main_chunk
    main_hi = tl.minimum(main_lo + main_chunk, main_len)

    for k_start in tl.range(main_lo, main_hi, BLOCK_K, num_stages=NUM_STAGES):
        k_pos = k_start + k_offsets
        in_range = k_pos < main_hi
        slot = tl.load(main_indices_ptr + main_start + k_pos, mask=in_range, other=-1)
        valid = in_range & (slot >= 0) & (slot < main_num_rows)
        safe_slot = tl.where(valid, slot, 0)

        block_idx = safe_slot // main_block_size
        pos_in_block = safe_slot % main_block_size
        cache_block_ptr = main_cache_ptr + block_idx.to(tl.int64) * main_cache_stride0
        token_data_ptr = cache_block_ptr + pos_in_block * 576
        token_scale_ptr = cache_block_ptr + main_block_size * 576 + pos_in_block * 8

        x_uint8 = tl.load(
            token_data_ptr[:, None] + nope_offsets[None, :],
            mask=valid[:, None] & nope_mask[None, :],
            other=0,
        )
        k_vals = _decode_fp8_lut(x_uint8, IS_FNUZ_MAIN, fp8_lut_ptr)
        encoded_scales = tl.load(
            token_scale_ptr[:, None] + nope_offsets[None, :] // 64,
            mask=valid[:, None] & nope_mask[None, :],
            other=127,
        )
        scales = tl.exp2(encoded_scales.to(tl.float32) - 127.0)
        k_nope = k_vals.to(tl.bfloat16) * scales.to(tl.bfloat16)
        k_nope = tl.where(valid[:, None] & nope_mask[None, :], k_nope, zero_nope)
        k_nope = tl.where(k_nope == k_nope, k_nope, zero_nope)

        rope_ptr = (token_data_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
        k_rope = tl.load(
            rope_ptr[:, None] + rope_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        k_rope = tl.where(valid[:, None], k_rope, zero_rope)
        k_rope = tl.where(k_rope == k_rope, k_rope, zero_rope)

        scores = (tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(q_rope, tl.trans(k_rope))) * scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc_nope = acc_nope * alpha[:, None] + tl.dot(p.to(k_nope.dtype), k_nope)
        acc_rope = acc_rope * alpha[:, None] + tl.dot(p.to(k_rope.dtype), k_rope)
        m_i = m_new
        l_i = l_new

    if HAS_EXTRA:
        extra_start = tl.load(extra_indptr_ptr + query_idx)
        extra_end = tl.load(extra_indptr_ptr + query_idx + 1)
        extra_len = extra_end - extra_start
        extra_chunk = (extra_len + NUM_SPLITS - 1) // NUM_SPLITS
        extra_lo = split_id * extra_chunk
        extra_hi = tl.minimum(extra_lo + extra_chunk, extra_len)

        for k_start in tl.range(extra_lo, extra_hi, BLOCK_K, num_stages=NUM_STAGES):
            k_pos = k_start + k_offsets
            in_range = k_pos < extra_hi
            slot = tl.load(
                extra_indices_ptr + extra_start + k_pos, mask=in_range, other=-1
            )
            valid = in_range & (slot >= 0) & (slot < extra_num_rows)
            safe_slot = tl.where(valid, slot, 0)

            block_idx = safe_slot // extra_block_size
            pos_in_block = safe_slot % extra_block_size
            cache_block_ptr = (
                extra_cache_ptr + block_idx.to(tl.int64) * extra_cache_stride0
            )
            token_data_ptr = cache_block_ptr + pos_in_block * 576
            token_scale_ptr = (
                cache_block_ptr + extra_block_size * 576 + pos_in_block * 8
            )

            x_uint8 = tl.load(
                token_data_ptr[:, None] + nope_offsets[None, :],
                mask=valid[:, None] & nope_mask[None, :],
                other=0,
            )
            k_vals = _decode_fp8_lut(x_uint8, IS_FNUZ_EXTRA, fp8_lut_ptr)
            encoded_scales = tl.load(
                token_scale_ptr[:, None] + nope_offsets[None, :] // 64,
                mask=valid[:, None] & nope_mask[None, :],
                other=127,
            )
            scales = tl.exp2(encoded_scales.to(tl.float32) - 127.0)
            k_nope = k_vals.to(tl.bfloat16) * scales.to(tl.bfloat16)
            k_nope = tl.where(valid[:, None] & nope_mask[None, :], k_nope, zero_nope)
            k_nope = tl.where(k_nope == k_nope, k_nope, zero_nope)

            rope_ptr = (token_data_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
            k_rope = tl.load(
                rope_ptr[:, None] + rope_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            k_rope = tl.where(valid[:, None], k_rope, zero_rope)
            k_rope = tl.where(k_rope == k_rope, k_rope, zero_rope)

            scores = (tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(q_rope, tl.trans(k_rope))) * scale
            scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

            m_block = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_block)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
            l_new = l_i * alpha + tl.sum(p, axis=1)

            acc_nope = acc_nope * alpha[:, None] + tl.dot(p.to(k_nope.dtype), k_nope)
            acc_rope = acc_rope * alpha[:, None] + tl.dot(p.to(k_rope.dtype), k_rope)
            m_i = m_new
            l_i = l_new

    pm_base = query_idx * pm_stride0 + split_id * pm_stride_s + head_offsets
    tl.store(part_m_ptr + pm_base, m_i, mask=head_mask)
    tl.store(part_l_ptr + pm_base, l_i, mask=head_mask)
    acc_base = (
        part_acc_ptr
        + query_idx * pa_stride0
        + split_id * pa_stride_s
        + head_offsets[:, None] * pa_stride_h
    )
    tl.store(
        acc_base + nope_offsets[None, :],
        acc_nope,
        mask=head_mask[:, None] & nope_mask[None, :],
    )
    tl.store(
        acc_base + NOPE_DIM + rope_offsets[None, :],
        acc_rope,
        mask=head_mask[:, None],
    )


@triton.jit
def _decode_e8m0_scales_bf16_triton(encoded_scales):
    # Bit-exact IEEE-754 bfloat16 construction for 2^(encoded_scales - 127)
    scale_u16 = tl.where(
        encoded_scales == 0,
        0x0040,
        encoded_scales.to(tl.uint16) << 7,
    )
    return scale_u16.to(tl.bfloat16, bitcast=True)


@triton.jit
def _sparse_attn_decode_partial_sm80_kernel(
    q_ptr,
    main_cache_ptr,
    main_indices_ptr,
    main_indptr_ptr,
    extra_cache_ptr,
    extra_indices_ptr,
    extra_indptr_ptr,
    part_m_ptr,
    part_l_ptr,
    part_acc_ptr,
    fp8_lut_ptr,
    q_stride0,
    q_stride1,
    main_cache_stride0,
    extra_cache_stride0,
    pm_stride0,
    pm_stride_s,
    pa_stride0,
    pa_stride_s,
    pa_stride_h,
    main_num_rows,
    extra_num_rows,
    main_block_size: tl.constexpr,
    extra_block_size: tl.constexpr,
    scale,
    num_heads: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    COMB_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    IS_FNUZ_MAIN: tl.constexpr,
    IS_FNUZ_EXTRA: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    query_idx = tl.program_id(0)
    split_id = tl.program_id(1)
    pid_h = tl.program_id(2)

    tl.static_assert(NOPE_DIM == 448)
    tl.static_assert(ROPE_DIM == 64)
    tl.static_assert(COMB_DIM == 512)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    comb_offsets = tl.arange(0, COMB_DIM)
    nope_mask = comb_offsets < NOPE_DIM

    q_row_ptr = q_ptr + query_idx * q_stride0 + head_offsets[:, None] * q_stride1
    q_combined = tl.load(
        q_row_ptr + comb_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, COMB_DIM), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)
    scale_offsets = tl.arange(0, 8)
    zero_comb = tl.zeros((BLOCK_K, COMB_DIM), dtype=tl.bfloat16)

    main_start = tl.load(main_indptr_ptr + query_idx)
    main_end = tl.load(main_indptr_ptr + query_idx + 1)
    main_len = main_end - main_start
    main_chunk = (main_len + NUM_SPLITS - 1) // NUM_SPLITS
    main_lo = split_id * main_chunk
    main_hi = tl.minimum(main_lo + main_chunk, main_len)

    for k_start in tl.range(main_lo, main_hi, BLOCK_K, num_stages=NUM_STAGES):
        k_pos = k_start + k_offsets
        in_range = k_pos < main_hi
        slot = tl.load(main_indices_ptr + main_start + k_pos, mask=in_range, other=-1)
        valid = in_range & (slot >= 0) & (slot < main_num_rows)
        safe_slot = tl.where(valid, slot, 0)

        block_idx = safe_slot // main_block_size
        pos_in_block = safe_slot % main_block_size
        cache_block_ptr = main_cache_ptr + block_idx.to(tl.int64) * main_cache_stride0
        token_data_ptr = cache_block_ptr + pos_in_block * 576
        token_scale_ptr = cache_block_ptr + main_block_size * 576 + pos_in_block * 8

        x_uint8 = tl.load(
            token_data_ptr[:, None] + comb_offsets[None, :],
            mask=valid[:, None] & nope_mask[None, :],
            other=0,
        )
        k_vals = _decode_fp8_lut(x_uint8, IS_FNUZ_MAIN, fp8_lut_ptr)
        encoded_scales = tl.load(
            token_scale_ptr[:, None] + scale_offsets[None, :],
            mask=valid[:, None],
            other=127,
        )
        scales = _decode_e8m0_scales_bf16_triton(encoded_scales)
        scales = tl.reshape(
            tl.broadcast_to(scales[:, :, None], (BLOCK_K, 8, 64)),
            (BLOCK_K, COMB_DIM),
        )
        k_nope = k_vals.to(tl.bfloat16) * scales

        rope_ptr = (token_data_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
        k_rope_full = tl.load(
            rope_ptr[:, None] + (comb_offsets[None, :] - NOPE_DIM),
            mask=valid[:, None] & (~nope_mask[None, :]),
            other=0.0,
        )
        k_combined = tl.where(nope_mask[None, :], k_nope, k_rope_full)
        k_combined = tl.where(valid[:, None], k_combined, zero_comb)
        k_combined = tl.where(k_combined == k_combined, k_combined, zero_comb)

        scores = tl.dot(q_combined, tl.trans(k_combined)) * scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), k_combined)
        m_i = m_new
        l_i = l_new

    if HAS_EXTRA:
        extra_start = tl.load(extra_indptr_ptr + query_idx)
        extra_end = tl.load(extra_indptr_ptr + query_idx + 1)
        extra_len = extra_end - extra_start
        extra_chunk = (extra_len + NUM_SPLITS - 1) // NUM_SPLITS
        extra_lo = split_id * extra_chunk
        extra_hi = tl.minimum(extra_lo + extra_chunk, extra_len)

        for k_start in tl.range(extra_lo, extra_hi, BLOCK_K, num_stages=NUM_STAGES):
            k_pos = k_start + k_offsets
            in_range = k_pos < extra_hi
            slot = tl.load(
                extra_indices_ptr + extra_start + k_pos, mask=in_range, other=-1
            )
            valid = in_range & (slot >= 0) & (slot < extra_num_rows)
            safe_slot = tl.where(valid, slot, 0)

            block_idx = safe_slot // extra_block_size
            pos_in_block = safe_slot % extra_block_size
            cache_block_ptr = (
                extra_cache_ptr + block_idx.to(tl.int64) * extra_cache_stride0
            )
            token_data_ptr = cache_block_ptr + pos_in_block * 576
            token_scale_ptr = (
                cache_block_ptr + extra_block_size * 576 + pos_in_block * 8
            )

            x_uint8 = tl.load(
                token_data_ptr[:, None] + comb_offsets[None, :],
                mask=valid[:, None] & nope_mask[None, :],
                other=0,
            )
            k_vals = _decode_fp8_lut(x_uint8, IS_FNUZ_EXTRA, fp8_lut_ptr)
            encoded_scales = tl.load(
                token_scale_ptr[:, None] + scale_offsets[None, :],
                mask=valid[:, None],
                other=127,
            )
            scales = _decode_e8m0_scales_bf16_triton(encoded_scales)
            scales = tl.reshape(
                tl.broadcast_to(scales[:, :, None], (BLOCK_K, 8, 64)),
                (BLOCK_K, COMB_DIM),
            )
            k_nope = k_vals.to(tl.bfloat16) * scales

            rope_ptr = (token_data_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
            k_rope_full = tl.load(
                rope_ptr[:, None] + (comb_offsets[None, :] - NOPE_DIM),
                mask=valid[:, None] & (~nope_mask[None, :]),
                other=0.0,
            )
            k_combined = tl.where(nope_mask[None, :], k_nope, k_rope_full)
            k_combined = tl.where(valid[:, None], k_combined, zero_comb)
            k_combined = tl.where(k_combined == k_combined, k_combined, zero_comb)

            scores = tl.dot(q_combined, tl.trans(k_combined)) * scale
            scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

            m_block = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_block)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
            l_new = l_i * alpha + tl.sum(p, axis=1)

            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), k_combined)
            m_i = m_new
            l_i = l_new

    pm_base = query_idx * pm_stride0 + split_id * pm_stride_s + head_offsets
    tl.store(part_m_ptr + pm_base, m_i, mask=head_mask)
    tl.store(part_l_ptr + pm_base, l_i, mask=head_mask)
    acc_base = (
        part_acc_ptr
        + query_idx * pa_stride0
        + split_id * pa_stride_s
        + head_offsets[:, None] * pa_stride_h
    )
    tl.store(
        acc_base + comb_offsets[None, :],
        acc,
        mask=head_mask[:, None],
    )


def test_sparse_mla_decode():
    print("\n" + "=" * 80)
    print("BENCHMARK 2: Sparse MLA Decode (41x per decode step in Target + DSpark)")
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
    # Allocate realistic uint8 fp8_ds_mla cache (block_size*576B data + block_size*8B scales per block)
    raw_swa = torch.zeros((num_blocks, main_block_size * 584), dtype=torch.uint8, device=device)
    swa_data = raw_swa[:, : main_block_size * 576].view(num_blocks, main_block_size, 576)
    swa_data[:, :, :448] = torch.randint(0, 120, (num_blocks, main_block_size, 448), dtype=torch.uint8, device=device)
    swa_data[:, :, 448:] = torch.randn(num_blocks, main_block_size, rope_dim, dtype=torch.bfloat16, device=device).view(torch.uint8)
    raw_swa[:, main_block_size * 576 :] = torch.randint(117, 121, (num_blocks, main_block_size * 8), dtype=torch.uint8, device=device)
    swa_k_cache = raw_swa.view(num_blocks, main_block_size, 584)

    raw_kv = torch.zeros((num_blocks, extra_block_size * 584), dtype=torch.uint8, device=device)
    kv_data = raw_kv[:, : extra_block_size * 576].view(num_blocks, extra_block_size, 576)
    kv_data[:, :, :448] = torch.randint(0, 120, (num_blocks, extra_block_size, 448), dtype=torch.uint8, device=device)
    kv_data[:, :, 448:] = torch.randn(num_blocks, extra_block_size, rope_dim, dtype=torch.bfloat16, device=device).view(torch.uint8)
    raw_kv[:, extra_block_size * 576 :] = torch.randint(117, 121, (num_blocks, extra_block_size * 8), dtype=torch.uint8, device=device)
    kv_cache = raw_kv.view(num_blocks, extra_block_size, 584)

    attn_sink = torch.zeros(num_heads, dtype=torch.float32, device=device)
    fp8_lut = get_e4m3fn_bf16_lut(device)

    for B in [1, 6, 12, 24]:
        q = torch.randn(B, num_heads, head_dim, dtype=torch.bfloat16, device=device)
        out_base = torch.empty_like(q)
        out_opt = torch.empty_like(q)

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

        block_h = _decode_block_h(num_heads)
        heads_blocks = triton.cdiv(num_heads, block_h)
        block_k = 32
        num_splits = _decode_num_splits(B, heads_blocks, float(swa_len), float(topk_len), block_k)

        def run_base():
            part_m = torch.empty((B, num_splits, num_heads), dtype=torch.float32, device=device)
            part_l = torch.empty_like(part_m)
            part_acc = torch.empty((B, num_splits, num_heads, head_dim), dtype=torch.float32, device=device)
            tmp_out = torch.empty_like(q)
            _sparse_attn_decode_partial_base_kernel[(B, num_splits, heads_blocks)](
                q,
                swa_k_cache,
                swa_ragged_indices,
                swa_ragged_indptr,
                kv_cache,
                topk_ragged_indices,
                topk_ragged_indptr,
                part_m,
                part_l,
                part_acc,
                fp8_lut,
                q.stride(0),
                q.stride(1),
                swa_k_cache.stride(0),
                kv_cache.stride(0),
                part_m.stride(0),
                part_m.stride(1),
                part_acc.stride(0),
                part_acc.stride(1),
                part_acc.stride(2),
                swa_k_cache.shape[0] * swa_k_cache.shape[1],
                kv_cache.shape[0] * kv_cache.shape[1],
                swa_k_cache.shape[1],
                kv_cache.shape[1],
                scale,
                num_heads,
                HAS_EXTRA=True,
                NOPE_DIM=nope_dim,
                NOPE_BLOCK=head_dim,
                ROPE_DIM=rope_dim,
                IS_FNUZ_MAIN=False,
                IS_FNUZ_EXTRA=False,
                BLOCK_H=block_h,
                BLOCK_K=block_k,
                NUM_SPLITS=num_splits,
                NUM_STAGES=1,
                num_warps=8,
            )
            _sparse_attn_decode_reduce_kernel[(B, num_heads)](
                part_m,
                part_l,
                part_acc,
                attn_sink,
                tmp_out,
                tmp_out.stride(0),
                tmp_out.stride(1),
                part_m.stride(0),
                part_m.stride(1),
                part_acc.stride(0),
                part_acc.stride(1),
                part_acc.stride(2),
                num_heads,
                HAS_ATTN_SINK=True,
                ADAPTIVE_SPLITS=False,
                COMB_DIM=head_dim,
                BLOCK_H=1,
                NUM_SPLITS=num_splits,
                SPLITS_PAD=triton.next_power_of_2(num_splits),
                num_warps=4,
            )
            out_base.copy_(tmp_out)

        def make_opt_runner(n_warps):
            def _run():
                part_m = torch.empty((B, num_splits, num_heads), dtype=torch.float32, device=device)
                part_l = torch.empty_like(part_m)
                part_acc = torch.empty((B, num_splits, num_heads, head_dim), dtype=torch.float32, device=device)
                _sparse_attn_decode_partial_sm80_kernel[(B, num_splits, heads_blocks)](
                    q,
                    swa_k_cache,
                    swa_ragged_indices,
                    swa_ragged_indptr,
                    kv_cache,
                    topk_ragged_indices,
                    topk_ragged_indptr,
                    part_m,
                    part_l,
                    part_acc,
                    fp8_lut,
                    q.stride(0),
                    q.stride(1),
                    swa_k_cache.stride(0),
                    kv_cache.stride(0),
                    part_m.stride(0),
                    part_m.stride(1),
                    part_acc.stride(0),
                    part_acc.stride(1),
                    part_acc.stride(2),
                    swa_k_cache.shape[0] * swa_k_cache.shape[1],
                    kv_cache.shape[0] * kv_cache.shape[1],
                    swa_k_cache.shape[1],
                    kv_cache.shape[1],
                    scale,
                    num_heads,
                    HAS_EXTRA=True,
                    NOPE_DIM=nope_dim,
                    COMB_DIM=head_dim,
                    ROPE_DIM=rope_dim,
                    IS_FNUZ_MAIN=False,
                    IS_FNUZ_EXTRA=False,
                    BLOCK_H=block_h,
                    BLOCK_K=block_k,
                    NUM_SPLITS=num_splits,
                    NUM_STAGES=1,
                    num_warps=n_warps,
                )
                _sparse_attn_decode_reduce_kernel[(B, num_heads)](
                    part_m,
                    part_l,
                    part_acc,
                    attn_sink,
                    out_opt,
                    out_opt.stride(0),
                    out_opt.stride(1),
                    part_m.stride(0),
                    part_m.stride(1),
                    part_acc.stride(0),
                    part_acc.stride(1),
                    part_acc.stride(2),
                    num_heads,
                    HAS_ATTN_SINK=True,
                    ADAPTIVE_SPLITS=False,
                    COMB_DIM=head_dim,
                    BLOCK_H=1,
                    NUM_SPLITS=num_splits,
                    SPLITS_PAD=triton.next_power_of_2(num_splits),
                    num_warps=4,
                )
            return _run

        run_base()
        run_opt_w8 = make_opt_runner(8)
        run_opt_w4 = make_opt_runner(4)
        run_opt_w8()
        err_w8 = (out_base.float() - out_opt.float()).abs().max().item()
        run_opt_w4()
        err_w4 = (out_base.float() - out_opt.float()).abs().max().item()

        t_base = bench_fn(run_base, warmup=20, rep=100)
        t_w8 = bench_fn(run_opt_w8, warmup=20, rep=100)
        t_w4 = bench_fn(run_opt_w4, warmup=20, rep=100)
        print(
            f"B={B:2d} (splits={num_splits:2d}) | Base: {t_base:7.2f} us | "
            f"SM80-w8: {t_w8:7.2f} us ({t_base/t_w8:.2f}x, err={err_w8:.1e}) | "
            f"SM80-w4: {t_w4:7.2f} us ({t_base/t_w4:.2f}x, err={err_w4:.1e})"
        )


# ============================================================================
# 3. Fused Triton combine_topk_swa_indices vs Torch 12-op chain
# ============================================================================
@triton.jit
def _combine_topk_swa_indices_sm80_kernel(
    topk_indices_ptr,
    topk_stride0,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    combined_indices_ptr,
    combined_stride0,
    combined_lens_ptr,
    num_reqs,
    window_size: tl.constexpr,
    compress_ratio: tl.constexpr,
    topk: tl.constexpr,
    logical_topk_width: tl.constexpr,
    M,
    N,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_SWA: tl.constexpr,
):
    token_idx = tl.program_id(0)
    q_base = tl.load(query_start_loc_ptr)

    # Binary search or linear scan over num_reqs to find req_id for token_idx
    req_id = 0
    q_start = 0
    q_end = 0
    for r in range(num_reqs):
        s = tl.load(query_start_loc_ptr + r) - q_base
        e = tl.load(query_start_loc_ptr + r + 1) - q_base
        if token_idx >= s and token_idx < e:
            req_id = r
            q_start = s
            q_end = e

    q_len = q_end - q_start
    seq_len = tl.load(seq_lens_ptr + req_id)
    gather_len = tl.load(gather_lens_ptr + req_id)
    pos = seq_len - q_len + (token_idx - q_start)

    topk_len = tl.minimum((pos + 1) // compress_ratio, logical_topk_width)
    topk_len = tl.maximum(topk_len, 0)

    swa_len = tl.minimum(pos + 1, window_size)
    swa_len = tl.maximum(swa_len, 0)

    tl.store(combined_lens_ptr + token_idx, topk_len + swa_len)

    req_base = M * req_id
    out_row_ptr = combined_indices_ptr + token_idx * combined_stride0
    in_row_ptr = topk_indices_ptr + token_idx * topk_stride0

    # Write topk indices [0 .. topk_len)
    for off in range(0, logical_topk_width, BLOCK_TOPK):
        cols = off + tl.arange(0, BLOCK_TOPK)
        mask = cols < topk_len
        val = tl.load(in_row_ptr + cols, mask=cols < logical_topk_width, other=-1).to(tl.int32)
        valid = mask & (val >= 0) & (val < N)
        out_val = tl.where(valid, val + req_base, -1)
        tl.store(out_row_ptr + cols, out_val, mask=cols < logical_topk_width)

    # Write swa indices at [topk_len .. topk_len + swa_len)
    gather_start = seq_len - gather_len
    swa_base_val = req_base + N + pos - swa_len + 1 - gather_start
    for off in range(0, window_size, BLOCK_SWA):
        swa_offs = off + tl.arange(0, BLOCK_SWA)
        swa_mask = swa_offs < swa_len
        dst_cols = topk_len + swa_offs
        swa_vals = swa_base_val + swa_offs
        tl.store(out_row_ptr + dst_cols, swa_vals, mask=swa_mask)


def triton_combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    num_tokens = topk_indices.shape[0]
    alignment = 64
    combined_topk = ((topk + window_size + alignment - 1) // alignment) * alignment
    combined_indices = torch.full(
        (num_tokens, combined_topk),
        fill_value=-1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    combined_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )
    if num_tokens == 0:
        return combined_indices, combined_lens
    logical_topk_width = min(topk, topk_indices.shape[1])
    _combine_topk_swa_indices_sm80_kernel[(num_tokens,)](
        topk_indices,
        topk_indices.stride(0),
        query_start_loc,
        seq_lens,
        gather_lens,
        combined_indices,
        combined_indices.stride(0),
        combined_lens,
        seq_lens.shape[0],
        window_size=window_size,
        compress_ratio=compress_ratio,
        topk=topk,
        logical_topk_width=logical_topk_width,
        M=M,
        N=N,
        BLOCK_TOPK=256,
        BLOCK_SWA=256,
        num_warps=4,
    )
    return combined_indices, combined_lens


def torch_combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpatched baseline 12-op PyTorch implementation for speedup & parity comparison."""
    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    num_tokens = topk_indices.shape[0]
    alignment = 128
    combined_topk = ((topk + window_size + alignment - 1) // alignment) * alignment
    combined_indices = torch.full(
        (num_tokens, combined_topk),
        fill_value=-1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    combined_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    req_ids = torch.repeat_interleave(
        torch.arange(seq_lens.shape[0], device=seq_lens.device), query_lens
    )
    query_starts = query_start_loc[:-1] - query_start_loc[0]
    token_offsets = torch.arange(num_tokens, device=seq_lens.device) - (
        torch.repeat_interleave(query_starts, query_lens)
    )
    positions = seq_lens[req_ids] - query_lens[req_ids] + token_offsets

    logical_topk_width = min(topk, topk_indices.shape[1])
    topk_lens = torch.minimum(
        (positions + 1) // compress_ratio,
        torch.full_like(positions, logical_topk_width),
    ).clamp_min(0)
    topk_offsets = torch.arange(logical_topk_width, device=seq_lens.device)
    topk_mask = topk_offsets[None, :] < topk_lens[:, None]
    topk_values = topk_indices[:, :logical_topk_width].to(torch.int32)
    topk_valid = topk_mask & (topk_values >= 0) & (topk_values < N)
    combined_indices[:, :logical_topk_width] = torch.where(
        topk_valid,
        topk_values + (M * req_ids).to(torch.int32)[:, None],
        -1,
    )

    swa_lens = torch.minimum(
        positions + 1, torch.full_like(positions, window_size)
    ).clamp_min(0)
    swa_offsets = torch.arange(window_size, device=seq_lens.device)
    swa_mask = swa_offsets[None, :] < swa_lens[:, None]
    swa_columns = topk_lens[:, None] + swa_offsets[None, :]
    gather_starts = seq_lens - gather_lens
    swa_values = (
        M * req_ids[:, None]
        + N
        + swa_offsets[None, :]
        + positions[:, None]
        - swa_lens[:, None]
        + 1
        - gather_starts[req_ids, None]
    ).to(torch.int32)
    rows = torch.arange(num_tokens, device=seq_lens.device)[:, None].expand_as(
        swa_columns
    )
    flat_dst = rows[swa_mask] * combined_topk + swa_columns[swa_mask]
    combined_indices.view(-1).index_copy_(0, flat_dst, swa_values[swa_mask])
    combined_lens.copy_((topk_lens + swa_lens).to(torch.int32))
    return combined_indices, combined_lens


def test_combine_topk_swa():
    print("\n" + "=" * 80)
    print("BENCHMARK 3: combine_topk_swa_indices (30x per prefill chunk)")
    print("=" * 80)
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    for num_tokens in [512, 1536, 4096]:
        topk = 1024
        window_size = 512
        compress_ratio = 4
        N = 4096
        M = N + window_size + num_tokens
        topk_indices = torch.randint(0, N, (num_tokens, topk), dtype=torch.int32, device=device)
        query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
        seq_lens = torch.tensor([8192 + num_tokens], dtype=torch.int32, device=device)
        gather_lens = torch.tensor([window_size + num_tokens], dtype=torch.int32, device=device)

        idx_ref, len_ref = torch_combine_topk_swa_indices(
            topk_indices, query_start_loc, seq_lens, gather_lens,
            window_size, compress_ratio, topk, M, N
        )
        idx_tri, len_tri = patched_combine_topk_swa_indices(
            topk_indices, query_start_loc, seq_lens, gather_lens,
            window_size, compress_ratio, topk, M, N
        )

        exact_idx = torch.equal(idx_ref, idx_tri)
        exact_len = torch.equal(len_ref, len_tri)

        t_ref = bench_fn(
            lambda: torch_combine_topk_swa_indices(
                topk_indices, query_start_loc, seq_lens, gather_lens,
                window_size, compress_ratio, topk, M, N
            ),
            warmup=20, rep=100,
        )
        t_tri = bench_fn(
            lambda: patched_combine_topk_swa_indices(
                topk_indices, query_start_loc, seq_lens, gather_lens,
                window_size, compress_ratio, topk, M, N
            ),
            warmup=20, rep=100,
        )
        print(
            f"T={num_tokens:4d} | Torch (12 ops): {t_ref:7.2f} us | "
            f"Triton (1 op): {t_tri:7.2f} us | Speedup: {t_ref/t_tri:5.2f}x | "
            f"Bit-exact: {exact_idx and exact_len}"
        )


if __name__ == "__main__":
    test_mhc_optimizations()
    test_sparse_mla_decode()
    test_combine_topk_swa()
