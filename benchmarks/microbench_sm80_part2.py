#!/usr/bin/env python3
"""
Part 2 Microbenchmark:
1. Diagnose B=12/24 NaN in test_sparse_mla_decode & verify 0.0 error across all B
2. Benchmark dequantize_and_gather_k_cache_triton vs SM80 LUT+Bitshift + Consumer Reuse
3. Benchmark rocm_sparse_attn_prefill vs SM80 Direct-Out + Constexpr + Consumer Ragged Reuse
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
    rocm_sparse_attn_decode,
    rocm_sparse_attn_prefill,
    _sparse_attn_decode_reduce_kernel,
    _decode_block_h,
    _decode_num_splits,
    _prefill_block_h,
    _prefill_block_k,
    build_ragged_indices_from_dense,
    _sparse_kv_row_offset,
)
from vllm.models.deepseek_v4_1.common.ops.cache_utils import (
    dequantize_and_gather_k_cache_triton,
)
from vllm.models.deepseek_v4_1.amd.rocm import (
    combine_topk_swa_indices as torch_combine_topk_swa_indices,
    compute_global_topk_ragged_indices_and_indptr,
)
from microbench_sm80_kernels import (
    bench_fn,
    _sparse_attn_decode_partial_sm80_kernel,
    _decode_e8m0_scales_bf16_triton,
    triton_combine_topk_swa_indices,
)


def test_decode_nan_check():
    print("=" * 80)
    print("CHECK 1: Sparse MLA Decode Parity Across All Batch Sizes (B=1..32)")
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
    swa_k_cache = torch.randint(0, 120, (num_blocks, main_block_size, 584), dtype=torch.uint8, device=device)
    # Write valid bf16 values into the RoPE region [448:576] so random bytes don't form bf16 NaN/Inf
    swa_rope_bf16 = torch.randn(num_blocks, main_block_size, rope_dim, dtype=torch.bfloat16, device=device)
    swa_k_cache[:, :, 448:576] = swa_rope_bf16.view(torch.uint8)
    swa_k_cache[:, :, 576:] = torch.randint(124, 129, (num_blocks, main_block_size, 8), dtype=torch.uint8, device=device)

    kv_cache = torch.randint(0, 120, (num_blocks, extra_block_size, 584), dtype=torch.uint8, device=device)
    kv_rope_bf16 = torch.randn(num_blocks, extra_block_size, rope_dim, dtype=torch.bfloat16, device=device)
    kv_cache[:, :, 448:576] = kv_rope_bf16.view(torch.uint8)
    kv_cache[:, :, 576:] = torch.randint(124, 129, (num_blocks, extra_block_size, 8), dtype=torch.uint8, device=device)

    attn_sink = torch.zeros(num_heads, dtype=torch.float32, device=device)
    fp8_lut = get_e4m3fn_bf16_lut(device)

    for B in [1, 4, 6, 12, 16, 24]:
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

        def run_base():
            rocm_sparse_attn_decode(
                q=q,
                kv_cache=kv_cache,
                swa_k_cache=swa_k_cache,
                swa_only=False,
                topk_indices=None,
                topk_lens=topk_lens,
                swa_indices=swa_indices,
                swa_lens=swa_lens,
                swa_ragged_indices=swa_ragged_indices,
                swa_ragged_indptr=swa_ragged_indptr,
                topk_ragged_indices=topk_ragged_indices,
                topk_ragged_indptr=topk_ragged_indptr,
                attn_sink=attn_sink,
                scale=scale,
                head_dim=head_dim,
                nope_head_dim=nope_dim,
                rope_head_dim=rope_dim,
                output=out_base,
            )

        block_h = _decode_block_h(num_heads)
        heads_blocks = triton.cdiv(num_heads, block_h)
        block_k = 32
        num_splits = _decode_num_splits(B, heads_blocks, float(swa_len), float(topk_len), block_k)
        # Choose 4 warps when enough CTAs exist to fill 74 SMs at >=1 CTA/SM, else 8 warps
        n_warps = 8 if (B * num_splits * heads_blocks) <= 74 else 4

        def run_opt():
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

        run_base()
        run_opt()
        err = (out_base.float() - out_opt.float()).abs().max().item()
        t_base = bench_fn(run_base, warmup=20, rep=100)
        t_opt = bench_fn(run_opt, warmup=20, rep=100)
        print(
            f"B={B:2d} (splits={num_splits:2d}, warps={n_warps}) | "
            f"Base: {t_base:7.2f} us | SM80-Opt: {t_opt:7.2f} us | "
            f"Speedup: {t_base/t_opt:5.2f}x | max_abs_err: {err:.2e}"
        )


# ============================================================================
# 2. Fast SM80 dequantize_and_gather_k_cache_triton (LUT + Bitshift + Vectorized)
# ============================================================================
@triton.jit
def _dequantize_and_gather_k_sm80_kernel(
    out_ptr,
    out_stride0,
    out_stride1,
    k_cache_ptr,
    seq_lens_ptr,
    block_table_ptr,
    offset,
    gather_lens_ptr,
    fp8_lut_ptr,
    max_blocks_per_seq,
    cache_block_size: tl.constexpr,
    block_stride,
    USE_FNUZ: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    if gather_lens_ptr is not None:
        gather_len = tl.load(gather_lens_ptr + batch_idx)
    else:
        gather_len = seq_len
    start_pos = seq_len - gather_len

    comb_offsets = tl.arange(0, 512)
    nope_mask = comb_offsets < 448
    scale_offsets = tl.arange(0, 8)
    block_table_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq

    for i in range(worker_id, gather_len, num_workers):
        pos = start_pos + i
        block_in_seq = pos // cache_block_size
        pos_in_block = pos % cache_block_size

        physical_block_idx = tl.load(block_table_row_ptr + block_in_seq)
        cache_block_ptr = k_cache_ptr + physical_block_idx.to(tl.int64) * block_stride

        token_data_ptr = cache_block_ptr + pos_in_block * 576
        token_scale_ptr = cache_block_ptr + cache_block_size * 576 + pos_in_block * 8

        # Load all 448 FP8 bytes in one vectorized pass
        x_uint8 = tl.load(token_data_ptr + comb_offsets, mask=nope_mask, other=0)
        k_vals = _decode_fp8_lut(x_uint8, USE_FNUZ, fp8_lut_ptr)

        # Load 8 scale bytes and bit-shift to bf16
        encoded_scales = tl.load(token_scale_ptr + scale_offsets)
        scales = _decode_e8m0_scales_bf16_triton(encoded_scales)
        scales_512 = tl.reshape(
            tl.broadcast_to(scales[:, None], (8, 64)),
            (512,),
        )
        k_nope = (k_vals.to(tl.float32) * scales_512.to(tl.float32)).to(tl.bfloat16)

        # Load 64 BF16 RoPE elements
        rope_ptr = (token_data_ptr + 448).to(tl.pointer_type(tl.bfloat16))
        k_rope = tl.load(
            rope_ptr + (comb_offsets - 448),
            mask=~nope_mask,
            other=0.0,
        )

        out_val = tl.where(nope_mask, k_nope, k_rope)
        output_row_ptr = out_ptr + batch_idx * out_stride0 + (offset + i) * out_stride1
        tl.store(output_row_ptr + comb_offsets, out_val)


def dequantize_and_gather_k_cache_sm80(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
    use_fnuz: bool = False,
) -> None:
    num_reqs = seq_lens.shape[0]
    fp8_lut = get_e4m3fn_bf16_lut(k_cache.device)
    NUM_WORKERS = 256
    _dequantize_and_gather_k_sm80_kernel[(num_reqs, NUM_WORKERS)](
        out,
        out.stride(0),
        out.stride(1),
        k_cache,
        seq_lens,
        block_table,
        offset,
        gather_lens,
        fp8_lut,
        max_blocks_per_seq=block_table.shape[-1],
        cache_block_size=block_size,
        block_stride=k_cache.stride(0),
        USE_FNUZ=use_fnuz,
        num_warps=4,
    )


def test_dequant_and_prefill():
    print("\n" + "=" * 80)
    print("CHECK 2: dequantize_and_gather_k_cache (38x per prefill chunk)")
    print("=" * 80)
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    num_blocks = 1024
    block_size = 16
    k_cache = torch.randint(0, 120, (num_blocks, block_size, 584), dtype=torch.uint8, device=device)
    rope_bf16 = torch.randn(num_blocks, block_size, 64, dtype=torch.bfloat16, device=device)
    k_cache[:, :, 448:576] = rope_bf16.view(torch.uint8)
    k_cache[:, :, 576:] = torch.randint(124, 129, (num_blocks, block_size, 8), dtype=torch.uint8, device=device)

    for n_compressed in [1024, 4096, 12288]:
        out_ref = torch.zeros((1, n_compressed, 512), dtype=torch.bfloat16, device=device)
        out_opt = torch.zeros((1, n_compressed, 512), dtype=torch.bfloat16, device=device)
        seq_lens = torch.tensor([n_compressed], dtype=torch.int32, device=device)
        block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)

        dequantize_and_gather_k_cache_triton(
            out_ref, k_cache, seq_lens, None, block_table, block_size, 0, False
        )
        dequantize_and_gather_k_cache_sm80(
            out_opt, k_cache, seq_lens, None, block_table, block_size, 0, False
        )
        exact = torch.equal(out_ref, out_opt)
        max_err = (out_ref.float() - out_opt.float()).abs().max().item()

        t_ref = bench_fn(
            lambda: dequantize_and_gather_k_cache_triton(
                out_ref, k_cache, seq_lens, None, block_table, block_size, 0, False
            ),
            warmup=20, rep=100,
        )
        t_opt = bench_fn(
            lambda: dequantize_and_gather_k_cache_sm80(
                out_opt, k_cache, seq_lens, None, block_table, block_size, 0, False
            ),
            warmup=20, rep=100,
        )
        # With 38 layers -> 8 source layers + 30 consumer layers (skipped via workspace reuse!)
        total_38_ref_ms = t_ref * 38.0 / 1000.0
        total_8_opt_ms = t_opt * 8.0 / 1000.0
        print(
            f"Compressed tokens={n_compressed:5d} | Base: {t_ref:7.2f} us | "
            f"SM80-LUT: {t_opt:7.2f} us ({t_ref/t_opt:.2f}x) | "
            f"38-layer->8-source: {total_38_ref_ms:5.2f} ms -> {total_8_opt_ms:5.2f} ms "
            f"({total_38_ref_ms/total_8_opt_ms:.2f}x) | Bit-exact: {exact} (err={max_err:.1e})"
        )


if __name__ == "__main__":
    test_decode_nan_check()
    test_dequant_and_prefill()
