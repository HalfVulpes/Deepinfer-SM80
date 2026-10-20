# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

import vllm.envs as envs
from vllm.utils.torch_utils import direct_register_custom_op

# The prenorm GEMM is L2-bound on re-reads of `fn`, not CUDA-core bound: at
# (block_m=2, tile_n=12) it moves 1.745 GB in 472.7 us = 3.69 TB/s, near the L2
# read ceiling, with only 18.2% of the fp32 pipe busy. `fn` traffic scales
# 1/block_m and `x` traffic scales 1/tile_n, and `fn` is 12x larger, so raising
# block_m is the lever — but the traffic model alone does not pick the winner.
# Swept on A100 (benchmark_dsv4_sm80.py --kernel prenorm-gemm): (8, 4) is
# fastest at T<=256 yet collapses above 1024 (620 us at T=2048 vs 459 for the
# old (2, 12)); (8, 6) is within 5% of the best measured config at every T from
# 64 to 2048 (26.9 us @128, 201.6 @1024, 442.2 @2048) and is the keeper.
_PRENORM_BLOCK_M = 8
_PRENORM_BLOCK_M_TILE_N = 6

# Below this many tokens the one-CTA-per-token tilelang kernel wins on launch
# latency (7.0 us at T=1 vs 10.5 for cuBLAS); at and above it the cuBLAS
# route wins (T=32 is a dead heat at 16.8 vs 16.7 us, then 31.8 vs 17.4 at
# T=64 and 442 vs 85 at T=2048). Measured crossover at T=32.
_PRENORM_SMALL_T = 32

# Escape hatch: routing back to the fused tilelang (8, 6) kernel if the
# cuBLAS route misbehaves on some stack. Numerics differ between the two
# (bf16 fn vs fp32 fn); the parity test covers both.
_PRENORM_USE_CUBLAS = True

# On the cuBLAS route the sqrsum is a second full read of the post-mapped
# residual (_row_sqrsum_kernel: 135 us x 86 calls = 11.6 ms of an 8K prefill,
# at the HBM ceiling, so nothing is left to tune inside it). mhc_post holds
# those values in registers one kernel earlier, so folding the reduction in
# there removes the pass and a launch: measured 498.9 -> 350.6 us at T=8192.
# Read per call rather than at import so a serving A/B only needs a restart.

# (tile_n, split_k, n_thr) for the small-token fused post+prenorm kernel.
# Swept over the full product at the shapes decode runs (m=5, 6, 12, 16;
# benchmark_dsv4_sm80.py --kernel mhc-fused). Scored on the *pair*, because
# split_k becomes the n_splits the pre kernel then reduces serially -- that
# coupling costs only 0.41 us going from 8 splits to 32, so it is close to
# free. Boundary pair at m=6: 13.73 us at the old (2, 8, 256) against 11.88
# here, -13.5%; the config also wins at m=5, 12 and 16. Same kernel and same
# arithmetic as before, so numerics are untouched (rel err 2.9e-07 either way).
_SMALL_FMA_CONFIG = (6, 16, 128)


def _torch_hc_prenorm_gemm(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
) -> None:
    assert out.shape[0] == 1
    assert sqrsum.shape[0] == 1
    x_float = x.float()
    out[0].copy_(x_float @ fn.t())
    sqrsum[0].copy_(x_float.square().sum(dim=-1))


def _tilelang_hc_prenorm_gemm(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    hidden_size: int,
    hc_mult: int,
    tile_n: int = 12,
    n_thr: int = 512,
    n_splits: int = 1,
    sqrsum_ready: bool = False,
) -> None:
    """Route the prenorm GEMM.

    ``sqrsum_ready`` says the caller already filled ``sqrsum``; only the cuBLAS
    route, where the reduction is a separate pass over ``x``, can act on it.
    The other routes produce the same values as a side effect of work they do
    anyway, so they ignore the flag and overwrite.
    """
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        hc_prenorm_gemm_block_m_tilelang,
        hc_prenorm_gemm_tilelang,
    )

    assert out.shape[0] == n_splits
    assert sqrsum.shape[0] == n_splits
    assert x.shape[1] == hc_mult * hidden_size
    assert x.shape[1] % n_splits == 0
    if (x.shape[1] // n_splits) % n_thr != 0:
        # Shape the tilelang kernels cannot tile; the prenorm GEMM output is
        # tiny ([T, hc_mult3] + [T]), so torch is a cheap universal fallback.
        _torch_hc_prenorm_gemm(x, fn, out, sqrsum)
        return
    use_default_config = tile_n == 12 and n_thr == 512
    if (
        n_splits == 1
        and use_default_config
        and x.shape[0] < _PRENORM_SMALL_T
        and x.shape[1] % 1024 == 0
    ):
        hc_prenorm_gemm_tilelang(
            x,
            fn,
            out,
            sqrsum,
            hidden_size,
            hc_mult,
            fn.shape[0],
            1024,
            4,
            n_splits,
        )
        return
    if n_splits == 1 and x.dtype == torch.bfloat16 and _PRENORM_USE_CUBLAS:
        # cuBLAS bf16 GEMM + one-pass sqrsum: 17.4 us at T=64, 85 at T=2048
        # vs 26.2 / 442 for the best fused tilelang config (8, 6) — the fused
        # kernels re-read fn per token tile and are L2-bound at large T. The
        # block_m tilelang route below stays in-tree (tested and benched) as
        # the escape.
        from vllm.model_executor.kernels.mhc.triton import hc_prenorm_gemm_cublas

        hc_prenorm_gemm_cublas(x, fn, out, None if sqrsum_ready else sqrsum)
        return
    if n_splits == 1 and use_default_config:
        # No upper token-count gate: num_tokens is dynamic and the kernel guards
        # token_idx < num_tokens, so this is correct at any T.
        hc_prenorm_gemm_block_m_tilelang(
            x,
            fn,
            out,
            sqrsum,
            hidden_size,
            hc_mult,
            fn.shape[0],
            n_thr,
            _PRENORM_BLOCK_M_TILE_N,
            _PRENORM_BLOCK_M,
        )
        return
    hc_prenorm_gemm_tilelang(
        x,
        fn,
        out,
        sqrsum,
        hidden_size,
        hc_mult,
        fn.shape[0],
        n_thr,
        tile_n,
        n_splits,
    )


def mhc_pre_delayed_tilelang(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None = None,
    x: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run mHC pre with a carried pre-mix and return the next pre-mix.

    Args:
        residual: BF16 residual streams of shape (tokens, hc_mult, hidden_size).
        fn: FP32 projection of shape (hc_mult * (hc_mult + 2), input_size).
        hc_scale: FP32 scales of shape (3,).
        hc_base: FP32 bias of shape (hc_mult * (hc_mult + 2),).
        rms_eps: RMS normalization epsilon.
        hc_pre_eps: Pre-mix epsilon.
        hc_sinkhorn_eps: Sinkhorn epsilon.
        hc_post_mult_value: Post-mix multiplier.
        sinkhorn_repeat: Number of Sinkhorn iterations.
        pre_mix: FP32 coefficients from the previous sublayer, or None to
            select residual stream zero at model entry.
        x: Optional BF16 projection input of shape (tokens, input_size), for
            the first layer's broadcast embedding and summed projection.
        norm_weight: Optional BF16 RMSNorm weight for the collapsed input.
        norm_eps: RMSNorm epsilon for the collapsed input.

    Returns:
        Post and residual coefficients, optionally normalized BF16 layer input,
        and the next FP32 pre-mix, with shapes (tokens, hc_mult, 1),
        (tokens, hc_mult, hc_mult), (tokens, hidden_size), and (tokens, hc_mult).
    """
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        mhc_pre_big_fuse_tilelang,
    )
    from vllm.model_executor.kernels.mhc.warmup import (
        MHC_PRE_NORM_KERNEL,
        compute_mhc_pre_num_splits,
    )
    from vllm.utils.deep_gemm import (
        is_deep_gemm_supported,
        tf32_hc_prenorm_gemm,
    )

    assert residual.ndim == 3 and residual.dtype == torch.bfloat16
    assert residual.is_contiguous()
    num_tokens, hc_mult, hidden_size = residual.shape
    if x is None:
        x = residual.view(num_tokens, hc_mult * hidden_size)
    assert x.ndim == 2 and x.dtype == torch.bfloat16 and x.is_contiguous()
    assert x.shape[0] == num_tokens
    input_size = x.shape[1]
    mix_size = hc_mult * (hc_mult + 2)
    assert fn.shape == (mix_size, input_size) and fn.dtype == torch.float32
    assert hc_scale.shape == (3,) and hc_scale.dtype == torch.float32
    assert hc_base.shape == (mix_size,) and hc_base.dtype == torch.float32
    if pre_mix is not None:
        assert pre_mix.shape == (num_tokens, hc_mult)
        assert pre_mix.dtype == torch.float32 and pre_mix.is_contiguous()

    next_pre_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    post = torch.empty_like(next_pre_mix)
    comb = torch.empty(
        num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )
    outputs = (
        post.unsqueeze(-1),
        comb.view(num_tokens, hc_mult, hc_mult),
        layer_input,
        next_pre_mix,
    )
    if num_tokens == 0:
        return outputs

    use_deep_gemm = is_deep_gemm_supported()
    n_splits = (
        compute_mhc_pre_num_splits(input_size, num_tokens) if use_deep_gemm else 1
    )
    mixes = torch.empty(
        n_splits, num_tokens, mix_size, dtype=torch.float32, device=residual.device
    )
    sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )
    if use_deep_gemm:
        tf32_hc_prenorm_gemm(x, fn, mixes, sqrsum, n_splits)
    else:
        _tilelang_hc_prenorm_gemm(x, fn, mixes, sqrsum, input_size, 1)
    if norm_weight is not None:
        assert norm_weight.shape == (hidden_size,)
        assert norm_weight.dtype == torch.bfloat16 and norm_weight.is_contiguous()
        MHC_PRE_NORM_KERNEL(
            mixes,
            sqrsum,
            hc_scale,
            hc_base,
            residual,
            post,
            comb,
            layer_input,
            norm_weight,
            pre_mix if pre_mix is not None else post,
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
            rms_numel=input_size,
        )
        return outputs
    mhc_pre_big_fuse_tilelang(
        mixes,
        sqrsum,
        hc_scale,
        hc_base,
        residual,
        post,
        comb,
        layer_input,
        pre_mix if pre_mix is not None else post,
        next_pre_mix,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        hc_mult,
        use_pre_mix_in=pre_mix is not None,
        save_pre_mix=True,
        rms_numel=input_size,
    )
    return outputs


def _mhc_pre_delayed_tilelang_fake(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None = None,
    x: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, hc_mult, hidden_size = residual.shape
    return (
        torch.empty(
            num_tokens, hc_mult, 1, dtype=torch.float32, device=residual.device
        ),
        torch.empty(
            num_tokens, hc_mult, hc_mult, dtype=torch.float32, device=residual.device
        ),
        torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
        ),
        torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=residual.device),
    )


def mhc_pre_tilelang(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass for mHC pre block.

    Args:
        residual: shape (..., hc_mult, hidden_size), dtype torch.bfloat16
        fn: shape (hc_mult3, hc_mult * hidden_size), dtype torch.float32
        hc_scale: shape (3,), dtype torch.float32
        hc_base: shape (hc_mult3,), dtype torch.float32
        rms_eps: RMS normalization epsilon
        hc_pre_eps: pre-mix epsilon
        hc_sinkhorn_eps: sinkhorn epsilon
        hc_post_mult_value: post-mix multiplier value
        sinkhorn_repeat: number of sinkhorn iterations
        n_splits: split-k factor;
        norm_weight: optional RMSNorm weight, shape (hidden_size,), dtype
            torch.bfloat16. When provided, RMSNorm is fused into the
            layer_input write path of the big_fuse kernel.
        norm_eps: epsilon for the fused RMSNorm; only consulted when
            norm_weight is given.

    Returns:
        post_mix: shape (..., hc_mult), dtype torch.float32
        comb_mix: shape (..., hc_mult, hc_mult), dtype torch.float32
        layer_input: shape (..., hidden_size), dtype torch.bfloat16
    """
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        compute_num_split,
        mhc_pre_big_fuse_tilelang,
        mhc_pre_big_fuse_with_norm_tilelang,
    )
    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm
    from vllm.utils.math_utils import cdiv

    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape[0] == hc_mult3
    assert fn.shape[1] == hc_hidden_size
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    if norm_weight is not None:
        assert norm_weight.shape == (hidden_size,)
        if norm_weight.dtype != torch.bfloat16:
            norm_weight = norm_weight.to(torch.bfloat16)
        if not norm_weight.is_contiguous():
            norm_weight = norm_weight.contiguous()

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    from vllm.utils.deep_gemm import is_deep_gemm_supported

    use_deep_gemm = is_deep_gemm_supported()
    if use_deep_gemm:
        # these numbers are from deepgemm kernel impl
        block_k = 64
        block_m = 64
        n_splits = compute_num_split(block_k, hc_hidden_size, cdiv(num_tokens, block_m))
    else:
        n_splits = 1

    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )

    gemm_out_mul = torch.empty(
        n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=residual.device
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )

    residual_2d = residual_flat.view(num_tokens, hc_mult * hidden_size)
    if use_deep_gemm:
        tf32_hc_prenorm_gemm(
            residual_2d,
            fn,
            gemm_out_mul,
            gemm_out_sqrsum,
            n_splits,
        )
    else:
        _tilelang_hc_prenorm_gemm(
            residual_2d,
            fn,
            gemm_out_mul,
            gemm_out_sqrsum,
            hidden_size,
            hc_mult,
        )

    if norm_weight is None:
        mhc_pre_big_fuse_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_flat,
            post_mix,
            comb_mix,
            layer_input,
            # Pre-mix buffers are unused in V4 mode.
            post_mix,
            post_mix,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            hc_mult,
        )
    else:
        mhc_pre_big_fuse_with_norm_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_flat,
            post_mix,
            comb_mix,
            layer_input,
            norm_weight,
            post_mix,
            post_mix,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_eps,
            n_splits,
            hc_mult,
        )

    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


def _mhc_pre_tilelang_fake(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    # Create empty tensors with correct shapes for meta device / shape inference
    post_mix = torch.empty(
        *outer_shape,
        hc_mult,
        1,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix = torch.empty(
        *outer_shape,
        hc_mult,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input = torch.empty(
        *outer_shape,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    return post_mix, comb_mix, layer_input


def mhc_pre_broadcast_tilelang(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    fn_broadcast: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """First-layer mHC pre for a residual broadcast from ``(T, H)``."""
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        compute_num_split,
        mhc_pre_big_fuse_broadcast_with_norm_tilelang,
    )
    from vllm.utils.math_utils import cdiv

    assert norm_weight is not None, "broadcast mHC pre currently requires fused RMSNorm"
    assert residual.dtype == torch.bfloat16
    assert residual.dim() == 2
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hidden_size = residual.shape[-1]
    hc_mult = fn.shape[1] // hidden_size
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    assert fn.shape == (hc_mult3, hc_mult * hidden_size)
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)
    assert fn_broadcast is not None
    assert fn_broadcast.dtype == torch.float32
    assert fn_broadcast.shape == (hc_mult3, hidden_size)

    if norm_weight.dtype != torch.bfloat16:
        norm_weight = norm_weight.to(torch.bfloat16)
    if not norm_weight.is_contiguous():
        norm_weight = norm_weight.contiguous()

    residual_flat = residual
    num_tokens = residual.shape[0]

    from vllm.utils.deep_gemm import is_deep_gemm_supported

    use_deep_gemm = is_deep_gemm_supported()
    if use_deep_gemm:
        n_splits = compute_num_split(64, hidden_size, cdiv(num_tokens, 64))
    else:
        n_splits = 1

    residual_out = torch.empty(
        num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=residual.device
    )
    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )
    gemm_out_mul = torch.empty(
        n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=residual.device
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )

    if use_deep_gemm:
        from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm

        tf32_hc_prenorm_gemm(
            residual_flat,
            fn_broadcast,
            gemm_out_mul,
            gemm_out_sqrsum,
            n_splits,
        )
    else:
        # Broadcast GEMM has K = hidden_size (hc_mult factor is 1).
        _tilelang_hc_prenorm_gemm(
            residual_flat,
            fn_broadcast,
            gemm_out_mul,
            gemm_out_sqrsum,
            hidden_size,
            1,
        )
    mhc_pre_big_fuse_broadcast_with_norm_tilelang(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_flat,
        residual_out,
        post_mix,
        comb_mix,
        layer_input,
        norm_weight,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_eps,
        n_splits,
        hc_mult,
    )
    return (
        residual_out,
        post_mix.unsqueeze(-1),
        comb_mix.view(num_tokens, hc_mult, hc_mult),
        layer_input,
    )


def mhc_post_tilelang(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        mhc_post_tilelang as _mhc_post_kernel,
    )

    out = torch.empty_like(residual)
    _mhc_post_kernel(
        comb_res_mix,
        residual,
        post_layer_mix.squeeze(-1),
        x,
        out,
        residual.shape[-2],
        residual.shape[-1],
    )
    return out


def mhc_fused_post_pre_tilelang(
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
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run one MHC post block followed by the next MHC pre block.

    When ``norm_weight`` is provided, the layer_input_cur output is the
    RMSNorm'd activation (fused into the kernel); otherwise it is the
    raw pre-norm activation as before.

    Returns:
        residual_cur: post-mapped residual, shape (..., hc_mult, hidden_size)
        post_mix_cur: shape (..., hc_mult, 1)
        comb_mix_cur: shape (..., hc_mult, hc_mult)
        layer_input_cur: shape (..., hidden_size)
    """

    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        compute_num_split,
        mhc_fused_tilelang,
        mhc_post_tilelang,
        mhc_pre_big_fuse_tilelang,
        mhc_pre_big_fuse_with_norm_tilelang,
    )
    from vllm.utils.math_utils import cdiv

    assert residual.dtype == torch.bfloat16
    assert x.dtype == torch.bfloat16
    assert post_layer_mix.dtype == torch.float32
    assert comb_res_mix.dtype == torch.float32
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    hc_hidden_size = hc_mult * hidden_size
    outer_shape = residual.shape[:-2]

    assert x.shape == (*outer_shape, hidden_size)
    assert post_layer_mix.shape in (
        (*outer_shape, hc_mult, 1),
        (*outer_shape, hc_mult),
    )
    assert comb_res_mix.shape == (*outer_shape, hc_mult, hc_mult)
    assert fn.shape == (hc_mult3, hc_hidden_size)
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    if norm_weight is not None:
        assert norm_weight.shape == (hidden_size,)
        if norm_weight.dtype != torch.bfloat16:
            norm_weight = norm_weight.to(torch.bfloat16)
        if not norm_weight.is_contiguous():
            norm_weight = norm_weight.contiguous()

    assert n_splits in (1, 2, 4, 8)
    assert hidden_size % n_splits == 0

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    x_flat = x.view(num_tokens, hidden_size)
    post_layer_mix_flat = post_layer_mix.view(num_tokens, hc_mult)
    comb_res_mix_flat = comb_res_mix.view(num_tokens, hc_mult, hc_mult)

    from vllm.utils.deep_gemm import is_deep_gemm_supported

    use_deep_gemm = is_deep_gemm_supported()
    use_small_fma = num_tokens <= 16
    if use_small_fma:
        tile_n, n_splits, fma_n_thr = _SMALL_FMA_CONFIG
        if (
            hc_mult3 % tile_n
            or hidden_size % n_splits
            or (hidden_size // n_splits) % fma_n_thr
        ):
            # The kernel walks its h slice in whole n_thr strides and drops any
            # remainder, so a shape the swept config cannot tile exactly has to
            # fall back rather than silently compute part of the reduction.
            tile_n = 2 if num_tokens < 8 else 3
            n_splits = 8 if (num_tokens < 8 and hidden_size <= 4096) else 4
            fma_n_thr = 256
    else:
        if use_deep_gemm:
            # these number are from deepgemm kernel impl
            block_k = 64
            block_m = 64
            n_splits = compute_num_split(
                block_k, hc_hidden_size, cdiv(num_tokens, block_m)
            )
        else:
            n_splits = 1

    gemm_out_mul = torch.empty(
        n_splits,
        num_tokens,
        hc_mult3,
        dtype=torch.float32,
        device=residual.device,
    )
    gemm_out_sqrsum = torch.empty(
        n_splits,
        num_tokens,
        dtype=torch.float32,
        device=residual.device,
    )
    residual_cur = torch.empty_like(residual_flat)
    post_mix_cur = torch.empty(
        num_tokens,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix_cur = torch.empty(
        num_tokens,
        hc_mult2,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input_cur = torch.empty(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    if use_small_fma:
        mhc_fused_tilelang(
            comb_res_mix_flat,
            residual_flat,
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
        residual_cur_2d = residual_cur.view(num_tokens, hc_mult * hidden_size)
        fuse_sqrsum = envs.VLLM_MHC_POST_FUSE_SQRSUM and not use_deep_gemm
        if fuse_sqrsum:
            from vllm.model_executor.kernels.mhc.tilelang_kernels import (
                mhc_post_sqrsum_tilelang,
            )

            mhc_post_sqrsum_tilelang(
                comb_res_mix_flat,
                residual_flat,
                post_layer_mix_flat,
                x_flat,
                residual_cur,
                gemm_out_sqrsum,
                residual.shape[-2],
                residual.shape[-1],
            )
        else:
            mhc_post_tilelang(
                comb_res_mix_flat,
                residual_flat,
                post_layer_mix_flat,
                x_flat,
                residual_cur,
                residual.shape[-2],
                residual.shape[-1],
            )

        if use_deep_gemm:
            from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm

            tf32_hc_prenorm_gemm(
                residual_cur_2d,
                fn,
                gemm_out_mul,
                gemm_out_sqrsum,
                n_splits,
            )
        else:
            _tilelang_hc_prenorm_gemm(
                residual_cur_2d,
                fn,
                gemm_out_mul,
                gemm_out_sqrsum,
                hidden_size,
                hc_mult,
                sqrsum_ready=fuse_sqrsum,
            )

    if norm_weight is None:
        mhc_pre_big_fuse_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_cur,
            post_mix_cur,
            comb_mix_cur,
            layer_input_cur,
            # Pre-mix buffers are unused in V4 mode.
            post_mix_cur,
            post_mix_cur,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            hc_mult,
        )
    else:
        mhc_pre_big_fuse_with_norm_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_cur,
            post_mix_cur,
            comb_mix_cur,
            layer_input_cur,
            norm_weight,
            post_mix_cur,
            post_mix_cur,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_eps,
            n_splits,
            hc_mult,
        )

    return (
        residual_cur.view(*outer_shape, hc_mult, hidden_size),
        post_mix_cur.view(*outer_shape, hc_mult, 1),
        comb_mix_cur.view(*outer_shape, hc_mult, hc_mult),
        layer_input_cur.view(*outer_shape, hidden_size),
    )


def _mhc_fused_post_pre_tilelang_fake(
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
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    residual_cur = torch.empty_like(residual)
    post_mix_cur = torch.empty(
        *outer_shape,
        hc_mult,
        1,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix_cur = torch.empty(
        *outer_shape,
        hc_mult,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input_cur = torch.empty(
        *outer_shape,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur


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
    """Run mHC post followed by mHC pre-delayed as a fused SM80 boundary pair."""
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        compute_num_split,
        mhc_fused_tilelang,
        mhc_post_sqrsum_tilelang,
        mhc_pre_big_fuse_tilelang,
    )
    from vllm.model_executor.kernels.mhc.warmup import (
        MHC_PRE_NORM_KERNEL,
    )
    from vllm.utils.deep_gemm import (
        is_deep_gemm_supported,
        tf32_hc_prenorm_gemm,
    )
    from vllm.utils.math_utils import cdiv

    assert residual.ndim == 3 and residual.dtype == torch.bfloat16
    assert residual.is_contiguous()
    assert x.dtype == torch.bfloat16
    assert post_layer_mix.dtype == torch.float32
    assert comb_res_mix.dtype == torch.float32
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    num_tokens, hc_mult, hidden_size = residual.shape
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * (hc_mult + 2)
    hc_hidden_size = hc_mult * hidden_size

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
    outputs = (
        residual_cur,
        post_mix_cur.unsqueeze(-1),
        comb_mix_cur.view(num_tokens, hc_mult, hc_mult),
        layer_input_cur,
        next_pre_mix,
    )
    if num_tokens == 0:
        return outputs

    x_flat = x.view(num_tokens, hidden_size)
    post_layer_mix_flat = post_layer_mix.view(num_tokens, hc_mult)
    comb_res_mix_flat = comb_res_mix.view(num_tokens, hc_mult, hc_mult)

    use_deep_gemm = is_deep_gemm_supported()
    use_small_fma = num_tokens <= 16
    if use_small_fma:
        tile_n, n_splits, fma_n_thr = _SMALL_FMA_CONFIG
        if (
            hc_mult3 % tile_n
            or hidden_size % n_splits
            or (hidden_size // n_splits) % fma_n_thr
        ):
            if (
                hc_mult3 % 6 == 0
                and hidden_size % 8 == 0
                and (hidden_size // 8) % 128 == 0
            ):
                tile_n, n_splits, fma_n_thr = 6, 8, 128
            else:
                tile_n = 2 if num_tokens < 8 else 3
                n_splits = 8 if (num_tokens < 8 and hidden_size <= 4096) else 4
                fma_n_thr = 256
                if (
                    hc_mult3 % tile_n
                    or hidden_size % n_splits
                    or (hidden_size // n_splits) % fma_n_thr
                ):
                    use_small_fma = False
                    n_splits = 1
    else:
        if use_deep_gemm:
            n_splits = compute_num_split(
                64, hc_hidden_size, cdiv(num_tokens, 64)
            )
        else:
            n_splits = 1

    gemm_out_mul = torch.empty(
        n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=residual.device
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
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
        if use_deep_gemm:
            from vllm.model_executor.kernels.mhc.tilelang_kernels import (
                mhc_post_tilelang as _mhc_post_kernel,
            )

            _mhc_post_kernel(
                comb_res_mix_flat,
                residual,
                post_layer_mix_flat,
                x_flat,
                residual_cur,
                hc_mult,
                hidden_size,
            )
            tf32_hc_prenorm_gemm(
                residual_cur_2d,
                fn,
                gemm_out_mul,
                gemm_out_sqrsum,
                n_splits,
            )
        else:
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

    if norm_weight is not None:
        assert norm_weight.shape == (hidden_size,)
        assert norm_weight.dtype == torch.bfloat16 and norm_weight.is_contiguous()
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
        return outputs

    mhc_pre_big_fuse_tilelang(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_cur,
        post_mix_cur,
        comb_mix_cur,
        layer_input_cur,
        pre_mix if pre_mix is not None else post_mix_cur,
        next_pre_mix,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        hc_mult,
        use_pre_mix_in=pre_mix is not None,
        save_pre_mix=True,
        rms_numel=hc_hidden_size,
    )
    return outputs


def _mhc_fused_post_pre_delayed_tilelang_fake(
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
    return (
        torch.empty_like(residual),
        torch.empty(
            num_tokens, hc_mult, 1, dtype=torch.float32, device=residual.device
        ),
        torch.empty(
            num_tokens, hc_mult, hc_mult, dtype=torch.float32, device=residual.device
        ),
        torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
        ),
        torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=residual.device),
    )


def _mhc_post_tilelang_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(residual)


def hc_head_fused_kernel_tilelang(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """Apply the fused hc_head kernel and return the (T, H) bf16 result."""
    num_tokens, hc_mult, hidden_size = hs_flat.shape
    out = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=hs_flat.device
    )
    if num_tokens == 0:
        return out
    from vllm.model_executor.kernels.mhc.tilelang_kernels import hc_head_fuse_tilelang

    hc_head_fuse_tilelang(
        hs_flat,
        fn,
        hc_scale,
        hc_base,
        out,
        hidden_size,
        rms_eps,
        hc_eps,
        hc_mult,
    )
    return out


def _hc_head_fused_kernel_tilelang_fake(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    num_tokens, _, hidden_size = hs_flat.shape
    return torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=hs_flat.device
    )


direct_register_custom_op(
    op_name="mhc_pre_delayed_tilelang",
    op_func=mhc_pre_delayed_tilelang,
    mutates_args=[],
    fake_impl=_mhc_pre_delayed_tilelang_fake,
)
direct_register_custom_op(
    op_name="mhc_pre_tilelang",
    op_func=mhc_pre_tilelang,
    mutates_args=[],
    fake_impl=_mhc_pre_tilelang_fake,
)
direct_register_custom_op(
    op_name="mhc_post_tilelang",
    op_func=mhc_post_tilelang,
    mutates_args=[],
    fake_impl=_mhc_post_tilelang_fake,
)

direct_register_custom_op(
    op_name="mhc_fused_post_pre_tilelang",
    op_func=mhc_fused_post_pre_tilelang,
    mutates_args=[],
    fake_impl=_mhc_fused_post_pre_tilelang_fake,
)
direct_register_custom_op(
    op_name="mhc_fused_post_pre_delayed_tilelang",
    op_func=mhc_fused_post_pre_delayed_tilelang,
    mutates_args=[],
    fake_impl=_mhc_fused_post_pre_delayed_tilelang_fake,
)

direct_register_custom_op(
    op_name="hc_head_fused_kernel_tilelang",
    op_func=hc_head_fused_kernel_tilelang,
    mutates_args=[],
    fake_impl=_hc_head_fused_kernel_tilelang_fake,
)
