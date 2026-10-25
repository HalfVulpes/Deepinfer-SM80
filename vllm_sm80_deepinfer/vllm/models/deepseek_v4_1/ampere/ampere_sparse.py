# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 sparse MLA attention for SM8x (Ampere: A100/A800/CMP 170HX).

Same construction as ``deepseek_v4/ampere``: the ROCm Triton sparse-MLA
implementation is reused wholesale. Its kernels, ragged metadata builders and
the bf16 ``rocm_inv_rope_einsum`` o_proj path are plain Triton/torch (the
aiter-only preshuffle GEMMs self-disable off ROCm), and
``vllm.v1.attention.ops.fp8_sm80`` supplies e4m3 encode/decode below SM89
where Triton refuses native fp8 converts.

What differs from the V4 Ampere layer is inherited from the V4.1 ROCm class:
ratio-0/1/2 layer types, KV sharing through ``kv_source_layer_ids`` (the
consumer reads the source layer's compressed cache and metadata), the
two-level candidate-block filtering, and the indexer-K-from-latent store.

The dense projections of the V4.1 checkpoint are MXFP8 (``[32, 32]`` blocks,
UE8M0 scales). On SM8x they run through Marlin W8A16 (upstream
``MarlinMxfp8LinearKernel``, sm_75+). ``wo_a`` is the exception: the einsum
o_proj reads the raw ``[N, K]`` weight, which a Marlin repack would destroy,
so it is dequantized to bf16 once at load instead (``mxfp8_dequant_to_bf16``,
honoured by ``ModelOptLinearMethod.process_weights_after_loading``).
"""

from vllm.models.deepseek_v4_1.amd.rocm import (
    DeepseekV4ROCMAiterMLASparseBackend,
    DeepseekV4ROCMAiterSparseSWAMetadataBuilder,
    DeepseekV41ROCMAiterMLAAttention,
    DeepseekV41ROCMAiterSparseSWABackend,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.sparse_swa import (
    _LAYER_TYPE_C1A,
    _LAYER_TYPE_C2A,
    _LAYER_TYPE_C4A,
    _LAYER_TYPE_C128A,
    _LAYER_TYPE_SWAONLY,
    FlashMLASchedMeta,
)


class DeepseekV41AmpereMLASparseBackend(DeepseekV4ROCMAiterMLASparseBackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_DSV41"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 8


class DeepseekV41AmpereSparseSWAMetadataBuilder(
    DeepseekV4ROCMAiterSparseSWAMetadataBuilder
):
    def build_tile_scheduler(
        self, num_decode_tokens: int
    ) -> dict[str, FlashMLASchedMeta | None]:
        # The base builder plans FlashMLA tile schedules for every layer type
        # on CUDA; the ragged Triton decode never calls FlashMLA (which is
        # not even compiled for sm_80), so skip the planner entirely.
        return dict.fromkeys(
            (
                _LAYER_TYPE_SWAONLY,
                _LAYER_TYPE_C4A,
                _LAYER_TYPE_C128A,
                _LAYER_TYPE_C1A,
                _LAYER_TYPE_C2A,
            )
        )


class DeepseekV41AmpereSparseSWABackend(DeepseekV41ROCMAiterSparseSWABackend):
    @staticmethod
    def get_builder_cls() -> type["DeepseekV41AmpereSparseSWAMetadataBuilder"]:
        return DeepseekV41AmpereSparseSWAMetadataBuilder


class DeepseekV41AmpereMLAAttention(DeepseekV41ROCMAiterMLAAttention):
    """SM8x DeepSeek V4.1 attention: ROCm Triton path on CUDA Ampere."""

    backend_cls = DeepseekV41AmpereMLASparseBackend
    swa_backend_cls = DeepseekV41AmpereSparseSWABackend

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The ROCm parent already cleared ``wo_a.is_bmm`` so the ordinary
        # MXFP8 linear method owns the layer. On SM8x that method's kernel
        # walk lands on Marlin, whose repack replaces ``weight`` with the
        # Marlin tile layout -- but ``rocm_inv_rope_einsum`` reads the plain
        # ``[N, K]`` weight (+ E8M0 scale) to build its cached bf16 copy.
        # Ask for the emulation kernel instead: it dequantizes the weight to
        # bf16 in place at load, and the einsum helper then uses it as is.
        self.wo_a.mxfp8_dequant_to_bf16 = True
