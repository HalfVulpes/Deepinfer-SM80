# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os

import torch
import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = init_logger(__name__)

# Checkpoint names the token embedding can appear under, most specific first.
_EMBED_KEYS = (
    "embed.weight",
    "model.embed_tokens.weight",
    "model.embed.weight",
    "embed_tokens.weight",
)


def _has_real_weight(module) -> bool:
    """True for a materialised layer; False for PPMissingLayer / None.

    Under pipeline parallelism vLLM replaces the layers a rank does not own with
    PPMissingLayer, which has no ``weight``. Aliasing one of those into the draft
    silently produces a no-op layer rather than an error, so check explicitly.
    """
    return module is not None and getattr(module, "weight", None) is not None


def _load_embed_from_checkpoint(embed: nn.Module, model_path: str) -> None:
    """Fill the draft's own token embedding straight from the checkpoint.

    Only needed under PP: the drafter runs on the LAST pipeline rank, but the
    target's ``embed_tokens`` lives on the FIRST, so there is nothing local to
    alias. Reading the one tensor off disk (~1 GB) avoids adding a cross-rank
    collective to model load, which would have to be ordered against every other
    rank's initialisation.
    """
    from safetensors import safe_open

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    key = shard = None
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        for cand in _EMBED_KEYS:
            if cand in weight_map:
                key, shard = cand, os.path.join(model_path, weight_map[cand])
                break
    if key is None:
        raise RuntimeError(
            f"DSpark+PP: could not find a token-embedding tensor in {index_path}; "
            f"looked for {_EMBED_KEYS}. The draft needs its own copy because the "
            "target's embedding lives on PP rank 0."
        )

    with safe_open(shard, framework="pt") as f:
        w = f.get_tensor(key)
    with torch.no_grad():
        # VocabParallelEmbedding pads the vocab dimension, so copy into the
        # leading rows rather than assigning the whole tensor.
        # Use the layer's official loader: it selects this TP rank's rows
        # and clears vocab padding. A full-matrix copy is only valid for TP1.
        loader = getattr(embed.weight, "weight_loader", embed.weight_loader)
        loader(embed.weight, w)
    logger.info(
        "DSpark+PP: loaded draft token embedding %s from key %r", tuple(w.shape), key
    )


def _resolve_dspark_attention_backend(
    draft_model_config: ModelConfig,
    draft_backend: AttentionBackendEnum | None,
    target_backend: AttentionBackendEnum | None,
) -> AttentionBackendEnum | None:
    if draft_backend is not None:
        return draft_backend
    # DeepSeek-V4(.1) draft layers share the target's KV-cache layout. Other
    # DSpark architectures may use a different attention kind.
    if draft_model_config.hf_config.model_type in ("deepseek_v4", "deepseek_v41"):
        if target_backend is not None:
            logger.info_once(
                "Using the target model's %s attention backend for the "
                "DeepSeek-V4 DSpark drafter.",
                target_backend.name,
            )
        return target_backend
    return None


def load_dspark_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config

    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.model_loader import get_model
    from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal
    from vllm.model_executor.models.utils import get_draft_quant_config
    from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
        _should_share,
        get_target_lm_head,
    )

    draft_attention_backend = _resolve_dspark_attention_backend(
        draft_model_config,
        speculative_config.attention_backend,
        vllm_config.attention_config.backend,
    )

    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=draft_attention_backend,
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            )
            if speculative_config.kv_cache_dtype is not None
            else vllm_config.cache_config
        ),
    )
    # VllmConfig post-init restores the target's quant config because the target
    # config is retained for DSpark's target-layer metadata, so we must override it.
    draft_vllm_config.quant_config = get_draft_quant_config(vllm_config)

    with set_model_tag("dspark_head"):
        draft_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    # NOTE(pp): DSpark used to hard-refuse pipeline parallelism here. It works
    # provided (a) the drafter is built on the last PP rank only -- the model
    # runner already does exactly that -- and (b) the draft can still reach a
    # token embedding and an lm_head. For DeepSeek-V4/V4.1 the aux-hidden-state
    # taps (dspark_target_layer_ids = the last three backbone layers) and
    # lm_head both land on that same last rank, so only the embedding must be
    # sourced separately.
    is_pp = get_pp_group().world_size != 1

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = draft_model.model
    target_vocab_size = vllm_config.model_config.get_vocab_size()

    target_embed = getattr(target_inner, "embed_tokens", None)
    draft_embed = getattr(draft_inner, "embed_tokens", None)
    if (
        _has_real_weight(target_embed)
        and draft_model_config.get_vocab_size() <= target_vocab_size
        and _should_share(
            draft_model, "has_own_embed_tokens", draft_embed, target_embed
        )
    ):
        if draft_embed is not None:
            del draft_inner.embed_tokens
        draft_inner.embed_tokens = target_embed
    elif is_pp and draft_embed is not None and not _has_real_weight(target_embed):
        # PP: the target's embedding is a PPMissingLayer on this rank. Keep the
        # draft's own module and populate it from the checkpoint -- the draft's
        # load_weights() only consumes mtp.* tensors, so it is still empty here.
        _load_embed_from_checkpoint(draft_embed, draft_model_config.model)

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(draft_model, "lm_head", None)
    draft_output_vocab_size = (
        getattr(draft_model_config.hf_config, "draft_vocab_size", None)
        or draft_model_config.get_vocab_size()
    )
    if (
        _has_real_weight(target_lm_head)
        and draft_output_vocab_size == target_vocab_size
        and _should_share(draft_model, "has_own_lm_head", draft_lm_head, target_lm_head)
    ):
        if draft_lm_head is not None:
            del draft_model.lm_head
        draft_model.lm_head = target_lm_head
    elif is_pp and not _has_real_weight(target_lm_head):
        # Should not happen: the drafter runs on the last rank, which owns
        # lm_head. Fail loudly rather than silently drafting through a no-op.
        raise RuntimeError(
            "DSpark+PP: the target lm_head is not materialised on this rank. The "
            "drafter is expected to run on the last pipeline rank."
        )

    return draft_model
