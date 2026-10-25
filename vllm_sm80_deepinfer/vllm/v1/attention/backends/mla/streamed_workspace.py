"""Experimental one-window gather allocation with request-level chunking.

Only capacity and request partitioning change. Never use a compressed coordinate
budget larger than the actual compressed gather allocation.
"""
def eligible(config):
    text=getattr(config.model_config,'hf_text_config',None)
    parallel=config.parallel_config
    return (text is not None and getattr(text,'model_type',None) in ('deepseek_v41','deepseek_v41_text')
            and getattr(text,'hidden_size',None)==5120
            and getattr(text,'num_hidden_layers',None)==40
            and getattr(text,'n_routed_experts',None)==384
            and getattr(text,'compress_ratios',None)
            and all(r in (0,1,2) for r in text.compress_ratios)
            and config.cache_config.block_size==128
            and parallel.decode_context_parallel_size==1
            and parallel.prefill_context_parallel_size==1)


def gather_capacity(config):
    length=int(config.model_config.max_model_len)
    if not eligible(config):return length*40
    drafts=int(config.num_speculative_tokens)
    if drafts<0:raise ValueError('Negative draft count')
    grain=128*max(1,max(config.model_config.hf_text_config.compress_ratios))
    return ((length+drafts+1+grain-1)//grain)*grain


def chunk_capacity(config,uncompressed_capacity,ratio):
    if not eligible(config):return uncompressed_capacity
    if ratio not in (1,2):raise ValueError('Unqualified indexer compression ratio')
    return uncompressed_capacity//ratio
