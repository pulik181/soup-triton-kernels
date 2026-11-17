"""
Solution package for Soup Kernel Challenge (Decentrathon 6.0).
"""

from solution.custom_kernels import (
    patch_qwen2_rmsnorm,
    patch_qwen2_rope,
    patch_cross_entropy,
    patch_soup_chunked_head_loss,
    patch_allocator_warmup,
    patch_embed_tokens_fp16,
)


def apply():
    patch_allocator_warmup()
    patch_embed_tokens_fp16()
    patch_qwen2_rmsnorm()
    patch_qwen2_rope()
    patch_cross_entropy()
    patch_soup_chunked_head_loss(chunk_size=128)
