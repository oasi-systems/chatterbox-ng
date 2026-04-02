"""
CUDA/GPU optimizations for ChatterBox TTS models.

Provides:
- BF16 inference precision
- torch.compile() on critical sub-modules
- SDPA (Scaled Dot-Product Attention) upgrade for encoder attention
- Streaming-specific optimizations (reduced ODE steps for intermediate chunks)

Usage:
    model = ChatterboxMultilingualTTS.from_pretrained("cuda")
    optimize_for_cuda(model)

    # Then use model normally — all optimizations are applied in-place
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def optimize_for_cuda(
    model,
    compile_mode: str = "reduce-overhead",
    use_bf16: bool = True,
    compile_models: bool = True,
):
    """Apply CUDA-specific optimizations to a ChatterBox model in-place.

    Args:
        model: ChatterboxTTS, ChatterboxMultilingualTTS, or ChatterboxTurboTTS
        compile_mode: torch.compile mode ("reduce-overhead", "max-autotune", "default")
        use_bf16: convert to bfloat16 for 2x memory bandwidth
        compile_models: apply torch.compile to sub-modules

    Returns:
        The same model, optimized.
    """
    device = model.device
    if not (isinstance(device, str) and "cuda" in device) and not (
        isinstance(device, torch.device) and device.type == "cuda"
    ):
        logger.warning(f"optimize_for_cuda called on non-CUDA device ({device}), skipping")
        return model

    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping optimizations")
        return model

    # --- BF16 ---
    if use_bf16 and torch.cuda.is_bf16_supported():
        logger.info("Converting to bfloat16...")
        _convert_to_bf16(model)
    elif use_bf16:
        logger.info("BF16 not supported on this GPU, using fp16...")
        _convert_to_fp16(model)

    # --- torch.compile ---
    if compile_models:
        logger.info(f"Compiling sub-modules with mode={compile_mode}...")
        _compile_submodules(model, compile_mode)

    # --- SDPA upgrade for encoder attention ---
    logger.info("Upgrading encoder attention to SDPA...")
    _upgrade_encoder_attention(model)

    logger.info("CUDA optimizations applied.")
    return model


def _convert_to_bf16(model):
    """Convert model components to bfloat16."""
    # S3Gen (flow encoder + CFM decoder + HiFiGAN)
    if hasattr(model, 's3gen'):
        model.s3gen.to(dtype=torch.bfloat16)

    # T3 backbone
    if hasattr(model, 't3'):
        model.t3.to(dtype=torch.bfloat16)

    # Voice encoder stays fp32 (used for embedding, not inference-critical)


def _convert_to_fp16(model):
    """Convert model components to float16."""
    if hasattr(model, 's3gen'):
        model.s3gen.to(dtype=torch.float16)
    if hasattr(model, 't3'):
        model.t3.to(dtype=torch.float16)


def _compile_submodules(model, mode):
    """Apply torch.compile to performance-critical sub-modules.

    We compile individual sub-modules rather than the full model because:
    - The inference loop has data-dependent control flow (EOS check)
    - Generators can't be compiled
    - Sub-module compilation avoids these issues while still fusing kernels
    """
    try:
        # S3Gen encoder (UpsampleConformerEncoder)
        if hasattr(model, 's3gen') and hasattr(model.s3gen, 'flow'):
            flow = model.s3gen.flow
            flow.encoder = torch.compile(flow.encoder, mode=mode, dynamic=True)
            logger.info("  Compiled: S3Gen encoder")

            # CFM decoder (ConditionalDecoder) — the biggest bottleneck
            if hasattr(flow, 'decoder') and hasattr(flow.decoder, 'estimator'):
                flow.decoder.estimator = torch.compile(
                    flow.decoder.estimator, mode=mode, dynamic=True
                )
                logger.info("  Compiled: CFM decoder estimator")

        # HiFiGAN vocoder
        if hasattr(model, 's3gen') and hasattr(model.s3gen, 'mel2wav'):
            model.s3gen.mel2wav = torch.compile(model.s3gen.mel2wav, mode=mode, dynamic=True)
            logger.info("  Compiled: HiFiGAN vocoder")

        # T3 backbone (LlamaModel or GPT2Model)
        if hasattr(model, 't3') and hasattr(model.t3, 'tfmr'):
            model.t3.tfmr = torch.compile(model.t3.tfmr, mode=mode, dynamic=True)
            logger.info("  Compiled: T3 backbone")

    except Exception as e:
        logger.warning(f"torch.compile failed (will continue without): {e}")


def _upgrade_encoder_attention(model):
    """Replace manual matmul attention in S3Gen encoder with F.scaled_dot_product_attention.

    The RelPositionMultiHeadedAttention uses two-matrix scoring:
        scores = (matrix_ac + matrix_bd) / sqrt(d_k)
    where matrix_ac is content-based and matrix_bd is position-based.

    We pre-compute matrix_bd as an attention bias and pass it to SDPA,
    which can use memory-efficient or math backends for the core Q@K^T computation.
    """
    if not hasattr(model, 's3gen') or not hasattr(model.s3gen, 'flow'):
        return

    encoder = model.s3gen.flow.encoder

    # Upgrade attention in both encoder stacks
    for layer_list in [encoder.encoders, encoder.up_encoders]:
        for layer in layer_list:
            attn = layer.self_attn
            _patch_attention_forward(attn)


def _patch_attention_forward(attn_module):
    """Monkey-patch the forward method of RelPositionMultiHeadedAttention
    to use F.scaled_dot_product_attention for the content attention,
    with position bias passed as attn_mask.
    """
    import math
    from torch.nn import functional as F
    from .models.s3gen.transformer.attention import RelPositionMultiHeadedAttention

    if not isinstance(attn_module, RelPositionMultiHeadedAttention):
        return

    original_forward = attn_module.forward

    def sdpa_forward(
        query, key, value,
        mask=torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb=torch.empty(0),
        cache=torch.zeros((0, 0, 0, 0)),
    ):
        q, k, v = attn_module.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        # Handle KV-cache
        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        new_cache = torch.cat((k, v), dim=-1)

        # Position encoding
        n_batch_pos = pos_emb.size(0)
        p = attn_module.linear_pos(pos_emb).view(n_batch_pos, -1, attn_module.h, attn_module.d_k)
        p = p.transpose(1, 2)  # (batch, head, time2, d_k)

        # Position-based scores (pre-computed as bias)
        q_with_bias_v = (q + attn_module.pos_bias_v.to(q.device)).transpose(1, 2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))

        # Content query with bias_u
        q_with_bias_u = (q + attn_module.pos_bias_u.to(q.device)).transpose(1, 2)

        # Check if rel_shift needed (content vs position shape mismatch)
        matrix_ac_shape = (q_with_bias_u.size(0), q_with_bias_u.size(1),
                           q_with_bias_u.size(2), k.size(2))
        if matrix_ac_shape[2:] != matrix_bd.shape[2:]:
            matrix_bd = attn_module.rel_shift(matrix_bd)

        # Position bias (divided by sqrt(d_k) because SDPA also divides by sqrt(d_k))
        # SDPA computes: softmax(Q@K^T/sqrt(d_k) + attn_mask) @ V
        # We want:       softmax((Q@K^T + matrix_bd)/sqrt(d_k)) @ V
        # So attn_mask = matrix_bd / sqrt(d_k)
        attn_bias = matrix_bd / math.sqrt(attn_module.d_k)

        # Add padding mask
        if mask.size(2) > 0:
            mask_expanded = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            mask_expanded = mask_expanded[:, :, :, :k.size(2)]
            padding_bias = mask_expanded.to(attn_bias.dtype) * -1e10
            attn_bias = attn_bias + padding_bias

        # SDPA — will dispatch to best available backend (Flash, MemEfficient, Math)
        n_batch = query.size(0)
        dropout_p = attn_module.dropout.p if attn_module.training else 0.0

        output = F.scaled_dot_product_attention(
            q_with_bias_u, k, v,
            attn_mask=attn_bias,
            dropout_p=dropout_p,
        )

        # Reshape output
        output = output.transpose(1, 2).contiguous().view(n_batch, -1, attn_module.h * attn_module.d_k)
        output = attn_module.linear_out(output)

        return output, new_cache

    attn_module.forward = sdpa_forward
