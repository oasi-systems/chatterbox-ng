"""
CUDA/GPU optimizations for ChatterBox TTS models.

Provides:
- torch.autocast for automatic mixed-precision (BF16/FP16) inference
- torch.compile() on critical sub-modules (kernel fusion)
- SDPA (Scaled Dot-Product Attention) upgrade for encoder attention
- Flash Attention / Memory-Efficient SDPA backend selection
- CUDA-specific flags (cuDNN benchmark, TF32)
- Weight norm removal for clean inference
- Warmup pass to trigger JIT compilation before serving

Usage:
    model = ChatterboxMultilingualTTS.from_pretrained("cuda")
    optimize_for_cuda(model)

    # Optional: pre-warm the compiled kernels (recommended for production)
    warmup_model(model, device="cuda")

    # Then use model normally — all optimizations are applied in-place
"""

import logging
import time
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def optimize_for_cuda(
    model,
    compile_mode: str = "default",
    use_bf16: bool = True,
    compile_models: bool = True,
    use_tensorrt: bool = False,
    trt_engine_dir: str = None,
):
    """Apply CUDA-specific optimizations to a ChatterBox model in-place.

    Args:
        model: ChatterboxMultilingualTTS instance
        compile_mode: torch.compile mode. Default: "default" (kernel fusion without
            CUDA graphs — safe for streaming with dynamic shapes).
            "max-autotune" uses CUDA graphs and is INCOMPATIBLE with streaming
            (crashes on dynamic tensor shapes). Only use for monolithic generate().
        use_bf16: convert to bfloat16 for 2x memory bandwidth
        compile_models: apply torch.compile to sub-modules
        use_tensorrt: replace HiFiGAN and CFM estimator with TRT/ORT engines
        trt_engine_dir: directory containing exported .onnx/.trt files (required if use_tensorrt=True)

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

    # --- CUDA backend flags ---
    _set_cuda_flags()

    # --- Mixed precision via autocast ---
    if use_bf16 and torch.cuda.is_bf16_supported():
        logger.info("Enabling autocast (bfloat16)...")
        _setup_autocast(model, torch.bfloat16)
    elif use_bf16:
        logger.info("BF16 not supported, enabling autocast (float16)...")
        _setup_autocast(model, torch.float16)

    # --- TensorRT / ONNX Runtime acceleration ---
    if use_tensorrt:
        if not trt_engine_dir:
            logger.warning("use_tensorrt=True but no trt_engine_dir specified, skipping")
        else:
            from .trt_runtime import load_trt_modules
            trt_result = load_trt_modules(model, trt_engine_dir)
            logger.info(f"TensorRT modules: {trt_result}")
            # Don't torch.compile modules that are already using TRT
            if trt_result.get("hifigan"):
                compile_models = False  # TRT handles these, skip compile
                logger.info("Skipping torch.compile for TRT-accelerated modules")

    # --- torch.compile ---
    if compile_models:
        logger.info(f"Compiling sub-modules with mode={compile_mode}...")
        _compile_submodules(model, compile_mode)

    # --- SDPA upgrade for encoder attention ---
    logger.info("Upgrading encoder attention to SDPA...")
    _upgrade_encoder_attention(model)

    logger.info("CUDA optimizations applied.")
    return model


def warmup_model(model, device="cuda", n_warmup: int = 3):
    """Run dummy inference passes to trigger JIT compilation of all torch.compile'd modules.

    Call this once at server boot, after optimize_for_cuda(). The first inference
    through compiled modules is slow (kernel autotuning + CUDA graph capture).
    Subsequent calls hit the cached kernels.

    Args:
        model: optimized ChatterBox model
        device: target device
        n_warmup: number of warmup passes (3 is enough for stable compile caches)
    """
    logger.info(f"Warming up compiled kernels ({n_warmup} passes)...")
    t0 = time.time()

    # Need conditionals loaded for warmup
    if model.conds is None:
        logger.warning("No conditionals loaded — skipping warmup. Call prepare_conditionals() first.")
        return

    # Synthetic short text for warmup
    warmup_text = "Test warmup."

    with torch.inference_mode():
        for i in range(n_warmup):
            try:
                if hasattr(model, 'generate'):
                    is_multilingual = hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'cangjie_converter')
                    kwargs = {"text": warmup_text}
                    if is_multilingual:
                        kwargs["language_id"] = "en"
                    _ = model.generate(**kwargs)
                logger.info(f"  Warmup pass {i+1}/{n_warmup} done")
            except Exception as e:
                logger.warning(f"  Warmup pass {i+1} failed: {e}")
                break

    elapsed = time.time() - t0
    logger.info(f"Warmup complete in {elapsed:.1f}s — kernels cached for production speed.")


def _set_cuda_flags():
    """Set global CUDA flags for maximum throughput."""
    # TF32: use Tensor Cores for fp32 matmul (19-bit mantissa, ~8x throughput)
    # Safe for inference — negligible quality impact
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # cuDNN benchmark: auto-select fastest convolution algorithm
    # Cost: ~1s at first conv call per shape. Pays off for repeated inference.
    torch.backends.cudnn.benchmark = True

    # Enable Flash Attention and Memory-Efficient SDPA backends
    # L4/A10/A100/H100 all support Flash Attention v2
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    # Keep math SDPA as fallback — some shapes/dtypes can't use Flash/MemEfficient
    torch.backends.cuda.enable_math_sdp(True)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
    logger.info(f"CUDA flags set: TF32=on, cuDNN benchmark=on, Flash SDPA=on ({gpu_name})")


def _setup_autocast(model, dtype):
    """Configure model for torch.autocast mixed-precision inference.

    Instead of manually casting weights to bf16/fp16 (which requires explicit
    fp32 exceptions and boundary casts everywhere), we keep all weights in fp32
    and let torch.autocast handle precision automatically per-op:
    - matmul, conv, linear → bf16 (fast on Tensor Cores)
    - FFT, STFT, layernorm, softmax → stays fp32 (needs precision)

    This eliminates all dtype mismatch bugs and is the foundation for TensorRT.

    We also remove weight_norm parametrizations as a standalone optimization
    (fusing weight_g/weight_v into a single weight tensor for faster inference).
    """
    # Remove weight_norm parametrizations — standalone optimization.
    # Fuses weight_g/weight_v into a plain weight tensor.
    for top_attr in ('s3gen', 't3'):
        top = getattr(model, top_attr, None)
        if top is None:
            continue
        for module in top.modules():
            if torch.nn.utils.parametrize.is_parametrized(module, 'weight'):
                try:
                    torch.nn.utils.parametrize.remove_parametrizations(module, 'weight')
                except Exception:
                    pass

    # Store autocast config on sub-modules — used by inference entry points
    if hasattr(model, 's3gen'):
        model.s3gen._autocast_dtype = dtype
    if hasattr(model, 't3'):
        model.t3._autocast_dtype = dtype
    model._autocast_dtype = dtype
    logger.info(f"Autocast configured: dtype={dtype}, weights stay fp32")


def _compile_submodules(model, mode):
    """Apply torch.compile to performance-critical sub-modules.

    We compile individual sub-modules rather than the full model because:
    - The inference loop has data-dependent control flow (EOS check)
    - Generators can't be compiled
    - Sub-module compilation avoids these issues while still fusing kernels

    IMPORTANT: "max-autotune" and "reduce-overhead" use CUDA graphs which
    are INCOMPATIBLE with streaming (dynamic tensor shapes change each chunk).
    Use "default" mode for streaming — it does kernel fusion via Triton
    without CUDA graphs.

    Strategy by bottleneck analysis (profiled on L4):
    - Encoder (53% of time): kernel fusion is critical
    - CFM estimator (29%): benefits from fused attention + conv ops
    - HiFiGAN (5%): lightweight, skip compile (overhead not worth it)
    - T3 (13%): transformer backbone benefits from kernel fusion
    """
    try:
        # S3Gen flow (encoder + CFM estimator) — NOT compiled.
        # Streaming re-encodes ALL accumulated tokens each chunk, so the
        # encoder input shape grows every call. torch.compile with dynamic=True
        # still fails on shape-dependent operations (clamp, make_pad_mask).
        # The 53% + 29% = 82% of S3Gen time runs in eager mode.
        # Future: TensorRT with dynamic shape profiles can handle this.

        # HiFiGAN vocoder (5%) — skip, not worth the compile overhead

        # T3 backbone (LlamaModel or GPT2Model) — safe to compile.
        # T3 generates one token at a time (fixed shape per step).
        if hasattr(model, 't3') and hasattr(model.t3, 'tfmr'):
            model.t3.tfmr = torch.compile(model.t3.tfmr, mode=mode, dynamic=True)
            logger.info(f"  Compiled: T3 backbone (mode={mode})")

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
