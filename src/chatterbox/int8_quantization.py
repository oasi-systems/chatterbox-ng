"""
INT8 weight-only quantization for T3 model.

Reduces T3 transformer memory by ~2x and can accelerate matmul on
CUDA Tensor Cores (sm >= 75 — L4, A10, A100, H100).

Three backends (auto-selected by priority):
1. torchao — best GPU INT8 via CUDA Tensor Cores (recommended)
2. torch.ao.quantization — dynamic INT8 (CPU speedup, GPU limited)
3. Manual weight-only — INT8 storage, dequantize at inference

Usage:
    from chatterbox.int8_quantization import quantize_t3_int8
    quantize_t3_int8(model)  # modifies model in-place
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def quantize_t3_int8(model, include_heads: bool = True) -> dict:
    """Apply INT8 weight-only quantization to the T3 transformer backbone.

    Targets all nn.Linear layers in the T3 transformer (Llama/GPT2 backbone).
    Optionally includes speech_head (large output projection: hidden → 8194 vocab).

    Args:
        model: ChatterboxMultilingualTTS instance
        include_heads: also quantize speech_head (default True)

    Returns:
        dict with quantization info:
        - backend: which quantization backend was used
        - n_layers: number of linear layers quantized
        - memory_saved_mb: estimated memory reduction
    """
    if not hasattr(model, 't3'):
        logger.warning("Model has no T3 module, skipping INT8 quantization")
        return {"backend": "none", "n_layers": 0, "memory_saved_mb": 0}

    t3 = model.t3

    # Count parameters before
    param_bytes_before = sum(
        p.numel() * p.element_size()
        for p in t3.parameters()
    )

    # Try backends in priority order
    result = _try_torchao(t3, include_heads)
    if result is None:
        result = _try_torch_ao_dynamic(t3, include_heads)
    if result is None:
        result = _try_manual_int8(t3, include_heads)
    if result is None:
        logger.warning("No INT8 backend available — model unchanged")
        return {"backend": "none", "n_layers": 0, "memory_saved_mb": 0}

    # Estimate memory saved
    param_bytes_after = sum(
        p.numel() * p.element_size()
        for p in t3.parameters()
        if p.is_floating_point()
    )
    # INT8 buffers
    buffer_bytes = sum(
        b.numel() * b.element_size()
        for b in t3.buffers()
    )
    total_after = param_bytes_after + buffer_bytes
    saved_mb = (param_bytes_before - total_after) / 1024 / 1024

    result["memory_saved_mb"] = max(0, saved_mb)
    logger.info(
        f"INT8 quantization: backend={result['backend']}, "
        f"layers={result['n_layers']}, "
        f"saved={result['memory_saved_mb']:.0f}MB"
    )
    return result


def _try_torchao(t3, include_heads: bool):
    """Try torchao INT8 weight-only quantization (best for GPU)."""
    try:
        import torchao
        from torchao.quantization import int8_weight_only

        n_layers = _count_linear_layers(t3.tfmr)

        # Quantize transformer backbone
        torchao.quantize_(t3.tfmr, int8_weight_only())

        # Optionally quantize speech head
        if include_heads and hasattr(t3, 'speech_head'):
            torchao.quantize_(t3.speech_head, int8_weight_only())
            n_layers += 1

        logger.info(f"torchao INT8 weight-only applied to {n_layers} layers")
        return {"backend": "torchao", "n_layers": n_layers}

    except ImportError:
        logger.debug("torchao not available")
        return None
    except Exception as e:
        logger.warning(f"torchao INT8 failed: {e}")
        return None


def _try_torch_ao_dynamic(t3, include_heads: bool):
    """Try torch.ao dynamic INT8 quantization."""
    try:
        from torch.ao.quantization import quantize_dynamic

        n_layers = _count_linear_layers(t3.tfmr)

        # Dynamic quantization: weights INT8, activations computed at runtime
        t3.tfmr = quantize_dynamic(
            t3.tfmr,
            {nn.Linear},
            dtype=torch.qint8,
        )

        if include_heads and hasattr(t3, 'speech_head'):
            # Wrap speech_head manually (single layer)
            t3.speech_head = quantize_dynamic(
                nn.Sequential(t3.speech_head),
                {nn.Linear},
                dtype=torch.qint8,
            )[0]
            n_layers += 1

        logger.info(f"torch.ao dynamic INT8 applied to {n_layers} layers")
        return {"backend": "torch_ao_dynamic", "n_layers": n_layers}

    except Exception as e:
        logger.warning(f"torch.ao dynamic INT8 failed: {e}")
        return None


def _try_manual_int8(t3, include_heads: bool):
    """Manual INT8 weight-only: store weights as INT8 + scale, dequantize at inference."""
    n_layers = 0

    def _quantize_linear(module: nn.Module):
        nonlocal n_layers
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                quantized = Int8WeightLinear.from_float(child)
                setattr(module, name, quantized)
                n_layers += 1
            else:
                _quantize_linear(child)

    _quantize_linear(t3.tfmr)

    if include_heads and hasattr(t3, 'speech_head') and isinstance(t3.speech_head, nn.Linear):
        t3.speech_head = Int8WeightLinear.from_float(t3.speech_head)
        n_layers += 1

    if n_layers == 0:
        return None

    logger.info(f"Manual INT8 weight-only applied to {n_layers} layers")
    return {"backend": "manual_int8", "n_layers": n_layers}


class Int8WeightLinear(nn.Module):
    """Linear layer with INT8 weight storage, dequantized to compute dtype at inference.

    Weights are stored as INT8 (1 byte per param) with per-channel scale factors.
    At inference, weights are dequantized to the input dtype (BF16/FP16/FP32)
    for the matmul. This halves weight memory with minimal quality impact.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('weight_int8', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.ones(out_features, dtype=torch.float32))
        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None

    @classmethod
    def from_float(cls, linear: nn.Linear) -> 'Int8WeightLinear':
        """Convert a float Linear to INT8 weight-only."""
        out_features, in_features = linear.weight.shape
        has_bias = linear.bias is not None

        layer = cls(in_features, out_features, bias=has_bias)

        # Per-channel symmetric quantization
        weight = linear.weight.detach().float()
        scale = weight.abs().amax(dim=1) / 127.0
        scale = scale.clamp(min=1e-8)  # avoid div by zero
        weight_int8 = (weight / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)

        layer.weight_int8 = weight_int8
        layer.weight_scale = scale

        if has_bias:
            layer.bias = linear.bias.detach().float()

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weights to input dtype
        weight = self.weight_int8.to(x.dtype) * self.weight_scale.to(x.dtype).unsqueeze(1)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return torch.nn.functional.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, dtype=int8'


def _count_linear_layers(module: nn.Module) -> int:
    """Count nn.Linear layers in a module."""
    return sum(1 for m in module.modules() if isinstance(m, nn.Linear))
