"""
ChatterBox NG — ONNX / TensorRT Export

Exports HiFiGAN and CFM estimator (ConditionalDecoder) as ONNX models,
optionally compiles them to TensorRT engines for 2-4x inference speedup.

Encoder and T3 stay in PyTorch (too dynamic for static graph export).

Usage:
    # Export ONNX models
    python -m chatterbox.trt_export --output-dir ./trt_engines --device cuda

    # Export + compile TensorRT (requires tensorrt package)
    python -m chatterbox.trt_export --output-dir ./trt_engines --device cuda --compile-trt

    # Use in code
    from chatterbox.trt_export import export_hifigan_onnx, export_estimator_onnx
    export_hifigan_onnx(model.s3gen.mel2wav, "hifigan.onnx")
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class _HiFiGANExportWrapper(nn.Module):
    """Wraps HiFTGenerator.inference() for ONNX tracing.

    Removes @torch.inference_mode() decorator and cache_source conditional
    (we export the no-cache path; cache logic stays in Python at runtime).
    """

    def __init__(self, hifigan):
        super().__init__()
        self.f0_predictor = hifigan.f0_predictor
        self.f0_upsamp = hifigan.f0_upsamp
        self.m_source = hifigan.m_source
        self.decode = hifigan.decode

    def forward(self, speech_feat: torch.Tensor) -> torch.Tensor:
        """No-cache inference path.

        Args:
            speech_feat: (B, 80, T) mel spectrogram

        Returns:
            generated_speech: (B, 1, T_audio)
        """
        f0 = self.f0_predictor(speech_feat)
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = self.m_source(s)
        s = s.transpose(1, 2)
        generated_speech = self.decode(x=speech_feat, s=s)
        return generated_speech


class _EstimatorExportWrapper(nn.Module):
    """Wraps ConditionalDecoder.forward() for ONNX tracing.

    Replaces einops.pack/rearrange/repeat with equivalent torch ops
    so the graph is fully traceable without third-party op registration.
    """

    def __init__(self, estimator):
        super().__init__()
        self._estimator = estimator

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        r: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Direct forward — einops ops trace correctly through torch.onnx.export
        because they decompose to standard torch operations at trace time.

        Args:
            x: (B, 80, T) noisy mel
            mask: (B, 1, T)
            mu: (B, 80, T) target mu
            t: (B,) timestep
            spks: (B, 80) speaker embedding
            cond: (B, 80, T) conditioning
            r: (B,) end time (meanflow only)
        """
        return self._estimator.forward(
            x=x, mask=mask, mu=mu, t=t, spks=spks, cond=cond, r=r,
        )


def export_hifigan_onnx(
    hifigan: nn.Module,
    output_path: str,
    device: str = "cuda",
    opset_version: int = 17,
    T: int = 100,
) -> str:
    """Export HiFiGAN vocoder to ONNX.

    Args:
        hifigan: HiFTGenerator instance
        output_path: path for .onnx file
        device: device for dummy inputs
        opset_version: ONNX opset
        T: time dimension for dummy input (dynamic axes handle any size)

    Returns:
        Path to exported ONNX file
    """
    output_path = str(output_path)
    wrapper = _HiFiGANExportWrapper(hifigan).to(device).eval()

    # Dummy input
    speech_feat = torch.randn(1, 80, T, device=device, dtype=next(hifigan.parameters()).dtype)

    logger.info(f"Exporting HiFiGAN to {output_path} (opset {opset_version})...")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (speech_feat,),
            output_path,
            input_names=["speech_feat"],
            output_names=["audio"],
            dynamic_axes={
                "speech_feat": {0: "batch", 2: "mel_frames"},
                "audio": {0: "batch", 2: "audio_samples"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )

    logger.info(f"HiFiGAN exported: {output_path}")
    return output_path


def export_estimator_onnx(
    estimator: nn.Module,
    output_path: str,
    device: str = "cuda",
    opset_version: int = 17,
    T: int = 100,
    meanflow: bool = False,
) -> str:
    """Export CFM estimator (ConditionalDecoder) to ONNX.

    The ODE solver loop stays in PyTorch — only the neural network
    estimator (the expensive part per step) gets exported.

    Args:
        estimator: ConditionalDecoder instance
        output_path: path for .onnx file
        device: device for dummy inputs
        opset_version: ONNX opset
        T: time dimension for dummy input
        meanflow: if True, include r input (end time)

    Returns:
        Path to exported ONNX file
    """
    output_path = str(output_path)
    wrapper = _EstimatorExportWrapper(estimator).to(device).eval()
    dtype = next(estimator.parameters()).dtype

    # For CFG, the solver doubles batch to 2*B. Export with B=2 to cover that.
    B = 2
    x = torch.randn(B, 80, T, device=device, dtype=dtype)
    mask = torch.ones(B, 1, T, device=device, dtype=dtype)
    mu = torch.randn(B, 80, T, device=device, dtype=dtype)
    t = torch.rand(B, device=device, dtype=dtype)
    spks = torch.randn(B, 80, device=device, dtype=dtype)
    cond = torch.randn(B, 80, T, device=device, dtype=dtype)

    input_names = ["x", "mask", "mu", "t", "spks", "cond"]
    args = (x, mask, mu, t, spks, cond)
    dynamic_axes = {
        "x": {0: "batch", 2: "frames"},
        "mask": {0: "batch", 2: "frames"},
        "mu": {0: "batch", 2: "frames"},
        "t": {0: "batch"},
        "spks": {0: "batch"},
        "cond": {0: "batch", 2: "frames"},
        "output": {0: "batch", 2: "frames"},
    }

    if meanflow:
        r = torch.rand(B, device=device, dtype=dtype)
        args = args + (r,)
        input_names.append("r")
        dynamic_axes["r"] = {0: "batch"}

    logger.info(f"Exporting CFM estimator to {output_path} (opset {opset_version}, meanflow={meanflow})...")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            args,
            output_path,
            input_names=input_names,
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
        )

    logger.info(f"CFM estimator exported: {output_path}")
    return output_path


def compile_trt_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    max_batch: int = 4,
    max_frames: int = 1000,
    min_frames: int = 10,
    opt_frames: int = 200,
) -> str:
    """Compile ONNX model to TensorRT engine.

    Requires: pip install tensorrt

    Args:
        onnx_path: path to .onnx file
        engine_path: output path for .trt engine
        fp16: enable FP16 precision (2x speedup on most GPUs)
        max_batch: maximum batch size
        max_frames: maximum mel frames (time dimension)
        min_frames: minimum mel frames
        opt_frames: optimal mel frames (TRT optimizes kernels for this size)

    Returns:
        Path to compiled TensorRT engine
    """
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError(
            "TensorRT not installed. Install with: pip install tensorrt\n"
            "Or use ONNX Runtime as an alternative: pip install onnxruntime-gpu"
        )

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    logger.info(f"Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(f"  ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError(f"Failed to parse ONNX model: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2 GB

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("FP16 enabled")

    # Set dynamic shape profiles
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        name = inp.name
        shape = list(inp.shape)

        # Build min/opt/max shapes based on dimension semantics
        min_shape = list(shape)
        opt_shape = list(shape)
        max_shape = list(shape)

        for d in range(len(shape)):
            if shape[d] == -1:  # Dynamic dimension
                if d == 0:  # Batch
                    min_shape[d] = 1
                    opt_shape[d] = 2  # CFG doubles batch
                    max_shape[d] = max_batch
                else:  # Frames (time dimension)
                    min_shape[d] = min_frames
                    opt_shape[d] = opt_frames
                    max_shape[d] = max_frames

        profile.set_shape(name, tuple(min_shape), tuple(opt_shape), tuple(max_shape))
        logger.info(f"  {name}: min={min_shape}, opt={opt_shape}, max={max_shape}")

    config.add_optimization_profile(profile)

    logger.info("Building TensorRT engine (this may take several minutes)...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("TensorRT engine build failed")

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    logger.info(f"TensorRT engine saved: {engine_path}")
    return engine_path


def export_all(
    model,
    output_dir: str,
    device: str = "cuda",
    compile_trt: bool = False,
    fp16: bool = True,
) -> dict:
    """Export both HiFiGAN and CFM estimator, optionally compile to TensorRT.

    Args:
        model: ChatterboxMultilingualTTS instance
        output_dir: directory for exported files
        device: target device
        compile_trt: if True, also compile TensorRT engines
        fp16: FP16 for TensorRT (ignored if compile_trt=False)

    Returns:
        dict with paths: {"hifigan_onnx", "estimator_onnx", "hifigan_trt", "estimator_trt"}
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    meanflow = getattr(model.s3gen.flow, "meanflow", False) or \
               getattr(model.s3gen.flow.decoder, "meanflow", False)

    result = {}

    # Export HiFiGAN
    hifigan_onnx = str(out / "hifigan.onnx")
    export_hifigan_onnx(model.s3gen.mel2wav, hifigan_onnx, device=device)
    result["hifigan_onnx"] = hifigan_onnx

    # Export CFM estimator
    estimator_onnx = str(out / "estimator.onnx")
    export_estimator_onnx(
        model.s3gen.flow.decoder.estimator,
        estimator_onnx,
        device=device,
        meanflow=meanflow,
    )
    result["estimator_onnx"] = estimator_onnx

    # Compile TensorRT
    if compile_trt:
        hifigan_trt = str(out / "hifigan.trt")
        compile_trt_engine(hifigan_onnx, hifigan_trt, fp16=fp16)
        result["hifigan_trt"] = hifigan_trt

        estimator_trt = str(out / "estimator.trt")
        compile_trt_engine(
            estimator_onnx, estimator_trt, fp16=fp16,
            max_batch=4,  # 2*B for CFG
        )
        result["estimator_trt"] = estimator_trt

    return result


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(description="Export ChatterBox models to ONNX/TensorRT")
    parser.add_argument("--output-dir", default="./trt_engines", help="Output directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--meanflow", action="store_true", help="Use meanflow S3Gen weights")
    parser.add_argument("--compile-trt", action="store_true", help="Also compile TensorRT engines")
    parser.add_argument("--fp16", action="store_true", default=True, help="FP16 for TensorRT")
    parser.add_argument("--no-fp16", action="store_false", dest="fp16")
    args = parser.parse_args()

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    logger.info(f"Loading model on {args.device}...")
    model = ChatterboxMultilingualTTS.from_pretrained(args.device, meanflow=args.meanflow)

    result = export_all(
        model,
        output_dir=args.output_dir,
        device=args.device,
        compile_trt=args.compile_trt,
        fp16=args.fp16,
    )

    logger.info("Export complete:")
    for k, v in result.items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
