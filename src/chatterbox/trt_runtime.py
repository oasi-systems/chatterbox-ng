"""
ChatterBox NG — TensorRT / ONNX Runtime Wrappers

Drop-in replacements for HiFiGAN and CFM estimator that use TensorRT
or ONNX Runtime for inference. Same forward() interface as PyTorch modules.

The ODE solver loop stays in PyTorch — only the estimator neural network
(called per ODE step) and HiFiGAN vocoder are accelerated.

Usage:
    from chatterbox.trt_runtime import load_trt_modules
    load_trt_modules(model, engine_dir="./trt_engines")
    # model now uses TRT for HiFiGAN and CFM estimator
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TRTEngine:
    """TensorRT engine wrapper with CUDA stream and memory management."""

    def __init__(self, engine_path: str):
        import tensorrt as trt

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        # Map input/output names to binding indices
        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        logger.info(
            f"TRT engine loaded: {engine_path} "
            f"(inputs={self.input_names}, outputs={self.output_names})"
        )

    def __call__(self, inputs: dict) -> dict:
        """Run inference.

        Args:
            inputs: {name: torch.Tensor} for each input

        Returns:
            {name: torch.Tensor} for each output
        """
        import tensorrt as trt

        # Set input shapes and bind memory
        for name in self.input_names:
            tensor = inputs[name].contiguous()
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())

        # Allocate output buffers
        outputs = {}
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            torch_dtype = _trt_dtype_to_torch(dtype)
            out = torch.empty(tuple(shape), dtype=torch_dtype, device="cuda")
            self.context.set_tensor_address(name, out.data_ptr())
            outputs[name] = out

        # Execute
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        return outputs


class OrtEngine:
    """ONNX Runtime engine wrapper (fallback when TensorRT is unavailable)."""

    def __init__(self, onnx_path: str):
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Enable CUDA memory arena for reduced allocation overhead
        sess_options.enable_mem_pattern = True

        self.session = ort.InferenceSession(
            onnx_path, sess_options=sess_options, providers=providers,
        )

        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        active_provider = self.session.get_providers()[0]
        logger.info(
            f"ORT engine loaded: {onnx_path} "
            f"(provider={active_provider}, inputs={self.input_names})"
        )

    def __call__(self, inputs: dict) -> dict:
        """Run inference.

        Args:
            inputs: {name: torch.Tensor} for each input

        Returns:
            {name: torch.Tensor} for each output (on same device as first input)
        """
        device = next(iter(inputs.values())).device

        # ORT expects numpy arrays
        feed = {
            name: inputs[name].detach().cpu().float().numpy()
            for name in self.input_names
            if name in inputs
        }

        results = self.session.run(self.output_names, feed)

        return {
            name: torch.from_numpy(arr).to(device)
            for name, arr in zip(self.output_names, results)
        }


class TRTHiFiGAN(nn.Module):
    """Drop-in replacement for HiFTGenerator using TRT/ORT engine.

    Handles cache_source logic in Python (not exported to engine).
    """

    def __init__(self, engine, original_hifigan):
        super().__init__()
        self.engine = engine
        # Keep original for cache_source logic and f0 source model
        self._original = original_hifigan

    @torch.inference_mode()
    def inference(self, speech_feat, cache_source=torch.zeros(1, 1, 0)):
        """Same interface as HiFTGenerator.inference()."""
        result = self.engine({"speech_feat": speech_feat})
        generated_speech = result["audio"]

        # Cache source handling stays in Python
        # For streaming with cache, fall back to original
        if cache_source.shape[2] != 0:
            return self._original.inference(speech_feat, cache_source)

        # Return dummy source (not used in streaming pipeline)
        return generated_speech, torch.zeros(1, 1, 0, device=speech_feat.device)

    def forward(self, *args, **kwargs):
        return self.inference(*args, **kwargs)


class TRTEstimator(nn.Module):
    """Drop-in replacement for ConditionalDecoder using TRT/ORT engine.

    Called per ODE step by the solver. The solver loop stays in PyTorch.
    """

    def __init__(self, engine, meanflow: bool = False):
        super().__init__()
        self.engine = engine
        self.meanflow = meanflow
        # Needed by cast_all in the solver
        self.dtype = torch.float32

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
        inputs = {
            "x": x.float(),
            "mask": mask.float(),
            "mu": mu.float(),
            "t": t.float(),
            "spks": spks.float(),
            "cond": cond.float(),
        }
        if r is not None and self.meanflow:
            inputs["r"] = r.float()

        result = self.engine(inputs)
        output = result["output"]

        # Match input dtype
        return output.to(x.dtype)


def _load_engine(onnx_path: str, trt_path: str):
    """Load TRT engine if available, fall back to ORT, then to ONNX file."""
    trt_path = Path(trt_path)
    onnx_path = Path(onnx_path)

    # Try TensorRT first
    if trt_path.exists():
        try:
            return TRTEngine(str(trt_path))
        except ImportError:
            logger.warning("TensorRT not installed, trying ONNX Runtime...")
        except Exception as e:
            logger.warning(f"TRT engine load failed ({e}), trying ONNX Runtime...")

    # Fall back to ONNX Runtime
    if onnx_path.exists():
        try:
            return OrtEngine(str(onnx_path))
        except ImportError:
            raise ImportError(
                "Neither TensorRT nor ONNX Runtime installed.\n"
                "Install one of:\n"
                "  pip install tensorrt\n"
                "  pip install onnxruntime-gpu"
            )

    raise FileNotFoundError(
        f"No engine files found. Expected:\n"
        f"  TRT: {trt_path}\n"
        f"  ONNX: {onnx_path}\n"
        f"Run: python -m chatterbox.trt_export --output-dir {trt_path.parent}"
    )


def load_trt_modules(model, engine_dir: str) -> dict:
    """Replace HiFiGAN and CFM estimator with TRT/ORT accelerated versions.

    Args:
        model: ChatterboxMultilingualTTS instance
        engine_dir: directory containing exported .onnx / .trt files

    Returns:
        dict with loaded module names: {"hifigan": True/False, "estimator": True/False}
    """
    engine_dir = Path(engine_dir)
    result = {"hifigan": False, "estimator": False}

    # HiFiGAN
    try:
        hifigan_engine = _load_engine(
            engine_dir / "hifigan.onnx",
            engine_dir / "hifigan.trt",
        )
        original_hifigan = model.s3gen.mel2wav
        model.s3gen.mel2wav = TRTHiFiGAN(hifigan_engine, original_hifigan)
        result["hifigan"] = True
        logger.info("HiFiGAN replaced with accelerated engine")
    except (FileNotFoundError, ImportError) as e:
        logger.warning(f"HiFiGAN: keeping PyTorch ({e})")

    # CFM estimator
    try:
        meanflow = getattr(model.s3gen.flow, "meanflow", False) or \
                   getattr(model.s3gen.flow.decoder, "meanflow", False)

        estimator_engine = _load_engine(
            engine_dir / "estimator.onnx",
            engine_dir / "estimator.trt",
        )
        model.s3gen.flow.decoder.estimator = TRTEstimator(estimator_engine, meanflow=meanflow)
        result["estimator"] = True
        logger.info("CFM estimator replaced with accelerated engine")
    except (FileNotFoundError, ImportError) as e:
        logger.warning(f"CFM estimator: keeping PyTorch ({e})")

    return result


def _trt_dtype_to_torch(trt_dtype):
    """Convert TensorRT dtype to torch dtype."""
    import tensorrt as trt
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    if hasattr(trt, "bfloat16"):
        mapping[trt.bfloat16] = torch.bfloat16
    return mapping.get(trt_dtype, torch.float32)
