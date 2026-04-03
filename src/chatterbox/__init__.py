try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # For Python <3.8

__version__ = version("chatterbox-ng")


from .tts import ChatterboxTTS
from .vc import ChatterboxVC
from .mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
from .streaming import ChatterboxStreamingTTS, StreamingResampler
from .audio_processing import post_process, lufs_normalize, de_ess, match_room_tone
from .cuda_optimizations import optimize_for_cuda, warmup_model
from .trt_export import export_hifigan_onnx, export_estimator_onnx, export_all
from .trt_runtime import load_trt_modules