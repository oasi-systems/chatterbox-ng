try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # For Python <3.8

__version__ = version("chatterbox-ng")


from .vc import ChatterboxVC
from .mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
from .streaming import ChatterboxStreamingTTS, StreamingResampler
from .audio_processing import post_process, lufs_normalize, de_ess, match_room_tone
from .cuda_optimizations import optimize_for_cuda, warmup_model
from .int8_quantization import quantize_t3_int8
from .trt_export import export_hifigan_onnx, export_estimator_onnx, export_all
from .trt_runtime import load_trt_modules
from .g2p import G2PPipeline, CustomDictionary, process_text
from .phoneme_tokens import PHONEME_LIST, N_PHONEMES, get_all_new_tokens
from .ssml import SSMLParser, SSMLSegment, parse_ssml, is_ssml