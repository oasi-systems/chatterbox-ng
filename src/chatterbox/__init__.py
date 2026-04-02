try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # For Python <3.8

__version__ = version("chatterbox-ng")


from .tts import ChatterboxTTS
from .vc import ChatterboxVC
from .mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
from .streaming import ChatterboxStreamingTTS
from .audio_processing import post_process, lufs_normalize, de_ess, match_room_tone
from .cuda_optimizations import optimize_for_cuda