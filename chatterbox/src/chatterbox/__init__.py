try:
    from importlib.metadata import version
    __version__ = version("chatterbox-tts")
except:
    # Fallback for local installations without package metadata
    __version__ = "local"

from .tts import ChatterboxTTS
from .vc import ChatterboxVC
from .mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES