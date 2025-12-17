"""
Chatterbox Turbo TTS Handler for Ultimate TTS Studio
Provides integration with Chatterbox Turbo - a faster distilled version of ChatterboxTTS
Model: SUP3RMASS1VE/turbo-chatterbox
"""

import os
import sys
import warnings
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Add current directory to path (same as launch.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Try to import Chatterbox Turbo TTS
try:
    from chatterbox.src.chatterbox.tts_turbo import ChatterboxTurboTTS
    CHATTERBOX_BASE_AVAILABLE = True
except ImportError as e:
    CHATTERBOX_BASE_AVAILABLE = False
    print(f"⚠️ ChatterboxTurboTTS not available: {e}")
    print("   Make sure the chatterbox folder exists with the required files")

# Import for audio file saving
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    try:
        from scipy.io.wavfile import write as wav_write
        SOUNDFILE_AVAILABLE = False
    except ImportError:
        SOUNDFILE_AVAILABLE = None

# Import for MP3 conversion
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


# Turbo model repository
TURBO_REPO_ID = "SUP3RMASS1VE/turbo-chatterbox"
TURBO_MODEL_FILES = [
    ".gitattributes",
    "added_tokens.json",
    "conds.pt",
    "merges.txt",
    "s3gen.safetensors",
    "s3gen_meanflow.safetensors",
    "special_tokens_map.json",
    "t3_turbo_v1.safetensors",
    "t3_turbo_v1.yaml",
    "tokenizer_config.json",
    "ve.safetensors",
    "vocab.json"
]


class ChatterboxTurboHandler:
    """Handler for Chatterbox Turbo TTS system"""
    
    def __init__(self):
        self.model = None
        self.device = self._get_device()
        self.sample_rate = 24000  # Chatterbox default sample rate
        self.model_dir = Path(current_dir) / "checkpoints" / "chatterbox_turbo"
        self.is_loaded = False
        
    def _get_device(self) -> str:
        """Get the best available device"""
        if torch.cuda.is_available():
            try:
                test_tensor = torch.tensor([1.0], device="cuda")
                del test_tensor
                return "cuda"
            except Exception:
                return "cpu"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def check_models_downloaded(self) -> bool:
        """Check if essential turbo model files are downloaded"""
        if not self.model_dir.exists():
            return False
        
        # Only check essential model files (the large safetensors files)
        essential_files = [
            "ve.safetensors",
            "s3gen.safetensors",
            "s3gen_meanflow.safetensors",
            "t3_turbo_v1.safetensors",
            "t3_turbo_v1.yaml",
            "conds.pt"
        ]
        
        for filename in essential_files:
            if not (self.model_dir / filename).exists():
                return False
        return True
    
    def download_models(self, progress_callback=None) -> Tuple[bool, str]:
        """Download Chatterbox Turbo model files from HuggingFace"""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            return False, "❌ Please install huggingface-hub: pip install huggingface-hub"
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"📥 Downloading Chatterbox Turbo model")
        print(f"📦 Repository: {TURBO_REPO_ID}")
        print(f"📁 Target: {self.model_dir}")
        print(f"{'='*60}\n")
        
        try:
            for i, filename in enumerate(TURBO_MODEL_FILES, 1):
                file_path = self.model_dir / filename
                
                if file_path.exists():
                    print(f"✅ [{i}/{len(TURBO_MODEL_FILES)}] {filename} already exists")
                    if progress_callback:
                        progress_callback(f"✓ {filename} already exists")
                    continue
                
                print(f"⬇️  [{i}/{len(TURBO_MODEL_FILES)}] Downloading {filename}...")
                if progress_callback:
                    progress_callback(f"Downloading {filename}...")
                
                downloaded_path = hf_hub_download(
                    repo_id=TURBO_REPO_ID,
                    filename=filename,
                    local_dir=str(self.model_dir),
                    local_dir_use_symlinks=False
                )
                
                print(f"✅ [{i}/{len(TURBO_MODEL_FILES)}] Downloaded {filename}")
                if progress_callback:
                    progress_callback(f"✓ Downloaded {filename}")
            
            print(f"\n{'='*60}")
            print(f"✅ Chatterbox Turbo model downloaded successfully")
            print(f"{'='*60}\n")
            
            return True, "✅ Chatterbox Turbo model downloaded successfully"
            
        except Exception as e:
            error_msg = f"❌ Failed to download Chatterbox Turbo: {str(e)}"
            print(error_msg)
            return False, error_msg
    
    def initialize(self, auto_download: bool = True) -> Tuple[bool, str]:
        """Initialize the Chatterbox Turbo model
        
        Args:
            auto_download: If True, download models if not found. If False, fail if models missing.
        """
        if not CHATTERBOX_BASE_AVAILABLE:
            return False, "❌ ChatterboxTTS base library not available"
        
        if self.is_loaded and self.model is not None:
            return True, "✅ Chatterbox Turbo already loaded"
        
        # Check if models are downloaded
        if not self.check_models_downloaded():
            if auto_download:
                print("🔄 Turbo model not found, downloading...")
                success, msg = self.download_models()
                if not success:
                    return False, msg
            else:
                return False, "❌ Chatterbox Turbo models not downloaded. Click 'Load' to download."
        
        try:
            print(f"🚀 Loading Chatterbox Turbo on {self.device}...")
            
            # Load from local checkpoint directory
            self.model = ChatterboxTurboTTS.from_local(str(self.model_dir), self.device)
            self.sample_rate = self.model.sr
            self.is_loaded = True
            
            print(f"✅ Chatterbox Turbo loaded successfully (sample rate: {self.sample_rate})")
            return True, "✅ Chatterbox Turbo loaded successfully"
            
        except Exception as e:
            error_msg = f"❌ Failed to load Chatterbox Turbo: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return False, error_msg
    
    def unload(self) -> str:
        """Unload the model to free memory"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            self.is_loaded = False
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return "✅ Chatterbox Turbo unloaded - memory freed"
        except Exception as e:
            return f"⚠️ Error unloading Chatterbox Turbo: {str(e)}"
    
    def generate_speech(
        self,
        text: str,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        cfg_weight: float = 0.5,
        repetition_penalty: float = 1.2,
        min_p: float = 0.05,
        top_p: float = 1.0,
        seed: Optional[int] = None
    ) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
        """
        Generate speech using Chatterbox Turbo
        
        Args:
            text: Text to synthesize
            audio_prompt_path: Path to reference audio for voice cloning
            exaggeration: Emotion exaggeration (0.0-1.0)
            temperature: Sampling temperature
            cfg_weight: Classifier-free guidance weight
            repetition_penalty: Repetition penalty for generation
            min_p: Minimum probability threshold
            top_p: Top-p sampling threshold
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (sample_rate, audio_array) and status message
        """
        if not self.is_loaded or self.model is None:
            return None, "❌ Chatterbox Turbo not loaded - please load the model first"
        
        if not text.strip():
            return None, "❌ Please provide text to synthesize"
        
        try:
            # Set seed if provided
            if seed is not None and seed != 0:
                torch.manual_seed(seed)
                if self.device == "cuda":
                    torch.cuda.manual_seed(seed)
                np.random.seed(seed)
            
            print(f"🚀 Generating with Chatterbox Turbo: {text[:50]}...")
            
            # Generate audio
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                
                wav = self.model.generate(
                    text,
                    audio_prompt_path=audio_prompt_path if audio_prompt_path else None,
                    exaggeration=exaggeration,
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                    repetition_penalty=repetition_penalty,
                    min_p=min_p,
                    top_p=top_p,
                )
            
            # Convert to numpy
            audio_np = wav.squeeze(0).numpy()
            
            return (self.sample_rate, audio_np), "✅ Generated with Chatterbox Turbo"
            
        except Exception as e:
            error_msg = f"❌ Chatterbox Turbo error: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return None, error_msg
    
    def get_status(self) -> Dict[str, Any]:
        """Get current handler status"""
        return {
            'available': CHATTERBOX_BASE_AVAILABLE,
            'loaded': self.is_loaded,
            'device': self.device,
            'sample_rate': self.sample_rate,
            'models_downloaded': self.check_models_downloaded(),
            'model_dir': str(self.model_dir)
        }


# Global handler instance
_handler_instance = None


def get_chatterbox_turbo_handler() -> ChatterboxTurboHandler:
    """Get or create the global handler instance"""
    global _handler_instance
    if _handler_instance is None:
        _handler_instance = ChatterboxTurboHandler()
    return _handler_instance


def init_chatterbox_turbo(auto_download: bool = True) -> Tuple[bool, str]:
    """Initialize Chatterbox Turbo model
    
    Args:
        auto_download: If True, download models if not found (default behavior when clicking Load)
    """
    handler = get_chatterbox_turbo_handler()
    return handler.initialize(auto_download=auto_download)


def unload_chatterbox_turbo() -> str:
    """Unload Chatterbox Turbo model"""
    handler = get_chatterbox_turbo_handler()
    return handler.unload()


def generate_chatterbox_turbo_tts(
    text: str,
    audio_prompt_path: Optional[str] = None,
    exaggeration: float = 0.5,
    temperature: float = 0.8,
    cfg_weight: float = 0.5,
    repetition_penalty: float = 1.2,
    min_p: float = 0.05,
    top_p: float = 1.0,
    seed: Optional[int] = None,
    chunk_size: int = 300,
    effects_settings: Optional[Dict] = None,
    audio_format: str = "wav",
    skip_file_saving: bool = False
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    """
    Generate TTS audio using Chatterbox Turbo with chunking support
    
    This is the main function to be called from launch.py
    """
    handler = get_chatterbox_turbo_handler()
    
    if not handler.is_loaded:
        return None, "❌ Chatterbox Turbo not loaded - please click 'Load' button first"
    
    try:
        # Split text into chunks if needed
        text_chunks = split_text_into_chunks(text, max_chunk_length=chunk_size)
        audio_chunks = []
        
        print(f"🚀 Generating Chatterbox Turbo audio for {len(text_chunks)} chunk(s)...")
        
        for i, chunk in enumerate(text_chunks):
            if len(text_chunks) > 1:
                print(f"📝 Processing chunk {i+1}/{len(text_chunks)}: {chunk[:50]}...")
            
            result, msg = handler.generate_speech(
                chunk,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                seed=seed
            )
            
            if result is None:
                return None, msg
            
            audio_chunks.append(result[1])
            
            if len(text_chunks) > 1:
                print(f"✅ Chunk {i+1}/{len(text_chunks)} completed")
        
        # Concatenate chunks
        if len(audio_chunks) == 1:
            final_audio = audio_chunks[0]
        else:
            silence_samples = int(handler.sample_rate * 0.05)
            silence = np.zeros(silence_samples)
            
            concatenated = []
            for i, chunk in enumerate(audio_chunks):
                concatenated.append(chunk)
                if i < len(audio_chunks) - 1:
                    concatenated.append(silence)
            
            final_audio = np.concatenate(concatenated)
        
        # Apply effects if provided
        if effects_settings:
            try:
                # Import apply_audio_effects from launch.py context
                from launch import apply_audio_effects
                final_audio = apply_audio_effects(final_audio, handler.sample_rate, effects_settings)
            except ImportError:
                pass  # Effects not available
        
        # Save audio file if not skipped
        if skip_file_saving:
            status_message = "✅ Generated with Chatterbox Turbo"
        else:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_base = f"chatterbox_turbo_{timestamp}"
                filepath, filename = save_turbo_audio(
                    final_audio, handler.sample_rate, audio_format, "outputs", filename_base
                )
                status_message = f"✅ Generated with Chatterbox Turbo - Saved as: {filename}"
            except Exception as e:
                print(f"Warning: Could not save audio file: {e}")
                status_message = "✅ Generated with Chatterbox Turbo (file saving failed)"
        
        return (handler.sample_rate, final_audio), status_message
        
    except Exception as e:
        return None, f"❌ Chatterbox Turbo error: {str(e)}"


def split_text_into_chunks(text: str, max_chunk_length: int = 300) -> list:
    """Split text into chunks at sentence boundaries"""
    if len(text) <= max_chunk_length:
        return [text]
    
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if not sentence.strip():
            continue
        
        if current_chunk and len(current_chunk + " " + sentence) > max_chunk_length:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text]


def save_turbo_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    audio_format: str = "wav",
    output_folder: str = "outputs",
    filename_base: str = None
) -> Tuple[str, str]:
    """Save audio to file"""
    if filename_base is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"chatterbox_turbo_{timestamp}"
    
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    audio_format = audio_format.lower()
    
    if audio_format == "wav":
        filename = f"{filename_base}.wav"
        filepath = output_path / filename
        
        if SOUNDFILE_AVAILABLE:
            sf.write(str(filepath), audio_data, sample_rate)
        elif SOUNDFILE_AVAILABLE is False:
            from scipy.io.wavfile import write
            # Convert to int16 for scipy
            audio_int16 = (audio_data * 32767).astype(np.int16)
            write(str(filepath), sample_rate, audio_int16)
        else:
            raise RuntimeError("No audio writing library available")
    
    elif audio_format == "mp3":
        if not PYDUB_AVAILABLE:
            # Fallback to WAV
            filename = f"{filename_base}.wav"
            filepath = output_path / filename
            if SOUNDFILE_AVAILABLE:
                sf.write(str(filepath), audio_data, sample_rate)
            else:
                from scipy.io.wavfile import write
                audio_int16 = (audio_data * 32767).astype(np.int16)
                write(str(filepath), sample_rate, audio_int16)
        else:
            import tempfile
            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_wav.close()
            
            try:
                if SOUNDFILE_AVAILABLE:
                    sf.write(temp_wav.name, audio_data, sample_rate)
                else:
                    from scipy.io.wavfile import write
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    write(temp_wav.name, sample_rate, audio_int16)
                
                audio_segment = AudioSegment.from_wav(temp_wav.name)
                filename = f"{filename_base}.mp3"
                filepath = output_path / filename
                audio_segment.export(str(filepath), format="mp3", bitrate="320k")
            finally:
                try:
                    os.unlink(temp_wav.name)
                except:
                    pass
    else:
        # Default to WAV
        filename = f"{filename_base}.wav"
        filepath = output_path / filename
        if SOUNDFILE_AVAILABLE:
            sf.write(str(filepath), audio_data, sample_rate)
        else:
            from scipy.io.wavfile import write
            audio_int16 = (audio_data * 32767).astype(np.int16)
            write(str(filepath), sample_rate, audio_int16)
    
    return str(filepath), filename


def get_chatterbox_turbo_status() -> str:
    """Get status string for UI"""
    handler = get_chatterbox_turbo_handler()
    status = handler.get_status()
    
    if not status['available']:
        return "❌ ChatterboxTTS base library not available"
    
    if status['loaded']:
        return f"✅ Loaded on {status['device']} (SR: {status['sample_rate']}Hz)"
    
    if status['models_downloaded']:
        return "⏸️ Ready to load (click Load)"
    
    return "📥 Click Load to download (~4.8GB)"


# Check availability on import
CHATTERBOX_TURBO_AVAILABLE = CHATTERBOX_BASE_AVAILABLE
if CHATTERBOX_TURBO_AVAILABLE:
    print("✅ Chatterbox Turbo handler loaded")
else:
    print("⚠️ Chatterbox Turbo not available - ChatterboxTTS base library required")
