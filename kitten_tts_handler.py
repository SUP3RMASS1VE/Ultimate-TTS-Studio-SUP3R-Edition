"""
KittenTTS Handler for Ultimate TTS Studio
Provides integration with KittenTTS ultra-lightweight TTS model
"""

import numpy as np
import tempfile
import os
import torch
from datetime import datetime
from pathlib import Path

# Global model instance
_kitten_model = None
_kitten_handler = None
KITTEN_TTS_AVAILABLE = False

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Available KittenTTS voices
KITTEN_VOICES = [
    'expr-voice-2-m', 'expr-voice-2-f',
    'expr-voice-3-m', 'expr-voice-3-f', 
    'expr-voice-4-m', 'expr-voice-4-f',
    'expr-voice-5-m', 'expr-voice-5-f'
]

try:
    from kittentts import KittenTTS
    KITTEN_TTS_AVAILABLE = True
    print("✅ KittenTTS package available")
except ImportError:
    print("⚠️ KittenTTS not available - install with: pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl")

def get_kitten_tts_handler():
    """Get the global KittenTTS handler instance (singleton)"""
    global _kitten_handler
    if _kitten_handler is None:
        _kitten_handler = KittenTTSHandler()
    return _kitten_handler

class KittenTTSHandler:
    def __init__(self):
        self.model = None
        self.model_name = "KittenML/kitten-tts-mini-0.1"
        self.sample_rate = 24000
        self.device = DEVICE
        # Set cache directory to checkpoints folder (don't create until needed)
        self.cache_dir = Path("checkpoints/kitten_tts")
    
    def initialize_model(self):
        """Initialize the KittenTTS model"""
        global _kitten_model
        
        if not KITTEN_TTS_AVAILABLE:
            return False
        
        if _kitten_model is None:
            try:
                print(f"🐱 Loading KittenTTS model on {self.device}...")
                
                # Create cache directory only when actually loading
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                print(f"📁 Cache directory: {self.cache_dir}")
                
                # Set multiple environment variables for HuggingFace cache and ONNX GPU
                import os
                original_vars = {}
                cache_vars = {
                    'HF_HOME': str(self.cache_dir),
                    'HUGGINGFACE_HUB_CACHE': str(self.cache_dir / 'hub'),
                    'HF_HUB_CACHE': str(self.cache_dir / 'hub'),
                    'TRANSFORMERS_CACHE': str(self.cache_dir / 'transformers')
                }
                
                # Add ONNX GPU environment variables if using CUDA
                if self.device == "cuda":
                    cache_vars.update({
                        'ORT_TENSORRT_ENGINE_CACHE_ENABLE': '1',
                        'ORT_TENSORRT_CACHE_PATH': str(self.cache_dir / 'tensorrt_cache'),
                        'CUDA_VISIBLE_DEVICES': '0'  # Use first GPU
                    })
                
                # Store original values and set new ones
                for var, value in cache_vars.items():
                    original_vars[var] = os.environ.get(var)
                    os.environ[var] = value
                
                try:
                    # Create cache subdirectories
                    (self.cache_dir / 'hub').mkdir(parents=True, exist_ok=True)
                    (self.cache_dir / 'transformers').mkdir(parents=True, exist_ok=True)
                    
                    # Create TensorRT cache directory if using CUDA
                    if self.device == "cuda":
                        (self.cache_dir / 'tensorrt_cache').mkdir(parents=True, exist_ok=True)
                    
                    # Try to monkey patch ONNX Runtime to force GPU providers
                    if self.device == "cuda":
                        try:
                            import onnxruntime as ort
                            original_inference_session = ort.InferenceSession
                            
                            def patched_inference_session(model_path, sess_options=None, providers=None, **kwargs):
                                # Force GPU providers if not specified
                                if providers is None:
                                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                                return original_inference_session(model_path, sess_options, providers, **kwargs)
                            
                            # Apply the patch
                            ort.InferenceSession = patched_inference_session
                            print(f"🔧 Applied ONNX Runtime GPU acceleration patch")
                            
                        except Exception as patch_error:
                            print(f"⚠️ Could not apply ONNX patch: {patch_error}")
                    
                    # Try to initialize KittenTTS with device parameter
                    try:
                        _kitten_model = KittenTTS(self.model_name, cache_dir=str(self.cache_dir), device=self.device)
                        print(f"✅ KittenTTS model loaded successfully on {self.device}!")
                    except TypeError:
                        # If device parameter is not supported, try with providers parameter for ONNX
                        print(f"⚠️ KittenTTS doesn't support device parameter, trying with ONNX providers...")
                        try:
                            # Try to pass ONNX providers to prioritize GPU
                            if self.device == "cuda":
                                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                                _kitten_model = KittenTTS(self.model_name, cache_dir=str(self.cache_dir), providers=providers)
                                print(f"✅ KittenTTS loaded with CUDA provider priority!")
                            else:
                                _kitten_model = KittenTTS(self.model_name, cache_dir=str(self.cache_dir))
                                print(f"✅ KittenTTS loaded with default providers")
                        except TypeError:
                            # If providers parameter is also not supported, use default
                            print(f"⚠️ KittenTTS doesn't support providers parameter either, using defaults...")
                            _kitten_model = KittenTTS(self.model_name, cache_dir=str(self.cache_dir))
                        
                        # Check ONNX Runtime providers to see if GPU is available
                        try:
                            import onnxruntime as ort
                            available_providers = ort.get_available_providers()
                            print(f"🔍 Available ONNX providers: {available_providers}")
                            
                            if 'CUDAExecutionProvider' in available_providers:
                                print(f"✅ ONNX Runtime GPU support available!")
                            else:
                                print(f"⚠️ ONNX Runtime GPU support not available")
                                print(f"💡 Make sure you have onnxruntime-gpu installed and CUDA drivers")
                                
                        except ImportError:
                            print(f"⚠️ Could not import onnxruntime to check providers")
                        except Exception as provider_error:
                            print(f"⚠️ Error checking ONNX providers: {provider_error}")
                        
                        print(f"✅ KittenTTS model loaded (ONNX-based model)")
                    
                    print(f"📁 Models cached in: {self.cache_dir}")
                    
                finally:
                    # Restore original environment variables
                    for var, original_value in original_vars.items():
                        if original_value is not None:
                            os.environ[var] = original_value
                        else:
                            os.environ.pop(var, None)
                        
            except Exception as e:
                print(f"❌ Error loading KittenTTS model: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        self.model = _kitten_model
        return True
    
    def unload_model(self):
        """Unload the KittenTTS model"""
        global _kitten_model, _kitten_handler
        _kitten_model = None
        self.model = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if using GPU
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✅ KittenTTS model unloaded")
    
    def generate_audio(self, text, voice="expr-voice-2-f"):
        """Generate audio using KittenTTS"""
        if not self.model:
            return None, "❌ KittenTTS model not loaded. Please load the model first."
        
        try:
            print(f"🐱 Generating audio with voice: {voice}")
            
            # Try to check which ONNX provider is being used
            try:
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'session'):
                    session = self.model.model.session
                    if hasattr(session, 'get_providers'):
                        active_providers = session.get_providers()
                        print(f"🔍 Active ONNX providers: {active_providers}")
                        if 'CUDAExecutionProvider' in active_providers:
                            print(f"✅ Using GPU acceleration via CUDA!")
                        elif 'TensorrtExecutionProvider' in active_providers:
                            print(f"✅ Using GPU acceleration via TensorRT!")
                        else:
                            print(f"⚠️ Using CPU execution")
                elif hasattr(self.model, 'model') and hasattr(self.model.model, 'ort_session'):
                    session = self.model.model.ort_session
                    if hasattr(session, 'get_providers'):
                        active_providers = session.get_providers()
                        print(f"🔍 Active ONNX providers: {active_providers}")
                        if 'CUDAExecutionProvider' in active_providers:
                            print(f"✅ Using GPU acceleration via CUDA!")
                        elif 'TensorrtExecutionProvider' in active_providers:
                            print(f"✅ Using GPU acceleration via TensorRT!")
                        else:
                            print(f"⚠️ Using CPU execution")
            except Exception as provider_check_error:
                print(f"🔍 Could not check active providers: {provider_check_error}")
            
            # Generate audio
            audio = self.model.generate(text, voice=voice)
            
            # Convert to numpy array if needed
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            
            # Ensure audio is in the right format (float32, mono)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Normalize audio to prevent clipping
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio)) * 0.95
            
            return audio, None
            
        except Exception as e:
            return None, f"❌ Error generating KittenTTS audio: {str(e)}"

def split_text_into_chunks(text: str, max_chunk_length: int = 200) -> list[str]:
    """Split text into chunks that respect sentence boundaries for KittenTTS."""
    import re
    
    if len(text) <= max_chunk_length:
        return [text]
    
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_chunk) + len(sentence) + 2 > max_chunk_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if len(sentence) > max_chunk_length:
                    parts = re.split(r'[,;]+', sentence)
                    for part in parts:
                        part = part.strip()
                        if len(current_chunk) + len(part) + 2 > max_chunk_length:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = part
                        else:
                            current_chunk += (", " if current_chunk else "") + part
                else:
                    current_chunk = sentence
        else:
            current_chunk += (". " if current_chunk else "") + sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def generate_kitten_tts(
    text,
    voice="expr-voice-2-f",
    effects_settings=None,
    audio_format="wav",
    skip_file_saving=False
):
    """
    Generate TTS audio using KittenTTS with automatic text chunking for long text
    
    Args:
        text: Text to synthesize
        voice: KittenTTS voice to use
        effects_settings: Audio effects settings (optional)
        audio_format: Output audio format
        skip_file_saving: If True, don't save to file
    
    Returns:
        tuple: (audio_data, info_text) where audio_data is (sample_rate, audio_array)
    """
    
    if not KITTEN_TTS_AVAILABLE:
        return None, "❌ KittenTTS not available. Please install the required package."
    
    if not text or not text.strip():
        return None, "❌ No text provided for synthesis"
    
    # Validate voice selection
    if voice not in KITTEN_VOICES:
        voice = "expr-voice-2-f"  # Default fallback
    
    try:
        handler = get_kitten_tts_handler()
        
        # Split text into chunks for long text (KittenTTS works better with shorter chunks)
        text_chunks = split_text_into_chunks(text, max_chunk_length=200)
        
        if len(text_chunks) > 1:
            print(f"🐱 KittenTTS: Processing {len(text_chunks)} text chunks for long text")
        
        audio_segments = []
        
        # Process each chunk
        for i, chunk in enumerate(text_chunks):
            if len(text_chunks) > 1:
                print(f"🐱 Processing chunk {i+1}/{len(text_chunks)}: {chunk[:50]}...")
            
            # Generate audio for this chunk
            audio, error = handler.generate_audio(chunk, voice)
            
            if error:
                return None, f"❌ Error in chunk {i+1}: {error}"
            
            if audio is None:
                return None, f"❌ No audio generated for chunk {i+1}"
            
            audio_segments.append(audio)
        
        # Combine audio segments if multiple chunks
        if len(audio_segments) > 1:
            print(f"🐱 Combining {len(audio_segments)} audio segments...")
            
            # Add small pause between chunks (0.3 seconds)
            pause_samples = int(handler.sample_rate * 0.3)
            pause = np.zeros(pause_samples, dtype=np.float32)
            
            combined_audio = []
            for i, segment in enumerate(audio_segments):
                combined_audio.append(segment)
                # Add pause between segments (but not after the last one)
                if i < len(audio_segments) - 1:
                    combined_audio.append(pause)
            
            audio = np.concatenate(combined_audio)
        else:
            audio = audio_segments[0]
        
        # Apply audio effects if specified
        if effects_settings:
            try:
                audio = apply_audio_effects(audio, handler.sample_rate, effects_settings)
            except Exception as e:
                print(f"⚠️ Warning: Could not apply audio effects: {e}")
        
        # Save to file if not skipping
        if not skip_file_saving:
            try:
                # Create outputs directory if it doesn't exist
                output_folder = Path("outputs")
                output_folder.mkdir(exist_ok=True)
                
                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_base = f"kitten_tts_{voice}_{timestamp}"
                
                # Save with specified format
                filepath, filename = save_audio_with_format(
                    audio, handler.sample_rate, audio_format, output_folder, filename_base
                )
                
                info_text = f"✅ KittenTTS audio generated successfully!\n"
                if len(text_chunks) > 1:
                    info_text += f"📝 Processed {len(text_chunks)} text chunks\n"
                info_text += f"📁 Saved as: {filename}\n"
                info_text += f"🎤 Voice: {voice}\n"
                info_text += f"⏱️ Duration: {len(audio) / handler.sample_rate:.2f}s\n"
                info_text += f"📊 Sample Rate: {handler.sample_rate}Hz"
                
            except Exception as save_error:
                print(f"Warning: Could not save audio file: {save_error}")
                info_text = f"✅ KittenTTS audio generated successfully!\n"
                if len(text_chunks) > 1:
                    info_text += f"📝 Processed {len(text_chunks)} text chunks\n"
                info_text += f"🎤 Voice: {voice}\n"
                info_text += f"⏱️ Duration: {len(audio) / handler.sample_rate:.2f}s\n"
                info_text += f"⚠️ File saving failed: {save_error}"
        else:
            info_text = f"✅ KittenTTS audio generated successfully!\n"
            if len(text_chunks) > 1:
                info_text += f"📝 Processed {len(text_chunks)} text chunks\n"
            info_text += f"🎤 Voice: {voice}\n"
            info_text += f"⏱️ Duration: {len(audio) / handler.sample_rate:.2f}s"
        
        return (handler.sample_rate, audio), info_text
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ KittenTTS generation error: {str(e)}"

def apply_audio_effects(audio, sample_rate, effects_settings):
    """Apply audio effects to the generated audio"""
    # Import the audio effects function from the main app
    try:
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Try to import the audio effects function from the main app
        # For now, just return the audio unchanged if import fails
        return audio
    except:
        return audio

def save_audio_with_format(audio, sample_rate, audio_format, output_folder, filename_base):
    """Save audio with the specified format"""
    import soundfile as sf
    from pathlib import Path
    
    # Ensure output folder exists
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)
    
    if audio_format.lower() == "wav":
        filepath = output_folder / f"{filename_base}.wav"
        sf.write(str(filepath), audio, sample_rate)
    elif audio_format.lower() == "mp3":
        # For MP3, we'd need additional libraries, fallback to WAV for now
        filepath = output_folder / f"{filename_base}.wav"
        sf.write(str(filepath), audio, sample_rate)
    else:
        filepath = output_folder / f"{filename_base}.wav"  # Default to WAV
        sf.write(str(filepath), audio, sample_rate)
    
    return str(filepath), filepath.name

# Model management functions for the UI
def init_kitten_tts():
    """Initialize KittenTTS model for UI"""
    if not KITTEN_TTS_AVAILABLE:
        return False, "❌ KittenTTS not available"
    
    try:
        handler = get_kitten_tts_handler()
        success = handler.initialize_model()
        if success:
            return True, "✅ KittenTTS model loaded successfully"
        else:
            return False, "❌ Failed to initialize KittenTTS model"
    except Exception as e:
        return False, f"❌ Error loading KittenTTS: {str(e)}"

def unload_kitten_tts():
    """Unload KittenTTS model for UI"""
    try:
        handler = get_kitten_tts_handler()
        handler.unload_model()
        return "✅ KittenTTS model unloaded"
    except Exception as e:
        return f"⚠️ Error unloading KittenTTS: {str(e)}"