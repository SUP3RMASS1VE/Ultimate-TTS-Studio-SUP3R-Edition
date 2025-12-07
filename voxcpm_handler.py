"""
VoxCPM Handler for Ultimate TTS Studio
Provides integration with VoxCPM Text-to-Speech system with voice cloning capabilities
"""

import os
import sys
import warnings
import numpy as np
import torch
import tempfile
import json
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, List
from datetime import datetime
import librosa
import soundfile as sf
import time

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup VoxCPM cache directories
current_dir = os.path.dirname(os.path.abspath(__file__))
voxcpm_cache_dir = os.path.join(current_dir, 'checkpoints', 'voxcpm', 'cache')
os.makedirs(voxcpm_cache_dir, exist_ok=True)

# Store original environment variables to restore later
original_env = {
    'MODELSCOPE_CACHE': os.environ.get('MODELSCOPE_CACHE'),
    'HF_HOME': os.environ.get('HF_HOME'),
    'TRANSFORMERS_CACHE': os.environ.get('TRANSFORMERS_CACHE'),
    'HF_DATASETS_CACHE': os.environ.get('HF_DATASETS_CACHE')
}

# Global handler instance
_voxcpm_handler = None
VOXCPM_AVAILABLE = False

# Try to import VoxCPM and Whisper
def try_import_voxcpm():
    """Try to import VoxCPM and dependencies"""
    global VOXCPM_AVAILABLE
    
    try:
        from voxcpm import VoxCPM
        import whisper
        VOXCPM_AVAILABLE = True
        print("✅ VoxCPM loaded successfully")
        return VoxCPM, whisper
    except ImportError as e:
        print(f"❌ VoxCPM not available: {e}")
        print("   Install with: pip install voxcpm openai-whisper")
        VOXCPM_AVAILABLE = False
        return None, None

# Try to import VoxCPM
VoxCPM, whisper = try_import_voxcpm()

def get_voxcpm_handler():
    """Get the global VoxCPM handler instance (singleton)"""
    global _voxcpm_handler
    if _voxcpm_handler is None:
        _voxcpm_handler = VoxCPMHandler()
    return _voxcpm_handler

class VoxCPMHandler:
    """Handler for VoxCPM TTS system with voice cloning capabilities"""
    
    def __init__(self):
        self.model = None
        self.whisper_model = None
        self.device = self._get_device()
        self.sample_rate = 44100  # VoxCPM1.5 uses 44.1kHz Audio VAE
        self.model_id = "openbmb/VoxCPM1.5"
        self.checkpoints_dir = Path("checkpoints/voxcpm")
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Create cache subdirectory
        self.cache_dir = self.checkpoints_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Conversation mode state
        self.conversation_mode = False
        self.conversation_seed = None
        self.conversation_reference_audio = None
        self.conversation_reference_text = None
        
        # Per-speaker seed tracking for conversations
        self.speaker_seeds = {}  # Maps speaker name to their consistent seed
        
        # Enable faster matmul precision if GPU supports it
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('high')
        
    def _get_device(self):
        """Get the appropriate device for inference"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def initialize_model(self):
        """Initialize the VoxCPM model and Whisper"""
        if not VOXCPM_AVAILABLE or VoxCPM is None:
            return False, "❌ VoxCPM not available"
        
        try:
            print("🎯 Initializing VoxCPM model...")
            
            # Set VoxCPM-specific cache directories
            voxcpm_cache = str(self.checkpoints_dir / "cache")
            os.makedirs(voxcpm_cache, exist_ok=True)
            
            # Temporarily set environment variables for VoxCPM
            temp_env = {
                'MODELSCOPE_CACHE': voxcpm_cache,
                'MODELSCOPE_LOG_LEVEL': '40',
                'HF_HOME': voxcpm_cache,
                'TRANSFORMERS_CACHE': os.path.join(voxcpm_cache, 'transformers'),
                'HF_DATASETS_CACHE': os.path.join(voxcpm_cache, 'datasets')
            }
            
            # Store current env and set VoxCPM env
            current_env = {}
            for key, value in temp_env.items():
                current_env[key] = os.environ.get(key)
                os.environ[key] = value
            
            try:
                # Load VoxCPM model
                print(f"📥 Loading VoxCPM model from {self.model_id}...")
                print(f"📁 Using cache directory: {voxcpm_cache}")
                self.model = VoxCPM.from_pretrained(self.model_id, cache_dir=str(self.checkpoints_dir))
                print(f"✅ VoxCPM model loaded")
                
                # Load Whisper model for transcription
                print("📥 Loading Whisper model for transcription...")
                whisper_cache_dir = self.checkpoints_dir / "whisper"
                whisper_cache_dir.mkdir(exist_ok=True)
                self.whisper_model = whisper.load_model("tiny", download_root=str(whisper_cache_dir))
                print("✅ Whisper model loaded")
                
            finally:
                # Restore original environment variables
                for key, value in current_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            
            return True, "✅ VoxCPM models loaded successfully"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ Error initializing VoxCPM: {e}")
            return False, f"❌ Error initializing VoxCPM: {str(e)}"
    
    def unload_model(self):
        """Unload the VoxCPM model to free memory"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.whisper_model is not None:
                del self.whisper_model
                self.whisper_model = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return "✅ VoxCPM models unloaded successfully"
            
        except Exception as e:
            return f"⚠️ Error unloading VoxCPM: {str(e)}"
    
    def is_model_loaded(self):
        """Check if the model is loaded"""
        return self.model is not None
    
    def start_conversation(self, seed: Optional[int] = None, reference_audio: Optional[str] = None, reference_text: Optional[str] = None):
        """
        Start conversation mode with consistent seed and voice across all turns
        
        Args:
            seed: Seed to use for all conversation turns (if None, generates random seed)
            reference_audio: Reference audio for voice cloning (optional)
            reference_text: Reference text for voice cloning (optional, will auto-transcribe if not provided)
        """
        self.conversation_mode = True
        
        # Set or generate conversation seed
        if seed is not None and seed != -1:
            self.conversation_seed = int(seed)
        else:
            self.conversation_seed = np.random.randint(0, 2147483647)
        
        # Set conversation voice reference
        self.conversation_reference_audio = reference_audio
        
        # Handle reference text
        if reference_audio and os.path.exists(reference_audio):
            if not reference_text or not reference_text.strip():
                # Auto-transcribe if reference text not provided
                self.conversation_reference_text = self.transcribe_audio(reference_audio)
            else:
                self.conversation_reference_text = reference_text.strip()
        else:
            self.conversation_reference_text = None
        
        print(f"🎭 Started conversation mode with seed: {self.conversation_seed}")
        if self.conversation_reference_audio:
            print(f"🎤 Using voice reference: {self.conversation_reference_audio}")
            if self.conversation_reference_text:
                print(f"📝 Reference text: {self.conversation_reference_text}")
        else:
            print("🎵 Using default voice for conversation")
        
        return {
            'seed': self.conversation_seed,
            'reference_audio': self.conversation_reference_audio,
            'reference_text': self.conversation_reference_text
        }
    
    def end_conversation(self):
        """End conversation mode"""
        self.conversation_mode = False
        self.conversation_seed = None
        self.conversation_reference_audio = None
        self.conversation_reference_text = None
        self.speaker_seeds.clear()  # Clear per-speaker seeds
        print("🎭 Ended conversation mode")
    
    def is_in_conversation(self):
        """Check if currently in conversation mode"""
        return self.conversation_mode
    
    def get_conversation_info(self):
        """Get current conversation mode information"""
        if not self.conversation_mode:
            return None
        
        return {
            'active': True,
            'seed': self.conversation_seed,
            'reference_audio': self.conversation_reference_audio,
            'reference_text': self.conversation_reference_text,
            'speaker_seeds': self.speaker_seeds.copy()
        }
    
    def get_or_set_speaker_seed(self, speaker_name: str, provided_seed: Optional[int] = None) -> int:
        """
        Get existing seed for speaker or set a new one if first time
        
        Args:
            speaker_name: Name of the speaker
            provided_seed: Optional seed to use (if None, generates random)
            
        Returns:
            The seed to use for this speaker
        """
        if speaker_name in self.speaker_seeds:
            # Speaker already has a seed, use it
            seed = self.speaker_seeds[speaker_name]
            print(f"🎭 Using existing seed {seed} for speaker '{speaker_name}'")
            return seed
        else:
            # First time for this speaker, set their seed
            if provided_seed is not None and provided_seed != -1:
                seed = int(provided_seed)
            else:
                seed = np.random.randint(0, 2147483647)
            
            self.speaker_seeds[speaker_name] = seed
            print(f"🎭 Set new seed {seed} for speaker '{speaker_name}'")
            return seed
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using Whisper"""
        if not audio_path or not os.path.exists(audio_path):
            return ""
        
        # Initialize Whisper if not loaded
        if self.whisper_model is None:
            try:
                print("📥 Loading Whisper model for transcription...")
                whisper_cache_dir = self.checkpoints_dir / "whisper"
                whisper_cache_dir.mkdir(exist_ok=True)
                self.whisper_model = whisper.load_model("tiny", download_root=str(whisper_cache_dir))
                print("✅ Whisper model loaded")
            except Exception as e:
                print(f"❌ Failed to load Whisper: {e}")
                return ""
        
        try:
            print(f"🎤 Transcribing audio: {audio_path}")
            result = self.whisper_model.transcribe(audio_path)
            transcription = result["text"].strip()
            print(f"📝 Transcription: {transcription}")
            return transcription
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            return ""
    
    def chunk_text(self, text: str, max_chars: int = 500) -> List[str]:
        """Split text into chunks that won't overflow the KV cache"""
        # For ebook content, use more conservative chunking to avoid badcase issues
        conservative_max = min(max_chars, 400)  # Even smaller chunks for ebooks
        
        if len(text) <= conservative_max:
            return [text]
        
        chunks = []
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed conservative_max, save current chunk
            if len(current_chunk) + len(sentence) + 1 > conservative_max and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def generate_speech(
        self,
        text: str,
        reference_audio: Optional[str] = None,
        reference_text: Optional[str] = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        normalize: bool = True,
        denoise: bool = True,
        retry_badcase: bool = True,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        seed: Optional[int] = None,
        max_chars_per_chunk: int = 500,
        speaker_name: Optional[str] = None
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        Generate speech using VoxCPM with optional voice cloning
        
        Args:
            text: Text to synthesize
            reference_audio: Path to reference audio for voice cloning
            reference_text: Reference text (if None, will auto-transcribe)
            cfg_value: LM guidance on LocDiT, higher for better adherence to prompt
            inference_timesteps: Higher for better quality, lower for faster speed
            normalize: Enable external TN tool
            denoise: Enable external Denoise tool
            retry_badcase: Enable retrying for bad cases
            retry_badcase_max_times: Maximum retry attempts
            retry_badcase_ratio_threshold: Retry ratio threshold
            seed: Random seed for reproducibility
            max_chars_per_chunk: Maximum characters per text chunk
        """
        if not self.is_model_loaded():
            return None, "❌ VoxCPM model not loaded. Please initialize first."
        
        if not text or not text.strip():
            return None, "❌ No text provided"
        
        try:
            print(f"🎯 Generating speech with VoxCPM...")
            print(f"   Text: {text[:50]}...")
            print(f"   Parameters: cfg={cfg_value}, steps={inference_timesteps}")
            
            # Handle reference audio and text
            prompt_wav_path = None
            final_prompt_text = None
            
            # In conversation mode, use conversation settings unless explicitly overridden
            if self.conversation_mode:
                if reference_audio is None and self.conversation_reference_audio:
                    reference_audio = self.conversation_reference_audio
                if reference_text is None and self.conversation_reference_text:
                    reference_text = self.conversation_reference_text
                print(f"🎭 Conversation mode: using consistent voice settings")
            
            if reference_audio and os.path.exists(reference_audio):
                prompt_wav_path = reference_audio
                
                # Auto-transcribe if reference text not provided
                if not reference_text or not reference_text.strip():
                    print("🎤 Auto-transcribing reference audio...")
                    final_prompt_text = self.transcribe_audio(reference_audio)
                    if not final_prompt_text:
                        print("⚠️ Failed to transcribe reference audio, using default voice")
                        prompt_wav_path = None
                else:
                    final_prompt_text = reference_text.strip()
                
                if prompt_wav_path and final_prompt_text:
                    print(f"🎤 Using voice cloning with audio: {prompt_wav_path}")
                    print(f"📝 Using prompt text: {final_prompt_text}")
                else:
                    print("🎵 Using default voice generation")
            else:
                print("🎵 Using default voice generation (no reference audio)")
            
            # Split long text into chunks to avoid KV cache overflow
            text_chunks = self.chunk_text(text, max_chars_per_chunk)
            print(f"📄 Split text into {len(text_chunks)} chunks")
            
            # Determine the seed to use for ALL chunks
            if speaker_name and (self.conversation_mode or len(self.speaker_seeds) > 0):
                # Use per-speaker seed for consistency
                actual_seed = self.get_or_set_speaker_seed(speaker_name, seed)
            elif self.conversation_mode and self.conversation_seed is not None:
                # In conversation mode, use the conversation seed as fallback
                actual_seed = self.conversation_seed
                print(f"🎭 Using conversation seed: {actual_seed}")
            elif seed is not None and seed != -1:
                actual_seed = int(seed)
                print(f"🎲 Using provided seed: {actual_seed}")
            else:
                actual_seed = np.random.randint(0, 2147483647)
                print(f"🎲 Generated random seed: {actual_seed}")
            
            all_wavs = []
            
            for i, chunk in enumerate(text_chunks):
                print(f"🎵 Generating speech for chunk {i+1}/{len(text_chunks)}: {chunk[:50]}...")
                
                # Set the SAME seed for every chunk for consistency
                torch.manual_seed(actual_seed)
                np.random.seed(actual_seed)
                print(f"🎲 Using seed {actual_seed} for chunk {i+1}")
                
                # Generate speech for this chunk
                wav = self.model.generate(
                    text=chunk,
                    prompt_wav_path=prompt_wav_path,
                    prompt_text=final_prompt_text,
                    cfg_value=cfg_value,
                    inference_timesteps=int(inference_timesteps),
                    normalize=normalize,
                    denoise=denoise,
                    retry_badcase=retry_badcase,
                    retry_badcase_max_times=int(retry_badcase_max_times),
                    retry_badcase_ratio_threshold=retry_badcase_ratio_threshold
                )
                
                # Convert to numpy if needed
                if torch.is_tensor(wav):
                    wav = wav.cpu().numpy()
                
                # Ensure it's 1D array
                if wav.ndim > 1:
                    wav = wav.squeeze()
                
                all_wavs.append(wav)
                
                # Clear any cached states between chunks to prevent memory buildup
                if hasattr(self.model, 'tts_model') and hasattr(self.model.tts_model, 'base_lm'):
                    if hasattr(self.model.tts_model.base_lm, 'kv_cache'):
                        try:
                            self.model.tts_model.base_lm.kv_cache.clear()
                            print(f"🧹 Cleared KV cache after chunk {i+1}")
                        except:
                            pass  # If clearing fails, continue anyway
            
            # Concatenate all audio chunks
            if len(all_wavs) > 1:
                # Add small silence between chunks (0.2 seconds)
                silence = np.zeros(int(self.sample_rate * 0.2))
                wav = np.concatenate([np.concatenate([chunk, silence]) for chunk in all_wavs[:-1]] + [all_wavs[-1]])
                print(f"🔗 Concatenated {len(all_wavs)} audio chunks")
            else:
                wav = all_wavs[0]
            
            print(f"✅ Speech generation completed!")
            print(f"Generated wav shape: {wav.shape}, dtype: {wav.dtype}")
            
            # Normalize audio to prevent clipping
            if wav.max() > 1.0 or wav.min() < -1.0:
                wav = wav / np.max(np.abs(wav))
            
            return wav, "✅ VoxCPM speech generated successfully"
            
        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            print(f"❌ Full traceback:\n{tb_str}")
            error_msg = str(e) if str(e) else f"Exception type: {type(e).__name__}"
            print(f"❌ Error generating speech: {error_msg}")
            return None, f"❌ Error generating speech: {error_msg}\n{tb_str}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the VoxCPM model"""
        return {
            'name': 'VoxCPM',
            'version': '1.5',
            'description': 'VoxCPM 1.5 Text-to-Speech with voice cloning capabilities',
            'sample_rate': self.sample_rate,
            'supports_voice_cloning': True,
            'supports_emotion_control': False,
            'supports_speed_control': False,
            'max_text_length': 2000,  # Recommended max for chunking
            'languages': ['English', 'Chinese', 'Multilingual'],
            'model_id': self.model_id
        }

# Convenience functions for integration with the main app
def init_voxcpm():
    """Initialize VoxCPM model"""
    handler = get_voxcpm_handler()
    return handler.initialize_model()

def unload_voxcpm():
    """Unload VoxCPM model"""
    handler = get_voxcpm_handler()
    return handler.unload_model()

def get_voxcpm_status():
    """Get VoxCPM model status"""
    if not VOXCPM_AVAILABLE:
        return "❌ VoxCPM not available"
    
    handler = get_voxcpm_handler()
    if handler.is_model_loaded():
        return "✅ VoxCPM model loaded"
    else:
        return "⚪ VoxCPM model not loaded"

def generate_voxcpm_tts(
    text: str,
    reference_audio: Optional[str] = None,
    reference_text: Optional[str] = None,
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
    normalize: bool = True,
    denoise: bool = True,
    retry_badcase: bool = True,
    retry_badcase_max_times: int = 3,
    retry_badcase_ratio_threshold: float = 6.0,
    seed: Optional[int] = None,
    output_path: Optional[str] = None,
    speaker_name: Optional[str] = None
) -> Tuple[Optional[str], str]:
    """
    Generate VoxCPM TTS and save to file
    
    Returns:
        Tuple of (output_path, status_message)
    """
    handler = get_voxcpm_handler()
    
    # Generate speech
    audio_data, message = handler.generate_speech(
        text=text,
        reference_audio=reference_audio,
        reference_text=reference_text,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
        normalize=normalize,
        denoise=denoise,
        retry_badcase=retry_badcase,
        retry_badcase_max_times=retry_badcase_max_times,
        retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
        seed=seed,
        speaker_name=speaker_name
    )
    
    if audio_data is None:
        return None, message
    
    # Save to file
    try:
        if output_path is None:
            # Generate output path
            outputs_dir = Path("outputs")
            outputs_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(outputs_dir / f"voxcpm_tts_{timestamp}.wav")
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save audio
        sf.write(output_path, audio_data, handler.sample_rate)
        
        return output_path, f"✅ VoxCPM TTS saved to {output_path}"
        
    except Exception as e:
        return None, f"❌ Error saving audio: {str(e)}"

def transcribe_voxcpm_audio(audio_path: str) -> str:
    """Transcribe audio using VoxCPM's Whisper model"""
    handler = get_voxcpm_handler()
    return handler.transcribe_audio(audio_path)

def start_voxcpm_conversation(seed: Optional[int] = None, reference_audio: Optional[str] = None, reference_text: Optional[str] = None) -> Dict[str, Any]:
    """
    Start VoxCPM conversation mode with consistent seed and voice
    
    Args:
        seed: Seed to use for all conversation turns (if None, generates random seed)
        reference_audio: Reference audio for voice cloning (optional)
        reference_text: Reference text for voice cloning (optional, will auto-transcribe if not provided)
    
    Returns:
        Dictionary with conversation settings
    """
    handler = get_voxcpm_handler()
    return handler.start_conversation(seed=seed, reference_audio=reference_audio, reference_text=reference_text)

def end_voxcpm_conversation():
    """End VoxCPM conversation mode"""
    handler = get_voxcpm_handler()
    handler.end_conversation()

def get_voxcpm_conversation_info() -> Optional[Dict[str, Any]]:
    """Get current VoxCPM conversation mode information"""
    handler = get_voxcpm_handler()
    return handler.get_conversation_info()

def is_voxcpm_in_conversation() -> bool:
    """Check if VoxCPM is currently in conversation mode"""
    handler = get_voxcpm_handler()
    return handler.is_in_conversation()

def set_voxcpm_speaker_seed(speaker_name: str, seed: Optional[int] = None) -> int:
    """
    Set or get a consistent seed for a specific speaker
    
    Args:
        speaker_name: Name of the speaker
        seed: Optional seed to use (if None, generates random)
        
    Returns:
        The seed assigned to this speaker
    """
    handler = get_voxcpm_handler()
    return handler.get_or_set_speaker_seed(speaker_name, seed)

def get_voxcpm_speaker_seeds() -> Dict[str, int]:
    """Get all current speaker seeds"""
    handler = get_voxcpm_handler()
    return handler.speaker_seeds.copy()

def clear_voxcpm_speaker_seeds():
    """Clear all speaker seeds"""
    handler = get_voxcpm_handler()
    handler.speaker_seeds.clear()
    print("🎭 Cleared all speaker seeds")

# Export main functions
__all__ = [
    'VoxCPMHandler',
    'get_voxcpm_handler',
    'init_voxcpm',
    'unload_voxcpm',
    'get_voxcpm_status',
    'generate_voxcpm_tts',
    'transcribe_voxcpm_audio',
    'start_voxcpm_conversation',
    'end_voxcpm_conversation',
    'get_voxcpm_conversation_info',
    'is_voxcpm_in_conversation',
    'set_voxcpm_speaker_seed',
    'get_voxcpm_speaker_seeds',
    'clear_voxcpm_speaker_seeds',
    'VOXCPM_AVAILABLE'
]