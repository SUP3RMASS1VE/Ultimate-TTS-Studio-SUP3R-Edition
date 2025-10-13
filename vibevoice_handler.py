"""
VibeVoice Handler for TTS Application
Provides integration with VibeVoice TTS engine
"""

import os
import sys
import numpy as np
import torch
import tempfile
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

# Suppress warnings
warnings.filterwarnings('ignore')

# Add vibevoice paths to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
vibevoice_root = os.path.join(current_dir, 'vibevoice')
vibevoice_demo_path = os.path.join(current_dir, 'vibevoice', 'demo')
vibevoice_package_path = os.path.join(current_dir, 'vibevoice', 'vibevoice')

# Add all necessary paths
for path in [vibevoice_root, vibevoice_demo_path, vibevoice_package_path]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Try to import VibeVoice components
try:
    # Import from the demo folder
    from gradio_demo import VibeVoiceDemo
    VIBEVOICE_AVAILABLE = True
    print("✅ VibeVoice handler loaded")
except ImportError as e:
    try:
        # Fallback: try importing with full path
        sys.path.insert(0, os.path.join(current_dir, 'vibevoice'))
        from demo.gradio_demo import VibeVoiceDemo
        VIBEVOICE_AVAILABLE = True
        print("✅ VibeVoice handler loaded (fallback import)")
    except ImportError as e2:
        VIBEVOICE_AVAILABLE = False
        print(f"⚠️ VibeVoice not available: {e}")
        print(f"   Fallback also failed: {e2}")
        print(f"   Checked paths: {[vibevoice_root, vibevoice_demo_path, vibevoice_package_path]}")

class VibeVoiceHandler:
    """Handler for VibeVoice TTS operations"""
    
    def __init__(self):
        self.demo = None
        self.model_loaded = False
        self.available_models = []
        self.current_model_path = ""
        
    def initialize_demo(self, model_path: str = "models/VibeVoice-1.5B") -> Tuple[bool, str]:
        """Initialize the VibeVoice demo instance"""
        if not VIBEVOICE_AVAILABLE:
            return False, "❌ VibeVoice not available"
        
        try:
            self.demo = VibeVoiceDemo(
                model_path=model_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                inference_steps=5
            )
            self.current_model_path = model_path
            return True, "✅ VibeVoice demo initialized"
        except Exception as e:
            return False, f"❌ Error initializing VibeVoice: {str(e)}"
    
    def load_model(self, model_path: str = None) -> Tuple[bool, str]:
        """Load a VibeVoice model"""
        if not self.demo:
            success, msg = self.initialize_demo(model_path or "models/VibeVoice-1.5B")
            if not success:
                return False, msg
        
        try:
            if model_path and model_path != self.current_model_path:
                self.demo.model_path = model_path
                self.current_model_path = model_path
            
            print(f"🔄 Loading VibeVoice model from: {self.current_model_path}")
            
            self.demo.load_model()
            
            if self.demo.model is not None:
                self.model_loaded = True
                print(f"✅ VibeVoice model loaded successfully!")
                return True, f"✅ VibeVoice model loaded from {self.current_model_path}"
            else:
                self.model_loaded = False
                return False, "❌ Failed to load VibeVoice model"
        except Exception as e:
            self.model_loaded = False
            return False, f"❌ Error loading model: {str(e)}"
    
    def unload_model(self) -> str:
        """Unload the current model"""
        if not self.demo:
            return "ℹ️ No demo instance available"
        
        try:
            result = self.demo.unload_model()
            self.model_loaded = False
            print(f"✅ VibeVoice model unloaded successfully")
            return result
        except Exception as e:
            return f"❌ Error unloading model: {str(e)}"
    
    def get_model_status(self) -> str:
        """Get current model status"""
        if not self.demo:
            return "❌ Demo not initialized"
        
        return self.demo.get_model_status()
    
    def scan_for_models(self) -> List[str]:
        """Scan for available models on disk without requiring demo initialization."""
        try:
            model_paths: List[str] = []
            default_dirs = ["models", "./models"]
            for base_dir in default_dirs:
                if os.path.exists(base_dir):
                    for item in os.listdir(base_dir):
                        item_path = os.path.join(base_dir, item)
                        if not os.path.isdir(item_path):
                            continue
                        # Consider it a VibeVoice model dir if it has a config.json and any safetensors file
                        has_config = os.path.exists(os.path.join(item_path, "config.json")) or os.path.exists(os.path.join(item_path, "preprocessor_config.json"))
                        has_weights = any(
                            f.endswith('.safetensors') for f in os.listdir(item_path)
                        ) or os.path.exists(os.path.join(item_path, "model.safetensors.index.json"))
                        if has_config and has_weights:
                            model_paths.append(item_path)

            # De-duplicate and sort for stable display
            unique_paths = []
            seen = set()
            for p in model_paths:
                ap = os.path.abspath(p)
                if ap not in seen:
                    unique_paths.append(p)
                    seen.add(ap)

            return unique_paths
        except Exception as e:
            return [f"Error scanning models: {str(e)}"]
    
    def download_model(self, model_name: str = "VibeVoice-1.5B") -> Tuple[bool, str]:
        """Download a VibeVoice model from HuggingFace"""
        try:
            from huggingface_hub import snapshot_download
            import os
            
            # Model repository mapping
            model_repos = {
                "VibeVoice-1.5B": "microsoft/VibeVoice-1.5B",
                "VibeVoice-Large": "vibevoice/VibeVoice-7B"
            }
            
            if model_name not in model_repos:
                return False, f"❌ Unknown model: {model_name}. Available: {list(model_repos.keys())}"
            
            repo_id = model_repos[model_name]
            local_dir = f"models/{model_name}"
            
            # Create models directory if it doesn't exist
            os.makedirs("models", exist_ok=True)
            
            # Check if model already exists
            if os.path.exists(local_dir) and os.listdir(local_dir):
                return True, f"✅ Model {model_name} already exists at {local_dir}"
            
            print(f"📥 Downloading {model_name} from HuggingFace...")
            
            # Download the model
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
            
            print(f"✅ Successfully downloaded {model_name}")
            return True, f"✅ Successfully downloaded {model_name} to {local_dir}"
            
        except Exception as e:
            return False, f"❌ Error downloading model: {str(e)}"
    
    def get_available_voices(self) -> List[str]:
        """Get list of available voice presets"""
        # Try to load voices directly from the voices directory
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            voices_dir = os.path.join(current_dir, 'vibevoice', 'demo', 'voices')
            
            if not os.path.exists(voices_dir):
                return ["Voices directory not found"]
            
            # Get all audio files in the voices directory
            audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']
            voice_files = []
            
            for file in os.listdir(voices_dir):
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    # Remove extension to get voice name
                    voice_name = os.path.splitext(file)[0]
                    voice_files.append(voice_name)
            
            if voice_files:
                return sorted(voice_files)
            else:
                return ["No voice files found"]
                
        except Exception as e:
            # Fallback to demo initialization
            if not self.demo:
                try:
                    self.demo = VibeVoiceDemo(
                        model_path="models/VibeVoice-1.5B",  # Default path, model won't be loaded
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        inference_steps=5
                    )
                    # Don't load the model, just initialize for voice access
                except Exception as demo_error:
                    return [f"Error initializing demo: {str(demo_error)}"]
            
            try:
                if hasattr(self.demo, 'available_voices') and self.demo.available_voices:
                    return list(self.demo.available_voices.keys())
                else:
                    # Try to initialize voice presets if not already done
                    self.demo.setup_voice_presets()
                    if hasattr(self.demo, 'available_voices') and self.demo.available_voices:
                        return list(self.demo.available_voices.keys())
                    else:
                        return [f"Error loading voices: {str(e)}"]
            except Exception as voice_error:
                return [f"Error loading voices: {str(voice_error)}"]
    
    def add_custom_voice(self, audio_file, voice_name: str) -> str:
        """Add a custom voice from audio file"""
        if not audio_file:
            return "❌ Please upload an audio file"
        
        if voice_name is None:
            return "❌ Please provide a voice name (voice_name is None)"
        
        if not isinstance(voice_name, str):
            return f"❌ Voice name must be text (got {type(voice_name)})"
        
        if not voice_name.strip():
            return "❌ Please provide a voice name (empty string)"
        
        # Clean the voice name
        voice_name = voice_name.strip().replace(' ', '-')
        
        # Validate voice name
        if not voice_name.replace('-', '').replace('_', '').isalnum():
            return "❌ Voice name should only contain letters, numbers, hyphens, and underscores"
        
        try:
            # Get the voices directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            voices_dir = os.path.join(current_dir, 'vibevoice', 'demo', 'voices')
            
            if not os.path.exists(voices_dir):
                return f"❌ Voices directory not found: {voices_dir}"
            
            # Create output filename
            output_filename = f"custom-{voice_name}.wav"
            output_path = os.path.join(voices_dir, output_filename)
            
            # Check if voice name already exists
            existing_voices = self.get_available_voices()
            if f"custom-{voice_name}" in existing_voices or voice_name in existing_voices:
                return f"❌ Voice name '{voice_name}' already exists. Please choose a different name."
            
            # Handle different audio file types
            import librosa
            import soundfile as sf
            
            # Load audio with librosa (handles multiple formats)
            audio_data, sample_rate = librosa.load(audio_file, sr=24000, mono=True)
            
            if len(audio_data) == 0:
                return "❌ Failed to read audio file. Please check the file format."
            
            # Validate audio length (recommend 3-30 seconds)
            duration = len(audio_data) / sample_rate
            if duration < 1:
                return f"❌ Audio too short ({duration:.1f}s). Please use audio that's at least 1 second long."
            elif duration > 60:
                return f"⚠️ Audio is quite long ({duration:.1f}s). Consider using a shorter sample (3-30s recommended) for better results."
            
            # Save the processed audio as WAV
            sf.write(output_path, audio_data, sample_rate)
            
            # Verify file was saved
            if os.path.exists(output_path):
                # Update the demo's available_voices dictionary if demo is initialized
                if self.demo and hasattr(self.demo, 'available_voices'):
                    self.demo.available_voices[f"custom-{voice_name}"] = output_path
                    print(f"✅ Updated demo's available_voices dictionary with '{f'custom-{voice_name}'}'")
                
                return f"✅ Successfully added custom voice '{voice_name}' ({duration:.1f}s)\n💡 You can now select it in the Speaker dropdowns!"
            else:
                return "❌ Failed to save voice file"
            
        except Exception as e:
            return f"❌ Error processing audio file: {str(e)}"
    
    def generate_podcast(self,
                        num_speakers: int,
                        script: str,
                        speaker_voices: List[str],
                        cfg_scale: float = 1.3,
                        seed: Optional[int] = None,
                        output_folder: str = "outputs",
                        audio_format: str = "wav") -> Tuple[Optional[Tuple], str]:
        """Generate podcast audio using VibeVoice (non-streaming mode)"""
        print("\n" + "="*60)
        print("🎙️ VIBEVOICE PODCAST GENERATION STARTED")
        print("="*60)

        if not self.demo:
            print("❌ Demo not initialized")
            return None, "❌ Demo not initialized"

        if not self.model_loaded:
            print("❌ No model loaded")
            return None, "❌ No model loaded"

        try:
            # Ensure we have enough speakers
            if len(speaker_voices) < num_speakers:
                error_msg = f"❌ Need {num_speakers} speakers, only {len(speaker_voices)} provided"
                print(error_msg)
                return None, error_msg

            # Print generation parameters
            print(f"📊 Generation Parameters:")
            print(f"   🎤 Number of speakers: {num_speakers}")
            print(f"   🎛️ CFG Scale: {cfg_scale}")
            print(f"   🎲 Seed: {seed if seed is not None else 'Random'}")
            print(f"   🎭 Speakers: {', '.join(speaker_voices[:num_speakers])}")
            print(f"   📝 Script length: {len(script)} characters")
            print(f"   📁 Output folder: {output_folder}")

            print(f"\n🔄 Starting VibeVoice generation...")
            print(f"⏳ This may take several minutes depending on script length...")

            # Refresh voice presets to ensure custom voices are available
            if hasattr(self.demo, 'setup_voice_presets'):
                self.demo.setup_voice_presets()
                print(f"✅ Refreshed voice presets. Available voices: {len(self.demo.available_voices)}")

            # Collect all results from streaming generator
            final_audio = None
            final_log = ""
            complete_audio = None

            # Process all streaming results to get the final complete audio
            for result in self.demo.generate_podcast_streaming(
                num_speakers=num_speakers,
                script=script,
                speaker_1=speaker_voices[0] if len(speaker_voices) > 0 else None,
                speaker_2=speaker_voices[1] if len(speaker_voices) > 1 else None,
                speaker_3=speaker_voices[2] if len(speaker_voices) > 2 else None,
                speaker_4=speaker_voices[3] if len(speaker_voices) > 3 else None,
                cfg_scale=cfg_scale,
                seed=seed
            ):
                # Handle different return formats from the streaming generator
                if isinstance(result, tuple):
                    if len(result) == 4:
                        # Format: streaming_audio, complete_audio, log, ui_update
                        streaming_audio, complete_audio, log_result, _ = result
                        final_log = log_result
                        if complete_audio is not None:
                            break  # We got the final complete audio, exit the loop
                    elif len(result) == 3:
                        # Format: audio, log, ui_update
                        audio_result, log_result, _ = result
                        if audio_result is not None:
                            complete_audio = audio_result
                        final_log = log_result
                    elif len(result) == 2:
                        # Format: audio, log
                        audio_result, log_result = result
                        if audio_result is not None:
                            complete_audio = audio_result
                        final_log = log_result
                else:
                    # Single value, assume it's audio
                    if result is not None:
                        complete_audio = result
                        final_log = "Generated audio"

            # Use complete audio if available, otherwise fall back to streaming audio
            if complete_audio is not None:
                final_audio = complete_audio
                print(f"✅ Complete audio received")
            elif final_audio is not None:
                print(f"✅ Using final streaming audio")
            else:
                print(f"⚠️ No audio generated")
                return None, "❌ No audio was generated"

            print(f"🎉 Generation completed!")

            # Save the audio to outputs folder
            if final_audio is not None:
                try:
                    print(f"\n💾 Saving audio to file...")
                    os.makedirs(output_folder, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename_base = f"vibevoice_podcast_{timestamp}"

                    # Import the save function from launch.py
                    import sys
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    if current_dir not in sys.path:
                        sys.path.insert(0, current_dir)
                    
                    try:
                        from launch import save_audio_with_format
                        
                        sample_rate, audio_data = final_audio

                        # Calculate audio duration
                        duration = len(audio_data) / sample_rate
                        file_size_mb = len(audio_data) * 4 / (1024 * 1024)  # Rough estimate for 32-bit float

                        print(f"   📊 Audio info:")
                        print(f"      🎵 Sample rate: {sample_rate} Hz")
                        print(f"      ⏱️ Duration: {duration:.2f} seconds")
                        print(f"      📏 Samples: {len(audio_data):,}")
                        print(f"      💽 Estimated size: {file_size_mb:.1f} MB")
                        print(f"      🎵 Format: {audio_format.upper()}")

                        # Use the format-aware save function
                        filepath, filename = save_audio_with_format(
                            audio_data, sample_rate, audio_format, output_folder, filename_base
                        )

                        # Get actual file size
                        actual_size_mb = os.path.getsize(filepath) / (1024 * 1024)

                        print(f"   ✅ Successfully saved: {filename}")
                        print(f"   📁 Location: {filepath}")
                        print(f"   💽 File size: {actual_size_mb:.1f} MB")

                        final_log += f"\n💾 Saved as: {filename} ({audio_format.upper()})"
                        
                    except ImportError:
                        # Fallback to WAV if save_audio_with_format is not available
                        print(f"   ⚠️ Using fallback WAV saving...")
                        import soundfile as sf
                        sample_rate, audio_data = final_audio
                        filename = f"{filename_base}.wav"
                        filepath = os.path.join(output_folder, filename)
                        sf.write(filepath, audio_data, sample_rate)
                        final_log += f"\n💾 Saved as: {filename} (WAV fallback)"
                        
                except Exception as save_error:
                    print(f"   ❌ Error saving file: {save_error}")
                    final_log += f"\n⚠️ Could not save file: {save_error}"
            else:
                print(f"\n⚠️ No audio generated to save")

            print("\n" + "="*60)
            print("🎉 VIBEVOICE PODCAST GENERATION COMPLETED")
            print("="*60)

            return final_audio, final_log

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return None, f"❌ Error generating podcast: {str(e)}\n\nDetails:\n{error_details}"

# Global handler instance
_vibevoice_handler = None

def get_vibevoice_handler() -> VibeVoiceHandler:
    """Get the global VibeVoice handler instance"""
    global _vibevoice_handler
    if _vibevoice_handler is None:
        _vibevoice_handler = VibeVoiceHandler()
    return _vibevoice_handler

def generate_vibevoice_podcast(num_speakers: int,
                              script: str,
                              speaker_voices: List[str],
                              cfg_scale: float = 1.3,
                              seed: Optional[int] = None,
                              output_folder: str = "outputs",
                              audio_format: str = "wav") -> Tuple[Optional[Tuple], str]:
    """Generate podcast using VibeVoice"""
    handler = get_vibevoice_handler()
    return handler.generate_podcast(
        num_speakers=num_speakers,
        script=script,
        speaker_voices=speaker_voices,
        cfg_scale=cfg_scale,
        seed=seed,
        output_folder=output_folder,
        audio_format=audio_format
    )

def init_vibevoice(model_path: str = "models/VibeVoice-1.5B") -> Tuple[bool, str]:
    """Initialize VibeVoice with a model"""
    handler = get_vibevoice_handler()
    return handler.load_model(model_path)

def unload_vibevoice() -> str:
    """Unload VibeVoice model"""
    handler = get_vibevoice_handler()
    return handler.unload_model()

def get_vibevoice_status() -> str:
    """Get VibeVoice status"""
    handler = get_vibevoice_handler()
    return handler.get_model_status()

def get_vibevoice_voices() -> List[str]:
    """Get available VibeVoice voices"""
    handler = get_vibevoice_handler()
    return handler.get_available_voices()

def scan_vibevoice_models() -> List[str]:
    """Scan for available VibeVoice models"""
    handler = get_vibevoice_handler()
    return handler.scan_for_models()

def download_vibevoice_model(model_name: str = "VibeVoice-1.5B") -> Tuple[bool, str]:
    """Download a VibeVoice model"""
    handler = get_vibevoice_handler()
    return handler.download_model(model_name)
