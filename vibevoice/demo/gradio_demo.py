"""
VibeVoice Gradio Demo - High-Quality Dialogue Generation Interface with Streaming Support
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Iterator
from datetime import datetime
import threading
import numpy as np
import gradio as gr
import librosa
import soundfile as sf
import torch
import os
import traceback
from huggingface_hub import snapshot_download, hf_hub_download
import shutil

from transformers.utils import logging
from transformers import set_seed

# Suppress the APEX FusedRMSNorm warning message BEFORE importing VibeVoice modules
logging.set_verbosity_info()
tokenizer_logger = logging.get_logger("vibevoice.modular.modular_vibevoice_tokenizer")
tokenizer_logger.setLevel(logging.ERROR)  # Only show ERROR level and above

logger = logging.get_logger(__name__)

# Import VibeVoice modules AFTER logging configuration
from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AudioStreamer


class VibeVoiceDemo:
    def __init__(self, model_path: str, device: str = "cuda", inference_steps: int = 5):
        """Initialize the VibeVoice demo with model loading."""
        self.model_path = model_path
        self.device = device
        self.inference_steps = inference_steps
        self.is_generating = False  # Track generation state
        self.stop_generation = False  # Flag to stop generation
        self.current_streamer = None  # Track current audio streamer
        self.is_downloading = False  # Track download state
        self.use_flash_attention = True  # Default to Flash Attention
        
        # Available models for download
        self.available_models = {
            "VibeVoice-1.5B": {
                "repo_id": "microsoft/VibeVoice-1.5B",
                "size": "~2.7B parameters",
                "description": "Compact model, good for most use cases"
            },
            "VibeVoice-Large": {
                "repo_id": "vibevoice/VibeVoice-7B",
                "size": "~9.34B parameters",
                "description": "Large model, higher quality, ~45 min generation"
            }
        }
        
        # Don't auto-load model - let user choose
        self.model = None
        self.processor = None
        print("🎙️ VibeVoice Demo initialized without loading a model.")
        print("💡 Use the Model Status section to select and load a model.")
        
        self.setup_voice_presets()
        self.load_example_scripts()  # Load example scripts
        
    def load_model(self):
        """Load the VibeVoice model and processor."""
        print(f"Loading processor & model from {self.model_path}")
        
        # Check if model path exists
        if not os.path.exists(self.model_path):
            print(f"⚠️ Model path does not exist: {self.model_path}")
            print("🔄 The demo will start without a loaded model.")
            print("💡 Use the Model Download section in the UI to download a model first.")
            self.model = None
            self.processor = None
            return
        
        # Check if path contains model files
        if not self._is_valid_model_path(self.model_path):
            print(f"⚠️ No valid model files found at: {self.model_path}")
            print("🔄 The demo will start without a loaded model.")
            print("💡 Use the Model Download section in the UI to download a model first.")
            self.model = None
            self.processor = None
            return
        
        try:
            # Load processor
            self.processor = VibeVoiceProcessor.from_pretrained(
                self.model_path,
            )
            
            # Load model with user-selected attention implementation
            import platform
            if self.use_flash_attention:
                attn_implementation = 'flash_attention_2'
                print("⚡ Using Flash Attention 2 for optimal performance")
            else:
                attn_implementation = 'sdpa'
                print("🔧 Using SDPA attention implementation")
            
            # Override on Windows if Flash Attention is selected but not available
            if platform.system() == "Windows" and self.use_flash_attention:
                print("🪟 Windows detected: If Flash Attention 2 is not available, Will fall back to SDPA")
            
            try:
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    device_map='cuda',
                    attn_implementation=attn_implementation
                )
            except Exception as e:
                if 'flash_attn' in str(e) and attn_implementation == 'flash_attention_2':
                    print(f"⚠️ Flash Attention 2 not available: {e}")
                    print("🔄 Falling back to SDPA attention implementation...")
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.bfloat16,
                        device_map='cuda',
                        attn_implementation='sdpa'
                    )
                else:
                    print(f"❌ Error loading model: {type(e).__name__}: {e}")
                    print(traceback.format_exc())
                    raise e
            self.model.eval()
            
            # Use SDE solver by default
            self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
                self.model.model.noise_scheduler.config, 
                algorithm_type='sde-dpmsolver++',
                beta_schedule='squaredcos_cap_v2'
            )
            self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)
            
            if hasattr(self.model.model, 'language_model'):
                print(f"Language model attention: {self.model.model.language_model.config._attn_implementation}")
                
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("🔄 The demo will start without a loaded model.")
            print("💡 Use the Model Download section in the UI to download a model first.")
            self.model = None
            self.processor = None
    
    def _is_valid_model_path(self, path: str) -> bool:
        """Check if the path contains valid model files."""
        if not os.path.exists(path) or not os.path.isdir(path):
            return False
        
        # Check for essential model files
        required_files = ['config.json']
        model_files = ['pytorch_model.bin', 'model.safetensors']
        
        files_in_dir = os.listdir(path)
        
        # Check for config file
        has_config = any(f in files_in_dir for f in required_files)
        
        # Check for at least one model file
        has_model = any(f in files_in_dir for f in model_files) or \
                   any(f.endswith('.bin') or f.endswith('.safetensors') for f in files_in_dir)
        
        return has_config and has_model
    
    def unload_model(self):
        """Unload the current model from memory."""
        if self.model is None and self.processor is None:
            return "ℹ️ No model is currently loaded."
        
        print("🗑️ Unloading current model from memory...")
        
        try:
            # Clear the model and processor references
            if hasattr(self, 'model') and self.model is not None:
                # Move model to CPU first to free GPU memory
                if hasattr(self.model, 'cpu'):
                    self.model.cpu()
                del self.model
                self.model = None
            
            if hasattr(self, 'processor') and self.processor is not None:
                del self.processor
                self.processor = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("🧹 Cleared CUDA cache")
            
            print("✅ Model unloaded successfully")
            return "✅ Model unloaded successfully. GPU/CPU memory has been freed."
            
        except Exception as e:
            error_msg = f"❌ Error during model unloading: {str(e)}"
            print(error_msg)
            # Ensure variables are reset even if there was an error
            self.model = None
            self.processor = None
            return error_msg
    
    def reload_model(self, new_model_path: str = None):
        """Reload the model from a new path."""
        # First unload the current model if one is loaded
        if self.model is not None or self.processor is not None:
            print("🔄 Unloading current model before loading new one...")
            self.unload_model()
        
        if new_model_path:
            self.model_path = new_model_path
        
        print(f"🔄 Loading model from {self.model_path}")
        self.load_model()
        
        if self.model is not None:
            return f"✅ Model successfully loaded from {self.model_path}"
        else:
            return f"❌ Failed to load model from {self.model_path}"
    
    def get_model_status(self) -> str:
        """Get current model loading status."""
        if self.model is None or self.processor is None:
            return f"❌ No model loaded"
        else:
            return f"✅ Model loaded successfully (Path: {self.model_path})"
    
    def scan_for_models(self) -> List[str]:
        """Scan for available downloaded models."""
        model_paths = []
        
        # Check default model directories
        default_dirs = ["models", "./models"]
        for base_dir in default_dirs:
            if os.path.exists(base_dir):
                for item in os.listdir(base_dir):
                    item_path = os.path.join(base_dir, item)
                    if os.path.isdir(item_path) and self._is_valid_model_path(item_path):
                        model_paths.append(item_path)
        
        # Check if the current model_path is valid and not already in list
        if self.model_path and self._is_valid_model_path(self.model_path):
            abs_path = os.path.abspath(self.model_path)
            if abs_path not in [os.path.abspath(p) for p in model_paths]:
                model_paths.append(self.model_path)
        
        # Also check common HuggingFace cache locations
        try:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            if os.path.exists(cache_dir):
                for item in os.listdir(cache_dir):
                    if "vibevoice" in item.lower():
                        item_path = os.path.join(cache_dir, item)
                        if os.path.isdir(item_path):
                            # Look for snapshots subdirectory
                            snapshots_dir = os.path.join(item_path, "snapshots")
                            if os.path.exists(snapshots_dir):
                                for snapshot in os.listdir(snapshots_dir):
                                    snapshot_path = os.path.join(snapshots_dir, snapshot)
                                    if os.path.isdir(snapshot_path) and self._is_valid_model_path(snapshot_path):
                                        display_name = f"🗂️ Cache: {item}"
                                        model_paths.append(f"{snapshot_path}|{display_name}")
        except Exception:
            pass  # Ignore cache scanning errors
        
        return sorted(model_paths) if model_paths else ["No models found"]
    
    def get_available_models_for_dropdown(self) -> List[str]:
        """Get formatted model list for dropdown."""
        models = self.scan_for_models()
        if models == ["No models found"]:
            return ["No models found - Download one first"]
        
        formatted_models = []
        for model_path in models:
            if "|" in model_path:
                # This is a cache entry with display name
                actual_path, display_name = model_path.split("|", 1)
                formatted_models.append(f"{display_name}")
            else:
                # Regular path
                model_name = os.path.basename(model_path)
                if not model_name:
                    model_name = os.path.basename(os.path.dirname(model_path))
                formatted_models.append(f"📁 {model_name}")
        
        return formatted_models
    
    def get_model_path_from_selection(self, selection: str) -> str:
        """Get actual model path from dropdown selection."""
        if "No models found" in selection:
            return ""
        
        models = self.scan_for_models()
        formatted_models = self.get_available_models_for_dropdown()
        
        try:
            index = formatted_models.index(selection)
            model_path = models[index]
            
            if "|" in model_path:
                # Cache entry
                actual_path, _ = model_path.split("|", 1)
                return actual_path
            else:
                return model_path
        except (ValueError, IndexError):
            return ""
    
    def add_custom_voice(self, audio_file, voice_name: str) -> str:
        """Add a custom voice from uploaded audio file."""
        if not audio_file:
            return "❌ Please upload an audio file"
        
        if not voice_name or not voice_name.strip():
            return "❌ Please provide a voice name"
        
        # Clean the voice name
        voice_name = voice_name.strip().replace(' ', '-')
        
        # Validate voice name
        if not voice_name.replace('-', '').replace('_', '').isalnum():
            return "❌ Voice name should only contain letters, numbers, hyphens, and underscores"
        
        try:
            # Get the voices directory
            voices_dir = os.path.join(os.path.dirname(__file__), "voices")
            os.makedirs(voices_dir, exist_ok=True)
            
            # Get file extension from uploaded file
            original_filename = audio_file.name if hasattr(audio_file, 'name') else str(audio_file)
            file_extension = os.path.splitext(original_filename)[1].lower()
            
            # If no extension or unsupported extension, default to .wav
            supported_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']
            if file_extension not in supported_extensions:
                file_extension = '.wav'
            
            # Create output filename
            output_filename = f"custom-{voice_name}{file_extension}"
            output_path = os.path.join(voices_dir, output_filename)
            
            # Check if voice name already exists
            if voice_name in self.available_voices or f"custom-{voice_name}" in self.available_voices:
                return f"❌ Voice name '{voice_name}' already exists. Please choose a different name."
            
            # Read and process the audio
            audio_data = self.read_audio(audio_file, target_sr=24000)
            
            if len(audio_data) == 0:
                return "❌ Failed to read audio file. Please check the file format."
            
            # Validate audio length (recommend 3-30 seconds)
            duration = len(audio_data) / 24000
            if duration < 1:
                return f"❌ Audio too short ({duration:.1f}s). Please use audio that's at least 1 second long."
            elif duration > 60:
                return f"⚠️ Audio is quite long ({duration:.1f}s). Consider using a shorter sample (3-30s recommended) for better results."
            
            # Save the processed audio as WAV
            output_path_wav = os.path.join(voices_dir, f"custom-{voice_name}.wav")
            sf.write(output_path_wav, audio_data, 24000)
            
            # Update available voices
            self.available_voices[f"custom-{voice_name}"] = output_path_wav
            
            return f"✅ Successfully added custom voice '{voice_name}' ({duration:.1f}s)\n💡 You can now select it in the Speaker dropdowns!"
            
        except Exception as e:
            return f"❌ Error processing audio file: {str(e)}"
    
    def refresh_voice_list(self) -> List[str]:
        """Refresh the list of available voices including custom ones."""
        self.setup_voice_presets()  # Re-scan the voices directory
        return list(self.available_voices.keys())
    
    def setup_voice_presets(self):
        """Setup voice presets by scanning the voices directory."""
        voices_dir = os.path.join(os.path.dirname(__file__), "voices")
        
        # Check if voices directory exists
        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            self.voice_presets = {}
            self.available_voices = {}
            return
        
        # Scan for all WAV files in the voices directory
        self.voice_presets = {}
        
        # Get all .wav files in the voices directory
        wav_files = [f for f in os.listdir(voices_dir) 
                    if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')) and os.path.isfile(os.path.join(voices_dir, f))]
        
        # Create dictionary with filename (without extension) as key
        for wav_file in wav_files:
            # Remove .wav extension to get the name
            name = os.path.splitext(wav_file)[0]
            # Create full path
            full_path = os.path.join(voices_dir, wav_file)
            self.voice_presets[name] = full_path
        
        # Sort the voice presets alphabetically by name for better UI
        self.voice_presets = dict(sorted(self.voice_presets.items()))
        
        # Filter out voices that don't exist (this is now redundant but kept for safety)
        self.available_voices = {
            name: path for name, path in self.voice_presets.items()
            if os.path.exists(path)
        }
        
        if not self.available_voices:
            raise gr.Error("No voice presets found. Please add .wav files to the demo/voices directory.")
        
        print(f"Found {len(self.available_voices)} voice files in {voices_dir}")
        print(f"Available voices: {', '.join(self.available_voices.keys())}")
    
    def read_audio(self, audio_path: str, target_sr: int = 24000) -> np.ndarray:
        """Read and preprocess audio file."""
        try:
            wav, sr = sf.read(audio_path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            return wav
        except Exception as e:
            print(f"Error reading audio {audio_path}: {e}")
            return np.array([])
    
    def generate_podcast_streaming(self, 
                                 num_speakers: int,
                                 script: str,
                                 speaker_1: str = None,
                                 speaker_2: str = None,
                                 speaker_3: str = None,
                                 speaker_4: str = None,
                                 cfg_scale: float = 1.3,
                                 seed: int = None) -> Iterator[tuple]:
        try:
            
            # Check if model is loaded
            if self.model is None or self.processor is None:
                self.is_generating = False
                raise gr.Error("❌ No model loaded! Please download a model first using the Model Download section.")
            
            # Set seed for reproducibility
            actual_seed = seed
            if seed is not None:
                set_seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
            else:
                # Generate a random seed and use it
                import random
                actual_seed = random.randint(0, 2**31 - 1)
                set_seed(actual_seed)
                torch.manual_seed(actual_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(actual_seed)
                np.random.seed(actual_seed)
            
            # Reset stop flag and set generating state
            self.stop_generation = False
            self.is_generating = True
            
            # Validate inputs
            if not script.strip():
                self.is_generating = False
                raise gr.Error("Error: Please provide a script.")

            # Defend against common mistake
            script = script.replace("’", "'")
            
            if num_speakers < 1 or num_speakers > 4:
                self.is_generating = False
                raise gr.Error("Error: Number of speakers must be between 1 and 4.")
            
            # Collect selected speakers
            selected_speakers = [speaker_1, speaker_2, speaker_3, speaker_4][:num_speakers]
            
            # Validate speaker selections
            for i, speaker in enumerate(selected_speakers):
                if not speaker or speaker not in self.available_voices:
                    self.is_generating = False
                    raise gr.Error(f"Error: Please select a valid speaker for Speaker {i+1}.")
            
            # Build initial log
            log = f"🎙️ Generating podcast with {num_speakers} speakers\n"
            seed_info = f"Seed={seed}" if seed is not None else "Seed=Random"
            log += f"📊 Parameters: CFG Scale={cfg_scale}, Inference Steps={self.inference_steps}, {seed_info}\n"
            log += f"🎭 Speakers: {', '.join(selected_speakers)}\n"
            
            # Check for stop signal
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            
            # Load voice samples
            voice_samples = []
            for speaker_name in selected_speakers:
                audio_path = self.available_voices[speaker_name]
                audio_data = self.read_audio(audio_path)
                if len(audio_data) == 0:
                    self.is_generating = False
                    raise gr.Error(f"Error: Failed to load audio for {speaker_name}")
                voice_samples.append(audio_data)
            
            # log += f"✅ Loaded {len(voice_samples)} voice samples\n"
            
            # Check for stop signal
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            
            # Parse script to assign speaker ID's
            lines = script.strip().split('\n')
            formatted_script_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if line already has speaker format
                if line.startswith('Speaker ') and ':' in line:
                    formatted_script_lines.append(line)
                else:
                    # Auto-assign to speakers in rotation
                    speaker_id = len(formatted_script_lines) % num_speakers
                    formatted_script_lines.append(f"Speaker {speaker_id}: {line}")
            
            formatted_script = '\n'.join(formatted_script_lines)
            log += f"📝 Formatted script with {len(formatted_script_lines)} turns\n\n"
            log += "🔄 Processing with VibeVoice (streaming mode)...\n"
            
            # Check for stop signal before processing
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            
            start_time = time.time()
            
            inputs = self.processor(
                text=[formatted_script],
                voice_samples=[voice_samples],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            
            # Create audio streamer
            audio_streamer = AudioStreamer(
                batch_size=1,
                stop_signal=None,
                timeout=None
            )
            
            # Store current streamer for potential stopping
            self.current_streamer = audio_streamer
            
            # Start generation in a separate thread
            generation_thread = threading.Thread(
                target=self._generate_with_streamer,
                args=(inputs, cfg_scale, audio_streamer)
            )
            generation_thread.start()
            
            # Wait for generation to actually start producing audio
            time.sleep(1)  # Reduced from 3 to 1 second

            # Check for stop signal after thread start
            if self.stop_generation:
                audio_streamer.end()
                generation_thread.join(timeout=5.0)  # Wait up to 5 seconds for thread to finish
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", gr.update(visible=False)
                return

            # Collect audio chunks as they arrive
            sample_rate = 24000
            all_audio_chunks = []  # For final statistics
            pending_chunks = []  # Buffer for accumulating small chunks
            chunk_count = 0
            last_yield_time = time.time()
            min_yield_interval = 15 # Yield every 15 seconds
            min_chunk_size = sample_rate * 30 # At least 2 seconds of audio
            
            # Get the stream for the first (and only) sample
            audio_stream = audio_streamer.get_stream(0)
            
            has_yielded_audio = False
            has_received_chunks = False  # Track if we received any chunks at all
            
            for audio_chunk in audio_stream:
                # Check for stop signal in the streaming loop
                if self.stop_generation:
                    audio_streamer.end()
                    break
                    
                chunk_count += 1
                has_received_chunks = True  # Mark that we received at least one chunk
                
                # Convert tensor to numpy
                if torch.is_tensor(audio_chunk):
                    # Convert bfloat16 to float32 first, then to numpy
                    if audio_chunk.dtype == torch.bfloat16:
                        audio_chunk = audio_chunk.float()
                    audio_np = audio_chunk.cpu().numpy().astype(np.float32)
                else:
                    audio_np = np.array(audio_chunk, dtype=np.float32)
                
                # Ensure audio is 1D and properly normalized
                if len(audio_np.shape) > 1:
                    audio_np = audio_np.squeeze()
                
                # Convert to 16-bit for Gradio
                audio_16bit = convert_to_16_bit_wav(audio_np)
                
                # Store for final statistics
                all_audio_chunks.append(audio_16bit)
                
                # Add to pending chunks buffer
                pending_chunks.append(audio_16bit)
                
                # Calculate pending audio size
                pending_audio_size = sum(len(chunk) for chunk in pending_chunks)
                current_time = time.time()
                time_since_last_yield = current_time - last_yield_time
                
                # Decide whether to yield
                should_yield = False
                if not has_yielded_audio and pending_audio_size >= min_chunk_size:
                    # First yield: wait for minimum chunk size
                    should_yield = True
                    has_yielded_audio = True
                elif has_yielded_audio and (pending_audio_size >= min_chunk_size or time_since_last_yield >= min_yield_interval):
                    # Subsequent yields: either enough audio or enough time has passed
                    should_yield = True
                
                if should_yield and pending_chunks:
                    # Concatenate and yield only the new audio chunks
                    new_audio = np.concatenate(pending_chunks)
                    new_duration = len(new_audio) / sample_rate
                    total_duration = sum(len(chunk) for chunk in all_audio_chunks) / sample_rate
                    
                    log_update = log + f"🎵 Streaming: {total_duration:.1f}s generated (chunk {chunk_count})\n"
                    
                    # Yield streaming audio chunk and keep complete_audio as None during streaming
                    yield (sample_rate, new_audio), None, log_update, gr.update(visible=True)
                    
                    # Clear pending chunks after yielding
                    pending_chunks = []
                    last_yield_time = current_time
            
            # Yield any remaining chunks
            if pending_chunks:
                final_new_audio = np.concatenate(pending_chunks)
                total_duration = sum(len(chunk) for chunk in all_audio_chunks) / sample_rate
                log_update = log + f"🎵 Streaming final chunk: {total_duration:.1f}s total\n"
                yield (sample_rate, final_new_audio), None, log_update, gr.update(visible=True)
                has_yielded_audio = True  # Mark that we yielded audio
            
            # Wait for generation to complete (with timeout to prevent hanging)
            generation_thread.join(timeout=5.0)  # Increased timeout to 5 seconds

            # If thread is still alive after timeout, force end
            if generation_thread.is_alive():
                print("Warning: Generation thread did not complete within timeout")
                audio_streamer.end()
                generation_thread.join(timeout=5.0)

            # Clean up
            self.current_streamer = None
            self.is_generating = False
            
            generation_time = time.time() - start_time
            
            # Check if stopped by user
            if self.stop_generation:
                yield None, None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            
            # Debug logging
            # print(f"Debug: has_received_chunks={has_received_chunks}, chunk_count={chunk_count}, all_audio_chunks length={len(all_audio_chunks)}")
            
            # Check if we received any chunks but didn't yield audio
            if has_received_chunks and not has_yielded_audio and all_audio_chunks:
                # We have chunks but didn't meet the yield criteria, yield them now
                complete_audio = np.concatenate(all_audio_chunks)
                final_duration = len(complete_audio) / sample_rate
                
                final_log = log + f"⏱️ Generation completed in {generation_time:.2f} seconds\n"
                final_log += f"🎵 Final audio duration: {final_duration:.2f} seconds\n"
                final_log += f"📊 Total chunks: {chunk_count}\n"
                final_log += "✨ Generation successful! Complete audio is ready.\n"
                final_log += "💡 Not satisfied? You can regenerate or adjust the CFG scale for different results."
                
                # Yield the complete audio
                yield None, (sample_rate, complete_audio), final_log, gr.update(visible=False)
                return
            
            if not has_received_chunks:
                error_log = log + f"\n❌ Error: No audio chunks were received from the model. Generation time: {generation_time:.2f}s"
                yield None, None, error_log, gr.update(visible=False)
                return
            
            if not has_yielded_audio:
                error_log = log + f"\n❌ Error: Audio was generated but not streamed. Chunk count: {chunk_count}"
                yield None, None, error_log, gr.update(visible=False)
                return

            # Prepare the complete audio
            if all_audio_chunks:
                complete_audio = np.concatenate(all_audio_chunks)
                final_duration = len(complete_audio) / sample_rate
                
                final_log = log + f"⏱️ Generation completed in {generation_time:.2f} seconds\n"
                final_log += f"🎵 Final audio duration: {final_duration:.2f} seconds\n"
                final_log += f"📊 Total chunks: {chunk_count}\n"
                final_log += "✨ Generation successful! Complete audio is ready in the 'Complete Audio' tab.\n"
                final_log += "💡 Not satisfied? You can regenerate or adjust the CFG scale for different results."
                
                # Final yield: Clear streaming audio and provide complete audio
                yield None, (sample_rate, complete_audio), final_log, gr.update(visible=False)
            else:
                final_log = log + "❌ No audio was generated."
                yield None, None, final_log, gr.update(visible=False)

        except gr.Error as e:
            # Handle Gradio-specific errors (like input validation)
            self.is_generating = False
            self.current_streamer = None
            error_msg = f"❌ Input Error: {str(e)}"
            print(error_msg)
            yield None, None, error_msg, gr.update(visible=False)
            
        except Exception as e:
            self.is_generating = False
            self.current_streamer = None
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            yield None, None, error_msg, gr.update(visible=False)
    
    def _generate_with_streamer(self, inputs, cfg_scale, audio_streamer):
        """Helper method to run generation with streamer in a separate thread."""
        try:
            # Check for stop signal before starting generation
            if self.stop_generation:
                audio_streamer.end()
                return
                
            # Define a stop check function that can be called from generate
            def check_stop_generation():
                return self.stop_generation
                
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={
                    'do_sample': False,
                },
                audio_streamer=audio_streamer,
                stop_check_fn=check_stop_generation,  # Pass the stop check function
                verbose=False,  # Disable verbose in streaming mode
                refresh_negative=True,
            )
            
        except Exception as e:
            print(f"Error in generation thread: {e}")
            traceback.print_exc()
            # Make sure to end the stream on error
            audio_streamer.end()
    
    def stop_audio_generation(self):
        """Stop the current audio generation process."""
        self.stop_generation = True
        if self.current_streamer is not None:
            try:
                self.current_streamer.end()
            except Exception as e:
                print(f"Error stopping streamer: {e}")
        print("🛑 Audio generation stop requested")
    
    def load_example_scripts(self):
        """Load example scripts from the text_examples directory."""
        examples_dir = os.path.join(os.path.dirname(__file__), "text_examples")
        self.example_scripts = []
        
        # Check if text_examples directory exists
        if not os.path.exists(examples_dir):
            print(f"Warning: text_examples directory not found at {examples_dir}")
            return
        
        # Get all .txt files in the text_examples directory
        txt_files = sorted([f for f in os.listdir(examples_dir) 
                          if f.lower().endswith('.txt') and os.path.isfile(os.path.join(examples_dir, f))])
        
        for txt_file in txt_files:
            file_path = os.path.join(examples_dir, txt_file)
            
            import re
            # Check if filename contains a time pattern like "45min", "90min", etc.
            time_pattern = re.search(r'(\d+)min', txt_file.lower())
            if time_pattern:
                minutes = int(time_pattern.group(1))
                if minutes > 15:
                    print(f"Skipping {txt_file}: duration {minutes} minutes exceeds 15-minute limit")
                    continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    script_content = f.read().strip()
                
                # Remove empty lines and lines with only whitespace
                script_content = '\n'.join(line for line in script_content.split('\n') if line.strip())
                
                if not script_content:
                    continue
                
                # Parse the script to determine number of speakers
                num_speakers = self._get_num_speakers_from_script(script_content)
                
                # Add to examples list as [num_speakers, script_content]
                self.example_scripts.append([num_speakers, script_content])
                print(f"Loaded example: {txt_file} with {num_speakers} speakers")
                
            except Exception as e:
                print(f"Error loading example script {txt_file}: {e}")
        
        if self.example_scripts:
            print(f"Successfully loaded {len(self.example_scripts)} example scripts")
        else:
            print("No example scripts were loaded")
    
    def _get_num_speakers_from_script(self, script: str) -> int:
        """Determine the number of unique speakers in a script."""
        import re
        speakers = set()
        
        lines = script.strip().split('\n')
        for line in lines:
            # Use regex to find speaker patterns
            match = re.match(r'^Speaker\s+(\d+)\s*:', line.strip(), re.IGNORECASE)
            if match:
                speaker_id = int(match.group(1))
                speakers.add(speaker_id)
        
        # If no speakers found, default to 1
        if not speakers:
            return 1
        
        # Return the maximum speaker ID + 1 (assuming 0-based indexing)
        # or the count of unique speakers if they're 1-based
        max_speaker = max(speakers)
        min_speaker = min(speakers)
        
        if min_speaker == 0:
            return max_speaker + 1
        else:
            # Assume 1-based indexing, return the count
            return len(speakers)
    
    def download_model(self, model_name: str, download_path: str = None) -> Iterator[str]:
        """
        Download a VibeVoice model from HuggingFace Hub.
        
        Args:
            model_name: Name of the model to download (e.g., "VibeVoice-1.5B")
            download_path: Path where to download the model (optional)
            
        Yields:
            Status messages during download
        """
        if self.is_downloading:
            yield "❌ Another download is already in progress. Please wait."
            return
            
        if model_name not in self.available_models:
            yield f"❌ Unknown model: {model_name}"
            return
            
        try:
            self.is_downloading = True
            
            model_info = self.available_models[model_name]
            repo_id = model_info["repo_id"]
            
            if download_path is None:
                download_path = f"./models/{model_name}"
            
            # Create download directory if it doesn't exist
            os.makedirs(download_path, exist_ok=True)
            
            yield f"🔄 Starting download of {model_name} from {repo_id}..."
            yield f"📂 Download location: {download_path}"
            yield f"📊 Model size: {model_info['size']}"
            yield f"ℹ️ {model_info['description']}"
            
            # Check if model already exists
            if os.path.exists(download_path) and os.listdir(download_path):
                yield f"⚠️ Model directory already exists at {download_path}"
                yield f"🔄 Checking for updates and downloading missing files..."
            
            # Download the model
            try:
                downloaded_path = snapshot_download(
                    repo_id=repo_id,
                    local_dir=download_path,
                    local_dir_use_symlinks=False,  # Use actual files instead of symlinks
                    resume_download=True,  # Resume if interrupted
                    cache_dir=None,  # Don't use cache, download directly
                )
                
                yield f"✅ Successfully downloaded {model_name}!"
                yield f"📁 Model saved to: {downloaded_path}"
                yield f"💡 You can now use this model by setting --model_path {downloaded_path}"
                
            except Exception as download_error:
                yield f"❌ Download failed: {str(download_error)}"
                yield f"💡 This might be due to network issues or insufficient disk space."
                
        except Exception as e:
            yield f"❌ An error occurred during download: {str(e)}"
            
        finally:
            self.is_downloading = False
    
    def get_model_download_status(self, model_name: str, download_path: str = None) -> str:
        """Check if a model is already downloaded."""
        if download_path is None:
            download_path = f"./models/{model_name}"
            
        if os.path.exists(download_path) and os.listdir(download_path):
            # Check for key model files
            key_files = ["config.json", "pytorch_model.bin", "model.safetensors"]
            has_model_files = any(
                any(f.endswith(key_file) for f in os.listdir(download_path)) 
                for key_file in key_files
            )
            
            if has_model_files:
                return f"✅ Already downloaded at {download_path}"
            else:
                return f"⚠️ Directory exists but may be incomplete at {download_path}"
        else:
            return f"❌ Not downloaded"
    
    def set_attention_implementation(self, use_flash_attention: bool):
        """Set the attention implementation preference."""
        self.use_flash_attention = use_flash_attention
        attention_type = "Flash Attention 2" if use_flash_attention else "SDPA"
        return f"✅ Attention implementation set to: {attention_type}"
    

def create_demo_interface(demo_instance: VibeVoiceDemo):
    """Create the Gradio interface with streaming support."""
    
    # Custom CSS for both light and dark mode support
    custom_css = """
    /* Base theme with CSS variables for light/dark mode */
    .gradio-container {
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Light mode (default) */
    .gradio-container {
        --bg-primary: #f8fafc;
        --bg-secondary: #e2e8f0;
        --bg-card: rgba(255, 255, 255, 0.8);
        --bg-card-border: rgba(226, 232, 240, 0.8);
        --text-primary: #1e293b;
        --text-secondary: #374151;
        --text-muted: #64748b;
        --speaker-bg: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%);
        --speaker-border: rgba(148, 163, 184, 0.4);
        --audio-bg: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        --complete-audio-bg: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        --complete-audio-border: rgba(34, 197, 94, 0.3);
        --slider-bg: rgba(248, 250, 252, 0.8);
        --slider-border: rgba(226, 232, 240, 0.6);
        --queue-bg: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        --queue-border: rgba(14, 165, 233, 0.3);
        --queue-text: #0369a1;
        
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
    }
    
    /* Dark mode */
    .dark .gradio-container {
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --bg-card: rgba(30, 41, 59, 0.8);
        --bg-card-border: rgba(71, 85, 105, 0.8);
        --text-primary: #f1f5f9;
        --text-secondary: #e2e8f0;
        --text-muted: #94a3b8;
        --speaker-bg: linear-gradient(135deg, #334155 0%, #475569 100%);
        --speaker-border: rgba(100, 116, 139, 0.4);
        --audio-bg: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        --complete-audio-bg: linear-gradient(135deg, #14532d 0%, #166534 100%);
        --complete-audio-border: rgba(34, 197, 94, 0.5);
        --slider-bg: rgba(30, 41, 59, 0.8);
        --slider-border: rgba(71, 85, 105, 0.6);
        --queue-bg: linear-gradient(135deg, #0c4a6e 0%, #075985 100%);
        --queue-border: rgba(14, 165, 233, 0.5);
        --queue-text: #38bdf8;
        
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Card styling */
    .settings-card, .generation-card {
        background: var(--bg-card);
        backdrop-filter: blur(10px);
        border: 1px solid var(--bg-card-border);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Speaker selection styling */
    .speaker-grid {
        display: grid;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .speaker-item {
        background: var(--speaker-bg);
        border: 1px solid var(--speaker-border);
        border-radius: 12px;
        padding: 1rem;
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    /* Streaming indicator */
    .streaming-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #22c55e;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.1); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    /* Queue status styling */
    .queue-status {
        background: var(--queue-bg);
        border: 1px solid var(--queue-border);
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        text-align: center;
        font-size: 0.9rem;
        color: var(--queue-text);
    }
    
    .generate-btn {
        background: linear-gradient(135deg, #059669 0%, #0d9488 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 20px rgba(5, 150, 105, 0.4);
        transition: all 0.3s ease;
    }
    
    .generate-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(5, 150, 105, 0.6);
    }
    
    .stop-btn {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.4);
        transition: all 0.3s ease;
    }
    
    .stop-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(239, 68, 68, 0.6);
    }
    
    /* Audio player styling */
    .audio-output {
        background: var(--audio-bg);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid var(--speaker-border);
    }
    
    .complete-audio-section {
        margin-top: 1rem;
        padding: 1rem;
        background: var(--complete-audio-bg);
        border: 1px solid var(--complete-audio-border);
        border-radius: 12px;
    }
    
    /* Text areas - adaptive colors */
    .script-input, .log-output {
        background: var(--bg-card) !important;
        border: 1px solid var(--speaker-border) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .script-input::placeholder {
        color: var(--text-muted) !important;
    }
    
    /* Sliders */
    .slider-container {
        background: var(--slider-bg);
        border: 1px solid var(--slider-border);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Labels and text - adaptive colors */
    .gradio-container label {
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
    }
    
    .gradio-container .markdown {
        color: var(--text-primary) !important;
    }
    
    /* Dark mode specific text overrides */
    .dark .gradio-container .markdown h1,
    .dark .gradio-container .markdown h2,
    .dark .gradio-container .markdown h3,
    .dark .gradio-container .markdown h4 {
        color: var(--text-primary) !important;
    }
    
    .dark .gradio-container .markdown p,
    .dark .gradio-container .markdown li {
        color: var(--text-secondary) !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 2rem; }
        .settings-card, .generation-card { padding: 1rem; }
    }
    
    /* Random example button styling - more subtle professional color */
    .random-btn {
        background: linear-gradient(135deg, #64748b 0%, #475569 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: white;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 20px rgba(100, 116, 139, 0.3);
        transition: all 0.3s ease;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .random-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(100, 116, 139, 0.4);
        background: linear-gradient(135deg, #475569 0%, #334155 100%);
    }
    
    /* Dark mode button adjustments */
    .dark .random-btn {
        background: linear-gradient(135deg, #475569 0%, #334155 100%);
        box-shadow: 0 4px 20px rgba(71, 85, 105, 0.3);
    }
    
    .dark .random-btn:hover {
        background: linear-gradient(135deg, #334155 0%, #1e293b 100%);
        box-shadow: 0 6px 25px rgba(71, 85, 105, 0.4);
    }
    """
    
    with gr.Blocks(
        title="VibeVoice - AI Podcast Generator",
        css=custom_css,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="slate",
        )
    ) as interface:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🎙️ Vibe Podcasting </h1>
            <p>Generating Long-form Multi-speaker AI Podcast with VibeVoice</p>
        </div>
        """)
        
        with gr.Row():
            # Left column - Settings
            with gr.Column(scale=1, elem_classes="settings-card"):
                gr.Markdown("### 🎛️ **Podcast Settings**")
                
                # Number of speakers
                num_speakers = gr.Slider(
                    minimum=1,
                    maximum=4,
                    value=2,
                    step=1,
                    label="Number of Speakers",
                    elem_classes="slider-container"
                )
                
                # Speaker selection
                gr.Markdown("### 🎭 **Speaker Selection**")
                
                # Custom voice upload section
                with gr.Accordion("🎤 Upload Custom Voices", open=False):
                    gr.Markdown("""
                    Upload your own voice samples to create custom speakers:
                    - **Format**: WAV, MP3, FLAC, OGG, M4A, AAC
                    - **Length**: 3-30 seconds recommended
                    - **Quality**: Clear speech, minimal background noise
                    - **Content**: Natural speaking voice works best
                    """)
                    
                    custom_voice_upload = gr.Audio(
                        label="Upload Voice Sample",
                        sources=["upload", "microphone"],
                        type="filepath",
                        interactive=True,
                        elem_classes="script-input"
                    )
                    
                    custom_voice_name = gr.Textbox(
                        label="Voice Name",
                        placeholder="e.g., 'My-Voice' or 'John-Narrator'",
                        elem_classes="script-input"
                    )
                    
                    add_custom_voice_btn = gr.Button(
                        "➕ Add Custom Voice",
                        size="sm",
                        variant="secondary",
                        elem_classes="random-btn"
                    )
                    
                    custom_voice_status = gr.Textbox(
                        label="Upload Status",
                        lines=3,
                        interactive=False,
                        elem_classes="log-output"
                    )
                
                available_speaker_names = list(demo_instance.available_voices.keys())
                # default_speakers = available_speaker_names[:4] if len(available_speaker_names) >= 4 else available_speaker_names
                default_speakers = ['en-Alice_woman', 'en-Carter_man', 'en-Frank_man', 'en-Maya_woman']

                speaker_selections = []
                for i in range(4):
                    default_value = default_speakers[i] if i < len(default_speakers) else None
                    speaker = gr.Dropdown(
                        choices=available_speaker_names,
                        value=default_value,
                        label=f"Speaker {i+1}",
                        visible=(i < 2),  # Initially show only first 2 speakers
                        elem_classes="speaker-item"
                    )
                    speaker_selections.append(speaker)
                
                # Model Status Section
                gr.Markdown("### 📊 **Model Management**")
                
                model_status_display = gr.Textbox(
                    label="Current Model Status",
                    value=demo_instance.get_model_status(),
                    interactive=False,
                    elem_classes="log-output"
                )
                
                # Model selection dropdown
                available_models = demo_instance.get_available_models_for_dropdown()
                model_selection_dropdown = gr.Dropdown(
                    choices=available_models,
                    value=available_models[0] if available_models else None,
                    label="Select Model to Load",
                    elem_classes="speaker-item"
                )
                
                # Refresh models button
                refresh_models_btn = gr.Button(
                    "🔄 Refresh Model List",
                    size="sm",
                    variant="secondary",
                    elem_classes="random-btn"
                )
                
                # Model action buttons row
                with gr.Row():
                    load_selected_model_btn = gr.Button(
                        "📂 Load Selected Model",
                        size="lg",
                        variant="primary",
                        elem_classes="generate-btn",
                        scale=1
                    )
                    
                    unload_model_btn = gr.Button(
                        "🗑️ Unload Model Only works before a generation",
                        size="lg",
                        variant="secondary",
                        elem_classes="stop-btn",
                        scale=2
                    )
                
                # Attention Implementation Settings
                gr.Markdown("**Attention Implementation:**")
                use_flash_attention = gr.Checkbox(
                    label="Use Flash Attention 2 (Uncheck for SDPA)",
                    value=True,  # Default to Flash Attention
                    info="Enable for better performance and memory efficiency. ⚠️UNCHECK BEFORE LOADING A MODEL⚠️",
                    elem_classes="speaker-item"
                )
                
                # Advanced model path input (for custom paths)
                with gr.Accordion("Advanced: Custom Model Path", open=False):
                    model_path_input = gr.Textbox(
                        label="Custom Model Path",
                        placeholder="Enter path to VibeVoice model",
                        elem_classes="script-input"
                    )
                    
                    load_custom_model_btn = gr.Button(
                        "🔄 Load Custom Model",
                        size="sm",
                        variant="secondary",
                        elem_classes="random-btn"
                    )
                
                # Model Download Section
                gr.Markdown("### 📥 **Model Download**")
                
                with gr.Accordion("Download VibeVoice Models", open=False):
                    gr.Markdown("""
                    Download pre-trained VibeVoice models from HuggingFace Hub:
                    
                    - **VibeVoice-1.5B**: Compact model (~2.7B parameters), good for most use cases, ~90 min generation
                    - **VibeVoice-Large**: Large model (~9.34B parameters), higher quality, ~45 min generation
                    """)
                    
                    # Model selection for download
                    model_selection = gr.Radio(
                        choices=["VibeVoice-1.5B", "VibeVoice-Large"],
                        value="VibeVoice-1.5B",
                        label="Select Model to Download",
                        elem_classes="speaker-item"
                    )
                    
                    # Download path input
                    download_path_input = gr.Textbox(
                        label="Download Path (optional)",
                        placeholder="demo/models/VibeVoice-1.5B (leave empty for default)",
                        elem_classes="script-input"
                    )
                    
                    # Download button
                    download_btn = gr.Button(
                        "📥 Download Model",
                        size="lg",
                        variant="secondary",
                        elem_classes="random-btn"
                    )
                    
                    # Download status/log
                    download_status = gr.Textbox(
                        label="Download Status",
                        lines=8,
                        max_lines=15,
                        interactive=False,
                        elem_classes="log-output"
                    )
                
                # Advanced settings
                gr.Markdown("### ⚙️ **Advanced Settings**")
                
                # Sampling parameters (contains all generation settings)
                with gr.Accordion("Generation Parameters", open=open):
                    cfg_scale = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        value=1.3,
                        step=0.05,
                        label="CFG Scale (Guidance Strength)",
                        # info="Higher values increase adherence to text",
                        elem_classes="slider-container"
                    )
                    
                    with gr.Row():
                        seed_input = gr.Number(
                            label="Seed (for reproducible results)",
                            value=44,
                            precision=0,
                            info="Leave empty for random seed. Use same seed to reproduce results.",
                            elem_classes="slider-container",
                            scale=3
                        )
                        
                        random_seed_btn = gr.Button(
                            "🎲 Random Seed",
                            size="sm",
                            variant="secondary",
                            elem_classes="random-btn",
                            scale=1
                        )
                
            # Right column - Generation
            with gr.Column(scale=2, elem_classes="generation-card"):
                gr.Markdown("### 📝 **Script Input**")
                
                script_input = gr.Textbox(
                    label="Conversation Script",
                    placeholder="""Enter your podcast script here. You can format it as:

Speaker 1: Welcome to our podcast today!
Speaker 2: Thanks for having me. I'm excited to discuss...

Or paste text directly and it will auto-assign speakers.""",
                    lines=12,
                    max_lines=20,
                    elem_classes="script-input"
                )
                
                # Button row with Random Example on the left and Generate on the right
                with gr.Row():
                    # Random example button (now on the left)
                    random_example_btn = gr.Button(
                        "🎲 Random Example",
                        size="lg",
                        variant="secondary",
                        elem_classes="random-btn",
                        scale=1  # Smaller width
                    )
                    
                    # Generate button (now on the right)
                    generate_btn = gr.Button(
                        "🚀 Generate Podcast",
                        size="lg",
                        variant="primary",
                        elem_classes="generate-btn",
                        scale=2  # Wider than random button
                    )
                
                # Stop button
                stop_btn = gr.Button(
                    "🛑 Stop Generation",
                    size="lg",
                    variant="stop",
                    elem_classes="stop-btn",
                    visible=False
                )
                
                # Streaming status indicator
                streaming_status = gr.HTML(
                    value="""
                    <div style="background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); 
                                border: 1px solid rgba(34, 197, 94, 0.3); 
                                border-radius: 8px; 
                                padding: 0.75rem; 
                                margin: 0.5rem 0;
                                text-align: center;
                                font-size: 0.9rem;
                                color: #166534;">
                        <span class="streaming-indicator"></span>
                        <strong>LIVE STREAMING</strong> - Audio is being generated in real-time
                    </div>
                    """,
                    visible=False,
                    elem_id="streaming-status"
                )
                
                # Output section
                gr.Markdown("### 🎵 **Generated Podcast**")
                
                # Streaming audio output (outside of tabs for simpler handling)
                audio_output = gr.Audio(
                    label="Streaming Audio (Real-time)",
                    type="numpy",
                    elem_classes="audio-output",
                    streaming=True,  # Enable streaming mode
                    autoplay=True,
                    show_download_button=False,  # Explicitly show download button
                    visible=True
                )
                
                # Complete audio output (non-streaming)
                complete_audio_output = gr.Audio(
                    label="Complete Podcast (Download after generation)",
                    type="numpy",
                    elem_classes="audio-output complete-audio-section",
                    streaming=False,  # Non-streaming mode
                    autoplay=False,
                    show_download_button=True,  # Explicitly show download button
                    visible=False  # Initially hidden, shown when audio is ready
                )
                
                gr.Markdown("""
                *💡 **Streaming**: Audio plays as it's being generated (may have slight pauses)  
                *💡 **Complete Audio**: Will appear below after generation finishes*
                """)
                
                # Generation log
                log_output = gr.Textbox(
                    label="Generation Log",
                    lines=8,
                    max_lines=15,
                    interactive=False,
                    elem_classes="log-output"
                )
        
        def update_speaker_visibility(num_speakers):
            updates = []
            for i in range(4):
                updates.append(gr.update(visible=(i < num_speakers)))
            return updates
        
        num_speakers.change(
            fn=update_speaker_visibility,
            inputs=[num_speakers],
            outputs=speaker_selections
        )
        
        # Main generation function with streaming
        def generate_podcast_wrapper(num_speakers, script, *speakers_and_params):
            """Wrapper function to handle the streaming generation call."""
            try:
                # Extract speakers and parameters
                speakers = speakers_and_params[:4]  # First 4 are speaker selections
                cfg_scale = speakers_and_params[4]   # CFG scale
                seed = speakers_and_params[5]        # Seed
                
                # Clear outputs and reset visibility at start
                yield None, gr.update(value=None, visible=False), "🎙️ Starting generation...", gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
                
                # The generator will yield multiple times
                final_log = "Starting generation..."
                
                for streaming_audio, complete_audio, log, streaming_visible in demo_instance.generate_podcast_streaming(
                    num_speakers=int(num_speakers),
                    script=script,
                    speaker_1=speakers[0],
                    speaker_2=speakers[1],
                    speaker_3=speakers[2],
                    speaker_4=speakers[3],
                    cfg_scale=cfg_scale,
                    seed=int(seed) if seed is not None else None
                ):
                    final_log = log
                    
                    # Check if we have complete audio (final yield)
                    if complete_audio is not None:
                        # Final state: clear streaming, show complete audio
                        yield None, gr.update(value=complete_audio, visible=True), log, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
                    else:
                        # Streaming state: update streaming audio only
                        if streaming_audio is not None:
                            yield streaming_audio, gr.update(visible=False), log, streaming_visible, gr.update(visible=False), gr.update(visible=True)
                        else:
                            # No new audio, just update status
                            yield None, gr.update(visible=False), log, streaming_visible, gr.update(visible=False), gr.update(visible=True)

            except Exception as e:
                error_msg = f"❌ A critical error occurred in the wrapper: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                # Reset button states on error
                yield None, gr.update(value=None, visible=False), error_msg, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        
        def stop_generation_handler():
            """Handle stopping generation."""
            demo_instance.stop_audio_generation()
            # Return values for: log_output, streaming_status, generate_btn, stop_btn
            return "🛑 Generation stopped.", gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        
        # Add a clear audio function
        def clear_audio_outputs():
            """Clear both audio outputs before starting new generation."""
            return None, gr.update(value=None, visible=False)

        # Connect generation button with streaming outputs
        generate_btn.click(
            fn=clear_audio_outputs,
            inputs=[],
            outputs=[audio_output, complete_audio_output],
            queue=False
        ).then(  # Immediate UI update to hide Generate, show Stop (non-queued)
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
            inputs=[],
            outputs=[generate_btn, stop_btn],
            queue=False
        ).then(
            fn=generate_podcast_wrapper,
            inputs=[num_speakers, script_input] + speaker_selections + [cfg_scale, seed_input],
            outputs=[audio_output, complete_audio_output, log_output, streaming_status, generate_btn, stop_btn],
            queue=True  # Enable Gradio's built-in queue
        )
        
        # Connect stop button
        stop_btn.click(
            fn=stop_generation_handler,
            inputs=[],
            outputs=[log_output, streaming_status, generate_btn, stop_btn],
            queue=False  # Don't queue stop requests
        ).then(
            # Clear both audio outputs after stopping
            fn=lambda: (None, None),
            inputs=[],
            outputs=[audio_output, complete_audio_output],
            queue=False
        )
        
        # Function to randomly select an example
        def load_random_example():
            """Randomly select and load an example script."""
            import random
            
            # Get available examples
            if hasattr(demo_instance, 'example_scripts') and demo_instance.example_scripts:
                example_scripts = demo_instance.example_scripts
            else:
                # Fallback to default
                example_scripts = [
                    [2, "Speaker 0: Welcome to our AI podcast demonstration!\nSpeaker 1: Thanks for having me. This is exciting!"]
                ]
            
            # Randomly select one
            if example_scripts:
                selected = random.choice(example_scripts)
                num_speakers_value = selected[0]
                script_value = selected[1]
                
                # Return the values to update the UI
                return num_speakers_value, script_value
            
            # Default values if no examples
            return 2, ""
        
        # Function to generate a random seed
        def generate_random_seed():
            """Generate a random seed for reproducible results."""
            import random
            return random.randint(0, 2**31 - 1)
        
        # Connect random example button
        random_example_btn.click(
            fn=load_random_example,
            inputs=[],
            outputs=[num_speakers, script_input],
            queue=False  # Don't queue this simple operation
        )
        
        # Connect random seed button
        random_seed_btn.click(
            fn=generate_random_seed,
            inputs=[],
            outputs=[seed_input],
            queue=False  # Don't queue this simple operation
        )
        

        
        def update_download_placeholder(model_name):
            """Update placeholder text based on selected model."""
            return gr.update(placeholder=f"/models/{model_name} (leave empty for default)")
        
        # Connect model selection to update placeholder
        model_selection.change(
            fn=update_download_placeholder,
            inputs=[model_selection],
            outputs=[download_path_input],
            queue=False
        )
        
        # Model management functions
        def refresh_model_list():
            """Refresh the list of available models."""
            available_models = demo_instance.get_available_models_for_dropdown()
            return gr.update(choices=available_models, value=available_models[0] if available_models else None)
        
        def handle_load_selected_model(selected_model):
            """Handle loading the selected model from dropdown."""
            if not selected_model or "No models found" in selected_model:
                return "❌ No model selected or no models available", demo_instance.get_model_status()
            
            model_path = demo_instance.get_model_path_from_selection(selected_model)
            if not model_path:
                return "❌ Could not determine model path", demo_instance.get_model_status()
            
            result = demo_instance.reload_model(model_path)
            return result, demo_instance.get_model_status()
        
        def handle_load_custom_model(model_path):
            """Handle model reload from custom path."""
            if not model_path.strip():
                return "❌ Please provide a model path", demo_instance.get_model_status()
            
            result = demo_instance.reload_model(model_path.strip())
            return result, demo_instance.get_model_status()
        
        def handle_unload_model():
            """Handle unloading the current model."""
            result = demo_instance.unload_model()
            return result, demo_instance.get_model_status()
        
        def handle_model_download_with_refresh(model_name, download_path):
            """Handle model download and refresh model list after completion."""
            # Clear download path if it's empty
            if not download_path.strip():
                download_path = None
                
            final_download_path = download_path if download_path else f"./models/{model_name}"
                
            # Check current status first
            status = demo_instance.get_model_download_status(model_name, download_path)
            yield f"📋 Current status: {status}\n", demo_instance.get_model_status(), gr.update()
            
            # If already downloaded, ask user if they want to re-download
            if "✅ Already downloaded" in status:
                yield f"{status}\n🔄 Proceeding to check for updates...", demo_instance.get_model_status(), gr.update()
            
            # Stream download progress
            download_completed = False
            for progress_msg in demo_instance.download_model(model_name, download_path):
                yield progress_msg, demo_instance.get_model_status(), gr.update()
                if "✅ Successfully downloaded" in progress_msg:
                    download_completed = True
            
            # If download completed successfully, refresh the model list
            if download_completed:
                yield f"📥 Download completed! Model saved to {final_download_path}", demo_instance.get_model_status(), gr.update()
                # Refresh model list
                available_models = demo_instance.get_available_models_for_dropdown()
                yield f"📥 Download completed! Model saved to {final_download_path}\n🔄 Model list refreshed.", demo_instance.get_model_status(), gr.update(choices=available_models, value=f"📁 {model_name}")
        
        # Connect refresh button
        refresh_models_btn.click(
            fn=refresh_model_list,
            inputs=[],
            outputs=[model_selection_dropdown],
            queue=False
        )
        
        # Connect load selected model button
        load_selected_model_btn.click(
            fn=handle_load_selected_model,
            inputs=[model_selection_dropdown],
            outputs=[download_status, model_status_display],
            queue=False
        )
        
        # Connect unload model button
        unload_model_btn.click(
            fn=handle_unload_model,
            inputs=[],
            outputs=[download_status, model_status_display],
            queue=False
        )
        
        # Connect custom model load button
        load_custom_model_btn.click(
            fn=handle_load_custom_model,
            inputs=[model_path_input],
            outputs=[download_status, model_status_display],
            queue=False
        )
        
        # Connect download button (now with model list refresh)
        download_btn.click(
            fn=handle_model_download_with_refresh,
            inputs=[model_selection, download_path_input],
            outputs=[download_status, model_status_display, model_selection_dropdown],
            queue=True  # Queue download requests
        )
        
        # Custom voice upload functions
        def handle_custom_voice_upload(audio_file, voice_name):
            """Handle custom voice upload and refresh speaker dropdowns."""
            result = demo_instance.add_custom_voice(audio_file, voice_name)
            
            # Refresh the voice list
            updated_voices = demo_instance.refresh_voice_list()
            
            # Create updates for all speaker dropdowns
            dropdown_updates = []
            for i in range(4):
                dropdown_updates.append(gr.update(choices=updated_voices))
            
            return result, *dropdown_updates
        
        # Connect custom voice upload button
        add_custom_voice_btn.click(
            fn=handle_custom_voice_upload,
            inputs=[custom_voice_upload, custom_voice_name],
            outputs=[custom_voice_status] + speaker_selections,
            queue=False
        )
        
        # Function to handle attention implementation change
        def handle_attention_change(use_flash_attention):
            """Handle changing the attention implementation preference."""
            result = demo_instance.set_attention_implementation(use_flash_attention)
            return result
        
        # Connect attention implementation checkbox
        use_flash_attention.change(
            fn=handle_attention_change,
            inputs=[use_flash_attention],
            outputs=[download_status],  # Use download_status to show the result
            queue=False
        )
        
        # Add usage tips in a closed accordion
        with gr.Accordion("💡 Usage Tips", open=False):
            gr.Markdown("""
            **Getting Started:**
            1. **📥 Download Models**: Use the Model Download section to get VibeVoice-1.5B (compact) or VibeVoice-7B-Preview (larger, higher quality)
            2. **📂 Select Model**: Choose which downloaded model to load from the dropdown
            3. **🔄 Load Model**: Click "Load Selected Model" to activate your chosen model
            4. **📊 Check Status**: Monitor loading status in the Model Management section
            
            **Model Management:**
            - **🔄 Refresh Model List**: Scans for newly downloaded models
            - **📂 Load Selected Model**: Load a model from the dropdown list
            - **🗑️ Unload Model**: Remove the current model from memory to free up GPU/CPU resources
            - **📁 Custom Path**: Load models from custom locations using the Advanced section
            - Models are automatically detected in `.demo/models/` and HuggingFace cache
            - **💡 Memory Management**: Unload models before loading new ones to prevent memory issues
            
            **Attention Implementation:**
            - **⚡ Flash Attention 2**: Faster and more memory-efficient (recommended when available)
            - **🔧 SDPA**: Scaled Dot Product Attention (more compatible fallback)
            - Setting is located below the Load/Unload Model buttons
            - Change takes effect when you next load/reload a model
            - Flash Attention 2 may not be available on all systems (e.g., Windows)
            
            **Custom Voices:**
            - **🎤 Upload Voice Samples**: Add your own voice samples (WAV, MP3, FLAC, etc.)
            - **📏 Recommended Length**: 3-30 seconds of clear speech
            - **✨ Best Quality**: Use high-quality recordings with minimal background noise
            - **🔄 Auto-Update**: Custom voices automatically appear in speaker dropdowns
            
            **Generating Podcasts:**
            - Click **🚀 Generate Podcast** to start audio generation (model must be loaded first)
            - **Live Streaming** shows audio as it's generated (may have slight pauses)
            - **Complete Audio** provides the full, uninterrupted podcast after generation
            - During generation, you can click **🛑 Stop Generation** to interrupt the process
            
            **Seeds for Reproducibility:**
            - **🎲 Random Seed**: Leave seed empty for different results each time
            - **🔒 Fixed Seed**: Enter a specific number to reproduce the same generation
            - **🎲 Random Seed Button**: Click to generate a random seed value
            - **💡 Pro Tip**: Save seeds from generations you like to recreate similar results
            """)
        
        # Add example scripts
        gr.Markdown("### 📚 **Example Scripts**")
        
        # Use dynamically loaded examples if available, otherwise provide a default
        if hasattr(demo_instance, 'example_scripts') and demo_instance.example_scripts:
            example_scripts = demo_instance.example_scripts
        else:
            # Fallback to a simple default example if no scripts loaded
            example_scripts = [
                [1, "Speaker 1: Welcome to our AI podcast demonstration! This is a sample script showing how VibeVoice can generate natural-sounding speech."]
            ]
        
        gr.Examples(
            examples=example_scripts,
            inputs=[num_speakers, script_input],
            label="Try these example scripts:"
        )

        # --- Risks & limitations (footer) ---
        gr.Markdown(
            """
## Risks and limitations

While efforts have been made to optimize it through various techniques, it may still produce outputs that are unexpected, biased, or inaccurate. VibeVoice inherits any biases, errors, or omissions produced by its base model (specifically, Qwen2.5 1.5b in this release).
Potential for Deepfakes and Disinformation: High-quality synthetic speech can be misused to create convincing fake audio content for impersonation, fraud, or spreading disinformation. Users must ensure transcripts are reliable, check content accuracy, and avoid using generated content in misleading ways. Users are expected to use the generated content and to deploy the models in a lawful manner, in full compliance with all applicable laws and regulations in the relevant jurisdictions. It is best practice to disclose the use of AI when sharing AI-generated content.
            """,
            elem_classes="generation-card",  # 可选：复用卡片样式
        )
    return interface


def convert_to_16_bit_wav(data):
    # Check if data is a tensor and move to cpu
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    
    # Ensure data is numpy array
    data = np.array(data)

    # Normalize to range [-1, 1] if it's not already
    if np.max(np.abs(data)) > 1.0:
        data = data / np.max(np.abs(data))
    
    # Scale to 16-bit integer range
    data = (data * 32767).astype(np.int16)
    return data


def parse_args():
    parser = argparse.ArgumentParser(description="VibeVoice Gradio Demo")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models",
        help="Path to the VibeVoice model directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=10,
        help="Number of inference steps for DDPM (not exposed to users)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Share the demo publicly via Gradio",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the demo on",
    )
    
    return parser.parse_args()


def main():
    """Main function to run the demo."""
    args = parse_args()
    
    set_seed(42)  # Set a fixed seed for reproducibility

    print("🎙️ Initializing VibeVoice Demo with Streaming Support...")
    
    # Initialize demo instance
    demo_instance = VibeVoiceDemo(
        model_path=args.model_path,
        device=args.device,
        inference_steps=args.inference_steps
    )
    
    # Create interface
    interface = create_demo_interface(demo_instance)
    
    print(f"🚀 Launching demo on port {args.port}")
    print(f"📁 Default model directory: {args.model_path}")
    print(f"🎭 Available voices: {len(demo_instance.available_voices)}")
    print(f"🔴 Streaming mode: ENABLED")
    print(f"🔒 Session isolation: ENABLED")
    print(f"💡 Models can be downloaded and managed through the web interface")
    
    # Launch the interface
    try:
        interface.queue(
            max_size=20,  # Maximum queue size
            default_concurrency_limit=1  # Process one request at a time
        ).launch(
            share=args.share,
            # server_port=args.port,
            server_name="0.0.0.0" if args.share else "127.0.0.1",
            show_error=True,
            show_api=False  # Hide API docs for cleaner interface
        )
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Server error: {e}")
        raise


if __name__ == "__main__":
    main()
