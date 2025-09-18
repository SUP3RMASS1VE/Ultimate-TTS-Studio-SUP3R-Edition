"""
F5-TTS Integration Handler for Ultimate TTS Studio
Handles F5-TTS model management, loading, and inference
"""

import os
import sys
import torch
import torchaudio
import numpy as np
from pathlib import Path
import tempfile
import json
import shutil
from typing import Optional, Tuple, Dict, List
import warnings
import subprocess
import requests
from tqdm import tqdm
import time

# Suppress warnings
warnings.filterwarnings('ignore')

class F5TTSHandler:
    """Handler for F5-TTS integration"""
    
    # Model information based on F5-TTS HuggingFace repository structure
    AVAILABLE_MODELS = {
        'F5-TTS Base': {
            'repo_id': 'SWivid/F5-TTS',
            'model_id': 'F5TTS_Base',
            'files': ['F5TTS_Base/model_1200000.safetensors', 'F5TTS_Base/vocab.txt'],
            'size': '1.35GB',
            'description': 'Base F5-TTS model with good quality',
            'config': 'F5-TTS'  # Changed from 'F5TTS_Base' to 'F5-TTS'
        },
        'F5-TTS v1 Base': {
            'repo_id': 'SWivid/F5-TTS',
            'model_id': 'F5TTS_v1_Base',
            'files': ['F5TTS_v1_Base/model_1250000.safetensors', 'F5TTS_v1_Base/vocab.txt'],
            'size': '1.35GB',
            'description': 'F5-TTS v1 base model (newer version)',
            'config': 'F5-TTS'  # Changed from 'F5TTS_Base' to 'F5-TTS'
        },
        'F5-TTS French': {
            'repo_id': 'RASPIAUDIO/F5-French-MixedSpeakers-reduced',
            'model_id': 'F5TTS_French',
            'files': ['model_last_reduced.pt', 'vocab.txt'],
            'size': '1.35GB',
            'description': 'French F5-TTS model trained on LibriVox data',
            'config': 'F5-TTS'
        },
        'F5-TTS German': {
            'repo_id': 'hvoss-techfak/F5-TTS-German',
            'model_id': 'F5TTS_German',
            'files': ['model_f5tts_german.pt', 'vocab.txt'],
            'size': '1.35GB',
            'description': 'German F5-TTS model trained on Mozilla Common Voice 19.0 & 800 hours Crowdsourced',
            'config': 'F5-TTS'
        },
        'F5-TTS Japanese': {
            'repo_id': 'Jmica/F5TTS',
            'model_id': 'F5TTS_Japanese',
            'files': ['JA_21999120/model_21999120.pt', 'JA_21999120/vocab_japanese.txt'],
            'size': '1.35GB',
            'description': 'Japanese F5-TTS model trained on Emilia 1.7k JA & Galgame Dataset',
            'config': 'F5-TTS'
        },
        'F5-TTS Spanish': {
            'repo_id': 'jpgallegoar/F5-Spanish', 
            'model_id': 'F5TTS_Spanish',
            'files': ['model_1250000.safetensors', 'vocab.txt'],
            'size': '1.35GB',
            'description': 'Spanish F5-TTS model trained on Voxpopuli & Crowdsourced & TEDx data',
            'config': 'F5-TTS'
        }
    }
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models_dir = Path('F5-Models')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        # Isolated cache for third-party dependencies (Whisper)
        self.cache_dir = Path('checkpoints') / 'f5' / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.loaded_models = {}
        self.current_model = None
        self.model = None
        self.vocoder = None
        
        # Try to import F5-TTS
        self.f5_tts_available = self._check_f5_tts_available()
        # Ensure Whisper dependency available for pipelines used by F5 (local snapshot, HF not ModelScope)
        try:
            from huggingface_hub import snapshot_download
            original_env = {
                'TRANSFORMERS_OFFLINE': os.environ.get('TRANSFORMERS_OFFLINE'),
                'HF_HUB_OFFLINE': os.environ.get('HF_HUB_OFFLINE'),
                'HF_HOME': os.environ.get('HF_HOME'),
                'TRANSFORMERS_CACHE': os.environ.get('TRANSFORMERS_CACHE'),
                'HF_HUB_CACHE': os.environ.get('HF_HUB_CACHE'),
                'HUGGINGFACE_HUB_CACHE': os.environ.get('HUGGINGFACE_HUB_CACHE')
            }
            try:
                os.environ.pop('TRANSFORMERS_OFFLINE', None)
                os.environ.pop('HF_HUB_OFFLINE', None)
                os.environ['HF_HOME'] = str(self.cache_dir)
                os.environ['TRANSFORMERS_CACHE'] = str(self.cache_dir)
                os.environ['HF_HUB_CACHE'] = str(self.cache_dir)
                os.environ['HUGGINGFACE_HUB_CACHE'] = str(self.cache_dir)
                # Pre-fetch the Whisper model to avoid ModelScope fallback
                self.whisper_repo = 'openai/whisper-large-v3-turbo'
                self.whisper_local = snapshot_download(
                    repo_id=self.whisper_repo,
                    cache_dir=str(self.cache_dir),
                    local_dir_use_symlinks=False
                )
            except Exception as _:
                self.whisper_local = None
            finally:
                for k, v in original_env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
        except Exception:
            self.whisper_local = None
        # Install redirect patches so any request for the HF id uses the local snapshot path
        try:
            self._install_whisper_redirect_patch()
        except Exception:
            pass

    def _install_whisper_redirect_patch(self):
        """Redirect HF id 'openai/whisper-large-v3-turbo' to a local snapshot path.
        This avoids ModelScope wrappers by patching Transformers Auto classes directly.
        """
        if not self.whisper_local or not os.path.isdir(self.whisper_local):
            return
        try:
            import transformers
            from transformers import AutoConfig, AutoTokenizer, AutoModelForSpeechSeq2Seq
            _target_id = 'openai/whisper-large-v3-turbo'
            _local = self.whisper_local

            def _redir_from_pretrained(orig):
                def wrapper(pretrained_model_name_or_path, *args, **kwargs):
                    if isinstance(pretrained_model_name_or_path, str) and pretrained_model_name_or_path.strip() == _target_id:
                        pretrained_model_name_or_path = _local
                        kwargs.setdefault('local_files_only', True)
                    return orig(pretrained_model_name_or_path, *args, **kwargs)
                return wrapper

            # Patch the most used classes for Whisper pipelines
            AutoConfig.from_pretrained = _redir_from_pretrained(AutoConfig.from_pretrained)
            AutoTokenizer.from_pretrained = _redir_from_pretrained(AutoTokenizer.from_pretrained)
            AutoModelForSpeechSeq2Seq.from_pretrained = _redir_from_pretrained(AutoModelForSpeechSeq2Seq.from_pretrained)
        except Exception:
            pass
        
    def _check_f5_tts_available(self):
        """Check if F5-TTS is installed"""
        try:
            # First, try the new API (as of v1.1.0+)
            try:
                # Try the CLI/API approach first
                import f5_tts
                from f5_tts.api import F5TTS
                
                self.use_new_api = True
                self.F5TTS = F5TTS
                print("✅ F5-TTS API loaded (new version)")
                
                # Still try to import legacy functions for compatibility
                try:
                    from f5_tts.infer.utils_infer import (
                        load_vocoder,
                        load_model,
                        preprocess_ref_audio_text,
                        infer_process
                    )
                    self.f5_imports = {
                        'load_vocoder': load_vocoder,
                        'load_model': load_model,
                        'preprocess_ref_audio_text': preprocess_ref_audio_text,
                        'infer_process': infer_process
                    }
                except:
                    self.f5_imports = None
                
                return True
                
            except ImportError:
                # Fall back to legacy imports
                self.use_new_api = False
                from f5_tts.infer.utils_infer import (
                    load_vocoder,
                    load_model,
                    preprocess_ref_audio_text,
                    infer_process
                )
                
                self.f5_imports = {
                    'load_vocoder': load_vocoder,
                    'load_model': load_model,
                    'preprocess_ref_audio_text': preprocess_ref_audio_text,
                    'infer_process': infer_process
                }
                print("✅ F5-TTS loaded (legacy version)")
                return True
                
        except ImportError as e:
            print(f"⚠️ F5-TTS not installed or import error: {e}")
            print("Please install with: pip install f5-tts")
            return False
    
    def get_model_status(self) -> Dict[str, Dict]:
        """Get status of all available models"""
        status = {}
        for model_name, model_info in self.AVAILABLE_MODELS.items():
            model_path = self.models_dir / model_info['model_id']
            # Check if files exist using just the filename, not full path
            is_downloaded = all(
                (model_path / os.path.basename(file)).exists() 
                for file in model_info['files']
            )
            
            status[model_name] = {
                'downloaded': is_downloaded,
                'loaded': model_name in self.loaded_models,
                'size': model_info['size'],
                'description': model_info['description'],
                'path': str(model_path) if is_downloaded else None
            }
        
        return status
    
    def download_model(self, model_name: str, progress_callback=None) -> Tuple[bool, str]:
        """Download a specific F5-TTS model"""
        if model_name not in self.AVAILABLE_MODELS:
            return False, f"Unknown model: {model_name}"
        
        model_info = self.AVAILABLE_MODELS[model_name]
        model_path = self.models_dir / model_info['model_id']
        model_path.mkdir(parents=True, exist_ok=True)
        
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            return False, "Please install huggingface-hub: pip install huggingface-hub"
        
        print(f"\n{'='*60}")
        print(f"📥 Starting download: {model_name}")
        print(f"📦 Repository: {model_info['repo_id']}")
        print(f"📊 Model size: {model_info['size']}")
        print(f"📝 Description: {model_info['description']}")
        print(f"{'='*60}\n")
        
        try:
            for i, file in enumerate(model_info['files'], 1):
                # Extract just the filename from the full path
                filename = os.path.basename(file)
                file_path = model_path / filename
                
                if file_path.exists():
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    print(f"✅ [{i}/{len(model_info['files'])}] {filename} already exists ({file_size_mb:.1f} MB)")
                    if progress_callback:
                        progress_callback(f"✓ {filename} already exists")
                    continue
                
                print(f"⬇️  [{i}/{len(model_info['files'])}] Downloading {filename}...")
                print(f"   From: {model_info['repo_id']}/{file}")
                if progress_callback:
                    progress_callback(f"Downloading {filename}...")
                
                # Download directly to target location to avoid duplication
                print(f"   Connecting to Hugging Face Hub...")
                print(f"   Expected size: ~{model_info['size']}")
                
                # Create a simple animated progress indicator
                import threading
                import urllib.request
                from huggingface_hub import hf_hub_url
                
                download_complete = False
                
                def show_progress_animation():
                    """Show animated progress while downloading"""
                    animation = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
                    idx = 0
                    start_time = time.time()
                    
                    while not download_complete:
                        elapsed = time.time() - start_time
                        mins, secs = divmod(int(elapsed), 60)
                        time_str = f"{mins:02d}:{secs:02d}"
                        
                        # Print progress animation
                        print(f"\r   {animation[idx % len(animation)]} Downloading... [{time_str}]", 
                              end='', flush=True)
                        
                        idx += 1
                        time.sleep(0.1)
                    
                    # Clear the line when done
                    print(f"\r   ✓ Download complete!{' ' * 20}", flush=True)
                
                # Get the download URL from Hugging Face
                download_url = hf_hub_url(
                    repo_id=model_info['repo_id'],
                    filename=file,
                    repo_type="model"
                )
                
                # Start progress animation in background thread
                progress_thread = threading.Thread(target=show_progress_animation)
                progress_thread.daemon = True
                progress_thread.start()
                
                try:
                    # Download directly to target location
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                    
                    # Create a request with headers
                    request = urllib.request.Request(download_url, headers=headers)
                    
                    # Download the file directly to our target location
                    with urllib.request.urlopen(request) as response:
                        # Make sure parent directory exists
                        if not file_path.parent.exists():
                            file_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Write directly to target file
                        with open(file_path, 'wb') as out_file:
                            # Read in chunks to handle large files
                            chunk_size = 8192
                            while True:
                                chunk = response.read(chunk_size)
                                if not chunk:
                                    break
                                out_file.write(chunk)
                    
                    downloaded_path = str(file_path)
                    
                finally:
                    # Stop progress animation
                    download_complete = True
                    progress_thread.join(timeout=0.5)
                
                # Get file size for display
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                
                print(f"✅ [{i}/{len(model_info['files'])}] Downloaded {filename} ({file_size_mb:.1f} MB)")
                if progress_callback:
                    progress_callback(f"✓ Downloaded {filename}")
                print()  # Add spacing between files
            
            # Save config
            config_path = model_path / 'config.json'
            config_data = {
                'model_name': model_name,
                'model_id': model_info['model_id'],
                'config': model_info['config'],
                'files': model_info['files']
            }
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            print(f"\n{'='*60}")
            print(f"✅ Successfully downloaded {model_name}")
            print(f"📁 Location: {model_path}")
            print(f"{'='*60}\n")
            
            return True, f"Successfully downloaded {model_name}"
            
        except KeyboardInterrupt:
            print(f"\n\n❌ Download interrupted by user")
            return False, "Download cancelled by user"
            
        except Exception as e:
            print(f"\n\n❌ Error downloading {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, f"Failed to download {model_name}: {str(e)}"
    
    def load_model(self, model_name: str) -> Tuple[bool, str]:
        """Load a specific F5-TTS model"""
        if not self.f5_tts_available:
            return False, "F5-TTS is not installed"
        
        if model_name not in self.AVAILABLE_MODELS:
            return False, f"Unknown model: {model_name}"
        
        model_info = self.AVAILABLE_MODELS[model_name]
        model_path = self.models_dir / model_info['model_id']
        
        # Check if model is downloaded (use basename for checking)
        if not all((model_path / os.path.basename(file)).exists() for file in model_info['files']):
            return False, f"Model {model_name} not downloaded. Please download it first."
        
        try:
            if self.use_new_api:
                # Use new API
                print(f"Loading F5-TTS model using new API: {model_name}")
                
                # Find model file
                model_files = [model_path / os.path.basename(f) for f in model_info['files']]
                model_file = model_files[0]  # Main model file
                
                # Map our model names to F5-TTS model identifiers
                model_map = {
                    'F5-TTS Base': 'F5TTS_Base',
                    'F5-TTS v1 Base': 'F5TTS_v1_Base',
                    'F5-TTS French': 'F5TTS_Base',
                    'F5-TTS German': 'F5TTS_Base',
                    'F5-TTS Japanese': 'F5TTS_Base',
                    'F5-TTS Spanish': 'F5TTS_Base'
                }
                
                model_id = model_map.get(model_name, 'F5TTS_Base')
                
                # Find vocab file if it exists
                vocab_file = None
                for f in model_files:
                    if 'vocab' in str(f) and f.exists():
                        vocab_file = str(f)
                        break
                
                # Initialize F5TTS with model path
                # Force Whisper to load from local snapshot if available
                whisper_path_override = getattr(self, 'whisper_local', None)
                if whisper_path_override and os.path.isdir(whisper_path_override):
                    os.environ['HF_HOME'] = str(self.cache_dir)
                    os.environ['TRANSFORMERS_CACHE'] = str(self.cache_dir)
                    os.environ['HF_HUB_CACHE'] = str(self.cache_dir)
                    os.environ['HUGGINGFACE_HUB_CACHE'] = str(self.cache_dir)
                self.model = self.F5TTS(
                    model=model_id,
                    ckpt_file=str(model_file),
                    vocab_file=vocab_file or '',  # Empty string if no vocab file
                    device=self.device
                )
                
                print(f"✅ Model loaded successfully using new API: {model_name}")
                
            else:
                # Use legacy method
                # Load vocoder if not already loaded
                if self.vocoder is None and self.f5_imports:
                    self.vocoder = self.f5_imports['load_vocoder'](
                        vocoder_name="vocos",
                        device=self.device
                    )
                
                # Load the model (use basename to get actual filename)
                model_files = [model_path / os.path.basename(f) for f in model_info['files']]
                model_file = model_files[0]  # Main model file is always first
                
                # Check if vocab file exists
                vocab_file = None
                for f in model_files:
                    if 'vocab' in str(f):
                        vocab_file = f
                        break
                
                # Load the model with proper parameters
                print(f"Loading F5-TTS model from: {model_file}")
                if vocab_file and vocab_file.exists():
                    print(f"Using vocab file: {vocab_file}")
                
                # Try different parameter combinations for load_model
                if vocab_file and vocab_file.exists():
                    # Try with vocab_file parameter
                    self.model = self.f5_imports['load_model'](
                        model_cls=model_info['config'],
                        model_cfg={},  # Empty dict instead of None
                        ckpt_path=str(model_file),
                        vocab_file=str(vocab_file),
                        device=self.device
                    )
                else:
                    # Try without vocab_file
                    self.model = self.f5_imports['load_model'](
                        model_cls=model_info['config'],
                        model_cfg={},
                        ckpt_path=str(model_file),
                        device=self.device
                    )
                
                print(f"✅ Model loaded successfully (legacy method): {model_name}")
            
            self.current_model = model_name
            self.loaded_models[model_name] = self.model
            
            return True, f"Successfully loaded {model_name}"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"Failed to load {model_name}: {str(e)}"
    
    def unload_model(self, model_name: str = None) -> str:
        """Unload a specific model or all models"""
        if model_name is None:
            # If no specific model is given, unload ALL models
            if self.loaded_models or self.model or self.vocoder:
                models_to_unload = list(self.loaded_models.keys()) if self.loaded_models else []
                
                # Clear all loaded models
                for model in models_to_unload:
                    del self.loaded_models[model]
                
                # Clear the main model
                if self.model is not None:
                    # If using new API, try to clean up the F5TTS instance
                    if self.use_new_api and hasattr(self.model, '__del__'):
                        try:
                            del self.model
                        except:
                            pass
                    self.model = None
                
                # Clear vocoder (used in legacy mode)
                if self.vocoder is not None:
                    del self.vocoder
                    self.vocoder = None
                
                self.current_model = None
                
                # Clear F5-TTS cached models (vocos and whisper)
                # These are typically cached in the transformers/huggingface cache
                # We'll try to clear them from memory
                try:
                    # Try to clear any cached models in the global scope
                    import sys
                    # Look for and delete F5-TTS related modules to force cleanup
                    modules_to_clear = []
                    for module_name in sys.modules:
                        if 'f5_tts' in module_name or 'vocos' in module_name or 'whisper' in module_name:
                            modules_to_clear.append(module_name)
                    
                    # Clear the modules
                    for module_name in modules_to_clear:
                        if module_name in sys.modules:
                            del sys.modules[module_name]
                except:
                    pass
                
                # Force garbage collection multiple times to ensure cleanup
                import gc
                for _ in range(3):
                    gc.collect()
                
                # Clear CUDA cache if using GPU
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # Try to reset CUDA memory
                    try:
                        torch.cuda.reset_peak_memory_stats()
                    except:
                        pass
                
                unload_msg = f"Unloaded all {len(models_to_unload)} F5-TTS model(s)"
                if self.vocoder is not None:
                    unload_msg += " and vocoder"
                unload_msg += " (cleared vocos-mel-24khz and whisper-large-v3-turbo from memory)"
                
                return unload_msg
            else:
                return "No models to unload"
        else:
            # Unload specific model
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
                if self.current_model == model_name:
                    if self.model is not None:
                        if self.use_new_api and hasattr(self.model, '__del__'):
                            try:
                                del self.model
                            except:
                                pass
                        self.model = None
                    self.current_model = None
                
                # Force garbage collection
                import gc
                gc.collect()
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                
                return f"Unloaded {model_name}"
            else:
                return f"Model {model_name} not loaded"
    
    def generate_speech(
        self,
        text: str,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        target_sample_rate: int = 24000,
        speed: float = 1.0,
        cross_fade_duration: float = 0.15,
        remove_silence: bool = False,
        seed: Optional[int] = None
    ) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
        """Generate speech using F5-TTS"""
        
        if not self.f5_tts_available:
            return None, "F5-TTS is not installed"
        
        if self.model is None:
            return None, "No model loaded. Please load a model first."
        
        if not text.strip():
            return None, "Please provide text to synthesize"
        
        print(f"🎵 F5-TTS generating speech with model: {self.current_model}")
        print(f"   Text: {text[:50]}...")
        print(f"   Ref audio: {ref_audio_path}")
        print(f"   Speed: {speed}, Cross-fade: {cross_fade_duration}")
        
        try:
            # Set seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                if self.device == 'cuda':
                    torch.cuda.manual_seed(seed)
                np.random.seed(seed)
            
            if self.use_new_api:
                # Use new API
                print("Using F5-TTS new API for generation...")
                
                # Prepare reference audio path - default to sample if not provided
                if not ref_audio_path or not os.path.exists(ref_audio_path):
                    # Try to use a default sample
                    sample_path = Path("sample/Sample.wav")
                    if sample_path.exists():
                        ref_audio_path = str(sample_path)
                        print(f"Using default sample audio: {ref_audio_path}")
                
                # Generate using new API
                # F5-TTS infer returns (audio, sample_rate, spectrogram)
                result = self.model.infer(
                    ref_file=ref_audio_path,
                    ref_text=ref_text or "",
                    gen_text=text,
                    speed=speed,
                    cross_fade_duration=cross_fade_duration,
                    seed=seed,
                    remove_silence=remove_silence
                )
                
                # Unpack the result
                if isinstance(result, tuple) and len(result) >= 2:
                    audio_data = result[0]
                    sample_rate = result[1]
                    # result[2] is spectrogram, which we don't need
                else:
                    # Fallback if format is different
                    audio_data = result
                    sample_rate = target_sample_rate
                
                # Convert to numpy if needed
                if isinstance(audio_data, torch.Tensor):
                    audio_np = audio_data.cpu().numpy()
                else:
                    audio_np = audio_data
                
                if len(audio_np.shape) > 1:
                    audio_np = audio_np.squeeze()
                
            else:
                # Use legacy method
                if not self.f5_imports:
                    return None, "Legacy F5-TTS imports not available"
                
                # Preprocess reference audio and text
                if ref_audio_path and os.path.exists(ref_audio_path):
                    ref_audio, ref_text_processed = self.f5_imports['preprocess_ref_audio_text'](
                        ref_audio_path,
                        ref_text or "",
                        device=self.device
                    )
                else:
                    # Use default or generate placeholder
                    ref_audio = torch.randn(1, 1, 16000).to(self.device)  # 1 second of noise
                    ref_text_processed = ""
                
                # Generate speech
                # F5-TTS infer_process returns (audio, sr, spectrogram)
                result = self.f5_imports['infer_process'](
                    ref_audio,
                    ref_text_processed,
                    text,
                    self.model,
                    self.vocoder,
                    speed=speed,
                    cross_fade_duration=cross_fade_duration,
                    device=self.device
                )
                
                # Handle different return types
                if isinstance(result, tuple) and len(result) >= 2:
                    generated_audio = result[0]  # audio
                    sample_rate = result[1] if len(result) > 1 else target_sample_rate
                else:
                    generated_audio = result
                    sample_rate = target_sample_rate
                
                # Convert to numpy array
                if isinstance(generated_audio, torch.Tensor):
                    audio_np = generated_audio.cpu().numpy()
                    if len(audio_np.shape) > 1:
                        audio_np = audio_np.squeeze()
                else:
                    audio_np = generated_audio
                
                # Remove silence if requested
                if remove_silence:
                    audio_np = self._remove_silence(audio_np, sample_rate)
            
            # Normalize audio
            audio_np = self._normalize_audio(audio_np)
            
            # Ensure we return the correct sample rate
            final_sample_rate = sample_rate if 'sample_rate' in locals() else target_sample_rate
            
            return (final_sample_rate, audio_np), f"Generated with {self.current_model}"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"Generation failed: {str(e)}"
    
    def _remove_silence(self, audio: np.ndarray, sr: int, threshold: float = 0.01) -> np.ndarray:
        """Remove silence from beginning and end of audio"""
        # Simple energy-based silence removal
        energy = np.abs(audio)
        indices = np.where(energy > threshold)[0]
        
        if len(indices) > 0:
            return audio[indices[0]:indices[-1] + 1]
        return audio
    
    def _normalize_audio(self, audio: np.ndarray, target_level: float = -3.0) -> np.ndarray:
        """Normalize audio to target level"""
        # Find peak
        peak = np.max(np.abs(audio))
        if peak == 0:
            return audio
        
        # Calculate target based on dB
        target_linear = 10 ** (target_level / 20.0)
        
        # Normalize
        normalized = audio * (target_linear / peak)
        
        # Soft clipping
        normalized = np.clip(normalized, -0.99, 0.99)
        
        return normalized
    
    def list_voices(self) -> List[str]:
        """List available voice presets (for future implementation)"""
        # F5-TTS uses reference audio for voice cloning
        # This could be expanded to include preset voices
        return ["Custom (Reference Audio)"]
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        if self.current_model:
            return {
                'model': self.current_model,
                'loaded': True,
                'device': self.device,
                'info': self.AVAILABLE_MODELS[self.current_model]
            }
        return {
            'model': None,
            'loaded': False,
            'device': self.device
        }

# Global instance
f5_tts_handler = None

def get_f5_tts_handler():
    """Get or create F5-TTS handler instance"""
    global f5_tts_handler
    if f5_tts_handler is None:
        print("Creating new F5-TTS handler instance...")
        f5_tts_handler = F5TTSHandler()
    return f5_tts_handler 