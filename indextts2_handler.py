"""
IndexTTS2 Handler for Ultimate TTS Studio
Provides integration with IndexTTS-2 Text-to-Speech system with advanced emotion control
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

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup IndexTTS2 module path and cache directories
current_dir = os.path.dirname(os.path.abspath(__file__))
indextts2_base_path = os.path.join(current_dir, 'indextts2')
indextts2_module_path = os.path.join(indextts2_base_path, 'indextts')

# Set up cache directory early to avoid ModelScope conflicts
cache_dir = os.path.join(current_dir, 'checkpoints', 'indextts2', 'cache')
os.makedirs(cache_dir, exist_ok=True)

# Set environment variables before any imports
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HUB_CACHE'] = cache_dir
os.environ['MODELSCOPE_CACHE'] = cache_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir

# Add paths to sys.path for proper module resolution
# Ensure the app root is included so implicit namespace package 'indextts2' works
paths_to_add = [current_dir, indextts2_base_path, indextts2_module_path]
for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

# Global handler instance
_indextts2_handler = None
INDEXTTS2_AVAILABLE = False

# Try multiple import strategies
def try_import_indextts2():
    """Try different import strategies for IndexTTS2"""
    global INDEXTTS2_AVAILABLE
    
    # Strategy 1: Set up proper module structure and import
    try:
        import importlib
        # Import the bundled package and alias as top-level 'indextts' for internal absolute imports
        indextts_pkg = importlib.import_module('indextts2.indextts')
        sys.modules['indextts'] = indextts_pkg
        
        # Import IndexTTS2
        from indextts2.indextts.infer_v2 import IndexTTS2
        INDEXTTS2_AVAILABLE = True
        print("✅ IndexTTS2 loaded successfully (Strategy 1)")
        return IndexTTS2
    except Exception as e1:
        print(f"⚠️ Strategy 1 failed: {e1}")
    
    # Strategy 2: Direct file loading with module setup
    try:
        import importlib
        import importlib.util
        
        # Ensure 'indextts' alias points to bundled package for internal imports
        try:
            indextts_pkg = importlib.import_module('indextts2.indextts')
            sys.modules['indextts'] = indextts_pkg
        except Exception:
            pass
        
        # Load infer_v2.py directly but set up the environment first
        infer_v2_path = os.path.join(indextts2_module_path, "infer_v2.py")
        spec = importlib.util.spec_from_file_location("indextts2.indextts.infer_v2", infer_v2_path)
        infer_v2_module = importlib.util.module_from_spec(spec)
        
        # Register both names for compatibility
        sys.modules['indextts2.indextts.infer_v2'] = infer_v2_module
        sys.modules['indextts.infer_v2'] = infer_v2_module
        
        # Execute the module
        spec.loader.exec_module(infer_v2_module)
        IndexTTS2 = infer_v2_module.IndexTTS2
        INDEXTTS2_AVAILABLE = True
        print("✅ IndexTTS2 loaded successfully (Strategy 2)")
        return IndexTTS2
    except Exception as e2:
        print(f"⚠️ Strategy 2 failed: {e2}")
    
    # All strategies failed
    INDEXTTS2_AVAILABLE = False
    print("❌ All import strategies failed")
    print("   IndexTTS2 requires complex module dependencies that couldn't be resolved")
    print("   This is a known limitation with the current IndexTTS2 package structure")
    return None

# Try to import IndexTTS2
IndexTTS2 = try_import_indextts2()

def check_indextts2_models():
    """Check if IndexTTS2 models are available"""
    model_dir = Path("checkpoints/indextts2")
    config_path = model_dir / "config.yaml"
    
    if not config_path.exists():
        return False
    
    # Check for essential IndexTTS-2 model files
    essential_files = ["gpt.pth", "s2mel.pth", "bpe.model"]
    
    for filename in essential_files:
        if not (model_dir / filename).exists():
            return False
    
    return True

def ensure_indextts2_dependencies():
    """Ensure all IndexTTS2 dependencies are available"""
    try:
        from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
        from huggingface_hub import hf_hub_download, snapshot_download
        
        model_dir = Path("checkpoints/indextts2")
        cache_dir = model_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables to use local cache
        os.environ['HF_HOME'] = str(cache_dir)
        os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
        os.environ['HF_HUB_CACHE'] = str(cache_dir)
        os.environ['MODELSCOPE_CACHE'] = str(cache_dir)
        
        print("🔧 Ensuring IndexTTS2 dependencies are available...")
        
        # Check and download facebook/w2v-bert-2.0 (feature extractor and model)
        try:
            print("   🔍 Checking facebook/w2v-bert-2.0...")
            repo_id = "facebook/w2v-bert-2.0"
            # Prefer a concrete snapshot path inside our cache if it already exists
            snapshots_dir = cache_dir / f"models--{repo_id.replace('/', '--')}" / "snapshots"
            local_repo_path = None
            if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
                # pick most recent snapshot
                local_repo_path = str(max(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime))
            if local_repo_path is None:
                local_repo_path = snapshot_download(repo_id=repo_id, cache_dir=str(cache_dir))

            # Validate both processor and model load locally (no remote access)
            SeamlessM4TFeatureExtractor.from_pretrained(local_repo_path, local_files_only=True)
            Wav2Vec2BertModel.from_pretrained(local_repo_path, local_files_only=True)
            print("   ✅ facebook/w2v-bert-2.0 ready")
        except Exception:
            # Non-fatal: real loading will happen later with fallback logic
            print("   ⚠️ facebook/w2v-bert-2.0 not fully validated; will download on demand")
        
        # Check and download MaskGCT semantic codec
        try:
            print("   🔍 Checking amphion/MaskGCT...")
            hf_hub_download(
                "amphion/MaskGCT",
                filename="semantic_codec/model.safetensors",
                cache_dir=str(cache_dir)
            )
            print("   ✅ MaskGCT semantic codec ready")
        except Exception as e:
            print(f"   ⚠️ MaskGCT issue: {e}")
        
        # Check and download campplus
        try:
            print("   🔍 Checking funasr/campplus...")
            hf_hub_download(
                "funasr/campplus",
                filename="campplus_cn_common.bin",
                cache_dir=str(cache_dir)
            )
            print("   ✅ campplus ready")
        except Exception as e:
            print(f"   ⚠️ campplus issue: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error ensuring dependencies: {e}")
        return False

def download_indextts2_models():
    """Download IndexTTS2 models and dependencies from HuggingFace"""
    try:
        from huggingface_hub import hf_hub_download
        from transformers import SeamlessM4TFeatureExtractor
        import requests
    except ImportError:
        print("⚠️ Cannot auto-download IndexTTS2 models - missing huggingface_hub or transformers")
        print("   Install with: pip install huggingface_hub transformers requests")
        return False
    
    repo_id = "IndexTeam/IndexTTS-2"
    model_dir = Path("checkpoints/indextts2")
    
    # Create directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎯 Auto-downloading IndexTTS-2 models and dependencies...")
    print("   This may take several minutes on first run...")
    
    # Step 1: Download main IndexTTS-2 model files
    required_files = [
        "config.yaml",
        "bpe.model", 
        "gpt.pth",
        "s2mel.pth",
        "feat1.pt",
        "feat2.pt",
        "wav2vec2bert_stats.pt"
    ]
    
    # Qwen emotion model files (in subfolder)
    qwen_files = [
        "qwen0.6bemo4-merge/config.json",
        "qwen0.6bemo4-merge/generation_config.json",
        "qwen0.6bemo4-merge/model.safetensors",
        "qwen0.6bemo4-merge/tokenizer.json",
        "qwen0.6bemo4-merge/tokenizer_config.json",
        "qwen0.6bemo4-merge/vocab.json"
    ]
    
    all_files = required_files + qwen_files
    
    print("📥 Downloading IndexTTS-2 main models...")
    for filename in all_files:
        file_path = model_dir / filename
        
        if file_path.exists():
            print(f"   ✅ {filename} already exists")
            continue
            
        try:
            print(f"   ⬇️ Downloading {filename}...")
            
            # Create subdirectories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(model_dir),
                local_dir_use_symlinks=False
            )
            
            print(f"   ✅ {filename} downloaded")
            
        except Exception as e:
            print(f"   ❌ Failed to download {filename}: {e}")
            continue
    
    # Step 2: Download dependency models
    print("📥 Downloading IndexTTS-2 dependencies...")
    
    try:
        print("   ⬇️ Downloading facebook/w2v-bert-2.0 feature extractor...")
        # Set cache directory before downloading
        cache_dir = str(model_dir / "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Set environment variables to use local cache
        os.environ['HF_HOME'] = cache_dir
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        os.environ['HF_HUB_CACHE'] = cache_dir
        
        # Download to local cache
        SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0",
            cache_dir=cache_dir
        )
        print("   ✅ facebook/w2v-bert-2.0 downloaded to local cache")
    except Exception as e:
        print(f"   ❌ Failed to download facebook/w2v-bert-2.0: {e}")
        print("   💡 This model will be downloaded automatically when needed")
    
    try:
        print("   ⬇️ Downloading amphion/MaskGCT semantic codec...")
        # Download the semantic codec model
        semantic_codec_path = hf_hub_download(
            "amphion/MaskGCT", 
            filename="semantic_codec/model.safetensors",
            cache_dir=str(model_dir / "cache")
        )
        print(f"   ✅ MaskGCT semantic codec downloaded to {semantic_codec_path}")
    except Exception as e:
        print(f"   ❌ Failed to download MaskGCT semantic codec: {e}")
    
    try:
        print("   ⬇️ Downloading funasr/campplus model...")
        # Download campplus model
        campplus_path = hf_hub_download(
            "funasr/campplus",
            filename="campplus_cn_common.bin",
            cache_dir=str(model_dir / "cache")
        )
        print(f"   ✅ campplus model downloaded to {campplus_path}")
    except Exception as e:
        print(f"   ❌ Failed to download campplus model: {e}")
    
    # Step 3: Check if essential files are present
    essential_files = ["config.yaml", "gpt.pth", "s2mel.pth", "bpe.model"]
    missing_essential = []
    
    for filename in essential_files:
        if not (model_dir / filename).exists():
            missing_essential.append(filename)
    
    if missing_essential:
        print(f"❌ Essential files missing: {missing_essential}")
        return False
    
    print("🎉 IndexTTS-2 models and dependencies ready!")
    print("💡 Note: Some models are cached by transformers/huggingface_hub")
    return True

def get_indextts2_handler():
    """Get the global IndexTTS2 handler instance (singleton)"""
    global _indextts2_handler
    if _indextts2_handler is None:
        _indextts2_handler = IndexTTS2Handler()
    return _indextts2_handler

class IndexTTS2Handler:
    """Handler for IndexTTS2 TTS system with advanced emotion control"""
    
    def __init__(self):
        self.model = None
        self.device = self._get_device()
        self.sample_rate = 22050
        self.model_path = "IndexTeam/IndexTTS-2"
        self.checkpoints_dir = Path("checkpoints/indextts2")
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Emotion control modes
        self.emotion_modes = {
            'audio_reference': 'Use audio file for emotion reference',
            'vector_control': 'Manual emotion vector adjustment',
            'text_description': 'Natural language emotion description'
        }
        
        # Emotion vectors for manual control
        self.emotion_vectors = {
            'happy': 0.0,
            'angry': 0.0,
            'sad': 0.0,
            'afraid': 0.0,
            'disgusted': 0.0,
            'melancholic': 0.0,
            'surprised': 0.0,
            'calm': 0.0
        }
        
    def _get_device(self):
        """Get the appropriate device for inference"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def initialize_model(self):
        """Initialize the IndexTTS2 model"""
        if not INDEXTTS2_AVAILABLE or IndexTTS2 is None:
            return False, "❌ IndexTTS2 not available"
        
        try:
            print("🎯 Initializing IndexTTS2 model...")
            
            # Set up cache directories to avoid ModelScope conflicts
            cache_dir = str(self.checkpoints_dir / "cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Set environment variables to use local cache (multiple cache systems)
            os.environ['HF_HOME'] = cache_dir
            os.environ['TRANSFORMERS_CACHE'] = cache_dir
            os.environ['HF_HUB_CACHE'] = cache_dir
            os.environ['MODELSCOPE_CACHE'] = cache_dir
            os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
            
            # Also set the cache directory in the IndexTTS2 infer_v2.py file
            # This overrides the hardcoded cache path
            original_hf_cache = os.environ.get('HF_HUB_CACHE', './checkpoints/hf_cache')
            os.environ['HF_HUB_CACHE'] = cache_dir
            
            # Check if models are available, download if needed
            if not check_indextts2_models():
                print("📥 IndexTTS2 models not found, downloading...")
                if not download_indextts2_models():
                    return False, "❌ Failed to download IndexTTS2 models"
            
            # Ensure all dependencies are available
            print("🔧 Ensuring dependencies are ready...")
            try:
                ensure_indextts2_dependencies()
            except Exception as dep_error:
                print(f"⚠️ Dependency check failed: {dep_error}")
                print("💡 Continuing with initialization - dependencies will be downloaded as needed")
            
            # Initialize IndexTTS2 with config path and model directory
            config_path = str(self.checkpoints_dir / "config.yaml")
            model_dir = str(self.checkpoints_dir)
            
            print(f"🔧 Using config: {config_path}")
            print(f"🔧 Using model dir: {model_dir}")
            print(f"🔧 Using cache dir: {cache_dir}")
            
            self.model = IndexTTS2(
                cfg_path=config_path,
                model_dir=model_dir,
                device=self.device,
                use_fp16=self.device != "cpu"
            )
            
            print(f"✅ IndexTTS2 model loaded on {self.device}")
            return True, "✅ IndexTTS2 model loaded successfully"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ Error initializing IndexTTS2: {e}")
            return False, f"❌ Error initializing IndexTTS2: {str(e)}"
    
    def unload_model(self):
        """Unload the IndexTTS2 model to free memory"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return "✅ IndexTTS2 model unloaded successfully"
            
        except Exception as e:
            return f"⚠️ Error unloading IndexTTS2: {str(e)}"
    
    def is_model_loaded(self):
        """Check if the model is loaded"""
        return self.model is not None
    
    def preprocess_audio(self, audio_path: str, max_duration: float = 15.0):
        """Preprocess reference audio for optimal performance"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Trim silence
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            # Limit duration for optimal performance
            max_samples = int(max_duration * self.sample_rate)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            
            return audio, sr
            
        except Exception as e:
            print(f"❌ Error preprocessing audio: {e}")
            return None, None
    
    def _preprocess_text_for_tensor_safety(self, text: str) -> str:
        """Preprocess text to avoid tensor dimension mismatches in IndexTTS2"""
        import re
        
        # Remove excessive punctuation that might cause issues
        text = re.sub(r'[.]{3,}', '...', text)  # Limit ellipsis
        text = re.sub(r'[!]{2,}', '!', text)    # Limit exclamation marks
        text = re.sub(r'[?]{2,}', '?', text)    # Limit question marks
        
        # Clean up excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove or replace problematic character sequences
        text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)  # Keep only safe characters
        
        # Ensure text doesn't end abruptly without punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        # Limit very long sentences that might cause tensor issues
        sentences = re.split(r'([.!?]+)', text)
        processed_sentences = []
        
        for i in range(0, len(sentences), 2):
            if i < len(sentences):
                sentence = sentences[i].strip()
                punctuation = sentences[i + 1] if i + 1 < len(sentences) else '.'
                
                # If sentence is too long, split it at commas or conjunctions
                if len(sentence) > 150:
                    # Split at commas, semicolons, or conjunctions
                    parts = re.split(r'(,|;|\s+and\s+|\s+but\s+|\s+or\s+)', sentence)
                    current_part = ""
                    
                    for j, part in enumerate(parts):
                        if part.strip() in [',', ';', 'and', 'but', 'or']:
                            current_part += part
                            if len(current_part) > 80:  # Split here
                                processed_sentences.append(current_part.strip() + '.')
                                current_part = ""
                        else:
                            if len(current_part + part) > 120 and current_part:
                                processed_sentences.append(current_part.strip() + '.')
                                current_part = part
                            else:
                                current_part += part
                    
                    if current_part.strip():
                        processed_sentences.append(current_part.strip() + punctuation)
                else:
                    processed_sentences.append(sentence + punctuation)
        
        return ' '.join(processed_sentences)
    
    def generate_speech(
        self,
        text: str,
        reference_audio: Optional[str] = None,
        emotion_mode: str = "audio_reference",
        emotion_audio: Optional[str] = None,
        emotion_vectors: Optional[Dict[str, float]] = None,
        emotion_description: str = "",
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        max_mel_tokens: int = 1500,
        seed: Optional[int] = None,
        use_random: bool = True,
        emo_alpha: float = 1.0
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        Generate speech using IndexTTS2 with emotion control
        
        Args:
            text: Text to synthesize
            reference_audio: Path to reference audio for voice cloning
            emotion_mode: Emotion control mode ('audio_reference', 'vector_control', 'text_description')
            emotion_audio: Path to emotion reference audio
            emotion_vectors: Dictionary of emotion intensities
            emotion_description: Natural language emotion description
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            max_mel_tokens: Maximum mel tokens to generate
            seed: Random seed for reproducibility
            use_random: Enable random sampling
            emo_alpha: Emotion blending alpha
        """
        if not self.is_model_loaded():
            return None, "❌ IndexTTS2 model not loaded. Please initialize first."
        
        if not reference_audio or not os.path.exists(reference_audio):
            return None, "❌ Reference audio is required for IndexTTS2"
        
        try:
            print(f"🎯 Generating speech with IndexTTS2...")
            print(f"   Text: {text[:50]}...")
            print(f"   Emotion mode: {emotion_mode}")
            
            # Preprocess long text to avoid tensor dimension mismatch
            # Split very long text into smaller chunks to prevent issues
            if len(text) > 500:  # If text is longer than 500 characters
                print(f"   ⚠️ Long text detected ({len(text)} chars), splitting into chunks...")
                # Split by sentences first, then by length if needed
                import re
                sentences = re.split(r'[.!?]+', text)
                processed_chunks = []
                current_chunk = ""
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    # If adding this sentence would make chunk too long, save current chunk
                    if len(current_chunk) + len(sentence) > 300 and current_chunk:
                        processed_chunks.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        current_chunk += (" " + sentence if current_chunk else sentence)
                
                # Add the last chunk
                if current_chunk.strip():
                    processed_chunks.append(current_chunk.strip())
                
                # If we have multiple chunks, process them separately and concatenate
                if len(processed_chunks) > 1:
                    print(f"   📝 Processing {len(processed_chunks)} text chunks...")
                    chunk_audios = []
                    
                    for i, chunk in enumerate(processed_chunks):
                        print(f"   🔄 Processing chunk {i+1}/{len(processed_chunks)}: {chunk[:30]}...")
                        
                        # Recursive call with shorter text
                        chunk_audio, chunk_message = self.generate_speech(
                            chunk, reference_audio, emotion_mode, emotion_audio,
                            emotion_vectors, emotion_description, temperature, top_p, top_k,
                            repetition_penalty, min(max_mel_tokens, 800), seed, use_random, emo_alpha
                        )
                        
                        if chunk_audio is None:
                            return None, f"❌ Error processing chunk {i+1}: {chunk_message}"
                        
                        chunk_audios.append(chunk_audio)
                    
                    # Concatenate all chunk audios with small pauses
                    print(f"   🎵 Combining {len(chunk_audios)} audio chunks...")
                    pause_samples = int(0.3 * self.sample_rate)  # 0.3 second pause
                    pause_audio = np.zeros(pause_samples)
                    
                    combined_audio = chunk_audios[0]
                    for chunk_audio in chunk_audios[1:]:
                        combined_audio = np.concatenate([combined_audio, pause_audio, chunk_audio])
                    
                    return combined_audio, "✅ Long text processed successfully in chunks"
                else:
                    # Single chunk, continue with normal processing
                    text = processed_chunks[0]
            
            # Set random seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
            
            # Prepare generation parameters
            generation_kwargs = {
                'do_sample': True,
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k,
                'repetition_penalty': repetition_penalty,
                'max_mel_tokens': max_mel_tokens
            }
            
            # Handle emotion control based on mode
            emo_audio_prompt = None
            emo_vector = None
            use_emo_text = False
            emo_text = None
            
            if emotion_mode == "audio_reference" and emotion_audio and os.path.exists(emotion_audio):
                emo_audio_prompt = emotion_audio
                    
            elif emotion_mode == "vector_control" and emotion_vectors:
                # Convert emotion vectors to the format expected by IndexTTS2
                emo_vector = []
                for emotion in ['happy', 'angry', 'sad', 'afraid', 'disgusted', 'melancholic', 'surprised', 'calm']:
                    emo_vector.append(emotion_vectors.get(emotion, 0.0))
                
            elif emotion_mode == "text_description" and emotion_description:
                use_emo_text = True
                emo_text = emotion_description
            
            # Preprocess text to avoid tensor dimension issues
            # Clean up text that might cause tensor mismatches
            original_text = text
            text = self._preprocess_text_for_tensor_safety(text)
            if text != original_text:
                print(f"🔧 Text preprocessed to avoid tensor issues")
            
            # Generate speech using the actual IndexTTS2 API with retry logic for tensor dimension issues
            # Use smaller max_text_tokens_per_segment to prevent tensor dimension issues
            max_text_tokens_per_segment = min(80, max_mel_tokens // 20) if len(text) > 200 else 120
            
            # Implement retry logic with progressively smaller parameters
            max_retries = 3
            retry_count = 0
            result = None
            
            while retry_count < max_retries and result is None:
                try:
                    # Adjust parameters based on retry count
                    if retry_count > 0:
                        print(f"   🔄 Tensor dimension mismatch (attempt {retry_count}/{max_retries})")
                        # Progressively reduce parameters to avoid tensor issues
                        max_text_tokens_per_segment = max(20, max_text_tokens_per_segment // 2)
                        generation_kwargs['max_mel_tokens'] = max(300, generation_kwargs['max_mel_tokens'] // 2)
                        print(f"   🔧 Retrying with max_text_tokens_per_segment={max_text_tokens_per_segment}, max_mel_tokens={generation_kwargs['max_mel_tokens']}")
                    
                    result = self.model.infer(
                        spk_audio_prompt=reference_audio,
                        text=text,
                        output_path=None,  # Return audio data instead of saving
                        emo_audio_prompt=emo_audio_prompt,
                        emo_alpha=emo_alpha,
                        emo_vector=emo_vector,
                        use_emo_text=use_emo_text,
                        emo_text=emo_text,
                        use_random=use_random,
                        max_text_tokens_per_segment=max_text_tokens_per_segment,
                        **generation_kwargs
                    )
                    
                except Exception as retry_error:
                    retry_error_msg = str(retry_error)
                    
                    # Check if this is a tensor dimension mismatch error
                    if ("Sizes of tensors must match" in retry_error_msg or 
                        "Expected size" in retry_error_msg or
                        "dimension" in retry_error_msg.lower()):
                        
                        retry_count += 1
                        if retry_count >= max_retries:
                            # Final attempt: force text chunking with very small segments
                            print(f"   🔄 Final attempt: forcing text chunking...")
                            try:
                                # Split text into very small chunks and process separately
                                words = text.split()
                                chunk_size = max(5, len(words) // 4)  # Very small chunks
                                text_chunks = []
                                
                                for i in range(0, len(words), chunk_size):
                                    chunk = " ".join(words[i:i + chunk_size])
                                    text_chunks.append(chunk)
                                
                                if len(text_chunks) > 1:
                                    print(f"   📝 Processing {len(text_chunks)} micro-chunks...")
                                    chunk_audios = []
                                    
                                    for j, chunk in enumerate(text_chunks):
                                        print(f"   🔄 Micro-chunk {j+1}/{len(text_chunks)}: {chunk[:20]}...")
                                        
                                        chunk_result = self.model.infer(
                                            spk_audio_prompt=reference_audio,
                                            text=chunk,
                                            output_path=None,
                                            emo_audio_prompt=emo_audio_prompt,
                                            emo_alpha=emo_alpha,
                                            emo_vector=emo_vector,
                                            use_emo_text=use_emo_text,
                                            emo_text=emo_text,
                                            use_random=use_random,
                                            max_text_tokens_per_segment=20,
                                            max_mel_tokens=300
                                        )
                                        
                                        if chunk_result is not None:
                                            if isinstance(chunk_result, tuple) and len(chunk_result) == 2:
                                                _, chunk_audio = chunk_result
                                                if isinstance(chunk_audio, torch.Tensor):
                                                    chunk_audio = chunk_audio.cpu().numpy()
                                                
                                                # Ensure chunk_audio is 1D
                                                if chunk_audio.ndim == 2:
                                                    if chunk_audio.shape[0] == 1:
                                                        chunk_audio = chunk_audio.flatten()
                                                    elif chunk_audio.shape[1] == 1:
                                                        chunk_audio = chunk_audio.flatten()
                                                    else:
                                                        # Take first channel if stereo
                                                        chunk_audio = chunk_audio[0] if chunk_audio.shape[0] < chunk_audio.shape[1] else chunk_audio[:, 0]
                                                
                                                chunk_audios.append(chunk_audio)
                                    
                                    if chunk_audios:
                                        try:
                                            # Ensure all chunks are 1D arrays before combining
                                            normalized_chunks = []
                                            for i, chunk_audio in enumerate(chunk_audios):
                                                if chunk_audio.ndim > 1:
                                                    chunk_audio = chunk_audio.flatten()
                                                
                                                # Ensure it's a valid audio array
                                                if len(chunk_audio) == 0:
                                                    print(f"   ⚠️ Skipping empty chunk {i+1}")
                                                    continue
                                                
                                                normalized_chunks.append(chunk_audio)
                                            
                                            if normalized_chunks:
                                                # Combine chunks with small pauses
                                                pause_samples = int(0.1 * self.sample_rate)
                                                pause_audio = np.zeros(pause_samples)
                                                
                                                combined_audio = normalized_chunks[0]
                                                for chunk_audio in normalized_chunks[1:]:
                                                    combined_audio = np.concatenate([combined_audio, pause_audio, chunk_audio])
                                                
                                                print(f"   ✅ Successfully combined {len(normalized_chunks)} micro-chunks")
                                                result = (self.sample_rate, combined_audio)
                                                break
                                            else:
                                                print(f"   ❌ No valid chunks to combine")
                                        
                                        except Exception as combine_error:
                                            print(f"   ❌ Error combining chunks: {combine_error}")
                                            # Continue to raise the original error
                                
                            except Exception as chunk_error:
                                print(f"   ❌ Chunking also failed: {chunk_error}")
                                raise retry_error  # Re-raise original error
                        else:
                            continue  # Try again with smaller parameters
                    else:
                        # Not a tensor dimension error, re-raise immediately
                        raise retry_error
            
            if result is None:
                return None, "❌ Failed to generate audio"
            
            # IndexTTS2 returns (sample_rate, audio_data) tuple
            if isinstance(result, tuple) and len(result) == 2:
                sample_rate, audio_data = result
                
                # Convert to numpy array if needed
                if isinstance(audio_data, torch.Tensor):
                    audio_data = audio_data.cpu().numpy()
                
                # Handle different audio data formats more robustly
                if audio_data.ndim == 2:
                    # If stereo or transposed, take first channel or transpose
                    if audio_data.shape[0] == 2:
                        audio_data = audio_data[0]  # Take first channel
                    elif audio_data.shape[1] == 1:
                        audio_data = audio_data.flatten()  # Flatten single channel
                    elif audio_data.shape[0] == 1:
                        audio_data = audio_data.flatten()  # Flatten single channel
                    else:
                        # Choose the dimension that makes more sense for audio
                        if audio_data.shape[0] < audio_data.shape[1]:
                            audio_data = audio_data[0]  # Take first row
                        else:
                            audio_data = audio_data[:, 0]  # Take first column
                elif audio_data.ndim > 2:
                    # Handle higher dimensional arrays by flattening
                    audio_data = audio_data.flatten()
                
                # Normalize audio to prevent clipping
                if len(audio_data) > 0:
                    max_val = np.max(np.abs(audio_data))
                    if max_val > 0:
                        audio_data = audio_data / max_val * 0.95
                
                print(f"✅ Generated {len(audio_data)} samples at {sample_rate}Hz")
                return audio_data, "✅ Speech generated successfully"
            else:
                return None, "❌ Unexpected audio format returned"
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            
            # Provide specific guidance for common tensor dimension errors
            if "Sizes of tensors must match" in error_msg:
                print("🔍 Tensor dimension mismatch detected - this is likely due to long text processing")
                print("💡 Try reducing max_mel_tokens or splitting your text into shorter segments")
                error_msg = f"Tensor dimension mismatch (likely due to long text): {error_msg}"
            elif "Expected size" in error_msg and "but got size" in error_msg:
                print("🔍 Tensor size mismatch detected")
                print("💡 This may be resolved by using shorter text segments")
                error_msg = f"Tensor size mismatch: {error_msg}"
            
            traceback.print_exc()
            return None, f"❌ Error generating speech: {error_msg}"
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if not self.is_model_loaded():
            return "❌ No model loaded"
        
        info = {
            'model_name': 'IndexTTS-2',
            'device': self.device,
            'sample_rate': self.sample_rate,
            'emotion_modes': list(self.emotion_modes.keys()),
            'supported_languages': ['English', 'Chinese'],
            'features': [
                'Zero-shot voice cloning',
                'Advanced emotion control',
                'Emotion-speaker disentanglement',
                'Duration control',
                'Multi-modal emotion input'
            ]
        }
        
        return info

# Model management functions
def init_indextts2():
    """Initialize IndexTTS2 model"""
    if not INDEXTTS2_AVAILABLE:
        return False, "❌ IndexTTS2 not available"
    
    try:
        handler = get_indextts2_handler()
        success, message = handler.initialize_model()
        return success, message
    except Exception as e:
        return False, f"❌ Error initializing IndexTTS2: {str(e)}"

def unload_indextts2():
    """Unload IndexTTS2 model"""
    try:
        handler = get_indextts2_handler()
        return handler.unload_model()
    except Exception as e:
        return f"⚠️ Error unloading IndexTTS2: {str(e)}"

def generate_indextts2_tts(
    text: str,
    reference_audio: Optional[str] = None,
    emotion_mode: str = "audio_reference",
    emotion_audio: Optional[str] = None,
    emotion_vectors: Optional[Dict[str, float]] = None,
    emotion_description: str = "",
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    max_mel_tokens: int = 1500,
    seed: Optional[int] = None,
    use_random: bool = True,
    emo_alpha: float = 1.0,
    effects_settings: Optional[Dict] = None,
    audio_format: str = "wav",
    skip_file_saving: bool = False
) -> Tuple[Optional[Union[str, Tuple]], str]:
    """
    Generate TTS using IndexTTS2 with comprehensive emotion control
    
    Returns:
        Tuple of (audio_data, info_message)
        audio_data can be either file path (str) or (sample_rate, audio_array) tuple
    """
    try:
        handler = get_indextts2_handler()
        
        if not handler.is_model_loaded():
            return None, "❌ IndexTTS2 model not loaded. Please load the model first."
        
        # Generate speech
        audio_array, message = handler.generate_speech(
            text=text,
            reference_audio=reference_audio,
            emotion_mode=emotion_mode,
            emotion_audio=emotion_audio,
            emotion_vectors=emotion_vectors,
            emotion_description=emotion_description,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_mel_tokens=max_mel_tokens,
            seed=seed,
            use_random=use_random,
            emo_alpha=emo_alpha
        )
        
        if audio_array is None:
            return None, message
        
        # Apply audio effects if specified
        if effects_settings:
            try:
                # Import audio effects processing (assuming it exists in the main app)
                from launch import apply_audio_effects
                audio_array = apply_audio_effects(audio_array, handler.sample_rate, effects_settings)
            except ImportError:
                print("⚠️ Audio effects not available")
        
        if skip_file_saving:
            # Return audio data directly as tuple
            return (handler.sample_rate, audio_array), message
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"indextts2_output_{timestamp}.{audio_format}"
        output_path = os.path.join("outputs", filename)
        
        # Ensure outputs directory exists
        os.makedirs("outputs", exist_ok=True)
        
        # Save audio file
        if audio_format.lower() == "wav":
            sf.write(output_path, audio_array, handler.sample_rate)
        else:
            # For other formats, use librosa
            sf.write(output_path, audio_array, handler.sample_rate, format=audio_format)
        
        return output_path, f"✅ Audio saved to {output_path}"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error in IndexTTS2 generation: {str(e)}"

def get_indextts2_status():
    """Get IndexTTS2 model status"""
    if not INDEXTTS2_AVAILABLE:
        return "❌ IndexTTS2 not available"
    
    handler = get_indextts2_handler()
    if handler.is_model_loaded():
        return "✅ IndexTTS2 model loaded and ready"
    else:
        return "⚠️ IndexTTS2 model not loaded"

# Emotion presets for easy use
EMOTION_PRESETS = {
    'neutral': {'happy': 0.0, 'angry': 0.0, 'sad': 0.0, 'afraid': 0.0, 'disgusted': 0.0, 'melancholic': 0.0, 'surprised': 0.0, 'calm': 1.0},
    'happy': {'happy': 1.0, 'angry': 0.0, 'sad': 0.0, 'afraid': 0.0, 'disgusted': 0.0, 'melancholic': 0.0, 'surprised': 0.2, 'calm': 0.0},
    'sad': {'happy': 0.0, 'angry': 0.0, 'sad': 1.0, 'afraid': 0.0, 'disgusted': 0.0, 'melancholic': 0.8, 'surprised': 0.0, 'calm': 0.0},
    'angry': {'happy': 0.0, 'angry': 1.0, 'sad': 0.0, 'afraid': 0.0, 'disgusted': 0.3, 'melancholic': 0.0, 'surprised': 0.0, 'calm': 0.0},
    'excited': {'happy': 0.8, 'angry': 0.0, 'sad': 0.0, 'afraid': 0.0, 'disgusted': 0.0, 'melancholic': 0.0, 'surprised': 0.6, 'calm': 0.0},
    'melancholic': {'happy': 0.0, 'angry': 0.0, 'sad': 0.6, 'afraid': 0.0, 'disgusted': 0.0, 'melancholic': 1.0, 'surprised': 0.0, 'calm': 0.2},
    'surprised': {'happy': 0.2, 'angry': 0.0, 'sad': 0.0, 'afraid': 0.3, 'disgusted': 0.0, 'melancholic': 0.0, 'surprised': 1.0, 'calm': 0.0},
    'afraid': {'happy': 0.0, 'angry': 0.0, 'sad': 0.3, 'afraid': 1.0, 'disgusted': 0.0, 'melancholic': 0.0, 'surprised': 0.4, 'calm': 0.0}
}