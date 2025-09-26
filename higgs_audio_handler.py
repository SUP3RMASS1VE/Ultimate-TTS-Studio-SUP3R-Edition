"""
Higgs Audio TTS Handler for Ultimate TTS Studio
Provides integration with Higgs Audio Text-to-Speech system
"""

import os
import sys
import warnings
import numpy as np
import torch
import base64
import tempfile
import json
import re
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, List
from datetime import datetime
from contextlib import contextmanager

# Import for audio file saving
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    try:
        from scipy.io.wavfile import write
        SOUNDFILE_AVAILABLE = False
    except ImportError:
        SOUNDFILE_AVAILABLE = None

# Import for MP3 conversion
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')

# Add higgs_audio to path
current_dir = os.path.dirname(os.path.abspath(__file__))
higgs_audio_path = os.path.join(current_dir, 'higgs_audio')
if higgs_audio_path not in sys.path:
    sys.path.insert(0, higgs_audio_path)

try:
    from higgs_audio.serve.serve_engine import HiggsAudioServeEngine
    from higgs_audio.data_types import ChatMLSample, AudioContent, Message
    HIGGS_AUDIO_AVAILABLE = True
    print("✅ Higgs Audio loaded successfully")
except ImportError as e:
    HIGGS_AUDIO_AVAILABLE = False
    print(f"⚠️ Higgs Audio not available: {e}")

# Optional: local snapshot downloader (keeps Higgs isolated from global HF cache)
try:
    from huggingface_hub import snapshot_download
    HF_HUB_AVAILABLE = True
except Exception:
    HF_HUB_AVAILABLE = False

def split_text_for_higgs(text: str, max_length: int = 100) -> List[str]:
    """
    Split text into chunks suitable for Higgs Audio processing.
    Ensures no text is lost during chunking.
    
    Args:
        text: Input text to split
        max_length: Maximum length per chunk (in characters)
    
    Returns:
        List of text chunks
    """
    if len(text) <= max_length:
        return [text]
    
    # Simple sentence-based chunking that preserves all text
    # Split on sentence endings but keep the punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would exceed max_length, save current chunk and start new one
        if current_chunk and len(current_chunk + " " + sentence) > max_length:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Add the last chunk if it has content
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # If we still have chunks that are too long, split them by words
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_length:
            final_chunks.append(chunk)
        else:
            # Split long chunks by words while preserving all content
            words = chunk.split()
            temp_chunk = ""
            
            for word in words:
                # If adding this word would exceed max_length, save current chunk
                if temp_chunk and len(temp_chunk + " " + word) > max_length:
                    if temp_chunk.strip():
                        final_chunks.append(temp_chunk.strip())
                    temp_chunk = word
                else:
                    if temp_chunk:
                        temp_chunk += " " + word
                    else:
                        temp_chunk = word
            
            # Add the final word chunk
            if temp_chunk.strip():
                final_chunks.append(temp_chunk.strip())
    
    # Ensure we haven't lost any text
    original_text_clean = re.sub(r'\s+', ' ', text.strip())
    reconstructed_text = ' '.join(final_chunks)
    reconstructed_clean = re.sub(r'\s+', ' ', reconstructed_text.strip())
    
    if len(reconstructed_clean) < len(original_text_clean) * 0.95:
        print(f"⚠️ WARNING: Chunking may have lost text!")
        print(f"   Original length: {len(original_text_clean)}")
        print(f"   Reconstructed length: {len(reconstructed_clean)}")
        # Fallback: use simple character-based chunking
        return [text[i:i+max_length] for i in range(0, len(text), max_length)]
    
    return final_chunks

def save_higgs_audio_file(audio_data: np.ndarray, sample_rate: int, audio_format: str = "wav", output_folder: str = "outputs", filename_base: str = None) -> Tuple[str, str]:
    """
    Save Higgs Audio output to file
    
    Args:
        audio_data: Audio data array (int16 format)
        sample_rate: Sample rate
        audio_format: Output format ("wav" or "mp3")
        output_folder: Output directory
        filename_base: Base filename without extension
    
    Returns:
        Tuple of (filepath, filename)
    """
    if filename_base is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"higgs_audio_{timestamp}"
    
    # Ensure output folder exists
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)
    
    # Convert int16 audio back to float32 range [-1, 1] for saving
    if audio_data.dtype == np.int16:
        audio_float = audio_data.astype(np.float32) / 32767.0
    else:
        audio_float = audio_data
    
    audio_format = audio_format.lower()
    
    if audio_format == "wav":
        filename = f"{filename_base}.wav"
        filepath = output_folder / filename
        
        if SOUNDFILE_AVAILABLE:
            sf.write(str(filepath), audio_float, sample_rate)
        elif SOUNDFILE_AVAILABLE is False:
            # Use scipy fallback
            from scipy.io.wavfile import write
            write(str(filepath), sample_rate, audio_data)  # Use original int16 data
        else:
            raise RuntimeError("No audio writing library available (soundfile or scipy)")
    
    elif audio_format == "mp3":
        # Convert to MP3 using pydub
        if not PYDUB_AVAILABLE:
            # Fallback to WAV if pydub not available
            print("⚠️ Warning: pydub not available, saving as WAV instead")
            filename = f"{filename_base}.wav"
            filepath = output_folder / filename
            
            if SOUNDFILE_AVAILABLE:
                sf.write(str(filepath), audio_float, sample_rate)
            elif SOUNDFILE_AVAILABLE is False:
                from scipy.io.wavfile import write
                write(str(filepath), sample_rate, audio_data)
            else:
                raise RuntimeError("No audio writing library available")
        else:
            # Save as temporary WAV first, then convert to MP3
            import tempfile
            
            # Create temporary WAV file
            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_wav.close()
            
            try:
                # Save as high-quality WAV first
                if SOUNDFILE_AVAILABLE:
                    sf.write(temp_wav.name, audio_float, sample_rate)
                elif SOUNDFILE_AVAILABLE is False:
                    from scipy.io.wavfile import write
                    write(temp_wav.name, sample_rate, audio_data)
                else:
                    raise RuntimeError("No audio writing library available")
                
                # Convert WAV to MP3 with high quality settings
                audio_segment = AudioSegment.from_wav(temp_wav.name)
                filename = f"{filename_base}.mp3"
                filepath = output_folder / filename
                
                # Export with high quality settings
                audio_segment.export(
                    str(filepath),
                    format="mp3",
                    bitrate="320k",  # High quality
                    parameters=["-q:a", "0"]  # Highest quality
                )
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_wav.name)
                except:
                    pass
    
    else:
        # Default to WAV
        filename = f"{filename_base}.wav"
        filepath = output_folder / filename
        
        if SOUNDFILE_AVAILABLE:
            sf.write(str(filepath), audio_float, sample_rate)
        elif SOUNDFILE_AVAILABLE is False:
            from scipy.io.wavfile import write
            write(str(filepath), sample_rate, audio_data)
        else:
            raise RuntimeError("No audio writing library available")
    
    return str(filepath), filename

class HiggsAudioHandler:
    """Handler for Higgs Audio TTS system"""
    
    def __init__(self):
        self.engine = None
        self.model_path = "bosonai/higgs-audio-v2-generation-3B-base"
        self.audio_tokenizer_path = "bosonai/higgs-audio-v2-tokenizer"
        self.sample_rate = 24000
        self.device = self._get_device()
        self.voice_presets = self._load_voice_presets()
        # Create an isolated cache for Higgs only
        self.higgs_cache_dir = os.path.join(current_dir, 'checkpoints', 'higgs_audio', 'cache')
        try:
            os.makedirs(self.higgs_cache_dir, exist_ok=True)
        except Exception:
            pass
        
        # Default system prompt
        self.default_system_prompt = (
            "Generate audio following instruction.\n\n"
            "<|scene_desc_start|>\n"
            "Audio is recorded from a quiet room.\n"
            "<|scene_desc_end|>"
        )
        
        # Default stop strings
        self.default_stop_strings = ["<|end_of_text|>", "<|eot_id|>"]
    
    def _get_device(self) -> str:
        """Get the best available device"""
        if torch.cuda.is_available():
            try:
                # Test if we can actually create a tensor on CUDA
                test_tensor = torch.tensor([1.0], device="cuda")
                del test_tensor
                return "cuda"
            except Exception:
                return "cpu"
        return "cpu"
    
    def _load_voice_presets(self) -> Dict[str, str]:
        """Load voice presets from config file"""
        try:
            config_path = os.path.join(current_dir, "higgs_audio", "voice_examples", "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    voice_dict = json.load(f)
                voice_presets = {k: v["transcript"] for k, v in voice_dict.items()}
                voice_presets["EMPTY"] = "No reference voice"
                return voice_presets
            else:
                return {"EMPTY": "No reference voice"}
        except Exception as e:
            print(f"Warning: Could not load voice presets: {e}")
            return {"EMPTY": "No reference voice"}
    
    def initialize_engine(self) -> bool:
        """Initialize the Higgs Audio engine"""
        if not HIGGS_AUDIO_AVAILABLE:
            return False
            
        try:
            print(f"🎤 Initializing Higgs Audio engine on {self.device}...")
            # Resolve local snapshot paths (download into an isolated cache if needed)
            local_model_path, local_tokenizer_path = self._ensure_local_higgs_snapshots()

            # Ensure any additional dependencies (e.g., HuBERT/WavLM) download into the isolated cache
            with self._temporary_hf_env_for_higgs():
                self.engine = HiggsAudioServeEngine(
                    model_name_or_path=local_model_path or self.model_path,
                    audio_tokenizer_name_or_path=local_tokenizer_path or self.audio_tokenizer_path,
                    device=self.device,
                )
            print(f"✅ Higgs Audio engine initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize Higgs Audio engine: {e}")
            # Try fallback to CPU if CUDA failed
            if self.device == "cuda":
                try:
                    print("🔄 Attempting CPU fallback...")
                    self.device = "cpu"
                    # Re-resolve local paths (ensures availability for CPU path too)
                    local_model_path, local_tokenizer_path = self._ensure_local_higgs_snapshots()
                    with self._temporary_hf_env_for_higgs():
                        self.engine = HiggsAudioServeEngine(
                            model_name_or_path=local_model_path or self.model_path,
                            audio_tokenizer_name_or_path=local_tokenizer_path or self.audio_tokenizer_path,
                            device="cpu",
                        )
                    print("✅ Higgs Audio engine initialized on CPU")
                    return True
                except Exception as cpu_e:
                    print(f"❌ CPU fallback also failed: {cpu_e}")
                    return False
            return False

    @contextmanager
    def _temporary_hf_env_for_higgs(self):
        """Temporarily set HF caches to the Higgs cache and disable offline mode."""
        original_env = {
            'TRANSFORMERS_OFFLINE': os.environ.get('TRANSFORMERS_OFFLINE'),
            'HF_HUB_OFFLINE': os.environ.get('HF_HUB_OFFLINE'),
            'HF_HOME': os.environ.get('HF_HOME'),
            'TRANSFORMERS_CACHE': os.environ.get('TRANSFORMERS_CACHE'),
            'HF_HUB_CACHE': os.environ.get('HF_HUB_CACHE'),
            'HUGGINGFACE_HUB_CACHE': os.environ.get('HUGGINGFACE_HUB_CACHE'),
        }
        try:
            os.environ.pop('TRANSFORMERS_OFFLINE', None)
            os.environ.pop('HF_HUB_OFFLINE', None)
            os.environ['HF_HOME'] = self.higgs_cache_dir
            os.environ['TRANSFORMERS_CACHE'] = self.higgs_cache_dir
            os.environ['HF_HUB_CACHE'] = self.higgs_cache_dir
            os.environ['HUGGINGFACE_HUB_CACHE'] = self.higgs_cache_dir
            yield
        finally:
            for k, v in original_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def _ensure_local_higgs_snapshots(self) -> Tuple[Optional[str], Optional[str]]:
        """Ensure Higgs model and tokenizer are available locally under the isolated cache.

        Returns a tuple of (model_path, tokenizer_path). If downloads are not possible, returns (None, None)
        and the caller should fall back to remote repo IDs. This method temporarily clears offline flags
        and points HF caches to the isolated directory, then restores the original environment.
        """
        # If downloader not available, attempt best-effort by letting downstream loaders fetch into our cache
        if not HF_HUB_AVAILABLE:
            print("⚠️ huggingface_hub not available; will attempt on-the-fly download during load")
            # Temporarily allow online and set caches so that any downstream download lands in our folder
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
                os.environ['HF_HOME'] = self.higgs_cache_dir
                os.environ['TRANSFORMERS_CACHE'] = self.higgs_cache_dir
                os.environ['HF_HUB_CACHE'] = self.higgs_cache_dir
                os.environ['HUGGINGFACE_HUB_CACHE'] = self.higgs_cache_dir
            except Exception:
                pass
            finally:
                # Do not restore yet; the caller will load immediately and then we restore here
                # Restore right away because we pass back None to indicate remote IDs will be used.
                for k, v in original_env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            return None, None

        # With huggingface_hub available, use snapshot_download for both repos
        original_env = {
            'TRANSFORMERS_OFFLINE': os.environ.get('TRANSFORMERS_OFFLINE'),
            'HF_HUB_OFFLINE': os.environ.get('HF_HUB_OFFLINE'),
            'HF_HOME': os.environ.get('HF_HOME'),
            'TRANSFORMERS_CACHE': os.environ.get('TRANSFORMERS_CACHE'),
            'HF_HUB_CACHE': os.environ.get('HF_HUB_CACHE'),
            'HUGGINGFACE_HUB_CACHE': os.environ.get('HUGGINGFACE_HUB_CACHE')
        }
        try:
            # Ensure online and route caches to the isolated directory for the download only
            os.environ.pop('TRANSFORMERS_OFFLINE', None)
            os.environ.pop('HF_HUB_OFFLINE', None)
            os.environ['HF_HOME'] = self.higgs_cache_dir
            os.environ['TRANSFORMERS_CACHE'] = self.higgs_cache_dir
            os.environ['HF_HUB_CACHE'] = self.higgs_cache_dir
            os.environ['HUGGINGFACE_HUB_CACHE'] = self.higgs_cache_dir

            print("🔽 Ensuring local Higgs snapshots (isolated cache)...")
            model_local = snapshot_download(
                repo_id=self.model_path,
                cache_dir=self.higgs_cache_dir,
                local_dir_use_symlinks=False
            )
            tokenizer_local = snapshot_download(
                repo_id=self.audio_tokenizer_path,
                cache_dir=self.higgs_cache_dir,
                local_dir_use_symlinks=False
            )
            # Also ensure required semantic model is cached locally
            try:
                print("   🔽 Ensuring HuBERT general model snapshot...")
                snapshot_download(
                    repo_id="ZhenYe234/hubert_base_general_audio",
                    cache_dir=self.higgs_cache_dir,
                    local_dir_use_symlinks=False
                )
            except Exception as he:
                print(f"   ⚠️ HuBERT general model snapshot issue: {he}")
            print("   ✅ Higgs snapshots ready")
            return model_local, tokenizer_local
        except Exception as e:
            print(f"   ❌ Higgs snapshot check failed: {e}")
            return None, None
        finally:
            # Restore original env to avoid impacting other handlers (e.g., IndexTTS2)
            for k, v in original_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    
    def _encode_audio_file(self, file_path: str) -> str:
        """Encode an audio file to base64"""
        try:
            with open(file_path, "rb") as audio_file:
                return base64.b64encode(audio_file.read()).decode("utf-8")
        except Exception as e:
            print(f"Error encoding audio file {file_path}: {e}")
            return ""
    
    def _get_voice_preset(self, voice_preset: str) -> Tuple[Optional[str], str]:
        """Get the voice path and text for a given voice preset"""
        voice_path = os.path.join(current_dir, "higgs_audio", "voice_examples", f"{voice_preset}.wav")
        if os.path.exists(voice_path):
            text = self.voice_presets.get(voice_preset, "No transcript available")
            return voice_path, text
        return None, "Voice preset not found"
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for TTS processing"""
        # Convert Chinese punctuation to English
        chinese_to_english_punct = {
            "，": ", ", "。": ".", "：": ":", "；": ";", "？": "?", "！": "!",
            "（": "(", "）": ")", "【": "[", "】": "]", "《": "<", "》": ">",
            """: '"', """: '"', "'": "'", "'": "'", "、": ",", "—": "-",
            "…": "...", "·": ".", "「": '"', "」": '"', "『": '"', "』": '"',
        }
        
        for zh_punct, en_punct in chinese_to_english_punct.items():
            text = text.replace(zh_punct, en_punct)
        
        # Other normalizations
        text = text.replace("(", " ").replace(")", " ")
        text = text.replace("°F", " degrees Fahrenheit")
        text = text.replace("°C", " degrees Celsius")
        
        # Handle special tags
        tag_replacements = [
            ("[laugh]", "<SE>[Laughter]</SE>"),
            ("[humming start]", "<SE>[Humming]</SE>"),
            ("[humming end]", "<SE_e>[Humming]</SE_e>"),
            ("[music start]", "<SE_s>[Music]</SE_s>"),
            ("[music end]", "<SE_e>[Music]</SE_e>"),
            ("[music]", "<SE>[Music]</SE>"),
            ("[sing start]", "<SE_s>[Singing]</SE_s>"),
            ("[sing end]", "<SE_e>[Singing]</SE_e>"),
            ("[applause]", "<SE>[Applause]</SE>"),
            ("[cheering]", "<SE>[Cheering]</SE>"),
            ("[cough]", "<SE>[Cough]</SE>"),
        ]
        
        for tag, replacement in tag_replacements:
            text = text.replace(tag, replacement)
        
        # Clean up whitespace (more conservative approach)
        lines = text.split("\n")
        # Only remove completely empty lines, preserve lines with some content
        filtered_lines = [line for line in lines if line.strip()]
        if filtered_lines:
            text = "\n".join([" ".join(line.split()) for line in filtered_lines])
        text = text.strip()
        
        # Add ending punctuation if needed
        if not any([text.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]]):
            text += "."
        
        return text
    
    def _prepare_chatml_sample(
        self,
        text: str,
        voice_preset: str = "EMPTY",
        reference_audio: Optional[str] = None,
        reference_text: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> ChatMLSample:
        """Prepare a ChatML sample for the engine"""
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        elif system_prompt != "":  # Use default if not explicitly set to empty
            messages.append(Message(role="system", content=self.default_system_prompt))
        
        # Add reference audio if provided
        audio_base64 = None
        ref_text = ""
        
        if reference_audio and os.path.exists(reference_audio):
            # Custom reference audio
            audio_base64 = self._encode_audio_file(reference_audio)
            ref_text = reference_text or ""
        elif voice_preset != "EMPTY":
            # Voice preset
            voice_path, ref_text = self._get_voice_preset(voice_preset)
            if voice_path:
                audio_base64 = self._encode_audio_file(voice_path)
        
        # Only add reference audio if we have it
        if audio_base64:
            # Add user message with reference text
            messages.append(Message(role="user", content=ref_text))
            
            # Add assistant message with audio content
            audio_content = AudioContent(raw_audio=audio_base64, audio_url="")
            messages.append(Message(role="assistant", content=[audio_content]))
        
        # Add the main user message
        normalized_text = self._normalize_text(text)
        # Debug: check if normalization is changing the text significantly
        if len(normalized_text) < len(text) * 0.8:  # If text shrunk by more than 20%
            print(f"   ⚠️ Text normalization significantly changed length: {len(text)} -> {len(normalized_text)}")
            print(f"   Original: '{text}'")
            print(f"   Normalized: '{normalized_text}'")
        messages.append(Message(role="user", content=normalized_text))
        
        return ChatMLSample(messages=messages)
    
    def generate_speech(
        self,
        text: str,
        voice_preset: str = "EMPTY",
        reference_audio: Optional[str] = None,
        reference_text: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
        max_tokens: int = 1024,
        ras_win_len: int = 7,
        ras_win_max_num_repeat: int = 2,
        chunk_length: int = 100
    ) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
        """
        Generate speech using Higgs Audio with automatic text chunking for long texts
        
        Args:
            chunk_length: Maximum character length per chunk for processing long texts
        
        Returns:
            Tuple of (audio_data, info_message)
            audio_data is (sample_rate, audio_array) or None if failed
        """
        if not HIGGS_AUDIO_AVAILABLE:
            return None, "❌ Higgs Audio not available"
        
        # Initialize engine if needed
        if self.engine is None:
            if not self.initialize_engine():
                return None, "❌ Failed to initialize Higgs Audio engine"
        
        try:
            # Split text into chunks if it's long
            text_chunks = split_text_for_higgs(text, max_length=chunk_length)
            print(f"🔍 Original text length: {len(text)} chars")
            print(f"🔍 Split into {len(text_chunks)} chunks with max length {chunk_length}")
            total_chunk_chars = sum(len(chunk) for chunk in text_chunks)
            print(f"🔍 Total characters in chunks: {total_chunk_chars} (ratio: {total_chunk_chars/len(text):.2f})")
            
            # If we have multiple chunks, process them separately and concatenate
            if len(text_chunks) > 1:
                print(f"🎤 Higgs Audio: Processing {len(text_chunks)} text chunks for long text")
                chunk_audios = []
                
                for i, chunk in enumerate(text_chunks):
                    print(f"🎤 Processing chunk {i+1}/{len(text_chunks)} ({len(chunk)} chars): {chunk[:50]}...")
                    print(f"   Full chunk text: '{chunk}'")  # Debug: show full chunk
                    
                    # Generate audio for this chunk with increased token limit for small chunks
                    chunk_max_tokens = min(max_tokens, 2048)  # Increase token limit for chunks
                    chunk_result, chunk_message = self._generate_single_chunk(
                        chunk, voice_preset, reference_audio, reference_text, 
                        system_prompt, temperature, top_p, top_k, chunk_max_tokens,
                        ras_win_len, ras_win_max_num_repeat
                    )
                    
                    if chunk_result is None:
                        print(f"   ❌ Chunk {i+1} failed: {chunk_message}")
                        return None, f"❌ Error processing chunk {i+1}: {chunk_message}"
                    
                    print(f"   ✅ Chunk {i+1} processed successfully, audio length: {len(chunk_result[1])} samples")
                    # Extract the audio array (second element of the tuple)
                    chunk_audios.append(chunk_result[1])
                
                # Concatenate all chunk audios with small pauses
                print(f"🎵 Combining {len(chunk_audios)} audio chunks...")
                pause_samples = int(0.3 * self.sample_rate)  # 0.3 second pause
                pause_audio = np.zeros(pause_samples, dtype=np.int16)
                
                combined_audio = chunk_audios[0]
                for chunk_audio in chunk_audios[1:]:
                    combined_audio = np.concatenate([combined_audio, pause_audio, chunk_audio])
                
                print(f"✅ Generated {len(combined_audio)} total audio samples from {len(text_chunks)} chunks")
                return (self.sample_rate, combined_audio), f"✅ Long text processed successfully in {len(text_chunks)} chunks"
            
            else:
                # Single chunk, process normally
                print(f"🎤 Generating speech with Higgs Audio: {text[:50]}...")
                return self._generate_single_chunk(
                    text, voice_preset, reference_audio, reference_text, 
                    system_prompt, temperature, top_p, top_k, max_tokens,
                    ras_win_len, ras_win_max_num_repeat
                )
                
        except Exception as e:
            error_msg = f"❌ Error generating speech: {str(e)}"
            print(error_msg)
            return None, error_msg
    
    def _generate_single_chunk(
        self,
        text: str,
        voice_preset: str = "EMPTY",
        reference_audio: Optional[str] = None,
        reference_text: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
        max_tokens: int = 1024,
        ras_win_len: int = 7,
        ras_win_max_num_repeat: int = 2
    ) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
        """
        Generate speech for a single text chunk using Higgs Audio
        
        Returns:
            Tuple of (audio_data, info_message)
            audio_data is (sample_rate, audio_array) or None if failed
        """
        try:
            # Prepare ChatML sample
            chatml_sample = self._prepare_chatml_sample(
                text, voice_preset, reference_audio, reference_text, system_prompt
            )
            
            # Generate using the engine
            response = self.engine.generate(
                chat_ml_sample=chatml_sample,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k if top_k > 0 else None,
                top_p=top_p,
                stop_strings=self.default_stop_strings,
                ras_win_len=ras_win_len if ras_win_len > 0 else None,
                ras_win_max_num_repeat=max(ras_win_len, ras_win_max_num_repeat),
            )
            
            if response.audio is not None:
                # Convert to int16 for compatibility
                audio_data = (response.audio * 32767).astype(np.int16)
                
                # Check if audio is not silent
                if np.all(audio_data == 0):
                    return None, "⚠️ Generated audio is silent"
                
                return (response.sampling_rate, audio_data), f"✅ Speech generated successfully"
            else:
                return None, "❌ No audio generated"
                
        except Exception as e:
            error_msg = f"❌ Error generating speech chunk: {str(e)}"
            print(error_msg)
            return None, error_msg
    
    def get_available_voice_presets(self) -> list:
        """Get list of available voice presets"""
        if not HIGGS_AUDIO_AVAILABLE:
            return ["EMPTY"]
        return list(self.voice_presets.keys())
    
    def is_available(self) -> bool:
        """Check if Higgs Audio is available"""
        return HIGGS_AUDIO_AVAILABLE
    
    def get_status(self) -> str:
        """Get current status of Higgs Audio"""
        if not HIGGS_AUDIO_AVAILABLE:
            return "❌ Not Available - Please install Higgs Audio dependencies"
        elif self.engine is None:
            return "⭕ Available - Not Initialized"
        else:
            return f"✅ Ready - Device: {self.device}"

# Global handler instance
_higgs_audio_handler = None

def get_higgs_audio_handler() -> HiggsAudioHandler:
    """Get the global Higgs Audio handler instance"""
    global _higgs_audio_handler
    if _higgs_audio_handler is None:
        _higgs_audio_handler = HiggsAudioHandler()
    return _higgs_audio_handler

def generate_higgs_audio_tts(
    text: str,
    reference_audio: str = "",
    reference_text: str = "",
    voice_preset: str = "EMPTY",
    system_prompt: str = "",
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 50,
    max_tokens: int = 1024,
    ras_win_len: int = 7,
    ras_win_max_num_repeat: int = 2,
    chunk_length: int = 100,
    effects_settings: Optional[Dict[str, Any]] = None,
    audio_format: str = "wav",
    skip_file_saving: bool = False
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    """
    Generate TTS using Higgs Audio - Compatible with TTS Studio interface
    
    Args:
        text: Text to synthesize
        reference_audio: Path to reference audio file
        reference_text: Transcript of reference audio
        voice_preset: Voice preset name
        system_prompt: System prompt for generation
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        max_tokens: Maximum tokens to generate
        ras_win_len: RAS window length
        ras_win_max_num_repeat: RAS max repetitions
        chunk_length: Maximum character length per chunk for processing long texts
        effects_settings: Audio effects (not used by Higgs Audio)
        audio_format: Output format (not used by Higgs Audio)
        skip_file_saving: Whether to skip saving file
    
    Returns:
        Tuple of (audio_data, info_message)
    """
    handler = get_higgs_audio_handler()
    
    # Use reference_audio if provided, otherwise use voice_preset
    ref_audio = reference_audio if reference_audio and reference_audio.strip() else None
    preset = voice_preset if not ref_audio else "EMPTY"
    
    # Generate speech
    result = handler.generate_speech(
        text=text,
        voice_preset=preset,
        reference_audio=ref_audio,
        reference_text=reference_text if reference_text.strip() else None,
        system_prompt=system_prompt if system_prompt.strip() else None,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        ras_win_len=ras_win_len,
        ras_win_max_num_repeat=ras_win_max_num_repeat,
        chunk_length=chunk_length
    )
    
    # Unpack result
    audio_data, status_message = result
    
    if audio_data is None:
        return None, status_message
    
    # Save to file if not skipping
    if not skip_file_saving:
        try:
            sample_rate, audio_array = audio_data
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = f"higgs_audio_{timestamp}"
            
            # Save audio file
            filepath, filename = save_higgs_audio_file(
                audio_array, sample_rate, audio_format, "outputs", filename_base
            )
            
            # Calculate duration
            duration = len(audio_array) / sample_rate
            
            # Update status message with file info
            enhanced_status = f"{status_message}\n"
            enhanced_status += f"📁 Saved as: {filename}\n"
            enhanced_status += f"⏱️ Duration: {duration:.2f}s\n"
            enhanced_status += f"📊 Sample Rate: {sample_rate}Hz"
            
            return audio_data, enhanced_status
            
        except Exception as save_error:
            print(f"⚠️ Warning: Could not save audio file: {save_error}")
            return audio_data, f"{status_message} (file saving failed)"
    
    return result