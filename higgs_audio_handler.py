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
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any
from datetime import datetime

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

class HiggsAudioHandler:
    """Handler for Higgs Audio TTS system"""
    
    def __init__(self):
        self.engine = None
        self.model_path = "bosonai/higgs-audio-v2-generation-3B-base"
        self.audio_tokenizer_path = "bosonai/higgs-audio-v2-tokenizer"
        self.sample_rate = 24000
        self.device = self._get_device()
        self.voice_presets = self._load_voice_presets()
        
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
            self.engine = HiggsAudioServeEngine(
                model_name_or_path=self.model_path,
                audio_tokenizer_name_or_path=self.audio_tokenizer_path,
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
                    self.engine = HiggsAudioServeEngine(
                        model_name_or_path=self.model_path,
                        audio_tokenizer_name_or_path=self.audio_tokenizer_path,
                        device="cpu",
                    )
                    print("✅ Higgs Audio engine initialized on CPU")
                    return True
                except Exception as cpu_e:
                    print(f"❌ CPU fallback also failed: {cpu_e}")
                    return False
            return False
    
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
        
        # Clean up whitespace
        lines = text.split("\n")
        text = "\n".join([" ".join(line.split()) for line in lines if line.strip()])
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
        ras_win_max_num_repeat: int = 2
    ) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
        """
        Generate speech using Higgs Audio
        
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
            # Prepare ChatML sample
            chatml_sample = self._prepare_chatml_sample(
                text, voice_preset, reference_audio, reference_text, system_prompt
            )
            
            print(f"🎤 Generating speech with Higgs Audio: {text[:50]}...")
            
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
                
                print(f"✅ Generated {len(audio_data)} audio samples")
                return (response.sampling_rate, audio_data), f"✅ Speech generated successfully"
            else:
                return None, "❌ No audio generated"
                
        except Exception as e:
            error_msg = f"❌ Error generating speech: {str(e)}"
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
    
    return handler.generate_speech(
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
        ras_win_max_num_repeat=ras_win_max_num_repeat
    )