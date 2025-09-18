import os
import shutil
import sys
import threading
import time
import requests
import huggingface_hub
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import yaml
import warnings
import torch
import io
import random
import json

# Custom stderr filter to hide specific CUDA errors
class FilteredStderr:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.blacklist = [
            "LINK : fatal error LNK1181: cannot open input file 'aio.lib'",
            "LINK : fatal error LNK1181: cannot open input file 'cufile.lib'",
            "NOTE: Redirects are currently not supported in Windows or MacOs",
            "test.c"
        ]
    
    def write(self, message):
        if not any(error in message for error in self.blacklist):
            self.original_stderr.write(message)
    
    def flush(self):
        self.original_stderr.flush()
        
    def isatty(self):
        return self.original_stderr.isatty()
        
    def fileno(self):
        return self.original_stderr.fileno()

# Install our custom filter
sys.stderr = FilteredStderr(sys.stderr)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)  # Added to suppress FutureWarnings
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning,ignore::DeprecationWarning,ignore::FutureWarning'

# Monkey patch torch._pytree to silence FutureWarning messages
import torch.utils._pytree
if hasattr(torch.utils._pytree, "_register_pytree_node"):
    original_register = torch.utils._pytree._register_pytree_node
    def patched_register(*args, **kwargs):
        # Suppress the warning by using the new API if available
        if hasattr(torch.utils._pytree, "register_pytree_node"):
            return torch.utils._pytree.register_pytree_node(*args, **kwargs)
        else:
            return original_register(*args, **kwargs)
    torch.utils._pytree._register_pytree_node = patched_register

# Suppress CUDA compilation warnings by redirecting stderr temporarily during model loading
import contextlib

@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr"""
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

# Add more specific CUDA warning suppression
if torch.cuda.is_available():
    # Suppress CUDA initialization messages
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # More controlled CUDA launches
    os.environ['TORCH_CUDA_ARCH_LIST'] = 'All'  # Avoid architecture-specific warnings
    
    # Suppress NVRTC warnings
    if hasattr(torch.cuda, 'nvrtc'):
        torch._C._jit_set_nvrtc_options(["--display-error-number", "--suppress-warnings"])

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

# Add command line argument parsing
import argparse
parser = argparse.ArgumentParser(description="IndexTTS WebUI")
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="checkpoints", help="Model checkpoints directory")
cmd_args = parser.parse_args()

def download_model_files():
    """Download model files from Hugging Face on first run"""
    repo_id = "IndexTeam/IndexTTS-2"
    os.makedirs(cmd_args.model_dir, exist_ok=True)
    
    # Check if config file exists, download if not
    config_path = os.path.join(cmd_args.model_dir, "config.yaml")
    if not os.path.exists(config_path):
        print("Downloading config.yaml...")
        try:
            hf_hub_download(repo_id=repo_id, filename="config.yaml", local_dir=cmd_args.model_dir, local_dir_use_symlinks=False)
        except Exception as e:
            print(f"Error downloading config.yaml: {e}")
            return False
    
    # Load config to get model filenames
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return False
    
    # Get model filenames from config - Updated for IndexTTS-2
    model_files = [
        "bpe.model",
        "gpt.pth",
        "s2mel.pth",
        "feat1.pt",
        "feat2.pt",
        "wav2vec2bert_stats.pt"
    ]
    
    # Download each model file
    for file in model_files:
        file_path = os.path.join(cmd_args.model_dir, file)
        if not os.path.exists(file_path):
            print(f"Downloading {file}...")
            try:
                hf_hub_download(repo_id=repo_id, filename=file, local_dir=cmd_args.model_dir, local_dir_use_symlinks=False)
                print(f"Successfully downloaded {file}")
            except Exception as e:
                print(f"Error downloading {file}: {e}")
                return False
    
    # Download QwenEmo model directory
    qwen_dir = os.path.join(cmd_args.model_dir, "qwen0.6bemo4-merge")
    if not os.path.exists(qwen_dir):
        print("Downloading QwenEmo model...")
        try:
            # Download all files in the qwen0.6bemo4-merge directory
            qwen_files = [
                "qwen0.6bemo4-merge/config.json",
                "qwen0.6bemo4-merge/generation_config.json", 
                "qwen0.6bemo4-merge/model.safetensors",
                "qwen0.6bemo4-merge/tokenizer.json",
                "qwen0.6bemo4-merge/tokenizer_config.json",
                "qwen0.6bemo4-merge/vocab.json"
            ]
            for file in qwen_files:
                try:
                    hf_hub_download(repo_id=repo_id, filename=file, local_dir=cmd_args.model_dir, local_dir_use_symlinks=False)
                except Exception as e:
                    print(f"Warning: Could not download {file}: {e}")
        except Exception as e:
            print(f"Warning: Error downloading QwenEmo model: {e}")
    
    print("All model files downloaded successfully!")
    return True

# Call the download function before importing the TTS model - Updated for IndexTTS-2
required_files = ["config.yaml", "bpe.model", "gpt.pth", "s2mel.pth", "feat1.pt", "feat2.pt", "wav2vec2bert_stats.pt"]
if not all(os.path.exists(os.path.join(cmd_args.model_dir, f)) for f in required_files):
    print("First run detected. Downloading model files...")
    if not download_model_files():
        print("Error downloading model files. Please check your internet connection and try again.")
        sys.exit(1)

import gradio as gr

try:
    # Prefer local bundled package path
    import importlib, sys as _sys
    try:
        _indextts_pkg = importlib.import_module('indextts2.indextts')
        _sys.modules['indextts'] = _indextts_pkg
    except Exception:
        pass
    from indextts2.indextts.infer_v2 import IndexTTS2
    from tools.i18n.i18n import I18nAuto
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please install the required dependencies manually:")
    print("pip install -r requirements.txt")
    sys.exit(1)

i18n = I18nAuto(language="en")  # Changed to English
MODE = 'local'

print("Loading IndexTTS-2 model...")
# More extensive suppression during model loading
try:
    # Completely silence all output during model loading
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    
    # Try loading the model - Updated to use IndexTTS2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = IndexTTS2(
        model_dir=cmd_args.model_dir, 
        cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
        device=device,
        use_fp16=True if device == "cuda" else False,
        use_cuda_kernel=False,  # Disable CUDA kernel to avoid compilation issues
        use_deepspeed=False
    )
except Exception as e:
    # Restore stderr and stdout before printing errors
    sys.stdout, sys.stderr = original_stdout, original_stderr
    print(f"Warning: Error during model initialization: {e}")
    print("Falling back to CPU mode")
    try:
        tts = IndexTTS2(
            model_dir=cmd_args.model_dir, 
            cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
            device="cpu",
            use_fp16=False,
            use_cuda_kernel=False,
            use_deepspeed=False
        )
    except Exception as e:
        print(f"Fatal error loading model: {e}")
        sys.exit(1)
finally:
    # Restore stdout and stderr
    if 'original_stdout' in locals() and 'original_stderr' in locals():
        sys.stdout, sys.stderr = original_stdout, original_stderr

os.makedirs("outputs/tasks", exist_ok=True)
os.makedirs("prompts", exist_ok=True)

def infer(voice, text, output_path=None, seed=None, emo_audio=None, emo_alpha=1.0, emo_vector=None, use_emo_text=False, emo_text=None, use_random=False):
    if not output_path:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
    
    # Set seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    # Use IndexTTS2 inference with new parameters
    tts.infer(
        spk_audio_prompt=voice, 
        text=text, 
        output_path=output_path,
        emo_audio_prompt=emo_audio,
        emo_alpha=emo_alpha,
        emo_vector=emo_vector,
        use_emo_text=use_emo_text,
        emo_text=emo_text,
        use_random=use_random
    )
    return output_path

def gen_single(prompt, text, seed=None, emo_audio=None, emo_alpha=1.0, emotion_mode="none", emo_text="", 
               happy=0.0, angry=0.0, sad=0.0, afraid=0.0, disgusted=0.0, melancholic=0.0, surprised=0.0, calm=0.0, use_random=False):
    # Use random seed if seed is 0 or None
    if seed is None or seed == 0:
        seed = None
    else:
        seed = int(seed)  # Ensure it's an integer
    
    # Handle emotion modes
    emo_vector = None
    use_emo_text = False
    emo_text_param = None
    final_emo_alpha = emo_alpha
    
    if emotion_mode == "audio" and emo_audio:
        # Use emotion audio reference
        pass  # emo_audio is already set, emo_alpha is already set
    elif emotion_mode == "vector":
        # Use emotion vector
        emo_vector = [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
        emo_audio = None  # Don't use audio when using vector
        final_emo_alpha = 1.0  # Vector mode handles alpha internally
    elif emotion_mode == "text":
        # Use text-based emotion
        use_emo_text = True
        emo_text_param = emo_text if emo_text.strip() else None
        emo_audio = None  # Don't use audio when using text
        final_emo_alpha = 0.6  # Recommended for text mode
    else:
        # No emotion control
        emo_audio = None
        final_emo_alpha = 1.0
    
    output_path = infer(
        voice=prompt, 
        text=text, 
        seed=seed,
        emo_audio=emo_audio,
        emo_alpha=final_emo_alpha,
        emo_vector=emo_vector,
        use_emo_text=use_emo_text,
        emo_text=emo_text_param,
        use_random=use_random
    )
    return gr.update(value=output_path, visible=True)

def update_prompt_audio():
    # Always keep button interactive
    return gr.update(interactive=True, variant="primary")

# Custom CSS for better UI - Dark Theme
css = """
:root {
    --primary-color: #3b82f6;
    --secondary-color: #2563eb;
    --accent-color: #60a5fa;
    --background-color: #121212;
    --surface-color: #1e1e1e;
    --text-color: #e2e8f0;
    --secondary-text-color: #94a3b8;
    --border-color: #2d3748;
    --border-radius: 8px;
    --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -1px rgba(0, 0, 0, 0.1);
    --content-width: 1100px;
}

body {
    background-color: var(--background-color);
    color: var(--text-color);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', sans-serif;
}

#main-container {
    width: var(--content-width) !important;
    max-width: 90% !important;
    margin: 0 auto !important;
}

.gradio-container {
    width: 100% !important;
    margin: 0 auto !important;
}

/* Tab and container styling */
.tab-nav {
    margin-bottom: 1.5rem !important;
    border-bottom: 1px solid var(--border-color) !important;
}

.tabitem {
    padding: 1.5rem !important;
}

/* Fix audio elements */
.audio-player {
    width: 100% !important;
    height: auto !important;
    margin-bottom: 10px !important;
    border-radius: var(--border-radius) !important;
    overflow: hidden !important;
}

.audio-player .wrapper {
    width: 100% !important;
}

.audio-player .controls {
    padding: 10px !important;
}

/* Better column balancing */
.gr-row {
    display: flex !important;
    flex-wrap: nowrap !important;
    gap: 20px !important;
    width: 100% !important;
    margin-bottom: 20px !important;
}

/* Button styling */
.gr-button {
    background-color: var(--primary-color) !important;
    border: none !important;
    color: white !important;
    border-radius: var(--border-radius) !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: var(--shadow) !important;
    width: 100% !important;
    height: 46px !important;
}

.gr-button:hover {
    background-color: var(--secondary-color) !important;
    transform: translateY(-2px) !important;
}

/* Card layout */
.card-title {
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    margin-bottom: 1rem !important;
    color: var(--text-color) !important;
    border-bottom: 1px solid var(--border-color) !important;
    padding-bottom: 0.5rem !important;
}

.header {
    background: linear-gradient(90deg, #2563eb, #1e40af);
    color: white;
    padding: 1.5rem;
    border-radius: var(--border-radius);
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow);
    text-align: center;
}

.header h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 700;
}

.header p {
    margin-top: 0.5rem;
    opacity: 0.9;
    font-size: 1.1rem;
}

.gr-box, .gr-form, .gr-panel {
    border-radius: var(--border-radius) !important;
    box-shadow: var(--shadow) !important;
    background-color: var(--surface-color) !important;
    border: 1px solid var(--border-color) !important;
}

.gr-padded {
    padding: 1.5rem !important;
}

.gr-input, .gr-textarea {
    border-radius: var(--border-radius) !important;
    border: 1px solid var(--border-color) !important;
    background-color: #2d3748 !important;
    color: var(--text-color) !important;
    padding: 0.75rem !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}

.gr-input:focus, .gr-textarea:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
}

.gr-form > div {
    margin-bottom: 1.5rem !important;
}

.gr-label {
    font-weight: 600 !important;
    margin-bottom: 0.5rem !important;
    color: var(--text-color) !important;
}

/* Fix slider appearance */
.gr-slider {
    width: 100% !important;
}

/* Textarea placeholder */
.gr-textarea::placeholder {
    color: var(--secondary-text-color) !important;
    opacity: 0.7 !important;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 2rem;
    padding: 1rem;
    font-size: 0.9rem;
    color: var(--secondary-text-color);
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate", font=["Inter", "sans-serif"]), elem_id="main-container") as demo:
    mutex = threading.Lock()
    
    # Header
    gr.HTML('''
    <div class="header">
        <h1>IndexTTS-2</h1>
        <p>A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech</p>
    </div>
    ''')
    
    # Main content
    with gr.Tab("Audio Generation"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=350):
                gr.HTML('<div class="card-title">Voice Reference</div>')
                os.makedirs("prompts", exist_ok=True)
                gr.HTML('<div style="margin-bottom: 10px; color: #94a3b8; font-size: 0.9rem;">For best results, use clean voice audio less than 15 seconds long.</div>')
                prompt_audio = gr.Audio(label="Upload reference audio", 
                                      sources=["upload", "microphone"], 
                                      type="filepath",
                                      elem_id="audio_component")
                
                gr.HTML('<div class="card-title">Emotion Control</div>')
                emotion_mode = gr.Radio(
                    choices=["none", "audio", "vector", "text"],
                    value="none",
                    label="Emotion Mode",
                    info="Choose how to control emotion"
                )
                
                # Emotion Audio Reference
                with gr.Group(visible=False) as emo_audio_group:
                    emo_audio = gr.Audio(label="Emotion Reference Audio", 
                                       sources=["upload", "microphone"], 
                                       type="filepath")
                    emo_alpha = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1,
                        value=1.0,
                        label="Emotion Strength",
                        info="How much emotion to apply (0.0-1.0)"
                    )
                
                # Emotion Vector Controls
                with gr.Group(visible=False) as emo_vector_group:
                    gr.HTML('<div style="margin-bottom: 10px; color: #94a3b8; font-size: 0.9rem;">Adjust emotion intensities (0.0-1.2):</div>')
                    with gr.Row():
                        happy = gr.Slider(0.0, 1.2, 0.0, step=0.1, label="Happy")
                        angry = gr.Slider(0.0, 1.2, 0.0, step=0.1, label="Angry")
                    with gr.Row():
                        sad = gr.Slider(0.0, 1.2, 0.0, step=0.1, label="Sad")
                        afraid = gr.Slider(0.0, 1.2, 0.0, step=0.1, label="Afraid")
                    with gr.Row():
                        disgusted = gr.Slider(0.0, 1.2, 0.0, step=0.1, label="Disgusted")
                        melancholic = gr.Slider(0.0, 1.2, 0.0, step=0.1, label="Melancholic")
                    with gr.Row():
                        surprised = gr.Slider(0.0, 1.2, 0.0, step=0.1, label="Surprised")
                        calm = gr.Slider(0.0, 1.2, 1.0, step=0.1, label="Calm")
                
                # Text-based Emotion
                with gr.Group(visible=False) as emo_text_group:
                    emo_text = gr.Textbox(
                        label="Emotion Description",
                        placeholder="Describe the desired emotion (e.g., 'excited and happy', 'sad and melancholic')",
                        lines=2
                    )
                    emo_text_alpha = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1,
                        value=0.6,
                        label="Emotion Strength",
                        info="Recommended: 0.6 or lower for text mode"
                    )
            
            with gr.Column(scale=2, min_width=500):
                gr.HTML('<div class="card-title">Text Input</div>')
                input_text_single = gr.Textbox(
                    label="Enter text to convert to speech",
                    placeholder="Type your text here...",
                    lines=5
                )
                
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.HTML('<div class="card-title">Generation Settings</div>')
                        seed_input = gr.Slider(
                            minimum=0,
                            maximum=1000000,
                            step=1,
                            label="Seed",
                            value=0,
                            info="Set to 0 for random seed"
                        )
                        use_random = gr.Checkbox(
                            label="Use Random Sampling",
                            value=False,
                            info="Adds randomness but may reduce voice cloning fidelity"
                        )
                    with gr.Column(scale=1):
                        gr.HTML('<div style="height: 38px;"></div>')
                        gen_button = gr.Button("🔊 Generate Speech", variant="primary", size="lg")
        
        gr.HTML('<div class="card-title">Generated Output</div>')
        output_audio = gr.Audio(label="", visible=True, elem_id="output_audio")
    
    with gr.Tab("About IndexTTS-2"):
        gr.Markdown("""
        # IndexTTS-2: Advanced Features
        
        ## 🎭 Emotion Control Modes
        
        **None**: Standard voice cloning without emotion control
        
        **Audio Reference**: Use a separate audio file to control the emotion of the generated speech
        - Upload an emotional reference audio
        - Adjust emotion strength (0.0-1.0)
        
        **Vector Control**: Manually adjust 8 different emotion intensities
        - Happy, Angry, Sad, Afraid, Disgusted, Melancholic, Surprised, Calm
        - Each emotion can be set from 0.0 to 1.2
        
        **Text Description**: Describe emotions in natural language
        - The AI will automatically detect emotions from your description
        - Recommended emotion strength: 0.6 or lower for natural results
        
        ## 🎯 Key Features
        
        - **Zero-shot voice cloning**: Generate speech matching any reference voice
        - **Emotional expression control**: Independent control over timbre and emotion
        - **Duration control**: Precise control over speech timing (coming soon)
        - **Multi-language support**: English and Chinese text synthesis
        - **High-quality audio**: 22kHz output with BigVGAN vocoder
        
        ## 📚 Tips for Best Results
        
        1. **Voice Reference**: Use clean, clear audio under 15 seconds
        2. **Emotion Audio**: Choose audio that clearly expresses the desired emotion
        3. **Text Mode**: Use descriptive emotion words for better detection
        4. **Vector Mode**: Start with one primary emotion, then fine-tune others
        5. **Random Sampling**: Disable for consistent results, enable for variation
        
        ## 🔬 Technical Details
        
        - **Model**: IndexTTS-2 (GPT-style autoregressive TTS)
        - **Vocoder**: BigVGAN v2 22kHz 80-band
        - **Emotion Engine**: Qwen-based emotion detection
        - **Audio Processing**: 22kHz sampling rate, mel-spectrogram features
        """)
    
    # Footer
    gr.HTML('''
    <div class="footer">
        <p>IndexTTS-2 - A Breakthrough in Emotionally Expressive Zero-Shot Text-to-Speech</p>
        <p><a href="https://github.com/index-tts/index-tts" target="_blank">GitHub</a> | 
           <a href="https://arxiv.org/abs/2506.21619" target="_blank">Paper</a> | 
           <a href="https://index-tts.github.io/index-tts2.github.io/" target="_blank">Demo</a></p>
    </div>
    ''')

    # Handle emotion mode switching
    def update_emotion_visibility(mode):
        return (
            gr.update(visible=(mode == "audio")),  # emo_audio_group
            gr.update(visible=(mode == "vector")), # emo_vector_group
            gr.update(visible=(mode == "text"))    # emo_text_group
        )
    
    emotion_mode.change(
        update_emotion_visibility,
        inputs=[emotion_mode],
        outputs=[emo_audio_group, emo_vector_group, emo_text_group]
    )

    prompt_audio.upload(update_prompt_audio,
                         inputs=[],
                         outputs=[gen_button])

    gen_button.click(gen_single,
                     inputs=[
                         prompt_audio, input_text_single, seed_input, 
                         emo_audio, emo_alpha, emotion_mode, emo_text,
                         happy, angry, sad, afraid, disgusted, melancholic, surprised, calm,
                         use_random
                     ],
                     outputs=[output_audio])


if __name__ == "__main__":
    # Disable Gradio's update message
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    
    try:
        # Ensure our custom filter is active
        if not isinstance(sys.stderr, FilteredStderr):
            sys.stderr = FilteredStderr(sys.stderr)
            
        # Disable the torch distributed elastic warning
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
        
        demo.queue(20)
        demo.launch(server_name="127.0.0.1", share=False)
    except Exception as e:
        print(f"Error launching the web interface: {e}")
        print("Try running with CPU only mode if you're having GPU-related issues")
