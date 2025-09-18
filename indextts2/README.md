
# IndexTTS-2 Web UI
<img width="1515" height="1300" alt="Screenshot 2025-09-13 132950" src="https://github.com/user-attachments/assets/32e28605-ba93-499f-ace1-969772f3005b" />

A comprehensive web interface for [IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2), a breakthrough in emotionally expressive and duration-controlled auto-regressive zero-shot text-to-speech.

<p align="center">
<a href='https://arxiv.org/abs/2506.21619'><img src='https://img.shields.io/badge/ArXiv-2506.21619-red'></a>
<a href='https://github.com/index-tts/index-tts'><img src='https://img.shields.io/badge/GitHub-Code-orange?logo=github'></a>
<a href='https://huggingface.co/IndexTeam/IndexTTS-2'><img src='https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface' /></a>
</p>

## 🎭 New Features in IndexTTS-2

- **Advanced Emotion Control**: Multiple ways to control emotional expression
  - Audio-based emotion reference
  - Manual emotion vector adjustment (8 emotions)
  - Natural language emotion descriptions
- **Emotion-Speaker Disentanglement**: Independent control over voice timbre and emotional expression
- **Enhanced Audio Quality**: Improved stability and clarity in emotional expressions
- **Smart Audio Processing**: Automatic audio length optimization (15s max for optimal performance)
- **Multi-modal Emotion Input**: Choose the emotion control method that works best for you

## Features

- **Zero-shot voice cloning**: Generate speech that matches any reference audio file
- **Emotional expression control**: Add emotions without changing the speaker's voice characteristics
- **Duration control**: Precise timing control (coming in future updates)
- **Simple web interface**: Easy-to-use Gradio interface with advanced controls
- **Automatic model downloading**: Downloads IndexTTS-2 models from Hugging Face automatically
- **Multi-language support**: English and Chinese text synthesis
- **High-quality output**: 22kHz audio with BigVGAN v2 vocoder

## Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)

### Installation

1. Install the requirements:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

2. Run the web UI:
```bash
python webui.py
```

On first run, the script will automatically download the necessary model files from the Hugging Face repository. This might take some time depending on your internet connection speed.

## Usage

1. Upload a reference audio file (voice you want to clone)
2. Enter the text you want to convert to speech
3. Click "Generate Speech"
4. Listen to the generated audio

## 🎯 Emotion Control Modes

### 1. Audio Reference Mode
Upload an emotional reference audio file to control the emotion of generated speech while maintaining the target speaker's voice characteristics.

### 2. Vector Control Mode
Manually adjust 8 different emotion intensities:
- **Happy**: Joy, excitement, positive emotions
- **Angry**: Anger, frustration, aggressive emotions  
- **Sad**: Sadness, sorrow, melancholy
- **Afraid**: Fear, anxiety, nervousness
- **Disgusted**: Disgust, revulsion, distaste
- **Melancholic**: Deep sadness, contemplative mood
- **Surprised**: Surprise, shock, amazement
- **Calm**: Neutral, peaceful, relaxed state

### 3. Text Description Mode
Describe emotions in natural language and let the AI automatically detect and apply them:
- "excited and happy"
- "sad and melancholic" 
- "angry and frustrated"
- "calm and peaceful"

## Model Information

IndexTTS-2 is an advanced GPT-style autoregressive text-to-speech model with breakthrough features:
- **Emotion-Speaker Disentanglement**: Independent control over timbre and emotional expression
- **Duration Control**: Precise synthesis timing control (autoregressive model-friendly method)
- **Enhanced Stability**: Three-stage training paradigm for improved speech quality
- **Multi-modal Emotion Control**: Audio, vector, and text-based emotion input
- **High-Quality Vocoder**: BigVGAN v2 22kHz 80-band for superior audio quality

For more information, visit the [IndexTTS-2 Hugging Face page](https://huggingface.co/IndexTeam/IndexTTS-2) or read the [research paper](https://arxiv.org/abs/2506.21619).

## 💡 Usage Tips

1. **Voice Reference**: Use clean, clear audio under 15 seconds for best results
2. **Emotion Audio**: Choose audio that clearly expresses the desired emotion
3. **Text Emotion Mode**: Use descriptive emotion words for better detection
4. **Vector Mode**: Start with one primary emotion, then fine-tune others
5. **Random Sampling**: Disable for consistent results, enable for variation

## Citation

IndexTTS-2:
```
@article{zhou2025indextts2,
  title={IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech},
  author={Siyi Zhou, Yiquan Zhou, Yi He, Xun Zhou, Jinchao Wang, Wei Deng, Jingchen Shu},
  journal={arXiv preprint arXiv:2506.21619},
  year={2025}
}
```

IndexTTS (Original):
```
@article{deng2025indextts,
  title={IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System},
  author={Wei Deng, Siyi Zhou, Jingchen Shu, Jinchao Wang, Lu Wang},
  journal={arXiv preprint arXiv:2502.05512},
  year={2025}
}
```
