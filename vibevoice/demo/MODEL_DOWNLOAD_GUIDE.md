# VibeVoice Model Download Guide

The VibeVoice Gradio demo now includes built-in model download functionality that allows you to download pre-trained models directly from HuggingFace Hub.

## Available Models

### VibeVoice-1.5B
- **Repository**: [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B)
- **Size**: ~2.7B parameters
- **Description**: Compact model, good for most use cases
- **Recommended for**: General usage, faster inference

### VibeVoice-7B-Preview
- **Repository**: [WestZhang/VibeVoice-Large-pt](https://huggingface.co/WestZhang/VibeVoice-Large-pt)
- **Size**: ~9.34B parameters  
- **Description**: Larger model, potentially higher quality
- **Recommended for**: Higher quality output, when computational resources allow

## How to Use

1. **Launch the Gradio Demo**:
   ```bash
   python demo/gradio_demo.py
   ```
   
   Or specify a custom model directory:
   ```bash
   python demo/gradio_demo.py --model_path ./my_models
   ```

2. **Access Model Download**:
   - In the web interface, look for the "📥 Model Download" section in the left sidebar
   - Click on "Download VibeVoice Models" accordion to expand

3. **Select and Download**:
   - Choose your preferred model (VibeVoice-1.5B or VibeVoice-7B-Preview)
   - Optionally specify a custom download path, or leave empty for default (`./models/ModelName`)
   - Click "📥 Download Model" to start the download
   - Monitor progress in the "Download Status" text area

4. **Use Downloaded Model**:
   After download completes, the model will appear in the Model Management dropdown. Simply:
   - Select the downloaded model from the "Select Model to Load" dropdown
   - Click "📂 Load Selected Model" 
   - Start generating podcasts!
   
   No need to restart the demo!

## Features

- **Progress Tracking**: Real-time download progress updates
- **Resume Support**: Interrupted downloads can be resumed
- **Path Validation**: Automatic check for existing models
- **Error Handling**: Comprehensive error messages and troubleshooting
- **Update Detection**: Checks for updates if model already exists
- **Custom Voice Upload**: Add your own voice samples for personalized speakers

## Download Locations

By default, models are downloaded to the `./models/` directory:
- `./models/VibeVoice-1.5B/` for the 1.5B model
- `./models/VibeVoice-7B-Preview/` for the 7B model

You can customize the download location using the "Download Path" field. The demo automatically scans this directory and makes downloaded models available in the dropdown selector.

## Custom Voice Upload

The demo supports uploading your own voice samples to create personalized speakers:

### **Upload Process:**
1. Click on "🎤 Upload Custom Voices" accordion in the Speaker Selection section
2. Upload an audio file (WAV, MP3, FLAC, OGG, M4A, AAC)
3. Enter a unique voice name (e.g., "My-Voice" or "John-Narrator")
4. Click "➕ Add Custom Voice"
5. The custom voice will automatically appear in speaker dropdowns

### **Voice Requirements:**
- **Length**: 3-30 seconds recommended (1-60 seconds supported)
- **Quality**: Clear speech with minimal background noise
- **Content**: Natural speaking voice works best
- **Format**: Automatic conversion to 24kHz WAV for optimal compatibility

### **File Storage:**
- Custom voices are saved to `demo/voices/` directory
- Files are prefixed with "custom-" to distinguish from built-in voices
- Audio is automatically normalized and converted to the required format

## Troubleshooting

- **Network Issues**: Download will automatically retry and resume if interrupted
- **Disk Space**: Ensure sufficient disk space (1.5B model: ~5-6GB, 7B model: ~18-20GB)
- **Permissions**: Make sure you have write permissions to the download directory
- **HuggingFace Authentication**: Some models may require HuggingFace login (use `huggingface-cli login`)
