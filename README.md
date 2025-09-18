## 💖 Support & Donations

If you enjoy using **Ultimate TTS Studio** and would like to support its ongoing development, your generosity is deeply appreciated. Any amount — big or small — makes a difference!

### 🪙 Bitcoin Donations Welcome

**Bitcoin Address:** `1N942jHr6vVuR2KAe2JEf3nN59eR21tpKv`

<table>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/a93b7a18-1a42-461c-bbe2-466f425ce350" alt="Donate via Bitcoin" width="150"/>
    </td>
    <td style="padding-left: 20px;">
      <p>Your donations help cover:</p>
      <ul>
        <li>🧠 Development time and continuous feature updates</li>
        <li>💾 Storage upgrades and data handling</li>
        <li>🧾 Ongoing subscription costs that support development tools and AI services</li>
        <li>🚀 Hardware improvements like GPU upgrades for faster and more powerful models</li>
      </ul>
      <p>
        Every contribution directly supports the creation of more innovative apps like this.<br>
        <strong>Thank you for helping keep the project alive and growing!</strong>
      </p>
    </td>
  </tr>
</table>

---

## ⚠️ Notice: Major App Update


### 📅 Sept 13, 2025
This update brings KittenTTS
<img width="1758" height="377" alt="Screenshot 2025-09-13 231901" src="https://github.com/user-attachments/assets/5a701acf-5639-46dd-9323-b04402e31e2e" />

### 📅 July 29, 2025

This recent update brings Higgs-Audio TTS:

### 📅 June 23, 2025

This recent update brings a few UI improvements focused on clarity and usability:

### 🎛️ TTS UI Refactor

* The **TTS engine selector** is now organized into a **tabbed interface**, making it easier to navigate and less overwhelming.
![Screenshot 2025-06-23 204234](https://github.com/user-attachments/assets/2b0382eb-c358-4a1c-85d6-cb99aa4217f8)

* The **audiobook feature** has been moved into its own tab to **reduce visual clutter** and improve user experience.
![Screenshot 2025-06-23 204240](https://github.com/user-attachments/assets/05a9df2d-2573-418a-9dbe-b82f6c5e8e1b)


---

### 📅 June 18, 2025

We’ve pushed another exciting update packed with new functionality and improvements!

### 🆕 New Additions & Improvements

### 🗣️ TTS Integration Expanded

* **F5-TTS** has now been added as a **fifth supported engine**, and it works seamlessly across all modes.
* **Index-TTS** has been added as a supported speech engine.
* All **TTS engines now work across all modes**, including narration, conversation, and ambient.

### 💬 Kokoro Conversation Mode

* **Kokoro** now fully supports **conversation mode**, offering a more dynamic and interactive experience.

### ✅ Recommended Setup

For the **smoothest installation and full feature compatibility**:

* Use a **Conda environment**, or
* Install via **[Pinokio](https://pinokio.co)** for the easiest experience.

---

### 📅 June 10, 2025

We’re excited to announce a major update to the app!

### 🎧 New Feature: eBook to Audiobook

Bring your favorite eBooks to life with our brand-new **custom voice audiobook** feature. Instantly convert any eBook into a personalized listening experience—perfect for learning, multitasking, or relaxing on the go.
![Screenshot 2025-06-10 204108](https://github.com/user-attachments/assets/7aa08f03-4c23-4772-a1cd-6e9967fa8882)

---

### 📅 June 7, 2025

This update brings key improvements to **performance**, **model management**, and the **user interface**. Here's what's new:

### 🔧 Model Management

* Models are **no longer auto-loaded into GPU memory** at app launch.
* You can now **manually load and unload models**, giving you more precise control over memory usage.

### 🎨 UI Enhancements

* A **refreshed interface** is now live.
* The app is now **optimized for dark mode**. It still works in light mode, but some visuals may not display as intended.

### 🐟 Fish Speech Fix

* Fixed a bug where **Fish Speech** did not chunk text correctly, which could cause processing issues.

### 📥 Model Download Behavior

* **Chatterbox** and **Kokoro** models will **automatically download** the first time you click "Load."
* **Fish Speech** models must still be **downloaded manually** and are **not included** in the auto-download process.

### 🗣️ New Feature: Custom Kokoro Voices

* **Kokoro** now supports **custom `.pt` voice models**!
* Use the **Custom Voice Upload** section in the Kokoro interface to upload your own compatible voice files.

---
![Screenshot 2025-06-22 183117](https://github.com/user-attachments/assets/fb0d87cf-1289-4dd1-9e51-0a8aa2f8fbc9)
![Screenshot 2025-06-22 183134](https://github.com/user-attachments/assets/30999da1-0c51-4030-978e-30e311781f79)
![Screenshot 2025-06-22 183147](https://github.com/user-attachments/assets/d78b3aa4-09f7-4845-80c9-d621f32f7d56)



# ✨ Ultimate TTS Studio SUP3R Edition ✨

**Ultimate TTS Studio** is a powerful all-in-one text-to-speech studio that brings together **ChatterboxTTS**, **Kokoro TTS**, and **Fish Speech** under one interactive Gradio interface.

🎭 Reference Audio Cloning
🗣️ Pre-trained Multi-Language Voices
🐟 Natural TTS with Audio Effects
🎵 Real-time Voice Synthesis & Export

---

## 🚀 Features

* 🎤 **ChatterboxTTS**: Custom voice cloning using short reference clips.
* 🗣️ **Kokoro TTS**: High-quality, multilingual pre-trained voices.
* 🐟 **Fish Speech**: Advanced TTS engine.
* 🎛️ **Professional Audio Effects**: Reverb, Echo, EQ, Pitch shift, Gain.

---

> ## 🚨🚨 **WARNING / IMPORTANT NOTES** 🚨🚨
>
> ⚠️ **Tested Hardware:** This project has **only** been tested on a **Windows 11** machine with an **RTX 4090** GPU.
> 💻 Performance or compatibility on other systems is **not guaranteed**.
>
> 🔊 **Audio Caution:** The **Fish Speech** feature may occasionally produce **extremely loud** or **muffled** audio.
> 🎧 **Please lower your volume and avoid using headphones** during initial tests.

---

## 🛠️ Installation

> ⚠️ **Windows Users — Important Note on `pynini`**
> If you encounter the following error when installing `pynini`:
> `ERROR: Failed building wheel for pynini`
> You can fix this by installing it via conda:
> Pynini and wetextprocessing is needed for index-tts to work at its best

```bash
# After activating your conda environment (e.g., conda activate index-tts)
conda install -c conda-forge pynini==2.1.6
pip install WeTextProcessing --no-deps
```
---

## Option 1.

Install via [Pinokio](https://pinokio.co)
You can use the Pinokio script here for one-click setup:
[Pinokio App Installer](https://pinokio-home.netlify.app/item?uri=https://github.com/SUP3RMASS1VE/Ultimate-TTS-Studio-SUP3R-Edition-Pinokio)

### Option 1a: Install via [Dione](https://getdione.app)

You can also use [Dione](https://getdione.app) for an easy one-click installation experience:

---

## Option 2

### 🔁 **Auto-Installer Method (Recommended)**

This is the fastest way to get started. It uses a built-in installer script for automatic setup and app launching.

> 🛠️ **Before You Begin:**
> Make sure you have **Miniconda** or **Anaconda** installed on your system.
> You can download Miniconda here: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

#### 1. Clone the Repository

```bash
git clone https://github.com/SUP3RMASS1VE/Ultimate-TTS-Studio-SUP3R-Edition.git
cd Ultimate-TTS-Studio-SUP3R-Edition
```

#### 2. Run the Installer

👉 Double-click `RUN_INSTALLER` in the project folder.
This will automatically set up everything for you — dependencies, environment, etc.

#### 3. Launch the App

👉 Double-click `RUN_APP` to open the app.

#### 4. Update the App (When Needed)

👉 Double-click `RUN_UPDATE` to update the app to the latest version.

---

## Option 3

[Espeak-ng](https://github.com/espeak-ng/espeak-ng) is needed for Kokoro to work at its best.

### 1. Clone the repository

```bash
git clone https://github.com/SUP3RMASS1VE/Ultimate-TTS-Studio-SUP3R-Edition.git
cd Ultimate-TTS-Studio-SUP3R-Edition
```

### 2. Create a Python virtual environment

```bash
python -m venv env
```

### 3. Activate the environment

* **Windows**:

  ```bash
  env\Scripts\activate
  ```

### 4. Install `uv` (optional but recommended for speed)

```bash
pip install uv
```

### 5. Install dependencies

#### Install PyTorch (CUDA 12.8 build)

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

#### Install other requirements

```bash
uv pip install -r requirements.txt
```

> 💡 If you're not using `uv`, you can just use `pip install` in its place.

---

## 🧠 First-Time Setup Tips

### 📥 Download Fish Speech Model (one-time)

To use **Fish Speech**, you must download the model checkpoint from Hugging Face. This requires a Hugging Face account and access token.

### 🔐 Step-by-Step:

1. **Create an account (if needed):**
   [https://huggingface.co/join](https://huggingface.co/join)

2. **Get your access token:**
   Visit [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a **read token**.

3. **Log in via CLI:**

   ```bash
   huggingface-cli login
   ```

   Paste your token when prompted.

4. **(Optional)** Accept the model license:
   Visit [https://huggingface.co/fishaudio/openaudio-s1-mini](https://huggingface.co/fishaudio/openaudio-s1-mini) and click **"Access repository"** if prompted.

5. **Download the model:**

   ```bash
   huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
   ```


---

## ▶️ Run the Studio

```bash
python launch.py
```

This will launch a local Gradio interface at:
📍 `http://127.0.0.1:7860`

---

## 💡 Notes

* All engines are optional — the app will gracefully disable missing engines.
* ChatterboxTTS and Fish Speech support reference audio input.
* Audio effects are applied post-synthesis for professional-quality output.
* Custom Kokoro voices can be added to `custom_voices/` as `.pt` files.

---

## 📜 License

MIT License © SUP3RMASS1VE

---

## 🙏 Acknowledgments

This project proudly integrates and builds upon the amazing work of:

* [Fish Speech by fishaudio](https://github.com/fishaudio/fish-speech) – Natural and expressive TTS engine.
  📜 License: [MIT License](https://github.com/fishaudio/fish-speech/blob/main/LICENSE)

* [Kokoro TTS by hexgrad](https://github.com/hexgrad/kokoro) – High-quality multilingual voice synthesis.
  📜 License: [Apache 2.0 License](https://github.com/hexgrad/kokoro/blob/main/LICENSE)

* [ChatterboxTTS by Resemble AI](https://github.com/resemble-ai/chatterbox) – Custom voice cloning from short reference clips.
  📜 License: [Apache 2.0 License](https://github.com/resemble-ai/chatterbox/blob/main/LICENSE)

* [F5-TTS by SWivid](https://github.com/SWivid/F5-TTS) – Efficient and lightweight TTS model focused on real-time synthesis.
  📜 License: [MIT License](https://github.com/SWivid/F5-TTS/blob/main/LICENSE)

* [Index TTS](https://github.com/index-tts/index-tts) – Modular and scalable text-to-speech system with advanced voice capabilities.
  📜 License: [Apache 2.0 License](https://github.com/index-tts/index-tts/blob/main/LICENSE)

We deeply thank the authors and contributors to these projects for making this work possible.

---
