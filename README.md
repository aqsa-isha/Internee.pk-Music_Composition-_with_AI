# 🎵 Text to Music Generator

This repository contains a **Text to Music Generator** app that allows users to generate music tracks based on text descriptions using Meta's [Audiocraft](https://github.com/facebookresearch/audiocraft) library, specifically the **MusicGen** model. With this app, you can describe a mood, style, or theme, and the AI will create a melody that matches your description.

## Features
- Generate music tracks using natural language descriptions.
- Adjustable duration for the generated music (up to 20 seconds).
- Download the generated music in `.wav` format.
- Deployed with a modern, interactive Streamlit interface.
- Run in Google Colab using **ngrok** for easy public access.

## Demo
🎵 [Access the app on Google Colab](https://colab.research.google.com/drive/1_b-puDdTFBmRJDEewQLDVX9h6LNRZtOK#scrollTo=uT9Zj0n-vYtg)

---

## Installation

### Prerequisites
1. **Python 3.8+**  
2. Install the following dependencies:
   - `streamlit`
   - `torch`
   - `torchaudio`
   - `audiocraft`
   - `pyngrok`

To install dependencies, run:
```bash
pip install streamlit torch torchaudio audiocraft pyngrok
