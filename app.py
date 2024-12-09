from audiocraft.models import MusicGen
import streamlit as st
import torch
import torchaudio
import os
import base64

# Cache the model to avoid reloading it on every interaction
@st.cache_resource
def load_model():
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    return model


def generate_music_tensors(description, duration: int):
    """Generates music tensors based on input description and duration."""
    st.write(f"Description: {description}")
    st.write(f"Duration: {duration}")
    model = load_model()

    # Set generation parameters
    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=duration
    )

    output = model.generate(
        descriptions=[description],
        progress=True,
        return_tokens=True
    )
    return output[0]


def save_audio(samples: torch.Tensor, sample_rate=32000, save_path="audio_output/"):
    """Saves generated audio to a specified path."""
    st.write("Processing audio samples...")
    os.makedirs(save_path, exist_ok=True)  # Ensure directory exists

    samples = samples.detach().cpu()
    if samples.dim() == 2:
        samples = samples[None, ...]

    for idx, audio in enumerate(samples):
        audio_path = os.path.join(save_path, f"audio_{idx}.wav")
        torchaudio.save(audio_path, audio, sample_rate)
        return audio_path


def get_binary_file_downloader_html(bin_file, file_label='File'):
    """Generates an HTML link for downloading a file."""
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href


st.set_page_config(
    page_icon="🎵",
    page_title="Text to Music Generator",
    layout="wide"
)

# Custom CSS for a purple and white theme
custom_css = """
<style>
    body {
        background-color: #4B0082; /* Purple background */
        color: white; /* White text */
        font-family: 'Arial', sans-serif;
    }
    header {
        color: white !important;
    }
    .stTextArea, .stSlider, .stButton {
        border-radius: 10px;
        background: #6A0DAD; /* Bright purple elements */
        color: white !important;
    }
    .stSlider .stSliderValue {
        color: white !important;
    }
    .stExpander {
        background: #4B0082; /* Dark purple for expanders */
        color: white;
    }
    .stAudio {
        background: white !important;
        border-radius: 10px;
    }
    h1, h2, h3, h4, h5 {
        text-align: center;
        color: white;
        text-shadow: 2px 2px #6A0DAD;
    }
    .stButton > button {
        background: #8A2BE2 !important; /* Bright purple */
        color: white !important;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }
    .stDownloadButton {
        background: #6A0DAD !important;
        color: white !important;
        font-weight: bold;
        border: 2px solid #FFFFFF !important;
        border-radius: 10px;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)


def main():
    st.title("🎵 Text to Music Generator 🎵")

    # Explanation section
    with st.expander("💡 What does this app do?"):
        st.markdown("""
        <div style='text-align: center;'>
            <p>🎶 **Welcome to the Text to Music Generator!** 🎶</p>
            <p>This app allows you to create music from text descriptions using Meta's Audiocraft library (<b>MusicGen</b>). 
            Simply describe the music style or mood you'd like, and AI will compose a melody for you.</p>
        </div>
        """, unsafe_allow_html=True)

    # User inputs
    st.markdown("<h2 style='text-align: center;'>🎤 Describe Your Music 🎵</h2>", unsafe_allow_html=True)
    text_area = st.text_area(
        "Enter your music description:",
        placeholder="e.g., upbeat classical symphony or relaxing lo-fi beats..."
    )
    time_slider = st.slider("⏱ Select duration (in seconds):", 1, 20, 10)

    # Generate music when inputs are provided
    if text_area and time_slider:
        st.markdown("<h3 style='text-align: center;'>📋 Your Inputs:</h3>", unsafe_allow_html=True)
        st.json({
            'Your Description': text_area,
            'Selected Time Duration (in Seconds)': time_slider
        })

        st.markdown("<h3 style='text-align: center;'>🎶 Generated Music 🎶</h3>", unsafe_allow_html=True)
        music_tensors = generate_music_tensors(text_area, time_slider)
        audio_filepath = save_audio(music_tensors)

        # Audio playback and download
        st.audio(audio_filepath)
        st.markdown(
            get_binary_file_downloader_html(audio_filepath, 'Generated Music'),
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
