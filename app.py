"""
ISOM5240 Individual Assignment
Storytelling Application using Hugging Face Pipelines
Target audience: Kids aged 3-10 years old
"""

import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS
import tempfile
import os
import requests


# ──────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="✨ Magic Story Time!",
    page_icon="📖",
    layout="centered",
)

# ──────────────────────────────────────────────
# Custom CSS – kid-friendly colourful theme
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fredoka+One&family=Nunito:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Nunito', sans-serif; background-color: #FFF9F0; }
    h1 { font-family: 'Fredoka One', cursive; color: #FF6B6B; text-align: center; font-size: 3rem !important; }
    .subtitle { text-align: center; color: #6C63FF; font-size: 1.2rem; margin-bottom: 2rem; }
    .story-box {
        background: linear-gradient(135deg, #FFE0EC, #E0F0FF);
        border-radius: 20px; padding: 1.5rem 2rem;
        font-size: 1.15rem; line-height: 1.8; color: #333;
        border: 3px solid #FFB6C1; margin-top: 1rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #FF6B6B, #FF8E53);
        color: white; font-family: 'Fredoka One', cursive;
        font-size: 1.2rem; border-radius: 50px; border: none;
        padding: 0.6rem 2rem; width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Model loading — only ONE local model (captioner)
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading image captioning model…")
def load_captioner():
    """
    Load only the BLIP image-to-text pipeline locally.
    Keeping just one model reduces memory usage on Streamlit Cloud free tier.
    """
    return pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
    )


# ──────────────────────────────────────────────
# Core functions
# ──────────────────────────────────────────────

def generate_caption(image: Image.Image, captioner) -> str:
    """
    Generate a short descriptive caption from the uploaded image.

    Args:
        image:     PIL Image uploaded by the user.
        captioner: Hugging Face image-to-text pipeline.

    Returns:
        Caption string.
    """
    results = captioner(image)
    return results[0]["generated_text"]


def generate_story(caption: str) -> str:
    """
    Expand the caption into a child-friendly story using a simple template.
    No second model is loaded — avoids OOM crash on Streamlit Cloud free tier.

    Args:
        caption: Image caption from generate_caption().

    Returns:
        A short story string (50-100 words).
    """
    story = (
        f"Once upon a time, there was {caption}. "
        "It was a bright and sunny day, and everyone was feeling happy. "
        "The little friends decided to go on a magical adventure together. "
        "They skipped through the flowers, laughed under the rainbow, "
        "and discovered a treasure chest full of yummy cookies! "
        "At the end of the day, they hugged each other and went home. "
        "And they all lived happily ever after. The End! 🌈"
    )
    return story


def text_to_speech(text: str) -> str:
    """
    Convert story text to an MP3 audio file using gTTS.

    Args:
        text: Story text to synthesise.

    Returns:
        File path of the generated temporary MP3 file.
    """
    tts = gTTS(text=text, lang="en", slow=False)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp_file.name)
    return tmp_file.name


# ──────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────

def main():
    st.markdown("<h1>✨ Magic Story Time! 📖</h1>", unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Upload a picture and listen to your very own magical story! 🌈🦄</p>',
        unsafe_allow_html=True,
    )

    captioner = load_captioner()

    uploaded_file = st.file_uploader(
        "📷 Upload your picture here!",
        type=["jpg", "jpeg", "png", "webp"],
        help="Choose any fun picture – an animal, a toy, or a place you love!",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Your picture 🎨", use_container_width=True)

        if st.button("🪄 Create My Story!"):
            # Step 1 – caption
            with st.spinner("👀 Looking at your picture…"):
                caption = generate_caption(image, captioner)
            st.info(f"**I see:** {caption}")

            # Step 2 – story
            with st.spinner("✍️ Writing your magical story…"):
                story = generate_story(caption)

            st.markdown("### 📖 Your Story")
            st.markdown(f'<div class="story-box">{story}</div>', unsafe_allow_html=True)

            # Step 3 – TTS
            with st.spinner("🔊 Recording the story for you…"):
                audio_path = text_to_speech(story)

            st.markdown("### 🎧 Listen to Your Story!")
            with open(audio_path, "rb") as audio_file:
                st.audio(audio_file.read(), format="audio/mp3")

            os.unlink(audio_path)

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#aaa; font-size:0.85rem;'>"
        "Made with ❤️ for curious little minds · ISOM5240</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
