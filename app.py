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

    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif;
        background-color: #FFF9F0;
    }
    h1 {
        font-family: 'Fredoka One', cursive;
        color: #FF6B6B;
        text-align: center;
        font-size: 3rem !important;
    }
    .subtitle {
        text-align: center;
        color: #6C63FF;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .story-box {
        background: linear-gradient(135deg, #FFE0EC, #E0F0FF);
        border-radius: 20px;
        padding: 1.5rem 2rem;
        font-size: 1.15rem;
        line-height: 1.8;
        color: #333;
        border: 3px solid #FFB6C1;
        margin-top: 1rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #FF6B6B, #FF8E53);
        color: white;
        font-family: 'Fredoka One', cursive;
        font-size: 1.2rem;
        border-radius: 50px;
        border: none;
        padding: 0.6rem 2rem;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #FF8E53, #FF6B6B);
        transform: scale(1.02);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Model loading (cached so it only loads once)
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading image captioning model…")
def load_captioner():
    """
    Load the image-text-to-text pipeline using BLIP.
    This task type correctly accepts a PIL Image in newer transformers versions.
    """
    return pipeline(
        "image-text-to-text",
        model="Salesforce/blip-image-captioning-base",
    )


@st.cache_resource(show_spinner="Loading story generation model…")
def load_story_generator():
    """
    Load the text-generation pipeline from Hugging Face.
    Uses a lightweight GPT-2 variant suitable for Streamlit Cloud free tier.
    """
    return pipeline(
        "text-generation",
        model="sshleifer/tiny-gpt2",
    )


# ──────────────────────────────────────────────
# Core functions
# ──────────────────────────────────────────────

def generate_caption(image: Image.Image, captioner) -> str:
    """
    Generate a short descriptive caption from an uploaded image.

    Args:
        image:     PIL Image object uploaded by the user.
        captioner: Hugging Face image-text-to-text pipeline.

    Returns:
        A short caption string describing the image.
    """
    results = captioner(image, generate_kwargs={"max_new_tokens": 50})
    caption = results[0]["generated_text"]
    return caption


def generate_story(caption: str, story_generator) -> str:
    """
    Expand a caption into a short, child-friendly story (50-100 words).

    Args:
        caption:         Image caption produced by generate_caption().
        story_generator: Hugging Face text-generation pipeline.

    Returns:
        A generated story string.
    """
    prompt = (
        f"Write a fun and magical short story for young children (aged 3 to 10) "
        f"based on the following scene: {caption}. "
        f"The story should be simple, cheerful, and between 50 and 100 words."
    )

    output = story_generator(
        prompt,
        max_new_tokens=120,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.85,
        top_p=0.92,
        repetition_penalty=1.3,
    )

    # Extract only the newly generated text (strip the original prompt)
    full_text: str = output[0]["generated_text"]
    story = full_text[len(prompt):].strip()

    # Fallback: if the model returns nothing useful, craft a simple story
    if not story:
        story = (
            f"Once upon a time, {caption}. "
            "It was a wonderful adventure full of joy and laughter! "
            "All the friends played together and lived happily ever after. The End."
        )

    return story


def text_to_speech(text: str) -> str:
    """
    Convert story text to an MP3 audio file using gTTS.

    Args:
        text: The story text to synthesise.

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
    # Title and subtitle
    st.markdown("<h1>✨ Magic Story Time! 📖</h1>", unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Upload a picture and listen to your very own magical story! 🌈🦄</p>',
        unsafe_allow_html=True,
    )

    # Load models
    captioner = load_captioner()
    story_generator = load_story_generator()

    # File uploader
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
                story = generate_story(caption, story_generator)

            st.markdown("### 📖 Your Story")
            st.markdown(f'<div class="story-box">{story}</div>', unsafe_allow_html=True)

            # Step 3 – TTS
            with st.spinner("🔊 Recording the story for you…"):
                audio_path = text_to_speech(story)

            st.markdown("### 🎧 Listen to Your Story!")
            with open(audio_path, "rb") as audio_file:
                st.audio(audio_file.read(), format="audio/mp3")

            # Clean up temporary audio file
            os.unlink(audio_path)

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#aaa; font-size:0.85rem;'>"
        "Made with ❤️ for curious little minds · ISOM5240</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
