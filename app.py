# ISOM5240 Individual Assignment
# Storytelling Application for 3-10 year-old children
import streamlit as st
from transformers import pipeline
from gtts import gTTS
from PIL import Image

st.set_page_config(page_title="Kids Story App", page_icon="📖")
st.title("📖 Image Storytelling for Kids")
st.write("Upload a picture, and I will create a short story for you!")

# ----------------------
# Load Hugging Face Models
# ----------------------
@st.cache_resource
def load_models():
    captioner = pipeline("image-text-to-text", model="Salesforce/blip-image-captioning-base")
    story_generator = pipeline("text-generation", model="distilgpt2")
    return captioner, story_generator

captioner, story_generator = load_models()

# Upload image
uploaded = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, use_column_width=True)

    # ----------------------
    # 1. Image Caption
    # ----------------------
    with st.spinner("Analyzing image..."):
        caption = captioner(text="a photo of", images=image)[0]["generated_text"]
        st.info(f"Image Caption: {caption}")

    # ----------------------
    # 2. Generate story (NO PROMPT!!! Only caption)
    # ----------------------
    with st.spinner("Writing story..."):

        # ✅ NO PROMPT — ONLY use caption to make story
        story = story_generator(
            caption,
            max_new_tokens=80,
            temperature=0.6,
            repetition_penalty=1.2,
            pad_token_id=50256
        )[0]["generated_text"]

        # Clean
        story = story.replace(caption, "").strip()

        # Ensure complete sentence
        if "." in story:
            story = story[:story.rfind(".") + 1]

        # Keep 50-100 words
        words = story.split()
        if len(words) > 100:
            story = " ".join(words[:100])
        if len(words) < 50:
            story = " ".join(words[:50])

        st.subheader("✨ Your Story")
        st.write(story)

    # ----------------------
    # 3. Text-to-Speech
    # ----------------------
    with st.spinner("Generating audio..."):
        tts = gTTS(text=story, lang="en")
        tts.save("story.mp3")
        with open("story.mp3", "rb") as f:
            st.audio(f.read(), format="audio/mp3")

    st.success("Story completed! 😊")
