# ISOM5240 Individual Assignment
# Storytelling Application for 3-10 year-old kids
import streamlit as st
from transformers import pipeline
from gtts import gTTS
from PIL import Image

# Page config
st.set_page_config(page_title="Kids Story App", page_icon="📖")
st.title("📖 Image Storytelling for Kids")
st.write("Upload an image, and I will make a lovely story for you!")

# Load models (cache to avoid reloading)
@st.cache_resource
def load_models():
    # 使用支持image-to-text的专用模型，或者用pipeline的正确参数
    captioner = pipeline("image-text-to-text", model="Salesforce/blip-image-captioning-base")
    story_generator = pipeline("text-generation", model="distilgpt2", max_new_tokens=150)
    return captioner, story_generator

captioner, story_generator = load_models()

# Upload image
img = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])

if img:
    image = Image.open(img)
    st.image(image, caption="Your picture", use_column_width=True)

    # Step 1: Image caption
    with st.spinner("Analyzing picture..."):
        # 正确调用方式：只传图片
        caption = captioner(image)[0]["generated_text"]
        st.info(f"Image caption: {caption}")

    # Step 2: Generate story (50–100 words, kid-friendly)
    with st.spinner("Writing story..."):
        prompt = f"Write a short, sweet, simple story for young kids based on this: {caption}. Keep it 50-100 words, happy and easy."
        story = story_generator(prompt)[0]["generated_text"]
        story = story.replace(prompt, "").strip()

        # Ensure length 50–100 words
        words = story.split()
        if len(words) > 100:
            story = " ".join(words[:100])
        if len(words) < 30:
            story += " The little friends played happily all day. It was a warm and wonderful day full of fun."

        st.subheader("Your Story ✨")
        st.write(story)

    # Step 3: Text to speech
    with st.spinner("Making voice..."):
        tts = gTTS(text=story, lang="en")
        tts.save("story.mp3")

        with open("story.mp3", "rb") as f:
            st.audio(f.read(), format="audio/mp3")

    st.success("Done! Hope you like the story 😊")
