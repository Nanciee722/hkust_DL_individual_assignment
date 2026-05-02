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
    captioner = pipeline("image-text-to-text", model="Salesforce/blip-image-captioning-base")
    
    # ✅ YOUR NEW MODEL: Qwen3-0.6B
    story_generator = pipeline(
        "text-generation", 
        model="Qwen/Qwen3-0.6B",
        max_new_tokens=100,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.1
    )
    return captioner, story_generator

captioner, story_generator = load_models()

# Upload image
img = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])

if img:
    image = Image.open(img)
    st.image(image, caption="Your picture", use_column_width=True)

    # Step 1: Image caption
    with st.spinner("Analyzing picture..."):
        result = captioner(
            text="a picture of",
            images=image
        )
        caption = result[0]["generated_text"]
        st.info(f"Image caption: {caption}")

    # Step 2: Generate story with Qwen3-0.6B (NO extra prompt, NO backup)
    with st.spinner("Writing story..."):
        
        # ✅ SUPER CLEAN PROMPT (you wanted no complicated prompt)
        prompt = f"Write a simple 50-100 word kids' story about: {caption}"
        
        outputs = story_generator(prompt, max_new_tokens=100, pad_token_id=151643)
        full_text = outputs[0]["generated_text"]
        
        story = full_text.replace(prompt, "").strip()

        # Ensure clean ending
        if "." in story:
            story = story[:story.rfind(".") + 1]

        # Keep 50-100 words
        words = story.split()
        if len(words) > 100:
            story = " ".join(words[:100])

        st.subheader("Your Story ✨")
        st.write(story)

    # Step 3: Text to speech
    with st.spinner("Making voice..."):
        tts = gTTS(text=story, lang="en")
        tts.save("story.mp3")
        with open("story.mp3", "rb") as f:
            st.audio(f.read(), format="audio/mp3")

    st.success("Done! Hope you like the story 😊")
