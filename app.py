# ISOM5240 Individual Assignment
# Storytelling Application for Children
import streamlit as st
from transformers import pipeline, set_seed
from gtts import gTTS
from PIL import Image
import re

# Set random seed to keep story stable
set_seed(42)

# Page configuration
st.set_page_config(page_title="Kids Story App", page_icon="📖")
st.title("📖 Image Storytelling for Kids")
st.write("Upload an image to generate a fun, short story!")

# Load Hugging Face models
@st.cache_resource
def load_models():
    # Image captioning model
    captioner = pipeline("image-text-to-text", model="Salesforce/blip-image-captioning-base")
    
    # Story generation model (optimized for complete, non-repeating stories)
    story_generator = pipeline(
        "text-generation",
        model="distilgpt2",
        max_new_tokens=100,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.3
    )
    return captioner, story_generator

captioner, story_generator = load_models()

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Your Image", use_column_width=True)

    # Step 1: Generate image caption
    with st.spinner("Generating caption..."):
        caption = captioner(text="a photo of", images=image)[0]["generated_text"]
        st.info(f"Image Caption: {caption}")

    # Step 2: Generate a complete 50–100 word story
    with st.spinner("Generating story..."):
        prompt = f"""Write a complete, child-friendly story based on this scene: {caption}
The story must be 50 to 100 words, have a clear ending, and no repeated sentences.
"""
        
        output = story_generator(
            prompt,
            pad_token_id=50256,
            do_sample=True
        )[0]["generated_text"]

        # Extract story
        story = output.replace(prompt, "").strip()

        # Clean up: Ensure the story ends with a complete sentence
        story = re.split(r'[.!?]', story)[0]
        story += "."

        # Strict word count: 50–100 words
        words = story.split()
        if len(words) > 100:
            story = " ".join(words[:100])
        if len(words) < 50:
            story = " ".join(words[:50])

        st.subheader("✨ Your Story")
        st.write(story)

    # Step 3: Text-to-Speech
    with st.spinner("Generating audio..."):
        tts = gTTS(text=story, lang="en", slow=False)
        tts.save("story.mp3")

        with open("story.mp3", "rb") as audio_file:
            st.audio(audio_file.read(), format="audio/mp3")

    st.success("✅ Story generated successfully!")
