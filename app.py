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
    # Keep your original image-to-text + teacher model
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    # Upgrade to gpt2-medium (much better for kids story)
    story_generator = pipeline(
        "text-generation", 
        model="gpt2-medium",
        temperature=0.6,
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
        result = captioner(image)
        caption = result[0]["generated_text"]
        st.info(f"Image caption: {caption}")

    # Step 2: Generate 50-100 words kids story
    with st.spinner("Writing story..."):
        # Simple clean prompt that model understands
        prompt = f"Once upon a time, {caption}."

        outputs = story_generator(
            prompt,
            max_new_tokens=130,
            pad_token_id=50256,
            do_sample=True
        )
        full_text = outputs[0]["generated_text"]

        # Cut off at the last complete sentence
        for punc in [".", "!", "?"]:
            if punc in full_text:
                full_text = full_text[:full_text.rfind(punc) + 1]

        # Control word count between 50-100
        words = full_text.split()
        if len(words) > 100:
            full_text = " ".join(words[:100])
        if len(words) < 50:
            pass

        st.subheader("Your Story ✨")
        st.write(full_text)

    # Step 3: Text to speech
    with st.spinner("Making voice..."):
        tts = gTTS(text=full_text, lang="en")
        tts.save("story.mp3")

        with open("story.mp3", "rb") as f:
            st.audio(f.read(), format="audio/mp3")

    st.success("Done! Hope you like the story 😊")
