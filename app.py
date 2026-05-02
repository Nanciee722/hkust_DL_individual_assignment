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
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    story_generator = pipeline(
        "text-generation", 
        model="openai-community/gpt2",
        temperature=0.5,
        repetition_penalty=1.2
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

    # Step 2: Generate story
    with st.spinner("Writing story..."):
        # 极简指令，GPT2 能听懂不跑偏
        prompt = f"Once upon a time, {caption}"
        
        outputs = story_generator(
            prompt,
            max_new_tokens=60,
            pad_token_id=50256,
            do_sample=True
        )
        full_text = outputs[0]["generated_text"]
        
        story = full_text.replace(prompt, "").strip()
        
        # 截断到第一个完整句号，杜绝无限重复
        if "." in story:
            story = story[:story.find(".") + 1]

        st.subheader("Your Story ✨")
        st.write(prompt + story)

    # Step 3: Text to speech
    with st.spinner("Making voice..."):
        tts = gTTS(text=prompt + story, lang="en")
        tts.save("story.mp3")

        with open("story.mp3", "rb") as f:
            st.audio(f.read(), format="audio/mp3")

    st.success("Done! Hope you like the story 😊")
