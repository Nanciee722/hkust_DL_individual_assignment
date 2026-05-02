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
    # 只改这里 → 换成 gpt2 原版模型
    story_generator = pipeline(
        "text-generation", 
        model="openai-community/gpt2"
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

    # Step 2: Generate story (fixed logic)
    with st.spinner("Writing story..."):
        # 更清晰的prompt，引导模型生成
        prompt = f"Write a short, happy story for kids based on: {caption}. Keep it simple, 50-100 words."
        outputs = story_generator(prompt, max_new_tokens=100, pad_token_id=50256)
        full_text = outputs[0]["generated_text"]
        
        # 提取生成的故事部分，避免prompt残留
        story = full_text.replace(prompt, "").strip()
        
        # 如果生成为空，用备用prompt重试
        if len(story) < 20:
            retry_prompt = f"Tell a simple story about this scene: {caption}."
            retry_output = story_generator(retry_prompt, max_new_tokens=100, pad_token_id=50256)
            story = retry_output[0]["generated_text"].replace(retry_prompt, "").strip()

        # 确保长度在50-100词之间
        words = story.split()
        if len(words) > 100:
            story = " ".join(words[:100])
        if len(words) < 30:
            story = f"Once upon a time, {caption}. The children laughed and played together. They made new friends and had a wonderful time. The sun shone brightly, and everyone felt happy. It was a perfect day full of joy and adventure."

        st.subheader("Your Story ✨")
        st.write(story)

    # Step 3: Text to speech
    with st.spinner("Making voice..."):
        tts = gTTS(text=story, lang="en")
        tts.save("story.mp3")

        with open("story.mp3", "rb") as f:
            st.audio(f.read(), format="audio/mp3")

    st.success("Done! Hope you like the story 😊")
