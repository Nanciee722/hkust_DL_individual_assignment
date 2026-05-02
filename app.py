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
        result = captioner(image)
        caption = result[0]["generated_text"]
        st.info(f"Image caption: {caption}")

    # Step 2: Generate story 50-100 words
    with st.spinner("Writing story..."):
        # 温和易懂prompt，适合gpt2写长一点
        prompt = f"Once upon a time, {caption}. The bright sunny day brought joy to everyone around."

        outputs = story_generator(
            prompt,
            max_new_tokens=120,      # 给足长度，够50-100词
            temperature=0.6,         # 不乱编
            repetition_penalty=1.1,  # 防重复
            pad_token_id=50256,
            do_sample=True
        )
        full_text = outputs[0]["generated_text"]
        
        # 清理prompt前缀
        story = full_text.replace(prompt, "").strip()
        
        # 截取到最后一个完整标点，不砍太短
        for end_mark in [".", "!", "?"]:
            if end_mark in story:
                story = story[:story.rfind(end_mark) + 1]

        # 拼接成完整故事
        full_story = prompt + " " + story
        
        # 控制在50-100词
        words = full_story.split()
        if len(words) > 100:
            full_story = " ".join(words[:100])
            if "." in full_story:
                full_story = full_story[:full_story.rfind(".") + 1]

        st.subheader("Your Story ✨")
        st.write(full_story)

    # Step 3: Text to speech
    with st.spinner("Making voice..."):
        tts = gTTS(text=full_story, lang="en")
        tts.save("story.mp3")
        with open("story.mp3", "rb") as f:
            st.audio(f.read(), format="audio/mp3")

    st.success("Done! Hope you like the story 😊")
