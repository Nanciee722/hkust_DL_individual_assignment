# ISOM5240 Individual Assignment
# Storytelling Application for 3-10 year-old kids
import streamlit as st
from transformers import pipeline
from gtts import gTTS
from PIL import Image

st.set_page_config(page_title="Kids Story App", page_icon="📖")
st.title("📖 Image Storytelling for Kids")
st.write("Upload an image, and I will make a lovely story for you!")

@st.cache_resource
def load_models():
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    story_generator = pipeline(
        "text-generation",
        model="openai-community/gpt2",
        repetition_penalty=1.2
    )
    return captioner, story_generator

captioner, story_generator = load_models()

img = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])

if img:
    image = Image.open(img)
    st.image(image, caption="Your picture", use_column_width=True)

    with st.spinner("Analyzing picture..."):
        result = captioner(image)
        caption = result[0]["generated_text"]
        st.info(f"Image caption: {caption}")

    with st.spinner("Writing story..."):
        # 固定最简单童话开头，GPT2 最听话
        prompt = f"Once upon a time, there were {caption}."

        outputs = story_generator(
            prompt,
            max_new_tokens=50,
            temperature=0.4,
            pad_token_id=50256,
            do_sample=True
        )

        full_text = outputs[0]["generated_text"]
        # 只保留到第一个句号，后面乱编的全部删掉
        if "." in full_text:
            story = full_text[:full_text.find(".") + 1]
        else:
            story = full_text

        st.subheader("Your Story ✨")
        st.write(story)

    with st.spinner("Making voice..."):
        tts = gTTS(text=story, lang="en")
        tts.save("story.mp3")
        with open("story.mp3", "rb") as f:
            st.audio(f.read(), format="audio/mp3")

    st.success("Done! Hope you like the story 😊")
