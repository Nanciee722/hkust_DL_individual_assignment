# ISOM5240 Individual Assignment
# Storytelling Application for 3-10 year-old kids
import streamlit as st
from transformers import pipeline
from gtts import gTTS
from PIL import Image

# --------------------------
# function part
# --------------------------

# img2text
def img2text(url):
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text

# text2story (适配所有图片、不跑题、50-100词儿童故事)
def text2story(text):
    story_model = pipeline(
        "text-generation",
        model="distilgpt2",
        temperature=0.5,
        repetition_penalty=1.2,
        pad_token_id=50256
    )
    
    # 通用、极简、不固定场景的prompt，只让模型围绕图片续写
    prompt = f"Once upon a time, {text}."
    
    story = story_model(
        prompt,
        max_new_tokens=90,
        do_sample=True
    )[0]["generated_text"]

    # 去掉prompt部分，避免重复
    story = story.replace(prompt, "").strip()

    # 只保留完整句子，截断在最后一个标点
    for punc in [".", "!", "?"]:
        if punc in story:
            story = story[:story.rfind(punc) + 1]

    # 拼接成完整故事，控制词数在50-100之间
    full_story = prompt + " " + story
    words = full_story.split()
    if len(words) > 100:
        full_story = " ".join(words[:100])
        if "." in full_story:
            full_story = full_story[:full_story.rfind(".") + 1]
    # 如果词数不足50，自动补一句不跑题的通用结尾
    if len(words) < 50:
        full_story = full_story + " It was a wonderful day full of fun and happiness."

    return full_story

# text2audio
def text2audio(story_text):
    tts = gTTS(text=story_text, lang="en")
    tts.save("story.mp3")
    
    with open("story.mp3", "rb") as f:
        audio_data = f.read()
    
    return audio_data

# --------------------------
# main part
# --------------------------
st.set_page_config(page_title="Kids Story App", page_icon="📖")
st.title("📖 Image Storytelling for Kids")
st.write("Upload any image to generate a fun story!")

uploaded_img = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])

if uploaded_img is not None:
    image = Image.open(uploaded_img)
    st.image(image, caption="Your Image", use_column_width=True)

    # Run functions
    with st.spinner("Generating caption..."):
        caption = img2text(image)
        st.info(f"Image Caption: {caption}")

    with st.spinner("Generating story..."):
        story = text2story(caption)
        st.subheader("Your Story ✨")
        st.write(story)

    with st.spinner("Generating audio..."):
        audio = text2audio(story)
        st.audio(audio, format="audio/mp3")

    st.success("All done! 🎉")
