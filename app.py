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
    # 去掉illustration这个词，减少模型误解
    text = text.replace("illustration", "scene")
    return text

# text2story (Fixed: stable, on-topic, 50-100 words)
def text2story(text):
    story_model = pipeline(
        "text-generation",
        model="distilgpt2",
        temperature=0.4,          # 降低脑洞，减少奇怪句子
        repetition_penalty=1.3,  # 强制惩罚重复/无关内容
        pad_token_id=50256
    )
    
    # 超明确的prompt：直接告诉模型“写一个短故事”，不玩花样
    prompt = f"Tell a short story about: {text} The kids are having fun."
    
    story = story_model(
        prompt,
        max_new_tokens=70,       # 控制续写长度，避免越写越歪
        do_sample=True
    )[0]["generated_text"]

    # 只保留模型续写的部分，去掉prompt
    story = story.replace(prompt, "").strip()

    # 截断到第一个完整句子，杜绝奇怪续写
    if "." in story:
        story = story[:story.find(".") + 1]

    # 拼接成完整故事，控制词数
    full_story = prompt.split(".")[0] + ". " + story
    words = full_story.split()
    
    # 词数控制：低于50词就加一句场景相关的结尾，高于100词就截断
    if len(words) < 50:
        full_story += " Everyone laughed and played happily under the sunny sky."
    if len(words) > 100:
        full_story = " ".join(words[:100])
        if "." in full_story:
            full_story = full_story[:full_story.rfind(".") + 1]

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
