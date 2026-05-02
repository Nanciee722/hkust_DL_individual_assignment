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

# text2story (BETTER MODEL: gpt2-medium, 50-100 words)
def text2story(text):
    story_model = pipeline(
        "text-generation",
        model="gpt2-medium",
        temperature=0.6,
        repetition_penalty=1.1,
        pad_token_id=50256
    )
    
    prompt = f"Once upon a time, {text}."
    
    story = story_model(
        prompt,
        max_new_tokens=120,
        do_sample=True
    )[0]["generated_text"]

    # Clean story to end at a complete sentence
    for punc in [".", "!", "?"]:
        if punc in story:
            story = story[:story.rfind(punc) + 1]

    # Keep 50-100 words
    words = story.split()
    if len(words) > 100:
        story = " ".join(words[:100])
        if "." in story:
            story = story[:story.rfind(".") + 1]
    
    return story

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
st.write("Upload an image to generate a fun story!")

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
