# Program overview: A storytelling app for kids aged 3-10.
# Functions included: Uploading an image -> generating a story -> playing the audio

# Import
import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Main Functions
# image-to-text
def img2text(url):
    """Generate a caption from the uploaded image."""
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text

# text-to-story
def text2story(text):
    """Generate a kid-friendly short story based on the image caption."""
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

    # Prompt designed for safe, fun, kid-friendly stories
    prompt = (
        f"Once upon a time, there was {text}. "
        f"It was a bright sunny day and everyone was happy. "
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=150,
        num_beams=1,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    story_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # limit to 50-100 words
    words = story_text.split()
    if len(words) > 100:
        story_text = " ".join(words[:100])
        if "." in story_text:
            story_text = story_text[:story_text.rfind(".") + 1]
        else:
            story_text += "."

    return story_text

# text-to-audio
def text2audio(story_text):
    """Convert the story text into speech audio."""
    audio_pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    audio_data = audio_pipe(story_text)
    return audio_data

# Main part
st.set_page_config(page_title="Kids Story App", page_icon="https://img.icons8.com/?size=100&id=jfFu3i8zJXfN&format=png&color=000000")
st.markdown(
    '<img src="https://img.icons8.com/?size=100&id=LlgB5a8aAr0G&format=png&color=000000" width="60"> '
    '<span style="font-size:28px; font-weight:bold;">Image Storytelling for Kids</span>',
    unsafe_allow_html=True
)
st.write("Upload an image, and I will make a lovely story for you!")

uploaded_file = st.file_uploader("Select an Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    with open(uploaded_file.name, "wb") as file:
        file.write(bytes_data)

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Stage 1: Image to Text
    scenario = img2text(uploaded_file.name)
    st.write(f"**Scenario:** {scenario}")

    # Stage 2: Text to Story
    story = text2story(scenario)
    st.write(f"**Story:** {story}")

    # Stage 3: Story to Audio
    audio_data = text2audio(story)

    # Play button
    if st.button("Play Audio"):
        audio_array = audio_data["audio"]
        sample_rate = audio_data["sampling_rate"]
        st.audio(audio_array, sample_rate=sample_rate)

    # Success mark
    st.markdown("---")
    st.markdown("Done! Hope you like the story 😊")
