import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import pipeline

# Load models
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define mood labels
mood_labels = ["casual", "elegant", "bold", "retro", "edgy", "vintage", "chic", "grunge"]

st.set_page_config(page_title="FashionMood AI", layout="wide")
st.title("🧠 FashionMood AI - Detect the Mood of Your Outfit")

uploaded_image = st.file_uploader("📸 Upload an Outfit Photo", type=["png", "jpg", "jpeg"])
if uploaded_image:
    img = Image.open(uploaded_image).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Analyzing fashion mood..."):
        # Generate caption
        inputs = caption_processor(images=img, return_tensors="pt")
        out = caption_model.generate(**inputs)
        caption = caption_processor.decode(out[0], skip_special_tokens=True)

        # Mood classification
        result = classifier(caption, mood_labels)
        top_mood = result["labels"][0]
        confidence = result["scores"][0]

    st.subheader("📝 Detected Mood & Caption")
    st.markdown(f"**Mood:** {top_mood.capitalize()} ({confidence*100:.1f}% confidence)")
    st.markdown(f"**Caption:** {caption}")

    st.subheader("📌 Suggested Hashtags")
    hashtags = f"#{top_mood} #style #fashion #OOTD #{top_mood}Vibes"
    st.code(hashtags, language="markdown")
