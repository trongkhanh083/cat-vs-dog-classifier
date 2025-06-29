import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="wide"
)

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

model = load_model("checkpoints/best_model_scratch.h5")
st.title("Cat vs Dog Classifier")
st.write("Upload an image and I'll tell you if it's a cat or dog.")

uploaded = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
if uploaded is not None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded image")
        x = np.array(img.resize((128,128))) / 255.0
        pred = model.predict(x[np.newaxis,...])[0][0]
        label = "Dog" if pred > 0.5 else "Cat"
        conf  = pred if pred>0.5 else 1-pred
        st.markdown(f"## Prediction: **{label}** ({conf*100:.1f}% confidence)")
