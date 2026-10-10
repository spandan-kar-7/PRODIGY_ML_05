import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "food_recognition_model.keras"
CLASSES_PATH = BASE_DIR / "food_classes.json"
IMG_SIZE = (128, 128)

CALORIE_INFO = {
    "pizza": {"calories": 285, "serving": "1 medium slice (roughly 100 g)"},
    "hamburger": {"calories": 295, "serving": "1 standard burger (roughly 100 g)"},
    "sushi": {"calories": 150, "serving": "roughly 100 g"},
    "ice_cream": {"calories": 207, "serving": "roughly 100 g"},
    "french_fries": {"calories": 312, "serving": "roughly 100 g"},
}

st.set_page_config(page_title="Food Recognition & Calorie Estimate", page_icon="🍽️")
st.title("🍽️ Food Image Recognition")
st.write("Upload an image to predict one of five food categories and view a rough calorie reference.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Place food_recognition_model.keras in the project folder.")
    st.stop()

if not CLASSES_PATH.exists():
    st.error("food_classes.json not found.")
    st.stop()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

uploaded_file = st.file_uploader("Choose a food image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    resized = image.resize(IMG_SIZE)
    image_array = np.asarray(resized, dtype=np.float32)
    image_batch = np.expand_dims(image_array, axis=0)

    model = load_model()
    probabilities = model.predict(image_batch, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_food = class_names[predicted_index]
    confidence = float(probabilities[predicted_index]) * 100
    pretty_food = predicted_food.replace("_", " ").title()

    st.subheader(f"Prediction: {pretty_food}")
    st.metric("Model confidence", f"{confidence:.1f}%")

    info = CALORIE_INFO.get(predicted_food)
    if info:
        st.metric("Approximate calorie reference", f"{info['calories']} kcal")
        st.caption(f"Reference serving: {info['serving']}.")

    st.warning(
        "This is an estimate based on the predicted category, not a measurement of the image. "
        "Actual calories vary with portion size, ingredients, and preparation. "
        "The model may also misclassify an image."
    )
