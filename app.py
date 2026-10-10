import json
import os
from pathlib import Path

import gdown
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


# ==========================================
# 1. APPLICATION CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Food Recognition & Calorie Estimation",
    page_icon="🍽️",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "food_recognition_model.keras"
CLASSES_PATH = BASE_DIR / "food_classes.json"

# Google Drive file ID
GOOGLE_DRIVE_FILE_ID = "1N0MGnh_uei5_sUAlESAaf5TvueEA1Lex"

# Model input dimensions
IMG_SIZE = (128, 128)


# ==========================================
# 2. FOOD CALORIE REFERENCE
# ==========================================

CALORIE_INFO = {
    "pizza": {
        "calories": 285,
        "serving": "1 medium slice (approximately 100 g)"
    },
    "hamburger": {
        "calories": 295,
        "serving": "1 standard burger (approximately 100 g)"
    },
    "sushi": {
        "calories": 150,
        "serving": "approximately 100 g"
    },
    "ice_cream": {
        "calories": 207,
        "serving": "approximately 100 g"
    },
    "french_fries": {
        "calories": 312,
        "serving": "approximately 100 g"
    }
}


# ==========================================
# 3. DOWNLOAD TRAINED MODEL
# ==========================================

def download_model():
    """Download the trained model from Google Drive if necessary."""

    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return True

    try:
        with st.spinner("Downloading the trained model..."):

            downloaded_file = gdown.download(
                id=GOOGLE_DRIVE_FILE_ID,
                output=str(MODEL_PATH),
                quiet=False,
            )

        if (
            downloaded_file
            and MODEL_PATH.exists()
            and MODEL_PATH.stat().st_size > 0
        ):
            return True

        if MODEL_PATH.exists():
            MODEL_PATH.unlink()

        st.error(
            "The model could not be downloaded. "
            "Check the Google Drive sharing settings and file availability."
        )
        return False

    except Exception as error:
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()

        st.error(f"Model download failed: {error}")
        return False


# ==========================================
# 4. LOAD MODEL AND CLASS NAMES
# ==========================================

@st.cache_resource
def load_model():
    """Load the trained TensorFlow model."""
    return tf.keras.models.load_model(MODEL_PATH)


def load_class_names():
    """Read class names from the JSON file."""

    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            "food_classes.json was not found in the repository."
        )

    with open(CLASSES_PATH, "r", encoding="utf-8") as file:
        names = json.load(file)

    if not isinstance(names, list) or not names:
        raise ValueError("The class names file is invalid.")

    return names


# ==========================================
# 5. PREDICT FOOD CATEGORY
# ==========================================

def predict_food(image, model, class_names):
    """Predict a food category from a PIL image."""

    resized_image = image.resize(IMG_SIZE)

    image_array = np.asarray(
        resized_image,
        dtype=np.float32
    )

    image_batch = np.expand_dims(image_array, axis=0)

    predictions = model.predict(
        image_batch,
        verbose=0
    )[0]

    if len(predictions) != len(class_names):
        raise ValueError(
            "The number of model output classes does not match "
            "the number of class names."
        )

    predicted_index = int(np.argmax(predictions))

    predicted_food = class_names[predicted_index]

    confidence = float(predictions[predicted_index]) * 100

    return predicted_food, confidence


# ==========================================
# 6. MAIN USER INTERFACE
# ==========================================

st.title("🍽️ Food Image Recognition")

st.markdown(
    """
    Upload a food image to identify its category and view an
    approximate calorie reference.

    **Supported categories:** Pizza, Hamburger, Sushi,
    Ice Cream and French Fries.
    """
)

st.divider()

uploaded_file = st.file_uploader(
    "📸 Upload a food image",
    type=["jpg", "jpeg", "png"],
    help="Choose a JPG, JPEG or PNG image."
)


# ==========================================
# 7. RUN PREDICTION
# ==========================================

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded Image")

        st.image(
            image,
            caption="Your selected food image",
            use_container_width=True
        )

        if st.button("🔍 Recognize Food", type="primary"):

            if not download_model():
                st.stop()

            try:
                class_names = load_class_names()
                model = load_model()

                with st.spinner("Analyzing your image..."):

                    predicted_food, confidence = predict_food(
                        image,
                        model,
                        class_names
                    )

                display_name = predicted_food.replace(
                    "_", " "
                ).title()

                st.divider()

                st.subheader("🎯 Prediction Results")

                st.success(f"Predicted Food: {display_name}")

                st.metric(
                    label="Model Confidence",
                    value=f"{confidence:.2f}%"
                )

                st.progress(
                    min(max(confidence / 100, 0.0), 1.0)
                )

                calorie_details = CALORIE_INFO.get(
                    predicted_food
                )

                if calorie_details:

                    st.subheader("🔥 Calorie Reference")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            label="Estimated Calories",
                            value=f"{calorie_details['calories']} kcal"
                        )

                    with col2:
                        st.metric(
                            label="Reference Serving",
                            value=calorie_details["serving"]
                        )

                st.info(
                    "Calorie values are approximate reference values, "
                    "not measurements derived from the image. Actual "
                    "calories depend on portion size, ingredients and "
                    "preparation. The model may also misclassify images."
                )

            except Exception as error:
                st.error(
                    f"Prediction failed: {error}"
                )

    except (UnidentifiedImageError, OSError):
        st.error(
            "Unable to read this image. Please upload a valid JPG, "
            "JPEG or PNG file."
        )

    except Exception as error:
        st.error(f"An unexpected error occurred: {error}")


# ==========================================
# 8. FOOTER
# ==========================================

st.divider()

st.caption(
    "Food Image Recognition & Calorie Estimation | "
    "Built with Python, TensorFlow and Streamlit"
)
