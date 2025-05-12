import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load model
model = tf.keras.models.load_model("currency_detector_model.h5")

# Load label map
with open("label_map.json", "r") as f:
    label_map = json.load(f)
inv_map = {int(v): k for k, v in label_map.items()}

# UI
st.set_page_config(page_title="Indian Currency Detector", layout="centered")
st.title(" Indian Currency Detector")
st.write("Upload an image of an Indian currency note to predict its denomination.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Note", use_column_width=True)

        # Preprocess
        img = image.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        st.success(f"Predicted Denomination: ₹{inv_map[predicted_class]}")
        st.info(f"Confidence: {confidence*100:.2f}%")

    except Exception as e:
        st.error(f" Something went wrong: {e}")


# Note: The model and label map files should be in the same directory as this script.