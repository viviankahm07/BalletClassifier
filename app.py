"""
app.py
------
Streamlit demo app — upload an image, get a ballet position prediction.

Run:
    streamlit run demo/app.py
"""

import streamlit as st
import numpy as np
import cv2
import tempfile
import os
from PIL import Image

from src.extraction.pose_extractor import PoseExtractor
from src.preprocessing.normalizer import extract_all_features
from src.models.classifier import BalletClassifierBase

st.set_page_config(page_title="Ballet Pose Classifier", layout="centered")

st.title("🩰 Ballet Pose Classifier")
st.write("Upload a photo of a ballet position and the model will identify it.")

# Load model (cached so it only loads once)
@st.cache_resource
def load_model():
    model = BalletClassifierBase.load("models/saved/best_model_random_forest.pkl")
    classes = np.load("data/splits/label_classes.npy", allow_pickle=True)
    return model, classes

try:
    model, class_names = load_model()
    model_loaded = True
except Exception as e:
    st.warning(f"Model not loaded yet: {e}. Run training first.")
    model_loaded = False

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file and model_loaded:
    # Save to temp file for OpenCV
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    extractor = PoseExtractor()
    result = extractor.extract(tmp_path)

    if not result.success:
        st.error("Could not detect a human pose in this image. Try a clearer photo.")
    else:
        angles = extract_all_features(result.keypoints)
        proba = model.predict_proba([angles])[0]
        predicted = class_names[proba.argmax()]
        confidence = proba.max()

        with col2:
            st.subheader("Prediction")
            label_display = predicted.replace("_", " ").title()
            st.metric(label="Detected Position", value=label_display,
                      delta=f"{confidence:.0%} confidence")

            st.subheader("All Probabilities")
            sorted_probs = sorted(zip(class_names, proba), key=lambda x: -x[1])
            for cls, prob in sorted_probs:
                st.progress(float(prob), text=f"{cls.replace('_', ' ').title()}: {prob:.1%}")

    os.unlink(tmp_path)
