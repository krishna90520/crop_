

import os
os.environ["STREAMLIT_SERVER_ENABLE_WATCHER"] = "false"  # Disable problematic watcher
import cv2  # Ensure this import is at the top of your file
from ultralytics import YOLO

import streamlit as st
from PIL import Image
import torch
import numpy as np

# Load YOLOv5 model with absolute path
@st.cache_resource
def load_model():
    try:
        #model_path = os.path.abspath("yolov5best_aug_false.pt")
        model_path = os.path.abspath("agmarks_final.pt")
        # Load model using a direct path instead of torch.hub
        model = torch.hub.load(
            'ultralytics/yolov5',
            'custom',
            path=model_path,
            force_reload=True,
            trust_repo=True
        )
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None


# Detection function
def detect_objects(image, conf_threshold):
    model = load_model()
    if model is None:
        return None
    model.conf = conf_threshold
    results = model(image)
    return results

# Streamlit UI
st.markdown("""
    <style>
    .title { text-align: center; color: #4CAF50; font-size: 36px; }
    .red-label { color: red; font-size: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Crop Disease Detection</div>', unsafe_allow_html=True)
st.markdown('<style>.red-label {color: red; font-weight: bold;}</style>', unsafe_allow_html=True)
st.markdown('<div class="red-label">Diseases Trained on: <br> Brown spots <br> Rice\'s hispa <br> Sheath blight</div>', unsafe_allow_html=True)
st.markdown('<style>.red-label {color: green; font-weight: bold;}</style>', unsafe_allow_html=True)
st.markdown('<div class="red-label">Select the crop</div>', unsafe_allow_html=True)
crop_selection = st.selectbox("Select the crop", ["Paddy", "Wheat", "Maize"], label_visibility="hidden")
st.write(f"Selected Crop: {crop_selection}")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
conf_threshold = st.slider("Set Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# Precautions dictionary (fill this with your actual data)
precautions_dict = {
    "disease1": ["Precaution 1", "Precaution 2"],
    "disease2": ["Precaution A", "Precaution B"],
    # Add your actual disease precautions here
}

if uploaded_image:
    img = Image.open(uploaded_image).convert("RGB")
    st.image(img, caption="Uploaded Image.", use_container_width=True)

    if st.button("Run Detection"):
        with st.spinner("Running detection..."):  # Show loading spinner
            results = detect_objects(img, conf_threshold)
            if results is None:
                st.error("Detection failed. Check model logs.")
            else:
                st.subheader("Detection Results")
                inferenced_img = np.squeeze(results.render())
                st.image(inferenced_img, caption="Detected Objects", use_container_width=True)

                # Display predictions
                preds = results.pandas().xyxy[0]
                if not preds.empty:
                    max_conf_row = preds.loc[preds['confidence'].idxmax()]
                    st.success(f"Prediction: {max_conf_row['name']} (Confidence: {max_conf_row['confidence']:.2f})")

                    if max_conf_row['name'] in precautions_dict:
                        st.subheader("Precautions/Remedies:")
                        for item in precautions_dict[max_conf_row['name']]:
                            st.write(f"- {item}")
                    else:
                        st.write("No precautions available.")
                else:
                    st.warning("No objects detected.")
