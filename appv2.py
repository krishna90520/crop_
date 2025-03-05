import os
import pathlib
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path

# Ensure compatibility with Windows paths
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Disable Streamlit watcher to prevent reload issues
os.environ["STREAMLIT_SERVER_ENABLE_WATCHER"] = "false"

# Define class labels for each crop type
CLASS_LABELS = {
    "Paddy": ["brown_spot", "leaf_blast", "rice_hispa", "sheath_blight"],
    "GroundNut": ["alternaria_leaf_spot", "leaf_spot", "rosette", "rust"],
    "Cotton": ["bacterial_blight", "curl_virus", "herbicide_growth_damage", "leaf_hopper_jassids", "leaf_redding", "leaf_variegation"]
}

# Define model paths for each crop (use Pathlib for OS compatibility)
MODEL_PATHS = {
    "Paddy": Path(r"D:\agmarks_deploy\classification_4Disease_best.pt"),
    "GroundNut": Path(r"D:\agmarks_deploy\groundnut_best.pt"),
    "Cotton": Path(r"D:\agmarks_deploy\re_do_cotton_2best.pt")
}

# Load the appropriate YOLOv5 classification model
@st.cache_resource
def load_model(crop_type):
    try:
        model_path = MODEL_PATHS.get(crop_type)
        if not model_path or not model_path.exists():
            st.error(f"Model not found: {model_path}")
            return None

        model = torch.hub.load("ultralytics/yolov5", "custom", path=str(model_path), force_reload=True)
        model.eval()  # Set to evaluation mode
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize for YOLOv5 classification input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Classification function
def classify_image(image, crop_type):
    model = load_model(crop_type)
    if model is None:
        return None, None

    image_tensor = preprocess_image(image)

    with torch.no_grad():
        results = model(image_tensor)  # Perform inference

    # Ensure results are formatted correctly
    if isinstance(results, torch.Tensor):
        results = results.squeeze(0)  # Remove batch dimension if necessary

    # Convert logits to probabilities
    probabilities = F.softmax(results, dim=0)  # YOLO may return single-dim tensor

    # Get the predicted class index
    predicted_idx = torch.argmax(probabilities).item()
    predicted_label = CLASS_LABELS[crop_type][predicted_idx]  # Map index to label

    return predicted_label, probabilities.tolist()

# Streamlit UI
st.markdown("""
    <style>
    .title { text-align: center; color: #4CAF50; font-size: 36px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Crop Disease Classification</div>', unsafe_allow_html=True)

st.markdown("### Select the Crop Type")
crop_selection = st.selectbox("Select the crop", ["Paddy", "GroundNut", "Cotton"], label_visibility="hidden")
st.write(f"Selected Crop: {crop_selection}")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    img = Image.open(uploaded_image).convert("RGB")
    st.image(img, caption="Uploaded Image.", use_container_width=True)

    if st.button("Classify Disease"):
        with st.spinner("Classifying..."):
            predicted_label, probabilities = classify_image(img, crop_selection)

            if predicted_label:
                st.success(f"Predicted Disease: {predicted_label}")
                st.write(f"Confidence Scores: {probabilities}")
            else:
                st.error("Error in classification.")

















