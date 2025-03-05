import os
import torch
import streamlit as st
from PIL import Image
import numpy as np
from torchvision import transforms  # For preprocessing the image before inference
import requests

# Ensure the cache directory exists (optional, as Streamlit Cloud uses a temp directory for caching)
cache_dir = os.path.expanduser('~/.cache/torch/hub/')
os.makedirs(cache_dir, exist_ok=True)

# Set up environment variables and model paths
os.environ["STREAMLIT_SERVER_ENABLE_WATCHER"] = "false"  # Disable problematic watcher

# Mapping of crop to the corresponding model file path
crop_model_mapping = {
    "Paddy": "https://your-storage-location/classification_4Disease_best.pt",  # Replace with actual model URL
    "Cotton": "https://your-storage-location/re_do_cotton_2best.pt",  # Replace with actual model URL
    "Groundnut": "https://your-storage-location/groundnut_best.pt"  # Replace with actual model URL
}

# Define class labels for each crop
CLASS_LABELS = {
    "Paddy": ["brown_spot", "leaf_blast", "rice_hispa", "sheath_blight"],
    "Groundnut": ["alternaria_leaf_spot", "leaf_spot", "rosette", "rust"],
    "Cotton": ["bacterial_blight", "curl_virus", "herbicide_growth_damage",
               "leaf_hopper_jassids", "leaf_redding", "leaf_variegation"]
}

# Cache model loading to avoid reloading on every classification
@st.cache_resource
def load_model(crop_name):
    """Loads the YOLOv5 model only once per crop type."""
    try:
        # Handle special cases where capitalization might fail
        crop_name = crop_name.strip().capitalize()  # Capitalizes only the first letter

        # Handle special cases for crop names
        crop_name = {"Groundnut": "Groundnut", "Cotton": "Cotton", "Paddy": "Paddy"}.get(crop_name, crop_name)

        model_url = crop_model_mapping.get(crop_name, None)
        if model_url is None:
            raise ValueError(f"No model found for crop: {crop_name}")

        # Download the model to the temporary directory
        model_path = os.path.join("/tmp", f"{crop_name}_model.pt")
        if not os.path.exists(model_path):
            response = requests.get(model_url)
            with open(model_path, 'wb') as f:
                f.write(response.content)

        # Load the model using Ultralytics YOLOv5
        model = torch.hub.load('ultralytics/yolov5:v7.0', 'custom', path=model_path, force_reload=True, device='cpu')

        # Set the model to evaluation mode
        model.eval()
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None


# Preprocess image for model input (resize and normalize)
def preprocess_image(img):
    img = img.convert('RGB')  # Ensure the image is in RGB format
    preprocess = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize to 640x640 as required by YOLOv5
        transforms.ToTensor(),
    ])
    img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
    return img_tensor

# Perform classification and get predicted class and confidence
def classify_image(img, crop_name):
    model = load_model(crop_name)
    if model is None:
        return None, None

    # Preprocess the image
    img_tensor = preprocess_image(img)

    # Perform inference on the image
    with torch.no_grad():
        results = model(img_tensor)

    # Get results from the model
    output = results[0]  # This contains the raw output (class logits)
    
    # Get the class index with the highest confidence
    confidence, class_idx = torch.max(output, dim=0)
    
    # Map the class index to the corresponding label
    try:
        class_label = CLASS_LABELS[crop_name][class_idx.item()]
    except KeyError:
        st.error(f"Error: '{crop_name}' not found in class labels. Please check the crop name.")
        return None, None
    
    return class_label, confidence.item()

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

# Crop selection dropdown
crop_selection = st.selectbox("Select the crop", ["Paddy", "Cotton", "Groundnut"], label_visibility="hidden")
st.write(f"Selected Crop: {crop_selection}")

# Image upload
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Run classification when user clicks button
if uploaded_image:
    img = Image.open(uploaded_image).convert("RGB")
    st.image(img, caption="Uploaded Image.", use_container_width=True)

    if st.button("Run Classification"):
        with st.spinner("Running classification..."):  # Show loading spinner
            predicted_class, confidence = classify_image(img, crop_selection)
            if predicted_class is None:
                st.error("Classification failed. Check model logs.")
            else:
                st.subheader("Prediction Results")
                st.success(f"Prediction: {predicted_class} (Confidence: {confidence:.2f})")

                # Display precautions for the disease (example)
                precautions_dict = {
                    "brown_spot": ["Use resistant varieties", "Apply fungicides"],
                    "leaf_blast": ["Use resistant varieties", "Avoid excess nitrogen fertilization"],
                    "rice_hispa": ["Use insecticides", "Manual removal of larvae"],
                    "sheath_blight": ["Use fungicides", "Improve water management"],
                }

                if predicted_class in precautions_dict:
                    st.subheader("Precautions/Remedies:")
                    for item in precautions_dict[predicted_class]:
                        st.write(f"- {item}")
                else:
                    st.write("No precautions available.")
