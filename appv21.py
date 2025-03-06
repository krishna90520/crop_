# import os
# import torch
# import streamlit as st
# from PIL import Image
# import numpy as np
# from torchvision import transforms  # For preprocessing the image before inference
# import requests

# # GitHub Personal Access Token (replace with your own token)
# GITHUB_TOKEN = "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"  # Replace with your token

# # Ensure the cache directory exists
# cache_dir = os.path.expanduser('~/.cache/torch/hub/')
# os.makedirs(cache_dir, exist_ok=True)

# # Create the trusted_list file if missing
# trusted_list_path = os.path.join(cache_dir, "trusted_list")
# if not os.path.exists(trusted_list_path):
#     with open(trusted_list_path, 'w') as f:
#         f.write("[]")  # Empty JSON array for trusted list

# # Set environment variable to avoid cache-related issues
# os.environ["TORCH_HOME"] = "/tmp/torch_cache"  # Avoid the default cache directory for Torch

# # Set up environment variables and model paths
# os.environ["STREAMLIT_SERVER_ENABLE_WATCHER"] = "false"  # Disable problematic watcher

# # Mapping of crop to the corresponding model file path (GitHub raw URLs)
# crop_model_mapping = {
#     "Paddy": "https://github.com/krishna90520/crop_/raw/refs/heads/main/classification_4Disease_best.pt",
#     "Cotton": "https://github.com/krishna90520/crop_/raw/refs/heads/main/re_do_cotton_2best.pt",
#     "Groundnut": "https://github.com/krishna90520/crop_/raw/refs/heads/main/groundnut_best.pt"
# }

# # Define class labels for each crop
# CLASS_LABELS = {
#     "Paddy": ["brown_spot", "leaf_blast", "rice_hispa", "sheath_blight"],
#     "Groundnut": ["alternaria_leaf_spot", "leaf_spot", "rosette", "rust"],
#     "Cotton": ["bacterial_blight", "curl_virus", "herbicide_growth_damage",
#                "leaf_hopper_jassids", "leaf_redding", "leaf_variegation"]
# }

# # Modify model loading to use GitHub token for authentication
# def download_model_with_token(model_url, model_path):
#     headers = {'Authorization': f'token {GITHUB_TOKEN}'}
#     response = requests.get(model_url, headers=headers)
#     if response.status_code == 200:
#         with open(model_path, 'wb') as f:
#             f.write(response.content)
#     else:
#         st.error(f"Failed to download model: {model_url}. Status Code: {response.status_code}")
#         return None

# # Cache model loading to avoid reloading on every classification
# @st.cache_resource
# def load_model(crop_name):
#     """Loads the YOLOv5 model only once per crop type."""
#     try:
#         crop_name = crop_name.strip().capitalize()  # Capitalizes only the first letter

#         # Handle special cases for crop names
#         crop_name = {"Groundnut": "Groundnut", "Cotton": "Cotton", "Paddy": "Paddy"}.get(crop_name, crop_name)

#         model_url = crop_model_mapping.get(crop_name, None)
#         if model_url is None:
#             raise ValueError(f"No model found for crop: {crop_name}")

#         # Download the model to the temporary directory using the token
#         model_path = os.path.join("/tmp", f"{crop_name}_model.pt")
#         if not os.path.exists(model_path):
#             download_model_with_token(model_url, model_path)

#         # Load the model using Ultralytics YOLOv5
#         model = torch.hub.load('ultralytics/yolov5:v7.0', 'custom', path=model_path, force_reload=True, device='cpu')

#         # Set the model to evaluation mode
#         model.eval()
#         return model
#     except Exception as e:
#         st.error(f"Model loading failed: {str(e)}")
#         return None


# # Preprocess image for model input (resize and normalize)
# def preprocess_image(img):
#     img = img.convert('RGB')  # Ensure the image is in RGB format
#     preprocess = transforms.Compose([
#         transforms.Resize((640, 640)),  # Resize to 640x640 as required by YOLOv5
#         transforms.ToTensor(),
#     ])
#     img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
#     return img_tensor

# # Perform classification and get predicted class and confidence
# def classify_image(img, crop_name):
#     model = load_model(crop_name)
#     if model is None:
#         return None, None

#     # Preprocess the image
#     img_tensor = preprocess_image(img)

#     # Perform inference on the image
#     with torch.no_grad():
#         results = model(img_tensor)

#     # Get results from the model
#     output = results[0]  # This contains the raw output (class logits)
    
#     # Get the class index with the highest confidence
#     confidence, class_idx = torch.max(output, dim=0)
    
#     # Map the class index to the corresponding label
#     try:
#         class_label = CLASS_LABELS[crop_name][class_idx.item()]
#     except KeyError:
#         st.error(f"Error: '{crop_name}' not found in class labels. Please check the crop name.")
#         return None, None
    
#     return class_label, confidence.item()

# # Streamlit UI
# st.markdown("""
#     <style>
#     .title { text-align: center; color: #4CAF50; font-size: 36px; }
#     .red-label { color: red; font-size: 20px; }
#     </style>
# """, unsafe_allow_html=True)

# st.markdown('<div class="title">Crop Disease Detection</div>', unsafe_allow_html=True)
# st.markdown('<style>.red-label {color: red; font-weight: bold;}</style>', unsafe_allow_html=True)
# st.markdown('<style>.red-label {color: green; font-weight: bold;}</style>', unsafe_allow_html=True)
# st.markdown('<div class="red-label">Select the crop</div>', unsafe_allow_html=True)

# # Crop selection dropdown
# crop_selection = st.selectbox("Select the crop", ["Paddy", "Cotton", "Groundnut"], label_visibility="hidden")
# st.write(f"Selected Crop: {crop_selection}")

# # Image upload
# uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# # Run classification when user clicks button
# if uploaded_image:
#     img = Image.open(uploaded_image).convert("RGB")
#     st.image(img, caption="Uploaded Image.", use_container_width=True)

#     if st.button("Run Classification"):
#         with st.spinner("Running classification..."):  # Show loading spinner
#             predicted_class, confidence = classify_image(img, crop_selection)
#             if predicted_class is None:
#                 st.error("Classification failed. Check model logs.")
#             else:
#                 st.subheader("Prediction Results")
#                 st.success(f"Prediction: {predicted_class} (Confidence: {confidence:.2f})")

#                 # Display precautions for the disease (example)
#                 precautions_dict = {
#                     "brown_spot": ["Use resistant varieties", "Apply fungicides"],
#                     "leaf_blast": ["Use resistant varieties", "Avoid excess nitrogen fertilization"],
#                     "rice_hispa": ["Use insecticides", "Manual removal of larvae"],
#                     "sheath_blight": ["Use fungicides", "Improve water management"],
#                 }

#                 if predicted_class in precautions_dict:
#                     st.subheader("Precautions/Remedies:")
#                     for item in precautions_dict[predicted_class]:
#                         st.write(f"- {item}")
#                 else:
#                     st.write("No precautions available.")




import os
import torch
import streamlit as st
from PIL import Image
import numpy as np
from torchvision import transforms
import requests

# Confidence threshold for predictions
CONFIDENCE_THRESHOLD = 0.40 

# GitHub Token (Replace with your actual token)
GITHUB_TOKEN = "your_github_token_here"

# Mapping of crop to model file path (GitHub raw URLs)
crop_model_mapping = {
    "Paddy": "https://github.com/krishna90520/crop_/raw/refs/heads/main/classification_4Disease_best.pt",
    "Cotton": "https://github.com/krishna90520/crop_/raw/refs/heads/main/re_do_cotton_2best.pt",
    "Groundnut": "https://github.com/krishna90520/crop_/raw/refs/heads/main/groundnut_best.pt"
}

# Define class labels for each crop
CLASS_LABELS = {
    "Paddy": ["brown_spot", "leaf_blast", "rice_hispa", "sheath_blight"],
    "Groundnut": ["alternaria_leaf_spot", "leaf_spot", "rosette", "rust"],
    "Cotton": ["bacterial_blight", "curl_virus", "herbicide_growth_damage",
               "leaf_hopper_jassids", "leaf_redding", "leaf_variegation"]
}

# Function to download model with GitHub token
def download_model_with_token(model_url, model_path):
    headers = {'Authorization': f'token {GITHUB_TOKEN}'}
    response = requests.get(model_url, headers=headers)
    if response.status_code == 200:
        with open(model_path, 'wb') as f:
            f.write(response.content)
    else:
        st.error(f"Failed to download model: {model_url}. Status Code: {response.status_code}")
        return None

# Cache model loading to avoid reloading
@st.cache_resource
def load_model(crop_name):
    """Loads the YOLOv5 model once per crop type."""
    try:
        crop_name = crop_name.strip().capitalize()

        model_url = crop_model_mapping.get(crop_name)
        if not model_url:
            raise ValueError(f"No model found for crop: {crop_name}")

        model_path = os.path.join("/tmp", f"{crop_name}_model.pt")
        if not os.path.exists(model_path):
            download_model_with_token(model_url, model_path)

        # Load model
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True, device='cpu')
        model.eval()
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# Preprocess image
def preprocess_image(img):
    img = img.convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((640, 640)),  
        transforms.ToTensor(),
    ])
    img_tensor = preprocess(img).unsqueeze(0)  
    return img_tensor

# Perform classification
def classify_image(img, crop_name):
    model = load_model(crop_name)
    if model is None:
        return None, None

    img_tensor = preprocess_image(img)

    # Ensure input tensor type matches model weight type
    img_tensor = img_tensor.to(torch.float32)  # Convert to float32 for CPU inference

    with torch.no_grad():
        results = model(img_tensor)

    # Extract raw prediction scores
    output = results[0]  
    st.write(output, "this is the output")
    softmax = torch.nn.functional.softmax(output, dim=0)  # Apply softmax to get probabilities
    st.write(softmax , "softmax")
    confidence, class_idx = torch.max(softmax, dim=0)  # Get highest confidence prediction
    st.write(confidence, class_idx , "confidence, class_idx")

    # Ensure confidence score is properly extracted
    confidence = confidence.item()  # Convert tensor to float

    # Ensure index is within range
    try:
        class_label = CLASS_LABELS[crop_name][class_idx.item()]
    except IndexError:
        return None, confidence

    return class_label, confidence

# Streamlit UI
st.markdown("""
    <style>
    .title { text-align: center; color: #4CAF50; font-size: 36px; }
    .red-label { color: red; font-size: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Crop Disease Detection</div>', unsafe_allow_html=True)

crop_selection = st.selectbox("Select the crop", ["Paddy", "Cotton", "Groundnut"])
st.write(f"Selected Crop: {crop_selection}")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    img = Image.open(uploaded_image).convert("RGB")
    st.image(img, caption="Uploaded Image.", use_container_width=True)

    if st.button("Run Classification"):
        with st.spinner("Running classification..."):
            predicted_class, confidence = classify_image(img, crop_selection)
            st.write(predicted_class, confidence, "predicted_class, confidence")

            if predicted_class is None or confidence < CONFIDENCE_THRESHOLD:
                st.warning(f"No disease detected (Confidence: {confidence:.2f})")
            else:
                st.subheader("Prediction Results")
                st.success(f"Prediction: {predicted_class} (Confidence: {confidence:.2f})")

                # Display precautions
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

