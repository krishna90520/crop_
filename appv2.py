import os
import torch
import streamlit as st
from PIL import Image
import numpy as np

# Mapping of crop to the corresponding model file path
crop_model_mapping = {
    "Paddy": "classification_4Disease_best.pt",  # Replace with actual path to paddy model
    "cotton": "re_do_cotton_2best.pt",  # Replace with actual path to cotton model
    "ground nut": "groundnut_best.pt"  # Replace with actual path to ground nut model
}

# Labels for each crop
CLASS_LABELS = {
    "Paddy": ["brown_spot", "leaf_blast", "rice_hispa", "sheath_blight"],
    "GroundNut": ["alternaria_leaf_spot", "leaf_spot", "rosette", "rust"],
    "Cotton": ["bacterial_blight", "curl_virus", "herbicide_growth_damage",
               "leaf_hopper_jassids", "leaf_redding", "leaf_variegation"]
}

# Load classification model with absolute path
@st.cache_resource
def load_model(crop_name):
    try:
        model_path = crop_model_mapping.get(crop_name, None)
        if model_path is None:
            raise ValueError(f"No model found for crop: {crop_name}")

        # Load the classification model
        model = torch.load(model_path)  # Load the model using torch.load (since it's a classification model)
        model.eval()  # Set model to evaluation mode
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None


# Classification function
def classify_image(image, crop_name):
    model = load_model(crop_name)
    if model is None:
        return None

    # Preprocess the image for classification
    img_tensor = preprocess_image(image)

    with torch.no_grad():
        # Perform the classification
        outputs = model(img_tensor)  # Run the image through the model
        probabilities = torch.nn.Softmax(dim=1)(outputs)  # Convert logits to probabilities
        max_prob, predicted_class_idx = torch.max(probabilities, dim=1)

    predicted_class = CLASS_LABELS[crop_name][predicted_class_idx.item()]
    confidence = max_prob.item()

    return predicted_class, confidence


# Function to preprocess the image for classification
def preprocess_image(image):
    # Convert PIL image to a tensor and normalize it if required by your model
    # Assuming the model expects input size [1, 3, 224, 224] for RGB image (resizing and normalization may be different)
    transform = torch.nn.Sequential(
        torch.transforms.Resize((224, 224)),  # Resize to the expected input size
        torch.transforms.ToTensor(),
        torch.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return img_tensor


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
crop_selection = st.selectbox("Select the crop", ["Paddy", "cotton", "ground nut"], label_visibility="hidden")
st.write(f"Selected Crop: {crop_selection}")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    img = Image.open(uploaded_image).convert("RGB")
    st.image(img, caption="Uploaded Image.", use_container_width=True)

    if st.button("Run Classification"):
        with st.spinner("Running classification..."):  # Show loading spinner
            predicted_class, confidence = classify_image(img, crop_selection)

            if predicted_class is None:
                st.error("Classification failed. Check model logs.")
            else:
                st.subheader("Classification Result")
                st.success(f"Prediction: {predicted_class} (Confidence: {confidence:.2f})")

                # Display the precautionary measures for the disease
                precautions_dict = {
                    "brown_spot": ["Use resistant varieties", "Apply fungicides"],
                    "leaf_blast": ["Use resistant varieties", "Avoid excess nitrogen fertilization"],
                    "rice_hispa": ["Use insecticides", "Manual removal of larvae"],
                    "sheath_blight": ["Use fungicides", "Improve water management"],
                    # Add more diseases with precautions as needed
                }

                if predicted_class in precautions_dict:
                    st.subheader("Precautions/Remedies:")
                    for item in precautions_dict[predicted_class]:
                        st.write(f"- {item}")
                else:
                    st.write("No precautions available.")
