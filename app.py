import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="AI Crop Disease Detection",
    layout="centered"
)

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("crop_disease_model.keras")

model = load_model()

# -----------------------------
# Disease classes
# -----------------------------
class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy"
]

# -----------------------------
# Recommendations
# -----------------------------
recommendations = {
    "Apple___Apple_scab": {
        "disease": "Apple Scab",
        "symptoms": "Olive or dark spots may appear on leaves and fruit.",
        "management": "Remove infected plant material and maintain good air circulation.",
        "prevention": "Keep the growing area clean and monitor plants regularly."
    },

    "Apple___Black_rot": {
        "disease": "Apple Black Rot",
        "symptoms": "Dark circular spots may appear on leaves and fruit.",
        "management": "Remove infected plant material and prune affected branches.",
        "prevention": "Maintain good orchard sanitation and remove dead or infected material."
    },

    "Apple___Cedar_apple_rust": {
        "disease": "Apple Cedar Rust",
        "symptoms": "Yellow-orange spots may appear on apple leaves.",
        "management": "Remove severely affected plant material and maintain orchard sanitation.",
        "prevention": "Monitor plants regularly and follow locally recommended disease-management practices."
    },

    "Apple___healthy": {
        "disease": "Healthy Apple Leaf",
        "symptoms": "No obvious disease symptoms detected.",
        "management": "Continue normal plant care and monitoring.",
        "prevention": "Maintain good sanitation and regularly inspect leaves."
    }
}

# -----------------------------
# Website title
# -----------------------------
st.title("🌱 AI Crop Disease Detection")
st.write("Upload an apple leaf image to detect possible disease.")

# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(
        img,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )

    if st.button("🔍 Detect Disease"):

        # Resize image
        img_resized = img.resize((128, 128))

        # Convert to array
        img_array = np.array(img_resized)

        # Normalize
        img_array = img_array / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array, verbose=0)

        predicted_index = np.argmax(prediction[0])
        predicted_class = class_names[predicted_index]

        confidence = prediction[0][predicted_index] * 100

        result = recommendations[predicted_class]

        # -----------------------------
        # Display result
        # -----------------------------
        st.success(f"Detected Disease: {result['disease']}")

        st.write(f"**Confidence:** {confidence:.2f}%")

        st.subheader("🔍 Symptoms")
        st.write(result["symptoms"])

        st.subheader("🛠️ Management")
        st.write(result["management"])

        st.subheader("🛡️ Prevention")
        st.write(result["prevention"])

        st.info(
            "Note: This AI result is for educational/project purposes. "
            "For actual crop treatment, consult a qualified agricultural expert "
            "and follow locally approved guidance."
        )