import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Clothing Classifier",
    page_icon="👕",
    layout="centered"
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "dress",
    "hat",
    "longsleeve",
    "outwear",
    "pants",
    "shirt",
    "shoes",
    "shorts",
    "skirt",
    "t-shirt"
]


# ============================================================
# MODEL PATH
# ============================================================

# Automatically find the model in the same folder as app.py
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_clothing_cnn.keras"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found!\n\n"
            f"Expected location:\n{MODEL_PATH}\n\n"
            f"Please place 'final_clothing_cnn.keras' "
            f"in the same folder as app.py."
        )
        st.stop()

    model = tf.keras.models.load_model(MODEL_PATH)
    return model


model = load_model()


# ============================================================
# TITLE
# ============================================================

st.title("👕 Clothing Image Classifier")

st.write(
    "Upload an image of a clothing item and the CNN model "
    "will predict its category."
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("📋 Model Information"):

    st.write("**Model:** CNN-based Image Classifier")

    st.write("**Input Image Size:** 128 × 128 pixels")

    st.write("**Number of Classes:** 10")

    st.write("**Optimizer:** Adam")

    st.write("**Loss Function:** Sparse Categorical Crossentropy")

    st.write("**Activation:** ReLU + Softmax")

    st.write("**Dropout:** 0.5")

    st.write("**Normalization:** Pixel values divided by 255")


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Choose a clothing image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    # Display original image
    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Input Image",
        width=400
    )

    # --------------------------------------------------------
    # PREPROCESS IMAGE
    # --------------------------------------------------------

    # Resize to the same size used during training
    resized_image = image.resize((128, 128))

    # Convert image to NumPy array
    image_array = np.array(resized_image)

    # Normalize pixel values
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    predictions = model.predict(image_array, verbose=0)

    # Get probabilities
    probabilities = predictions[0]

    # Find predicted class
    predicted_index = np.argmax(probabilities)

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = probabilities[predicted_index] * 100


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.subheader("Prediction")

    st.success(
        f"Predicted Class: **{predicted_class}**"
    )

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )


    # ========================================================
    # TOP 3 PREDICTIONS
    # ========================================================

    st.subheader("Top 3 Predictions")

    top_3_indices = np.argsort(probabilities)[-3:][::-1]

    for index in top_3_indices:

        class_name = CLASS_NAMES[index]

        probability = probabilities[index] * 100

        st.write(
            f"**{class_name}** — {probability:.2f}%"
        )

        st.progress(
            float(probabilities[index])
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "CNN-based Clothing Image Classification | "
    "Streamlit Deployment"
)