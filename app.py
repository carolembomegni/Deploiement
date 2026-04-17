import os
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# CONFIG
# =========================
MODEL_PATH = "modelefinal.keras"
MODEL_URL = "https://drive.google.com/uc?id=1k6SXisbC2qErIYncGZ3J6a9viJZjnvAy"
IMG_SIZE = (224, 224)
THRESHOLD = 0.3

st.set_page_config(page_title="Détection tumeur cérébrale", layout="centered")
st.title("Application de prédiction - ResNet")
st.write("Charge une image MRI pour obtenir une prédiction.")

# =========================
# TÉLÉCHARGEMENT DU MODÈLE
# =========================
if not os.path.exists(MODEL_PATH):
    with st.spinner("Téléchargement du modèle en cours..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# =========================
# CHARGEMENT DU MODÈLE
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# =========================
# PRÉTRAITEMENT IMAGE
# =========================
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =========================
# UPLOAD IMAGE
# =========================
uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image chargée", use_container_width=True)

    x = preprocess_image(image)
    pred = model.predict(x)

    st.subheader("Résultat")

    # Cas 1 : modèle binaire avec 1 sortie (sigmoid)
    if pred.shape[-1] == 1:
        prob = float(pred[0][0])
        classe = "tumor" if prob >= THRESHOLD else "no_tumor"
        confiance = prob if classe == "tumor" else 1 - prob

        st.write(f"**Classe prédite :** {classe}")
        st.write(f"**Confiance :** {confiance:.4f}")
        st.write(f"**Probabilité tumor :** {prob:.4f}")

    # Cas 2 : modèle avec 2 sorties (softmax)
    else:
        classes = ["no_tumor", "tumor"]
        probs = pred[0]
        idx = int(np.argmax(probs))

        st.write(f"**Classe prédite :** {classes[idx]}")
        st.write(f"**Confiance :** {float(probs[idx]):.4f}")
        st.write("**Probabilités par classe :**")
        for i, cls in enumerate(classes):
            st.write(f"- {cls}: {float(probs[i]):.4f}")