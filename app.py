import os
import urllib.request
import cv2
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# CONFIG
# =========================
MODEL_PATH = "modelefinal.keras"
MODEL_URL = "https://github.com/carolembomegni/Deploiement/releases/download/v1.0/modelefinal.keras?raw=1"
IMG_SIZE = (224, 224)
DEFAULT_THRESHOLD = 0.5

st.set_page_config(
    page_title="Analyse IRM cérébrale",
    page_icon="🧠",
    layout="centered"
)

# =========================
# STYLE CSS
# =========================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #f4fbf6 0%, #edf7ef 100%);
    }

    .block-container {
        max-width: 980px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1f6f43 0%, #2d8a57 100%);
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    .top-banner {
        background: linear-gradient(135deg, #1f6f43 0%, #3d9b63 65%, #8fca9f 100%);
        border-radius: 26px;
        padding: 26px 30px;
        color: white;
        box-shadow: 0 10px 24px rgba(31, 111, 67, 0.18);
        margin-bottom: 22px;
    }

    .top-banner-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }

    .top-banner-subtitle {
        font-size: 1.05rem;
        line-height: 1.6;
        opacity: 0.97;
    }

    .deploy-banner {
        background: white;
        border: 2px solid #d7eadc;
        border-radius: 22px;
        padding: 18px 22px;
        margin-bottom: 20px;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.05);
    }

    .deploy-banner-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: #1f6f43;
        margin-bottom: 0.2rem;
    }

    .deploy-banner-text {
        font-size: 1rem;
        color: #355e43;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1f6f43;
        margin-top: 0.5rem;
        margin-bottom: 0.8rem;
    }

    .image-card {
        background: white;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.06);
        margin-top: 10px;
        margin-bottom: 18px;
        border: 1px solid #e4f0e7;
    }

    .result-card {
        border-radius: 22px;
        padding: 24px;
        margin-top: 18px;
        margin-bottom: 18px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    }

    .result-danger {
        background: linear-gradient(135deg, #fff3f2 0%, #ffe5e1 100%);
        border-left: 10px solid #d95c4f;
    }

    .result-success {
        background: linear-gradient(135deg, #effcf3 0%, #ddf6e4 100%);
        border-left: 10px solid #2f8f57;
    }

    .result-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: #1e2b20;
        margin-bottom: 10px;
    }

    .result-text {
        font-size: 1.06rem;
        line-height: 1.7;
        color: #2c3a2e;
    }

    .note {
        margin-top: 10px;
        font-size: 0.94rem;
        color: #617066;
    }

    .metric-card {
        background: white;
        border-radius: 18px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 5px 14px rgba(0,0,0,0.05);
        border-top: 5px solid #2f8f57;
    }

    .metric-label {
        font-size: 0.95rem;
        color: #6a7a6d;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: #1f6f43;
    }

    .footer-note {
        text-align: center;
        color: #66756a;
        font-size: 0.92rem;
        margin-top: 28px;
    }

    div[data-testid="stFileUploader"] > section {
        background: white;
        border-radius: 18px;
        border: 2px dashed #b8d9c0;
        padding: 8px;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="top-banner">
    <div class="top-banner-title">🧠 Analyse d’image IRM cérébrale</div>
    <div class="top-banner-subtitle">
        Outil d’aide automatisée à l’interprétation d’une image IRM cérébrale.
        Chargez une image pour obtenir un résultat simple, lisible et visuel.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="deploy-banner">
    <div class="deploy-banner-title">✅ Projet déployé</div>
    <div class="deploy-banner-text">
        Projet déployé par les étudiants du Collège La Cité.
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Paramètres")
    threshold = st.slider(
        "Seuil de décision",
        min_value=0.10,
        max_value=0.90,
        value=DEFAULT_THRESHOLD,
        step=0.05
    )
    show_details = st.checkbox("Afficher les détails techniques", value=False)
    st.markdown("---")
    st.info(
        "Cet outil est une aide académique à la décision. "
        "Il ne remplace pas l’interprétation d’un professionnel de santé."
    )

# =========================
# MODEL DOWNLOAD
# =========================
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Téléchargement du modèle en cours..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

download_model()

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False,
        custom_objects={
            "preprocess_input": tf.keras.applications.resnet50.preprocess_input
        }
    )

model = load_model()

# =========================
# PREPROCESSING
# =========================
def apply_clahe_bgr(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge((l_clahe, a, b))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return img_clahe

def preprocess_image(image):
    img_rgb = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_clahe_bgr = apply_clahe_bgr(img_bgr)
    img_clahe_rgb = cv2.cvtColor(img_clahe_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_clahe_rgb, IMG_SIZE)
    img_array = img_resized.astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =========================
# USER GUIDE
# =========================
with st.expander("📘 Guide d’utilisation"):
    st.markdown("""
**Étapes d’utilisation :**
1. Chargez une image IRM cérébrale au format JPG, JPEG ou PNG.
2. Attendez quelques secondes pendant l’analyse.
3. Consultez le résultat affiché par l’application.

**Résultats possibles :**
- **Image potentiellement compatible avec une tumeur**
- **Aucune tumeur détectée par le modèle**

**Important :**  
Ce système est une aide automatique à la décision.  
Le résultat doit toujours être interprété dans un contexte clinique global.
""")

# =========================
# UPLOADER
# =========================
st.markdown('<div class="section-title">Image à analyser</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choisir une image IRM",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.markdown('<div class="image-card">', unsafe_allow_html=True)
    st.image(image, caption="Image IRM chargée", width=380)
    st.markdown('</div>', unsafe_allow_html=True)

    x = preprocess_image(image)

    with st.spinner("Analyse en cours..."):
        pred = model.predict(x, verbose=0)

    prob = float(pred[0][0])
    predicted_class = "tumor" if prob >= threshold else "no_tumor"
    confidence = prob if predicted_class == "tumor" else 1 - prob

    st.markdown('<div class="section-title">Résultat de l’analyse</div>', unsafe_allow_html=True)

    if predicted_class == "tumor":
        st.markdown(f"""
        <div class="result-card result-danger">
            <div class="result-title">⚠️ Image potentiellement compatible avec une tumeur</div>
            <div class="result-text">
                Le modèle a détecté des caractéristiques pouvant être associées à une tumeur sur cette image.
            </div>
            <div class="note">
                Une validation par interprétation clinique et radiologique est recommandée.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card result-success">
            <div class="result-title">✅ Aucune tumeur détectée par le modèle</div>
            <div class="result-text">
                Le modèle n’a pas identifié d’élément évocateur de tumeur sur cette image.
            </div>
            <div class="note">
                Ce résultat constitue une aide automatisée et doit être confirmé dans le contexte clinique.
            </div>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Niveau de confiance</div>
            <div class="metric-value">{confidence:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Probabilité estimée de tumeur</div>
            <div class="metric-value">{prob:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.progress(max(0.0, min(prob, 1.0)))

    if show_details:
        with st.expander("Détails techniques"):
            st.write(f"Classe interne prédite : `{predicted_class}`")
            st.write(f"Valeur brute : `{pred}`")
            st.write(f"Seuil utilisé : `{threshold}`")

st.markdown(
    '<div class="footer-note">Application académique de classification binaire d’IRM cérébrale</div>',
    unsafe_allow_html=True
)
   

  


 
  




   
    

 
