import os
import gdown
import cv2
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
DEFAULT_THRESHOLD = 0.5

st.set_page_config(
    page_title="Analyse IRM cérébrale",
    page_icon="🧠",
    layout="centered"
)

# =========================
# STYLE - inspiré de La Cité
# =========================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #eef4fb 0%, #f8fbff 100%);
    }

    .block-container {
        max-width: 960px;
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #17365d 0%, #244b78 100%);
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    .hero {
        background: linear-gradient(135deg, #17365d 0%, #244b78 70%, #ef7d57 100%);
        border-radius: 24px;
        padding: 28px 30px;
        color: white;
        box-shadow: 0 10px 24px rgba(23, 54, 93, 0.18);
        margin-bottom: 24px;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        line-height: 1.6;
        opacity: 0.96;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #17365d;
        margin-top: 0.5rem;
        margin-bottom: 0.8rem;
    }

    .image-card {
        background: white;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 22px rgba(17, 24, 39, 0.08);
        margin-top: 12px;
        margin-bottom: 18px;
    }

    .result-card {
        border-radius: 22px;
        padding: 24px;
        margin-top: 20px;
        margin-bottom: 18px;
        box-shadow: 0 10px 22px rgba(0,0,0,0.08);
    }

    .result-danger {
        background: linear-gradient(135deg, #fff1f2 0%, #ffe3e6 100%);
        border-left: 10px solid #ef7d57;
    }

    .result-success {
        background: linear-gradient(135deg, #ecfdf5 0%, #dcfce7 100%);
        border-left: 10px solid #10b981;
    }

    .result-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: #17365d;
        margin-bottom: 10px;
    }

    .result-text {
        font-size: 1.06rem;
        line-height: 1.7;
        color: #1f2937;
    }

    .note {
        margin-top: 10px;
        font-size: 0.94rem;
        color: #5b6470;
    }

    .metric-card {
        background: white;
        border-radius: 18px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
        border-top: 5px solid #17365d;
    }

    .metric-label {
        font-size: 0.95rem;
        color: #6b7280;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: #17365d;
    }

    .footer-note {
        text-align: center;
        color: #5b6470;
        font-size: 0.9rem;
        margin-top: 26px;
    }

    div[data-testid="stFileUploader"] > section {
        background: white;
        border-radius: 18px;
        border: 2px dashed #c6d4e5;
        padding: 8px;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="hero">
    <div class="hero-title">🧠 Analyse d’image IRM cérébrale</div>
    <div class="hero-subtitle">
        Outil d’aide automatisée à l’interprétation d’une image IRM cérébrale.
        Chargez une image pour obtenir un résultat simple, clair et visuel.
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander("📘 Guide d’utilisation"):
    st.markdown("""
###  Objectif
Analyser une image IRM cérébrale pour détecter la présence potentielle d’une tumeur.

###  Étapes
1. Importer une image IRM  
2. Attendre l’analyse automatique  
3. Lire le résultat affiché  

### Résultats
- ⚠️ Compatible avec une tumeur → attention requise  
- ✅ Pas de tumeur détectée → résultat rassurant  

###  Paramètre
Le seuil de décision permet d’ajuster la sensibilité du modèle.

### Important
Ce système est une aide à la décision et ne remplace pas un médecin.
""")
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
        "Cet outil est un prototype académique d’aide à la décision. "
        "Il ne remplace pas l’interprétation d’un professionnel de santé."
    )

# =========================
# MODEL
# =========================
if not os.path.exists(MODEL_PATH):
    with st.spinner("Téléchargement du modèle en cours..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

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
# UPLOAD
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
                Une validation par examen clinique et interprétation radiologique est recommandée.
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
                Ce résultat reste une aide automatisée et doit être interprété dans le contexte clinique global.
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
    '<div class="footer-note">Prototype académique — Application de classification binaire d’IRM cérébrale</div>',
    unsafe_allow_html=True
)
