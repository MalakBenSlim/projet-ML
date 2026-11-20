"""
================================================================================
INTERFACE INTERACTIVE DE CLASSIFICATION DU DUBAS DU PALMIER DATTIER
Projet Machine Learning & Vision par Ordinateur
================================================================================
Application Web développée avec Streamlit permettant de :
1. Téléverser une photo de foliole de palmier dattier ou tester des exemples.
2. Visualiser le processus d'extraction de descripteurs (Original, HOG, LBP, HSV).
3. Obtenir en temps réel la prédiction de la maladie et le score de confiance.
4. Consulter les probabilités détaillées pour chacune des 4 classes.
5. Obtenir un diagnostic biologique et des recommandations agronomiques de traitement.
================================================================================
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import joblib

from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray, rgb2hsv

# ------------------------------------------------------------------------------
# 1. CONFIGURATION DE LA PAGE STREAMLIT
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Détection du Dubas - Palmier Dattier",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1b5e20;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4b6b53;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #f1f8e9;
        border-left: 5px solid #2e7d32;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .recommendation-box {
        background-color: #fffde7;
        border-left: 5px solid #fbc02d;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. CHARGEMENT DES RESSOURCES & MODÈLES
# ------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATASET_PATH = BASE_DIR / "Image dataset of infected date palm leaves" / "Image dataset of infected date palm leaves by dubas insects" / "Palm Leaves Dataset"

@st.cache_resource
def load_ml_pipeline():
    """Charge en cache les modèles et transformateurs entraînés."""
    try:
        scaler = joblib.load(MODELS_DIR / "standard_scaler.joblib")
        selector = joblib.load(MODELS_DIR / "feature_selector.joblib")
        le = joblib.load(MODELS_DIR / "label_encoder.joblib")
        rf_model = joblib.load(MODELS_DIR / "random_optimal_model.joblib")
        svm_model = joblib.load(MODELS_DIR / "svm_optimal_model.joblib")
        knn_model = joblib.load(MODELS_DIR / "k-nearest_optimal_model.joblib")
        return {
            'scaler': scaler,
            'selector': selector,
            'le': le,
            'Random Forest': rf_model,
            'SVM (Noyau RBF)': svm_model,
            'K-Nearest Neighbors': knn_model
        }
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles : {e}")
        return None

pipeline = load_ml_pipeline()

# Diagnostic biologique et recommandations agronomiques
DIAGNOSIS_DATA = {
    'Healthy (Saine)': {
        'badge': '🟢 EXCELLENT ÉTAT SANITAIRE',
        'color': '#2e7d32',
        'icon': '✅',
        'summary': "Feuille saine et vigoureuse, tissu végétal homogène, absence totale de parasite.",
        'actions': [
            "Poursuivre la surveillance périodique dans la palmeraie.",
            "Maintenir une irrigation oasienne et une fertilisation équilibrées.",
            "Aucun traitement phytosanitaire requis."
        ]
    },
    'Bug (Insectes seuls)': {
        'badge': '🟡 PRÉSENCE DE NOUVELLE COLONIE DE DUBAS',
        'color': '#f57f17',
        'icon': '🐛',
        'summary': "Présence active de nymphes ou adultes de dubas (Ommatissus lybicus) suçant la sève foliaire.",
        'actions': [
            "Installer des pièges jaunes englués pour monitorer la population d'adultes.",
            "Favoriser la faune auxiliaire prédatrice (coccinelles, chrysopes).",
            "En cas de forte prolifération, envisager un traitement ciblé avec un bio-insecticide."
        ]
    },
    'Honey (Miellat)': {
        'badge': '🟠 DÉPÔTS DE MIELLAT COLLANT',
        'color': '#e65100',
        'icon': '🍯',
        'summary': "Dépôts liquides sucrés et collants exsudés par les insectes. Risque imminent de développement de la fumagine.",
        'actions': [
            "Effectuer un lavage sous pression à l'eau ou savon noir pour dissoudre les résidus de miellat.",
            "Aérer les couronnes de palmiers pour réduire l'humidité favorable aux champignons.",
            "Inspecter les palmes avoisinantes pour localiser les foyers d'insectes actifs."
        ]
    },
    'Mixed (Infestation mixte)': {
        'badge': '🔴 INFESTATION CRITIQUE SÉVÈRE',
        'color': '#c62828',
        'icon': '⚠️',
        'summary': "Stade d'attaque le plus sévère : présence conjointe de nombreuses colonies d'insectes et d'une nappe épaisse de miellat et fumagine bloquant la photosynthèse.",
        'actions': [
            "Intervention curative urgente requise pour préserver la vigueur du palmier et la récolte de dattes.",
            "Nettoyage foliaire complet suivi d'une pulvérisation phytosanitaire homologuée.",
            "Isoler et traiter l'ensemble des arbres du périmètre oasien contaminé."
        ]
    }
}


# ------------------------------------------------------------------------------
# 3. EXTRACTION EN TEMPS RÉEL SUR IMAGE
# ------------------------------------------------------------------------------
def process_and_extract(image: Image.Image, target_size=(128, 128)):
    """Traite l'image et extrait les descripteurs HOG, LBP, HSV."""
    img_resized = image.convert('RGB').resize(target_size)
    arr = np.array(img_resized)
    
    gray = rgb2gray(arr)
    gray_uint8 = (gray * 255).astype(np.uint8)
    hsv = rgb2hsv(arr)
    
    # 1. HOG avec image visuelle
    hf, hog_img = hog(
        gray,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=True,
        feature_vector=True
    )
    
    # 2. LBP
    lbp = local_binary_pattern(gray_uint8, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    
    # 3. HSV
    h_hist, _ = np.histogram(hsv[:, :, 0], bins=16, range=(0, 1), density=True)
    s_hist, _ = np.histogram(hsv[:, :, 1], bins=16, range=(0, 1), density=True)
    v_hist, _ = np.histogram(hsv[:, :, 2], bins=16, range=(0, 1), density=True)
    
    # 4. Moments statistiques
    rgb_norm = arr / 255.0
    color_moments = [
        float(np.mean(rgb_norm[:, :, 0])), float(np.std(rgb_norm[:, :, 0])),
        float(np.mean(rgb_norm[:, :, 1])), float(np.std(rgb_norm[:, :, 1])),
        float(np.mean(rgb_norm[:, :, 2])), float(np.std(rgb_norm[:, :, 2])),
        float(np.mean(hsv[:, :, 1])), float(np.std(hsv[:, :, 1])),
        float(np.mean(hsv[:, :, 0])), float(np.std(hsv[:, :, 0]))
    ]
    
    feat_vector = np.hstack([hf, lbp_hist, h_hist, s_hist, v_hist, color_moments]).reshape(1, -1)
    
    visuals = {
        'resized': img_resized,
        'gray': (gray * 255).astype(np.uint8),
        'hog_img': (hog_img / np.max(hog_img) * 255).astype(np.uint8),
        'lbp_img': ((lbp / lbp.max()) * 255).astype(np.uint8),
        'saturation_mask': (hsv[:, :, 1] * 255).astype(np.uint8)
    }
    
    return feat_vector, visuals


# ------------------------------------------------------------------------------
# 4. BARRE LATÉRALE (SIDEBAR)
# ------------------------------------------------------------------------------
st.sidebar.image("generale.jpg" if Path("generale.jpg").exists() else "🌴", use_container_width=True)
st.sidebar.markdown("### ⚙️ Paramètres d'Analyse")

selected_model_name = st.sidebar.selectbox(
    "🤖 Choisir le modèle :",
    ["Random Forest", "SVM (Noyau RBF)", "K-Nearest Neighbors"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔬 À Propos du Système")
st.sidebar.info("""
**Système de Vision par Ordinateur & Machine Learning**  
Classification automatisée des folioles de palmier dattier (*Phoenix dactylifera*).  
Modèles supervisés entraînés sur des descripteurs HOG, LBP et HSV.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 Échantillons de Test")
sample_choice = st.sidebar.radio(
    "Tester directement un cas type :",
    ["(Aucun - Charger mon image)", "Exemple Saine (Healthy)", "Exemple Insectes (Bug)", "Exemple Miellat (Honey)", "Exemple Mixte (Mixed)"]
)


# ------------------------------------------------------------------------------
# 5. CONTENU PRINCIPAL
# ------------------------------------------------------------------------------
st.markdown('<p class="main-title">🌴 Détection du Dubas du Palmier Dattier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Système automatisé de vision par ordinateur pour le diagnostic phytosanitaire des palmeraies</p>', unsafe_allow_html=True)

col_input, col_pred = st.columns([1, 1.2], gap="large")

image_to_analyze = None
image_origin = ""

with col_input:
    st.markdown("### 📸 1. Entrée de l'Image")
    uploaded_file = st.file_uploader(
        "Déposer une photo de foliole de palmier (JPG, PNG) :",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        image_to_analyze = Image.open(uploaded_file)
        image_origin = f"Image importée : {uploaded_file.name}"
    elif sample_choice != "(Aucun - Charger mon image)":
        folder_map = {
            "Exemple Saine (Healthy)": "Healthy",
            "Exemple Insectes (Bug)": "Bug",
            "Exemple Miellat (Honey)": "Honey",
            "Exemple Mixte (Mixed)": "Dubas"
        }
        target_folder = DATASET_PATH / folder_map[sample_choice]
        if target_folder.exists():
            sample_files = list(target_folder.glob("*.jpg")) + list(target_folder.glob("*.JPG"))
            if sample_files:
                image_to_analyze = Image.open(sample_files[0])
                image_origin = f"Cas type chargé : {sample_choice} ({sample_files[0].name})"
                
    if image_to_analyze is not None:
        st.image(image_to_analyze, caption=image_origin, use_container_width=True)
    else:
        st.info("👆 Déposez une photo foliaire ci-dessus ou choisissez un exemple dans le menu latéral pour lancer le diagnostic.")


# ------------------------------------------------------------------------------
# 6. ANALYSE ET PRÉDICTIONS
# ------------------------------------------------------------------------------
if image_to_analyze is not None and pipeline is not None:
    feat_vector, visuals = process_and_extract(image_to_analyze)
    
    # Prétraitement et sélection
    scaled_feats = pipeline['scaler'].transform(feat_vector)
    selected_feats = pipeline['selector'].transform(scaled_feats)
    
    # Inférence avec le modèle sélectionné
    model = pipeline[selected_model_name]
    pred_idx = model.predict(selected_feats)[0]
    predicted_label = pipeline['le'].inverse_transform([pred_idx])[0]
    
    # Probabilités
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(selected_feats)[0]
    else:
        probas = [0.25, 0.25, 0.25, 0.25]
        
    class_names = pipeline['le'].classes_
    proba_dict = dict(zip(class_names, probas))
    confidence = proba_dict[predicted_label] * 100
    
    diag_info = DIAGNOSIS_DATA.get(predicted_label, {})
    
    with col_pred:
        st.markdown("### 🎯 2. Diagnostic du Modèle")
        
        # Badge de prédiction
        st.markdown(f"""
        <div style="background-color: {diag_info.get('color', '#333')}15; 
                    border: 2px solid {diag_info.get('color', '#333')}; 
                    border-radius: 10px; padding: 18px; text-align: center; margin-bottom: 20px;">
            <span style="font-size: 2.2rem;">{diag_info.get('icon', '🔍')}</span>
            <h2 style="color: {diag_info.get('color', '#333')}; margin: 5px 0 0 0;">
                {predicted_label}
            </h2>
            <p style="font-weight: 700; color: #555; margin-top: 5px; font-size: 1.05rem;">
                Indice de confiance : <span style="color: {diag_info.get('color', '#333')};">{confidence:.1f} %</span> (Modèle : {selected_model_name})
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Répartition des probabilités
        st.markdown("#### 📊 Probabilités Estimées par Classe :")
        for cls, prob in proba_dict.items():
            col_bar1, col_bar2 = st.columns([3, 1])
            with col_bar1:
                st.progress(float(prob))
            with col_bar2:
                is_best = (cls == predicted_label)
                label_text = f"**{prob*100:.1f} %**" if is_best else f"{prob*100:.1f} %"
                st.markdown(f"{cls.split(' ')[0]} : {label_text}")
                
        # Recommandations agronomiques
        st.markdown("#### 🌾 Recommandations & Mesures Agronomiques :")
        st.markdown(f"**Diagnostic :** {diag_info.get('summary', '')}")
        for action in diag_info.get('actions', []):
            st.markdown(f"- {action}")
            
    # --------------------------------------------------------------------------
    # 7. EXPLICABILITÉ VISUELLE DES DESCRIPTEURS (XAI)
    # --------------------------------------------------------------------------
    st.markdown("---")
    with st.expander("🔍 Voir les coulisses de l'extraction des caractéristiques (HOG, LBP, HSV)", expanded=False):
        st.markdown("""
        Avant la classification, l'image subit une extraction multi-dimensionnelle permettant à l'algorithme 
        d'analyser la géométrie, la rugosité et la couleur sans dépendre uniquement des pixels bruts :
        """)
        vcols = st.columns(4)
        with vcols[0]:
            st.image(visuals['resized'], caption="Image Redimensionnée (128x128)", use_container_width=True)
        with vcols[1]:
            st.image(visuals['hog_img'], caption="Contours HOG (Détection formes d'insectes)", use_container_width=True)
        with vcols[2]:
            st.image(visuals['lbp_img'], caption="Texture LBP (Rugosité fumagine)", use_container_width=True)
        with vcols[3]:
            st.image(visuals['saturation_mask'], caption="Masque Saturation HSV (Détection miellat)", use_container_width=True)
            
st.markdown("---")
st.caption("🌴 Système de Diagnostic Foliaire du Palmier Dattier — Machine Learning Appliqué")
