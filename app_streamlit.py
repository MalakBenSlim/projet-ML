"""
================================================================================
APPLICATION WEB - DÉTECTION DU DUBAS DU PALMIER DATTIER PAR DEEP LEARNING
================================================================================
Application Web Streamlit permettant de :
1. Téléverser une photo de foliole de palmier dattier ou tester des cas types.
2. Évaluer en direct les 4 modèles de Deep Learning :
   - ConvNeXt-Tiny (Champion - 92.89%)
   - MobileNetV3-ECA (Edge-AI - 90.89%)
   - Swin-T (Vision Transformer - 90.67%)
   - EfficientNet-V2-S (Multi-échelle - 88.89%)
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
import torch
import torch.nn as nn
from torchvision import transforms, models

# ------------------------------------------------------------------------------
# 1. CONFIGURATION GÉNÉRALE & CONSTANTES
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="🌴 Diagnostic Phytosanitaire du Palmier Dattier",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = (BASE_DIR / "Image dataset of infected date palm leaves" 
               / "Image dataset of infected date palm leaves by dubas insects" 
               / "Palm Leaves Dataset")
MODELS_DIR = BASE_DIR / "models"
GEN_DIR = BASE_DIR / "generation"

CLASS_NAMES = ['Healthy (Saine)', 'Bug (Insectes seuls)', 'Honey (Miellat)', 'Mixed (Infestation mixte)']

DIAGNOSIS_DATA = {
    'Healthy (Saine)': {
        'icon': '🌿',
        'color': '#2ECC71',
        'summary': 'Feuille saine. Tissu chlorophyllien intact, aucune trace de ravageur ni de sécrétion de miellat.',
        'actions': [
            'Poursuivre la surveillance régulière du feuillage.',
            'Maintenir une irrigation équilibrée et une fertilisation adaptée.',
            'Aucune intervention phytosanitaire chimique requise.'
        ]
    },
    'Bug (Insectes seuls)': {
        'icon': '🐛',
        'color': '#E74C3C',
        'summary': 'Présence d\'individus Dubas (Ommatissus lybicus) sans accumulation critique de miellat.',
        'actions': [
            'Intervention précoce recommandée pour bloquer la ponte et la multiplication larvaire.',
            'Application ciblée d\'insecticides biologiques (huile de neem ou savon noir horticole).',
            'Installation de pièges visuels jaunes englués pour monitorer la population d\'adultes volants.'
        ]
    },
    'Honey (Miellat)': {
        'icon': '🍯',
        'color': '#F39C12',
        'summary': 'Excrétion abondante de miellat collant et brillant par les larves de Dubas.',
        'actions': [
            'Lavage à grande eau sous pression pour lessiver le miellat avant le développement de la fumagine.',
            'Traitement larvicide spécifique ciblé sous la face inférieure des palmes.',
            'Élagage aéré pour améliorer la ventilation de la couronne du palmier.'
        ]
    },
    'Mixed (Infestation mixte)': {
        'icon': '🥀',
        'color': '#9B59B6',
        'summary': 'Infestation sévère : insectes actifs, miellat et développement de fumagine opaque.',
        'actions': [
            'Traitement d\'urgence combiné (insecticide homologué et agent fongique pour stopper la fumagine).',
            'Élimination et incinération des palmes basales les plus atteintes.',
            'Alerte immédiate du service phytosanitaire pour traitement coordonné de la parcelle oasienne.'
        ]
    }
}

# ------------------------------------------------------------------------------
# 2. CHARGEMENT DES 4 MODÈLES DEEP LEARNING (MIS EN CACHE)
# ------------------------------------------------------------------------------
class DeepClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes=4, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.net(x)

@st.cache_resource
def load_all_models():
    """Charge en cache les 4 modèles de Deep Learning."""
    dl_models = {}
    
    try:
        # A. ConvNeXt-Tiny (Champion - 92.89%)
        m_conv = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        backbone_conv = nn.Sequential(m_conv.features, m_conv.avgpool, m_conv.classifier[0], m_conv.classifier[1])
        for p in backbone_conv.parameters(): p.requires_grad_(False)
        backbone_conv.eval()
        h_conv = DeepClassifierHead(768, 4)
        h_conv.load_state_dict(torch.load(MODELS_DIR / "convnext_tiny_head.pth", weights_only=True, map_location='cpu'))
        h_conv.eval()
        dl_models['🏆 ConvNeXt-Tiny (Champion - 92.89%)'] = {'base': backbone_conv, 'head': h_conv}

        # B. MobileNetV3-ECA (Edge-AI - 90.89%)
        m_mob = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        backbone_mob = nn.Sequential(m_mob.features, m_mob.avgpool, nn.Flatten(1))
        for p in backbone_mob.parameters(): p.requires_grad_(False)
        backbone_mob.eval()
        h_mob = DeepClassifierHead(960, 4)
        h_mob.load_state_dict(torch.load(MODELS_DIR / "mobilenetv3_eca_head.pth", weights_only=True, map_location='cpu'))
        h_mob.eval()
        dl_models['🚀 MobileNetV3-ECA (Edge-AI - 90.89%)'] = {'base': backbone_mob, 'head': h_mob}

        # C. Swin-T (Vision Transformer - 90.67%)
        m_swin = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        backbone_swin = nn.Sequential(m_swin.features, m_swin.norm, m_swin.permute, m_swin.avgpool, m_swin.flatten)
        for p in backbone_swin.parameters(): p.requires_grad_(False)
        backbone_swin.eval()
        h_swin = DeepClassifierHead(768, 4)
        h_swin.load_state_dict(torch.load(MODELS_DIR / "swin_t_head.pth", weights_only=True, map_location='cpu'))
        h_swin.eval()
        dl_models['🔮 Swin-T (Vision Transformer - 90.67%)'] = {'base': backbone_swin, 'head': h_swin}

        # D. EfficientNet-V2-S (Multi-échelle - 88.89%)
        m_eff = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        backbone_eff = nn.Sequential(m_eff.features, m_eff.avgpool, nn.Flatten(1))
        for p in backbone_eff.parameters(): p.requires_grad_(False)
        backbone_eff.eval()
        h_eff = DeepClassifierHead(1280, 4)
        h_eff.load_state_dict(torch.load(MODELS_DIR / "efficientnet_v2_s_head.pth", weights_only=True, map_location='cpu'))
        h_eff.eval()
        dl_models['🧠 EfficientNet-V2-S (Multi-échelle - 88.89%)'] = {'base': backbone_eff, 'head': h_eff}

    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles Deep Learning : {e}")

    return dl_models

dl_models_dict = load_all_models()

# ------------------------------------------------------------------------------
# 3. TRANSFORMATION & INFÉRENCE
# ------------------------------------------------------------------------------
dl_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_deep_learning(image: Image.Image, dl_model_dict):
    """Inférence via le backbone gelé et la tête entraînée."""
    t_img = dl_transform(image.convert('RGB')).unsqueeze(0)
    base = dl_model_dict['base']
    head = dl_model_dict['head']
    with torch.no_grad():
        feat = base(t_img)
        if feat.dim() > 2:
            feat = torch.flatten(feat, 1)
        logits = head(feat)
        probas = torch.softmax(logits, dim=1).numpy()[0]
    return probas

# ------------------------------------------------------------------------------
# 4. BARRE LATÉRALE (SIDEBAR)
# ------------------------------------------------------------------------------
st.sidebar.image("generale.jpg" if Path("generale.jpg").exists() else "🌴", use_container_width=True)
st.sidebar.markdown("### ⚙️ Choix de l'Algorithme Deep Learning")

available_models = list(dl_models_dict.keys())
selected_model_name = st.sidebar.selectbox(
    "Modèle actif :",
    available_models,
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 Échantillons de Test du Dataset")
sample_choice = st.sidebar.radio(
    "Tester un cas type :",
    ["(Aucun - Importer une image)", "Exemple Saine (Healthy)", "Exemple Insectes (Bug)", "Exemple Miellat (Honey)", "Exemple Mixte (Mixed)"]
)

st.sidebar.markdown("---")
st.sidebar.caption("🌴 **Projet ML Palmier Dattier** — Détection automatique du ravageur Dubas (*Ommatissus lybicus*).")

# ------------------------------------------------------------------------------
# 5. CONTENU PRINCIPAL
# ------------------------------------------------------------------------------
st.markdown("""
<style>
.main-title { font-size: 2.2rem; font-weight: 800; color: #2C3E50; margin-bottom: 0px; }
.sub-title { font-size: 1.1rem; color: #7F8C8D; margin-bottom: 25px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🌴 Détection du Dubas du Palmier Dattier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Système de Vision Artificielle par Deep Learning (SOTA > 88-93%)</p>', unsafe_allow_html=True)

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
    elif sample_choice != "(Aucun - Importer une image)":
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
# 6. ANALYSE ET PRÉDICTION
# ------------------------------------------------------------------------------
if image_to_analyze is not None and selected_model_name in dl_models_dict:
    dl_model = dl_models_dict[selected_model_name]
    probas = predict_deep_learning(image_to_analyze, dl_model)
    proba_dict = dict(zip(CLASS_NAMES, probas))
    pred_idx = int(np.argmax(probas))
    predicted_label = CLASS_NAMES[pred_idx]
    confidence = probas[pred_idx] * 100
    diag_info = DIAGNOSIS_DATA.get(predicted_label, {})
    
    with col_pred:
        st.markdown("### 🎯 2. Diagnostic du Modèle")
        st.markdown(f"""
        <div style="background-color: {diag_info.get('color', '#333')}15; 
                    border: 2px solid {diag_info.get('color', '#333')}; 
                    border-radius: 10px; padding: 18px; text-align: center; margin-bottom: 20px;">
            <span style="font-size: 2.2rem;">{diag_info.get('icon', '🔍')}</span>
            <h2 style="color: {diag_info.get('color', '#333')}; margin: 5px 0 0 0;">
                {predicted_label}
            </h2>
            <p style="font-weight: 700; color: #555; margin-top: 5px; font-size: 1.05rem;">
                Indice de confiance : <span style="color: {diag_info.get('color', '#333')};">{confidence:.1f} %</span><br/>
                <small style="color: #666;">Modèle actif : {selected_model_name}</small>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 📊 Probabilités Estimées par Classe :")
        for cls, prob in proba_dict.items():
            col_bar1, col_bar2 = st.columns([3, 1])
            with col_bar1:
                st.progress(float(prob))
            with col_bar2:
                is_best = (cls == predicted_label)
                label_text = f"**{prob*100:.1f} %**" if is_best else f"{prob*100:.1f} %"
                st.markdown(f"{cls.split(' ')[0]} : {label_text}")
                
        st.markdown("#### 🌾 Recommandations & Mesures Agronomiques :")
        st.markdown(f"**Diagnostic :** {diag_info.get('summary', '')}")
        for action in diag_info.get('actions', []):
            st.markdown(f"- {action}")

st.markdown("---")
st.caption("🌴 Système de Diagnostic Foliaire du Palmier Dattier — Deep Learning Appliqué (ConvNeXt, MobileNetV3, Swin-T, EfficientNet-V2)")
