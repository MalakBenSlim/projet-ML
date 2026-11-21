"""
Backend API FastAPI pour le projet de détection des maladies du palmier dattier.
Fournit :
1. /api/predict : inférence avec les 4 modèles Deep Learning
2. /api/chat : chatbot expert Gemini avec premier message "Salut !"
3. /api/auth/login & /api/auth/verify : Authentification JWT (admin/admin en mode dev)
4. /api/gemini/models : chargement dynamique des modèles Gemini depuis la clé API Google
5. /api/config : gestion de la clé API Gemini et du modèle (protégé par JWT)
6. /api/dashboard : statistiques, benchmark, métriques par classe
7. /api/samples : cas types d'images du dataset
8. /api/image : service des images locales
"""

import os
import json
import base64
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import requests
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import jwt

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
GEN_DIR = BASE_DIR / "generation"
CONFIG_FILE = BASE_DIR / ".gemini_config.json"
DATASET_PATH = (BASE_DIR / "Image dataset of infected date palm leaves" 
               / "Image dataset of infected date palm leaves by dubas insects" 
               / "Palm Leaves Dataset")

# Configuration JWT (Mode Dev)
JWT_SECRET = "palm-dubas-secret-key-dev-mode-2026"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

CLASS_NAMES = ['Healthy (Saine)', 'Bug (Insectes seuls)', 'Honey (Miellat)', 'Mixed (Infestation mixte)']

DIAGNOSIS_DATA = {
    'Healthy (Saine)': {
        'icon': '🌿',
        'color': '#2ECC71',
        'urgency': 'Normale',
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
        'urgency': 'Surveillance active',
        'summary': 'Présence d\'individus Dubas (Ommatissus lybicus) sans accumulation critique de miellat.',
        'actions': [
            'Intervention précoce recommandée pour bloquer la ponte et la multiplication larvaire.',
            'Application ciblée d\'insecticides biologiques (savon noir horticole ou huile de neem).',
            'Installation de pièges visuels jaunes englués pour monitorer les adultes volants.'
        ]
    },
    'Honey (Miellat)': {
        'icon': '🍯',
        'color': '#F39C12',
        'urgency': 'Élevée',
        'summary': 'Excrétion abondante de miellat collant et brillant par les larves de Dubas.',
        'actions': [
            'Lavage à grande eau sous pression pour lessiver le miellat avant l\'apparition de la fumagine.',
            'Traitement larvicide spécifique ciblé sous la face inférieure des palmes.',
            'Élagage aéré pour améliorer la ventilation de la couronne du palmier.'
        ]
    },
    'Mixed (Infestation mixte)': {
        'icon': '🥀',
        'color': '#9B59B6',
        'urgency': 'Critique',
        'summary': 'Infestation sévère : insectes actifs, miellat et développement de fumagine opaque.',
        'actions': [
            'Traitement d\'urgence combiné (insecticide homologué et agent fongique pour stopper la fumagine).',
            'Élimination et incinération des palmes basales sévèrement nécrosées.',
            'Alerte immédiate du service phytosanitaire pour traitement coordonné de la palmeraie.'
        ]
    }
}

SYSTEM_INSTRUCTION = """Tu es l'Assistant officiel et expert du projet PalmVision, dédié à la détection et au diagnostic des maladies du palmier dattier (Phoenix dactylifera L.) causées par le ravageur Dubas bug (Ommatissus lybicus).

CONSIGNE STRICTE DE PÉRIMÈTRE & SÉCURITÉ :
Tu réponds EXCLUSIVEMENT aux questions portant sur :
1. Le projet PalmVision et la recherche associée : les 4 modèles de Deep Learning étudiés (ConvNeXt-Tiny champion avec 92.89% d'accuracy et 93.00% de Macro F1, MobileNetV3-ECA à 90.89%, Swin-T Vision Transformer à 90.67%, EfficientNet-V2-S à 88.89%), l'architecture (Transfer Learning, backbones gelés ImageNet, tête dense 256 avec BatchNorm et Dropout 0.2, optimiseur AdamW lr=1e-3, CosineAnnealingLR sur 40 époques).
2. Le jeu de données (Dataset) : 3 000 images haute résolution (896x896, 892x892, 1792x1792 px), réparties en 4 classes : Healthy (800 images), Bug (600 images), Honey (800 images), Mixed (800 images dans le dossier 'Dubas/'). Split stratifié 70% train (2100 images), 15% val (450 images), 15% test (450 images).
3. La biologie du palmier dattier, le cycle de vie du Dubas (Ommatissus lybicus), les symptômes (piqûres foliaires, miellat collant, champignon saprophyte de la fumagine noire bloquant la photosynthèse) et les protocoles de traitement agronomique (savon noir, huile de neem, lavage sous pression, pièges jaunes, élagage).
4. Les statistiques, métriques par classe, matrices de confusion et comparatifs.
5. L'analyse des photos foliaires fournies par l'utilisateur et les prédictions du modèle de vision du backend.

RÈGLE IMPÉRATIVE EN CAS DE QUESTION HORS SUJET :
Si l'utilisateur pose une question hors de ce domaine (par exemple sur la cuisine, le sport, la politique, des mathématiques sans rapport, du code général sans lien avec le projet, etc.), tu dois poliment et fermement REFUSER de répondre :
"Je suis l'Assistant du projet PalmVision, spécialisé exclusivement dans le diagnostic des maladies du palmier dattier (Dubas bug, vision par ordinateur et statistiques du projet). Je ne peux répondre qu'aux questions relevant de ce domaine. N'hésitez pas à me questionner sur les 4 modèles Deep Learning, le dataset ou à me transmettre des photos pour analyse !"

STYLE DE RÉPONSE OBLIGATOIRE :
Sois TOUJOURS COURT, CONCIS, DIRECT et TRÈS PROFESSIONNEL.
- Évite absolument les longues introductions, les bavardages et les gros pavés de texte.
- Donne la réponse précise en 2 à 5 lignes claires ou avec quelques tirets concis.
- Va droit au but avec des chiffres précis et un ton scientifique impeccable."""

# ------------------------------------------------------------------------------
# CHARGEMENT DES MODÈLES PYTORCH
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

dl_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dl_models: Dict[str, Any] = {}

def load_models_cache():
    global dl_models
    if dl_models:
        return
    print("-> Chargement des 4 modèles Deep Learning en mémoire...")
    try:
        # ConvNeXt-Tiny
        m_conv = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        bb_conv = nn.Sequential(m_conv.features, m_conv.avgpool, m_conv.classifier[0], m_conv.classifier[1])
        for p in bb_conv.parameters(): p.requires_grad_(False)
        bb_conv.eval()
        h_conv = DeepClassifierHead(768, 4)
        h_conv.load_state_dict(torch.load(MODELS_DIR / "convnext_tiny_head.pth", weights_only=True, map_location='cpu'))
        h_conv.eval()
        dl_models['ConvNeXt-Tiny'] = {'base': bb_conv, 'head': h_conv, 'name': 'ConvNeXt-Tiny (Champion)', 'acc': 92.89}

        # MobileNetV3-ECA
        m_mob = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        bb_mob = nn.Sequential(m_mob.features, m_mob.avgpool, nn.Flatten(1))
        for p in bb_mob.parameters(): p.requires_grad_(False)
        bb_mob.eval()
        h_mob = DeepClassifierHead(960, 4)
        h_mob.load_state_dict(torch.load(MODELS_DIR / "mobilenetv3_eca_head.pth", weights_only=True, map_location='cpu'))
        h_mob.eval()
        dl_models['MobileNetV3-ECA'] = {'base': bb_mob, 'head': h_mob, 'name': 'MobileNetV3-ECA (Edge-AI)', 'acc': 90.89}

        # Swin-T
        m_swin = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        bb_swin = nn.Sequential(m_swin.features, m_swin.norm, m_swin.permute, m_swin.avgpool, m_swin.flatten)
        for p in bb_swin.parameters(): p.requires_grad_(False)
        bb_swin.eval()
        h_swin = DeepClassifierHead(768, 4)
        h_swin.load_state_dict(torch.load(MODELS_DIR / "swin_t_head.pth", weights_only=True, map_location='cpu'))
        h_swin.eval()
        dl_models['Swin-T'] = {'base': bb_swin, 'head': h_swin, 'name': 'Swin-T (Vision Transformer)', 'acc': 90.67}

        # EfficientNet-V2-S
        m_eff = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        bb_eff = nn.Sequential(m_eff.features, m_eff.avgpool, nn.Flatten(1))
        for p in bb_eff.parameters(): p.requires_grad_(False)
        bb_eff.eval()
        h_eff = DeepClassifierHead(1280, 4)
        h_eff.load_state_dict(torch.load(MODELS_DIR / "efficientnet_v2_s_head.pth", weights_only=True, map_location='cpu'))
        h_eff.eval()
        dl_models['EfficientNet-V2-S'] = {'base': bb_eff, 'head': h_eff, 'name': 'EfficientNet-V2-S (Multi-échelle)', 'acc': 88.89}
        print("-> Tous les modèles Deep Learning sont chargés avec succès !")
    except Exception as e:
        print(f"Erreur chargement modèles : {e}")

# ------------------------------------------------------------------------------
# CONFIGURATION GEMINI
# ------------------------------------------------------------------------------
def get_gemini_config() -> Dict[str, Any]:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    env_key = os.environ.get("GEMINI_API_KEY", "")
    return {
        "api_key": env_key,
        "model": "gemini-1.5-flash",
        "temperature": 0.3,
        "max_output_tokens": 1024
    }

def save_gemini_config(cfg: Dict[str, Any]):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

# ------------------------------------------------------------------------------
# AUTHENTIFICATION JWT (MODE DEV : admin / admin)
# ------------------------------------------------------------------------------
class LoginRequest(BaseModel):
    username: str
    password: str

def create_jwt_token(username: str) -> str:
    exp = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    payload = {
        "sub": username,
        "exp": exp,
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt_token(authorization: Optional[str] = Header(None)) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Jeton d'authentification manquant")
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Schéma d'authentification invalide")
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload["sub"]
    except Exception:
        raise HTTPException(status_code=401, detail="Session expirée ou jeton JWT invalide")

# ------------------------------------------------------------------------------
# APPLICATION FASTAPI
# ------------------------------------------------------------------------------
app = FastAPI(
    title="PalmVision API",
    description="API PalmVision avec Inférence Deep Learning, Authentification JWT et Assistant IA",
    version="2.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    load_models_cache()

# ------------------------------------------------------------------------------
# ROUTES AUTHENTIFICATION JWT
# ------------------------------------------------------------------------------
@app.post("/api/auth/login")
@app.post("/api/auth/login/")
def login(creds: LoginRequest):
    u = creds.username.strip().lower()
    p = creds.password.strip()
    if (u in ["admin", "admin@palm.ai", "administrateur"]) and p == "admin":
        token = create_jwt_token(creds.username.strip())
        return {
            "access_token": token,
            "token_type": "bearer",
            "user": {
                "username": creds.username.strip(),
                "role": "Administrateur"
            }
        }
    raise HTTPException(
        status_code=401, 
        detail="Identifiant ou mot de passe incorrect."
    )

@app.get("/api/auth/verify")
def verify_auth(user: str = Depends(verify_jwt_token)):
    return {"valid": True, "user": user}

# ------------------------------------------------------------------------------
# ROUTES PRÉDICTION
# ------------------------------------------------------------------------------
@app.get("/api/models")
def get_available_models():
    load_models_cache()
    return [
        {
            "id": k,
            "name": v['name'],
            "accuracy": v['acc'],
            "is_default": (k == 'ConvNeXt-Tiny')
        }
        for k, v in dl_models.items()
    ]

def predict_pil_image(image: Image.Image, model_name: str = "ConvNeXt-Tiny") -> dict:
    load_models_cache()
    if model_name not in dl_models:
        model_name = "ConvNeXt-Tiny"

    m = dl_models[model_name]
    t_img = dl_transform(image.convert('RGB')).unsqueeze(0)
    
    with torch.no_grad():
        feat = m['base'](t_img)
        if feat.dim() > 2:
            feat = torch.flatten(feat, 1)
        logits = m['head'](feat)
        probas = torch.softmax(logits, dim=1).numpy()[0]

    pred_idx = int(np.argmax(probas))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(probas[pred_idx] * 100)
    
    proba_dict = {
        cls: round(float(prob) * 100, 2)
        for cls, prob in zip(CLASS_NAMES, probas)
    }

    diag = DIAGNOSIS_DATA.get(pred_label, {})

    return {
        "model_used": model_name,
        "predicted_class": pred_label,
        "confidence": round(confidence, 2),
        "probabilities": proba_dict,
        "diagnosis": {
            "icon": diag.get('icon', '🔍'),
            "color": diag.get('color', '#333'),
            "urgency": diag.get('urgency', 'Normale'),
            "summary": diag.get('summary', ''),
            "actions": diag.get('actions', [])
        }
    }

@app.post("/api/predict")
async def predict_image(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    model_name: str = Form("ConvNeXt-Tiny")
):
    load_models_cache()
    image = None
    if file is not None:
        try:
            image = Image.open(file.file).convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image invalide: {e}")
    elif sample_id:
        sample_map = {
            "healthy": ("Healthy", "Healthy(1).jpg"),
            "bug": ("Bug", "Bug(1).JPG"),
            "honey": ("Honey", "Honey(1).jpg"),
            "mixed": ("Dubas", "Mixed(1).jpg")
        }
        if sample_id.lower() in sample_map:
            folder, fname = sample_map[sample_id.lower()]
            img_path = DATASET_PATH / folder / fname
            if not img_path.exists():
                d = DATASET_PATH / folder
                if d.exists():
                    files = list(d.glob("*.jpg")) + list(d.glob("*.JPG"))
                    if files:
                        img_path = files[0]
            if img_path.exists():
                image = Image.open(img_path).convert('RGB')
    
    if image is None:
        raise HTTPException(status_code=400, detail="Veuillez fournir une image ou un sample_id valide.")

    return predict_pil_image(image, model_name=model_name)

# ------------------------------------------------------------------------------
# ROUTES SAMPLES DU DATASET
# ------------------------------------------------------------------------------
@app.get("/api/samples")
def get_dataset_samples():
    sample_categories = [
        {"id": "healthy", "title": "Feuille Saine (Healthy)", "class": "Healthy (Saine)", "folder": "Healthy", "tag": "Saine"},
        {"id": "bug", "title": "Présence Insectes (Bug)", "class": "Bug (Insectes seuls)", "folder": "Bug", "tag": "Insectes"},
        {"id": "honey", "title": "Sécrétion Miellat (Honey)", "class": "Honey (Miellat)", "folder": "Honey", "tag": "Miellat"},
        {"id": "mixed", "title": "Infestation Mixte (Mixed)", "class": "Mixed (Infestation mixte)", "folder": "Dubas", "tag": "Mixte"}
    ]
    results = []
    for sc in sample_categories:
        fld = DATASET_PATH / sc['folder']
        thumb_base64 = ""
        filename = ""
        if fld.exists():
            files = list(fld.glob("*.jpg")) + list(fld.glob("*.JPG"))
            if files:
                first_file = files[0]
                filename = first_file.name
                try:
                    with Image.open(first_file) as im:
                        im.thumbnail((180, 180))
                        import io
                        buf = io.BytesIO()
                        im.convert('RGB').save(buf, format="JPEG", quality=80)
                        thumb_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                except Exception:
                    pass
        results.append({
            "id": sc['id'],
            "title": sc['title'],
            "class_name": sc['class'],
            "tag": sc['tag'],
            "filename": filename,
            "thumbnail": f"data:image/jpeg;base64,{thumb_base64}" if thumb_base64 else None
        })
    return results

# ------------------------------------------------------------------------------
# ROUTES CHATBOT AVEC API GEMINI
# ------------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    text: str

class ChatRequest(BaseModel):
    message: Optional[str] = ""
    history: Optional[List[ChatMessage]] = []
    images: Optional[List[str]] = []  # Photos transmises au robot (base64 ou data URL)
    api_key: Optional[str] = None
    model: Optional[str] = None
    dl_model: Optional[str] = "ConvNeXt-Tiny"

@app.post("/api/chat")
async def chat_gemini(req: ChatRequest):
    import io
    user_msg = (req.message or "").strip()
    dl_model_name = req.dl_model if (req.dl_model and req.dl_model in dl_models) else "ConvNeXt-Tiny"

    # 1. Traitement des photos envoyées au robot via le backend Deep Learning
    dl_analysis_results = []
    gemini_inline_images = []
    if req.images:
        for idx, img_str in enumerate(req.images):
            try:
                mime_type = "image/jpeg"
                raw_b64 = img_str
                if "data:" in img_str and ";base64," in img_str:
                    header, raw_b64 = img_str.split(";base64,", 1)
                    mime_type = header.replace("data:", "")
                
                img_bytes = base64.b64decode(raw_b64)
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                pred = predict_pil_image(pil_img, model_name=dl_model_name)
                dl_analysis_results.append({
                    "index": idx + 1,
                    "prediction": pred
                })
                gemini_inline_images.append({
                    "mime_type": mime_type,
                    "data": raw_b64
                })
            except Exception as ex:
                print(f"Erreur analyse photo #{idx+1} : {ex}")

    # Synthèse textuelle de l'analyse Deep Learning locale
    dl_summary_text = ""
    if dl_analysis_results:
        parts = []
        for item in dl_analysis_results:
            i = item["index"]
            p = item["prediction"]
            diag = p["diagnosis"]
            parts.append(
                f"• **Photo {i} : {p['predicted_class']}** ({p['confidence']}%)\n"
                f"  - Diagnostic : {diag['summary']}\n"
                f"  - Action : {diag['actions'][0] if diag['actions'] else 'Surveillance'}"
            )
        dl_summary_text = "\n\n".join(parts)

    cfg = get_gemini_config()
    api_key = req.api_key.strip() if (req.api_key and req.api_key.strip()) else cfg.get("api_key", "").strip()

    # Si pas de clé Gemini configurée
    if not api_key:
        if dl_analysis_results:
            nb = len(dl_analysis_results)
            reply = (
                f"🔬 **Analyse Deep Learning PalmVision ({dl_model_name}) — {nb} photo{'s' if nb > 1 else ''} :**\n\n"
                f"{dl_summary_text}\n\n"
                f"> *(Clé Google Gemini non configurée : analyse réalisée directement par le modèle neuronal local)*"
            )
            return {"reply": reply, "status": "success", "model": dl_model_name}
        return {
            "reply": "⚠️ Clé API Gemini non configurée dans le Dashboard. Vous pouvez tout de même me transmettre des photos de palmes pour une analyse Deep Learning instantanée !",
            "status": "missing_api_key"
        }

    chosen_model = req.model if req.model else cfg.get("model", "gemini-1.5-flash")
    if chosen_model.startswith("models/"):
        chosen_model = chosen_model.replace("models/", "")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{chosen_model}:generateContent?key={api_key}"

    contents = []
    if req.history:
        for m in req.history[-10:]:
            role = "user" if m.role == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": m.text}]
            })

    current_parts = []
    for img_obj in gemini_inline_images:
        current_parts.append({
            "inlineData": {
                "mimeType": img_obj["mime_type"],
                "data": img_obj["data"]
            }
        })

    if dl_analysis_results:
        prompt_with_dl = (
            f"L'utilisateur a envoyé {len(dl_analysis_results)} photo(s) foliaire(s) de palmier dattier.\n"
            f"Notre modèle Deep Learning ({dl_model_name}) a analysé les images avec ces résultats précis :\n"
            f"{dl_summary_text}\n\n"
            f"Question / Instruction de l'utilisateur : '{user_msg if user_msg else 'Analyse ces photos.'}'\n\n"
            f"Consigne : Rédige une réponse courte, professionnelle et directe (2 à 5 lignes max). "
            f"Confirme l'état détecté, résume les symptômes et donne les actions phytosanitaires immédiates."
        )
        current_parts.append({"text": prompt_with_dl})
    else:
        current_parts.append({"text": user_msg if user_msg else "Bonjour"})

    contents.append({
        "role": "user",
        "parts": current_parts
    })

    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_INSTRUCTION}]
        },
        "contents": contents,
        "generationConfig": {
            "temperature": float(cfg.get("temperature", 0.3)),
            "maxOutputTokens": int(cfg.get("max_output_tokens", 1024))
        }
    }

    try:
        resp = requests.post(url, json=payload, timeout=25)
        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                text_part = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                return {"reply": text_part, "status": "success", "model": chosen_model}
            elif dl_summary_text:
                return {"reply": f"🔬 **Analyse Deep Learning PalmVision ({dl_model_name}) :**\n\n{dl_summary_text}", "status": "success"}
            return {"reply": "Désolé, aucune réponse générée par le modèle.", "status": "empty"}
        else:
            if dl_summary_text:
                return {
                    "reply": f"🔬 **Analyse Deep Learning PalmVision ({dl_model_name}) :**\n\n{dl_summary_text}",
                    "status": "success"
                }
            err_data = resp.json()
            err_msg = err_data.get("error", {}).get("message", resp.text)
            return {
                "reply": f"⚠️ Erreur API Gemini ({resp.status_code}) : {err_msg}.",
                "status": "error"
            }
    except Exception as e:
        if dl_summary_text:
            return {
                "reply": f"🔬 **Analyse Deep Learning PalmVision ({dl_model_name}) :**\n\n{dl_summary_text}",
                "status": "success"
            }
        return {
            "reply": f"Erreur de communication réseau avec l'API Gemini : {str(e)}",
            "status": "network_error"
        }

# ------------------------------------------------------------------------------
# CHARGEMENT DYNAMIQUE DES MODÈLES GEMINI DEPUIS LA CLÉ API
# ------------------------------------------------------------------------------
@app.get("/api/gemini/models")
def get_gemini_models_from_key(key: Optional[str] = None):
    cfg = get_gemini_config()
    api_key = key.strip() if (key and key.strip()) else cfg.get("api_key", "").strip()
    
    fallback_models = [
        {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash (Recommandé, rapide)", "desc": "Multimodal rapide et économique"},
        {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash (Nouvelle génération)", "desc": "Dernière version ultra-rapide de Google"},
        {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro (Haute capacité)", "desc": "Raisonnement complexe et grande fenêtre de contexte"}
    ]
    
    if not api_key:
        return {"models": fallback_models, "from_api": False}

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            raw_models = data.get("models", [])
            extracted = []
            for m in raw_models:
                methods = m.get("supportedGenerationMethods", [])
                if "generateContent" in methods:
                    m_id = m.get("name", "").replace("models/", "")
                    # Prioriser les modèles Gemini
                    if "gemini" in m_id.lower():
                        extracted.append({
                            "id": m_id,
                            "name": m.get("displayName", m_id),
                            "desc": m.get("description", "")[:120]
                        })
            if extracted:
                return {"models": extracted, "from_api": True}
    except Exception as e:
        print(f"Erreur requête modèles Gemini: {e}")
        
    return {"models": fallback_models, "from_api": False}

# ------------------------------------------------------------------------------
# ROUTES CONFIGURATION & DASHBOARD (PROTÉGÉES PAR JWT)
# ------------------------------------------------------------------------------
class ConfigUpdate(BaseModel):
    api_key: str
    model: Optional[str] = "gemini-1.5-flash"
    temperature: Optional[float] = 0.3

@app.get("/api/config")
def get_config(user: str = Depends(verify_jwt_token)):
    cfg = get_gemini_config()
    raw_key = cfg.get("api_key", "")
    masked_key = (raw_key[:6] + "..." + raw_key[-4:]) if len(raw_key) > 10 else ("Configurée" if raw_key else "Non configurée")
    return {
        "has_key": bool(raw_key),
        "masked_key": masked_key,
        "model": cfg.get("model", "gemini-1.5-flash"),
        "temperature": cfg.get("temperature", 0.3),
        "authenticated_user": user
    }

@app.post("/api/config")
def update_config(body: ConfigUpdate, user: str = Depends(verify_jwt_token)):
    cfg = get_gemini_config()
    if body.api_key is not None and body.api_key.strip():
        cfg["api_key"] = body.api_key.strip()
    if body.model:
        cfg["model"] = body.model.replace("models/", "")
    if body.temperature is not None:
        cfg["temperature"] = body.temperature
    save_gemini_config(cfg)
    return {"status": "saved", "has_key": bool(cfg.get("api_key")), "user": user}

@app.post("/api/config/test")
def test_gemini_key(body: ConfigUpdate, user: str = Depends(verify_jwt_token)):
    key = body.api_key.strip() if body.api_key else get_gemini_config().get("api_key", "")
    if not key:
        return {"ok": False, "message": "Clé API vide."}
    m = body.model if body.model else "gemini-1.5-flash"
    m_clean = m.replace("models/", "")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_clean}:generateContent?key={key}"
    payload = {"contents": [{"parts": [{"text": "Réponds simplement par 'OK'."}]}]}
    try:
        r = requests.post(url, json=payload, timeout=8)
        if r.status_code == 200:
            return {"ok": True, "message": f"Clé API valide et opérationnelle avec {m_clean} !"}
        else:
            return {"ok": False, "message": f"Erreur {r.status_code}: {r.json().get('error', {}).get('message', r.text)}"}
    except Exception as e:
        return {"ok": False, "message": f"Erreur de connexion : {e}"}

@app.get("/api/dashboard")
def get_dashboard_data():
    return {
        "benchmark": [
            {"rank": 1, "model": "ConvNeXt-Tiny", "type": "CNN Moderne (SOTA)", "params": "28.6 M", "accuracy": 92.89, "f1": 93.00, "is_best": True},
            {"rank": 2, "model": "MobileNetV3-ECA", "type": "Edge-AI Léger", "params": "5.4 M", "accuracy": 90.89, "f1": 91.05, "is_best": False},
            {"rank": 3, "model": "Swin-T", "type": "Vision Transformer", "params": "28.3 M", "accuracy": 90.67, "f1": 90.70, "is_best": False},
            {"rank": 4, "model": "EfficientNet-V2-S", "type": "CNN Multi-Échelle", "params": "21.5 M", "accuracy": 88.89, "f1": 88.98, "is_best": False}
        ],
        "dataset_stats": {
            "total_images": 3000,
            "classes": [
                {"name": "Healthy (Saine)", "count": 800, "percentage": 26.67, "color": "#2ECC71"},
                {"name": "Bug (Insectes seuls)", "count": 600, "percentage": 20.00, "color": "#E74C3C"},
                {"name": "Honey (Miellat)", "count": 800, "percentage": 26.67, "color": "#F39C12"},
                {"name": "Mixed (Infestation mixte)", "count": 800, "percentage": 26.67, "color": "#9B59B6"}
            ],
            "split": {"train": 2100, "val": 450, "test": 450}
        },
        "convnext_metrics": [
            {"class_name": "Healthy (Saine)", "precision": 0.9836, "recall": 1.0000, "f1": 0.9917, "support": 120},
            {"class_name": "Bug (Insectes seuls)", "precision": 0.9271, "recall": 0.9889, "f1": 0.9570, "support": 90},
            {"class_name": "Honey (Miellat)", "precision": 0.9123, "recall": 0.8667, "f1": 0.8889, "support": 120},
            {"class_name": "Mixed (Infestation mixte)", "precision": 0.8898, "recall": 0.8750, "f1": 0.8824, "support": 120}
        ]
    }

@app.get("/api/image/{folder}/{filename}")
def get_image_file(folder: str, filename: str):
    file_path = None
    if folder == "benchmark":
        file_path = GEN_DIR / "benchmark" / filename
    elif folder in ["ConvNeXt-Tiny", "MobileNetV3-ECA", "Swin-T", "EfficientNet-V2-S"]:
        file_path = GEN_DIR / folder / filename
    elif folder in ["Healthy", "Bug", "Honey", "Dubas"]:
        file_path = DATASET_PATH / folder / filename
    
    if file_path and file_path.exists():
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Image non trouvée")

# ------------------------------------------------------------------------------
# SERVICE DU FRONTEND VITE.JS (SPA)
# ------------------------------------------------------------------------------
DIST_DIR = BASE_DIR / "web-app" / "dist"
if DIST_DIR.exists():
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Endpoint API non trouvé")
        file_path = DIST_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        index_file = DIST_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        raise HTTPException(status_code=404, detail="Frontend non trouvé")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=True)
