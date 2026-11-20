"""
================================================================================
APPLICATION DESKTOP - DÉTECTION DU DUBAS DU PALMIER DATTIER PAR DEEP LEARNING
================================================================================
Interface graphique locale Tkinter :
- Utilise les 4 modèles Deep Learning (ConvNeXt-Tiny, MobileNetV3-ECA, Swin-T, EfficientNet-V2-S).
- Permet de charger n'importe quelle photo de foliole et d'afficher le résultat instantanément.
================================================================================
"""

import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms, models

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

CLASS_NAMES = ['Healthy (Saine)', 'Bug (Insectes seuls)', 'Honey (Miellat)', 'Mixed (Infestation mixte)']

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

class DubasClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🌴 Diagnostic du Dubas - Palmier Dattier (Deep Learning)")
        self.root.geometry("820x650")
        self.root.minsize(700, 550)
        
        self.current_image = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.load_models()
        self.create_widgets()
        
    def load_models(self):
        try:
            self.models = {}
            # ConvNeXt-Tiny
            m_conv = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
            bb_conv = nn.Sequential(m_conv.features, m_conv.avgpool, m_conv.classifier[0], m_conv.classifier[1])
            for p in bb_conv.parameters(): p.requires_grad_(False)
            bb_conv.eval()
            h_conv = DeepClassifierHead(768, 4)
            h_conv.load_state_dict(torch.load(MODELS_DIR / "convnext_tiny_head.pth", weights_only=True, map_location='cpu'))
            h_conv.eval()
            self.models["ConvNeXt-Tiny (Champion - 92.89%)"] = {'base': bb_conv, 'head': h_conv}

            # MobileNetV3-ECA
            m_mob = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
            bb_mob = nn.Sequential(m_mob.features, m_mob.avgpool, nn.Flatten(1))
            for p in bb_mob.parameters(): p.requires_grad_(False)
            bb_mob.eval()
            h_mob = DeepClassifierHead(960, 4)
            h_mob.load_state_dict(torch.load(MODELS_DIR / "mobilenetv3_eca_head.pth", weights_only=True, map_location='cpu'))
            h_mob.eval()
            self.models["MobileNetV3-ECA (Edge-AI - 90.89%)"] = {'base': bb_mob, 'head': h_mob}

            # Swin-T
            m_swin = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
            bb_swin = nn.Sequential(m_swin.features, m_swin.norm, m_swin.permute, m_swin.avgpool, m_swin.flatten)
            for p in bb_swin.parameters(): p.requires_grad_(False)
            bb_swin.eval()
            h_swin = DeepClassifierHead(768, 4)
            h_swin.load_state_dict(torch.load(MODELS_DIR / "swin_t_head.pth", weights_only=True, map_location='cpu'))
            h_swin.eval()
            self.models["Swin-T (Vision Transformer - 90.67%)"] = {'base': bb_swin, 'head': h_swin}

            # EfficientNet-V2-S
            m_eff = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
            bb_eff = nn.Sequential(m_eff.features, m_eff.avgpool, nn.Flatten(1))
            for p in bb_eff.parameters(): p.requires_grad_(False)
            bb_eff.eval()
            h_eff = DeepClassifierHead(1280, 4)
            h_eff.load_state_dict(torch.load(MODELS_DIR / "efficientnet_v2_s_head.pth", weights_only=True, map_location='cpu'))
            h_eff.eval()
            self.models["EfficientNet-V2-S (Multi-échelle - 88.89%)"] = {'base': bb_eff, 'head': h_eff}

        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger les modèles Deep Learning : {e}")
            self.models = {}

    def create_widgets(self):
        header = tk.Frame(self.root, bg="#2C3E50", height=60)
        header.pack(fill=tk.X)
        title = tk.Label(header, text="🌴 Diagnostic Phytosanitaire du Palmier Dattier", 
                         font=("Arial", 16, "bold"), fg="white", bg="#2C3E50")
        title.pack(pady=15)
        
        ctrl = tk.Frame(self.root, pady=10)
        ctrl.pack(fill=tk.X, padx=20)
        
        btn_open = ttk.Button(ctrl, text="📁 Ouvrir une image...", command=self.load_image)
        btn_open.pack(side=tk.LEFT, padx=5)
        
        lbl_mod = tk.Label(ctrl, text="Modèle Deep Learning :", font=("Arial", 10, "bold"))
        lbl_mod.pack(side=tk.LEFT, padx=(20, 5))
        
        self.model_var = tk.StringVar()
        model_names = list(self.models.keys())
        self.combo_model = ttk.Combobox(ctrl, textvariable=self.model_var, values=model_names, state="readonly", width=38)
        if model_names:
            self.combo_model.current(0)
        self.combo_model.pack(side=tk.LEFT, padx=5)
        
        btn_predict = ttk.Button(ctrl, text="⚡ Diagnostiquer", command=self.predict)
        btn_predict.pack(side=tk.LEFT, padx=10)
        
        content = tk.Frame(self.root)
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.img_label = tk.Label(content, text="Aucune image chargée\n\nCliquez sur 'Ouvrir une image' pour démarrer.", 
                                  bg="#ECF0F1", relief=tk.SOLID, bd=1, font=("Arial", 11))
        self.img_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        res_frame = tk.LabelFrame(content, text="Résultats du Diagnostic", font=("Arial", 12, "bold"), padx=15, pady=15)
        res_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.res_class = tk.Label(res_frame, text="En attente...", font=("Arial", 14, "bold"), fg="#7F8C8D")
        self.res_class.pack(pady=(10, 5))
        
        self.res_conf = tk.Label(res_frame, text="", font=("Arial", 11))
        self.res_conf.pack(pady=5)
        
        self.details_label = tk.Label(res_frame, text="", justify=tk.LEFT, font=("Arial", 10), wraplength=280)
        self.details_label.pack(pady=15, fill=tk.BOTH, expand=True)

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.JPG *.PNG")]
        )
        if not path:
            return
        try:
            img = Image.open(path)
            self.current_image = img
            display_img = img.copy()
            display_img.thumbnail((360, 360))
            self.tk_img = ImageTk.PhotoImage(display_img)
            self.img_label.config(image=self.tk_img, text="")
            self.res_class.config(text="Prêt pour analyse", fg="#2C3E50")
            self.res_conf.config(text="")
            self.details_label.config(text="")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur de lecture : {e}")

    def predict(self):
        if self.current_image is None:
            messagebox.showwarning("Attention", "Veuillez d'abord ouvrir une image.")
            return
        m_name = self.model_var.get()
        if m_name not in self.models:
            messagebox.showerror("Erreur", "Modèle non sélectionné.")
            return
        try:
            m = self.models[m_name]
            t_img = self.transform(self.current_image.convert('RGB')).unsqueeze(0)
            with torch.no_grad():
                feats = m['base'](t_img)
                if feats.dim() > 2:
                    feats = torch.flatten(feats, 1)
                logits = m['head'](feats)
                probas = torch.softmax(logits, dim=1).numpy()[0]
            
            pred_idx = int(np.argmax(probas))
            pred_class = CLASS_NAMES[pred_idx]
            conf = probas[pred_idx] * 100
            
            colors = {
                'Healthy (Saine)': '#2ECC71',
                'Bug (Insectes seuls)': '#E74C3C',
                'Honey (Miellat)': '#F39C12',
                'Mixed (Infestation mixte)': '#9B59B6'
            }
            c = colors.get(pred_class, "#2C3E50")
            self.res_class.config(text=pred_class, fg=c)
            self.res_conf.config(text=f"Confiance : {conf:.1f} %\nModèle : {m_name.split(' ')[0]}")
            
            details = "Probabilités détaillées :\n" + "-"*30 + "\n"
            for cls, p in zip(CLASS_NAMES, probas):
                details += f"• {cls.split(' ')[0]:10s} : {p*100:5.1f} %\n"
            self.details_label.config(text=details)
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'inférence : {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DubasClassifierApp(root)
    root.mainloop()
