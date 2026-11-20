"""
================================================================================
APPLICATION DESKTOP - DÉTECTION DU DUBAS DU PALMIER DATTIER
Projet Machine Learning & Vision par Ordinateur
================================================================================
Interface graphique locale ne nécessitant aucun serveur web.
Permet de charger une image et d'afficher le résultat instantanément.
================================================================================
"""

import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import joblib

from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray, rgb2hsv

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

class DubasClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🌴 Diagnostic du Dubas - Palmier Dattier")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        self.root.configure(bg="#f4f6f9")
        
        self.current_image_path = None
        self.tk_image = None
        
        # Chargement des modèles
        self.load_models()
        
        # Construction de l'interface
        self.create_widgets()
        
    def load_models(self):
        try:
            self.scaler = joblib.load(MODELS_DIR / "standard_scaler.joblib")
            self.selector = joblib.load(MODELS_DIR / "feature_selector.joblib")
            self.le = joblib.load(MODELS_DIR / "label_encoder.joblib")
            self.models = {
                "Random Forest": joblib.load(MODELS_DIR / "random_optimal_model.joblib"),
                "SVM (Noyau RBF)": joblib.load(MODELS_DIR / "svm_optimal_model.joblib"),
                "K-Nearest Neighbors": joblib.load(MODELS_DIR / "k-nearest_optimal_model.joblib")
            }
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger les modèles : {e}")
            self.root.destroy()
            
    def create_widgets(self):
        # En-tête
        header = tk.Frame(self.root, bg="#1b5e20", pady=15)
        header.pack(fill="x")
        
        lbl_title = tk.Label(header, text="🌴 Classification des Feuilles de Palmier Dattier", 
                             font=("Segoe UI", 16, "bold"), fg="white", bg="#1b5e20")
        lbl_title.pack()
        lbl_sub = tk.Label(header, text="Diagnostic automatisé du ravageur Dubas (Ommatissus lybicus)", 
                           font=("Segoe UI", 10), fg="#c8e6c9", bg="#1b5e20")
        lbl_sub.pack()
        
        # Conteneur principal
        main_pane = tk.Frame(self.root, bg="#f4f6f9", padx=20, pady=15)
        main_pane.pack(fill="both", expand=True)
        
        # Panneau gauche : Image & Boutons
        left_box = tk.LabelFrame(main_pane, text=" Image à Analyser ", font=("Segoe UI", 11, "bold"),
                                 bg="white", padx=15, pady=15, relief="groove")
        left_box.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        self.canvas_img = tk.Label(left_box, bg="#e0e0e0", text="Aucune image sélectionnée\n\nCliquez sur 'Parcourir...' ci-dessous",
                                   font=("Segoe UI", 11), width=35, height=15)
        self.canvas_img.pack(fill="both", expand=True, pady=(0, 10))
        
        btn_frame = tk.Frame(left_box, bg="white")
        btn_frame.pack(fill="x")
        
        btn_load = tk.Button(btn_frame, text="📁 Parcourir une photo...", command=self.load_image,
                             font=("Segoe UI", 10, "bold"), bg="#2e7d32", fg="white", padx=12, pady=6, cursor="hand2")
        btn_load.pack(side="left", padx=5)
        
        btn_classify = tk.Button(btn_frame, text="⚡ Analyser & Prédire", command=self.classify_image,
                                 font=("Segoe UI", 10, "bold"), bg="#1565c0", fg="white", padx=12, pady=6, cursor="hand2")
        btn_classify.pack(side="right", padx=5)
        
        # Panneau droit : Paramètres et Résultats
        right_box = tk.LabelFrame(main_pane, text=" Résultat du Diagnostic ", font=("Segoe UI", 11, "bold"),
                                  bg="white", padx=15, pady=15, relief="groove")
        right_box.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        # Choix du modèle
        mod_frame = tk.Frame(right_box, bg="white")
        mod_frame.pack(fill="x", pady=(0, 15))
        tk.Label(mod_frame, text="Modèle d'IA :", font=("Segoe UI", 10, "bold"), bg="white").pack(side="left", padx=(0, 10))
        self.model_var = tk.StringVar(value="Random Forest")
        model_cb = ttk.Combobox(mod_frame, textvariable=self.model_var, values=list(self.models.keys()), state="readonly")
        model_cb.pack(side="left", fill="x", expand=True)
        
        # Affichage du statut prédit
        self.result_badge = tk.Label(right_box, text="En attente d'image...", font=("Segoe UI", 14, "bold"),
                                     bg="#e0e0e0", fg="#555", pady=12, relief="ridge")
        self.result_badge.pack(fill="x", pady=5)
        
        self.lbl_conf = tk.Label(right_box, text="Confiance : -- %", font=("Segoe UI", 11, "bold"), bg="white", fg="#333")
        self.lbl_conf.pack(pady=5)
        
        # Probabilités
        tk.Label(right_box, text="Détail des Probabilités par Classe :", font=("Segoe UI", 10, "bold"), bg="white").pack(anchor="w", pady=(10, 5))
        self.probas_frame = tk.Frame(right_box, bg="white")
        self.probas_frame.pack(fill="x", pady=5)
        
        self.proba_labels = {}
        for cls in self.le.classes_:
            row = tk.Frame(self.probas_frame, bg="white")
            row.pack(fill="x", pady=2)
            lbl_name = tk.Label(row, text=f"• {cls.split(' ')[0]} :", width=12, anchor="w", font=("Segoe UI", 9), bg="white")
            lbl_name.pack(side="left")
            pbar = ttk.Progressbar(row, orient="horizontal", mode="determinate", length=140)
            pbar.pack(side="left", padx=5)
            lbl_val = tk.Label(row, text="0.0 %", font=("Segoe UI", 9, "bold"), width=7, anchor="e", bg="white")
            lbl_val.pack(side="right")
            self.proba_labels[cls] = (pbar, lbl_val)
            
        # Recommandations
        tk.Label(right_box, text="Recommandation Agronomique :", font=("Segoe UI", 10, "bold"), bg="white").pack(anchor="w", pady=(15, 5))
        self.txt_recom = tk.Label(right_box, text="Veuillez sélectionner et classifier une image.", 
                                  font=("Segoe UI", 9), bg="#fffde7", fg="#333", justify="left", 
                                  wraplength=350, padx=10, pady=10, relief="solid", bd=1)
        self.txt_recom.pack(fill="both", expand=True)
        
    def load_image(self):
        fpath = filedialog.askopenfilename(
            title="Choisir une photo de feuille de palmier",
            filetypes=[("Images", "*.jpg *.jpeg *.png")]
        )
        if fpath:
            self.current_image_path = fpath
            img = Image.open(fpath)
            img.thumbnail((320, 320))
            self.tk_image = ImageTk.PhotoImage(img)
            self.canvas_img.configure(image=self.tk_image, text="")
            
    def classify_image(self):
        if not self.current_image_path:
            messagebox.showwarning("Attention", "Veuillez d'abord charger une photo.")
            return
            
        try:
            img = Image.open(self.current_image_path).convert('RGB').resize((128, 128))
            arr = np.array(img)
            gray = rgb2gray(arr)
            gray_uint8 = (gray * 255).astype(np.uint8)
            hsv = rgb2hsv(arr)
            
            hf = hog(gray, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), feature_vector=True)
            lbp = local_binary_pattern(gray_uint8, P=8, R=1, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
            h_hist, _ = np.histogram(hsv[:, :, 0], bins=16, range=(0, 1), density=True)
            s_hist, _ = np.histogram(hsv[:, :, 1], bins=16, range=(0, 1), density=True)
            v_hist, _ = np.histogram(hsv[:, :, 2], bins=16, range=(0, 1), density=True)
            rgb_norm = arr / 255.0
            cm = [
                float(np.mean(rgb_norm[:, :, 0])), float(np.std(rgb_norm[:, :, 0])),
                float(np.mean(rgb_norm[:, :, 1])), float(np.std(rgb_norm[:, :, 1])),
                float(np.mean(rgb_norm[:, :, 2])), float(np.std(rgb_norm[:, :, 2])),
                float(np.mean(hsv[:, :, 1])), float(np.std(hsv[:, :, 1])),
                float(np.mean(hsv[:, :, 0])), float(np.std(hsv[:, :, 0]))
            ]
            vec = np.hstack([hf, lbp_hist, h_hist, s_hist, v_hist, cm]).reshape(1, -1)
            
            vec_scaled = self.scaler.transform(vec)
            vec_sel = self.selector.transform(vec_scaled)
            
            model = self.models[self.model_var.get()]
            pred_idx = model.predict(vec_sel)[0]
            pred_label = self.le.inverse_transform([pred_idx])[0]
            
            probas = model.predict_proba(vec_sel)[0] if hasattr(model, 'predict_proba') else [0.25]*4
            proba_dict = dict(zip(self.le.classes_, probas))
            
            # Couleurs et textes
            color_map = {
                'Healthy (Saine)': ('#2e7d32', '#e8f5e9'),
                'Bug (Insectes seuls)': ('#f57f17', '#fffde7'),
                'Honey (Miellat)': ('#e65100', '#fff3e0'),
                'Mixed (Infestation mixte)': ('#c62828', '#ffebee')
            }
            fg_col, bg_col = color_map.get(pred_label, ('#333', '#eee'))
            
            self.result_badge.configure(text=f"Résultat : {pred_label}", fg=fg_col, bg=bg_col)
            conf = proba_dict[pred_label] * 100
            self.lbl_conf.configure(text=f"Indice de Confiance : {conf:.1f} %", fg=fg_col)
            
            for cls, prob in proba_dict.items():
                pbar, lbl_val = self.proba_labels[cls]
                pbar['value'] = prob * 100
                lbl_val.configure(text=f"{prob*100:.1f} %")
                
            recom_map = {
                'Healthy (Saine)': "Feuille saine. Aucun traitement chimique requis. Maintenir une surveillance périodique dans les palmeraies.",
                'Bug (Insectes seuls)': "Présence d'insectes vivants. Poser des pièges englués et privilégier la lutte biologique (coccinelles/chrysopes).",
                'Honey (Miellat)': "Dépôts de miellat visqueux. Effectuer un lavage foliaire sous pression pour empêcher la formation de fumagine.",
                'Mixed (Infestation mixte)': "Infestation critique sévère. Intervention curative urgente recommandée pour protéger la récolte de dattes."
            }
            self.txt_recom.configure(text=recom_map.get(pred_label, "Aucune information."), bg=bg_col)
            
        except Exception as e:
            messagebox.showerror("Erreur de Classification", f"Une anomalie est survenue : {e}")

if __name__ == '__main__':
    root = tk.Tk()
    app = DubasClassifierApp(root)
    root.mainloop()
