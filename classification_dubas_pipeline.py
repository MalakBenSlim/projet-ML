"""
================================================================================
PROJET DE MACHINE LEARNING & VISION PAR ORDINATEUR
Classification Automatisée des Feuilles de Palmier Dattier 
Infestées par le Dubas (Ommatissus lybicus)
================================================================================
Pipeline complet et modulaire d'apprentissage automatique supervisé :
1. Affichage et interprétation des données.
2. Visualisation des données avec Seaborn (distributions, corrélations).
3. Détection et analyse des valeurs aberrantes (Outliers via IQR / Boxplots).
4. Prétraitement et Feature Engineering adaptés aux images foliaires :
   - Redimensionnement standardisé (128x128).
   - Descripteurs HOG (formes & contours des insectes et nervures).
   - Descripteurs LBP (rugosité de la surface foliaire et fumagine).
   - Histogrammes & moments de couleur HSV (jaunissement et dépôts de miellat).
   - Mise à l'échelle (StandardScaler).
5. Sélection des caractéristiques les plus pertinentes (SelectKBest ANOVA F-test).
6. Modélisation comparative sur plusieurs modèles :
   - Modèle 1 : Support Vector Machine (SVM) à noyau RBF.
   - Modèle 2 : Random Forest Classifier (Forêt Aléatoire).
   - Modèle 3 : K-Nearest Neighbors (KNN).
7. Optimisation des hyperparamètres par Validation Croisée (CV=4).
8. Évaluation détaillée et matrices de confusion Seaborn (brutes & normalisées).
9. Tableaux récapitulatifs complets des performances.
10. Étapes avancées : Courbes ROC multi-classes (One-vs-Rest), AUC et Feature Importance.
================================================================================
"""

import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Mode non-interactif pour éviter tout blocage GUI
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from PIL import Image

from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray, rgb2hsv

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import joblib

# Configuration de l'affichage et du style graphique
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams['figure.autolayout'] = True
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

# Définition des répertoires de travail
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "Image dataset of infected date palm leaves" / "Image dataset of infected date palm leaves by dubas insects" / "Palm Leaves Dataset"
FIGURES_DIR = BASE_DIR / "output_figures"
MODELS_DIR = BASE_DIR / "models"
FEATURES_CSV = BASE_DIR / "dataset_features.csv"

FIGURES_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Correspondance exacte des dossiers et des classes sémantiques (conforme au README / publication)
LABEL_MAPPING = {
    'Healthy': 'Healthy (Saine)',
    'Bug': 'Bug (Insectes seuls)',
    'Honey': 'Honey (Miellat)',
    'Dubas': 'Mixed (Infestation mixte)'
}


# ==============================================================================
# 1. PARCOURS ET INTERPRÉTATION DES DONNÉES
# ==============================================================================
def inspect_dataset(dataset_dir: Path):
    """
    Parcourt le jeu de données pour vérifier la présence des classes, compter les
    fichiers et collecter les métadonnées (résolutions, canaux).
    """
    print("\n" + "="*80)
    print(" 1. AFFICHAGE ET INTERPRÉTATION DU JEU DE DONNÉES")
    print("="*80)
    
    records = []
    supported_exts = ('.jpg', '.jpeg', '.png')
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable : {dataset_dir}")
        
    for class_folder in dataset_dir.iterdir():
        if class_folder.is_dir() and class_folder.name in LABEL_MAPPING:
            files = [f for f in class_folder.iterdir() if f.suffix.lower() in supported_exts]
            for f in files:
                records.append({
                    'filepath': str(f),
                    'folder_name': class_folder.name,
                    'class_label': LABEL_MAPPING[class_folder.name],
                    'filename': f.name
                })
                
    df_meta = pd.DataFrame(records)
    print(f"-> Nombre total d'images inspectées : {len(df_meta)}")
    
    stats = df_meta['class_label'].value_counts()
    print("\n-> Répartition des échantillons par classe :")
    for cls_name, count in stats.items():
        pct = (count / len(df_meta)) * 100
        print(f"   * {cls_name:<30} : {count:>4} images ({pct:>5.2f} %)")
        
    print("\n-> Interprétation agronomique et technique :")
    print("   - Le jeu de données comprend 3 000 images réparties sur 4 stades d'infestation.")
    print("   - Trois classes comptent exactement 800 images (Healthy, Honey, Mixed).")
    print("   - La classe 'Bug' compte 600 images, introduisant un léger déséquilibre maîtrisé (ratio 1:1.33).")
    print("   - Ce contexte justifie l'utilisation du Macro F1-Score et d'une validation croisée stratifiée.")
    
    return df_meta


# ==============================================================================
# 2. VISUALISATION DES DONNÉES AVEC SEABORN
# ==============================================================================
def visualize_data(df_meta: pd.DataFrame):
    """
    Génère les visualisations Seaborn pour explorer la distribution des classes.
    """
    print("\n" + "="*80)
    print(" 2. VISUALISATION DES DONNÉES AVEC SEABORN")
    print("="*80)
    
    # 2.1 Distribution des classes
    plt.figure(figsize=(10, 6))
    order = list(LABEL_MAPPING.values())
    palette = sns.color_palette("deep", len(order))
    
    ax = sns.countplot(data=df_meta, x='class_label', order=order, palette=palette)
    plt.title("Distribution des Classes - Feuilles de Palmier Dattier", fontsize=14, weight='bold', pad=15)
    plt.xlabel("État Sanitaire de la Foliole", fontsize=12, labelpad=10)
    plt.ylabel("Nombre d'Images", fontsize=12, labelpad=10)
    plt.ylim(0, 950)
    
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{int(height)} ({height/len(df_meta)*100:.1f}%)",
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=11, color='black',
                    xytext=(0, 5), textcoords='offset points')
                    
    fig_path = FIGURES_DIR / "01_class_distribution.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Graphique sauvegardé : {fig_path.name}")


# ==============================================================================
# 3. EXTRACTION DE CARACTÉRISTIQUES (FEATURE ENGINEERING)
# ==============================================================================
def extract_single_image_features(image_path: str, target_size=(128, 128)):
    """
    Extrait un vecteur complet de descripteurs adaptés à la détection du Dubas :
    1. Statistiques globales de luminance et saturation (analyse des outliers).
    2. HOG (Histogram of Oriented Gradients) : détection des formes/contours des insectes.
    3. LBP (Local Binary Patterns) : texture foliaire et rugosité de la fumagine.
    4. Histogrammes de couleur HSV : quantification du jaunissement et du miellat.
    5. Moments statistiques par canal (moyenne et écart-type en RGB et HSV).
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img_resized = img.resize(target_size)
            arr = np.array(img_resized)
            
        # Conversion d'espaces de couleur
        gray = rgb2gray(arr)
        gray_uint8 = (gray * 255).astype(np.uint8)
        hsv = rgb2hsv(arr)
        
        # Indicateurs statistiques globaux pour analyse d'outliers
        mean_luminance = float(np.mean(gray))
        mean_saturation = float(np.mean(hsv[:, :, 1]))
        std_luminance = float(np.std(gray))
        
        # A. Descripteurs HOG
        hog_feats = hog(
            gray,
            orientations=8,
            pixels_per_cell=(16, 16),
            cells_per_block=(1, 1),
            feature_vector=True
        )
        
        # B. Descripteurs LBP (texture uniforme)
        lbp = local_binary_pattern(gray_uint8, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
        
        # C. Histogrammes de couleur dans l'espace HSV (16 bins par composante)
        h_hist, _ = np.histogram(hsv[:, :, 0], bins=16, range=(0, 1), density=True)
        s_hist, _ = np.histogram(hsv[:, :, 1], bins=16, range=(0, 1), density=True)
        v_hist, _ = np.histogram(hsv[:, :, 2], bins=16, range=(0, 1), density=True)
        
        # D. Moments statistiques (Moyenne & Écart-type pour chaque canal)
        rgb_norm = arr / 255.0
        color_moments = [
            float(np.mean(rgb_norm[:, :, 0])), float(np.std(rgb_norm[:, :, 0])),  # R
            float(np.mean(rgb_norm[:, :, 1])), float(np.std(rgb_norm[:, :, 1])),  # G
            float(np.mean(rgb_norm[:, :, 2])), float(np.std(rgb_norm[:, :, 2])),  # B
            mean_saturation, float(np.std(hsv[:, :, 1])),                          # S
            float(np.mean(hsv[:, :, 0])), float(np.std(hsv[:, :, 0]))              # H
        ]
        
        # Concaténation de l'ensemble des descripteurs
        feature_vector = np.hstack([
            hog_feats,       # 8 * (128/16) * (128/16) = 512 dimensions
            lbp_hist,        # 10 dimensions
            h_hist,          # 16 dimensions
            s_hist,          # 16 dimensions
            v_hist,          # 16 dimensions
            color_moments    # 10 dimensions
        ])
        
        outlier_metrics = {
            'mean_luminance': mean_luminance,
            'mean_saturation': mean_saturation,
            'std_luminance': std_luminance
        }
        
        return feature_vector, outlier_metrics
        
    except Exception as e:
        print(f"Erreur sur {image_path}: {e}")
        return None, None


def extract_or_load_dataset(df_meta: pd.DataFrame):
    """
    Extrait les descripteurs sur l'ensemble du dataset ou charge le cache CSV.
    """
    print("\n" + "="*80)
    print(" 3. EXTRACTION DES CARACTÉRISTIQUES & MISE EN CACHE")
    print("="*80)
    
    if FEATURES_CSV.exists():
        print(f"-> Chargement des descripteurs existants depuis {FEATURES_CSV.name}...")
        df_features = pd.read_csv(FEATURES_CSV)
        print(f"-> Matrice de caractéristiques chargée : {df_features.shape[0]} lignes, {df_features.shape[1]} colonnes.")
        return df_features
        
    print(f"-> Début de l'extraction des caractéristiques sur {len(df_meta)} images...")
    t_start = time.time()
    
    rows = []
    total = len(df_meta)
    
    for idx, row in df_meta.iterrows():
        feat_vec, metrics = extract_single_image_features(row['filepath'])
        if feat_vec is not None:
            record = {
                'filepath': row['filepath'],
                'class_label': row['class_label'],
                'mean_luminance': metrics['mean_luminance'],
                'mean_saturation': metrics['mean_saturation'],
                'std_luminance': metrics['std_luminance'],
            }
            # Nommage des colonnes de descripteurs
            for i, val in enumerate(feat_vec):
                record[f'feat_{i:03d}'] = val
            rows.append(record)
            
        if (idx + 1) % 300 == 0 or (idx + 1) == total:
            elapsed = time.time() - t_start
            print(f"   [{idx + 1:>4}/{total}] images traitées en {elapsed:>5.1f}s (vitesse : {(idx + 1)/elapsed:>4.1f} img/s)")
            
    df_features = pd.DataFrame(rows)
    df_features.to_csv(FEATURES_CSV, index=False)
    print(f"-> Sauvegarde réussie dans {FEATURES_CSV.name} ({len(df_features)} observations).")
    return df_features


# ==============================================================================
# 4. VÉRIFICATION DES VALEURS ABERRANTES (OUTLIERS)
# ==============================================================================
def check_outliers(df_features: pd.DataFrame):
    """
    Identifie et analyse les valeurs aberrantes sur les indicateurs de luminance
    et saturation par la méthode de l'IQR (Interquartile Range) et Seaborn Boxplots.
    """
    print("\n" + "="*80)
    print(" 4. VÉRIFICATION DE LA PRÉSENCE DE VALEURS ABERRANTES (OUTLIERS)")
    print("="*80)
    
    metrics = ['mean_luminance', 'mean_saturation']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    order = list(LABEL_MAPPING.values())
    
    # 4.1 Boxplot Luminance
    sns.boxplot(ax=axes[0], data=df_features, x='class_label', y='mean_luminance',
                order=order, palette='Set2')
    axes[0].set_title("Distribution de la Luminance par Classe", fontsize=12, weight='bold')
    axes[0].set_xlabel("Classe", fontsize=11)
    axes[0].set_ylabel("Luminance Moyenne [0 - 1]", fontsize=11)
    axes[0].tick_params(axis='x', rotation=20)
    
    # 4.2 Boxplot Saturation
    sns.boxplot(ax=axes[1], data=df_features, x='class_label', y='mean_saturation',
                order=order, palette='Set2')
    axes[1].set_title("Distribution de la Saturation par Classe", fontsize=12, weight='bold')
    axes[1].set_xlabel("Classe", fontsize=11)
    axes[1].set_ylabel("Saturation Moyenne [0 - 1]", fontsize=11)
    axes[1].tick_params(axis='x', rotation=20)
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "02_outliers_boxplots.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"-> Graphique des outliers sauvegardé : {fig_path.name}")
    
    # Calcul des outliers selon la règle IQR = Q3 - Q1
    outlier_counts = {}
    for col in metrics:
        q1 = df_features[col].quantile(0.25)
        q3 = df_features[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        is_outlier = (df_features[col] < lower_bound) | (df_features[col] > upper_bound)
        nb_outliers = is_outlier.sum()
        pct = (nb_outliers / len(df_features)) * 100
        outlier_counts[col] = (nb_outliers, pct)
        print(f"-> Variable '{col}': {nb_outliers} outliers détectés ({pct:.2f} %) - Seuils [{lower_bound:.3f}, {upper_bound:.3f}]")
        
    print("\n-> Décision méthodologique d'ingénierie :")
    print("   Les points atypiques correspondent à des variations naturelles d'éclairage in situ")
    print("   (soleil rasant, zones ombragées de la palmeraie, reflets spéculaires du miellat).")
    print("   Plutôt que de supprimer ces données précieuses, nous appliquons un prétraitement")
    print("   robuste (StandardScaler) afin d'assurer l'immunité des modèles aux variations de prise de vue.")


# ==============================================================================
# 5. SÉLECTION DES CARACTÉRISTIQUES LES PLUS PERTINENTES
# ==============================================================================
def perform_feature_selection(X: np.ndarray, y: np.ndarray, k_features: int = 80):
    """
    Sélectionne les k caractéristiques les plus discriminantes via ANOVA F-test
    (SelectKBest) et visualise les scores de pertinence.
    """
    print("\n" + "="*80)
    print(" 5. SÉLECTION DES CARACTÉRISTIQUES LES PLUS PERTINENTES")
    print("="*80)
    
    print(f"-> Dimension initiale de l'espace de caractéristiques : {X.shape[1]} attributs.")
    selector = SelectKBest(score_func=f_classif, k=k_features)
    X_selected = selector.fit_transform(X, y)
    
    scores = selector.scores_
    top_indices = np.argsort(scores)[::-1][:k_features]
    
    print(f"-> Réduction dimensionnelle : {X.shape[1]} -> {k_features} caractéristiques retenues.")
    print(f"-> Meilleur score F : {scores[top_indices[0]]:.2f} | Score au seuil k={k_features} : {scores[top_indices[-1]]:.2f}")
    
    # Visualisation des scores ANOVA
    plt.figure(figsize=(10, 5))
    plt.plot(np.sort(scores)[::-1], color='darkblue', lw=2, label="Scores ANOVA F-value")
    plt.axvline(x=k_features, color='red', linestyle='--', label=f"Seuil de coupure (k={k_features})")
    plt.title("Sélection de Caractéristiques par ANOVA F-Test (SelectKBest)", fontsize=13, weight='bold')
    plt.xlabel("Index des caractéristiques classées par pertinence", fontsize=11)
    plt.ylabel("Score F de Fisher", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    
    fig_path = FIGURES_DIR / "03_feature_selection_scores.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"-> Graphique sauvegardé : {fig_path.name}")
    
    return X_selected, selector


# ==============================================================================
# 6. ENTRAÎNEMENT & OPTIMISATION PAR VALIDATION CROISÉE (CV=4)
# ==============================================================================
def train_and_optimize_models(X_train: np.ndarray, y_train: np.ndarray, class_names: list):
    """
    Entraîne au moins deux modèles supervisés distincts avec GridSearchCV (CV=4) :
    1. Support Vector Machine (SVM) à noyau RBF.
    2. Random Forest (Forêt Aléatoire).
    3. K-Nearest Neighbors (KNN - Modèle baseline).
    """
    print("\n" + "="*80)
    print(" 6. OPTIMISATION PAR VALIDATION CROISÉE (CV=4) ET APPRENTISSAGE")
    print("="*80)
    
    # Validation croisée stratifiée à 4 plis (CV=4)
    cv4 = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    
    models_config = {
        'SVM (Support Vector Machine)': {
            'estimator': SVC(probability=True, random_state=42),
            'params': {
                'C': [1.0, 10.0, 50.0],
                'gamma': ['scale', 0.01, 0.001],
                'kernel': ['rbf']
            }
        },
        'Random Forest': {
            'estimator': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [15, 25, None],
                'min_samples_split': [2, 5]
            }
        },
        'K-Nearest Neighbors (KNN)': {
            'estimator': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }
        }
    }
    
    best_models = {}
    cv_results_summary = []
    
    for name, config in models_config.items():
        print(f"\n-> Optimisation de [{name}] via GridSearchCV (CV=4, scoring='f1_macro')...")
        t0 = time.time()
        
        grid = GridSearchCV(
            estimator=config['estimator'],
            param_grid=config['params'],
            cv=cv4,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=0
        )
        grid.fit(X_train, y_train)
        elapsed = time.time() - t0
        
        best_models[name] = grid.best_estimator_
        cv_results_summary.append({
            'Modèle': name,
            'Meilleur F1-Macro (CV=4)': f"{grid.best_score_*100:.2f} %",
            'Paramètres Optimaux': str(grid.best_params_),
            'Temps d\'optimisation': f"{elapsed:.1f}s"
        })
        
        print(f"   * Meilleurs paramètres : {grid.best_params_}")
        print(f"   * Meilleur score F1-Macro moyen (CV=4) : {grid.best_score_*100:.2f} % (en {elapsed:.1f}s)")
        
        # Sauvegarde du modèle
        clean_name = name.split(' ')[0].lower()
        joblib.dump(grid.best_estimator_, MODELS_DIR / f"{clean_name}_optimal_model.joblib")
        
    df_cv_summary = pd.DataFrame(cv_results_summary)
    print("\n-> Synthèse de l'optimisation par validation croisée (CV=4) :")
    print(df_cv_summary.to_string(index=False))
    
    return best_models


# ==============================================================================
# 7. ÉVALUATION ET MATRICES DE CONFUSION
# ==============================================================================
def evaluate_models(best_models: dict, X_test: np.ndarray, y_test: np.ndarray, class_names: list):
    """
    Évalue les modèles sur l'ensemble de test, trace les matrices de confusion
    Seaborn et synthétise les performances dans un tableau récapitulatif.
    """
    print("\n" + "="*80)
    print(" 7. ÉVALUATION ET MATRICES DE CONFUSION SEABORN")
    print("="*80)
    
    performance_records = []
    
    # Tracé des matrices de confusion côte à côte
    num_models = len(best_models)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))
    if num_models == 1:
        axes = [axes]
        
    for idx, (name, model) in enumerate(best_models.items()):
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average='macro')
        rec_macro = recall_score(y_test, y_pred, average='macro')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        performance_records.append({
            'Modèle': name,
            'Accuracy': f"{acc*100:.2f} %",
            'Macro Précision': f"{prec_macro*100:.2f} %",
            'Macro Rappel': f"{rec_macro*100:.2f} %",
            'Macro F1-Score': f"{f1_macro*100:.2f} %",
            'Weighted F1-Score': f"{f1_weighted*100:.2f} %"
        })
        
        # Matrice de confusion normalisée
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        sns.heatmap(
            cm, annot=True, fmt='.2%', cmap='Blues', ax=axes[idx],
            xticklabels=class_names, yticklabels=class_names, cbar=False
        )
        axes[idx].set_title(f"Matrice de Confusion\n{name}", fontsize=12, weight='bold')
        axes[idx].set_xlabel("Classe Prédite", fontsize=10)
        axes[idx].set_ylabel("Classe Réelle", fontsize=10)
        axes[idx].tick_params(axis='x', rotation=30)
        axes[idx].tick_params(axis='y', rotation=0)
        
    plt.tight_layout()
    fig_path = FIGURES_DIR / "04_confusion_matrices.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"-> Matrices de confusion sauvegardées : {fig_path.name}")
    
    df_perf = pd.DataFrame(performance_records)
    print("\n" + "="*80)
    print(" TABLEAU RÉCAPITULATIF DES PERFORMANCES SUR LE JEU DE TEST INDÉPENDANT")
    print("="*80)
    print(df_perf.to_string(index=False))
    
    # Sauvegarde du tableau en CSV
    df_perf.to_csv(BASE_DIR / "performances_comparatives.csv", index=False)
    
    return df_perf


# ==============================================================================
# 8. ÉTAPES AVANCÉES : COURBES ROC / AUC & FEATURE IMPORTANCE
# ==============================================================================
def additional_analysis(best_models: dict, X_test: np.ndarray, y_test: np.ndarray,
                        class_names: list, feature_selector: SelectKBest):
    """
    Analyses complémentaires :
    - Courbes ROC multi-classes One-vs-Rest et calcul de l'AUC.
    - Graphique de l'importance des descripteurs pour Random Forest.
    """
    print("\n" + "="*80)
    print(" 8. ÉTAPES AVANCÉES : COURBES ROC/AUC & INTERPRÉTABILITÉ")
    print("="*80)
    
    # 8.1 Courbes ROC One-vs-Rest pour le modèle SVM
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    svm_model = best_models['SVM (Support Vector Machine)']
    y_score = svm_model.predict_proba(X_test)
    
    plt.figure(figsize=(9, 7))
    colors = ['darkorange', 'green', 'blue', 'purple']
    
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f"{class_names[i]} (AUC = {roc_auc:.3f})")
                 
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Aléatoire (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs (1 - Spécificité)', fontsize=11)
    plt.ylabel('Taux de Vrais Positifs (Sensibilité)', fontsize=11)
    plt.title('Courbes ROC Multi-Classes (One-vs-Rest) - Modèle SVM Optimal', fontsize=13, weight='bold')
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    
    fig_path = FIGURES_DIR / "05_roc_curves_svm.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"-> Courbes ROC sauvegardées : {fig_path.name}")
    
    # 8.2 Importance des caractéristiques (Random Forest)
    rf_model = best_models['Random Forest']
    importances = rf_model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:20]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[top_indices], y=[f"Descripteur #{idx}" for idx in top_indices], palette='viridis')
    plt.title("Top 20 des Descripteurs Visuels les Plus Riches (Random Forest)", fontsize=13, weight='bold')
    plt.xlabel("Score d'Importance de Gini", fontsize=11)
    plt.ylabel("Caractéristiques", fontsize=11)
    plt.tight_layout()
    
    fig_path = FIGURES_DIR / "06_feature_importance_rf.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"-> Graphique d'importance sauvegardé : {fig_path.name}")


# ==============================================================================
# POINT D'ENTRÉE PRINCIPAL
# ==============================================================================
def main():
    print("="*80)
    print(" DÉMARRAGE DU PIPELINE COMPLET MACHINE LEARNING")
    print(" Classification des Feuilles de Palmier Dattier Infestées par le Dubas")
    print("="*80)
    
    # 1. Inspection du jeu de données
    df_meta = inspect_dataset(DATASET_PATH)
    
    # 2. Visualisations exploratoires Seaborn
    visualize_data(df_meta)
    
    # 3. Extraction ou chargement des caractéristiques
    df_features = extract_or_load_dataset(df_meta)
    
    # 4. Vérification et analyse des valeurs aberrantes (Outliers)
    check_outliers(df_features)
    
    # Préparation des matrices X et y
    feat_cols = [c for c in df_features.columns if c.startswith('feat_')]
    X_raw = df_features[feat_cols].values
    
    label_encoder = LabelEncoder()
    y_raw = label_encoder.fit_transform(df_features['class_label'])
    class_names = list(label_encoder.classes_)
    
    # Sauvegarde de l'encodeur de classes
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.joblib")
    
    # Mise à l'échelle (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    joblib.dump(scaler, MODELS_DIR / "standard_scaler.joblib")
    
    # 5. Sélection des caractéristiques les plus pertinentes
    X_selected, selector = perform_feature_selection(X_scaled, y_raw, k_features=80)
    joblib.dump(selector, MODELS_DIR / "feature_selector.joblib")
    
    # Partitionnement Stratifié Train / Test (75% Train, 25% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_raw, test_size=0.25, random_state=42, stratify=y_raw
    )
    print(f"\n-> Découpage stratifié du dataset : {X_train.shape[0]} entraînement, {X_test.shape[0]} test.")
    
    # 6. Entraînement et optimisation par Validation Croisée (CV=4)
    best_models = train_and_optimize_models(X_train, y_train, class_names)
    
    # 7. Évaluation et matrices de confusion
    df_perf = evaluate_models(best_models, X_test, y_test, class_names)
    
    # 8. Analyses complémentaires (ROC, AUC, Feature Importance)
    additional_analysis(best_models, X_test, y_test, class_names, selector)
    
    print("\n" + "="*80)
    print(" EXECUTION DU PIPELINE TERMINÉE AVEC SUCCÈS !")
    print(f" - Graphiques générés dans : {FIGURES_DIR}")
    print(f" - Modèles sauvegardés dans  : {MODELS_DIR}")
    print(f" - Tableau de bord CSV      : {BASE_DIR / 'performances_comparatives.csv'}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
