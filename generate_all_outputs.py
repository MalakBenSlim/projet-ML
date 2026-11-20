"""
Script unifié de génération des résultats pour les 4 algorithmes Deep Learning :
1. ConvNeXt-Tiny (CNN Moderne SOTA)
2. MobileNetV3-ECA (Edge-AI Léger)
3. Swin-T (Vision Transformer)
4. EfficientNet-V2-S (CNN Multi-Échelle)

Génère tous les artefacts dans generation/ :
- generation/benchmark/ (benchmark_global.png, benchmark_global.csv, confusion_matrices_combined.png, dataset_distribution.png, dataset_samples.png)
- generation/<nom-du-modele>/ (confusion_matrix.png, per_class_metrics.png, resultats.csv)
"""
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent
DATASET_DIR = (BASE / 'Image dataset of infected date palm leaves'
               / 'Image dataset of infected date palm leaves by dubas insects'
               / 'Palm Leaves Dataset')
MODELS_DIR = BASE / 'models'
GEN_DIR    = BASE / 'generation'
BENCH_DIR  = GEN_DIR / 'benchmark'

LABEL_MAPPING = {
    'Healthy': 'Healthy (Saine)',
    'Bug':     'Bug (Insectes seuls)',
    'Honey':   'Honey (Miellat)',
    'Dubas':   'Mixed (Infestation mixte)',
}
CLASS_NAMES = ['Healthy (Saine)', 'Bug (Insectes seuls)', 'Honey (Miellat)', 'Mixed (Infestation mixte)']
CLASS_SHORT = ['Healthy', 'Bug', 'Honey', 'Mixed']
CLASS_COLORS = ['#2ECC71', '#E74C3C', '#F39C12', '#9B59B6']
SEED = 42
DEVICE = torch.device('cpu')

class ClassifierHead(nn.Module):
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

def get_dl_backbone(model_name):
    if model_name == 'ConvNeXt-Tiny':
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        backbone = nn.Sequential(m.features, m.avgpool, m.classifier[0], m.classifier[1])
        in_f = 768
    elif model_name == 'MobileNetV3-ECA':
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        backbone = nn.Sequential(m.features, m.avgpool, nn.Flatten(1))
        in_f = 960
    elif model_name == 'Swin-T':
        m = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        backbone = nn.Sequential(m.features, m.norm, m.permute, m.avgpool, m.flatten)
        in_f = 768
    elif model_name == 'EfficientNet-V2-S':
        m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        backbone = nn.Sequential(m.features, m.avgpool, nn.Flatten(1))
        in_f = 1280
    else:
        raise ValueError(f"Modèle inconnu: {model_name}")
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()
    return backbone, in_f

def main():
    print("=== Démarrage de la génération unifiée (4 Algorithmes Deep Learning) ===")
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    
    paths, labels = [], []
    for folder, lbl in LABEL_MAPPING.items():
        d = DATASET_DIR / folder
        if not d.exists(): continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                paths.append(f); labels.append(lbl)
    paths, labels = np.array(paths), np.array(labels)
    idx = np.arange(len(paths))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=labels, random_state=SEED)
    val, te  = train_test_split(tmp, test_size=0.50, stratify=labels[tmp], random_state=SEED)
    
    test_paths = paths[te]
    test_labels = labels[te]
    print(f"Jeu de test prêt : {len(test_paths)} échantillons.")

    # 1. Dataset Visualizations
    counts = [np.sum(labels == c) for c in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(CLASS_SHORT, counts, color=CLASS_COLORS, edgecolor='white', width=0.6)
    ax.set_ylabel("Nombre d'images", fontsize=12)
    ax.set_title("Distribution des classes (Total : 3 000 images)", fontsize=13, fontweight='bold')
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8, str(cnt), ha='center', fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4); ax.set_axisbelow(True)
    plt.tight_layout(); fig.savefig(BENCH_DIR / 'dataset_distribution.png', dpi=150); plt.close(fig)

    fig, axes = plt.subplots(4, 3, figsize=(10, 12))
    for row, (cls, short, col) in enumerate(zip(CLASS_NAMES, CLASS_SHORT, CLASS_COLORS)):
        cls_paths = paths[labels == cls]
        samples   = cls_paths[:3] if len(cls_paths) >= 3 else cls_paths
        for c in range(3):
            ax = axes[row][c]
            if c < len(samples):
                img = Image.open(samples[c]).convert('RGB')
                img.thumbnail((300, 300))
                ax.imshow(img)
                if c == 0:
                    ax.set_ylabel(short, fontsize=12, fontweight='bold', rotation=90, labelpad=10, color=col)
            ax.axis('off')
    fig.suptitle("Exemples d'images par classe sanitaire du palmier dattier", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout(); fig.savefig(BENCH_DIR / 'dataset_samples.png', dpi=150); plt.close(fig)

    # 2. Deep Learning Evaluation
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    class PalmDataset(torch.utils.data.Dataset):
        def __init__(self, ps, ls): self.ps=ps; self.ls=ls
        def __len__(self): return len(self.ps)
        def __getitem__(self, i):
            img = Image.open(self.ps[i]).convert('RGB')
            return tfm(img), CLASS_NAMES.index(self.ls[i])

    test_ds = PalmDataset(test_paths, test_labels)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)
    
    dl_configs = {
        'ConvNeXt-Tiny':     ('convnext_tiny_head.pth',     'CNN Moderne (SOTA)'),
        'MobileNetV3-ECA':   ('mobilenetv3_eca_head.pth',   'Edge-AI Léger (Inverted Res.)'),
        'Swin-T':            ('swin_t_head.pth',            'Vision Transformer (Shifted)'),
        'EfficientNet-V2-S': ('efficientnet_v2_s_head.pth', 'CNN Multi-Échelle (NAS)'),
    }
    
    benchmark_data = []
    dl_preds_dict = {}

    for name, (pth, paradigme) in dl_configs.items():
        print(f"Évaluation du modèle : {name}...")
        out_dir = GEN_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        backbone, in_f = get_dl_backbone(name)
        head = ClassifierHead(in_f)
        head.load_state_dict(torch.load(MODELS_DIR / pth, map_location=DEVICE, weights_only=True))
        head.eval()

        preds, trues = [], []
        with torch.no_grad():
            for X, y in loader:
                feats = backbone(X)
                out = head(feats)
                preds.extend(out.argmax(1).cpu().numpy())
                trues.extend(y.numpy())
        y_true, y_pred = np.array(trues), np.array(preds)
        dl_preds_dict[name] = (y_true, y_pred)
        
        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average='macro')
        benchmark_data.append({'Modèle': name, 'Paradigme': paradigme, 'Accuracy (%)': round(acc*100, 2), 'Macro F1 (%)': round(f1*100, 2)})

        # Matrice de confusion individuelle
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_SHORT, yticklabels=CLASS_SHORT, linewidths=.5, ax=ax)
        ax.set_xlabel('Prédiction', fontsize=12); ax.set_ylabel('Vérité terrain', fontsize=12)
        ax.set_title(f"Matrice de confusion — {name} ({acc*100:.2f}%)", fontsize=13, fontweight='bold')
        plt.tight_layout(); fig.savefig(out_dir / 'confusion_matrix.png', dpi=150); plt.close(fig)

        # Barplot métriques par classe
        rep = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
        x = np.arange(len(CLASS_NAMES)); w = 0.25
        fig, ax = plt.subplots(figsize=(11, 5))
        for i, (mn, col) in enumerate([('precision', '#3498DB'), ('recall', '#E74C3C'), ('f1-score', '#2ECC71')]):
            vals = [rep[c][mn] for c in CLASS_NAMES]
            ax.bar(x + i*w, vals, w, label=mn.capitalize(), color=col, alpha=0.85, edgecolor='white')
        ax.set_xticks(x + w); ax.set_xticklabels(CLASS_SHORT, fontsize=11)
        ax.set_ylim(0, 1.09); ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f"Métriques par classe — {name}", fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right'); ax.yaxis.grid(True, linestyle='--', alpha=0.5); ax.set_axisbelow(True)
        plt.tight_layout(); fig.savefig(out_dir / 'per_class_metrics.png', dpi=150); plt.close(fig)

        # CSV par modèle
        rows = [{'Classe': c, 'Precision': round(rep[c]['precision'], 4), 'Recall': round(rep[c]['recall'], 4), 'F1-Score': round(rep[c]['f1-score'], 4), 'Support': int(rep[c]['support'])} for c in CLASS_NAMES]
        rows.append({'Classe': 'GLOBAL (Macro)', 'Precision': round(rep['macro avg']['precision'], 4), 'Recall': round(rep['macro avg']['recall'], 4), 'F1-Score': round(rep['macro avg']['f1-score'], 4), 'Support': int(rep['macro avg']['support'])})
        pd.DataFrame(rows).to_csv(out_dir / 'resultats.csv', index=False)

    # 3. Grille combinée 2x2 des matrices de confusion
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.ravel()
    cmaps = ['Blues', 'Greens', 'Purples', 'Oranges']
    for i, (name, (y_t, y_p)) in enumerate(dl_preds_dict.items()):
        cm = confusion_matrix(y_t, y_p)
        ax = axes[i]
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmaps[i], xticklabels=CLASS_SHORT, yticklabels=CLASS_SHORT, linewidths=.5, ax=ax, cbar=False)
        ax.set_title(f"{name} — Accuracy: {accuracy_score(y_t, y_p)*100:.2f}%", fontsize=12, fontweight='bold')
        ax.set_xlabel('Prédiction', fontsize=10); ax.set_ylabel('Vérité terrain', fontsize=10)
    fig.suptitle('Matrices de Confusion Comparatives — 4 Algorithmes Deep Learning', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout(); fig.savefig(BENCH_DIR / 'confusion_matrices_combined.png', dpi=150); plt.close(fig)

    # 4. Benchmark global
    bench_df = pd.DataFrame(benchmark_data).sort_values('Accuracy (%)', ascending=False)
    bench_df.insert(0, 'Rang', range(1, len(bench_df) + 1))
    bench_df.to_csv(BENCH_DIR / 'benchmark_global.csv', index=False)
    print("\nTableau comparatif final :")
    print(bench_df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = ['#2ECC71', '#3498DB', '#9B59B6', '#E67E22']
    for ax, vals, title in [(axes[0], bench_df['Accuracy (%)'].tolist(), 'Accuracy (%)'), (axes[1], bench_df['Macro F1 (%)'].tolist(), 'Macro F1 (%)')]:
        bars = ax.barh(bench_df['Modèle'].tolist()[::-1], vals[::-1], color=colors[::-1], edgecolor='white', height=0.55)
        ax.set_xlabel(title, fontsize=12); ax.set_xlim(85, 96); ax.xaxis.grid(True, linestyle='--', alpha=0.4); ax.set_axisbelow(True)
        ax.set_title(title, fontsize=13, fontweight='bold')
        for bar, val in zip(bars, vals[::-1]):
            ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2, f"{val:.2f}%", va='center', fontsize=11, fontweight='bold')
    fig.suptitle('Benchmark Comparatif des 4 Algorithmes Deep Learning (Test : 450 images)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(); fig.savefig(BENCH_DIR / 'benchmark_global.png', dpi=150); plt.close(fig)
    print("\n✅ Génération complète terminée avec succès dans generation/ !")

if __name__ == '__main__':
    main()
