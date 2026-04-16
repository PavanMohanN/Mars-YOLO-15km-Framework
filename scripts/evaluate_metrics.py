"""
evaluate_metrics_uniform25.py

Validation pipeline for Mars-YOLO-15km.
Executes model evaluation, generates performance plots, and exports
a standalone confusion matrix with controlled plotting behavior.
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# 1. GLOBAL CONFIGURATION
# Defines consistent plotting behavior across all figures
# =====================================================================

plt.rcParams.update({
    "font.size": 25,
    "axes.labelsize": 25,
    "axes.titlesize": 25,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
    "legend.fontsize": 25,
    "legend.title_fontsize": 25,
    "figure.titlesize": 25,
    "font.family": "serif",

    "axes.linewidth": 1.2,
    "lines.linewidth": 2.0,

    "xtick.direction": "in",
    "ytick.direction": "in",

    "axes.grid": False,

    "figure.dpi": 350,
    "savefig.dpi": 350,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# =====================================================================
# 2. SAVEFIG OVERRIDE
# Ensures all figures are exported with consistent resolution/settings
# =====================================================================

_original_savefig = plt.savefig
def patched_savefig(*args, **kwargs):
    kwargs["dpi"] = 350
    kwargs["format"] = "png"
    kwargs["bbox_inches"] = "tight"
    kwargs["pad_inches"] = 0.02
    return _original_savefig(*args, **kwargs)

plt.savefig = patched_savefig

# =====================================================================
# 3. AXIS TEXT CONTROL
# Standardizes axis titles and labels during plotting
# =====================================================================

_original_title = matplotlib.axes.Axes.set_title
def patched_title(self, label, *args, **kwargs):
    kwargs["fontsize"] = 25
    return _original_title(self, label, *args, **kwargs)
matplotlib.axes.Axes.set_title = patched_title

_original_xlabel = matplotlib.axes.Axes.set_xlabel
def patched_xlabel(self, label, *args, **kwargs):
    kwargs["fontsize"] = 25
    return _original_xlabel(self, label, *args, **kwargs)
matplotlib.axes.Axes.set_xlabel = patched_xlabel

_original_ylabel = matplotlib.axes.Axes.set_ylabel
def patched_ylabel(self, label, *args, **kwargs):
    kwargs["fontsize"] = 25
    return _original_ylabel(self, label, *args, **kwargs)
matplotlib.axes.Axes.set_ylabel = patched_ylabel

# =====================================================================
# 4. LEGEND CONTROL
# Forces legend placement inside axes at bottom-left corner
# =====================================================================

_original_legend = matplotlib.axes.Axes.legend
def patched_legend(self, *args, **kwargs):
    kwargs["fontsize"] = 25
    kwargs["title_fontsize"] = 25

    # Anchor legend explicitly inside axes (bottom-left)
    kwargs["loc"] = "lower left"
    kwargs["bbox_to_anchor"] = (0.02, 0.02)  # slight offset to avoid clipping
    kwargs["borderaxespad"] = 0.0

    return _original_legend(self, *args, **kwargs)

matplotlib.axes.Axes.legend = patched_legend

# =====================================================================
# 5. HEATMAP CONTROL
# Ensures annotation consistency in confusion matrix
# =====================================================================

_original_heatmap = sns.heatmap
def patched_heatmap(*args, **kwargs):
    kwargs["annot_kws"] = {"size": 25}
    return _original_heatmap(*args, **kwargs)

sns.heatmap = patched_heatmap

# =====================================================================
# 6. IMPORT MODEL (AFTER PATCHES)
# =====================================================================

from ultralytics import YOLO

# =====================================================================
# 7. PATH INITIALIZATION
# Resolves model, dataset, and output directories
# =====================================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best.pt')
DATA_YAML = os.path.join(BASE_DIR, 'data', 'dataset.yaml')
OUTPUT_DIR = os.path.join(BASE_DIR, 'validation_metrics')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# 8. CONFUSION MATRIX EXPORT
# Generates and saves a standalone confusion matrix visualization
# =====================================================================

def export_confusion_matrix(matrix, labels, save_path):
    plt.figure(figsize=(12, 10))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Count"}
    )

    plt.xlabel("Predicted Class", labelpad=15)
    plt.ylabel("Actual Class", labelpad=15)
    plt.title("Geomorphometric Confusion Matrix", pad=20)

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.savefig(save_path)
    plt.close()

# =====================================================================
# 9. EVALUATION PIPELINE
# Executes validation and triggers all plot generation
# =====================================================================

def evaluate_model():
    print("[*] Loading model weights...")
    model = YOLO(MODEL_PATH)

    print("[*] Running validation...")

    metrics = model.val(
        data=DATA_YAML,
        conf=0.7,
        iou=0.5,
        plots=True,
        device='cpu'
    )

    print("\n[+] Validation Complete")
    print(f"    mAP@50:    {metrics.box.map50:.4f}")
    print(f"    Precision: {metrics.box.mp:.4f}")
    print(f"    Recall:    {metrics.box.mr:.4f}")

    # Export confusion matrix
    print("\n[*] Exporting confusion matrix...")

    matrix = metrics.confusion_matrix.matrix
    labels = ['Crater', 'Background']

    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_350dpi.png")

    export_confusion_matrix(matrix, labels, cm_path)

    print(f"[+] Saved: {cm_path}")
    print(f"[+] YOLO plot directory: {metrics.save_dir}")

# =====================================================================
# 10. ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    evaluate_model()