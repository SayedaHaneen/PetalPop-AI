"""
Model Evaluation and Metrics Visualization Module for PetalPop AI:
- Accuracy, Precision, Recall, F1-Score (macro & weighted)
- High-resolution Confusion Matrix visualization
- Training & Validation Loss/Accuracy curves
- Classification Report generation and saving
"""

import os
import json
import logging
from typing import Dict, Any, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless mode for server / scripts
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
import tensorflow as tf

from src.flower_info import get_all_flower_names

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PetalPopEvaluation")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
GRAPHS_DIR = os.path.join(OUTPUTS_DIR, "graphs")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")

# Set aesthetic plot styling (Pastel & botanical accents)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
PALETTE = {
    "primary": "#E06D8C",     # Rose
    "secondary": "#8A6FE8",   # Lavender
    "accent": "#F5A8B8",      # Pastel pink
    "dark": "#3D2B3D",        # Deep plum
    "bg": "#FFF9FB"           # Soft cream
}

def plot_training_history(history_dict: Dict[str, list], save_path: str = None) -> str:
    """
    Plots training vs validation accuracy and loss side-by-side with elegant styling.
    """
    if save_path is None:
        os.makedirs(GRAPHS_DIR, exist_ok=True)
        save_path = os.path.join(GRAPHS_DIR, "training_history.png")

    epochs = range(1, len(history_dict["accuracy"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=PALETTE["bg"])
    for ax in (ax1, ax2):
        ax.set_facecolor("#FFFFFF")
        ax.grid(True, linestyle="--", alpha=0.5, color="#E8D5E0")

    # Accuracy subplot
    ax1.plot(epochs, [x * 100 for x in history_dict["accuracy"]], label="Training Accuracy",
             color=PALETTE["primary"], linewidth=2.5, marker="o", markersize=4)
    if "val_accuracy" in history_dict:
        ax1.plot(epochs, [x * 100 for x in history_dict["val_accuracy"]], label="Validation Accuracy",
                 color=PALETTE["secondary"], linewidth=2.5, marker="s", markersize=4)
    ax1.set_title("Model Accuracy Across Epochs", fontsize=14, fontweight="bold", color=PALETTE["dark"], pad=12)
    ax1.set_xlabel("Epoch", fontsize=12, color=PALETTE["dark"])
    ax1.set_ylabel("Accuracy (%)", fontsize=12, color=PALETTE["dark"])
    ax1.legend(loc="lower right", frameon=True, facecolor=PALETTE["bg"], edgecolor="#E8D5E0")

    # Loss subplot
    ax2.plot(epochs, history_dict["loss"], label="Training Loss",
             color=PALETTE["primary"], linewidth=2.5, marker="o", markersize=4)
    if "val_loss" in history_dict:
        ax2.plot(epochs, history_dict["val_loss"], label="Validation Loss",
                 color=PALETTE["secondary"], linewidth=2.5, marker="s", markersize=4)
    ax2.set_title("Model Loss Across Epochs", fontsize=14, fontweight="bold", color=PALETTE["dark"], pad=12)
    ax2.set_xlabel("Epoch", fontsize=12, color=PALETTE["dark"])
    ax2.set_ylabel("Cross-Entropy Loss", fontsize=12, color=PALETTE["dark"])
    ax2.legend(loc="upper right", frameon=True, facecolor=PALETTE["bg"], edgecolor="#E8D5E0")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved training history curves to {save_path}")
    return save_path

def evaluate_on_test_set(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: list = None
) -> Dict[str, Any]:
    """
    Evaluates model on the test dataset, computes metrics, confusion matrix, and classification report.
    """
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    if class_names is None:
        class_names = get_all_flower_names()

    logger.info("Running evaluation on test dataset...")
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    test_acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    # Classification report
    unique_labels = sorted(list(set(y_true)))
    target_names_subset = [class_names[i] for i in unique_labels] if class_names else None
    clf_report_dict = classification_report(
        y_true, y_pred,
        labels=unique_labels,
        target_names=target_names_subset,
        output_dict=True,
        zero_division=0
    )
    clf_report_text = classification_report(
        y_true, y_pred,
        labels=unique_labels,
        target_names=target_names_subset,
        zero_division=0
    )

    # Save classification report
    report_file = os.path.join(REPORTS_DIR, "classification_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(clf_report_text)

    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    
    plt.figure(figsize=(16, 14), facecolor=PALETTE["bg"])
    # If 102 classes, plotting all labels would overlap, so display heatmap with aesthetic palette
    sns.heatmap(
        cm,
        cmap="Purples",
        cbar=True,
        square=True,
        xticklabels=False,
        yticklabels=False
    )
    plt.title(f"PetalPop AI - Confusion Matrix ({len(unique_labels)} Active Flower Classes)", fontsize=16, fontweight="bold", pad=15)
    plt.xlabel("Predicted Flower Class Index", fontsize=12)
    plt.ylabel("True Flower Class Index", fontsize=12)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    metrics = {
        "test_accuracy": float(test_acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "f1_weighted": float(f1_weighted),
        "total_test_samples": int(len(y_true)),
        "confusion_matrix_path": cm_path,
        "classification_report_path": report_file
    }

    # Save metrics JSON
    metrics_json_path = os.path.join(REPORTS_DIR, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Evaluation complete! Test Accuracy: {test_acc*100:.2f}%, F1-Score: {f1_weighted:.4f}")
    return metrics
