"""
Evaluation & visualization module for the Autism Detection System.
Computes metrics on the test set and generates plots.
"""

import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

import config


def evaluate_model(model, test_loader):
    """
    Evaluate the model on the test set.
    Returns all predictions and true labels for further analysis.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    return all_preds, all_labels, all_probs


def print_metrics(all_preds, all_labels):
    """Print detailed classification metrics."""
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    # Calculate sensitivity and specificity from confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sensitivity = 0.0
        specificity = 0.0

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy:    {accuracy * 100:.2f}%")
    print(f"  Precision:   {precision * 100:.2f}%")
    print(f"  Recall:      {recall * 100:.2f}%")
    print(f"  F1-Score:    {f1 * 100:.2f}%")
    print(f"  Sensitivity: {sensitivity * 100:.2f}%")
    print(f"  Specificity: {specificity * 100:.2f}%")
    print("\n" + "-" * 60)
    print("DETAILED CLASSIFICATION REPORT")
    print("-" * 60)
    print(
        classification_report(
            all_labels, all_preds, target_names=config.CLASS_NAMES, zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def plot_confusion_matrix(all_preds, all_labels):
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 16},
    )
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
    plt.ylabel("True Label", fontsize=13)
    plt.xlabel("Predicted Label", fontsize=13)
    plt.tight_layout()

    save_path = os.path.join(config.RESULTS_DIR, f"{config.MODEL_NAME}_confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Confusion matrix saved to: {save_path}")


def plot_roc_curve(all_labels, all_probs):
    """Generate and save the ROC curve with AUC."""
    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title('Receiver Operating Characteristic', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(config.RESULTS_DIR, f"{config.MODEL_NAME}_roc_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ ROC curve saved to: {save_path}")


def plot_training_history(history):
    """Generate and save training/validation loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Loss plot ──
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=4)
    axes[0].set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Accuracy plot ──
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=4)
    axes[1].plot(epochs, history["val_acc"], "r-o", label="Val Acc", markersize=4)
    axes[1].set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(config.RESULTS_DIR, f"{config.MODEL_NAME}_training_history.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Training history plot saved to: {save_path}")


def run_evaluation(model, test_loader, history=None):
    """
    Full evaluation pipeline:
      1. Predict on test set
      2. Print metrics
      3. Plot confusion matrix
      4. Plot ROC curve
      5. Plot training history (if available)
    """
    print("\nRunning evaluation on test set...")
    all_preds, all_labels, all_probs = evaluate_model(model, test_loader)

    metrics = print_metrics(all_preds, all_labels)
    plot_confusion_matrix(all_preds, all_labels)
    plot_roc_curve(all_labels, all_probs)

    if history is not None:
        plot_training_history(history)

    return metrics
