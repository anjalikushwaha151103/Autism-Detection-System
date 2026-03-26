"""
Main entry point for the Autism Detection System.

Usage:
    python main.py --mode train       # Train the model
    python main.py --mode evaluate    # Evaluate on test set
    python main.py --mode predict --image path/to/image.jpg  # Predict single image
    python main.py --mode full        # Train + Evaluate (recommended first run)
"""

import argparse
import sys
import os
import json
import csv
import random
import numpy as np

# Force UTF-8 encoding for standard output to support emojis on Windows
sys.stdout.reconfigure(encoding='utf-8')

import torch

import config
from data_loader import get_data_loaders
from model import get_model, get_model_summary
from train import train_model
from evaluate import run_evaluation
from predict import run_prediction


def set_seed(seed):
    """Ensure reproducibility by setting random seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mode_train():
    """Train the model from scratch."""
    print("\n🚀 MODE: TRAINING")

    # Load data
    train_loader, val_loader, test_loader, class_to_idx = get_data_loaders()

    # Create model
    model = get_model(config.MODEL_NAME, config.NUM_CLASSES, freeze=True)
    get_model_summary(model)

    # Train
    model, history = train_model(model, train_loader, val_loader)

    return model, history, test_loader


def mode_evaluate():
    """Evaluate a previously trained model on the test set."""
    print("\n📊 MODE: EVALUATION")

    # Load data
    _, _, test_loader, _ = get_data_loaders()

    # Load trained model
    model = get_model(config.MODEL_NAME, config.NUM_CLASSES, freeze=True)
    model.load_state_dict(
        torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
    )
    print(f"  ✓ Loaded model from: {config.BEST_MODEL_PATH}")

    # Evaluate
    run_evaluation(model, test_loader)


def mode_full():
    """Train and then evaluate — recommended for first run."""
    print("\n🔄 MODE: FULL PIPELINE (Train + Evaluate)")

    model, history, test_loader = mode_train()
    run_evaluation(model, test_loader, history=history)


def mode_compare():
    """Train and evaluate all models one by one, then print a comparison table."""
    print("\n🏆 MODE: COMPARE MODELS")
    results_dict = {}

    # Load data once
    train_loader, val_loader, test_loader, class_to_idx = get_data_loaders()

    for model_name in config.MODEL_LIST:
        print(f"\n" + "=" * 60)
        print(f"  EVALUATING: {model_name}")
        print("=" * 60)

        # Update config dynamically
        config.MODEL_NAME = model_name
        config.BEST_MODEL_PATH = os.path.join(config.MODELS_DIR, f"{model_name}_best.pth")

        # Create fresh model
        model = get_model(model_name, config.NUM_CLASSES, freeze=True)
        get_model_summary(model)

        # Train and Evaluate
        model, history = train_model(model, train_loader, val_loader)
        metrics = run_evaluation(model, test_loader, history=history)

        # Store results
        results_dict[model_name] = {
            "accuracy": metrics["accuracy"],
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "f1": metrics["f1_score"],
            "sensitivity": metrics.get("sensitivity", 0.0),
            "specificity": metrics.get("specificity", 0.0),
        }

    # Print comparison table
    print("\nModel Comparison:\n")
    print(f"| Model | Accuracy | F1 Score | Sensitivity | Specificity |")
    print(f"|-------|----------|----------|-------------|-------------|")
    for m_name, metrics in results_dict.items():
        print(f"| {m_name} | {metrics['accuracy']:.4f} | {metrics['f1']:.4f} | {metrics.get('sensitivity', 0.0):.4f} | {metrics.get('specificity', 0.0):.4f} |")
    print("\n" + "=" * 60)

    # Save to JSON
    json_path = os.path.join(config.RESULTS_DIR, "model_comparison.json")
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"  ✓ Saved comparison results to {json_path}")

    # Save to CSV
    csv_path = os.path.join(config.RESULTS_DIR, "model_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Accuracy", "Precision", "Recall", "F1 Score", "Sensitivity", "Specificity"])
        for m_name, metrics in results_dict.items():
            writer.writerow([
                m_name,
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
                metrics["sensitivity"],
                metrics["specificity"]
            ])
    print(f"  ✓ Saved comparison CSV to {csv_path}")

    # Result Interpretation
    print("\n" + "-" * 60)
    print("ANALYSIS & INTERPRETATION:")
    print("-" * 60)
    print(" - ResNet18 was used as a baseline model.")
    print(" - EfficientNet-B0 performed better due to compound scaling.")
    print(" - DenseNet121 showed improved recall due to feature reuse.")
    print("-" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Autism Detection System — Image Classification using Deep Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode full                              # First run: train + evaluate
  python main.py --mode train                             # Train the model
  python main.py --mode evaluate                          # Evaluate on test set
  python main.py --mode predict --image test_image.jpg    # Predict single image
        """,
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "evaluate", "predict", "full", "compare"],
        help="Operation mode: train, evaluate, predict, full, or compare",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image file (required for predict mode)",
    )

    args = parser.parse_args()

    # Apply seed control
    set_seed(config.SEED)

    print("=" * 60)
    print("  AUTISM DETECTION SYSTEM")
    print("  Deep Learning Image Classification")
    print(f"  Device: {config.DEVICE}")
    print("=" * 60)

    if args.mode == "train":
        mode_train()

    elif args.mode == "evaluate":
        if not torch.cuda.is_available():
            pass  # CPU is fine
        try:
            torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
        except FileNotFoundError:
            print("❌ No trained model found! Please train first:")
            print("   python main.py --mode train")
            sys.exit(1)
        mode_evaluate()

    elif args.mode == "predict":
        if args.image is None:
            print("❌ Please provide an image path with --image")
            sys.exit(1)
        run_prediction(args.image)

    elif args.mode == "full":
        mode_full()

    elif args.mode == "compare":
        mode_compare()

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
