"""
Prediction / inference module for the Autism Detection System.
Loads a trained model and predicts on a single image.
"""

import argparse

import torch
from PIL import Image

import config
from model import get_model
from data_loader import get_eval_transforms


def load_trained_model(model_path=None):
    """Load the best saved model from disk."""
    # 3. Load model
    print("Loading model...")
    model = get_model(config.MODEL_NAME, config.NUM_CLASSES, freeze=True)
    
    load_path = model_path if model_path is not None else config.BEST_MODEL_PATH

    try:
        model.load_state_dict(
            torch.load(load_path, map_location=config.DEVICE)
        )
        print(f"  ✓ Loaded model weights from: {load_path}")
    except FileNotFoundError:
        print(f"  ✗ Error: Model weights not found at {load_path}")
        raise
    except Exception as e:
        print(f"  ✗ Error loading model weights: {e}")
        raise

    model.eval()
    return model


def predict_image(model, image_path):
    """
    Predict the class of a single image.

    Args:
        model: Trained model
        image_path: Path to the image file

    Returns:
        predicted_class: "Autistic" or "Non_Autistic"
        confidence: Confidence score (0-100%)
        probabilities: Raw probability for each class
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    transform = get_eval_transforms()
    image_tensor = transform(image).unsqueeze(0).to(config.DEVICE)

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = config.CLASS_NAMES[predicted_idx.item()]
    confidence_pct = confidence.item() * 100
    probs = {
        config.CLASS_NAMES[i]: f"{probabilities[0][i].item() * 100:.2f}%"
        for i in range(config.NUM_CLASSES)
    }

    return predicted_class, confidence_pct, probs


def run_prediction(image_path, model_path=None):
    """Full prediction pipeline for a single image."""
    print("\n" + "=" * 50)
    print("AUTISM DETECTION — SINGLE IMAGE PREDICTION")
    print("=" * 50)
    print(f"  Image: {image_path}")

    # Load model
    model = load_trained_model(model_path)
    print("  ✓ Model loaded successfully")

    # Predict
    predicted_class, confidence, probs = predict_image(model, image_path)

    print(f"\n  Prediction:  {predicted_class}")
    print(f"  Confidence:  {confidence:.2f}%")
    print(f"  Probabilities:")
    for cls, prob in probs.items():
        print(f"    - {cls}: {prob}")
    print("=" * 50)

    return predicted_class, confidence, probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict autism from a single image")
    parser.add_argument("--image", type=str, required=True, help="Path to the image")
    parser.add_argument("--model", type=str, default=None, help="Path to model weights")
    args = parser.parse_args()

    run_prediction(args.image, args.model)
