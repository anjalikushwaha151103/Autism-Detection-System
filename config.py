"""
Configuration file for Autism Detection System.
Central place for all hyperparameters, paths, and settings.
"""

import os
import torch

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

TRAIN_DIR = os.path.join(DATA_DIR, "train-20260325T170535Z-3-001", "train")
TEST_DIR  = os.path.join(DATA_DIR, "test-20260325T170536Z-3-001", "test")
VALID_DIR = os.path.join(DATA_DIR, "valid-20260325T170538Z-3-001", "valid")

MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_LIST = ["resnet50"]
MODEL_NAME = "resnet50"  # can be changed to efficientnet_b0 or densenet121

BEST_MODEL_PATH = os.path.join(MODELS_DIR, f"{MODEL_NAME}_best.pth")

# ──────────────────────────────────────────────
# CLASS NAMES
# ──────────────────────────────────────────────
CLASS_NAMES = ["Autistic", "Non_Autistic"]
NUM_CLASSES = len(CLASS_NAMES)

# ──────────────────────────────────────────────
# HYPERPARAMETERS
# ──────────────────────────────────────────────
IMAGE_SIZE = 224          # ResNet expected input size
BATCH_SIZE = 16           # Smaller batch for CPU training
NUM_EPOCHS = 20           # Max epochs (early stopping may kick in sooner)
LEARNING_RATE = 1e-5      # Reduced learning rate for fine-tuning
WEIGHT_DECAY = 1e-4       # L2 regularization
SEED = 42

# Early stopping
PATIENCE = 5              # Stop after 5 epochs without improvement

# Dropout in classifier head
DROPOUT_RATE = 0.5

# ──────────────────────────────────────────────
# DEVICE
# ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# DATA AUGMENTATION SETTINGS
# ──────────────────────────────────────────────
# ImageNet normalization stats (used by pretrained ResNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

ROTATION_DEGREES = 15
COLOR_JITTER = {
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.2,
    "hue": 0.1,
}
