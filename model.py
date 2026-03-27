"""
Model architecture module for the Autism Detection System.
Uses ResNet-18 with transfer learning (pretrained on ImageNet).
"""

import torch.nn as nn
from torchvision import models

import config


def get_model(model_name, num_classes, freeze=True):
    """
    Create a model fine-tuned for classification based on model_name.

    Args:
        model_name: "resnet18", "efficientnet_b0", or "densenet121"
        num_classes: Number of output classes (e.g., 2)
        freeze: If True, freeze all convolutional layers.

    Returns:
        model: The modified model
    """
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    else:
        # Unfreeze all parameters for fine-tuning
        for param in model.parameters():
            param.requires_grad = True

    if model_name == "resnet18":
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(256, num_classes),
        )
        # Ensure classifier is trainable
        for param in model.fc.parameters():
            param.requires_grad = True
            
    elif model_name == "efficientnet_b0":
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
        # Ensure classifier is trainable
        for param in model.classifier[1].parameters():
            param.requires_grad = True
            
    elif model_name == "densenet121":
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
        # Ensure classifier is trainable
        for param in model.classifier.parameters():
            param.requires_grad = True
            
    elif model_name == "resnet50":
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        # Ensure classifier is trainable
        for param in model.fc.parameters():
            param.requires_grad = True

    # Move model to device
    model = model.to(config.DEVICE)

    return model


def get_model_summary(model):
    """Print a summary of the model architecture and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print("=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)
    print(f"  Architecture:       {model.__class__.__name__} (Transfer Learning)")
    print(f"  Total parameters:   {total_params:,}")
    print(f"  Trainable params:   {trainable_params:,}")
    print(f"  Frozen params:      {frozen_params:,}")
    print(f"  Output classes:     {config.NUM_CLASSES} ({', '.join(config.CLASS_NAMES)})")
    print(f"  Device:             {config.DEVICE}")
    print("=" * 50)
