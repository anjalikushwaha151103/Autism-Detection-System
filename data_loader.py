"""
Data loading and preprocessing module for the Autism Detection System.
Handles image transformations, augmentation, and DataLoader creation.
"""

import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import config


# Valid image extensions supported by torchvision
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def is_valid_file(path):
    """Filter out files inside .ipynb_checkpoints or other hidden directories."""
    if ".ipynb_checkpoints" in path:
        return False
    return path.lower().endswith(VALID_EXTENSIONS)


def get_train_transforms():
    """
    Training transforms with data augmentation to improve model generalization.
    Augmentation is critical for small datasets to reduce overfitting.
    """
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=config.ROTATION_DEGREES),
        transforms.ColorJitter(**config.COLOR_JITTER),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def get_eval_transforms():
    """
    Validation/test transforms — no augmentation, just resize and normalize.
    """
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def load_datasets():
    """
    Load train, validation, and test datasets using ImageFolder.
    Returns the dataset objects (useful for inspecting class-to-index mapping).
    """
    train_dataset = datasets.ImageFolder(
        root=config.TRAIN_DIR,
        transform=get_train_transforms(),
        is_valid_file=is_valid_file,
    )
    valid_dataset = datasets.ImageFolder(
        root=config.VALID_DIR,
        transform=get_eval_transforms(),
        is_valid_file=is_valid_file,
    )
    test_dataset = datasets.ImageFolder(
        root=config.TEST_DIR,
        transform=get_eval_transforms(),
        is_valid_file=is_valid_file,
    )

    return train_dataset, valid_dataset, test_dataset


def create_data_loaders(train_dataset, valid_dataset, test_dataset):
    """
    Create DataLoader objects for each split.
    Training data is shuffled; validation/test data is not.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,    # 0 workers for CPU compatibility on Windows
        pin_memory=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, valid_loader, test_loader


def get_data_loaders():
    """
    Convenience function: load datasets and create data loaders in one call.
    Also prints dataset statistics.
    """
    train_dataset, valid_dataset, test_dataset = load_datasets()
    train_loader, valid_loader, test_loader = create_data_loaders(
        train_dataset, valid_dataset, test_dataset
    )

    print("=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)
    print(f"  Training samples:   {len(train_dataset)}")
    print(f"  Validation samples: {len(valid_dataset)}")
    print(f"  Test samples:       {len(test_dataset)}")
    print(f"  Classes:            {train_dataset.classes}")
    print(f"  Class-to-Index:     {train_dataset.class_to_idx}")
    print("=" * 50)

    return train_loader, valid_loader, test_loader, train_dataset.class_to_idx
