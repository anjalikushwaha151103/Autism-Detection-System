"""
Training pipeline for the Autism Detection System.
Handles the training loop, validation, early stopping, and checkpointing.
"""

import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config


class EarlyStopping:
    """
    Stop training when validation loss stops improving.
    Saves the best model checkpoint automatically.
    """

    def __init__(self, patience=config.PATIENCE, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"  ⚠ EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_one_epoch(model, train_loader, criterion, optimizer):
    """Train the model for one epoch. Returns average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="  Training", leave=False)
    for images, labels in progress_bar:
        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100 * correct / total:.1f}%",
        )

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion):
    """Evaluate the model on the validation set. Returns average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader):
    """
    Full training pipeline with:
      - CrossEntropy loss
      - Adam optimizer with weight decay
      - Learning rate scheduling (reduce on plateau)
      - Early stopping
      - Best model checkpointing

    Returns:
        model: The best model (by validation accuracy)
        history: Dictionary with training/validation loss and accuracy per epoch
    """
    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer — only optimize parameters that require gradients
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Learning rate scheduler — reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=config.PATIENCE)

    # Track metrics
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print(f"  Epochs: {config.NUM_EPOCHS} | Batch Size: {config.BATCH_SIZE}")
    print(f"  Learning Rate: {config.LEARNING_RATE} | Device: {config.DEVICE}")
    print("=" * 60)

    start_time = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        print("-" * 40)

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion)

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Print epoch results
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"  ✓ Best model saved! (Val Acc: {val_acc:.2f}%)")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\n⛔ Early stopping triggered at epoch {epoch}")
            break

    # Training complete
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  Total time:        {total_time / 60:.1f} minutes")
    print(f"  Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"  Model saved to:    {config.BEST_MODEL_PATH}")
    print("=" * 60)

    # Load best model weights
    model.load_state_dict(best_model_weights)

    return model, history
