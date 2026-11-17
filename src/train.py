"""
Full Training and Fine-Tuning Pipeline for PetalPop AI:
- Transfer Learning with MobileNetV2
- Classification Head Training
- Upper-Layer Fine-Tuning
- Real Evaluation on Unseen Test Set
- Generates True Training Curves, Metrics, and Confusion Matrix
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import build_datasets
from src.model import build_transfer_learning_model, unfreeze_for_fine_tuning
from src.evaluation import plot_training_history, evaluate_on_test_set

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PetalPopTraining")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.keras")
HISTORY_JSON_PATH = os.path.join(PROJECT_ROOT, "outputs", "reports", "training_history.json")

def run_training_pipeline(
    epochs_head: int = 4,
    epochs_finetune: int = 4,
    batch_size: int = 32,
    steps_per_epoch: int = None,
    val_steps: int = None,
    unfreeze_from_layer: int = 110
):
    """
    Executes real training and fine-tuning pipeline on Oxford 102 flowers dataset.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "outputs", "graphs"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "outputs", "reports"), exist_ok=True)

    # 1. Build data pipelines
    logger.info("Loading and batching Oxford 102 dataset...")
    train_ds, val_ds, test_ds = build_datasets(batch_size=batch_size)

    # 2. Build MobileNetV2 Model
    logger.info("Building MobileNetV2 Transfer Learning Architecture...")
    model = build_transfer_learning_model(num_classes=102, dropout_rate=0.35)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    logger.info("Model compiled for Phase 1 (Classification Head Warmup).")

    # Callbacks
    callbacks_head = [
        EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        ModelCheckpoint(BEST_MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1)
    ]

    logger.info(f"Phase 1: Training Classification Head for {epochs_head} epochs...")
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_head,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks_head
    )

    # 3. Fine-Tuning Phase
    logger.info(f"Phase 2: Fine-Tuning from layer {unfreeze_from_layer} onwards for {epochs_finetune} epochs...")
    model = unfreeze_for_fine_tuning(
        model,
        unfreeze_from_layer=unfreeze_from_layer,
        learning_rate=5e-5
    )

    callbacks_ft = [
        EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
        ModelCheckpoint(BEST_MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1)
    ]

    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_head + epochs_finetune,
        initial_epoch=epochs_head,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks_ft
    )

    # Combine metrics
    combined_history = {
        "accuracy": [float(x) for x in (history_head.history["accuracy"] + history_ft.history["accuracy"])],
        "val_accuracy": [float(x) for x in (history_head.history["val_accuracy"] + history_ft.history["val_accuracy"])],
        "loss": [float(x) for x in (history_head.history["loss"] + history_ft.history["loss"])],
        "val_loss": [float(x) for x in (history_head.history["val_loss"] + history_ft.history["val_loss"])]
    }

    with open(HISTORY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(combined_history, f, indent=2)

    # Plot training and validation loss & accuracy graphs
    logger.info("Generating and saving training history graphs...")
    plot_training_history(combined_history)

    # 4. Strict Evaluation on Unseen Test Set
    logger.info("Performing final evaluation on independent test partition...")
    # Load best checkpointed model
    if os.path.exists(BEST_MODEL_PATH):
        best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
    else:
        best_model = model

    metrics = evaluate_on_test_set(best_model, test_ds)
    logger.info(f"Test Set Evaluation Results: {metrics}")
    print("\n" + "="*50)
    print(f"🎉 Model Training & Evaluation Complete!")
    print(f"Test Accuracy: {metrics['test_accuracy']*100:.2f}%")
    print(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}")
    print("="*50 + "\n")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PetalPop AI Training Pipeline")
    parser.add_argument("--epochs_head", type=int, default=3, help="Epochs for training classification head")
    parser.add_argument("--epochs_finetune", type=int, default=3, help="Epochs for fine-tuning")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--steps_per_epoch", type=int, default=None, help="Steps per epoch (optional for fast testing)")
    parser.add_argument("--val_steps", type=int, default=None, help="Validation steps (optional)")
    parser.add_argument("--unfreeze_from", type=int, default=110, help="Layer index to unfreeze")
    args = parser.parse_args()

    run_training_pipeline(
        epochs_head=args.epochs_head,
        epochs_finetune=args.epochs_finetune,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        val_steps=args.val_steps,
        unfreeze_from_layer=args.unfreeze_from
    )
