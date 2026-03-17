"""
train.py
--------
Full training pipeline for the Automatic Image Caption Generator.
"""

import os
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from data_preprocessing import (
    load_captions, add_start_end_tokens, Vocabulary,
    extract_image_features, create_sequences, MAX_CAPTION_LEN,
)
from model import build_captioning_model, get_callbacks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Paths & hyper-parameters
# ------------------------------------------------------------------
CAPTIONS_FILE  = os.path.join("data", "captions", "captions.txt")
IMAGE_DIR      = os.path.join("data", "images")
FEATURES_PATH  = os.path.join("models", "image_features.pkl")
VOCAB_PATH     = os.path.join("models", "vocabulary.pkl")
MODEL_PATH     = os.path.join("models", "best_caption_model.keras")
OUTPUTS_DIR    = "outputs"

EMBED_DIM   = 256
LSTM_UNITS  = 512
BATCH_SIZE  = 64
EPOCHS      = 30
TEST_SPLIT  = 0.1


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    os.makedirs("models",  exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # ---- Captions ----
    logger.info("Loading captions …")
    captions = load_captions(CAPTIONS_FILE)
    captions = add_start_end_tokens(captions)

    # ---- Vocabulary ----
    vocab = Vocabulary(min_freq=2)
    vocab.build(captions)
    vocab.save(VOCAB_PATH)

    # ---- Image features ----
    if os.path.exists(FEATURES_PATH):
        logger.info("Loading cached image features …")
        with open(FEATURES_PATH, "rb") as f:
            features = pickle.load(f)
    else:
        features = extract_image_features(IMAGE_DIR, FEATURES_PATH)

    feature_dim = next(iter(features.values())).shape[0]

    # ---- Create sequences ----
    logger.info("Creating training sequences …")
    X_img, X_seq, y = create_sequences(captions, features, vocab, MAX_CAPTION_LEN)

    # ---- Train / val split ----
    (X_img_tr, X_img_val,
     X_seq_tr, X_seq_val,
     y_tr,     y_val) = train_test_split(X_img, X_seq, y, test_size=TEST_SPLIT, random_state=42)

    logger.info(f"Train samples: {len(y_tr)}, Val samples: {len(y_val)}")

    # ---- Model ----
    model = build_captioning_model(
        vocab_size=vocab.size,
        max_caption_len=MAX_CAPTION_LEN,
        embed_dim=EMBED_DIM,
        lstm_units=LSTM_UNITS,
        feature_dim=feature_dim,
    )

    # ---- Training ----
    history = model.fit(
        [X_img_tr, X_seq_tr], y_tr,
        validation_data=([X_img_val, X_seq_val], y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=get_callbacks(MODEL_PATH),
        verbose=1,
    )

    # ---- Plots ----
    plot_training_history(history)
    logger.info("Training complete.")


def plot_training_history(history) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(history.history["accuracy"],     label="Train Acc")
    axes[1].plot(history.history["val_accuracy"], label="Val Acc")
    axes[1].set_title("Accuracy over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "training_history.png"), dpi=150)
    plt.close()
    logger.info("Training history plot saved.")


if __name__ == "__main__":
    main()
