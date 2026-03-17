"""
data_preprocessing.py
---------------------
Prepares image-caption pairs for the Automatic Image Caption Generator.
Handles vocabulary building, image loading/augmentation, and data generators.
"""

import os
import re
import json
import string
import logging
import numpy as np
import pickle
from collections import Counter

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import preprocess_input

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
IMAGE_SIZE    = (299, 299)     # InceptionV3 / EfficientNet input
MAX_CAPTION_LEN = 40
MIN_WORD_FREQ   = 2            # Discard rare words
PAD_TOKEN     = "<pad>"
START_TOKEN   = "<start>"
END_TOKEN     = "<end>"
UNK_TOKEN     = "<unk>"


# ------------------------------------------------------------------
# Caption utilities
# ------------------------------------------------------------------

def clean_caption(caption: str) -> str:
    """Lowercase, strip punctuation, normalise whitespace."""
    caption = caption.lower()
    caption = caption.translate(str.maketrans("", "", string.punctuation))
    caption = re.sub(r"\s+", " ", caption).strip()
    return caption


def load_captions(filepath: str) -> dict[str, list[str]]:
    """
    Load captions file.  Expected format (one per line):
        image_id.jpg  A dog plays in the park .
    Returns dict  image_id → [caption1, caption2, …]
    """
    captions: dict[str, list[str]] = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            image_id, caption = parts[0].split("#")[0], parts[1]
            captions.setdefault(image_id, []).append(clean_caption(caption))
    logger.info(f"Loaded captions for {len(captions)} images.")
    return captions


def add_start_end_tokens(captions: dict) -> dict:
    """Wrap each caption with <start> and <end> tokens."""
    return {
        img_id: [f"{START_TOKEN} {cap} {END_TOKEN}" for cap in caps]
        for img_id, caps in captions.items()
    }


# ------------------------------------------------------------------
# Vocabulary
# ------------------------------------------------------------------

class Vocabulary:
    """Simple word ↔ integer vocabulary."""

    def __init__(self, min_freq: int = MIN_WORD_FREQ):
        self.min_freq  = min_freq
        self.word2idx  : dict[str, int] = {}
        self.idx2word  : dict[int, str] = {}
        self.size      : int = 0

    def build(self, captions: dict) -> None:
        counter: Counter = Counter()
        for caps in captions.values():
            for cap in caps:
                counter.update(cap.split())

        special = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]
        words = special + [w for w, f in counter.items() if f >= self.min_freq and w not in special]

        self.word2idx = {w: i for i, w in enumerate(words)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.size     = len(words)
        logger.info(f"Vocabulary size: {self.size}")

    def encode(self, caption: str) -> list[int]:
        unk = self.word2idx[UNK_TOKEN]
        return [self.word2idx.get(w, unk) for w in caption.split()]

    def decode(self, indices: list[int]) -> str:
        return " ".join(
            self.idx2word[i]
            for i in indices
            if self.idx2word.get(i) not in (PAD_TOKEN, START_TOKEN, END_TOKEN)
        )

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Vocabulary saved → {path}")

    @staticmethod
    def load(path: str) -> "Vocabulary":
        with open(path, "rb") as f:
            return pickle.load(f)


# ------------------------------------------------------------------
# Image utilities
# ------------------------------------------------------------------

def load_image(image_path: str, size: tuple = IMAGE_SIZE) -> np.ndarray:
    """Load and pre-process a single image for InceptionV3."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, size)
    img = preprocess_input(img)
    return img.numpy()


def augment_image(image: np.ndarray) -> np.ndarray:
    """Apply random flips & brightness jitter (training only)."""
    img = tf.convert_to_tensor(image)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.clip_by_value(img, -1.0, 1.0)
    return img.numpy()


# ------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------

def extract_image_features(image_dir: str, save_path: str) -> dict:
    """
    Extract CNN feature vectors for every image using InceptionV3
    (pre-trained on ImageNet, no top layer).
    Saves a dict  {image_id: feature_vector}  to disk.
    """
    from tensorflow.keras.applications import InceptionV3
    from tensorflow.keras import Model

    base   = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    encoder = Model(inputs=base.input, outputs=base.output)

    features: dict[str, np.ndarray] = {}
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png"))]
    logger.info(f"Extracting features for {len(image_files)} images …")

    for i, fname in enumerate(image_files):
        path = os.path.join(image_dir, fname)
        img  = load_image(path)
        vec  = encoder.predict(img[np.newaxis, ...], verbose=0)[0]
        features[fname] = vec
        if (i + 1) % 500 == 0:
            logger.info(f"  {i + 1} / {len(image_files)}")

    with open(save_path, "wb") as f:
        pickle.dump(features, f)
    logger.info(f"Features saved → {save_path}")
    return features


# ------------------------------------------------------------------
# Dataset generator
# ------------------------------------------------------------------

def create_sequences(
    captions: dict,
    features: dict,
    vocab: Vocabulary,
    max_len: int = MAX_CAPTION_LEN,
):
    """
    Yield (image_feature, input_sequence, target_word) triples for training.
    """
    X_img, X_seq, y = [], [], []

    for img_id, caps in captions.items():
        if img_id not in features:
            continue
        feat = features[img_id]
        for cap in caps:
            seq = vocab.encode(cap)
            for end in range(1, len(seq)):
                in_seq  = pad_sequences([seq[:end]],  maxlen=max_len, padding="post")[0]
                out_word = seq[end] if end < len(seq) else vocab.word2idx[PAD_TOKEN]
                X_img.append(feat)
                X_seq.append(in_seq)
                y.append(out_word)

    return np.array(X_img), np.array(X_seq), np.array(y)


if __name__ == "__main__":
    # Quick smoke-test with dummy data
    dummy_captions = {
        "img1.jpg": ["a cat on a mat", "white cat sitting"],
        "img2.jpg": ["dogs running in park", "two dogs playing"],
    }
    dummy_captions = add_start_end_tokens(dummy_captions)
    vocab = Vocabulary(min_freq=1)
    vocab.build(dummy_captions)
    print(f"Vocab size: {vocab.size}")
    print(vocab.decode(vocab.encode("<start> a cat on a mat <end>")))
