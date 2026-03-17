"""
caption_generator.py
--------------------
Inference script: load a trained model and generate captions for new images.
Supports greedy and beam-search decoding.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from data_preprocessing import Vocabulary, load_image
from model import generate_caption_greedy, generate_caption_beam

VOCAB_PATH  = os.path.join("models", "vocabulary.pkl")
MODEL_PATH  = os.path.join("models", "best_caption_model.keras")
FEATURE_DIM = 2048


# ------------------------------------------------------------------
# Feature extraction (single image at inference time)
# ------------------------------------------------------------------

_encoder_cache = None

def get_encoder():
    global _encoder_cache
    if _encoder_cache is None:
        from tensorflow.keras.applications import InceptionV3
        from tensorflow.keras import Model
        base = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
        _encoder_cache = Model(inputs=base.input, outputs=base.output)
    return _encoder_cache


def extract_single_feature(image_path: str) -> np.ndarray:
    encoder = get_encoder()
    img     = load_image(image_path)
    feature = encoder.predict(img[np.newaxis, ...], verbose=0)[0]
    return feature


# ------------------------------------------------------------------
# Caption generation
# ------------------------------------------------------------------

def caption_image(
    image_path: str,
    model,
    vocab: Vocabulary,
    method: str = "beam",
    beam_width: int = 5,
) -> str:
    """
    Generate a caption for the given image.

    Parameters
    ----------
    image_path : path to the image file
    model      : loaded Keras captioning model
    vocab      : Vocabulary instance
    method     : "greedy" or "beam"
    beam_width : number of beams (beam search only)
    """
    feature = extract_single_feature(image_path)
    if method == "beam":
        return generate_caption_beam(model, feature, vocab, beam_width=beam_width)
    else:
        return generate_caption_greedy(model, feature, vocab)


def display_captioned_image(image_path: str, caption: str, save_path: str = None) -> None:
    img = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(caption, fontsize=14, wrap=True, pad=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def batch_caption(image_dir: str, output_csv: str, method: str = "beam") -> None:
    """Caption all images in a directory and save results to CSV."""
    import pandas as pd

    model = tf.keras.models.load_model(MODEL_PATH)
    vocab = Vocabulary.load(VOCAB_PATH)

    records = []
    files   = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png"))]
    print(f"Captioning {len(files)} images …")

    for fname in files:
        path    = os.path.join(image_dir, fname)
        caption = caption_image(path, model, vocab, method=method)
        records.append({"filename": fname, "caption": caption})
        print(f"  {fname}: {caption}")

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\nCaptions saved → {output_csv}")


# ------------------------------------------------------------------
# BLEU evaluation helper
# ------------------------------------------------------------------

def evaluate_bleu(captions_dict: dict, model, vocab: Vocabulary) -> dict:
    """
    Compute BLEU-1 to BLEU-4 on a reference caption dict.
    Requires nltk: pip install nltk
    """
    from nltk.translate.bleu_score import corpus_bleu

    references, hypotheses = [], []

    for img_id, ref_caps in captions_dict.items():
        feature_path = os.path.join("models", "image_features.pkl")
        import pickle
        with open(feature_path, "rb") as f:
            features = pickle.load(f)

        if img_id not in features:
            continue

        feature    = features[img_id]
        hypothesis = generate_caption_beam(model, feature, vocab).split()
        refs       = [cap.split() for cap in ref_caps]

        references.append(refs)
        hypotheses.append(hypothesis)

    scores = {
        "BLEU-1": corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0)),
        "BLEU-2": corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0)),
        "BLEU-3": corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0)),
        "BLEU-4": corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25)),
    }
    return scores


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic Image Caption Generator")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--batch", type=str, help="Directory of images for batch captioning")
    parser.add_argument("--output", type=str, default="outputs/captions.csv")
    parser.add_argument("--method", choices=["greedy", "beam"], default="beam")
    parser.add_argument("--beam_width", type=int, default=5)
    args = parser.parse_args()

    if args.batch:
        batch_caption(args.batch, args.output, method=args.method)
    elif args.image:
        model = tf.keras.models.load_model(MODEL_PATH)
        vocab = Vocabulary.load(VOCAB_PATH)
        cap   = caption_image(args.image, model, vocab, method=args.method,
                              beam_width=args.beam_width)
        print(f"\nGenerated Caption: {cap}")
        display_captioned_image(args.image, cap, save_path="outputs/captioned_image.png")
    else:
        print("Provide --image or --batch argument.")
