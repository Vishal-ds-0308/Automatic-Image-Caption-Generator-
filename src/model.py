"""
model.py
--------
CNN-LSTM image captioning model built with TensorFlow / Keras.
Architecture: InceptionV3 encoder → dense projection → LSTM decoder.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Dense, LSTM, Embedding, Dropout,
    Add, LayerNormalization, Reshape,
)
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard,
)
import logging

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Encoder (CNN — InceptionV3)
# ------------------------------------------------------------------

class ImageEncoder(tf.keras.layers.Layer):
    """
    Wraps InceptionV3 (frozen) + a Dense projection to embed_dim.
    Input  : raw image batch (B, 299, 299, 3)  OR pre-extracted features (B, 2048)
    Output : (B, embed_dim)
    """

    def __init__(self, embed_dim: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.project   = Dense(embed_dim, activation="relu")
        self.norm      = LayerNormalization()

    def call(self, features, training=False):
        x = self.project(features)
        return self.norm(x)


# ------------------------------------------------------------------
# Decoder (LSTM)
# ------------------------------------------------------------------

class CaptionDecoder(tf.keras.layers.Layer):
    """
    LSTM-based caption generator.
    Input  : image embedding (B, embed_dim)  +  partial caption sequence (B, T)
    Output : logits over vocabulary (B, vocab_size)
    """

    def __init__(self, vocab_size: int, embed_dim: int = 256, units: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.embedding = Embedding(vocab_size, embed_dim, mask_zero=True)
        self.lstm      = LSTM(units, return_sequences=False, return_state=False)
        self.dropout   = Dropout(0.5)
        self.add       = Add()
        self.norm      = LayerNormalization()
        self.fc        = Dense(vocab_size, activation="softmax")
        self.units     = units

    def call(self, image_embed, seq, training=False):
        # Embed the partial caption
        seq_embed = self.embedding(seq)                       # (B, T, embed_dim)

        # Broadcast image embedding to match sequence length
        img_expand = tf.expand_dims(image_embed, 1)           # (B, 1, embed_dim)
        img_tiled  = tf.tile(img_expand, [1, tf.shape(seq_embed)[1], 1])

        # Merge image context into sequence
        merged = self.add([img_tiled, seq_embed])             # (B, T, embed_dim)
        merged = self.norm(merged)

        # LSTM over merged context
        lstm_out = self.lstm(merged)                          # (B, units)
        lstm_out = self.dropout(lstm_out, training=training)

        return self.fc(lstm_out)                              # (B, vocab_size)


# ------------------------------------------------------------------
# Full captioning model
# ------------------------------------------------------------------

def build_captioning_model(
    vocab_size: int,
    max_caption_len: int,
    embed_dim: int = 256,
    lstm_units: int = 512,
    feature_dim: int = 2048,
) -> Model:
    """
    Build and compile the CNN-LSTM captioning model.

    Parameters
    ----------
    vocab_size       : size of the vocabulary
    max_caption_len  : maximum sequence length (time steps)
    embed_dim        : embedding / projection dimension
    lstm_units       : LSTM hidden state size
    feature_dim      : dimension of pre-extracted CNN features

    Returns
    -------
    Compiled Keras Model
    """
    # Inputs
    image_input = Input(shape=(feature_dim,), name="image_features")
    seq_input   = Input(shape=(max_caption_len,), name="caption_input")

    # Layers
    encoder = ImageEncoder(embed_dim=embed_dim, name="image_encoder")
    decoder = CaptionDecoder(vocab_size, embed_dim, lstm_units, name="caption_decoder")

    # Forward pass
    img_embed = encoder(image_input)
    outputs   = decoder(img_embed, seq_input)

    model = Model(inputs=[image_input, seq_input], outputs=outputs, name="ImageCaptioner")
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    logger.info(model.summary())
    return model


# ------------------------------------------------------------------
# Callbacks factory
# ------------------------------------------------------------------

def get_callbacks(checkpoint_path: str = "models/best_caption_model.keras") -> list:
    return [
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        TensorBoard(log_dir="outputs/logs", histogram_freq=1),
    ]


# ------------------------------------------------------------------
# Greedy inference
# ------------------------------------------------------------------

def generate_caption_greedy(
    model: Model,
    image_feature: np.ndarray,
    vocab,
    max_len: int = 40,
) -> str:
    """
    Greedy decoder: at each step take the argmax token.

    Parameters
    ----------
    model         : trained captioning model
    image_feature : (feature_dim,) numpy array
    vocab         : Vocabulary instance
    max_len       : maximum caption length

    Returns
    -------
    str  generated caption
    """
    start_idx = vocab.word2idx["<start>"]
    end_idx   = vocab.word2idx["<end>"]

    seq = [start_idx]

    from tensorflow.keras.preprocessing.sequence import pad_sequences

    for _ in range(max_len):
        padded  = pad_sequences([seq], maxlen=max_len, padding="post")
        img_in  = np.expand_dims(image_feature, 0)
        pred    = model.predict([img_in, padded], verbose=0)
        next_id = int(np.argmax(pred[0]))
        if next_id == end_idx:
            break
        seq.append(next_id)

    return vocab.decode(seq[1:])   # skip <start>


# ------------------------------------------------------------------
# Beam-search inference
# ------------------------------------------------------------------

def generate_caption_beam(
    model: Model,
    image_feature: np.ndarray,
    vocab,
    beam_width: int = 5,
    max_len: int = 40,
) -> str:
    """
    Beam-search decoder for higher quality captions.
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    start_idx = vocab.word2idx["<start>"]
    end_idx   = vocab.word2idx["<end>"]
    img_in    = np.expand_dims(image_feature, 0)

    # Each beam: (log_prob, token_list)
    beams = [(0.0, [start_idx])]
    completed = []

    for _ in range(max_len):
        candidates = []
        for score, seq in beams:
            if seq[-1] == end_idx:
                completed.append((score, seq))
                continue
            padded = pad_sequences([seq], maxlen=max_len, padding="post")
            probs  = model.predict([img_in, padded], verbose=0)[0]
            top_k  = np.argsort(probs)[-beam_width:]
            for idx in top_k:
                candidates.append((score + np.log(probs[idx] + 1e-10), seq + [idx]))

        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]
        if not beams:
            break

    completed += beams
    best = max(completed, key=lambda x: x[0] / len(x[1]))
    return vocab.decode(best[1][1:])   # skip <start>
