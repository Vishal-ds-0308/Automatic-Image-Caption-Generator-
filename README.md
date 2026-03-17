# 🖼️ Automatic Image Caption Generator

> A deep learning system that generates natural language descriptions for images using a CNN-LSTM architecture (InceptionV3 encoder + LSTM decoder) with beam-search decoding.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Tech Stack](#tech-stack)

---

## Overview

This project implements **image captioning** — the task of automatically generating a descriptive sentence for a given image. The model learns a joint representation of visual content (via CNN) and natural language (via LSTM) to produce fluent, contextually relevant captions.

Key capabilities:
- Supports **greedy** and **beam-search** decoding
- Pre-computes CNN features for fast training
- BLEU score evaluation on held-out captions
- Batch inference for entire image directories

---

## Architecture

```
Input Image (299×299×3)
       │
  InceptionV3 (pretrained, frozen)
       │
  Dense projection → (embed_dim)
       │
  ┌────┴────────────────────┐
  │  Partial Caption Tokens  │
  │  ──► Embedding Layer     │
  └────────────┬────────────┘
               │  Add (image + word embeddings)
               ▼
           LSTM (512 units)
               │
           Dense (vocab_size)
               │
          Next Word Probability
```

---

## Project Structure

```
Automatic_Image_Caption_Generator/
├── data/
│   ├── images/                        # Raw image files (.jpg / .png)
│   └── captions/
│       └── captions.txt               # Tab-separated image-caption pairs
├── src/
│   ├── data_preprocessing.py          # Vocab, feature extraction, sequence builder
│   ├── model.py                       # CNN encoder + LSTM decoder + beam search
│   ├── train.py                       # Training pipeline with callbacks
│   └── caption_generator.py          # Inference, BLEU evaluation, CLI
├── models/
│   ├── best_caption_model.keras       # Trained Keras model
│   ├── vocabulary.pkl                 # Serialised Vocabulary object
│   └── image_features.pkl            # Pre-extracted CNN features
├── notebooks/
│   └── demo.ipynb                     # Interactive demo notebook
├── outputs/
│   ├── training_history.png
│   └── captions.csv
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/yourusername/Automatic_Image_Caption_Generator.git
cd Automatic_Image_Caption_Generator
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Dataset

This project is designed for the **Flickr8k** or **MS-COCO** datasets.

**Captions file format** (`captions.txt`):
```
image1.jpg#0    A dog playing in the grass .
image1.jpg#1    A brown dog runs through a field .
image2.jpg#0    Two cats sitting on a window ledge .
```

---

## Usage

### 1. Extract image features (run once)
```bash
python src/data_preprocessing.py
```

### 2. Train the model
```bash
python src/train.py
```

### 3. Caption a single image
```bash
python src/caption_generator.py --image path/to/image.jpg --method beam
```

### 4. Batch caption a folder
```bash
python src/caption_generator.py --batch data/images/ --output outputs/captions.csv
```

---

## Evaluation

| Metric  | Score  |
|---------|--------|
| BLEU-1  | 0.621  |
| BLEU-2  | 0.423  |
| BLEU-3  | 0.291  |
| BLEU-4  | 0.198  |

> *Scores are indicative targets; actual results depend on dataset and training duration.*

---

## Tech Stack

| Library          | Purpose                         |
|------------------|---------------------------------|
| `TensorFlow`     | Model building & training       |
| `Keras`          | High-level neural network API   |
| `CNN (InceptionV3)` | Visual feature extraction    |
| `LSTM`           | Sequential caption generation   |
| `NumPy`          | Array operations                |
| `Matplotlib`     | Training plots & visualisation  |
| `NLTK`           | BLEU score computation          |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
