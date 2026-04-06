"""
predictor.py — CNN inference module
=====================================
Loaded once at Flask startup. Reads class_names.json written by train.py,
so it works automatically with ANY dataset folder naming.

Call predict(image_path) → dict with label, confidence, treatments hint.
"""

import os
import json
import numpy as np
from PIL import Image, UnidentifiedImageError

# Lazy-load TensorFlow to keep Flask startup fast
_model       = None
_class_names = None   # list of friendly labels e.g. ["Apple Scab", "Healthy", ...]
_img_size    = 224
MODEL_PATH = os.path.join(os.path.dirname(__file__), "apple_disease_model.h5")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "class_names.json")

# ── Keyword → MySQL disease name mapping ─────────────────────────────────────
# Maps substrings of CNN labels (lowercase) to the diseases.name column in MySQL.
# Add more entries here if you add diseases to the database.
LABEL_TO_DB = {
    "scab"    : "Apple Scab",
    "black"   : "Black Rot",
    "rot"     : "Black Rot",
    "cedar"   : "Cedar Apple Rust",
    "rust"    : "Cedar Apple Rust",
    "healthy" : None,   # healthy trees have no disease record
}

def _label_to_db_name(label: str):
    """Map a CNN output label to the diseases.name value in MySQL."""
    lower = label.lower()
    for keyword, db_name in LABEL_TO_DB.items():
        if keyword in lower:
            return db_name
    return None   # unknown — will show a generic warning in the UI


# ── Model loading ─────────────────────────────────────────────────────────────
def _load():
    global _model, _class_names, _img_size

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Run  python cnn/train.py  first."
        )
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            f"class_names.json not found at {CONFIG_PATH}. "
            "Re-run  python cnn/train.py  to regenerate it."
        )

    import tensorflow as tf
    _model = tf.keras.models.load_model(MODEL_PATH)

    with open(CONFIG_PATH) as f:
        cfg = json.load(f)

    _class_names = cfg["class_names"]
    _img_size    = cfg.get("img_size", 224)


def is_ready() -> bool:
    """Return True only if both the model file AND class map exist."""
    return os.path.exists(MODEL_PATH) and os.path.exists(CONFIG_PATH)


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(image_path: str) -> dict:
    """
    Run the CNN on one image file.

    Returns:
    {
        "label"      : "Apple Scab",        # top predicted class
        "confidence" : 94.2,                 # percent, 0-100
        "is_healthy" : False,
        "db_name"    : "Apple Scab",         # matches diseases.name in MySQL
        "all_scores" : [                     # all classes, best first
            {"label": "Apple Scab", "confidence": 94.2},
            ...
        ]
    }
    """
    global _model, _class_names

    if _model is None:
        _load()

    # Pre-process
    try:
        img = Image.open(image_path).convert("RGB")
    except (UnidentifiedImageError, Exception) as e:
        raise ValueError(f"Cannot open image: {e}")

    img   = img.resize((_img_size, _img_size))
    arr   = np.array(img, dtype=np.float32) / 255.0
    batch = np.expand_dims(arr, axis=0)   # (1, H, W, 3)

    # Inference
    probs    = _model.predict(batch, verbose=0)[0]
    top_idx  = int(np.argmax(probs))
    top_label = _class_names[top_idx]
    top_conf  = float(probs[top_idx]) * 100.0

    all_scores = sorted(
        [
            {
                "label"      : _class_names[i],
                "confidence" : round(float(probs[i]) * 100, 2),
            }
            for i in range(len(_class_names))
        ],
        key=lambda x: x["confidence"],
        reverse=True,
    )

    return {
        "label"      : top_label,
        "confidence" : round(top_conf, 2),
        "is_healthy" : "healthy" in top_label.lower(),
        "db_name"    : _label_to_db_name(top_label),
        "all_scores" : all_scores,
    }


# Keep this for backward compatibility with app.py
def db_disease_name(cnn_label: str):
    return _label_to_db_name(cnn_label)