"""
CNN Training Script — Apple Leaf Disease Detection
====================================================
Dataset : Any apple disease dataset with one subfolder per class.
          Works with PlantVillage naming (Apple___Apple_scab/)
          AND with plain naming (apple_scab/, scab/, Scab/, etc.)

          Tested with:
            https://www.kaggle.com/datasets/showravdhar/apple-disease-dataset
            https://www.kaggle.com/datasets/emmarex/plantdisease

Model   : MobileNetV2 (transfer learning) — runs on CPU, no GPU needed
Output  : cnn/apple_disease_model.h5  +  cnn/class_names.json

Usage:
    pip install tensorflow pillow scikit-learn matplotlib seaborn
    python cnn/train.py --data_dir data/ --epochs 15
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")   # works without a display / GUI
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 32
SEED       = 42

# ── Smart label normaliser ────────────────────────────────────────────────────
# Maps many possible folder-name variations → one clean label.
# Any folder name containing a keyword (case-insensitive) gets that label.
KEYWORD_TO_LABEL = {
    "scab"    : "Apple Scab",
    "black"   : "Black Rot",
    "rot"     : "Black Rot",
    "cedar"   : "Cedar Apple Rust",
    "rust"    : "Cedar Apple Rust",
    "healthy" : "Healthy",
}

def folder_to_label(folder_name: str) -> str:
    """Convert any folder name variant to a clean human-readable label."""
    lower = folder_name.lower()
    for keyword, label in KEYWORD_TO_LABEL.items():
        if keyword in lower:
            return label
    # Fallback: strip Apple___ prefix, replace underscores, title-case
    clean = folder_name.replace("Apple___", "").replace("_", " ").replace("-", " ").strip()
    return clean.title()


# ── Dataset finder ────────────────────────────────────────────────────────────
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def _count_images(folder: str) -> int:
    """Count images directly inside folder OR one level deeper (handles nested datasets)."""
    count = 0
    try:
        for entry in os.listdir(folder):
            full = os.path.join(folder, entry)
            if entry.lower().endswith(IMG_EXTS):
                count += 1
            elif os.path.isdir(full):
                # one level deeper — e.g. Apple___Apple_scab/Apple___Apple_scab/*.jpg
                for f in os.listdir(full):
                    if f.lower().endswith(IMG_EXTS):
                        count += 1
    except PermissionError:
        pass
    return count


def find_data_dir(root: str) -> str:
    """
    Walk up to 3 levels inside root to find a folder that directly contains
    class subfolders (each holding images at any depth).
    """
    root = os.path.abspath(root)

    def _has_image_subfolders(path: str, min_classes: int = 2) -> bool:
        try:
            subdirs = [
                d for d in os.listdir(path)
                if os.path.isdir(os.path.join(path, d)) and not d.startswith(".")
            ]
            if len(subdirs) < min_classes:
                return False
            # ALL subdirs must contain images (not just one)
            hits = sum(1 for sd in subdirs if _count_images(os.path.join(path, sd)) > 0)
            return hits >= min_classes
        except PermissionError:
            return False

    if _has_image_subfolders(root):
        return root

    for dirpath, _, _ in os.walk(root):
        depth = os.path.relpath(dirpath, root).count(os.sep)
        if depth > 3:
            continue
        if _has_image_subfolders(dirpath):
            print(f"  Auto-detected class folders inside: {dirpath}")
            return dirpath

    print(f"\n  No image class folders found under: {root}")
    print("  Top-level contents:")
    try:
        for name in sorted(os.listdir(root))[:20]:
            sub = os.path.join(root, name)
            imgs = _count_images(sub) if os.path.isdir(sub) else 0
            print(f"    {name}/  ({imgs} images found)")
    except Exception:
        pass
    print(
        "\n  Your data folder should look like:\n"
        "    data/\n"
        "     Apple___Apple_scab/   <- images directly inside, or in a subfolder\n"
        "      Apple___Black_rot/\n"
        "      Apple___Cedar_apple_rust/\n"
        "      Apple___healthy/\n"
    )
    sys.exit(1)


def list_classes(data_dir: str) -> list:
    """Return sorted list of class folder names that contain images."""
    classes = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if not os.path.isdir(path) or name.startswith("."):
            continue
        count = _count_images(path)
        if count > 0:
            classes.append(name)
            print(f"  [{name}]  ->  label: '{folder_to_label(name)}'  ({count} images)")
    return classes


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(num_classes: int) -> tf.keras.Model:
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # frozen in Phase 1

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ], name="apple_disease_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def fine_tune(model: tf.keras.Model) -> tf.keras.Model:
    """Unfreeze top 30 layers of MobileNetV2 for fine-tuning."""
    base = model.layers[0]
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Data generators ───────────────────────────────────────────────────────────
def get_generators(data_dir: str, class_folders: list):
    """
    Build train/val generators. Handles two layouts:
      Layout A: data/Apple___Apple_scab/img001.jpg        (images directly inside)
      Layout B: data/Apple___Apple_scab/Apple___Apple_scab/img001.jpg  (one extra level)
    For Layout B we temporarily symlink/copy to a flat structure.
    Keras flow_from_directory only looks ONE level deep for images.
    """
    import shutil, tempfile

    # Check if images are directly in class folders or nested one deeper
    sample_class = class_folders[0]
    sample_path  = os.path.join(data_dir, sample_class)
    direct_imgs  = [f for f in os.listdir(sample_path) if f.lower().endswith(IMG_EXTS)]

    if not direct_imgs:
        # Images are nested — flatten into a temp directory
        tmp_dir = tempfile.mkdtemp(prefix="orchard_flat_")
        print(f"  Images are nested inside class folders.")
        print(f"  Flattening dataset into temp dir: {tmp_dir}")
        for cls in class_folders:
            cls_src = os.path.join(data_dir, cls)
            cls_dst = os.path.join(tmp_dir, cls)
            os.makedirs(cls_dst, exist_ok=True)
            # Collect images from all subfolders inside this class folder
            for root, _, files in os.walk(cls_src):
                for fname in files:
                    if fname.lower().endswith(IMG_EXTS):
                        src = os.path.join(root, fname)
                        # Use relative path as part of new filename to avoid collisions
                        rel  = os.path.relpath(src, cls_src).replace(os.sep, "_")
                        dst  = os.path.join(cls_dst, rel)
                        shutil.copy2(src, dst)
            count = len(os.listdir(cls_dst))
            print(f"    {cls}: {count} images copied")
        data_dir = tmp_dir

    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        validation_split=0.2,
    )
    val_aug = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    common = dict(
        directory=data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=class_folders,
        seed=SEED,
    )
    train_gen = train_aug.flow_from_directory(subset="training",   shuffle=True,  **common)
    val_gen   = val_aug.flow_from_directory(  subset="validation", shuffle=False, **common)
    return train_gen, val_gen


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_history(h1, h2, save_path: str):
    combined = {k: h1.history[k] + h2.history[k] for k in h1.history}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(combined["accuracy"],     label="Train")
    axes[0].plot(combined["val_accuracy"], label="Val")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(combined["loss"],     label="Train")
    axes[1].plot(combined["val_loss"], label="Val")
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    plt.suptitle("Apple Disease CNN — Training History", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Training plot saved  -> {save_path}")


def plot_confusion(model, val_gen, class_names: list, save_path: str):
    val_gen.reset()
    preds  = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Confusion Matrix — Validation Set")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Confusion matrix saved -> {save_path}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train Apple Leaf Disease CNN")
    parser.add_argument("--data_dir", default="data/",
                        help="Folder containing one subfolder per disease class")
    parser.add_argument("--epochs",   default=15, type=int,
                        help="Phase-1 training epochs (Phase 2 always runs 10 more)")
    parser.add_argument("--out_dir",  default="cnn/",
                        help="Where to save model + config files")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. Find & list classes ────────────────────────────────────────────────
    print(f"\nSearching for class folders inside: {args.data_dir}")
    data_dir      = find_data_dir(args.data_dir)
    print(f"\nClasses found:")
    class_folders = list_classes(data_dir)
    class_names   = [folder_to_label(f) for f in class_folders]
    num_classes   = len(class_folders)

    if num_classes < 2:
        print("Need at least 2 classes. Check your --data_dir.")
        sys.exit(1)

    print(f"\n{num_classes} classes: {class_names}")

    # ── 2. Save class map ─────────────────────────────────────────────────────
    config_path = os.path.join(args.out_dir, "class_names.json")
    model_path  = os.path.join(args.out_dir, "apple_disease_model.h5")

    with open(config_path, "w") as f:
        json.dump({
            "class_names"  : class_names,
            "folder_names" : class_folders,
            "img_size"     : IMG_SIZE,
        }, f, indent=2)
    print(f"Class map saved -> {config_path}\n")

    # ── 3. Load data ──────────────────────────────────────────────────────────
    print("Loading images...")
    train_gen, val_gen = get_generators(data_dir, class_folders)
    print(f"Train: {train_gen.samples} images  |  Val: {val_gen.samples} images\n")

    if train_gen.samples == 0:
        print("No images loaded. Make sure class folders contain .jpg/.png files.")
        sys.exit(1)

    callbacks = [
        ModelCheckpoint(model_path, save_best_only=True,
                        monitor="val_accuracy", verbose=1),
        EarlyStopping(monitor="val_loss", patience=5,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                          patience=3, min_lr=1e-7, verbose=1),
    ]

    # ── 4. Phase 1: train head ────────────────────────────────────────────────
    print("=" * 55)
    print("Phase 1: Training classification head (base frozen)")
    print("=" * 55)
    model    = build_model(num_classes)
    history1 = model.fit(train_gen, validation_data=val_gen,
                         epochs=args.epochs, callbacks=callbacks)

    # ── 5. Phase 2: fine-tune ─────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("Phase 2: Fine-tuning top 30 MobileNetV2 layers")
    print("=" * 55)
    model    = fine_tune(model)
    history2 = model.fit(train_gen, validation_data=val_gen,
                         epochs=10, callbacks=callbacks)

    # ── 6. Evaluate & plot ────────────────────────────────────────────────────
    plot_history(history1, history2,
                 save_path=os.path.join(args.out_dir, "training_plot.png"))

    print("\nEvaluating on validation set...")
    loss, acc = model.evaluate(val_gen, verbose=0)
    print(f"Validation Accuracy : {acc * 100:.2f}%")
    print(f"Validation Loss     : {loss:.4f}")

    plot_confusion(model, val_gen, class_names,
                   save_path=os.path.join(args.out_dir, "confusion_matrix.png"))

    print(f"\n{'=' * 55}")
    print(f"Model saved  -> {model_path}")
    print(f"Class map    -> {config_path}")
    print(f"{'=' * 55}")
    print("\nNext:  python app.py   then open /scan to use the leaf scanner.")


if __name__ == "__main__":
    main()