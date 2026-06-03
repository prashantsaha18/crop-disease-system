"""
train.py
--------
Fine-tune EfficientNetV2-S on PlantVillage (38 disease classes, ~54 k images).

Pipeline
────────
1. Load & split PlantVillage with tf.data (80 / 10 / 10 train/val/test).
2. Apply augmentation pipeline on training data.
3. Phase 1 – freeze base, train classification head  → 5 epochs.
4. Phase 2 – unfreeze top-20 layers, fine-tune end-to-end → 10 epochs.
5. Evaluate on val set; assert ≥ 90 % accuracy.
6. Save full Keras model → saved_model/crop_disease_model/

Usage
─────
  python train.py --data_dir /path/to/PlantVillage \
                  --output_dir saved_model/crop_disease_model \
                  --batch_size 32

PlantVillage directory layout expected:
  <data_dir>/
      Apple___Apple_scab/   *.jpg
      Apple___Black_rot/    *.jpg
      ...                   (38 sub-dirs total)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

# ─────────────────────────── constants ────────────────────────────────────── #

IMG_SIZE      = (224, 224)
NUM_CLASSES   = 38
AUTOTUNE      = tf.data.AUTOTUNE
SEED          = 42

PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 10
UNFREEZE_N    = 20       # unfreeze top-N layers of the base

PHASE1_LR     = 1e-3
PHASE2_LR     = 1e-4

# ─────────────────────────── augmentation ─────────────────────────────────── #

def build_augmentation() -> keras.Sequential:
    """Random augmentation applied only during training."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=30 / 360),        # ±30°
            layers.RandomZoom(height_factor=0.2),
            layers.RandomBrightness(factor=0.2),
            layers.RandomContrast(factor=0.1),
        ],
        name="augmentation",
    )


# ─────────────────────────── dataset ──────────────────────────────────────── #

def load_dataset(
    data_dir: str,
    batch_size: int,
    val_split: float = 0.1,
    test_split: float = 0.1,
):
    """
    Load PlantVillage from directory structure using image_dataset_from_directory.
    Returns (train_ds, val_ds, test_ds, class_names).
    """
    base_dir = Path(data_dir)
    if not base_dir.exists():
        sys.exit(f"[ERROR] data_dir not found: {data_dir}")

    # ── collect all class dirs ──────────────────────────────────────────── #
    class_dirs = sorted([d.name for d in base_dir.iterdir() if d.is_dir()])
    print(f"[INFO] Found {len(class_dirs)} class directories.")

    # ── full dataset (no validation_split first; we do manual split) ─────── #
    full_ds = keras.utils.image_dataset_from_directory(
        base_dir,
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=None,           # unbatched so we can shuffle+split
        shuffle=True,
        seed=SEED,
    )
    class_names = full_ds.class_names
    n_total = full_ds.cardinality().numpy()
    print(f"[INFO] Total images: {n_total}, classes: {len(class_names)}")

    # ── manual split ────────────────────────────────────────────────────── #
    n_val  = int(n_total * val_split)
    n_test = int(n_total * test_split)
    n_train = n_total - n_val - n_test

    full_ds = full_ds.shuffle(buffer_size=min(n_total, 10_000), seed=SEED)
    train_ds  = full_ds.take(n_train)
    val_ds    = full_ds.skip(n_train).take(n_val)
    test_ds   = full_ds.skip(n_train + n_val)

    # ── preprocessing helper ─────────────────────────────────────────────── #
    # EfficientNetV2 expects pixels in [0, 255]; preprocess_input handles the rest.
    preprocess = tf.keras.applications.efficientnet_v2.preprocess_input

    augment = build_augmentation()

    def prepare_train(img, lbl):
        img = tf.cast(img, tf.float32)
        img = augment(img, training=True)
        img = preprocess(img)
        lbl = tf.one_hot(lbl, NUM_CLASSES)
        return img, lbl

    def prepare_eval(img, lbl):
        img = tf.cast(img, tf.float32)
        img = preprocess(img)
        lbl = tf.one_hot(lbl, NUM_CLASSES)
        return img, lbl

    train_ds = (
        train_ds.map(prepare_train, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        val_ds.map(prepare_eval, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    test_ds = (
        test_ds.map(prepare_eval, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds, test_ds, class_names


# ─────────────────────────── model ────────────────────────────────────────── #

def build_model(num_classes: int = NUM_CLASSES) -> keras.Model:
    """
    EfficientNetV2-S base + custom classification head.
    Base is frozen initially (Phase 1).
    """
    base = EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
        include_preprocessing=False,   # we do our own preprocess_input
    )
    base.trainable = False             # freeze entire base

    inputs  = keras.Input(shape=(*IMG_SIZE, 3), name="input_image")
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D(name="gap")(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(0.4, name="top_dropout")(x)
    x       = layers.Dense(512, activation="relu", name="dense_512")(x)
    x       = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="CropDiseaseNet")
    return model, base


def unfreeze_top_n(base: keras.Model, n: int) -> None:
    """Unfreeze the top-N layers of the base model for fine-tuning."""
    base.trainable = True
    # freeze all but the last n layers
    for layer in base.layers[:-n]:
        layer.trainable = False
    frozen = sum(1 for l in base.layers if not l.trainable)
    total  = len(base.layers)
    print(f"[INFO] Base layers frozen: {frozen}/{total}  (top {n} unfrozen)")


# ─────────────────────────── callbacks ────────────────────────────────────── #

def get_callbacks(phase: int, ckpt_dir: str) -> list:
    ckpt_path = os.path.join(ckpt_dir, f"phase{phase}_best.keras")
    return [
        ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


# ─────────────────────────── train ────────────────────────────────────────── #

def train(args):
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # Set mixed precision only if GPU is available to prevent CPU numerical issues
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"[INFO] Compute dtype: {policy.compute_dtype}")
    else:
        print("[INFO] GPU not available. Using default float32 precision.")

    # ── data ─────────────────────────────────────────────────────────────── #
    print("\n[INFO] Loading dataset ...")
    train_ds, val_ds, test_ds, class_names = load_dataset(
        args.data_dir, args.batch_size
    )
    print(f"[INFO] Classes ({len(class_names)}): {class_names[:5]} ... {class_names[-3:]}")

    # ── model ─────────────────────────────────────────────────────────────── #
    print("\n[INFO] Building model ...")
    model, base = build_model(len(class_names))
    model.summary(line_length=100)

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ======================================================================== #
    #  PHASE 1 - train head only
    # ======================================================================== #
    print("\n" + "=" * 60)
    print("  PHASE 1  -  Training classification head (base frozen)")
    print("=" * 60)

    model.compile(
        optimizer=keras.optimizers.Adam(PHASE1_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    t0 = time.time()
    hist1 = model.fit(
        train_ds,
        epochs=PHASE1_EPOCHS,
        validation_data=val_ds,
        callbacks=get_callbacks(1, args.ckpt_dir),
        verbose=1,
    )
    print(f"[INFO] Phase 1 done in {(time.time()-t0)/60:.1f} min")
    print(f"       Best val_accuracy: {max(hist1.history['val_accuracy']):.4f}")

    # ======================================================================== #
    #  PHASE 2 - unfreeze top-20 layers, fine-tune
    # ======================================================================== #
    print("\n" + "=" * 60)
    print("  PHASE 2  -  Fine-tuning top-20 base layers")
    print("=" * 60)

    unfreeze_top_n(base, UNFREEZE_N)

    model.compile(
        optimizer=keras.optimizers.Adam(PHASE2_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    t0 = time.time()
    hist2 = model.fit(
        train_ds,
        epochs=PHASE2_EPOCHS,
        validation_data=val_ds,
        callbacks=get_callbacks(2, args.ckpt_dir),
        verbose=1,
    )
    print(f"[INFO] Phase 2 done in {(time.time()-t0)/60:.1f} min")
    best_val_acc = max(hist2.history["val_accuracy"])
    print(f"       Best val_accuracy: {best_val_acc:.4f}")

    # ── target check ─────────────────────────────────────────────────────── #
    if best_val_acc < 0.90:
        print(
            f"\n[WARNING] Val accuracy {best_val_acc:.4f} < 0.90 target. "
            "Consider more epochs or different augmentation."
        )
    else:
        print(f"\n[SUCCESS] >= 90 % validation accuracy achieved: {best_val_acc:.4f}")

    # ── test evaluation ───────────────────────────────────────────────────── #
    print("\n[INFO] Evaluating on held-out test set ...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"[INFO] Test accuracy: {test_acc:.4f}")

    # ── save model ────────────────────────────────────────────────────────── #
    out_dir = args.output_dir
    parent_dir = os.path.dirname(out_dir)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    model.save(out_dir)
    print(f"\n[INFO] Full model saved -> {out_dir}")

    # ── save class names ──────────────────────────────────────────────────── #
    labels_path = os.path.join(parent_dir, "class_names.txt")
    with open(labels_path, "w") as f:
        f.write("\n".join(class_names))
    print(f"[INFO] Class names saved -> {labels_path}")

    return model, class_names


# ─────────────────────────── CLI ──────────────────────────────────────────── #

def parse_args():
    p = argparse.ArgumentParser(description="Train crop disease classifier.")
    p.add_argument("--data_dir",    default="data/PlantVillage",
                   help="Root directory of PlantVillage dataset.")
    p.add_argument("--output_dir",  default="saved_model/crop_disease_model.keras",
                   help="Where to save the final Keras model.")
    p.add_argument("--ckpt_dir",    default="checkpoints",
                   help="Directory to save best-epoch checkpoints.")
    p.add_argument("--batch_size",  type=int, default=32)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"[INFO] TensorFlow {tf.__version__}")
    print(f"[INFO] GPUs available: {tf.config.list_physical_devices('GPU')}")
    train(args)
