"""
convert_to_tflite.py
────────────────────
Convert the trained Keras model to TFLite with float16 post-training
quantization, then benchmark original vs quantized on 100 sample images.

Outputs
───────
  crop_model.tflite          – float16 quantized TFLite model
  crop_model_float32.tflite  – baseline float32 TFLite model (for comparison)

Usage
─────
  python convert_to_tflite.py \
      --model_dir  saved_model/crop_disease_model \
      --data_dir   data/PlantVillage \
      --output_dir tflite_models

Benchmark output (printed to stdout)
─────────────────────────────────────
  ┌─────────────────────────────────────────────────────────┐
  │  Benchmark Results (n=100 images)                       │
  │  Float32 model: size=XX.X MB, latency=XX.X ms/img       │
  │  Float16 model: size=XX.X MB, latency=XX.X ms/img       │
  │  Size reduction:    XX.X %                              │
  │  Latency improvement: XX.X %                            │
  └─────────────────────────────────────────────────────────┘
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

IMG_SIZE   = (224, 224)
N_BENCH    = 100        # number of images for benchmarking
AUTOTUNE   = tf.data.AUTOTUNE
SEED       = 42


# ─────────────────────────── conversion ───────────────────────────────────── #

def convert_float32(model: keras.Model, out_path: str) -> str:
    """Convert to TFLite float32 (no quantization) for baseline comparison."""
    print("[Convert] Building float32 TFLite model ...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    _save(tflite_model, out_path)
    mb = os.path.getsize(out_path) / 1e6
    print(f"[Convert] float32 TFLite  ->  {out_path}  ({mb:.2f} MB)")
    return out_path


def convert_float16(model: keras.Model, out_path: str) -> str:
    """
    Post-training float16 quantization.
    Weights are quantized to float16; activations remain float32 at runtime
    on most CPU/GPU targets, but the model file is ~50 % smaller.
    """
    print("[Convert] Building float16 quantized TFLite model ...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # -- post-training quantization settings ------------------------------- #
    converter.optimizations         = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    # allow TF ops not yet in the TFLite kernel set (needed for some EfficientNet ops)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    tflite_model = converter.convert()
    _save(tflite_model, out_path)
    mb = os.path.getsize(out_path) / 1e6
    print(f"[Convert] float16 TFLite  ->  {out_path}  ({mb:.2f} MB)")
    return out_path


def _save(model_bytes: bytes, path: str) -> None:
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, "wb") as f:
        f.write(model_bytes)


# ─────────────────────────── inference helpers ────────────────────────────── #

class TFLiteRunner:
    """Thin wrapper around tf.lite.Interpreter for single-image inference."""

    def __init__(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_idx  = self.input_details[0]["index"]
        self.output_idx = self.output_details[0]["index"]
        self.input_dtype = self.input_details[0]["dtype"]

    def predict(self, img_array: np.ndarray) -> np.ndarray:
        """
        img_array : (H, W, 3) uint8 or float32, will be pre-processed internally.
        Returns   : (num_classes,) softmax probabilities.
        """
        # Scale to [-1, 1] as expected by the wrapped model
        prep = (img_array.astype(np.float32) / 127.5) - 1.0
        batch = np.expand_dims(prep, axis=0).astype(self.input_dtype)
        self.interpreter.set_tensor(self.input_idx, batch)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_idx)[0]


# ─────────────────────────── benchmark ────────────────────────────────────── #

def load_sample_images(data_dir: str, n: int = N_BENCH) -> list:
    """
    Collect up to `n` image paths from the PlantVillage directory.
    Samples uniformly across all class sub-directories.
    """
    base = Path(data_dir)
    all_paths = []
    for cls_dir in sorted(base.iterdir()):
        if not cls_dir.is_dir():
            continue
        imgs = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.JPG")) \
             + sorted(cls_dir.glob("*.png"))
        all_paths.extend(imgs)

    rng = np.random.default_rng(SEED)
    chosen = rng.choice(all_paths, size=min(n, len(all_paths)), replace=False)
    print(f"[Bench] Sampled {len(chosen)} images from {data_dir}")
    return list(chosen)


def preload_images(paths: list) -> np.ndarray:
    """Load + resize all images to (N, 224, 224, 3) uint8."""
    import cv2
    imgs = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        imgs.append(img)
    return np.array(imgs, dtype=np.uint8)


def benchmark_model(runner: TFLiteRunner, images: np.ndarray) -> dict:
    """
    Run inference on all images, measure per-image latency.
    First 5 runs are warm-up (excluded from stats).
    """
    warmup = min(5, len(images))
    for i in range(warmup):
        runner.predict(images[i])

    latencies = []
    for img in images:
        t0 = time.perf_counter()
        runner.predict(img)
        latencies.append((time.perf_counter() - t0) * 1000)   # ms

    return {
        "mean_ms":   np.mean(latencies),
        "median_ms": np.median(latencies),
        "p95_ms":    np.percentile(latencies, 95),
        "n":         len(latencies),
    }


def run_benchmark(
    fp32_path: str,
    fp16_path: str,
    data_dir: str,
    n: int = N_BENCH,
) -> None:
    """Full benchmark: load models + images, measure, print table."""

    # ── load images ───────────────────────────────────────────────────────── #
    print(f"\n[Bench] Loading {n} sample images ...")
    if os.path.isdir(data_dir):
        paths  = load_sample_images(data_dir, n)
        images = preload_images(paths)
    else:
        print("[Bench] data_dir not found - generating random noise images.")
        rng    = np.random.default_rng(SEED)
        images = rng.integers(0, 255, size=(n, *IMG_SIZE, 3), dtype=np.uint8)

    print(f"[Bench] Images loaded: {images.shape}")

    # -- run ----------------------------------------------------------------- #
    print("\n[Bench] Benchmarking float32 model ...")
    fp32_runner = TFLiteRunner(fp32_path)
    fp32_stats  = benchmark_model(fp32_runner, images)

    print("[Bench] Benchmarking float16 model ...")
    fp16_runner = TFLiteRunner(fp16_path)
    fp16_stats  = benchmark_model(fp16_runner, images)

    # ── sizes ─────────────────────────────────────────────────────────────── #
    fp32_mb = os.path.getsize(fp32_path) / 1e6
    fp16_mb = os.path.getsize(fp16_path) / 1e6

    size_reduction   = (1 - fp16_mb / fp32_mb) * 100
    latency_improvement = (
        (1 - fp16_stats["mean_ms"] / fp32_stats["mean_ms"]) * 100
        if fp32_stats["mean_ms"] > 0 else 0.0
    )

    # ── report ────────────────────────────────────────────────────────────── #
    border = "-" * 60
    print(f"\n{'='*60}")
    print(f"  Benchmark Results  (n={fp32_stats['n']} images)")
    print(f"{'='*60}")
    print(f"  {'Model':<20}  {'Size (MB)':>10}  {'Mean ms':>10}  {'P95 ms':>10}")
    print(f"  {border}")
    print(f"  {'Float32 baseline':<20}  {fp32_mb:>10.2f}  "
          f"{fp32_stats['mean_ms']:>10.2f}  {fp32_stats['p95_ms']:>10.2f}")
    print(f"  {'Float16 quantized':<20}  {fp16_mb:>10.2f}  "
          f"{fp16_stats['mean_ms']:>10.2f}  {fp16_stats['p95_ms']:>10.2f}")
    print(f"  {border}")
    print(f"  Size reduction      : {size_reduction:+.1f} %")
    if latency_improvement >= 0:
        print(f"  Latency improvement : {latency_improvement:+.1f} %  (lower is better)")
    else:
        print(f"  Latency overhead    : {-latency_improvement:.1f} %  "
              "(float16 slower on this CPU - normal for CPU-only runtimes)")
    print(f"{'='*60}\n")


# ─────────────────────────── accuracy check ───────────────────────────────── #

def verify_accuracy_parity(
    original_model: keras.Model,
    tflite_runner: TFLiteRunner,
    images: np.ndarray,
    tolerance: float = 0.01,
) -> None:
    """
    Spot-check that TFLite predictions match the original Keras model
    on a small random subset (top-1 agreement rate).
    """
    n = min(20, len(images))
    matches = 0
    for img in images[:n]:
        # Scale to [-1, 1] as expected by the wrapped model
        prep = (img.astype(np.float32) / 127.5) - 1.0
        keras_pred = int(np.argmax(
            original_model.predict(np.expand_dims(prep, 0), verbose=0)[0]
        ))
        tflite_pred = int(np.argmax(tflite_runner.predict(img)))
        if keras_pred == tflite_pred:
            matches += 1
    agreement = matches / n * 100
    print(f"[Verify] Keras ↔ TFLite top-1 agreement on {n} images: {agreement:.1f}%")
    if agreement < (1 - tolerance) * 100:
        print("[Verify] WARNING: agreement below 99% – check conversion settings.")


# ─────────────────────────── main ─────────────────────────────────────────── #

def parse_args():
    p = argparse.ArgumentParser(description="Convert Keras model to TFLite.")
    p.add_argument("--model_dir",  default="streamlit_app/saved_model/crop_disease_model.keras")
    p.add_argument("--data_dir",   default="data/PlantVillage",
                   help="PlantVillage root dir (for benchmark images).")
    p.add_argument("--output_dir", default="tflite_models")
    p.add_argument("--n_bench",    type=int, default=N_BENCH,
                   help="Number of images for benchmark.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"[INFO] TensorFlow {tf.__version__}")
    print(f"[INFO] Loading Keras model from {args.model_dir} ...")
    model = keras.models.load_model(args.model_dir)
    model.summary(line_length=100, expand_nested=False)

    os.makedirs(args.output_dir, exist_ok=True)

    fp32_path = os.path.join(args.output_dir, "crop_model_float32.tflite")
    fp16_path = os.path.join(args.output_dir, "crop_model.tflite")

    # ── convert ───────────────────────────────────────────────────────────── #
    convert_float32(model, fp32_path)
    convert_float16(model, fp16_path)

    # ── benchmark ─────────────────────────────────────────────────────────── #
    run_benchmark(fp32_path, fp16_path, args.data_dir, n=args.n_bench)

    # ── copy final model for Flutter asset ────────────────────────────────── #
    flutter_asset = os.path.join(
        os.path.dirname(__file__), "..", "flutter_app", "assets", "models", "crop_model.tflite"
    )
    os.makedirs(os.path.dirname(flutter_asset), exist_ok=True)
    import shutil
    shutil.copy2(fp16_path, flutter_asset)
    print(f"[INFO] Copied model to Flutter assets: {flutter_asset}")

    print("\n[DONE] TFLite conversion complete.")
    print(f"       Quantized model (float16): {fp16_path}")
    print(f"       Bundle this file as a Flutter asset in: assets/models/crop_model.tflite")
