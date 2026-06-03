"""
gradcam.py
----------
Grad-CAM heatmap generation for the crop disease classifier.

Produces a colour heatmap overlaid on the original leaf image to highlight
which regions drove the prediction.  Uses the LAST convolutional layer of
EfficientNetV2-S as the target layer.

Usage (standalone)
──────────────────
  python gradcam.py \
      --model_dir saved_model/crop_disease_model \
      --image     path/to/leaf.jpg \
      --out_path  output/gradcam_result.jpg

Programmatic
────────────
  from gradcam import GradCAM
  gcam = GradCAM(model)
  overlay = gcam.explain(img_array)          # img_array: (H, W, 3) uint8
  cv2.imwrite("overlay.jpg", overlay[:, :, ::-1])
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# ─────────────────────────── constants ────────────────────────────────────── #

IMG_SIZE              = (224, 224)
GRADCAM_TARGET_LAYER  = "top_conv"      # last conv in EfficientNetV2-S backbone
# fallback layer names in case architecture differs between TF versions
FALLBACK_LAYER_NAMES  = [
    "top_conv",
    "block7b_project_conv",
    "block7a_project_conv",
    "block6d_project_conv",
]
HEATMAP_ALPHA         = 0.45           # blending weight for heatmap overlay
COLORMAP              = cv2.COLORMAP_JET


# ─────────────────────────── GradCAM class ────────────────────────────────── #

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Parameters
    ----------
    model       : trained Keras model
    target_layer: name of the convolutional layer to hook.
                  If None, the class auto-detects the last conv layer.
    """

    def __init__(
        self,
        model: keras.Model,
        target_layer: Optional[str] = None,
    ):
        self.model = model
        self.target_layer_name = target_layer or self._find_last_conv_layer()
        self._grad_model = self._build_grad_model()
        print(f"[GradCAM] Target layer: '{self.target_layer_name}'")

    # ── private helpers ────────────────────────────────────────────────────── #

    def _find_last_conv_layer(self) -> str:
        """
        Walk the (possibly nested) model to find the last Conv2D layer.
        Tries known EfficientNetV2 layer names first, then falls back to
        a full traversal.
        """
        # 1) try known names
        for name in FALLBACK_LAYER_NAMES:
            try:
                self.model.get_layer(name)
                return name
            except ValueError:
                pass

        # 2) full traversal (handles nested sub-models)
        last_conv_name = None
        for layer in self.model.layers:
            if isinstance(layer, keras.Model):
                for sub in layer.layers:
                    if isinstance(sub, (keras.layers.Conv2D,
                                        keras.layers.DepthwiseConv2D)):
                        last_conv_name = sub.name
            elif isinstance(layer, (keras.layers.Conv2D,
                                    keras.layers.DepthwiseConv2D)):
                last_conv_name = layer.name

        if last_conv_name is None:
            raise ValueError(
                "Could not find any convolutional layer. "
                "Pass target_layer explicitly."
            )
        return last_conv_name

    def _get_layer_from_nested(self, name: str) -> keras.layers.Layer:
        """Retrieve a layer that may live inside a sub-model."""
        try:
            return self.model.get_layer(name)
        except ValueError:
            for layer in self.model.layers:
                if isinstance(layer, keras.Model):
                    try:
                        return layer.get_layer(name)
                    except ValueError:
                        pass
        raise ValueError(f"Layer '{name}' not found in model or sub-models.")

    def _build_grad_model(self) -> keras.Model:
        """
        Build a secondary model that outputs:
          [<target_conv_layer_output>, <final_softmax_output>]
        This is required for gradient computation via GradientTape.
        """
        target_layer = self._get_layer_from_nested(self.target_layer_name)
        self.is_nested = False
        self.sub_model = None
        self.sub_model_idx = -1

        try:
            return keras.Model(
                inputs=self.model.inputs,
                outputs=[target_layer.output, self.model.outputs[0]],
                name="gradcam_model",
            )
        except (ValueError, AttributeError):
            self.is_nested = True
            for i, layer in enumerate(self.model.layers):
                if isinstance(layer, keras.Model):
                    try:
                        _ = layer.get_layer(target_layer.name)
                        self.sub_model = layer
                        self.sub_model_idx = i
                        break
                    except ValueError:
                        pass

            if self.sub_model is None:
                raise ValueError(
                    f"Layer '{self.target_layer_name}' not found in top-level or nested sub-models."
                )

            return keras.Model(
                inputs=self.sub_model.inputs,
                outputs=[target_layer.output, self.sub_model.outputs[0]],
                name="gradcam_sub_model",
            )

    def _forward_preprocess(self, inputs):
        x = inputs
        for layer in self.model.layers[:self.sub_model_idx]:
            if isinstance(layer, keras.layers.InputLayer):
                continue
            try:
                x = layer(x, training=False)
            except TypeError:
                x = layer(x)
        return x

    def _forward_head(self, sub_output):
        x = sub_output
        for layer in self.model.layers[self.sub_model_idx + 1:]:
            try:
                x = layer(x, training=False)
            except TypeError:
                x = layer(x)
        return x

    # ── core ───────────────────────────────────────────────────────────────── #

    def _compute_heatmap(
        self,
        img_array: np.ndarray,
        class_idx: Optional[int] = None,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """
        Compute the raw (grayscale) Grad-CAM heatmap.

        Parameters
        ----------
        img_array : preprocessed float32 array of shape (1, H, W, 3)
        class_idx : target class (defaults to argmax of prediction)
        eps       : small value to avoid division by zero

        Returns
        -------
        heatmap : float32 array in [0, 1], shape (hmap_H, hmap_W)
        """
        img_tensor = tf.cast(img_array, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            if not self.is_nested:
                conv_outputs, predictions = self._grad_model(img_tensor, training=False)
            else:
                sub_inputs = self._forward_preprocess(img_tensor)
                conv_outputs, sub_outputs = self._grad_model(sub_inputs, training=False)
                predictions = self._forward_head(sub_outputs)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0]).numpy()
            class_channel = predictions[:, class_idx]

        # gradients of the class score w.r.t. the conv feature map
        grads = tape.gradient(class_channel, conv_outputs)   # (1, h, w, C)
        if grads is None:
            return np.zeros(conv_outputs.shape[1:3], dtype=np.float32)

        # global average pooling over spatial dims → (C,)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # weight each feature map by its pooled gradient → (h, w)
        conv_outputs = conv_outputs[0]                        # (h, w, C)
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)                         # (h, w)

        # ReLU + normalise to [0, 1]
        heatmap = tf.maximum(heatmap, 0)
        heatmap_max = tf.reduce_max(heatmap)
        heatmap = heatmap / (heatmap_max + eps)

        return heatmap.numpy()

    def explain(
        self,
        original_img: np.ndarray,
        class_idx: Optional[int] = None,
        alpha: float = HEATMAP_ALPHA,
        colormap: int = COLORMAP,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a Grad-CAM overlay on top of the original image.

        Parameters
        ----------
        original_img : uint8 RGB array of shape (H, W, 3) — the raw leaf photo
        class_idx    : if None, uses the top-predicted class
        alpha        : blending weight for the heatmap
        colormap     : OpenCV colormap constant (default: COLORMAP_JET)

        Returns
        -------
        overlay  : uint8 RGB array (H, W, 3) — original + heatmap blended
        heatmap  : uint8 RGB heatmap before blending (H, W, 3)
        """
        H, W = original_img.shape[:2]

        # ── preprocess for model ─────────────────────────────────────────── #
        # Input is expected to be RGB (as documented); resize preserves channel order
        resized_rgb = cv2.resize(original_img, IMG_SIZE)
        # Scale to [-1, 1] as expected by the wrapped model
        preprocessed = (resized_rgb.astype(np.float32) / 127.5) - 1.0
        img_batch = np.expand_dims(preprocessed, 0)

        # ── compute heatmap ──────────────────────────────────────────────── #
        raw_heatmap = self._compute_heatmap(img_batch, class_idx=class_idx)

        # ── upsample to original resolution ─────────────────────────────── #
        heatmap_upsampled = cv2.resize(raw_heatmap, (W, H))

        # ── apply colour map ──────────────────────────────────────────────── #
        heatmap_uint8 = np.uint8(255 * heatmap_upsampled)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)  # BGR
        heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # ── blend with original ───────────────────────────────────────────── #
        original_float = original_img.astype(np.float32)
        heatmap_float  = heatmap_colored_rgb.astype(np.float32)
        overlay = (1 - alpha) * original_float + alpha * heatmap_float
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        return overlay, heatmap_colored_rgb

    def predict_with_explain(
        self,
        original_img: np.ndarray,
    ) -> dict:
        """
        Full pipeline: predict class + confidence + generate overlay.

        Parameters
        ----------
        original_img : uint8 RGB array (H, W, 3)

        Returns
        -------
        dict with keys:
          class_idx, class_name (if class_names set), confidence,
          overlay (H, W, 3), heatmap (H, W, 3)
        """
        # preprocess
        resized = cv2.resize(original_img, IMG_SIZE)
        # Scale to [-1, 1] as expected by the wrapped model
        prep = (resized.astype(np.float32) / 127.5) - 1.0
        batch = np.expand_dims(prep, 0)

        # predict
        preds = self.model.predict(batch, verbose=0)[0]
        class_idx  = int(np.argmax(preds))
        confidence = float(preds[class_idx])

        # explain
        overlay, heatmap = self.explain(original_img, class_idx=class_idx)

        return {
            "class_idx":  class_idx,
            "confidence": confidence,
            "overlay":    overlay,
            "heatmap":    heatmap,
            "top5": sorted(
                enumerate(preds), key=lambda x: -x[1]
            )[:5],
        }


# ─────────────────────────── batch helper ─────────────────────────────────── #

def generate_gradcam_grid(
    gcam: GradCAM,
    images: list,
    labels: list,
    class_names: list,
    out_dir: str,
    max_images: int = 16,
) -> None:
    """
    Save individual Grad-CAM images for a list of (image, label) pairs.
    Useful for qualitative evaluation.
    """
    os.makedirs(out_dir, exist_ok=True)
    for i, (img, lbl) in enumerate(zip(images[:max_images], labels[:max_images])):
        result = gcam.predict_with_explain(img)
        overlay = result["overlay"]
        pred_name = class_names[result["class_idx"]] if class_names else str(result["class_idx"])
        true_name = class_names[lbl] if class_names else str(lbl)
        conf = result["confidence"]

        # annotate
        annotated = overlay.copy()
        cv2.putText(
            annotated,
            f"Pred: {pred_name[:30]} ({conf*100:.1f}%)",
            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"True: {true_name[:30]}",
            (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (200, 255, 200) if pred_name == true_name else (255, 100, 100),
            2, cv2.LINE_AA,
        )
        out_path = os.path.join(out_dir, f"gradcam_{i:04d}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    print(f"[GradCAM] Saved {min(max_images, len(images))} images → {out_dir}")


# ─────────────────────────── CLI ──────────────────────────────────────────── #

def parse_args():
    p = argparse.ArgumentParser(description="Generate Grad-CAM for a single image.")
    p.add_argument("--model_dir", default="streamlit_app/saved_model/crop_disease_model.keras",
                   help="Path to the saved Keras model file.")
    p.add_argument("--image",     required=True,
                   help="Path to input leaf image.")
    p.add_argument("--out_path",  default="gradcam_output.jpg",
                   help="Where to save the overlay image.")
    p.add_argument("--layer",     default=None,
                   help="Target layer name (auto-detected if omitted).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # load model
    print(f"[INFO] Loading model from {args.model_dir} …")
    model = keras.models.load_model(args.model_dir)

    # load class names (optional)
    parent_dir = os.path.dirname(args.model_dir)
    labels_path = os.path.join(parent_dir, "class_names.txt")
    class_names = None
    if os.path.exists(labels_path):
        class_names = open(labels_path).read().strip().splitlines()
        print(f"[INFO] {len(class_names)} class names loaded.")

    # load image
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load image: {args.image}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # run GradCAM
    gcam   = GradCAM(model, target_layer=args.layer)
    result = gcam.predict_with_explain(img_rgb)

    pred_name  = class_names[result["class_idx"]] if class_names else str(result["class_idx"])
    confidence = result["confidence"]
    print(f"\n[Result] Predicted: {pred_name}")
    print(f"         Confidence: {confidence*100:.2f}%")
    print(f"         Top-5:")
    for idx, prob in result["top5"]:
        name = class_names[idx] if class_names else str(idx)
        print(f"           {name:40s}  {prob*100:5.2f}%")

    # save overlay
    os.makedirs(Path(args.out_path).parent, exist_ok=True)
    overlay_bgr = cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.out_path, overlay_bgr)
    print(f"\n[INFO] Grad-CAM overlay saved → {args.out_path}")
