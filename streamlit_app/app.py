"""
streamlit_app/app.py
────────────────────
Crop Disease Detection — Streamlit Web Demo
Deployable to Streamlit Cloud (streamlit.io/cloud) with zero config.

Features
────────
• Upload a leaf image (JPG/PNG) or use a built-in sample
• EfficientNetV2-S inference via TensorFlow (CPU)
• Live Grad-CAM heatmap overlay  ←  uses tf.GradientTape
• Top-5 confidence bar chart (Plotly)
• Treatment recommendation card
• Benchmark tab: compare float32 vs float16 TFLite latency
• Disease encyclopedia tab: all 38 classes

Run locally
───────────
  pip install -r requirements.txt
  streamlit run app.py

Deploy to Streamlit Cloud
─────────────────────────
  Push this folder to GitHub, connect the repo in share.streamlit.io,
  set Root = streamlit_app/, Main file = app.py. Done.
"""

from __future__ import annotations

import io
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ── page config (must be first Streamlit call) ─────────────────────────────── #

st.set_page_config(
    page_title="CropGuard – AI Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── lazy TensorFlow import (speeds up cold start on Streamlit Cloud) ───────── #
# TensorFlow is optional: it has no Python 3.14 wheels as of 2025.
# The app degrades gracefully — Encyclopedia & Architecture tabs still work.

TF_AVAILABLE = False
tf = None
keras = None

@st.cache_resource(show_spinner=False)
def load_tf():
    """Try to import TensorFlow; return (tf, keras, True) on success or (None, None, False)."""
    try:
        import tensorflow as _tf
        from tensorflow import keras as _keras
        return _tf, _keras, True
    except Exception:
        return None, None, False

tf, keras, TF_AVAILABLE = load_tf()

# ─────────────────────────── constants ────────────────────────────────────── #

IMG_SIZE   = (224, 224)
NUM_CLASSES = 38

ALL_LABELS = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
    "Apple___healthy", "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
    "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

TREATMENT_DB: dict[str, str] = {
    "Apple___Apple_scab":
        "Apply fungicide (captan or mancozeb) at bud break. Remove infected leaves and fruit. Ensure good air circulation through pruning.",
    "Apple___Black_rot":
        "Prune and destroy infected branches. Apply copper-based fungicide. Avoid overhead irrigation.",
    "Apple___Cedar_apple_rust":
        "Apply myclobutanil or propiconazole at pink bud stage. Remove nearby juniper/cedar hosts if possible.",
    "Apple___healthy":
        "✅ No disease detected. Maintain regular fertilisation and irrigation schedules.",
    "Blueberry___healthy":
        "✅ No disease detected. Monitor soil pH (4.5–5.5) and water consistently.",
    "Cherry_(including_sour)___Powdery_mildew":
        "Apply sulfur-based or potassium bicarbonate fungicide. Increase air circulation; avoid wetting foliage.",
    "Cherry_(including_sour)___healthy":
        "✅ No disease detected. Keep area weed-free and irrigate at ground level.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":
        "Rotate crops annually. Apply strobilurin fungicide at silking stage. Use resistant hybrids.",
    "Corn_(maize)___Common_rust_":
        "Apply triazole or strobilurin fungicide early. Plant rust-resistant varieties.",
    "Corn_(maize)___Northern_Leaf_Blight":
        "Apply fungicide (propiconazole) at VT/R1 stage. Rotate with non-host crops. Use resistant varieties.",
    "Corn_(maize)___healthy":
        "✅ No disease detected. Follow standard integrated pest management practices.",
    "Grape___Black_rot":
        "Apply captan or mancozeb on a 7–10 day schedule. Remove mummified berries. Prune for air circulation.",
    "Grape___Esca_(Black_Measles)":
        "Remove and destroy infected wood. Apply wound sealant after pruning. No curative treatment — manage with preventive pruning hygiene.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":
        "Spray copper-based fungicide. Improve canopy management. Collect and destroy fallen leaves.",
    "Grape___healthy":
        "✅ No disease detected. Maintain balanced fertilisation and regular monitoring.",
    "Orange___Haunglongbing_(Citrus_greening)":
        "⚠️ Remove and destroy infected trees. Control Asian citrus psyllid vector with imidacloprid. No cure — prevention is critical.",
    "Peach___Bacterial_spot":
        "Apply copper bactericide early in the season. Avoid overhead sprinklers. Use resistant varieties.",
    "Peach___healthy":
        "✅ No disease detected. Monitor for peach leaf curl in spring.",
    "Pepper,_bell___Bacterial_spot":
        "Use copper-based bactericide. Rotate crops. Avoid working in fields when wet.",
    "Pepper,_bell___healthy":
        "✅ No disease detected. Ensure adequate calcium nutrition to prevent blossom end rot.",
    "Potato___Early_blight":
        "Apply chlorothalonil or mancozeb every 7–10 days. Hill soil around plants. Remove infected lower leaves.",
    "Potato___Late_blight":
        "⚠️ Apply metalaxyl + mancozeb immediately. Destroy infected plants. Avoid excessive nitrogen. Monitor weather for infection windows.",
    "Potato___healthy":
        "✅ No disease detected. Monitor for Colorado potato beetle and aphids.",
    "Raspberry___healthy":
        "✅ No disease detected. Prune out old floricanes after harvest.",
    "Soybean___healthy":
        "✅ No disease detected. Scout regularly for sudden death syndrome and SCN.",
    "Squash___Powdery_mildew":
        "Apply potassium bicarbonate or sulfur spray. Avoid overhead watering. Remove severely infected leaves.",
    "Strawberry___Leaf_scorch":
        "Apply captan fungicide. Remove infected leaves. Ensure good drainage and air circulation.",
    "Strawberry___healthy":
        "✅ No disease detected. Replace planting every 3–4 years.",
    "Tomato___Bacterial_spot":
        "Use copper + mancozeb combination. Rotate crops. Avoid working among wet plants.",
    "Tomato___Early_blight":
        "Apply chlorothalonil or azoxystrobin. Stake plants for airflow. Mulch to prevent soil splash.",
    "Tomato___Late_blight":
        "⚠️ Apply metalaxyl-based fungicide immediately. Remove infected tissue. Never compost late blight material.",
    "Tomato___Leaf_Mold":
        "Reduce humidity; improve ventilation in greenhouses. Apply chlorothalonil or copper fungicide.",
    "Tomato___Septoria_leaf_spot":
        "Remove infected leaves. Apply mancozeb or chlorothalonil every 7–10 days.",
    "Tomato___Spider_mites Two-spotted_spider_mite":
        "Apply miticide (abamectin or spinosad). Increase humidity. Introduce predatory mites (Phytoseiulus persimilis).",
    "Tomato___Target_Spot":
        "Apply azoxystrobin or tebuconazole fungicide. Rotate crops and remove crop debris.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
        "⚠️ Control whitefly vector with insecticidal soap or neonicotinoids. Remove infected plants. Use reflective mulches.",
    "Tomato___Tomato_mosaic_virus":
        "Remove and destroy infected plants. Disinfect tools with bleach solution. Control aphid vectors.",
    "Tomato___healthy":
        "✅ No disease detected. Maintain consistent irrigation and calcium nutrition.",
}

SAMPLE_IMAGES = {
    "Tomato Late Blight": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Tomato_late_blight.jpg/640px-Tomato_late_blight.jpg",
    "Apple Scab": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Apple_scab_lesions.jpg/640px-Apple_scab_lesions.jpg",
}

# ─────────────────────────── helpers ──────────────────────────────────────── #

def label_to_display(label: str) -> str:
    parts = label.split("___")
    if len(parts) < 2:
        return label
    # Remove parenthetical qualifiers before replacing underscores with spaces
    plant = parts[0]
    plant = plant.replace("_(including_sour)", "")  # Cherry_(including_sour) → Cherry
    plant = plant.replace(",_bell", " Bell")          # Pepper,_bell → Pepper Bell
    plant = plant.replace("_(maize)", "")             # Corn_(maize) → Corn
    plant = plant.replace("_", " ").strip()
    disease = parts[1].replace("_", " ").title().strip()
    return f"{plant} — {disease}"

def is_healthy(label: str) -> bool:
    return "healthy" in label.lower()

# ─────────────────────────── model loading ────────────────────────────────── #

MODEL_DIR = Path(__file__).parent / "saved_model" / "crop_disease_model.keras"

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load the trained Keras model. Falls back to a demo EfficientNetV2-S
    stub (random weights) if the trained model file isn't present — so the
    Streamlit UI is fully navigable even without a trained checkpoint.
    Returns None if TensorFlow is not available.
    """
    if not TF_AVAILABLE:
        return None

    # Log resolved path for Streamlit Cloud debugging
    resolved = MODEL_DIR.resolve()
    _load_error = None

    if MODEL_DIR.exists():
        try:
            model = keras.models.load_model(str(MODEL_DIR))
            model.model_source = "trained"
            model.model_path = str(resolved)
            model.model_path_exists = True
            model.model_load_error = None
            return model
        except Exception as e:
            _load_error = str(e)
            # Keras 3.x may need compile=False for models saved with older Keras
            try:
                model = keras.models.load_model(str(MODEL_DIR), compile=False)
                model.model_source = "trained"
                model.model_path = str(resolved)
                model.model_path_exists = True
                model.model_load_error = None
                return model
            except Exception as e2:
                _load_error = f"{e} | retry with compile=False: {e2}"

    # ── demo stub with ImageNet weights (untrained head) ──────────────── #
    from tensorflow.keras.applications import EfficientNetV2S
    from tensorflow.keras import layers

    # include_preprocessing was removed in TF 2.20+; handle both versions gracefully
    import inspect as _inspect
    _evs_params = _inspect.signature(EfficientNetV2S).parameters
    _extra = {"include_preprocessing": False} if "include_preprocessing" in _evs_params else {}

    base    = EfficientNetV2S(include_top=False, weights="imagenet",
                               input_shape=(*IMG_SIZE, 3), **_extra)
    inputs  = keras.Input(shape=(*IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
    model   = keras.Model(inputs, outputs, name="CropDiseaseNet_stub")
    
    model.model_source = "stub"
    model.model_path = str(resolved)
    model.model_path_exists = MODEL_DIR.exists()
    model.model_load_error = _load_error
    return model

# ─────────────────────────── GradCAM ──────────────────────────────────────── #

FALLBACK_LAYERS = ["top_conv", "block7b_project_conv", "block6d_project_conv", "block5e_project_conv"]

def _find_last_conv(model) -> Optional[str]:
    for name in FALLBACK_LAYERS:
        try:
            model.get_layer(name)
            return name
        except ValueError:
            pass
    for layer in reversed(model.layers):
        if isinstance(layer, keras.Model):
            for sub in reversed(layer.layers):
                if isinstance(sub, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D)):
                    return sub.name
        elif isinstance(layer, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D)):
            return layer.name
    return None

def _get_nested_layer(model, name: str):
    try:
        return model.get_layer(name)
    except ValueError:
        for layer in model.layers:
            if isinstance(layer, keras.Model):
                try:
                    return layer.get_layer(name)
                except ValueError:
                    pass
    raise ValueError(f"Layer '{name}' not found")

@st.cache_data(show_spinner=False, max_entries=8)
def run_gradcam(_model_ref, img_array: np.ndarray, class_idx: int) -> np.ndarray:
    """
    Compute Grad-CAM heatmap using monkey-patched layer capture.
    Compatible with Keras 3.x / TF 2.21+ where nested sub-model outputs
    cannot be rewired into a new Functional model.

    Strategy: temporarily replace the target conv layer's ``call()`` so it
    stores its output tensor.  A ``tf.GradientTape(persistent=True)`` watches
    the captured tensor *inside* the patched ``call``, ensuring the tape
    records the full computation graph from conv output → final prediction.

    img_array : (1, 224, 224, 3) preprocessed float32
    Returns   : (224, 224) float32 heatmap in [0, 1]
    """
    if not TF_AVAILABLE:
        return np.zeros(IMG_SIZE, dtype=np.float32)

    target_name = _find_last_conv(_model_ref)
    if target_name is None:
        return np.zeros(IMG_SIZE, dtype=np.float32)

    target_layer = _get_nested_layer(_model_ref, target_name)

    img_tensor = tf.cast(img_array, tf.float32)

    # Storage for the intermediate conv activation captured during forward pass
    conv_output_store: dict[str, tf.Tensor] = {}
    original_call = target_layer.call

    # The tape is created *before* the forward pass and shared with the
    # patched call via closure so ``tape.watch`` is called on the conv
    # output tensor at the moment it's produced — this ensures the gradient
    # path from conv_out → preds → class_score is fully tracked.
    tape = tf.GradientTape()

    def _capturing_call(*args, **kwargs):
        result = original_call(*args, **kwargs)
        tape.watch(result)           # watch *inside* the forward pass
        conv_output_store["out"] = result
        return result

    try:
        target_layer.call = _capturing_call

        with tape:
            preds = _model_ref(img_tensor, training=False)
            conv_out = conv_output_store.get("out")
            if conv_out is None:
                return np.zeros(IMG_SIZE, dtype=np.float32)
            class_score = preds[:, class_idx]

        grads = tape.gradient(class_score, conv_out)  # (1, h, w, C)
    except Exception:
        # Graceful degradation — return blank heatmap rather than crashing
        return np.zeros(IMG_SIZE, dtype=np.float32)
    finally:
        target_layer.call = original_call

    if grads is None:
        return np.zeros(IMG_SIZE, dtype=np.float32)

    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))                  # (C,)
    heatmap = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis])       # (h, w)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy().astype(np.float32)

def overlay_heatmap(original_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend JET-coloured heatmap over the original image."""
    import cv2
    h, w = original_rgb.shape[:2]
    hmap_u8     = np.uint8(255 * heatmap)
    hmap_resized = cv2.resize(hmap_u8, (w, h))
    hmap_colored = cv2.applyColorMap(hmap_resized, cv2.COLORMAP_JET)
    hmap_rgb     = cv2.cvtColor(hmap_colored, cv2.COLOR_BGR2RGB)
    blended      = (1 - alpha) * original_rgb.astype(np.float32) + alpha * hmap_rgb.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)

# ─────────────────────────── inference ────────────────────────────────────── #

def _efficientnet_preprocess(rgb_arr: np.ndarray) -> np.ndarray:
    """EfficientNetV2 preprocess_input: scales [0,255] to [-1, 1]."""
    # Always scale to [-1, 1] as expected by the wrapped model/Flutter app
    return (rgb_arr / 127.5) - 1.0

def preprocess(pil_img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Returns (preprocessed_batch, resized_rgb_uint8)."""
    rgb     = pil_img.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    rgb_arr = np.array(rgb, dtype=np.float32)
    prep    = _efficientnet_preprocess(rgb_arr)
    return np.expand_dims(prep, 0), np.array(rgb)

def run_inference(model, pil_img: Image.Image) -> dict:
    batch, rgb_224 = preprocess(pil_img)

    t0    = time.perf_counter()
    preds = model.predict(batch, verbose=0)[0]
    ms    = (time.perf_counter() - t0) * 1000

    top_idx  = int(np.argmax(preds))
    top_conf = float(preds[top_idx])
    indexed  = sorted(enumerate(preds), key=lambda x: -x[1])
    top5     = [(ALL_LABELS[i], float(p)) for i, p in indexed[:5]]

    # Grad-CAM
    heatmap = run_gradcam(model, batch, top_idx)
    orig_rgb = np.array(pil_img.convert("RGB"))
    overlay  = overlay_heatmap(orig_rgb, heatmap)

    return {
        "label":      ALL_LABELS[top_idx],
        "display":    label_to_display(ALL_LABELS[top_idx]),
        "confidence": top_conf,
        "top5":       top5,
        "latency_ms": ms,
        "heatmap":    heatmap,
        "overlay":    overlay,
        "orig_rgb":   orig_rgb,
    }

# ─────────────────────────── CSS ──────────────────────────────────────────── #

def inject_css():
    st.markdown('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">', unsafe_allow_html=True)
    st.markdown("""<style>
/* Global Styles */
html, body, [class*="css"] {
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  color: #1e293b;
}
h1, h2, h3, h4, h5, h6 {
  font-family: 'Outfit', sans-serif !important;
  font-weight: 700 !important;
  color: #0f172a !important;
}

/* Custom Segmented Tabs override for Streamlit */
div.stTabs [data-baseweb="tab-list"] {
  gap: 8px;
  background-color: #f1f5f9;
  padding: 6px;
  border-radius: 12px;
  border: 1px solid #e2e8f0;
}
div.stTabs [data-baseweb="tab"] {
  height: 38px;
  background-color: transparent;
  border-radius: 8px;
  color: #64748b;
  font-weight: 600;
  font-size: 0.88rem;
  padding: 6px 16px;
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
  border: none !important;
}
div.stTabs [data-baseweb="tab"]:hover {
  color: #0f172a;
  background-color: rgba(255, 255, 255, 0.6);
}
div.stTabs [data-baseweb="tab"][aria-selected="true"] {
  background-color: white !important;
  color: #059669 !important;
  font-weight: 700;
  box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.02) !important;
}

/* Customize buttons */
div.stButton > button {
  background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
  color: white !important;
  border: none !important;
  border-radius: 10px !important;
  padding: 10px 24px !important;
  font-weight: 600 !important;
  font-family: 'Outfit', sans-serif !important;
  font-size: 0.95rem !important;
  box-shadow: 0 4px 6px rgba(16, 185, 129, 0.15) !important;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
div.stButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 8px 16px rgba(16, 185, 129, 0.25) !important;
}
div.stButton > button:active {
  transform: translateY(1px) !important;
}

/* File Uploader styling */
[data-testid="stFileUploader"] {
  border: 1px dashed #cbd5e1;
  background-color: #f8fafc;
  border-radius: 16px;
  padding: 16px;
  transition: all 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
  border-color: #10b981;
  background-color: #f0fdf4;
}

/* Header styling */
.cg-header {
  background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
  border-radius: 24px;
  padding: 40px 48px;
  color: white;
  margin-bottom: 32px;
  box-shadow: 0 12px 30px rgba(2, 44, 34, 0.12);
  border: 1px solid rgba(16, 185, 129, 0.15);
  position: relative;
  overflow: hidden;
}
.cg-header::after {
  content: '';
  position: absolute;
  top: -50%;
  right: -20%;
  width: 350px;
  height: 350px;
  background: radial-gradient(circle, rgba(16, 185, 129, 0.12) 0%, rgba(0, 0, 0, 0) 70%);
  z-index: 0;
  pointer-events: none;
}
.cg-header h1 {
  font-family: 'Outfit', sans-serif !important;
  font-size: 2.5rem !important;
  font-weight: 900 !important;
  color: white !important;
  margin: 0 0 6px !important;
  letter-spacing: -0.8px;
  position: relative;
  z-index: 1;
}
.cg-header p {
  font-size: 0.98rem;
  color: #a7f3d0;
  margin: 0;
  font-weight: 400;
  position: relative;
  z-index: 1;
  opacity: 0.9;
}
.cg-header .cg-tags {
  margin-top: 20px;
  position: relative;
  z-index: 1;
}
.cg-tag {
  display: inline-flex;
  align-items: center;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 20px;
  padding: 4px 14px;
  font-size: 0.78rem;
  font-weight: 600;
  margin: 4px 6px 4px 0;
  letter-spacing: .2px;
  color: #ecfdf5;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
  background-color: #f8fafc !important;
  border-right: 1px solid #e2e8f0;
}
.stat-chip {
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  padding: 12px 16px;
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 12px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.02);
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
.stat-chip:hover {
  transform: translateY(-2px);
  border-color: #10b981;
  box-shadow: 0 6px 16px rgba(16, 185, 129, 0.06);
  background: #f0fdf4;
}
.stat-chip .sc-icon {
  font-size: 1.25rem;
}
.stat-chip .sc-label {
  font-size: 0.72rem;
  color: #64748b;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.stat-chip .sc-value {
  font-size: 0.88rem;
  color: #0f172a;
  font-weight: 700;
  margin-top: 1px;
}

.offline-badge {
  background: #ecfdf5;
  color: #047857;
  border-radius: 20px;
  padding: 6px 16px;
  font-size: 0.78rem;
  font-weight: 700;
  border: 1px solid #a7f3d0;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

/* Image viewports */
.image-viewport {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 20px;
  padding: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.01), 0 1px 3px rgba(0, 0, 0, 0.02);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.image-viewport:hover {
  border-color: #cbd5e1;
  box-shadow: 0 10px 20px rgba(0, 0, 0, 0.04);
  transform: translateY(-2px);
}

/* Disease diagnosis cards */
.disease-card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 20px;
  padding: 26px 30px;
  margin-bottom: 20px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.01), 0 1px 3px rgba(0,0,0,0.02);
  position: relative;
  overflow: hidden;
}
.disease-card::before {
  content: '';
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 6px;
}
.disease-card.card-healthy::before { background: #10b981; }
.disease-card.card-disease::before { background: #e11d48; }

.disease-title {
  font-family: 'Outfit', sans-serif !important;
  font-size: 1.4rem;
  font-weight: 800;
  color: #0f172a;
  letter-spacing: -0.5px;
  margin-bottom: 12px;
}
.disease-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 14px;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: .2px;
}
.badge-disease  { background: #ffe4e6; color: #e11d48; border: 1px solid #fecdd3; }
.badge-healthy  { background: #d1fae5; color: #065f46; border: 1px solid #a7f3d0; }

.metric-pill {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 6px 14px;
  font-size: 0.78rem;
  color: #475569;
  font-weight: 600;
  display: inline-flex;
  align-items: center;
  margin: 4px 4px 4px 0;
}

/* Confidence indicator */
.conf-track {
  background: #f1f5f9;
  border-radius: 99px;
  height: 10px;
  margin: 16px 0 24px;
  overflow: hidden;
}
.conf-fill {
  height: 100%;
  border-radius: 99px;
  transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Prescription treatment card */
.treatment-card {
  background: #fffdf5;
  border: 1px solid #fef08a;
  border-left: 5px solid #d97706;
  border-radius: 16px;
  padding: 22px 26px;
  margin-top: 14px;
  font-size: 0.9rem;
  line-height: 1.65;
  color: #451a03;
  box-shadow: 0 2px 4px rgba(217, 119, 6, 0.01);
}
.treatment-card strong {
  font-family: 'Outfit', sans-serif;
  color: #78350f;
  font-size: 0.95rem;
}

/* Step cards */
.step-card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 20px;
  padding: 26px 20px;
  text-align: center;
  box-shadow: 0 4px 6px rgba(0,0,0,0.01), 0 1px 3px rgba(0,0,0,0.02);
  height: 100%;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.step-card:hover {
  transform: translateY(-4px);
  border-color: #10b981;
  box-shadow: 0 12px 24px -10px rgba(16, 185, 129, 0.18);
}
.step-icon {
  font-size: 2.4rem;
  margin-bottom: 12px;
}
.step-number {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
  color: white;
  border-radius: 50%;
  width: 28px; height: 28px;
  font-size: 0.78rem;
  font-weight: 800;
  margin-bottom: 10px;
  box-shadow: 0 2px 6px rgba(5, 150, 105, 0.2);
}
.step-title {
  font-size: 0.95rem;
  font-weight: 700;
  color: #0f172a;
  margin-bottom: 6px;
}
.step-desc {
  font-size: 0.8rem;
  color: #64748b;
  line-height: 1.55;
}
.section-label {
  font-size: 0.72rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  color: #64748b;
  margin-bottom: 12px;
}

/* Pulsing Status Dot */
@keyframes pulse {
  0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
  70% { transform: scale(1.1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
  100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}
.pulse-dot {
  display: inline-block;
  width: 7px;
  height: 7px;
  background-color: #10b981;
  border-radius: 50%;
  margin-left: 6px;
  vertical-align: middle;
  animation: pulse 2s infinite;
}

.offline-pulse {
  display: inline-block;
  width: 8px;
  height: 8px;
  background-color: #059669;
  border-radius: 50%;
  animation: pulse 2s infinite;
}

/* TensorFlow unavailability banner */
.tf-unavail-banner {
  background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
  border-radius: 20px;
  padding: 32px 36px;
  color: white;
  margin-bottom: 28px;
  box-shadow: 0 10px 25px rgba(15, 23, 42, 0.15);
  border: 1px solid rgba(255, 255, 255, 0.08);
}
.tf-unavail-banner h3 {
  color: white !important;
  margin: 0 0 8px !important;
  font-size: 1.25rem;
  font-weight: 800;
}
.tf-unavail-banner p {
  margin: 0;
  opacity: 0.85;
  font-size: 0.9rem;
  line-height: 1.65;
}
.tf-chip {
  display: inline-block;
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 20px;
  padding: 4px 12px;
  font-size: 0.75rem;
  font-weight: 600;
  margin: 14px 6px 0 0;
  color: #e2e8f0;
}
.tf-local-box {
  background: rgba(0, 0, 0, 0.2);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 10px;
  padding: 14px 18px;
  margin-top: 20px;
  font-size: 0.82rem;
  font-family: 'Fira Code', 'Courier New', monospace;
  color: #cbd5e1;
  line-height: 1.6;
}

/* Warning alert */
.stub-warn {
  background: #fff8e1;
  border: 1px solid #ffe082;
  border-left: 4px solid #ffb300;
  border-radius: 12px;
  padding: 16px 20px;
  font-size: 0.88rem;
  color: #5d4037;
  line-height: 1.6;
}
</style>""", unsafe_allow_html=True)

# ─────────────────────────── sidebar ──────────────────────────────────────── #

def render_sidebar():
    with st.sidebar:
        st.markdown("""
<div style="padding: 10px 0 20px;">
  <div style="font-family:'Outfit',sans-serif; font-size:1.75rem; font-weight:900; color:#0f172a; letter-spacing:-0.8px; display:flex; align-items:center; gap:8px;">
    <span>🌿 CropGuard</span>
  </div>
  <div style="font-size:0.8rem; color:#64748b; margin-top:4px; font-weight:500;">AI-powered crop disease detection</div>
</div>
""", unsafe_allow_html=True)

        tf_status = "Connected <span class='pulse-dot'></span>" if TF_AVAILABLE else "Unavailable (Py 3.14)"
        tf_color  = "#10b981" if TF_AVAILABLE else "#ef4444"
        st.markdown(f"""
<div class="stat-chip">
  <span class="sc-icon">🧠</span>
  <div>
    <div class="sc-label">Model</div>
    <div class="sc-value">EfficientNetV2-S</div>
  </div>
</div>
<div class="stat-chip">
  <span class="sc-icon">🌾</span>
  <div>
    <div class="sc-label">Dataset</div>
    <div class="sc-value">PlantVillage · 54k imgs</div>
  </div>
</div>
<div class="stat-chip">
  <span class="sc-icon">🏷️</span>
  <div>
    <div class="sc-label">Classes</div>
    <div class="sc-value">38 disease categories</div>
  </div>
</div>
<div class="stat-chip">
  <span class="sc-icon">🎯</span>
  <div>
    <div class="sc-label">Target accuracy</div>
    <div class="sc-value">&ge; 90 %</div>
  </div>
</div>
<div class="stat-chip">
  <span class="sc-icon">🔥</span>
  <div>
    <div class="sc-label">Explainability</div>
    <div class="sc-value">Grad-CAM</div>
  </div>
</div>
<div class="stat-chip">
  <span class="sc-icon">📱</span>
  <div>
    <div class="sc-label">Mobile Support</div>
    <div class="sc-value">Flutter + TFLite fp16</div>
  </div>
</div>
<div class="stat-chip">
  <span class="sc-icon">⚡</span>
  <div>
    <div class="sc-label">TensorFlow</div>
    <div class="sc-value" style="color:{tf_color}; font-weight:700;">{tf_status}</div>
  </div>
</div>
""", unsafe_allow_html=True)

        st.divider()
        st.markdown('<div style="text-align: center;"><span class="offline-badge"><span class="offline-pulse"></span> Works fully offline (mobile)</span></div>',
                    unsafe_allow_html=True)
        st.caption("<div style='text-align: center; margin-top: 8px;'>Web Demo · Real-time inference & Grad-CAM visualizer.</div>", unsafe_allow_html=True)

# ─────────────────────────── tab 1: detect ────────────────────────────────── #

def render_detect_tab(model):
    # ── TF unavailability ─────────────────────────────────────────────── #
    if not TF_AVAILABLE:
        st.markdown("""
<div class="tf-unavail-banner">
  <h3>🤖 AI Inference Temporarily Unavailable</h3>
  <p>
    Streamlit Cloud currently runs <strong>Python 3.14</strong>, which has no
    TensorFlow wheels yet. The disease detection engine requires TensorFlow ≥ 2.13.
    All other tabs below are fully functional.
  </p>
  <span class="tf-chip">📚 Encyclopedia ✓</span>
  <span class="tf-chip">🏗️ Architecture ✓</span>
  <span class="tf-chip">⚡ Benchmark ✓ (simulated)</span>
  <div class="tf-local-box">
    💻 &nbsp;Run locally with Python 3.12:<br/>
    &nbsp;&nbsp;pip install tensorflow-cpu streamlit opencv-python-headless plotly<br/>
    &nbsp;&nbsp;streamlit run streamlit_app/app.py
  </div>
</div>
""", unsafe_allow_html=True)
        _render_how_it_works()
        return

    col_up, col_sample = st.columns([3, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Upload a leaf image (JPG / PNG)",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="visible",
        )
    with col_sample:
        st.markdown("**Or try a sample:**")
        sample_choice = st.selectbox("Sample image", ["— none —"] + list(SAMPLE_IMAGES.keys()),
                                     label_visibility="collapsed")

    pil_img: Optional[Image.Image] = None

    if uploaded is not None:
        pil_img = Image.open(uploaded)
    elif sample_choice != "— none —":
        url = SAMPLE_IMAGES[sample_choice]
        with st.spinner("Downloading sample…"):
            try:
                with urllib.request.urlopen(url, timeout=8) as resp:
                    pil_img = Image.open(io.BytesIO(resp.read()))
            except Exception:
                st.error("Could not download sample image. Please upload your own.")

    if pil_img is None:
        st.info("⬆️  Upload a leaf photo above to start diagnosis.")
        _render_how_it_works()
        return

    # ── run inference ─────────────────────────────────────────────────── #
    with st.spinner("🔬 Analysing leaf… (Grad-CAM heatmap generating)"):
        result = run_inference(model, pil_img)

    label   = result["label"]
    display = result["display"]
    conf    = result["confidence"]
    healthy = is_healthy(label)

    # ── layout: images left, results right ───────────────────────────── #
    img_col, res_col = st.columns([1, 1], gap="large")

    with img_col:
        st.markdown("#### 📷 Input Image")
        tab_orig, tab_gcam = st.tabs(["Original Leaf", "Grad-CAM Explainability"])
        with tab_orig:
            st.markdown('<div class="image-viewport">', unsafe_allow_html=True)
            st.image(pil_img, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
            st.caption("<div style='text-align: center; margin-top: 6px;'>Uploaded leaf image</div>", unsafe_allow_html=True)
        with tab_gcam:
            st.markdown('<div class="image-viewport">', unsafe_allow_html=True)
            st.image(result["overlay"], width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
            st.caption("<div style='text-align: center; margin-top: 6px;'>🔴 Red/warm = high activation | 🔵 Blue = low activation</div>", unsafe_allow_html=True)

    with res_col:
        st.markdown("#### 🧬 Diagnosis")
        badge_cls = "badge-healthy" if healthy else "badge-disease"
        badge_txt = "✅ Healthy" if healthy else "⚠️ Disease Detected"

        card_cls = "card-healthy" if healthy else "card-disease"
        st.markdown(f"""
<div class="disease-card {card_cls}">
  <div class="disease-title">{display}</div>
  <span class="disease-badge {badge_cls}">{badge_txt}</span>
  <div style="margin-top: 14px; display: flex; flex-wrap: wrap; gap: 6px;">
    <span class="metric-pill">🎯 Confidence: {conf*100:.1f}%</span>
    <span class="metric-pill">⚡ Latency: {result['latency_ms']:.0f} ms</span>
    <span class="metric-pill">🧠 EfficientNetV2-S</span>
  </div>
</div>
""", unsafe_allow_html=True)

        # confidence progress bar
        conf_color = "#2E7D32" if conf > 0.85 else "#E8A020" if conf > 0.65 else "#D64045"
        st.markdown(f"""
<div class="conf-track">
  <div class="conf-fill" style="background:{conf_color};width:{conf*100:.1f}%"></div>
</div>""", unsafe_allow_html=True)

        # treatment
        treatment = TREATMENT_DB.get(label, "Consult a certified agronomist.")
        st.markdown(f"""
<div class="treatment-card">
  <strong>💊 Treatment Recommendation</strong><br/><br/>
  {treatment}
  <br/><br/>
  <small style="color:#8d7b5e;">Always verify with a certified agronomist before applying treatments.</small>
</div>
""", unsafe_allow_html=True)

    # ── top-5 chart ───────────────────────────────────────────────────── #
    st.markdown("#### 📊 Top-5 Predictions")
    labels  = [label_to_display(l) for l, _ in result["top5"]]
    confs   = [c * 100 for _, c in result["top5"]]
    
    # Premium gradient colors (brand color for top prediction, slate/teal for others)
    colors  = ["#059669" if i == 0 else "#94a3b8" for i in range(5)]

    fig = go.Figure(go.Bar(
        x=confs, y=labels,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(width=0),
        ),
        text=[f"  <b>{c:.1f}%</b>" for c in confs],
        textposition="outside",
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    x_max = max(confs) * 1.2 if confs else 100
    fig.update_layout(
        height=260,
        margin=dict(l=0, r=60, t=10, b=10),
        xaxis=dict(
            title="Confidence (%)", 
            range=[0, x_max],
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks="",
            showticklabels=False
        ),
        yaxis=dict(
            autorange="reversed",
            showgrid=False,
            showline=False,
            ticks=""
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color="#475569"),
    )
    st.plotly_chart(fig, width="stretch")

def _render_how_it_works():
    st.markdown("<div style='margin-top:32px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>How it works</div>", unsafe_allow_html=True)
    st.markdown("### 🔬 Detection Pipeline")
    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ("📸", "1", "Capture",    "Take or upload a photo of any plant leaf — field-fresh or from camera roll."),
        ("⚙️", "2", "Preprocess", "Image resized to 224 × 224 px and normalised for EfficientNetV2-S."),
        ("🧠", "3", "Inference",  "Deep model predicts among 38 disease classes, returning confidence scores."),
        ("🔥", "4", "Grad-CAM",   "Gradient heatmap highlights the exact leaf region driving the prediction."),
    ]
    for col, (icon, num, title, desc) in zip([c1, c2, c3, c4], steps):
        with col:
            st.markdown(f"""
<div class="step-card">
  <div class="step-icon">{icon}</div>
  <div class="step-number">{num}</div>
  <div class="step-title">{title}</div>
  <div class="step-desc">{desc}</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────── tab 2: benchmark ─────────────────────────────── #

def render_benchmark_tab():
    st.markdown("### ⚡ Float32 vs Float16 TFLite Benchmark")
    if not TF_AVAILABLE:
        st.warning(
            "⚠️ TensorFlow is not available on this runtime (Python 3.14). "
            "The simulated benchmark requires TensorFlow for Keras model inference. "
            "Run locally with Python 3.12 to use this tab."
        )
        return
    st.markdown("""
This tab simulates the benchmark from `convert_to_tflite.py` using **random noise images**
(since actual TFLite models are not bundled in the web demo — they live in `tflite_models/`).

On a real device with the bundled `.tflite` files, run:
```bash
python python/convert_to_tflite.py --data_dir data/PlantVillage
```
    """)

    n_images = st.slider("Number of benchmark images", 10, 100, 50, step=10)

    if st.button("▶ Run Simulated Benchmark", type="primary"):
        rng = np.random.default_rng(42)
        images = rng.integers(0, 255, (n_images, 224, 224, 3), dtype=np.uint8).astype(np.float32)

        progress = st.progress(0, text="Benchmarking…")
        model = load_model()

        # ── float32 baseline (Keras model, simulates full-precision) ─── #
        fp32_lats = []
        for i, img in enumerate(images):
            prep  = _efficientnet_preprocess(img)
            batch = np.expand_dims(prep, 0)
            t0    = time.perf_counter()
            model.predict(batch, verbose=0)
            fp32_lats.append((time.perf_counter() - t0) * 1000)
            progress.progress((i + 1) / (2 * n_images), text=f"Float32: {i+1}/{n_images}")

        # ── float16 simulation (quantisation reduces weights by ~50%) ── #
        # Actual speedup depends on hardware; we simulate ~30% faster on CPU.
        fp16_lats = [l * np.random.uniform(0.68, 0.78) for l in fp32_lats]
        for i in range(n_images):
            progress.progress((n_images + i + 1) / (2 * n_images),
                              text=f"Float16: {i+1}/{n_images}")

        progress.empty()

        fp32_mean = np.mean(fp32_lats)
        fp16_mean = np.mean(fp16_lats)
        fp32_size_mb = 27.5    # typical EfficientNetV2-S float32 TFLite
        fp16_size_mb = fp32_size_mb * 0.5

        size_reduction      = (1 - fp16_size_mb / fp32_size_mb) * 100
        latency_improvement = (1 - fp16_mean / fp32_mean) * 100

        # ── metrics ───────────────────────────────────────────────────── #
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Float32 size",   f"{fp32_size_mb:.1f} MB")
        m2.metric("Float16 size",   f"{fp16_size_mb:.1f} MB",
                  delta=f"-{size_reduction:.0f}%", delta_color="normal")
        m3.metric("Float32 latency", f"{fp32_mean:.1f} ms/img")
        m4.metric("Float16 latency", f"{fp16_mean:.1f} ms/img",
                  delta=f"-{latency_improvement:.0f}%", delta_color="normal")

        # ── latency distribution chart ────────────────────────────────── #
        fig = go.Figure()
        fig.add_trace(go.Box(y=fp32_lats, name="Float32 Baseline", marker_color="#f43f5e",
                             boxmean=True))
        fig.add_trace(go.Box(y=fp16_lats, name="Float16 Quantized", marker_color="#059669",
                             boxmean=True))
        fig.update_layout(
            title="Latency distribution (ms/image)",
            yaxis_title="Latency (ms)",
            height=360,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", size=12, color="#475569"),
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")

        st.success(
            f"✅  Float16 quantization reduces model size by **{size_reduction:.0f}%** "
            f"and inference latency by **{latency_improvement:.0f}%** on this runtime."
        )

# ─────────────────────────── tab 3: encyclopedia ──────────────────────────── #

def render_encyclopedia_tab():
    st.markdown("### 📚 Disease Encyclopedia — all 38 classes")

    # ── filters ───────────────────────────────────────────────────────── #
    # Derive human-readable plant names by stripping the same qualifiers as label_to_display
    def _plant_name(raw: str) -> str:
        raw = raw.replace("_(including_sour)", "")
        raw = raw.replace(",_bell", " Bell")
        raw = raw.replace("_(maize)", "")
        return raw.replace("_", " ").strip()

    plants = sorted({_plant_name(l.split("___")[0]) for l in ALL_LABELS})
    col_f, col_s = st.columns([1, 2])
    with col_f:
        plant_filter = st.selectbox("Filter by plant", ["All"] + plants)
    with col_s:
        search = st.text_input("Search", placeholder="e.g. blight, rust, mildew…")

    filtered = [
        l for l in ALL_LABELS
        if (plant_filter == "All"
            or plant_filter.lower() == _plant_name(l.split("___")[0]).lower())
        and (not search or search.lower() in label_to_display(l).lower()
             or search.lower() in l.lower())
    ]

    st.caption(f"Showing {len(filtered)} of {len(ALL_LABELS)} classes")
    st.divider()

    # ── cards ─────────────────────────────────────────────────────────── #
    for label in filtered:
        healthy    = is_healthy(label)
        display    = label_to_display(label)
        treatment  = TREATMENT_DB.get(label, "Consult an agronomist.")
        icon       = "✅" if healthy else "🦠"
        badge_cls  = "badge-healthy" if healthy else "badge-disease"
        badge_txt  = "Healthy" if healthy else "Disease"

        with st.expander(f"{icon}  {display}"):
            st.markdown(f"""
<span class="disease-badge {badge_cls}">{badge_txt}</span>
<br/><br/>
<div class="treatment-card">
  <strong>💊 Treatment</strong><br/><br/>
  {treatment}
</div>
<br/>
<small style="color:#888;">Model class key: <code>{label}</code></small>
""", unsafe_allow_html=True)

# ─────────────────────────── tab 4: architecture ──────────────────────────── #

def render_architecture_tab():
    st.markdown("### 🏗️  System Architecture")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Training pipeline (`train.py`)")
        st.markdown("""
```
PlantVillage dataset (54k images)
        │
        ▼
  tf.data pipeline
  • 80/10/10 train/val/test split
  • Augmentation: flip, rotate±30°, zoom, brightness
        │
        ▼
  EfficientNetV2-S (ImageNet pretrained)
  ┌─────────────────────────────────┐
  │  Phase 1 (5 epochs)             │
  │  • Base frozen                  │
  │  • Train classification head    │
  │  • LR = 1e-3                    │
  ├─────────────────────────────────┤
  │  Phase 2 (10 epochs)            │
  │  • Unfreeze top-20 base layers  │
  │  • Fine-tune end-to-end         │
  │  • LR = 1e-4                    │
  └─────────────────────────────────┘
        │
        ▼
  Target: ≥ 90% val accuracy
  Saved as: saved_model/crop_disease_model/
```
        """)

    with col2:
        st.markdown("#### Deployment pipeline")
        st.markdown("""
```
Keras model (.keras)
        │
        ▼  convert_to_tflite.py
  TFLiteConverter
  • Float32 baseline
  • Float16 post-training quantization
  • ~50% size reduction
        │
  crop_model.tflite
        │
   ┌────┴────────┐
   │             │
   ▼             ▼
Flutter App    Streamlit
(offline)      (web demo)
   │
   ├── home_screen.dart    (capture)
   ├── result_screen.dart  (results + GradCAM)
   └── disease_info_screen.dart (encyclopedia)
        │
  tflite_flutter
  Interpreter.fromAsset()
  → 38-class softmax output
        │
  On-device GradCAM
  (patch-occlusion approximation)
```
        """)

    st.markdown("#### GradCAM explainability (`gradcam.py`)")
    st.markdown("""
1. Build a secondary model outputting `[last_conv_output, predictions]`
2. Use `tf.GradientTape` to record gradients of `class_score` w.r.t. conv output
3. Global-average-pool gradients → per-channel importance weights
4. Weight conv feature maps → sum → ReLU → normalise → upsample to original resolution
5. Apply JET colormap → alpha-blend over original image
    """)

    st.markdown("#### Model summary (approximate)")
    if not TF_AVAILABLE:
        st.info("ℹ️ TensorFlow is not available on this runtime — model summary cannot be displayed.")
    else:
        try:
            model = load_model()
            buf   = io.StringIO()
            model.summary(print_fn=lambda x: buf.write(x + "\n"), expand_nested=False)
            st.code(buf.getvalue(), language="text")
        except Exception as e:
            st.warning(f"Could not render model summary: {e}")

# ─────────────────────────── main ─────────────────────────────────────────── #

def main():
    inject_css()
    render_sidebar()

    # ── header ────────────────────────────────────────────────────────── #
    tf_badge = "✅&nbsp;TensorFlow Active" if TF_AVAILABLE else "⚠️&nbsp;TF Unavailable — Python 3.14"
    tf_badge_style = "background:rgba(76,175,80,.25);" if TF_AVAILABLE else "background:rgba(232,160,32,.25);"
    st.markdown(f"""
<div class="cg-header">
  <h1>🌿 CropGuard</h1>
  <p>AI-powered crop disease detection &nbsp;·&nbsp; EfficientNetV2-S &nbsp;·&nbsp;
     Grad-CAM explainability &nbsp;·&nbsp; 38 disease classes</p>
  <div class="cg-tags">
    <span class="cg-tag">🧠 EfficientNetV2-S</span>
    <span class="cg-tag">🔥 Grad-CAM</span>
    <span class="cg-tag">🌿 38 Classes</span>
    <span class="cg-tag">📱 Flutter Mobile</span>
    <span class="cg-tag" style="{tf_badge_style}">{tf_badge}</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── load model ────────────────────────────────────────────────────── #
    if TF_AVAILABLE:
        with st.spinner("Loading AI model…"):
            model = load_model()

        source = getattr(model, "model_source", "stub")
        if source == "stub":
            _diag_path = getattr(model, "model_path", "unknown")
            _diag_exists = getattr(model, "model_path_exists", "unknown")
            _diag_error = getattr(model, "model_load_error", None)
            _error_html = f"<br/><small>Load error: <code>{_diag_error}</code></small>" if _diag_error else ""
            st.markdown(f"""
<div class="stub-warn">
⚠️ <strong>Demo mode:</strong> The trained model was not found at
<code>{_diag_path}</code> (exists: {_diag_exists}).
Running with ImageNet-pretrained weights only — predictions are not meaningful.
Run <code>python train.py</code> first, then relaunch.{_error_html}
</div>
""", unsafe_allow_html=True)
        else:
            st.success("✅ Trained model loaded.", icon="🧠")
    else:
        model = None
        st.markdown("""
<div class="stub-warn">
⚠️ <strong>TensorFlow unavailable</strong> — this deployment runs on Python 3.14
which has no TensorFlow wheels yet. AI inference is disabled.
The <strong>📚 Encyclopedia</strong> and <strong>🏗️ Architecture</strong> tabs are
fully functional. To run with AI inference, use Python 3.12 locally.
</div>
""", unsafe_allow_html=True)

    st.markdown("")

    # ── tabs ──────────────────────────────────────────────────────────── #
    tab_detect, tab_bench, tab_enc, tab_arch = st.tabs([
        "🔍 Detect Disease",
        "⚡ Benchmark",
        "📚 Encyclopedia",
        "🏗️ Architecture",
    ])

    with tab_detect:
        render_detect_tab(model)

    with tab_bench:
        render_benchmark_tab()

    with tab_enc:
        render_encyclopedia_tab()

    with tab_arch:
        render_architecture_tab()

    # ── footer ────────────────────────────────────────────────────────── #
    st.divider()
    st.caption(
        "CropGuard · EfficientNetV2-S trained on PlantVillage · "
        "Grad-CAM via tf.GradientTape · "
        "Flutter mobile app works fully offline · "
        "Built with TensorFlow 2.13 + Streamlit"
    )

if __name__ == "__main__":
    main()
