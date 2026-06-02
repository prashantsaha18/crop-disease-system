# CropGuard — Streamlit Deployment Guide

## Deploy to Streamlit Cloud in 3 steps

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "CropGuard crop disease detection system"
git remote add origin https://github.com/YOUR_USERNAME/cropguard.git
git push -u origin main
```

### 2. Connect to Streamlit Cloud
1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Click **New app**
3. Choose your GitHub repo
4. Set **Root directory** → `streamlit_app`
5. Set **Main file path** → `app.py`
6. Click **Deploy**

Streamlit Cloud will automatically:
- Install Python packages from `streamlit_app/requirements.txt`
- Install system packages from `streamlit_app/packages.txt` (needed for OpenCV)

### 3. (Optional) Add trained model
After training with `python python/train.py`, the model will be at:
```
python/saved_model/crop_disease_model/
```
Push this directory to GitHub (use Git LFS for large files) and the
Streamlit app will detect and load it automatically. Without it, the
app runs in demo mode with ImageNet weights (predictions not meaningful).

---

## Local development

```bash
# Install system deps (Ubuntu/WSL)
# sudo apt-get install libgl1-mesa-glx libglib2.0-0

# Install Python deps
pip install -r streamlit_app/requirements.txt

# Run
streamlit run streamlit_app/app.py
```

## Repository structure for Streamlit Cloud

When connecting to Streamlit Cloud, set:
- **Root directory**: `streamlit_app`
- **Main file**: `app.py`

The `streamlit_app/` folder contains everything Streamlit Cloud needs:

```
streamlit_app/
├── app.py                  # ← Main Streamlit app (entry point)
├── requirements.txt        # Python dependencies
├── packages.txt            # System-level apt packages (for OpenCV)
└── .streamlit/
    └── config.toml         # Theme, server settings
```

## Full project structure

```
crop_disease_system/
├── python/
│   ├── train.py                    # EfficientNetV2-S training
│   ├── gradcam.py                  # GradCAM explainability
│   └── convert_to_tflite.py        # TFLite conversion + benchmark
├── streamlit_app/
│   ├── app.py                      # ← Streamlit web demo (deployable)
│   ├── requirements.txt            # Python deps (flexible version ranges)
│   ├── packages.txt                # System deps (libgl1 for OpenCV)
│   └── .streamlit/config.toml     # Theme + server config
└── flutter_app/
    ├── pubspec.yaml
    ├── lib/
    │   ├── main.dart
    │   ├── screens/
    │   │   ├── home_screen.dart
    │   │   ├── result_screen.dart
    │   │   └── disease_info_screen.dart
    │   └── utils/
    │       └── classifier.dart
    ├── assets/
    │   └── models/
    │       ├── crop_model.tflite   # from convert_to_tflite.py
    │       └── labels.txt
    ├── android/
    │   └── app/
    │       ├── build.gradle
    │       └── src/main/AndroidManifest.xml
    └── ios/
        └── Runner/Info.plist
```
