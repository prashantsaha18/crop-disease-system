import requests
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

model_path = "streamlit_app/saved_model/crop_disease_model.keras"
model = keras.models.load_model(model_path)

# Direct raw image links from a public apple leaf disease dataset repository on GitHub
urls = {
    "Apple Scab Leaf": "https://raw.githubusercontent.com/abhi2425/disease-detection-for-apple-leaves/master/images/Apple___Apple_scab/f34d9619-f182-4b2f-b1a3-1658cc298f9b___FREC_Scab%203440.JPG",
    "Apple Healthy Leaf": "https://raw.githubusercontent.com/abhi2425/disease-detection-for-apple-leaves/master/images/Apple___healthy/0055dd26-23a7-4415-ac61-e0b44ebfaf80___RS_HL%205672.JPG"
}

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

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224), Image.LANCZOS)
    rgb_arr = np.array(img, dtype=np.float32)
    # Scale inputs to [-1, 1] as expected by the wrapped model
    prep = (rgb_arr / 127.5) - 1.0
    return np.expand_dims(prep, 0)

headers = {
    'User-Agent': 'Mozilla/5.0'
}

for name, url in urls.items():
    print(f"\nTesting {name} ...")
    temp_path = f"temp_{name.replace(' ', '_')}.jpg"
    
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    with open(temp_path, 'wb') as out_file:
        out_file.write(r.content)
        
    batch = preprocess(temp_path)
    preds = model.predict(batch)
    top_idx = np.argmax(preds[0])
    print(f"Top predicted index: {top_idx} ({ALL_LABELS[top_idx]})")
    print(f"Confidence: {preds[0][top_idx] * 100:.2f}%")
    
    # Print top 5
    indexed = sorted(enumerate(preds[0]), key=lambda x: -x[1])
    print("Top 5 predictions:")
    for idx, prob in indexed[:5]:
         print(f"  - {ALL_LABELS[idx]}: {prob * 100:.2f}%")
