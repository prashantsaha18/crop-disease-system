import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def main():
    model_path = "streamlit_app/saved_model/crop_disease_model.keras"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    original_model = keras.models.load_model(model_path)
    print("Original model input shape:", original_model.input_shape)
    
    # We wrap it in a functional model that prepends a Rescaling layer:
    # Inputs in [-1, 1] -> Rescaling(scale=127.5, offset=127.5) -> Outputs in [0, 255] -> Original Model
    inputs = keras.Input(shape=(224, 224, 3), name="input_image_neg1_to_pos1")
    rescaled = layers.Rescaling(scale=127.5, offset=127.5, name="rescale_to_0_255")(inputs)
    outputs = original_model(rescaled)
    
    wrapped_model = keras.Model(inputs, outputs, name="WrappedCropDiseaseNet")
    print("Wrapped model input shape:", wrapped_model.input_shape)
    
    # Save the wrapped model
    wrapped_model.save(model_path)
    print("Wrapped model saved successfully!")

if __name__ == "__main__":
    main()
