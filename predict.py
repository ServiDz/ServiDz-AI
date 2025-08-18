import os
import tensorflow as tf
import numpy as np
from PIL import Image

# Absolute path to model (update with your actual location)
MODEL_PATH = r"D:\ServiDZ_AI\servidz_model.h5"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

SERVICE_CATEGORIES = ["gardener", "painter", "plumber"]

def predict_service(image_path):
    try:
        img = Image.open(image_path).resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        service_index = np.argmax(prediction)
        return SERVICE_CATEGORIES[service_index]
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print("Predicted service:", predict_service(sys.argv[1]))
    else:
        print("Please provide an image path as argument")
