import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model('servidz_model.h5')

# Define your service categories (MUST match the order used during training)
SERVICE_CATEGORIES = ["gardener", "painter", "plumber" ]  # Update with your actual classes

def predict_service(image_path):
    try:
        # Load and preprocess image
        img = Image.open(image_path).resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(img_array)
        service_index = np.argmax(prediction)
        return SERVICE_CATEGORIES[service_index]
    except Exception as e:
        return f"Error: {str(e)}"

# Test
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print("Predicted service:", predict_service(sys.argv[1]))
    else:
        print("Please provide an image path as argument")