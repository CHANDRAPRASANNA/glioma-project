import tensorflow as tf
import os
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'cnn_model_v2.h5')

print(f"Loading model from {MODEL_PATH}")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
    
    # Create dummy input
    dummy_input = np.random.rand(1, 128, 128, 3).astype(np.float32)
    
    print("Predicting...")
    pred = model.predict(dummy_input)
    print(f"Prediction result: {pred}")
    
except Exception as e:
    print(f"Error: {e}")
