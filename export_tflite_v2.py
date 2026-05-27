"""
Optional speed upgrade: convert the trained SignBridge v2 Keras model to TensorFlow Lite.
Run after training:
    python export_tflite_v2.py
"""

import tensorflow as tf
from sign_config import MODEL_PATH

OUTPUT_PATH = 'signbridge_model_v2.tflite'

model = tf.keras.models.load_model(MODEL_PATH)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(OUTPUT_PATH, 'wb') as f:
    f.write(tflite_model)

print(f'✅ Saved {OUTPUT_PATH}')
