import tensorflow as tf
import numpy as np

# Load model
print("Loading model...")
model = tf.keras.models.load_model('signbridge_model.keras')

print("\n=== Model Summary ===")
model.summary()

print("\n=== Model Output Shape ===")
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")

print("\n=== Model Configuration ===")
config = model.get_config()
print(f"Number of layers: {len(config['layers'])}")

# Try to find training info
print("\n=== Model Metadata ===")
if hasattr(model, 'history'):
    print("Training history available")
else:
    print("No training history embedded in model")

# Check if there are logs
import os
if os.path.exists('logs'):
    print("\n✅ Training logs found in 'logs/' directory")
    print("Review training with: tensorboard --logdir=logs")
else:
    print("\n⚠️ No training logs found - model might be old")
