"""
STEP 2: Convert and Quantize the Model
========================================
This script takes the trained Keras model (.h5) and converts it
to TensorFlow Lite format (.tflite) with INT8 quantization.

What is quantization?
    The trained model uses 32-bit floats (very precise numbers).
    INT8 quantization converts them to 8-bit integers (whole numbers).
    Result: ~4x smaller file, faster inference, tiny accuracy loss.

Requirements:
    pip install tensorflow

Run:
    python 2_convert_model.py

Input:  wake_model.h5    (from step 1)
Output: wake_model.tflite
"""

import tensorflow as tf
import os

print("=" * 50)
print("TinyML Wake Word Detector — Step 2: Convert")
print("=" * 50)

# Check input file exists
if not os.path.exists("wake_model.h5"):
    print("\n❌ Error: wake_model.h5 not found.")
    print("   Please run 1_train_model.py first.")
    exit(1)

# Load the trained Keras model
print("\nLoading wake_model.h5...")
model = tf.keras.models.load_model("wake_model.h5")
print(f"Original model loaded.")

# Get original size estimate
original_size = sum(
    tf.size(w).numpy() * w.dtype.size
    for w in model.weights
)
print(f"Approximate original size: {original_size} bytes ({original_size/1024:.1f} KB)")

# -------------------------------------------------------
# CONVERT TO TFLITE WITH INT8 QUANTIZATION
# tf.lite.Optimize.DEFAULT applies the best available
# optimization — currently dynamic range INT8 quantization.
# -------------------------------------------------------
print("\nConverting to TFLite with INT8 quantization...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save the .tflite file
with open("wake_model.tflite", "wb") as f:
    f.write(tflite_model)

tflite_size = len(tflite_model)
print(f"\nQuantized model size: {tflite_size} bytes ({tflite_size/1024:.1f} KB)")
print(f"Size reduction: ~{original_size // tflite_size}x smaller")

# Quick sanity check — load and test the converted model
print("\nRunning sanity check on converted model...")
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input shape:  {input_details[0]['shape']}  (13 MFCC features)")
print(f"Output shape: {output_details[0]['shape']}  (1 confidence score)")

# Run one test inference
import numpy as np
test_input = np.ones((1, 13), dtype=np.float32) * 0.5
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
test_output = interpreter.get_tensor(output_details[0]['index'])
print(f"Test inference output: {test_output[0][0]:.4f}")

print("\n✅ wake_model.tflite saved successfully")
print("   Next step: run 3_make_header.py")
