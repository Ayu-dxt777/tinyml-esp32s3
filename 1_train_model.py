"""
STEP 1: Train the Neural Network
=================================
This script trains a small neural network to detect a wake word.
It uses synthetic (fake) data so you can run the full pipeline
without needing a real microphone or audio dataset.

In a real project, replace X and y with real MFCC audio features.

Requirements:
    pip install tensorflow numpy

Run:
    python 1_train_model.py
"""

import numpy as np
import tensorflow as tf

print("=" * 50)
print("TinyML Wake Word Detector — Step 1: Training")
print("=" * 50)

# -------------------------------------------------------
# GENERATE SYNTHETIC TRAINING DATA
# Each row = 13 MFCC features describing one audio frame.
# MFCC (Mel-Frequency Cepstral Coefficients) is the standard
# way to represent audio as numbers for speech recognition.
# We have 1000 training examples total.
# -------------------------------------------------------
np.random.seed(42)  # Makes results reproducible
X = np.random.randn(1000, 13).astype(np.float32)

# Labels: 1 = wake word, 0 = background noise
# Using first MFCC feature as a simple decision boundary
y = (X[:, 0] > 0).astype(np.float32)

print(f"\nTraining data shape: {X.shape}")
print(f"Wake word samples: {int(y.sum())}")
print(f"Background noise samples: {int((1 - y).sum())}")

# -------------------------------------------------------
# BUILD THE NEURAL NETWORK
#
# Architecture: 13 inputs → 16 → 8 → 1 output
#
# Dense(16, relu)  — 16 pattern detectors
# Dense(8,  relu)  — 8 pattern summarizers
# Dense(1, sigmoid) — final score 0.0 to 1.0
#
# We keep it small because the ESP32-S3 only has ~384KB RAM.
# -------------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(13,)),
    tf.keras.layers.Dense(8,  activation='relu'),
    tf.keras.layers.Dense(1,  activation='sigmoid')
], name="wake_word_detector")

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel architecture:")
model.summary()

# -------------------------------------------------------
# TRAIN THE MODEL
# epochs=20 means we loop through all 1000 examples 20 times.
# validation_split=0.2 holds back 200 examples for testing.
# -------------------------------------------------------
print("\nTraining...")
history = model.fit(
    X, y,
    epochs=20,
    validation_split=0.2,
    verbose=1
)

final_accuracy = history.history['val_accuracy'][-1]
print(f"\nFinal validation accuracy: {final_accuracy:.1%}")

# Save the trained model
model.save("wake_model.h5")
print("\n✅ Model saved as wake_model.h5")
print("   Next step: run 2_convert_model.py")
