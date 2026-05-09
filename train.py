"""
SignSpeak — Gesture Classifier Trainer
=======================================
Reads landmark CSVs collected by collect.py, trains a small MLP,
and exports a .tflite model + label file ready for Flutter.

Usage:
    python train.py

Requirements:
    pip install tensorflow scikit-learn numpy pandas matplotlib
"""

import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

DATA_DIR    = 'data'
CSV_FILE    = os.path.join(DATA_DIR, 'landmarks.csv')
LABEL_FILE  = os.path.join(DATA_DIR, 'labels.txt')
OUTPUT_DIR  = 'exported_model'
TFLITE_FILE = os.path.join(OUTPUT_DIR, 'gesture_classifier.tflite')
LABELS_OUT  = os.path.join(OUTPUT_DIR, 'gesture_labels.txt')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(CSV_FILE)

with open(LABEL_FILE) as f:
    all_labels = [l.strip() for l in f.readlines()]

# Use only classes present in the CSV
present = sorted(df['label'].unique())
labels = [all_labels[i] for i in present]
num_classes = len(labels)

print(f"Classes ({num_classes}): {labels}")
print(f"Total samples: {len(df)}")
print(f"Samples per class:")
for i, name in enumerate(labels):
    print(f"  {name}: {(df['label']==i).sum()}")

X = df.drop('label', axis=1).values.astype(np.float32)
y = df['label'].values.astype(np.int32)

# ── Augment — add small noise to improve robustness ──────────────────────────
print("\nAugmenting data...")
noise_factor = 0.01
X_noisy = X + np.random.normal(0, noise_factor, X.shape).astype(np.float32)
X = np.vstack([X, X_noisy])
y = np.hstack([y, y])
print(f"After augmentation: {len(X)} samples")

# ── Train/val/test split ──────────────────────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"\nSplit: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

# ── Build model ───────────────────────────────────────────────────────────────
print("\nBuilding model...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(126,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax'),
], name='gesture_classifier')

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

# ── Train ─────────────────────────────────────────────────────────────────────
print("\nTraining...")
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=15, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6
    ),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1,
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\nEvaluating...")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {acc*100:.1f}%  |  Loss: {loss:.4f}")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
print("\nClassification report:")
print(classification_report(y_test, y_pred, labels=list(range(num_classes)), target_names=labels))

# ── Plot training curves ──────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Val')
ax1.set_title('Accuracy'); ax1.legend(); ax1.set_xlabel('Epoch')
ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Val')
ax2.set_title('Loss'); ax2.legend(); ax2.set_xlabel('Epoch')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=120)
print(f"✓ Training curves saved")

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm, cmap='Greens')
ax.set_xticks(range(num_classes)); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(num_classes)); ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
plt.colorbar(im)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=120)
print(f"✓ Confusion matrix saved")

# ── Export TFLite ─────────────────────────────────────────────────────────────
print("\nExporting TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # INT8 quantisation
tflite_model = converter.convert()

with open(TFLITE_FILE, 'wb') as f:
    f.write(tflite_model)

size_kb = os.path.getsize(TFLITE_FILE) / 1024
print(f"✓ TFLite model saved: {TFLITE_FILE}  ({size_kb:.1f} KB)")

# Save labels
with open(LABELS_OUT, 'w') as f:
    f.write('\n'.join(labels))
print(f"✓ Labels saved: {LABELS_OUT}")

print(f"""
{'='*50}
Done! Files to copy into Flutter:

  {TFLITE_FILE}  →  signspeak/assets/models/gesture_classifier.tflite
  {LABELS_OUT}   →  signspeak/assets/models/gesture_labels.txt

Test accuracy: {acc*100:.1f}%
{'='*50}
""")