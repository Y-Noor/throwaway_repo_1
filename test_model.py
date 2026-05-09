"""
SignSpeak — Model Tester
=========================
Tests the trained TFLite model live using your webcam.
Shows the predicted sign, confidence, and top-3 predictions in real time.

Usage:
    python test_model.py

Requirements:
    pip install mediapipe opencv-python numpy
"""

import cv2
import numpy as np
import time
import urllib.request
import os

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, HandLandmarker

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = 'exported_model/gesture_classifier.tflite'
LABELS_PATH = 'exported_model/gesture_labels.txt'
LANDMARK_MODEL = 'hand_landmarker.task'
CONFIDENCE_THRESHOLD = 0.75

# ── Load labels ───────────────────────────────────────────────────────────────
with open(LABELS_PATH) as f:
    labels = [l.strip() for l in f.readlines()]
print(f"Loaded {len(labels)} classes: {labels}")

# ── Load TFLite model ─────────────────────────────────────────────────────────
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"Model input shape:  {input_details[0]['shape']}")
print(f"Model output shape: {output_details[0]['shape']}")

# ── Download landmark model if needed ────────────────────────────────────────
if not os.path.exists(LANDMARK_MODEL):
    print("Downloading hand_landmarker.task (~29MB)...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, LANDMARK_MODEL)
    print("Downloaded.")

# ── MediaPipe setup ───────────────────────────────────────────────────────────
latest_result = None

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=LANDMARK_MODEL),
    running_mode=mp_vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=result_callback,
)
landmarker = HandLandmarker.create_from_options(options)

# ── Feature extraction (must match collect.py exactly) ────────────────────────
def hand_to_coords(lm_list):
    wrist   = np.array([lm_list[0].x, lm_list[0].y, lm_list[0].z])
    mid_mcp = np.array([lm_list[9].x, lm_list[9].y, lm_list[9].z])
    span    = np.linalg.norm(mid_mcp - wrist) + 1e-6
    coords  = []
    for pt in lm_list:
        coords.extend([
            (pt.x - wrist[0]) / span,
            (pt.y - wrist[1]) / span,
            (pt.z - wrist[2]) / span,
        ])
    return coords

def extract_features(result):
    if not result or not result.hand_landmarks:
        return None
    hands = result.hand_landmarks
    hand0 = hand_to_coords(hands[0]) if len(hands) > 0 else [0.0] * 63
    hand1 = hand_to_coords(hands[1]) if len(hands) > 1 else [0.0] * 63
    return np.array(hand0 + hand1, dtype=np.float32)

def run_inference(features):
    """Returns list of (label, confidence) sorted by confidence descending."""
    interpreter.set_tensor(input_details[0]['index'], features.reshape(1, -1))
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]['index'])[0]
    ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    return [(labels[i], float(p)) for i, p in ranked]

# ── Drawing helpers ───────────────────────────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def draw_landmarks(frame, result):
    if not result or not result.hand_landmarks:
        return
    h, w = frame.shape[:2]
    for hand in result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 200, 100), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 255, 150), -1)
            cv2.circle(frame, pt, 4, (0, 0, 0), 1)

def draw_prediction(frame, predictions):
    """Draws the top prediction large + top 3 smaller below."""
    if not predictions:
        return

    top_label, top_conf = predictions[0]
    is_confident = top_conf >= CONFIDENCE_THRESHOLD and top_label != 'none'

    # Bottom panel background
    cv2.rectangle(frame, (0, 370), (640, 480), (15, 15, 25), -1)

    # Top prediction
    color = (50, 255, 160) if is_confident else (100, 100, 100)
    label_text = top_label.upper() if is_confident else '—'
    cv2.putText(frame, label_text,
                (20, 430), cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 4)
    cv2.putText(frame, f"{top_conf*100:.0f}%",
                (20, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Top 3 ranked list on the right
    for i, (lbl, conf) in enumerate(predictions[:3]):
        bar_color = (50, 255, 160) if i == 0 and is_confident else (80, 80, 120)
        y = 390 + i * 28
        # Bar
        bar_w = int(conf * 200)
        cv2.rectangle(frame, (340, y - 14), (340 + bar_w, y + 6), bar_color, -1)
        cv2.rectangle(frame, (340, y - 14), (540, y + 6), (60, 60, 80), 1)
        # Label + %
        cv2.putText(frame, f"{lbl}", (350, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(frame, f"{conf*100:.1f}%", (500, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # FPS tracking
    fps_counter = 0
    fps_start   = time.time()
    fps_display = 0.0

    print("\nModel tester running — press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker.detect_async(mp_image, int(time.time() * 1000))

        draw_landmarks(frame, latest_result)

        # Run inference
        features = extract_features(latest_result)
        predictions = run_inference(features) if features is not None else []

        draw_prediction(frame, predictions)

        # Top HUD
        num_hands = len(latest_result.hand_landmarks) if latest_result and latest_result.hand_landmarks else 0
        cv2.rectangle(frame, (0, 0), (640, 36), (15, 15, 25), -1)
        cv2.putText(frame, f"SignSpeak Model Tester",
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 160), 2)
        cv2.putText(frame, f"FPS: {fps_display:.0f}  Hands: {num_hands}  Q=quit",
                    (320, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        # FPS
        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            fps_display = fps_counter / (time.time() - fps_start)
            fps_counter = 0
            fps_start   = time.time()

        cv2.imshow('SignSpeak Model Tester', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == '__main__':
    main()