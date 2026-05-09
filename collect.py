"""
SignSpeak — Gesture Data Collector (MediaPipe Tasks API)
=========================================================
Compatible with mediapipe 0.10.13+
Supports 1-handed and 2-handed signs.
Feature vector: 126 values (2 hands x 21 landmarks x xyz)

SPACE = 5s countdown, then 10 captures x 0.5s apart
N     = next sign
Q     = quit and save
"""

import cv2
import numpy as np
import csv
import os
import time
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, HandLandmarker

# ── Config ────────────────────────────────────────────────────────────────────
COUNTDOWN_SECS = 5
BURST_SHOTS    = 10
BURST_DELAY    = 0.5   # seconds between each captured frame

SIGNS = [
    'none',           # 1-hand: relaxed hand / no sign
    'welcome',        # 2-hands: both arms open outward
    'build',          # 2-hands: fists stacking
    'a',              # 1-hand: ASL letter A
    'i',              # 1-hand: ASL letter I (pinky up)
    'my',             # 1-hand: flat hand on chest
    'name',           # 2-hands: index+middle tap together
    'noor',           # 1-hand: custom pose you define
    'this',           # 1-hand: index pointing down
    'sign_speak',     # 1-hand: custom pose for app name
    'thank_you',      # 1-hand: flat hand from chin outward
    'listen',         # 1-hand: index finger to ear
    'hello',          # 1-hand: open palm wave
    'please',         # 1-hand: flat hand circular on chest
    'AI',             

]

TWO_HANDED = {'welcome', 'build', 'name', 'no', 'help', 'more', 'stop', 'AI'}

SAMPLES_PER_SIGN = 150
OUTPUT_DIR = 'data'
CSV_FILE   = os.path.join(OUTPUT_DIR, 'landmarks.csv')
LABEL_FILE = os.path.join(OUTPUT_DIR, 'labels.txt')
MODEL_PATH = 'hand_landmarker.task'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Download model if needed ──────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task (~29MB)...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Downloaded.")

# ── MediaPipe setup ───────────────────────────────────────────────────────────
latest_result = None

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=result_callback,
)
landmarker = HandLandmarker.create_from_options(options)

# ── Feature extraction ────────────────────────────────────────────────────────
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

def extract_landmarks(result):
    # Default to 126 zeros (63 for hand0 + 63 for hand1)
    # This ensures the data row is always the correct length
    hand0 = [0.0] * 63
    hand1 = [0.0] * 63

    if result and result.hand_landmarks:
        hands = result.hand_landmarks
        if len(hands) > 0:
            hand0 = hand_to_coords(hands[0])
        if len(hands) > 1:
            hand1 = hand_to_coords(hands[1])
            
    return hand0 + hand1

# ── Drawing ───────────────────────────────────────────────────────────────────
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

def draw_hud(frame, sign, count, is_two_handed, num_hands, status_text):
    hand_color = (150, 150, 255) if is_two_handed else (50, 255, 160)
    hand_label = "TWO-HANDED" if is_two_handed else "ONE-HANDED"
    need = 2 if is_two_handed else 1
    hd_color = (0, 255, 100) if num_hands >= need else (0, 80, 255)

    cv2.rectangle(frame, (0, 0), (640, 85), (15, 15, 25), -1)
    cv2.putText(frame, f"{sign.upper()}  ({count}/{SAMPLES_PER_SIGN})  [{hand_label}]",
                (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, hand_color, 2)
    cv2.putText(frame, status_text,
                (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    cv2.putText(frame, f"Hands: {num_hands}",
                (530, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hd_color, 2)

    bar = int((count / SAMPLES_PER_SIGN) * 616)
    cv2.rectangle(frame, (12, 72), (628, 80), (40, 40, 40), -1)
    cv2.rectangle(frame, (12, 72), (12 + bar, 80), hand_color, -1)

# ── Burst capture ─────────────────────────────────────────────────────────────
def run_burst(cap, sign_idx, sign, is_two_handed, all_rows, count_so_far):
    """
    5s countdown then 10 captures 0.5s apart.
    Returns (num_captured, quit_requested).
    """
    captured = 0
    #  timestamp now uses wall clock directly

    # Countdown phase
    start = time.time()
    while True:
        elapsed   = time.time() - start
        remaining = COUNTDOWN_SECS - elapsed
        if remaining <= 0:
            break

        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        landmarker.detect_async(mp_image, int(time.time() * 1000))
        draw_landmarks(frame, latest_result)

        num_hands = len(latest_result.hand_landmarks) if latest_result and latest_result.hand_landmarks else 0

        # Dim overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (640, 480), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # Big countdown number
        cv2.putText(frame, str(int(remaining) + 1),
                    (285, 290), cv2.FONT_HERSHEY_SIMPLEX, 7.0, (50, 255, 160), 10)

        draw_hud(frame, sign, count_so_far + captured, is_two_handed, num_hands,
                 f"GET READY... capturing in {int(remaining)+1}s")

        cv2.imshow('SignSpeak Collector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return captured, True

    # Burst capture phase
    for shot in range(BURST_SHOTS):
        shot_start = time.time()

        while time.time() - shot_start < BURST_DELAY:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            landmarker.detect_async(mp_image, int(time.time() * 1000))
            
            draw_landmarks(frame, latest_result)
            num_hands = len(latest_result.hand_landmarks) if latest_result and latest_result.hand_landmarks else 0
            
            if (time.time() - shot_start) < 0.08:
                cv2.rectangle(frame, (3, 90), (637, 477), (255, 255, 255), 4)

            draw_hud(frame, sign, count_so_far + captured, is_two_handed, num_hands,
                     f"CAPTURING  {shot+1}/{BURST_SHOTS}")
            cv2.imshow('SignSpeak Collector', frame)
            cv2.waitKey(1)

        # Grabbing sample — it will now ALWAYS return 126 values
        coords = extract_landmarks(latest_result)
        all_rows.append([sign_idx] + coords)
        captured += 1
        print(f"  [{sign}] shot {shot+1}/{BURST_SHOTS} recorded   total={count_so_far+captured}", end='\r')

    print(f"\n  Burst complete: {BURST_SHOTS} samples added.")
    return captured, False

# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    all_rows = []
    sign_idx = 0
    count    = 0
    # timestamp uses wall clock

    bursts_needed = -(-SAMPLES_PER_SIGN // BURST_SHOTS)  # ceiling division

    print(f"\n{'='*50}")
    print(f"SignSpeak Data Collector")
    print(f"SPACE = {COUNTDOWN_SECS}s countdown then {BURST_SHOTS} shots x {BURST_DELAY}s")
    print(f"Target: {SAMPLES_PER_SIGN} samples/sign  (~{bursts_needed} bursts each)")
    print(f"N=next sign  Q=quit")
    print(f"{'='*50}\n")

    while True:
        if sign_idx >= len(SIGNS):
            print("All signs collected!")
            break

        sign = SIGNS[sign_idx]
        is_two_handed = sign in TWO_HANDED

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        landmarker.detect_async(mp_image, int(time.time() * 1000))
        draw_landmarks(frame, latest_result)

        num_hands = len(latest_result.hand_landmarks) if latest_result and latest_result.hand_landmarks else 0
        draw_hud(frame, sign, count, is_two_handed, num_hands,
                 "SPACE=start burst  N=next sign  Q=quit")

        cv2.imshow('SignSpeak Collector', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('n'):
            print(f"  > {sign}: {count} samples saved, moving on")
            sign_idx += 1
            count = 0
        elif key == ord(' '):
            if is_two_handed and num_hands < 2:
                print(f"  ! Only {num_hands} hand(s) — use both hands for {sign}")
            captured, quit_signal = run_burst(cap, sign_idx, sign, is_two_handed, all_rows, count)
            count += captured
            if quit_signal:
                break
            if count >= SAMPLES_PER_SIGN:
                print(f"  {sign} complete ({count} samples)! Press N for next sign.")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    if all_rows:
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['label'] + [f'f{i}' for i in range(126)])
            writer.writerows(all_rows)
        print(f"\nSaved {len(all_rows)} samples to {CSV_FILE}")

    with open(LABEL_FILE, 'w') as f:
        f.write('\n'.join(SIGNS))
    print(f"Labels saved to {LABEL_FILE}")
    print(f"\nNext: python train.py")

if __name__ == '__main__':
    main()