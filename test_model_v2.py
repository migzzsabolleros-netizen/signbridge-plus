"""
SignBridge v2 Flask inference server.

Keeps your existing routes:
    /              -> render_template('test_model.html', signs=SIGNS)
    /video_feed    -> MJPEG webcam stream
    /prediction    -> JSON prediction state
    /clear         -> clear sentence/history

Main upgrades:
    - no FaceMesh
    - normalized hand + minimal pose features
    - motion-based gesture segmentation
    - optional static model router
    - dynamic BiLSTM model
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from flask import Flask, Response, jsonify, render_template

from sign_config import (
    FEATURE_SIZE,
    LABELS_PATH,
    MODEL_PATH,
    SEQUENCE_LENGTH,
    SIGNS,
    STATIC_LABELS_PATH,
    STATIC_MODEL_PATH,
)
from sign_features import (
    KeypointSmoother,
    calculate_motion_magnitude,
    classify_gesture_type,
    extract_keypoints,
    hand_detection_confidence,
    has_detected_hand,
    normalize_sequence_length,
)


# -------------------- SETTINGS --------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

MIN_HAND_DETECTION_CONFIDENCE = 0.60
MOTION_THRESHOLD = 0.008
IDLE_FRAMES_REQUIRED = 10
MIN_GESTURE_FRAMES = 12
MAX_GESTURE_FRAMES = 90

DYNAMIC_CONFIDENCE_THRESHOLD = 0.70
STATIC_CONFIDENCE_THRESHOLD = 0.75
DUPLICATE_COOLDOWN_SECONDS = 1.25

KEYPOINT_SMOOTHING_ALPHA = 0.65
TOP_K_TO_PRINT = 3


# -------------------- APP + MODELS --------------------
app = Flask(__name__)

print('Loading SignBridge v2 model...')
if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(
        f"{MODEL_PATH} not found. Train first with: python train_model_v2.py\n"
        "Note: v2 requires data collected with collect_data_v2.py."
    )

model = tf.keras.models.load_model(MODEL_PATH)
print(f'✅ Dynamic/unified model loaded: {MODEL_PATH}')

if Path(LABELS_PATH).exists():
    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        dynamic_labels = json.load(f)
else:
    dynamic_labels = SIGNS

static_model = None
static_labels = []
if Path(STATIC_MODEL_PATH).exists() and Path(STATIC_LABELS_PATH).exists():
    static_model = tf.keras.models.load_model(STATIC_MODEL_PATH)
    with open(STATIC_LABELS_PATH, 'r', encoding='utf-8') as f:
        static_labels = json.load(f)
    print(f'✅ Optional static model loaded: {STATIC_MODEL_PATH}')
else:
    print('ℹ️ Optional static model not found. Static gestures will use the dynamic/unified model.')


# -------------------- MEDIAPIPE --------------------
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
pose = mp_pose.Pose(
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)


# -------------------- SHARED STATE --------------------
latest_frame = None
frame_lock = threading.Lock()
state_lock = threading.Lock()

prediction_state = {
    'sign': '',
    'confidence': 0.0,
    'sentence': [],
    'history': [],
    'status': 'Starting camera...',
    'gesture_type': '',
    'top_predictions': [],
}

keypoint_smoother = KeypointSmoother(alpha=KEYPOINT_SMOOTHING_ALPHA)

last_confirmed_sign = ''
last_confirmed_time = 0.0


def update_state(**kwargs) -> None:
    with state_lock:
        prediction_state.update(kwargs)


def add_sign_to_sentence(sign: str, confidence: float, gesture_type: str, top_predictions: List[dict]) -> None:
    global last_confirmed_sign, last_confirmed_time

    now = time.time()
    if sign == last_confirmed_sign and (now - last_confirmed_time) < DUPLICATE_COOLDOWN_SECONDS:
        update_state(
            sign=sign,
            confidence=round(confidence * 100, 1),
            status=f'Duplicate filtered: {sign}',
            gesture_type=gesture_type,
            top_predictions=top_predictions,
        )
        return

    with state_lock:
        if not prediction_state['sentence'] or prediction_state['sentence'][-1] != sign:
            prediction_state['sentence'].append(sign)

        prediction_state['history'].insert(0, {
            'sign': sign,
            'confidence': round(confidence * 100, 1),
            'type': gesture_type,
        })
        prediction_state['history'] = prediction_state['history'][:10]

        prediction_state['sign'] = sign
        prediction_state['confidence'] = round(confidence * 100, 1)
        prediction_state['status'] = f'Detected {sign}'
        prediction_state['gesture_type'] = gesture_type
        prediction_state['top_predictions'] = top_predictions

    last_confirmed_sign = sign
    last_confirmed_time = now
    print(f'✅ Detected: {sign} ({confidence:.2%}) [{gesture_type}]')


def build_top_predictions(prediction: np.ndarray, labels: List[str]) -> List[dict]:
    top_indices = np.argsort(prediction)[-TOP_K_TO_PRINT:][::-1]
    top = []
    for idx in top_indices:
        if idx < len(labels):
            top.append({'sign': labels[idx], 'confidence': round(float(prediction[idx]) * 100, 1)})
    return top


def predict_dynamic(sequence: np.ndarray) -> Tuple[str, float, List[dict]]:
    input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
    prediction = model.predict(input_data, verbose=0)[0]
    idx = int(np.argmax(prediction))
    confidence = float(prediction[idx])
    labels = dynamic_labels
    sign = labels[idx] if idx < len(labels) else ''
    return sign, confidence, build_top_predictions(prediction, labels)


def predict_static(sequence: np.ndarray) -> Tuple[str, float, List[dict]]:
    # Static model uses a stable average of the middle frames.
    if static_model is None:
        return predict_dynamic(sequence)

    mid = len(sequence) // 2
    sample = np.mean(sequence[max(0, mid - 3): min(len(sequence), mid + 4)], axis=0)
    input_data = np.expand_dims(sample, axis=0).astype(np.float32)
    prediction = static_model.predict(input_data, verbose=0)[0]
    idx = int(np.argmax(prediction))
    confidence = float(prediction[idx])
    sign = static_labels[idx] if idx < len(static_labels) else ''
    return sign, confidence, build_top_predictions(prediction, static_labels)


def process_gesture(recorded_sequence: List[np.ndarray]) -> None:
    if len(recorded_sequence) < MIN_GESTURE_FRAMES:
        update_state(status='Gesture too short')
        return

    try:
        sequence = normalize_sequence_length(recorded_sequence, target_length=SEQUENCE_LENGTH)
    except Exception as exc:
        print(f'❌ Sequence normalization error: {exc}')
        update_state(status='Sequence error')
        return

    if sequence.shape != (SEQUENCE_LENGTH, FEATURE_SIZE):
        print(f'❌ Bad sequence shape: {sequence.shape}')
        update_state(status='Bad feature shape')
        return

    gesture_type = classify_gesture_type(sequence)

    if gesture_type == 'static':
        sign, confidence, top = predict_static(sequence)
        threshold = STATIC_CONFIDENCE_THRESHOLD
    else:
        sign, confidence, top = predict_dynamic(sequence)
        threshold = DYNAMIC_CONFIDENCE_THRESHOLD

    print('\n📊 Top predictions:')
    for item in top:
        print(f"   {item['sign']:35} {item['confidence']:5.1f}%")

    if not sign or confidence < threshold:
        update_state(
            sign='',
            confidence=round(confidence * 100, 1),
            status=f'Low confidence ({confidence:.0%})',
            gesture_type=gesture_type,
            top_predictions=top,
        )
        return

    add_sign_to_sentence(sign, confidence, gesture_type, top)


def draw_overlay(frame, status: str, confidence: float, signing: bool, gesture_frames: int) -> None:
    color = (0, 180, 255) if signing else (0, 255, 100)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)

    cv2.putText(
        frame,
        status,
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
    )

    if signing:
        cv2.putText(
            frame,
            f'Recording gesture... {gesture_frames} frames',
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
    else:
        bar_w = int((frame.shape[1] - 20) * min(confidence / 100.0, 1.0))
        cv2.rectangle(frame, (10, 58), (frame.shape[1] - 10, 70), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 58), (10 + bar_w, 70), color, -1)


def capture_loop() -> None:
    global latest_frame

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print('❌ Cannot open camera')
        update_state(status='Cannot open camera')
        return

    print('✅ Camera opened!')
    update_state(status='Ready')

    prev_keypoints = None
    signing = False
    idle_counter = 0
    recorded_sequence: List[np.ndarray] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        hand_res = hands.process(rgb)
        pose_res = pose.process(rgb)

        rgb.flags.writeable = True
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if hand_res.multi_hand_landmarks:
            for hand_landmarks in hand_res.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 200), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(0, 180, 255), thickness=2),
                )

        if pose_res.pose_landmarks:
            # Drawing full pose is okay visually; feature extraction only uses shoulders/elbows/wrists.
            mp_draw.draw_landmarks(
                frame,
                pose_res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(255, 200, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(255, 140, 0), thickness=2),
            )

        hand_conf = hand_detection_confidence(hand_res)
        if not has_detected_hand(hand_res) or hand_conf < MIN_HAND_DETECTION_CONFIDENCE:
            if signing and len(recorded_sequence) >= MIN_GESTURE_FRAMES:
                process_gesture(recorded_sequence)
            signing = False
            idle_counter = 0
            recorded_sequence = []
            prev_keypoints = None
            keypoint_smoother.reset()
            update_state(status='Show your hand clearly')
        else:
            try:
                keypoints = extract_keypoints(hand_res, pose_res)
                keypoints = keypoint_smoother.smooth(keypoints)

                motion = calculate_motion_magnitude(prev_keypoints, keypoints)
                prev_keypoints = keypoints.copy()

                if motion > MOTION_THRESHOLD:
                    if not signing:
                        recorded_sequence = []
                        update_state(status='Recording gesture...')
                    signing = True
                    idle_counter = 0
                    recorded_sequence.append(keypoints)

                else:
                    if signing:
                        idle_counter += 1
                        recorded_sequence.append(keypoints)

                        if idle_counter >= IDLE_FRAMES_REQUIRED or len(recorded_sequence) >= MAX_GESTURE_FRAMES:
                            process_gesture(recorded_sequence)
                            signing = False
                            idle_counter = 0
                            recorded_sequence = []
                    else:
                        update_state(status='Ready')

            except Exception as exc:
                print(f'❌ Feature/prediction error: {type(exc).__name__}: {exc}')
                update_state(status='Processing error')

        with state_lock:
            status = prediction_state.get('status', 'Ready')
            confidence = float(prediction_state.get('confidence', 0.0))

        draw_overlay(frame, status, confidence, signing, len(recorded_sequence))

        with frame_lock:
            latest_frame = frame.copy()

    cap.release()


def generate_frames():
    while True:
        with frame_lock:
            if latest_frame is None:
                frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                cv2.putText(frame, 'Starting camera...', (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            else:
                frame = latest_frame.copy()

        ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template('test_model.html', signs=SIGNS)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/prediction')
def prediction():
    with state_lock:
        return jsonify(prediction_state)


@app.route('/clear', methods=['POST'])
def clear():
    global last_confirmed_sign, last_confirmed_time
    with state_lock:
        prediction_state['sentence'] = []
        prediction_state['history'] = []
        prediction_state['sign'] = ''
        prediction_state['confidence'] = 0.0
        prediction_state['top_predictions'] = []
        prediction_state['status'] = 'Cleared'
    last_confirmed_sign = ''
    last_confirmed_time = 0.0
    return jsonify({'status': 'cleared'})


if __name__ == '__main__':
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()
    print('✅ Open http://localhost:5001 in your browser')
    app.run(port=5001, debug=False, threaded=True)
