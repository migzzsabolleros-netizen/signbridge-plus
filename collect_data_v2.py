"""
Collect SignBridge v2 training data using the upgraded feature extractor.

Usage examples:
    python collect_data_v2.py --sign Hello --sequences 100
    python collect_data_v2.py --all --sequences 50

Controls:
    q = quit
    s = start recording the current sign

Dataset output:
    dataset/<SIGN>/<SEQUENCE_INDEX>/<FRAME>.npy
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from sign_config import DATA_PATH, SEQUENCE_LENGTH, SIGNS
from sign_features import extract_keypoints


mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
COUNTDOWN_SECONDS = 2


def next_sequence_index(sign_dir: Path) -> int:
    sign_dir.mkdir(parents=True, exist_ok=True)
    existing = [int(p.name) for p in sign_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    return max(existing) + 1 if existing else 0


def show_countdown(cap: cv2.VideoCapture, sign: str, seconds: int) -> bool:
    end_time = time.monotonic() + seconds

    while True:
        remaining = end_time - time.monotonic()
        if remaining <= 0:
            return True

        ret, frame = cap.read()
        if not ret:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False
            continue

        frame = cv2.flip(frame, 1)
        countdown = int(remaining) + 1

        cv2.putText(frame, f"Sign: {sign}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, "Get ready", (220, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(frame, str(countdown), (295, 285), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 5)
        cv2.putText(frame, "Press Q to quit", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('SignBridge v2 Data Collection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False


def collect_for_sign(sign: str, num_sequences: int, camera_index: int, output_dir: str) -> None:
    if sign not in SIGNS:
        print(f"❌ Unknown sign: {sign}")
        print("Edit sign_config.py if you need to add new signs.")
        return

    sign_dir = Path(output_dir) / sign
    start_idx = next_sequence_index(sign_dir)

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

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

    print(f"\nCollecting: {sign}")
    print("Press 's' to record each sequence. Press 'q' to quit.")

    sequence_id = start_idx
    recorded = 0

    while recorded < num_sequences:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        display = frame.copy()
        cv2.putText(display, f"Sign: {sign}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(display, f"Recorded: {recorded}/{num_sequences}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, "Press S to record, Q to quit", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('SignBridge v2 Data Collection', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key != ord('s'):
            continue

        print(f"Starting {COUNTDOWN_SECONDS}-second countdown for {sign} sequence {sequence_id}...")
        if not show_countdown(cap, sign, COUNTDOWN_SECONDS):
            break

        seq_dir = sign_dir / str(sequence_id)
        seq_dir.mkdir(parents=True, exist_ok=True)

        print(f"Recording {sign} sequence {sequence_id}...")

        frames_saved = 0
        while frames_saved < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            hand_res = hands.process(rgb)
            pose_res = pose.process(rgb)
            rgb.flags.writeable = True

            keypoints = extract_keypoints(hand_res, pose_res)
            np.save(seq_dir / f"{frames_saved}.npy", keypoints)

            if hand_res.multi_hand_landmarks:
                for hand_landmarks in hand_res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if pose_res.pose_landmarks:
                mp_draw.draw_landmarks(frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.putText(frame, f"Recording {frames_saved + 1}/{SEQUENCE_LENGTH}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow('SignBridge v2 Data Collection', frame)
            cv2.waitKey(1)
            frames_saved += 1

        print(f"✅ Saved: {seq_dir}")
        sequence_id += 1
        recorded += 1

    hands.close()
    pose.close()
    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sign', type=str, help='Single sign to collect')
    parser.add_argument('--all', action='store_true', help='Collect all signs in sign_config.py')
    parser.add_argument('--sequences', type=int, default=30, help='Sequences to record per sign')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--output', type=str, default=DATA_PATH, help='Dataset output directory')
    args = parser.parse_args()

    if not args.sign and not args.all:
        parser.error("Use --sign <SIGN> or --all")

    signs = SIGNS if args.all else [args.sign]
    for sign in signs:
        collect_for_sign(sign, args.sequences, args.camera, args.output)


if __name__ == '__main__':
    main()
