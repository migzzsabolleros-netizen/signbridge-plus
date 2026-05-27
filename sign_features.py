"""
Feature extraction utilities for SignBridge v2.

All scripts must use these functions so training-time and inference-time features match.
The extractor intentionally removes face landmarks for speed and stability.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

from sign_config import FEATURE_SIZE, SEQUENCE_LENGTH


FINGER_JOINTS = [
    (1, 2, 4),    # thumb
    (5, 6, 8),    # index
    (9, 10, 12),  # middle
    (13, 14, 16), # ring
    (17, 18, 20), # pinky
]

FINGERTIPS = [4, 8, 12, 16, 20]
POSE_POINTS = [11, 12, 13, 14, 15, 16]  # shoulders, elbows, wrists


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Returns the angle ABC in degrees."""
    ba = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    bc = np.asarray(c, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cos_angle = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def normalize_hand(coords: np.ndarray) -> np.ndarray:
    """Wrist-relative, scale-normalized 21x3 hand landmark array."""
    coords = np.asarray(coords, dtype=np.float32).reshape(21, 3)
    wrist = coords[0].copy()
    coords = coords - wrist

    # Palm-size proxy: distance from wrist to middle MCP.
    scale = np.linalg.norm(coords[9])
    if scale < 1e-6:
        scale = np.linalg.norm(coords[5]) + 1e-6
    return coords / (scale + 1e-6)


def hand_features(hand: np.ndarray) -> List[float]:
    """Builds per-hand features: normalized landmarks + bend angles + fingertip distances."""
    hand = np.asarray(hand, dtype=np.float32).reshape(21, 3)
    features: List[float] = []
    features.extend(hand.flatten().tolist())

    for a, b, c in FINGER_JOINTS:
        features.append(calculate_angle(hand[a], hand[b], hand[c]) / 180.0)

    for i in range(len(FINGERTIPS) - 1):
        p1 = hand[FINGERTIPS[i]]
        p2 = hand[FINGERTIPS[i + 1]]
        features.append(float(np.linalg.norm(p1 - p2)))

    return features


def extract_keypoints(hand_res, pose_res) -> np.ndarray:
    """
    Extracts a 162-dimensional feature vector from MediaPipe Hands + Pose results.

    Output order:
    left hand features, right hand features, minimal upper-body pose features.
    Missing landmarks are filled with zeros.
    """
    left_hand = np.zeros((21, 3), dtype=np.float32)
    right_hand = np.zeros((21, 3), dtype=np.float32)

    if hand_res is not None and hand_res.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(hand_res.multi_hand_landmarks):
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
            coords = normalize_hand(coords)

            # MediaPipe's handedness is mirrored when the webcam frame is flipped. This code assumes
            # you flip the image before processing, like your original test_model.py did.
            label = hand_res.multi_handedness[i].classification[0].label
            if label == 'Left':
                left_hand = coords
            else:
                right_hand = coords

    features: List[float] = []
    features.extend(hand_features(left_hand))
    features.extend(hand_features(right_hand))

    pose_features: List[float] = []
    if pose_res is not None and pose_res.pose_landmarks:
        pose = pose_res.pose_landmarks.landmark
        shoulder_center = np.array([
            (pose[11].x + pose[12].x) / 2.0,
            (pose[11].y + pose[12].y) / 2.0,
            (pose[11].z + pose[12].z) / 2.0,
        ], dtype=np.float32)

        shoulder_width = np.linalg.norm(np.array([
            pose[11].x - pose[12].x,
            pose[11].y - pose[12].y,
            pose[11].z - pose[12].z,
        ], dtype=np.float32)) + 1e-6

        for idx in POSE_POINTS:
            lm = pose[idx]
            pt = np.array([lm.x, lm.y, lm.z], dtype=np.float32)
            pt = (pt - shoulder_center) / shoulder_width
            pose_features.extend(pt.tolist())
    else:
        pose_features = [0.0] * (len(POSE_POINTS) * 3)

    features.extend(pose_features)
    arr = np.asarray(features, dtype=np.float32)

    if arr.shape[0] != FEATURE_SIZE:
        raise ValueError(f'Feature size mismatch: got {arr.shape[0]}, expected {FEATURE_SIZE}')

    return arr


def calculate_motion_magnitude(prev_keypoints: Optional[np.ndarray], curr_keypoints: Optional[np.ndarray]) -> float:
    """Average hand-feature movement between two frames."""
    if prev_keypoints is None or curr_keypoints is None:
        return 0.0
    prev = np.asarray(prev_keypoints, dtype=np.float32)
    curr = np.asarray(curr_keypoints, dtype=np.float32)
    if prev.shape != curr.shape:
        return 0.0
    # First 144 values correspond to both hands including hand angles/distances.
    return float(np.mean(np.abs(curr[:144] - prev[:144])))


class KeypointSmoother:
    """Exponential moving average smoother for landmark features."""
    def __init__(self, alpha: float = 0.65):
        self.alpha = alpha
        self.prev_keypoints: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.prev_keypoints = None

    def smooth(self, keypoints: np.ndarray) -> np.ndarray:
        keypoints = np.asarray(keypoints, dtype=np.float32)
        if self.prev_keypoints is None:
            self.prev_keypoints = keypoints.copy()
            return keypoints
        smoothed = self.alpha * keypoints + (1.0 - self.alpha) * self.prev_keypoints
        self.prev_keypoints = smoothed.copy()
        return smoothed.astype(np.float32)


def normalize_sequence_length(sequence: Iterable[np.ndarray], target_length: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Resamples any gesture sequence to a fixed length."""
    sequence = np.asarray(list(sequence), dtype=np.float32)
    if sequence.ndim != 2:
        raise ValueError(f'Expected sequence shape (frames, features), got {sequence.shape}')
    if len(sequence) == target_length:
        return sequence.astype(np.float32)
    if len(sequence) < 2:
        return np.repeat(sequence, target_length, axis=0).astype(np.float32)

    old_steps = np.linspace(0, 1, len(sequence))
    new_steps = np.linspace(0, 1, target_length)
    interpolator = interp1d(old_steps, sequence, axis=0, kind='linear', fill_value='extrapolate')
    return interpolator(new_steps).astype(np.float32)


def classify_gesture_type(sequence: Iterable[np.ndarray], motion_threshold: float = 0.010) -> str:
    """Simple static/dynamic router based on average hand motion."""
    sequence = np.asarray(list(sequence), dtype=np.float32)
    if len(sequence) < 2:
        return 'static'
    motion = float(np.mean(np.abs(np.diff(sequence[:, :144], axis=0))))
    return 'dynamic' if motion >= motion_threshold else 'static'


def has_detected_hand(hand_res) -> bool:
    return bool(hand_res is not None and hand_res.multi_hand_landmarks)


def hand_detection_confidence(hand_res) -> float:
    if hand_res is None or not hand_res.multi_handedness:
        return 0.0
    scores = [h.classification[0].score for h in hand_res.multi_handedness]
    return float(np.mean(scores)) if scores else 0.0
