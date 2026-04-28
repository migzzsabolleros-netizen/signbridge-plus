import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ---- SETTINGS ----
SIGNS = ['good_morning', 'good_afternoon', 'good_evening']  # your FSL vocabulary
SEQUENCES = 30       # 30 videos per sign
SEQUENCE_LENGTH = 30 # 30 frames per video

DATA_PATH = 'dataset'

# ---- SETUP ----
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
face = mp_face.FaceMesh(min_detection_confidence=0.7)
pose = mp_pose.Pose(min_detection_confidence=0.7)

# Create folders
for sign in SIGNS:
    for seq in range(SEQUENCES):
        os.makedirs(os.path.join(DATA_PATH, sign, str(seq)), exist_ok=True)

def extract_keypoints(hand_results, face_results, pose_results):
    # Hands (21 landmarks x 3 coords x 2 hands = 126 values)
    lh = np.zeros(63)
    rh = np.zeros(63)
    if hand_results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            arr = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            label = hand_results.multi_handedness[i].classification[0].label
            if label == 'Left':  lh = arr
            else:                rh = arr

    # Pose (33 landmarks x 4 coords = 132 values)
    pose_kp = np.zeros(132)
    if pose_results.pose_landmarks:
        pose_kp = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                             for lm in pose_results.pose_landmarks.landmark]).flatten()

    # Face (468 landmarks x 3 coords = 1404 values)
    face_kp = np.zeros(1404)
    if face_results.multi_face_landmarks:
        face_kp = np.array([[lm.x, lm.y, lm.z]
                             for lm in face_results.multi_face_landmarks[0].landmark]).flatten()

    return np.concatenate([lh, rh, pose_kp, face_kp])  # 1662 total values

# ---- CAPTURE LOOP ----
cap = cv2.VideoCapture(1)

for sign in SIGNS:
    for seq in range(SEQUENCES):

        # Countdown before each sequence
        for countdown in range(3, 0, -1):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.rectangle(frame, (0,0), (640, 80), (0,0,0), -1)
            cv2.putText(frame, f'Sign: {sign.upper()}  |  Set: {seq+1}/{SEQUENCES}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,200), 2)
            cv2.putText(frame, f'Get ready... {countdown}',
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)
            cv2.imshow('Collecting Data', frame)
            cv2.waitKey(1000)

        # Collect frames
        for frame_num in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hand_results = hands.process(rgb)
            face_results = face.process(rgb)
            pose_results = pose.process(rgb)

            # Draw landmarks
            if hand_results.multi_hand_landmarks:
                for hl in hand_results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
            if face_results.multi_face_landmarks:
                for fl in face_results.multi_face_landmarks:
                    mp_draw.draw_landmarks(frame, fl, mp_face.FACEMESH_CONTOURS,
                                           landmark_drawing_spec=None)
            if pose_results.pose_landmarks:
                mp_draw.draw_landmarks(frame, pose_results.pose_landmarks,
                                       mp_pose.POSE_CONNECTIONS)

            # Save keypoints
            keypoints = extract_keypoints(hand_results, face_results, pose_results)
            save_path = os.path.join(DATA_PATH, sign, str(seq), str(frame_num))
            np.save(save_path, keypoints)

            # UI
            cv2.rectangle(frame, (0,0), (640, 80), (0,0,0), -1)
            cv2.putText(frame, f'Sign: {sign.upper()}  |  Set: {seq+1}/{SEQUENCES}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,200), 2)
            cv2.putText(frame, f'Frame {frame_num+1}/{SEQUENCE_LENGTH}',
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow('Collecting Data', frame)
            cv2.waitKey(1)

        print(f'✅ Saved: {sign} | set {seq+1}')

cap.release()
cv2.destroyAllWindows()
print('\n🎉 Data collection complete!')