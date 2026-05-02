import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ---- SETTINGS ----
SIGNS = ['pamilya', 'lola', 'lolo', 'mama', 'papa']  # Add more signs as needed
SEQUENCES = 30
SEQUENCE_LENGTH = 40

DATA_PATH = 'dataset'
BREAK_SECONDS = 60  # 1 minute break between signs

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
    lh = np.zeros(63)
    rh = np.zeros(63)
    if hand_results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            arr = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            label = hand_results.multi_handedness[i].classification[0].label
            if label == 'Left':  lh = arr
            else:                rh = arr

    pose_kp = np.zeros(132)
    if pose_results.pose_landmarks:
        pose_kp = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                             for lm in pose_results.pose_landmarks.landmark]).flatten()

    # REDUCED FACE: Only 6 key points for hand-to-face proximity detection
    face_kp = np.zeros(18)
    if face_results.multi_face_landmarks:
        landmarks = face_results.multi_face_landmarks[0].landmark
        # Key points for detecting hand-to-face contact
        key_indices = [1, 152, 234, 454, 33, 263]  # nose, chin, cheeks, eyes
        face_kp = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] 
                             for i in key_indices]).flatten()

    return np.concatenate([lh, rh, pose_kp, face_kp])

# ---- CAPTURE LOOP ----
cap = cv2.VideoCapture(1)

for sign_idx, sign in enumerate(SIGNS):

    # ---- SIGN INTRO SCREEN ----
    for _ in range(30):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
        cv2.putText(frame, f'NEXT SIGN: {sign.upper()}',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 200), 3)
        cv2.putText(frame, f'Sign {sign_idx+1} of {len(SIGNS)}  |  Get in position!',
                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Collecting Data', frame)
        cv2.waitKey(100)

    for seq in range(SEQUENCES):

        # Countdown before each sequence
        for countdown in range(3, 0, -1):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
            cv2.putText(frame, f'Sign: {sign.upper()}  |  Set: {seq+1}/{SEQUENCES}',
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2)
            cv2.putText(frame, f'Get ready... {countdown}',
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
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
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
            cv2.putText(frame, f'Sign: {sign.upper()}  |  Set: {seq+1}/{SEQUENCES}',
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2)
            cv2.putText(frame, f'Frame {frame_num+1}/{SEQUENCE_LENGTH}',
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Warn if no hands detected
            if not hand_results.multi_hand_landmarks:
                cv2.putText(frame, '⚠ NO HANDS DETECTED',
                            (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow('Collecting Data', frame)
            cv2.waitKey(1)

        print(f'✅ Saved: {sign} | set {seq+1}/{SEQUENCES}')

    print(f'\n🎉 Done collecting: {sign.upper()}')

    # ---- BREAK SCREEN (skip after last sign) ----
    if sign_idx < len(SIGNS) - 1:
        print(f'💤 Take a {BREAK_SECONDS} second break...')
        for remaining in range(BREAK_SECONDS, 0, -1):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 160), (0, 0, 0), -1)
            cv2.putText(frame, '💤 BREAK TIME! Rest your hands.',
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2)
            cv2.putText(frame, f'Next sign: {SIGNS[sign_idx+1].upper()}',
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            cv2.putText(frame, f'Starting in: {remaining}s',
                        (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            cv2.imshow('Collecting Data', frame)
            cv2.waitKey(1000)
            print(f'\r⏳ Next sign in: {remaining}s ', end='')
        print('\n')

cap.release()
cv2.destroyAllWindows()
print('\n🎉 All signs collected! Run check_data.py to verify.')
