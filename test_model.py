from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import threading

app = Flask(__name__)

# ---- SETTINGS ----
SIGNS = ['pamilya', 'lola', 'lolo', 'mama', 'papa']
SEQUENCE_LENGTH = 40
THRESHOLD = 0.7

# ---- Load Model ----
print("Loading model...")
model = tf.keras.models.load_model('signbridge_model.keras')
print("✅ Model loaded!")

# ---- MediaPipe ----
mp_hands = mp.solutions.hands
mp_face  = mp.solutions.face_mesh
mp_pose  = mp.solutions.pose
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
face  = mp_face.FaceMesh(min_detection_confidence=0.7)
pose  = mp_pose.Pose(min_detection_confidence=0.7)

# ---- Shared state ----
latest_frame     = None
frame_lock       = threading.Lock()
prediction_state = {'sign': '', 'confidence': 0.0, 'sentence': [], 'history': []}
sequence         = []

def extract_keypoints(hand_res, face_res, pose_res):
    lh = np.zeros(63)
    rh = np.zeros(63)
    if hand_res.multi_hand_landmarks:
        for i, hl in enumerate(hand_res.multi_hand_landmarks):
            arr   = np.array([[lm.x, lm.y, lm.z] for lm in hl.landmark]).flatten()
            label = hand_res.multi_handedness[i].classification[0].label
            if label == 'Left': lh = arr
            else:               rh = arr

    pose_kp = np.zeros(132)
    if pose_res.pose_landmarks:
        pose_kp = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                             for lm in pose_res.pose_landmarks.landmark]).flatten()

    face_kp = np.zeros(18)
    if face_res.multi_face_landmarks:
        lms     = face_res.multi_face_landmarks[0].landmark
        indices = [1, 152, 234, 454, 33, 263]
        face_kp = np.array([[lms[i].x, lms[i].y, lms[i].z] for i in indices]).flatten()

    return np.concatenate([lh, rh, pose_kp, face_kp])

def capture_loop():
    global latest_frame, sequence
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    print("✅ Camera opened!")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        hand_res = hands.process(rgb)
        face_res = face.process(rgb)
        pose_res = pose.process(rgb)

        rgb.flags.writeable = True
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        if hand_res.multi_hand_landmarks:
            for hl in hand_res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0,255,200), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(0,180,255), thickness=2))
        if face_res.multi_face_landmarks:
            for fl in face_res.multi_face_landmarks:
                mp_draw.draw_landmarks(frame, fl, mp_face.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_draw.DrawingSpec(color=(255,100,200), thickness=1))
        if pose_res.pose_landmarks:
            mp_draw.draw_landmarks(frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(255,200,0), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(255,140,0), thickness=2))

        # Build sequence
        keypoints = extract_keypoints(hand_res, face_res, pose_res)
        sequence.append(keypoints)
        sequence = sequence[-SEQUENCE_LENGTH:]

        # Predict every 3rd frame
        if len(sequence) == SEQUENCE_LENGTH and frame_count % 3 == 0:
            try:
                input_data = np.expand_dims(sequence, axis=0)
                prediction = model.predict(input_data, verbose=0)[0]
                idx        = np.argmax(prediction)
                confidence = float(prediction[idx])
                sign       = SIGNS[idx] if confidence > THRESHOLD else ''
            except Exception as e:
                print(f"❌ Prediction error: {e}")
                sign       = ''
                confidence = 0.0

            if sign and (not prediction_state['sentence'] or
                         prediction_state['sentence'][-1] != sign):
                prediction_state['sentence'].append(sign)
                prediction_state['history'].insert(0, {
                    'sign': sign,
                    'confidence': round(confidence * 100, 1)
                })
                prediction_state['history'] = prediction_state['history'][:10]

            prediction_state['sign']       = sign
            prediction_state['confidence'] = round(confidence * 100, 1)

        # Draw overlay
        sign       = prediction_state['sign']
        confidence = prediction_state['confidence']
        label      = f"{sign.upper()} ({confidence:.0f}%)" if sign else "Detecting..."
        color      = (0, 255, 100) if sign else (0, 100, 255)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0,0,0), -1)
        cv2.putText(frame, label, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

        bar_w = int((frame.shape[1] - 20) * (confidence / 100))
        cv2.rectangle(frame, (10, 45), (frame.shape[1]-10, 50), (50,50,50), -1)
        cv2.rectangle(frame, (10, 45), (10 + bar_w, 50), color, -1)

        with frame_lock:
            latest_frame = frame.copy()

    cap.release()

def generate_frames():
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            _, buffer = cv2.imencode('.jpg', latest_frame,
                                     [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('test_model.html', signs=SIGNS)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/prediction')
def prediction():
    return jsonify(prediction_state)

@app.route('/clear', methods=['POST'])
def clear():
    prediction_state['sentence'] = []
    prediction_state['history']  = []
    return jsonify({'status': 'cleared'})

if __name__ == '__main__':
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()
    print("✅ Open http://localhost:5001 in your browser")
    app.run(port=5001, debug=False)