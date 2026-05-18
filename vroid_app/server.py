from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import mediapipe as mp
import numpy as np
import math
import threading

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(model_complexity=1, smooth_landmarks=True,
                    min_detection_confidence=0.6, min_tracking_confidence=0.6)
face = mp_face.FaceMesh(refine_landmarks=True, min_detection_confidence=0.6)
hands_detector = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7,
                                 min_tracking_confidence=0.6)

latest_frame = None
frame_lock = threading.Lock()

class Smoother:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.prev = {}
    def smooth(self, key, val):
        if key not in self.prev:
            self.prev[key] = val
        self.prev[key] = self.alpha * val + (1 - self.alpha) * self.prev[key]
        return self.prev[key]

smoother = Smoother(alpha=0.2)

def lm_to_np(lm):
    return np.array([lm.x, lm.y, -lm.z])

def vec_to_quat(v1, v2):
    v1 = v1 / (np.linalg.norm(v1) + 1e-6)
    v2 = v2 / (np.linalg.norm(v2) + 1e-6)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    axis = np.cross(v1, v2)
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-6:
        return (0.0, 0.0, 0.0, 1.0)
    axis = axis / axis_len
    angle = math.acos(dot)
    s = math.sin(angle / 2)
    return (float(axis[0]*s), float(axis[1]*s),
            float(axis[2]*s), float(math.cos(angle/2)))

def smooth_quat(name, qx, qy, qz, qw):
    return (smoother.smooth(name+'x', qx), smoother.smooth(name+'y', qy),
            smoother.smooth(name+'z', qz), smoother.smooth(name+'w', qw))

# Finger bone names mapping (MediaPipe index → VRM bone name)
FINGER_BONES = {
    'Right': [
        ('rightIndexProximal',    1, 2),
        ('rightIndexIntermediate',2, 3),
        ('rightIndexDistal',      3, 4),
        ('rightMiddleProximal',   5, 6),
        ('rightMiddleIntermediate',6,7),
        ('rightMiddleDistal',     7, 8),
        ('rightRingProximal',     9, 10),
        ('rightRingIntermediate', 10,11),
        ('rightRingDistal',       11,12),
        ('rightLittleProximal',   13,14),
        ('rightLittleIntermediate',14,15),
        ('rightLittleDistal',     15,16),
        ('rightThumbProximal',    1, 2),
        ('rightThumbMetacarpal',  2, 3),
        ('rightThumbDistal',      3, 4),
    ],
    'Left': [
        ('leftIndexProximal',     1, 2),
        ('leftIndexIntermediate', 2, 3),
        ('leftIndexDistal',       3, 4),
        ('leftMiddleProximal',    5, 6),
        ('leftMiddleIntermediate',6, 7),
        ('leftMiddleDistal',      7, 8),
        ('leftRingProximal',      9, 10),
        ('leftRingIntermediate',  10,11),
        ('leftRingDistal',        11,12),
        ('leftLittleProximal',    13,14),
        ('leftLittleIntermediate',14,15),
        ('leftLittleDistal',      15,16),
        ('leftThumbProximal',     1, 2),
        ('leftThumbMetacarpal',   2, 3),
        ('leftThumbDistal',       3, 4),
    ]
}

def process_fingers(hand_landmarks, label, data):
    lm = hand_landmarks.landmark
    ref = np.array([0.0, -1.0, 0.0])
    bones = FINGER_BONES[label]
    for bone_name, idx_a, idx_b in bones:
        a = lm_to_np(lm[idx_a])
        b = lm_to_np(lm[idx_b])
        direction = b - a
        qx,qy,qz,qw = vec_to_quat(ref, direction)
        data[bone_name] = smooth_quat(bone_name, qx,qy,qz,qw)

def capture_loop():
    global latest_frame
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
    ref_down = np.array([0.0, -1.0, 0.0])
    frame_count = 0
    display_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        pose_results  = pose.process(rgb)
        face_results  = face.process(rgb)
        hand_results  = hands_detector.process(rgb)

        rgb.flags.writeable = True
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        data = {}

        if pose_results.pose_landmarks:
            mp_draw.draw_landmarks(frame, pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,255,200), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(0,180,255), thickness=2))

            lm = pose_results.pose_landmarks.landmark
            rs = lm_to_np(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER])
            re = lm_to_np(lm[mp_pose.PoseLandmark.RIGHT_ELBOW])
            rw = lm_to_np(lm[mp_pose.PoseLandmark.RIGHT_WRIST])
            qx,qy,qz,qw = vec_to_quat(ref_down, re-rs)
            data['RightUpperArm'] = smooth_quat('rua',qx,qy,qz,qw)
            qx,qy,qz,qw = vec_to_quat(ref_down, rw-re)
            data['RightLowerArm'] = smooth_quat('rla',qx,qy,qz,qw)

            ls = lm_to_np(lm[mp_pose.PoseLandmark.LEFT_SHOULDER])
            le = lm_to_np(lm[mp_pose.PoseLandmark.LEFT_ELBOW])
            lw = lm_to_np(lm[mp_pose.PoseLandmark.LEFT_WRIST])
            qx,qy,qz,qw = vec_to_quat(ref_down, le-ls)
            data['LeftUpperArm'] = smooth_quat('lua',qx,qy,qz,qw)
            qx,qy,qz,qw = vec_to_quat(ref_down, lw-le)
            data['LeftLowerArm'] = smooth_quat('lla',qx,qy,qz,qw)

            nose    = lm_to_np(lm[mp_pose.PoseLandmark.NOSE])
            l_ear   = lm_to_np(lm[mp_pose.PoseLandmark.LEFT_EAR])
            r_ear   = lm_to_np(lm[mp_pose.PoseLandmark.RIGHT_EAR])
            mid_ear = (l_ear + r_ear) / 2
            qx,qy,qz,qw = vec_to_quat(np.array([0.,0.,1.]), nose-mid_ear)
            data['Head'] = smooth_quat('head',qx,qy,qz,qw)

        if hand_results.multi_hand_landmarks:
            for i, hand_lm in enumerate(hand_results.multi_hand_landmarks):
                label = hand_results.multi_handedness[i].classification[0].label
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(255,200,0), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(255,140,0), thickness=2))
                process_fingers(hand_lm, label, data)

        if face_results.multi_face_landmarks:
            for fl_lm in face_results.multi_face_landmarks:
                mp_draw.draw_landmarks(frame, fl_lm, mp_face.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_draw.DrawingSpec(
                        color=(255,100,200), thickness=1))
            fl = face_results.multi_face_landmarks[0].landmark
            mouth   = float(np.clip(abs(fl[13].y - fl[14].y) * 25, 0, 1))
            l_blink = float(np.clip(1.0 - abs(fl[159].y - fl[145].y) * 35, 0, 1))
            r_blink = float(np.clip(1.0 - abs(fl[386].y - fl[374].y) * 35, 0, 1))
            data['blendshapes'] = {
                'A':       smoother.smooth('mouth', mouth),
                'Blink_L': smoother.smooth('bl', l_blink),
                'Blink_R': smoother.smooth('br', r_blink),
            }

        with frame_lock:
            latest_frame = frame.copy()

        if data:
            # Debug: print what pose data we're sending
            if 'RightUpperArm' in data or 'Head' in data:
                print(f"📊 Sent pose data with keys: {list(data.keys())}")
            socketio.emit('pose', data)

    cap.release()

def generate_frames():
    global latest_frame
    frame_skip = 0
    skip_rate = 2  # Send every 2nd frame to reduce bandwidth
    
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            frame_to_send = latest_frame.copy()
        
        # Skip frames to reduce server load
        frame_skip += 1
        if frame_skip < skip_rate:
            continue
        frame_skip = 0
        
        # Compress with lower quality for faster transmission
        _, buffer = cv2.imencode('.jpg', frame_to_send,
                                 [cv2.IMWRITE_JPEG_QUALITY, 60])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()
    print("✅ Open http://localhost:5000 in your browser")
    socketio.run(app, port=5000)