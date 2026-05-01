import cv2
import mediapipe as mp
import numpy as np
from pythonosc import udp_client
import math

# ---- OSC Client ----
client = udp_client.SimpleUDPClient("127.0.0.1", 39539)

# ---- MediaPipe ----
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
face = mp_face.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.6
)

# ---- Smoothing filter ----
class Smoother:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.prev = {}

    def smooth(self, key, value):
        if key not in self.prev:
            self.prev[key] = value
        self.prev[key] = self.alpha * value + (1 - self.alpha) * self.prev[key]
        return self.prev[key]

smoother = Smoother(alpha=0.25)  # lower = smoother but more lag

def vec_to_quat(v1, v2):
    """Rotation quaternion from vector v1 to vector v2"""
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
    return (axis[0]*s, axis[1]*s, axis[2]*s, math.cos(angle / 2))

def send_bone(name, qx, qy, qz, qw):
    qx = smoother.smooth(name+'x', qx)
    qy = smoother.smooth(name+'y', qy)
    qz = smoother.smooth(name+'z', qz)
    qw = smoother.smooth(name+'w', qw)
    client.send_message("/VMC/Ext/Bone/Pos",
        [name, 0.0, 0.0, 0.0, float(qx), float(qy), float(qz), float(qw)])

def send_blend(name, value):
    value = smoother.smooth('blend_'+name, value)
    client.send_message("/VMC/Ext/Blend/Val", [name, float(np.clip(value, 0.0, 1.0))])

def send_time():
    client.send_message("/VMC/Ext/Blend/Apply", [])
    client.send_message("/VMC/Ext/OK", [1])

def lm_to_np(lm):
    return np.array([lm.x, lm.y, -lm.z])

def process_pose(landmarks):
    lm = landmarks.landmark
    ref_down = np.array([0.0, -1.0, 0.0])
    ref_fwd  = np.array([0.0,  0.0, 1.0])

    # ---- RIGHT ARM ----
    rs = lm_to_np(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER])
    re = lm_to_np(lm[mp_pose.PoseLandmark.RIGHT_ELBOW])
    rw = lm_to_np(lm[mp_pose.PoseLandmark.RIGHT_WRIST])

    r_upper_dir = re - rs
    r_lower_dir = rw - re

    qx,qy,qz,qw = vec_to_quat(ref_down, r_upper_dir)
    send_bone("RightUpperArm", qx, qy, qz, qw)

    qx,qy,qz,qw = vec_to_quat(ref_down, r_lower_dir)
    send_bone("RightLowerArm", qx, qy, qz, qw)

    # ---- LEFT ARM ----
    ls = lm_to_np(lm[mp_pose.PoseLandmark.LEFT_SHOULDER])
    le = lm_to_np(lm[mp_pose.PoseLandmark.LEFT_ELBOW])
    lw = lm_to_np(lm[mp_pose.PoseLandmark.LEFT_WRIST])

    l_upper_dir = le - ls
    l_lower_dir = lw - le

    qx,qy,qz,qw = vec_to_quat(ref_down, l_upper_dir)
    send_bone("LeftUpperArm", qx, qy, qz, qw)

    qx,qy,qz,qw = vec_to_quat(ref_down, l_lower_dir)
    send_bone("LeftLowerArm", qx, qy, qz, qw)

    # ---- SPINE ----
    lh = lm_to_np(lm[mp_pose.PoseLandmark.LEFT_HIP])
    rh = lm_to_np(lm[mp_pose.PoseLandmark.RIGHT_HIP])
    spine_dir = (ls + rs) / 2 - (lh + rh) / 2

    qx,qy,qz,qw = vec_to_quat(ref_fwd, spine_dir)
    send_bone("Spine", qx*0.3, qy*0.3, qz*0.3, qw)

def process_face(landmarks):
    lm = landmarks.landmark

    # Mouth
    mouth = abs(lm[13].y - lm[14].y) * 25
    send_blend("A", mouth)

    # Blink
    l_blink = 1.0 - min(abs(lm[159].y - lm[145].y) * 35, 1.0)
    r_blink = 1.0 - min(abs(lm[386].y - lm[374].y) * 35, 1.0)
    send_blend("Blink_L", l_blink)
    send_blend("Blink_R", r_blink)

    # Head tilt (using nose tip vs chin)
    nose  = lm[1]
    chin  = lm[152]
    tilt  = (nose.x - chin.x) * 0.5
    send_bone("Head", float(tilt*0.2), 0.0, 0.0, 1.0)

# ---- MAIN LOOP ----
cap = cv2.VideoCapture(1)
print("✅ VRoid controller running!")
print("Make sure VSeeFace VMC Receiver is ON at port 39539")
print("Press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False

    pose_results = pose.process(rgb)
    face_results = face.process(rgb)

    rgb.flags.writeable = True
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if pose_results.pose_landmarks:
        mp_draw.draw_landmarks(frame, pose_results.pose_landmarks,
                               mp_pose.POSE_CONNECTIONS)
        process_pose(pose_results.pose_landmarks)

    if face_results.multi_face_landmarks:
        process_face(face_results.multi_face_landmarks[0])

    send_time()

    cv2.rectangle(frame, (0,0), (frame.shape[1], 35), (0,0,0), -1)
    cv2.putText(frame, 'VRoid Controller | Q to quit',
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,200), 1)

    cv2.imshow('VRoid Controller', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()