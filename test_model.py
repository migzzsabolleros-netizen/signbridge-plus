from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import threading
from collections import Counter
import math

app = Flask(__name__)

# ---- SETTINGS ----
SIGNS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
         'Ako', 'Ako ay mabuti', 'Alin', 'Ano', 'Bakit', 'Basahin', 'Because', 'Estudyante', 'Guro', 'Hello', 'Hindi masama', 'Ikaw', 'Kailan',
         'Kaklase', 'Kamusta ang buhay', 'Kamusta kana', 'Klase', 'lola', 'lolo', 'Mabuti ako', 'Mag-aral', 'Magaling ako', 'Maghintay',
         'mama', 'Masama ang aking pakiramdam', 'Mula sa', 'No', 'Okay lang ako', 'Opo', 'pamilya', 'papa', 'Saan', 'Salamat po', 'Siguro',
         'Sila', 'Sino', 'Siya', 'Walang anuman']
SEQUENCE_LENGTH = 40
FAST_CONFIRM_THRESHOLD = 0.75   # Immediately confirm if we get 75%+ confidence
THRESHOLD_NEW_SIGN = 0.60       # Confidence for NEW sign (higher = stable)
THRESHOLD_SAME_SIGN = 0.50      # Confidence for SAME sign (hysteresis)
MIN_HAND_DETECTION_CONFIDENCE = 0.65  # Ignore predictions if hand confidence is too low
STABILIZATION_FRAMES = 18       # ~0.6 seconds
TEMPORAL_SMOOTHING = 4          # Smooth predictions over N frames
MOTION_THRESHOLD = 0.015        # Threshold for detecting motion (hand movement)
KEYPOINT_SMOOTHING_ALPHA = 0.6  # Exponential smoothing for landmarks

# ---- Load Model ----
print("Loading model...")
model = tf.keras.models.load_model('signbridge_model.keras')
print("✅ Model loaded!")
print(f"Model output size: {model.output_shape}")
print(f"Number of SIGNS: {len(SIGNS)}")

# ---- MediaPipe ----
mp_hands = mp.solutions.hands
mp_face  = mp.solutions.face_mesh
mp_pose  = mp.solutions.pose
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
face  = mp_face.FaceMesh(min_detection_confidence=0.7)
pose  = mp_pose.Pose(min_detection_confidence=0.7)

# ---- Prediction Buffer for Stabilization ----
class PredictionBuffer:
    """Manages predictions over a window for stable detection"""
    def __init__(self, buffer_size=STABILIZATION_FRAMES):
        self.buffer_size = buffer_size
        self.predictions = []      # List of (sign_index, confidence) tuples
        self.sign_votes = Counter() 
        self.confidence_sum = {}
        self.current_confirmed_sign = ''
        self.consecutive_count = {}  # Track consecutive same predictions
    
    def add_prediction(self, sign_idx, confidence):
        """Add a new prediction to the buffer"""
        self.predictions.append((sign_idx, confidence))
        
        # Keep buffer size limited
        if len(self.predictions) > self.buffer_size:
            old_sign_idx, _ = self.predictions.pop(0)
            self.sign_votes[old_sign_idx] -= 1
        
        # Update votes and confidence
        self.sign_votes[sign_idx] += 1
        if sign_idx not in self.confidence_sum:
            self.confidence_sum[sign_idx] = 0.0
        self.confidence_sum[sign_idx] += confidence
        
        # Track consecutive predictions
        if not self.consecutive_count or list(self.consecutive_count.keys())[0] != sign_idx:
            self.consecutive_count = {sign_idx: 1}
        else:
            self.consecutive_count[sign_idx] += 1
    
    def get_consensus(self):
        """Get the consensus prediction with voting"""
        if not self.predictions:
            return -1, 0.0, 0.0
        
        if not self.sign_votes:
            return -1, 0.0, 0.0
        
        most_common_sign_idx = self.sign_votes.most_common(1)[0][0]
        vote_count = self.sign_votes[most_common_sign_idx]
        avg_confidence = self.confidence_sum[most_common_sign_idx] / vote_count
        
        # Consistency = how dominant this prediction is
        consistency_ratio = vote_count / len(self.predictions)
        
        return most_common_sign_idx, avg_confidence, consistency_ratio
    
    def has_enough_data(self):
        """Check if buffer has enough predictions for consensus"""
        return len(self.predictions) >= max(3, self.buffer_size // 4)
    
    def reset(self):
        """Reset the buffer"""
        self.predictions = []
        self.sign_votes.clear()
        self.confidence_sum.clear()
        self.consecutive_count.clear()

class KeypointSmoother:
    """Smooth MediaPipe landmarks to reduce noise"""
    def __init__(self, alpha=KEYPOINT_SMOOTHING_ALPHA):
        self.alpha = alpha
        self.prev_keypoints = None
    
    def smooth(self, keypoints):
        """Apply exponential moving average to keypoints"""
        if self.prev_keypoints is None:
            self.prev_keypoints = keypoints.copy()
            return keypoints
        
        smoothed = self.alpha * keypoints + (1 - self.alpha) * self.prev_keypoints
        self.prev_keypoints = smoothed.copy()
        return smoothed

def calculate_motion_magnitude(prev_keypoints, curr_keypoints):
    """Calculate if hands moved significantly"""
    if prev_keypoints is None or curr_keypoints is None:
        return 0.0
    
    # Extract hand keypoints only (first 126 values)
    prev_hands = prev_keypoints[:126]
    curr_hands = curr_keypoints[:126]
    
    # Calculate average distance moved
    diff = np.abs(curr_hands - prev_hands)
    motion = np.mean(diff)
    return motion

# ---- Shared state ----
latest_frame     = None
frame_lock       = threading.Lock()
prediction_state = {'sign': '', 'confidence': 0.0, 'sentence': [], 'history': []}
sequence         = []
prediction_history = []
prediction_buffer = PredictionBuffer()
keypoint_smoother = KeypointSmoother()
last_confirmed_sign = ''
prev_keypoints = None


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
    global latest_frame, sequence, prediction_history, prediction_buffer, last_confirmed_sign, prev_keypoints
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
        
        # Smooth keypoints to reduce noise
        smoothed_keypoints = keypoint_smoother.smooth(keypoints)
        
        # Track motion
        motion = calculate_motion_magnitude(prev_keypoints, smoothed_keypoints)
        prev_keypoints = smoothed_keypoints.copy()
        
        sequence.append(smoothed_keypoints)
        sequence = sequence[-SEQUENCE_LENGTH:]

        # Check hand detection quality
        hand_confidence = 0.0
        if hand_res.multi_hand_landmarks:
            # Get average confidence of all detected hands
            confidences = [h.classification[0].score for h in hand_res.multi_handedness]
            hand_confidence = np.mean(confidences) if confidences else 0.0

        # Predict every 3rd frame
        if len(sequence) == SEQUENCE_LENGTH and frame_count % 3 == 0:
            # Only predict if hands are detected AND hand confidence is good
            if not hand_res.multi_hand_landmarks or hand_confidence < MIN_HAND_DETECTION_CONFIDENCE:
                prediction_buffer.reset()
                sign = ''
                confidence = 0.0
                if hand_res.multi_hand_landmarks:
                    print(f"⚠️  Hand detection quality too low: {hand_confidence:.2%} (need >= {MIN_HAND_DETECTION_CONFIDENCE:.2%})")
            else:
                try:
                    input_data = np.array(sequence)
                    
                    # Validate data
                    if np.any(np.isnan(input_data)):
                        prediction_buffer.reset()
                        sign = ''
                        confidence = 0.0
                    else:
                        # Apply same preprocessing as training
                        input_data = np.expand_dims(input_data, axis=0)
                        input_data[:, :, 0:126] *= 4.0      # Hand keypoints: 4x weight
                        input_data[:, :, 258:276] *= 0.05   # Face keypoints: 0.05x weight
                        
                        prediction = model.predict(input_data, verbose=0)[0]
                        
                        # Validate prediction matches SIGNS count
                        if len(prediction) != len(SIGNS):
                            print(f"❌ Model output size mismatch: {len(prediction)} vs {len(SIGNS)} SIGNS")
                            prediction_buffer.reset()
                            sign = ''
                            confidence = 0.0
                        else:
                            # Add to history for temporal smoothing
                            prediction_history.append(prediction.copy())
                            prediction_history = prediction_history[-TEMPORAL_SMOOTHING:]
                            
                            # Compute smoothed prediction using weighted average
                            if len(prediction_history) > 1:
                                # Weight recent predictions more heavily
                                weights = np.linspace(0.5, 1.5, len(prediction_history))
                                weights /= weights.sum()
                                smoothed_prediction = np.average(prediction_history, axis=0, weights=weights)
                            else:
                                smoothed_prediction = np.mean(prediction_history, axis=0)
                            
                            top_idx = np.argmax(smoothed_prediction)
                            top_confidence = float(smoothed_prediction[top_idx])
                            
                            # Get top 3 predictions
                            top_3_indices = np.argsort(smoothed_prediction)[-3:][::-1]
                            print(f"\n📊 Frame {frame_count} - Top 3 predictions:")
                            for rank, idx in enumerate(top_3_indices, 1):
                                if idx < len(SIGNS):
                                    conf = float(smoothed_prediction[idx])
                                    print(f"   {rank}. {SIGNS[idx]:35} ({conf:.2%})")
                            
                            # Add to prediction buffer for stabilization
                            prediction_buffer.add_prediction(top_idx, top_confidence)
                            
                            # ===== FAST-TRACK: High confidence = immediate confirmation =====
                            if top_confidence >= FAST_CONFIRM_THRESHOLD:
                                sign = SIGNS[top_idx] if top_idx < len(SIGNS) else ''
                                confidence = top_confidence
                                print(f"⚡ FAST CONFIRMED: {sign} ({confidence:.2%}) - HIGH CONFIDENCE!")
                                prediction_buffer.reset()  # Reset for next sign
                            
                            # ===== STANDARD: Wait for consensus =====
                            elif prediction_buffer.has_enough_data():
                                consensus_idx, avg_conf, consistency = prediction_buffer.get_consensus()
                                
                                # Apply hysteresis: different thresholds for new vs same sign
                                if last_confirmed_sign == SIGNS[consensus_idx]:
                                    threshold = THRESHOLD_SAME_SIGN
                                else:
                                    threshold = THRESHOLD_NEW_SIGN
                                
                                # Decision logic with multiple criteria
                                if (consensus_idx < len(SIGNS) and 
                                    avg_conf >= threshold and 
                                    consistency >= 0.35):
                                    sign = SIGNS[consensus_idx]
                                    confidence = avg_conf
                                    print(f"✅ CONFIRMED: {sign} ({confidence:.2%}) [Consistency: {consistency:.0%}]")
                                    prediction_buffer.reset()  # Reset for next sign
                                else:
                                    sign = ''
                                    confidence = 0.0
                                    reason = []
                                    if consensus_idx >= len(SIGNS):
                                        reason.append(f"Invalid index {consensus_idx}")
                                    if avg_conf < threshold:
                                        reason.append(f"Conf {avg_conf:.2%} < {threshold:.2%}")
                                    if consistency < 0.35:
                                        reason.append(f"Consistency {consistency:.0%}")
                                    print(f"⚠️  Not confident: {', '.join(reason)}")
                            else:
                                # Still building buffer
                                sign = ''
                                confidence = 0.0
                                progress = len(prediction_buffer.predictions) / STABILIZATION_FRAMES
                                print(f"🔄 Buffering... {progress:.0%} | Top: {SIGNS[top_idx]} ({top_confidence:.2%})")
                            
                            
                            # Debug: Show keypoint info
                            hand_count = len(hand_res.multi_hand_landmarks) if hand_res.multi_hand_landmarks else 0
                            pose_detected = "Yes" if pose_res.pose_landmarks else "No"
                            print(f"   Hands: {hand_count} | Pose: {pose_detected} | Buffer: {len(prediction_buffer.predictions)}/{STABILIZATION_FRAMES}")
                
                except Exception as e:
                    print(f"❌ Prediction error: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    sign       = ''
                    confidence = 0.0

            # Update confirmed sign only when we have a valid detection
            if sign:
                last_confirmed_sign = sign
                if (not prediction_state['sentence'] or
                    prediction_state['sentence'][-1] != sign):
                    prediction_state['sentence'].append(sign)
                    prediction_state['history'].insert(0, {
                        'sign': sign,
                        'confidence': round(confidence * 100, 1)
                    })
                    prediction_state['history'] = prediction_state['history'][:10]
                    print(f"📝 Added to sentence: {sign}")
                    # Reset buffer after confirming a sign for next detection
                    prediction_buffer.reset()

            prediction_state['sign']       = sign
            prediction_state['confidence'] = round(confidence * 100, 1)

        # Draw overlay
        sign       = prediction_state['sign']
        confidence = prediction_state['confidence']
        
        # Status indicator
        if prediction_buffer.predictions:
            status = f"Stabilizing... {len(prediction_buffer.predictions)}/{STABILIZATION_FRAMES}"
            status_color = (0, 150, 255)
        elif sign:
            status = f"{sign.upper()} ({confidence:.0f}%)"
            status_color = (0, 255, 100)
        else:
            status = "Detecting..."
            status_color = (100, 100, 255)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0,0,0), -1)
        cv2.putText(frame, status, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, status_color, 2)

        # Confidence bar
        bar_w = int((frame.shape[1] - 20) * min(confidence / 100, 1.0))
        cv2.rectangle(frame, (10, 50), (frame.shape[1]-10, 60), (50,50,50), -1)
        cv2.rectangle(frame, (10, 50), (10 + bar_w, 60), status_color, -1)

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