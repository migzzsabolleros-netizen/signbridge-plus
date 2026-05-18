import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ---- SETTINGS (must match collect_data.py) ----
SIGNS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
         'Ako', 'Ako ay mabuti', 'Alin', 'Ano', 'Bakit', 'Basahin', 'Because', 'Estudyante', 'Guro', 'Hello', 'Hindi masama', 'Ikaw', 'Kailan',
         'Kaklase', 'Kamusta ang buhay', 'Kamusta kana', 'Klase', 'lola', 'lolo', 'Mabuti ako', 'Mag-aral', 'Magaling ako', 'Maghintay',
         'mama', 'Masama ang aking pakiramdam', 'Mula sa', 'No', 'Okay lang ako', 'Opo', 'pamilya', 'papa', 'Saan', 'Salamat po', 'Siguro',
         'Sila', 'Sino', 'Siya', 'Walang anuman']
SEQUENCES = 30
SEQUENCE_LENGTH = 40
DATA_PATH = 'dataset'

# ---- LOAD DATA ----
print('Loading dataset...')
label_map = {sign: i for i, sign in enumerate(SIGNS)}

X, y = [], []
for sign in SIGNS:
    sign_path = os.path.join(DATA_PATH, sign)
    
    if not os.path.exists(sign_path):
        print(f"⚠️ {sign}: Directory not found")
        continue
    
    try:
        contents = sorted(os.listdir(sign_path))
        if not contents:
            print(f"⚠️ {sign}: Empty directory")
            continue
        
        # Check for sequence subdirectories (0, 1, 2, etc.) 
        sequence_dirs = [d for d in contents if os.path.isdir(os.path.join(sign_path, d)) and d.isdigit()]
        has_sequences = len(sequence_dirs) > 0
        
        sequences_loaded = 0
        
        if has_sequences:
            # Has subdirectories (like 0, 1, 2, etc. with frame files inside)
            for seq_name in sequence_dirs:
                seq_path = os.path.join(sign_path, seq_name)
                if not os.path.isdir(seq_path):
                    continue
                
                frames = []
                for frame_num in range(SEQUENCE_LENGTH):
                    path = os.path.join(seq_path, str(frame_num) + '.npy')
                    if os.path.exists(path):
                        frames.append(np.load(path))
                    else:
                        break
                
                if len(frames) == SEQUENCE_LENGTH:
                    X.append(frames)
                    y.append(label_map[sign])
                    sequences_loaded += 1
        else:
            # Direct frames in folder (like numeric signs '0', '1', etc.)
            frames = []
            for frame_num in range(SEQUENCE_LENGTH):
                path = os.path.join(sign_path, str(frame_num) + '.npy')
                if os.path.exists(path):
                    frames.append(np.load(path))
                else:
                    break
            
            if len(frames) == SEQUENCE_LENGTH:
                X.append(frames)
                y.append(label_map[sign])
                sequences_loaded = 1
        
        if sequences_loaded > 0:
            print(f"✅ {sign:35} | Loaded {sequences_loaded} sequence(s)")
        else:
            print(f"⚠️ {sign:35} | No complete sequences found")
    
    except Exception as e:
        print(f"❌ {sign:35} | Error: {type(e).__name__}: {e}")
        continue

print(f"\nLoaded {len(X)} sequences from {len(set(y))} signs")
if len(X) == 0:
    print("❌ ERROR: No sequences loaded! Check dataset structure.")
    exit(1)

X = np.array(X)
y = to_categorical(y, num_classes=len(SIGNS))
print(f'Dataset shape: {X.shape}')

# ---- DATA AUGMENTATION ----
print('\nApplying data augmentation...')
X_augmented = [X]

# Augmentation 1: Small random noise
X_noise = X + np.random.normal(0, 0.02, X.shape)
X_augmented.append(X_noise)

# Augmentation 2: Small random rotations (rotate keypoints slightly)
X_rotated = X.copy()
for i in range(len(X_rotated)):
    # Add small random variations to simulate different angles
    rotation_factor = np.random.uniform(0.95, 1.05)
    X_rotated[i] *= rotation_factor
X_augmented.append(X_rotated)

# Augmentation 3: Temporal scaling (speed variation - compress/expand time)
X_temporal = X.copy()
for i in range(len(X_temporal)):
    if np.random.random() > 0.5:
        # Speed up: sample frames
        step = 2
        indices = np.arange(0, SEQUENCE_LENGTH, step)
        indices = np.append(indices, SEQUENCE_LENGTH - 1)
        indices = np.unique(indices)
        
        fast = X_temporal[i][indices]
        # Interpolate back to original length
        x_old = np.linspace(0, 1, len(fast))
        x_new = np.linspace(0, 1, SEQUENCE_LENGTH)
        f = interp1d(x_old, fast.T, kind='linear', fill_value='extrapolate')
        X_temporal[i] = f(x_new).T
X_augmented.append(X_temporal)

# Augmentation 4: Dropout (missing keypoints simulation)
X_dropout = X.copy()
for i in range(len(X_dropout)):
    mask = np.random.random(X_dropout[i].shape) > 0.1  # 10% dropout
    X_dropout[i] = X_dropout[i] * mask
X_augmented.append(X_dropout)

X = np.vstack(X_augmented)
y = np.vstack([y] * len(X_augmented))

print(f'✅ Augmented dataset shape: {X.shape}')

# ---- WEIGHT KEYPOINTS: Prioritize hands over faces ----
# Hand keypoints (0-125): left hand (0-62) + right hand (63-125) -> 4x weight
# Pose keypoints (126-257): normal weight (1x)
# Face keypoints (258-275): minimal weight (0.05x)
X[:, :, 0:126] *= 4.0      # Hand keypoints: 4x weight
X[:, :, 258:276] *= 0.05   # Face keypoints: 0.05x weight (20x less relevant)
print('✅ Applied keypoint weighting: Hands 4x, Face 0.05x')

# ---- SPLIT ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- BUILD MODEL (1D CNN - better for gesture recognition) ----
model = Sequential([
    # Conv block 1
    Conv1D(64, 3, activation='relu', input_shape=(SEQUENCE_LENGTH, 276), padding='same', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.4),
    
    # Conv block 2
    Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.4),
    
    # Conv block 3
    Conv1D(256, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Conv1D(256, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.4),
    
    # Dense layers
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.4),
    Dense(len(SIGNS), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---- TRAIN ----
callbacks = [
    TensorBoard(log_dir='logs'),
    EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
]

print('\nTraining...')
history = model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# ---- EVALUATE ----
loss, accuracy = model.evaluate(X_test, y_test)
print(f'\n✅ Test Accuracy: {accuracy * 100:.2f}%')

# ---- SAVE MODEL ----
model.save('signbridge_model.keras')
print('✅ Model saved as signbridge_model.keras')

# ---- PLOT ----
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig('training_results.png')
plt.show()
print('✅ Training chart saved as training_results.png')
