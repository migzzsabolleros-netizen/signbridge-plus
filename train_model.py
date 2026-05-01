import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import matplotlib.pyplot as plt

# ---- SETTINGS (must match collect_data.py) ----
SIGNS = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q']
SEQUENCES = 30
SEQUENCE_LENGTH = 30
DATA_PATH = 'dataset'

# ---- LOAD DATA ----
print('Loading dataset...')
label_map = {sign: i for i, sign in enumerate(SIGNS)}

X, y = [], []
for sign in SIGNS:
    for seq in range(SEQUENCES):
        frames = []
        for frame_num in range(SEQUENCE_LENGTH):
            path = os.path.join(DATA_PATH, sign, str(seq), str(frame_num) + '.npy')
            frames.append(np.load(path))
        X.append(frames)
        y.append(label_map[sign])

X = np.array(X)
y = to_categorical(y, num_classes=len(SIGNS))
print(f'Dataset shape: {X.shape}')  # should be (150, 30, 1662)

# ---- SPLIT ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- BUILD MODEL ----
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, 1662)),
    Dropout(0.2),
    LSTM(128, return_sequences=True, activation='relu'),
    Dropout(0.2),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(len(SIGNS), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---- TRAIN ----
callbacks = [
    TensorBoard(log_dir='logs'),
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
]

print('\nTraining...')
history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=callbacks
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
