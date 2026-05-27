"""
Train the SignBridge v2 dynamic/unified gesture model.

This replaces your original Conv1D-only model with a BiLSTM model and uses the
new 162-feature extractor. It keeps the same dataset folder layout:
    dataset/<SIGN>/<SEQUENCE>/<FRAME>.npy

Important:
    Old v1 276-feature data is not compatible. Recollect with collect_data_v2.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import BatchNormalization, Bidirectional, Dense, Dropout, LSTM, Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from sign_config import DATA_PATH, FEATURE_SIZE, LABELS_PATH, METADATA_PATH, MODEL_PATH, SEQUENCE_LENGTH, SIGNS
from sign_features import normalize_sequence_length


np.random.seed(42)
tf.random.set_seed(42)


def load_sequence(seq_path: Path) -> np.ndarray | None:
    frames = []
    for frame_num in range(SEQUENCE_LENGTH):
        path = seq_path / f'{frame_num}.npy'
        if not path.exists():
            return None
        arr = np.load(path).astype(np.float32)
        if arr.shape[0] != FEATURE_SIZE:
            raise ValueError(
                f"Feature size mismatch in {path}: got {arr.shape[0]}, expected {FEATURE_SIZE}. "
                "This usually means the data was collected with the old extractor. "
                "Recollect using collect_data_v2.py."
            )
        frames.append(arr)
    return np.asarray(frames, dtype=np.float32)


def load_dataset(data_path: str = DATA_PATH) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X, y = [], []
    label_map = {sign: i for i, sign in enumerate(SIGNS)}
    loaded_signs = []

    print('Loading dataset...')
    for sign in SIGNS:
        sign_path = Path(data_path) / sign
        if not sign_path.exists():
            print(f'⚠️ {sign:35} | Directory not found')
            continue

        sequence_dirs = sorted([p for p in sign_path.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))
        loaded = 0

        if sequence_dirs:
            for seq_dir in sequence_dirs:
                seq = load_sequence(seq_dir)
                if seq is not None:
                    X.append(seq)
                    y.append(label_map[sign])
                    loaded += 1
        else:
            # Backward-compatible fallback for dataset/<SIGN>/<FRAME>.npy
            seq = load_sequence(sign_path)
            if seq is not None:
                X.append(seq)
                y.append(label_map[sign])
                loaded += 1

        if loaded:
            loaded_signs.append(sign)
            print(f'✅ {sign:35} | Loaded {loaded} sequence(s)')
        else:
            print(f'⚠️ {sign:35} | No complete sequences found')

    if not X:
        raise RuntimeError('No sequences loaded. Check your dataset path and run collect_data_v2.py first.')

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    print(f'\nLoaded {len(X)} sequences from {len(set(y))} signs')
    print(f'Dataset shape: {X.shape}')
    return X, y, loaded_signs


def augment_sequence(seq: np.ndarray) -> List[np.ndarray]:
    """Safe sequence augmentations for normalized landmark features."""
    augmented = [seq]

    # Small feature noise.
    augmented.append((seq + np.random.normal(0, 0.01, seq.shape)).astype(np.float32))

    # Slight global hand-feature scaling.
    scaled = seq.copy()
    scaled[:, :144] *= np.random.uniform(0.95, 1.05)
    augmented.append(scaled.astype(np.float32))

    # Temporal speed variation.
    if np.random.random() > 0.3:
        source_len = len(seq)
        factor = np.random.uniform(0.75, 1.25)
        warped_len = max(8, int(source_len * factor))
        old = np.linspace(0, 1, source_len)
        mid = np.linspace(0, 1, warped_len)
        new = np.linspace(0, 1, SEQUENCE_LENGTH)
        f1 = interp1d(old, seq, axis=0, kind='linear', fill_value='extrapolate')
        warped = f1(mid)
        f2 = interp1d(mid, warped, axis=0, kind='linear', fill_value='extrapolate')
        augmented.append(f2(new).astype(np.float32))

    # Random missing landmark simulation.
    dropped = seq.copy()
    mask = (np.random.random(dropped.shape) > 0.05).astype(np.float32)
    dropped *= mask
    augmented.append(dropped.astype(np.float32))

    return augmented


def augment_dataset(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    print('\nApplying data augmentation...')
    X_out, y_out = [], []
    for seq, label in zip(X, y):
        for aug in augment_sequence(seq):
            X_out.append(aug)
            y_out.append(label)
    X_aug = np.asarray(X_out, dtype=np.float32)
    y_aug = np.asarray(y_out, dtype=np.int64)
    print(f'✅ Augmented dataset shape: {X_aug.shape}')
    return X_aug, y_aug


def build_model(num_classes: int) -> Sequential:
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(SEQUENCE_LENGTH, FEATURE_SIZE)),

        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.30),
        BatchNormalization(),

        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.30),
        BatchNormalization(),

        Bidirectional(LSTM(64)),
        Dropout(0.40),

        Dense(256, activation='relu'),
        Dropout(0.40),

        Dense(128, activation='relu'),
        Dropout(0.30),

        Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def main() -> None:
    X, y_raw, loaded_signs = load_dataset(DATA_PATH)
    X, y_raw = augment_dataset(X, y_raw)
    y = to_categorical(y_raw, num_classes=len(SIGNS))

    stratify = y_raw if len(set(y_raw)) > 1 and min(np.bincount(y_raw)) >= 2 else None
    X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
        X, y, y_raw, test_size=0.2, random_state=42, stratify=stratify
    )

    classes_present = np.unique(y_raw_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes_present, y=y_raw_train)
    class_weight = {int(cls): float(weight) for cls, weight in zip(classes_present, weights)}

    model = build_model(len(SIGNS))
    model.summary()

    callbacks = [
        TensorBoard(log_dir='logs_v2'),
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
    ]

    print('\nTraining SignBridge v2 model...')
    history = model.fit(
        X_train,
        y_train,
        epochs=300,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'\n✅ Test Accuracy: {accuracy * 100:.2f}%')

    model.save(MODEL_PATH)
    print(f'✅ Model saved as {MODEL_PATH}')

    with open(LABELS_PATH, 'w', encoding='utf-8') as f:
        json.dump(SIGNS, f, ensure_ascii=False, indent=2)

    metadata = {
        'model_path': MODEL_PATH,
        'labels_path': LABELS_PATH,
        'feature_size': FEATURE_SIZE,
        'sequence_length': SEQUENCE_LENGTH,
        'loaded_signs': loaded_signs,
        'num_training_sequences_after_augmentation': int(len(X)),
        'test_accuracy': float(accuracy),
    }
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

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
    plt.savefig('training_results_v2.png')
    print('✅ Training chart saved as training_results_v2.png')


if __name__ == '__main__':
    main()
