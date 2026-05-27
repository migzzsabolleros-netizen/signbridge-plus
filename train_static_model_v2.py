"""
Optional static-sign model for SignBridge v2.

Train this after collecting v2 data if you want faster/more accurate classification
for static signs. test_model_v2.py will automatically use this model if it exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from sign_config import DATA_PATH, FEATURE_SIZE, SEQUENCE_LENGTH, STATIC_LABELS_PATH, STATIC_MODEL_PATH, STATIC_SIGNS


np.random.seed(42)
tf.random.set_seed(42)


def load_static_dataset() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X, y = [], []
    labels = [s for s in STATIC_SIGNS if (Path(DATA_PATH) / s).exists()]
    label_map = {sign: i for i, sign in enumerate(labels)}

    if not labels:
        raise RuntimeError('No static sign folders found. Check DATA_PATH and STATIC_SIGNS in sign_config.py.')

    print('Loading static-sign data...')
    for sign in labels:
        sign_path = Path(DATA_PATH) / sign
        sequence_dirs = sorted([p for p in sign_path.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))
        loaded = 0

        for seq_dir in sequence_dirs:
            frames = []
            for frame_num in range(SEQUENCE_LENGTH):
                f = seq_dir / f'{frame_num}.npy'
                if not f.exists():
                    frames = []
                    break
                arr = np.load(f).astype(np.float32)
                if arr.shape[0] != FEATURE_SIZE:
                    raise ValueError(f'Feature size mismatch in {f}: got {arr.shape[0]}, expected {FEATURE_SIZE}')
                frames.append(arr)
            if not frames:
                continue

            seq = np.asarray(frames, dtype=np.float32)
            # Use a stable middle-window average, not just one frame.
            mid = len(seq) // 2
            start = max(0, mid - 3)
            end = min(len(seq), mid + 4)
            feature = np.mean(seq[start:end], axis=0)
            X.append(feature)
            y.append(label_map[sign])
            loaded += 1

        print(f'✅ {sign:20} | Loaded {loaded} static samples')

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), labels


def augment_static(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for sample, label in zip(X, y):
        Xs.append(sample)
        ys.append(label)
        Xs.append((sample + np.random.normal(0, 0.01, sample.shape)).astype(np.float32))
        ys.append(label)
        scaled = sample.copy()
        scaled[:144] *= np.random.uniform(0.95, 1.05)
        Xs.append(scaled.astype(np.float32))
        ys.append(label)
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def build_static_model(num_classes: int) -> Sequential:
    model = Sequential([
        Dense(256, activation='relu', input_shape=(FEATURE_SIZE,)),
        BatchNormalization(),
        Dropout(0.30),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.30),

        Dense(64, activation='relu'),
        Dropout(0.20),

        Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def main() -> None:
    X, y_raw, labels = load_static_dataset()
    X, y_raw = augment_static(X, y_raw)
    y = to_categorical(y_raw, num_classes=len(labels))

    stratify = y_raw if len(set(y_raw)) > 1 and min(np.bincount(y_raw)) >= 2 else None
    X_train, X_test, y_train, y_test, y_raw_train, _ = train_test_split(
        X, y, y_raw, test_size=0.2, random_state=42, stratify=stratify
    )

    classes_present = np.unique(y_raw_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes_present, y=y_raw_train)
    class_weight = {int(cls): float(weight) for cls, weight in zip(classes_present, weights)}

    model = build_static_model(len(labels))
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, min_lr=1e-6, verbose=1),
        ModelCheckpoint(STATIC_MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
    ]

    model.fit(
        X_train,
        y_train,
        epochs=250,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f'\n✅ Static model test accuracy: {acc * 100:.2f}%')
    model.save(STATIC_MODEL_PATH)

    with open(STATIC_LABELS_PATH, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print(f'✅ Saved {STATIC_MODEL_PATH}')
    print(f'✅ Saved {STATIC_LABELS_PATH}')


if __name__ == '__main__':
    main()
