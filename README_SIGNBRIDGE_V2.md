# SignBridge v2 Upgrade Files

These files upgrade your Filipino Sign Language translator while keeping your existing Flask routes and frontend connection style.

## Files included

| File | Purpose |
|---|---|
| `sign_config.py` | Shared labels, paths, model settings, sequence length, feature size |
| `sign_features.py` | New feature extractor: normalized hands, finger angles, fingertip distances, minimal pose |
| `collect_data_v2.py` | Collects dataset using the new 162-feature extractor |
| `train_model_v2.py` | Trains the upgraded BiLSTM dynamic/unified model |
| `train_static_model_v2.py` | Optional static-sign model for faster/more accurate static letters |
| `test_model_v2.py` | Flask inference app with motion segmentation and optional static/dynamic routing |
| `export_tflite_v2.py` | Optional TensorFlow Lite export for speed |

## Important compatibility note

Your old `dataset/*.npy` files from the previous system probably have 276 features because they used hands + pose + face.

The v2 model uses 162 features:

- both hands
- finger angles
- fingertip distances
- shoulders, elbows, wrists
- no face mesh

So you should recollect training data with:

```bash
python collect_data_v2.py --sign Hello --sequences 100
```

or:

```bash
python collect_data_v2.py --all --sequences 50
```

## Suggested workflow

1. Put all v2 files in the same folder as your existing project.
2. Keep your `templates/test_model.html` file as-is.
3. Recollect dataset using `collect_data_v2.py`.
4. Train the main model:

```bash
python train_model_v2.py
```

5. Optional: train the static model:

```bash
python train_static_model_v2.py
```

6. Run the upgraded app:

```bash
python test_model_v2.py
```

7. Open:

```text
http://localhost:5001
```

## Why v2 should be more accurate

The old version predicted continuously every few frames. That causes flickering, duplicate words, and unstable sentence building.

The v2 version uses a gesture state machine:

```text
IDLE → RECORDING → CLASSIFY → OUTPUT → IDLE
```

This means the model classifies after the sign is completed, not while the hand is still moving.

## What to tune first

In `test_model_v2.py`, adjust these if needed:

```python
MOTION_THRESHOLD = 0.008
IDLE_FRAMES_REQUIRED = 10
DYNAMIC_CONFIDENCE_THRESHOLD = 0.70
STATIC_CONFIDENCE_THRESHOLD = 0.75
DUPLICATE_COOLDOWN_SECONDS = 1.25
```

If it records too easily, increase `MOTION_THRESHOLD`.
If it misses small gestures, decrease `MOTION_THRESHOLD`.
If it cuts signs too early, increase `IDLE_FRAMES_REQUIRED`.
If it outputs wrong predictions, increase the confidence thresholds.

## Dataset recommendation

For best results, aim for at least:

- 100+ sequences per sign for testing
- 300–500 sequences per sign for better accuracy
- multiple people, lighting conditions, backgrounds, and distances

## Existing frontend compatibility

`test_model_v2.py` keeps these routes:

- `/`
- `/video_feed`
- `/prediction`
- `/clear`

So your existing frontend should still work if it was built around your original `test_model.py`.
