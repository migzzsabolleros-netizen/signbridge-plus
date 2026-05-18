import os
import numpy as np
from pathlib import Path

# ---- SETTINGS ----
SIGNS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
         'Ako', 'Ako ay mabuti', 'Alin', 'Ano', 'Bakit', 'Basahin', 'Because', 'Estudyante', 'Guro', 'Hello', 'Hindi masama', 'Ikaw', 'Kailan',
         'Kaklase', 'Kamusta ang buhay', 'Kamusta kana', 'Klase', 'lola', 'lolo', 'Mabuti ako', 'Mag-aral', 'Magaling ako', 'Maghintay',
         'mama', 'Masama ang aking pakiramam', 'Mula sa', 'No', 'Okay lang ako', 'Opo', 'pamilya', 'papa', 'Saan', 'Salamat po', 'Siguro',
         'Sila', 'Sino', 'Siya', 'Walang anuman']
SEQUENCE_LENGTH = 40
DATA_PATH = 'dataset'

print("Analyzing dataset for dynamic vs static gestures...\n")
print("=" * 80)

results = []

for sign in SIGNS:
    sign_path = os.path.join(DATA_PATH, sign)
    
    # Determine if sign has sequence subdirectories
    try:
        contents = os.listdir(sign_path)
        first_item = contents[0]
        first_item_path = os.path.join(sign_path, first_item)
        
        has_sequences = os.path.isdir(first_item_path)
        
        # Load samples
        variances = []
        samples_found = 0
        
        if has_sequences:
            # Has subdirectories (0/, 1/, etc.)
            for seq in range(min(5, 30)):  # Check first 5 sequences
                seq_path = os.path.join(sign_path, str(seq))
                if not os.path.exists(seq_path):
                    continue
                
                frames = []
                for frame_num in range(SEQUENCE_LENGTH):
                    path = os.path.join(seq_path, str(frame_num) + '.npy')
                    if os.path.exists(path):
                        frames.append(np.load(path))
                
                if len(frames) == SEQUENCE_LENGTH:
                    frames = np.array(frames)
                    # Calculate variance across time (how much the gesture changes frame-to-frame)
                    variance = np.var(frames)
                    variances.append(variance)
                    samples_found += 1
        else:
            # Direct frames in folder
            frames = []
            for frame_num in range(SEQUENCE_LENGTH):
                path = os.path.join(sign_path, str(frame_num) + '.npy')
                if os.path.exists(path):
                    frames.append(np.load(path))
            
            if len(frames) == SEQUENCE_LENGTH:
                frames = np.array(frames)
                variance = np.var(frames)
                variances.append(variance)
                samples_found = 1
        
        if variances:
            avg_variance = np.mean(variances)
            # Threshold: 0.33 separates letters/words (static) from phrases (dynamic)
            gesture_type = "DYNAMIC (moving)" if avg_variance > 0.33 else "STATIC (held)"
            
            results.append((sign, avg_variance, gesture_type))
            print(f"{sign:35} | Variance: {avg_variance:.6f} | {gesture_type}")
    except Exception as e:
        print(f"{sign:35} | ERROR: {e}")

print("=" * 80)
print("\nSUMMARY:")
print("-" * 80)

dynamic = [r for r in results if "DYNAMIC" in r[2]]
static = [r for r in results if "STATIC" in r[2]]

print(f"\n📊 DYNAMIC (Moving) Signs ({len(dynamic)}):")
for sign, var, gesture in dynamic:
    print(f"  - {sign}")

print(f"\n📊 STATIC (Held) Signs ({len(static)}):")
for sign, var, gesture in static:
    print(f"  - {sign}")

print("\n⚠️  RECOMMENDATION:")
print("During testing, move your hands for DYNAMIC signs and hold still for STATIC signs.")
print("If you want consistent predictions, recollect data to make ALL signs either dynamic or static.")
