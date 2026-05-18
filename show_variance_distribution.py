import os
import numpy as np

DATA_PATH = 'dataset'

SIGNS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
         'Ako', 'Ako ay mabuti', 'Alin', 'Ano', 'Bakit', 'Basahin', 'Because', 'Estudyante', 'Guro', 'Hello', 'Hindi masama', 'Ikaw', 'Kailan',
         'Kaklase', 'Kamusta ang buhay', 'Kamusta kana', 'Klase', 'lola', 'lolo', 'Mabuti ako', 'Mag-aral', 'Magaling ako', 'Maghintay',
         'mama', 'Masama ang aking pakiramam', 'Mula sa', 'No', 'Okay lang ako', 'Opo', 'pamilya', 'papa', 'Saan', 'Salamat po', 'Siguro',
         'Sila', 'Sino', 'Siya', 'Walang anuman']
SEQUENCE_LENGTH = 40

print("Analyzing variance distribution...\n")

# Collect all variances
all_variances = {}

for sign in SIGNS:
    sign_path = os.path.join(DATA_PATH, sign)
    
    if not os.path.exists(sign_path):
        continue
    
    try:
        contents = sorted(os.listdir(sign_path))
        if not contents:
            continue
        
        first_item_path = os.path.join(sign_path, contents[0])
        has_sequences = os.path.isdir(first_item_path)
        
        variances = []
        
        if has_sequences:
            for seq_name in contents:
                seq_path = os.path.join(sign_path, seq_name)
                if not os.path.isdir(seq_path):
                    continue
                
                frames = []
                for frame_num in range(SEQUENCE_LENGTH):
                    path = os.path.join(seq_path, str(frame_num) + '.npy')
                    if os.path.exists(path):
                        frames.append(np.load(path))
                
                if len(frames) == SEQUENCE_LENGTH:
                    frames = np.array(frames)
                    variance = np.var(frames)
                    variances.append(variance)
        else:
            frames = []
            for frame_num in range(SEQUENCE_LENGTH):
                path = os.path.join(sign_path, str(frame_num) + '.npy')
                if os.path.exists(path):
                    frames.append(np.load(path))
            
            if len(frames) == SEQUENCE_LENGTH:
                frames = np.array(frames)
                variance = np.var(frames)
                variances.append(variance)
        
        if variances:
            avg_variance = np.mean(variances)
            all_variances[sign] = avg_variance
            
    except Exception as e:
        continue

# Sort by variance
sorted_signs = sorted(all_variances.items(), key=lambda x: x[1])

print("=" * 70)
print("VARIANCE BY SIGN (sorted low to high):")
print("=" * 70)

for sign, variance in sorted_signs:
    bar_length = int(variance * 500)  # Scale for visualization
    bar = "█" * bar_length
    print(f"{sign:35} | {variance:.6f} | {bar}")

print("\n" + "=" * 70)
print("STATISTICS:")
print("=" * 70)

variances_list = list(all_variances.values())
print(f"Min variance:  {min(variances_list):.6f}")
print(f"Max variance:  {max(variances_list):.6f}")
print(f"Mean variance: {np.mean(variances_list):.6f}")
print(f"Median variance: {np.median(variances_list):.6f}")

print("\n" + "=" * 70)
print("RECOMMENDATION:")
print("=" * 70)
print(f"\nLook at the variance values above.")
print(f"Set a threshold between static and dynamic gestures.")
print(f"For example:")
print(f"  - If letters have variance < 0.003, use threshold 0.003")
print(f"  - If phrases have variance > 0.01, use threshold 0.01")
print(f"\nCurrent threshold in analyze_dataset.py: 0.005")
