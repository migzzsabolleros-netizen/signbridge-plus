import os

DATA_PATH = 'dataset'

print("Checking which signs have actual data...\n")

signs_with_data = []

for sign in sorted(os.listdir(DATA_PATH)):
    sign_path = os.path.join(DATA_PATH, sign)
    if not os.path.isdir(sign_path):
        continue
    
    contents = os.listdir(sign_path)
    
    # Check if has data
    has_data = False
    
    # Check for direct .npy files
    npy_files = [f for f in contents if f.endswith('.npy')]
    if npy_files:
        has_data = True
        print(f"✅ {sign:35} | {len(npy_files):3} .npy files")
    
    # Check for subdirectories with .npy files
    subdirs = [d for d in contents if os.path.isdir(os.path.join(sign_path, d))]
    if subdirs and not npy_files:
        # Check if subdirs have .npy files
        sample_dir = os.path.join(sign_path, subdirs[0])
        sample_files = [f for f in os.listdir(sample_dir) if f.endswith('.npy')]
        if sample_files:
            has_data = True
            print(f"✅ {sign:35} | {len(subdirs):3} sequences x {len(sample_files)} frames")
    
    if has_data:
        signs_with_data.append(sign)

print(f"\n{'='*70}")
print(f"Total signs with data: {len(signs_with_data)}")
print(f"\nSigns to use in training:")
print(f"SIGNS = {signs_with_data}")
