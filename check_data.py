import numpy as np
import os

<<<<<<< HEAD
SIGNS = ['pamilya','lola','lolo','mama','papa']
=======
SIGNS = ['good_morning', 'good_afternoon', 'good_evening']
>>>>>>> 2c383ee2c00e73545b0df12cd499f5186665dc52
DATA_PATH = 'dataset'

for sign in SIGNS:
    empty_frames = 0
    for seq in range(30):
        for frame in range(30):
            path = os.path.join(DATA_PATH, sign, str(seq), str(frame) + '.npy')
            data = np.load(path)
            if data[:126].sum() == 0:  # no hand landmarks detected
                empty_frames += 1
    print(f'{sign}: {empty_frames} frames with no hand data')