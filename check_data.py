import numpy as np
import os

SIGNS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
         'Ako', 'Ako ay mabuti', 'Alin', 'Ano', 'Bakit', 'Basahin', 'Because', 'Estudyante', 'Guro', 'Hello', 'Hindi masama', 'Ikaw', 'Kailan',
         'Kaklase', 'Kamusta ang buhay', 'Kamusta kana', 'Klase', 'lola', 'lolo', 'Mabuti ako', 'Mag-aral', 'Magaling ako', 'Maghintay',
         'mama', 'Masama ang aking pakiramam', 'Mula sa', 'No', 'Okay lang ako', 'Opo', 'pamilya', 'papa', 'Saan', 'Salamat po', 'Siguro',
         'Sila', 'Sino', 'Siya', 'Walang anuman']
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
