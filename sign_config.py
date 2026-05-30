"""
Shared configuration for SignBridge v2.
Keep this file imported by collection, training, and testing so labels stay aligned.
"""

SIGNS = [
    

]

# Signs that are usually static. You can edit this after observing your own FSL dataset.
# In many sign-language alphabets, J and Z are dynamic, so they are excluded here.
STATIC_SIGNS = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y'
]

DYNAMIC_SIGNS = ['z', 'Magandang umaga', 'Magandang tanghali', 'Oo', 'Hindi', 'Salamat po', 'Walang anuman']

SEQUENCE_LENGTH = 40
DATA_PATH = 'dataset'

# New feature vector:
# 2 hands * 21 landmarks * 3 coords = 126
# 2 hands * 5 finger bend angles = 10
# 2 hands * 4 fingertip-neighbor distances = 8
# 6 upper-body pose points * 3 coords = 18
# Total = 162
FEATURE_SIZE = 162

MODEL_PATH = 'signbridge_model_v2.h5'
STATIC_MODEL_PATH = 'signbridge_static_model_v2.h5'
LABELS_PATH = 'signbridge_labels_v2.json'
STATIC_LABELS_PATH = 'signbridge_static_labels_v2.json'
METADATA_PATH = 'signbridge_metadata_v2.json'
