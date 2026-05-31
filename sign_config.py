"""
Shared configuration for SignBridge v2.
Keep this file imported by collection, training, and testing so labels stay aligned.
"""

# Combined master list of all signs you intend to collect or train
SIGNS = [
    'Ako', 'Ikaw', 'Magandang umaga', 'Magandang tanghali', 'Oo', 'Hindi', 'Salamat po', 'Walang anuman', 'Estudyante', 'Guro', 'Klase',
    'Mag-aaral', 'Magbabasa'
]

# Signs that are usually static.
STATIC_SIGNS = [
    'Ako', 'Ikaw'
]

# Your 7 specific dynamic target signs
DYNAMIC_SIGNS = [
    'Magandang umaga', 'Magandang tanghali', 'Oo', 'Hindi', 'Salamat po', 'Walang anuman', 'Estudyante', 'Guro', 'Klase',
    'Mag-aaral', 'Magbabasa'
]

SEQUENCE_LENGTH = 40
DATA_PATH = 'dataset'

FEATURE_SIZE = 162

MODEL_PATH = 'signbridge_model_v2.h5'
STATIC_MODEL_PATH = 'signbridge_static_model_v2.h5'
LABELS_PATH = 'signbridge_labels_v2.json'
STATIC_LABELS_PATH = 'signbridge_static_labels_v2.json'
METADATA_PATH = 'signbridge_metadata_v2.json'