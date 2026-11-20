import pickle
from pathlib import Path

import cv2

HERE = Path(__file__).parent
EMPTY = True
NOT_EMPTY = False

# Written by train.py: {'model', 'scaler', 'model_type'}
with open(HERE / "svm_model.pkl", "rb") as f:
    _PACKAGE = pickle.load(f)
MODEL = _PACKAGE["model"]
SCALER = _PACKAGE["scaler"]


def empty_or_not(spot_bgr):
    # Must match training preprocessing: cv2 resize to 15x15 BGR, flatten to 675 features, 0-255 range
    flat = cv2.resize(spot_bgr, (15, 15)).flatten().reshape(1, -1)
    if SCALER is not None:
        flat = SCALER.transform(flat)
    # Class 0 = "empty" (training sorts category folders: empty, non_empty)
    return EMPTY if MODEL.predict(flat)[0] == 0 else NOT_EMPTY


def get_parking_spots_bboxes(connected_components):
    """(x, y, w, h) per mask component. List index = component label - 1 (the slot numbers crop_cars.py uses)."""
    total, _, stats, _ = connected_components
    return [(int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
             int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
            for i in range(1, total)]
