"""Build the training set: crop hand-labelled slots from parking_1920_1080.mp4 into data/empty and data/non_empty.

WARNING: deletes and recreates both data folders on every run.
"""
import shutil

import cv2

import utils

FRAME_INTERVAL = 30          # sample every 30th frame for lighting/position variety
MAX_SAMPLES_PER_SLOT = 100

# Slot numbers = index into utils.get_parking_spots_bboxes(). Labelled once by watching the video:
# these slots stay empty / occupied for the whole clip.
EMPTY_SLOTS = [
    132, 147, 164, 180, 344, 360, 377, 385, 341, 179, 131, 106, 91,
    61, 4, 89, 129, 161, 185, 201, 224, 271, 303, 319, 335, 351, 389,
    29, 12, 32, 72, 281, 280, 157, 223, 26,
]
OCCUPIED_SLOTS = [
    18, 31, 46, 76, 120, 119, 148, 163, 188, 187, 232, 229, 278, 258,
    277, 253, 295, 312, 326, 327, 370, 378, 384, 392, 391, 5, 16, 13,
    105, 145, 178, 146, 127, 160, 175, 184, 200, 199, 176, 159, 144,
    128, 17, 14, 28, 41, 56, 55, 85, 115, 116, 125, 142, 219, 221,
    67, 356, 358, 65, 243, 270, 3, 7,
]
# (filename prefix, slots, output folder) — folder names set the class order in training
CLASSES = [("empty", EMPTY_SLOTS, utils.HERE / "data" / "empty"),
           ("occupied", OCCUPIED_SLOTS, utils.HERE / "data" / "non_empty")]

mask = cv2.imread(str(utils.HERE / "mask_1920_1080.png"), cv2.IMREAD_GRAYSCALE)
cap = cv2.VideoCapture(str(utils.HERE / "parking_1920_1080.mp4"))
if mask is None or not cap.isOpened():
    raise SystemExit("Could not load mask_1920_1080.png or parking_1920_1080.mp4")
video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
if mask.shape[::-1] != video_size:
    mask = cv2.resize(mask, video_size)
slots = utils.get_parking_spots_bboxes(cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S))
print(f"Slots in mask: {len(slots)}")

for _, _, out_dir in CLASSES:
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)  # OneDrive can hold the folder briefly after rmtree

saved = {prefix: 0 for prefix, _, _ in CLASSES}
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for frame_nmr in range(0, total_frames, FRAME_INTERVAL):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nmr)
    success, frame = cap.read()
    if not success:
        break
    if frame_nmr // FRAME_INTERVAL >= MAX_SAMPLES_PER_SLOT:
        break  # every slot already has its max samples
    for prefix, slot_list, out_dir in CLASSES:
        for slot_nmr in slot_list:
            x, y, w, h = slots[slot_nmr]
            cv2.imwrite(str(out_dir / f"{prefix}_slot_{slot_nmr}_frame_{frame_nmr}.jpg"), frame[y:y + h, x:x + w])
            saved[prefix] += 1
    if frame_nmr % 300 == 0:
        print(f"frame {frame_nmr}/{total_frames}  saved {saved}")

cap.release()
print(f"Done. Saved {saved}")
