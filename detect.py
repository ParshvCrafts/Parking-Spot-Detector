import cv2
import numpy as np

import utils

video = cv2.VideoCapture(str(utils.HERE / "parking_1920_1080.mp4"))
mask = cv2.imread(str(utils.HERE / "mask_1920_1080.png"), 0)
spots = utils.get_parking_spots_bboxes(cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S))

STEP = 30            # classify every STEP frames
DIFF_RATIO = 0.4     # re-classify spots whose change is > 40% of this step's largest change
DISPLAY_SCALE = 0.6  # processing stays full-res; only the window is scaled
DISPLAY_SIZE = (int(1920 * DISPLAY_SCALE), int(1080 * DISPLAY_SCALE))

spots_status = np.zeros(len(spots), dtype=int)  # 1 = empty
diffs = np.zeros(len(spots))
previous_frame = None
frame_nmr = 0
print("Press 'q' to quit")


def crop(img, spot):
    x, y, w, h = spot
    return img[y:y + h, x:x + w, :]


while True:
    success, frame = video.read()
    if not success:  # end of the 28 s clip: rewind and keep the spot states (the lot looks the same at frame 0)
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, frame = video.read()
        if not success:
            raise SystemExit("Could not read parking_1920_1080.mp4")

    if frame_nmr % STEP == 0:
        if previous_frame is None:
            to_check = range(len(spots))
        else:
            # Mean brightness change since the last classified frame; skip spots that barely changed
            for i, spot in enumerate(spots):
                diffs[i] = abs(np.mean(crop(frame, spot)) - np.mean(crop(previous_frame, spot)))
            to_check = np.flatnonzero(diffs / max(diffs.max(), 1e-9) > DIFF_RATIO)
        for i in to_check:
            spots_status[i] = utils.empty_or_not(crop(frame, spots[i]))
        previous_frame = frame.copy()

    for (x, y, w, h), empty in zip(spots, spots_status):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0) if empty else (0, 0, 255), 2)

    display_frame = cv2.resize(frame, DISPLAY_SIZE)
    cv2.rectangle(display_frame, (5, 10), (200, 40), (0, 0, 0), -1)
    cv2.putText(display_frame, f"Empty: {spots_status.sum()} / {len(spots)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Parking Spot Detector", display_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    frame_nmr += 1

video.release()
cv2.destroyAllWindows()
