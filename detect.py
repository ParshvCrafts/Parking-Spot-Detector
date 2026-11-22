"""Live parking detector: loops the clip, marks each mask spot empty (green) or occupied (red).

    python detect.py                                  # window, press q to quit
    python detect.py --save out.mp4 --frames 300 --no-window   # headless render to a video file
"""
import argparse

import cv2
import numpy as np

import utils

STEP = 30            # classify every STEP frames
DIFF_RATIO = 0.4     # re-classify spots whose change is > 40% of this step's largest change
DISPLAY_SCALE = 0.6  # processing stays full-res; only the output is scaled


def crop(img, spot):
    x, y, w, h = spot
    return img[y:y + h, x:x + w, :]


def spots_to_check(frame, previous_frame, spots):
    """Indices of spots worth re-classifying: all on the first step, else those whose mean brightness
    changed by more than DIFF_RATIO of the largest change since the last classified frame."""
    if previous_frame is None:
        return np.arange(len(spots))
    diffs = np.array([abs(np.mean(crop(frame, s)) - np.mean(crop(previous_frame, s))) for s in spots])
    return np.flatnonzero(diffs / max(diffs.max(), 1e-9) > DIFF_RATIO)


def draw(frame, spots, spots_status):
    """Spot boxes on the full-res frame, then a downscaled copy with the empty-spot counter."""
    for (x, y, w, h), empty in zip(spots, spots_status):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0) if empty else (0, 0, 255), 2)
    out = cv2.resize(frame, (int(frame.shape[1] * DISPLAY_SCALE), int(frame.shape[0] * DISPLAY_SCALE)))
    cv2.rectangle(out, (5, 10), (200, 40), (0, 0, 0), -1)
    cv2.putText(out, f"Empty: {int(spots_status.sum())} / {len(spots)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save", help="write the annotated output to this .mp4")
    parser.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = run until q)")
    parser.add_argument("--no-window", action="store_true", help="do not open a display window")
    args = parser.parse_args()
    if args.no_window and not args.frames:
        parser.error("--no-window needs --frames, otherwise nothing would stop it")

    video = cv2.VideoCapture(str(utils.HERE / "parking_1920_1080.mp4"))
    mask = cv2.imread(str(utils.HERE / "mask_1920_1080.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None or not video.isOpened():
        raise SystemExit("Could not load mask_1920_1080.png or parking_1920_1080.mp4")
    spots = utils.get_parking_spots_bboxes(cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S))
    fps = video.get(cv2.CAP_PROP_FPS) or 25

    spots_status = np.zeros(len(spots), dtype=int)  # 1 = empty
    previous_frame = None
    writer = None
    frame_nmr = 0
    if not args.no_window:
        print("Press 'q' to quit")

    while not args.frames or frame_nmr < args.frames:
        success, frame = video.read()
        if not success:  # end of the 28 s clip: rewind and keep the spot states (the lot looks the same at frame 0)
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = video.read()
            if not success:
                raise SystemExit("Could not read parking_1920_1080.mp4")

        if frame_nmr % STEP == 0:
            for i in spots_to_check(frame, previous_frame, spots):
                spots_status[i] = utils.empty_or_not(crop(frame, spots[i]))
            previous_frame = frame.copy()

        out = draw(frame, spots, spots_status)
        if args.save:
            if writer is None:
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, out.shape[1::-1])
            writer.write(out)
        if not args.no_window:
            cv2.imshow("Parking Spot Detector", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_nmr += 1

    video.release()
    if writer is not None:
        writer.release()
        print(f"Saved {frame_nmr} frames to {args.save}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
