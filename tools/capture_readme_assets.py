"""Regenerate docs/assets/ from the real detector output (needs ffmpeg on PATH).

    python tools/capture_readme_assets.py

hero.png     detect.py's annotated output at frame 0
zoom.gif     full-res close-up of spot 183 while a car parks in it (frames 120-270)
crops.png    training crops from data/, top row empty, bottom row occupied
social-preview.png  1280x640 GitHub social card
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import detect  # noqa: E402
import utils  # noqa: E402

OUT = ROOT / "docs" / "assets"
FRAMES = 300
ZOOM_SPOT, ZOOM_FRAMES = 183, range(120, 270)
ZOOM_BOX = (330, 430, 480, 270)  # x, y, w, h around spot 183 at full res


def gif(src, dst, fps, width):
    pal = f"fps={fps},scale={width}:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf",
                    f"{pal},split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                    "-loop", "0", str(dst)], check=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mask = cv2.imread(str(ROOT / "mask_1920_1080.png"), cv2.IMREAD_GRAYSCALE)
    spots = utils.get_parking_spots_bboxes(cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S))
    video = cv2.VideoCapture(str(ROOT / "parking_1920_1080.mp4"))
    tmp = Path(tempfile.mkdtemp())
    zx, zy, zw, zh = ZOOM_BOX
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    zoom_w = None

    status, previous, last = np.zeros(len(spots), int), None, None
    for n in range(FRAMES):
        ok, frame = video.read()
        if not ok:
            raise SystemExit("Could not read parking_1920_1080.mp4")
        if n % detect.STEP == 0:
            for i in detect.spots_to_check(frame, previous, spots):
                status[i] = utils.empty_or_not(detect.crop(frame, spots[i]))
            previous = frame.copy()
        out = detect.draw(frame, spots, status)  # also draws the boxes onto frame at full res
        if zoom_w is None:
            zoom_w = cv2.VideoWriter(str(tmp / "zoom.mp4"), fourcc, 30, (zw, zh))
        if n in ZOOM_FRAMES:
            sx, sy, sw, sh = spots[ZOOM_SPOT]
            zoom = frame.copy()
            cv2.rectangle(zoom, (sx - 4, sy - 4), (sx + sw + 4, sy + sh + 4), (0, 220, 255), 2)  # yellow marker
            zoom = zoom[zy:zy + zh, zx:zx + zw]
            label = "EMPTY" if status[ZOOM_SPOT] else "OCCUPIED"
            cv2.rectangle(zoom, (0, 0), (250, 30), (0, 0, 0), -1)
            cv2.putText(zoom, f"spot {ZOOM_SPOT}: {label}", (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if status[ZOOM_SPOT] else (0, 0, 255), 2, cv2.LINE_AA)
            zoom_w.write(zoom)
        if n == 0:
            last = out
    zoom_w.release()
    video.release()
    cv2.imwrite(str(OUT / "hero.png"), last)
    gif(tmp / "zoom.mp4", OUT / "zoom.gif", 12, 480)

    # Training crops: 10 slots per class, frame 0, upscaled with nearest so pixels stay honest
    rows = []
    for folder in ("empty", "non_empty"):
        files = sorted((ROOT / "data" / folder).glob("*_frame_0.jpg"))[:10]
        rows.append(np.hstack([cv2.copyMakeBorder(cv2.resize(cv2.imread(str(f)), (136, 64)), 4, 4, 4, 4,
                                                  cv2.BORDER_CONSTANT, value=(255, 255, 255)) for f in files]))
    cv2.imwrite(str(OUT / "crops.png"), np.vstack(rows))

    # Social preview: real annotated frame on the right, title on the left, soft fade between
    card = np.full((640, 1280, 3), (24, 20, 18), np.uint8)
    shot = cv2.resize(last, (1138, 640))[:, 300:1138]  # right part of the lot, 838 px wide
    fade = np.clip(np.linspace(-0.2, 1.5, shot.shape[1]), 0, 1)[None, :, None]
    card[:, 442:] = (shot * fade + card[:, 442:] * (1 - fade)).astype(np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    for text, y, scale, color in [("Parking Spot", 230, 1.9, (255, 255, 255)),
                                  ("Detector", 300, 1.9, (255, 255, 255)),
                                  ("396 spots, empty or occupied", 380, 0.85, (200, 200, 200)),
                                  ("SVM + frame-difference filter", 420, 0.85, (200, 200, 200)),
                                  ("94% on unseen slots", 480, 0.95, (120, 230, 120))]:
        cv2.putText(card, text, (60, y), font, scale, color, 2 if scale > 1 else 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / "social-preview.png"), card)
    for p in sorted(OUT.iterdir()):
        print(f"{p.name}: {p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
