import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import detect  # noqa: E402
import utils  # noqa: E402


@pytest.fixture(scope="module")
def spots():
    mask = cv2.imread(str(ROOT / "mask_1920_1080.png"), cv2.IMREAD_GRAYSCALE)
    return utils.get_parking_spots_bboxes(cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S))


def test_mask_gives_396_spots(spots):
    # crop_cars.py slot numbers index this list; a different count means the labels point at the wrong spots
    assert len(spots) == 396


@pytest.mark.parametrize("folder, expected", [("empty", utils.EMPTY), ("non_empty", utils.NOT_EMPTY)])
def test_classifier_class_mapping(folder, expected):
    # Catches an inverted class index (train.py sorts folders: 0 = empty). One crop per slot, first frame.
    files = sorted((ROOT / "data" / folder).glob("*_frame_0.jpg"))
    assert files
    preds = [utils.empty_or_not(cv2.imread(str(f))) for f in files]
    assert np.mean([p == expected for p in preds]) > 0.9


def test_change_filter_first_step_checks_everything(spots):
    frame = np.zeros((1080, 1920, 3), np.uint8)
    assert len(detect.spots_to_check(frame, None, spots)) == len(spots)


def test_change_filter_picks_only_changed_spot(spots):
    before = np.zeros((1080, 1920, 3), np.uint8)
    after = before.copy()
    x, y, w, h = spots[42]
    after[y:y + h, x:x + w] = 200
    assert detect.spots_to_check(after, before, spots).tolist() == [42]
    assert detect.spots_to_check(before, before, spots).size == 0  # nothing changed, nothing re-classified


def test_change_filter_matches_classify_everything(spots):
    # Real clip, first 10 classification steps: the filtered states must agree with classifying every spot
    video = cv2.VideoCapture(str(ROOT / "parking_1920_1080.mp4"))
    filtered = np.zeros(len(spots), int)
    previous, agree, calls = None, [], 0
    for frame_nmr in range(300):
        ok, frame = video.read()
        assert ok
        if frame_nmr % detect.STEP:
            continue
        idx = detect.spots_to_check(frame, previous, spots)
        calls += len(idx)
        for i in idx:
            filtered[i] = utils.empty_or_not(detect.crop(frame, spots[i]))
        full = np.array([utils.empty_or_not(detect.crop(frame, s)) for s in spots], int)
        agree.append(np.mean(filtered == full))
        previous = frame.copy()
    video.release()
    assert np.mean(agree) > 0.99
    assert calls < 10 * len(spots) / 3  # filter must cut model calls substantially
