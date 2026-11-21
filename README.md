# Parking Spot Detector

Marks each of 396 parking spots in a fixed-camera video as empty (green) or occupied (red), with a live empty-spot count. Spot boxes come from a mask image; an RBF SVM classifies each 15x15 spot crop, and a frame-difference filter cuts model calls about 13x.

![Python](https://img.shields.io/badge/python-3.11-blue)
![OpenCV](https://img.shields.io/badge/opencv-4.11-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-orange)
![License](https://img.shields.io/badge/license-MIT-blue)

## Quick start

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python detect.py               # press q to quit
```

The trained model (`svm_model.pkl`) and dataset (`data/`) are included, so detection runs straight away. Scripts resolve paths relative to their own folder, so you can run them from anywhere.

## Pipeline

| Step | Command | Input -> output |
| --- | --- | --- |
| 1. Build dataset (optional, deletes `data/`) | `python crop_cars.py` | `parking_1920_1080.mp4` + hand-labelled slot lists -> `data/empty`, `data/non_empty` (1044 / 1827 crops) |
| 2. Train (optional, overwrites the model) | `python train.py` | `data/` -> `svm_model.pkl` (grid-searched RBF SVM + StandardScaler) |
| 3. Detect | `python detect.py` | `parking_1920_1080.mp4` (looped) + `mask_1920_1080.png` + model -> live window |

`utils.py` holds the shared pieces: model loading, `empty_or_not()` (its preprocessing must match training) and `get_parking_spots_bboxes()` (the slot numbers in `crop_cars.py` index this list).

## How detection stays fast

`detect.py` classifies only every 30th frame, and then only the spots whose mean brightness changed by more than 40% of that step's biggest change. Over 1800 frames this made about 13x fewer model calls than classifying every spot, while matching the classify-everything run on 99.9% of spot-frames (never more than 2 spots off in any frame).

## Results and limits

- Steps 1 to 3 run end to end; step 1 reproduces the committed `data/` byte for byte.
- **Accuracy on unseen parking slots: 94.1% ± 4.2%** (5-fold cross-validation grouped by slot, hyperparameters tuned inside each fold). An earlier random image split reported 100%, but each slot gives ~29 near-identical crops, so the same slot sat in both train and test. `train.py` now splits by slot.
- Misses cluster in about 10 of the 99 labelled slots. The labels were checked by eye and look correct; the hard cases are cars only partly inside the box, empty spots with painted markings or dark pavement, and spots next to bright cars.
- Tried, same grouped CV: 30x30 pixels 94.8%, HOG 93.3%, HOG + HSV colour stats 96.2%, pixels + edge density 94.6%. None beat the 15x15 baseline by more than the fold-to-fold spread, so the simplest features stay.
- Still untested: other lots, other lighting, other camera angles. All data comes from one 28 s clip, so treat 94% as an upper bound for new footage. The best next step is labelling more slots or a second clip.
- Random Forest and CNN classifiers were tried and dropped in favour of the SVM.

## Project structure

```
detect.py                live detector
train.py                 SVM training + evaluation
crop_cars.py             dataset builder
utils.py                 shared model loading and spot extraction
mask_1920_1080.png       one white blob per parking spot
parking_1920_1080.mp4    28 s source clip
svm_model.pkl            trained model
data/                    labelled 15x15 crops (empty / non_empty)
plots/                   histograms from tuning the change filter
```

## License

[MIT](LICENSE)
