"""Train the empty/occupied SVM on data/<category>/*.jpg and write svm_model.pkl (read by utils.py).

Evaluation splits by parking slot, not by image: each slot contributes ~29 near-identical crops from the same clip,
so a random image split puts the same slot in train and test and reports a meaningless 100%.
Here every test slot is unseen during training and hyperparameter search.
Random Forest and a small CNN were also tried earlier and dropped; SVM was kept.
"""
import os
import pickle
import re
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).parent
input_dir = HERE / "data"
SEED = 0
rng = np.random.default_rng(SEED)

# Sorted so class indices are stable: 0 = empty, 1 = non_empty (utils.empty_or_not relies on this)
categories = sorted(os.listdir(input_dir))
files = {c: sorted(os.listdir(input_dir / c)) for c in categories}
per_class = min(len(f) for f in files.values())  # balance classes by downsampling
for c in categories:
    print(f"{c}: {len(files[c])} images, using {per_class}")
if per_class == 0:
    raise SystemExit("A category folder is empty; run crop_cars.py first")

data, labels, slots = [], [], []
for label, c in enumerate(categories):
    for name in rng.choice(files[c], per_class, replace=False):
        image = cv2.imread(str(input_dir / c / name))  # cv2 read + resize to match utils.empty_or_not
        data.append(cv2.resize(image, (15, 15)).flatten())
        labels.append(label)
        slots.append(int(re.search(r"slot_(\d+)_", name).group(1)))  # crop_cars.py names: <class>_slot_<n>_frame_<f>.jpg
X, y, groups = np.array(data, dtype=float), np.array(labels), np.array(slots)
print(f"Data: {X.shape[0]} crops from {len(set(slots))} slots")

pipe = make_pipeline(StandardScaler(), SVC())
grid = {"svc__C": [0.1, 1, 10], "svc__gamma": [0.1, 0.01, 0.001]}

# Nested, slot-grouped CV: outer folds estimate accuracy on unseen slots, inner folds pick C/gamma
outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
y_true, y_pred, fold_acc, wrong_slots = [], [], [], set()
for train_idx, test_idx in outer.split(X, y, groups):
    search = GridSearchCV(pipe, grid, cv=StratifiedGroupKFold(n_splits=4))
    search.fit(X[train_idx], y[train_idx], groups=groups[train_idx])
    pred = search.predict(X[test_idx])
    fold_acc.append(accuracy_score(y[test_idx], pred))
    wrong_slots |= set(groups[test_idx][pred != y[test_idx]].tolist())
    y_true.extend(y[test_idx])
    y_pred.extend(pred)
    print(f"fold {len(fold_acc)}: {len(set(groups[test_idx]))} unseen slots, acc {fold_acc[-1]:.4f}, best {search.best_params_}")

print(classification_report(y_true, y_pred, target_names=categories, digits=4))
print("Confusion matrix (all outer folds):\n", confusion_matrix(y_true, y_pred))
print(f"Unseen-slot accuracy: {np.mean(fold_acc):.4f} +/- {np.std(fold_acc):.4f}")
print(f"Slots with any misclassified crop: {sorted(wrong_slots) or 'none'}")

# Final model: same search on all slots, then save scaler + SVC separately (format utils.py loads)
final = GridSearchCV(pipe, grid, cv=StratifiedGroupKFold(n_splits=5)).fit(X, y, groups=groups)
print("Final model params:", final.best_params_)
scaler, model = final.best_estimator_.named_steps["standardscaler"], final.best_estimator_.named_steps["svc"]
with open(HERE / "svm_model.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler, "model_type": "SVM"}, f)
print("Saved svm_model.pkl")
