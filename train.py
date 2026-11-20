"""Train the empty/occupied SVM on data/<category>/*.jpg and write svm_model.pkl (read by utils.py).

Random Forest and a small CNN were also tried (commented code in commit f7d69f73); SVM was kept.
"""
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).parent
input_dir = HERE / "data"

# Sorted so class indices are stable: 0 = empty, 1 = non_empty (utils.empty_or_not relies on this)
categories = sorted(os.listdir(input_dir))
files = {c: os.listdir(input_dir / c) for c in categories}
per_class = min(len(f) for f in files.values())  # balance classes by downsampling
for c in categories:
    print(f"{c}: {len(files[c])} images, using {per_class}")
if per_class == 0:
    raise SystemExit("A category folder is empty; run crop_cars.py first")

data, labels = [], []
for label, c in enumerate(categories):
    for name in np.random.choice(files[c], per_class, replace=False):
        image = cv2.imread(str(input_dir / c / name))  # cv2 read + resize to match utils.empty_or_not
        data.append(cv2.resize(image, (15, 15)).flatten())
        labels.append(label)
data, labels = np.array(data, dtype=float), np.array(labels)
print("Data shape:", data.shape)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

parameters = {"C": [0.1, 1, 10], "gamma": [0.1, 0.01, 0.001], "kernel": ["rbf"]}
grid_search = GridSearchCV(SVC(), parameters, refit=True, verbose=2)
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_
print("Best parameters found:", grid_search.best_params_)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=categories))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)

with open(HERE / "svm_model.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler, "model_type": "SVM"}, f)
print(f"Saved svm_model.pkl  ACCURACY: {accuracy:.4f}")
