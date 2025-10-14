#!/usr/bin/env python3
"""
Predict Hogwarts House using a one-vs-all Logistic Regression model
trained by src/logreg_train.py.

- Loads the JSON model (weights, feature order, means/stds, etc.)
- Reads dataset_test.csv
- Imputes any missing/non-numeric feature with the training mean
- Applies the same standardization (if enabled in the model)
- Computes probabilities for each class and picks argmax
- Writes outputs/houses.csv with headers: Index,Hogwarts House
- (Optional) Also writes per-class probabilities to a CSV

Usage:
  # Default paths
  python src/logreg_predict.py data/dataset_test.csv --model models/logreg_model.json

  # Custom output path
  python src/logreg_predict.py data/dataset_test.csv --out outputs/houses.csv

  # Also save class probabilities
  python src/logreg_predict.py data/dataset_test.csv --proba-out outputs/probas.csv

Notes:
- No sklearn; all math is manual and consistent with training.
"""

import argparse
import csv
import json
import math
import os
import sys
from typing import List, Dict, Tuple

# ----------------------------- Utilities ---------------------------------

def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)

def try_float(s: str):
    try:
        return float(s)
    except Exception:
        return None

# ----------------------------- Model math --------------------------------

def sigmoid(z: float) -> float:
    # Numerically safe sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)

def dot(theta: List[float], x: List[float]) -> float:
    s = 0.0
    for j in range(len(theta)):
        s += theta[j] * x[j]
    return s

def add_bias_row(z: List[float]) -> List[float]:
    # Prepend 1.0 for bias term
    return [1.0] + z

# ----------------------------- IO helpers --------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="Path to dataset_test.csv")
    ap.add_argument("--model", default="models/logreg_model.json", help="Path to trained JSON model")
    ap.add_argument("--out", default="outputs/houses.csv", help="Output CSV with predictions (Index,Hogwarts House)")
    ap.add_argument("--proba-out", default=None, help="Optional: write per-class probabilities CSV")
    ap.add_argument("--verbose", action="store_true", default=False, help="Print a few sanity checks")
    return ap.parse_args()

def load_model(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        model = json.load(f)
    # sanity checks
    required = ["houses", "feature_names", "standardize", "means", "stds", "thetas"]
    for k in required:
        if k not in model:
            raise ValueError(f"Model file missing '{k}'")
    houses = model["houses"]
    feats = model["feature_names"]
    means = model["means"]
    stds = model["stds"]
    if len(means) != len(feats) or len(stds) != len(feats):
        raise ValueError("Model means/stds length mismatch with feature_names")
    for h in houses:
        th = model["thetas"].get(h)
        if th is None:
            raise ValueError(f"Missing theta for class '{h}'")
        if len(th) != len(feats) + 1:  # +1 for bias
            raise ValueError(f"Theta length mismatch for class '{h}'")
    return model

def read_test_rows(path: str) -> Tuple[List[str], List[dict]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)
    return headers, rows

# ----------------------------- Prediction --------------------------------

def build_feature_vector(row: dict, feature_names: List[str], means: List[float]) -> List[float]:
    """
    Returns a raw (unstandardized) feature vector.
    For any missing/non-numeric value, impute with the training mean.
    """
    vec = []
    for j, name in enumerate(feature_names):
        v = row.get(name, "")
        fv = try_float(v)
        if fv is None:
            fv = means[j]  # impute
        vec.append(fv)
    return vec

def standardize_vector(x: List[float], means: List[float], stds: List[float]) -> List[float]:
    z = []
    for j in range(len(x)):
        sd = stds[j] if stds[j] != 0.0 else 1.0
        z.append((x[j] - means[j]) / sd)
    return z

def predict_row_probas(
    row: dict,
    houses: List[str],
    feature_names: List[str],
    standardize: bool,
    means: List[float],
    stds: List[float],
    thetas: Dict[str, List[float]],
) -> Dict[str, float]:
    """
    Returns a dict house->probability (one-vs-all scores) for a single row.
    These are NOT guaranteed to sum to 1.0 (OVA), but argmax is meaningful.
    """
    x = build_feature_vector(row, feature_names, means)
    z = standardize_vector(x, means, stds) if standardize else x
    xb = add_bias_row(z)

    probs = {}
    for h in houses:
        theta = thetas[h]
        p = sigmoid(dot(theta, xb))
        probs[h] = p
    return probs

def argmax_label(probas: Dict[str, float]) -> str:
    best_h = None
    best_p = -1.0
    for h, p in probas.items():
        if p > best_p:
            best_p = p
            best_h = h
    return best_h

# ----------------------------- Writing outputs ---------------------------

def write_predictions(out_path: str, rows: List[dict], preds: List[str]):
    ensure_dir(os.path.dirname(out_path))
    # Try to preserve 'Index' if present, else use row order
    has_index = "Index" in (rows[0].keys() if rows else {})
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Hogwarts House"])
        for i, r in enumerate(rows):
            idx = r.get("Index", i)
            writer.writerow([idx, preds[i]])
    print(f"Saved predictions → {out_path}")

def write_probabilities(out_path: str, rows: List[dict], houses: List[str], all_probas: List[Dict[str, float]]):
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Index"] + houses)
        for i, r in enumerate(rows):
            idx = r.get("Index", i)
            row_out = [idx] + [f"{all_probas[i].get(h, 0.0):.6f}" for h in houses]
            writer.writerow(row_out)
    print(f"Saved per-class probabilities → {out_path}")

# ----------------------------- Main --------------------------------------

def main():
    args = parse_args()

    # Load model
    model = load_model(args.model)
    houses: List[str] = model["houses"]
    features: List[str] = model["feature_names"]
    standardize: bool = bool(model.get("standardize", True))
    means: List[float] = model["means"]
    stds: List[float] = model["stds"]
    thetas: Dict[str, List[float]] = model["thetas"]

    # Read test set
    headers, rows = read_test_rows(args.csv_path)
    if not rows:
        print("Empty test CSV.")
        sys.exit(1)

    # Predict each row
    preds: List[str] = []
    all_probas: List[Dict[str, float]] = []
    for r in rows:
        probas = predict_row_probas(
            r, houses, features, standardize, means, stds, thetas
        )
        label = argmax_label(probas)
        preds.append(label)
        all_probas.append(probas)

    # Write outputs
    write_predictions(args.out, rows, preds)
    if args.proba_out:
        write_probabilities(args.proba_out, rows, houses, all_probas)

    # Optional sanity report
    n = len(rows)
    uniq = {p: preds.count(p) for p in houses}
    print(f"Predicted {n} rows. Class counts: {uniq}")

if __name__ == "__main__":
    main()
