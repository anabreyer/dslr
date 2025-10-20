#!/usr/bin/env python3
"""
Predict Hogwarts House using a one-vs-all Logistic Regression model trained by src/logreg_train.py.

This version STRICTLY uses models/weights.json (no alternate schemas or paths).

- Loads models/weights.json (keys: classes, features, mu, sigma, thetas, optional standardize)
- Reads dataset_test.csv
- Imputes any missing/non-numeric feature with the training mean (mu)
- Applies the same standardization (if enabled)
- Computes OVA probabilities and picks argmax
- Writes outputs/houses.csv with headers: Index,Hogwarts House
- (Optional) Also writes per-class probabilities to a CSV

Usage:
  python3 src/logreg_predict.py data/dataset_test.csv
  python3 src/logreg_predict.py data/dataset_test.csv --out outputs/houses.csv
  python3 src/logreg_predict.py data/dataset_test.csv --proba-out outputs/probas.csv
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
    ap.add_argument(
        "--out",
        default="outputs/houses.csv",
        help="Output CSV with predictions (Index,Hogwarts House)"
    )
    ap.add_argument(
        "--proba-out",
        default=None,
        help="Optional: write per-class probabilities CSV"
    )
    ap.add_argument("--verbose", action="store_true", default=False, help="Print a few sanity checks")
    return ap.parse_args()

def load_weights_json() -> dict:
    model_path = "models/weights.json"
    if not os.path.exists(model_path):
        print("❌ Expected model at models/weights.json (not found).")
        sys.exit(1)

    with open(model_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Strict schema: classes, features, mu, sigma, thetas
    required = ["classes", "features", "mu", "sigma", "thetas"]
    for k in required:
        if k not in raw:
            raise ValueError(f"models/weights.json is missing required key '{k}'")

    houses = raw["classes"]
    feats = raw["features"]
    means = raw["mu"]
    stds  = raw["sigma"]
    thetas = raw["thetas"]
    standardize = bool(raw.get("standardize", True))

    if not isinstance(houses, list) or not isinstance(feats, list):
        raise ValueError("classes and features must be lists in models/weights.json")
    if len(means) != len(feats) or len(stds) != len(feats):
        raise ValueError("mu/sigma length must match features length in models/weights.json")

    for h in houses:
        th = thetas.get(h)
        if th is None:
            raise ValueError(f"models/weights.json: missing theta for class '{h}'")
        if len(th) != len(feats) + 1:
            raise ValueError(f"models/weights.json: theta length mismatch for class '{h}' "
                             f"(expected {len(feats)+1}, got {len(th)})")

    # Normalize to predictor's internal keys
    model = {
        "houses": houses,
        "feature_names": feats,
        "standardize": standardize,
        "means": means,
        "stds": stds,
        "thetas": thetas,
        "_source": "models/weights.json",
    }
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
    has_index = bool(rows) and "Index" in rows[0]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Hogwarts House"])
        for i, r in enumerate(rows):
            idx = r.get("Index", i) if has_index else i
            writer.writerow([idx, preds[i]])
    print(f"Saved predictions → {out_path}")

def write_probabilities(out_path: str, rows: List[dict], houses: List[str], all_probas: List[Dict[str, float]]):
    ensure_dir(os.path.dirname(out_path))
    has_index = bool(rows) and "Index" in rows[0]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Index"] + houses)
        for i, r in enumerate(rows):
            idx = r.get("Index", i) if has_index else i
            row_out = [idx] + [f"{all_probas[i].get(h, 0.0):.6f}" for h in houses]
            writer.writerow(row_out)
    print(f"Saved per-class probabilities → {out_path}")

# ----------------------------- Main --------------------------------------

def main():
    args = parse_args()

    # Load STRICT weights.json
    model = load_weights_json()
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
