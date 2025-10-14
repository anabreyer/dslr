"""
Train one-vs-all Logistic Regression (from scratch) on Hogwarts dataset.

Usage:
  python src/logreg_train.py data/dataset_train.csv

Optional flags:
  --alpha 0.1           # learning rate (default: 0.1)
  --max-iter 20000      # max gradient descent iterations (default: 15000)
  --tol 1e-6            # early-stopping tolerance on cost improvement (default: 1e-6)
  --lambda 0.0          # L2 regularization strength (default: 0.0)
  --no-standardize      # disable feature standardization (NOT recommended)
  --out models/weights.json

What it does:
- Reads dataset_train.csv
- Picks numeric features (excludes "Hogwarts House", "Index")
- Standardizes features (mean=0, std=1) unless --no-standardize
- Trains 4 binary logistic models (one-vs-all) with batch gradient descent + optional L2
- Saves model, feature order, means/stds to JSON for prediction
"""

import argparse
import csv
import json
import math
import os
import sys
from typing import List, Dict, Tuple

# ---------------- Constants ----------------

HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
TARGET_COL = "Hogwarts House"
EXCLUDE_COLUMNS = {"Hogwarts House", "Index"}
MISSING_TOKENS = {"", "na", "nan", "null", "none", "NaN", "NA", "NULL", "None"}

DEFAULT_ALPHA = 0.1
DEFAULT_MAX_ITER = 15000
DEFAULT_TOL = 1e-6
DEFAULT_L2 = 0.0
DEFAULT_OUT = "models/weights.json"

# ---------------- CLI ----------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="Path to dataset_train.csv")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Learning rate")
    ap.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER, help="Max iterations")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL, help="Early stopping tolerance on cost")
    ap.add_argument("--lambda", dest="l2", type=float, default=DEFAULT_L2, help="L2 regularization strength")
    ap.add_argument("--no-standardize", action="store_true", help="Disable feature standardization")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Where to save learned weights (JSON)")
    return ap.parse_args()

# ---------------- IO Helpers ----------------

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def is_missing(s: str) -> bool:
    return s.strip() in MISSING_TOKENS

def try_float(s: str):
    try:
        return float(s)
    except Exception:
        return None

def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)
    return headers, rows

def detect_numeric_features(headers, rows):
    numeric = []
    for h in headers:
        if h in EXCLUDE_COLUMNS:
            continue
        ok = True
        for r in rows:
            v = r.get(h, "")
            if is_missing(v):
                continue
            if try_float(v) is None:
                ok = False
                break
        if ok:
            numeric.append(h)
    return numeric

# ---------------- Small numerics (manual) ----------------

def kahan_sum(values: List[float]) -> float:
    total = 0.0
    c = 0.0
    for x in values:
        y = x - c
        t = total + y
        c = (t - total) - y
        total = t
    return total

def mean(values: List[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    return kahan_sum(values) / n

def stddev(values: List[float], mu: float) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    acc = 0.0
    for v in values:
        d = v - mu
        acc += d * d
    # population std (ok for standardization)
    return math.sqrt(acc / n)

# ---------------- Data assembly ----------------

def build_design_matrix(rows, features: List[str], standardize: bool):
    """
    Returns:
      X: List[List[float]] of shape (m, n)  (WITHOUT bias)
      y_house: List[str] length m
      mu: List[float] per feature
      sigma: List[float] per feature (>= 1e-12)
    - Drops rows with missing target or missing feature value.
    - Standardizes X if requested: x' = (x - mu) / sigma
    """
    X_raw = []
    y_house = []
    for r in rows:
        house = r.get(TARGET_COL, "")
        if house not in HOUSES:
            continue
        row_vals = []
        keep = True
        for feat in features:
            v = r.get(feat, "")
            if is_missing(v):
                keep = False
                break
            f = try_float(v)
            if f is None:
                keep = False
                break
            row_vals.append(f)
        if keep:
            X_raw.append(row_vals)
            y_house.append(house)

    m = len(X_raw)
    n = len(features)
    if m == 0 or n == 0:
        return [], [], [], []

    # compute mu/sigma per feature
    mu = [0.0] * n
    sigma = [1.0] * n
    for j in range(n):
        col = [X_raw[i][j] for i in range(m)]
        mu[j] = mean(col)
        s = stddev(col, mu[j])
        if s < 1e-12:
            s = 1.0  # avoid divide-by-zero; constant feature stays zero after centering
        sigma[j] = s

    # standardize
    if standardize:
        X = []
        for i in range(m):
            row = []
            for j in range(n):
                row.append((X_raw[i][j] - mu[j]) / sigma[j])
            X.append(row)
    else:
        X = X_raw

    return X, y_house, mu, sigma

def add_bias_column(X: List[List[float]]) -> List[List[float]]:
    """Add bias=1 as the first column."""
    return [[1.0] + row for row in X]

# ---------------- Logistic regression core ----------------

def sigmoid(z: float) -> float:
    # stable sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)

def dot(u: List[float], v: List[float]) -> float:
    s = 0.0
    for a, b in zip(u, v):
        s += a * b
    return s

def predict_proba_row(theta: List[float], xrow: List[float]) -> float:
    return sigmoid(dot(theta, xrow))

def compute_cost_and_grad(theta: List[float], Xb: List[List[float]], y: List[int], l2: float) -> Tuple[float, List[float]]:
    """
    Binary cross-entropy + L2 (excluding bias).
    J = -1/m sum y log(h) + (1-y) log(1-h) + (λ/(2m)) sum_{j>=1} θ_j^2
    grad_j = 1/m * sum (h - y) * x_j + (λ/m) * θ_j for j>=1 ; grad_0 has no reg
    """
    m = len(Xb)
    n = len(theta)
    eps = 1e-15

    # predictions and residuals
    h = [predict_proba_row(theta, Xb[i]) for i in range(m)]
    # cost
    cost = 0.0
    for i in range(m):
        hi = min(max(h[i], eps), 1.0 - eps)  # clip
        yi = y[i]
        cost += -(yi * math.log(hi) + (1 - yi) * math.log(1 - hi))
    cost /= m

    # L2 reg (no bias)
    reg = 0.0
    for j in range(1, n):
        reg += theta[j] * theta[j]
    cost += (l2 / (2.0 * m)) * reg

    # gradient
    grad = [0.0] * n
    for j in range(n):
        s = 0.0
        for i in range(m):
            s += (h[i] - y[i]) * Xb[i][j]
        s /= m
        if j >= 1:  # regularize all but bias
            s += (l2 / m) * theta[j]
        grad[j] = s

    return cost, grad

def gradient_descent(Xb: List[List[float]], y: List[int], alpha: float, max_iter: int, tol: float, l2: float, verbose: bool=False) -> Tuple[List[float], float]:
    """
    Batch gradient descent to minimize logistic cost.
    Returns final theta and final cost.
    Early-stops if improvement < tol for several checks.
    """
    n = len(Xb[0])  # num params incl. bias
    theta = [0.0] * n
    prev = float("inf")
    patience = 5
    stable = 0
    final_cost = float("inf")

    for it in range(1, max_iter + 1):
        cost, grad = compute_cost_and_grad(theta, Xb, y, l2)
        # update
        for j in range(n):
            theta[j] -= alpha * grad[j]

        # early stopping
        if prev - cost < tol:
            stable += 1
        else:
            stable = 0
        prev = cost
        final_cost = cost

        if verbose and (it % 1000 == 0 or it == 1):
            print(f"[iter {it:5d}] cost={cost:.6f}")

        if stable >= patience:
            if verbose:
                print(f"Early stop at iter {it} (Δcost < {tol} for {patience} checks)")
            break

    return theta, final_cost

def train_one_vs_all(X: List[List[float]], y_house: List[str], l2: float, alpha: float, max_iter: int, tol: float) -> Dict[str, Dict]:
    """
    Trains 4 binary classifiers (one per house: house vs rest).
    Returns dict: house -> {"theta": [...], "cost": float}
    """
    # Add bias
    Xb = add_bias_column(X)
    m = len(Xb)
    n = len(Xb[0])

    results = {}
    for house in HOUSES:
        # Build binary labels
        y = [1 if yh == house else 0 for yh in y_house]
        theta, cost = gradient_descent(Xb, y, alpha=alpha, max_iter=max_iter, tol=tol, l2=l2, verbose=True)
        results[house] = {"theta": theta, "final_cost": cost}
    return results

# ---------------- Main ----------------

def main():
    args = parse_args()
    headers, rows = read_csv_rows(args.csv_path)
    if not headers or not rows:
        print("Empty CSV or missing headers.")
        return

    # pick numeric features
    features = detect_numeric_features(headers, rows)
    if not features:
        print("No numeric features found.")
        return

    # assemble X, y, and standardize
    standardize = not args.no_standardize
    X, y_house, mu, sigma = build_design_matrix(rows, features, standardize=standardize)
    if not X or not y_house:
        print("No usable rows after filtering missing values.")
        return

    print(f"Training on {len(X)} samples with {len(features)} features (standardize={standardize})")
    print(f"Hyperparams: alpha={args.alpha}, max_iter={args.max_iter}, tol={args.tol}, lambda={args.l2}")

    # train one-vs-all models
    results = train_one_vs_all(X, y_house, l2=args.l2, alpha=args.alpha, max_iter=args.max_iter, tol=args.tol)

    # package weights
    payload = {
        "classes": HOUSES,
        "features": features,      # order matters!
        "mu": mu,                  # for standardization
        "sigma": sigma,
        "standardize": standardize,
        "alpha": args.alpha,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "lambda": args.l2,
        "thetas": {house: results[house]["theta"] for house in HOUSES},
        "final_costs": {house: results[house]["final_cost"] for house in HOUSES},
    }

    ensure_dir(args.out)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved model to {args.out}")
    for house in HOUSES:
        print(f"- {house}: final cost = {payload['final_costs'][house]:.6f}")

if __name__ == "__main__":
    main()
