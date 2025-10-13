#!/usr/bin/env python3
"""
Scatter plot — What are the two features that are similar?

Usage:
  # Auto-select the most similar pair (by |Pearson r|), save + show
  python src/scatter_plot.py data/dataset_train.csv --show

  # Explicit features
  python src/scatter_plot.py data/dataset_train.csv --x "Astronomy" --y "Defense Against the Dark Arts"

  # Also list top-N pairs
  python src/scatter_plot.py data/dataset_train.csv --rank 10

Notes:
- No heavy-lifting stats functions; Pearson r is computed manually.
- Colors points by Hogwarts House.
- --show is cross-platform safe (macOS/Linux); figures are always saved to outputs/figures/.
"""

import argparse
import csv
import math
import os
import sys

# --- Constants ---

HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
EXCLUDE_COLUMNS = {"Hogwarts House", "Index"}
MISSING_TOKENS = {"", "na", "nan", "null", "none", "NaN", "NA", "NULL", "None"}

# Canonical Hogwarts house colors (Option 1)
HOUSE_COLORS = {
    "Gryffindor": "#AE0001",   # Scarlet
    "Hufflepuff": "#FFD800",   # Yellow
    "Ravenclaw":  "#222F5B",   # Blue
    "Slytherin":  "#2A623D",   # Green
}

# ---------------------- Backend & FS helpers (cross-platform) ----------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def configure_matplotlib_backend(wants_show: bool) -> bool:
    """
    Configure a safe Matplotlib backend across macOS/Linux.

    Returns:
        bool: True if interactive showing is enabled (plt.show() will be attempted),
              False if we fell back to a non-interactive backend (Agg).
    """
    try:
        import matplotlib
        try:
            import matplotlib.backends  # noqa: F401
        except Exception:
            pass
    except Exception:
        return False

    if not wants_show:
        try:
            matplotlib.use("Agg", force=True)
        except Exception:
            pass
        return False

    try:
        # On macOS, prefer native Cocoa and don't require DISPLAY.
        if sys.platform == "darwin":
            try:
                matplotlib.use("MacOSX", force=True)
            except Exception:
                pass
            return True

        # On Linux servers without X, show isn't possible.
        if os.name != "nt" and not os.environ.get("DISPLAY"):
            matplotlib.use("Agg", force=True)
            return False

        return True  # desktop Linux/Windows with GUI
    except Exception:
        try:
            matplotlib.use("Agg", force=True)
        except Exception:
            pass
        return False

# ---------------------- CSV & parsing helpers ----------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="Path to dataset_train.csv")
    ap.add_argument("--x", help="Feature name for X axis (optional)")
    ap.add_argument("--y", help="Feature name for Y axis (optional)")
    ap.add_argument("--rank", type=int, default=0, help="Print top-N most similar feature pairs")
    ap.add_argument("--outdir", default="outputs/figures", help="Directory to save figures")
    ap.add_argument("--show", action="store_true", help="Also show plot interactively")
    return ap.parse_args()

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

# ---------------------- Data alignment & math ----------------------

def collect_pair_values(rows, xfeat, yfeat):
    """
    Return aligned arrays (x, y, house_labels).
    Only keep rows where BOTH features are present and numeric.
    """
    X, Y, L = [], [], []
    for r in rows:
        vx = r.get(xfeat, "")
        vy = r.get(yfeat, "")
        if is_missing(vx) or is_missing(vy):
            continue
        fx = try_float(vx)
        fy = try_float(vy)
        if fx is None or fy is None:
            continue
        house = r.get("Hogwarts House", None)
        X.append(fx)
        Y.append(fy)
        L.append(house)
    return X, Y, L

def safe_sum(values):
    total = 0.0
    c = 0.0  # Kahan-like
    for x in values:
        y = x - c
        t = total + y
        c = (t - total) - y
        total = t
    return total

def mean(values):
    n = len(values)
    if n == 0:
        return float("nan")
    return safe_sum(values) / n

def variance(values, mu=None):
    n = len(values)
    if n == 0:
        return float("nan")
    if mu is None:
        mu = mean(values)
    acc = 0.0
    for v in values:
        d = v - mu
        acc += d * d
    # population variance is fine for correlation computation
    return acc / n

def covariance(xs, ys, mux=None, muy=None):
    n = len(xs)
    if n == 0:
        return float("nan")
    if mux is None:
        mux = mean(xs)
    if muy is None:
        muy = mean(ys)
    acc = 0.0
    for i in range(n):
        acc += (xs[i] - mux) * (ys[i] - muy)
    return acc / n

def pearson_r(xs, ys):
    """
    Pearson correlation coefficient r = cov(X,Y)/sqrt(var(X)*var(Y)).
    Returns (r, n_used).
    """
    if len(xs) != len(ys):
        raise ValueError("pearson_r: xs and ys must have same length")
    n = len(xs)
    if n == 0:
        return float("nan"), 0
    mux, muy = mean(xs), mean(ys)
    vx = variance(xs, mux)
    vy = variance(ys, muy)
    if vx == 0.0 or vy == 0.0:
        # perfectly constant axis -> undefined correlation, treat as 0
        return 0.0, n
    cov = covariance(xs, ys, mux, muy)
    r = cov / math.sqrt(vx * vy)
    # numeric clamp for tiny rounding issues
    if r > 1.0:
        r = 1.0
    elif r < -1.0:
        r = -1.0
    return r, n

def all_pairwise_correlations(headers, rows, numeric_features):
    """
    Compute |r| for every unordered pair of numeric features.
    Returns list of tuples: (abs_r, r, n, xfeat, yfeat)
    """
    out = []
    m = len(numeric_features)
    for i in range(m):
        for j in range(i + 1, m):
            xf, yf = numeric_features[i], numeric_features[j]
            X, Y, _ = collect_pair_values(rows, xf, yf)
            r, n = pearson_r(X, Y)
            if n > 0 and not math.isnan(r):
                out.append((abs(r), r, n, xf, yf))
    out.sort(key=lambda t: t[0], reverse=True)  # highest |r| first
    return out

# ---------------------- Plotting ----------------------

def scatter_plot(xfeat, yfeat, rows, outdir, show=False):
    """
    Create a scatter plot for xfeat vs yfeat colored by Hogwarts House.
    Saves outputs/figures/scatter_<xfeat>__<yfeat>.png
    """
    import matplotlib.pyplot as plt

    # Build per-house arrays for plotting
    by_house_x = {h: [] for h in HOUSES}
    by_house_y = {h: [] for h in HOUSES}

    X, Y, L = collect_pair_values(rows, xfeat, yfeat)
    for x, y, house in zip(X, Y, L):
        if house in by_house_x:
            by_house_x[house].append(x)
            by_house_y[house].append(y)

    plt.figure(figsize=(7.5, 5.5))
    for house in HOUSES:
        plt.scatter(
            by_house_x[house],
            by_house_y[house],
            s=20,
            alpha=0.8,
            label=house,
            color=HOUSE_COLORS.get(house)
        )

    # Correlation for the title
    r, n = pearson_r(X, Y)
    plt.title(f"{xfeat} vs {yfeat} — Pearson r={r:.3f} (n={n})")
    plt.xlabel(xfeat)
    plt.ylabel(yfeat)
    plt.legend()
    plt.tight_layout()

    ensure_dir(outdir)
    safe_x = xfeat.replace("/", "_").replace(" ", "_")
    safe_y = yfeat.replace("/", "_").replace(" ", "_")
    out_path = os.path.join(outdir, f"scatter_{safe_x}__{safe_y}.png")
    plt.savefig(out_path, dpi=160)
    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close()
    return out_path, r, n

# ---------------------- Main ----------------------

def main():
    args = parse_args()
    ensure_dir(args.outdir)
    show_enabled = configure_matplotlib_backend(args.show)

    headers, rows = read_csv_rows(args.csv_path)
    if not headers or not rows:
        print("Empty CSV or missing headers.")
        return

    numeric_features = detect_numeric_features(headers, rows)
    if len(numeric_features) < 2:
        print("Not enough numeric features to compare.")
        return

    # Optional: list top-N pairs
    if args.rank and args.rank > 0:
        pairs = all_pairwise_correlations(headers, rows, numeric_features)
        print(f"\n=== Top {args.rank} most similar pairs (by |Pearson r|) ===")
        for k, (abs_r, r, n, xf, yf) in enumerate(pairs[:args.rank], start=1):
            print(f"{k:2d}. |r|={abs_r:.3f} (r={r:.3f}, n={n}) — {xf} vs {yf}")

    # Choose features: explicit or auto
    if args.x and args.y:
        xfeat, yfeat = args.x, args.y
        if xfeat not in numeric_features or yfeat not in numeric_features:
            print(f"Error: --x and --y must be numeric features. Available: {numeric_features}")
            return
    else:
        pairs = all_pairwise_correlations(headers, rows, numeric_features)
        if not pairs:
            print("Could not compute correlations.")
            return
        _, r, n, xfeat, yfeat = pairs[0]
        print(f"\nAuto-selected most similar pair by |r|: {xfeat} vs {yfeat} (r={r:.3f}, n={n})")

    out_path, r, n = scatter_plot(xfeat, yfeat, rows, args.outdir, show=show_enabled)
    print(f"\nSaved scatter to: {out_path}")

if __name__ == "__main__":
    main()