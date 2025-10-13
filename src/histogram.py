#!/usr/bin/env python3
"""
Histogram — Which Hogwarts course has a homogeneous score distribution between all four houses?

Usage:
  python src/histogram.py data/dataset_train.csv
  # This will save one PNG per course under outputs/figures/
  # and print a ranked list of courses by a simple "homogeneity score"
  # (lower is more homogeneous).

Notes:
- We overlay 4 house histograms (same bin edges) for each course.
- We also compute a simple homogeneity metric: the average variance across houses
  of per-bin normalized frequencies. Lower means more similar distributions.
- We avoid "describe" heavy-lifting; but for plotting we rely on matplotlib.
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

def configure_matplotlib_backend(wants_show: bool) -> bool:
    """
    Configure a safe Matplotlib backend across macOS/Linux.

    Returns:
        bool: True if interactive showing is enabled (plt.show() will be attempted),
              False if we fell back to a non-interactive backend (Agg).
    Notes:
        - On headless Linux (no DISPLAY), we force 'Agg' and disable interactive show.
        - When --show is not requested, we force 'Agg' to avoid any GUI requirements.
    """
    try:
        import matplotlib
        try:
            import matplotlib.backends
        except Exception:
            pass
    except Exception:
        # If matplotlib is not installed, the caller will hit the import error later.
        return False

    if not wants_show:
        try:
            matplotlib.use("Agg", force=True)
        except Exception:
            pass
        return False

    # wants_show == True
    try:
        import sys as _sys
        # On macOS, prefer the native Cocoa backend and do NOT require DISPLAY.
        if _sys.platform == "darwin":
            try:
                matplotlib.use("MacOSX", force=True)
            except Exception:
                # If MacOSX backend isn't available, keep default and try to show anyway.
                pass
            return True

        # On Linux servers without X (no DISPLAY), interactive backends won't work.
        if os.name != "nt" and not os.environ.get("DISPLAY"):
            matplotlib.use("Agg", force=True)
            return False

        # Otherwise (Windows or desktop Linux with DISPLAY), allow interactive showing.
        return True
    except Exception:
        # Any issue configuring -> fall back to non-interactive
        try:
            matplotlib.use("Agg", force=True)
        except Exception:
            pass
        return False

# --- Constants ---
EXCLUDE_COLUMNS = {"Hogwarts House", "Index"}
HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
MISSING_TOKENS = {"", "na", "nan", "null", "none", "NaN", "NA", "NULL", "None"}

# Canonical Hogwarts house colors (Option 1)
HOUSE_COLORS = {
    "Gryffindor": "#AE0001",   # Scarlet
    "Hufflepuff": "#FFD800",   # Yellow
    "Ravenclaw":  "#222F5B",   # Blue
    "Slytherin":  "#2A623D",   # Green
}

# --- CLI parsing ---

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="Path to dataset_train.csv")
    ap.add_argument("--bins", type=int, default=20, help="Number of bins per histogram (default: 20)")
    ap.add_argument("--outdir", default="outputs/figures", help="Directory to save figures")
    ap.add_argument("--show", action="store_true", help="Also show plots interactively")
    return ap.parse_args()

# --- CSV / parsing helpers ---

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
        all_numeric_or_missing = True
        for r in rows:
            v = r.get(h, "")
            if is_missing(v):
                continue
            if try_float(v) is None:
                all_numeric_or_missing = False
                break
        if all_numeric_or_missing:
            numeric.append(h)
    return numeric

def group_values_by_house(rows, feature):
    """
    Returns dict: house -> list[float] for given feature.
    Ignores missing/non-parsable values or rows without a valid house.
    """
    by_house = {h: [] for h in HOUSES}
    for r in rows:
        house = r.get("Hogwarts House", "")
        if house not in by_house:
            continue
        val_raw = r.get(feature, "")
        if is_missing(val_raw):
            continue
        val = try_float(val_raw)
        if val is not None:
            by_house[house].append(val)
    return by_house

# --- Simple numeric utilities (manual-ish) ---

def safe_min(values):
    m = values[0]
    for x in values[1:]:
        if x < m:
            m = x
    return m

def safe_max(values):
    m = values[0]
    for x in values[1:]:
        if x > m:
            m = x
    return m

def make_linspace(start, stop, num):
    """
    Manual-ish linspace: inclusive of start and stop, returns 'num+1' edges when used for bins.
    We’ll use it so bins = make_linspace(min, max, bins) produces len = bins+1 edges.
    """
    if num <= 0:
        return [start, stop]
    step = (stop - start) / float(num)
    out = [start + i * step for i in range(num + 1)]
    # ensure last is exactly stop to avoid FP drift
    out[-1] = stop
    return out

def histogram_counts(values, bin_edges):
    """
    Manual histogram counts per bin given bin_edges.
    - bin_edges length = B+1 for B bins.
    - We count value x into bin i if edges[i] <= x < edges[i+1], except the last bin is inclusive.
    """
    B = len(bin_edges) - 1
    counts = [0] * B
    if not values:
        return counts
    lo = bin_edges[0]
    hi = bin_edges[-1]
    for x in values:
        if x < lo:
            continue
        if x > hi:
            # If numerically slightly above hi, put it in the last bin.
            counts[-1] += 1
            continue
        # find bin by linear search (B is small so it’s ok)
        placed = False
        for i in range(B - 1):
            if bin_edges[i] <= x < bin_edges[i + 1]:
                counts[i] += 1
                placed = True
                break
        if not placed:
            # Either x is exactly hi or at last edge due to FP -> last bin
            counts[-1] += 1
    return counts

def normalize_counts(counts):
    total = 0
    for c in counts:
        total += c
    if total == 0:
        return [0.0 for _ in counts]
    return [c / float(total) for c in counts]

def variance(values):
    # simple population variance here for our homogeneity metric (no need for ddof=1)
    n = len(values)
    if n == 0:
        return 0.0
    mu = sum(values) / n
    acc = 0.0
    for v in values:
        d = v - mu
        acc += d * d
    return acc / n

# --- Homogeneity metric ---

def homogeneity_score(house_density_by_bin):
    """
    Input: list of lists with shape (H, B) -> H houses, B bins, each row sums to 1 (or 0 if empty).
    We compute, for each bin b, the variance across houses of the densities at that bin.
    Then average across bins. Lower = more similar distributions.
    """
    if not house_density_by_bin:
        return math.inf
    H = len(house_density_by_bin)
    if H == 0:
        return math.inf
    B = len(house_density_by_bin[0]) if H > 0 else 0
    if B == 0:
        return math.inf

    # transpose: per-bin arrays across houses
    avg_var = 0.0
    for b in range(B):
        col = [house_density_by_bin[h][b] for h in range(H)]
        avg_var += variance(col)
    avg_var /= B
    return avg_var

# --- Plotting ---

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def plot_hist_per_course(course, by_house, bin_edges, outdir, show=False):
    """
    Overlay 4 house histograms for a single course using the same bin edges.
    Saves as outputs/figures/hist_<course>.png
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    # Build densities for each house
    densities = {}
    for house, vals in by_house.items():
        counts = histogram_counts(vals, bin_edges)
        dens = normalize_counts(counts)
        densities[house] = dens
        # For visualization: plot as stepped histogram (approximate) by repeating edges and values
        # We'll multiply density by total count just for visual scale with 'step' style,
        # but to keep it intuitive we can plot the density itself as a bar-like step using width.
        # Simpler: use plt.hist for visibility only, but we compute binning ourselves already.
        # Here, we’ll just plot as a line over bin centers.

    # Compute bin centers for plotting lines
    centers = []
    for i in range(len(bin_edges) - 1):
        centers.append(0.5 * (bin_edges[i] + bin_edges[i + 1]))

    # Plot one line per house
    for house in HOUSES:
        vals = by_house.get(house, [])
        counts = histogram_counts(vals, bin_edges)
        dens = normalize_counts(counts)
        plt.plot(
            centers,
            dens,
            marker="o",
            linewidth=1.5,
            label=house,
            alpha=0.9,
            color=HOUSE_COLORS.get(house)
        )

    plt.title(f"Histogram (density) per house — {course}")
    plt.xlabel(course)
    plt.ylabel("Density per bin")
    plt.legend()
    plt.tight_layout()

    ensure_dir(outdir)
    safe_name = course.replace("/", "_").replace(" ", "_")
    out_path = os.path.join(outdir, f"hist_{safe_name}.png")
    plt.savefig(out_path, dpi=160)
    if show:
        try:
            plt.show()
        except Exception:
            # Fall back silently: image already saved to disk
            pass
    plt.close()

def main():
    args = parse_args()
    ensure_dir(args.outdir)
    show_enabled = configure_matplotlib_backend(args.show)
    headers, rows = read_csv_rows(args.csv_path)
    if not headers or not rows:
        print("Empty CSV or missing headers.")
        return

    numeric_features = detect_numeric_features(headers, rows)
    if not numeric_features:
        print("No numeric features found.")
        return

    # For each course/feature, group by house and build shared bin edges
    results = []  # (course, score)
    for course in numeric_features:
        by_house = group_values_by_house(rows, course)

        # Gather all values to define a shared min/max for bins
        all_vals = []
        for vals in by_house.values():
            all_vals.extend(vals)
        if len(all_vals) == 0:
            continue

        # min/max per course
        mn = safe_min(all_vals)
        mx = safe_max(all_vals)
        if mx == mn:
            # constant column, bins would degenerate; make a small padding
            eps = 1e-9
            mn -= eps
            mx += eps

        bin_edges = make_linspace(mn, mx, args.bins)

        # Compute per-house densities (for homogeneity score)
        house_density_by_bin = []
        for house in HOUSES:
            vals = by_house.get(house, [])
            counts = histogram_counts(vals, bin_edges)
            dens = normalize_counts(counts)
            house_density_by_bin.append(dens)

        score = homogeneity_score(house_density_by_bin)
        results.append((course, score))

        # Plot per course
        plot_hist_per_course(course, by_house, bin_edges, args.outdir, show=show_enabled)

    # Rank courses by homogeneity (lower is better)
    results.sort(key=lambda x: x[1])
    print("\n=== Homogeneity ranking (lower is more homogeneous) ===")
    for rank, (course, score) in enumerate(results, 1):
        print(f"{rank:2d}. {course}: {score:.6f}")

    if results:
        best_course, best_score = results[0]
        print(f"\n👉 Most homogeneous (by this metric): {best_course} (score={best_score:.6f})")
        print(f"See per-course histograms in: {args.outdir}")

if __name__ == "__main__":
    main()