#!/usr/bin/env python3
"""
Pair plot — scatter matrix of selected features colored by Hogwarts House.

Usage:
  # Auto-pick up to 6 highest-variance numeric features, save + show if possible
  python src/pair_plot.py data/dataset_train.csv --show

  # Explicit subset in order
  python src/pair_plot.py data/dataset_train.csv --features "Astronomy,Defense Against the Dark Arts,Herbology"

  # Change the cap of auto-selected features
  python src/pair_plot.py data/dataset_train.csv --max 5
"""

import argparse
import csv
import math
import os
import sys

# ---------------- Constants ----------------
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

# ---------------- Backend & FS helpers (same pattern as other scripts) ----------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def configure_matplotlib_backend(wants_show: bool) -> bool:
    """
    Configure a safe Matplotlib backend across macOS/Linux.

    Returns:
        True if interactive showing is enabled (plt.show() will be attempted),
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
        if sys.platform == "darwin":
            try:
                matplotlib.use("MacOSX", force=True)
            except Exception:
                pass
            return True

        if os.name != "nt" and not os.environ.get("DISPLAY"):
            matplotlib.use("Agg", force=True)
            return False

        return True
    except Exception:
        try:
            matplotlib.use("Agg", force=True)
        except Exception:
            pass
        return False

# ---------------- CSV / parsing helpers ----------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="Path to dataset_train.csv")
    ap.add_argument("--features", help='Comma-separated feature names to include, e.g. "Astronomy,Herbology"', default=None)
    ap.add_argument("--max", dest="max_features", type=int, default=6, help="Max number of auto-selected features by variance (default: 6)")
    ap.add_argument("--outdir", default="outputs/figures", help="Directory to save the figure")
    ap.add_argument("--outfile", default=None, help="Output filename (PNG). Default is auto-generated from feature names.")
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

# ---------------- Small numeric helpers (manual) ----------------

def kahan_sum(values):
    total = 0.0
    c = 0.0
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
    return kahan_sum(values) / n

def variance(values):
    n = len(values)
    if n == 0:
        return float("nan")
    mu = mean(values)
    acc = 0.0
    for v in values:
        d = v - mu
        acc += d * d
    # population variance (sufficient for ranking by spread)
    return acc / n

# ---------------- Data extraction ----------------

def collect_feature_values(rows, feature):
    """Return list[float] for a single feature, ignoring missing."""
    out = []
    for r in rows:
        v = r.get(feature, "")
        if is_missing(v):
            continue
        f = try_float(v)
        if f is not None:
            out.append(f)
    return out

def collect_xy_by_house(rows, fx, fy):
    """Return dicts per house: x-values and y-values filtered & aligned."""
    by_house_x = {h: [] for h in HOUSES}
    by_house_y = {h: [] for h in HOUSES}
    for r in rows:
        vx = r.get(fx, "")
        vy = r.get(fy, "")
        if is_missing(vx) or is_missing(vy):
            continue
        fxv = try_float(vx)
        fyv = try_float(vy)
        if fxv is None or fyv is None:
            continue
        house = r.get("Hogwarts House", "")
        if house in by_house_x:
            by_house_x[house].append(fxv)
            by_house_y[house].append(fyv)
    return by_house_x, by_house_y

def collect_hist_per_house(rows, feature):
    """Return dict house->list[float] for diagonal histograms."""
    by_house = {h: [] for h in HOUSES}
    for r in rows:
        v = r.get(feature, "")
        if is_missing(v):
            continue
        f = try_float(v)
        if f is None:
            continue
        house = r.get("Hogwarts House", "")
        if house in by_house:
            by_house[house].append(f)
    return by_house

# ---------------- Plotting (matplotlib, no seaborn) ----------------

def pair_plot(features, rows, outdir, outfile=None, show=False, bins=20):
    """
    Draw a scatter-matrix:
      - diagonal: per-house histograms (densities)
      - off-diagonal: per-house scatter plots
      - optimized for compact layout (small markers, tight grid)
    """
    import matplotlib.pyplot as plt

    n = len(features)
    if n < 1:
        print("No features to plot.")
        return None

    # ↓↓↓ Compact figure scaling ↓↓↓
    fig_w = max(6.0, 1.2 * n)   # was 2.2 * n
    fig_h = max(6.0, 1.2 * n)
    fig, axes = plt.subplots(nrows=n, ncols=n, figsize=(fig_w, fig_h))

    # Ensure 2D array even for n=1
    if n == 1:
        axes = [[axes]]

    house_labels = HOUSES

    # Precompute global min/max per feature
    feature_minmax = {}
    for f in features:
        values = collect_feature_values(rows, f)
        if not values:
            mn, mx = 0.0, 1.0
        else:
            mn = min(values)
            mx = max(values)
            if mx == mn:
                eps = 1e-9
                mn -= eps
                mx += eps
        feature_minmax[f] = (mn, mx)

    def make_edges(mn, mx, b):
        step = (mx - mn) / float(b)
        edges = [mn + i * step for i in range(b + 1)]
        edges[-1] = mx
        return edges

    def hist_counts(values, edges):
        B = len(edges) - 1
        counts = [0] * B
        if not values:
            return counts
        lo, hi = edges[0], edges[-1]
        for x in values:
            if x < lo:
                continue
            if x > hi:
                counts[-1] += 1
                continue
            for i in range(B - 1):
                if edges[i] <= x < edges[i + 1]:
                    counts[i] += 1
                    break
            else:
                counts[-1] += 1
        return counts

    def to_density(counts):
        tot = sum(counts)
        if tot == 0:
            return [0.0] * len(counts)
        return [c / float(tot) for c in counts]

    # Build the grid
    for i, yi in enumerate(features):
        for j, xj in enumerate(features):
            ax = axes[i][j]
            if i == j:
                # Diagonal: per-house histogram density lines
                mn, mx = feature_minmax[yi]
                edges = make_edges(mn, mx, bins)
                centers = [(edges[k] + edges[k+1]) * 0.5 for k in range(len(edges) - 1)]
                by_house = collect_hist_per_house(rows, yi)
                for house in house_labels:
                    vals = by_house[house]
                    counts = hist_counts(vals, edges)
                    dens = to_density(counts)
                    ax.plot(
                        centers, dens,
                        marker="",
                        linewidth=0.9,
                        alpha=0.8,
                        color=HOUSE_COLORS.get(house),
                    )
                if j == n - 1:
                    ax.legend(fontsize=6, loc="upper right", frameon=False)
            else:
                # Off-diagonal: scatter with smaller points & low alpha
                by_house_x, by_house_y = collect_xy_by_house(rows, xj, yi)
                for house in house_labels:
                    ax.scatter(
                        by_house_x[house],
                        by_house_y[house],
                        s=4,           # smaller point size
                        alpha=0.5,     # more transparency
                        linewidths=0,
                        color=HOUSE_COLORS.get(house)
                    )
            # Axis labeling every edge only
            if i == n - 1:
                ax.set_xlabel(xj, fontsize=7)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(yi, fontsize=7)
            else:
                ax.set_yticklabels([])

            ax.tick_params(axis="both", which="both", labelsize=6, length=1.5)

    plt.tight_layout(pad=0.5, w_pad=0.2, h_pad=0.2)

    ensure_dir(outdir)
    if outfile is None:
        safe = "__".join(f.replace("/", "_").replace(" ", "_") for f in features)
        outfile = f"pairplot__{safe}.png"
    out_path = os.path.join(outdir, outfile)
    fig.savefig(out_path, dpi=160)

    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)
    return out_path

# ---------------- Feature selection ----------------

def pick_features(headers, rows, numeric_features, explicit_list, max_features):
    """
    If explicit_list is provided, validate and return them (in given order).
    Otherwise, pick up to max_features with highest variance.
    """
    if explicit_list:
        chosen = []
        for name in explicit_list:
            name = name.strip()
            if name not in numeric_features:
                raise ValueError(f"--features requested '{name}' is not a numeric feature. Available: {numeric_features}")
            chosen.append(name)
        return chosen

    # Auto-pick by variance
    scored = []
    for f in numeric_features:
        vals = collect_feature_values(rows, f)
        v = variance(vals) if vals else float("-inf")
        scored.append((v, f))
    # Highest variance first
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    chosen = [f for _, f in scored[:max_features]]
    return chosen

# ---------------- Main ----------------

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

    explicit_list = None
    if args.features:
        explicit_list = [s for s in args.features.split(",") if s.strip()]

    # >>> ADD THIS BLOCK <<<
    # If user asked to show and didn't specify features, plot ALL numeric features
    if show_enabled and not explicit_list:
        features = numeric_features  # all of them
    else:
        try:
            features = pick_features(headers, rows, numeric_features, explicit_list, args.max_features)
        except ValueError as e:
            print(f"Error: {e}")
            return
    # <<< END ADD >>>

    if len(features) == 0:
        print("No features selected.")
        return

    print(f"Using features ({len(features)}): {features}")
    out_path = pair_plot(features, rows, args.outdir, outfile=args.outfile, show=show_enabled)
    if out_path:
        print(f"Saved pair plot to: {out_path}")


if __name__ == "__main__":
    main()