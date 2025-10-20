"""
Histogram — Which Hogwarts course has a homogeneous score distribution between all four houses?

Usage:
  # Save PNGs (no GUI)
  python src/histogram.py data/dataset_train.csv

  # Single interactive window (←/h prev, →/l next, s save, a save all, q/Esc quit)
  python src/histogram.py data/dataset_train.csv --show

  # Legacy behavior: one window per course (requires closing each)
  python src/histogram.py data/dataset_train.csv --show --multi-windows

Notes:
- Overlays 4 house histograms (same bin edges) for each course.
- Computes a simple homogeneity metric: average per-bin variance across houses of normalized frequencies.
- Cross-platform backend handling: GUI on macOS/desktop Linux, save-only on headless servers.
"""

import argparse
import csv
import math
import os
import sys

# ---------------- Backend configuration ----------------

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
        # If matplotlib isn't installed, import will fail later in plotting.
        return False

    if not wants_show:
        try:
            matplotlib.use("Agg", force=True)
        except Exception:
            pass
        return False

    try:
        # On macOS, prefer native Cocoa and do NOT require DISPLAY.
        if sys.platform == "darwin":
            try:
                matplotlib.use("MacOSX", force=True)
            except Exception:
                pass
            return True

        # On Linux servers without X, interactive backends won't work.
        if os.name != "nt" and not os.environ.get("DISPLAY"):
            matplotlib.use("Agg", force=True)
            return False

        return True  # desktop Linux/Windows with GUI
    except Exception:
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
        except Exception:
            pass
        return False

# ---------------- Constants ----------------

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

# ---------------- CLI ----------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="Path to dataset_train.csv")
    ap.add_argument("--bins", type=int, default=20, help="Number of bins per histogram (default: 20)")
    ap.add_argument("--outdir", default="outputs/figures", help="Directory to save figures")
    ap.add_argument("--show", action="store_true", help="Show a single interactive window (use arrows to change course)")
    ap.add_argument("--multi-windows", action="store_true", help="Legacy mode: open one window per course")
    return ap.parse_args()

# ---------------- CSV helpers ----------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

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
        v = r.get(feature, "")
        if is_missing(v):
            continue
        f = try_float(v)
        if f is not None:
            by_house[house].append(f)
    return by_house

# ---------------- Numeric helpers ----------------

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
    Inclusive start/stop; returns num+1 edges for num bins.
    """
    if num <= 0:
        return [start, stop]
    step = (stop - start) / float(num)
    out = [start + i * step for i in range(num + 1)]
    out[-1] = stop
    return out

def histogram_counts(values, bin_edges):
    """
    Manual histogram counts per bin given bin_edges (B+1 edges -> B bins).
    Bin rule: [edge_i, edge_{i+1}) except the last bin is inclusive.
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
            counts[-1] += 1
            continue
        placed = False
        for i in range(B - 1):
            if bin_edges[i] <= x < bin_edges[i + 1]:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    return counts

def normalize_counts(counts):
    tot = sum(counts)
    if tot == 0:
        return [0.0] * len(counts)
    return [c / float(tot) for c in counts]

def variance(values):
    # population variance (sufficient for our homogeneity metric)
    n = len(values)
    if n == 0:
        return 0.0
    mu = sum(values) / n
    acc = 0.0
    for v in values:
        d = v - mu
        acc += d * d
    return acc / n

# ---------------- Homogeneity metric ----------------

def homogeneity_score(house_density_by_bin):
    """
    Input: list of lists with shape (H, B).
    For each bin b, compute variance across houses of densities; average across bins.
    Lower = more similar distributions.
    """
    if not house_density_by_bin:
        return math.inf
    H = len(house_density_by_bin)
    if H == 0:
        return math.inf
    B = len(house_density_by_bin[0]) if H > 0 else 0
    if B == 0:
        return math.inf

    avg_var = 0.0
    for b in range(B):
        col = [house_density_by_bin[h][b] for h in range(H)]
        avg_var += variance(col)
    avg_var /= B
    return avg_var

# ---------------- Precompute & drawing ----------------

def precompute_course_data(rows, features, bins):
    """
    Returns a list of dicts, one per course:
      { 'course': str, 'by_house': dict[house -> list[float]], 'edges': list[float], 'centers': list[float] }
    """
    data = []
    for course in features:
        by_house = group_values_by_house(rows, course)
        all_vals = []
        for vals in by_house.values():
            all_vals.extend(vals)
        if len(all_vals) == 0:
            continue
        mn = safe_min(all_vals)
        mx = safe_max(all_vals)
        if mx == mn:
            eps = 1e-9
            mn -= eps
            mx += eps
        edges = make_linspace(mn, mx, bins)
        centers = [(edges[i] + edges[i+1]) * 0.5 for i in range(len(edges) - 1)]
        data.append({"course": course, "by_house": by_house, "edges": edges, "centers": centers})
    return data

def draw_course_on_axes(ax, item):
    """
    Draw one course as overlaid bar histograms (same bin edges for all houses),
    with transparent fills and colored outlines — like the screenshot.
    """
    import matplotlib.pyplot as plt  # local import for safety
    ax.clear()
    course = item["course"]
    by_house = item["by_house"]
    edges = item["edges"]

    # Draw bars per house
    for house in HOUSES:
        vals = by_house.get(house, [])
        if not vals:
            continue
        ax.hist(
            vals,
            bins=edges,            # use precomputed common edges
            density=True,          # density to compare shapes
            alpha=0.35,            # transparent fill
            color=HOUSE_COLORS.get(house),
            edgecolor=HOUSE_COLORS.get(house),
            linewidth=1.4,
            label=house
        )

    ax.set_title(f"Score Distribution for {course}")
    ax.set_xlabel(course)
    ax.set_ylabel("Density")
    ax.legend(title="Hogwarts House", loc="best")
    ax.figure.tight_layout()

def save_current_course_png(fig, ax, item, outdir):
    ensure_dir(outdir)
    safe_name = item["course"].replace("/", "_").replace(" ", "_")
    out_path = os.path.join(outdir, f"hist_{safe_name}.png")
    fig.savefig(out_path, dpi=160)
    print(f"Saved: {out_path}")

# ---------------- Interactive navigator (single window) ----------------

def interactive_hist_navigator(courses_data, outdir):
    """
    Launch one interactive window that lets you flip through courses with keyboard:
      - Left/Right arrows (or h/l): previous/next course
      - s: save current course PNG
      - a: save all courses' PNGs
      - q or Esc: quit
    """
    import matplotlib.pyplot as plt

    if not courses_data:
        print("No courses to display.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    idx = 0

    # initial draw
    draw_course_on_axes(ax, courses_data[idx])

    # instructions overlay
    help_txt = ("←/h prev | →/l next | s save | a save all | q/Esc quit")
    fig.text(0.01, 0.01, help_txt, fontsize=8, ha="left", va="bottom", alpha=0.7)

    def on_key(event):
        nonlocal idx
        if event.key in ("right", "l"):
            idx = (idx + 1) % len(courses_data)
            draw_course_on_axes(ax, courses_data[idx])
            fig.canvas.draw_idle()
        elif event.key in ("left", "h"):
            idx = (idx - 1) % len(courses_data)
            draw_course_on_axes(ax, courses_data[idx])
            fig.canvas.draw_idle()
        elif event.key == "s":
            save_current_course_png(fig, ax, courses_data[idx], outdir)
        elif event.key == "a":
            # save all; redraw current after
            for item in courses_data:
                draw_course_on_axes(ax, item)
                fig.canvas.draw_idle()
                save_current_course_png(fig, ax, item, outdir)
            draw_course_on_axes(ax, courses_data[idx])
            fig.canvas.draw_idle()
        elif event.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

# ---------------- Legacy per-course plotting (kept for --multi-windows) ----------------

def plot_hist_per_course(course, by_house, bin_edges, outdir, show=False):
    """
    Overlay 4 house histograms for a single course (bar/column style),
    using identical bin edges for fair comparison.
    Saves as outputs/figures/hist_<course>.png
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    for house in HOUSES:
        vals = by_house.get(house, [])
        if not vals:
            continue
        ax.hist(
            vals,
            bins=bin_edges,
            density=True,
            alpha=0.35,
            color=HOUSE_COLORS.get(house),
            edgecolor=HOUSE_COLORS.get(house),
            linewidth=1.4,
            label=house
        )

    ax.set_title(f"Score Distribution for {course}")
    ax.set_xlabel(course)
    ax.set_ylabel("Density")
    ax.legend(title="Hogwarts House", loc="best")
    fig.tight_layout()

    ensure_dir(outdir)
    safe_name = course.replace("/", "_").replace(" ", "_")
    out_path = os.path.join(outdir, f"hist_{safe_name}.png")
    fig.savefig(out_path, dpi=160)

    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)

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

    # Precompute once for speed / interactivity
    courses_data = precompute_course_data(rows, numeric_features, args.bins)

    # Compute homogeneity ranking (independent of UI choice)
    results = []
    for item in courses_data:
        edges = item["edges"]
        by_house = item["by_house"]
        house_density_by_bin = []
        for house in HOUSES:
            vals = by_house.get(house, [])
            counts = histogram_counts(vals, edges)
            dens = normalize_counts(counts)
            house_density_by_bin.append(dens)
        score = homogeneity_score(house_density_by_bin)
        results.append((item["course"], score))

    results.sort(key=lambda x: x[1])
    print("\n=== Homogeneity ranking (lower is more homogeneous) ===")
    for rank, (course, score) in enumerate(results, 1):
        print(f"{rank:2d}. {course}: {score:.6f}")
    if results:
        best_course, best_score = results[0]
        print(f"\n👉 Most homogeneous (by this metric): {best_course} (score={best_score:.6f})")
        print(f"Figures save to: {args.outdir}")

    # UI behavior
    if args.show and show_enabled and not args.multi_windows:
        # One interactive window (keyboard navigation)
        interactive_hist_navigator(courses_data, args.outdir)
    else:
        # Legacy behavior: save + show each (if requested)
        for item in courses_data:
            # Save + optional show
            plot_hist_per_course(
                item["course"], item["by_house"], item["edges"], args.outdir,
                show=(args.show and show_enabled)
            )

if __name__ == "__main__":
    main()

"""
coisas para mudar: mudar para formato pilar (talvez fazer formato ponto e pilar ao mesmo tempo como o da jisu)

"""