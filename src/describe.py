import csv
import math
import argparse
import sys

EXCLUDE_COLUMNS = {"Hogwarts House", "Index"}

MISSING_TOKENS = {"", "na", "nan", "null", "none", "NaN", "NA", "NULL", "None"}

def parse_arg():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help=("Path to dataset CSV (eg., data/dataset_train.csv)"))
    return ap.parse_args()

def is_missing(s: str) -> bool:
    return s.strip() in MISSING_TOKENS

def try_parse_float(s: str):
    try:
        return float(s)
    except Exception:
        return None

def is_numeric_column(values):
    """
    Decide is a colum in numeric:
    - True if all NON_MISSING entries parse as float.
    - False otherwise
    """
    for v in values:
        if is_missing(v):
            continue
        if try_parse_float(v) is None:
            return False
    return True

def to_float_list(values):
    """Return list[float] with missing filtered out."""
    out = []
    for v in values:
        if is_missing(v):
            continue
        fv = try_parse_float(v)
        if fv is not None:
            out.append(fv)
    return out

def safe_min(values):
    # manual min
    m = values[0]
    for x in values[1:]:
        if x < m:
            m = x
    return m 

def safe_max(values):
    #manual max
    m = values[0]
    for x in values[1:]:
        if x < m:
            m = x
    return m

def safe_sum(values):
    total = 0.0
    c = 0.0
    for x in values:
        y = x - c
        t = total + y
        c = (t - total) - y
        total = y
    return total

def mean(values):
    return safe_sum(values) / len(values) if values else float("nan")

def variance_sample(values, mu):
    """
    Sample variance (ddof=1): sum((x - mu)^2) / (n - 1), for n >= 2
    If n < 2, return 0.0 by convention here.
    """
    n = len(values)
    if n < 2:
        return 0.0
    acc = 0.0
    for x in values:
        d = x - mu
        acc += d * d
    return acc / (n - 1)

def percentile_linear(values_sorted, p):
    """
    Linear interpolation at index = p*(n-1), p in [0,1], n>=1.
    If n==0 caller should not call this.
    """
    n = len(values_sorted)
    if n == 1:
        return values_sorted[0]
    idx = p * (n - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return values_sorted[lo]
    frac = idx - lo
    return values_sorted[lo] + frac * (values_sorted[hi] - values_sorted[lo])

def describe_column(values_float):
    """
    Compute Count, Mean, Std (sample), Min, 25%, 50%, 75%, Max for a float list.
    Assumes values_float is non-empty.
    """
    cnt = len(values_float)
    mu = mean(values_float)
    var = variance_sample(values_float, mu)
    std = math.sqrt(var)

    sorted_vals = sorted(values_float)
    mn = sorted_vals[0]
    mx = sorted_vals[-1]
    q25 = percentile_linear(sorted_vals, 0.25)
    q50 = percentile_linear(sorted_vals, 0.50)
    q75 = percentile_linear(sorted_vals, 0.75)

    return {
        "Count": float(cnt),
        "Mean": mu,
        "Std": std,
        "Min": mn,
        "25%": q25,
        "50%": q50,
        "75%": q75,
        "Max": mx,
    }

def format_number(x):
    # match the sample style: 6 decimals for everything
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)

def main():
    args = parse_arg()

    # Read CSV
    try:
        with open(args.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            rows = list(reader)
    except FileNotFoundError:
        print(f"Error: file not found: {args.csv_path}", file=sys.stderr)
        sys.exit(1)

    if not headers or not rows:
        print("Error: empty CSV or no headers.", file=sys.stderr)
        sys.exit(1)

    # Collect column -> raw strings
    col_values = {h: [] for h in headers}
    for row in rows:
        for h in headers:
            col_values[h].append(row.get(h, ""))

    # Determine numeric features (exclude known non-features)
    numeric_headers = []
    for h in headers:
        if h in EXCLUDE_COLUMNS:
            continue
        if is_numeric_column(col_values[h]):
            numeric_headers.append(h)

    if not numeric_headers:
        print("No numeric features detected.", file=sys.stderr)
        sys.exit(1)

    # Compute stats per numeric column
    stats_per_col = {}
    for h in numeric_headers:
        vals = to_float_list(col_values[h])
        if len(vals) == 0:
            continue  # skip entirely empty numeric columns
        stats_per_col[h] = describe_column(vals)

    if not stats_per_col:
        print("No numeric data to describe.", file=sys.stderr)
        sys.exit(1)

    # Prepare output table:
    # Rows in this order; columns are numeric_headers in CSV order
    metrics_order = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]

    # Header line
    print(" " * 8, end="")
    for h in numeric_headers:
        print(f"{h:>16}", end="")
    print()

    # Each metric row
    for metric in metrics_order:
        print(f"{metric:<8}", end="")
        for h in numeric_headers:
            val = stats_per_col[h][metric] if h in stats_per_col else float("nan")
            print(f"{format_number(val):>16}", end="")
        print()

if __name__ == "__main__":
    main()