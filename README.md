# dslr
# 🧙‍♀️ DSLR — Data Science & Logistic Regression (Hogwarts Sorting Hat)

This project recreates the famous **Hogwarts Sorting Hat** using **Logistic Regression (one-vs-all)** trained on students’ course grades.

You’ll explore data, visualize patterns, and build a classifier to predict which house (Gryffindor, Hufflepuff, Ravenclaw, or Slytherin) a student belongs to — just like the Sorting Hat itself.

---

## 🪄 Project Structure

```
dslr/
├── data/
│   ├── dataset_train.csv
│   ├── dataset_test.csv
├── src/
│   ├── describe.py
│   ├── histogram.py
│   ├── scatter_plot.py
│   ├── pair_plot.py
│   ├── logreg_train.py
│   ├── logreg_predict.py
├── outputs/
│   └── figures/
└── README.md
```

---

## 🧠 Project Goal

To predict the Hogwarts House of a student based on their grades, using your own **implementation** of logistic regression (without scikit-learn or high-level functions).

You’ll:
1. Perform **data analysis** and descriptive statistics.
2. Create **visualizations** to understand patterns.
3. Implement **logistic regression** and train your model.
4. Evaluate predictions on unseen data.

---

## 📊 Data Visualization Scripts

### 1️⃣ `describe.py`
**Purpose:**  
Manually compute key statistics for each numerical feature without using built-in pandas/numpy functions like `.mean()` or `.describe()`.

**Computed metrics:**
- Count  
- Mean  
- Standard deviation  
- Min / Max  
- 25%, 50%, 75% percentiles  

**Usage:**
```bash
python3 src/describe.py data/dataset_train.csv
```

---

### 2️⃣ `histogram.py`
**Purpose:**  
Visualize score distributions for each course across the four houses.

Each plot overlays the four houses using Hogwarts-themed colors:
- Gryffindor → Scarlet `#AE0001`
- Hufflepuff → Yellow `#FFD800`
- Ravenclaw → Blue `#222F5B`
- Slytherin → Green `#2A623D`

Also computes a simple **homogeneity score** — lower values mean more similar distributions across houses.

**Usage:**
```bash
# Generate one histogram per course (20 bins by default)
python3 src/histogram.py data/dataset_train.csv

# Show interactively (macOS/Linux-safe)
python3 src/histogram.py data/dataset_train.csv --show

# Change bin count
python3 src/histogram.py data/dataset_train.csv --bins 30
```

**Homogeneity metric logic:**
- For each course, compute per-house normalized frequency (density) histograms.
- Measure the average variance across houses for each bin.
- Lower = more homogeneous = houses perform similarly.

---

### 3️⃣ `scatter_plot.py`
**Purpose:**  
Identify the two most similar features (by absolute Pearson correlation |r|) and visualize their relationship.

**What the output shows:**
- Each point = one student.
- Color = their Hogwarts House.
- The title shows the **correlation coefficient r** and the number of samples.

**Pearson correlation coefficient:**
$\[
r = \frac{cov(X, Y)}{\sqrt{var(X) \cdot var(Y)}}
\]$

- **r = +1:** Perfect positive correlation (one increases → the other increases)
- **r = -1:** Perfect negative correlation (one increases → the other decreases)
- **r = 0:** No linear correlation

**Usage:**
```bash
# Auto-select most similar pair
python3 src/scatter_plot.py data/dataset_train.csv --show

# Show top 10 most correlated pairs
python3 src/scatter_plot.py data/dataset_train.csv --rank 10

# Specify pair manually
python3 src/scatter_plot.py data/dataset_train.csv --x "Astronomy" --y "Defense Against the Dark Arts" --show
```

**Interpreting the scatter plot:**
- **Diagonal line (positive slope):** features increase together.
- **Downward slope:** one increases, the other decreases (negative correlation).
- **Clustered colors:** good feature separation between houses.
- **Overlapping colors:** less predictive power.

---

### 4️⃣ `pair_plot.py`
**Purpose:**  
Display all pairwise relationships between selected features — like a grid of scatter plots (off-diagonal) and histograms (diagonal).

**Why it matters:**
- Helps you visually spot correlated or redundant features.
- Highlights which features separate houses well (useful for model input selection).

**Usage:**
```bash
# Auto-pick up to 6 highest-variance features
python3 src/pair_plot.py data/dataset_train.csv --show

# Choose specific features
python3 src/pair_plot.py data/dataset_train.csv --features "Astronomy,Defense Against the Dark Arts,Herbology"

# Change number of features auto-selected
python3 src/pair_plot.py data/dataset_train.csv --max 5
```

**Interpreting the pair plot:**

| Plot type | Meaning |
|------------|----------|
| Diagonal | Histogram per course per house — differences indicate predictive power |
| Off-diagonal | Scatter of two courses — patterns show correlation |
| Distinct color clusters | Houses separate well → good features for classification |
| Overlap | Poor separation → less useful |

---

## 🧮 Mathematical Foundations

### Logistic Regression Hypothesis:
$\[
h_θ(x) = g(θ^T x) = \frac{1}{1 + e^{-θ^T x}}
\]$

### Cost Function:
$\[
J(θ) = -\frac{1}{m}\sum_i [y_i \log(h_θ(x_i)) + (1 - y_i)\log(1 - h_θ(x_i))]
\]$

### Gradient:
$\[
\frac{∂J(θ)}{∂θ_j} = \frac{1}{m}\sum_i (h_θ(x_i) - y_i)x_i^j
\]$

### Gradient Descent Update Rule:
$\[
θ := θ - α \frac{∂J(θ)}{∂θ}
\]$

Where:
- \( α \) = learning rate  
- \( θ \) = model parameters  
- \( m \) = number of samples  

---

## 📈 Model Training and Prediction (coming next)

| Script | Purpose |
|---------|----------|
| `logreg_train.py` | Train the logistic regression (one-vs-all) and save weights |
| `logreg_predict.py` | Load weights and predict houses for new students |

These will build on your visual and statistical insights to perform **multi-class classification**.

---

## 🧪 Development Notes

- Compatible with **macOS** and **Linux** (handles GUI / headless environments automatically).
- Figures are always saved in `outputs/figures/`.
- For GUI display on macOS: `--show` uses the `MacOSX` backend.
- For servers or WSL: the script falls back to saving-only mode (`Agg` backend).

---

## 🏁 Quick Reference

| Script | Description | Example Command |
|---------|--------------|----------------|
| `describe.py` | Compute summary stats | `python3 src/describe.py data/dataset_train.csv` |
| `histogram.py` | Plot per-course histograms per house | `python3 src/histogram.py data/dataset_train.csv --show` |
| `scatter_plot.py` | Find and visualize most correlated features | `python3 src/scatter_plot.py data/dataset_train.csv --show` |
| `pair_plot.py` | Show scatter matrix of multiple features | `python3 src/pair_plot.py data/dataset_train.csv --show` |

---

## ⚙️ Requirements

Install dependencies:
```bash
pip install matplotlib
```

(Optional but recommended):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install matplotlib
```

---

## ✨ Credits

Developed as part of the **École 42 Data Science & Logistic Regression (DSLR)** project.  
Author: Ana Breyer  
Year: 2025

---