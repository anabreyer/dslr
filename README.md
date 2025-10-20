# 🧙‍♀️ DSLR — Data Science & Logistic Regression (Hogwarts Sorting Hat)

This project recreates the **Hogwarts Sorting Hat** using a **from‑scratch one‑vs‑all Logistic Regression** trained on students’ course grades. No scikit‑learn; all math and plotting are implemented manually.

---

## 📂 Project Structure

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
│   ├── test.py
├── models/                # saved weights (e.g., weights.json)
├── outputs/
│   ├── houses.csv         # predictions
│   └── figures/           # PNGs
├── Makefile
└── README.md
```

---

## 🧠 Goal

Predict a student’s **Hogwarts House** (Gryffindor, Hufflepuff, Ravenclaw, Slytherin) from their course grades by:
1) exploring the data, 2) visualizing important patterns, 3) training a logistic regression classifier, and 4) predicting on unseen data.

---

## 🛠 Make Commands (exactly as wired)

From the repository root:

```bash
make help
make describe
make histogram
make scatter
make pair
make train
make predict
make test
make fclean
```

### What each target does

| Target       | Command run                                                                                  | Notes |
|--------------|-----------------------------------------------------------------------------------------------|-------|
| `describe`   | `python3 src/describe.py data/dataset_train.csv`                                              | prints summary stats for every numeric column |
| `histogram`  | `python3 src/histogram.py data/dataset_train.csv --show`                                      | overlaid per‑house histograms for each course (also saves PNGs) |
| `scatter`    | `python3 src/scatter_plot.py data/dataset_train.csv --show`                                   | finds & shows the most correlated pair; accepts `--rank`, `--x`, `--y` when run directly |
| `pair`       | `python3 src/pair_plot.py data/dataset_train.csv --show`                                      | compact pair plot (all numeric features when `--show`) |
| `train`      | `python3 src/logreg_train.py data/dataset_train.csv`                                          | trains OVA logistic regression and writes `models/weights.json` |
| `predict`    | `python3 src/logreg_predict.py data/dataset_test.csv`                                         | loads `models/weights.json` and writes `outputs/houses.csv` |
| `test`       | `python3 src/test.py`                                                                         | optional correctness/sanity checks for the project |
| `fclean`     | deletes **all files** inside `outputs/` and `models/`                                         | keeps the folders present |

> Tip: If you prefer running scripts directly, the commands in the middle column are exactly what Make executes.

---

## 📊 Visualization Scripts

### `describe.py`
Manually computes: **count, mean, std, min, 25%, 50%, 75%, max** for all numeric columns.

```bash
python3 src/describe.py data/dataset_train.csv
```

### `histogram.py`
Overlaid histograms (same bin edges) per course for all four houses. Also prints a **homogeneity score** (lower ≙ more similar distributions).

```bash
python3 src/histogram.py data/dataset_train.csv --show
```

### `scatter_plot.py`
Searches the most linearly related pair of courses (by |Pearson r|) and plots it, colored by house.

```bash
python3 src/scatter_plot.py data/dataset_train.csv --show
```

### `pair_plot.py`
Grid of **scatter plots** (off‑diagonal) and **per‑house histograms** (diagonal). In `--show` mode, it uses **all** numeric features and a compact layout (tiny markers, rotated labels).

```bash
python3 src/pair_plot.py data/dataset_train.csv --show
```

---

## 🧮 Math — Logistic Regression (One‑vs‑All)

We train **one binary classifier per house**. For an input row \(x \in \mathbb{R}^{d}\) with a bias term \(x_0=1\) and parameters \(\theta \in \mathbb{R}^{d+1}\):

### Hypothesis (sigmoid)
\[
h_\theta(x)=\sigma(\theta^\top x)=\frac{1}{1+e^{-\theta^\top x}}
\]

### Binary Cross‑Entropy Cost (with optional L2)
\[
J(\theta)= -\frac{1}{m}\sum_{i=1}^{m}\big[y^{(i)}\log h_\theta(x^{(i)})+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))\big] + \frac{\lambda}{2m}\sum_{j=1}^{d}\theta_j^2
\]

> The bias \(\theta_0\) is **not** regularized.

### Gradient
\[
\frac{\partial J}{\partial \theta_j}=\frac{1}{m}\sum_{i=1}^{m}\big(h_\theta(x^{(i)})-y^{(i)}\big)\,x_j^{(i)} + \begin{cases}
0 & j=0\\[2pt]
\frac{\lambda}{m}\theta_j & j\ge1
\end{cases}
\]

### Gradient Descent Update
\[
\theta \leftarrow \theta - \alpha \,\nabla J(\theta)
\]

- \( \alpha \): learning rate  
- \( \lambda \): L2 penalty (optional)  
- Standardization: features may be z‑scored with training mean \(\mu\) and std \(\sigma\), and the same transform is applied at prediction time.

### One‑vs‑All Prediction
Compute a probability per house \(p_k=\sigma(\theta_k^\top x)\) and pick the **argmax**: \(\hat{y}=\arg\max_k p_k\).
(OVA scores don’t sum to 1; they are calibrated per classifier.)

---

## 🧭 Training & Prediction

### Train
```bash
python3 src/logreg_train.py data/dataset_train.csv
```
- Saves model to `models/weights.json` with keys:  
  `classes`, `features`, `mu`, `sigma`, `thetas`, and `standardize` flag.

### Predict
```bash
python3 src/logreg_predict.py data/dataset_test.csv
```
- Loads `models/weights.json`, imputes missing values with training means, applies the same standardization (if enabled), writes:
  - `outputs/houses.csv` (columns: `Index,Hogwarts House`)
  - optional `outputs/probas.csv` if your script supports `--proba-out`

---

## ⚙️ Requirements

```bash
python3 -m pip install matplotlib
```

(Optional)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install matplotlib
```

---

## 📝 Notes

- Backends are auto‑handled: GUI on macOS/Linux desktop; headless falls back to `Agg` and still saves PNGs.  
- Colors use canonical house palettes for consistency across plots.  
- All figures are written to `outputs/figures/`.

---

## ✨ Credits

Built for the **École 42 — DSLR** project.  
Author: **Ana Breyer**, 2025.
