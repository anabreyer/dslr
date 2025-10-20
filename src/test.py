import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# 1. predicted file
df_pred = pd.read_csv("outputs/houses.csv")  # ['Index', 'Hogwarts House']

# 2. answer file
df_true = pd.read_csv("data/dataset_truth.csv")  # ['Index', 'Hogwarts House']

# 3. merge predicted and answer on 'Index' column
merged = pd.merge(df_pred, df_true, on="Index", suffixes=("_pred", "_true"))

# 4. evaluate the prediction
#    - accuracy score
y_true = merged["Hogwarts House_true"]
y_pred = merged["Hogwarts House_pred"]

print("Accuracy:", accuracy_score(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred))


# Accuracy= Total number of predictions / Number of correct predictions