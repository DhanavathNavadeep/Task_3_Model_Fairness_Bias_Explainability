import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

try:
    from fairlearn.metrics import (
        MetricFrame,
        selection_rate,
        demographic_parity_difference
    )
except ImportError:
    print("Install fairlearn first:")
    print("pip install fairlearn")
    raise

# -------------------------
# Load model and dataset
# -------------------------

model = joblib.load("../outputs/model.pkl")

df = pd.read_csv("../data/dataset.csv")

TARGET = "Income"
SENSITIVE = "Gender"

# -------------------------
# Features & Target
# -------------------------

# Convert categorical columns to dummy variables
X = pd.get_dummies(
    df.drop(columns=[TARGET]),
    drop_first=True
)

# Match the exact feature columns used during training
X = X.reindex(columns=model.feature_names_in_, fill_value=0)

y = df[TARGET]

# -------------------------
# Predict
# -------------------------

pred = model.predict(X)

# -------------------------
# Accuracy
# -------------------------

acc = accuracy_score(y, pred)
print(f"\nAccuracy : {acc:.4f}")

# -------------------------
# Fairness Metrics
# -------------------------

metric = MetricFrame(
    metrics=selection_rate,
    y_true=y,
    y_pred=pred,
    sensitive_features=df[SENSITIVE]
)

print("\nSelection Rate by Group")
print(metric.by_group)

dp = demographic_parity_difference(
    y_true=y,
    y_pred=pred,
    sensitive_features=df[SENSITIVE]
)

print("\nDemographic Parity Difference:", dp)

# -------------------------
# Save Results
# -------------------------

metric.by_group.to_csv("../outputs/fairness_metrics.csv")

with open("../outputs/bias_report.txt", "w") as f:
    f.write("MODEL FAIRNESS REPORT\n")
    f.write("=========================\n\n")
    f.write(f"Accuracy : {acc:.4f}\n")
    f.write(f"Demographic Parity Difference : {dp:.4f}\n\n")
    f.write(metric.by_group.to_string())

print("\nBias Analysis Completed Successfully!")