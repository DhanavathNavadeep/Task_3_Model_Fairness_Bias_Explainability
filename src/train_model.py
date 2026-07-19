import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

try:
    from fairlearn.metrics import (
        MetricFrame,
        selection_rate,
        demographic_parity_difference,
    )
except ImportError:
    print("Install Fairlearn:")
    print("pip install fairlearn")
    raise

# ----------------------------
# Load model
# ----------------------------
model = joblib.load("../outputs/model.pkl")

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("../data/dataset.csv")

TARGET = "Income"
SENSITIVE = "Gender"

# ----------------------------
# Features and Target
# ----------------------------
X = df.drop(columns=[TARGET])
X = pd.get_dummies(X, drop_first=True)

y = df[TARGET]

# ----------------------------
# Predict
# ----------------------------
pred = model.predict(X)

# ----------------------------
# Accuracy
# ----------------------------
acc = accuracy_score(y, pred)

print("=" * 50)
print("MODEL FAIRNESS REPORT")
print("=" * 50)
print(f"Accuracy : {acc:.4f}")

# ----------------------------
# Fairness Metrics
# ----------------------------
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

print("\nDemographic Parity Difference :", dp)

metric.by_group.to_csv(
    "../outputs/fairness_metrics.csv"
)

with open("../outputs/bias_report.txt", "w") as f:
    f.write("MODEL FAIRNESS REPORT\n")
    f.write("=========================\n\n")
    f.write(f"Accuracy : {acc:.4f}\n")
    f.write(f"Demographic Parity Difference : {dp:.4f}\n\n")
    f.write(metric.by_group.to_string())

print("\nBias Analysis Completed Successfully!")