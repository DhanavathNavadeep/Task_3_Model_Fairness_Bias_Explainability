import joblib
import shap
import pandas as pd

# ----------------------------
# Load trained model
# ----------------------------
model = joblib.load("../outputs/model.pkl")

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("../data/dataset.csv")

TARGET = "Income"

# ----------------------------
# Prepare features
# ----------------------------
X = pd.get_dummies(
    df.drop(columns=[TARGET]),
    drop_first=True
)

print("=" * 50)
print("MODEL EXPLAINABILITY")
print("=" * 50)

# ----------------------------
# Feature Importance
# ----------------------------
if hasattr(model, "feature_importances_"):

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

elif hasattr(model, "coef_"):

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": abs(model.coef_[0])
    })

else:
    raise Exception("This model does not support feature importance.")

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)

# ----------------------------
# Save CSV
# ----------------------------
importance.to_csv(
    "../outputs/feature_importance.csv",
    index=False
)

# ----------------------------
# Save Report
# ----------------------------
with open("../outputs/explainability_report.txt", "w") as f:
    f.write("MODEL EXPLAINABILITY REPORT\n")
    f.write("============================\n\n")
    f.write(importance.to_string(index=False))

print("\nFeature importance saved.")
print("Explainability Completed Successfully!")