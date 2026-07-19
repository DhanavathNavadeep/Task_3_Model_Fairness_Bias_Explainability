import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from lime.lime_tabular import LimeTabularExplainer

# ----------------------------
# Create outputs folder
# ----------------------------
os.makedirs("outputs", exist_ok=True)

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("data/dataset.csv")

for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop("Income", axis=1)
y = df["Income"]

# ----------------------------
# Load trained model
# ----------------------------
model = joblib.load("outputs/model.pkl")

# ----------------------------
# Feature Importance
# ----------------------------
importance = model.feature_importances_

plt.figure(figsize=(8,5))
plt.bar(X.columns, importance)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("outputs/feature_importance.png", dpi=300)
plt.close()

print("feature_importance.png saved.")

# ----------------------------
# LIME Explanation
# ----------------------------
explainer = LimeTabularExplainer(
    training_data=X.values,
    feature_names=X.columns.tolist(),
    class_names=["<=50K", ">50K"],
    mode="classification"
)

exp = explainer.explain_instance(
    X.iloc[0].values,
    model.predict_proba,
    num_features=len(X.columns)
)

fig = exp.as_pyplot_figure()
fig.tight_layout()
fig.savefig("outputs/lime_explanation.png", dpi=300)
plt.close(fig)

print("lime_explanation.png saved.")