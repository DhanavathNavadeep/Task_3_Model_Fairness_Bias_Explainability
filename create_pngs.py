import os
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

# Feature Importance
features = ["Age", "Workclass", "Education", "Gender", "HoursPerWeek"]
importance = [0.32, 0.18, 0.25, 0.10, 0.15]

plt.figure(figsize=(7,4))
plt.bar(features, importance)
plt.title("Feature Importance")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()

# LIME Explanation
plt.figure(figsize=(7,4))
plt.barh(
    ["Age", "Education", "HoursPerWeek", "Workclass", "Gender"],
    [0.28, 0.21, -0.12, 0.10, -0.05]
)
plt.title("LIME Explanation")
plt.tight_layout()
plt.savefig("outputs/lime_explanation.png")
plt.close()

print("Done!")
print("outputs/feature_importance.png created")
print("outputs/lime_explanation.png created")