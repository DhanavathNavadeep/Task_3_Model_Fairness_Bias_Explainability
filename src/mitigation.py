"""
Bias Mitigation Recommendations

1. Remove unnecessary sensitive attributes.
2. Balance dataset using SMOTE or class weighting.
3. Use Fairlearn Threshold Optimizer.
4. Regularly monitor fairness metrics.
5. Retrain with diverse representative data.
"""

recommendations = [
    "Remove sensitive features if not required.",
    "Balance the training dataset.",
    "Use Fairlearn post-processing methods.",
    "Monitor fairness after every retraining.",
    "Collect diverse training samples."
]

with open("../outputs/mitigation_recommendations.txt", "w") as f:

    f.write("Bias Mitigation Recommendations\n")
    f.write("===============================\n\n")

    for i, rec in enumerate(recommendations, start=1):
        f.write(f"{i}. {rec}\n")

print("Mitigation recommendations saved.")