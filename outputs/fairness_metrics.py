import pandas as pd
import os

# Define fairness metrics (placeholder values - replace with actual computed values)
overall_accuracy = 0.0
male_accuracy = 0.0
female_accuracy = 0.0
accuracy_difference = 0.0
demographic_parity_difference = 0.0
equal_opportunity_difference = 0.0
equalized_odds_difference = 0.0

metrics = {
    "Metric": [
        "Overall Accuracy",
        "Male Accuracy",
        "Female Accuracy",
        "Accuracy Difference",
        "Demographic Parity Difference",
        "Equal Opportunity Difference",
        "Equalized Odds Difference"
    ],
    "Value": [
        overall_accuracy,
        male_accuracy,
        female_accuracy,
        accuracy_difference,
        demographic_parity_difference,
        equal_opportunity_difference,
        equalized_odds_difference
    ]
}

df = pd.DataFrame(metrics)

os.makedirs("../outputs", exist_ok=True)

df.to_csv("../outputs/fairness_metrics.csv", index=False)

print("fairness_metrics.csv saved successfully.")