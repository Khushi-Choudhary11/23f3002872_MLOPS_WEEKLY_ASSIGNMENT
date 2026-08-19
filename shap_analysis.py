"""
Task 3 -- SHAP summary plots for the IRIS classifier.

Uses a full-dataset shap.TreeExplainer (exact, since the model is a
DecisionTreeClassifier) to explain predictions for all three IRIS classes.
Only the four real features are explained -- `location` is never a model
input, so it is excluded here too.

For each class, this produces a beeswarm summary plot: one dot per sample,
x-axis = SHAP value (how much that feature's value pushed the prediction
toward/away from the class), color = the feature's value for that sample
(red = high, blue = low), features ranked top-to-bottom by mean |SHAP
value| (overall importance).
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

DATA_DIR = Path("data/week9")
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def main():
    model = joblib.load("models/week9_model.joblib")
    df = pd.read_csv(DATA_DIR / "iris_with_location.csv")
    X = df[FEATURES]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)  # Explanation: (n_samples, n_features, n_classes)

    classes = list(model.classes_)
    print(f"Explaining {len(X)} samples across classes: {classes}")

    for idx, class_name in enumerate(classes):
        plt.figure()
        class_shap = shap_values[:, :, idx]
        shap.plots.beeswarm(class_shap, show=False)
        plt.title(f"SHAP Summary -- class: {class_name}")
        plt.tight_layout()
        out_path = DATA_DIR / f"shap_summary_{class_name}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved -> {out_path}")

        mean_abs = pd.Series(
            abs(class_shap.values).mean(axis=0), index=FEATURES
        ).sort_values(ascending=False)
        print(f"  Mean |SHAP| for '{class_name}':")
        print(mean_abs.to_string())


if __name__ == "__main__":
    main()
