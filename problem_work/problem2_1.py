# %%
"""Problem Sheet 2."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn import datasets, metrics
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split

# Load digits dataset
digits = datasets.load_digits()

# %%
# Process our data
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))


class RandomGuessClassifier(BaseEstimator):  # noqa: D101
    def fit(self, X, y) -> None:  # noqa: ANN001, ARG002, D102, N803
        # Store unique classes from y to randomly select from them
        self.classes_ = np.unique(y)

    def predict(self, X):  # noqa: ANN001, ANN201, D102, N803
        # Return random choices from available classes
        return np.random.choice(self.classes_, size=len(X))  # noqa: NPY002


# Models and parameters to experiment with
models = [
    {"name": "SGDClassifier", "model": SGDClassifier, "params": {"penalty": ["l2", None]}},
    {
        "name": "LogisticRegression",
        "model": LogisticRegression,
        "params": {"penalty": ["l2", None]},
    },
    {"name": "RandomGuessClassifier", "model": RandomGuessClassifier, "params": {}},
]


results_df = pd.DataFrame()

X_train, X_test, y_train, y_test = train_test_split(
    data,
    digits.target,
    test_size=0.5,
    shuffle=False,
)
for model_info in models:
    model_name = model_info["name"]
    ModelClass = model_info["model"]
    param_grid = model_info.get("params", {})

    # If there are no parameters (e.g., for RandomGuessClassifier), skip parameter looping
    if not param_grid:
        clf = ModelClass()
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)

        # Calculate metrics
        accuracy = metrics.accuracy_score(y_test, predicted)
        report = metrics.classification_report(y_test, predicted, output_dict=True)

        # Store results for random guessing
        report_df = pd.DataFrame(report).transpose()
        report_df["model"] = model_name
        report_df["penalty"] = "None"  # No penalty for random guessing
        report_df["accuracy"] = accuracy

        # Append results to main DataFrame
        results_df = pd.concat(
            [results_df, report_df.assign(index=np.arange(len(report_df)))],
            axis=0,
            ignore_index=True,
        )

    else:
        # Loop through each parameter combination for other models
        for penalty in param_grid["penalty"]:
            clf = ModelClass(penalty=penalty, max_iter=1000, tol=1e-3)

            # Fit the model
            clf.fit(X_train, y_train)

            # Make predictions
            predicted = clf.predict(X_test)

            # Calculate metrics
            accuracy = metrics.accuracy_score(y_test, predicted)
            report = metrics.classification_report(y_test, predicted, output_dict=True)

            # Store results
            report_df = pd.DataFrame(report).transpose()
            report_df["model"] = model_name
            report_df["penalty"] = penalty if penalty is not None else "None"
            report_df["accuracy"] = accuracy

            # Append results to main DataFrame
            results_df = pd.concat(
                [results_df, report_df.assign(index=np.arange(len(report_df)))],
                axis=0,
                ignore_index=True,
            )

# Display results
logger.info(f"Results DataFrame:\n{results_df}\n")

results_df.pivot_table(index="penalty", columns="model", values="accuracy").plot(
    kind="bar",
    title="Model Penalty Accuracy Comp",
)
# %%
# ? We can look at an example of which ones we didn't get

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image_reshaped = image.reshape(8, 8)
    ax.imshow(image_reshaped, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

# %%
