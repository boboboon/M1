# %%
"""Problem Sheet 2."""

from __future__ import annotations

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


class RandomGuessClassifier(BaseEstimator):
    """A classifier that makes random guesses.

    This classifier randomly selects from the unique classes found in the
    training data during the `fit` method and then uses these classes to
    make random predictions for each sample in the test data.

    Attributes:
        classes_ (np.ndarray): Array of unique classes derived from the training labels.
        rng (np.random.Generator): Random number generator instance for creating predictions.
    """

    def __init__(self, random_state: int | None = None) -> None:
        """Initializes the RandomGuessClassifier with a random seed.

        Args:
            random_state (Optional[int]): Seed for the random number generator.
                Defaults to None for randomness.
        """
        self.rng = np.random.default_rng(random_state)

    def fit(self, _: np.array, y: np.array) -> None:
        """Fits the classifier by storing unique classes from y.

        Args:
            _ (array-like): Training data of shape (n_samples, n_features).
            y (array-like): Training labels of shape (n_samples,).
        """
        self.classes_ = np.unique(y)

    def predict(self, x: np.array) -> np.ndarray:
        """Predicts random classes for each sample in X.

        Args:
            x (array-like): Data to predict, shape (n_samples, n_features).

        Returns:
            np.ndarray: Randomly chosen predictions of shape (n_samples,).
        """
        return self.rng.choice(self.classes_, size=len(x))


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
