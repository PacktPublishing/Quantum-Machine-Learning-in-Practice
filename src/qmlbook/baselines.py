"""Small classical baselines for fair QML comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from numpy.typing import ArrayLike

from qmlbook.metrics import classification_scores


def build_classical_baselines(*, random_state: int = 7) -> dict[str, Any]:
    """Build lightweight baseline models used before interpreting QML results."""

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC, SVC

    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=random_state),
        ),
        "linear_svc": make_pipeline(
            StandardScaler(),
            LinearSVC(dual="auto", max_iter=5000, random_state=random_state),
        ),
        "rbf_svc": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", gamma="scale"),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
        ),
    }


def evaluate_classical_baselines(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_test: ArrayLike,
    y_test: ArrayLike,
    *,
    models: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, float]]:
    """Fit baseline models and return comparable classification scores."""

    from sklearn.base import clone

    baseline_models = build_classical_baselines() if models is None else models
    results: dict[str, dict[str, float]] = {}
    for name, model in baseline_models.items():
        fitted = clone(model)
        fitted.fit(X_train, y_train)
        results[name] = classification_scores(y_test, fitted.predict(X_test))
    return results
