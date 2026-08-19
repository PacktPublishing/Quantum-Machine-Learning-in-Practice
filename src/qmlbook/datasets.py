"""Dataset helpers shared by notebooks and scripts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DatasetSplit:
    """Train/test arrays with fitted preprocessing kept beside the data."""

    X_train: NDArray[np.float64]
    X_test: NDArray[np.float64]
    y_train: NDArray[np.int_]
    y_test: NDArray[np.int_]
    scaler: object


def make_toy_classification(
    *,
    n_samples: int = 120,
    n_features: int = 2,
    class_sep: float = 1.0,
    random_state: int = 7,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Create a small binary classification dataset for quick examples."""

    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        class_sep=class_sep,
        random_state=random_state,
    )
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def split_and_scale(
    X: NDArray[np.float64],
    y: NDArray[np.int_],
    *,
    test_size: float = 0.25,
    random_state: int = 7,
) -> DatasetSplit:
    """Make a stratified train/test split and fit scaling on training data only."""

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return DatasetSplit(
        X_train=np.asarray(X_train_scaled, dtype=float),
        X_test=np.asarray(X_test_scaled, dtype=float),
        y_train=np.asarray(y_train, dtype=int),
        y_test=np.asarray(y_test, dtype=int),
        scaler=scaler,
    )
