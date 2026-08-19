"""Unit tests for the reusable qmlbook helpers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from sklearn.dummy import DummyClassifier

from qmlbook.baselines import build_classical_baselines, evaluate_classical_baselines
from qmlbook.datasets import make_toy_classification, split_and_scale
from qmlbook.metrics import classification_scores, confusion_counts


class DatasetTests(unittest.TestCase):
    def test_toy_dataset_is_reproducible(self) -> None:
        first_features, first_labels = make_toy_classification(random_state=11)
        second_features, second_labels = make_toy_classification(random_state=11)

        np.testing.assert_array_equal(first_features, second_features)
        np.testing.assert_array_equal(first_labels, second_labels)

    def test_split_scales_training_data_only(self) -> None:
        features, labels = make_toy_classification(n_samples=80, random_state=5)
        split = split_and_scale(features, labels, test_size=0.25, random_state=5)

        self.assertEqual(split.X_train.shape, (60, 2))
        self.assertEqual(split.X_test.shape, (20, 2))
        np.testing.assert_allclose(split.X_train.mean(axis=0), 0.0, atol=1e-12)
        self.assertEqual(set(split.y_train), {0, 1})
        self.assertEqual(set(split.y_test), {0, 1})


class MetricTests(unittest.TestCase):
    def test_classification_scores_use_plain_floats(self) -> None:
        scores = classification_scores([0, 1, 1], [0, 1, 1])

        self.assertEqual(scores, {
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "f1_weighted": 1.0,
        })
        self.assertTrue(all(type(value) is float for value in scores.values()))

    def test_confusion_counts_preserves_observed_non_contiguous_labels(self) -> None:
        counts = confusion_counts([2, 2, 7], [2, 9, 7])

        self.assertEqual(counts["labels"], [2, 7, 9])
        self.assertEqual(
            counts["matrix"],
            [
                [1, 0, 1],
                [0, 1, 0],
                [0, 0, 0],
            ],
        )


class BaselineTests(unittest.TestCase):
    def test_builds_expected_baseline_families(self) -> None:
        self.assertEqual(
            set(build_classical_baselines()),
            {"logistic_regression", "linear_svc", "rbf_svc", "random_forest"},
        )

    def test_evaluates_supplied_cloneable_models(self) -> None:
        features, labels = make_toy_classification(n_samples=60, random_state=3)
        split = split_and_scale(features, labels, random_state=3)

        results = evaluate_classical_baselines(
            split.X_train,
            split.y_train,
            split.X_test,
            split.y_test,
            models={"dummy": DummyClassifier(strategy="most_frequent")},
        )

        self.assertEqual(set(results), {"dummy"})
        self.assertEqual(
            set(results["dummy"]),
            {"accuracy", "balanced_accuracy", "f1_weighted"},
        )

    def test_explicit_empty_models_mapping_returns_empty_results(self) -> None:
        with patch("qmlbook.baselines.build_classical_baselines") as build_defaults:
            results = evaluate_classical_baselines([], [], [], [], models={})

        self.assertEqual(results, {})
        build_defaults.assert_not_called()


if __name__ == "__main__":
    unittest.main()
