#!/usr/bin/env python3
"""Minimal setup checks for the Appendix IV environments."""

from __future__ import annotations

import os

# TensorFlow reads this flag while it is importing.  Set it before importlib can
# load TensorFlow directly or as a TensorFlow Quantum dependency.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import argparse
import importlib
import importlib.metadata as metadata
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PackageCheck:
    module_name: str
    distribution_name: str


CHECKS: dict[str, list[PackageCheck]] = {
    "base": [
        PackageCheck("qmlbook", "packt-qml-listings"),
        PackageCheck("numpy", "numpy"),
        PackageCheck("scipy", "scipy"),
        PackageCheck("pandas", "pandas"),
        PackageCheck("sklearn", "scikit-learn"),
        PackageCheck("matplotlib", "matplotlib"),
        PackageCheck("seaborn", "seaborn"),
        PackageCheck("jupyterlab", "jupyterlab"),
        PackageCheck("ipykernel", "ipykernel"),
        PackageCheck("nbconvert", "nbconvert"),
        PackageCheck("nbclient", "nbclient"),
        PackageCheck("tqdm", "tqdm"),
    ],
    "pennylane": [
        PackageCheck("pennylane", "pennylane"),
        PackageCheck("torch", "torch"),
    ],
    "qiskit": [
        PackageCheck("qiskit", "qiskit"),
        PackageCheck("qiskit_aer", "qiskit-aer"),
        PackageCheck("qiskit_machine_learning", "qiskit-machine-learning"),
    ],
    "tfq": [
        PackageCheck("tensorflow", "tensorflow"),
        PackageCheck("tf_keras", "tf-keras"),
        PackageCheck("tensorflow_quantum", "tensorflow-quantum"),
        PackageCheck("cirq", "cirq-core"),
        PackageCheck("sympy", "sympy"),
    ],
    "cirq": [
        PackageCheck("cirq", "cirq-core"),
        PackageCheck("sympy", "sympy"),
    ],
}


def exception_summary(exc: Exception, *, limit: int = 160) -> str:
    """Return a bounded, single-line diagnostic for a caught exception."""

    message = " ".join(str(exc).split()) or "no message"
    summary = f"{type(exc).__name__}: {message}"
    if len(summary) > limit:
        return summary[: limit - 3] + "..."
    return summary


def check_packages(framework: str) -> list[str]:
    """Import selected packages and report their installed versions."""

    if framework == "tfq":
        # Restore the required value if a caller changed the process environment
        # after importing this module but before starting the TFQ checks.
        os.environ["TF_USE_LEGACY_KERAS"] = "1"

    missing: list[str] = []
    for check in CHECKS[framework]:
        try:
            importlib.import_module(check.module_name)
            version = metadata.version(check.distribution_name)
        except Exception as exc:
            print(f"{check.module_name}: missing ({exception_summary(exc)})")
            missing.append(check.module_name)
        else:
            print(f"{check.module_name}: {version}")
    return missing


def base_smoke() -> None:
    """Exercise the installed project helpers and core scientific stack."""

    from qmlbook.datasets import make_toy_classification, split_and_scale
    from qmlbook.metrics import classification_scores

    features, labels = make_toy_classification(n_samples=40, random_state=7)
    split = split_and_scale(features, labels, random_state=7)
    scores = classification_scores(split.y_test, split.y_test)
    print(
        "qmlbook baseline: "
        f"train={split.X_train.shape}, test={split.X_test.shape}, "
        f"accuracy={scores['accuracy']:.1f}"
    )


def pennylane_smoke() -> None:
    import pennylane as qml

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def one_qubit_score(theta: float) -> float:
        qml.RX(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    value = float(one_qubit_score(0.25))
    expected = math.cos(0.25)
    if not math.isclose(value, expected, rel_tol=1e-7, abs_tol=1e-7):
        raise RuntimeError(f"unexpected Pauli-Z expectation: {value:.8f}")
    print(f"pennylane expectation: {value:.6f}")


def qiskit_smoke() -> None:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    circuit = QuantumCircuit(1, 1)
    circuit.x(0)
    circuit.measure(0, 0)
    counts = AerSimulator().run(circuit, shots=16, seed_simulator=7).result().get_counts()
    if counts != {"1": 16}:
        raise RuntimeError(f"unexpected measurement counts: {counts}")
    print("qiskit counts: {'1': 16}")


def cirq_smoke() -> None:
    import cirq

    qubit = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(qubit), cirq.measure(qubit, key="m"))
    result = cirq.Simulator(seed=7).run(circuit, repetitions=16)
    measurements = result.measurements["m"].tolist()
    if measurements != [[1]] * 16:
        raise RuntimeError(f"unexpected measurements: {measurements}")
    print("cirq measurements: 16 ones")


def tfq_smoke() -> None:
    os.environ["TF_USE_LEGACY_KERAS"] = "1"

    import cirq
    import tensorflow_quantum as tfq

    qubit = cirq.GridQubit(0, 0)
    circuit_tensor = tfq.convert_to_tensor([cirq.Circuit(cirq.X(qubit))])
    operator_tensor = tfq.convert_to_tensor([[cirq.Z(qubit)]])
    expectation = tfq.layers.Expectation()(
        circuit_tensor,
        operators=operator_tensor,
    )
    value = float(expectation.numpy()[0, 0])
    if not math.isclose(value, -1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise RuntimeError(f"unexpected Pauli-Z expectation: {value:.8f}")
    print(f"tfq expectation: {value:.1f}")


SMOKE_TESTS: dict[str, Callable[[], None]] = {
    "base": base_smoke,
    "pennylane": pennylane_smoke,
    "qiskit": qiskit_smoke,
    "cirq": cirq_smoke,
    "tfq": tfq_smoke,
}


def selected_frameworks(name: str) -> list[str]:
    """Expand the command-line framework selection."""

    if name == "all":
        return list(CHECKS)
    return [name]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--framework",
        choices=["base", "pennylane", "qiskit", "tfq", "cirq", "all"],
        default="base",
        help="Environment to check.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when a selected package or circuit check fails.",
    )
    args = parser.parse_args(argv)

    failures: list[str] = []
    for framework in selected_frameworks(args.framework):
        print(f"\n[{framework}]")
        missing = check_packages(framework)
        failures.extend(f"{framework}:{module_name}" for module_name in missing)
        if missing:
            continue

        smoke_test = SMOKE_TESTS.get(framework)
        if smoke_test is None:
            continue
        try:
            smoke_test()
        except Exception as exc:
            failures.append(f"{framework}:smoke")
            print(f"{framework} smoke: failed ({exception_summary(exc)})")

    if failures and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
