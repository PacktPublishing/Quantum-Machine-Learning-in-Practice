# Requirements

These pip-compatible files mirror the base dependencies and optional extras in
`pyproject.toml`. Use Python 3.12 and install one framework environment at a
time. Start with the base scientific stack, then add the framework-specific
file for the chapter you are running. See the README chapter/environment map.
The tested interpreter is Python 3.12.13; the project supports Python 3.12 only.
PennyLane additionally pins Autograd 1.8.0 and Lightning 0.45.0; the Cirq and
TFQ profiles include Cirq Google 1.5.0, matching Appendix IV.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements/base.txt
python -m pip install --no-deps -e .
python scripts/smoke_test.py --framework base --strict
```

Framework examples:

```bash
python -m pip install -r requirements/pennylane.txt
python -m pip install --no-deps -e .
python scripts/smoke_test.py --framework pennylane --strict

python -m pip install -r requirements/qiskit.txt
python -m pip install --no-deps -e .
python scripts/smoke_test.py --framework qiskit --strict

python -m pip install -r requirements/tensorflow-quantum.txt
python -m pip install --no-deps -e .
python scripts/smoke_test.py --framework tfq --strict

python -m pip install -r requirements/cirq.txt
python -m pip install --no-deps -e .
python scripts/smoke_test.py --framework cirq --strict
```

`pennylane.txt` uses PyTorch's official CPU wheel index when a compatible wheel
is available. Linux and Windows therefore avoid pulling CUDA runtime packages.
macOS falls back to PyPI, where the pinned PyTorch release supports Apple
silicon on macOS 14 or newer but does not publish an Intel Mac wheel.

TensorFlow Quantum 0.7.6 is supported only on Linux x86_64 in this repository.
Its requirements are ignored on unsupported platforms so the base environment
can still be installed there.

After a working install, record the realized environment:

```bash
python scripts/record_environment.py qml-env
```

The files in `requirements/locks/` are for machine-specific lock snapshots
created after an environment has passed smoke tests.
