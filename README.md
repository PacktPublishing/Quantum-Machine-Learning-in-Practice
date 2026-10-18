# Quantum Machine Learning in Practice

Companion repository for the book **Quantum Machine Learning in Practice**,
published by Packt.

The book treats quantum machine learning as an engineering discipline: build
small simulator-first experiments, compare against strong classical baselines,
record the environment, and make narrow claims that match the evidence. This
repository supports that workflow with runnable notebooks, shared helper code,
environment files, smoke tests, and folders for data and experiment artifacts.

## What is in this repository

- `notebooks/` contains Jupyter notebooks for the code listings from the book.
- `src/qmlbook/` contains reusable helpers for datasets, metrics, and classical
  baselines.
- `requirements/` contains pip-compatible environment files split by framework.
- `scripts/smoke_test.py` checks imports and minimal framework functionality.
- `scripts/check_notebooks.py` verifies listing sources and optionally runs notebooks.
- `scripts/record_environment.py` records package, platform, and dependency-check
  output after an environment works.
- `data/raw/` and `data/processed/` separate source data from derived data.
- `results/experiments/`, `results/figures/`, and `results/logs/` hold generated
  artifacts and environment records.

No quantum hardware account is required for the core workflow. The examples are
designed to run locally with simulators first.

## How to use this repo with the book

Use the repository as an experimental companion, not just as a code dump.

1. Read the chapter section before running its notebook.
2. Start Jupyter from the repository root so local modules and relative paths
   resolve consistently.
3. Open the matching notebook in `notebooks/`.
4. Restart the kernel and run all cells before trusting a result.
5. Compare QML examples against the classical baselines discussed in the text.
6. Save generated figures, tables, and logs under `results/`.
7. Record the environment after a successful run.

The most useful habit is to keep an experiment log beside the notebook. Record
the dataset, preprocessing, random seed, circuit family, simulator or backend,
shot count, package environment, baseline results, and final test metric.

Cells tagged `Listing X.Y` reproduce the final book code blocks in chapter order.
Appendix cells tagged `Test` provide additional checks. The listing manifest
records the final source hashes so synchronization can be checked without access
to the manuscript. The release notebooks start with cleared outputs. Run the
cells to reproduce the book examples; timings and some sampled values can vary
across platforms and reruns.

## Notebook map

| Book material | Notebook |
| --- | --- |
| Chapter 1: Why Quantum Machine Learning (and Why Not Yet?) | `notebooks/chapter01.ipynb` |
| Chapter 2: Quantum Computing Essentials for ML Practitioners | `notebooks/chapter02.ipynb` |
| Chapter 3: Encoding Classical Data into Quantum Circuits | `notebooks/chapter03.ipynb` |
| Chapter 4: Variational Quantum Circuits as Machine Learning Models | `notebooks/chapter04.ipynb` |
| Chapter 5: PennyLane: Differentiable Quantum Programming | `notebooks/chapter05.ipynb` |
| Chapter 6: Qiskit Machine Learning | `notebooks/chapter06.ipynb` |
| Chapter 7: TensorFlow Quantum | `notebooks/chapter07.ipynb` |
| Chapter 8: Cirq | `notebooks/chapter08.ipynb` |
| Chapter 9: Quantum Kernels and Similarity-Based Learning | `notebooks/chapter09.ipynb` |
| Chapter 10: Generative Modeling with Quantum Circuits | `notebooks/chapter10.ipynb` |
| Chapter 11: Benchmarking, Evaluation, and When Classical ML Wins | `notebooks/chapter11.ipynb` |
| Chapter 12: End-to-End Mini Project | `notebooks/chapter12.ipynb` |
| Chapter 13: Information-Theoretic and Privacy Perspectives | `notebooks/chapter13.ipynb` |
| Chapter 14: Beyond Simulation: Hardware and the Road Ahead | `notebooks/chapter14.ipynb` |
| Appendix I: Linear Algebra Refresher | `notebooks/appendixI.ipynb` |
| Appendix II: Probability and Expectation Values | `notebooks/appendixII.ipynb` |
| Appendix III: Optimization and Gradients | `notebooks/appendixIII.ipynb` |

## Choose the environment for your chapter

| Environment | Notebooks |
| --- | --- |
| Base scientific stack | Chapters 1–4, 11, 13; Appendices I–III |
| `pennylane` | Chapters 5, 10, 12 |
| `qiskit` | Chapters 6, 9, 14 |
| `tfq` | Chapter 7 (Linux x86_64) |
| `cirq` | Chapter 8 |

The pinned Autograd, Lightning, and Cirq Google versions match the tested book
setup. Use the framework profiles and CPU-only PyTorch configuration below.
Keep TensorFlow Quantum in its own environment.

## Recommended setup: uv

The repository pins the tested Python 3.12.13 interpreter and provides a committed
`uv.lock`. Install the base notebook and classical-ML environment from the
repository root. This is the lock, smoke-test, and environment-recording workflow
used in Appendix IV.
If needed, install or update uv using its
[official installation guide](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv python install 3.12.13
uv sync --locked
uv run --locked python scripts/smoke_test.py --framework base --strict
uv run --locked jupyter lab
```

`uv sync` installs this project in editable form, so `qmlbook` is importable in
the notebooks without a separate `PYTHONPATH` or package-install step. Do not run
ad-hoc `uv pip install` commands after syncing: that would make the environment
differ from the committed lockfile.

### Framework extras

Install only the extra needed for the chapter or experiment you are running.
Keep the same `--extra` on `uv sync` and `uv run`:

```bash
# PennyLane
uv sync --locked --extra pennylane
uv run --locked --extra pennylane python scripts/smoke_test.py --framework pennylane --strict

# Qiskit
uv sync --locked --extra qiskit
uv run --locked --extra qiskit python scripts/smoke_test.py --framework qiskit --strict

# Cirq
uv sync --locked --extra cirq
uv run --locked --extra cirq python scripts/smoke_test.py --framework cirq --strict

# TensorFlow Quantum
uv sync --locked --extra tfq
uv run --locked --extra tfq python scripts/smoke_test.py --framework tfq --strict
```

The uv configuration resolves CPU-only PyTorch wheels for the `pennylane`
extra; no CUDA toolkit is required. The pinned PyTorch release provides macOS
wheels only for Apple silicon on macOS 14 or newer, not Intel Macs. The base,
Qiskit, and Cirq profiles do not inherit that PyTorch restriction.

Chapter 7's locked `tfq` environment is supported only on Python 3.12 running
Linux x86_64. On macOS, Windows, ARM, or another unsupported platform, use the
[official TensorFlow Quantum installation page](https://www.tensorflow.org/quantum/install)
to run the tutorials in Google Colab. If you need a native environment and can
meet its supported build prerequisites, follow the source-build instructions on
that page. Keep TFQ separate from the other framework extras.

To launch Jupyter with a framework extra, use the matching flag, for example:

```bash
uv run --locked --extra pennylane jupyter lab
```

## Alternative setup: pip and venv

`uv` is the reproducible path. If it is unavailable, create a clean Python 3.12
environment and install this project as an editable package:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python scripts/smoke_test.py --framework base --strict
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Install one
framework extra with the same name used by uv:

```bash
python -m pip install -e ".[pennylane]"  # or qiskit, cirq, tfq
python scripts/smoke_test.py --framework pennylane --strict
python -m jupyter lab
```

The editable install makes `qmlbook` importable. Unlike uv, pip does not use
`uv.lock` or uv's CPU-only PyTorch source configuration, so record the realized
environment carefully. The TFQ platform restriction still applies.

## Record a working environment

After the strict smoke test succeeds, save the realized package, compatibility,
and platform records. Use a short label that identifies the environment:

```bash
uv run --locked --extra pennylane python scripts/record_environment.py pennylane-env
```

For the pip/venv path, run `python scripts/record_environment.py pennylane-env`.
The script writes a package snapshot under `requirements/locks/` and diagnostic
files under `results/logs/`. Treat these as records of a working machine, not as
portable lockfiles across operating systems and CPU architectures. Record labels
are write-once; use a new label rather than overwriting evidence from an earlier
run.

## Reproducible experiments

Keep source data in `data/raw/` and write derived datasets to `data/processed/`;
never overwrite raw inputs. Store generated outputs under `results/experiments/`,
`results/figures/`, and `results/logs/` as appropriate.

For each run, record enough context to reproduce and interpret the result:

- dataset source or checksum, preprocessing, and train/validation/test split;
- all random seeds, circuit family, observable, optimizer, and hyperparameters;
- simulator or hardware backend, shot count, transpilation settings, and date;
- classical baseline configuration and comparable evaluation metrics; and
- environment label, package snapshot, platform record, and relevant logs.

Restart the kernel and run all notebook cells before trusting a result. If a
notebook fails, reduce it to an import check and a one-qubit deterministic
simulation before adding shots, optimization, and data. Installation failures
are software diagnostics, not evidence for or against a modeling claim.

Most examples are simulator-first. If you use a remote provider, keep API tokens
out of notebooks, source files, figures, and logs. Record non-secret job and
calibration metadata needed to interpret the run.

## Verify the notebooks

Check the notebook inventory, Python syntax, saved errors, and exact listing
hashes without importing quantum frameworks:

```bash
uv run --locked python scripts/check_notebooks.py
```

Execute a profile with a fresh Jupyter kernel and temporary working directory for
each notebook. Use the same extra as the environment being tested:

```bash
uv run --locked python scripts/check_notebooks.py --profile base --execute
uv run --locked --extra pennylane python scripts/check_notebooks.py --profile pennylane --execute
uv run --locked --extra qiskit python scripts/check_notebooks.py --profile qiskit --execute
uv run --locked --extra cirq python scripts/check_notebooks.py --profile cirq --execute
uv run --locked --extra tfq python scripts/check_notebooks.py --profile tfq --execute
```

These checks leave the source notebooks unchanged. The capstone and framework
training examples can take several minutes. Their documented numerical checks
run as part of execution; runtime and memory measurements are machine-dependent.

Release validation on 2026-10-18 passed all 17 notebooks on Linux x86_64 with
Python 3.12.13, using a separate locked environment for each profile. All five
profiles also passed the strict smoke tests and dependency consistency checks.

## Final chapters and generated records

Chapter 12 writes `chapter12_results.json`; Chapter 14 writes per-job JSON files
and `summary.json` in `chapter14_results/`. These paths are relative to the
kernel's working directory, normally `notebooks/` when opened in JupyterLab.
They are ignored by Git. Archive them under `results/experiments/` for a run you
want to retain, and start Chapter 14 in a fresh working directory for each
independent run.

Chapter 13 uses a fixed classifier to audit representations, readouts,
interpretability, and privacy calculations. It does not release private data.
Chapter 14's default execution is entirely local and records physical hardware
as `not_run`. Its optional provider adapter requires a separately configured
account, backend, and sampler. The book and notebook do not claim that this
optional path was validated on a physical device; inspect the saved job ID after
any retrieval failure before considering another submission.
