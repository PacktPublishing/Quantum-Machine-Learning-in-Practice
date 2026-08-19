"""Dependency-light unit tests for repository utility scripts."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str) -> types.ModuleType:
    """Load a script as a module without requiring scripts to be a package."""

    module_name = f"test_loaded_{name}"
    spec = importlib.util.spec_from_file_location(
        module_name, REPOSITORY_ROOT / "scripts" / f"{name}.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


SMOKE_TEST = load_script("smoke_test")
RECORD_ENVIRONMENT = load_script("record_environment")


class FakeDistribution:
    def __init__(
        self,
        name: str,
        version: str,
        requires: list[str] | None = None,
    ) -> None:
        self.metadata = {"Name": name}
        self.version = version
        self.requires = requires


def create_dependency_inputs(root: Path) -> None:
    """Create the complete dependency-file set expected by the recorder."""

    for relative_path in RECORD_ENVIRONMENT.DEPENDENCY_FILES:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"test content for {relative_path.as_posix()}\n",
            encoding="utf-8",
        )


class SmokeTestScriptTests(unittest.TestCase):
    def test_tensorflow_flag_is_set_before_tfq_package_imports(self) -> None:
        observed_values: list[str | None] = []

        def observe_import(_module_name: str) -> object:
            observed_values.append(os.environ.get("TF_USE_LEGACY_KERAS"))
            return object()

        with (
            mock.patch.dict(os.environ, {"TF_USE_LEGACY_KERAS": "0"}),
            mock.patch.object(
                SMOKE_TEST.importlib, "import_module", side_effect=observe_import
            ),
            mock.patch.object(SMOKE_TEST.metadata, "version", return_value="1.0"),
        ):
            self.assertEqual(SMOKE_TEST.check_packages("tfq"), [])

        self.assertTrue(observed_values)
        self.assertEqual(set(observed_values), {"1"})

    def test_base_checks_installed_project_and_has_functional_smoke(self) -> None:
        base_modules = {check.module_name for check in SMOKE_TEST.CHECKS["base"]}

        self.assertTrue({"qmlbook", "nbconvert", "nbclient"} <= base_modules)
        self.assertIs(SMOKE_TEST.SMOKE_TESTS["base"], SMOKE_TEST.base_smoke)

    def test_strict_mode_fails_for_missing_package(self) -> None:
        with mock.patch.object(
            SMOKE_TEST, "check_packages", return_value=["missing-package"]
        ):
            self.assertEqual(SMOKE_TEST.main(["--framework", "base", "--strict"]), 1)

    def test_import_failure_reports_bounded_single_line_diagnostic(self) -> None:
        checks = {
            "broken": [SMOKE_TEST.PackageCheck("broken_module", "broken-package")]
        }
        output = io.StringIO()
        with (
            mock.patch.object(SMOKE_TEST, "CHECKS", checks),
            mock.patch.object(
                SMOKE_TEST.importlib,
                "import_module",
                side_effect=ImportError("first line\n" + "x" * 300),
            ),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(SMOKE_TEST.check_packages("broken"), ["broken_module"])

        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertIn("ImportError: first line", lines[0])
        self.assertTrue(lines[0].endswith("...)"))
        self.assertLessEqual(len(lines[0]), 190)

    def test_strict_mode_fails_with_bounded_smoke_exception(self) -> None:
        output = io.StringIO()
        failing_smoke = mock.Mock(
            side_effect=RuntimeError("circuit failed\n" + "x" * 300)
        )
        with (
            mock.patch.object(SMOKE_TEST, "check_packages", return_value=[]),
            mock.patch.dict(
                SMOKE_TEST.SMOKE_TESTS,
                {"base": failing_smoke},
            ),
            contextlib.redirect_stdout(output),
        ):
            returncode = SMOKE_TEST.main(["--framework", "base", "--strict"])

        self.assertEqual(returncode, 1)
        diagnostic = next(
            line for line in output.getvalue().splitlines() if "smoke: failed" in line
        )
        self.assertIn("RuntimeError: circuit failed", diagnostic)
        self.assertNotIn("\n", diagnostic)
        self.assertTrue(diagnostic.endswith("...)"))
        self.assertLessEqual(len(diagnostic), 190)


class RecordEnvironmentScriptTests(unittest.TestCase):
    def test_label_rejects_path_traversal_and_separators(self) -> None:
        self.assertEqual(
            RECORD_ENVIRONMENT.parse_label("experiment-2.1"),
            "experiment-2.1",
        )
        for unsafe_label in ("../escape", "nested/name", r"nested\name", ".", ""):
            with self.subTest(label=unsafe_label):
                with self.assertRaises(argparse.ArgumentTypeError):
                    RECORD_ENVIRONMENT.parse_label(unsafe_label)

    def test_package_snapshot_uses_sorted_stdlib_metadata(self) -> None:
        distributions = [
            FakeDistribution("zeta", "2"),
            FakeDistribution("Alpha", "1"),
        ]

        self.assertEqual(
            RECORD_ENVIRONMENT.installed_package_lines(distributions),
            ["Alpha==1", "zeta==2"],
        )

    def test_stdlib_fallback_reports_missing_unconditional_dependencies(self) -> None:
        distributions = [
            FakeDistribution(
                "example",
                "1",
                ["missing>=2", 'conditional; python_version < "3"'],
            )
        ]

        result = RECORD_ENVIRONMENT.stdlib_dependency_check(distributions)

        self.assertEqual(result.returncode, 1)
        self.assertIn("example: missing>=2", result.output)
        self.assertNotIn("conditional;", result.output)

    def test_uv_check_targets_the_active_interpreter(self) -> None:
        completed = subprocess.CompletedProcess([], 0, "All packages are compatible")
        with (
            mock.patch.object(RECORD_ENVIRONMENT.shutil, "which", return_value="/bin/uv"),
            mock.patch.object(
                RECORD_ENVIRONMENT.subprocess, "run", return_value=completed
            ) as run,
        ):
            result = RECORD_ENVIRONMENT.check_dependencies([])

        self.assertEqual(result.tool, "uv pip check")
        self.assertEqual(result.returncode, 0)
        self.assertTrue(result.output.endswith("\n"))
        self.assertEqual(
            run.call_args.args[0],
            ["/bin/uv", "pip", "check", "--python", sys.executable],
        )

    def test_uv_missing_dispatches_to_stdlib_fallback(self) -> None:
        distributions = [FakeDistribution("Example", "1")]
        with (
            mock.patch.object(RECORD_ENVIRONMENT.shutil, "which", return_value=None),
            mock.patch.object(
                RECORD_ENVIRONMENT,
                "stdlib_dependency_check",
                return_value=RECORD_ENVIRONMENT.DependencyCheck(
                    "fallback", 0, "fallback output\n"
                ),
            ) as fallback,
        ):
            result = RECORD_ENVIRONMENT.check_dependencies(distributions)

        self.assertEqual(result.tool, "fallback")
        fallback.assert_called_once_with(distributions)

    def test_uv_oserror_dispatches_to_stdlib_fallback(self) -> None:
        distributions = [FakeDistribution("Example", "1")]
        with (
            mock.patch.object(
                RECORD_ENVIRONMENT.shutil, "which", return_value="/bin/uv"
            ),
            mock.patch.object(
                RECORD_ENVIRONMENT.subprocess,
                "run",
                side_effect=OSError("launch denied"),
            ),
            mock.patch.object(
                RECORD_ENVIRONMENT,
                "stdlib_dependency_check",
                return_value=RECORD_ENVIRONMENT.DependencyCheck(
                    "fallback", 0, "fallback output\n"
                ),
            ) as fallback,
        ):
            result = RECORD_ENVIRONMENT.check_dependencies(distributions)

        self.assertEqual(result.tool, "fallback")
        fallback.assert_called_once()
        self.assertIn(
            "could not launch uv (OSError)",
            fallback.call_args.kwargs["reason"],
        )

    def test_uv_nonzero_output_is_preserved(self) -> None:
        completed = subprocess.CompletedProcess(
            [], 3, "incompatible dependency\nsecond line\n"
        )
        with (
            mock.patch.object(
                RECORD_ENVIRONMENT.shutil, "which", return_value="/bin/uv"
            ),
            mock.patch.object(
                RECORD_ENVIRONMENT.subprocess, "run", return_value=completed
            ),
        ):
            result = RECORD_ENVIRONMENT.check_dependencies([])

        self.assertEqual(result.returncode, 3)
        self.assertEqual(result.output, completed.stdout)

    def test_main_rejects_root_missing_dependency_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaises(FileNotFoundError):
                RECORD_ENVIRONMENT.dependency_file_hashes(root)

            stderr = io.StringIO()
            with (
                contextlib.redirect_stderr(stderr),
                self.assertRaisesRegex(SystemExit, "2"),
            ):
                RECORD_ENVIRONMENT.main(["trial", "--root", str(root)])

            self.assertIn("missing required dependency files", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertFalse((root / "requirements/locks").exists())
            self.assertFalse((root / "results/logs").exists())

    def test_main_writes_records_beneath_project_root(self) -> None:
        distributions = [FakeDistribution("Example", "1.2.3")]
        dependency_check = RECORD_ENVIRONMENT.DependencyCheck(
            "unit checker", 0, "compatible\n"
        )
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            mock.patch.object(
                RECORD_ENVIRONMENT.metadata,
                "distributions",
                return_value=distributions,
            ),
            mock.patch.object(
                RECORD_ENVIRONMENT,
                "check_dependencies",
                return_value=dependency_check,
            ),
        ):
            root = Path(temporary_directory)
            create_dependency_inputs(root)
            returncode = RECORD_ENVIRONMENT.main(
                ["trial", "--root", str(root)]
            )

            self.assertEqual(returncode, 0)
            self.assertEqual(
                (root / "requirements/locks/trial-lock.txt").read_text(
                    encoding="utf-8"
                ),
                "Example==1.2.3\n",
            )
            self.assertEqual(
                (root / "results/logs/trial-dependency-check.txt").read_text(
                    encoding="utf-8"
                ),
                "compatible\n",
            )
            platform_record = (
                root / "results/logs/trial-platform.txt"
            ).read_text(encoding="utf-8")
            self.assertIn("dependency_check_tool=unit checker", platform_record)
            self.assertIn("installed_distribution_count=1", platform_record)
            for relative_path in RECORD_ENVIRONMENT.DEPENDENCY_FILES:
                self.assertIn(
                    f"dependency_file_sha256[{relative_path.as_posix()}]=",
                    platform_record,
                )

    def test_main_refuses_to_reuse_a_label(self) -> None:
        dependency_check = RECORD_ENVIRONMENT.DependencyCheck(
            "unit checker", 0, "compatible\n"
        )
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            mock.patch.object(
                RECORD_ENVIRONMENT.metadata,
                "distributions",
                return_value=[],
            ),
            mock.patch.object(
                RECORD_ENVIRONMENT,
                "check_dependencies",
                return_value=dependency_check,
            ),
        ):
            root = Path(temporary_directory)
            create_dependency_inputs(root)
            self.assertEqual(
                RECORD_ENVIRONMENT.main(["trial", "--root", str(root)]), 0
            )
            lock_path = root / "requirements/locks/trial-lock.txt"
            original_lock = lock_path.read_text(encoding="utf-8")

            stderr = io.StringIO()
            with (
                contextlib.redirect_stderr(stderr),
                self.assertRaisesRegex(SystemExit, "2"),
            ):
                RECORD_ENVIRONMENT.main(["trial", "--root", str(root)])

            self.assertIn("refusing to overwrite existing record", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertEqual(lock_path.read_text(encoding="utf-8"), original_lock)

    def test_exclusive_write_failure_cleans_files_from_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = [root / f"record-{index}.txt" for index in range(3)]
            real_open = Path.open
            exclusive_open_count = 0

            def fail_second_exclusive_open(
                path: Path, mode: str = "r", *args: object, **kwargs: object
            ) -> object:
                nonlocal exclusive_open_count
                if mode == "x":
                    exclusive_open_count += 1
                    if exclusive_open_count == 2:
                        raise OSError("injected write failure")
                return real_open(path, mode, *args, **kwargs)

            with mock.patch.object(Path, "open", new=fail_second_exclusive_open):
                with self.assertRaisesRegex(OSError, "injected write failure"):
                    RECORD_ENVIRONMENT.write_payloads_exclusively(
                        [
                            (path, f"payload {index}\n")
                            for index, path in enumerate(paths)
                        ]
                    )

            self.assertTrue(all(not path.exists() for path in paths))

    def test_main_propagates_nonzero_check_after_writing_records(self) -> None:
        dependency_check = RECORD_ENVIRONMENT.DependencyCheck(
            "unit checker", 4, "incompatible\n"
        )
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            mock.patch.object(
                RECORD_ENVIRONMENT.metadata,
                "distributions",
                return_value=[],
            ),
            mock.patch.object(
                RECORD_ENVIRONMENT,
                "check_dependencies",
                return_value=dependency_check,
            ),
        ):
            root = Path(temporary_directory)
            create_dependency_inputs(root)
            returncode = RECORD_ENVIRONMENT.main(
                ["broken", "--root", str(root)]
            )

            self.assertEqual(returncode, 4)
            self.assertTrue((root / "requirements/locks/broken-lock.txt").is_file())
            self.assertEqual(
                (root / "results/logs/broken-dependency-check.txt").read_text(
                    encoding="utf-8"
                ),
                "incompatible\n",
            )
            self.assertTrue((root / "results/logs/broken-platform.txt").is_file())


if __name__ == "__main__":
    unittest.main()
