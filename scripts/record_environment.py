#!/usr/bin/env python3
"""Record package and platform details for a working environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import locale
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


LABEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
REQUIREMENT_NAME_PATTERN = re.compile(
    r"\s*([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)"
)
DEPENDENCY_FILES = (
    Path("pyproject.toml"),
    Path("uv.lock"),
    Path("requirements/base.txt"),
    Path("requirements/pennylane.txt"),
    Path("requirements/qiskit.txt"),
    Path("requirements/tensorflow-quantum.txt"),
    Path("requirements/cirq.txt"),
)


@dataclass(frozen=True)
class DependencyCheck:
    """Result of checking the active interpreter's installed distributions."""

    tool: str
    returncode: int
    output: str


def parse_label(value: str) -> str:
    """Validate an artifact label so it cannot change the output directory."""

    if not LABEL_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "label must be 1-64 characters and contain only letters, numbers, "
            "dots, underscores, or hyphens; it must start with a letter or number"
        )
    return value


def distribution_name(distribution: metadata.Distribution) -> str | None:
    """Return the standardized display name exposed by a distribution."""

    name = distribution.metadata.get("Name")
    return str(name).strip() if name else None


def installed_package_lines(
    distributions: Iterable[metadata.Distribution],
) -> list[str]:
    """Format installed distributions without requiring pip."""

    packages = {
        f"{name}=={distribution.version}"
        for distribution in distributions
        if (name := distribution_name(distribution))
    }
    return sorted(packages, key=str.casefold)


def normalize_distribution_name(name: str) -> str:
    """Apply the distribution-name equivalence rules used by package indexes."""

    return re.sub(r"[-_.]+", "-", name).lower()


def stdlib_dependency_check(
    distributions: Iterable[metadata.Distribution],
    *,
    reason: str = "uv was not found on PATH",
) -> DependencyCheck:
    """Conservatively check unconditional dependency presence using stdlib data."""

    distribution_list = list(distributions)
    installed = {
        normalize_distribution_name(name)
        for distribution in distribution_list
        if (name := distribution_name(distribution))
    }
    missing: set[str] = set()
    for distribution in distribution_list:
        owner = distribution_name(distribution) or "unknown distribution"
        for requirement in distribution.requires or ():
            requirement_text, separator, _marker = requirement.partition(";")
            if separator:
                # Evaluating environment markers correctly requires a PEP 508
                # parser, so the dependency-safe fallback ignores conditional
                # requirements rather than reporting false failures.
                continue
            match = REQUIREMENT_NAME_PATTERN.match(requirement_text)
            if match and normalize_distribution_name(match.group(1)) not in installed:
                missing.add(f"{owner}: {requirement.strip()}")

    lines = [
        f"uv unavailable ({reason}); used the stdlib metadata fallback.",
        "The fallback checks presence of unconditional dependencies only; "
        "it does not evaluate version constraints or environment markers.",
    ]
    if missing:
        lines.append("Missing unconditional dependencies:")
        lines.extend(f"- {requirement}" for requirement in sorted(missing, key=str.casefold))
        returncode = 1
    else:
        lines.append("No missing unconditional dependencies found.")
        returncode = 0
    return DependencyCheck("stdlib metadata", returncode, "\n".join(lines) + "\n")


def check_dependencies(
    distributions: Iterable[metadata.Distribution],
) -> DependencyCheck:
    """Prefer uv's dependency checker and fall back safely when unavailable."""

    distribution_list = list(distributions)
    uv_executable = shutil.which("uv")
    if uv_executable is None:
        return stdlib_dependency_check(distribution_list)

    try:
        completed = subprocess.run(
            [uv_executable, "pip", "check", "--python", sys.executable],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as exc:
        return stdlib_dependency_check(
            distribution_list,
            reason=f"could not launch uv ({type(exc).__name__})",
        )

    output = completed.stdout
    if not output.endswith("\n"):
        output += "\n"
    return DependencyCheck("uv pip check", completed.returncode, output)


def file_sha256(path: Path) -> str:
    """Hash one dependency input without copying its contents into the log."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_file_hashes(root: Path) -> dict[Path, str]:
    """Validate and hash every dependency input required for a record."""

    missing = [
        relative_path
        for relative_path in DEPENDENCY_FILES
        if not (root / relative_path).is_file()
    ]
    if missing:
        missing_text = ", ".join(path.as_posix() for path in missing)
        raise FileNotFoundError(
            f"project root is missing required dependency files: {missing_text}"
        )
    return {
        relative_path: file_sha256(root / relative_path)
        for relative_path in DEPENDENCY_FILES
    }


def git_revision(root: Path) -> str:
    """Return the checked-out commit without exposing remotes or credentials."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return "unavailable"
    revision = completed.stdout.strip()
    if completed.returncode == 0 and re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
        return revision.lower()
    return "unavailable"


def platform_report(
    root: Path,
    package_count: int,
    check_tool: str,
    dependency_hashes: dict[Path, str] | None = None,
) -> str:
    """Build an allowlisted reproducibility report without dumping the environment."""

    hashes = (
        dependency_file_hashes(root)
        if dependency_hashes is None
        else dependency_hashes
    )
    missing_hashes = [path for path in DEPENDENCY_FILES if path not in hashes]
    if missing_hashes:
        missing_text = ", ".join(path.as_posix() for path in missing_hashes)
        raise ValueError(f"dependency hashes are incomplete: {missing_text}")

    python_version = " ".join(sys.version.split())
    tf_legacy_value = os.environ.get("TF_USE_LEGACY_KERAS")
    if tf_legacy_value not in {None, "0", "1"}:
        tf_legacy_value = "custom value set"

    lines = [
        f"recorded_at_utc={datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"python_version={python_version}",
        f"python_implementation={platform.python_implementation()}",
        f"python_executable={sys.executable}",
        f"python_prefix={sys.prefix}",
        f"python_base_prefix={sys.base_prefix}",
        f"platform={platform.platform()}",
        f"machine={platform.machine()}",
        f"processor={platform.processor()}",
        f"cpu_count={os.cpu_count()}",
        f"byteorder={sys.byteorder}",
        f"filesystem_encoding={sys.getfilesystemencoding()}",
        f"locale_encoding={locale.getencoding()}",
        f"tf_use_legacy_keras={tf_legacy_value or 'unset'}",
        f"installed_distribution_count={package_count}",
        f"dependency_check_tool={check_tool}",
        f"git_revision={git_revision(root)}",
    ]
    for relative_path in DEPENDENCY_FILES:
        lines.append(
            f"dependency_file_sha256[{relative_path.as_posix()}]="
            f"{hashes[relative_path]}"
        )
    return "\n".join(lines) + "\n"


def project_output_directory(root: Path, relative_path: Path) -> Path:
    """Resolve an output directory and reject symlinks that escape the root."""

    directory = (root / relative_path).resolve()
    try:
        directory.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"output directory escapes project root: {relative_path}") from exc
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_payloads_exclusively(payloads: Sequence[tuple[Path, str]]) -> None:
    """Create all record files without overwriting or leaving a partial set."""

    for path, _payload in payloads:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite existing record: {path}")

    created: list[Path] = []
    try:
        for path, payload in payloads:
            with path.open("x", encoding="utf-8") as output:
                created.append(path)
                output.write(payload)
    except Exception:
        for path in reversed(created):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "label",
        nargs="?",
        default="qml-env",
        type=parse_label,
        help="Prefix for generated environment records.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root that contains requirements/ and results/.",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    try:
        hashes = dependency_file_hashes(root)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    distributions = list(metadata.distributions())
    package_lines = installed_package_lines(distributions)
    lock_payload = "\n".join(package_lines) + "\n"

    dependency_check = check_dependencies(distributions)
    check_payload = dependency_check.output
    platform_payload = platform_report(
        root,
        len(package_lines),
        dependency_check.tool,
        hashes,
    )

    # All report contents are complete before output directories or files are
    # created, so a collection failure cannot leave a mixed record behind.
    locks_dir = project_output_directory(root, Path("requirements/locks"))
    logs_dir = project_output_directory(root, Path("results/logs"))
    lock_path = locks_dir / f"{args.label}-lock.txt"
    check_path = logs_dir / f"{args.label}-dependency-check.txt"
    platform_path = logs_dir / f"{args.label}-platform.txt"
    try:
        write_payloads_exclusively(
            [
                (lock_path, lock_payload),
                (check_path, check_payload),
                (platform_path, platform_payload),
            ]
        )
    except FileExistsError as exc:
        parser.error(str(exc))

    print(f"wrote {lock_path}")
    print(f"wrote {check_path}")
    print(f"wrote {platform_path}")
    return dependency_check.returncode


if __name__ == "__main__":
    sys.exit(main())
