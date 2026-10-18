#!/usr/bin/env python3
"""Check final-book listing parity and optionally execute notebooks in isolation.

Use --profile to select notebooks supported by the active environment. Execution
uses a fresh kernel and temporary working directory for each notebook; source
notebooks and their saved outputs are never overwritten.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ('base', 'pennylane', 'qiskit', 'tfq', 'cirq')


def check_sources(path: Path, expected: dict) -> None:
    """Validate executable cells against the final book's listing hashes."""
    notebook = json.loads(path.read_text())
    hashes = []
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        ast.parse(source, filename=path.name)
        tags = cell.get('metadata', {}).get('tags', [])
        if any(tag.startswith('Listing ') for tag in tags):
            hashes.append(hashlib.sha256(source.encode()).hexdigest())
        elif tags != ['Test']:
            raise ValueError(f'{path.name}: unclassified executable cell')
        if any(output.get('output_type') == 'error' for output in cell.get('outputs', [])):
            raise ValueError(f'{path.name}: saved error output')
    if hashes != expected['listing_sha256']:
        raise ValueError(f'{path.name}: listing sources differ from the final book')


def execute_notebook(path: Path) -> None:
    """Run one notebook with the active environment's kernel in a fresh folder."""
    import nbformat
    from nbclient import NotebookClient

    os.environ.setdefault('MPLBACKEND', 'Agg')
    notebook = nbformat.read(path, as_version=4)
    with tempfile.TemporaryDirectory(prefix=f'qmlbook-{path.stem}-') as work:
        NotebookClient(
            notebook, timeout=900, kernel_name='python3', allow_errors=False,
            resources={'metadata': {'path': work}},
        ).execute()


def main() -> int:
    """Check all listings or a selected profile, returning a failing exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES)
    parser.add_argument('--execute', action='store_true',
                        help='Run selected notebooks in fresh Jupyter kernels.')
    args = parser.parse_args()
    if args.execute and args.profile is None:
        parser.error('--execute requires --profile; keep framework environments separate')
    manifest = json.loads((ROOT/'notebooks/listing_manifest.json').read_text())
    actual = {p.name for p in (ROOT/'notebooks').glob('*.ipynb')}
    if actual != set(manifest):
        raise ValueError('Notebook inventory differs from the release manifest')
    count = 0
    for name, expected in manifest.items():
        if args.profile and expected['profile'] != args.profile:
            continue
        path = ROOT/'notebooks'/name
        check_sources(path, expected)
        if args.execute:
            execute_notebook(path)
        count += 1
        print(f'PASS {name}' + (' (executed)' if args.execute else ' (source parity)'), flush=True)
    print(f'{count} notebooks passed.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
