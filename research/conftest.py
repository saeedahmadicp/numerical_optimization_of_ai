"""Run every research study in one pytest process.

Each study folder (`research/<slug>/`) is self-contained: its tests import their siblings by bare
name (`from method import ...`, `import run`, `import problems`), and several studies use the
same names. Before pytest imports a study's test file, and before each of its tests runs, this
file makes that study's folder the first entry of `sys.path` (and removes the other study
folders), and drops from `sys.modules` any bare-name module that another study loaded. Each test
then sees its own study's `method.py`, as it does when the study runs alone.

`pytest.ini` next to this file selects `--import-mode=importlib`, so the eight `test_method.py`
files do not collide either.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from hypothesis import settings

# CI (GitHub Actions sets CI=true) runs a fixed, reproducible set of examples: derandomized, no
# example database, no deadline (the shared runners' timing varies).
settings.register_profile("ci", derandomize=True, database=None, deadline=None)
if os.environ.get("CI", "").lower() == "true":
    settings.load_profile("ci")

RESEARCH = Path(__file__).resolve().parent
STUDIES = frozenset(p for p in RESEARCH.iterdir() if p.is_dir() and (p / "README.md").is_file())


def _study_of(path: Path) -> Path | None:
    path = path.resolve()
    for parent in (path, *path.parents):
        if parent in STUDIES:
            return parent
    return None


def _enter(study: Path) -> None:
    """Make `study` the folder that bare-name imports resolve to."""
    others = {str(s) for s in STUDIES if s != study}
    sys.path[:] = [p for p in sys.path if p not in others]
    if not sys.path or sys.path[0] != str(study):
        if str(study) in sys.path:
            sys.path.remove(str(study))
        sys.path.insert(0, str(study))
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if not file or "." in name:
            continue
        owner = _study_of(Path(file))
        if owner is not None and owner != study:
            del sys.modules[name]


@pytest.hookimpl(tryfirst=True)
def pytest_collectstart(collector: pytest.Collector) -> None:
    if isinstance(collector, pytest.Module) and (study := _study_of(collector.path)):
        _enter(study)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    if study := _study_of(item.path):
        _enter(study)
