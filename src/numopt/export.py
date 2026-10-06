"""Export the registry, the problem catalog and parity fixtures as JSON for the web app.

Each family package may define ``FIXTURE_CASES``: a list of
``(method_id, problem_id, params)`` tuples. ``numopt export web/src/generated`` runs
every case and writes:

* ``registry.json`` — every MethodSpec (id, family, params, ...)
* ``problems.json`` — metadata of every registered problem
* ``fixtures/<family>.json`` — ``[{method, problem, params, result}]`` with full traces

The web test-suite replays each case with the TypeScript port and checks parity.

File format: every file is a JSON array with one element per line, and each element is
compact JSON (no indentation, no spaces after ``,`` and ``:``). Floats are written with
Python's shortest round-trip ``repr``, so a reader recovers each value bit-exactly. One
element per line keeps the files small and still gives a readable per-case ``git diff``.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from . import problems as problem_lib
from .core.registry import get_method, list_methods, run
from .core.types import to_jsonable

PACKAGES = (
    "roots",
    "scalar",
    "line_search",
    "unconstrained",
    "stochastic",
    "constrained",
    "lp",
    "combinatorial",
    "linalg",
    "integration",
    "differentiation",
    "interpolation",
    "regression",
)


def fixture_cases() -> list[tuple[str, str, dict[str, Any]]]:
    cases: list[tuple[str, str, dict[str, Any]]] = []
    for pkg in PACKAGES:
        mod = importlib.import_module(f"numopt.{pkg}")
        cases.extend(getattr(mod, "FIXTURE_CASES", []))
    return cases


def build_fixtures() -> dict[str, list[dict[str, Any]]]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for method_id, problem_id, params in fixture_cases():
        spec = get_method(method_id)
        result = run(method_id, problem_lib.get(problem_id), **params)
        by_family.setdefault(spec.family, []).append(
            {
                "method": method_id,
                "problem": problem_id,
                "params": to_jsonable(params),
                "result": result.to_dict(),
            }
        )
    return by_family


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def dumps_lines(items: list[Any]) -> str:
    """A JSON array with one compact element per line (see the module docstring)."""
    if not items:
        return "[]\n"
    return "[\n" + ",\n".join(_compact(item) for item in items) + "\n]\n"


def export(out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    (out / "fixtures").mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def dump(path: Path, data: list[Any]) -> None:
        path.write_text(dumps_lines(data), encoding="utf-8")
        written.append(path)

    dump(out / "registry.json", [s.to_dict() for s in list_methods()])
    dump(
        out / "problems.json",
        [{"kind": problem_lib.kind_of(p.id), **p.to_dict()} for p in problem_lib.list_problems()],
    )
    for family, cases in build_fixtures().items():
        dump(out / "fixtures" / f"{family}.json", cases)
    return written
