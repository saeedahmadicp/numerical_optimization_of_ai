"""Check that the committed ``web/src/generated/`` is what ``numopt export`` writes today.

Run from the repository root (CI runs it in the "parity" job)::

    .venv/bin/python web/tests/fixtures/check_generated.py

It exports into a temporary folder and compares the result with the committed files. The
comparison uses the parity rules of docs/architecture.md, not byte identity, because NumPy
may round the last bits differently on another CPU or BLAS:

* the same files, the same registry and problem metadata (floats to 1e-12 relative),
* the same cases ``(method, problem, params)`` in the same order,
* per case: the same ``converged`` flag and trace length, the final ``x`` to 1e-6
  (relative), the first ``min(10, n)`` trace iterates to 1e-8, and the same ``n_iter``
  for deterministic methods (stochastic methods: the final ``fun`` to 1e-6 instead).

Exit status 1 lists every difference: the committed files are stale, so run
``npm run gen`` in web/ and commit the result. ``--strict`` also requires byte identity.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

from numopt.export import export

ROOT = Path(__file__).resolve().parents[3]
COMMITTED = ROOT / "web" / "src" / "generated"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _close(a: Any, b: Any, rel: float, path: str, out: list[str]) -> None:
    """Structural equality; floats within ``rel`` (relative, with an absolute floor)."""
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or a is None:
        if a != b:
            out.append(f"{path}: {a!r} != {b!r}")
        return
    if isinstance(a, int | float) and isinstance(b, int | float):
        fa, fb = float(a), float(b)
        if fa == fb:
            return
        if not (math.isfinite(fa) and math.isfinite(fb)):
            out.append(f"{path}: {a!r} != {b!r}")
            return
        if abs(fa - fb) > rel * max(abs(fa), abs(fb), 1e-300) and abs(fa - fb) > 1e-300:
            out.append(f"{path}: {a!r} != {b!r}")
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _close(x, y, rel, f"{path}[{i}]", out)
        return
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            out.append(f"{path}: keys differ {sorted(set(a) ^ set(b))}")
            return
        for k in a:
            _close(a[k], b[k], rel, f"{path}.{k}", out)
        return
    if a != b:
        out.append(f"{path}: {a!r} != {b!r}")


def _compare_fixture(
    name: str, old: list[Any], new: list[Any], deterministic: dict[str, bool], out: list[str]
) -> None:
    keys_old = [(c["method"], c["problem"], c["params"]) for c in old]
    keys_new = [(c["method"], c["problem"], c["params"]) for c in new]
    if keys_old != keys_new:
        out.append(f"{name}: the case list differs ({len(old)} committed, {len(new)} exported)")
        return
    for c_old, c_new in zip(old, new, strict=True):
        tag = f"{name}: {c_old['method']} on {c_old['problem']}"
        r_old, r_new = c_old["result"], c_new["result"]
        if r_old["converged"] != r_new["converged"]:
            out.append(f"{tag}: converged {r_old['converged']} != {r_new['converged']}")
        if len(r_old["trace"]) != len(r_new["trace"]):
            out.append(f"{tag}: trace length {len(r_old['trace'])} != {len(r_new['trace'])}")
        if deterministic.get(c_old["method"], True):
            if r_old["n_iter"] != r_new["n_iter"]:
                out.append(f"{tag}: n_iter {r_old['n_iter']} != {r_new['n_iter']}")
        else:
            _close(r_old["fun"], r_new["fun"], 1e-6, f"{tag}: fun", out)
        _close(r_old["x"], r_new["x"], 1e-6, f"{tag}: x", out)
        n = min(10, len(r_old["trace"]), len(r_new["trace"]))
        for k in range(n):
            _close(
                r_old["trace"][k]["x"], r_new["trace"][k]["x"], 1e-8, f"{tag}: trace[{k}].x", out
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--strict", action="store_true", help="also require byte-identical files")
    ap.add_argument("--committed", type=Path, default=COMMITTED, help="folder to check")
    args = ap.parse_args()
    committed: Path = args.committed

    out: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp)
        written = export(fresh)
        names = sorted(str(p.relative_to(fresh)) for p in written)
        have = sorted(
            str(p.relative_to(committed))
            for p in [*committed.glob("*.json"), *committed.glob("fixtures/*.json")]
        )
        if names != have:
            out.append(
                f"file set differs: missing {sorted(set(names) - set(have))}, "
                f"extra {sorted(set(have) - set(names))}"
            )
        registry = _load(fresh / "registry.json")
        deterministic = {m["id"]: bool(m.get("deterministic", True)) for m in registry}
        identical = 0
        for name in names:
            old_path, new_path = committed / name, fresh / name
            if not old_path.exists():
                continue
            if old_path.read_bytes() == new_path.read_bytes():
                identical += 1
                continue
            if args.strict:
                out.append(f"{name}: not byte-identical")
            old, new = _load(old_path), _load(new_path)
            if name.startswith("fixtures/"):
                _compare_fixture(name, old, new, deterministic, out)
            else:
                _close(old, new, 1e-12, name, out)
        print(f"numopt export: {len(names)} files, {identical} byte-identical to {committed}")

    if out:
        print(f"\n{len(out)} difference(s); the committed files are stale:")
        for line in out[:200]:
            print("  " + line)
        if len(out) > 200:
            print(f"  ... and {len(out) - 200} more")
        print("\nFix: cd web && npm run gen, then commit web/src/generated.")
        return 1
    print("OK: the committed export matches the Python package (parity tolerances).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
