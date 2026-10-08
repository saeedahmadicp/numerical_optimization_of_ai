"""Check that the committed ``web/src/generated/`` is what ``numopt export`` writes today.

Run from the repository root (CI runs it in the "parity" job, on x86-64)::

    .venv/bin/python web/tests/fixtures/check_generated.py

It exports into a temporary folder and compares the result with the committed files (the
canonical export, written on aarch64). Another CPU or BLAS rounds the last bits differently,
so the comparison is not byte identity. It is exact for everything that is not a rounded
float, and it compares floats with a tolerance that is far above rounding and far below any
change of an algorithm.

Exact (any difference fails):

* the file set; the case list ``(method, problem, params)`` and its order;
* the structure of every file and case: the same keys, the same list lengths (so the same
  trace length), the same JSON type at every leaf (int, float, str, bool, null);
* every int (``n_iter``, ``n_fev``, ``n_gev``, ``n_hev``, ``k``, indices and counts in
  ``info``), every bool (``converged``) and every null;
* every string of registry.json and problems.json; in the fixtures, every string with its
  decimal numerals masked. A message quotes rounded values (``‖r‖₂ = 0`` on one CPU,
  ``4.97e-16`` on another), so a numeral with a point or an exponent is not compared; a
  pair of integer numerals (``5 support points``, ``type (4, 4)``) must be equal.

Floats, per field group. A group g is one field over the whole run: ``x`` holds the final x
and every trace iterate x_k, ``fun`` the final value and every f_k, and so on. With
S_g = max |v| over the group (both files), every pair (a, b) in g must satisfy

    |a - b| ≤ RTOL·S_g + ATOL,    RTOL = 1e-9, ATOL = 1e-12.

* S_g is the natural magnitude of the field in that case (the size of the iterates along
  the run, the initial objective value, the initial gradient norm). A minimizer coordinate
  of 5.6e-17 on one CPU and 2.8e-17 on another (true value 0) is rounding noise relative
  to S_g ≈ 1, and passes.
* RTOL: the largest normwise difference measured between platforms is 3e-11 (cg_hager_zhang
  on rosenbrock; x86-64 AVX2 under emulation against aarch64 with its SVE, generic ARMv8
  and Neoverse-N1 OpenBLAS kernels, NumPy 2.4 and 2.5), so 1e-9 leaves a margin of 30; a
  change of a method moves its iterates far more than 1e-9.
* ATOL covers a group whose every value is rounding noise: the residual ‖b - Ax‖₂ that a
  direct solver reports as ``fun`` (0 or a few 1e-15), an objective that starts at 0. The
  test problems have data of size 1 to 1e3, so their rounding noise is below 1e-12.

The state groups ``x``, ``fun`` and ``grad_norm`` are compared this way. The diagnostics
(``step_size``, every ``info`` and ``extra`` float) are compared only for their class
(finite, +inf, -inf, NaN): many of them divide one rounding-level quantity by another once a
method has converged (the gain ratio ρ = ared/pred, Powell's restart ratio, error ratios,
an interpolated step length, a backward error), so they have no digits that two platforms
share (ρ differs by 0.9 relative between two OpenBLAS kernels on michaelis_menten). They are
functions of the state compared above, and the vitest parity tests check them against the
canonical files, where the TypeScript ports replay the aarch64 arithmetic.

registry.json and problems.json: floats with RTOL = 1e-12 per group (data constants such as
a Gram matrix A = MMᵀ, rounded once).

A fixture case whose state differs beyond this rule between platforms is chaotic with
respect to rounding (a different n_iter, or iterates that drift apart); it does not belong
in the parity fixtures (docs/architecture.md, "Parity fixtures"). The test for a new case:
rerun it with x0 changed by 1 to 3 ulp; if that breaks this rule, the case is out, unless
its iterates and values are bit-identical on every platform (only correctly rounded
arithmetic reaches them: adam on beale, powell, basin_hopping on rastrigin).

Exit status 1 lists every difference: the committed files are stale, so run ``npm run gen``
in web/ and commit the result. ``--strict`` also requires byte identity. ``--fresh DIR``
compares DIR (an existing export) instead of exporting.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
COMMITTED = ROOT / "web" / "src" / "generated"

#: Fixture floats: |a - b| ≤ RTOL·S_g + ATOL per field group g (see the module docstring).
RTOL = 1e-9
ATOL = 1e-12
#: registry.json / problems.json floats (data constants, rounded once).
META_RTOL = 1e-12
#: The state groups of a fixture case; every other float is a diagnostic.
STATE_GROUPS = frozenset({"x", "fun", "grad_norm"})

_NUMERAL = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|\b(?:nan|inf)\b")

# One float pair: (path, committed value, exported value).
FloatPair = tuple[str, float, float]


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fclass(v: float) -> str:
    return "nan" if math.isnan(v) else ("+inf" if v > 0 else "-inf") if math.isinf(v) else "fin"


def _same_text(a: str, b: str) -> bool:
    """Equal text with decimal numerals masked; integer numerals must match."""
    if a == b:
        return True
    if _NUMERAL.sub("#", a) != _NUMERAL.sub("#", b):
        return False
    for x, y in zip(_NUMERAL.findall(a), _NUMERAL.findall(b), strict=True):
        if x.lstrip("+-").isdigit() and y.lstrip("+-").isdigit() and int(x) != int(y):
            return False
    return True


def _walk(
    a: Any,
    b: Any,
    path: str,
    group: str,
    out: list[str],
    floats: dict[str, list[FloatPair]],
    same_text: Callable[[str, str], bool],
) -> None:
    """Exact comparison of everything but floats; floats are collected by group."""
    if type(a) is not type(b):
        out.append(f"{path}: type {type(a).__name__} != {type(b).__name__} ({a!r} != {b!r})")
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            out.append(f"{path}: keys differ {sorted(set(a) ^ set(b))}")
            return
        for k in a:
            sub = k if group == "" else f"{group}.{k}"
            _walk(a[k], b[k], f"{path}.{k}", sub, out, floats, same_text)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
            return
        # Every element of a list belongs to the same group: x[0] and x[1] are both "x".
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _walk(x, y, f"{path}[{i}]", group, out, floats, same_text)
    elif isinstance(a, float):
        floats.setdefault(group, []).append((path, a, b))
    elif isinstance(a, str):
        if not same_text(a, b):
            out.append(f"{path}: {a!r} != {b!r}")
    elif a != b:  # int, bool, None
        out.append(f"{path}: {a!r} != {b!r}")


def _check_group(pairs: list[FloatPair], rtol: float, atol: float, out: list[str]) -> None:
    """|a - b| ≤ rtol·S + atol with S = max |v| over the finite values of the group."""
    scale = max((abs(v) for _, a, b in pairs for v in (a, b) if math.isfinite(v)), default=0.0)
    bound = rtol * scale + atol
    for path, a, b in pairs:
        if _fclass(a) != _fclass(b):
            out.append(f"{path}: {a!r} != {b!r}")
        elif math.isfinite(a) and abs(a - b) > bound:
            out.append(f"{path}: {a!r} != {b!r} (|Δ| = {abs(a - b):.3g} > {bound:.3g})")


def _check_class(pairs: list[FloatPair], out: list[str]) -> None:
    for path, a, b in pairs:
        if _fclass(a) != _fclass(b):
            out.append(f"{path}: {a!r} != {b!r}")


def _group_of_case(key: str) -> str:
    """Field group of a result path: the trace level is merged with the result level."""
    return key.removeprefix("trace.")


def compare_case(tag: str, old: Any, new: Any, out: list[str]) -> None:
    """Compare one fixture case's ``result`` under the parity rules (module docstring)."""
    floats: dict[str, list[FloatPair]] = {}
    _walk(old, new, tag, "", out, floats, _same_text)
    merged: dict[str, list[FloatPair]] = {}
    for g, pairs in floats.items():
        merged.setdefault(_group_of_case(g), []).extend(pairs)
    for g, pairs in merged.items():
        if g in STATE_GROUPS:
            _check_group(pairs, RTOL, ATOL, out)
        else:
            _check_class(pairs, out)


def compare_fixture(name: str, old: list[Any], new: list[Any], out: list[str]) -> None:
    keys_old = [(c["method"], c["problem"], c["params"]) for c in old]
    keys_new = [(c["method"], c["problem"], c["params"]) for c in new]
    if keys_old != keys_new:
        out.append(f"{name}: the case list differs ({len(old)} committed, {len(new)} exported)")
        return
    for c_old, c_new in zip(old, new, strict=True):
        params = json.dumps(c_old["params"], separators=(",", ":"))
        tag = f"{name}: {c_old['method']} on {c_old['problem']} {params}"
        if c_old.keys() != c_new.keys():
            out.append(f"{tag}: keys differ {sorted(set(c_old) ^ set(c_new))}")
            continue
        compare_case(tag, c_old["result"], c_new["result"], out)


def compare_meta(name: str, old: Any, new: Any, out: list[str]) -> None:
    """registry.json / problems.json: exact but for floats (META_RTOL per entry and field)."""
    floats: dict[str, list[FloatPair]] = {}
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        for i, (a, b) in enumerate(zip(old, new, strict=True)):
            entry: dict[str, list[FloatPair]] = {}
            _walk(a, b, f"{name}[{i}]", "", out, entry, str.__eq__)
            floats.update({f"[{i}].{g}": p for g, p in entry.items()})
    else:
        _walk(old, new, name, "", out, floats, str.__eq__)
    for pairs in floats.values():
        _check_group(pairs, META_RTOL, 0.0, out)


def compare_dirs(committed: Path, fresh: Path, strict: bool = False) -> tuple[list[str], int, int]:
    """All differences between two export folders; also (files, byte-identical files)."""
    out: list[str] = []
    names = sorted(
        str(p.relative_to(fresh)) for p in [*fresh.glob("*.json"), *fresh.glob("fixtures/*.json")]
    )
    have = sorted(
        str(p.relative_to(committed))
        for p in [*committed.glob("*.json"), *committed.glob("fixtures/*.json")]
    )
    if names != have:
        out.append(
            f"file set differs: missing {sorted(set(names) - set(have))}, "
            f"extra {sorted(set(have) - set(names))}"
        )
    identical = 0
    for name in names:
        old_path, new_path = committed / name, fresh / name
        if not old_path.exists():
            continue
        if old_path.read_bytes() == new_path.read_bytes():
            identical += 1
            continue
        if strict:
            out.append(f"{name}: not byte-identical")
        old, new = _load(old_path), _load(new_path)
        if name.startswith("fixtures/"):
            compare_fixture(name, old, new, out)
        else:
            compare_meta(name, old, new, out)
    return out, len(names), identical


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--strict", action="store_true", help="also require byte-identical files")
    ap.add_argument("--committed", type=Path, default=COMMITTED, help="folder to check")
    ap.add_argument("--fresh", type=Path, help="compare this export instead of exporting")
    args = ap.parse_args()
    committed: Path = args.committed

    if args.fresh is not None:
        out, n_files, identical = compare_dirs(committed, args.fresh, args.strict)
    else:
        from numopt.export import export

        with tempfile.TemporaryDirectory() as tmp:
            export(Path(tmp))
            out, n_files, identical = compare_dirs(committed, Path(tmp), args.strict)
    print(f"numopt export: {n_files} files, {identical} byte-identical to {committed}")

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
