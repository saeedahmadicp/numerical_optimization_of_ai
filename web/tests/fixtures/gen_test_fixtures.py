"""Generate every Python reference dump that the web test-suite reads (git ignores them all).

Run from web/ with ``npm run gen:test-fixtures`` (or from anywhere with the repo's Python)::

    .venv/bin/python web/tests/fixtures/gen_test_fixtures.py            # all of them
    .venv/bin/python web/tests/fixtures/gen_test_fixtures.py --missing  # only absent files
    .venv/bin/python web/tests/fixtures/gen_test_fixtures.py --list     # print the table

``npm test`` calls ``--missing`` first (``pretest`` → tests/fixtures/ensure.mjs), so a fresh
clone with a Python environment needs no manual step. These dumps are separate from
``web/src/generated/`` (the parity fixtures written by ``npm run gen``, which stay committed).

Every generator is deterministic: it runs numopt with fixed inputs and seeds, so a rerun with
the same numopt source on the same platform gives the same values. Another CPU or BLAS rounds
the last bits differently: gen_platform.py records whether this Python reproduces the committed
parity fixtures byte for byte, and the tests read that (tests/fixtures/platform.ts). Add a new dump to ``GENERATORS`` below and to
the ignore list in web/.gitignore.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

WEB = Path(__file__).resolve().parents[2]

# (generator script, files it writes), both relative to web/.
GENERATORS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Whether this Python rounds like the canonical export (exact or tolerant web tests).
    ("tests/fixtures/gen_platform.py", ("tests/fixtures/platform_python.json",)),
    ("tests/fixtures/gen_rng_fixture.py", ("tests/fixtures/rng_python.json",)),
    (
        "tests/cg-trust-region/gen_cg_tr_fixture.py",
        ("tests/cg-trust-region/fixtures/cg_tr_python.json",),
    ),
    ("tests/combinatorial/gen_extra_fixture.py", ("tests/combinatorial/extra_python.json",)),
    (
        "tests/derivative-free/gen_derivative_free_fixture.py",
        ("tests/derivative-free/fixtures/derivative_free_python.json",),
    ),
    (
        "tests/first-order/gen_first_order_fixture.py",
        ("tests/first-order/fixtures/first_order_python.json",),
    ),
    ("tests/interpolation/gen_oracle.py", ("tests/interpolation/fixtures/oracle.json",)),
    (
        "tests/interpolation/gen_rational_oracle.py",
        ("tests/interpolation/fixtures/rational_oracle.json",),
    ),
    (
        "tests/least-squares/gen_least_squares_fixture.py",
        ("tests/least-squares/fixtures/least_squares_python.json",),
    ),
    ("tests/linalg/fixtures/gen_linalg_cross.py", ("tests/linalg/fixtures/linalg_cross.json",)),
    ("tests/lp/gen_lp_reference.py", ("tests/lp/fixtures/lp_reference.json",)),
    (
        "tests/newton_qn/gen_newton_qn_fixture.py",
        ("tests/newton_qn/fixtures/newton_qn_python.json",),
    ),
    ("tests/pdhg/gen_pdhg_reference.py", ("tests/pdhg/fixtures/pdhg_reference.json",)),
    ("tests/regression/gen_oracle.py", ("tests/regression/fixtures/oracle.json",)),
    ("tests/roots/gen_roots_oracle.py", ("tests/roots/fixtures/roots_oracle.json.gz",)),
    ("tests/scalar/gen_scalar_fixture.py", ("tests/scalar/fixtures/scalar_python.json",)),
    (
        "tests/shared-ports/gen_shared_ports_fixture.py",
        ("tests/shared-ports/fixtures/shared_ports_python.json",),
    ),
    (
        "tests/stochastic/gen_stochastic_fixture.py",
        ("tests/stochastic/fixtures/stochastic_python.json",),
    ),
    ("tests/systems/fixtures/gen_systems_extra.py", ("tests/systems/fixtures/systems_extra.json",)),
    (
        "tests/unconstrained-promoted/gen_anderson_lstsq_fixture.py",
        ("tests/unconstrained-promoted/fixtures/anderson_lstsq.json",),
    ),
    (
        "tests/unconstrained-promoted/gen_promoted_fixture.py",
        ("tests/unconstrained-promoted/fixtures/promoted_python.json",),
    ),
    (
        "src/methods/constrained/__fixtures__/gen_constrained_reference.py",
        ("src/methods/constrained/__fixtures__/constrained_reference.json",),
    ),
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate the web test-suite's Python dumps.")
    ap.add_argument("--missing", action="store_true", help="only run generators with absent output")
    ap.add_argument("--list", action="store_true", help="print the generators and exit")
    args = ap.parse_args()

    if args.list:
        for script, outs in GENERATORS:
            state = "ok" if all((WEB / o).exists() for o in outs) else "MISSING"
            print(f"{state:8} {script} -> {', '.join(outs)}")
        return 0

    todo = [
        (script, outs)
        for script, outs in GENERATORS
        if not args.missing or not all((WEB / o).exists() for o in outs)
    ]
    if not todo:
        print(f"test fixtures: all {len(GENERATORS)} Python reference dumps present")
        return 0
    failed: list[str] = []
    for script, outs in todo:
        for o in outs:
            (WEB / o).parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        # The generators print RuntimeWarnings from deliberately divergent runs: keep stderr
        # only when a generator fails.
        proc = subprocess.run(
            [sys.executable, str(WEB / script)], cwd=WEB.parent, capture_output=True, text=True
        )
        dt = time.perf_counter() - t0
        missing = [o for o in outs if not (WEB / o).exists()]
        if proc.returncode != 0 or missing:
            failed.append(script)
            print(f"FAIL {script} ({dt:.1f} s)\n{proc.stdout}{proc.stderr}", file=sys.stderr)
        else:
            print(f"ok   {script} ({dt:.1f} s)")
    if failed:
        print(f"{len(failed)} generator(s) failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"test fixtures: generated {len(todo)} of {len(GENERATORS)} Python reference dumps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
