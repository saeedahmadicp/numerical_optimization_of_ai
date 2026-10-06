"""Reference values for tests/unconstrained-promoted/anderson-lstsq.test.ts (test data only).

Run from the repository root:

    .venv/bin/python web/tests/unconstrained-promoted/gen_anderson_lstsq_fixture.py

It writes web/tests/unconstrained-promoted/fixtures/anderson_lstsq.json: ``np.linalg.lstsq(A, b,
rcond=None)`` (x and the singular values) and ``M @ v`` on random systems, which the TS port of
anderson_gd must reproduce bit for bit. The shapes cover every dgelsd path (QR first, LQ first,
direct bidiagonalization) and the rank-deficient and badly scaled cases.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("fixtures") / "anderson_lstsq.json"

SHAPES = [
    (1, 1),
    (1, 3),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (2, 7),
    (3, 2),
    (3, 3),
    (3, 4),
    (4, 6),
    (5, 5),
    (5, 7),
    (10, 1),
    (10, 3),
    (10, 5),
    (10, 11),
    (16, 5),
]


def main() -> None:
    rng = np.random.default_rng(20261006)
    lstsq = []
    for m, n in SHAPES:
        for kind in range(8):
            A = rng.standard_normal((m, n)) * 10.0 ** rng.uniform(-6, 2, size=(1, n))
            if kind == 1 and n > 1:
                A[:, 1] = A[:, 0]  # exactly rank-deficient
            elif kind == 2 and n > 1:
                A[:, 1] = A[:, 0] * (1 + 1e-15)  # numerically rank-deficient
            elif kind == 3:
                A[:, -1] = 0.0
            elif kind == 4:
                A *= 1e-200  # dgelsd scales A up
            elif kind == 5:
                A[0, :] = 0.0
            b = rng.standard_normal(m)
            if kind == 6:
                b[:] = 0.0
            x, _, _, s = np.linalg.lstsq(A, b, rcond=None)
            lstsq.append({"A": A.tolist(), "b": b.tolist(), "x": x.tolist(), "s": s.tolist()})
    matvec = []
    for m, n in [(2, 1), (2, 2), (2, 3), (2, 5), (2, 6), (6, 5), (10, 5), (10, 6)]:
        for _ in range(10):
            M = rng.standard_normal((m, n))
            v = rng.standard_normal(n)
            matvec.append({"M": M.tolist(), "v": v.tolist(), "y": (M @ v).tolist()})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"lstsq": lstsq, "matvec": matvec}) + "\n")
    print(f"wrote {len(lstsq)} lstsq and {len(matvec)} matvec cases to {OUT}")


if __name__ == "__main__":
    main()
