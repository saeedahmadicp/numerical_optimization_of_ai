"""Shared test helpers."""

from __future__ import annotations

import json

import numpy as np

from numopt.core.types import Result


def assert_valid_result(res: Result, *, max_iter: int | None = None) -> None:
    """Contract checks every method's Result must satisfy."""
    assert res.trace, "trace must not be empty"
    assert res.trace[0].k == 0, "trace must start at k = 0"
    ks = [s.k for s in res.trace]
    assert ks == sorted(ks), "trace indices must increase"
    if max_iter is not None:
        assert res.n_iter <= max_iter
        assert len(res.trace) <= max_iter + 1
    assert res.n_iter >= 0
    assert isinstance(res.converged, bool)
    assert res.message
    # Must round-trip to strict JSON (the web app consumes it).
    json.dumps(res.to_dict(), allow_nan=False)


def close(a, b, tol: float = 1e-8) -> bool:
    return bool(
        np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=tol, rtol=tol)
    )
