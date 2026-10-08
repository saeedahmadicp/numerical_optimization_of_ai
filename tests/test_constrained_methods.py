"""Tests for numopt.constrained.methods.

Accuracy oracle (method independent). At a listed minimizer x* with multipliers λ* and active
set A, the KKT system F(x, λ_A) = (∇f(x) + J_A(x)ᵀλ_A, c_A(x)) = 0 has the nonsingular
Jacobian K = [[∇²_xx L, J_Aᵀ], [J_A, 0]] (LICQ + second-order sufficiency, verified in
test_problems_constrained.py). First-order perturbation theory gives, for x near x*,

    ‖(x − x*, λ_A − λ*_A)‖₂ ≤ ‖K⁻¹‖₂ ‖F(x, λ_A)‖₂ (1 + O(‖F‖)),

for any λ_A. The tests compute F from the problem data (not from the method's report) and
allow a factor 2 for the O(‖F‖) term. SciPy's SLSQP is the external oracle for the
minimizers; a brute-force active-set enumeration is the oracle of the QP solver.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from conftest import assert_valid_result
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose
from scipy.optimize import minimize

import numopt
from numopt import problems
from numopt.constrained import FIXTURE_CASES, methods
from numopt.constrained.methods import solve_qp
from numopt.core.registry import get_method, list_methods
from numopt.core.types import Constraint, Problem, Result

EPS = float(np.finfo(float).eps)
METHODS = (
    "projected_gradient",
    "frank_wolfe",
    "quadratic_penalty",
    "augmented_lagrangian",
    "log_barrier",
    "sqp",
)
MULTIPLIER_METHODS = ("quadratic_penalty", "augmented_lagrangian", "log_barrier", "sqp")
PROBLEMS = [p.id for p in problems.list_problems("constrained")]
CONVEX = [pid for pid in PROBLEMS if "convex" in problems.get(pid).tags]

#: Pairs that are not applicable: ValueError with defaults (documented preconditions).
NOT_APPLICABLE = {
    ("frank_wolfe", "linear_eq_quadratic"),  # unbounded feasible set (a line)
    ("frank_wolfe", "halfplanes_quadratic"),  # unbounded wedge
    ("projected_gradient", "circle_eq"),  # nonconvex set, no projection
    ("frank_wolfe", "circle_eq"),
    ("log_barrier", "circle_eq"),  # nonlinear equality (B&V §11.1 needs Ax = b)
}
#: Legitimate failures with default parameters: converged=False at max_iter.
#: projected_gradient on Rosenbrock: a first-order method in the curved valley needs more
#: than 1000 steps. frank_wolfe: the O(1/k) rate with solutions on a face or a curved
#: boundary (Jaggi 2013, Thm. 1; hs21 has diameter ~140, so C_f is large).
KNOWN_FAILURES = {
    ("projected_gradient", "rosenbrock_disk"),
    ("frank_wolfe", "rosenbrock_disk"),
    ("frank_wolfe", "rosenbrock_unit_disk"),
    ("frank_wolfe", "hs21"),
    ("frank_wolfe", "mishra_bird_constrained"),
}
#: rosenbrock_disk has λ* = 0 on an active constraint (strict complementarity fails): the
#: barrier's KKT test (|λ_i c_i| = 1/t ≤ tol) does not make c(x) small; see
#: test_log_barrier_degenerate_rate for the exact asymptotics.
DEGENERATE = {("log_barrier", "rosenbrock_disk")}
CONVERGES = [
    (m, pid)
    for m in METHODS
    for pid in PROBLEMS
    if (m, pid) not in NOT_APPLICABLE and (m, pid) not in KNOWN_FAILURES
]


# ======================================================================================
# Helpers
# ======================================================================================


def _run(method: str, pid: str | Problem, **kw: Any) -> Result:
    prob = problems.get(pid) if isinstance(pid, str) else pid
    return numopt.run(method, prob, **kw)


def _cvals(p: Problem, x: np.ndarray) -> np.ndarray:
    return np.array([float(c.fun(x)) for c in p.constraints])


def _cjac(p: Problem, x: np.ndarray) -> np.ndarray:
    return np.array([np.asarray(c.grad(x), dtype=float) for c in p.constraints]).reshape(-1, 2)


def _is_eq(p: Problem) -> np.ndarray:
    return np.array([c.kind == "eq" for c in p.constraints], dtype=bool)


def _violation(p: Problem, x: np.ndarray) -> float:
    c = _cvals(p, x)
    eq = _is_eq(p)
    return float(np.max(np.where(eq, np.abs(c), np.maximum(c, 0.0))))


def _val(v: float | None) -> float:
    """A Step/Result objective value that must be present."""
    assert v is not None
    return v


def _grad(p: Problem, x: np.ndarray) -> np.ndarray:
    assert p.grad is not None
    return np.asarray(p.grad(x), dtype=float)


def _hess(p: Problem, x: np.ndarray) -> np.ndarray:
    assert p.hess is not None
    return np.asarray(p.hess(x), dtype=float)


def _nearest_minimum(p: Problem, x: np.ndarray) -> int:
    return int(np.argmin([np.linalg.norm(x - np.asarray(xm)) for xm in p.minima]))


def _kkt_matrix_bound(p: Problem, j: int) -> tuple[float, list[int]]:
    """‖K⁻¹‖₂ at the j-th listed minimizer and its active set."""
    xs = np.asarray(p.minima[j], dtype=float)
    lam = p.extra["multipliers"][j]
    act = list(p.extra["active"][j])
    H = _hess(p, xs) + sum(
        li * h(xs) for li, h in zip(lam, p.extra["constraint_hess"], strict=True)
    )
    JA = _cjac(p, xs)[act]
    q = len(act)
    K = np.block([[H, JA.T], [JA, np.zeros((q, q))]])
    return float(np.linalg.norm(np.linalg.inv(K), 2)), act


def _residual(p: Problem, x: np.ndarray, act: list[int], lam_A: np.ndarray | None) -> np.ndarray:
    """F(x, λ_A); λ_A = least-squares multipliers when None."""
    g = _grad(p, x)
    JA = _cjac(p, x)[act]
    if lam_A is None:
        lam_A = np.linalg.lstsq(JA.T, -g, rcond=None)[0] if act else np.zeros(0)
    return np.concatenate([g + JA.T @ lam_A, _cvals(p, x)[act]])


def _project_bruteforce(p: Problem, z: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the problem's simple set, independent of methods.py."""
    ex = p.extra
    if ex["projection"] == "box":
        lo, hi = np.asarray(ex["bounds"], dtype=float).T
        return np.minimum(np.maximum(z, lo), hi)
    if ex["projection"] == "disk":
        c0 = np.asarray(ex["disk"]["center"], dtype=float)
        d = z - c0
        r = float(np.linalg.norm(d))
        return z.copy() if r <= ex["disk"]["radius"] else c0 + d * ex["disk"]["radius"] / r
    x = _qp_bruteforce(np.eye(2), -z, ex["A_eq"], ex["b_eq"], ex["A_ub"], ex["b_ub"])
    assert x is not None
    return x[0]


def _qp_bruteforce(
    G: Any, a: Any, A_eq: Any, b_eq: Any, A_ub: Any, b_ub: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Strictly convex QP by enumerating active sets (exact oracle for tiny problems).

    For each subset S of inequality rows with |S| + m_eq ≤ n whose KKT matrix is
    nonsingular, solve the equality-constrained QP; return the first (x, λ_eq, λ_ub) that
    is primal feasible with λ_S ≥ 0. The solution of a strictly convex QP is unique, and an
    optimal active set with linearly independent rows exists (Carathéodory), so the
    enumeration finds x*.
    """
    G = np.asarray(G, dtype=float)
    a = np.asarray(a, dtype=float)
    n = a.size
    Ae = np.asarray(A_eq, dtype=float).reshape(-1, n)
    be = np.asarray(b_eq, dtype=float).reshape(-1)
    Ai = np.asarray(A_ub, dtype=float).reshape(-1, n)
    bi = np.asarray(b_ub, dtype=float).reshape(-1)
    # Drop equality rows that depend on earlier ones (consistent by construction here).
    keep: list[int] = []
    for i in range(be.size):
        if np.linalg.matrix_rank(Ae[[*keep, i]]) == len(keep) + 1:
            keep.append(i)
    Ae, be = Ae[keep], be[keep]
    me, mi = be.size, bi.size
    scale = 1.0 + float(np.abs(bi).max(initial=0.0)) + float(np.abs(be).max(initial=0.0))
    for size in range(0, min(n - me, mi) + 1):
        for S in itertools.combinations(range(mi), size):
            A = np.vstack([Ae, Ai[list(S)]])
            b = np.concatenate([be, bi[list(S)]])
            q = A.shape[0]
            K = np.block([[G, A.T], [A, np.zeros((q, q))]])
            if np.linalg.matrix_rank(K) < n + q:
                continue
            sol = np.linalg.solve(K, np.concatenate([-a, b]))
            x, lam = sol[:n], sol[n:]
            if np.all(Ai @ x <= bi + 1e-9 * scale) and np.all(lam[me:] >= -1e-9 * scale):
                lam_ub = np.zeros(mi)
                lam_ub[list(S)] = lam[me:]
                lam_eq = np.zeros(np.asarray(b_eq, dtype=float).size)
                lam_eq[keep] = lam[:me]
                return x, lam_eq, lam_ub
    return None


def _counting_problem(p: Problem) -> tuple[Problem, dict[str, int]]:
    cnt: dict[str, int] = dict.fromkeys(("f", "g", "h", "c", "cg", "ch"), 0)

    def wrap(fn: Callable[..., Any], key: str) -> Callable[..., Any]:
        def w(x: Any) -> Any:
            cnt[key] += 1
            return fn(x)

        return w

    cons = tuple(
        Constraint(c.kind, wrap(c.fun, "c"), wrap(c.grad, "cg"), c.latex) for c in p.constraints
    )
    extra = dict(p.extra)
    extra["constraint_hess"] = tuple(wrap(h, "ch") for h in p.extra["constraint_hess"])
    q = dataclasses.replace(
        p,
        f=wrap(p.f, "f"),
        grad=wrap(p.grad, "g"),  # type: ignore[arg-type]
        hess=wrap(p.hess, "h"),  # type: ignore[arg-type]
        constraints=cons,
        extra=extra,
    )
    return q, cnt


def _resolution(v: float) -> float:
    """The module's resolution 100ε·max(1, |v|) of a value of f."""
    return 100 * EPS * max(1.0, abs(v))


def _armijo_trials(prev: Any, s: Any, kind: str) -> list[tuple[float, float, float, bool]]:
    """(step, f, predicted decrease, rounding regime) of every Armijo trial of step s.

    Recomputed from the trace alone, with the method's arithmetic: projected_gradient
    predicts ∇f(x)ᵀ(x − z) along the arc, frank_wolfe γ·gap; a trial is in the rounding
    regime when the prediction is ≤ the resolution of f(x) and f(z) is finite.
    """
    x0 = np.asarray(prev.x)
    out = []
    for j, (step, fz) in enumerate(s.info["trials"]):
        if kind == "projected_gradient":
            pred = float(np.asarray(s.info["gradient"]) @ (x0 - np.asarray(s.info["arc"][j])))
        else:
            pred = step * prev.info["gap"]
        rounding = pred <= _resolution(_val(prev.fun)) and math.isfinite(fz)
        out.append((step, fz, pred, rounding))
    return out


def _extra_gradients(prev: Any, s: Any, kind: str) -> int:
    """Gradients evaluated at rejected rounding-regime trials (the accepted one is reused)."""
    f0 = _val(prev.fun)
    trials = _armijo_trials(prev, s, kind)
    used = [rnd and fz <= f0 + _resolution(f0) for _, fz, _, rnd in trials]
    return sum(used) - int(used[-1])


# ======================================================================================
# Registry and fixtures
# ======================================================================================


def test_registered_methods_and_params():
    ids = {s.id for s in list_methods("constrained")}
    assert set(METHODS) <= ids
    for m in METHODS:
        spec = get_method(m)
        assert spec.family == "constrained" and spec.summary and spec.references
        assert spec.fn.__doc__ and "Stops (converged)" in spec.fn.__doc__
        for ps in spec.params:
            if ps.kind in ("float", "int"):
                assert ps.min is not None and ps.max is not None
                assert ps.min <= ps.default <= ps.max, (m, ps.name)
        json.dumps([ps.to_dict() for ps in spec.params], allow_nan=False)


def test_fixture_cases_cover_every_method_and_run():
    covered = {m for m, _, _ in FIXTURE_CASES}
    assert covered == set(METHODS)
    assert 3 <= len(methods.FIXTURE_CASES) <= 8
    for m, pid, params in FIXTURE_CASES:
        res = _run(m, pid, **params)
        assert res.converged, (m, pid, res.message)
        assert len(res.trace) < 300
        assert_valid_result(res)


def test_every_info_key_is_documented():
    doc = methods.__doc__ or ""
    section = doc.split("Info keys")[1]
    for m in METHODS:
        for pid in PROBLEMS:
            if (m, pid) in NOT_APPLICABLE:
                continue
            res = _run(m, pid, max_iter=5)
            for s in res.trace:
                for key in s.info:
                    assert re.search(rf"\b{key}\b", section), f"{m}: info key {key!r} undocumented"


# ======================================================================================
# The full method × problem matrix with defaults
# ======================================================================================


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("pid", PROBLEMS)
def test_outcome_matrix(method, pid):
    if (method, pid) in NOT_APPLICABLE:
        with pytest.raises(ValueError):
            _run(method, pid)
        return
    res = _run(method, pid)
    spec = get_method(method)
    assert_valid_result(res, max_iter=spec.defaults()["max_iter"])
    if (method, pid) in KNOWN_FAILURES:
        assert not res.converged and "max_iter" in res.message
        # Even when not converged, every iterate is feasible (projection / convex hull).
        p = problems.get(pid)
        assert all(_violation(p, np.asarray(s.x)) <= 1e-12 for s in res.trace)
    else:
        assert res.converged, res.message


@pytest.mark.parametrize(("method", "pid"), CONVERGES)
def test_solution_certificate(method, pid):
    """Independent KKT certificate of a converged run (see the module docstring)."""
    p = problems.get(pid)
    res = _run(method, pid)
    tol = get_method(method).defaults()["tol"]
    x = np.asarray(res.x, dtype=float)
    # 1. Feasibility, recomputed from the problem.
    viol = _violation(p, x)
    assert viol <= tol
    assert math.isclose(res.extra["violation"], viol, rel_tol=1e-12, abs_tol=1e-300)
    # 2. Step.fun and Result.fun are f(x).
    assert res.fun == float(p.f(x)) and res.trace[-1].fun == res.fun
    # 3. Distance to the nearest listed minimizer is explained by the KKT residual.
    j = _nearest_minimum(p, x)
    C, act = _kkt_matrix_bound(p, j)
    F = _residual(p, x, act, None)
    err = float(np.linalg.norm(x - np.asarray(p.minima[j])))
    assert err <= 2.0 * C * float(np.linalg.norm(F)) + 10 * EPS
    # 4. ... and the residual is O(tol): each component is a stopping-test quantity times a
    # constraint-gradient scale, except c_A for the barrier, where |λ_i c_i| = 1/t ≤ tol
    # gives |c_i| ≤ tol/λ*_i.
    lam_star = np.asarray(p.extra["multipliers"][j])
    pos = [abs(lam_star[i]) for i in act if lam_star[i] != 0.0]
    inv_lam = max([1.0, *(1.0 / v for v in pos)])
    if (method, pid) not in DEGENERATE:
        assert float(np.linalg.norm(F)) <= 10.0 * inv_lam * tol * (1.0 + np.abs(_cjac(p, x)).max())
    # 5. The objective value.
    assert abs(_val(res.fun) - p.extra["minima_f"][j]) <= 2.0 * C * float(np.linalg.norm(F)) * (
        1.0 + float(np.linalg.norm(p.grad(x)))
    ) + 1e-12 * (1.0 + abs(_val(res.fun)))


@pytest.mark.parametrize(("method", "pid"), [c for c in CONVERGES if c[0] in MULTIPLIER_METHODS])
def test_multiplier_certificate(method, pid):
    p = problems.get(pid)
    res = _run(method, pid)
    tol = get_method(method).defaults()["tol"]
    x = np.asarray(res.x, dtype=float)
    lam = np.asarray(res.extra["multipliers"], dtype=float)
    eq = _is_eq(p)
    c = _cvals(p, x)
    J = _cjac(p, x)
    # The reported KKT residual is honest: recompute it from the problem data.
    stat = float(np.max(np.abs(p.grad(x) + J.T @ lam)))
    comp = float(np.max(np.abs(lam[~eq] * c[~eq]), initial=0.0))
    dual = float(np.max(np.maximum(-lam[~eq], 0.0), initial=0.0))
    kkt = max(stat, comp, dual)
    assert math.isclose(res.extra["kkt_residual"], kkt, rel_tol=1e-9, abs_tol=1e-15)
    assert kkt <= tol
    # Multipliers of the active set: same perturbation bound, with the method's λ_A.
    j = _nearest_minimum(p, x)
    C, act = _kkt_matrix_bound(p, j)
    inactive = [i for i in range(c.size) if i not in act]
    F = _residual(p, x, act, lam[act])
    F[:2] += J[inactive].T @ lam[inactive]  # F uses λ on A only; the rest is ≤ tol/|c_i|
    lam_star = np.asarray(p.extra["multipliers"][j])
    assert float(np.linalg.norm(lam[act] - lam_star[act])) <= 2.0 * C * float(
        np.linalg.norm(F)
    ) + 10 * EPS * (1.0 + float(np.abs(lam_star).max()))
    # Inactive multipliers vanish up to complementarity: λ_i ≤ tol/|c_i(x)|.
    for i in inactive:
        assert abs(lam[i]) <= tol / abs(c[i]) * (1 + 1e-12)


@pytest.mark.parametrize(("method", "pid"), CONVERGES)
def test_scipy_slsqp_oracle(method, pid):
    """The converged x agrees with SciPy's SLSQP started from the same x0."""
    p = problems.get(pid)
    res = _run(method, pid)
    cons = [
        {"type": "ineq", "fun": lambda x, c=c: -c.fun(x), "jac": lambda x, c=c: -c.grad(x)}
        if c.kind == "ineq"
        else {"type": "eq", "fun": c.fun, "jac": c.grad}
        for c in p.constraints
    ]
    x0 = np.asarray(p.x0, dtype=float)
    if method in ("projected_gradient", "frank_wolfe"):
        x0 = np.asarray(res.trace[0].x, dtype=float)  # the projected start
    ref = minimize(
        p.f,
        x0,
        jac=p.grad,
        constraints=cons,
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 500},
    )
    x = np.asarray(res.x, dtype=float)
    j = _nearest_minimum(p, x)
    C, act = _kkt_matrix_bound(p, j)
    bound = 2.0 * C * float(np.linalg.norm(_residual(p, x, act, None)))
    # SLSQP is accurate to 1e-6 here (test_problems_constrained.py).
    assert float(np.linalg.norm(x - ref.x)) <= bound + 1e-6
    assert _nearest_minimum(p, ref.x) == j  # both found the same local minimizer


# ======================================================================================
# Trace contract, info geometry and exact counts
# ======================================================================================

COMMON_KEYS = {"outer", "inner", "constraints", "active", "violation", "kkt_residual"}
STATE_KEYS = {
    "projected_gradient": set(),
    "frank_wolfe": {"lmo", "gap"},
    "quadratic_penalty": {"mu", "tau", "merit", "multipliers", "stationarity"},
    "augmented_lagrangian": {
        "mu",
        "lambda",
        "omega",
        "eta",
        "update",
        "merit",
        "multipliers",
        "stationarity",
    },
    "log_barrier": {"t", "gap", "merit", "multipliers", "stationarity"},
    "sqp": {"mu", "merit", "multipliers", "stationarity"},
}
MOVE_KEYS = {
    "projected_gradient": {"from", "gradient", "unprojected", "s", "arc", "trials"},
    "frank_wolfe": {"from", "vertex", "direction", "gamma", "trials"},
    "quadratic_penalty": {"from", "direction", "alpha", "trials"},
    "augmented_lagrangian": {"from", "direction", "alpha", "trials"},
    "log_barrier": {"from", "direction", "alpha", "trials", "newton_decrement", "hessian_shift"},
    "sqp": {"from", "direction", "alpha", "trials", "working_set", "theta"},
}
STEP_SIZE_KEY = {
    "projected_gradient": "s",
    "frank_wolfe": "gamma",
    "quadratic_penalty": "alpha",
    "augmented_lagrangian": "alpha",
    "log_barrier": "alpha",
    "sqp": "alpha",
}


@pytest.mark.parametrize(
    ("method", "pid"),
    [(m, pid) for m in METHODS for pid in PROBLEMS if (m, pid) not in NOT_APPLICABLE],
)
def test_trace_contract_and_info(method, pid):
    p = problems.get(pid)
    res = _run(method, pid, max_iter=60)
    assert_valid_result(res, max_iter=60)
    assert [s.k for s in res.trace] == list(range(len(res.trace)))
    assert res.n_iter == res.trace[-1].k
    eq = _is_eq(p)
    for s in res.trace:
        x = np.asarray(s.x, dtype=float)
        keys = set(s.info) - {"projected_from"}
        expected = COMMON_KEYS | STATE_KEYS[method] | (MOVE_KEYS[method] if s.k else set())
        assert keys == expected, (s.k, keys ^ expected)
        if "projected_from" in s.info:
            assert s.k == 0
        c = _cvals(p, x)
        assert_allclose(s.info["constraints"], c, rtol=0, atol=0)
        assert s.info["violation"] == _violation(p, x)
        assert s.info["active"] == [
            i for i in range(c.size) if eq[i] or c[i] >= -methods.ACTIVE_TOL
        ]
        assert s.fun == float(p.f(x))
        assert math.isclose(_val(s.grad_norm), float(np.linalg.norm(p.grad(x))), rel_tol=1e-15)
        if s.k:
            assert s.step_size == s.info[STEP_SIZE_KEY[method]]
            assert s.info["trials"][-1][0] == s.step_size  # the accepted trial is the last
            if method != "frank_wolfe":
                d = np.asarray(s.info.get("direction", [0, 0]), dtype=float)
                if method != "projected_gradient":  # x_k = x_{k−1} + α p
                    assert_allclose(
                        x, np.asarray(s.info["from"]) + s.step_size * d, rtol=1e-15, atol=1e-15
                    )
            else:  # x_k = (1 − γ) x_{k−1} + γ s_{k−1}
                g = s.info["gamma"]
                xp = (1 - g) * np.asarray(s.info["from"]) + g * np.asarray(s.info["vertex"])
                assert_allclose(x, xp, rtol=1e-15, atol=1e-15)
    json.dumps(res.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    ("method", "pid"),
    [(m, pid) for m in METHODS for pid in PROBLEMS if (m, pid) not in NOT_APPLICABLE],
)
def test_exact_evaluation_counts(method, pid):
    p, cnt = _counting_problem(problems.get(pid))
    res = _run(method, p, max_iter=40)
    assert (res.n_fev, res.n_gev, res.n_hev) == (cnt["f"], cnt["g"], cnt["h"])
    assert res.extra["n_cev"] == cnt["c"]
    assert res.extra["n_cgev"] == cnt["cg"]
    assert res.extra["n_chev"] == cnt["ch"]
    trials = sum(len(s.info["trials"]) for s in res.trace[1:])
    if method in ("projected_gradient", "frank_wolfe", "sqp"):
        # One f per line-search trial plus f(x0); one gradient per accepted iterate, plus
        # one per rejected Armijo trial in the rounding regime (projection methods).
        assert res.n_fev == 1 + trials
        extra = (  # frank_wolfe runs its default open-loop rule here: no Armijo trials
            sum(_extra_gradients(a, b, method) for a, b in itertools.pairwise(res.trace))
            if method == "projected_gradient"
            else 0
        )
        assert res.n_gev == 1 + res.n_iter + extra
    if method == "log_barrier":
        assert res.n_hev == 1 + res.n_iter  # one Newton system per step
        feasible_trials = sum(
            1 for s in res.trace[1:] for _, v in s.info["trials"] if v != math.inf
        )
        assert res.n_fev == 1 + feasible_trials  # f is not evaluated outside the domain


# ======================================================================================
# Failure paths
# ======================================================================================


@pytest.mark.parametrize("method", METHODS)
def test_max_iter_reported(method):
    pid = "rosenbrock_unit_disk" if method != "frank_wolfe" else "quadratic_disk"
    res = _run(method, pid, max_iter=3)
    assert not res.converged and "max_iter=3" in res.message
    assert res.n_iter == 3 and len(res.trace) == 4
    assert_valid_result(res, max_iter=3)


@pytest.mark.parametrize("method", METHODS)
def test_nonfinite_objective_at_start(method):
    p = dataclasses.replace(problems.get("box_quadratic"), f=lambda x: math.nan)
    res = _run(method, p)
    assert not res.converged and "non-finite" in res.message
    assert res.n_iter == 0
    assert_valid_result(res)


@pytest.mark.parametrize("method", ["projected_gradient", "frank_wolfe", "sqp"])
def test_nonfinite_region_is_never_accepted(method):
    # f is NaN for x ≥ 0.2: the line searches reject NaN trial values (a NaN comparison is
    # False), so no accepted iterate has a NaN value (Frank–Wolfe's open-loop step cannot
    # backtrack and stops with a clear message instead).
    base = problems.get("box_quadratic")
    p = dataclasses.replace(base, f=lambda x: base.f(x) if x[0] < 0.2 else math.nan)
    res = _run(method, p, max_iter=50)
    assert not res.converged
    assert_valid_result(res, max_iter=50)
    if method == "frank_wolfe":
        assert "non-finite" in res.message
    else:
        assert all(s.fun is not None and math.isfinite(s.fun) for s in res.trace)


def _infeasible_problem() -> Problem:
    """x ≤ −1 and x ≥ 1: no feasible point."""

    def lin(a: list[float], b: float) -> Constraint:
        av = np.asarray(a, dtype=float)
        return Constraint("ineq", lambda x: float(av @ x - b), lambda x: av.copy(), "")

    return Problem(
        id="infeasible",
        name="infeasible",
        latex="",
        f=lambda x: float(x @ x) + float(x[1]),
        grad=lambda x: 2 * np.asarray(x) + np.array([0.0, 1.0]),
        hess=lambda x: 2 * np.eye(2),
        dim=2,
        domain=((-2.0, 2.0), (-2.0, 2.0)),
        x0=[0.0, 0.0],
        constraints=(lin([1, 0], -1), lin([-1, 0], -1)),
        extra={
            "projection": "polyhedron",
            "A_ub": [[1, 0], [-1, 0]],
            "b_ub": [-1, -1],
            "affine": [True, True],
            "vertices": [[0.0, 0.0]],
        },
    )


def test_infeasible_problem():
    p = _infeasible_problem()
    for m in ("quadratic_penalty", "augmented_lagrangian"):
        res = _run(m, p)
        assert not res.converged and "mu_max" in res.message
        assert res.extra["violation"] >= 0.99  # ≥ 1 at every point; x stays near x = 0
        assert_valid_result(res)
    res = _run("sqp", p)
    assert not res.converged and "inconsistent" in res.message
    for m in ("projected_gradient", "frank_wolfe", "log_barrier"):
        with pytest.raises(ValueError):
            _run(m, p)


def test_float64_floors_are_reported_as_stalls():
    # Mishra's disk c = (x+5)² + (y+5)² − 25 cancels terms of size 25 at the boundary, so
    # δc ≈ 1e-14 (‖∇c‖ = 10 on the circle). At the boundary minimizers (λ* = 6.42, 8.01):
    # * quadratic_penalty needs μ ≥ λ*²/tol ≈ 4e7 for |λc| = λ*²/μ ≤ tol, and then ∇Q
    #   carries the rounding error μ‖∇c‖δc ≈ 1e-5 > tol: the iteration reaches a fixed point.
    # * log_barrier: λ = −1/(tc) has relative error tλ*δc, giving a stationarity error
    #   λ*²t‖∇c‖δc ≥ λ*²‖∇c‖δc/tol ≈ 6e-6 > tol once 1/t ≤ tol.
    p = problems.get("mishra_bird_constrained")
    res = _run("quadratic_penalty", p, x0=[-10.0, -10.0])
    assert not res.converged and res.message.startswith("stalled")
    assert np.linalg.norm(np.asarray(res.x) - np.asarray(p.minima[1])) <= 1e-6
    assert res.trace[-1].info["mu"] >= p.extra["multipliers"][1][0] ** 2 / 1e-6
    assert res.extra["kkt_residual"] <= 1e-5
    res = _run("log_barrier", p, x0=[-9.5, -4.0])
    assert not res.converged and res.message.startswith("stalled at the rounding level")
    assert np.linalg.norm(np.asarray(res.x) - np.asarray(p.minima[3])) <= 1e-6
    assert_valid_result(res)
    # Both reach the tolerance the floor allows.
    assert _run("quadratic_penalty", p, x0=[-10.0, -10.0], tol=1e-4).converged
    assert _run("log_barrier", p, x0=[-9.5, -4.0], tol=1e-4).converged


def test_sqp_stops_when_licq_fails():
    # At the centre of circle_eq, ∇c = 0: the linearized constraint 0ᵀp + c = −1 = 0 is
    # inconsistent, so the QP subproblem has no feasible point.
    res = _run("sqp", "circle_eq", x0=[0.0, 0.0])
    assert not res.converged and "inconsistent" in res.message and res.n_iter == 0


def test_quadratic_penalty_reports_mu_cap():
    res = _run("quadratic_penalty", "quadratic_disk", mu_max=1e3)
    assert not res.converged and "mu_max" in res.message
    # The penalty minimizer is infeasible by ≈ λ*/(2μ‖∇c‖)… still visibly infeasible.
    assert res.extra["violation"] > 1e-6


def test_invalid_input_raises():
    with pytest.raises(ValueError):
        _run("projected_gradient", "box_quadratic", beta=1.5)
    with pytest.raises(ValueError):
        _run("frank_wolfe", "box_quadratic", step="exact")
    with pytest.raises(ValueError):
        _run("quadratic_penalty", "box_quadratic", rho=1.0)
    with pytest.raises(ValueError):
        _run("augmented_lagrangian", "box_quadratic", mu_factor=0.5)
    with pytest.raises(ValueError):
        _run("log_barrier", "box_quadratic", mu=1.0)
    with pytest.raises(ValueError, match="strictly feasible"):
        _run("log_barrier", "quadratic_disk", x0=[1.0, 0.0])  # on the boundary
    with pytest.raises(ValueError, match="exact projection"):
        _run("projected_gradient", "circle_eq")
    with pytest.raises(ValueError, match="compact"):
        _run("frank_wolfe", "halfplanes_quadratic")
    with pytest.raises(TypeError):
        numopt.run("sqp", problems.get("hs21"), unknown=1)


def test_bare_callable_problems():
    def rosen(x):
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    for m in ("sqp", "log_barrier", "quadratic_penalty", "augmented_lagrangian"):
        res = numopt.run(m, rosen, x0=[-1.2, 1.0])  # unconstrained, finite differences
        assert res.converged, (m, res.message)
        assert_allclose(res.x, [1.0, 1.0], atol=1e-6)
        assert_valid_result(res)
    with pytest.raises(ValueError):
        numopt.run("projected_gradient", rosen, x0=[0.0, 0.0])


@pytest.mark.parametrize("n", [1, 3])
@pytest.mark.parametrize("method", MULTIPLIER_METHODS)
def test_bare_callable_of_any_dimension(method, n):
    # A bare callable takes its dimension from x0 (not the 2-D default of a Problem).
    def shifted_sphere(x):
        return float(np.sum((np.asarray(x) - 1.0) ** 2))

    res = numopt.run(method, shifted_sphere, x0=np.zeros(n))
    assert res.converged, res.message
    assert np.asarray(res.x).shape == (n,)
    # ∇f = 2(x − 1) and the stationarity test is ‖∇f‖∞ ≤ tol (finite-difference gradients,
    # whose error is far below tol for a quadratic).
    tol = get_method(method).defaults()["tol"]
    assert_allclose(res.x, np.ones(n), rtol=0, atol=tol)
    assert_valid_result(res)


@pytest.mark.parametrize("method", ["projected_gradient", "frank_wolfe"])
def test_bare_callable_of_any_dimension_needs_a_feasible_set(method):
    with pytest.raises(ValueError, match="exact projection"):
        numopt.run(method, lambda x: float(np.sum(np.asarray(x) ** 2)), x0=np.zeros(3))


def _kkt_from_reported_multipliers(p: Problem, res: Result) -> float:
    """The KKT residual of (Result.x, Result.extra['multipliers']), from the problem data."""
    x = np.asarray(res.x, dtype=float)
    lam = np.asarray(res.extra["multipliers"], dtype=float)
    eq = _is_eq(p)
    c = _cvals(p, x)
    stat = float(np.max(np.abs(_grad(p, x) + _cjac(p, x).T @ lam)))
    comp = float(np.max(np.abs(lam[~eq] * c[~eq]), initial=0.0))
    dual = float(np.max(np.maximum(-lam[~eq], 0.0), initial=0.0))
    return max(stat, comp, dual)


@pytest.mark.parametrize("method", MULTIPLIER_METHODS)
@pytest.mark.parametrize("pid", ["quadratic_disk", "hs21", "circle_eq"])
def test_reported_multipliers_match_reported_kkt_residual_at_every_stop(method, pid):
    # Every max_iter < n_iter stops the run at a different place, including right after an
    # outer (μ, λ or t) update; the reported (multipliers, kkt_residual) must be one pair.
    if (method, pid) in NOT_APPLICABLE:
        return
    p = problems.get(pid)
    full = _run(method, pid)
    assert full.converged, full.message
    for max_iter in range(0, full.n_iter + 1):
        res = _run(method, pid, max_iter=max_iter)
        kkt = _kkt_from_reported_multipliers(p, res)
        assert math.isclose(res.extra["kkt_residual"], kkt, rel_tol=1e-9, abs_tol=1e-15), (
            max_iter,
            res.message,
        )


def test_penalty_multipliers_after_an_outer_update_are_those_of_the_last_step():
    # quadratic_penalty stopped by max_iter right after μ ← ρμ: the estimate that belongs to
    # x is μ_k c(x) of the merit x minimized (N&W eq. 17.9), not ρμ_k c(x).
    p = problems.get("quadratic_disk")
    full = _run("quadratic_penalty", p)
    stops = [
        (prev.k, s.info["mu"] / prev.info["mu"])
        for prev, s in itertools.pairwise(full.trace)
        if s.info["outer"] != prev.info["outer"]
    ]
    assert stops  # at least one outer update happened
    for k, ratio in stops:
        assert ratio > 1.0
        res = _run("quadratic_penalty", p, max_iter=k)
        assert not res.converged and "max_iter" in res.message
        assert res.extra["multipliers"] == res.trace[-1].info["multipliers"]
        assert res.extra["kkt_residual"] == res.trace[-1].info["kkt_residual"]


def test_reported_kkt_is_consistent_after_a_stall():
    p = problems.get("mishra_bird_constrained")
    res = _run("quadratic_penalty", p, x0=[-10.0, -10.0])
    assert res.message.startswith("stalled")
    assert math.isclose(
        res.extra["kkt_residual"], _kkt_from_reported_multipliers(p, res), rel_tol=1e-9
    )


def test_log_barrier_start_within_rounding_of_the_boundary_stops_as_stalled():
    # x0 on the unit circle with c(x0) = −1.1e-16 < 0: strictly feasible in float64, but
    # λ = −1/(t c) ≈ 9e15 and the Newton step is below the spacing of the floats at x0.
    # The accepted step leaves x unchanged: a fixed point, reported at once.
    p = problems.get("quadratic_disk")
    x0 = np.asarray(p.minima[0], dtype=float)
    assert -1e-15 < float(p.constraints[0].fun(x0)) < 0.0
    res = _run("log_barrier", p, x0=x0)
    assert not res.converged and res.message.startswith("stalled")
    assert "max_I c_i(x)" in res.message
    assert res.n_iter == 0  # no repeated iterate is recorded
    assert_valid_result(res)


@settings(max_examples=1000, deadline=None)
@given(st.floats(0.0, 2.0 * math.pi))
def test_log_barrier_never_repeats_an_iterate_from_boundary_starts(theta):
    p = problems.get("quadratic_disk")
    x0 = np.array([math.cos(theta), math.sin(theta)])
    if not float(p.constraints[0].fun(x0)) < 0.0:
        return  # on or outside the circle in float64: ValueError (tested elsewhere)
    res = _run("log_barrier", p, x0=x0)
    assert_valid_result(res, max_iter=500)
    assert res.converged or res.message.startswith("stalled"), res.message
    xs = [np.asarray(s.x) for s in res.trace]
    assert not any(np.array_equal(a, b) for a, b in itertools.pairwise(xs))


def _circle_equality_problem(extra: dict[str, Any]) -> Problem:
    """min x + y s.t. x² + y² − 2 = 0 (a nonlinear equality), built from scratch."""
    return Problem(
        id="user_circle",
        name="user circle",
        latex="",
        f=lambda x: float(x[0] + x[1]),
        grad=lambda x: np.array([1.0, 1.0]),
        hess=lambda x: np.zeros((2, 2)),
        dim=2,
        domain=((-2.0, 2.0), (-2.0, 2.0)),
        x0=[0.5, -0.5],
        constraints=(
            Constraint("eq", lambda x: float(x[0] ** 2 + x[1] ** 2 - 2.0), lambda x: 2.0 * x),
        ),
        extra=extra,
    )


def test_log_barrier_rejects_equalities_not_declared_affine():
    ce = problems.get("circle_eq")
    without_flag = dataclasses.replace(
        ce, extra={k: v for k, v in ce.extra.items() if k != "affine"}
    )
    for p in (without_flag, _circle_equality_problem({})):
        with pytest.raises(ValueError, match="not declared affine"):
            _run("log_barrier", p)
    # An affine equality without the flag is rejected too (the precondition is the flag)…
    lin = dataclasses.replace(
        problems.get("linear_eq_quadratic"),
        extra={k: v for k, v in problems.get("linear_eq_quadratic").extra.items() if k != "affine"},
    )
    with pytest.raises(ValueError, match="not declared affine"):
        _run("log_barrier", lin)
    # … and accepted with it.
    assert _run("log_barrier", "linear_eq_quadratic").converged


def test_log_barrier_reports_ascent_when_an_equality_is_falsely_declared_affine():
    # With a nonlinear equality, ∇FᵀΔx = −ΔxᵀHΔx + wᵀc_E can be positive: the method must
    # stop with that reason instead of accepting the step (or blaming the boundary).
    p = _circle_equality_problem({"affine": [True]})
    res = _run("log_barrier", p)
    assert not res.converged and "not a descent direction" in res.message
    assert_valid_result(res)


def _assert_names_failing_quantities(res: Result) -> None:
    """A stall message (tol = 0) names exactly the failing quantities of the returned x."""
    kkt, viol = res.extra["kkt_residual"], res.extra["violation"]
    # _excess names a quantity only when it is > tol (checked from given values below).
    assert methods._excess(kkt, viol, 0.0) in res.message, (kkt, viol, res.message)


def test_stall_messages_name_the_failing_quantities():
    # tol = 0 cannot be met, so these runs end at the rounding level: at a fixed point
    # ("stalled") or, where the last bits cycle instead (it depends on the CPU's rounding), at
    # max_iter. Which of KKT residual and violation is exactly 0 there also depends on the
    # rounding (sqp on circle_eq: KKT residual 0 on one CPU, 2.2e-16 on another), so the
    # message is checked against the values the run reports.
    res = _run("sqp", "circle_eq", tol=0.0)
    assert not res.converged
    assert res.message.startswith(("stalled", "reached max_iter=200")), res.message
    assert res.extra["kkt_residual"] < 1e-12 and res.extra["violation"] < 1e-12
    if res.message.startswith("stalled"):
        _assert_names_failing_quantities(res)
    # log_barrier without inequalities: no barrier parameter or boundary in the message.
    res = _run("log_barrier", "linear_eq_quadratic", tol=0.0)
    assert not res.converged
    assert res.message.startswith(("stalled", "reached max_iter")), res.message
    assert "boundary" not in res.message and "t =" not in res.message
    if res.message.startswith("stalled"):
        _assert_names_failing_quantities(res)
    # The messages, from given values: only the failing quantities are named.
    assert methods._stalled_msg(0.0, 2e-3, 1e-6) == (
        "stalled: the accepted step leaves x unchanged (violation 0.002 > tol = 1e-06); the "
        "merit function cannot resolve further progress in float64"
    )
    # The helper names exactly the failing quantities.
    assert methods._excess(1e-3, 0.0, 1e-6) == "KKT residual 0.001 > tol = 1e-06"
    assert methods._excess(0.0, 2e-3, 1e-6) == "violation 0.002 > tol = 1e-06"
    assert methods._excess(1.0, 2.0, 0.5) == "KKT residual 1 and violation 2 > tol = 0.5"


@pytest.mark.parametrize("method", ["quadratic_penalty", "augmented_lagrangian"])
def test_penalty_ui_ranges_cannot_select_mu_max_below_mu0(method):
    spec = {ps.name: ps for ps in get_method(method).params}
    mu0, mu_max = spec["mu0"], spec["mu_max"]
    assert mu0.max is not None and mu_max.min is not None and mu_max.min >= mu0.max
    for m0 in (mu0.min, mu0.max):
        for mx in (mu_max.min, mu_max.max):
            res = _run(method, "quadratic_disk", mu0=m0, mu_max=mx, max_iter=50)
            assert_valid_result(res, max_iter=50)


def test_sqp_penalty_rule_18_36():
    # μ_k ≥ (∇f_kᵀp + ½pᵀB_k p)/((1 − ρ)‖c_k⁻‖₁) with pᵀB_k p ≥ 0 implies the descent
    # condition D = ∇f_kᵀp − μ_k‖c_k⁻‖₁ ≤ −ρμ_k‖c_k⁻‖₁ (ρ = 0.5); at a feasible x_k (18.36) is
    # void and μ keeps its value (documented NOTE in sqp).
    for pid, x0 in (
        ("quadratic_disk", [-0.5, 0.5]),
        ("rosenbrock_unit_disk", None),
        ("hs21", None),
    ):
        p = problems.get(pid)
        res = _run("sqp", pid, **({"x0": x0} if x0 is not None else {}))
        assert res.converged
        eq = _is_eq(p)
        for prev, s in itertools.pairwise(res.trace):
            c = np.asarray(prev.info["constraints"])
            v1 = float(np.sum(np.abs(c[eq])) + np.sum(np.maximum(c[~eq], 0.0)))
            mu = s.info["mu"]
            if v1 == 0.0:
                assert mu == prev.info["mu"]
                continue
            gp = float(_grad(p, np.asarray(prev.x)) @ np.asarray(s.info["direction"]))
            assert gp - mu * v1 <= -0.5 * mu * v1 + 1e-12 * (abs(gp) + mu * v1)


# ======================================================================================
# Method-specific mathematics
# ======================================================================================


def _documented_armijo(p: Problem, prev: Any, s: Any, kind: str, c1: float) -> list[bool]:
    """Acceptance of each trial by the documented rule, with an independent ∇f(z).

    Outside the rounding regime: Armijo f(x) − f(z) ≥ c₁·pred. Inside: f(z) ≤ f(x) + res
    and ∇f(z)ᵀ(z − x) ≤ (1 − 2c₁)·pred (approximate Wolfe, Hager & Zhang 2005).
    """
    x0 = np.asarray(prev.x)
    f0 = _val(prev.fun)
    out = []
    for j, (step, fz, pred, rounding) in enumerate(_armijo_trials(prev, s, kind)):
        if kind == "projected_gradient":
            z = np.asarray(s.info["arc"][j])
        else:
            z = (1.0 - step) * x0 + step * np.asarray(s.info["vertex"])
        if not rounding:
            out.append(f0 - fz >= c1 * pred)
        else:
            slope = float(_grad(p, z) @ (z - x0))
            out.append(fz <= f0 + _resolution(f0) and slope <= (1 - 2 * c1) * pred)
    return out


@pytest.mark.parametrize(
    ("pid", "kw", "regime"),
    [
        ("hs21", {}, False),
        # The audit's start: a decrease below the resolution of f from k ≈ 20 on.
        ("mishra_bird_constrained", {"x0": [-6.743790354647398, -5.4696050530085465]}, True),
    ],
)
def test_projected_gradient_steps_follow_the_armijo_rule_on_the_arc(pid, kw, regime):
    p = problems.get(pid)
    res = _run("projected_gradient", pid, tol=1e-12, **kw)
    n_regime = 0
    for prev, s in itertools.pairwise(res.trace):
        x0 = np.asarray(prev.x)
        g = np.asarray(s.info["gradient"])
        assert_allclose(g, p.grad(x0), rtol=0, atol=0)
        for (sk, _), z in zip(s.info["trials"], s.info["arc"], strict=True):
            assert_allclose(z, _project_bruteforce(p, x0 - sk * g), rtol=0, atol=1e-10)
        accepted = _documented_armijo(p, prev, s, "projected_gradient", 1e-4)
        assert accepted == [sk == s.step_size for sk, _ in s.info["trials"]]
        n_regime += sum(r for *_, r in _armijo_trials(prev, s, "projected_gradient"))
        # Decrease, or a change below the resolution of f in the rounding regime.
        assert _val(s.fun) <= _val(prev.fun) + _resolution(_val(prev.fun))
    assert res.converged, res.message
    assert (n_regime > 0) == regime


def test_projected_gradient_residual_is_the_documented_measure():
    for pid in (
        "quadratic_disk",
        "box_quadratic",
        "hs21",
        "halfplanes_quadratic",
        "linear_eq_quadratic",
    ):
        p = problems.get(pid)
        res = _run("projected_gradient", pid)
        for s in res.trace:
            x = np.asarray(s.x)
            r = float(np.max(np.abs(x - _project_bruteforce(p, x - p.grad(x)))))
            assert math.isclose(s.info["kkt_residual"], r, rel_tol=1e-6, abs_tol=1e-12)


def test_frank_wolfe_gap_bounds_the_primal_error():
    # Jaggi (2013), eq. (2): for convex f, g(x) = ∇f(x)ᵀ(x − s) ≥ f(x) − f*.
    for pid, step in [
        ("quadratic_disk", "open_loop"),
        ("box_quadratic", "armijo"),
        ("hs21", "open_loop"),
    ]:
        p = problems.get(pid)
        res = _run("frank_wolfe", pid, step=step, max_iter=200)
        for s in res.trace:
            assert s.info["gap"] >= _val(s.fun) - p.extra["f_min"] - 1e-12 * (1 + abs(_val(s.fun)))


def test_frank_wolfe_open_loop_rate():
    # Jaggi (2013), Thm. 1: f(x_k) − f* ≤ 2 C_f/(k + 2), C_f ≤ diam(C)² L. On the unit disk,
    # diam = 2 and L = 2 (∇²f = 2I), so f(x_k) − f* ≤ 16/(k + 2).
    p = problems.get("quadratic_disk")
    res = _run("frank_wolfe", "quadratic_disk", max_iter=200)
    for s in res.trace[1:]:
        assert _val(s.fun) - p.extra["f_min"] <= 16.0 / (s.k + 2)
        assert s.info["gamma"] == 2.0 / (s.k - 1 + 2)


def test_frank_wolfe_lmo_is_the_best_vertex():
    for pid in ("box_quadratic", "hs21", "quadratic_disk"):
        p = problems.get(pid)
        res = _run("frank_wolfe", pid, max_iter=20)
        for s in res.trace:
            g = p.grad(np.asarray(s.x))
            v = np.asarray(s.info["lmo"])
            if "vertices" in p.extra:
                best = min(float(g @ np.asarray(w)) for w in p.extra["vertices"])
            else:  # disk: min over the circle is c − r g/‖g‖
                best = -p.extra["disk"]["radius"] * float(np.linalg.norm(g))
            assert math.isclose(float(g @ v), best, rel_tol=1e-12, abs_tol=1e-12)


def test_quadratic_penalty_needs_mu_of_order_lambda_over_tol():
    # Penalty minimizers violate an active constraint by c ≈ λ*/μ (N&W eq. 17.9), so the
    # violation test c ≤ tol forces μ ≳ ‖λ*‖∞/tol; the augmented Lagrangian does not.
    for pid in ("quadratic_disk", "box_quadratic", "halfplanes_quadratic", "circle_eq", "hs21"):
        lam_inf = max(abs(v) for v in problems.get(pid).extra["multipliers"][0])
        rp = _run("quadratic_penalty", pid)
        ra = _run("augmented_lagrangian", pid)
        assert rp.trace[-1].info["mu"] >= 0.5 * lam_inf / 1e-6
        assert ra.trace[-1].info["mu"] <= 1e4
        # μ grows geometrically by rho per outer iteration.
        for s in rp.trace:
            assert math.isclose(s.info["mu"], 10.0 ** s.info["outer"], rel_tol=1e-12)


def test_augmented_lagrangian_outer_updates_follow_alg_17_4():
    res = _run("augmented_lagrangian", "halfplanes_quadratic")
    for prev, s in itertools.pairwise(res.trace):
        if s.info["outer"] == prev.info["outer"]:
            assert s.info["lambda"] == prev.info["lambda"] and s.info["mu"] == prev.info["mu"]
            # Armijo on L_A with fixed λ, μ, or a change at the rounding level of L_A (the
            # documented rule of the inner line search when the decrease is unresolvable).
            m0 = prev.info["merit"]
            assert s.info["merit"] <= m0 + 100 * EPS * max(1.0, abs(m0))
            continue
        if s.info["update"] == "multipliers":
            # λ ← λ̃(x) at the last iterate of the previous outer iteration (N&W 17.39).
            assert_allclose(s.info["lambda"], prev.info["multipliers"], rtol=1e-15, atol=0)
            assert s.info["mu"] == prev.info["mu"]
        else:
            assert s.info["update"] == "penalty"
            assert s.info["lambda"] == prev.info["lambda"]
            assert math.isclose(s.info["mu"], 100.0 * prev.info["mu"], rel_tol=1e-15)


def test_log_barrier_central_path_and_duality_gap():
    p = problems.get("box_quadratic")
    res = _run("log_barrier", "box_quadratic")
    m_in = len(p.constraints)
    for s in res.trace:
        t = s.info["t"]
        assert math.isclose(t, 10.0 ** s.info["outer"], rel_tol=1e-12)
        assert math.isclose(s.info["gap"], m_in / t, rel_tol=1e-15)
        lam = np.asarray(s.info["multipliers"])
        c = np.asarray(s.info["constraints"])
        assert np.all(c < 0.0)  # strictly feasible
        assert_allclose(lam * c, -1.0 / t, rtol=1e-14)  # λ_i = −1/(t c_i), B&V eq. 11.10
        # Weak duality: f(x) − f* ≤ gap of the dual point when x is centered (Newton
        # decrement tiny) — check at the first step of each new stage's predecessor.
    for prev, s in itertools.pairwise(res.trace):
        if s.info["outer"] == prev.info["outer"]:
            assert s.info["merit"] <= prev.info["merit"] * (1 + 1e-15) + 1e-300
    assert res.fun - p.extra["f_min"] <= m_in / res.trace[-1].info["t"] + 1e-12


def test_log_barrier_degenerate_rate():
    # rosenbrock_disk: x* = (1, 1) on the circle with λ* = 0. Near x*, the central point
    # solves H d = a/(t aᵀd) (a = ∇c(x*), H = ∇²f(x*)), so d = −H⁻¹a/√(t aᵀH⁻¹a):
    # the error decays like t^(−1/2), not t^(−1) as for a nondegenerate constraint.
    p = problems.get("rosenbrock_disk")
    res = _run("log_barrier", "rosenbrock_disk")
    assert res.converged
    xs = np.array([1.0, 1.0])
    t = res.trace[-1].info["t"]
    H = p.hess(xs)
    a = p.constraints[0].grad(xs)
    Ha = np.linalg.solve(H, a)
    d_pred = -Ha / math.sqrt(t * float(a @ Ha))
    d = np.asarray(res.x) - xs
    # Relative accuracy of the leading-order term: O(‖d‖) ≈ 2e-3, plus centering error.
    assert_allclose(d, d_pred, rtol=0, atol=5e-3 * float(np.linalg.norm(d_pred)))


def test_log_barrier_projects_x0_onto_affine_equalities():
    res = _run("log_barrier", "linear_eq_quadratic", x0=[0.0, 0.0])
    assert res.trace[0].info["projected_from"] == [0.0, 0.0]
    assert_allclose(res.trace[0].x, [0.5, 0.5], rtol=0, atol=1e-15)  # nearest point of x+y=1
    assert res.converged and res.n_iter == 1  # Newton is exact on an equality-constrained QP
    assert_allclose(res.x, [2 / 3, 1 / 3], rtol=0, atol=1e-15)


def test_log_barrier_hessian_shift_on_nonconvex_problem():
    # det ∇²f = 400 − 80000(y − x²) < 0 at (0, 1): t∇²f + ∇²φ is indefinite for t = 1, and
    # N&W Alg. 3.3 adds τI before the first Newton step.
    res = _run("log_barrier", "rosenbrock_disk", x0=[0.0, 1.0])
    assert res.converged
    assert res.trace[1].info["hessian_shift"] > 0
    for s in res.trace[1:]:
        assert s.info["hessian_shift"] >= 0.0
    res = _run("log_barrier", "quadratic_disk")
    assert all(s.info["hessian_shift"] == 0.0 for s in res.trace[1:])  # convex: no shift


def test_sqp_superlinear_on_rosenbrock_unit_disk():
    p = problems.get("rosenbrock_unit_disk")
    res = _run("sqp", "rosenbrock_unit_disk")
    assert res.converged
    xs = np.asarray(p.minima[0])
    err = [float(np.linalg.norm(np.asarray(s.x) - xs)) for s in res.trace]
    # The last steps contract by much more than any linear rate (N&W Thm. 18.5 for BFGS).
    ratios = [err[i + 1] / err[i] for i in range(len(err) - 4, len(err) - 1)]
    assert max(ratios) < 0.1
    assert res.trace[-1].step_size == 1.0  # full steps near the solution


@pytest.mark.parametrize(
    "pid", ["rosenbrock_unit_disk", "hs21", "circle_eq", "mishra_bird_constrained"]
)
def test_sqp_iteration_rules(pid):
    p = problems.get(pid)
    eq = _is_eq(p)
    res = _run("sqp", pid)
    assert res.converged
    for prev, s in itertools.pairwise(res.trace):
        xk = np.asarray(prev.x)
        pk = np.asarray(s.info["direction"])
        lam_hat = np.asarray(s.info["multipliers"])  # λ_{k+1} = λ̂ (QP multipliers)
        # The QP step satisfies the linearized constraints, with λ̂_I ≥ 0, λ̂ = 0 off the
        # working set and the working set active in the linearization (N&W 18.11).
        lin = _cjac(p, xk) @ pk + _cvals(p, xk)
        scale = 1e-10 * (1.0 + np.abs(_cvals(p, xk)).max())
        assert np.all(lin[~eq] <= scale) and np.all(np.abs(lin[eq]) <= scale)
        assert np.all(lam_hat[~eq] >= 0.0)
        ws = s.info["working_set"]
        assert all(lam_hat[i] == 0.0 for i in range(eq.size) if i not in ws)
        assert all(abs(lin[i]) <= scale for i in ws)
        # μ never decreases (18.36), and the accepted step satisfies the Armijo condition on
        # φ₁(·; μ_k) or changes φ₁ only at the rounding level.
        mu = s.info["mu"]
        assert mu >= prev.info["mu"]
        c_prev = np.asarray(prev.info["constraints"])
        l1_prev = float(np.sum(np.abs(c_prev[eq])) + np.sum(np.maximum(c_prev[~eq], 0.0)))
        phi_prev = prev.fun + mu * l1_prev
        assert s.info["merit"] <= phi_prev + 100 * EPS * max(1.0, abs(phi_prev))
        assert s.info["merit"] == s.info["trials"][-1][1]


# ======================================================================================
# QP solver (Goldfarb–Idnani) against brute-force enumeration
# ======================================================================================


def _check_qp(G, a, A_eq, b_eq, A_ub, b_ub) -> None:
    G = np.asarray(G, dtype=float)
    a = np.asarray(a, dtype=float)
    ref = _qp_bruteforce(G, a, A_eq, b_eq, A_ub, b_ub)
    res = solve_qp(G, a, A_eq, b_eq, A_ub, b_ub)
    assert ref is not None
    assert res.ok, res.message
    n = a.size
    Ae = np.asarray(A_eq, dtype=float).reshape(-1, n)
    be = np.asarray(b_eq, dtype=float).reshape(-1)
    Ai = np.asarray(A_ub, dtype=float).reshape(-1, n)
    bi = np.asarray(b_ub, dtype=float).reshape(-1)
    x = res.x
    scale = 1.0 + np.abs(a).max() + np.abs(G).max() * (1.0 + np.abs(x).max())
    # KKT of the returned solution.
    r = G @ x + a + Ae.T @ res.lam_eq + Ai.T @ res.lam_ub
    lam_scale = (
        1.0 + float(np.abs(res.lam_ub).max(initial=0)) + float(np.abs(res.lam_eq).max(initial=0))
    )
    assert np.abs(r).max() <= 1e-10 * scale * lam_scale
    assert np.all(np.abs(Ae @ x - be) <= 1e-10 * scale)
    assert np.all(Ai @ x - bi <= 1e-10 * scale)
    assert np.all(res.lam_ub >= 0.0)
    assert np.all(np.abs(res.lam_ub * (Ai @ x - bi)) <= 1e-9 * scale * lam_scale)
    # Positive multipliers belong to the final active set, whose rows hold with equality.
    assert {j for j in range(bi.size) if res.lam_ub[j] > 0} <= set(res.active)
    assert all(abs(Ai[j] @ x - bi[j]) <= 1e-10 * scale for j in res.active)

    # Strong convexity: q(x) − q(x*) ≥ (λ_min(G)/2)‖x − x*‖² for feasible x.
    def q(v):
        return 0.5 * v @ G @ v + a @ v

    dq = abs(q(x) - q(ref[0]))
    sigma = float(np.linalg.eigvalsh(G).min())
    assert float(np.linalg.norm(x - ref[0])) <= math.sqrt(2 * dq / sigma) + 1e-8 * scale


def test_qp_textbook_example():
    # N&W Example 16.4: min (x1 − 1)² + (x2 − 2.5)² s.t. five linear inequalities;
    # solution (1.4, 1.7) with only constraint 1 active, multiplier 0.8.
    A_ub = -np.array([[1.0, -2.0], [-1.0, -2.0], [-1.0, 2.0], [1.0, 0.0], [0.0, 1.0]])
    b_ub = -np.array([-2.0, -6.0, -2.0, 0.0, 0.0])
    G = 2 * np.eye(2)
    a = np.array([-2.0, -5.0])
    res = solve_qp(G, a, None, None, A_ub, b_ub)
    assert res.ok and res.active == (0,)
    assert_allclose(res.x, [1.4, 1.7], rtol=0, atol=1e-14)
    assert_allclose(res.lam_ub, [0.8, 0, 0, 0, 0], rtol=0, atol=1e-14)
    _check_qp(G, a, np.zeros((0, 2)), [], A_ub, b_ub)


def test_qp_degenerate_and_failure_cases():
    G = np.eye(2)
    a = np.array([-2.0, -2.0])
    # Duplicate active rows (degenerate vertex: x ≤ 1/4 twice, on the line x + y = 1).
    _check_qp(G, a, [[1.0, 1.0]], [1.0], [[1.0, 0.0], [1.0, 0.0]], [0.25, 0.25])
    res = solve_qp(G, a, [[1.0, 1.0]], [1.0], [[1.0, 0.0], [1.0, 0.0]], [0.25, 0.25])
    assert_allclose(res.x, [0.25, 0.75], atol=1e-15)
    # Gx + a = (−1.75, −1.25) = −ν(1, 1) − (λ₁ + λ₂)(1, 0): ν = 1.25, λ₁ + λ₂ = 0.5; only
    # the sum of the multipliers of the duplicated row is determined.
    assert_allclose(res.lam_ub.sum(), 0.5, atol=1e-15)
    # A redundant (linearly dependent, consistent) equality is skipped.
    res = solve_qp(G, a, [[1.0, 1.0], [2.0, 2.0]], [1.0, 2.0])
    assert res.ok
    assert_allclose(res.x, [0.5, 0.5], atol=1e-15)
    # Inconsistent inequalities / equalities.
    assert not solve_qp(G, a, None, None, [[1.0, 0.0], [-1.0, 0.0]], [0.0, -1.0]).ok
    bad = solve_qp(G, a, [[1.0, 1.0], [1.0, 1.0]], [1.0, 2.0])
    assert not bad.ok and "inconsistent" in bad.message
    # Not positive definite.
    assert not solve_qp(np.diag([1.0, -1.0]), a).ok
    with pytest.raises(ValueError):
        solve_qp(G, a, [[1.0, 0.0]], [1.0, 2.0])


def _equality_qp_oracle(G: np.ndarray, a: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """min ½xᵀGx + aᵀx s.t. Ax = b (A full row rank) from the KKT system, independent of G–I."""
    n, q = a.size, b.size
    K = np.block([[G, A.T], [A, np.zeros((q, q))]])
    return np.linalg.solve(K, np.concatenate([-a, b]))[:n]


@settings(max_examples=1000, deadline=None)
@given(
    n=st.integers(2, 6),
    data=st.data(),
)
def test_qp_redundant_and_paired_constraints_with_ill_conditioned_g(n, data):
    # Each of k independent affine equalities a_iᵀx = b_i is written redundantly: as an
    # equality plus a scaled copy, or as the two inequalities a_iᵀx ≤ b_i and −a_iᵀx ≤ −b_i
    # (plus scaled copies). The QP is consistent by construction and equals the equality QP
    # on the k independent rows. With cond(G) up to 1e5 and ‖G⁻¹a‖ up to 1e6, the slack of a
    # dependent row carries rounding far above 1e-12 relative; it must not be reported as
    # inconsistent (audit finding: s_p alone was tested).
    k = data.draw(st.integers(1, n - 1))
    seed = data.draw(st.integers(0, 2**32 - 1))
    rng = np.random.default_rng(seed)
    Qg, _ = np.linalg.qr(rng.standard_normal((n, n)))
    kappa = 10.0 ** data.draw(st.floats(0.0, 5.0))
    G = Qg @ np.diag(np.geomspace(1.0, kappa, n)) @ Qg.T
    G = 0.5 * (G + G.T)
    a = rng.standard_normal(n) * 10.0 ** data.draw(st.floats(0.0, 6.0))
    Qa, _ = np.linalg.qr(rng.standard_normal((n, k)))
    A = Qa.T * rng.uniform(0.5, 2.0, size=(k, 1))  # well-conditioned independent rows
    x_feas = rng.standard_normal(n) * 10.0
    b = A @ x_feas
    eq_rows: list[np.ndarray] = []
    eq_rhs: list[float] = []
    ub_rows: list[np.ndarray] = []
    ub_rhs: list[float] = []
    for i in range(k):
        scale = float(data.draw(st.sampled_from([2.0, -3.0, 0.5])))
        mode = data.draw(st.sampled_from(["eq_copy", "ineq_pair", "ineq_pair_copy"]))
        if mode == "eq_copy":
            eq_rows += [A[i], scale * A[i]]
            eq_rhs += [b[i], scale * b[i]]
        else:
            ub_rows += [A[i], -A[i]]
            ub_rhs += [b[i], -b[i]]
            if mode == "ineq_pair_copy":
                s = abs(scale)
                ub_rows += [s * A[i], -s * A[i]]
                ub_rhs += [s * b[i], -s * b[i]]
    Ae = np.array(eq_rows).reshape(-1, n)
    Ai = np.array(ub_rows).reshape(-1, n)
    res = solve_qp(G, a, Ae, np.array(eq_rhs), Ai, np.array(ub_rhs))
    assert res.ok, res.message
    x_ref = _equality_qp_oracle(G, a, A, b)
    # Forward error of a backward-stable KKT solve: ≈ κ(K)·ε relative (Higham 2002, §7.1);
    # G–I reach the same point through a sequence of rank-one active-set steps, which may
    # add a factor of the number of steps. Bound fixed before running: 1e3·κ(K)·ε.
    K = np.block([[G, A.T], [A, np.zeros((k, k))]])
    bound = 1e3 * float(np.linalg.cond(K)) * EPS * (1.0 + float(np.linalg.norm(x_ref)))
    assert float(np.linalg.norm(res.x - x_ref)) <= bound
    assert np.all(res.lam_ub >= 0.0)


@pytest.mark.parametrize("seed", range(20))
def test_qp_consistent_dependent_equality_after_a_long_step(seed):
    # Audit case A shape: a duplicated equality rᵀx = b, 2rᵀx = 2b, cond(G) = 3.2e4 and an
    # unconstrained minimizer x_unc = x* + 3.5e5·G⁻¹r/‖G⁻¹r‖ whose constrained minimizer x*
    # is small. After the long step onto rᵀx = b the residual of r is ≈ ε‖r‖‖x_unc‖ ≫
    # 1e-12(1 + |b| + ‖r‖‖x*‖), so a test on |s_p| alone called the copy inconsistent (in 183
    # of 200 seeds); s_p − r·s_N cancels that rounding.
    rng = np.random.default_rng(seed)
    n, kappa, dist = 2, 3.2e4, 3.5e5
    Qg, _ = np.linalg.qr(rng.standard_normal((n, n)))
    G = Qg @ np.diag([1.0, kappa]) @ Qg.T
    G = 0.5 * (G + G.T)
    r = rng.standard_normal(n)
    x_star = rng.standard_normal(n)
    b = float(r @ x_star)
    d = np.linalg.solve(G, r)
    x_unc = x_star + dist * d / np.linalg.norm(d)  # Gx* + a = −(dist/‖d‖) r: x* is optimal
    res = solve_qp(G, -G @ x_unc, np.array([r, 2.0 * r]), np.array([b, 2.0 * b]))
    assert res.ok, res.message
    # Forward error of solving with G at the scale of the step: κ(G)·ε·‖x_unc − x*‖.
    assert float(np.linalg.norm(res.x - x_star)) <= np.linalg.cond(G) * EPS * dist
    # The copy is skipped as redundant: the whole multiplier sits on the first row.
    assert res.lam_eq[1] == 0.0
    # Stationarity Gx + a + λ₀r = 0 up to the rounding of forming Gx + a, whose terms have
    # size ‖G‖‖x_unc‖ (n-term dot products: n·ε·‖G‖‖x_unc‖, Higham 2002, §3.1; factor 10).
    a = -G @ x_unc
    station = float(np.linalg.norm(G @ res.x + a + res.lam_eq[0] * r))
    assert station <= 10 * n * EPS * np.linalg.norm(G, 2) * np.linalg.norm(x_unc)


_small = st.integers(-3, 3)


@settings(max_examples=1000, deadline=None)
@given(
    n=st.integers(2, 3),
    data=st.data(),
)
def test_qp_matches_bruteforce(n, data):
    M = np.array(data.draw(st.lists(_small, min_size=n * n, max_size=n * n)), dtype=float).reshape(
        n, n
    )
    G = M.T @ M + 0.5 * np.eye(n)
    a = np.array(data.draw(st.lists(st.floats(-5, 5), min_size=n, max_size=n)))
    x_feas = np.array(data.draw(st.lists(st.floats(-2, 2), min_size=n, max_size=n)))
    mi = data.draw(st.integers(0, 5))
    me = data.draw(st.integers(0, 1))
    Ai = np.array(
        data.draw(st.lists(_small, min_size=mi * n, max_size=mi * n)), dtype=float
    ).reshape(mi, n)
    slack = np.array(
        data.draw(st.lists(st.sampled_from([0.0, 0.5, 1.0, 3.0]), min_size=mi, max_size=mi))
    )
    Ae = np.array(
        data.draw(st.lists(_small, min_size=me * n, max_size=me * n)), dtype=float
    ).reshape(me, n)
    _check_qp(G, a, Ae, Ae @ x_feas, Ai, Ai @ x_feas + slack)


# ======================================================================================
# Hypothesis invariants of the methods
# ======================================================================================

PROJECTABLE = [pid for pid in PROBLEMS if "projection" in problems.get(pid).extra]
INTERIOR_START = {
    "quadratic_disk": lambda u, v: [
        0.95 * math.sqrt(u) * math.cos(6.28 * v),
        0.95 * math.sqrt(u) * math.sin(6.28 * v),
    ],
    "box_quadratic": lambda u, v: [-0.99 + 1.98 * u, -0.99 + 1.98 * v],
    "halfplanes_quadratic": lambda u, v: [-2.0 + 2.6 * u - 0.3 * v, -2.0 + 2.6 * v - 0.3 * u],
    "hs21": lambda u, v: [2.5 + 47 * u, -49 + 98 * v * min(1.0, (10 * (2.5 + 47 * u) - 10) / 50)],
}


def _domain_point(p: Problem, u: float, v: float) -> list[float]:
    (x_lo, x_hi), (y_lo, y_hi) = p.domain
    return [x_lo + u * (x_hi - x_lo), y_lo + v * (y_hi - y_lo)]


@settings(max_examples=1000, deadline=None)
@given(st.sampled_from(PROJECTABLE), st.floats(-0.5, 1.5), st.floats(-0.5, 1.5))
def test_projection_variational_inequality(pid, u, v):
    # P_C(z) is characterized by (z − P(z))ᵀ(y − P(z)) ≤ 0 for all y ∈ C (Bertsekas Prop. 2.1.3).
    p = problems.get(pid)
    C = methods._FeasibleSet(p, "test")
    z = np.asarray(_domain_point(p, u, v))
    Pz = C.project(z)
    assert _violation(p, Pz) <= 1e-12 * (1 + np.abs(z).max())
    assert_allclose(Pz, _project_bruteforce(p, z), rtol=0, atol=1e-10 * (1 + np.abs(z).max()))
    assert_allclose(C.project(Pz), Pz, rtol=0, atol=1e-12 * (1 + np.abs(z).max()))
    ys = [np.asarray(xm) for xm in p.minima] + [np.asarray(w) for w in p.extra.get("vertices", [])]
    for y in ys:
        assert float((z - Pz) @ (y - Pz)) <= 1e-9 * (1 + np.abs(z).max()) ** 2


@settings(max_examples=1000, deadline=None)
@given(
    st.sampled_from([pid for pid in PROJECTABLE if pid in CONVEX]),
    st.floats(0.0, 1.0),
    st.floats(0.0, 1.0),
)
def test_projected_gradient_feasible_monotone_and_convergent(pid, u, v):
    p = problems.get(pid)
    res = _run("projected_gradient", pid, x0=_domain_point(p, u, v), max_iter=1000)
    assert_valid_result(res, max_iter=1000)
    fs = [_val(s.fun) for s in res.trace]
    # Armijo: decrease, or a change below the resolution of f in the rounding regime.
    assert all(b <= a + _resolution(a) for a, b in itertools.pairwise(fs))
    assert all(_violation(p, np.asarray(s.x)) <= 1e-10 for s in res.trace)
    assert res.converged, res.message
    C, act = _kkt_matrix_bound(p, 0)
    x = np.asarray(res.x)
    assert (
        np.linalg.norm(x - p.minima[0])
        <= 2 * C * np.linalg.norm(_residual(p, x, act, None)) + 1e-14
    )


@settings(max_examples=1000, deadline=None)
@given(
    st.sampled_from(["quadratic_disk", "box_quadratic", "hs21"]),
    st.sampled_from(["open_loop", "armijo"]),
    st.floats(-0.2, 1.2),
    st.floats(-0.2, 1.2),
)
def test_frank_wolfe_iterates_stay_feasible(pid, step, u, v):
    p = problems.get(pid)
    res = _run("frank_wolfe", pid, x0=_domain_point(p, u, v), step=step, max_iter=100)
    assert_valid_result(res, max_iter=100)
    assert all(_violation(p, np.asarray(s.x)) <= 1e-12 * (1 + np.abs(s.x).max()) for s in res.trace)
    for s in res.trace:  # Jaggi eq. (2)
        assert s.info["gap"] >= _val(s.fun) - p.extra["f_min"] - 1e-12 * (1 + abs(_val(s.fun)))
    if step == "armijo":
        assert all(_val(b.fun) <= _val(a.fun) for a, b in itertools.pairwise(res.trace))


@settings(max_examples=1000, deadline=None)
@given(st.sampled_from(sorted(INTERIOR_START)), st.floats(0.0, 1.0), st.floats(0.0, 1.0))
def test_log_barrier_strictly_feasible_and_convergent(pid, u, v):
    p = problems.get(pid)
    x0 = INTERIOR_START[pid](u, v)
    if not np.all(_cvals(p, np.asarray(x0)) < 0.0):
        return
    res = _run("log_barrier", pid, x0=x0)
    assert_valid_result(res)
    assert res.converged, res.message
    for s in res.trace:
        assert np.all(np.asarray(s.info["constraints"]) < 0.0)
        for sk, val in s.info.get("trials", []):
            if val != math.inf:  # a finite barrier value only at strictly feasible trials
                assert np.all(
                    _cvals(p, np.asarray(s.info["from"]) + sk * np.asarray(s.info["direction"])) < 0
                )
    for a, b in itertools.pairwise(res.trace):
        if a.info["outer"] == b.info["outer"]:
            assert b.info["merit"] <= a.info["merit"] + 100 * EPS * (1 + abs(a.info["merit"]))
    C, act = _kkt_matrix_bound(p, 0)
    x = np.asarray(res.x)
    assert (
        np.linalg.norm(x - p.minima[0])
        <= 2 * C * np.linalg.norm(_residual(p, x, act, None)) + 1e-14
    )


@settings(max_examples=1000, deadline=None)
@given(
    st.sampled_from(["sqp", "augmented_lagrangian", "quadratic_penalty"]),
    st.sampled_from(CONVEX),
    st.floats(0.0, 1.0),
    st.floats(0.0, 1.0),
)
def test_multiplier_methods_converge_from_any_start_on_convex_problems(method, pid, u, v):
    p = problems.get(pid)
    res = _run(method, pid, x0=_domain_point(p, u, v))
    assert_valid_result(res)
    assert res.converged, res.message
    tol = get_method(method).defaults()["tol"]
    x = np.asarray(res.x)
    assert _violation(p, x) <= tol
    C, act = _kkt_matrix_bound(p, 0)
    assert (
        np.linalg.norm(x - p.minima[0])
        <= 2 * C * np.linalg.norm(_residual(p, x, act, None)) + 1e-14
    )


# ======================================================================================
# Regression tests: rounding floors, parameter ranges, failure reporting
# ======================================================================================

#: Starts from the audit at which the old Armijo searches accepted null steps until max_iter.
MISHRA_PG_START = [-6.743790354647398, -5.4696050530085465]
MISHRA_FW_START = [-3.7490453339533305, -1.0278619903042454]


def _no_repeated_iterate(res: Result) -> bool:
    xs = [np.asarray(s.x) for s in res.trace]
    return not any(np.array_equal(a, b) for a, b in itertools.pairwise(xs))


#: Convex/smooth problems whose default x0 is strictly feasible for log_barrier.
BARRIER_PROBLEMS = (
    "quadratic_disk",
    "box_quadratic",
    "hs21",
    "halfplanes_quadratic",
    "rosenbrock_unit_disk",
    "linear_eq_quadratic",
)


@pytest.mark.parametrize(
    ("pid", "kw"),
    [
        ("halfplanes_quadratic", {"newton_tol": 1e-14}),
        ("hs21", {"t0": 1e3}),
        ("quadratic_disk", {"mu": 1.5}),
        ("box_quadratic", {"mu": 1.5}),
        *[(pid, {"newton_tol": 1e-14, "mu": 1.5, "t0": 1e3}) for pid in BARRIER_PROBLEMS],
    ],
)
def test_log_barrier_rounding_floor_of_a_stage_advances_t(pid, kw):
    # Newton centering cannot change the 1/t part of the KKT residual. A full step below
    # the resolution of F that leaves the residual at 1/t (stationarity 6e-14 on
    # halfplanes_quadratic, t = 1) once stopped the method as "stalled"; the centering is
    # complete there and the method must go on with t := μt.
    p = problems.get(pid)
    res = _run("log_barrier", pid, **kw)
    assert res.converged, res.message
    assert_valid_result(res, max_iter=500)
    if not _is_eq(p).all():  # with inequalities, only in the final stage (1/t ≤ tol)
        assert 1.0 / res.trace[-1].info["t"] <= 1e-6
    C, act = _kkt_matrix_bound(p, 0)
    x = np.asarray(res.x)
    assert np.linalg.norm(x - p.minima[0]) <= 2 * C * np.linalg.norm(_residual(p, x, act, None))


@settings(max_examples=1000, deadline=None)
@given(
    st.sampled_from(BARRIER_PROBLEMS),
    st.floats(-3.0, 3.0),  # log10 t0 over the UI range
    st.floats(math.log10(1.5), 2.0),  # log10 mu over the UI range
    st.floats(-14.0, -8.0),  # log10 newton_tol
)
def test_log_barrier_converges_over_the_parameter_ranges(pid, lt0, lmu, lnt):
    res = _run("log_barrier", pid, t0=10**lt0, mu=10**lmu, newton_tol=10**lnt)
    assert_valid_result(res, max_iter=500)
    assert res.converged, res.message


def test_augmented_lagrangian_needs_mu0_above_one():
    # N&W Alg. 17.4: ω ← ω/μ and η ← η/μ^0.9 tighten only for μ > 1. With μ₀ < 1 the inner
    # tolerance ω₀ = 1/μ₀ exceeded ‖∇L_A(x₀)‖, x never moved, and ω, η grew at each update.
    spec = {ps.name: ps for ps in get_method("augmented_lagrangian").params}["mu0"]
    assert spec.min is not None and spec.min > 1.0
    for pid in PROBLEMS:
        res = _run("augmented_lagrangian", pid, mu0=spec.min)
        assert res.converged, (pid, res.message)
        assert res.n_iter > 0
        assert all(s.info["omega"] <= 1.0 / spec.min for s in res.trace)  # ω ≤ ω₀
    for mu0 in (1e-2, 0.5, 1.0):
        with pytest.raises(ValueError, match="mu0 > 1"):
            _run("augmented_lagrangian", "quadratic_disk", mu0=mu0)


@pytest.mark.parametrize("tol", [1e-6, 1e-9, 1e-12])
def test_projected_gradient_reaches_tight_tol_without_null_steps(tol):
    # The old search shrank s until P(x − s∇f) = x and accepted that null step 977 times
    # (33394 f evaluations) to report "reached max_iter"; a fixed step reaches 8e-14 here.
    res = _run("projected_gradient", "mishra_bird_constrained", x0=MISHRA_PG_START, tol=tol)
    assert res.converged, res.message
    assert res.n_iter <= 50 and res.n_fev <= 400
    assert _no_repeated_iterate(res)
    assert_valid_result(res)


def test_frank_wolfe_armijo_rounding_regime():
    # The old search accepted γ = 5.6e-17 null steps 950 times from this start.
    p = problems.get("mishra_bird_constrained")
    res = _run("frank_wolfe", p, x0=MISHRA_FW_START, step="armijo")
    assert res.converged, res.message
    assert _no_repeated_iterate(res)
    n_regime = 0
    extra = 0
    for prev, s in itertools.pairwise(res.trace):
        accepted = _documented_armijo(p, prev, s, "frank_wolfe", 1e-4)
        assert accepted == [g == s.step_size for g, _ in s.info["trials"]]
        n_regime += sum(r for *_, r in _armijo_trials(prev, s, "frank_wolfe"))
        extra += _extra_gradients(prev, s, "frank_wolfe")
        assert _val(s.fun) <= _val(prev.fun) + _resolution(_val(prev.fun))
    assert n_regime > 0
    assert res.n_gev == 1 + res.n_iter + extra
    assert res.n_fev == 1 + sum(len(s.info["trials"]) for s in res.trace[1:])


@settings(max_examples=1000, deadline=None)
@given(
    st.sampled_from(["projected_gradient", "frank_wolfe"]),
    st.floats(-0.5, 1.5),
    st.floats(-0.5, 1.5),
)
def test_projection_methods_at_tol_1e9_converge_or_stop_at_the_floor(method, u, v):
    # tol = 1e-9 is in the UI range. At the floor of a monotone method the decrease per
    # step, ≈ r²/L for the residual r, is below ‖∇f‖·ε·‖x‖ ≈ 5e-16 on quadratic_disk
    # (the change of f caused by the rounding of the projected point), i.e. r ≲ 5e-8
    # with L = 2. The method must either converge or stop "stalled" at once at that floor;
    # it must never repeat an iterate.
    p = problems.get("quadratic_disk")
    kw = {"step": "armijo"} if method == "frank_wolfe" else {}
    res = _run(method, p, x0=_domain_point(p, u, v), tol=1e-9, **kw)
    assert_valid_result(res, max_iter=1000)
    assert _no_repeated_iterate(res)
    if not res.converged:
        assert res.message.startswith("stalled"), res.message
        assert res.extra["kkt_residual"] <= 1e-7
    if method == "projected_gradient":
        assert res.n_iter <= 100


def test_projected_gradient_trial_cap_scales_with_beta():
    # rosenbrock_disk needs s < 2/L ≈ 1.7e-3 at k ≈ 31, below the 60th trial 0.9⁵⁹ ≈ 2.0e-3
    # of β = 0.9: a fixed cap of 60 trials reported a line-search failure there.
    res = _run("projected_gradient", "rosenbrock_disk", beta=0.9)
    assert "max_iter" in res.message  # a known slow case, but no line-search failure
    assert min(_val(s.step_size) for s in res.trace[1:]) < 0.9**59
    assert max(len(s.info["trials"]) for s in res.trace[1:]) > 60
    # The cap is 1 + ⌈log(1e-18)/log β⌉ trials (down to s̄·1e-18): f is NaN except at x0.
    base = problems.get("box_quadratic")
    x0 = np.zeros(2)
    p = dataclasses.replace(
        base, f=lambda x: base.f(x) if np.array_equal(np.asarray(x), x0) else math.nan
    )
    for beta in (0.1, 0.5, 0.9):
        res = _run("projected_gradient", p, x0=x0, beta=beta)
        assert res.n_fev == 2 + math.ceil(math.log(1e-18) / math.log(beta))
        assert not res.converged and "non-finite" in res.message and res.n_iter == 0
    # With finite f and a wrong gradient, the message names the trial limit.
    q = dataclasses.replace(base, grad=lambda x: -1e3 * np.asarray(base.grad(x)))
    res = _run("projected_gradient", q, x0=x0, s_bar=100.0)
    assert not res.converged and "trial limit reached (61 trials" in res.message


@pytest.mark.parametrize("method", METHODS)
def test_nonfinite_derivative_at_an_accepted_point_returns_the_last_iterate(method):
    # ∇f (∇²f for log_barrier) is NaN for x₁ > 0.5 (> 0): the line search accepts such a
    # point because f is finite there. The Result must describe trace[-1], the last point
    # with finite values, not the rejected point (which only the message names).
    base = problems.get("box_quadratic")
    if method == "log_barrier":
        bad = 0.0
        p = dataclasses.replace(
            base, hess=lambda x: np.full((2, 2), np.nan) if x[0] > bad else _hess(base, x)
        )
    else:
        bad = 0.5
        p = dataclasses.replace(
            base, grad=lambda x: np.full(2, np.nan) if x[0] > bad else _grad(base, x)
        )
    res = _run(method, p)
    assert not res.converged and "non-finite" in res.message
    last = res.trace[-1]
    assert np.array_equal(res.x, last.x) and res.fun == last.fun
    assert float(np.asarray(res.x)[0]) <= bad
    assert res.extra["kkt_residual"] == last.info["kkt_residual"]
    assert res.extra["violation"] == last.info["violation"]
    if "multipliers" in res.extra:
        assert res.extra["multipliers"] == last.info["multipliers"]
    bad_x = [float(v) for v in re.findall(r"-?\d+\.\d+(?:e-?\d+)?", res.message.split("x =")[1])]
    assert bad_x[0] > bad  # the message names the rejected point
    assert_valid_result(res)
