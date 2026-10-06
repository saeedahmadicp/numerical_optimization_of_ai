"""Tests for numopt.lp.pdhg: restarted PDHG (PDLP-style).

Oracles: ``scipy.optimize.linprog`` (HiGHS) for primal optima and duals; a hand-computed
three-step PDHG trajectory; for the normalized duality gap, the Lagrangian dual bound
min_λ>0 (an independent formulation of the same trust-region problem) and a 50-digit mpmath
bisection on the radius equation; the Pock–Chambolle bound ‖Ã‖₂ ≤ 1; random LPs with a known
strictly complementary optimum; PDLP's scale invariance (Applegate et al. 2021, App. A); the
complexity statements of Applegate et al. (2023), Table 2 (the study's key property).
"""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any

import mpmath
import numpy as np
import pytest
from conftest import assert_valid_result
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from numpy.testing import assert_allclose
from scipy.optimize import linprog, minimize_scalar

import numopt
from numopt import problems
from numopt.core.registry import get_method
from numopt.core.types import LinearProgram
from numopt.lp import pdhg as pdhg_mod
from numopt.lp.pdhg import (
    kkt_error,
    normalized_duality_gap,
    primal_weight_update,
    restarted_pdhg,
    ruiz_pock_chambolle,
    standard_form,
)

LIBRARY = (
    "wyndor",
    "diet_2d",
    "degenerate_2d",
    "beale_cycling",
    "klee_minty_3",
    "transport_small",
    "ilp_knapsack_like_2d",
    "ilp_3var",
)
PLAIN = {"restart": "none", "primal_weight": "unit", "precondition": "none"}
UNSCALED = {"primal_weight": "unit", "precondition": "none"}
INFO_KEYS = {
    "x",
    "x_pdhg",
    "x_avg",
    "y",
    "kkt_last",
    "kkt_avg",
    "kkt",
    "primal_residual",
    "dual_residual",
    "gap",
    "omega",
    "tau",
    "sigma",
    "restarted",
    "epoch",
    "epoch_len",
    "normalized_gap",
    "restart_threshold",
    "matvecs",
}


def _linprog(lp: LinearProgram):
    sign = 1.0 if lp.sense == "min" else -1.0
    res = linprog(
        sign * np.asarray(lp.c),
        A_ub=lp.A_ub,
        b_ub=lp.b_ub,
        A_eq=lp.A_eq,
        b_eq=lp.b_eq,
        bounds=(0, None),
        method="highs",
    )
    assert res.status == 0
    return res, sign * res.fun


def random_lp(
    rng: np.random.Generator, m: int, n: int
) -> tuple[LinearProgram, np.ndarray, np.ndarray]:
    """Equality LP with a known strictly complementary optimum (z*, y*): b = Az*, c = Aᵀy* + s*."""
    A = rng.normal(size=(m, n))
    z = np.zeros(n)
    basis = rng.choice(n, size=m, replace=False)
    z[basis] = rng.uniform(0.5, 2.0, size=m)
    y = rng.normal(size=m)
    s = rng.uniform(0.5, 2.0, size=n)
    s[basis] = 0.0
    lp = LinearProgram(id="rand", name="rand", c=A.T @ y + s, A_eq=A, b_eq=A @ z)
    return lp, z, y


def _check_trace(res: Any, lp: LinearProgram) -> None:
    """Contract checks specific to this method (counts, info keys, z ≥ 0, x in original units)."""
    n = np.asarray(lp.c).size
    assert res.n_iter == res.trace[-1].k
    assert res.extra["n_matvec"] == 2 + 2 * res.n_iter
    for s in res.trace:
        assert set(s.info) == INFO_KEYS
        assert s.info["matvecs"] == 2 + 2 * s.k
        assert_allclose(s.info["x"], s.x, rtol=0, atol=0)
        assert np.asarray(s.x).shape == (n,)
        assert np.all(np.asarray(s.x) >= 0.0) and np.all(np.asarray(s.info["x_pdhg"]) >= 0.0)
        assert s.info["kkt"] == min(s.info["kkt_last"], s.info["kkt_avg"])
        assert s.info["kkt_last"] == max(
            s.info["primal_residual"], s.info["dual_residual"], s.info["gap"]
        )
        assert s.fun == pytest.approx(float(np.asarray(lp.c) @ s.x), rel=1e-12, abs=1e-12)
        if s.info["restarted"]:
            assert_allclose(s.x, s.info["x_avg"], rtol=0, atol=0)
    assert np.all(np.asarray(res.x) >= 0.0)


EPS = float(np.finfo(np.float64).eps)


@dataclasses.dataclass(frozen=True)
class Replayed:
    """Epoch quantities of one trace step, rebuilt from the trace alone (see ``_replay``)."""

    k: int
    x_avg: np.ndarray  # (N,) running average z̄ⁿ'ᵗ of the PDHG iterates of the epoch
    y_avg: np.ndarray  # (m,) running average ȳⁿ'ᵗ
    t: int  # epoch length at this step, before its restart
    x_start: np.ndarray  # (N,) epoch start zⁿ'⁰
    y_start: np.ndarray  # (m,) epoch start yⁿ'⁰
    x_prev: np.ndarray  # (N,) previous epoch start zⁿ⁻¹'⁰ (zⁿ'⁰ itself for n = 0)
    y_prev: np.ndarray  # (m,)
    prod_err: float  # t·eps·max‖(A zᵢ, Aᵀyᵢ)‖: a priori error of a running-mean product


def _replay(res: Any, A: np.ndarray, b: np.ndarray, c: np.ndarray) -> list[Replayed]:
    """Rebuild the PDHG iterates and the epoch averages of an *unscaled equality-only* run.

    With ``precondition="none"`` and no ``≤`` rows, z = x and the trace gives every state:
    ``info["x_pdhg"]`` is zⁿ'ᵗ, ``info["y"]`` is yⁿ'ᵗ except at a restart (it is then the
    average), where the PDHG dual is rebuilt from one step of the recurrence
    yⁿ'ᵗ = y + σ(b − A(2zⁿ'ᵗ − z)) from the previous trace point (z, y, σ). The primal
    recurrence zⁿ'ᵗ = max(z − τ(c − Aᵀy), 0) is asserted on the way, so the rebuilt states
    are anchored to the method's own step. The averages use fresh means of the rebuilt
    iterates; the method keeps running sums of z, y, A z and Aᵀy.
    """
    out: list[Replayed] = []
    x_start = np.asarray(res.trace[0].x)
    y_start = np.asarray(res.trace[0].info["y"])
    x_prev, y_prev = x_start, y_start
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    scale = restart_err = 0.0
    for p, s in zip(res.trace, res.trace[1:], strict=False):
        x_old, y_old = np.asarray(p.x), np.asarray(p.info["y"])
        x_new = np.asarray(s.info["x_pdhg"])
        assert s.step_size == p.info["tau"]
        # NOTE: after a restart the method steps from its running-mean product Aᵀȳ, not a
        # fresh one; they differ by ≤ restart_err (module docstring, "Cost"), and the step
        # multiplies that by τ. Otherwise only O(eps) roundings of the step itself remain.
        grad = c - A.T @ y_old
        tol_step = 8 * EPS * (np.abs(x_old).max() + p.info["tau"] * np.abs(grad).max())
        if p.info["restarted"]:
            tol_step += 4 * p.info["tau"] * restart_err
        x_rec = np.maximum(x_old - p.info["tau"] * grad, 0.0)
        assert_allclose(x_new, x_rec, rtol=1e-12, atol=tol_step)
        if s.info["restarted"]:
            y_new = y_old + p.info["sigma"] * (b - (2.0 * A @ x_new - A @ x_old))
        else:
            y_new = np.asarray(s.info["y"])
        xs.append(x_new)
        ys.append(y_new)
        scale = max(scale, float(np.linalg.norm(A @ x_new)), float(np.linalg.norm(A.T @ y_new)))
        t = len(xs)
        assert t == s.info["epoch_len"]
        x_avg, y_avg = np.mean(xs, axis=0), np.mean(ys, axis=0)
        out.append(
            Replayed(
                s.k, x_avg, y_avg, t, x_start, y_start, x_prev, y_prev, t * EPS * (1.0 + scale)
            )
        )
        if s.info["restarted"]:
            # The restart point zⁿ⁺¹'⁰ is the average (2023 paper, Algorithm 1 line 10).
            x_prev, y_prev = x_start, y_start
            x_start, y_start = np.asarray(s.x), np.asarray(s.info["y"])
            # NOTE: running sum vs np.mean (pairwise): ≤ γ_t Σ|vᵢ| (Higham, 2nd ed., §4.2).
            for got, ref, v in ((x_start, x_avg, xs), (y_start, y_avg, ys)):
                assert_allclose(got, ref, rtol=1e-13, atol=4 * t * EPS * np.abs(v).max())
            restart_err = out[-1].prod_err
            xs, ys, scale = [], [], 0.0
    return out


def _wnorm(dz: np.ndarray, dy: np.ndarray, omega: float) -> float:
    """‖(dz, dy)‖_ω = √(ω‖dz‖² + ‖dy‖²/ω) (Applegate et al. 2021, §2), written out here."""
    return math.sqrt(omega * float(np.sum(dz**2)) + float(np.sum(dy**2)) / omega)


# --------------------------------------------------------------------------------------
# Contract, registry and fixtures
# --------------------------------------------------------------------------------------


def test_registry_entry_and_defaults() -> None:
    spec = get_method("restarted_pdhg")
    assert spec.family == "lp" and spec.deterministic
    assert spec.defaults() == {
        "restart": "adaptive",
        "restart_period": 64,
        "beta": math.exp(-1.0),
        "restart_check_every": 40,
        "primal_weight": "balanced",
        "precondition": "ruiz_pc",
        "tol": 1e-8,
        "max_iter": 20_000,
    }
    for p in spec.params:
        if p.kind in ("float", "int"):
            assert p.min is not None and p.max is not None and p.min <= p.default <= p.max
        if p.kind == "choice":
            assert p.default in p.choices
    assert any("Algorithm 1" in r for r in spec.references)
    json.dumps(spec.to_dict(), allow_nan=False)


def test_fixture_cases_run_with_short_traces() -> None:
    assert 3 <= len(pdhg_mod.FIXTURE_CASES) <= 6
    for method, pid, params in pdhg_mod.FIXTURE_CASES:
        lp = problems.get(pid)
        res = numopt.run(method, lp, **params)
        assert_valid_result(res, max_iter=params.get("max_iter", 20_000))
        assert len(res.trace) < 400, (pid, params, len(res.trace))
        _check_trace(res, lp)
        if params.get("restart", "adaptive") != "none":
            assert res.converged, (pid, params, res.message)


def test_first_three_steps_match_hand_computation() -> None:
    # min x1 + 2 x2 s.t. x1 + x2 = 1, x ≥ 0; ‖A‖₂ = √2, ω = 1, so τ = σ = s = 0.9/√2.
    lp = LinearProgram(
        id="t", name="t", c=np.array([1.0, 2.0]), A_eq=np.array([[1.0, 1.0]]), b_eq=np.array([1.0])
    )
    res = restarted_pdhg(lp, max_iter=3, **PLAIN)
    s = 0.9 / math.sqrt(2.0)
    # k=1: z = max(−s c, 0) = 0, y = s·1.  k=2: z = max(−s(c − s), 0) = 0 (s < 1), y = 2s.
    # k=3: c − Aᵀy = (1 − 2s, 2 − 2s) with 1 − 2s < 0, so z = (s(2s − 1), 0) and
    #      y = 2s + s(1 − 2 s(2s − 1)).
    x_hand = [np.zeros(2), np.zeros(2), np.zeros(2), np.array([s * (2 * s - 1), 0.0])]
    y_hand = [0.0, s, 2 * s, 2 * s + s * (1 - 2 * s * (2 * s - 1))]
    for k in range(4):
        assert_allclose(res.trace[k].x, x_hand[k], rtol=1e-14, atol=1e-15)
        assert_allclose(res.trace[k].info["y"], [y_hand[k]], rtol=1e-14, atol=1e-15)
    # Running average of the three iterates, and the step sizes.
    assert_allclose(res.trace[3].info["x_avg"], sum(x_hand[1:]) / 3, rtol=1e-14, atol=1e-15)
    assert res.trace[3].info["tau"] == pytest.approx(s, rel=1e-15)
    assert res.trace[3].info["sigma"] == pytest.approx(s, rel=1e-15)
    assert res.trace[0].step_size is None and res.trace[1].step_size == pytest.approx(s)
    assert not res.converged and res.n_iter == 3 and len(res.trace) == 4
    assert res.extra["status"] == "max_iter" and "max_iter" in res.message


# --------------------------------------------------------------------------------------
# Oracle: linprog (HiGHS)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pid", LIBRARY)
@pytest.mark.parametrize(
    "variant",
    [
        {},
        {"restart": "adaptive", **UNSCALED},
        {"restart": "fixed", "restart_period": 64, "primal_weight": "unit"},
        {"primal_weight": "adaptive"},
    ],
    ids=["default", "adaptive-unscaled", "fixed64+pc", "adaptive-omega"],
)
def test_converges_to_linprog_optimum(pid: str, variant: dict[str, Any]) -> None:
    if pid == "klee_minty_3" and variant.get("precondition") == "none":
        pytest.skip("unpreconditioned PDHG is slow on klee_minty_3; see the preconditioning test")
    lp = problems.get(pid)
    res = restarted_pdhg(lp, max_iter=50_000, **variant)
    assert_valid_result(res, max_iter=50_000)
    assert res.converged, res.message
    assert res.extra["status"] == "optimal" and res.extra["kkt"] <= 1e-8
    _check_trace(res, lp)
    _, f_ref = _linprog(lp)
    # NOTE: a relative KKT error ≤ 1e-8 bounds the objective error by ~1e-8·(1 + |f|) only up
    # to the LP's Hoffman constant; 1e-6 relative leaves two digits of margin on these LPs
    # (the study's largest observed error over 1532 converged runs was 1.7e-7).
    assert abs(res.fun - f_ref) <= 1e-6 * (1 + abs(f_ref))
    if lp.optimum is not None and not lp.integer:  # unique optimum known
        assert_allclose(res.x, lp.optimum, rtol=1e-5, atol=1e-5)


def test_dual_matches_linprog_on_unique_dual() -> None:
    lp = problems.get("wyndor")  # nondegenerate optimum: unique duals (0, 1.5, 1)
    res = restarted_pdhg(lp)
    ref, _ = _linprog(lp)
    # linprog marginals of ≤ rows are ∂f/∂b ≤ 0 for min(−cᵀx); y is the dual of the same form.
    assert_allclose(res.extra["y"], ref.ineqlin.marginals, rtol=1e-6, atol=1e-6)


def test_equality_rows_and_dual_of_transport() -> None:
    lp = problems.get("transport_small")
    res = restarted_pdhg(lp)
    _, f_ref = _linprog(lp)
    assert res.converged
    std = standard_form(lp)
    z = np.concatenate([res.x, np.asarray(lp.b_ub) - np.asarray(lp.A_ub) @ res.x])
    # Primal feasibility of the returned point and weak duality of the returned dual.
    assert np.linalg.norm(std.A @ z - std.b) <= 1e-8 * (1 + np.linalg.norm(std.b)) * 10
    assert float(std.b @ res.extra["y"]) == pytest.approx(f_ref, rel=1e-6)


# --------------------------------------------------------------------------------------
# Failure paths and invalid input
# --------------------------------------------------------------------------------------


def test_max_iter_infeasible_and_unbounded_return_unconverged() -> None:
    res = restarted_pdhg(problems.get("wyndor"), max_iter=5)
    assert_valid_result(res, max_iter=5)
    assert not res.converged and res.extra["status"] == "max_iter" and res.n_iter == 5
    # The best point seen is returned: its KKT error is the smallest in the trace.
    assert res.extra["kkt"] == pytest.approx(min(s.info["kkt"] for s in res.trace), rel=1e-12)
    for pid in ("infeasible_2d", "unbounded_2d"):
        bad = restarted_pdhg(problems.get(pid), max_iter=2000)
        assert_valid_result(bad, max_iter=2000)
        assert not bad.converged and bad.extra["status"] == "max_iter"
        assert "infeasible or unbounded" in bad.message
        json.dumps(bad.to_dict(), allow_nan=False)


def test_optimal_start_stops_at_k0() -> None:
    # With c = 0 every feasible point is optimal with y = 0: a feasible x0 (its slack is 0)
    # has KKT error 0 at k = 0, so the method stops before the first PDHG step.
    lp = LinearProgram(
        id="t",
        name="t",
        c=np.zeros(2),
        A_ub=np.array([[-1.0, -1.0]]),
        b_ub=np.array([-1.0]),
    )
    res = restarted_pdhg(lp, x0=[0.5, 0.5])
    assert res.converged and res.n_iter == 0 and len(res.trace) == 1
    assert res.extra["n_matvec"] == 2


def test_nonfinite_iterate_stops_unconverged() -> None:
    # A huge c makes τ(c − Aᵀy) overflow in the first step: the method must stop cleanly.
    lp = LinearProgram(
        id="t",
        name="t",
        c=np.array([-1e308, -1e308]),
        A_ub=np.array([[1.0, 1.0]]),
        b_ub=np.array([1e-300]),
    )
    res = restarted_pdhg(lp, max_iter=50, **PLAIN)
    assert_valid_result(res, max_iter=50)
    # ‖c̃‖₂ = √2·1e308 is finite, so the dual term of the start is ≈ 1 (not (finite)/inf = 0).
    assert res.trace[0].info["dual_residual"] == pytest.approx(1.0, rel=1e-12)
    assert not res.converged
    assert res.extra["status"] == "nonfinite" and "non-finite" in res.message
    assert res.n_iter < 50 and res.n_iter == res.trace[-1].k


def test_invalid_input_raises() -> None:
    lp = problems.get("wyndor")
    for bad in (
        {"restart": "sometimes"},
        {"primal_weight": "heavy"},
        {"precondition": "ilu"},
        {"beta": 1.0},
        {"beta": 0.0},
        {"restart_period": 0},
        {"restart_check_every": 0},
        {"max_iter": 0},
        {"tol": 0.0},
        {"x0": [1.0]},
        {"x0": [1.0, math.nan]},
    ):
        with pytest.raises(ValueError):
            restarted_pdhg(lp, **bad)
    with pytest.raises(TypeError):
        restarted_pdhg(problems.get("rosenbrock"))  # type: ignore[arg-type]
    no_rows = LinearProgram(
        id="t", name="t", c=np.array([1.0]), A_eq=np.zeros((1, 1)), b_eq=np.zeros(1)
    )
    with pytest.raises(ValueError):
        restarted_pdhg(no_rows)
    with pytest.raises(ValueError):
        normalized_duality_gap(
            np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.ones(1), -1, 1
        )


def test_x0_sets_slacks_and_projects_negative_entries() -> None:
    lp = problems.get("wyndor")
    res = restarted_pdhg(lp, x0=[1.0, -3.0], max_iter=1, **PLAIN)
    assert_allclose(res.trace[0].x, [1.0, 0.0], rtol=0, atol=0)
    # z⁰ = (1, 0, slacks b − A_ub x) = (1, 0, 3, 12, 15); the KKT primal residual is 0.
    assert res.trace[0].info["primal_residual"] == 0.0
    assert res.trace[0].fun == 3.0


# --------------------------------------------------------------------------------------
# Restart schemes and primal weight
# --------------------------------------------------------------------------------------


def test_fixed_restarts_happen_exactly_every_period() -> None:
    res = restarted_pdhg(problems.get("diet_2d"), restart="fixed", restart_period=50, **UNSCALED)
    ks = [s.k for s in res.trace if s.info["restarted"]]
    assert ks and ks == list(range(50, 50 * len(ks) + 1, 50))
    assert res.extra["n_restarts"] == len(ks) == res.trace[-1].info["epoch"]
    for s in res.trace:
        if s.info["restarted"]:
            assert s.info["epoch_len"] == 50


def test_restart_point_is_the_epoch_average_of_the_pdhg_iterates() -> None:
    res = restarted_pdhg(problems.get("wyndor"), restart="fixed", restart_period=7, max_iter=30)
    start = 0
    for s in res.trace[1:]:
        if s.info["restarted"]:
            epoch = [np.asarray(t.info["x_pdhg"]) for t in res.trace[start + 1 : s.k + 1]]
            assert_allclose(s.x, np.mean(epoch, axis=0), rtol=1e-13, atol=1e-15)
            start = s.k


def test_adaptive_restart_fires_only_on_beta_decay() -> None:
    lp = problems.get("transport_small")
    res = restarted_pdhg(lp, restart="adaptive", restart_check_every=1, **UNSCALED)
    assert res.trace[1].info["restarted"]  # n = 0: τ⁰ = 1
    checked = 0
    for s in res.trace[2:]:
        rho, thr = s.info["normalized_gap"], s.info["restart_threshold"]
        if rho is None:
            assert not s.info["restarted"] or s.k == res.n_iter
            continue
        checked += 1
        assert s.info["restarted"] == (rho <= thr)
    assert checked > 10 and res.extra["n_restarts"] >= 3
    # With a check interval, every epoch length is a multiple of it.
    res40 = restarted_pdhg(lp, restart="adaptive", restart_check_every=40, **UNSCALED)
    lens = [s.info["epoch_len"] for s in res40.trace if s.info["restarted"]]
    assert lens and all(n % 40 == 0 for n in lens)
    assert lens[0] == 40  # n = 0 restarts at the first check


def _equality_lp() -> LinearProgram:
    # A 3×7 equality LP with a known optimum; balanced ω = ‖c‖/‖b‖ ≈ 3.2 (far from 1, so a
    # norm with ω and 1/ω swapped, or ω replaced by 1, changes the values checked below).
    # With restart_check_every=3 and balanced ω it stops on the running average.
    lp, _, _ = random_lp(np.random.default_rng(36), 3, 7)
    return lp


@pytest.mark.parametrize("primal_weight", ["balanced", "adaptive"])
def test_adaptive_restart_test_matches_eq30_recomputed_from_the_trace(primal_weight: str) -> None:
    # Eq. 30 of Applegate et al. (2023): restart when
    #   ρ_{‖z̄ⁿ'ᵗ − zⁿ'⁰‖_ω}(z̄ⁿ'ᵗ) ≤ β ρ_{‖zⁿ'⁰ − zⁿ⁻¹'⁰‖_ω}(zⁿ'⁰).
    # Both sides are recomputed from the trace points (fresh products, a written-out weighted
    # norm, the epoch starts of the last two restarts) and compared with the reported values.
    lp = _equality_lp()
    A, b, c = np.asarray(lp.A_eq), np.asarray(lp.b_eq), np.asarray(lp.c)
    beta, every = 0.3, 3
    res = restarted_pdhg(
        lp,
        restart="adaptive",
        beta=beta,
        restart_check_every=every,
        primal_weight=primal_weight,
        precondition="none",
    )
    assert res.converged, res.message
    checked = fired = far = 0
    for rep in _replay(res, A, b, c):
        s = res.trace[rep.k]
        rho, thr = s.info["normalized_gap"], s.info["restart_threshold"]
        is_check = s.info["epoch"] - s.info["restarted"] >= 1 and rep.t % every == 0
        if not is_check or s.k == res.n_iter:  # no test once the run has stopped
            assert rho is None and thr is None
            continue
        # The test runs in the norm of epoch n; a restart at this step then updates ω, so the
        # ω of the test is the one reported at the previous step.
        omega = res.trace[s.k - 1].info["omega"]
        r = _wnorm(rep.x_avg - rep.x_start, rep.y_avg - rep.y_start, omega)
        rho_ref = normalized_duality_gap(rep.x_avg, A @ rep.x_avg, A.T @ rep.y_avg, b, c, r, omega)
        r0 = _wnorm(rep.x_start - rep.x_prev, rep.y_start - rep.y_prev, omega)
        thr_ref = beta * normalized_duality_gap(
            rep.x_start, A @ rep.x_start, A.T @ rep.y_start, b, c, r0, omega
        )
        # NOTE: |Δρ| ≤ ‖Δg‖_{ω,*} (ρ_r is a max of gᵀd/r over ‖d‖_ω ≤ r), and the method's
        # running-mean products differ from fresh ones by ≤ prod_err; rtol 1e-12 covers
        # the O(eps) roundings of the sort and the square roots.
        atol = 4 * rep.prod_err * max(math.sqrt(omega), 1 / math.sqrt(omega))
        assert rho == pytest.approx(rho_ref, rel=1e-12, abs=atol), s.k
        assert thr == pytest.approx(thr_ref, rel=1e-12, abs=atol), s.k
        assert s.info["restarted"] == (rho <= thr)
        checked += 1
        fired += s.info["restarted"]
        far += abs(math.log(omega)) > 0.5
    assert checked >= 20 and 3 <= fired < checked and far >= 10


@pytest.mark.parametrize("primal_weight", ["balanced", "adaptive"])
def test_kkt_of_running_average_matches_fresh_products(primal_weight: str) -> None:
    # kkt_avg comes from running means of A z and Aᵀy; recompute it from the rebuilt average
    # (z̄, ȳ) with fresh products A z̄, Aᵀȳ. Also the returned point: Result.extra["kkt"] must
    # be the KKT error of (Result.x, Result.extra["y"]) — z = x for this LP.
    lp = _equality_lp()
    A, b, c = np.asarray(lp.A_eq), np.asarray(lp.b_eq), np.asarray(lp.c)
    std = standard_form(lp)
    res = restarted_pdhg(
        lp, restart_check_every=3, primal_weight=primal_weight, precondition="none"
    )
    assert res.converged, res.message
    reps = _replay(res, A, b, c)
    assert len(reps) == res.n_iter
    for rep in reps:
        s = res.trace[rep.k]
        fresh = kkt_error(std, rep.x_avg, rep.y_avg, A @ rep.x_avg, A.T @ rep.y_avg)
        # NOTE: each KKT term is a norm of a product divided by (1 + a norm ≥ 0); the running
        # means differ from fresh products by ≤ prod_err (Higham, 2nd ed., §4.2).
        assert s.info["kkt_avg"] == pytest.approx(fresh.error, rel=1e-12, abs=4 * rep.prod_err)
        last = kkt_error(
            std, s.info["x_pdhg"], s.info["y"], A @ s.info["x_pdhg"], A.T @ s.info["y"]
        )
        if not s.info["restarted"]:  # info["y"] is the PDHG dual only without a restart
            assert s.info["kkt_last"] == pytest.approx(last.error, rel=1e-12, abs=1e-15)
    x, y = np.asarray(res.x), np.asarray(res.extra["y"])
    out = kkt_error(std, x, y, A @ x, A.T @ y)
    atol = 4 * max(rep.prod_err for rep in reps)
    assert out.error <= 1e-8 + atol
    assert res.extra["kkt"] == pytest.approx(out.error, rel=1e-12, abs=atol)
    for key in ("primal_residual", "dual_residual", "gap"):
        assert res.extra[key] == pytest.approx(getattr(out, key.split("_")[0]), rel=1e-12, abs=atol)


def test_kkt_of_running_average_decides_a_returned_average() -> None:
    # The path above matters: on this LP the method stops on the average (output "average").
    lp = _equality_lp()
    res = restarted_pdhg(lp, restart_check_every=3, primal_weight="balanced", precondition="none")
    assert res.extra["output"] == "average"
    assert res.trace[-1].info["kkt_avg"] <= 1e-8 < res.trace[-1].info["kkt_last"]


def test_adaptive_primal_weight_follows_algorithm_3_at_restarts() -> None:
    # PDLP Algorithm 3 with θ = ½ at the restart into epoch n + 1:
    #   ω⁺ = √(ω Δy/Δz),  Δz = ‖zⁿ⁺¹'⁰ − zⁿ'⁰‖₂,  Δy = ‖yⁿ⁺¹'⁰ − yⁿ'⁰‖₂,
    # then τ = η/ω⁺, σ = ηω⁺ (Applegate et al. 2021, eq. 4), and the next step uses this τ.
    lp = _equality_lp()
    A = np.asarray(lp.A_eq)
    res = restarted_pdhg(lp, restart_check_every=3, primal_weight="adaptive", precondition="none")
    assert res.converged, res.message
    eta = 0.9 / np.linalg.svd(A, compute_uv=False)[0]
    # Balanced start ω = ‖c‖/‖b‖ (PDLP InitializePrimalWeight).
    omega = float(np.linalg.norm(np.asarray(lp.c)) / np.linalg.norm(np.asarray(lp.b_eq)))
    x_prev, y_prev = np.asarray(res.trace[0].x), np.asarray(res.trace[0].info["y"])
    updated = 0
    for s in res.trace:
        if s.info["restarted"]:
            x_new, y_new = np.asarray(s.x), np.asarray(s.info["y"])
            dz, dy = float(np.linalg.norm(x_new - x_prev)), float(np.linalg.norm(y_new - y_prev))
            if dz > 1e-10 and dy > 1e-10:  # PDLP's "zero" test; ω is kept otherwise
                # Mutation guard: the inverted rule √(ω Δz/Δy) differs by more than 1 %.
                updated += abs(math.log(dy / dz)) > 0.02
                omega = math.sqrt(omega * dy / dz)
            x_prev, y_prev = x_new, y_new
        assert s.info["omega"] == pytest.approx(omega, rel=1e-13), s.k
        assert s.info["tau"] == pytest.approx(eta / omega, rel=1e-13), s.k
        assert s.info["sigma"] == pytest.approx(eta * omega, rel=1e-13), s.k
    for p, s in zip(res.trace[1:], res.trace[2:], strict=False):
        assert s.step_size == p.info["tau"]
    assert updated >= 3
    assert res.extra["omega"] == pytest.approx(omega, rel=1e-13)


def test_restart_none_never_restarts_and_averages_the_whole_run() -> None:
    res = restarted_pdhg(problems.get("wyndor"), max_iter=60, **PLAIN)
    assert not any(s.info["restarted"] for s in res.trace) and res.extra["n_restarts"] == 0
    xs = np.array([s.info["x_pdhg"] for s in res.trace[1:]])
    assert_allclose(res.trace[-1].info["x_avg"], xs.mean(axis=0), rtol=1e-13, atol=1e-15)


def test_primal_weight_update_hand_values() -> None:
    # ω⁺ = exp(½ log(Δy/Δz) + ½ log ω) = √(ω Δy/Δz).
    assert primal_weight_update(1.0, 4.0, 1.0) == pytest.approx(2.0, rel=1e-15)
    assert primal_weight_update(2.0, 2.0, 16.0) == pytest.approx(4.0, rel=1e-15)
    assert primal_weight_update(0.0, 1.0, 3.0) == 3.0  # Δz ≤ zero: unchanged
    assert primal_weight_update(1.0, math.inf, 3.0) == 3.0  # overflowed Δ: unchanged


def test_balanced_primal_weight_and_step_sizes() -> None:
    lp = problems.get("wyndor")
    res = restarted_pdhg(lp, precondition="none", max_iter=1)
    std = standard_form(lp)
    omega = np.linalg.norm(std.c) / np.linalg.norm(std.b)
    eta = 0.9 / np.linalg.norm(std.A, 2)
    info = res.trace[0].info
    assert info["omega"] == pytest.approx(omega, rel=1e-14)
    assert info["tau"] == pytest.approx(eta / omega, rel=1e-14)
    assert info["sigma"] == pytest.approx(eta * omega, rel=1e-14)


def test_adaptive_primal_weight_changes_only_at_restarts() -> None:
    res = restarted_pdhg(problems.get("diet_2d"), primal_weight="adaptive")
    assert res.converged
    for prev, s in zip(res.trace, res.trace[1:], strict=False):
        if not s.info["restarted"]:
            assert s.info["omega"] == prev.info["omega"]
    assert len({s.info["omega"] for s in res.trace}) > 1


# --------------------------------------------------------------------------------------
# The study's key property (Applegate et al. 2023, Table 2) and preconditioning
# --------------------------------------------------------------------------------------


def test_restarts_beat_plain_pdhg_and_plain_average_is_sublinear() -> None:
    # Plain PDHG: last iterate linear (κ²), average Θ(κ/ε) sublinear; restarted: linear (κ).
    lp = problems.get("diet_2d")
    plain = restarted_pdhg(lp, tol=1e-300, max_iter=20_000, **PLAIN)
    last = np.array([s.info["kkt_last"] for s in plain.trace])
    avg = np.array([s.info["kkt_avg"] for s in plain.trace])
    assert last[-100:].min() < 1e-10 < avg.min()
    # O(1/k) for the average: log-log slope over the second decade is −1 within 0.1.
    k1, k2 = 2000, 20_000
    slope = math.log(avg[k2] / avg[k1]) / math.log(k2 / k1)
    assert slope == pytest.approx(-1.0, abs=0.1)
    first_plain = int(np.argmax(last <= 1e-8))
    for variant in (UNSCALED, {}):  # without and with preconditioning (the default)
        adaptive = restarted_pdhg(lp, restart="adaptive", **variant)
        assert adaptive.converged and adaptive.n_iter < first_plain
        assert adaptive.extra["n_restarts"] >= 3


def test_preconditioning_rescues_klee_minty() -> None:
    # The study's finding: b spans 1 … 1e4 and A spans 1 … 200; without diagonal scaling PDHG
    # is far from 1e-8 after 5000 iterations, with Ruiz + Pock–Chambolle it needs < 300.
    lp = problems.get("klee_minty_3")
    slow = restarted_pdhg(lp, max_iter=5000, **UNSCALED)
    fast = restarted_pdhg(lp)
    assert not slow.converged and slow.extra["kkt"] > 1e-4
    assert fast.converged and fast.n_iter < 300


# --------------------------------------------------------------------------------------
# Normalized duality gap
# --------------------------------------------------------------------------------------


def _gap_dual_oracle(z, gz, gy, r, omega) -> float:
    """min over λ > 0 of the Lagrangian bound λr²/2 + Σᵢ max_{dᵢ ≥ lᵢ}(gᵢdᵢ − λWᵢdᵢ²/2)."""

    def bound(log_lam: float) -> float:
        lam = math.exp(log_lam)
        dz = np.maximum(gz / (lam * omega), -z)
        dy = gy * omega / lam
        val = gz @ dz - lam * omega * (dz @ dz) / 2 + gy @ dy - lam * (dy @ dy) / (2 * omega)
        return float(lam * r * r / 2 + val)

    best = minimize_scalar(bound, bounds=(-40.0, 40.0), method="bounded", options={"xatol": 1e-12})
    return float(best.fun)


def _gap_instance(m, n, seed, zero_frac):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m, n))
    z = rng.uniform(0.0, 2.0, size=n) * (rng.uniform(size=n) > zero_frac)
    y = rng.normal(size=m)
    b, c = rng.normal(size=m), rng.normal(size=n)
    return A, z, y, b, c


@settings(max_examples=1000, deadline=None)
@given(
    m=st.integers(1, 4),
    n=st.integers(1, 6),
    seed=st.integers(0, 2**32 - 1),
    log_r=st.floats(-3.0, 2.0),
    log_omega=st.floats(-2.0, 2.0),
    zero_frac=st.sampled_from([0.0, 0.5, 1.0]),
)
def test_normalized_gap_matches_lagrangian_dual(m, n, seed, log_r, log_omega, zero_frac) -> None:
    r, omega = 10.0**log_r, 10.0**log_omega
    A, z, y, b, c = _gap_instance(m, n, seed, zero_frac)
    rho = normalized_duality_gap(z, A @ z, A.T @ y, b, c, r, omega)
    upper = _gap_dual_oracle(z, A.T @ y - c, b - A @ z, r, omega) / r
    assert rho >= -1e-15
    # Weak duality: the bound is ≥ the maximum; strong duality (Slater, r > 0) makes it tight.
    # NOTE: rel 1e-7 is the accuracy of the bounded 1-D search over log λ, not of rho.
    assert rho <= upper * (1 + 1e-9) + 1e-12
    assert rho == pytest.approx(upper, rel=1e-7, abs=1e-10)


def _gap_mpmath(z, gz, gy, r, omega) -> float:
    """50-digit bisection on t for ‖d(t)‖_ω = r, d(t) = (max(−z, t g_z/ω), t ω g_y)."""
    with mpmath.workdps(50):
        Z = [mpmath.mpf(float(v)) for v in z]
        Gz = [mpmath.mpf(float(v)) for v in gz]
        Gy = [mpmath.mpf(float(v)) for v in gy]
        w, R = mpmath.mpf(omega), mpmath.mpf(r)

        def d(t):
            return [max(-zi, t * gi / w) for zi, gi in zip(Z, Gz, strict=True)], [
                t * w * g for g in Gy
            ]

        def norm2(t):
            dz, dy = d(t)
            return w * sum(v * v for v in dz) + sum(v * v for v in dy) / w

        lo, hi = mpmath.mpf(0), mpmath.mpf(1)
        while norm2(hi) < R * R and hi < mpmath.mpf(10) ** 60:
            hi *= 2
        for _ in range(220):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if norm2(mid) < R * R else (lo, mid)
        dz, dy = d(hi)
        val = sum(g * v for g, v in zip(Gz, dz, strict=True)) + sum(
            g * v for g, v in zip(Gy, dy, strict=True)
        )
        return float(val / R)


@settings(max_examples=300, deadline=None)
@given(
    seed=st.integers(0, 2**32 - 1),
    n=st.integers(2, 8),
    log_small=st.floats(-9.0, -3.0),
    log_r=st.floats(-3.0, 1.0),
    log_omega=st.floats(-1.0, 1.0),
)
def test_normalized_gap_is_accurate_when_most_ascent_is_blocked(
    seed, n, log_small, log_r, log_omega
) -> None:
    # Near a solution the free part of g is tiny while blocked components (gᵢ < 0, zᵢ small)
    # are large: a "total − cumsum" evaluation of the t² coefficient cancels catastrophically
    # here (the study's version lost up to 4e-2 relative). The reverse sum must keep ~1e-13.
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.0, 1e-3, size=n)
    gz = -rng.uniform(1.0, 100.0, size=n)  # blocked directions
    free = rng.integers(0, n)
    gz[free] = 10.0**log_small * rng.choice([-1.0, 1.0])
    z[free] = 1.0
    gy = 10.0**log_small * rng.normal(size=2)
    r, omega = 10.0**log_r, 10.0**log_omega
    c = -gz  # ATy = 0 so that g_z = −c
    b = gy  # Az = 0 so that g_y = b
    rho = normalized_duality_gap(z, np.zeros(2), np.zeros(n), b, c, r, omega)
    ref = _gap_mpmath(z, gz, gy, r, omega)
    # NOTE: rtol 1e-12: a handful of O(eps) roundings in sums of ≤ 10 nonnegative terms.
    assert rho == pytest.approx(ref, rel=1e-12, abs=1e-300)


@settings(max_examples=1000, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), r1=st.floats(1e-3, 10.0), r2=st.floats(1e-3, 10.0))
def test_normalized_gap_nonincreasing_in_r_and_zero_at_optimum(seed, r1, r2) -> None:
    rng = np.random.default_rng(seed)
    lp, z_star, y_star = random_lp(rng, 3, 7)
    A, b, c = np.asarray(lp.A_eq), np.asarray(lp.b_eq), np.asarray(lp.c)
    for omega in (0.3, 1.0, 3.0):
        g0 = normalized_duality_gap(z_star, A @ z_star, A.T @ y_star, b, c, r1, omega)
        assert abs(g0) <= 1e-12 * (1 + np.abs(c).max())
    k = kkt_error(standard_form(lp), z_star, y_star, A @ z_star, A.T @ y_star)
    assert k.error <= 1e-13
    z = rng.uniform(0, 2, size=7)
    y = rng.normal(size=3)
    lo, hi = sorted((r1, r2))
    g_lo = normalized_duality_gap(z, A @ z, A.T @ y, b, c, lo, 1.0)
    g_hi = normalized_duality_gap(z, A @ z, A.T @ y, b, c, hi, 1.0)
    # v(r) = r ρ_r is concave with v(0) = 0, so ρ_r = v(r)/r is nonincreasing.
    assert g_hi <= g_lo * (1 + 1e-12) + 1e-14
    g_0 = normalized_duality_gap(z, A @ z, A.T @ y, b, c, 0.0, 1.0)
    assert g_lo <= g_0 * (1 + 1e-12) + 1e-14


# --------------------------------------------------------------------------------------
# Preconditioner, invariances, random LPs
# --------------------------------------------------------------------------------------


@settings(max_examples=1000, deadline=None)
@given(
    A=hnp.arrays(
        np.float64,
        hnp.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=6),
        # NOTE: magnitudes in [1e-6, 1e6] (dynamic range 1e12). Ruiz scales a lone tiny entry
        # up by 1/√|a| per pass, so ranges near 1e300 overflow the scaling vectors.
        elements=st.one_of(
            st.just(0.0),
            st.builds(lambda s, e: s * 10.0**e, st.sampled_from([-1.0, 1.0]), st.floats(-6.0, 6.0)),
        ),
    )
)
def test_ruiz_pock_chambolle_gives_unit_norm_bound(A) -> None:
    d1, d2 = ruiz_pock_chambolle(A)
    assert np.all(d1 > 0) and np.all(d2 > 0)
    K = A * d1[:, None] * d2[None, :]
    # Pock & Chambolle (2011, Lemma 2), α = 1: ‖Σ^½ K T^½‖₂ ≤ 1 with row/col ℓ1 scalings.
    assert np.linalg.norm(K, 2) <= 1.0 + 1e-12


@pytest.mark.parametrize("precondition", ["none", "ruiz_pc"])
def test_balanced_primal_weight_is_scale_invariant(precondition: str) -> None:
    # PDLP App. A: with ω = ‖c‖/‖b‖ and η = 0.9/‖A‖, scaling c by κ leaves x unchanged and
    # scales y by κ (iterates identical up to rounding).
    lp = problems.get("transport_small")
    lp1000 = dataclasses.replace(lp, c=1000.0 * np.asarray(lp.c))
    kw: dict[str, Any] = {
        "restart": "fixed",
        "restart_period": 16,
        "precondition": precondition,
        "max_iter": 60,
    }
    a, b = restarted_pdhg(lp, **kw), restarted_pdhg(lp1000, **kw)
    for sa, sb in zip(a.trace, b.trace, strict=True):
        assert_allclose(sb.x, sa.x, rtol=1e-10, atol=1e-10)
        assert_allclose(sb.info["y"], 1000.0 * np.asarray(sa.info["y"]), rtol=1e-10, atol=1e-8)


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), m=st.integers(2, 6), extra=st.integers(2, 8))
def test_random_sharp_lps_converge_only_to_the_known_optimum(seed, m, extra) -> None:
    # Honest flag: converged=True only at the known optimum. Restarted PDHG is linear with a
    # rate set by the LP's sharpness, so a fixed budget does not solve every instance: in a
    # survey of 300 such LPs, 3 (1%) were still at KKT error 1e-5 … 2e-5 after 50 000
    # iterations (see test_slow_sharp_lp_needs_a_larger_budget).
    rng = np.random.default_rng(seed)
    lp, z_star, _ = random_lp(rng, m, m + extra)
    res = restarted_pdhg(lp, max_iter=50_000)
    if not res.converged:
        assert res.extra["status"] == "max_iter" and res.extra["kkt"] > 1e-8
        assert res.extra["kkt"] <= 1e-3, res.message  # linear progress, only slow
        return
    f_star = float(np.asarray(lp.c) @ z_star)
    assert res.fun is not None
    assert abs(res.fun - f_star) <= 1e-6 * (1 + abs(f_star))
    # NOTE: the optimum is unique and strictly complementary; ‖x − x*‖ ≤ H·KKT with the
    # Hoffman constant H ≲ cond(B) ~ 1e3 here, so 1e-4 is the right scale for KKT 1e-8.
    assert_allclose(res.x, z_star, rtol=0, atol=1e-4)


def test_slow_sharp_lp_needs_a_larger_budget() -> None:
    # Hypothesis counterexample of the study's claim "every random sharp LP converges in
    # 50 000 iterations": seed 22562, 6×10, optimal basis with cond(B) ≈ 1.6e3.
    lp, z_star, _ = random_lp(np.random.default_rng(22562), 6, 10)
    short = restarted_pdhg(lp, max_iter=50_000)
    assert not short.converged and short.extra["kkt"] < 1e-6
    full = restarted_pdhg(lp, max_iter=100_000)
    assert full.converged and 50_000 < full.n_iter < 100_000
    f_star = float(np.asarray(lp.c) @ z_star)
    assert full.fun is not None and abs(full.fun - f_star) <= 1e-6 * (1 + abs(f_star))
