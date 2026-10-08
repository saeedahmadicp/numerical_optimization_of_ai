"""Simplex methods: oracle comparison with scipy.optimize.linprog(method="highs"),
tableau invariants, pivot-rule behaviour (Klee–Minty, Beale cycling), failure paths."""

from __future__ import annotations

import re
from dataclasses import replace
from itertools import pairwise
from typing import Literal

import numpy as np
import pytest
from conftest import assert_valid_result
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from scipy.optimize import linprog

import numopt
from numopt import problems
from numopt.core.types import LinearProgram
from numopt.lp import simplex as simplex_module
from numopt.lp.simplex import lu_factor, lu_solve

PRIMAL = ("two_phase_simplex", "big_m", "revised_simplex")
RULES = ("dantzig", "bland", "steepest_edge")
FEASIBLE = (
    "wyndor",
    "diet_2d",
    "degenerate_2d",
    "klee_minty_3",
    "transport_small",
    "beale_cycling",
)
SETTINGS = settings(max_examples=1000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
ORIGIN_FEASIBLE = ("wyndor", "degenerate_2d", "klee_minty_3", "unbounded_2d")


def _drop_trivial_rows(a, b, *, eq: bool):
    """Remove all-zero rows that hold trivially (0 = 0, or 0 ≤ b with b ≥ 0): an equivalent LP.

    HiGHS without presolve returns status 4 ("numerical difficulties") on such a row.
    """
    if a is None or b is None:
        return None, None
    zero = ~np.any(a != 0, axis=1)
    trivial = zero & ((b == 0) if eq else (b >= 0))
    if trivial.all():
        return None, None
    return a[~trivial], b[~trivial]


def oracle(lp: LinearProgram):
    sign = 1.0 if lp.sense == "min" else -1.0
    a_ub, b_ub = _drop_trivial_rows(lp.A_ub, lp.b_ub, eq=False)
    a_eq, b_eq = _drop_trivial_rows(lp.A_eq, lp.b_eq, eq=True)
    r = linprog(
        sign * np.asarray(lp.c),
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        method="highs",
        # NOTE: HiGHS presolve can label an unbounded LP "infeasible" (e.g. max x1 + x3 + x4
        # s.t. x1 − x3 + x4 ≤ 1, x1 + x3 − x4 ≤ 1, whose origin is feasible). Without presolve
        # the simplex solver reports the correct status.
        options={"presolve": False},
    )
    if r.status == 4:
        # HiGHS without presolve fails ("model_status is Unknown") on e.g. min −x1 − x2 − x3
        # s.t. −2x1 ≤ 1, −2x2 ≤ 1 (unbounded along x3); presolve settles such LPs.
        r = linprog(
            sign * np.asarray(lp.c), A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, method="highs"
        )
    return r.status, (None if r.status != 0 else sign * r.fun), r.x


def check_against_oracle(res, lp: LinearProgram, *, rtol: float = 1e-9):
    status, value, _ = oracle(lp)
    if status == 0:
        assert res.converged, res.message
        assert value is not None and res.fun is not None
        # Exact pivoting on small integer data: only rounding error (κ of the bases ≤ 1e4).
        np.testing.assert_allclose(res.fun, value, rtol=rtol, atol=rtol)
        x = np.asarray(res.x)
        assert np.all(x >= -1e-9)
        if lp.A_ub is not None and lp.b_ub is not None:
            assert np.all(lp.A_ub @ x <= lp.b_ub + 1e-8 * (1 + np.abs(lp.b_ub)))
        if lp.A_eq is not None and lp.b_eq is not None:
            np.testing.assert_allclose(lp.A_eq @ x, lp.b_eq, rtol=1e-9, atol=1e-9)
    elif status == 2:
        assert not res.converged and res.extra["status"] == "infeasible"
        assert "infeasible" in res.message
    elif status == 3:
        assert not res.converged and res.extra["status"] == "unbounded"
        assert "unbounded" in res.message
    else:  # pragma: no cover - oracle failure
        pytest.fail(f"linprog status {status}")


# --------------------------------------------------------------------------------------
# Library problems
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pid", ORIGIN_FEASIBLE)
@pytest.mark.parametrize("rule", ("bland", "steepest_edge", "dantzig"))
def test_simplex_matches_linprog(pid, rule):
    lp = problems.get(pid)
    res = numopt.run("simplex", lp, pivot_rule=rule)
    assert_valid_result(res, max_iter=200)
    check_against_oracle(res, lp)


@pytest.mark.parametrize("method", PRIMAL)
@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("pid", [*FEASIBLE, "unbounded_2d", "infeasible_2d"])
def test_primal_methods_match_linprog(method, rule, pid):
    lp = problems.get(pid)
    res = numopt.run(method, lp, pivot_rule=rule)
    assert_valid_result(res, max_iter=200)
    if pid == "beale_cycling" and rule == "dantzig":
        assert not res.converged and res.extra["status"] == "cycling"
        return
    check_against_oracle(res, lp)


@pytest.mark.parametrize("pid", ("diet_2d", "transport_small"))
@pytest.mark.parametrize("rule", ("dantzig", "bland"))
def test_dual_simplex_matches_linprog(pid, rule):
    lp = problems.get(pid)
    res = numopt.run("dual_simplex", lp, pivot_rule=rule)
    assert_valid_result(res, max_iter=200)
    check_against_oracle(res, lp)
    # Dual feasibility is kept on every tableau: reduced costs ≥ 0.
    for s in res.trace:
        assert np.min(np.asarray(s.info["tableau"])[0, :-1]) >= -1e-9


def test_wyndor_textbook_path():
    # Hillier & Lieberman §4.3: (0,0) → (0,6) → (2,6), Z = 0 → 30 → 36.
    res = numopt.run("simplex", problems.get("wyndor"))
    assert [list(s.x) for s in res.trace] == [[0, 0], [0, 6], [2, 6]]
    assert [s.fun for s in res.trace] == [0.0, 30.0, 36.0]
    first = res.trace[0].info
    assert first["col_labels"] == ["x1", "x2", "s1", "s2", "s3", "rhs"]
    assert first["row_labels"] == ["z", "s1", "s2", "s3"]
    assert first["entering"] == 1 and first["leaving"] == 3  # x2 enters, s2 leaves
    assert first["ratio_test"] == [None, 6.0, 9.0]
    assert first["pivot_row"] == 2
    assert res.trace[-1].info["entering"] is None
    assert res.trace[1].step_size == 6.0


def test_klee_minty_dantzig_visits_all_vertices():
    res = numopt.run("simplex", problems.get("klee_minty_3"), pivot_rule="dantzig")
    assert res.converged and res.n_iter == 7
    vertices = {tuple(s.x) for s in res.trace}
    assert len(vertices) == 8  # all 2³ vertices of the deformed cube
    values = [float(s.fun) for s in res.trace if s.fun is not None]
    assert len(values) == len(res.trace)
    assert all(b > a for a, b in pairwise(values))
    steep = numopt.run("simplex", problems.get("klee_minty_3"), pivot_rule="steepest_edge")
    assert steep.converged and steep.n_iter == 1


def test_beale_cycles_with_dantzig_and_bland_terminates():
    lp = problems.get("beale_cycling")
    res = numopt.run("simplex", lp, pivot_rule="dantzig", max_iter=50)
    assert_valid_result(res, max_iter=50)
    assert not res.converged and res.extra["status"] == "cycling"
    assert "cycling" in res.message
    assert res.n_iter == 6  # Chvátal p. 31: the 7th tableau repeats the first
    assert all(s.step_size in (None, 0.0) for s in res.trace)  # every pivot degenerate
    bland = numopt.run("simplex", lp, pivot_rule="bland")
    assert bland.converged
    np.testing.assert_allclose(bland.x, lp.optimum, atol=1e-12)


def test_degenerate_pivot_keeps_vertex():
    res = numopt.run("simplex", problems.get("degenerate_2d"))
    assert res.converged
    steps = res.trace
    degenerate = [i for i in range(1, len(steps)) if steps[i].step_size == 0.0]
    assert degenerate, "expected a degenerate pivot"
    i = degenerate[0]
    assert list(steps[i].x) == list(steps[i - 1].x) == [2.0, 0.0]
    assert steps[i].info["basis"] != steps[i - 1].info["basis"]


def test_unbounded_ray_is_a_recession_direction():
    lp = problems.get("unbounded_2d")
    for method in ("simplex", *PRIMAL):
        res = numopt.run(method, lp)
        ray = np.asarray(res.extra["ray"])
        assert np.all(ray >= 0) and np.all(lp.A_ub @ ray <= 1e-12)
        assert lp.c @ ray > 0  # max problem: objective grows along the ray
        np.testing.assert_allclose(res.trace[-1].info["ray"], ray)


def test_two_phase_phase_labels_and_infeasibility_measure():
    res = numopt.run("two_phase_simplex", problems.get("diet_2d"))
    phases = [s.info["phase"] for s in res.trace]
    assert phases[0] == 1 and phases[-1] == 2 and phases == sorted(phases)
    # Phase-1 objective (sum of artificials) never increases.
    w = [s.info["infeasibility"] for s in res.trace if s.info["phase"] == 1]
    assert all(b <= a + 1e-12 for a, b in pairwise(w))
    assert all("a" not in lab for lab in res.trace[-1].info["col_labels"])


def test_big_m_has_two_objective_rows():
    lp = problems.get("diet_2d")
    res = numopt.run("big_m", lp)
    info = res.trace[0].info
    assert info["row_labels"][:2] == ["zM", "z"]
    tab = np.asarray(info["tableau"])
    # Artificial of row i has M-cost 1/ρᵢ (ρ = 1, 3, 2): the M-row is −Σᵢ Aᵢ/ρᵢ on x and
    # −Σᵢ bᵢ/ρᵢ on the rhs (rows 2x1 + x2 ≥ 5 etc. flipped to b ≥ 0).
    np.testing.assert_allclose(tab[0, :2], [-(1 + 1 / 3 + 1), -(1 + 1 + 1 / 2)], rtol=1e-15)
    np.testing.assert_allclose(tab[0, -1], -(4 + 6 / 3 + 5 / 2), rtol=1e-15)
    # On the equilibrated rows (every ρᵢ = 1) it is Hillier–Lieberman's −(sum of the rows).
    assert lp.A_ub is not None and lp.b_ub is not None
    rho = np.max(np.abs(lp.A_ub), axis=1)
    unit = replace(lp, A_ub=lp.A_ub / rho[:, None], b_ub=lp.b_ub / rho)
    tab1 = np.asarray(numopt.run("big_m", unit).trace[0].info["tableau"])
    assert unit.A_ub is not None and unit.b_ub is not None
    np.testing.assert_allclose(tab1[0, :2], unit.A_ub.sum(axis=0), rtol=1e-15)
    np.testing.assert_allclose(tab1[0, -1], unit.b_ub.sum(), rtol=1e-15)


def test_big_m_row_is_exactly_zero_once_the_artificials_left():
    """M_j = cost_M[j] − cost_M[B]ᵀB⁻¹A_j = 0 exactly on real columns when cost_M[B] = 0."""
    for pid in ("diet_2d", "transport_small"):
        res = numopt.run("big_m", problems.get(pid))
        assert res.converged
        info = res.trace[-1].info
        T = np.asarray(info["tableau"])
        real = [j for j, lab in enumerate(info["col_labels"][:-1]) if not lab.startswith("a")]
        assert not any(info["col_labels"][j].startswith("a") for j in info["basis"])
        assert np.all(T[0, real] == 0.0) and T[0, -1] == 0.0


def test_redundant_equalities_are_removed():
    # Balanced transportation problem: supply rows sum to the demand rows → rank 4 of 5.
    cost = np.array([8.0, 6.0, 10.0, 9.0, 12.0, 13.0])
    a_eq = np.array(
        [
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
        ],
        dtype=float,
    )
    lp = LinearProgram("t", "t", cost, A_eq=a_eq, b_eq=np.array([20.0, 30.0, 10.0, 25.0, 15.0]))
    for method in ("two_phase_simplex", "revised_simplex"):
        res = numopt.run(method, lp)
        assert_valid_result(res)
        check_against_oracle(res, lp)
        removed = [s.info.get("removed_rows") for s in res.trace if "removed_rows" in s.info]
        assert removed and removed[0] is not None and len(removed[0]) == 1
    check_against_oracle(numopt.run("big_m", lp), lp)


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("pid", [*FEASIBLE, "unbounded_2d", "infeasible_2d"])
def test_revised_and_tableau_visit_the_same_bases(rule, pid):
    lp = problems.get(pid)
    tab = numopt.run("two_phase_simplex", lp, pivot_rule=rule)
    rev = numopt.run("revised_simplex", lp, pivot_rule=rule)
    assert [s.info["basis"] for s in tab.trace] == [s.info["basis"] for s in rev.trace]
    assert tab.n_iter == rev.n_iter and tab.extra["status"] == rev.extra["status"]
    for a, b in zip(tab.trace, rev.trace, strict=True):
        np.testing.assert_allclose(a.info["tableau"], b.info["tableau"], atol=1e-9)


@pytest.mark.parametrize("method", ("simplex", *PRIMAL))
def test_tableau_invariants(method):
    """Every Step: B⁻¹A has identity basic columns, row 0 = c̄ recomputed, vertex feasible."""
    pid = "klee_minty_3" if method == "simplex" else "transport_small"
    lp = problems.get(pid)
    res = numopt.run(method, lp)
    sign = 1.0 if lp.sense == "min" else -1.0
    for s in res.trace:
        info = s.info
        T = np.asarray(info["tableau"], dtype=float)
        n_obj = 2 if method == "big_m" else 1
        rows = T[n_obj:]
        basis = info["basis"]
        np.testing.assert_allclose(rows[:, basis], np.eye(len(basis)), atol=1e-12)
        if info["phase"] == 2 and method != "big_m":
            labels = info["col_labels"][:-1]
            m_ub = 0 if lp.A_ub is None else lp.A_ub.shape[0]
            c_full = np.concatenate([sign * lp.c, np.zeros(m_ub)])[: len(labels)]
            d = c_full - c_full[basis] @ rows[:, :-1]
            np.testing.assert_allclose(T[0, :-1], d, atol=1e-9)
            np.testing.assert_allclose(T[0, -1], -(c_full[basis] @ rows[:, -1]), atol=1e-9)
            x = np.asarray(s.x)
            assert np.all(rows[:, -1] >= -1e-9)  # primal feasible basis
            if lp.A_ub is not None:
                assert np.all(lp.A_ub @ x <= lp.b_ub + 1e-9)


# --------------------------------------------------------------------------------------
# Scale-invariant pivot tolerance (audit: a legitimate entry ≤ tol gave a false "unbounded")
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ("simplex", *PRIMAL))
@pytest.mark.parametrize("tol", (1e-9, 1e-4))
def test_tiny_legitimate_pivot_is_not_ignored(method, tol):
    """max x1 + x2 s.t. 1e-9·x1 ≤ 1, x2 ≤ 1 is bounded with optimum (1e9, 1).

    The only entry that bounds x1 is 1e-9 ≤ tol; an absolute pivot tolerance reported this
    LP unbounded with the "ray" (1, 0), for which A·ray = (1e-9, 0) > 0.
    """
    lp = LinearProgram(
        "tiny",
        "tiny",
        np.array([1.0, 1.0]),
        A_ub=np.array([[1e-9, 0.0], [0.0, 1.0]]),
        b_ub=np.array([1.0, 1.0]),
        sense="max",
    )
    res = numopt.run(method, lp, tol=tol)
    assert_valid_result(res)
    assert res.converged, res.message
    # One pivot per variable on exactly representable data: only rounding of 1/1e-9.
    np.testing.assert_allclose(res.x, [1e9, 1.0], rtol=1e-15)


@pytest.mark.parametrize("method", ("simplex", *PRIMAL))
def test_tiny_pivot_with_large_entry_in_the_same_column(method):
    """Column (1e-5, −1) at tol = 1e-4: the 1e-5 of the row 1e-5·x1 ≤ 1 bounds x1 = 1e5.

    A tolerance scaled only by the column norm (here 1) would still ignore it; the row
    equilibration in τᵢⱼ = tol·σⱼ/σ_B(i) does not.
    """
    lp = LinearProgram(
        "col",
        "col",
        np.array([1.0, 0.0]),
        A_ub=np.array([[1e-5, 0.0], [-1.0, 1.0]]),
        b_ub=np.array([1.0, 1.0]),
        sense="max",
    )
    res = numopt.run(method, lp, tol=1e-4)
    assert res.converged, res.message
    np.testing.assert_allclose(res.x, [1e5, 0.0], rtol=1e-12, atol=0)


def test_dual_simplex_tiny_negative_entry():
    """min x1 + x2 s.t. 1e-10·x1 + x2 ≥ 1, 1e-10·x1 ≥ 1: feasible, optimum x = (1e10, 0).

    The dual ratio test needs the entries −1e-10 < −tol (absolute) to find the entering
    column; with an absolute test it reported the LP infeasible.
    """
    lp = LinearProgram(
        "d",
        "d",
        np.array([1.0, 1.0]),
        A_ub=np.array([[-1e-10, -1.0], [-1e-10, 0.0]]),
        b_ub=np.array([-1.0, -1.0]),
    )
    res = numopt.run("dual_simplex", lp)
    assert res.converged, res.message
    assert lp.A_ub is not None and lp.b_ub is not None
    np.testing.assert_allclose(res.x, [1e10, 0.0], rtol=1e-12, atol=1e-12)
    assert np.all(lp.A_ub @ res.x <= lp.b_ub + 1e-12)


@pytest.mark.parametrize("method", ("simplex", *PRIMAL))
def test_tiny_costs_still_enter(method):
    """Wyndor with c×1e-12: every c̄ⱼ is below an absolute tol = 1e-9, so an absolute
    entering test stops at the origin; the test c̄ⱼ < −tol·σⱼ·γ scales with c."""
    lp = problems.get("wyndor")
    res = numopt.run(method, replace(lp, c=lp.c * 1e-12))
    assert res.converged, res.message
    np.testing.assert_allclose(res.x, [2.0, 6.0], rtol=1e-12)


@pytest.mark.parametrize("scale", (1.0, 1e-12))
def test_dual_simplex_rejects_negative_costs_of_any_scale(scale):
    """Audit: the admission test sign·c < −tol was absolute, so Wyndor with c×1e-12 was
    accepted and returned converged=True at the origin (fun 0, true 3.6e-11)."""
    lp = problems.get("wyndor")
    with pytest.raises(ValueError, match="dual-feasible"):
        numopt.run("dual_simplex", replace(lp, c=lp.c * scale))
    tiny = LinearProgram(
        "t",
        "t",
        np.array([1e-10, 0.0]),
        A_ub=np.eye(2),
        b_ub=np.array([5.0, 1.0]),
        sense="max",
    )
    with pytest.raises(ValueError, match="dual-feasible"):
        numopt.run("dual_simplex", tiny)


# --------------------------------------------------------------------------------------
# Phase 1 / big-M on badly scaled rows (audit: converged=True on infeasible LPs)
# --------------------------------------------------------------------------------------


def _ub_lp(a, b, c, sense: Literal["min", "max"] = "max"):
    return LinearProgram(
        "p", "p", np.array(c, float), A_ub=np.array(a, float), b_ub=np.array(b, float), sense=sense
    )


ROW_SCALED_INFEASIBLE = {
    # x1 ≥ 1 written in ppm, x1 ≤ 0.5: phase 1 with cost 1 stopped at a1 = 5e-7 ≤ 10·tol·‖b‖∞.
    "ppm_row": _ub_lp([[-1e-6, 0], [1, 0], [0, 1]], [-1e-6, 0.5, 1000], [1, 1]),
    "unit_rows": _ub_lp([[-1, 0], [1, 0], [0, 1]], [-1, 0.5, 1000], [1, 1]),
    # x1 ≥ 1 and x1 ≤ 0.95: Σa = 0.05 ≤ 10·tol·(1 + 100) at tol = 1e-4 was accepted.
    "large_rhs": _ub_lp([[-1, 0], [1, 0], [0, 1]], [-1, 0.95, 100], [1, 1]),
    # 1e-9·x1 ≥ 1e-9 and x1 ≤ 0.5: infeasible in exact arithmetic (the violation 5e-10 is
    # below HiGHS's absolute feasibility tolerance 1e-7, so HiGHS reports "unbounded").
    "nano_row": _ub_lp([[-1e-9, 0], [1, 0]], [-1e-9, 0.5], [1, 1]),
    # 6e5·x2 ≥ 40 and x2 ≤ 0: a test aᵢ ≤ tol·(ρᵢ + bᵢ) = 60 at tol = 1e-4 accepts a1 = 40.
    "large_column": _ub_lp([[0, -6e5], [0, 1]], [-40, 0], [1, 1], "min"),
}


@pytest.mark.parametrize("method", PRIMAL)
@pytest.mark.parametrize("tol", (1e-14, 1e-9, 1e-5, 1e-4))
@pytest.mark.parametrize("case", sorted(ROW_SCALED_INFEASIBLE))
def test_infeasible_lp_is_infeasible_at_every_row_scale_and_tol(method, tol, case):
    lp = ROW_SCALED_INFEASIBLE[case]
    res = numopt.run(method, lp, tol=tol)
    assert_valid_result(res)
    assert not res.converged and res.extra["status"] == "infeasible", res.message
    assert "infeasible" in res.message


@pytest.mark.parametrize("method", (*PRIMAL, "dual_simplex"))
@pytest.mark.parametrize("tol", (1e-14, 1e-9, 1e-5, 1e-4))
def test_row_scaled_lp_reaches_the_optimum(method, tol):
    """min x1 + x2 s.t. 1e-6·x1 ≥ 1e-6, x1 + x2 ≥ 5e-5: optimum (1, 0), value 1 (HiGHS).

    Audit (tol = 1e-5): phase 1 with cost 1 stopped with a1 ≈ 1e-6 (c̄ = −1e-6 > −tol),
    and the drive-out pivot on a negative entry returned x = (1, −0.99995), fun 5e-5;
    big_m and dual_simplex returned x = (5e-5, 0), which violates x1 ≥ 1.
    """
    lp = _ub_lp([[-1e-6, 0], [-1, -1]], [-1e-6, -5e-5], [1, 1], "min")
    res = numopt.run(method, lp, tol=tol)
    assert_valid_result(res)
    assert res.converged, res.message
    status, value, _ = oracle(lp)
    assert status == 0 and value is not None and res.fun is not None
    np.testing.assert_allclose(res.fun, value, rtol=1e-12)
    # One pivot on 1e-6 per row: x1 = 1e-6/1e-6 and x2 = 0 up to rounding.
    np.testing.assert_allclose(res.x, [1.0, 0.0], rtol=0, atol=1e-15)
    assert np.all(res.x >= 0.0)


@pytest.mark.parametrize("method", ("two_phase_simplex", "revised_simplex", "big_m"))
def test_small_accepted_artificial_is_zeroed_not_pivoted(method):
    """min x1 s.t. x1 ≤ 1, x1 = 1 + δ with δ = 2⁻³⁴ ≤ tol·(bᵢ + Σ|A_ij|zⱼ) ≈ 2e-9.

    Phase 1 ends at x1 = 1 with the artificial at δ. The textbook drive-out pivot on the
    entry −1 of s1 has θ = −δ: it made s1 = −δ and x1 = 1 + δ while the Step reported
    step_size 0. The accepted level is now set to 0 first, so the pivot is degenerate.
    """
    delta = 2.0**-34
    lp = LinearProgram(
        "d",
        "d",
        np.array([1.0]),
        A_ub=np.array([[1.0]]),
        b_ub=np.array([1.0]),
        A_eq=np.array([[1.0]]),
        b_eq=np.array([1.0 + delta]),
    )
    res = numopt.run(method, lp, tol=1e-9)
    assert_valid_result(res)
    assert res.converged, res.message
    assert res.x.tolist() == [1.0]  # A_ub x ≤ b_ub exactly; |A_eq x − b_eq| = δ ≤ cap
    drives = [i for i, s in enumerate(res.trace) if s.info.get("drive_out")]
    if method == "big_m":
        assert not drives
    else:
        assert len(drives) == 1
        before, after = res.trace[drives[0]], res.trace[drives[0] + 1]
        assert after.step_size == 0.0 and after.x.tolist() == before.x.tolist() == [1.0]
        assert np.all(np.asarray(after.info["tableau"])[1:, -1] >= 0.0)
    # δ above the accepted level: the LP is reported infeasible.
    bigger = replace(lp, b_eq=np.array([1.0 + 2.0**-26]))
    out = numopt.run(method, bigger, tol=1e-9)
    assert not out.converged and out.extra["status"] == "infeasible"


# --------------------------------------------------------------------------------------
# Symbolic big-M: rounding residue in the M row (audit: false "optimal" / "unbounded")
# --------------------------------------------------------------------------------------

# N(0, 1) data (numpy default_rng(7), audit cases 98 and 51). With the Gauss–Jordan update of
# the M row, its residue (≈1e-14) on real columns exceeded tol = 1e-14 after every artificial
# had left: case 98 was reported unbounded (optimum 5.3896845...) and case 51 cycling.
BIG_M_98 = LinearProgram(
    "m98",
    "m98",
    np.array(
        [
            -0.5932970789467739,
            1.1620278326784772,
            -0.29839467518434,
            -0.3091516143961749,
            0.12380104826825901,
        ]
    ),
    A_ub=np.array(
        [
            [
                1.673345147842607,
                -1.9090490809054177,
                1.1712236874228787,
                -0.5687168567540479,
                2.3490052720222456,
            ],
            [
                0.43600792104884084,
                -0.18784437134585374,
                1.105381765888225,
                0.5260284507497229,
                -0.5924559759824957,
            ],
            [
                0.710216943288975,
                -0.5106958703451107,
                -1.788487967143609,
                2.031592193247407,
                -0.34958310074179777,
            ],
            [
                -0.609014000667632,
                0.3387220634549792,
                -1.5596318578537094,
                -1.35357864819892,
                -0.4996467000687651,
            ],
            [
                -0.5197218065528668,
                0.0718892724861329,
                1.1225577131260291,
                -0.4740269731350362,
                0.4927984727498653,
            ],
        ]
    ),
    b_ub=np.array(
        [
            1.2030900018070332,
            -2.168150763195758,
            0.45691144582560245,
            -1.8853347586459248,
            2.1277974890955673,
        ]
    ),
    A_eq=np.array(
        [
            [
                0.4550898780724911,
                0.016504114361246543,
                -1.3396996537511157,
                0.11599001028531707,
                -0.7670275135766861,
            ]
        ]
    ),
    b_eq=np.array([-3.8171292433532833]),
)
BIG_M_51 = LinearProgram(
    "m51",
    "m51",
    np.array([0.6222162018519841, 0.5718545872905609, -1.7830817487195436, -0.3120637504487101]),
    A_ub=np.array(
        [
            [-1.3077856280692512, -1.265946471501941, -0.49012847417311295, -1.852578860174007],
            [-1.3474687028443724, -1.6351369290692526, 0.1822594637816633, 0.4079842295184183],
            [2.0071409631828265, -1.4975297992323828, -0.6795969589096627, 0.9126113893228037],
            [-0.2167738049889562, -0.3269979387077995, 1.709169703339553, -0.3384061595207278],
            [-1.1561948693340067, -1.3169399598669396, 0.3356943223139325, 0.30524160340596707],
        ]
    ),
    b_ub=np.array(
        [
            -2.7434047405723536,
            -1.9707925154526218,
            1.0694252756312241,
            1.1545335450502818,
            -1.2864441672079785,
        ]
    ),
)


@pytest.mark.parametrize("lp", (BIG_M_98, BIG_M_51), ids=("optimal_98", "unbounded_51"))
@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("tol", (1e-14, 1e-9, 1e-4))
def test_big_m_on_gaussian_data_matches_linprog(lp, rule, tol):
    res = numopt.run("big_m", lp, pivot_rule=rule, tol=tol)
    assert_valid_result(res)
    # κ of these 5×5 Gaussian bases is ≤ 1e3: ≥ 12 digits survive.
    check_against_oracle(res, lp, rtol=1e-10)
    if res.converged:
        T = np.asarray(res.trace[-1].info["tableau"])
        labels = res.trace[-1].info["col_labels"][:-1]
        real = [j for j, lab in enumerate(labels) if not lab.startswith("a")]
        assert np.all(T[0, real] == 0.0)  # recomputed M row: exact zeros


def test_big_m_with_rows_and_columns_scaled_by_up_to_1e4():
    """Audit case 68: integer data with rows ×R and columns ×C (10^±4). The Gauss–Jordan
    M row kept 7.45e-9 on x4 (c̄ = −1861) and big_m returned −2.0 instead of −2.2."""
    a_ub = np.array([[2.0, -2, 0, 1, 2, 2], [2, 2, 0, 1, -1, -2], [-2, 2, 2, 0, -1, -2]])
    b_ub = np.array([3.0, -2, -1])
    a_eq = np.array([[-1.0, 1, 2, 2, -2, 2], [2, 0, -2, 1, 1, -2]])
    b_eq = np.array([2.0, -2])
    c = np.array([3.0, 3, -2, 3, 0, -2])
    r_ub = np.array([0.42234169524727944, 0.005364713327003838, 0.020886465971608287])
    r_eq = np.array([5.966267116535948e-01, 4.125978506959315e03])
    col = np.array(
        [
            2.9926158988330758e03,
            1.0041781048581034e-03,
            1.3823201858489243e-04,
            1.8609817325311276e03,
            1.2296017794518414e03,
            3.0292071595535020e01,
        ]
    )
    base = LinearProgram("b", "b", c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq)
    scaled = LinearProgram(
        "s",
        "s",
        c * col,
        A_ub=a_ub * r_ub[:, None] * col,
        b_ub=b_ub * r_ub,
        A_eq=a_eq * r_eq[:, None] * col,
        b_eq=b_eq * r_eq,
    )
    status, value, _ = oracle(base)  # x_scaled = x_base / col has the same objective value
    assert status == 0 and value is not None
    for method in PRIMAL:
        for rule in RULES:
            res = numopt.run(method, scaled, pivot_rule=rule)
            assert res.converged and res.fun is not None, (method, rule, res.message)
            # Scalings up to 1e4·1e4 on rows and columns: ≈8 digits lost to κ of the bases.
            np.testing.assert_allclose(res.fun, value, rtol=1e-8)
            assert_feasible_within_tol(res, scaled, 1e-9)


@st.composite
def origin_feasible_lps(draw):
    """A_ub x ≤ b_ub with b_ub ≥ 0 (slack basis, no artificials), plus a row scaling."""
    n = draw(st.integers(1, 4))
    m = draw(st.integers(1, 4))
    a = draw(hnp.arrays(np.float64, (m, n), elements=st.integers(-4, 6)))
    b = draw(hnp.arrays(np.float64, (m,), elements=st.integers(0, 10)))
    c = draw(hnp.arrays(np.float64, (n,), elements=st.integers(-5, 5)))
    sense = draw(st.sampled_from(["min", "max"]))
    exponents = draw(hnp.arrays(np.int64, (m,), elements=st.integers(-9, 9)))
    lp = LinearProgram("o", "o", c, A_ub=a, b_ub=b, sense=sense)
    return lp, 10.0 ** exponents.astype(np.float64)


@SETTINGS
@given(origin_feasible_lps(), st.sampled_from(("simplex", *PRIMAL)))
def test_row_scaling_does_not_change_the_answer(lp_scale, method):
    """Multiplying row i by 10^kᵢ (|kᵢ| ≤ 9) leaves B⁻¹A, the pivots and the solution unchanged.

    The pivot tolerance τᵢⱼ is row-scale invariant and c̄ = c − (B⁻¹A)ᵀc_B does not depend on
    the row scaling, so the scaled LP must follow the same bases (b ≥ 0: no phase 1).
    """
    lp, scale_rows = lp_scale
    assert lp.A_ub is not None and lp.b_ub is not None
    scaled = replace(lp, A_ub=lp.A_ub * scale_rows[:, None], b_ub=lp.b_ub * scale_rows)
    ref = numopt.run(method, lp, pivot_rule="bland", max_iter=500)
    res = numopt.run(method, scaled, pivot_rule="bland", max_iter=500)
    check_against_oracle(ref, lp, rtol=1e-8)
    assert res.extra["status"] == ref.extra["status"]
    assert [s.info["basis"] for s in res.trace] == [s.info["basis"] for s in ref.trace]
    # Same bases; x_B = B⁻¹b is computed from 10^±9-scaled rows (κ grows with the scaling).
    np.testing.assert_allclose(res.x, ref.x, rtol=1e-6, atol=1e-6)
    if ref.extra["status"] == "unbounded":
        ray = np.asarray(res.extra["ray"])
        # η_B = −B⁻¹A_q: an entry the ratio test counts as 0 (|ā| ≤ τ) may be a rounding-level
        # ±1e-16, the same tolerance as for A·ray ≤ 0.
        assert np.all(ray >= -1e-9 * np.max(np.abs(ray))) and np.all(lp.A_ub @ ray <= 1e-9)


# --------------------------------------------------------------------------------------
# Failure paths and input validation
# --------------------------------------------------------------------------------------


def test_max_iter_is_reported():
    res = numopt.run("simplex", problems.get("klee_minty_3"), max_iter=3)
    assert_valid_result(res, max_iter=3)
    assert not res.converged and "max_iter" in res.message and res.n_iter == 3
    for method in PRIMAL:
        r = numopt.run(method, problems.get("transport_small"), max_iter=2)
        assert_valid_result(r, max_iter=2)
        assert not r.converged and r.extra["status"] == "max_iter"
    r = numopt.run("dual_simplex", problems.get("diet_2d"), max_iter=1)
    assert_valid_result(r, max_iter=1)
    assert not r.converged and r.extra["status"] == "max_iter"


def test_dual_simplex_detects_infeasibility():
    lp = LinearProgram(
        "inf",
        "inf",
        np.array([1.0, 1.0]),
        A_ub=np.array([[1.0, 1.0], [-1.0, -1.0]]),
        b_ub=np.array([2.0, -4.0]),
    )
    res = numopt.run("dual_simplex", lp)
    assert_valid_result(res)
    assert not res.converged and res.extra["status"] == "infeasible"


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="b_ub"):
        numopt.run("simplex", problems.get("diet_2d"))
    with pytest.raises(ValueError, match="A_ub"):
        numopt.run("simplex", problems.get("transport_small"))
    with pytest.raises(ValueError, match="dual-feasible"):
        numopt.run("dual_simplex", problems.get("wyndor"))
    with pytest.raises(ValueError, match="pivot_rule"):
        numopt.run("two_phase_simplex", problems.get("wyndor"), pivot_rule="largest")
    with pytest.raises(TypeError):
        numopt.run("two_phase_simplex", lambda x: x)
    bad = replace(problems.get("wyndor"), b_ub=np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        numopt.run("two_phase_simplex", bad)


def test_no_constraints():
    bounded = LinearProgram("free", "free", np.array([1.0, 2.0]))
    res = numopt.run("two_phase_simplex", bounded)
    assert res.converged and list(res.x) == [0.0, 0.0]
    unbounded = LinearProgram("free", "free", np.array([1.0, -2.0]))
    for method in PRIMAL:
        r = numopt.run(method, unbounded)
        assert_valid_result(r)
        assert r.extra["status"] == "unbounded"


# --------------------------------------------------------------------------------------
# Hypothesis: random LPs against linprog
# --------------------------------------------------------------------------------------


@st.composite
def random_lps(draw, *, dual_feasible: bool = False):
    n = draw(st.integers(1, 4))
    m_ub = draw(st.integers(0, 4))
    m_eq = 0 if dual_feasible else draw(st.integers(0, 2))
    ints = st.integers(-4, 6)
    a_ub = draw(hnp.arrays(np.float64, (m_ub, n), elements=ints))
    b_ub = draw(hnp.arrays(np.float64, (m_ub,), elements=st.integers(-5, 10)))
    a_eq = draw(hnp.arrays(np.float64, (m_eq, n), elements=ints))
    b_eq = draw(hnp.arrays(np.float64, (m_eq,), elements=st.integers(-5, 10)))
    if dual_feasible:
        c = draw(hnp.arrays(np.float64, (n,), elements=st.integers(0, 6)))
        sense = "min"
    else:
        c = draw(hnp.arrays(np.float64, (n,), elements=st.integers(-5, 5)))
        sense = draw(st.sampled_from(["min", "max"]))
    return LinearProgram(
        "rand",
        "rand",
        c,
        A_ub=a_ub if m_ub else None,
        b_ub=b_ub if m_ub else None,
        A_eq=a_eq if m_eq else None,
        b_eq=b_eq if m_eq else None,
        sense=sense,
    )


TOLS = st.sampled_from((1e-14, 1e-9, 1e-4))  # the ParamSpec ends and the default
EPS = np.finfo(np.float64).eps


def assert_feasible_within_tol(res, lp: LinearProgram, tol: float) -> None:
    """converged ⇒ x ≥ 0 and every row holds up to the residual the module docstring accepts.

    Two-phase / big-M: an accepted artificial changes bᵢ by at most tol·(bᵢ + Σⱼ|A_ij|zⱼ)
    at the phase-1 point z, so with x̄ = max |x| over the trace (that point included) a row
    is violated by at most 2·tol·(|bᵢ| + |aᵢ|·x̄); the 2 covers the surplus variable zⱼ of a
    flipped ≤ row. A 64·ε term covers the rounding of A x itself.
    """
    if not res.converged:
        return
    x = np.asarray(res.x)
    x_bar = np.max(np.abs(np.vstack([x, *[s.x for s in res.trace]])), axis=0)
    assert np.all(x >= -tol * (1.0 + x_bar)), x
    for a, b, eq in ((lp.A_ub, lp.b_ub, False), (lp.A_eq, lp.b_eq, True)):
        if a is None or b is None:
            continue
        r = a @ x - b
        viol = np.abs(r) if eq else np.maximum(r, 0.0)
        size = np.abs(b) + np.abs(a) @ x_bar
        assert np.all(viol <= (2.0 * tol + 64 * EPS) * size), (viol, size)


def assert_dual_feasible_within_tol(res, lp: LinearProgram, tol: float) -> None:
    """dual_simplex, converged ⇒ x_B ≥ −tol·|B⁻¹||b| (its documented stopping test), with
    x_B and |B⁻¹| recomputed by numpy from the final basis of A x + s = b (not the tableau)."""
    if not res.converged:
        return
    a_rows = [m for m in (lp.A_ub, lp.A_eq) if m is not None]
    b_rows = [v for v in (lp.b_ub, lp.b_eq) if v is not None]
    if lp.A_eq is not None and lp.b_eq is not None:
        a_rows.append(-lp.A_eq)
        b_rows.append(-lp.b_eq)
    if not a_rows:
        return
    a, b = np.vstack(a_rows), np.concatenate(b_rows)
    full = np.hstack([a, np.eye(b.size)])
    B = full[:, res.extra["basis"]]
    # Test-only: the entries of B⁻¹ are needed for |B⁻¹||b| (κ(B) ≤ 1e4 on this data).
    b_inv = np.linalg.solve(B, np.eye(b.size))
    x_b = np.linalg.solve(B, b)
    scale = np.abs(b_inv) @ np.abs(b)
    assert np.all(x_b >= -(tol * (1.0 + 1e-6) + 1e3 * EPS) * scale), (x_b, scale)


@SETTINGS
@given(random_lps(), st.sampled_from(PRIMAL), TOLS)
def test_random_lps_bland_matches_linprog(lp, method, tol):
    res = numopt.run(method, lp, pivot_rule="bland", max_iter=500, tol=tol)
    assert_valid_result(res, max_iter=500)
    check_against_oracle(res, lp, rtol=1e-8)
    assert_feasible_within_tol(res, lp, tol)


@SETTINGS
@given(random_lps(), st.sampled_from(("dantzig", "steepest_edge")), st.sampled_from(PRIMAL), TOLS)
def test_random_lps_other_rules_correct_unless_cycling(lp, rule, method, tol):
    res = numopt.run(method, lp, pivot_rule=rule, max_iter=500, tol=tol)
    if res.extra["status"] == "cycling":
        assert not res.converged
        return
    check_against_oracle(res, lp, rtol=1e-8)
    assert_feasible_within_tol(res, lp, tol)


@SETTINGS
@given(random_lps(dual_feasible=True), st.sampled_from(("dantzig", "bland")), TOLS)
def test_random_dual_simplex_matches_linprog(lp, rule, tol):
    res = numopt.run("dual_simplex", lp, pivot_rule=rule, max_iter=500, tol=tol)
    assert_valid_result(res, max_iter=500)
    if res.extra["status"] == "cycling":
        assert rule == "dantzig" and not res.converged
        return
    check_against_oracle(res, lp, rtol=1e-8)
    assert_dual_feasible_within_tol(res, lp, tol)


@st.composite
def row_scaled_lps(draw, *, dual_feasible: bool = False):
    """A random LP (b of any sign, equality rows) and the same LP with row i ×10^kᵢ."""
    lp = draw(random_lps(dual_feasible=dual_feasible))
    m_ub = 0 if lp.A_ub is None else lp.A_ub.shape[0]
    m_eq = 0 if lp.A_eq is None else lp.A_eq.shape[0]
    k = draw(hnp.arrays(np.int64, (m_ub + m_eq,), elements=st.integers(-9, 9)))
    scale = 10.0 ** k.astype(np.float64)
    r_ub, r_eq = scale[:m_ub], scale[m_ub:]
    scaled = replace(
        lp,
        A_ub=None if lp.A_ub is None else lp.A_ub * r_ub[:, None],
        b_ub=None if lp.b_ub is None else lp.b_ub * r_ub,
        A_eq=None if lp.A_eq is None else lp.A_eq * r_eq[:, None],
        b_eq=None if lp.b_eq is None else lp.b_eq * r_eq,
    )
    return lp, scaled


@SETTINGS
@given(row_scaled_lps(), st.sampled_from(PRIMAL), TOLS)
def test_row_scaling_with_phase_1_does_not_change_the_answer(pair, method, tol):
    """Rows with b < 0 and equality rows need phase 1 / big-M. Its costs 1/ρᵢ, the per-row
    accept test, the pivot tolerance τ and the relative ratio-test ties are all invariant
    under row scaling, so the scaled LP follows the same bases (audit: phase 1 with cost 1
    and the test Σa ≤ 10·tol·(1 + ‖b‖∞) were not, and gave converged=True on infeasible
    LPs). The oracle is HiGHS on the unscaled LP: HiGHS's absolute feasibility tolerance
    1e-7 accepts e.g. a row 1e-9·(x₁ − 4x₂) ≤ −4e-9 violated by 1e-9.
    """
    lp, scaled = pair
    ref = numopt.run(method, lp, pivot_rule="bland", max_iter=500, tol=tol)
    res = numopt.run(method, scaled, pivot_rule="bland", max_iter=500, tol=tol)
    check_against_oracle(ref, lp, rtol=1e-8)
    assert res.extra["status"] == ref.extra["status"]
    assert [s.info["basis"] for s in res.trace] == [s.info["basis"] for s in ref.trace]
    # Same bases; x_B = B⁻¹b from 10^±9-scaled rows (κ grows with the scaling).
    np.testing.assert_allclose(res.x, ref.x, rtol=1e-6, atol=1e-6)
    assert_feasible_within_tol(res, scaled, tol)


@SETTINGS
@given(row_scaled_lps(dual_feasible=True), TOLS)
def test_row_scaling_does_not_change_the_dual_simplex(pair, tol):
    lp, scaled = pair
    ref = numopt.run("dual_simplex", lp, pivot_rule="bland", max_iter=500, tol=tol)
    res = numopt.run("dual_simplex", scaled, pivot_rule="bland", max_iter=500, tol=tol)
    check_against_oracle(ref, lp, rtol=1e-8)
    assert res.extra["status"] == ref.extra["status"]
    assert [s.info["basis"] for s in res.trace] == [s.info["basis"] for s in ref.trace]
    np.testing.assert_allclose(res.x, ref.x, rtol=1e-6, atol=1e-6)


@settings(max_examples=300, deadline=None)
@given(random_lps())
def test_phase2_objective_is_monotone(lp):
    res = numopt.run("two_phase_simplex", lp, pivot_rule="bland", max_iter=500)
    vals = [s.fun for s in res.trace if s.info["phase"] == 2]
    better = (lambda a, b: b >= a - 1e-9) if lp.sense == "max" else (lambda a, b: b <= a + 1e-9)
    assert all(better(a, b) for a, b in pairwise(vals))


@settings(max_examples=1000, deadline=None)
@given(
    hnp.arrays(np.float64, st.tuples(st.integers(1, 6), st.just(6)), elements=st.floats(-10, 10)),
    st.booleans(),
)
def test_lu_solve_matches_numpy(block, trans):
    m = block.shape[0]
    B = block[:, :m] + 3.0 * np.eye(m) * np.sign(block[:, :m].diagonal() + 0.5)
    rhs = block[:, -1]
    factor = lu_factor(B)
    cond = np.linalg.cond(B)
    if factor is None:
        assert cond > 1e12
        return
    y = lu_solve(factor, rhs, trans=trans)
    ref = np.linalg.solve(B.T if trans else B, rhs)
    # Backward-stable LU: forward error ≲ κ(B)·ε (Higham, Thm 9.4); 1e3 covers growth factor.
    np.testing.assert_allclose(
        y, ref, rtol=0, atol=1e3 * cond * np.finfo(float).eps * (1 + np.abs(ref).max())
    )


def documented_info_keys(module) -> set[str]:
    """Names listed in the module docstring's "Info keys:" section."""
    doc = module.__doc__ or ""
    section = doc.split("Info keys", 1)[1]
    keys: set[str] = set()
    for line in section.splitlines():
        m = re.match(r"^\s{4}([a-z_]+(?:, [a-z_]+)*)(?::| —)", line)
        if m:
            keys.update(k.strip() for k in m.group(1).split(","))
    return keys


def test_every_info_key_is_documented():
    documented = documented_info_keys(simplex_module)
    cases = [
        ("simplex", "wyndor"),
        ("simplex", "unbounded_2d"),
        ("simplex", "beale_cycling"),
        ("two_phase_simplex", "transport_small"),
        ("two_phase_simplex", "infeasible_2d"),
        ("big_m", "diet_2d"),
        ("dual_simplex", "diet_2d"),
        ("revised_simplex", "transport_small"),
        ("revised_simplex", "unbounded_2d"),
    ]
    for method, pid in cases:
        res = numopt.run(method, problems.get(pid))
        for s in res.trace:
            assert set(s.info) <= documented, (method, set(s.info) - documented)
