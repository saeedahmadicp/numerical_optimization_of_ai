"""Tests for restarted PDHG (method.py).

Oracles: scipy.optimize.linprog (HiGHS) for LP optima; a hand-computed PDHG trajectory; the
Lagrangian dual bound and SLSQP for the normalized duality gap (an independent formulation of
the same trust-region problem); the Pock–Chambolle bound ‖Ã‖₂ ≤ 1; the fixed-point property of
the PDHG step at a primal-dual optimum; PDLP's scale invariance (Applegate et al. 2021, App. A).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from numpy.testing import assert_allclose
from scipy.optimize import linprog, minimize, minimize_scalar

from numopt import problems
from numopt.core.types import LinearProgram

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load("pdlp_restarted_pdhg_method", HERE / "method.py")
# The repository's contract checks (tests/conftest.py), loaded under a private name.
assert_valid_result = _load(
    "numopt_tests_conftest", ROOT / "tests" / "conftest.py"
).assert_valid_result

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
    """Equality LP with a known strictly complementary primal-dual optimum (z*, y*)."""
    A = rng.normal(size=(m, n))
    z = np.zeros(n)
    basis = rng.choice(n, size=m, replace=False)
    z[basis] = rng.uniform(0.5, 2.0, size=m)
    y = rng.normal(size=m)
    s = rng.uniform(0.5, 2.0, size=n)
    s[basis] = 0.0
    lp = LinearProgram(id="rand", name="rand", c=A.T @ y + s, A_eq=A, b_eq=A @ z)
    return lp, z, y


def _eq_data(lp: LinearProgram) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert lp.A_eq is not None and lp.b_eq is not None
    return np.asarray(lp.A_eq), np.asarray(lp.b_eq), np.asarray(lp.c)


# --------------------------------------------------------------------------------------
# Hand-computed trajectory and the contract
# --------------------------------------------------------------------------------------


def test_first_three_steps_match_hand_computation() -> None:
    # min x1 + 2 x2 s.t. x1 + x2 = 1, x ≥ 0; ‖A‖₂ = √2, ω = 1, so τ = σ = s = 0.9/√2.
    lp = LinearProgram(
        id="t", name="t", c=np.array([1.0, 2.0]), A_eq=np.array([[1.0, 1.0]]), b_eq=np.array([1.0])
    )
    res = M.restarted_pdhg(lp, max_iter=3, **PLAIN)
    s = 0.9 / math.sqrt(2.0)
    # k=1: z = max(−s c, 0) = 0, y = s·1.  k=2: z = max(−s(c − s), 0) = 0 (s < 1), y = 2s.
    # k=3: c − Aᵀy = (1 − 2s, 2 − 2s) with 1 − 2s < 0, so z = (s(2s − 1), 0) and
    #      y = 2s + s(1 − 2 s(2s − 1)).
    x_hand = [np.zeros(2), np.zeros(2), np.zeros(2), np.array([s * (2 * s - 1), 0.0])]
    y_hand = [0.0, s, 2 * s, 2 * s + s * (1 - 2 * s * (2 * s - 1))]
    for k in range(4):
        assert_allclose(res.trace[k].x, x_hand[k], rtol=1e-14, atol=1e-15)
        assert_allclose(res.trace[k].info["y"], [y_hand[k]], rtol=1e-14, atol=1e-15)
    assert not res.converged and res.n_iter == 3 and len(res.trace) == 4
    assert "max_iter" in res.message


@pytest.mark.parametrize("pid", LIBRARY)
@pytest.mark.parametrize(
    "variant",
    [
        {"restart": "adaptive"},
        {"restart": "adaptive", "primal_weight": "unit", "precondition": "none"},
        {
            "restart": "fixed",
            "restart_period": 64,
            "primal_weight": "unit",
            "precondition": "ruiz_pc",
        },
    ],
    ids=["pdlp", "adaptive", "fixed64+pc"],
)
def test_converges_to_linprog_optimum(pid: str, variant: dict) -> None:
    if pid == "klee_minty_3" and variant.get("precondition") == "none":
        # The study's finding: b spans 1 … 1e4 and A spans 1 … 200; without diagonal
        # preconditioning PDHG is still at KKT error ~1e-1 after 50 000 iterations.
        pytest.xfail("unpreconditioned PDHG does not reach 1e-8 on klee_minty_3 in 50k iterations")
    lp = problems.get(pid)
    res = M.restarted_pdhg(lp, max_iter=50_000, **variant)
    assert_valid_result(res, max_iter=50_000)
    assert res.converged, res.message
    _, f_ref = _linprog(lp)
    # NOTE: a relative KKT error ≤ 1e-8 bounds the objective error by ~1e-8·(1+|f|) only up to
    # the LP's Hoffman constant; 1e-6 relative leaves two digits of margin on these problems.
    assert abs(res.fun - f_ref) <= 1e-6 * (1 + abs(f_ref))
    if lp.optimum is not None and not lp.integer:  # unique optimum known
        assert_allclose(res.x, lp.optimum, rtol=1e-5, atol=1e-5)
    assert res.extra["kkt"] <= 1e-8
    assert res.extra["n_matvec"] == 2 + 2 * res.n_iter
    assert [s.info["matvecs"] for s in res.trace] == [2 + 2 * s.k for s in res.trace]
    assert np.all(res.x >= 0.0)


def test_dual_matches_linprog_on_unique_dual() -> None:
    lp = problems.get("wyndor")  # nondegenerate optimum: unique duals (0, 1.5, 1)
    res = M.restarted_pdhg(lp)
    ref, _ = _linprog(lp)
    # linprog marginals of ≤ rows are ∂f/∂b ≤ 0 for min(−cᵀx); our y is for the same min form.
    assert_allclose(res.extra["y"], ref.ineqlin.marginals, rtol=1e-6, atol=1e-6)


def test_max_iter_and_infeasible_paths_return_unconverged() -> None:
    res = M.restarted_pdhg(problems.get("wyndor"), max_iter=5)
    assert_valid_result(res, max_iter=5)
    assert not res.converged and res.extra["status"] == "max_iter"
    inf = M.restarted_pdhg(problems.get("infeasible_2d"), max_iter=2000)
    assert_valid_result(inf, max_iter=2000)
    assert not inf.converged
    json.dumps(inf.to_dict(), allow_nan=False)


def test_invalid_input_raises() -> None:
    lp = problems.get("wyndor")
    with pytest.raises(ValueError):
        M.restarted_pdhg(lp, restart="sometimes")
    with pytest.raises(ValueError):
        M.restarted_pdhg(lp, beta=1.0)
    with pytest.raises(ValueError):
        M.restarted_pdhg(lp, x0=[1.0])
    with pytest.raises(TypeError):
        M.restarted_pdhg(problems.get("rosenbrock"))


def test_fixed_restarts_happen_exactly_every_period() -> None:
    res = M.restarted_pdhg(
        problems.get("diet_2d"),
        restart="fixed",
        restart_period=50,
        **{k: v for k, v in PLAIN.items() if k != "restart"},
    )
    ks = [s.k for s in res.trace if s.info["restarted"]]
    assert ks and ks == list(range(50, 50 * len(ks) + 1, 50))
    # After a restart the iterate is the epoch average.
    for s in res.trace:
        if s.info["restarted"]:
            assert_allclose(s.x, s.info["x_avg"], rtol=0, atol=0)


def test_adaptive_restart_fires_only_on_beta_decay() -> None:
    base = {k: v for k, v in PLAIN.items() if k != "restart"}
    lp = problems.get("transport_small")
    res = M.restarted_pdhg(lp, restart="adaptive", restart_check_every=1, **base)
    assert res.trace[1].info["restarted"]  # n = 0: τ⁰ = 1
    for s in res.trace[2:]:
        rho, thr = s.info["normalized_gap"], s.info["restart_threshold"]
        if rho is None:
            continue
        assert s.info["restarted"] == (rho <= thr)
    assert res.extra["n_restarts"] >= 3
    # With a check interval, every epoch length is a multiple of it.
    res40 = M.restarted_pdhg(lp, restart="adaptive", restart_check_every=40, **base)
    lens = [s.info["epoch_len"] for s in res40.trace if s.info["restarted"]]
    assert lens and all(n % 40 == 0 for n in lens)
    assert lens[0] == 40  # n = 0 restarts at the first check


def test_primal_weight_update_hand_values() -> None:
    assert M.primal_weight_update(1.0, 4.0, 1.0) == pytest.approx(2.0, rel=1e-15)
    assert M.primal_weight_update(2.0, 2.0, 16.0) == pytest.approx(4.0, rel=1e-15)
    assert M.primal_weight_update(0.0, 1.0, 3.0) == 3.0  # Δz ≤ zero: unchanged


def test_plain_last_iterate_beats_average_and_restarts_beat_plain() -> None:
    # 2023 paper, Table 2: last iterate linear (κ²), average sublinear, restarted linear (κ).
    lp = problems.get("diet_2d")
    plain = M.restarted_pdhg(lp, tol=1e-300, max_iter=20_000, **PLAIN)
    last = [s.info["kkt_last"] for s in plain.trace]
    avg = [s.info["kkt_avg"] for s in plain.trace]
    assert min(last[-100:]) < 1e-10 < min(avg)
    adaptive = M.restarted_pdhg(lp, restart="adaptive", primal_weight="unit", precondition="none")
    first_plain = next(s.k for s in plain.trace if s.info["kkt_last"] <= 1e-8)
    assert adaptive.converged and adaptive.n_iter < first_plain


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


def _gap_instance(m, n, seed, r, omega, zero_frac):
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
    A, z, y, b, c = _gap_instance(m, n, seed, r, omega, zero_frac)
    rho = M.normalized_duality_gap(z, A @ z, A.T @ y, b, c, r, omega)
    upper = _gap_dual_oracle(z, A.T @ y - c, b - A @ z, r, omega) / r
    assert rho >= -1e-15
    # Weak duality: the bound is ≥ the maximum; strong duality (Slater, r > 0) makes it tight.
    # NOTE: rtol 1e-7 is the accuracy of the bounded 1-D search over log λ, not of rho.
    assert rho <= upper * (1 + 1e-9) + 1e-12
    assert rho == pytest.approx(upper, rel=1e-7, abs=1e-10)


@settings(max_examples=200, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), log_r=st.floats(-2.0, 1.0), log_omega=st.floats(-1.0, 1.0))
def test_normalized_gap_matches_slsqp(seed, log_r, log_omega) -> None:
    r, omega = 10.0**log_r, 10.0**log_omega
    A, z, y, b, c = _gap_instance(2, 3, seed, r, omega, 0.3)
    gz, gy = A.T @ y - c, b - A @ z
    g = np.concatenate([gz, gy])
    W = np.concatenate([np.full(3, omega), np.full(2, 1 / omega)])
    lo = np.concatenate([-z, np.full(2, -np.inf)])
    cons = [{"type": "ineq", "fun": lambda d: r * r - W @ (d * d), "jac": lambda d: -2 * W * d}]
    best = -np.inf
    for start in (np.zeros(5), r * g / (np.linalg.norm(g) + 1e-300) / np.sqrt(W.max())):
        sol = minimize(
            lambda d: -(g @ d),
            start,
            jac=lambda d: -g,
            method="SLSQP",
            bounds=[(lb, None) for lb in lo],
            constraints=cons,
            options={"ftol": 1e-14, "maxiter": 500},
        )
        if W @ (sol.x * sol.x) <= r * r * (1 + 1e-8) and np.all(sol.x >= lo - 1e-10):
            best = max(best, g @ sol.x)
    rho = M.normalized_duality_gap(z, A @ z, A.T @ y, b, c, r, omega)
    # NOTE: SLSQP is a local first-order-accurate oracle; 1e-6 relative is its stated ftol regime.
    assert rho * r == pytest.approx(best, rel=1e-6, abs=1e-9)


@settings(max_examples=1000, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), r1=st.floats(1e-3, 10.0), r2=st.floats(1e-3, 10.0))
def test_normalized_gap_is_nonincreasing_in_r_and_zero_at_optimum(seed, r1, r2) -> None:
    rng = np.random.default_rng(seed)
    lp, z_star, y_star = random_lp(rng, 3, 7)
    A, b, c = _eq_data(lp)
    for omega in (0.3, 1.0, 3.0):
        g0 = M.normalized_duality_gap(z_star, A @ z_star, A.T @ y_star, b, c, r1, omega)
        assert abs(g0) <= 1e-12 * (1 + np.abs(c).max())
    z = rng.uniform(0, 2, size=7)
    y = rng.normal(size=3)
    lo, hi = sorted((r1, r2))
    g_lo = M.normalized_duality_gap(z, A @ z, A.T @ y, b, c, lo, 1.0)
    g_hi = M.normalized_duality_gap(z, A @ z, A.T @ y, b, c, hi, 1.0)
    # The value v(r) = r ρ_r is concave with v(0) = 0, so ρ_r = v(r)/r is nonincreasing.
    assert g_hi <= g_lo * (1 + 1e-12) + 1e-14
    g_0 = M.normalized_duality_gap(z, A @ z, A.T @ y, b, c, 0.0, 1.0)
    assert g_lo <= g_0 * (1 + 1e-12) + 1e-14


# --------------------------------------------------------------------------------------
# PDHG step, preconditioner and invariances
# --------------------------------------------------------------------------------------


@settings(max_examples=1000, deadline=None)
@given(
    seed=st.integers(0, 2**32 - 1),
    m=st.integers(1, 5),
    extra=st.integers(1, 5),
    log_tau=st.floats(-3.0, 1.0),
    log_sigma=st.floats(-3.0, 1.0),
)
def test_pdhg_step_fixes_a_primal_dual_optimum(seed, m, extra, log_tau, log_sigma) -> None:
    # (z*, y*) solves the saddle point iff it is a fixed point of PDHG for any τ, σ > 0.
    rng = np.random.default_rng(seed)
    lp, z, y = random_lp(rng, m, m + extra)
    A, b, c = _eq_data(lp)
    tau, sigma = 10.0**log_tau, 10.0**log_sigma
    z_new = np.maximum(z - tau * (c - A.T @ y), 0.0)
    y_new = y + sigma * (b - A @ (2 * z_new - z))
    assert_allclose(z_new, z, rtol=1e-12, atol=1e-12 * (1 + tau))
    assert_allclose(y_new, y, rtol=1e-12, atol=1e-11 * (1 + sigma))
    k = M.kkt_error(M.standard_form(lp), z, y, A @ z, A.T @ y)
    assert k.error <= 1e-13


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
    d1, d2 = M.ruiz_pock_chambolle(A)
    assert np.all(d1 > 0) and np.all(d2 > 0)
    K = A * d1[:, None] * d2[None, :]
    # Pock & Chambolle (2011, Lemma 2), α = 1: ‖Σ^½ K T^½‖₂ ≤ 1 with row/col ℓ1 scalings.
    assert np.linalg.norm(K, 2) <= 1.0 + 1e-12


@pytest.mark.parametrize("precondition", ["none", "ruiz_pc"])
def test_balanced_primal_weight_is_scale_invariant(precondition: str) -> None:
    # PDLP App. A: with ω = ‖c‖/‖b‖ and η = 0.9/‖A‖, scaling c by κ leaves x unchanged and
    # scales y by κ (iterates identical up to rounding).
    lp = problems.get("transport_small")
    lp10 = dataclasses.replace(lp, c=1000.0 * np.asarray(lp.c))
    kw = {
        "restart": "fixed",
        "restart_period": 16,
        "primal_weight": "balanced",
        "precondition": precondition,
        "max_iter": 60,
    }
    a, b = M.restarted_pdhg(lp, **kw), M.restarted_pdhg(lp10, **kw)
    for sa, sb in zip(a.trace, b.trace, strict=True):
        assert_allclose(sb.x, sa.x, rtol=1e-10, atol=1e-10)
        assert_allclose(sb.info["y"], 1000.0 * np.asarray(sa.info["y"]), rtol=1e-10, atol=1e-8)


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), m=st.integers(2, 6), extra=st.integers(2, 8))
def test_random_sharp_lps_solved_to_known_optimum(seed, m, extra) -> None:
    rng = np.random.default_rng(seed)
    lp, z_star, _ = random_lp(rng, m, m + extra)
    # Restarted PDHG converges linearly at a rate set by the LP's sharpness, so an ill-conditioned
    # draw needs many more iterations: about 1 in 150 draws needs more than 50,000,
    # and the slowest of 2,000 draws needed 168,072. The budget only has to be large enough; it
    # does not shape the run.
    res = M.restarted_pdhg(lp, max_iter=500_000)
    assert res.converged, res.message
    assert abs(res.fun - float(lp.c @ z_star)) <= 1e-6 * (1 + abs(float(lp.c @ z_star)))


# --------------------------------------------------------------------------------------
# Experiment design (run.py): grids, configurations and the per-LP aggregation
# --------------------------------------------------------------------------------------

R = _load("pdlp_restarted_pdhg_run", HERE / "run.py")


def test_period_grids_are_nested() -> None:
    # The grid sensitivity study compares oracles on nested grids: 4^j ⊂ 2^j ⊂ 2^(j/2).
    pow4, pow2, half = (set(R.GRIDS[g]) for g in ("pow4", "pow2", "half_octave"))
    assert pow4 < pow2 < half
    assert min(half) == 4 and max(half) == 16384 and len(half) == 25
    assert list(R.PERIODS) == sorted(half)


def test_default_variant_is_the_pc_configuration_with_adaptive_restarts() -> None:
    # "default" (method defaults) must be the adaptive scheme of the pc configuration, so that
    # default ÷ fixed-T+pc and plain+pc ÷ default compare inside one configuration.
    lp = problems.get("diet_2d")
    a = M.restarted_pdhg(lp, max_iter=400)
    b = M.restarted_pdhg(
        lp, max_iter=400, restart="adaptive", restart_check_every=40, beta=math.exp(-1.0), **R.PC
    )
    assert a.n_iter == b.n_iter
    for sa, sb in zip(a.trace, b.trace, strict=True):
        assert_allclose(sa.x, sb.x, rtol=0, atol=0)


@settings(max_examples=1000, deadline=None)
@given(
    data=st.data(),
    groups=st.lists(st.integers(1, 3), min_size=1, max_size=6),
)
def test_per_lp_ratio_is_geomean_of_per_start_ratios(data, groups) -> None:
    names = [f"lp{g}@{s}" for g, k in enumerate(groups) for s in range(k)]
    vals = st.one_of(st.floats(1.0, 1e5), st.just(math.inf))
    a = np.array(data.draw(st.lists(vals, min_size=len(names), max_size=len(names))))
    b = np.array(data.draw(st.lists(st.floats(1.0, 1e5), min_size=len(names), max_size=len(names))))
    lps, ag = R.per_lp(a[:, None], names)
    _, bg = R.per_lp(b[:, None], names)
    assert lps == [f"lp{g}" for g in range(len(groups))]
    for i, lp in enumerate(lps):
        rows = [j for j, n in enumerate(names) if n.startswith(lp + "@")]
        if np.all(np.isfinite(a[rows])):
            want = float(np.exp(np.mean(np.log(a[rows] / b[rows]))))
            assert_allclose(ag[i, 0] / bg[i, 0], want, rtol=1e-12)
        else:
            assert ag[i, 0] == math.inf


def test_ratio_stats_counts() -> None:
    inf = math.inf
    s = R.ratio_stats(np.array([1.0, 4.0, inf, 2.0, inf]), np.array([2.0, 2.0, 1.0, inf, inf]))
    assert s["n_both_solved"] == 2 and s["n_ratio_le_1"] == 1 and s["n_ratio_gt_1"] == 1
    assert s["median_ratio"] == pytest.approx(1.25) and s["geomean_ratio"] == pytest.approx(1.0)
    assert s["n_only_numerator_solved"] == 1 and s["n_only_denominator_solved"] == 1


def test_siblings_are_held_out_and_of_the_same_lp_or_size() -> None:
    names = [n for n, _, _ in R.instances()]
    for i, n in enumerate(names):
        sib = R.siblings(names, i)
        assert i not in sib and len(sib) == (1 if n.startswith("rand_") else 2)
        for j in sib:
            if n.startswith("rand_"):
                assert names[j].split("_s")[0] == n.split("_s")[0] and names[j] != n
            else:
                assert R.lp_of(names[j]) == R.lp_of(n)
