"""Benchmark profiles as a shared research harness (extensions of ``numopt.bench``).

``numopt.bench`` runs solvers under a budget, records every evaluation, and computes the
Dolan–Moré performance profile and the Moré–Wild data profile. This module adds what a
research comparison needs on top of it:

1. **The BenDFO cutoff** (Moré & Wild 2009, §2; ``perf_profile.m`` / ``data_profile.m`` in
   https://github.com/POptUS/BenDFO). The convergence test (Moré–Wild eq. 2.2)

       f(x) ≤ f_L + τ (f(x₀) − f_L),                                           (2.2)

   uses f_L = the smallest value that *any* solver found on the instance (f(x₀) included),
   not the known global minimum. ``with_bendfo_cutoff`` rewrites a ``BenchmarkResult`` to
   that convention; ``numopt.bench`` uses the known minimum when the problem states one.

2. **A literal port of the BenDFO reference code** on a raw history array
   ``H[k, p, s]`` = f at evaluation k+1 of solver s on problem p (``bendfo_solve_counts``,
   ``bendfo_performance_ratios``, ``bendfo_data_values``), and a raw evaluation logger
   (``log_evaluations``) that fills H. This is an independent oracle for ``numopt.bench``.

3. **Scalar summaries of a profile**, so that "the ranking of the solvers" is a defined
   quantity. For a performance profile ρ_s (Dolan & Moré 2002, eq. 1, with
   r_{p,s} = t_{p,s} / min_σ t_{p,σ}) we use the normalized area on a log₂ axis,

       A_s = (1/L) ∫₀ᴸ ρ_s(2ᵘ) du = (1/|P|) Σ_p max(0, 1 − log₂(r_{p,s}) / L),
       L = log₂ α_max,

   and for a data profile d_s (Moré–Wild eq. 2.7, κ_{p,s} = t_{p,s} / (n_p + 1))

       D_s = (1/L) ∫₀ᴸ d_s(2ᵘ) du = (1/|P|) Σ_p max(0, 1 − log₂(max(κ_{p,s}, 1)) / L),
       L = log₂ κ_max.

   An unsolved instance (t = ∞) contributes 0. The closed forms follow from
   ∫₀ᴸ 1[log₂ r ≤ u] du = max(0, L − log₂ r) for log₂ r ≥ 0.

4. **Pairwise dominance** of profile curves (``dominance``): A dominates B when its curve is
   ≥ B's at every break point and > at one; otherwise the curves cross or are equal.

5. **A cluster bootstrap** of the area summaries (``bootstrap_order_support``). Instances of
   one problem (different start points and seeds) are correlated, so the bootstrap draws
   whole problems with replacement (``numopt.core.rng.Rng``, a documented draw order).

6. **The COCO runtime ECDF and ERT** (Hansen et al. 2021, §5–6; Hansen et al. 2016,
   "COCO: performance assessment", arXiv:1605.03560). For a target f_opt + Δf the runtime
   of a run is the number of evaluations until the best value is ≤ f_opt + Δf. The ECDF is
   the fraction of (run, target) pairs whose runtime / n is ≤ x. The expected running time
   is ERT(Δf) = (Σ_runs evaluations spent until success or stop) / #successes.

7. **An audit of f_L against the stated minimum** (``cutoff_audit``). A best value far below
   f_known means that f_known is not a minimum of f on ℝⁿ (for example, the minimum holds on a
   box only, and f is unbounded below). Then both cutoff conventions use a divergent f_L, and
   the COCO targets count every escape as a hit. A study must not use such an instance.

8. **Multiple comparisons and leave-one-problem-out** (``holm``, ``order_support``,
   ``leave_one_group_out``). A rank change between two settings is an intersection-union
   test: its p-value is 1 − min(support in setting a, support in setting b) (Berger 1982).
   ``holm`` adjusts a family of such p-values (Holm 1979). ``leave_one_group_out`` drops each
   problem in turn and recomputes the point estimates and the bootstrap supports.

Info keys: none (this module defines no iterative solver, so it emits no ``Step``).

``# NOTE:`` deviations from the references, each repeated where it is implemented:
  * BenDFO takes f(x₀) = H(1, p, 1), the first value of solver 1. We pass f(x₀) explicitly
    (``numopt.bench`` evaluates it once per instance), because a solver need not evaluate x₀
    first (CMA-ES samples around it).
  * BenDFO marks a failure with NaN; we use +∞ (same profile values, no NaN arithmetic).
  * COCO's runtime ECDF uses simulated restarts of unsuccessful runs (a bootstrap). We
    report single-run runtimes; at the end of the budget the ECDF is the success rate.
  * COCO measures Δf against the known global optimum f_opt. A problem without one uses the
    BenDFO f_L instead.
  * The bootstrap p-value of an order is the fraction of replicates against the point-estimate
    order (a percentile bootstrap p-value, Efron & Tibshirani 1993, §16), not an exact test.

References:
    E. D. Dolan, J. J. Moré, "Benchmarking optimization software with performance
    profiles", Math. Program. 91 (2002) 201–213, doi:10.1007/s101070100263.
    J. J. Moré, S. M. Wild, "Benchmarking derivative-free optimization algorithms",
    SIAM J. Optim. 20(1) (2009) 172–191, doi:10.1137/080724083.
    N. Hansen, A. Auger, R. Ros, O. Mersmann, T. Tušar, D. Brockhoff, "COCO: a platform for
    comparing continuous optimizers in a black-box setting", Optim. Methods Softw. 36
    (2021) 114–144, doi:10.1080/10556788.2020.1808977.
    S. Holm, "A simple sequentially rejective multiple test procedure", Scand. J. Statist. 6
    (1979) 65–70.
    R. L. Berger, "Multiparameter hypothesis testing and acceptance sampling",
    Technometrics 24 (1982) 295–300 (the intersection-union test).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from numopt import bench
from numopt.core.registry import ParamSpec, run
from numopt.core.rng import Rng
from numopt.core.types import Problem, Result

__all__ = [
    "PARAMS",
    "BootstrapSupport",
    "CutoffAudit",
    "Ecdf",
    "area_terms",
    "bendfo_data_values",
    "bendfo_performance_ratios",
    "bendfo_solve_counts",
    "bootstrap_order_support",
    "coco_targets",
    "cutoff_audit",
    "data_area",
    "dominance",
    "ecdf_area",
    "expected_running_time",
    "history_array",
    "holm",
    "kendall_tau_b",
    "leave_one_group_out",
    "log_evaluations",
    "order_support",
    "performance_area",
    "performance_ratios",
    "rank_scores",
    "runtime_ecdf",
    "with_bendfo_cutoff",
]

# Parameters of the harness, described as numopt ParamSpecs so the options can move into
# numopt.bench unchanged.
PARAMS: dict[str, ParamSpec] = {
    "tau": ParamSpec(
        "tau",
        1e-3,
        min=1e-12,
        max=0.5,
        log=True,
        help="Moré–Wild eq. 2.2 tolerance: solved when f ≤ f_L + τ (f(x₀) − f_L).",
    ),
    "f_L": ParamSpec(
        "f_L",
        "bendfo",
        kind="choice",
        choices=("bendfo", "known"),
        help="Cutoff f_L: best value any solver found (BenDFO) or the known minimum.",
    ),
    "alpha_max": ParamSpec(
        "alpha_max",
        32.0,
        min=2.0,
        max=2.0**20,
        log=True,
        help="Right end of the performance-profile area A_s (log₂ axis from α = 1).",
    ),
    "kappa_max": ParamSpec(
        "kappa_max",
        500.0,
        min=2.0,
        max=1e6,
        log=True,
        help="Right end of the data-profile area D_s, in simplex gradients (log₂ axis from κ = 1).",
    ),
    "n_boot": ParamSpec(
        "n_boot",
        2000,
        kind="int",
        min=100,
        max=100_000,
        help="Cluster-bootstrap replicates (problems drawn with replacement).",
    ),
}


# --------------------------------------------------------------------------------------
# 1. The BenDFO cutoff on a numopt.bench result
# --------------------------------------------------------------------------------------


def with_bendfo_cutoff(result: bench.BenchmarkResult) -> bench.BenchmarkResult:
    """Return a copy of ``result`` whose f_L is the BenDFO cutoff.

    f_L(p) = min(f(x₀), min over solvers s of the best value s found on p). The known
    minimum of the problem is ignored. Under this cutoff every instance is solved by at
    least one solver at every τ ∈ (0, 1), because that solver reaches f_L ≤ the threshold.
    """
    instances = []
    for inst in result.instances:
        bests = [result.runs[(lab, inst.id)].best for lab in result.labels]
        f_found = min([inst.f0] + [float(b[-1]) for b in bests if b.size])
        instances.append(dataclasses.replace(inst, f_star=f_found))
    return dataclasses.replace(result, instances=tuple(instances))


# --------------------------------------------------------------------------------------
# 2. A literal port of BenDFO (perf_profile.m, data_profile.m) and a raw evaluation logger
# --------------------------------------------------------------------------------------


def log_evaluations(
    method: str | Callable[..., Result],
    problem: Problem,
    *,
    x0: ArrayLike,
    budget: int,
    seed: int | None = None,
    params: Mapping[str, Any] | None = None,
) -> tuple[Result | None, list[float]]:
    """Run a solver and return its Result and the raw value of every f evaluation, in order.

    The run stops (Result ``None``) when the solver asks for evaluation ``budget + 1``. This
    is the history column H[:, p, s] of the BenDFO scripts (not best-so-far values).
    """
    values: list[float] = []

    class _Stop(Exception):
        pass

    def f(x: Any) -> float:
        if len(values) >= budget:
            raise _Stop
        v = float(problem.f(x))
        values.append(v)
        return v

    logged = dataclasses.replace(problem, f=f, grad=None, hess=None)
    kw: dict[str, Any] = dict(params or {})
    kw["x0"] = np.asarray(x0, dtype=np.float64).tolist()
    if seed is not None:
        kw["seed"] = seed
    try:
        res = run(method, logged, **kw) if isinstance(method, str) else method(logged, **kw)
    except _Stop:
        res = None
    return res, values


def history_array(
    columns: Sequence[Sequence[Sequence[float]]], n_evals: int
) -> NDArray[np.float64]:
    """Stack raw histories ``columns[p][s]`` into H with shape (n_evals, P, S).

    A history shorter than ``n_evals`` is padded with +∞ (no further evaluation). BenDFO's
    running minimum then carries its best value forward.
    """
    P = len(columns)
    S = len(columns[0]) if P else 0
    H = np.full((n_evals, P, S), np.inf)
    for p, row in enumerate(columns):
        if len(row) != S:
            raise ValueError("every problem needs one history per solver")
        for s, h in enumerate(row):
            v = np.asarray(h, dtype=np.float64)[:n_evals]
            H[: v.size, p, s] = np.where(np.isnan(v), np.inf, v)  # NaN never improves
    return H


def bendfo_solve_counts(
    H: ArrayLike, gate: float, f0: ArrayLike | None = None
) -> NDArray[np.float64]:
    """T[p, s] = first evaluation count k with min_{i≤k} H[i, p, s] ≤ cutoff_p (∞ if none).

    Port of the loop shared by ``perf_profile.m`` and ``data_profile.m`` (BenDFO)::

        H(i,:,j) = min(H(i,:,j), H(i-1,:,j))        % running minimum
        prob_min = min(min(H), [], 3)               % f_L over every solver
        prob_max = H(1,:,1)                         % f(x0)
        cutoff = prob_min(p) + gate*(prob_max(p) - prob_min(p))
        T(p,s) = find(H(:,p,s) <= cutoff, 1)        % NaN if empty

    Args:
        H: (n_evals, P, S) raw values (row k is evaluation k + 1).
        gate: τ ∈ (0, 1).
        f0: (P,) the values f(x₀). ``# NOTE:`` BenDFO uses H[0, :, 0]; that is the default.
    """
    if not 0.0 < gate < 1.0:
        raise ValueError(f"gate must be in (0, 1), got {gate!r}")
    Hm = np.minimum.accumulate(np.asarray(H, dtype=np.float64), axis=0)  # (nf, P, S)
    _, P, S = Hm.shape
    prob_max = Hm[0, :, 0] if f0 is None else np.asarray(f0, dtype=np.float64).reshape(P)
    prob_min = np.minimum(Hm.min(axis=(0, 2)), prob_max)  # f(x₀) is a value seen on p
    T = np.full((P, S), np.inf)  # NOTE: BenDFO uses NaN for "not solved"
    for p in range(P):
        cutoff = prob_min[p] + gate * (prob_max[p] - prob_min[p])
        for s in range(S):
            hit = np.flatnonzero(Hm[:, p, s] <= cutoff)
            if hit.size:
                T[p, s] = hit[0] + 1  # MATLAB's 1-based index = number of evaluations
    return T


def bendfo_performance_ratios(T: ArrayLike) -> NDArray[np.float64]:
    """r[p, s] = T[p, s] / min_σ T[p, σ] (``perf_profile.m``); ∞ where T is ∞."""
    T_arr = np.asarray(T, dtype=np.float64)
    minperf = T_arr.min(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(T_arr), T_arr / minperf, np.inf)


def bendfo_data_values(T: ArrayLike, n: ArrayLike) -> NDArray[np.float64]:
    """κ[p, s] = T[p, s] / (n_p + 1) (``data_profile.m`` with N(p) = n(p) + 1)."""
    T_arr = np.asarray(T, dtype=np.float64)
    return T_arr / (np.asarray(n, dtype=np.float64).reshape(-1, 1) + 1.0)


# --------------------------------------------------------------------------------------
# 3. Scalar summaries and rankings
# --------------------------------------------------------------------------------------


def performance_ratios(costs: ArrayLike) -> NDArray[np.float64]:
    """r[p, s] = t[p, s] / min_σ t[p, σ] for a cost table from ``numopt.bench`` (∞ = failed)."""
    return bendfo_performance_ratios(costs)


def performance_area(R: ArrayLike, alpha_max: float) -> NDArray[np.float64]:
    """A_s = (1/|P|) Σ_p max(0, 1 − log₂ r_{p,s} / log₂ α_max), shape (S,).

    Equals (1/L) ∫₀ᴸ ρ_s(2ᵘ) du with L = log₂ α_max: the mean height of the performance
    profile on a log₂ axis over [1, α_max].
    """
    if not alpha_max > 1.0:
        raise ValueError(f"alpha_max must be > 1, got {alpha_max!r}")
    return _clipped_log_area(np.asarray(R, dtype=np.float64), 1.0, alpha_max)


def data_area(K: ArrayLike, kappa_max: float) -> NDArray[np.float64]:
    """D_s = (1/|P|) Σ_p max(0, 1 − log₂ max(κ_{p,s}, 1) / log₂ κ_max), shape (S,).

    Equals (1/L) ∫₀ᴸ d_s(2ᵘ) du with L = log₂ κ_max. ``# NOTE:`` a solve within less
    than one simplex gradient (κ < 1) counts as κ = 1, the left end of the axis.
    """
    if not kappa_max > 1.0:
        raise ValueError(f"kappa_max must be > 1, got {kappa_max!r}")
    return _clipped_log_area(np.asarray(K, dtype=np.float64), 1.0, kappa_max)


def _clipped_log_area(V: NDArray[np.float64], lo: float, hi: float) -> NDArray[np.float64]:
    """Per-column mean of max(0, 1 − log₂(max(v, lo)/lo) / log₂(hi/lo)); ∞ gives 0."""
    terms = _clipped_log_terms(V, lo, hi)  # (P, S)
    # math.fsum is correctly rounded, so a column's area does not depend on the other
    # columns or on the summation order (a data area is then bitwise independent of the
    # other solvers, as d_s itself is).
    return np.array([math.fsum(col) for col in terms.T]) / terms.shape[0]


def _clipped_log_terms(V: NDArray[np.float64], lo: float, hi: float) -> NDArray[np.float64]:
    if V.ndim != 2 or V.size == 0:
        raise ValueError(f"expected a non-empty (problems, solvers) table, got shape {V.shape}")
    if np.isnan(V).any():
        raise ValueError("NaN in the table (use inf for 'not solved')")
    L = math.log2(hi / lo)
    with np.errstate(divide="ignore"):
        u = np.log2(np.maximum(V, lo) / lo)  # ∞ stays ∞
    return np.maximum(0.0, 1.0 - u / L)


def rank_scores(scores: ArrayLike) -> NDArray[np.float64]:
    """Rank 1 = the largest score; tied scores share the mean of their ranks."""
    x = np.asarray(scores, dtype=np.float64)
    ranks = np.empty(x.size)
    for i in range(x.size):
        ranks[i] = 1.0 + np.sum(x > x[i]) + 0.5 * (np.sum(x == x[i]) - 1)
    return ranks


def kendall_tau_b(a: ArrayLike, b: ArrayLike) -> float:
    """Kendall's τ_b between two score vectors (Kendall 1945 tie correction).

    τ_b = (n_c − n_d) / √((n₀ − n₁)(n₀ − n₂)), with n₀ = n(n−1)/2 and n₁, n₂ the tied pairs
    in a and b. NaN when a or b is constant.
    """
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("a and b must have the same length")
    i, j = np.triu_indices(x.size, k=1)
    sx = np.sign(x[i] - x[j])
    sy = np.sign(y[i] - y[j])
    n0 = i.size
    n1 = int(np.sum(sx == 0))
    n2 = int(np.sum(sy == 0))
    denom = math.sqrt((n0 - n1) * (n0 - n2))
    return float(np.sum(sx * sy) / denom) if denom > 0 else math.nan


def dominance(y: ArrayLike) -> NDArray[np.int64]:
    """Pairwise dominance of profile curves sampled at every break point, y shape (S, G).

    D[a, b] = 1 when curve a ≥ curve b everywhere and > somewhere; −1 for the reverse;
    0 when the curves are equal; 2 when they cross. Because a profile is a right-continuous
    step function whose jumps are all in the grid, the grid values decide this exactly.
    """
    Y = np.asarray(y, dtype=np.float64)
    S = Y.shape[0]
    D = np.zeros((S, S), dtype=np.int64)
    for a in range(S):
        for b in range(S):
            ge = bool(np.all(Y[a] >= Y[b]))
            le = bool(np.all(Y[a] <= Y[b]))
            D[a, b] = 0 if (ge and le) else 1 if ge else -1 if le else 2
    return D


@dataclass(frozen=True)
class BootstrapSupport:
    """Cluster-bootstrap support of the area summaries.

    ``p_greater[a, b]`` = fraction of replicates with score_a > score_b (ties count ½).
    ``scores`` has shape (n_boot, S); ``ci`` holds the 2.5 % and 97.5 % quantiles, (2, S).
    """

    p_greater: NDArray[np.float64]
    scores: NDArray[np.float64]
    ci: NDArray[np.float64]


def bootstrap_order_support(
    terms: ArrayLike, groups: Sequence[str], *, n_boot: int = 2000, seed: int = 0
) -> BootstrapSupport:
    """Bootstrap the per-solver mean of ``terms[p, s]`` by drawing whole groups.

    ``terms`` are the per-instance area contributions (``area_terms``); ``groups[p]`` names
    the problem of instance p. Draw order: for each replicate, G calls of
    ``Rng(seed).integers(G)`` pick the groups (G = number of distinct groups, sorted by
    name). The replicate score is Σ_g w_g Σ_{p∈g} terms[p] / Σ_g w_g |g|.
    """
    X = np.asarray(terms, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] != len(groups):
        raise ValueError("terms must be (instances, solvers) with one group name per instance")
    names = sorted(set(groups))
    G, S = len(names), X.shape[1]
    index = {g: i for i, g in enumerate(names)}
    gid = np.array([index[g] for g in groups])
    sums = np.zeros((G, S))
    np.add.at(sums, gid, X)  # (G, S) per-group sums
    sizes = np.bincount(gid, minlength=G).astype(np.float64)  # (G,)
    rng = Rng(seed)
    W = np.zeros((n_boot, G))
    for b in range(n_boot):
        for _ in range(G):
            W[b, rng.integers(G)] += 1.0
    scores = (W @ sums) / (W @ sizes)[:, None]  # (n_boot, S)
    diff = scores[:, :, None] - scores[:, None, :]  # (n_boot, S, S)
    p_greater = (diff > 0).mean(axis=0) + 0.5 * (diff == 0).mean(axis=0)
    ci = np.quantile(scores, [0.025, 0.975], axis=0)
    return BootstrapSupport(p_greater=p_greater, scores=scores, ci=ci)


def area_terms(V: ArrayLike, hi: float) -> NDArray[np.float64]:
    """Per-instance contributions max(0, 1 − log₂ max(v, 1) / log₂ hi); their column mean is
    ``performance_area`` (V = ratios, hi = α_max) or ``data_area`` (V = κ, hi = κ_max)."""
    return _clipped_log_terms(np.asarray(V, dtype=np.float64), 1.0, hi)


# --------------------------------------------------------------------------------------
# 6. COCO runtime ECDF and expected running time
# --------------------------------------------------------------------------------------


def coco_targets(
    lo_exp: float = -8.0, hi_exp: float = 2.0, per_decade: int = 5
) -> NDArray[np.float64]:
    """Δf targets 10^{hi_exp}, …, 10^{lo_exp}, ``per_decade`` per decade (COCO bbob: 51)."""
    m = round((hi_exp - lo_exp) * per_decade) + 1
    return 10.0 ** np.linspace(hi_exp, lo_exp, m)


def _f_opt(result: bench.BenchmarkResult, inst: bench.Instance) -> float:
    if inst.f_known is not None:
        return inst.f_known
    # NOTE: no known optimum → the BenDFO f_L (best value any solver found).
    bests = [result.runs[(lab, inst.id)].best for lab in result.labels]
    return min([inst.f0] + [float(b[-1]) for b in bests if b.size])


@dataclass(frozen=True)
class Ecdf:
    """Runtime ECDF: ``y[s, j]`` = fraction of (run, target) pairs with runtime/n ≤ x[j]."""

    labels: tuple[str, ...]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    runtimes: NDArray[np.float64]  # (S, runs, targets) evaluations / n; ∞ = target missed


def runtime_ecdf(result: bench.BenchmarkResult, deltas: ArrayLike) -> Ecdf:
    """COCO runtime ECDF over every instance (problem × start × seed) and target Δf.

    ``# NOTE:`` single-run runtimes, no simulated restarts (see the module docstring).
    """
    D = np.asarray(deltas, dtype=np.float64).reshape(-1)
    if D.size == 0 or (D <= 0).any():
        raise ValueError("deltas must be positive")
    S, P = len(result.labels), len(result.instances)
    RT = np.full((S, P, D.size), np.inf)
    for p, inst in enumerate(result.instances):
        f_opt = _f_opt(result, inst)
        for s, lab in enumerate(result.labels):
            h = result.runs[(lab, inst.id)]
            for j, d in enumerate(D):
                RT[s, p, j] = h.solve_cost(f_opt + d) / inst.n
    finite = RT[np.isfinite(RT)]
    hi = max(
        float(finite.max()) if finite.size else 1.0,
        result.budget / min(i.n for i in result.instances),
    )
    lo = float(finite.min()) if finite.size else 1.0
    grid = np.unique(np.concatenate([np.geomspace(lo / 2, hi, 200), finite]))
    flat = RT.reshape(S, -1)
    y = np.stack(
        [np.searchsorted(np.sort(flat[s]), grid, side="right") / flat.shape[1] for s in range(S)]
    )
    return Ecdf(labels=result.labels, x=grid, y=y, runtimes=RT)


def ecdf_area(ecdf: Ecdf, x_max: float) -> NDArray[np.float64]:
    """Mean height of the ECDF on a log axis over [1, x_max] (runtime/n < 1 counts as 1).

    The normalized area does not depend on the base of the logarithm.
    """
    return _clipped_log_area(ecdf.runtimes.reshape(len(ecdf.labels), -1).T, 1.0, x_max)


def expected_running_time(
    result: bench.BenchmarkResult, delta: float, instances: Sequence[str] | None = None
) -> NDArray[np.float64]:
    """ERT(Δf) per solver over the given instances (default: all), shape (S,).

    ERT = Σ_runs (evaluations until success, or all evaluations spent) / #successes
    (Hansen et al. 2021, §6; Auger & Hansen 2005); ∞ when no run succeeds.
    """
    ids = set(instances) if instances is not None else {i.id for i in result.instances}
    out = np.empty(len(result.labels))
    for s, lab in enumerate(result.labels):
        spent, wins = 0.0, 0
        for inst in result.instances:
            if inst.id not in ids:
                continue
            h = result.runs[(lab, inst.id)]
            t = h.solve_cost(_f_opt(result, inst) + delta)
            if math.isfinite(t):
                spent += t
                wins += 1
            else:
                spent += h.total_cost
        out[s] = spent / wins if wins else math.inf
    return out


# --------------------------------------------------------------------------------------
# 7. Audit of the cutoff f_L against the stated minimum
# --------------------------------------------------------------------------------------

_EPS = float(np.finfo(np.float64).eps)
_AUDIT_ULPS = 64.0  # rounding allowance of an f evaluation near its minimum, in ε·max(1, |f*|)


@dataclass(frozen=True)
class CutoffAudit:
    """The result of ``cutoff_audit``.

    ``gap[p]`` = (f_known − f_found) / (f(x₀) − f_known): how far below the stated minimum the
    best value of any solver is, in units of the eq. 2.2 scale (NaN when the problem states no
    minimum, ≤ 0 when no solver went below it). ``flagged`` lists the failing instance ids.
    """

    ids: tuple[str, ...]
    gap: NDArray[np.float64]
    flagged: tuple[str, ...]


def cutoff_audit(result: bench.BenchmarkResult, *, rtol: float = 1e-10) -> CutoffAudit:
    """Flag every instance whose best found value is far below the problem's stated minimum.

    With f_found = min(f(x₀), best value of every solver), instance p fails when

        f_known − f_found > rtol · (f(x₀) − f_known) + 64 ε max(1, |f_known|).

    An f_L that is δ below f_known moves the eq. 2.2 threshold by δ(1 − τ). The default
    rtol = 1e-10 keeps that shift below 10⁻³ of the threshold distance τ (f(x₀) − f_known) at
    τ = 10⁻⁷. The 64 ε term allows for rounding in f near its minimum. A failing instance
    means that f_known is not the infimum of f where the solvers searched (for example, a
    problem whose minimum holds on a box only). Both cutoff conventions then use a divergent
    f_L, and the result must not be used.
    """
    if not rtol >= 0.0:
        raise ValueError(f"rtol must be ≥ 0, got {rtol!r}")
    ids, gaps, flagged = [], [], []
    for inst in result.instances:
        ids.append(inst.id)
        if inst.f_known is None:
            gaps.append(math.nan)
            continue
        bests = [result.runs[(lab, inst.id)].best for lab in result.labels]
        f_found = min([inst.f0] + [float(b[-1]) for b in bests if b.size])
        scale = inst.f0 - inst.f_known
        below = inst.f_known - f_found
        tol = rtol * max(scale, 0.0) + _AUDIT_ULPS * _EPS * max(1.0, abs(inst.f_known))
        if scale > 0:
            gaps.append(below / scale)
        else:  # x₀ is at the stated minimum (gap 0) or below it (gap +∞, always a failure)
            gaps.append(math.inf if below > tol else 0.0)
        if below > tol:
            flagged.append(inst.id)
    return CutoffAudit(ids=tuple(ids), gap=np.array(gaps), flagged=tuple(flagged))


# --------------------------------------------------------------------------------------
# 8. Multiple comparisons and leave-one-problem-out
# --------------------------------------------------------------------------------------


def holm(p: ArrayLike) -> NDArray[np.float64]:
    """Holm (1979) adjusted p-values: reject H_i at family-wise level α iff ``holm(p)[i] ≤ α``.

    With p sorted ascending, p̃_(k) = max_{l ≤ k} min(1, (m − l + 1) p_(l)), k = 1, …, m.
    """
    x = np.asarray(p, dtype=np.float64).reshape(-1)
    if np.isnan(x).any() or (x < 0).any() or (x > 1).any():
        raise ValueError("p-values must be in [0, 1]")
    m = x.size
    order = np.argsort(x, kind="stable")
    adj_sorted = np.maximum.accumulate(np.minimum(1.0, (m - np.arange(m)) * x[order]))
    out = np.empty(m)
    out[order] = adj_sorted
    return out


def order_support(boot: BootstrapSupport, i: int, j: int, diff: float) -> float:
    """Bootstrap support of the point-estimate order of solvers i and j.

    ``diff`` = score_i − score_j of the point estimate. Returns P*(score_i > score_j) when
    diff > 0, P*(score_j > score_i) when diff < 0 (ties count ½), and 0.5 when diff = 0.
    """
    if diff > 0:
        return float(boot.p_greater[i, j])
    if diff < 0:
        return float(boot.p_greater[j, i])
    return 0.5


def _mean_terms(X: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([math.fsum(col) for col in X.T]) / X.shape[0]


def leave_one_group_out(
    terms_a: ArrayLike,
    terms_b: ArrayLike,
    groups: Sequence[str],
    i: int,
    j: int,
    *,
    n_boot: int,
    seed: int,
    confirm: float,
) -> dict[str, dict[str, Any]]:
    """Drop each group (problem) in turn and re-test the rank change of solvers i and j.

    For every group g, the instances of g are removed from both settings. The point estimates
    are the column means of the remaining terms. The supports come from
    ``bootstrap_order_support`` with the same seed in both settings (paired replicates).
    ``kept`` is true when the point estimates still flip and the support of each setting's
    order is ≥ ``confirm``.
    """
    A = np.asarray(terms_a, dtype=np.float64)
    B = np.asarray(terms_b, dtype=np.float64)
    if A.shape != B.shape or A.ndim != 2 or A.shape[0] != len(groups):
        raise ValueError("terms_a and terms_b must be (instances, solvers) with one group each")
    g_arr = np.asarray(groups)
    out: dict[str, dict[str, Any]] = {}
    for g in sorted(set(groups)):
        keep = g_arr != g
        if not keep.any():
            continue
        sub = [str(x) for x in g_arr[keep]]
        sa, sb = _mean_terms(A[keep]), _mean_terms(B[keep])
        da, db = float(sa[i] - sa[j]), float(sb[i] - sb[j])
        ba = bootstrap_order_support(A[keep], sub, n_boot=n_boot, seed=seed)
        bb = bootstrap_order_support(B[keep], sub, n_boot=n_boot, seed=seed)
        supp_a, supp_b = order_support(ba, i, j, da), order_support(bb, i, j, db)
        flip = da * db < 0
        out[g] = {
            "diff_a": da,
            "diff_b": db,
            "support_a": supp_a,
            "support_b": supp_b,
            "flip": bool(flip),
            "kept": bool(flip and min(supp_a, supp_b) >= confirm),
        }
    return out
