"""The symmetric Euclidean traveling-salesman problem (TSP).

Given n cities with coordinates cᵢ ∈ ℝ², find a tour (a cyclic permutation π of 0, …, n−1)
of minimum length

    L(π) = Σ_{k=0}^{n−1} d(π_k, π_{k+1 mod n}),     d(i, j) = ‖cᵢ − cⱼ‖₂.

Methods take a :class:`~numopt.problems.combinatorial.TspInstance` (or an n × 2 array of
coordinates) and return a :class:`Result` with ``x`` = the tour (list of n city indices; the
closing edge back to ``x[0]`` is implicit) and ``fun`` = its length. Instances whose
distances overflow, or underflow (a squared coordinate difference is subnormal while the
largest distance is below 2⁻⁴⁸⁵ ≈ 1e−146), raise ``ValueError``.

Extra keys (``Result.extra``):
    all methods, when the instance knows its optimum L*:
        optimum_length: float — L*.  gap: float — (L − L*)/L* (0 when L* = 0).
    tsp_nearest_neighbor:
        distance_evaluations: int — n(n−1)/2, the distances looked up by the n−1 scans of the
        unvisited cities.
    tsp_two_opt, tsp_or_opt:
        initial_length: float — length of the starting tour.
    tsp_simulated_annealing:
        final_temperature: float | None — T after the last proposal (None when the method
        returns at once: n ≤ 3 or all cities coincide).
    tsp_genetic:
        no method-specific keys.
    tsp_ant_colony:
        tau0: float — the initial pheromone τ₀ = m/C^nn (1 when C^nn = 0).
        nn_fallback_steps: int — construction steps that had τ_ij = 0 on every unvisited edge
        and moved to the nearest city instead (see the function docstring).
    tsp_held_karp:
        states: int — DP states C(S, j) computed, Σ_{s=1}^{n−1} s·C(n−1, s) (1 when n = 1).
        transitions: int — candidate predecessors examined, Σ_{s≥2} (s−1)·states_s plus the
        n−1 closing transitions (0 when n = 1).

Numerical conventions (shared with the TypeScript port):

* d(i, j) = sqrt(dx·dx + dy·dy), elementwise IEEE arithmetic (correctly rounded, so integer
  coordinates give bit-identical distances everywhere).
* Tour lengths are summed sequentially in tour order, k = 0, …, n−1, closing edge last.
* 2-opt deltas are evaluated as (d(a,c) + d(b,d)) − (d(a,b) + d(c,d)).
* A move "improves" only when its delta < −tol with tol = 1e−10 · max d(i, j); nearest-city
  ties within the same tol go to the smaller index.

  # NOTE: the tolerance is a deviation from the textbook "delta < 0". On instances with exact
  # ties (a circle, a grid) rounding noise of order 1e−14 would otherwise decide moves and
  # tie-breaks, so local search could cycle on zero-gain moves and the two ports could diverge.

Evaluation counts: ``n_fev`` counts candidate evaluations — one per 2-opt / Or-opt move delta,
SA proposal, GA child, or ant tour. Constructive and exact DP methods report 0 and give their
work counts in ``extra``.

Random numbers (SA, GA, ACO) come only from :class:`numopt.core.rng.Rng` (Mulberry32) in the
draw order each docstring lists, so the TypeScript port replays them exactly.

Info keys:
    all methods:
        tour: [int] — the tour (or partial path) shown at this step.
        length: float — its length (open-path length for partial paths).
    tsp_nearest_neighbor:
        current: int — the city just appended.  closed: bool — True on the final step that
        adds the closing edge.
    tsp_two_opt, tsp_or_opt, tsp_simulated_annealing:
        best_length: float — best tour length so far.
        move: dict | None — the move made (or proposed) at this step. 2-opt: {i, j, removed:
        [[a, b], [c, d]], added: [[a, c], [b, d]], delta}; the segment between positions i+1
        and j is reversed. Or-opt: {segment: [cities], removed: [[p, s0], [sL, q], [x, y]],
        added: [[p, q], [x, ·], [·, y]], reversed: bool, delta}, where an edge that is both
        removed and added (segment reinserted next to p or q) is left out of both lists, so
        each lists the 2 or 3 edges that really change. SA adds ``accepted: bool``.
    tsp_simulated_annealing:
        best_tour: [int] — the best tour so far.  temperature: float — T used at this step.
        acceptance_rate: float — accepted / proposed since the previous recorded step.
    tsp_genetic:
        tour: best individual of the population; length: its length.
        best_length: float — best length so far; elitism makes it equal to length exactly.
        mean_length: float, worst_length: float — population statistics.
        stall: int — generations since the best length last improved by more than tol.
    tsp_ant_colony:
        tour: best ant tour of this iteration; length: its length. At k = 0 these are the
        nearest-neighbor tour from city 0 and C^nn, which set τ₀.
        best_tour: [int], best_length: float — best ant tour so far ([] and None at k = 0).
        mean_length: float — mean ant tour length in this iteration (None at k = 0).
        pheromone: [[float]] | None — the n × n pheromone matrix τ after the update (None when
        n > 40, to keep traces small).  tau_min: float, tau_max: float — off-diagonal range of τ.
        stall: int — iterations since the best length last improved.
    tsp_held_karp:
        subset_size: int — |S|, the number of cities besides city 0 in the DP layer.
        states: int — number of states C(S, j) computed in this layer.
        tour: the cheapest path 0 → … → j over all states of the layer (the optimal tour on
        the closing step); length: its cost.  closed: bool — True on the closing step.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.registry import ParamSpec, register
from ..core.rng import Rng
from ..core.types import Result, Step
from ..problems.combinatorial import TspInstance

Matrix = NDArray[np.float64]

#: Relative tolerance for "improving" moves and nearest-city ties (see module docstring).
# NOTE: textbook local search accepts any delta < 0; we require delta < −REL_TOL·max d so that
# rounding noise on tied instances (circle, grid) cannot drive zero-gain moves or tie-breaks.
REL_TOL = 1e-10
#: Held–Karp needs O(2ⁿ·n) memory and O(2ⁿ·n²) time; larger n is refused. At n = 16 the
#: tables have 2¹⁵·15 entries (two 3.9 MB arrays) and the DP takes about 0.3 s in CPython.
HELD_KARP_MAX_N = 16
#: Largest n for which the ant-colony trace carries the full pheromone matrix.
PHEROMONE_MAX_N = 40
#: Smallest admissible largest distance, 2⁻⁴⁸⁵ ≈ 1.0e−146: below it gradual underflow in
#: dx·dx + dy·dy can perturb distances by more than eps·max d (see :func:`_distances`).
MIN_DISTANCE_SCALE = math.ldexp(1.0, -485)
#: Smallest positive normal double, 2⁻¹⁰²².
_TINY = float(np.finfo(np.float64).tiny)
#: UI range of the nearest-neighbor start city: the largest library instance
#: (tsp_cities_20) has 20 cities. Larger custom instances still accept any valid index.
NN_START_UI_MAX = 19

INIT_PARAM = ParamSpec(
    "init",
    "identity",
    kind="choice",
    choices=("identity", "nearest_neighbor"),
    help="Starting tour: cities in index order, or the nearest-neighbor tour from city 0.",
)


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


def _instance(problem: TspInstance | ArrayLike) -> TspInstance:
    """Accept a TspInstance or an (n, 2) coordinate array and validate the coordinates."""
    if isinstance(problem, TspInstance):
        inst = problem
    else:
        arr = np.asarray(problem, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"coordinates must have shape (n, 2), got {arr.shape}")
        inst = TspInstance("custom", "custom", tuple((float(a), float(b)) for a, b in arr))
    if inst.n == 0:
        raise ValueError(f"{inst.id}: the instance has no cities")
    coords = np.asarray(inst.coords, dtype=np.float64)
    if coords.shape != (inst.n, 2):
        raise ValueError(f"{inst.id}: coordinates must have shape (n, 2), got {coords.shape}")
    if not np.all(np.isfinite(coords)):
        raise ValueError(f"{inst.id}: coordinates must be finite")
    return inst


def distance_matrix(coords: ArrayLike) -> Matrix:
    """d(i, j) = sqrt(dx·dx + dy·dy) for every pair of cities (n × n, symmetric, zero diagonal)."""
    c = np.asarray(coords, dtype=np.float64)  # (n, 2)
    dx = c[:, 0][:, None] - c[:, 0][None, :]  # (n, n)
    dy = c[:, 1][:, None] - c[:, 1][None, :]  # (n, n)
    return np.sqrt(dx * dx + dy * dy)


def _distances(inst: TspInstance) -> Matrix:
    """The distance matrix of a validated instance; ``ValueError`` if it over- or underflows.

    Overflow: finite coordinates do not give finite distances. |dx| > ~1.3e154 makes dx·dx
    overflow to inf. Every comparison against tol = 1e−10·max d would then be NaN, so the
    methods would return non-permutations as "converged".

    Underflow: a nonzero square dx·dx (or dy·dy) below the smallest normal double 2⁻¹⁰²²
    (|dx| ≲ 1.5e−154) is rounded to a subnormal or to 0, with an absolute error of at most
    2⁻¹⁰⁷⁵. The sum of two such squares is exact, so the computed d(i, j) has an absolute
    error of at most sqrt(2⁻¹⁰⁷⁴) = 2⁻⁵³⁷ ≈ 2.2e−162 on top of the usual relative rounding
    (Higham 2002, §2.1, gradual underflow). That error is at most eps·max d — the size of an
    ordinary rounding of the largest distance — when max d ≥ 2⁻⁵³⁷/2⁻⁵² = 2⁻⁴⁸⁵ ≈ 1.0e−146.
    So an instance is refused when some square underflowed AND max d < 2⁻⁴⁸⁵. Below that
    scale distinct cities can get d = 0, and Held–Karp would certify a wrong tour.

    Both kinds of instance are refused, not solved. Squares that do not underflow carry no
    extra error, so cities that coincide exactly (dx = dy = 0) are always valid.
    """
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        D = distance_matrix(inst.coords)
    if not np.all(np.isfinite(D)):
        raise ValueError(
            f"{inst.id}: coordinate differences overflow the distance computation "
            "(|dx|, |dy| must stay below ~1e154); rescale the instance"
        )
    d_max = float(D.max())
    if d_max < MIN_DISTANCE_SCALE:
        c = np.asarray(inst.coords, dtype=np.float64)  # (n, 2)
        diff = c[:, None, :] - c[None, :, :]  # (n, n, 2): dx, dy
        with np.errstate(under="ignore"):
            underflowed = (diff != 0.0) & (diff * diff < _TINY)  # (n, n, 2)
        if np.any(underflowed):
            raise ValueError(
                f"{inst.id}: inter-city distances underflow the distance computation "
                f"(largest distance {d_max:.3g} < 2^-485 ≈ {MIN_DISTANCE_SCALE:.3g}); "
                "rescale the instance"
            )
    return D


def tour_length(D: Matrix, tour: list[int]) -> float:
    """Closed-tour length, summed sequentially in tour order (closing edge last)."""
    n = len(tour)
    total = 0.0
    for k in range(n - 1):
        total += float(D[tour[k], tour[k + 1]])
    if n > 1:
        total += float(D[tour[n - 1], tour[0]])
    return total


def _path_length(D: Matrix, path: list[int]) -> float:
    total = 0.0
    for k in range(len(path) - 1):
        total += float(D[path[k], path[k + 1]])
    return total


def _tolerance(D: Matrix) -> float:
    return REL_TOL * float(D.max()) if D.size else 0.0


def _nearest_neighbor_tour(D: Matrix, start: int, tol: float) -> list[int]:
    """Nearest-neighbor tour (ties within ``tol`` go to the smaller index)."""
    n = D.shape[0]
    visited = [False] * n
    visited[start] = True
    tour = [start]
    for _ in range(n - 1):
        cur = tour[-1]
        best_j, best_d = -1, math.inf
        for j in range(n):
            if not visited[j] and float(D[cur, j]) < best_d - tol:
                best_j, best_d = j, float(D[cur, j])
        visited[best_j] = True
        tour.append(best_j)
    return tour


def _initial_tour(D: Matrix, init: str, tol: float) -> list[int]:
    if init == "identity":
        return list(range(D.shape[0]))
    if init == "nearest_neighbor":
        return _nearest_neighbor_tour(D, 0, tol)
    raise ValueError(f"unknown init {init!r}; expected 'identity' or 'nearest_neighbor'")


def _check_int(name: str, value: int, low: int) -> int:
    if int(value) != value or int(value) < low:
        raise ValueError(f"{name} must be an integer ≥ {low}, got {value!r}")
    return int(value)


def _check_unit(
    name: str, value: float, *, closed_low: bool = True, closed_high: bool = True
) -> float:
    v = float(value)
    ok_low = v >= 0.0 if closed_low else v > 0.0
    ok_high = v <= 1.0 if closed_high else v < 1.0
    if not (math.isfinite(v) and ok_low and ok_high):
        raise ValueError(f"{name} must lie in the unit interval, got {value!r}")
    return v


def _two_opt_delta(D: Matrix, t: list[int], i: int, j: int) -> float:
    """Length change when edges (t[i], t[i+1]) and (t[j], t[j+1]) become (t[i], t[j]), (t[i+1], t[j+1])."""
    n = len(t)
    a, b, c, d = t[i], t[i + 1], t[j], t[(j + 1) % n]
    return (float(D[a, c]) + float(D[b, d])) - (float(D[a, b]) + float(D[c, d]))


def _two_opt_move(t: list[int], i: int, j: int, delta: float) -> dict[str, Any]:
    n = len(t)
    a, b, c, d = t[i], t[i + 1], t[j], t[(j + 1) % n]
    return {"i": i, "j": j, "removed": [[a, b], [c, d]], "added": [[a, c], [b, d]], "delta": delta}


def _net_exchange(
    removed: list[list[int]], added: list[list[int]]
) -> tuple[list[list[int]], list[list[int]]]:
    """Drop every edge (as an unordered pair) that occurs in both lists; keep the order."""
    common = {frozenset(e) for e in removed} & {frozenset(e) for e in added}
    return (
        [e for e in removed if frozenset(e) not in common],
        [e for e in added if frozenset(e) not in common],
    )


def _reverse(t: list[int], i: int, j: int) -> None:
    """Reverse t[i+1..j] in place (the 2-opt move for the edge pair at positions i, j)."""
    t[i + 1 : j + 1] = t[i + 1 : j + 1][::-1]


def _finish(
    method: str,
    inst: TspInstance,
    tour: list[int],
    length: float,
    converged: bool,
    message: str,
    n_iter: int,
    n_fev: int,
    trace: list[Step],
    **extra: Any,
) -> Result:
    info: dict[str, Any] = dict(extra)
    if inst.optimum_length is not None:
        info["optimum_length"] = inst.optimum_length
        info["gap"] = (
            (length - inst.optimum_length) / inst.optimum_length if inst.optimum_length > 0 else 0.0
        )
    return Result(
        method, list(tour), length, converged, message, n_iter, n_fev, trace=trace, extra=info
    )


def _trivial(
    method: str,
    inst: TspInstance,
    D: Matrix,
    tour: list[int],
    info: dict[str, Any],
    reason: str = "",
    **extra: Any,
) -> Result:
    """Every tour has the same length (n ≤ 3, or all cities coincide): return the start tour."""
    length = tour_length(D, tour)
    step = Step(0, list(tour), length, info={"tour": list(tour), "length": length, **info})
    message = reason or f"n = {inst.n} ≤ 3: every tour has the same length"
    return _finish(method, inst, tour, length, True, message, 0, 0, [step], **extra)


def _record(k: int, record_every: int) -> bool:
    return k % record_every == 0


# --------------------------------------------------------------------------------------
# Nearest neighbor
# --------------------------------------------------------------------------------------


@register(
    id="tsp_nearest_neighbor",
    family="combinatorial",
    name="Nearest neighbor",
    params=(
        ParamSpec(
            "start",
            0,
            kind="int",
            min=0,
            max=NN_START_UI_MAX,
            help="City where the tour starts (an index 0, …, n−1).",
        ),
    ),
    needs=("tsp",),
    order="heuristic, O(n²); length ≤ ½(⌈log₂ n⌉ + 1)·L⋆ on metric instances",
    summary="From the current city always travel to the closest city not yet visited.",
    references=(
        "Rosenkrantz, Stearns & Lewis (1977), SIAM J. Comput. 6(3), 563–581",
        "Johnson & McGeoch (1997), The TSP: a case study in local optimization",
    ),
)
def tsp_nearest_neighbor(problem: TspInstance | ArrayLike, *, start: int = 0) -> Result:
    """Nearest-neighbor construction (Rosenkrantz, Stearns & Lewis 1977).

    Start at city ``start``; repeatedly append the closest unvisited city (ties within tol go
    to the smaller index); finally close the tour. On metric instances the tour is at most
    ½(⌈log₂ n⌉ + 1) times the optimum (Rosenkrantz et al. 1977).

    Trace: k = 0 is the start city; step k = 1, …, n−1 appends one city (``fun`` = open path
    length); step k = n adds the closing edge (``fun`` = tour length). Stopping test: every
    city is visited. A heuristic: ``converged`` means the tour is complete, not optimal.
    """
    inst = _instance(problem)
    n = inst.n
    start = _check_int("start", start, 0)
    if start >= n:
        raise ValueError(f"start must be a city index in [0, {n - 1}], got {start}")
    D = _distances(inst)
    tol = _tolerance(D)

    visited = [False] * n
    visited[start] = True
    tour = [start]
    length = 0.0
    trace = [
        Step(
            0,
            [start],
            0.0,
            info={"tour": [start], "length": 0.0, "current": start, "closed": n == 1},
        )
    ]
    for k in range(1, n):
        cur = tour[-1]
        best_j, best_d = -1, math.inf
        for j in range(n):
            if not visited[j] and float(D[cur, j]) < best_d - tol:
                best_j, best_d = j, float(D[cur, j])
        visited[best_j] = True
        tour.append(best_j)
        length += best_d
        trace.append(
            Step(
                k,
                list(tour),
                length,
                info={"tour": list(tour), "length": length, "current": best_j, "closed": False},
            )
        )
    n_iter = 0
    if n > 1:
        length = tour_length(D, tour)
        n_iter = n
        trace.append(
            Step(
                n,
                list(tour),
                length,
                info={"tour": list(tour), "length": length, "current": start, "closed": True},
            )
        )
    return _finish(
        "tsp_nearest_neighbor",
        inst,
        tour,
        length,
        True,
        f"tour complete from city {start} (heuristic: no optimality guarantee)",
        n_iter,
        0,
        trace,
        distance_evaluations=n * (n - 1) // 2,
    )


# --------------------------------------------------------------------------------------
# 2-opt and Or-opt local search
# --------------------------------------------------------------------------------------


@register(
    id="tsp_two_opt",
    family="combinatorial",
    name="2-opt local search",
    params=(
        ParamSpec(
            "strategy",
            "first",
            kind="choice",
            choices=("first", "best"),
            help="Apply the first improving move found, or the best move of the full neighborhood.",
        ),
        INIT_PARAM,
        ParamSpec(
            "max_iter",
            300,
            kind="int",
            min=1,
            max=100_000,
            help="Maximum number of improving moves.",
        ),
        ParamSpec(
            "record_every",
            1,
            kind="int",
            min=1,
            max=10_000,
            help="Record every m-th move in the trace.",
        ),
    ),
    needs=("tsp",),
    order="local search; stops at a 2-optimal tour",
    summary="Remove two edges and reconnect the tour the other way whenever that shortens it.",
    references=(
        "Croes (1958), Oper. Res. 6(6), 791–812",
        "Johnson & McGeoch (1997), The TSP: a case study in local optimization",
    ),
)
def tsp_two_opt(
    problem: TspInstance | ArrayLike,
    *,
    strategy: str = "first",
    init: str = "identity",
    max_iter: int = 300,
    record_every: int = 1,
) -> Result:
    """2-opt local search (Croes 1958).

    A 2-opt move at tour positions i < j removes edges (t_i, t_{i+1}) and (t_j, t_{j+1}) and
    reconnects with (t_i, t_j), (t_{i+1}, t_{j+1}) by reversing t_{i+1..j}. Its gain is
    Δ = (d(t_i, t_j) + d(t_{i+1}, t_{j+1})) − (d(t_i, t_{i+1}) + d(t_j, t_{j+1})). The
    neighborhood is every pair 0 ≤ i < j ≤ n−1 with j ≥ i + 2, except (0, n−1) (those edges
    share city t_0), scanned with i outer and j inner, ascending: n(n−3)/2 moves.

    ``strategy="first"``: apply the first move with Δ < −tol in scan order, then rescan from
    the start. ``strategy="best"``: scan everything and apply the move with the smallest Δ
    (first in scan order on ties) if Δ < −tol.

    Stopping test: a full scan finds no move with Δ < −tol — the tour is 2-optimal
    (``converged=True``). Reaching ``max_iter`` moves first gives ``converged=False``.
    One iteration = one applied move; the trace records every ``record_every``-th move.
    """
    if strategy not in ("first", "best"):
        raise ValueError(f"unknown strategy {strategy!r}; expected 'first' or 'best'")
    inst = _instance(problem)
    max_iter = _check_int("max_iter", max_iter, 1)
    record_every = _check_int("record_every", record_every, 1)
    D = _distances(inst)
    tol = _tolerance(D)
    t = _initial_tour(D, init, tol)
    n = len(t)
    if n <= 3:
        length = tour_length(D, t)
        return _trivial(
            "tsp_two_opt",
            inst,
            D,
            t,
            {"best_length": length, "move": None},
            initial_length=length,
        )

    length = tour_length(D, t)
    initial_length = length
    trace = [
        Step(
            0,
            list(t),
            length,
            info={"tour": list(t), "length": length, "best_length": length, "move": None},
        )
    ]
    n_fev = 0
    k = 0
    converged = False
    last_move: dict[str, Any] | None = None
    while True:
        best: tuple[float, int, int] | None = None
        for i in range(n - 2):
            for j in range(i + 2, n if i > 0 else n - 1):
                delta = _two_opt_delta(D, t, i, j)
                n_fev += 1
                if delta < -tol and (best is None or delta < best[0]):
                    best = (delta, i, j)
                    if strategy == "first":
                        break
            if best is not None and strategy == "first":
                break
        if best is None:
            converged = True
            break
        if k == max_iter:
            break
        delta, i, j = best
        move = _two_opt_move(t, i, j, delta)
        _reverse(t, i, j)
        length = tour_length(D, t)
        k += 1
        if _record(k, record_every):
            trace.append(
                Step(
                    k,
                    list(t),
                    length,
                    info={"tour": list(t), "length": length, "best_length": length, "move": move},
                )
            )
        last_move = move
    if trace[-1].k != k:
        trace.append(
            Step(
                k,
                list(t),
                length,
                info={"tour": list(t), "length": length, "best_length": length, "move": last_move},
            )
        )
    msg = (
        f"2-optimal: no 2-opt move shortens the tour (after {k} moves)"
        if converged
        else f"reached max_iter={max_iter} improving moves before a 2-optimal tour"
    )
    return _finish(
        "tsp_two_opt",
        inst,
        t,
        length,
        converged,
        msg,
        k,
        n_fev,
        trace,
        initial_length=initial_length,
    )


def _or_opt_moves(n: int) -> list[tuple[int, int, int]]:
    """Or-opt neighborhood in scan order: (segment length L, start position i, insertion m)."""
    moves = []
    for L in (3, 2, 1):
        if L > n - 3:
            continue
        for i in range(n):
            for m in range(n - L - 1):
                moves.append((L, i, m))
    return moves


@register(
    id="tsp_or_opt",
    family="combinatorial",
    name="Or-opt local search",
    params=(
        INIT_PARAM,
        ParamSpec(
            "max_iter",
            300,
            kind="int",
            min=1,
            max=100_000,
            help="Maximum number of improving moves.",
        ),
        ParamSpec(
            "record_every",
            1,
            kind="int",
            min=1,
            max=10_000,
            help="Record every m-th move in the trace.",
        ),
    ),
    needs=("tsp",),
    order="local search; stops at an Or-optimal tour",
    summary="Move a chain of 1–3 consecutive cities to a better place in the tour (possibly reversed).",
    references=(
        "Or (1976), PhD thesis, Northwestern University",
        "Johnson & McGeoch (1997), The TSP: a case study in local optimization",
    ),
)
def tsp_or_opt(
    problem: TspInstance | ArrayLike,
    *,
    init: str = "identity",
    max_iter: int = 300,
    record_every: int = 1,
) -> Result:
    """Or-opt local search (Or 1976), first improvement.

    A move takes the segment s = (t_i, …, t_{i+L−1}) (positions mod n, L ∈ {3, 2, 1} as in Or's
    original order) with neighbors p = t_{i−1}, q = t_{i+L}, removes it, and reinserts it
    between consecutive cities x, y of the remaining cycle r = (q, …, p), forward or reversed:

        Δ = [d(x, s₀) + d(s_L, y) − d(x, y)] − [d(p, s₀) + d(s_L, q) − d(p, q)]   (forward)
        Δ = [d(x, s_L) + d(s₀, y) − d(x, y)] − [d(p, s₀) + d(s_L, q) − d(p, q)]   (reversed)

    where s_L is the last city of s. Insertion edges are (r_m, r_{m+1}) for m = 0, …, n−L−2,
    which excludes the edge (p, q) itself. Moves require L ≤ n − 3. Scan order: L, then i, then
    m, then forward before reversed (L = 1 has one orientation only, evaluated once). The first
    move with Δ < −tol is applied (the new tour is r₀..r_m, segment, r_{m+1}..), then the scan
    restarts.

    A move with m = 0 or m = n−L−2 (the segment reinserted next to q or p) can remove and add
    the same edge; ``info["move"]`` reports the net exchange, so it lists 2 or 3 edges each.

    Stopping test: a full scan finds no improving move — the tour is Or-optimal
    (``converged=True``). Reaching ``max_iter`` moves first gives ``converged=False``.
    """
    inst = _instance(problem)
    max_iter = _check_int("max_iter", max_iter, 1)
    record_every = _check_int("record_every", record_every, 1)
    D = _distances(inst)
    tol = _tolerance(D)
    t = _initial_tour(D, init, tol)
    n = len(t)
    if n <= 3:
        length = tour_length(D, t)
        return _trivial(
            "tsp_or_opt",
            inst,
            D,
            t,
            {"best_length": length, "move": None},
            initial_length=length,
        )

    def dd(a: int, b: int) -> float:
        return float(D[a, b])

    length = tour_length(D, t)
    initial_length = length
    trace = [
        Step(
            0,
            list(t),
            length,
            info={"tour": list(t), "length": length, "best_length": length, "move": None},
        )
    ]
    moves = _or_opt_moves(n)
    n_fev = 0
    k = 0
    converged = False
    last_move: dict[str, Any] | None = None
    while True:
        found: tuple[float, int, int, int, bool] | None = None
        for L, i, m in moves:
            seg = [t[(i + s) % n] for s in range(L)]
            p, q = t[(i - 1) % n], t[(i + L) % n]
            r = [t[(i + L + s) % n] for s in range(n - L)]  # remaining cycle q, …, p
            x, y = r[m], r[m + 1]
            removal_gain = dd(p, seg[0]) + dd(seg[-1], q) - dd(p, q)
            # A one-city segment reversed is the same segment: evaluate it once.
            for rev in (False,) if L == 1 else (False, True):
                first, last = (seg[-1], seg[0]) if rev else (seg[0], seg[-1])
                delta = (dd(x, first) + dd(last, y) - dd(x, y)) - removal_gain
                n_fev += 1
                if delta < -tol:
                    found = (delta, L, i, m, rev)
                    break
            if found is not None:
                break
        if found is None:
            converged = True
            break
        if k == max_iter:
            break
        delta, L, i, m, rev = found
        seg = [t[(i + s) % n] for s in range(L)]
        p, q = t[(i - 1) % n], t[(i + L) % n]
        r = [t[(i + L + s) % n] for s in range(n - L)]
        x, y = r[m], r[m + 1]
        placed = seg[::-1] if rev else seg
        t = [*r[: m + 1], *placed, *r[m + 1 :]]
        length = tour_length(D, t)
        k += 1
        removed, added = _net_exchange(
            [[p, seg[0]], [seg[-1], q], [x, y]], [[p, q], [x, placed[0]], [placed[-1], y]]
        )
        last_move = {
            "segment": seg,
            "removed": removed,
            "added": added,
            "reversed": rev,
            "delta": delta,
        }
        if _record(k, record_every):
            trace.append(
                Step(
                    k,
                    list(t),
                    length,
                    info={
                        "tour": list(t),
                        "length": length,
                        "best_length": length,
                        "move": last_move,
                    },
                )
            )
    if trace[-1].k != k:
        trace.append(
            Step(
                k,
                list(t),
                length,
                info={"tour": list(t), "length": length, "best_length": length, "move": last_move},
            )
        )
    msg = (
        f"Or-optimal: no segment move shortens the tour (after {k} moves)"
        if converged
        else f"reached max_iter={max_iter} improving moves before an Or-optimal tour"
    )
    return _finish(
        "tsp_or_opt",
        inst,
        t,
        length,
        converged,
        msg,
        k,
        n_fev,
        trace,
        initial_length=initial_length,
    )


# --------------------------------------------------------------------------------------
# Simulated annealing
# --------------------------------------------------------------------------------------


def _mean_distance(D: Matrix) -> float:
    """Mean of d(i, j) over i < j, summed sequentially in row-major order."""
    n = D.shape[0]
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += float(D[i, j])
    return total / (n * (n - 1) // 2)


@register(
    id="tsp_simulated_annealing",
    family="combinatorial",
    name="Simulated annealing (2-opt moves)",
    params=(
        ParamSpec(
            "t0",
            1.0,
            min=1e-2,
            max=100.0,
            log=True,
            help="Initial temperature, in units of the mean inter-city distance (must exceed "
            "t_min).",
        ),
        ParamSpec(
            "alpha",
            0.9995,
            min=0.9,
            max=0.99999,
            help="Geometric cooling factor: T ← αT after every proposal.",
        ),
        ParamSpec(
            "t_min",
            1e-3,
            min=1e-8,
            max=5e-3,  # < t0.min, so no UI setting gives the invalid t_min ≥ t0
            log=True,
            help="Freezing temperature, in units of the mean inter-city distance (must be below "
            "t0).",
        ),
        INIT_PARAM,
        ParamSpec(
            "max_iter",
            20_000,
            kind="int",
            min=1,
            max=1_000_000,
            help="Maximum number of proposals.",
        ),
        ParamSpec(
            "record_every",
            100,
            kind="int",
            min=1,
            max=100_000,
            help="Record every m-th proposal in the trace.",
        ),
    ),
    needs=("tsp",),
    order="stochastic metaheuristic",
    summary="Propose random 2-opt moves; always accept improvements and accept worse tours with "
    "probability exp(−Δ/T) while the temperature T slowly falls.",
    references=(
        "Kirkpatrick, Gelatt & Vecchi (1983), Science 220(4598), 671–680",
        "Černý (1985), J. Optim. Theory Appl. 45(1), 41–51",
        "Aarts & Korst (1989), Simulated Annealing and Boltzmann Machines",
    ),
    deterministic=False,
)
def tsp_simulated_annealing(
    problem: TspInstance | ArrayLike,
    *,
    seed: int = 0,
    t0: float = 1.0,
    alpha: float = 0.9995,
    t_min: float = 1e-3,
    init: str = "identity",
    max_iter: int = 20_000,
    record_every: int = 100,
) -> Result:
    """Simulated annealing with 2-opt neighborhood and geometric cooling.

    With d̄ the mean inter-city distance, T starts at T₀ = t0·d̄. Proposal k (k = 1, 2, …):

    1. i = rng.integers(n), then o = 2 + rng.integers(n − 3), j = (i + o) mod n; reorder so
       i < j. The two removed edges are therefore never adjacent and every non-adjacent edge
       pair has the same probability 2/(n(n−3)).
    2. Δ = 2-opt delta of reversing t_{i+1..j} (see :func:`tsp_two_opt`).
    3. Metropolis rule (Kirkpatrick et al. 1983): accept when Δ ≤ 0; otherwise draw
       u = rng.random() and accept when u < exp(−Δ/T). (u is drawn only when Δ > 0.)
    4. T ← α·T.

    The best tour seen (strictly shorter by more than tol) is kept and returned.

    Stopping test: T < T_min = t_min·d̄ — the system is frozen (``converged=True``; a heuristic
    stop, not an optimality certificate). Reaching ``max_iter`` proposals first gives
    ``converged=False``. Requires 0 < t_min < t0 (otherwise the schedule is empty and the
    "frozen" stop would be vacuous; ValueError). Requires n ≥ 4 for a proper 2-opt move;
    n ≤ 3 returns at once.
    """
    inst = _instance(problem)
    max_iter = _check_int("max_iter", max_iter, 1)
    record_every = _check_int("record_every", record_every, 1)
    if not (math.isfinite(t0) and t0 > 0.0 and math.isfinite(t_min) and t_min > 0.0):
        raise ValueError("t0 and t_min must be positive")
    if t_min >= t0:
        raise ValueError(
            f"t_min = {t_min:g} must be below t0 = {t0:g}: the annealing schedule would be empty"
        )
    _check_unit("alpha", alpha, closed_low=False, closed_high=False)
    D = _distances(inst)
    tol = _tolerance(D)
    t = _initial_tour(D, init, tol)
    n = len(t)
    info = {
        "best_length": tour_length(D, t),
        "best_tour": list(t),
        "temperature": None,
        "acceptance_rate": None,
        "move": None,
    }
    if n <= 3:
        return _trivial("tsp_simulated_annealing", inst, D, t, info, final_temperature=None)
    d_bar = _mean_distance(D)
    if d_bar == 0.0:
        reason = "all cities coincide: every tour has length 0"
        return _trivial("tsp_simulated_annealing", inst, D, t, info, reason, final_temperature=None)

    rng = Rng(seed)
    T, T_min = t0 * d_bar, t_min * d_bar
    length = tour_length(D, t)
    best_t, best_len = list(t), length

    def step(k: int, temp: float | None, rate: float | None, move: dict[str, Any] | None) -> Step:
        return Step(
            k,
            list(t),
            length,
            info={
                "tour": list(t),
                "length": length,
                "best_length": best_len,
                "best_tour": list(best_t),
                "temperature": temp,
                "acceptance_rate": rate,
                "move": move,
            },
        )

    trace = [step(0, T, None, None)]
    accepted_window = proposed_window = 0
    converged = False
    k = 0
    move: dict[str, Any] | None = None
    T_used = T
    while k < max_iter:
        k += 1
        i = rng.integers(n)
        j = (i + 2 + rng.integers(n - 3)) % n
        i, j = min(i, j), max(i, j)
        delta = _two_opt_delta(D, t, i, j)
        accept = delta <= 0.0 or rng.random() < math.exp(-delta / T)
        move = {**_two_opt_move(t, i, j, delta), "accepted": accept}
        proposed_window += 1
        if accept:
            accepted_window += 1
            _reverse(t, i, j)
            length = tour_length(D, t)
            if length < best_len - tol:
                best_t, best_len = list(t), length
        T_used = T
        T = alpha * T
        frozen = T < T_min
        if _record(k, record_every) or frozen or k == max_iter:
            trace.append(step(k, T_used, accepted_window / proposed_window, move))
            accepted_window = proposed_window = 0
        if frozen:
            converged = True
            break
    msg = (
        f"frozen: T = {T:.6g} < t_min·d̄ = {T_min:.6g} after {k} proposals (heuristic stop)"
        if converged
        else f"reached max_iter={max_iter} proposals before freezing (T = {T:.6g})"
    )
    return _finish(
        "tsp_simulated_annealing",
        inst,
        best_t,
        best_len,
        converged,
        msg,
        k,
        k,
        trace,
        final_temperature=T,
    )


# --------------------------------------------------------------------------------------
# Genetic algorithm
# --------------------------------------------------------------------------------------


def order_crossover(p1: list[int], p2: list[int], a: int, b: int) -> list[int]:
    """Davis' order crossover OX with cut points a ≤ b (inclusive).

    The child copies p1[a..b]; the other positions, starting at b + 1 and wrapping, get the
    cities of p2 in the order they appear in p2 starting at position b + 1 (wrapping), skipping
    cities already copied (Davis 1985; called OX1 in Larrañaga et al. 1999).
    """
    n = len(p1)
    child = [-1] * n
    used = [False] * n
    for pos in range(a, b + 1):
        child[pos] = p1[pos]
        used[p1[pos]] = True
    write = (b + 1) % n
    for s in range(n):
        city = p2[(b + 1 + s) % n]
        if not used[city]:
            child[write] = city
            used[city] = True
            write = (write + 1) % n
    return child


@register(
    id="tsp_genetic",
    family="combinatorial",
    name="Genetic algorithm (OX crossover)",
    params=(
        ParamSpec(
            "pop_size", 40, kind="int", min=4, max=1000, help="Number of tours in the population."
        ),
        ParamSpec(
            "crossover_rate",
            0.9,
            min=0.0,
            max=1.0,
            help="Probability that a child is made by order crossover.",
        ),
        ParamSpec(
            "mutation_rate", 0.2, min=0.0, max=1.0, help="Probability of a swap mutation per child."
        ),
        ParamSpec(
            "tournament_size",
            3,
            kind="int",
            min=1,
            max=20,
            help="Individuals drawn per tournament selection.",
        ),
        ParamSpec(
            "patience",
            60,
            kind="int",
            min=1,
            max=10_000,
            help="Stop after this many generations without improvement.",
        ),
        ParamSpec(
            "max_iter", 300, kind="int", min=1, max=100_000, help="Maximum number of generations."
        ),
        ParamSpec(
            "record_every",
            1,
            kind="int",
            min=1,
            max=10_000,
            help="Record every m-th generation in the trace.",
        ),
    ),
    needs=("tsp",),
    order="stochastic metaheuristic",
    summary="Evolve a population of tours: tournament selection, order crossover, swap mutation, elitism.",
    references=(
        "Davis (1985), Applying adaptive algorithms to epistatic domains, IJCAI-85, 162–164 (OX)",
        "Goldberg (1989), Genetic Algorithms in Search, Optimization and Machine Learning, Ch. 5",
        "Larrañaga et al. (1999), Artif. Intell. Rev. 13, 129–170 (TSP operators survey)",
    ),
    deterministic=False,
)
def tsp_genetic(
    problem: TspInstance | ArrayLike,
    *,
    seed: int = 0,
    pop_size: int = 40,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.2,
    tournament_size: int = 3,
    patience: int = 60,
    max_iter: int = 300,
    record_every: int = 1,
) -> Result:
    """Generational genetic algorithm with OX crossover, swap mutation and one elite.

    Draw order (TypeScript parity): the initial population is ``pop_size`` calls of
    ``rng.permutation(n)``. Each generation copies the best tour (elitism; first index on
    ties), then makes ``pop_size − 1`` children, each by:

    1. two tournament selections — each draws ``tournament_size`` indices with
       ``rng.integers(pop_size)`` (with replacement); the shortest tour wins, ties go to the
       earlier draw;
    2. u = rng.random(); if u < crossover_rate: a = rng.integers(n), b = rng.integers(n),
       swap so a ≤ b, child = OX(parent1, parent2, a, b); otherwise child = copy of parent1;
    3. u = rng.random(); if u < mutation_rate: i = rng.integers(n), j = rng.integers(n) and
       swap child[i], child[j] (a no-op when i = j).

    Stopping test: the best length has not improved by more than tol for ``patience``
    generations (``converged=True``; a stall test, not an optimality certificate). Reaching
    ``max_iter`` generations first gives ``converged=False``. Elitism makes the best length
    non-increasing.
    """
    inst = _instance(problem)
    pop_size = _check_int("pop_size", pop_size, 2)
    tournament_size = _check_int("tournament_size", tournament_size, 1)
    patience = _check_int("patience", patience, 1)
    max_iter = _check_int("max_iter", max_iter, 1)
    record_every = _check_int("record_every", record_every, 1)
    crossover_rate = _check_unit("crossover_rate", crossover_rate)
    mutation_rate = _check_unit("mutation_rate", mutation_rate)
    D = _distances(inst)
    tol = _tolerance(D)
    n = inst.n
    rng = Rng(seed)

    pop = [rng.permutation(n) for _ in range(pop_size)]
    lengths = [tour_length(D, p) for p in pop]
    n_fev = pop_size

    def argmin(vals: list[float]) -> int:
        best = 0
        for idx in range(1, len(vals)):
            if vals[idx] < vals[best]:
                best = idx
        return best

    def tournament() -> int:
        winner = rng.integers(pop_size)
        for _ in range(tournament_size - 1):
            c = rng.integers(pop_size)
            if lengths[c] < lengths[winner]:
                winner = c
        return winner

    def step(k: int, stall: int) -> Step:
        b = argmin(lengths)
        mean = sum(lengths) / pop_size
        return Step(
            k,
            list(pop[b]),
            lengths[b],
            info={
                "tour": list(pop[b]),
                "length": lengths[b],
                "best_length": best_len,
                "mean_length": mean,
                "worst_length": max(lengths),
                "stall": stall,
            },
        )

    best_len = lengths[argmin(lengths)]
    stall_ref = best_len  # best length at the last improvement by more than tol
    trace = [step(0, 0)]
    stall = 0
    converged = False
    k = 0
    while k < max_iter:
        k += 1
        elite = argmin(lengths)
        new_pop, new_len = [list(pop[elite])], [lengths[elite]]
        while len(new_pop) < pop_size:
            p1, p2 = tournament(), tournament()
            if rng.random() < crossover_rate:
                a, b = rng.integers(n), rng.integers(n)
                if a > b:
                    a, b = b, a
                child = order_crossover(pop[p1], pop[p2], a, b)
            else:
                child = list(pop[p1])
            if rng.random() < mutation_rate:
                i, j = rng.integers(n), rng.integers(n)
                child[i], child[j] = child[j], child[i]
            new_pop.append(child)
            new_len.append(tour_length(D, child))
            n_fev += 1
        pop, lengths = new_pop, new_len
        gen_best = lengths[argmin(lengths)]
        # Elitism makes gen_best non-increasing, so best_len = gen_best exactly; the stall test
        # compares against the level of the last improvement by more than tol.
        best_len = min(best_len, gen_best)
        if gen_best < stall_ref - tol:
            stall_ref, stall = gen_best, 0
        else:
            stall += 1
        done = stall >= patience
        if _record(k, record_every) or done or k == max_iter:
            trace.append(step(k, stall))
        if done:
            converged = True
            break
    b = argmin(lengths)
    msg = (
        f"stalled: best length unchanged for {patience} generations (after {k}; heuristic stop)"
        if converged
        else f"reached max_iter={max_iter} generations (best still improving within patience)"
    )
    return _finish("tsp_genetic", inst, pop[b], lengths[b], converged, msg, k, n_fev, trace)


# --------------------------------------------------------------------------------------
# Ant colony (Ant System)
# --------------------------------------------------------------------------------------


@register(
    id="tsp_ant_colony",
    family="combinatorial",
    name="Ant System (ant colony optimization)",
    params=(
        ParamSpec(
            "n_ants",
            20,
            kind="int",
            min=1,
            max=500,
            help="Ants per iteration (Dorigo et al. suggest m = n).",
        ),
        ParamSpec("alpha", 1.0, min=0.0, max=5.0, help="Pheromone exponent α."),
        ParamSpec("beta", 5.0, min=0.0, max=10.0, help="Visibility exponent β (η = 1/d)."),
        ParamSpec("rho", 0.5, min=0.01, max=1.0, help="Evaporation rate ρ: τ ← (1 − ρ)τ + Σ Δτ."),
        ParamSpec(
            "patience",
            30,
            kind="int",
            min=1,
            max=10_000,
            help="Stop after this many iterations without improvement.",
        ),
        ParamSpec(
            "max_iter",
            100,
            kind="int",
            min=1,
            max=100_000,
            help="Maximum number of colony iterations.",
        ),
        ParamSpec(
            "record_every",
            1,
            kind="int",
            min=1,
            max=10_000,
            help="Record every m-th iteration in the trace.",
        ),
    ),
    needs=("tsp",),
    order="stochastic metaheuristic",
    summary="Ants build tours city by city, preferring short edges with much pheromone; "
    "pheromone evaporates and short tours deposit more.",
    references=(
        "Dorigo, Maniezzo & Colorni (1996), IEEE Trans. SMC-B 26(1), 29–41 (Ant System)",
        "Dorigo & Stützle (2004), Ant Colony Optimization, Ch. 3 (§3.3.1 Ant System)",
    ),
    deterministic=False,
)
def tsp_ant_colony(
    problem: TspInstance | ArrayLike,
    *,
    seed: int = 0,
    n_ants: int = 20,
    alpha: float = 1.0,
    beta: float = 5.0,
    rho: float = 0.5,
    patience: int = 30,
    max_iter: int = 100,
    record_every: int = 1,
) -> Result:
    """Ant System (Dorigo, Maniezzo & Colorni 1996) for the symmetric TSP.

    Initialization: τ_ij = τ₀ = m / C^nn for i ≠ j, with C^nn the nearest-neighbor tour length
    from city 0 (the setting Dorigo & Stützle 2004, Ch. 3, recommend for AS); η_ij = 1/d_ij.

    Iteration (Dorigo & Stützle 2004, §3.3.1, Eqs. 3.1–3.3). For ants 0, …, m−1 in order:
    start city s = rng.integers(n); then n − 1 times, from city i, with weights
    w_j = τ_ij^α · η_ij^β over the unvisited j (ascending j), draw u = rng.random() and pick
    the first j whose running sum of w exceeds u·Σw (random-proportional rule
    p_ij = w_j / Σw, Eq. 3.1). Then evaporate τ ← (1 − ρ)τ (Eq. 3.2) and let every ant deposit
    Δτ = 1/L_k on both directions of each tour edge (Eq. 3.3).

    Scale-free evaluation of the rule. p_ij is unchanged when every w_j of a row is multiplied
    by the same positive factor, and AS is scale invariant: coordinates scaled by c scale τ and
    η by 1/c. Raw τ^α η^β therefore under- or overflows at large or small scales (all weights
    of a row become 0 or inf) although the probabilities do not change. The code keeps
    ℓ_ij = log(τ_ij/τ₀) (updated in log space, so evaporation cannot underflow it) and uses

        log w_ij = α·ℓ_ij + β·log(d_ref / d_ij),     d_ref = C^nn / n,

    and at each step w_j = exp(log w_ij − max_{unvisited j'} log w_ij') ∈ (0, 1], so Σw ≥ 1.
    In exact arithmetic this is the same p_ij as Eq. 3.1 at every coordinate scale.

    # NOTE: guards not in the paper — coincident cities (d_ij = 0, η_ij = ∞) use
    # d_ij = d_0 = min(tol, smallest positive d) (tol from the module docstring), so a
    # coincident city is never less attractive than a distinct one. Every d_ij > 0 enters
    # Eq. 3.1 exactly, however small (log d_ij is finite; where d_ref/d_ij overflows, the code
    # uses log d_ref − log d_ij). If every unvisited j has τ_ij = 0 exactly (possible only
    # with ρ = 1, α > 0), p_ij is 0/0 and the ant moves to the nearest unvisited city
    # instead. Such steps are counted in ``extra["nn_fallback_steps"]`` and named in the
    # message. If rounding leaves u·Σw beyond the running sum, the last unvisited city with
    # w_j > 0 is taken.

    Stopping test: the best length has not improved by more than tol for ``patience``
    iterations (``converged=True``; a stall test, not an optimality certificate). Reaching
    ``max_iter`` iterations first gives ``converged=False``.
    """
    inst = _instance(problem)
    n_ants = _check_int("n_ants", n_ants, 1)
    patience = _check_int("patience", patience, 1)
    max_iter = _check_int("max_iter", max_iter, 1)
    record_every = _check_int("record_every", record_every, 1)
    if not (math.isfinite(alpha) and alpha >= 0.0 and math.isfinite(beta) and beta >= 0.0):
        raise ValueError("alpha and beta must be non-negative")
    rho = _check_unit("rho", rho, closed_low=False)
    D = _distances(inst)
    tol = _tolerance(D)
    n = inst.n
    rng = Rng(seed)

    nn_tour = _nearest_neighbor_tour(D, 0, tol)
    nn_len = tour_length(D, nn_tour)
    tau0 = n_ants / nn_len if nn_len > 0.0 else 1.0
    d_ref = nn_len / n if nn_len > 0.0 else 1.0  # mean NN edge length: the instance's scale
    off_diag = ~np.eye(n, dtype=bool)
    # ℓ = log(τ/τ₀): 0 off the diagonal, −inf on it (τ_ii = 0), (n, n)
    log_tau_hat = np.where(off_diag, 0.0, -np.inf)
    positive = D[off_diag & (D > 0.0)]  # distinct-city distances, (≤ n(n−1),)
    d_zero = min(tol, float(positive.min())) if positive.size else 1.0  # d_0 (see NOTE)
    # only d_ij = 0 is replaced (Eq. 3.1 needs η = 1/d_ij exactly for every d_ij > 0), (n, n)
    safe_d = np.where(off_diag, np.where(D > 0.0, D, d_zero), d_ref)
    with np.errstate(over="ignore"):
        ratio = d_ref / safe_d  # η̂ = d_ref/d_ij, (n, n); overflows only for d_ij ≲ 1e−154·d_ref
    # log(d_ref/d) is exact-ratio-then-log where finite (keeps power-of-two scaling exact) and
    # log d_ref − log d where the ratio overflowed (both logs are finite: d_ij > 0)
    log_eta = np.where(np.isfinite(ratio), np.log(ratio), math.log(d_ref) - np.log(safe_d))
    beta_log_eta = beta * log_eta if beta > 0.0 else np.zeros((n, n))  # β·log η̂, (n, n)
    log_keep = math.log1p(-rho) if rho < 1.0 else -math.inf  # log(1 − ρ)
    nn_fallback_steps = 0

    def ant_tour(log_w: list[list[float]]) -> list[int]:
        nonlocal nn_fallback_steps
        start = rng.integers(n)
        visited = [False] * n
        visited[start] = True
        tour = [start]
        for _ in range(n - 1):
            i = tour[-1]
            row = log_w[i]
            cand = [j for j in range(n) if not visited[j]]
            top = max(row[j] for j in cand)
            u = rng.random()
            chosen = -1
            if math.isfinite(top):
                w = [math.exp(row[j] - top) for j in cand]  # w_j / max w ∈ (0, 1]
                total = 0.0
                for wj in w:
                    total += wj
                target = u * total
                run = 0.0
                for j, wj in zip(cand, w, strict=True):
                    run += wj
                    if target < run:
                        chosen = j
                        break
                if chosen < 0:
                    chosen = next(
                        j for j, wj in zip(reversed(cand), reversed(w), strict=True) if wj > 0.0
                    )
            else:  # every unvisited τ_ij = 0: p_ij = 0/0 (see NOTE)
                nn_fallback_steps += 1
                chosen = min(cand, key=lambda j: (float(D[i, j]), j))
            visited[chosen] = True
            tour.append(chosen)
        return tour

    def pheromone() -> Matrix:
        return tau0 * np.exp(log_tau_hat)  # τ = τ₀·exp(ℓ), (n, n)

    def tau_range(tau: Matrix) -> tuple[float, float]:
        if n < 2:
            return 0.0, 0.0
        vals = tau[off_diag]
        return float(vals.min()), float(vals.max())

    best_t: list[int] = []
    best_len = math.inf
    tau = pheromone()
    lo, hi = tau_range(tau)
    trace = [
        Step(
            0,
            list(nn_tour),
            nn_len,
            info={
                "tour": list(nn_tour),
                "length": nn_len,
                "best_tour": [],
                "best_length": None,
                "mean_length": None,
                "pheromone": tau.tolist() if n <= PHEROMONE_MAX_N else None,
                "tau_min": lo,
                "tau_max": hi,
                "stall": 0,
            },
        )
    ]
    n_fev = 0
    stall = 0
    converged = False
    k = 0
    while k < max_iter:
        k += 1
        # log w_ij = α·log(τ_ij/τ₀) + β·log(d_ref/d_ij); α = 0 skips the term (0·(−inf) is NaN)
        log_w = (alpha * log_tau_hat + beta_log_eta) if alpha > 0.0 else beta_log_eta.copy()
        log_w[~off_diag] = -math.inf
        tours = [ant_tour(log_w.tolist()) for _ in range(n_ants)]
        lens = [tour_length(D, tr) for tr in tours]
        n_fev += n_ants
        # Δτ/τ₀ = (1/L)·(C^nn/m), summed over ants, (n, n); scale-free, so no under/overflow
        delta_hat = np.zeros((n, n))
        for tr, L in zip(tours, lens, strict=True):
            if L <= 0.0:
                continue
            dep_hat = 1.0 / (L * tau0)
            for s in range(n):
                a, b = tr[s], tr[(s + 1) % n]
                if a != b:
                    delta_hat[a, b] += dep_hat
                    delta_hat[b, a] += dep_hat
        with np.errstate(divide="ignore"):
            log_delta = np.log(delta_hat)  # −inf where no ant deposited
        # ℓ ← log((1 − ρ)·e^ℓ + Δτ/τ₀)  (Eqs. 3.2–3.3 in log space)
        log_tau_hat = np.logaddexp(log_keep + log_tau_hat, log_delta)
        it_best = 0
        for idx in range(1, n_ants):
            if lens[idx] < lens[it_best]:
                it_best = idx
        if lens[it_best] < best_len - tol:
            best_t, best_len, stall = list(tours[it_best]), lens[it_best], 0
        else:
            stall += 1
        done = stall >= patience
        if _record(k, record_every) or done or k == max_iter:
            tau = pheromone()
            lo, hi = tau_range(tau)
            trace.append(
                Step(
                    k,
                    list(tours[it_best]),
                    lens[it_best],
                    info={
                        "tour": list(tours[it_best]),
                        "length": lens[it_best],
                        "best_tour": list(best_t),
                        "best_length": best_len,
                        "mean_length": sum(lens) / n_ants,
                        "pheromone": tau.tolist() if n <= PHEROMONE_MAX_N else None,
                        "tau_min": lo,
                        "tau_max": hi,
                        "stall": stall,
                    },
                )
            )
        if done:
            converged = True
            break
    msg = (
        f"stalled: best length unchanged for {patience} iterations (after {k}; heuristic stop)"
        if converged
        else f"reached max_iter={max_iter} iterations (best still improving within patience)"
    )
    if nn_fallback_steps:
        msg += (
            f"; {nn_fallback_steps} construction steps had zero pheromone on every unvisited "
            "edge and moved to the nearest city instead"
        )
    return _finish(
        "tsp_ant_colony",
        inst,
        best_t,
        best_len,
        converged,
        msg,
        k,
        n_fev,
        trace,
        tau0=tau0,
        nn_fallback_steps=nn_fallback_steps,
    )


# --------------------------------------------------------------------------------------
# Held–Karp exact dynamic programming
# --------------------------------------------------------------------------------------


@register(
    id="tsp_held_karp",
    family="combinatorial",
    name="Held–Karp dynamic programming (exact)",
    params=(),
    needs=("tsp",),
    order=f"exact, O(2ⁿ·n²) time, O(2ⁿ·n) memory (n ≤ {HELD_KARP_MAX_N})",
    summary="For every subset of cities and every last city, remember the shortest path from city 0.",
    references=(
        "Held & Karp (1962), J. SIAM 10(1), 196–210",
        "Bellman (1962), J. ACM 9(1), 61–63",
    ),
)
def tsp_held_karp(problem: TspInstance | ArrayLike) -> Result:
    """Held–Karp / Bellman dynamic programming for the exact TSP optimum.

    For S ⊆ {1, …, n−1} and j ∈ S, C(S, j) is the length of the shortest path that starts at
    city 0, visits every city of S exactly once and ends at j (Held & Karp 1962; Bellman 1962):

        C({j}, j) = d(0, j),
        C(S, j)   = min_{i ∈ S∖{j}} C(S∖{j}, i) + d(i, j),
        L*        = min_{j} C({1, …, n−1}, j) + d(j, 0).

    Subsets are bit masks over cities 1, …, n−1, processed in layers of equal size |S|; the
    minimizing predecessor (first index on ties) is stored to recover the tour.

    Trace: step k = s (s = 1, …, n−1) is DP layer |S| = s, showing the cheapest path of that
    layer; step k = n closes the tour (n = 1 has only k = 0). Stopping test: all layers are
    complete; the method is exact, so ``converged`` is always True.
    Raises ``ValueError`` for n > ``HELD_KARP_MAX_N`` = 16: the inner minimum over i is
    vectorized, so n = 16 (15·2¹⁴ ≈ 2.5·10⁵ states) takes about 0.3 s, but every further
    city doubles the memory and more than doubles the time.
    """
    inst = _instance(problem)
    n = inst.n
    if n > HELD_KARP_MAX_N:
        raise ValueError(
            f"{inst.id}: Held–Karp is limited to n ≤ {HELD_KARP_MAX_N} cities (got n = {n})"
        )
    D = _distances(inst)
    trace = [
        Step(
            0,
            [0],
            0.0,
            info={"subset_size": 0, "states": 1, "tour": [0], "length": 0.0, "closed": n == 1},
        )
    ]
    if n == 1:
        return _finish(
            "tsp_held_karp",
            inst,
            [0],
            0.0,
            True,
            "n = 1: the tour is the single city",
            0,
            0,
            trace,
            states=1,
            transitions=0,
        )

    m = n - 1  # cities 1..n−1 are bits 0..m−1
    size = 1 << m
    cost = np.full((size, m), np.inf)  # C(S, j)
    parent = np.full((size, m), -1, dtype=np.int64)  # predecessor bit of j in S, −1 for city 0
    d_from = D[1:, 1:]  # (m, m): d(i+1, j+1)
    layers: list[list[int]] = [[] for _ in range(m + 1)]
    for mask in range(1, size):
        layers[mask.bit_count()].append(mask)

    def path_of(mask: int, j: int) -> list[int]:
        path = []
        while j >= 0:
            path.append(j + 1)
            prev = int(parent[mask, j])
            mask ^= 1 << j
            j = prev
        return [0, *path[::-1]]

    transitions = 0
    states = 0
    for s in range(1, m + 1):
        layer_states = 0
        for mask in layers[s]:
            for j in range(m):
                if not mask >> j & 1:
                    continue
                layer_states += 1
                prev = mask ^ (1 << j)
                if prev == 0:
                    cost[mask, j] = D[0, j + 1]
                    continue
                cand = cost[prev] + d_from[:, j]  # inf where i ∉ prev
                i = int(np.argmin(cand))
                cost[mask, j] = cand[i]
                parent[mask, j] = i
                transitions += s - 1
        states += layer_states
        # Cheapest state of the layer (first mask, then first j, on ties).
        best_mask, best_j, best_c = -1, -1, math.inf
        for mask in layers[s]:
            row = cost[mask]
            j = int(np.argmin(row))
            if row[j] < best_c:
                best_mask, best_j, best_c = mask, j, float(row[j])
        path = path_of(best_mask, best_j)
        length = _path_length(D, path)
        trace.append(
            Step(
                s,
                path,
                length,
                info={
                    "subset_size": s,
                    "states": layer_states,
                    "tour": path,
                    "length": length,
                    "closed": False,
                },
            )
        )

    full = size - 1
    closing = cost[full] + D[1:, 0]
    j_last = int(np.argmin(closing))
    tour = path_of(full, j_last)
    length = tour_length(D, tour)
    trace.append(
        Step(
            n,
            tour,
            length,
            info={"subset_size": m, "states": m, "tour": tour, "length": length, "closed": True},
        )
    )
    return _finish(
        "tsp_held_karp",
        inst,
        tour,
        length,
        True,
        f"exact optimum by Held–Karp DP over {states} states",
        n,
        0,
        trace,
        states=states,
        transitions=transitions + m,
    )


#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("tsp_nearest_neighbor", "tsp_random_15", {}),
    ("tsp_two_opt", "tsp_circle_12", {"strategy": "first"}),
    ("tsp_two_opt", "tsp_cities_20", {"strategy": "best", "init": "nearest_neighbor"}),
    ("tsp_or_opt", "tsp_random_15", {"init": "nearest_neighbor"}),
    ("tsp_simulated_annealing", "tsp_random_15", {"seed": 1}),
    ("tsp_genetic", "tsp_grid_16", {"seed": 2}),
    ("tsp_ant_colony", "tsp_circle_12", {"seed": 3, "max_iter": 20}),
    ("tsp_held_karp", "tsp_circle_12", {}),
]
