"""Stochastic global minimization of f: ℝⁿ → ℝ over a search box (family ``global``).

Methods: simulated annealing, particle swarm optimization, differential evolution, CMA-ES and
basin hopping. They share these conventions:

* **Search box.** Every method needs the box [lo, hi] = ``problem.domain`` (one ``(lo, hi)`` pair
  per coordinate; ``ValueError`` without one). A bare callable has no box, so pass
  ``numopt.core.counting.vector_problem(f, x0=x0, domain=((lo, hi), ...))`` instead. Simulated annealing, particle swarm and
  differential evolution keep every iterate inside the box (each documents how). CMA-ES and
  basin hopping use the box only to scale their initial step sizes and may leave it, as in
  their references.
* **Start.** ``x0`` (or the problem's default) is the starting point of simulated annealing,
  CMA-ES (the initial mean) and basin hopping, and the first member of the initial
  population of particle swarm and differential evolution (if neither is given, every member
  is random). Where a method keeps iterates in the box, x0 must lie in it (else ``ValueError``).
* **Randomness.** All random numbers come from ``numopt.core.rng.Rng(seed)`` (Mulberry32,
  bit-identical in the web app), drawn in the order each method documents under "Draw order".
  ``Rng.normal()`` uses two uniforms per normal (Box–Muller); ``Rng.integers(m)`` one uniform.
* **dim = 1.** A :class:`Problem` with ``dim == 1`` follows the scalar convention of
  ``core.types``: every method, and the local Nelder–Mead searches of basin hopping, pass
  f the float x[0]. ``Result.x`` and the trace keep vectors of length 1.
* **Extreme barrier.** f values that are NaN or +∞ count as +∞ (never accepted, ranked last).
  A value −∞ stops the method with ``converged=False`` (f is unbounded below).
* **Trace.** One Step for ``k = 0`` and one per iteration (generation, temperature step, hop),
  ``n_iter == trace[-1].k``. ``Step.x`` and ``Step.fun`` are the best point found so far and
  its value; ``Result.x`` is the best point found.
* **Convergence.** ``converged=True`` means the documented stopping test of the method passed.
  That test detects that the search has *settled* (a frozen temperature, a collapsed
  population or one whose f values differ only by rounding, a stalled best minimum). It is
  not a certificate of global optimality.
* **Counts.** ``n_fev`` counts every evaluation of f exactly (basin hopping includes the
  evaluations of its local Nelder–Mead searches).

Info keys (every method, every step):
    best: [n]                the best point found so far (= Step.x).
    best_f: float            its value (= Step.fun).

Additional info keys per method:
    simulated_annealing:
        current: [n]          the state of the Markov chain after the step; current_f: f there.
        candidate: [n] | null the proposal y = x + σ_k⊙z of this step (null at k = 0).
        proposal_sd: [n] | null  σ_k, the proposal standard deviations of this step.
        candidate_f: float | null  f(y), or null when y lies outside the box (not evaluated).
        inside: bool | null   whether y lies in the box.
        accept_prob: float | null  min(1, exp(−(f(y) − f(x))/T)), 0 outside the box.
        accepted: bool | null whether the chain moved to y.
        temperature: float    the temperature T of this step (T₀ at k = 0).
    particle_swarm:
        particles: [[n] × N]  positions after the step; particles_f: [N] f at them.
        velocities: [[n] × N] velocities after the step (a coordinate clamped at the box has
                              velocity 0).
        personal_best: [[n] × N]  each particle's best position; personal_best_f: [N].
        global_best: [n]      the swarm's best position (= best); global_best_f: float.
    differential_evolution:
        population: [[n] × NP]  the population after selection; population_f: [NP].
        mutants: [[n] × NP]   the mutant vectors v_i of this generation ([] at k = 0).
        trials: [[n] × NP]    the trial vectors u_i after crossover and bound repair ([] at k = 0).
        trials_f: [NP]        f(u_i) ([] at k = 0).
        accepted: [bool] × NP whether u_i replaced x_i ([] at k = 0).
        best_index: int       index of the best member of ``population``.
    cma_es:
        mean: [n]             the distribution mean m after the update; sigma: float σ after.
        covariance: [[n] × n] C after the update (the search distribution is N(m, σ²C)).
        sample_mean, sample_sigma, sample_covariance: m, σ, C of the distribution the
                              ``population`` of this step was drawn from (equal to the
                              updated values at k = 0).
        population: [[n] × λ] the λ samples of this generation ([] at k = 0).
        population_f: [λ]     f at the samples ([] at k = 0).
        selected: [int] × μ   indices into ``population`` of the μ best, best first.
        p_sigma: [n], p_c: [n]  the evolution paths after the update.
        h_sigma: bool | null  Hansen's stall indicator h_σ of this generation.
    basin_hopping:
        current: [n]          the current local minimum after the hop; current_f: f there.
        start: [n]            the perturbed start y = x + δ of this hop (x0 at k = 0).
        local_min: [n]        the local minimum z reached from ``start``; local_f: f(z).
        local_path: [[n]...]  the best vertex of each Nelder–Mead iteration from ``start``.
        local_converged: bool whether that Nelder–Mead search met its tolerance.
        accept_prob: float | null  min(1, exp(−(f(z) − f(x))/T)) (null at k = 0).
        accepted: bool | null whether the hop moved the current minimum to z.
        minima: [{"x": [n], "f": float, "hits": int}]  the distinct local minima found so far,
                              in order of discovery, with the number of hops that reached each.
        stall: int            hops since the best minimum last improved (stopping counter).
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from typing import Any

import numpy as np

from ..core.counting import vector_problem
from ..core.registry import ParamSpec, register
from ..core.rng import Rng
from ..core.types import Problem, Result, Step, Vector, as_vector
from .derivative_free import _check_max_iter, _Objective, nelder_mead

#: Machine epsilon of float64.
EPS = sys.float_info.epsilon

VectorFn = Callable[[Any], float]


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


def _resolve(problem: Problem | VectorFn, x0: Any) -> tuple[Problem, Vector, Vector, Vector | None]:
    """Return (problem, lo, hi, start) where start is x0, the problem's x0, or None."""
    prob = vector_problem(problem, x0=x0)
    domain = prob.domain
    if not domain:
        raise ValueError(
            f"{prob.id}: global methods need a search box (problem.domain); for a bare callable "
            "pass numopt.core.counting.vector_problem(f, x0=x0, domain=((lo, hi), ...))"
        )
    # A 1-D scalar problem stores its domain as (lo, hi); n-D problems as ((lo, hi), ...).
    pairs = [domain] if len(domain) == 2 and np.isscalar(domain[0]) else list(domain)
    lo = np.array([float(p[0]) for p in pairs], dtype=np.float64)
    hi = np.array([float(p[1]) for p in pairs], dtype=np.float64)
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)) and np.all(lo < hi)):
        raise ValueError(f"{prob.id}: invalid search box lo={lo.tolist()}, hi={hi.tolist()}")
    start = x0 if x0 is not None else prob.x0
    x_start = None if start is None else as_vector(start)
    if x_start is not None and x_start.size != lo.size:
        raise ValueError(f"x0 has {x_start.size} entries, the search box has {lo.size}")
    return prob, lo, hi, x_start


def _require_inside(x: Vector, lo: Vector, hi: Vector) -> None:
    if not bool(np.all((x >= lo) & (x <= hi))):
        raise ValueError(
            f"x0 = {x.tolist()} lies outside the search box [{lo.tolist()}, {hi.tolist()}]"
        )


def _require_start(x: Vector | None, prob: Problem) -> Vector:
    if x is None:
        raise ValueError(f"{prob.id}: no starting point given and the problem has no default x0")
    return x


def _unbounded(method: str, x: Vector, trace: list[Step], k: int, nfev: int) -> Result:
    msg = f"f = -inf at x = {x.tolist()}: f is unbounded below"
    return Result(method, x, -math.inf, False, msg, k, nfev, trace=trace)


def _max_iter(
    method: str, x: Vector, fx: float, max_iter: int, nfev: int, trace: list[Step]
) -> Result:
    msg = f"reached max_iter={max_iter} (best f = {fx:.6g})"
    return Result(method, x, fx, False, msg, max_iter, nfev, trace=trace)


def _check_positive(**values: float) -> None:
    for name, v in values.items():
        if not v > 0.0:
            raise ValueError(f"{name} must be > 0, got {v}")


def _uniform_in_box(rng: Rng, lo: Vector, hi: Vector) -> Vector:
    """One uniform point of the box: n draws, coordinate j = lo_j + (hi_j − lo_j)·u_j."""
    return np.array([lo[j] + (hi[j] - lo[j]) * rng.random() for j in range(lo.size)])


def _collapsed(
    points: list[Vector], values: list[float], best: Vector, f_best: float, xtol: float, ftol: float
) -> tuple[bool, float, float]:
    """Population test: max_i ‖x_i − best‖∞ ≤ xtol + 2ε‖best‖∞ and max_i f_i − f_best ≤ ftol + 2ε|f_best|.

    Returns (passed, x-spread, f-spread).
    """
    x_spread = max(float(np.max(np.abs(p - best))) for p in points)
    f_spread = max(values) - f_best
    tol_x = xtol + 2.0 * EPS * float(np.max(np.abs(best)))
    tol_f = ftol + 2.0 * EPS * abs(f_best)
    return x_spread <= tol_x and f_spread <= tol_f, x_spread, f_spread


#: Rounding-plateau exit of particle swarm and differential evolution: the number of
#: consecutive generations over which every f value must stay within 2ε|f_best| of f_best.
_PLATEAU_GENERATIONS = 10


def _on_plateau(f_max_history: list[float], f_best: float) -> tuple[bool, float]:
    """Rounding-plateau test: max f over the last G generations − f_best ≤ 2ε|f_best|.

    ``f_max_history[k]`` is the largest f value of the population of generation k. f_best never
    increases, so the test also says that f_best fell by at most 2ε|f_best| in those G
    generations: f does not separate the members beyond rounding, and the search has stalled.
    Returns (passed, f-range over the window); the test needs G generations of history.
    """
    if len(f_max_history) < _PLATEAU_GENERATIONS:
        return False, math.inf
    f_range = max(f_max_history[-_PLATEAU_GENERATIONS:]) - f_best
    return f_range <= 2.0 * EPS * abs(f_best), f_range


def _plateau_message(what: str, x_spread: float, f_range: float, f_best: float) -> str:
    return (
        f"rounding plateau: for {_PLATEAU_GENERATIONS} generations every f of the {what} was "
        f"within 2ε|f_best| of f_best (f-range {f_range:.3g}), so f cannot rank the members; "
        f"x-spread {x_spread:.3g}; best f = {f_best:.6g}"
    )


def _rows(points: list[Vector]) -> list[Vector]:
    return [p.copy() for p in points]


_TOL_PARAMS = (
    ParamSpec(
        "xtol",
        1e-6,
        min=1e-12,
        max=1e-1,
        log=True,
        help="Population spread test: every member within xtol (∞-norm) of the best point.",
    ),
    ParamSpec(
        "ftol",
        1e-8,
        min=1e-14,
        max=1e-1,
        log=True,
        help="Population spread test: every member's f within ftol of the best f.",
    ),
)


# --------------------------------------------------------------------------------------
# Simulated annealing
# --------------------------------------------------------------------------------------


@register(
    id="simulated_annealing",
    family="global",
    name="Simulated annealing",
    params=(
        ParamSpec("T0", 10.0, min=1e-3, max=1e3, log=True, help="Initial temperature T₀."),
        ParamSpec(
            "cooling",
            "geometric",
            kind="choice",
            choices=("geometric", "logarithmic"),
            help="Schedule: Tₜ = T₀·αᵗ (geometric) or T₀·ln 2/ln(t + 2) (logarithmic).",
        ),
        ParamSpec(
            "alpha", 0.99, min=0.8, max=0.9999, help="Geometric cooling factor α (geometric only)."
        ),
        ParamSpec(
            "T_min",
            1e-3,
            min=1e-8,
            max=1.0,
            log=True,
            help="Stop (frozen) when the temperature of a step is ≤ Tₘᵢₙ (T_min).",
        ),
        ParamSpec(
            "step",
            0.1,
            min=1e-3,
            max=1.0,
            log=True,
            help="Proposal standard deviation at T = T₀, as a fraction of the box width (∝ √T after).",
        ),
        ParamSpec("max_iter", 2000, kind="int", min=1, max=100_000, help="Step limit."),
    ),
    needs=("f",),
    order="global convergence in probability only for logarithmic cooling (Hajek 1988)",
    summary="Random-walk proposals, accepted uphill with probability exp(−Δf/T) as T cools.",
    references=(
        "Kirkpatrick, Gelatt & Vecchi (1983), Science 220(4598), 671–680",
        "Ingber (1993), Math. Comput. Modelling 18(11), 29–57 (Boltzmann annealing)",
        "Metropolis et al. (1953), J. Chem. Phys. 21(6), 1087–1092",
        "Geman & Geman (1984), IEEE TPAMI 6(6), 721–741 (logarithmic schedule)",
        "Hajek (1988), Math. Oper. Res. 13(2), 311–329 (convergence of logarithmic cooling)",
    ),
    deterministic=False,
)
def simulated_annealing(
    problem: Problem | VectorFn,
    *,
    x0: Any = None,
    seed: int = 0,
    T0: float = 10.0,
    cooling: str = "geometric",
    alpha: float = 0.99,
    T_min: float = 1e-3,
    step: float = 0.1,
    max_iter: int = 2000,
) -> Result:
    """Simulated annealing with Gaussian (Boltzmann) proposals and the Metropolis rule.

    Step k = 1, 2, … uses the temperature T_k = T₀·α^{k−1} (geometric) or
    T_k = T₀·ln 2/ln(k + 1) (logarithmic, Geman & Geman 1984; T₁ = T₀). From the current state x:

        y = x + σ_k⊙z,  z ~ N(0, I),  σ_k,j = step·(hi_j − lo_j)·√(T_k/T₀);
        accept y with probability min(1, exp(−(f(y) − f(x))/T_k))   (Metropolis et al. 1953).

    The proposal variance is proportional to the temperature, the generating distribution of
    Boltzmann annealing (Ingber 1993), so the walk explores at high T and refines the
    best basin as T falls (Kirkpatrick et al. 1983 describe the cooling, not the proposal).

    A proposal outside the box is rejected without evaluating f: this is the Metropolis chain
    for the target ∝ exp(−f/T) restricted to the box (its density is 0 outside).

    Stops (converged) after the first step whose temperature is ≤ ``T_min`` (the schedule is
    frozen). With logarithmic cooling, T_k ≤ T_min ⇔ ln(k + 1) ≥ T₀ ln 2/T_min ⇔
    k ≥ 2^{T₀/T_min} − 1, so freezing takes about 2^{T₀/T_min} steps (2^{10⁴} for the defaults)
    and the run usually ends at ``max_iter`` (converged=False): the slow schedule of the
    convergence theory is impractical, which is the point it illustrates.

    Draw order (per step): z_1, …, z_n (``rng.normal()``, 2 uniforms each), then the acceptance
    uniform u (``rng.random()``), always drawn even when it is not needed; 2n + 1 uniforms.
    """
    _check_max_iter(max_iter)
    _check_positive(T0=T0, T_min=T_min, step=step)
    if cooling not in ("geometric", "logarithmic"):
        raise ValueError(f"cooling must be 'geometric' or 'logarithmic', got {cooling!r}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    prob, lo, hi, start = _resolve(problem, x0)
    x = _require_start(start, prob)
    _require_inside(x, lo, hi)
    n = x.size
    fobj = _Objective(prob.f, scalar=prob.dim == 1)
    rng = Rng(seed)
    sigma0 = step * (hi - lo)  # (n,) proposal standard deviation at T = T₀
    fx = fobj(x)
    if not math.isfinite(fx):
        msg = f"f(x0) is not finite (f = {fx!r}); cannot start"
        return Result(
            "simulated_annealing", x, fx, False, msg, 0, fobj.n, trace=[Step(0, x.copy(), fx)]
        )
    best, f_best = x.copy(), fx

    def temperature(k: int) -> float:
        if cooling == "geometric":
            return T0 * alpha ** (k - 1)
        return T0 * math.log(2.0) / math.log(k + 1.0)

    info0: dict[str, Any] = {"best": best.copy(), "best_f": f_best, "current": x.copy()}
    info0 |= {"current_f": fx, "candidate": None, "proposal_sd": None, "candidate_f": None}
    info0 |= {"inside": None}
    info0 |= {"accept_prob": None, "accepted": None, "temperature": T0}
    trace = [Step(0, best.copy(), f_best, info=info0)]
    k = 0
    while True:
        k += 1
        T = temperature(k)
        z = np.array([rng.normal() for _ in range(n)])
        u = rng.random()
        sigma = sigma0 * math.sqrt(T / T0)  # Boltzmann annealing: proposal variance ∝ T
        y = x + sigma * z
        inside = bool(np.all((y >= lo) & (y <= hi)))
        f_y: float | None = None
        if inside:
            f_y = fobj(y)
            delta = f_y - fx
            # exp(−Δ/T) underflows harmlessly to 0 for large Δ/T; Δ = +inf gives exp(−inf) = 0.
            prob_acc = 1.0 if delta <= 0.0 else math.exp(-delta / T)
        else:
            prob_acc = 0.0
        accepted = inside and u < prob_acc
        if accepted:
            assert f_y is not None
            x, fx = y, f_y
            if fx < f_best:
                best, f_best = x.copy(), fx
        info = {
            "best": best.copy(),
            "best_f": f_best,
            "current": x.copy(),
            "current_f": fx,
            "candidate": y,
            "proposal_sd": sigma,
            "candidate_f": f_y,
            "inside": inside,
            "accept_prob": prob_acc,
            "accepted": accepted,
            "temperature": T,
        }
        trace.append(
            Step(k, best.copy(), f_best, step_size=float(np.linalg.norm(sigma * z)), info=info)
        )
        if f_best == -math.inf:
            return _unbounded("simulated_annealing", best, trace, k, fobj.n)
        if T <= T_min:
            msg = f"temperature T = {T:.3g} ≤ Tₘᵢₙ = {T_min:.3g} (frozen); best f = {f_best:.6g}"
            return Result("simulated_annealing", best, f_best, True, msg, k, fobj.n, trace=trace)
        if k == max_iter:
            return _max_iter("simulated_annealing", best, f_best, max_iter, fobj.n, trace)


# --------------------------------------------------------------------------------------
# Particle swarm optimization
# --------------------------------------------------------------------------------------

#: Clerc–Kennedy constriction for φ₁ = φ₂ = 2.05: χ = 2/|2 − φ − √(φ² − 4φ)|, φ = φ₁ + φ₂.
_PHI = 4.1
_CHI = 2.0 / abs(2.0 - _PHI - math.sqrt(_PHI * _PHI - 4.0 * _PHI))  # 0.7298437881...
_C_CONSTRICTION = _CHI * 2.05  # 1.4961797656...


@register(
    id="particle_swarm",
    family="global",
    name="Particle swarm",
    params=(
        ParamSpec("n_particles", 20, kind="int", min=2, max=200, help="Swarm size N."),
        ParamSpec(
            "w",
            _CHI,
            min=0.0,
            max=1.2,
            help="Inertia weight w (default: Clerc–Kennedy constriction χ ≈ 0.7298).",
        ),
        ParamSpec(
            "c1",
            _C_CONSTRICTION,
            min=0.0,
            max=4.0,
            help="Cognitive coefficient: pull toward the particle's own best (default χ·2.05).",
        ),
        ParamSpec(
            "c2",
            _C_CONSTRICTION,
            min=0.0,
            max=4.0,
            help="Social coefficient: pull toward the swarm's best (default χ·2.05).",
        ),
        *_TOL_PARAMS,
        ParamSpec("max_iter", 1000, kind="int", min=1, max=100_000, help="Iteration limit."),
    ),
    needs=("f",),
    order="no rate; the swarm contracts geometrically when w, c₁, c₂ are in the stable region",
    summary="Particles fly through the box, pulled toward their own best and the swarm's best point.",
    references=(
        "Kennedy & Eberhart (1995), Proc. IEEE ICNN, 1942–1948",
        "Shi & Eberhart (1998), Proc. IEEE CEC, 69–73 (inertia weight)",
        "Clerc & Kennedy (2002), IEEE Trans. Evol. Comput. 6(1), 58–73 (constriction)",
    ),
    deterministic=False,
)
def particle_swarm(
    problem: Problem | VectorFn,
    *,
    x0: Any = None,
    seed: int = 0,
    n_particles: int = 20,
    w: float = _CHI,
    c1: float = _C_CONSTRICTION,
    c2: float = _C_CONSTRICTION,
    xtol: float = 1e-6,
    ftol: float = 1e-8,
    max_iter: int = 1000,
) -> Result:
    """Global-best particle swarm with inertia weight (Kennedy & Eberhart 1995; Shi & Eberhart 1998).

    Each particle i has a position x_i, velocity v_i and personal best p_i; g is the swarm's
    best. One iteration updates every particle with the g of the start of the iteration
    (synchronous update), then evaluates all particles and updates p_i and g:

        v_i ← w v_i + c₁ r₁ ⊙ (p_i − x_i) + c₂ r₂ ⊙ (g − x_i),   r₁, r₂ ~ U[0, 1)ⁿ
        x_i ← x_i + v_i

    The defaults w = χ ≈ 0.72984, c₁ = c₂ = χ·2.05 ≈ 1.49618 make this exactly the constriction
    form v ← χ(v + φ₁r₁(p − x) + φ₂r₂(g − x)) of Clerc & Kennedy (2002) with φ₁ = φ₂ = 2.05.

    Box: a coordinate that leaves [lo_j, hi_j] is set to the bound and its velocity to 0
    (absorbing walls, as in SPSO 2007). Initial velocities are v_i = (U(lo, hi) − x_i)/2.

    Stops (converged) when every particle is within xtol + 2ε‖g‖∞ of g (∞-norm) and every
    f(x_i) is within ftol + 2ε|f(g)| of f(g): the swarm has collapsed onto g. Also stops
    (converged) on a rounding plateau: for 10 consecutive iterations every f(x_i) is within
    2ε|f(g)| of f(g). Near a minimizer with f* ≠ 0, rounding makes f constant on a ball of radius
    ≈ √(2ε|f*|/curvature), which can exceed xtol (f = 10⁵ + ‖x‖²: 4.7e-6); the swarm cannot
    contract inside it because f no longer ranks the particles.

    Draw order: initial positions particle by particle, coordinates in order (n uniforms per
    particle; particle 0 is x0 when a start is available and draws nothing); then initial
    velocities for every particle (n uniforms each); then per iteration, for each particle i in
    order: r₁ (n uniforms) then r₂ (n uniforms).
    """
    _check_max_iter(max_iter)
    if n_particles < 2:
        raise ValueError(f"n_particles must be ≥ 2, got {n_particles}")
    for name, v in (("w", w), ("c1", c1), ("c2", c2)):
        if not v >= 0.0:
            raise ValueError(f"{name} must be ≥ 0, got {v}")
    _check_positive(xtol=xtol, ftol=ftol)
    prob, lo, hi, start = _resolve(problem, x0)
    n = lo.size
    fobj = _Objective(prob.f, scalar=prob.dim == 1)
    rng = Rng(seed)

    xs: list[Vector] = []
    if start is not None:
        _require_inside(start, lo, hi)
        xs.append(start.copy())
    while len(xs) < n_particles:
        xs.append(_uniform_in_box(rng, lo, hi))
    vs = [0.5 * (_uniform_in_box(rng, lo, hi) - x) for x in xs]
    fs = [fobj(x) for x in xs]
    pbest, pbest_f = _rows(xs), list(fs)
    gi = int(np.argmin(pbest_f))
    g, g_f = pbest[gi].copy(), pbest_f[gi]

    def info() -> dict[str, Any]:
        return {
            "best": g.copy(),
            "best_f": g_f,
            "particles": _rows(xs),
            "particles_f": list(fs),
            "velocities": _rows(vs),
            "personal_best": _rows(pbest),
            "personal_best_f": list(pbest_f),
            "global_best": g.copy(),
            "global_best_f": g_f,
        }

    trace = [Step(0, g.copy(), g_f, info=info())]
    if g_f == -math.inf:
        return _unbounded("particle_swarm", g.copy(), trace, 0, fobj.n)
    f_max_history: list[float] = []
    k = 0
    while True:
        k += 1
        for i in range(n_particles):
            r1 = np.array([rng.random() for _ in range(n)])
            r2 = np.array([rng.random() for _ in range(n)])
            v = w * vs[i] + c1 * r1 * (pbest[i] - xs[i]) + c2 * r2 * (g - xs[i])
            x = xs[i] + v
            low, high = x < lo, x > hi
            x = np.where(low, lo, np.where(high, hi, x))
            v = np.where(low | high, 0.0, v)
            xs[i], vs[i] = x, v
        fs = [fobj(x) for x in xs]
        for i in range(n_particles):
            if fs[i] < pbest_f[i]:
                pbest[i], pbest_f[i] = xs[i].copy(), fs[i]
        gi = int(np.argmin(pbest_f))
        g, g_f = pbest[gi].copy(), pbest_f[gi]
        step = float(np.max([np.linalg.norm(v) for v in vs]))
        trace.append(Step(k, g.copy(), g_f, step_size=step, info=info()))
        if g_f == -math.inf:
            return _unbounded("particle_swarm", g.copy(), trace, k, fobj.n)
        done, x_spread, f_spread = _collapsed(xs, fs, g, g_f, xtol, ftol)
        if done:
            msg = f"swarm collapsed: x-spread {x_spread:.3g}, f-spread {f_spread:.3g}; best f = {g_f:.6g}"
            return Result("particle_swarm", g.copy(), g_f, True, msg, k, fobj.n, trace=trace)
        f_max_history.append(max(fs))
        flat, f_range = _on_plateau(f_max_history, g_f)
        if flat:
            msg = _plateau_message("swarm", x_spread, f_range, g_f)
            return Result("particle_swarm", g.copy(), g_f, True, msg, k, fobj.n, trace=trace)
        if k == max_iter:
            return _max_iter("particle_swarm", g.copy(), g_f, max_iter, fobj.n, trace)


# --------------------------------------------------------------------------------------
# Differential evolution
# --------------------------------------------------------------------------------------


def _distinct_index(rng: Rng, size: int, exclude: list[int]) -> int:
    """Uniform index in [0, size) not in ``exclude``, by rejection (one uniform per draw)."""
    while True:
        r = rng.integers(size)
        if r not in exclude:
            return r


@register(
    id="differential_evolution",
    family="global",
    name="Differential evolution",
    params=(
        ParamSpec("pop_size", 20, kind="int", min=4, max=200, help="Population size NP."),
        ParamSpec("F", 0.8, min=0.0, max=2.0, help="Differential weight F (mutation scale)."),
        ParamSpec("CR", 0.9, min=0.0, max=1.0, help="Crossover probability CR."),
        ParamSpec(
            "strategy",
            "rand/1/bin",
            kind="choice",
            choices=("rand/1/bin", "best/1/bin"),
            help="Base vector: a random member (rand) or the best member (best).",
        ),
        *_TOL_PARAMS,
        ParamSpec("max_iter", 300, kind="int", min=1, max=100_000, help="Generation limit."),
    ),
    needs=("f",),
    order="no rate; a population method",
    summary="Mutate with scaled differences of population members, cross over, keep the better.",
    references=(
        "Storn & Price (1997), J. Global Optim. 11(4), 341–359",
        "Price, Storn & Lampinen (2005), Differential Evolution: A Practical Approach, Springer",
    ),
    deterministic=False,
)
def differential_evolution(
    problem: Problem | VectorFn,
    *,
    x0: Any = None,
    seed: int = 0,
    pop_size: int = 20,
    F: float = 0.8,
    CR: float = 0.9,
    strategy: str = "rand/1/bin",
    xtol: float = 1e-6,
    ftol: float = 1e-8,
    max_iter: int = 300,
) -> Result:
    """Differential evolution DE/rand/1/bin or DE/best/1/bin (Storn & Price 1997).

    For every target x_i of the generation (i = 0, …, NP − 1), with r₁, r₂, r₃ distinct and ≠ i:

        mutant   v_i = x_{r₁} + F(x_{r₂} − x_{r₃})          (rand/1)
                 v_i = x_best + F(x_{r₁} − x_{r₂})          (best/1, best of the old generation)
        trial    u_ij = v_ij if U_j ≤ CR or j = j_rand, else x_ij   (binomial crossover)
        select   x_i' = u_i if f(u_i) ≤ f(x_i), else x_i.

    The generation is synchronous, as in Storn & Price: all mutants use the old population and
    the winners form the next one.

    NOTE: a trial component outside [lo_j, hi_j] is reset halfway between the violated bound
    and the target's component, u_ij = (lo_j + x_ij)/2 (or (hi_j + x_ij)/2). This is a
    deterministic form of the bounce-back repair of Price, Storn & Lampinen (2005); Storn &
    Price (1997) do not treat bounds.
    NOTE: selection accepts ties (≤), as in Price, Storn & Lampinen (2005), so the population
    can drift across plateaus; Storn & Price (1997) state a strict decrease.

    Stops (converged) when every member is within xtol + 2ε‖x_best‖∞ of the best member
    (∞-norm) and every f within ftol + 2ε|f_best| of f_best: the population has collapsed.
    Also stops (converged) on a rounding plateau: for 10 consecutive generations every f is
    within 2ε|f_best| of f_best. Rounding in f makes f constant near a minimizer with f* ≠ 0
    on a ball of radius ≈ √(2ε|f*|/curvature), which can exceed xtol; there selection accepts
    every tie and the population cannot contract.

    Draw order: initial members in order, n uniforms each (member 0 is x0 when a start is
    available and draws nothing). Per generation, for each target i in order: r₁, r₂, (r₃)
    by rejection (``rng.integers(NP)``, redrawn while equal to i or an earlier r), then
    j_rand = ``rng.integers(n)``, then U_1, …, U_n (``rng.random()``, all n always drawn).
    """
    _check_max_iter(max_iter)
    if strategy not in ("rand/1/bin", "best/1/bin"):
        raise ValueError(f"strategy must be 'rand/1/bin' or 'best/1/bin', got {strategy!r}")
    need = 4 if strategy == "rand/1/bin" else 3
    if pop_size < need:
        raise ValueError(f"{strategy} needs pop_size ≥ {need}, got {pop_size}")
    if not F >= 0.0:
        raise ValueError(f"F must be ≥ 0, got {F}")
    if not 0.0 <= CR <= 1.0:
        raise ValueError(f"CR must lie in [0, 1], got {CR}")
    _check_positive(xtol=xtol, ftol=ftol)
    prob, lo, hi, start = _resolve(problem, x0)
    n = lo.size
    fobj = _Objective(prob.f, scalar=prob.dim == 1)
    rng = Rng(seed)

    pop: list[Vector] = []
    if start is not None:
        _require_inside(start, lo, hi)
        pop.append(start.copy())
    while len(pop) < pop_size:
        pop.append(_uniform_in_box(rng, lo, hi))
    pop_f = [fobj(x) for x in pop]
    bi = int(np.argmin(pop_f))

    def info(
        mutants: list[Vector], trials: list[Vector], trials_f: list[float], accepted: list[bool]
    ) -> dict[str, Any]:
        return {
            "best": pop[bi].copy(),
            "best_f": pop_f[bi],
            "population": _rows(pop),
            "population_f": list(pop_f),
            "mutants": mutants,
            "trials": trials,
            "trials_f": trials_f,
            "accepted": accepted,
            "best_index": bi,
        }

    trace = [Step(0, pop[bi].copy(), pop_f[bi], info=info([], [], [], []))]
    if pop_f[bi] == -math.inf:
        return _unbounded("differential_evolution", pop[bi].copy(), trace, 0, fobj.n)
    f_max_history: list[float] = []
    k = 0
    while True:
        k += 1
        mutants: list[Vector] = []
        trials: list[Vector] = []
        trials_f: list[float] = []
        accepted: list[bool] = []
        new_pop, new_f = _rows(pop), list(pop_f)
        x_best = pop[bi]
        for i in range(pop_size):
            r1 = _distinct_index(rng, pop_size, [i])
            r2 = _distinct_index(rng, pop_size, [i, r1])
            if strategy == "rand/1/bin":
                r3 = _distinct_index(rng, pop_size, [i, r1, r2])
                v = pop[r1] + F * (pop[r2] - pop[r3])
            else:
                v = x_best + F * (pop[r1] - pop[r2])
            j_rand = rng.integers(n)
            cross = np.array([rng.random() <= CR or j == j_rand for j in range(n)])
            u = np.where(cross, v, pop[i])
            u = np.where(u < lo, 0.5 * (lo + pop[i]), u)
            u = np.where(u > hi, 0.5 * (hi + pop[i]), u)
            f_u = fobj(u)
            take = f_u <= pop_f[i]
            if take:
                new_pop[i], new_f[i] = u.copy(), f_u
            mutants.append(v)
            trials.append(u)
            trials_f.append(f_u)
            accepted.append(bool(take))
        pop, pop_f = new_pop, new_f
        bi = int(np.argmin(pop_f))
        trace.append(
            Step(k, pop[bi].copy(), pop_f[bi], info=info(mutants, trials, trials_f, accepted))
        )
        if pop_f[bi] == -math.inf:
            return _unbounded("differential_evolution", pop[bi].copy(), trace, k, fobj.n)
        done, x_spread, f_spread = _collapsed(pop, pop_f, pop[bi], pop_f[bi], xtol, ftol)
        if done:
            msg = (
                f"population collapsed: x-spread {x_spread:.3g}, f-spread {f_spread:.3g}; "
                f"best f = {pop_f[bi]:.6g}"
            )
            return Result(
                "differential_evolution",
                pop[bi].copy(),
                pop_f[bi],
                True,
                msg,
                k,
                fobj.n,
                trace=trace,
            )
        f_max_history.append(max(pop_f))
        flat, f_range = _on_plateau(f_max_history, pop_f[bi])
        if flat:
            msg = _plateau_message("population", x_spread, f_range, pop_f[bi])
            return Result(
                "differential_evolution",
                pop[bi].copy(),
                pop_f[bi],
                True,
                msg,
                k,
                fobj.n,
                trace=trace,
            )
        if k == max_iter:
            return _max_iter(
                "differential_evolution", pop[bi].copy(), pop_f[bi], max_iter, fobj.n, trace
            )


# --------------------------------------------------------------------------------------
# CMA-ES
# --------------------------------------------------------------------------------------

#: Hansen (2016), App. B.3 ConditionCov: stop when κ(C) exceeds this.
_CMA_MAX_COND = 1e14


def _sym_sqrt(eigval: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Symmetric square root C^{1/2} = B D Bᵀ from the eigendecomposition C = B D² Bᵀ."""
    return (B * np.sqrt(eigval)) @ B.T


@register(
    id="cma_es",
    family="global",
    name="CMA-ES",
    params=(
        ParamSpec(
            "sigma0",
            0.3,
            min=1e-3,
            max=1.0,
            log=True,
            help="Initial step size σ₀ as a fraction of the mean box width (Hansen: ≈ 0.3).",
        ),
        ParamSpec(
            "pop_size",
            0,
            kind="int",
            min=0,
            max=200,
            help=(
                "Population size λ (0 or 1: the default 4 + ⌊3 ln n⌋, since λ ≥ 2 is needed; "
                "larger is more global)."
            ),
        ),
        ParamSpec(
            "xtol",
            1e-10,
            min=1e-14,
            max=1e-2,
            log=True,
            help="TolX: stop when σ·√C_ii and σ·|p_c,i| are ≤ xtol for every i.",
        ),
        ParamSpec(
            "ftol",
            1e-12,
            min=1e-15,
            max=1e-2,
            log=True,
            help="TolFun: stop when the recent best f values and the generation's f span ≤ ftol.",
        ),
        ParamSpec("max_iter", 1000, kind="int", min=1, max=100_000, help="Generation limit."),
    ),
    needs=("f",),
    order="linear (log-linear in f) on convex quadratics, invariant to rotations and scaling",
    summary="Sample a Gaussian, move its mean to the best samples and learn its covariance from them.",
    references=(
        "Hansen (2016), The CMA Evolution Strategy: A Tutorial, arXiv:1604.00772, Fig. 6, Table 1",
        "Hansen & Ostermeier (2001), Evol. Comput. 9(2), 159–195",
    ),
    deterministic=False,
)
def cma_es(
    problem: Problem | VectorFn,
    *,
    x0: Any = None,
    seed: int = 0,
    sigma0: float = 0.3,
    pop_size: int = 0,
    xtol: float = 1e-10,
    ftol: float = 1e-12,
    max_iter: int = 1000,
) -> Result:
    """The (μ/μ_w, λ)-CMA-ES of Hansen's tutorial (2016), Fig. 6, with the Table 1 defaults.

    Parameters (Table 1, eqs. 48–58): λ = 4 + ⌊3 ln n⌋, μ = ⌊λ/2⌋,
    w_i ∝ ln((λ + 1)/2) − ln i (i ≤ μ, Σw_i = 1), μ_eff = 1/Σw_i²,
    c_σ = (μ_eff + 2)/(n + μ_eff + 5), d_σ = 1 + 2 max(0, √((μ_eff − 1)/(n + 1)) − 1) + c_σ,
    c_c = (4 + μ_eff/n)/(n + 4 + 2μ_eff/n), c₁ = 2/((n + 1.3)² + μ_eff),
    c_μ = min(1 − c₁, 2(¼ + μ_eff + 1/μ_eff − 2)/((n + 2)² + μ_eff)), c_m = 1.

    Generation g = 1, 2, …:

        z_k ~ N(0, I),  y_k = C^{1/2} z_k,  x_k = m + σ y_k            k = 1, …, λ   (38–40)
        ⟨y⟩_w = Σ_{i≤μ} w_i y_{i:λ};  m ← m + σ⟨y⟩_w                               (41–42)
        p_σ ← (1 − c_σ)p_σ + √(c_σ(2 − c_σ)μ_eff) C^{−1/2}⟨y⟩_w                     (43)
        σ ← σ exp((c_σ/d_σ)(‖p_σ‖/E‖N(0, I)‖ − 1))                                  (44)
        p_c ← (1 − c_c)p_c + h_σ √(c_c(2 − c_c)μ_eff) ⟨y⟩_w                         (45)
        C ← (1 + c₁δ(h_σ) − c₁ − c_μ)C + c₁ p_c p_cᵀ + c_μ Σ_{i≤μ} w_i y_{i:λ}y_{i:λ}ᵀ  (47)

    with h_σ = 1 iff ‖p_σ‖/√(1 − (1 − c_σ)^{2g}) < (1.4 + 2/(n + 1))E‖N(0, I)‖,
    δ(h_σ) = (1 − h_σ)c_c(2 − c_c), and E‖N(0, I)‖ ≈ √n(1 − 1/(4n) + 1/(21n²)). The C used
    in (47) and the y_k are those of the old distribution. Start: m = x0,
    σ = sigma0 · mean_j(hi_j − lo_j), C = I, p_σ = p_c = 0; f(x0) is evaluated once so that
    the best point is defined at k = 0.

    NOTE: only the μ positive weights are used (w_i = 0 for i > μ), as in the tutorial's own
    reference code (App. C); Table 1's default adds negative ("active") weights.
    NOTE: samples are y_k = C^{1/2}z_k with the symmetric root C^{1/2} = B D Bᵀ instead of
    B D z_k. The distribution N(0, C) is the same and C^{−1/2}y_k = z_k still holds (so (43)
    uses Σ w_i z_{i:λ}), but the samples do not depend on the sign and order convention of
    the eigenvectors, which makes the method reproducible across implementations.
    NOTE: the exponent 2g counts this generation (g = 1 first), as in the tutorial's code.
    NOTE: no box handling (the tutorial has none); samples may leave the box.
    NOTE: ``pop_size`` 0 or 1 selects the default λ (λ = 1 gives μ = 0 parents, so it is not a
    valid population), which makes every value of the ParamSpec range [0, 200] valid.

    ``Result.x`` is the best sample ever evaluated (the tutorial suggests the best-ever point or
    the final mean). It need not be a stationary point: the mean may converge into another basin
    after an early lucky sample; ``info["mean"]`` shows where the distribution went.

    Stops (converged) on TolX (σ√C_ii ≤ xtol and σ|p_c,i| ≤ xtol for all i) or TolFun (the range
    of the best f of the last 10 + ⌈30n/λ⌉ generations together with all f of the current one
    is ≤ ftol), both from the tutorial's App. B.3. κ(C) > 10¹⁴ (ConditionCov), a σ or C that is
    not finite, or a σ that underflows to 0 stops with converged=False. C stays positive
    definite in exact arithmetic (eq. 47 is a combination with weights ≥ 0 of C and positive
    semidefinite terms); a computed λ_min(C) ≤ 0 is reported as a loss of definiteness.
    The eigendecomposition C = B D² Bᵀ of the check is reused for C^{1/2} in the next generation.

    Draw order: per generation, z_1, …, z_λ, each z_k's n coordinates in order
    (``rng.normal()``, 2 uniforms each): 2nλ uniforms.
    """
    _check_max_iter(max_iter)
    _check_positive(sigma0=sigma0, xtol=xtol, ftol=ftol)
    if pop_size < 0:
        raise ValueError(f"pop_size must be ≥ 0 (0 or 1: the default λ), got {pop_size}")
    prob, lo, hi, start = _resolve(problem, x0)
    m = _require_start(start, prob).copy()
    n = m.size
    fobj = _Objective(prob.f, scalar=prob.dim == 1)
    rng = Rng(seed)

    lam = pop_size if pop_size >= 2 else 4 + math.floor(3.0 * math.log(n))
    mu = lam // 2
    w_raw = np.array([math.log((lam + 1) / 2.0) - math.log(i) for i in range(1, mu + 1)])
    weights = w_raw / np.sum(w_raw)  # (μ,)
    mueff = 1.0 / float(np.sum(weights**2))
    c_sigma = (mueff + 2.0) / (n + mueff + 5.0)
    d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + c_sigma
    c_c = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
    c_1 = 2.0 / ((n + 1.3) ** 2 + mueff)
    c_mu = min(1.0 - c_1, 2.0 * (0.25 + mueff + 1.0 / mueff - 2.0) / ((n + 2.0) ** 2 + mueff))
    chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
    hist_len = 10 + math.ceil(30.0 * n / lam)

    sigma = sigma0 * float(np.mean(hi - lo))
    C = np.eye(n)
    eigval, B = np.linalg.eigh(C)  # C = B diag(eigval) Bᵀ, recomputed after every update
    p_sigma = np.zeros(n)
    p_c = np.zeros(n)
    # f(x0) = +∞ (or NaN, mapped to +∞) is allowed: the sampling does not use it, and the
    # first finite sample becomes the best point.
    f_m = fobj(m)
    if f_m == -math.inf:
        return _unbounded("cma_es", m.copy(), [Step(0, m.copy(), f_m)], 0, fobj.n)
    best, f_best = m.copy(), f_m
    best_history: list[float] = []

    def info(
        sample: tuple[Vector, float, np.ndarray],
        pop: list[Vector],
        pop_f: list[float],
        selected: list[int],
        h_sigma: bool | None,
    ) -> dict[str, Any]:
        return {
            "best": best.copy(),
            "best_f": f_best,
            "mean": m.copy(),
            "sigma": sigma,
            "covariance": C.copy(),
            "sample_mean": sample[0],
            "sample_sigma": sample[1],
            "sample_covariance": sample[2],
            "population": pop,
            "population_f": pop_f,
            "selected": selected,
            "p_sigma": p_sigma.copy(),
            "p_c": p_c.copy(),
            "h_sigma": h_sigma,
        }

    trace = [
        Step(
            0,
            best.copy(),
            f_best,
            step_size=sigma,
            info=info((m.copy(), sigma, C.copy()), [], [], [], None),
        )
    ]
    k = 0
    while True:
        k += 1
        sqrt_C = _sym_sqrt(eigval, B)
        sample = (m.copy(), sigma, C.copy())
        Z = np.array([[rng.normal() for _ in range(n)] for _ in range(lam)])  # (λ, n)
        Y = Z @ sqrt_C  # (λ, n); rows y_k = C^{1/2} z_k (C^{1/2} is symmetric)
        X = m + sigma * Y  # (λ, n)
        fs = [fobj(X[i]) for i in range(lam)]
        order = sorted(range(lam), key=lambda i: fs[i])  # stable: ties keep sampling order
        sel = order[:mu]
        if fs[order[0]] < f_best:
            best, f_best = X[order[0]].copy(), fs[order[0]]
        if f_best == -math.inf:
            trace.append(
                Step(
                    k,
                    best.copy(),
                    f_best,
                    step_size=sigma,
                    info=info(sample, list(X), fs, sel, None),
                )
            )
            return _unbounded("cma_es", best.copy(), trace, k, fobj.n)

        y_w = weights @ Y[sel]  # (n,)  ⟨y⟩_w
        z_w = weights @ Z[sel]  # (n,)  C^{−1/2}⟨y⟩_w
        m = m + sigma * y_w
        p_sigma = (1.0 - c_sigma) * p_sigma + math.sqrt(c_sigma * (2.0 - c_sigma) * mueff) * z_w
        norm_ps = float(np.linalg.norm(p_sigma))
        h_sigma = (
            norm_ps / math.sqrt(1.0 - (1.0 - c_sigma) ** (2 * k)) < (1.4 + 2.0 / (n + 1.0)) * chi_n
        )
        p_c = (1.0 - c_c) * p_c + (1.0 if h_sigma else 0.0) * math.sqrt(
            c_c * (2.0 - c_c) * mueff
        ) * y_w
        delta_h = (0.0 if h_sigma else 1.0) * c_c * (2.0 - c_c)
        rank_mu = (Y[sel].T * weights) @ Y[sel]  # Σ w_i y_i y_iᵀ, (n, n)
        C = (1.0 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * rank_mu
        C = 0.5 * (C + C.T)  # remove rounding asymmetry
        sigma = sigma * math.exp((c_sigma / d_sigma) * (norm_ps / chi_n - 1.0))

        trace.append(
            Step(
                k,
                best.copy(),
                f_best,
                step_size=sigma,
                info=info(sample, list(X), fs, sel, h_sigma),
            )
        )
        best_history.append(fs[order[0]])

        if not (math.isfinite(sigma) and sigma > 0.0 and np.all(np.isfinite(C))):
            msg = f"step size or covariance became non-finite or zero (σ = {sigma!r})"
            return Result("cma_es", best, f_best, False, msg, k, fobj.n, trace=trace)
        eigval, B = np.linalg.eigh(C)
        if not eigval[0] > 0.0:
            msg = f"covariance matrix C lost positive definiteness (λ_min(C) = {eigval[0]:.3g})"
            return Result("cma_es", best, f_best, False, msg, k, fobj.n, trace=trace)
        if eigval[-1] > _CMA_MAX_COND * eigval[0]:
            msg = f"ConditionCov: κ(C) = {eigval[-1] / eigval[0]:.3g} > 1e14"
            return Result("cma_es", best, f_best, False, msg, k, fobj.n, trace=trace)
        sd = sigma * np.sqrt(np.diag(C))
        if float(np.max(sd)) <= xtol and float(np.max(np.abs(sigma * p_c))) <= xtol:
            msg = f"TolX: σ·max√C_ii = {float(np.max(sd)):.3g} ≤ xtol; best f = {f_best:.6g}"
            return Result("cma_es", best, f_best, True, msg, k, fobj.n, trace=trace)
        if len(best_history) >= hist_len:
            recent = best_history[-hist_len:] + fs
            f_range = max(recent) - min(recent)
            if f_range <= ftol:
                msg = f"TolFun: f range {f_range:.3g} ≤ ftol over the last {hist_len} generations"
                return Result("cma_es", best, f_best, True, msg, k, fobj.n, trace=trace)
        if k == max_iter:
            return _max_iter("cma_es", best, f_best, max_iter, fobj.n, trace)


# --------------------------------------------------------------------------------------
# Basin hopping
# --------------------------------------------------------------------------------------

#: Local Nelder–Mead searches: tolerances and iteration limit.
_LOCAL_XTOL = 1e-8
_LOCAL_FTOL = 1e-10
_LOCAL_MAX_ITER = 2000
#: Two local minima are the same if ‖z − m‖∞ ≤ _SAME_MIN_TOL·(1 + ‖m‖∞).
_SAME_MIN_TOL = 1e-4
#: A hop improves the best minimum only if it lowers f_best by more than this·(1 + |f_best|).
_IMPROVE_TOL = 1e-8


@register(
    id="basin_hopping",
    family="global",
    name="Basin hopping",
    params=(
        ParamSpec(
            "T",
            1.0,
            min=1e-3,
            max=1e3,
            log=True,
            help="Temperature of the Metropolis test between local minima.",
        ),
        ParamSpec(
            "step",
            0.1,
            min=1e-3,
            max=1.0,
            log=True,
            help="Hop half-width per coordinate as a fraction of the box width.",
        ),
        ParamSpec(
            "patience",
            20,
            kind="int",
            min=1,
            max=1000,
            help="Stop when the best minimum has not improved for this many hops.",
        ),
        ParamSpec("max_iter", 100, kind="int", min=1, max=10_000, help="Hop limit."),
    ),
    needs=("f",),
    order="no rate; a Monte Carlo walk on local minima",
    summary="Jump randomly, slide downhill with Nelder–Mead, accept the new minimum by Metropolis.",
    references=(
        "Wales & Doye (1997), J. Phys. Chem. A 101(28), 5111–5116",
        "Li & Scheraga (1987), PNAS 84(19), 6611–6615 (Monte Carlo minimization)",
    ),
    deterministic=False,
)
def basin_hopping(
    problem: Problem | VectorFn,
    *,
    x0: Any = None,
    seed: int = 0,
    T: float = 1.0,
    step: float = 0.1,
    patience: int = 20,
    max_iter: int = 100,
) -> Result:
    """Basin hopping (Wales & Doye 1997): a Metropolis walk on the local minima of f.

    Start: x = the local minimum reached from x0. Hop k = 1, 2, …:

        y = x + δ,  δ_j = s_j(2U_j − 1) ~ U[−s_j, s_j],  s_j = step·(hi_j − lo_j)
        z = local minimum from y  (this module's Nelder–Mead, xtol 1e-8, ftol 1e-10,
            initial simplex edge ½·mean_j s_j)
        x ← z with probability min(1, exp(−(f(z) − f(x))/T))       (Metropolis on minima)

    The local search uses ``nelder_mead`` of :mod:`numopt.unconstrained.derivative_free`; when it
    stops at its iteration limit its best vertex is used and ``local_converged`` is False.

    Stops when the best minimum has not improved by more than 10⁻⁸(1 + |f_best|) for
    ``patience`` consecutive hops (SciPy's ``niter_success`` rule; every hop counts, also one
    whose local search did not converge). The stop is ``converged=True`` only when the local
    search that produced the best minimum met its tolerance. Otherwise the best point is not a
    certified local minimum and the result is ``converged=False``, as SciPy's ``basinhopping``
    sets ``success`` from the local result of its lowest minimum.

    Draw order: per hop, U_1, …, U_n (``rng.random()``), then the acceptance uniform
    (``rng.random()``, always drawn): n + 1 uniforms.
    """
    _check_max_iter(max_iter)
    _check_positive(T=T, step=step)
    if patience < 1:
        raise ValueError(f"patience must be ≥ 1, got {patience}")
    prob, lo, hi, start = _resolve(problem, x0)
    x_start = _require_start(start, prob)
    n = x_start.size
    rng = Rng(seed)
    s = step * (hi - lo)  # (n,)
    local_step = 0.5 * float(np.mean(s))
    nfev = 0

    def local(y: Vector) -> tuple[Vector, float, list[Vector], bool]:
        nonlocal nfev
        res = nelder_mead(
            prob,
            x0=y,
            xtol=_LOCAL_XTOL,
            ftol=_LOCAL_FTOL,
            initial_step=local_step,
            max_iter=_LOCAL_MAX_ITER,
        )
        nfev += res.n_fev
        fz = float(res.fun) if res.fun is not None else math.inf
        fz = math.inf if math.isnan(fz) else fz
        return as_vector(res.x), fz, [as_vector(st.x) for st in res.trace], res.converged

    x, fx, path, local_ok = local(x_start)
    if not math.isfinite(fx):
        msg = f"f is not finite at the first local minimum (f = {fx!r}); cannot start"
        if fx == -math.inf:
            msg = "f = -inf on the first local search: f is unbounded below"
        trace0 = [Step(0, x.copy(), fx)]
        return Result("basin_hopping", x, fx, False, msg, 0, nfev, trace=trace0)
    best, f_best, best_ok = x.copy(), fx, local_ok
    minima: list[dict[str, Any]] = [{"x": x.copy(), "f": fx, "hits": 1}]
    stall = 0

    def record(z: Vector, fz: float) -> None:
        for mrec in minima:
            m = mrec["x"]
            if float(np.max(np.abs(z - m))) <= _SAME_MIN_TOL * (1.0 + float(np.max(np.abs(m)))):
                mrec["hits"] += 1
                if fz < mrec["f"]:
                    mrec["x"], mrec["f"] = z.copy(), fz
                return
        minima.append({"x": z.copy(), "f": fz, "hits": 1})

    def info(
        y: Vector,
        z: Vector,
        fz: float,
        path: list[Vector],
        ok: bool,
        p: float | None,
        acc: bool | None,
    ) -> dict[str, Any]:
        return {
            "best": best.copy(),
            "best_f": f_best,
            "current": x.copy(),
            "current_f": fx,
            "start": y.copy(),
            "local_min": z.copy(),
            "local_f": fz,
            "local_path": path,
            "local_converged": ok,
            "accept_prob": p,
            "accepted": acc,
            "minima": [{"x": r["x"].copy(), "f": r["f"], "hits": r["hits"]} for r in minima],
            "stall": stall,
        }

    trace = [Step(0, best.copy(), f_best, info=info(x_start, x, fx, path, local_ok, None, None))]
    k = 0
    while True:
        k += 1
        u = np.array([rng.random() for _ in range(n)])
        u_acc = rng.random()
        y = x + s * (2.0 * u - 1.0)
        z, fz, path, local_ok = local(y)
        if fz == -math.inf:
            trace.append(Step(k, z.copy(), fz, info=info(y, z, fz, path, local_ok, None, None)))
            return _unbounded("basin_hopping", z, trace, k, nfev)
        delta = fz - fx
        p_acc = 1.0 if delta <= 0.0 else math.exp(-delta / T)
        accepted = u_acc < p_acc
        if math.isfinite(fz):
            record(z, fz)
        if fz < f_best - _IMPROVE_TOL * (1.0 + abs(f_best)):
            stall = 0
        else:
            stall += 1
        if fz < f_best:
            best, f_best, best_ok = z.copy(), fz, local_ok
        if accepted:
            x, fx = z, fz
        trace.append(
            Step(
                k,
                best.copy(),
                f_best,
                step_size=float(np.linalg.norm(y - z)),
                info=info(y, z, fz, path, local_ok, p_acc, accepted),
            )
        )
        if stall >= patience:
            msg = f"best minimum unchanged for {patience} hops; best f = {f_best:.6g}"
            if not best_ok:
                msg += (
                    f", but the local Nelder–Mead search that found it reached its limit of "
                    f"{_LOCAL_MAX_ITER} iterations, so the best point is not a converged "
                    "local minimum"
                )
            return Result("basin_hopping", best, f_best, best_ok, msg, k, nfev, trace=trace)
        if k == max_iter:
            return _max_iter("basin_hopping", best, f_best, max_iter, nfev, trace)


#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("simulated_annealing", "himmelblau", {"alpha": 0.96}),
    ("simulated_annealing", "rastrigin", {"alpha": 0.96, "T0": 3.0}),
    ("particle_swarm", "rastrigin", {"n_particles": 12, "xtol": 1e-4, "ftol": 1e-6}),
    ("particle_swarm", "ackley", {"n_particles": 12, "xtol": 1e-4, "ftol": 1e-6}),
    ("differential_evolution", "himmelblau", {}),
    ("differential_evolution", "rastrigin", {"strategy": "best/1/bin"}),
    ("cma_es", "ackley", {}),
    ("basin_hopping", "rastrigin", {"max_iter": 80}),
]
