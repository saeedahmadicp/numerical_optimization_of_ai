"""Benchmarking: run solvers on a set of problems and summarize them with profiles.

The benchmark protocol follows Moré & Wild (2009). A *problem instance* p is a problem with a
fixed start point x₀ (and a seed, for stochastic solvers). Every solver s runs on p with a
budget of μ_f cost units. A wrapper records every evaluation the solver makes, so the history
of the best value found against the cost spent is exact (it does not depend on how a method
fills its trace).

Cost models (``cost=``):

* ``"nfev"`` — one unit per evaluation of f (or of the residual r, for least squares);
  gradients and Hessians are free.
* ``"nfev+n*ngev"`` — the Moré–Wild "simplex gradient" convention: one gradient (or one
  Jacobian) costs n function evaluations, the price of a forward-difference gradient.

``# NOTE:`` Hessian evaluations are free under both models (the spec defines no price for
them); every run records ``n_hev`` so a reader can see how many it used.

Convergence test (Moré & Wild 2009, eq. 2.2). Solver s solves p at the first cost where the
best value found satisfies

    f(x) ≤ f_L + τ (f(x₀) − f_L),        0 < τ < 1,

where f_L is the known global minimum of the problem (``problem.extra["f_min"]``) or, when it
is unknown, the smallest value found by any solver within the budget (or f(x₀) if that is
smaller). t_{p,s} is that cost; t_{p,s} = ∞ when s never passes the test.

Performance profile (Dolan & Moré 2002). With r_{p,s} = t_{p,s} / min_σ t_{p,σ} (∞ when s
fails on p),

    ρ_s(α) = |{p ∈ P : r_{p,s} ≤ α}| / |P|,        α ≥ 1.

ρ_s(1) is the fraction of problems on which s is (one of) the cheapest; ρ_s(∞) is the
fraction s solves at all.

Data profile (Moré & Wild 2009, eq. 2.7). With n_p the dimension of p,

    d_s(κ) = |{p ∈ P : t_{p,s} / (n_p + 1) ≤ κ}| / |P|,

so κ counts *simplex gradients* (n_p + 1 evaluations each). Data profiles do not compare
solvers with each other, so adding a solver never changes another solver's curve.

Both profiles are right-continuous step functions of α (κ). The default grids contain every
break point, so the values at the grid points describe the step function exactly.

Seeds: a stochastic solver (a registered method with ``deterministic=False``, or a callable
that takes ``seed``) runs once per seed; every other solver runs once and its history is
shared by the seed copies of the instance. The instance set is cases × start points × seeds.

Quick start::

    from numopt import bench

    res = bench.run_benchmark(
        ["bfgs", "nelder_mead", "cma_es"],
        [bench.BenchmarkCase("rosenbrock"), bench.BenchmarkCase("beale", x0s=[[1, 1], [-2, 2]])],
        budget=2000, cost="nfev+n*ngev", seeds=(0, 1),
    )
    perf = bench.performance_profile(res, tau=1e-3)
    data = bench.data_profile(res, tau=1e-3)
    bench.plot_performance_profile(perf, "perf.svg")
    res.save("bench.json")

References:
    E. D. Dolan and J. J. Moré, "Benchmarking optimization software with performance
    profiles", Math. Program. 91 (2002) 201–213.
    J. J. Moré and S. M. Wild, "Benchmarking derivative-free optimization algorithms",
    SIAM J. Optim. 20(1) (2009) 172–191 (eq. 2.2: convergence test; eq. 2.7: data profile).
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core.registry import get_method, run
from .core.types import Problem, Result, as_vector, to_jsonable

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

CostModel = Literal["nfev", "nfev+n*ngev"]
COST_MODELS: tuple[str, ...] = ("nfev", "nfev+n*ngev")
Solver = Callable[..., Result]
MethodArg = str | tuple[str, Solver]

__all__ = [
    "COST_MODELS",
    "BenchmarkCase",
    "BenchmarkResult",
    "Instance",
    "Profile",
    "RunHistory",
    "data_profile",
    "data_profile_from_costs",
    "is_solved",
    "performance_profile",
    "performance_profile_from_costs",
    "plot_data_profile",
    "plot_performance_profile",
    "run_benchmark",
]

# Okabe–Ito colorblind-safe palette (yellow dropped: too light on white), with line styles
# that repeat after the colors so more than seven solvers stay distinguishable.
_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")
_STYLES = ("-", "--", "-.", ":")


# --------------------------------------------------------------------------------------
# Inputs and records
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkCase:
    """A problem (library id or :class:`Problem`) and the start points to run it from.

    ``x0s`` is a sequence of start points; ``None`` (or a ``None`` entry) means the problem's
    default ``x0``. Each start point gives one problem instance.
    """

    problem: str | Problem
    x0s: Sequence[ArrayLike | None] | None = None

    def resolve(self) -> Problem:
        """Return the :class:`Problem` (looking the id up in the library)."""
        if isinstance(self.problem, Problem):
            return self.problem
        from . import problems as problem_lib

        prob = problem_lib.get(self.problem)
        if not isinstance(prob, Problem):
            raise ValueError(f"{self.problem!r} is a {type(prob).__name__}, not a smooth Problem")
        return prob

    def start_points(self, problem: Problem) -> list[NDArray[np.float64]]:
        """The start points as float vectors (defaults resolved)."""
        raw: Sequence[ArrayLike | None] = [None] if self.x0s is None else self.x0s
        if len(raw) == 0:
            raise ValueError(f"{problem.id}: x0s is empty")
        out = []
        for x0 in raw:
            x = problem.x0 if x0 is None else x0
            if x is None:
                raise ValueError(f"{problem.id}: no start point given and no default x0")
            v = as_vector(x)
            if problem.dim and v.size != problem.dim:
                raise ValueError(f"{problem.id}: x0 has {v.size} entries, expected {problem.dim}")
            out.append(v)
        return out


@dataclass(frozen=True)
class Instance:
    """One problem instance p: a problem, a start point and a seed.

    ``f_star`` is the f_L of the convergence test: ``f_known`` when the problem states its
    minimum, else the best value found by any solver (see the module docstring).
    """

    id: str
    problem_id: str
    x0: tuple[float, ...]
    seed: int
    n: int
    f0: float
    f_known: float | None
    f_star: float

    def threshold(self, tau: float) -> float:
        """f_L + τ (f(x₀) − f_L), the right-hand side of Moré–Wild eq. 2.2."""
        return self.f_star + tau * (self.f0 - self.f_star)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Instance:
        return cls(
            id=str(d["id"]),
            problem_id=str(d["problem_id"]),
            x0=tuple(_float(v) for v in d["x0"]),
            seed=int(d["seed"]),
            n=int(d["n"]),
            f0=_float(d["f0"]),
            f_known=None if d["f_known"] is None else _float(d["f_known"]),
            f_star=_float(d["f_star"]),
        )


@dataclass(frozen=True)
class RunHistory:
    """The record of one solver on one instance.

    ``cost[i]`` is the cumulative cost when the best value found dropped to ``best[i]``;
    ``best`` is strictly decreasing, so the history is the step function "best f vs cost".
    ``stopped`` says why the run ended: ``"solver"`` (its own test or max_iter), ``"budget"``
    (the next evaluation would exceed the budget) or ``"error"`` (the solver raised).
    """

    method: str
    instance: str
    cost: NDArray[np.float64]
    best: NDArray[np.float64]
    total_cost: float
    n_fev: int
    n_gev: int
    n_hev: int
    stopped: Literal["solver", "budget", "error"]
    converged: bool
    fun: float | None
    message: str

    def best_at(self, cost: float) -> float:
        """Best value found after spending at most ``cost`` (∞ before the first evaluation)."""
        i = int(np.searchsorted(self.cost, cost, side="right"))
        return float(self.best[i - 1]) if i > 0 else math.inf

    def solve_cost(self, threshold: float) -> float:
        """The first cost at which the best value is ≤ ``threshold`` (∞ if never)."""
        hit = np.flatnonzero(self.best <= threshold)
        return float(self.cost[hit[0]]) if hit.size else math.inf

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(
            {
                "method": self.method,
                "instance": self.instance,
                "cost": self.cost,
                "best": self.best,
                "total_cost": self.total_cost,
                "n_fev": self.n_fev,
                "n_gev": self.n_gev,
                "n_hev": self.n_hev,
                "stopped": self.stopped,
                "converged": self.converged,
                "fun": self.fun,
                "message": self.message,
            }
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RunHistory:
        stopped = str(d["stopped"])
        if stopped not in ("solver", "budget", "error"):
            raise ValueError(f"bad 'stopped' value {stopped!r}")
        return cls(
            method=str(d["method"]),
            instance=str(d["instance"]),
            cost=np.array([_float(v) for v in d["cost"]], dtype=np.float64),
            best=np.array([_float(v) for v in d["best"]], dtype=np.float64),
            total_cost=_float(d["total_cost"]),
            n_fev=int(d["n_fev"]),
            n_gev=int(d["n_gev"]),
            n_hev=int(d["n_hev"]),
            stopped=stopped,  # type: ignore[arg-type]
            converged=bool(d["converged"]),
            fun=None if d["fun"] is None else _float(d["fun"]),
            message=str(d["message"]),
        )


@dataclass
class BenchmarkResult:
    """Every run of a benchmark. ``labels`` fixes the solver order (and plot colors)."""

    labels: tuple[str, ...]
    instances: tuple[Instance, ...]
    runs: dict[tuple[str, str], RunHistory]
    budget: float
    cost_model: str
    seeds: tuple[int, ...] = (0,)
    params: dict[str, dict[str, Any]] = field(default_factory=dict)

    def history(self, label: str, instance: str) -> RunHistory:
        return self.runs[(label, instance)]

    def solve_costs(self, tau: float) -> NDArray[np.float64]:
        """The cost table t_{p,s}, shape (|P|, |S|); ∞ where s does not solve p."""
        _check_tau(tau)
        T = np.full((len(self.instances), len(self.labels)), np.inf)
        for i, inst in enumerate(self.instances):
            thr = inst.threshold(tau)
            for j, label in enumerate(self.labels):
                T[i, j] = self.runs[(label, inst.id)].solve_cost(thr)
        return T

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(
            {
                "format": "numopt.bench/1",
                "labels": list(self.labels),
                "budget": self.budget,
                "cost_model": self.cost_model,
                "seeds": list(self.seeds),
                "params": self.params,
                "instances": [inst.to_dict() for inst in self.instances],
                "runs": [r.to_dict() for r in self.runs.values()],
            }
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> BenchmarkResult:
        if d.get("format") != "numopt.bench/1":
            raise ValueError(f"not a numopt benchmark (format={d.get('format')!r})")
        runs = [RunHistory.from_dict(r) for r in d["runs"]]
        return cls(
            labels=tuple(str(s) for s in d["labels"]),
            instances=tuple(Instance.from_dict(i) for i in d["instances"]),
            runs={(r.method, r.instance): r for r in runs},
            budget=_float(d["budget"]),
            cost_model=str(d["cost_model"]),
            seeds=tuple(int(s) for s in d["seeds"]),
            params={str(k): dict(v) for k, v in d["params"].items()},
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> BenchmarkResult:
        return cls.from_dict(json.loads(text))

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(self.to_json(), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkResult:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def _float(v: Any) -> float:
    """Inverse of :func:`to_jsonable` for floats (None → nan, "inf"/"-inf" → ±inf)."""
    if v is None:
        return math.nan
    if v == "inf":
        return math.inf
    if v == "-inf":
        return -math.inf
    return float(v)


# --------------------------------------------------------------------------------------
# Evaluation recording
# --------------------------------------------------------------------------------------


class _BudgetExhausted(Exception):
    """Raised by the recorder when the next evaluation would exceed the budget."""


class _Recorder:
    """Charge every evaluation against the budget and keep the history of the best f."""

    def __init__(self, budget: float, grad_cost: float) -> None:
        self.budget = budget
        self.grad_cost = grad_cost
        self.cost = 0.0
        self.best = math.inf
        self.costs: list[float] = []
        self.bests: list[float] = []
        self.n_fev = 0
        self.n_gev = 0
        self.n_hev = 0
        self.exhausted = False

    def _charge(self, c: float) -> None:
        # Sticky: once refused, every later evaluation is refused too (a solver that swallows
        # the exception still cannot spend past the budget).
        if self.exhausted or self.cost + c > self.budget:
            self.exhausted = True
            raise _BudgetExhausted
        self.cost += c

    def _record(self, value: float) -> None:
        if value < self.best:  # NaN never improves
            self.best = value
            self.costs.append(self.cost)
            self.bests.append(value)

    def wrap(self, problem: Problem) -> Problem:
        changes: dict[str, Any] = {"f": self._wrap_f(problem.f)}
        if problem.residual is not None:
            changes["residual"] = self._wrap_residual(problem.residual)
        if problem.grad is not None:
            changes["grad"] = self._wrap_grad(problem.grad)
        if problem.jac is not None:
            # A Jacobian of the residual is priced like a gradient (n evaluations of r).
            changes["jac"] = self._wrap_grad(problem.jac)
        if problem.hess is not None:
            changes["hess"] = self._wrap_hess(problem.hess)
        return dataclasses.replace(problem, **changes)

    def _wrap_f(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        def f(*args: Any, **kwargs: Any) -> Any:
            self._charge(1.0)
            self.n_fev += 1
            value = fn(*args, **kwargs)
            self._record(float(value))
            return value

        return f

    def _wrap_residual(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        def residual(*args: Any, **kwargs: Any) -> Any:
            self._charge(1.0)
            self.n_fev += 1
            r = fn(*args, **kwargs)
            v = np.asarray(r, dtype=np.float64).reshape(-1)
            self._record(0.5 * float(v @ v))  # f = ½‖r‖²
            return r

        return residual

    def _wrap_grad(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        def grad(*args: Any, **kwargs: Any) -> Any:
            self._charge(self.grad_cost)
            self.n_gev += 1
            return fn(*args, **kwargs)

        return grad

    def _wrap_hess(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        def hess(*args: Any, **kwargs: Any) -> Any:
            self._charge(0.0)  # NOTE: free under both cost models (see module docstring)
            self.n_hev += 1
            return fn(*args, **kwargs)

        return hess


# --------------------------------------------------------------------------------------
# Running a benchmark
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Solver:
    label: str
    fn: Solver
    method_id: str | None  # registry id, or None for a user callable
    stochastic: bool


def _resolve_solver(m: MethodArg) -> _Solver:
    if isinstance(m, str):
        spec = get_method(m)
        takes_seed = "seed" in inspect.signature(spec.fn).parameters
        return _Solver(m, spec.fn, m, stochastic=(not spec.deterministic) and takes_seed)
    label, fn = m
    if not callable(fn):
        raise TypeError(f"solver {label!r} is not callable")
    params = inspect.signature(fn).parameters
    takes_seed = "seed" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    return _Solver(str(label), fn, None, stochastic=takes_seed)


def _point(problem: Problem, x: NDArray[np.float64]) -> Any:
    return float(x[0]) if problem.dim == 1 else x.copy()


def _x0_arg(problem: Problem, x: NDArray[np.float64]) -> Any:
    return float(x[0]) if problem.dim == 1 else x.tolist()


def _run_one(
    solver: _Solver,
    problem: Problem,
    x0: NDArray[np.float64],
    seed: int | None,
    params: Mapping[str, Any],
    budget: float,
    grad_cost: float,
    instance_id: str,
) -> RunHistory:
    rec = _Recorder(budget, grad_cost)
    wrapped = rec.wrap(problem)
    kw: dict[str, Any] = dict(params)
    kw["x0"] = _x0_arg(problem, x0)
    if seed is not None:
        kw["seed"] = seed
    stopped: Literal["solver", "budget", "error"] = "solver"
    converged = False
    fun: float | None = None
    message = ""
    try:
        res = run(solver.method_id, wrapped, **kw) if solver.method_id else solver.fn(wrapped, **kw)
        converged = bool(res.converged)
        fun = None if res.fun is None else float(res.fun)
        message = res.message
    except _BudgetExhausted:
        stopped = "budget"
        message = f"budget of {budget:g} exhausted"
    except Exception as exc:  # a crashing solver must not abort the whole benchmark
        stopped = "error"
        message = f"{type(exc).__name__}: {exc}"
    if stopped == "solver" and rec.exhausted:  # the solver swallowed _BudgetExhausted
        stopped = "budget"
        converged = False
        message = f"budget of {budget:g} exhausted"
    return RunHistory(
        method=solver.label,
        instance=instance_id,
        cost=np.array(rec.costs, dtype=np.float64),
        best=np.array(rec.bests, dtype=np.float64),
        total_cost=rec.cost,
        n_fev=rec.n_fev,
        n_gev=rec.n_gev,
        n_hev=rec.n_hev,
        stopped=stopped,
        converged=converged,
        fun=fun,
        message=message,
    )


def run_benchmark(
    methods: Sequence[MethodArg],
    cases: Sequence[BenchmarkCase | str | Problem],
    *,
    budget: float,
    cost: CostModel = "nfev",
    params: Mapping[str, Mapping[str, Any]] | None = None,
    seeds: Sequence[int] = (0,),
) -> BenchmarkResult:
    """Run every solver on every instance (case × start point × seed) within ``budget``.

    Args:
        methods: registry ids (``"bfgs"``) or ``(label, solver)`` pairs, where
            ``solver(problem, *, x0, **params) -> Result`` (plus ``seed`` if it takes one).
        cases: :class:`BenchmarkCase` objects; a bare id or Problem means its default x0.
        budget: μ_f, the maximum cost per run, in units of the cost model.
        cost: ``"nfev"`` or ``"nfev+n*ngev"`` (one gradient = n function evaluations).
        params: per-label keyword parameters, e.g. ``{"bfgs": {"max_iter": 10_000}}``.
            A solver stops at its own tolerance or ``max_iter`` before the budget unless
            these allow it to continue.
        seeds: seeds for stochastic solvers; every instance is copied once per seed.

    Returns:
        A :class:`BenchmarkResult` with one :class:`RunHistory` per (solver, instance).
    """
    if cost not in COST_MODELS:
        raise ValueError(f"cost must be one of {COST_MODELS}, got {cost!r}")
    if not (budget > 0 and math.isfinite(budget)):
        raise ValueError(f"budget must be a positive finite number, got {budget!r}")
    if not methods:
        raise ValueError("no methods given")
    if not cases:
        raise ValueError("no cases given")
    seeds = tuple(int(s) for s in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be a non-empty sequence of distinct ints, got {seeds}")
    params = {k: dict(v) for k, v in (params or {}).items()}

    solvers = [_resolve_solver(m) for m in methods]
    labels = tuple(s.label for s in solvers)
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate solver labels: {labels}")
    unknown = set(params) - set(labels)
    if unknown:
        raise ValueError(f"params given for unknown solver(s) {sorted(unknown)}")

    norm_cases = [c if isinstance(c, BenchmarkCase) else BenchmarkCase(c) for c in cases]
    runs: dict[tuple[str, str], RunHistory] = {}
    pending: list[tuple[str, str, tuple[float, ...], int, int, float, float | None]] = []

    for case in norm_cases:
        problem = case.resolve()
        starts = case.start_points(problem)
        n = int(problem.dim) if problem.dim else int(starts[0].size)
        grad_cost = float(n) if cost == "nfev+n*ngev" else 0.0
        f_min = problem.extra.get("f_min")
        f_known = None if f_min is None else float(f_min)
        for i, x0 in enumerate(starts):
            f0 = float(problem.f(_point(problem, x0)))
            if not math.isfinite(f0):
                raise ValueError(f"{problem.id}: f(x0) = {f0} is not finite")
            base = problem.id + (f"[{i}]" if len(starts) > 1 else "")
            shared: dict[str, RunHistory] = {}  # deterministic solvers: one run for all seeds
            for seed in seeds:
                iid = base + (f"@seed{seed}" if len(seeds) > 1 else "")
                if any(p[0] == iid for p in pending):
                    raise ValueError(f"duplicate instance id {iid!r} (same problem twice?)")
                for s in solvers:
                    if s.stochastic or s.label not in shared:
                        h = _run_one(
                            s,
                            problem,
                            x0,
                            seed if s.stochastic else None,
                            params.get(s.label, {}),
                            float(budget),
                            grad_cost,
                            iid,
                        )
                        if not s.stochastic:
                            shared[s.label] = h
                    else:
                        h = dataclasses.replace(shared[s.label], instance=iid)
                    runs[(s.label, iid)] = h
                pending.append((iid, problem.id, tuple(x0.tolist()), seed, n, f0, f_known))

    instances = []
    for iid, pid, x0t, seed, n, f0, f_known in pending:
        # f_L: the smallest value seen on this instance (f(x0) included, as every solver
        # starts there in Moré–Wild's setting).
        f_found = min(
            [f0]
            + [float(runs[(lab, iid)].best[-1]) for lab in labels if runs[(lab, iid)].best.size]
        )
        # NOTE: when a solver goes below the stated minimum (rounding, or a wrong f_min),
        # f_L is the value found so the test stays satisfiable; otherwise f_L = f_known.
        f_star = f_known if f_known is not None and f_known <= f_found else f_found
        instances.append(Instance(iid, pid, x0t, seed, n, f0, f_known, f_star))

    return BenchmarkResult(
        labels=labels,
        instances=tuple(instances),
        runs=runs,
        budget=float(budget),
        cost_model=cost,
        seeds=seeds,
        params=to_jsonable(params),
    )


# --------------------------------------------------------------------------------------
# Convergence test and profiles
# --------------------------------------------------------------------------------------


def _check_tau(tau: float) -> None:
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must be in (0, 1), got {tau!r}")


def is_solved(f: float, f0: float, f_star: float, tau: float) -> bool:
    """Moré–Wild (2009) eq. 2.2: ``f ≤ f_L + τ (f(x₀) − f_L)``."""
    _check_tau(tau)
    return bool(f <= f_star + tau * (f0 - f_star))


@dataclass(frozen=True)
class Profile:
    """A performance or data profile, sampled on a grid that contains every break point.

    ``y[s, j]`` is ρ_s(x[j]) (or d_s(x[j])); ``solved[s]`` is the limit as x → ∞, the
    fraction of instances solver s solves within the budget.
    """

    kind: Literal["performance", "data"]
    labels: tuple[str, ...]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    solved: NDArray[np.float64]
    tau: float | None = None

    def at(self, value: float) -> NDArray[np.float64]:
        """The profile of every solver at ``value`` (exact if ``value`` is ≥ x[0])."""
        i = int(np.searchsorted(self.x, value, side="right"))
        if i == 0:
            raise ValueError(f"{value} is below the grid start {self.x[0]}")
        return self.y[:, i - 1].copy()


def _check_costs(costs: ArrayLike) -> NDArray[np.float64]:
    T = np.array(costs, dtype=np.float64)
    if T.ndim != 2 or T.shape[0] == 0 or T.shape[1] == 0:
        raise ValueError(f"costs must be a non-empty (problems, solvers) table, got {T.shape}")
    if np.isnan(T).any() or (T <= 0).any():
        raise ValueError("costs must be > 0 (use inf for 'not solved'); NaN is not allowed")
    return T


def _labels(labels: Sequence[str] | None, S: int) -> tuple[str, ...]:
    out = tuple(f"s{j}" for j in range(S)) if labels is None else tuple(labels)
    if len(out) != S:
        raise ValueError(f"{len(out)} labels for {S} solvers")
    return out


def _step_values(values: NDArray[np.float64], grid: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fraction of ``values[:, s]`` ≤ each grid point, for every column s: shape (S, G)."""
    P, S = values.shape
    y = np.empty((S, grid.size))
    for s in range(S):
        col = np.sort(values[:, s])  # inf sorts last and is never ≤ a finite grid point
        y[s] = np.searchsorted(col, grid, side="right") / P
    return y


def _log2_grid(a: int, hi: float, breaks: NDArray[np.float64]) -> NDArray[np.float64]:
    """A log₂-spaced grid from 2^a to one octave past ``hi``, plus every break point."""
    b = max(math.ceil(math.log2(hi)), a) + 1
    base = np.exp2(np.linspace(a, b, 8 * (b - a) + 1))
    return np.unique(np.concatenate([base, breaks]))


def performance_profile_from_costs(
    costs: ArrayLike,
    labels: Sequence[str] | None = None,
    *,
    alphas: ArrayLike | None = None,
    tau: float | None = None,
) -> Profile:
    """Dolan–Moré performance profile ρ_s(α) of a cost table ``costs[p, s]`` (∞ = failed).

    The default α grid runs over [1, 2·max r] on a log₂ scale and contains every finite
    performance ratio r_{p,s}, so it samples the step function at all its jumps.
    """
    T = _check_costs(costs)
    S = T.shape[1]
    best = T.min(axis=1, keepdims=True)  # (P, 1); ∞ when no solver solves p
    with np.errstate(invalid="ignore"):
        R = np.where(np.isfinite(T), T / best, np.inf)  # (P, S) performance ratios
    finite = R[np.isfinite(R)]
    if alphas is None:
        grid = _log2_grid(0, float(finite.max()) if finite.size else 1.0, finite)
    else:
        grid = np.unique(np.asarray(alphas, dtype=np.float64))
        if grid.size == 0 or grid[0] < 1.0:
            raise ValueError("alphas must be non-empty and ≥ 1")
    return Profile(
        kind="performance",
        labels=_labels(labels, S),
        x=grid,
        y=_step_values(R, grid),
        solved=np.isfinite(T).mean(axis=0),
        tau=tau,
    )


def data_profile_from_costs(
    costs: ArrayLike,
    n: ArrayLike,
    labels: Sequence[str] | None = None,
    *,
    kappas: ArrayLike | None = None,
    tau: float | None = None,
) -> Profile:
    """Moré–Wild data profile d_s(κ) of ``costs[p, s]`` with problem dimensions ``n[p]``.

    κ is measured in simplex gradients: t_{p,s} / (n_p + 1).
    """
    T = _check_costs(costs)
    P, S = T.shape
    n_arr = np.asarray(n, dtype=np.float64).reshape(-1)
    if n_arr.shape != (P,) or (n_arr < 1).any():
        raise ValueError(f"n must hold one dimension ≥ 1 per problem ({P}), got {n_arr}")
    K = T / (n_arr[:, None] + 1.0)  # (P, S), ∞ stays ∞
    finite = K[np.isfinite(K)]
    if kappas is None:
        # Start strictly below the first jump so the plot shows the zero level.
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (1.0, 1.0)
        grid = _log2_grid(math.ceil(math.log2(lo)) - 1, hi, finite)
    else:
        grid = np.unique(np.asarray(kappas, dtype=np.float64))
        if grid.size == 0 or grid[0] <= 0.0:
            raise ValueError("kappas must be non-empty and > 0")
    return Profile(
        kind="data",
        labels=_labels(labels, S),
        x=grid,
        y=_step_values(K, grid),
        solved=np.isfinite(T).mean(axis=0),
        tau=tau,
    )


def performance_profile(
    result: BenchmarkResult, tau: float, *, alphas: ArrayLike | None = None
) -> Profile:
    """Performance profile of a benchmark at tolerance τ (Moré–Wild eq. 2.2 test)."""
    return performance_profile_from_costs(
        result.solve_costs(tau), result.labels, alphas=alphas, tau=tau
    )


def data_profile(
    result: BenchmarkResult, tau: float, *, kappas: ArrayLike | None = None
) -> Profile:
    """Data profile of a benchmark at tolerance τ; κ in simplex gradients (cost / (n + 1))."""
    n = [inst.n for inst in result.instances]
    return data_profile_from_costs(
        result.solve_costs(tau), n, result.labels, kappas=kappas, tau=tau
    )


# --------------------------------------------------------------------------------------
# Plots (matplotlib is imported lazily, only here)
# --------------------------------------------------------------------------------------


def _plot_profile(
    profile: Profile,
    path: str | Path | None,
    ax: Axes | None,
    title: str | None,
) -> Figure:
    from matplotlib.figure import Figure

    if ax is None:
        fig = Figure(figsize=(6.4, 4.2), layout="constrained")
        axes = fig.add_subplot()
    else:
        axes = ax
        parent = ax.get_figure()
        if parent is None:
            raise ValueError("ax is not attached to a figure")
        fig = parent if isinstance(parent, Figure) else parent.figure  # SubFigure → root
    for j, label in enumerate(profile.labels):
        axes.step(
            profile.x,
            profile.y[j],
            where="post",
            color=_COLORS[j % len(_COLORS)],
            linestyle=_STYLES[(j // len(_COLORS)) % len(_STYLES)],
            linewidth=1.8,
            label=label,
        )
    axes.set_xscale("log", base=2)
    axes.set_xlim(float(profile.x[0]), float(profile.x[-1]))
    axes.set_ylim(-0.02, 1.02)
    if profile.kind == "performance":
        axes.set_xlabel(r"performance ratio $\alpha$")
        axes.set_ylabel(r"$\rho_s(\alpha)$")
        default_title = "Performance profile"
    else:
        axes.set_xlabel(r"simplex gradients $\kappa$ (cost / (n + 1))")
        axes.set_ylabel(r"$d_s(\kappa)$")
        default_title = "Data profile"
    if title is None:
        title = default_title + (f"  ($\\tau$ = {profile.tau:.0e})" if profile.tau else "")
    axes.set_title(title)
    axes.grid(True, which="major", alpha=0.3)
    axes.legend(loc="lower right", frameon=False)
    if path is not None:
        fig.savefig(str(path), dpi=150)  # format from the suffix: .png, .svg, .pdf
    return fig


def plot_performance_profile(
    profile: Profile,
    path: str | Path | None = None,
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> Figure:
    """Step plot of ρ_s(α) on a log₂ α axis; saves to ``path`` (PNG/SVG/PDF by suffix)."""
    if profile.kind != "performance":
        raise ValueError("expected a performance profile")
    return _plot_profile(profile, path, ax, title)


def plot_data_profile(
    profile: Profile,
    path: str | Path | None = None,
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> Figure:
    """Step plot of d_s(κ) on a log₂ κ axis; saves to ``path`` (PNG/SVG/PDF by suffix)."""
    if profile.kind != "data":
        raise ValueError("expected a data profile")
    return _plot_profile(profile, path, ax, title)
