"""Stochastic gradient methods for finite sums f(w) = (1/n) Σᵢ fᵢ(w).

Every method runs on a :class:`~numopt.problems.stochastic.FiniteSumProblem` and shares one
driver, so the methods differ only in their update rule. Conventions:

* **Sampling.** ``b = min(batch_size, n)``. Epoch e = 1, 2, ... first draws
  ``perm = Rng(seed).permutation(n)`` (one Fisher–Yates shuffle per epoch, from one generator
  that is created once per run), then performs U = ⌈n/b⌉ updates with the mini-batches
  ``perm[jb : (j+1)b]``, j = 0..U−1 (the last one is smaller when b ∤ n). Every sample is used
  exactly once per epoch (sampling without replacement). No other random numbers are drawn.
* **Learning rate.** Update t = 0, 1, ..., T−1 (T = epochs·U) uses η_t from
  :func:`learning_rate`, with τ = t/U the number of epochs completed before the update:
  ``constant`` η₀; ``step`` η₀·½^⌊⌊τ⌋/s⌋ with s = max(1, ⌊epochs/4⌋) (halve the rate every
  s epochs); ``inv_sqrt`` η₀/√(1 + τ); ``cosine`` η₀·½(1 + cos(πt/T)) (Loshchilov & Hutter
  2017, eq. 5 with η_min = 0 and no restarts). Here η₀ = ``lr``.
* **Trace.** Step k is the iterate after k updates (k = 0 is w₀). Step k is recorded when
  k is a multiple of the recording interval ``every`` and at the final iterate, so
  ``n_iter == trace[-1].k`` is the number of updates. ``every = max(record_every,
  ⌈T/(TRACE_MAX − 2)⌉, 1)`` = max(record_every, ⌈T/398⌉, 1): ``record_every = 0`` is
  automatic, and a ``record_every`` that is too small for T is raised, so the trace always
  has ≤ TRACE_MAX = 400 steps (k = 0, ≤ 398 multiples of ``every``, the final step). The
  value used is ``extra["record_every"]``. ``Step.fun`` is the FULL loss f(w_k),
  ``Step.grad_norm`` = ‖∇f(w_k)‖₂ and ``Step.step_size`` the learning rate of the update
  that produced w_k.
* **Stopping test (converged).** ‖∇f(w)‖₂ ≤ ``gtol`` for the full gradient, tested at w₀
  and after the last update of every epoch. Otherwise the run ends after ``epochs`` epochs
  with ``converged=False`` and the message "completed N epochs ...".
* **Start test.** When f(w₀) or ‖∇f(w₀)‖ is not finite (a start point so large that the
  loss overflows, or non-finite data), the run stops at k = 0 with ``converged=False`` and a
  message that begins "not finite at x0". No update was made, so this is not a divergence.
* **Divergence test.** The run stops at once with ``converged=False`` and a message
  "diverged: ...; try a smaller lr" when (a) the iterate, the loss or the gradient is not
  finite, or (b) it blows up: ‖∇f(w_k)‖ > BLOWUP·(1 + ‖∇f(w₀)‖) at a step where ∇f is
  evaluated (recorded steps and epoch ends), or f(w_k) > BLOWUP·(1 + |f(w₀)|) at a recorded
  step, with BLOWUP = 10⁸. Test (b) catches a run that grows geometrically but stays finite
  for all ``epochs`` (e.g. SAG with η·L_max ≫ 1).

  # NOTE: (b) is a heuristic, not a textbook test. A convergent run (or one that stalls in
  # the noise ball of constant-step SGD, radius O(√η)) never raises ‖∇f‖ or f by eight orders of
  # magnitude above the start, and 10⁸ leaves ~10³⁰⁰ before the floats overflow.
* **Counts.** ``n_gev`` counts the component gradients ∇fᵢ that the update rule evaluates
  (the "incremental first-order oracle" count; a full gradient counts n). ``n_fev`` counts
  the full-loss evaluations f(w) made for the trace. The full gradients evaluated only to
  record the trace and to apply the stopping test are counted in
  ``extra["monitor_grad_evals"]`` (each costs n component gradients); they do not change the
  iterates.

Each fᵢ includes the problem's ridge term (λ/2)‖w‖², so SAG, SAGA and SVRG store and
combine full component gradients ∇fᵢ = φ'(aᵢᵀw)aᵢ + λw.

Info keys (every method, every recorded step; ``null`` at k = 0 where marked):
    epoch: int              epoch (1-based) of the update that produced w_k; 0 at k = 0.
    batch: [int] | null     the mini-batch of that update, in sampled order (null when
                            b > 32, and at k = 0).
    stoch_grad: [d] | null  the gradient estimate g_k the update used (see each method).
    full_grad: [d]          ∇f(w_k), the full gradient at the recorded iterate.
    lr: float | null        η of the update that produced w_k.
    update: [d] | null      w_k − w_{k−1}, the displacement of that update.
    ifo: int                component gradients evaluated by the method so far (n_gev).

Additional info keys per method (the state after the update that produced w_k). A key
marked ``| null`` is null at k = 0 only (no update has defined it yet) and a [d] vector at
every k ≥ 1; the unmarked keys are never null.
    sgd_momentum:        velocity: [d]                v_k (zeros at k = 0).
    sgd_nesterov:        velocity: [d]                v_k (zeros at k = 0);
                         lookahead: [d] | null        w_{k−1} + βv_{k−1}, where g_k was evaluated.
    stochastic_adagrad:  accum: [d]                   r_k = Σ g⊙g (zeros at k = 0);
                         scaled_lr: [d] | null        η/(√r_k + ε) per coordinate.
    stochastic_rmsprop:  sq_avg: [d]                  r_k = ρr_{k−1} + (1−ρ)g⊙g (zeros at k = 0);
                         scaled_lr: [d] | null        η/(√r_k + ε).
    stochastic_adam:     m, v: [d]                    first and second moment estimates
                                                      (zeros at k = 0);
                         m_hat, v_hat: [d] | null     their bias-corrected values (0/0 at k = 0);
                         scaled_lr: [d] | null        η/(√v̂_k + ε).
    svrg:                snapshot: [d] | null         the snapshot w̃ of the current epoch (the
                                                      first one is taken when epoch 1 starts);
                         snapshot_grad: [d] | null    μ = ∇f(w̃).
    saga:                table_mean: [d]              (1/n) Σᵢ φᵢ, the mean of the gradient table
                                                      after the update (∇f(w₀) at k = 0).
    sag:                 seen: int                    m, the number of distinct samples visited.
"""

from __future__ import annotations

import math
import numbers
import operator
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.counting import finite
from ..core.registry import ParamSpec, register
from ..core.rng import Rng
from ..core.types import Result, Step, as_vector
from ..problems.stochastic import FiniteSumProblem

Array = NDArray[np.float64]
Schedule = Literal["constant", "step", "inv_sqrt", "cosine"]

SCHEDULES: tuple[str, ...] = ("constant", "step", "inv_sqrt", "cosine")
#: The ``step`` schedule multiplies η by STEP_GAMMA every ⌊epochs/STEP_DROPS⌋ epochs.
STEP_GAMMA = 0.5
STEP_DROPS = 4
#: Largest mini-batch whose indices are written to ``Step.info["batch"]``.
BATCH_INFO_MAX = 32
#: The recording interval is raised as needed so the trace has ≤ TRACE_MAX steps.
TRACE_MAX = 400
#: Divergence test (b): ‖∇f‖ or f above BLOWUP·(1 + its value at w₀) stops the run.
BLOWUP = 1e8


def learning_rate(
    schedule: str, lr: float, t: int, updates_per_epoch: int, total_updates: int
) -> float:
    """η_t of update t (0-based) for a run of ``total_updates`` = epochs·U updates.

    See the module docstring for the four schedules. τ = t/U counts completed epochs.
    """
    U, T = updates_per_epoch, total_updates
    if schedule == "constant":
        return lr
    if schedule == "step":
        s = max(1, (T // U) // STEP_DROPS)
        return lr * STEP_GAMMA ** ((t // U) // s)
    if schedule == "inv_sqrt":
        return lr / math.sqrt(1.0 + t / U)
    if schedule == "cosine":
        return lr * 0.5 * (1.0 + math.cos(math.pi * t / T))
    raise ValueError(f"unknown lr_schedule {schedule!r}; expected one of {SCHEDULES}")


def _common(lr: float) -> tuple[ParamSpec, ...]:
    return (
        ParamSpec(
            "lr", lr, min=1e-5, max=10.0, log=True, help="Base learning rate η₀ (step size)."
        ),
        ParamSpec(
            "batch_size",
            10,
            kind="int",
            min=1,
            max=1000,
            help="Samples per mini-batch (values above n use the full batch).",
        ),
        ParamSpec("epochs", 20, kind="int", min=1, max=1000, help="Passes over the data."),
        ParamSpec(
            "lr_schedule",
            "constant",
            kind="choice",
            choices=SCHEDULES,
            help="η_t: constant, halve every epochs/4 (step), η₀/√(1+τ), or cosine to 0.",
        ),
        ParamSpec(
            "gtol",
            1e-6,
            min=1e-14,
            max=1e-1,
            log=True,
            help="Stop when the full-gradient norm ‖∇f‖ ≤ gtol (tested after each epoch).",
        ),
        ParamSpec(
            "record_every",
            0,
            kind="int",
            min=0,
            max=100_000,
            help="Record every k-th update in the trace (0 = automatic). The trace is capped "
            "at 400 steps: k is raised to ⌈updates/398⌉ when it is smaller.",
        ),
    )


_EPS_PARAM = ParamSpec(
    "eps", 1e-8, min=1e-12, max=1e-2, log=True, help="ε in η/(√r + ε): guards the division."
)


# --------------------------------------------------------------------------------------
# Oracle and driver
# --------------------------------------------------------------------------------------


class _Oracle:
    """Component-gradient access with an exact count of the ∇fᵢ evaluated (n_gev)."""

    __slots__ = ("n", "p")

    def __init__(self, p: FiniteSumProblem) -> None:
        self.p = p
        self.n = 0

    def batch(self, w: Array, idx: NDArray[np.intp]) -> Array:
        """(1/|B|) Σ_{i∈B} ∇fᵢ(w)."""
        self.n += int(idx.size)
        return self.p.grad_batch(w, idx)

    def samples(self, w: Array, idx: NDArray[np.intp]) -> Array:
        """Rows ∇fᵢ(w), i ∈ idx (in the order of idx)."""
        self.n += int(idx.size)
        return self.p.grad_samples(w, idx)

    def full(self, w: Array) -> Array:
        """∇f(w) (n component gradients)."""
        self.n += self.p.n_samples
        return self.p.grad(w)


class _Rule:
    """An update rule: the part in which the methods differ."""

    def start(self, w: Array, oracle: _Oracle) -> None:
        """Called once before the first epoch."""

    def begin_epoch(self, w: Array, oracle: _Oracle) -> None:
        """Called at the start of every epoch, before its first update."""

    def update(
        self, w: Array, batch: NDArray[np.intp], eta: float, t: int, oracle: _Oracle
    ) -> tuple[Array, Array]:
        """Return (w_{t+1}, g_t): the new iterate and the gradient estimate it used."""
        raise NotImplementedError

    def info(self) -> dict[str, Any]:
        """Method-specific Step.info entries for the current state."""
        return {}


def _vec(v: Array | None) -> list[float] | None:
    return None if v is None else v.tolist()


def _drive(
    method: str,
    problem: FiniteSumProblem,
    rule: _Rule,
    *,
    x0: ArrayLike | None,
    seed: int,
    lr: float,
    batch_size: object,
    epochs: object,
    lr_schedule: str,
    gtol: float,
    record_every: object,
) -> Result:
    """Run ``rule`` with the shared sampling, schedule, trace and stopping test."""
    if not isinstance(problem, FiniteSumProblem):
        raise TypeError(f"{method}: problem must be a FiniteSumProblem (kind 'stochastic')")
    if lr_schedule not in SCHEDULES:
        raise ValueError(f"unknown lr_schedule {lr_schedule!r}; expected one of {SCHEDULES}")
    if not (lr > 0.0 and math.isfinite(lr)):
        raise ValueError(f"lr must be positive and finite, got {lr}")
    batch_size = _check_count("batch_size", batch_size, 1)
    epochs = _check_count("epochs", epochs, 1)
    record_every = _check_count("record_every", record_every, 0)
    if not gtol >= 0.0:
        raise ValueError(f"gtol must be ≥ 0, got {gtol}")
    w = as_vector(problem.x0 if x0 is None else x0)
    if w.size != problem.dim or not finite(w):
        raise ValueError(f"{problem.id}: x0 must be {problem.dim} finite numbers, got {w}")

    p = problem
    n = p.n_samples
    b = min(batch_size, n)
    U = -(-n // b)  # ⌈n/b⌉ updates per epoch
    T = epochs * U
    # 1 (k = 0) + ⌊T/every⌋ ≤ TRACE_MAX − 2 multiples + 1 (the final step) ≤ TRACE_MAX steps.
    every = max(record_every, -(-T // (TRACE_MAX - 2)), 1)
    rng = Rng(seed)
    oracle = _Oracle(p)
    n_fev = 0
    n_mon = 0  # full gradients evaluated only for monitoring

    def stop(converged: bool, message: str, k: int, fw: float, epochs_run: int) -> Result:
        extra = {
            "ifo": oracle.n,
            "monitor_grad_evals": n_mon,
            "epochs_run": epochs_run,
            "updates_per_epoch": U,
            "batch_size": b,
            "record_every": every,
        }
        return Result(method, w.copy(), fw, converged, message, k, n_fev, oracle.n, 0, trace, extra)

    with np.errstate(all="ignore"):  # overflow is detected below and reported as divergence
        rule.start(w, oracle)  # SAGA fills its gradient table at w₀ here (n gradients)
        g_full = p.grad(w)
        fw = float(p.f(w))
        n_mon, n_fev = 1, 1
        gnorm = float(np.linalg.norm(g_full))
        info0: dict[str, Any] = {
            "epoch": 0,
            "batch": None,
            "stoch_grad": None,
            "full_grad": g_full.tolist(),
            "lr": None,
            "update": None,
            "ifo": oracle.n,
        }
        trace = [Step(0, w.copy(), fw, gnorm, None, {**info0, **rule.info()})]
        if not finite(fw, gnorm):
            # Start test: no update was made, so this is not a divergence (no lr helps).
            msg = (
                f"not finite at x0: f(x0) = {fw:.3g}, ‖∇f(x0)‖ = {gnorm:.3g} (the start point "
                "or the data is out of range); no update was made"
            )
            return stop(False, msg, 0, fw, 0)
        if gnorm <= gtol:
            return stop(True, f"‖∇f(x0)‖ = {gnorm:.3g} ≤ gtol at the start", 0, fw, 0)

        # Divergence test (b): the blow-up thresholds, relative to the start.
        g_cap = BLOWUP * (1.0 + gnorm)
        f_cap = BLOWUP * (1.0 + abs(fw))
        nonfinite = "non-finite iterate, loss or gradient"
        t = 0
        for epoch in range(1, epochs + 1):
            perm = np.asarray(rng.permutation(n), dtype=np.intp)
            rule.begin_epoch(w, oracle)
            for j in range(U):
                batch = perm[j * b : (j + 1) * b]
                eta = learning_rate(lr_schedule, lr, t, U, T)
                w_prev = w
                w, g_est = rule.update(w, batch, eta, t, oracle)
                t += 1
                end_of_epoch = j == U - 1
                why = "" if finite(w) else nonfinite  # non-empty: the run diverged
                if t % every == 0 or end_of_epoch or why:
                    g_full = p.grad(w)
                    n_mon += 1
                    gnorm = float(np.linalg.norm(g_full))
                    if not (why or finite(gnorm)):
                        why = nonfinite
                    elif not why and gnorm > g_cap:
                        why = f"‖∇f‖ = {gnorm:.3g} > {BLOWUP:.0e}·(1 + ‖∇f(x0)‖) = {g_cap:.3g}"
                converged = end_of_epoch and not why and gnorm <= gtol
                done = bool(why) or converged or t == T
                if t % every != 0 and not done:
                    continue
                fw = float(p.f(w))
                n_fev += 1
                if not (why or finite(fw)):
                    why = nonfinite
                elif not why and fw > f_cap:
                    why = f"f = {fw:.3g} > {BLOWUP:.0e}·(1 + |f(x0)|) = {f_cap:.3g}"
                converged = converged and not why
                info = {
                    "epoch": epoch,
                    "batch": batch.tolist() if b <= BATCH_INFO_MAX else None,
                    "stoch_grad": _vec(g_est),
                    "full_grad": g_full.tolist(),
                    "lr": eta,
                    "update": (w - w_prev).tolist(),
                    "ifo": oracle.n,
                }
                trace.append(Step(t, w.copy(), fw, gnorm, eta, {**info, **rule.info()}))
                if why:
                    msg = f"diverged: {why} after update {t} (epoch {epoch}); try a smaller lr"
                    return stop(False, msg, t, fw, epoch)
                if converged:
                    msg = f"‖∇f‖ = {gnorm:.3g} ≤ gtol after {epoch} epochs ({t} updates)"
                    return stop(True, msg, t, fw, epoch)
        msg = f"completed {epochs} epochs ({T} updates); ‖∇f‖ = {gnorm:.3g} > gtol"
        return stop(False, msg, T, fw, epochs)


# --------------------------------------------------------------------------------------
# Update rules
# --------------------------------------------------------------------------------------


class _SGD(_Rule):
    def update(
        self, w: Array, batch: NDArray[np.intp], eta: float, t: int, oracle: _Oracle
    ) -> tuple[Array, Array]:
        g = oracle.batch(w, batch)
        return w - eta * g, g


class _Momentum(_Rule):
    def __init__(self, beta: float, nesterov: bool) -> None:
        self.beta = beta
        self.nesterov = nesterov
        self.v: Array | None = None
        self.lookahead: Array | None = None

    def start(self, w: Array, oracle: _Oracle) -> None:
        self.v = np.zeros_like(w)

    def update(
        self, w: Array, batch: NDArray[np.intp], eta: float, t: int, oracle: _Oracle
    ) -> tuple[Array, Array]:
        assert self.v is not None
        if self.nesterov:
            self.lookahead = w + self.beta * self.v
            g = oracle.batch(self.lookahead, batch)
        else:
            g = oracle.batch(w, batch)
        self.v = self.beta * self.v - eta * g
        return w + self.v, g

    def info(self) -> dict[str, Any]:
        out: dict[str, Any] = {"velocity": _vec(self.v) if self.v is not None else None}
        if self.nesterov:
            out["lookahead"] = _vec(self.lookahead)
        return out


class _Adaptive(_Rule):
    """AdaGrad, RMSProp and Adam: per-coordinate steps η/(√r + ε)."""

    def __init__(
        self, kind: str, eps: float, rho: float = 0.9, beta1: float = 0.9, beta2: float = 0.999
    ) -> None:
        self.kind = kind
        self.eps, self.rho, self.beta1, self.beta2 = eps, rho, beta1, beta2
        self.r: Array | None = None  # Σ g², EMA of g², or Adam's v
        self.m: Array | None = None
        self.m_hat: Array | None = None
        self.v_hat: Array | None = None
        self.scaled: Array | None = None

    def start(self, w: Array, oracle: _Oracle) -> None:
        self.r = np.zeros_like(w)
        self.m = np.zeros_like(w)

    def update(
        self, w: Array, batch: NDArray[np.intp], eta: float, t: int, oracle: _Oracle
    ) -> tuple[Array, Array]:
        assert self.r is not None and self.m is not None
        g = oracle.batch(w, batch)
        if self.kind == "adagrad":
            self.r = self.r + g * g
            self.scaled = eta / (np.sqrt(self.r) + self.eps)
            return w - self.scaled * g, g
        if self.kind == "rmsprop":
            self.r = self.rho * self.r + (1.0 - self.rho) * (g * g)
            self.scaled = eta / (np.sqrt(self.r) + self.eps)
            return w - self.scaled * g, g
        step = t + 1  # Adam's bias-correction counter starts at 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.r = self.beta2 * self.r + (1.0 - self.beta2) * (g * g)
        self.m_hat = self.m / (1.0 - self.beta1**step)
        self.v_hat = self.r / (1.0 - self.beta2**step)
        self.scaled = eta / (np.sqrt(self.v_hat) + self.eps)
        return w - self.scaled * self.m_hat, g

    def info(self) -> dict[str, Any]:
        if self.kind == "adagrad":
            return {"accum": _vec(self.r), "scaled_lr": _vec(self.scaled)}
        if self.kind == "rmsprop":
            return {"sq_avg": _vec(self.r), "scaled_lr": _vec(self.scaled)}
        return {
            "m": _vec(self.m),
            "v": _vec(self.r),
            "m_hat": _vec(self.m_hat),
            "v_hat": _vec(self.v_hat),
            "scaled_lr": _vec(self.scaled),
        }


class _SVRG(_Rule):
    def __init__(self) -> None:
        self.snapshot: Array | None = None
        self.mu: Array | None = None

    def begin_epoch(self, w: Array, oracle: _Oracle) -> None:
        # Option I of Johnson & Zhang (2013): the next snapshot is the last inner iterate.
        self.snapshot = w.copy()
        self.mu = oracle.full(self.snapshot)

    def update(
        self, w: Array, batch: NDArray[np.intp], eta: float, t: int, oracle: _Oracle
    ) -> tuple[Array, Array]:
        assert self.snapshot is not None and self.mu is not None
        v = oracle.batch(w, batch) - oracle.batch(self.snapshot, batch) + self.mu
        return w - eta * v, v

    def info(self) -> dict[str, Any]:
        return {"snapshot": _vec(self.snapshot), "snapshot_grad": _vec(self.mu)}


class _TableRule(_Rule):
    """SAG and SAGA: a table φ of the last component gradient seen for every sample."""

    def __init__(self, unbiased: bool) -> None:
        self.unbiased = unbiased  # True: SAGA, False: SAG
        self.table: Array | None = None  # (n, d)
        self.total: Array | None = None  # Σᵢ φᵢ, kept up to date incrementally
        self.seen: NDArray[np.bool_] | None = None

    def start(self, w: Array, oracle: _Oracle) -> None:
        n = oracle.p.n_samples
        if self.unbiased:
            # SAGA (Defazio et al. 2014, §2): φᵢ⁰ = w₀, i.e. the table holds ∇fᵢ(w₀).
            self.table = oracle.samples(w, np.arange(n, dtype=np.intp))
            self.seen = np.ones(n, dtype=bool)
        else:
            # SAG (Schmidt et al. 2017, Alg. 1): yᵢ = 0 and d = 0.
            self.table = np.zeros((n, w.size))
            self.seen = np.zeros(n, dtype=bool)
        self.total = self.table.sum(axis=0)

    def update(
        self, w: Array, batch: NDArray[np.intp], eta: float, t: int, oracle: _Oracle
    ) -> tuple[Array, Array]:
        assert self.table is not None and self.total is not None and self.seen is not None
        n = self.table.shape[0]
        new = oracle.samples(w, batch)  # (b, d) rows ∇fᵢ(w), i ∈ B
        delta = new - self.table[batch]  # (b, d)
        # SAGA: v = (1/b) Σ_{i∈B} (∇fᵢ(w) − φᵢ) + (1/n) Σⱼ φⱼ (the table before the update).
        v_saga = delta.mean(axis=0) + self.total / n
        self.table[batch] = new
        self.total = self.total + delta.sum(axis=0)
        self.seen[batch] = True
        # SAG: v = d/m with d = Σⱼ yⱼ after the update and m = #samples seen.
        v = v_saga if self.unbiased else self.total / int(self.seen.sum())
        return w - eta * v, v

    def info(self) -> dict[str, Any]:
        if self.unbiased:
            mean = (
                None if self.total is None or self.table is None else self.total / len(self.table)
            )
            return {"table_mean": _vec(mean)}
        return {"seen": 0 if self.seen is None else int(self.seen.sum())}


# --------------------------------------------------------------------------------------
# Registered methods
# --------------------------------------------------------------------------------------

_NEEDS = ("f", "grad_batch")
_FAMILY = "stochastic"


def _registered(
    id: str, name: str, lr: float, extra: tuple[ParamSpec, ...], **meta: Any
) -> Callable[[Callable[..., Result]], Callable[..., Result]]:
    return register(
        id=id,
        family=_FAMILY,
        name=name,
        params=(*_common(lr), *extra),
        needs=_NEEDS,
        deterministic=False,
        **meta,
    )


@_registered(
    "sgd",
    "Stochastic gradient descent",
    0.05,
    (),
    order="sublinear (O(1/k) with decaying η, strongly convex)",
    summary="Step against the gradient of a random mini-batch instead of the full gradient.",
    references=(
        "Robbins & Monro (1951)",
        "Bottou, Curtis & Nocedal (2018), SIAM Review 60(2), Alg. 4.1",
    ),
)
def sgd(
    problem: FiniteSumProblem,
    *,
    x0: ArrayLike | None = None,
    seed: int = 0,
    lr: float = 0.05,
    batch_size: int = 10,
    epochs: int = 20,
    lr_schedule: str = "constant",
    gtol: float = 1e-6,
    record_every: int = 0,
) -> Result:
    """Mini-batch SGD (Bottou, Curtis & Nocedal 2018, Alg. 4.1).

    g_k = (1/|B_k|) Σ_{i∈B_k} ∇fᵢ(w_k),   w_{k+1} = w_k − η_k g_k.

    With a constant η the iterates reach a noise ball of radius O(√η) around w⋆ (an O(η)
    loss floor); a decaying η (Σ η = ∞, Σ η² < ∞) is needed for convergence. With
    batch_size ≥ n every step is an exact gradient-descent step. Stopping test and conventions: see the module docstring.
    """
    return _drive("sgd", problem, _SGD(), **_kw(locals()))


@_registered(
    "sgd_momentum",
    "SGD with momentum (heavy ball)",
    0.02,
    (ParamSpec("momentum", 0.9, min=0.0, max=0.999, help="β: the velocity decay."),),
    order="sublinear (stochastic); accelerates ill-conditioned valleys",
    summary="Accumulate a velocity of past stochastic gradients and move along it.",
    references=(
        "Polyak (1964)",
        "Goodfellow, Bengio & Courville (2016), Deep Learning, Alg. 8.2",
    ),
)
def sgd_momentum(
    problem: FiniteSumProblem,
    *,
    x0: ArrayLike | None = None,
    seed: int = 0,
    lr: float = 0.02,
    batch_size: int = 10,
    epochs: int = 20,
    lr_schedule: str = "constant",
    gtol: float = 1e-6,
    record_every: int = 0,
    momentum: float = 0.9,
) -> Result:
    """SGD with heavy-ball momentum (Goodfellow et al. 2016, Alg. 8.2).

    g_k = ∇f_{B_k}(w_k),   v_{k+1} = βv_k − η_k g_k,   w_{k+1} = w_k + v_{k+1},   v_0 = 0.

    Stopping test and conventions: see the module docstring.
    """
    _check_unit("momentum", momentum)
    return _drive("sgd_momentum", problem, _Momentum(momentum, False), **_kw(locals()))


@_registered(
    "sgd_nesterov",
    "SGD with Nesterov momentum",
    0.02,
    (ParamSpec("momentum", 0.9, min=0.0, max=0.999, help="β: the velocity decay."),),
    order="sublinear (stochastic)",
    summary="Momentum that evaluates the gradient at the look-ahead point w + βv.",
    references=(
        "Nesterov (1983)",
        "Sutskever, Martens, Dahl & Hinton (2013), ICML, eqs. 3-4",
        "Goodfellow, Bengio & Courville (2016), Deep Learning, Alg. 8.3",
    ),
)
def sgd_nesterov(
    problem: FiniteSumProblem,
    *,
    x0: ArrayLike | None = None,
    seed: int = 0,
    lr: float = 0.02,
    batch_size: int = 10,
    epochs: int = 20,
    lr_schedule: str = "constant",
    gtol: float = 1e-6,
    record_every: int = 0,
    momentum: float = 0.9,
) -> Result:
    """SGD with Nesterov momentum (Sutskever et al. 2013; Goodfellow et al. 2016, Alg. 8.3).

    w̃_k = w_k + βv_k,  g_k = ∇f_{B_k}(w̃_k),  v_{k+1} = βv_k − η_k g_k,  w_{k+1} = w_k + v_{k+1}.

    Stopping test and conventions: see the module docstring.
    """
    _check_unit("momentum", momentum)
    return _drive("sgd_nesterov", problem, _Momentum(momentum, True), **_kw(locals()))


@_registered(
    "stochastic_adagrad",
    "AdaGrad",
    0.5,
    (_EPS_PARAM,),
    order="sublinear (O(1/√k) regret bound)",
    summary="Divide each coordinate's step by the root of its summed squared gradients.",
    references=(
        "Duchi, Hazan & Singer (2011), JMLR 12 (diagonal variant)",
        "Goodfellow, Bengio & Courville (2016), Deep Learning, Alg. 8.4",
    ),
)
def stochastic_adagrad(
    problem: FiniteSumProblem,
    *,
    x0: ArrayLike | None = None,
    seed: int = 0,
    lr: float = 0.5,
    batch_size: int = 10,
    epochs: int = 20,
    lr_schedule: str = "constant",
    gtol: float = 1e-6,
    record_every: int = 0,
    eps: float = 1e-8,
) -> Result:
    """Diagonal AdaGrad (Duchi et al. 2011; Goodfellow et al. 2016, Alg. 8.4).

    g_k = ∇f_{B_k}(w_k),  r_{k+1} = r_k + g_k⊙g_k,  w_{k+1} = w_k − η_k g_k / (√r_{k+1} + ε),
    r_0 = 0. The effective step decays like 1/√k by itself.

    # NOTE: ε is added outside the square root (Duchi et al.); Goodfellow et al. write
    # δ + √r with δ = 1e-7, which is the same form.
    Stopping test and conventions: see the module docstring.
    """
    _check_eps(eps)
    return _drive("stochastic_adagrad", problem, _Adaptive("adagrad", eps), **_kw(locals()))


@_registered(
    "stochastic_rmsprop",
    "RMSProp",
    0.01,
    (
        ParamSpec(
            "rho", 0.9, min=0.0, max=0.9999, help="ρ: decay of the squared-gradient average."
        ),
        _EPS_PARAM,
    ),
    order="sublinear (stochastic)",
    summary="AdaGrad with an exponential moving average, so the step does not die out.",
    references=(
        "Tieleman & Hinton (2012), COURSERA Neural Networks, Lecture 6.5",
        "Goodfellow, Bengio & Courville (2016), Deep Learning, Alg. 8.5",
    ),
)
def stochastic_rmsprop(
    problem: FiniteSumProblem,
    *,
    x0: ArrayLike | None = None,
    seed: int = 0,
    lr: float = 0.01,
    batch_size: int = 10,
    epochs: int = 20,
    lr_schedule: str = "constant",
    gtol: float = 1e-6,
    record_every: int = 0,
    rho: float = 0.9,
    eps: float = 1e-8,
) -> Result:
    """RMSProp (Tieleman & Hinton 2012; Goodfellow et al. 2016, Alg. 8.5).

    g_k = ∇f_{B_k}(w_k),  r_{k+1} = ρr_k + (1 − ρ)g_k⊙g_k,
    w_{k+1} = w_k − η_k g_k / (√r_{k+1} + ε),  r_0 = 0.

    # NOTE: Goodfellow et al. write η/√(δ + r); we use η/(√r + ε) like AdaGrad and Adam
    # (and PyTorch / Keras), so ε has the same meaning in the three adaptive methods.
    Stopping test and conventions: see the module docstring.
    """
    _check_eps(eps)
    _check_unit("rho", rho)
    return _drive(
        "stochastic_rmsprop", problem, _Adaptive("rmsprop", eps, rho=rho), **_kw(locals())
    )


@_registered(
    "stochastic_adam",
    "Adam",
    0.05,
    (
        ParamSpec("beta1", 0.9, min=0.0, max=0.999, help="β₁: decay of the first moment."),
        ParamSpec("beta2", 0.999, min=0.0, max=0.99999, help="β₂: decay of the second moment."),
        _EPS_PARAM,
    ),
    order="sublinear (stochastic)",
    summary="Momentum on the gradient plus RMSProp scaling, both with bias correction.",
    references=("Kingma & Ba (2015), ICLR, Algorithm 1",),
)
def stochastic_adam(
    problem: FiniteSumProblem,
    *,
    x0: ArrayLike | None = None,
    seed: int = 0,
    lr: float = 0.05,
    batch_size: int = 10,
    epochs: int = 20,
    lr_schedule: str = "constant",
    gtol: float = 1e-6,
    record_every: int = 0,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> Result:
    """Adam (Kingma & Ba 2015, Algorithm 1), with s = k + 1 the 1-based update count:

    m_{k+1} = β₁m_k + (1 − β₁)g_k,   v_{k+1} = β₂v_k + (1 − β₂)g_k⊙g_k,
    m̂ = m_{k+1}/(1 − β₁ˢ),   v̂ = v_{k+1}/(1 − β₂ˢ),   w_{k+1} = w_k − η_k m̂/(√v̂ + ε).

    The schedule multiplies Adam's step size α = η_k. Stopping test and conventions: see the
    module docstring.
    """
    _check_eps(eps)
    _check_unit("beta1", beta1)
    _check_unit("beta2", beta2)
    rule = _Adaptive("adam", eps, beta1=beta1, beta2=beta2)
    return _drive("stochastic_adam", problem, rule, **_kw(locals()))


@_registered(
    "svrg",
    "SVRG (stochastic variance-reduced gradient)",
    0.05,
    (),
    order="linear (strongly convex, η < 1/(4L_max))",
    summary="Correct each stochastic gradient with a full gradient taken once per epoch.",
    references=("Johnson & Zhang (2013), NeurIPS, Procedure SVRG (Fig. 1), option I",),
)
def svrg(
    problem: FiniteSumProblem,
    *,
    x0: ArrayLike | None = None,
    seed: int = 0,
    lr: float = 0.05,
    batch_size: int = 10,
    epochs: int = 20,
    lr_schedule: str = "constant",
    gtol: float = 1e-6,
    record_every: int = 0,
) -> Result:
    """SVRG (Johnson & Zhang 2013, Fig. 1, option I). At the start of every epoch:

    w̃ = w,   μ = ∇f(w̃)   (n component gradients);

    then for every mini-batch B of the epoch:

    v = ∇f_B(w) − ∇f_B(w̃) + μ,   w ← w − η v      (2|B| component gradients).

    E[v] = ∇f(w) and Var[v] → 0 as w, w̃ → w*, so a constant η gives linear convergence.

    # NOTE: the inner loop is one shuffled pass of ⌈n/b⌉ mini-batches (sampling without
    # replacement), not m = 2n uniform draws with replacement as in the paper's experiments.
    Stopping test and conventions: see the module docstring.
    """
    return _drive("svrg", problem, _SVRG(), **_kw(locals()))


@_registered(
    "saga",
    "SAGA",
    0.05,
    (),
    order="linear (strongly convex, η = 1/(3L_max))",
    summary="Keep the last gradient of every sample; correct each new one by the table.",
    references=("Defazio, Bach & Lacoste-Julien (2014), NeurIPS, §2 (SAGA update)",),
)
def saga(
    problem: FiniteSumProblem,
    *,
    x0: ArrayLike | None = None,
    seed: int = 0,
    lr: float = 0.05,
    batch_size: int = 10,
    epochs: int = 20,
    lr_schedule: str = "constant",
    gtol: float = 1e-6,
    record_every: int = 0,
) -> Result:
    """SAGA (Defazio, Bach & Lacoste-Julien 2014, §2), mini-batch form.

    The table φᵢ = ∇fᵢ(w₀) is filled once (n component gradients). For every mini-batch B:

    v = (1/|B|) Σ_{i∈B} [∇fᵢ(w) − φᵢ] + (1/n) Σⱼ φⱼ,   w ← w − η v,   φᵢ ← ∇fᵢ(w), i ∈ B.

    E[v] = ∇f(w) (unbiased, unlike SAG). The table sum is updated incrementally (O(bd) per step).
    Stopping test and conventions: see the module docstring.
    """
    return _drive("saga", problem, _TableRule(True), **_kw(locals()))


@_registered(
    "sag",
    "SAG (stochastic average gradient)",
    0.05,
    (),
    order="linear (strongly convex, η = 1/(16L_max))",
    summary="Step along the average of the last gradient seen for every sample.",
    references=(
        "Le Roux, Schmidt & Bach (2012), NeurIPS",
        "Schmidt, Le Roux & Bach (2017), Math. Programming 162, Alg. 1 and §4.1",
    ),
)
def sag(
    problem: FiniteSumProblem,
    *,
    x0: ArrayLike | None = None,
    seed: int = 0,
    lr: float = 0.05,
    batch_size: int = 10,
    epochs: int = 20,
    lr_schedule: str = "constant",
    gtol: float = 1e-6,
    record_every: int = 0,
) -> Result:
    """SAG (Schmidt, Le Roux & Bach 2017, Alg. 1), mini-batch form. yᵢ = 0, d = 0. For every
    mini-batch B:

    d ← d + Σ_{i∈B} [∇fᵢ(w) − yᵢ],   yᵢ ← ∇fᵢ(w) (i ∈ B),   w ← w − (η/m) d,

    with m the number of distinct samples visited so far.

    # NOTE: Alg. 1 divides d by n from the first step; we divide by m, the re-weighting on
    # early iterations of Schmidt et al. (2017), §4.1. After the first epoch m = n, so both agree.
    # NOTE: the paper samples uniformly with replacement; the per-epoch reshuffling used here
    # makes SAG stable only for smaller steps. Measured on linreg_2d with b = 1: it does not
    # converge for η ≥ 0.01 ≈ 0.1/L_max, while uniform sampling converges at η = 0.1 ≈ 1/L_max.
    # SAGA and SVRG converge at η = 0.1 under reshuffling.
    Stopping test and conventions: see the module docstring.
    """
    return _drive("sag", problem, _TableRule(False), **_kw(locals()))


# --------------------------------------------------------------------------------------
# Helpers for the registered wrappers
# --------------------------------------------------------------------------------------

_DRIVER_KEYS = ("x0", "seed", "lr", "batch_size", "epochs", "lr_schedule", "gtol", "record_every")


def _kw(local_vars: dict[str, Any]) -> dict[str, Any]:
    """The driver keywords from a wrapper's ``locals()``."""
    return {k: local_vars[k] for k in _DRIVER_KEYS}


def _check_count(name: str, value: object, minimum: int) -> int:
    """Validate an integer parameter ≥ ``minimum``; return it as an ``int``.

    An integral float (``100.0``, ``1e2`` as the CLI parses it) is accepted and converted;
    a bool, a non-number, a non-integral, non-finite or too small value raises ValueError.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(value)
        or value != math.floor(value)
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer ≥ {minimum}, got {value!r}")
    if isinstance(value, numbers.Integral):  # int, np.int64: exact at any size
        return operator.index(value)
    return int(float(value))  # an integral float (or np.float64): exact


def _check_unit(name: str, value: float) -> None:
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must lie in [0, 1), got {value}")


def _check_eps(eps: float) -> None:
    if not eps > 0.0:
        raise ValueError(f"eps must be positive, got {eps}")


#: Parity fixtures: (method_id, problem_id, params). Every case has ≤ 101 trace steps.
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("sgd", "linreg_2d", {"epochs": 10, "batch_size": 20, "seed": 1}),
    ("sgd", "logreg_2d", {"epochs": 5, "batch_size": 10, "lr": 0.5, "lr_schedule": "inv_sqrt"}),
    ("sgd_momentum", "linreg_2d", {"epochs": 10, "batch_size": 20, "lr_schedule": "step"}),
    ("sgd_nesterov", "huber_regression_2d", {"epochs": 10, "batch_size": 20}),
    ("stochastic_adagrad", "ill_conditioned_ls", {"epochs": 5, "batch_size": 10}),
    ("stochastic_rmsprop", "ill_conditioned_ls", {"epochs": 5, "batch_size": 10}),
    ("stochastic_adam", "logreg_2d", {"epochs": 5, "batch_size": 10, "lr_schedule": "cosine"}),
    ("svrg", "linreg_2d", {"epochs": 10, "batch_size": 20, "gtol": 1e-10}),
    ("saga", "logreg_2d", {"epochs": 10, "batch_size": 20, "lr": 0.3}),
    ("sag", "linreg_2d", {"epochs": 10, "batch_size": 20}),
]
