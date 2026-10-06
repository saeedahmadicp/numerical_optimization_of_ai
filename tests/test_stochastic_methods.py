"""Tests for the stochastic-gradient family (numopt.stochastic.methods).

Oracles:
    * an independent, per-sample-loop implementation of every update rule written from the
      paper equations (``_oracle_run``), which replays the documented Rng sampling order and
      computes the component gradients with SciPy's ``special.expit`` / the textbook formulas;
    * full-batch gradient descent (exact reduction of SGD with batch_size ≥ n);
    * ``numpy.linalg.lstsq`` and ``scipy.optimize`` minimizers for the converged
      variance-reduced methods.

Tolerances. The oracle and the method differ only in summation order and in the sigmoid
formula, i.e. by a few ulps per update; over ≤ 60 contractive updates that stays below
1e-12 relative, so iterates are compared at rtol = 1e-10, atol = 1e-12.
"""

from __future__ import annotations

import dataclasses
import inspect
import itertools
import json
import math
import re
from typing import Any

import numpy as np
import pytest
from conftest import assert_valid_result
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal
from scipy import optimize, special

import numopt
from numopt import problems
from numopt.core.registry import get_method, list_methods
from numopt.core.rng import Rng
from numopt.problems.stochastic import FiniteSumProblem
from numopt.stochastic import methods as sm

METHODS = (
    "sgd",
    "sgd_momentum",
    "sgd_nesterov",
    "stochastic_adagrad",
    "stochastic_rmsprop",
    "stochastic_adam",
    "svrg",
    "saga",
    "sag",
)
PROBLEMS = ("linreg_2d", "logreg_2d", "ill_conditioned_ls", "huber_regression_2d")
COMMON_INFO = {"epoch", "batch", "stoch_grad", "full_grad", "lr", "update", "ifo"}
METHOD_INFO = {
    "sgd": set(),
    "sgd_momentum": {"velocity"},
    "sgd_nesterov": {"velocity", "lookahead"},
    "stochastic_adagrad": {"accum", "scaled_lr"},
    "stochastic_rmsprop": {"sq_avg", "scaled_lr"},
    "stochastic_adam": {"m", "v", "m_hat", "v_hat", "scaled_lr"},
    "svrg": {"snapshot", "snapshot_grad"},
    "saga": {"table_mean"},
    "sag": {"seen"},
}


def _p(pid: str) -> FiniteSumProblem:
    p = problems.get(pid)
    assert isinstance(p, FiniteSumProblem)
    return p


def _xs(res) -> np.ndarray:
    return np.array([s.x for s in res.trace])


# --------------------------------------------------------------------------------------
# Independent oracle: per-sample loops straight from the equations
# --------------------------------------------------------------------------------------


def _grad_i(p: FiniteSumProblem, w: np.ndarray, i: int) -> np.ndarray:
    """∇fᵢ(w) = φ'(aᵢᵀw, yᵢ)aᵢ + λw, from the textbook derivatives."""
    a, y = np.asarray(p.X)[i], float(p.y[i])
    z = float(a @ w)
    if p.loss == "squared":
        s = z - y
    elif p.loss == "logistic":
        s = float(special.expit(z)) - y
    else:
        s = min(max(z - y, -p.huber_delta), p.huber_delta)
    return s * a + p.l2 * w


def _lr(schedule: str, lr: float, t: int, U: int, epochs: int) -> float:
    T = epochs * U
    if schedule == "constant":
        return lr
    if schedule == "step":
        return lr * 0.5 ** ((t // U) // max(1, epochs // 4))
    if schedule == "inv_sqrt":
        return lr / math.sqrt(1 + t / U)
    return lr * (1 + math.cos(math.pi * t / T)) / 2


def _oracle_run(method, p, *, lr, batch_size, epochs, seed=0, schedule="constant", **hp):
    """All iterates w_0..w_T of ``method``, computed sample by sample."""
    n = p.n_samples
    b = min(batch_size, n)
    U = -(-n // b)
    rng = Rng(seed)
    w = np.array(p.x0, dtype=float)
    out = [w.copy()]
    v = np.zeros(2)
    r = np.zeros(2)
    m = np.zeros(2)
    if method == "saga":
        table = [_grad_i(p, w, i) for i in range(n)]
    else:
        table = [np.zeros(2) for _ in range(n)]
    seen: set[int] = set()
    w_snap, mu = w.copy(), np.zeros(2)  # SVRG snapshot, refreshed every epoch
    t = 0
    for _ in range(epochs):
        perm = rng.permutation(n)
        if method == "svrg":
            w_snap = w.copy()
            mu = sum((_grad_i(p, w_snap, i) for i in range(n)), np.zeros(2)) / n
        for j in range(U):
            B = perm[j * b : (j + 1) * b]
            eta = _lr(schedule, lr, t, U, epochs)

            def gB(x, B=B):
                return sum((_grad_i(p, x, i) for i in B), np.zeros(2)) / len(B)

            if method == "sgd":
                w = w - eta * gB(w)
            elif method == "sgd_momentum":
                v = hp["momentum"] * v - eta * gB(w)
                w = w + v
            elif method == "sgd_nesterov":
                v = hp["momentum"] * v - eta * gB(w + hp["momentum"] * v)
                w = w + v
            elif method == "stochastic_adagrad":
                g = gB(w)
                r = r + g * g
                w = w - eta * g / (np.sqrt(r) + hp["eps"])
            elif method == "stochastic_rmsprop":
                g = gB(w)
                r = hp["rho"] * r + (1 - hp["rho"]) * g * g
                w = w - eta * g / (np.sqrt(r) + hp["eps"])
            elif method == "stochastic_adam":
                g = gB(w)
                b1, b2 = hp["beta1"], hp["beta2"]
                m = b1 * m + (1 - b1) * g
                r = b2 * r + (1 - b2) * g * g
                m_hat, v_hat = m / (1 - b1 ** (t + 1)), r / (1 - b2 ** (t + 1))
                w = w - eta * m_hat / (np.sqrt(v_hat) + hp["eps"])
            elif method == "svrg":
                w = w - eta * (gB(w) - gB(w_snap) + mu)
            elif method == "saga":
                new = {i: _grad_i(p, w, i) for i in B}
                corr = sum((new[i] - table[i] for i in B), np.zeros(2)) / len(B)
                w = w - eta * (corr + sum(table, np.zeros(2)) / n)
                for i in B:
                    table[i] = new[i]
            elif method == "sag":
                for i in B:
                    table[i] = _grad_i(p, w, i)
                    seen.add(i)
                w = w - eta * sum(table, np.zeros(2)) / len(seen)
            t += 1
            out.append(w.copy())
    return np.array(out)


HYPER: dict[str, dict[str, Any]] = {
    "sgd_momentum": {"momentum": 0.9},
    "sgd_nesterov": {"momentum": 0.8},
    "stochastic_adagrad": {"eps": 1e-8},
    "stochastic_rmsprop": {"rho": 0.9, "eps": 1e-8},
    "stochastic_adam": {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8},
}
ORACLE_CASES = [
    (m, pid, sched)
    for m in METHODS
    for pid, sched in (
        ("linreg_2d", "constant"),
        ("logreg_2d", "cosine"),
        ("huber_regression_2d", "step"),
    )
]


@pytest.mark.parametrize(("method", "pid", "schedule"), ORACLE_CASES)
def test_iterates_match_independent_oracle(method, pid, schedule):
    p = _p(pid)
    lr = {"stochastic_adagrad": 0.3, "stochastic_rmsprop": 0.02}.get(method, 0.03)
    kw: dict[str, Any] = {"lr": lr, "batch_size": 7, "epochs": 2, "seed": 5}  # 7 ∤ 200
    hp = HYPER.get(method, {})
    res = numopt.run(method, p, **kw, lr_schedule=schedule, gtol=0.0, record_every=1, **hp)
    ref = _oracle_run(method, p, schedule=schedule, **kw, **hp)
    assert res.n_iter == 2 * 29 and len(res.trace) == 59
    assert_allclose(_xs(res), ref, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("method", ["sag", "saga"])
def test_table_methods_match_oracle_with_single_samples(method):
    p = _p("linreg_2d")
    kw: dict[str, Any] = {"lr": 0.005, "batch_size": 1, "epochs": 1, "seed": 2}
    res = numopt.run(method, p, **kw, gtol=0.0, record_every=1)
    assert_allclose(_xs(res), _oracle_run(method, p, **kw), rtol=1e-10, atol=1e-12)


# --------------------------------------------------------------------------------------
# Registry and result contract
# --------------------------------------------------------------------------------------


def test_registry_entries():
    specs = {s.id: s for s in list_methods("stochastic")}
    assert set(METHODS) <= set(specs)
    for mid in METHODS:
        spec = get_method(mid)
        assert spec.family == "stochastic" and spec.deterministic is False
        assert spec.references and spec.summary
        names = [q.name for q in spec.params]
        for required in ("lr", "batch_size", "epochs", "lr_schedule", "gtol", "record_every"):
            assert required in names
        sched = next(q for q in spec.params if q.name == "lr_schedule")
        assert sched.choices == ("constant", "step", "inv_sqrt", "cosine")
        sig = inspect.signature(spec.fn)
        kw = {k: v.default for k, v in sig.parameters.items() if k not in ("problem", "x0", "seed")}
        assert kw == spec.defaults(), "every keyword is a ParamSpec with the same default"
        assert sig.parameters["seed"].default == 0
        for q in spec.params:
            if q.kind in ("float", "int"):
                assert q.min is not None and q.max is not None and q.min <= q.default <= q.max


def _finite_step(step) -> bool:
    return bool(
        np.all(np.isfinite(step.x))
        and step.fun is not None
        and math.isfinite(step.fun)
        and step.grad_norm is not None
        and math.isfinite(step.grad_norm)
    )


def _blew_up(step, start) -> bool:
    """Divergence test (b) of the module docstring at a recorded step."""
    g_cap = sm.BLOWUP * (1.0 + start.grad_norm)
    f_cap = sm.BLOWUP * (1.0 + abs(start.fun))
    return step.grad_norm > g_cap or step.fun > f_cap


def _assert_contract(res, p: FiniteSumProblem, method: str, gtol: float) -> None:
    """The documented Result contract, including the meaning of each stop reason."""
    assert_valid_result(res)
    assert len(res.trace) <= sm.TRACE_MAX
    assert res.n_iter == res.trace[-1].k
    assert res.n_fev == len(res.trace)
    assert res.extra["ifo"] == res.n_gev
    for step in res.trace:
        assert set(step.info) == COMMON_INFO | METHOD_INFO[method]
        if _finite_step(step):
            assert step.fun == p.f(step.x)
            assert step.grad_norm == float(np.linalg.norm(p.grad(step.x)))
            assert_array_equal(step.info["full_grad"], p.grad(step.x))
    start, last = res.trace[0], res.trace[-1]
    U = res.extra["updates_per_epoch"]
    if res.converged:
        # The stopping test passed, at w₀ or at the end of an epoch, and only there.
        assert last.grad_norm is not None and last.grad_norm <= gtol
        assert last.k % U == 0 and "≤ gtol" in res.message
        assert not any(_blew_up(s, start) for s in res.trace)
    elif res.message.startswith("not finite at x0"):
        # Start test: f(w₀) or ∇f(w₀) is not finite; the run stops before any update.
        assert res.n_iter == 0 and len(res.trace) == 1 and res.extra["epochs_run"] == 0
        assert not _finite_step(last) and "no update" in res.message
    elif res.message.startswith("diverged"):
        assert "smaller lr" in res.message
        # The run stopped at the first recorded step that failed test (a) or (b).
        assert not _finite_step(last) or _blew_up(last, start)
        assert all(_finite_step(s) and not _blew_up(s, start) for s in res.trace[:-1])
    else:
        assert res.message.startswith("completed") and "> gtol" in res.message
        assert last.grad_norm is not None and last.grad_norm > gtol
        assert res.n_iter == res.extra["epochs_run"] * U
        assert all(_finite_step(s) and not _blew_up(s, start) for s in res.trace)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("pid", PROBLEMS)
def test_contract_on_every_problem(method, pid):
    p = _p(pid)
    res = numopt.run(method, p)  # defaults: none of these 36 runs reaches gtol = 1e-6
    _assert_contract(res, p, method, gtol=1e-6)


CONVERGING_CASES: list[tuple[str, str, dict[str, Any]]] = [
    # Full-batch runs with η ≤ 1/L (L = 2.77 for linreg_2d, 0.694 for logreg_2d) are
    # deterministic descent methods; the adaptive ones need ~600 full-batch epochs.
    ("sgd", "linreg_2d", {"lr": 0.36, "batch_size": 200, "epochs": 200}),
    ("sgd_momentum", "linreg_2d", {"lr": 0.18, "momentum": 0.5, "batch_size": 200, "epochs": 200}),
    ("sgd_nesterov", "logreg_2d", {"lr": 1.44, "momentum": 0.5, "batch_size": 200, "epochs": 200}),
    ("stochastic_adagrad", "linreg_2d", {"batch_size": 200, "epochs": 1000}),
    (
        "stochastic_rmsprop",
        "linreg_2d",
        {"batch_size": 200, "epochs": 1000, "lr_schedule": "cosine"},
    ),
    ("stochastic_adam", "linreg_2d", {"batch_size": 200, "epochs": 1000}),
    ("svrg", "linreg_2d", {"lr": 0.1}),  # the default 20 epochs suffice
    ("saga", "logreg_2d", {"lr": 1.0, "epochs": 200}),
    ("sag", "logreg_2d", {"lr": 0.3, "epochs": 200}),
]


@pytest.mark.parametrize(("method", "pid", "params"), CONVERGING_CASES)
def test_contract_on_converging_runs(method, pid, params):
    # Regression: the default-parameter contract runs never converge, so the converged
    # branch of the contract needs its own cases.
    p = _p(pid)
    gtol = 1e-6 if method.startswith("stochastic_") else 1e-8
    res = numopt.run(method, p, gtol=gtol, **params)
    assert res.converged, res.message
    _assert_contract(res, p, method, gtol=gtol)


def test_info_keys_are_null_only_where_documented():
    # Regression: the "Info keys" section gave these keys as [d], but they are null at k = 0.
    null_at_start = {
        "sgd": set(),
        "sgd_momentum": set(),
        "sgd_nesterov": {"lookahead"},
        "stochastic_adagrad": {"scaled_lr"},
        "stochastic_rmsprop": {"scaled_lr"},
        "stochastic_adam": {"m_hat", "v_hat", "scaled_lr"},
        "svrg": {"snapshot", "snapshot_grad"},
        "saga": set(),
        "sag": set(),
    }
    always_null_at_start = {"batch", "stoch_grad", "lr", "update"}
    doc = sm.__doc__ or ""
    p = _p("logreg_2d")
    for method in METHODS:
        res = numopt.run(method, p, epochs=2, batch_size=10, record_every=1)
        nulls = {key for key, value in res.trace[0].info.items() if value is None}
        assert nulls == always_null_at_start | null_at_start[method], method
        for step in res.trace[1:]:
            assert all(value is not None for value in step.info.values()), (method, step.k)
        for key in always_null_at_start | null_at_start[method]:
            # "key: shape | null" (keys can share a line: "m_hat, v_hat: [d] | null").
            pattern = rf"\b{key}\b[\w, ]*: \S+ \| null"
            assert re.search(pattern, doc), f"{key} must be documented as '| null'"


@pytest.mark.parametrize("method", METHODS)
def test_deterministic_by_seed(method):
    p = _p("logreg_2d")
    a = numopt.run(method, p, epochs=3, seed=7, record_every=1)
    b = numopt.run(method, p, epochs=3, seed=7, record_every=1)
    c = numopt.run(method, p, epochs=3, seed=8, record_every=1)
    assert json.dumps(a.to_dict()) == json.dumps(b.to_dict())
    assert not np.array_equal(_xs(a), _xs(c))


def test_epochs_partition_the_data_in_rng_order():
    p = _p("linreg_2d")
    res = numopt.run("sgd", p, batch_size=8, epochs=3, seed=3, record_every=1, gtol=0.0)
    rng = Rng(3)
    for e in (1, 2, 3):
        batches = [s.info["batch"] for s in res.trace if s.info["epoch"] == e]
        assert len(batches) == 25
        assert list(itertools.chain.from_iterable(batches)) == rng.permutation(200)
    big = numopt.run("sgd", p, batch_size=40, epochs=1)
    assert all(s.info["batch"] is None for s in big.trace)


@pytest.mark.parametrize(
    ("method", "per_epoch", "setup"),
    [("sgd", 1, 0), ("stochastic_adam", 1, 0), ("svrg", 3, 0), ("saga", 1, 1), ("sag", 1, 0)],
)
def test_component_gradient_counts(method, per_epoch, setup):
    p = _p("linreg_2d")
    res = numopt.run(method, p, lr=0.01, batch_size=6, epochs=4, gtol=0.0, record_every=5)
    n = p.n_samples
    assert res.n_gev == setup * n + per_epoch * 4 * n
    U = 34  # ⌈200/6⌉
    assert res.n_iter == 4 * U
    recorded = [k for k in range(1, 4 * U + 1) if k % 5 == 0 or k == 4 * U]
    assert [s.k for s in res.trace] == [0, *recorded]
    epoch_ends = {e * U for e in range(1, 5)}
    assert res.extra["monitor_grad_evals"] == 1 + len(set(recorded) | epoch_ends)


@settings(max_examples=200, deadline=None)
@given(
    st.integers(1, 300),
    st.integers(1, 250),
    st.sampled_from(METHODS),
    st.sampled_from([0, 0, 1, 2, 7, 50, 100_000]),
)
def test_trace_length_is_capped(epochs, batch_size, method, record_every):
    p = _p("linreg_2d")
    res = numopt.run(method, p, lr=1e-3, epochs=epochs, batch_size=batch_size, gtol=0.0,
                     record_every=record_every)  # fmt: skip
    U = -(-200 // min(batch_size, 200))
    T = epochs * U
    assert res.n_iter == T == res.trace[-1].k
    assert len(res.trace) <= sm.TRACE_MAX
    every = res.extra["record_every"]
    assert every == max(record_every, -(-T // (sm.TRACE_MAX - 2)), 1)
    assert [s.k for s in res.trace] == sorted({*range(0, T + 1, every), T})


def test_small_record_every_is_raised_to_the_trace_cap():
    # Regression: record_every = 1 with batch_size = 1 gave one Step per update (20 001 steps,
    # 12.8 MB of JSON for 100 epochs; 200 001 steps at the UI maximum of 1000 epochs).
    p = _p("linreg_2d")
    res = numopt.run("stochastic_adam", p, lr=0.01, batch_size=1, epochs=100, record_every=1)
    assert res.n_iter == 20_000
    assert res.extra["record_every"] == 51 == -(-20_000 // 398)
    assert len(res.trace) == 394 <= sm.TRACE_MAX  # k = 0, 51, ..., 19 992, 20 000
    assert len(json.dumps(res.to_dict())) < 1_000_000
    small = numopt.run("sgd", p, epochs=2, batch_size=10, record_every=1)
    assert small.extra["record_every"] == 1 and len(small.trace) == 41  # honoured when it fits


# --------------------------------------------------------------------------------------
# Learning-rate schedules
# --------------------------------------------------------------------------------------


def test_schedule_values():
    U, E = 10, 20
    T = U * E
    lr = 0.4
    assert sm.learning_rate("constant", lr, 123, U, T) == lr
    assert sm.learning_rate("step", lr, 49, U, T) == lr  # s = ⌊20/4⌋ = 5 epochs
    assert sm.learning_rate("step", lr, 50, U, T) == lr / 2
    assert sm.learning_rate("step", lr, 199, U, T) == lr / 8
    assert sm.learning_rate("inv_sqrt", lr, 0, U, T) == lr
    assert sm.learning_rate("inv_sqrt", lr, 3 * U, U, T) == pytest.approx(lr / 2, rel=1e-15)
    assert sm.learning_rate("cosine", lr, 0, U, T) == lr
    assert sm.learning_rate("cosine", lr, T // 2, U, T) == pytest.approx(lr / 2, rel=1e-15)
    assert sm.learning_rate("step", lr, 2 * U, U, 3 * U) == lr / 4  # s = 1 for < 8 epochs
    with pytest.raises(ValueError):
        sm.learning_rate("linear", lr, 0, U, T)


@settings(max_examples=1000, deadline=None)
@given(
    st.sampled_from(sm.SCHEDULES),
    st.floats(1e-5, 10),
    st.integers(1, 50),
    st.integers(1, 200),
    st.data(),
)
def test_schedules_are_positive_and_nonincreasing(schedule, lr, U, epochs, data):
    T = U * epochs  # valid update indices are t = 0..T-1
    t = data.draw(st.integers(0, T - 1))
    a = sm.learning_rate(schedule, lr, t, U, T)
    assert 0 < a <= lr
    assert_allclose(a, _lr(schedule, lr, t, U, epochs), rtol=1e-15)
    if t + 1 < T:
        assert 0 < sm.learning_rate(schedule, lr, t + 1, U, T) <= a


def test_trace_lr_is_the_schedule():
    p = _p("linreg_2d")
    res = numopt.run("sgd", p, lr=0.05, epochs=8, batch_size=50, lr_schedule="cosine",
                     record_every=1)  # fmt: skip
    for s in res.trace[1:]:
        assert s.info["lr"] == s.step_size == sm.learning_rate("cosine", 0.05, s.k - 1, 4, 32)


# --------------------------------------------------------------------------------------
# Reductions and invariants
# --------------------------------------------------------------------------------------


@settings(max_examples=1000, deadline=None)
@given(
    st.sampled_from(PROBLEMS),
    st.floats(0.01, 1.9),
    st.integers(200, 1000),
    st.integers(1, 6),
    st.sampled_from(sm.SCHEDULES),
    st.integers(0, 2**32 - 1),
)
def test_full_batch_sgd_is_gradient_descent_exactly(pid, frac, b, epochs, schedule, seed):
    p = _p(pid)
    lr = frac / p.extra["L"]
    res = numopt.run("sgd", p, lr=lr, batch_size=b, epochs=epochs, lr_schedule=schedule,
                     seed=seed, gtol=0.0, record_every=1)  # fmt: skip
    w = np.array(p.x0)
    gd = [w]
    for t in range(epochs):
        w = w - sm.learning_rate(schedule, lr, t, 1, epochs) * p.grad(w)
        gd.append(w)
    assert_array_equal(_xs(res), np.array(gd))
    if frac <= 1.0:
        # Descent lemma (Nesterov 2004, Lemma 1.2.3): η ≤ 1/L gives f(w − η∇f) ≤ f(w) − (η/2)‖∇f‖².
        fs = [float(s.fun or 0.0) for s in res.trace]
        assert all(b_ <= a + 1e-15 * abs(a) for a, b_ in itertools.pairwise(fs))


@pytest.mark.parametrize("method", ["svrg", "saga", "sag"])
def test_variance_reduced_full_batch_is_gradient_descent(method):
    # With B = all samples, ∇f_B(w) − ∇f_B(w̃) + μ, the SAGA and the SAG directions are all
    # ∇f(w) (up to rounding in the table sums).
    p = _p("logreg_2d")
    res = numopt.run(method, p, lr=1.0, batch_size=200, epochs=15, gtol=0.0, record_every=1)
    w = np.array(p.x0)
    gd = [w]
    for _ in range(15):
        w = w - 1.0 * p.grad(w)
        gd.append(w)
    assert_allclose(_xs(res), np.array(gd), rtol=1e-12, atol=1e-14)


def test_adaptive_first_steps_are_sign_steps():
    # t = 1: AdaGrad gives −η g/(|g| + ε) and Adam gives −η m̂/(√v̂ + ε) = −η g/(|g| + ε):
    # each coordinate moves by η (1 − O(ε/|g|)).
    p = _p("ill_conditioned_ls")
    for method in ("stochastic_adagrad", "stochastic_adam"):
        res = numopt.run(method, p, lr=0.1, record_every=1, epochs=1)
        step = np.array(res.trace[1].info["update"])
        g = np.array(res.trace[1].info["stoch_grad"])
        assert_allclose(step, -0.1 * np.sign(g), rtol=1e-6)


def test_nesterov_lookahead_and_velocity():
    p = _p("linreg_2d")
    res = numopt.run("sgd_nesterov", p, lr=0.01, momentum=0.7, epochs=2, record_every=1)
    for prev, cur in zip(res.trace, res.trace[1:], strict=False):
        v_prev = np.array(prev.info["velocity"])
        assert_allclose(cur.info["lookahead"], prev.x + 0.7 * v_prev, rtol=1e-15, atol=1e-15)
        assert_allclose(cur.info["update"], cur.info["velocity"], rtol=1e-15, atol=1e-15)
        expected = 0.7 * v_prev - cur.info["lr"] * np.array(cur.info["stoch_grad"])
        assert_allclose(cur.info["velocity"], expected, rtol=1e-14, atol=1e-15)


def test_svrg_snapshot_info():
    p = _p("linreg_2d")
    res = numopt.run("svrg", p, lr=0.05, epochs=3, batch_size=50, record_every=1)
    U = 4
    for s in res.trace[1:]:
        start = res.trace[(s.info["epoch"] - 1) * U]  # the iterate the epoch started from
        assert_array_equal(s.info["snapshot"], start.x)
        assert_array_equal(s.info["snapshot_grad"], p.grad(start.x))


# --------------------------------------------------------------------------------------
# Convergence
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "pid", "lr"),
    [
        ("svrg", "linreg_2d", 0.1),
        ("saga", "linreg_2d", 0.1),
        ("sag", "linreg_2d", 0.03),
        ("svrg", "logreg_2d", 1.0),
        ("saga", "logreg_2d", 1.0),
        ("sag", "logreg_2d", 0.3),
        ("svrg", "huber_regression_2d", 0.1),
        ("saga", "huber_regression_2d", 0.1),
    ],
)
def test_variance_reduction_reaches_the_exact_minimizer(method, pid, lr):
    p = _p(pid)
    res = numopt.run(method, p, lr=lr, batch_size=10, epochs=200, gtol=1e-10)
    assert_valid_result(res)
    assert res.converged, res.message
    assert (res.trace[-1].grad_norm or 0.0) <= 1e-10
    # ‖w − w*‖ ≤ ‖∇f(w)‖/μ_local; μ_local = λ_min(∇²f(w*)) ≥ 0.06 on these problems.
    assert_allclose(res.x, p.minima[0], rtol=0, atol=2e-9)


def test_converged_minimizer_matches_numpy_and_scipy_oracles():
    lin = _p("linreg_2d")
    w_ls, *_ = np.linalg.lstsq(np.asarray(lin.X), np.asarray(lin.y), rcond=None)
    res = numopt.run("svrg", lin, lr=0.1, batch_size=1, epochs=100, gtol=1e-12)
    assert res.converged
    assert_allclose(res.x, w_ls, rtol=0, atol=1e-11)

    log = _p("logreg_2d")
    ref = optimize.minimize(log.f, np.zeros(2), jac=log.grad, method="BFGS",
                            options={"gtol": 1e-12})  # fmt: skip
    res = numopt.run("saga", log, lr=1.0, batch_size=1, epochs=100, gtol=1e-11)
    assert res.converged
    assert_allclose(res.x, ref.x, rtol=0, atol=1e-9)  # BFGS stops at ‖∇f‖ ≈ 1e-11


def test_sgd_with_decaying_lr_approaches_the_minimizer():
    p = _p("linreg_2d")
    w_star = np.array(p.minima[0])
    e0 = np.linalg.norm(np.array(p.x0) - w_star)

    def err(schedule: str, epochs: int, seed: int) -> float:
        res = numopt.run("sgd", p, lr=0.1, batch_size=10, epochs=epochs, lr_schedule=schedule,
                         seed=seed)  # fmt: skip
        assert not res.converged and res.message.startswith("completed")
        return float(np.linalg.norm(np.asarray(res.x) - w_star))

    const = np.mean([err("constant", 50, s) for s in range(5)])
    decay50 = np.mean([err("inv_sqrt", 50, s) for s in range(5)])
    decay200 = np.mean([err("inv_sqrt", 200, s) for s in range(5)])
    cosine = np.mean([err("cosine", 50, s) for s in range(5)])
    assert decay50 < 0.01 * e0 and cosine < 0.01 * e0
    assert decay50 < 0.3 * const and cosine < 0.3 * const  # constant η stalls in a noise ball
    assert decay200 < decay50  # and the decaying rate keeps going


@pytest.mark.parametrize("method", ["stochastic_adagrad", "stochastic_rmsprop", "stochastic_adam"])
def test_adaptive_methods_handle_ill_conditioning(method):
    p = _p("ill_conditioned_ls")
    w_star = np.array(p.minima[0])
    res = numopt.run(method, p, epochs=40)
    assert_valid_result(res)
    assert np.linalg.norm(np.asarray(res.x) - w_star) < 0.05
    assert res.fun - p.extra["f_min"] < 1e-2 * (p.f(p.x0) - p.extra["f_min"])


# --------------------------------------------------------------------------------------
# Failure paths and input validation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["sgd", "sgd_momentum", "sgd_nesterov", "svrg", "saga", "sag"])
def test_divergence_is_reported(method):
    # L = 941, L_max ≈ 7500 on ill_conditioned_ls: the default η = 0.02–0.05 is far above 2/L.
    # Regression (sag): SAG grew from f = 472 to f = 2.4e65 without overflow and was
    # reported as "completed 20 epochs"; the blow-up test (b) now stops it.
    p = _p("ill_conditioned_ls")
    res = numopt.run(method, p)
    _assert_contract(res, p, method, gtol=1e-6)
    assert not res.converged
    assert res.message.startswith("diverged") and "smaller lr" in res.message
    assert _blew_up(res.trace[-1], res.trace[0])
    assert res.n_iter == res.trace[-1].k < 20 * 20


def test_nonfinite_divergence_is_reported():
    # record_every > T: ∇f is evaluated only at the epoch end, so the blow-up test (b) never
    # runs before the iterate overflows; test (a) must catch the non-finite iterate at once.
    p = _p("ill_conditioned_ls")
    res = numopt.run("sgd", p, lr=10.0, batch_size=1, epochs=1, record_every=1000)
    _assert_contract(res, p, "sgd", gtol=1e-6)
    assert res.message.startswith("diverged: non-finite") and "smaller lr" in res.message
    assert res.n_iter < 200 and not _finite_step(res.trace[-1])


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("x0", [[1e160, 1e160], [1e200, -1e200]])
def test_nonfinite_start_point_is_not_divergence(method, x0):
    # Regression: f(x0) overflows, but no update was made, so the run did not diverge and
    # "try a smaller lr" would be wrong advice.
    p = _p("linreg_2d")
    res = numopt.run(method, p, x0=x0)
    _assert_contract(res, p, method, gtol=1e-6)
    assert not res.converged and res.message.startswith("not finite at x0")
    assert "diverged" not in res.message and "smaller lr" not in res.message
    assert res.n_iter == 0 and res.n_fev == 1


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("field", ["X", "y"])
def test_nonfinite_data_stops_at_the_start(method, field):
    # Regression: a NaN in the data gave the same false "diverged" message as a huge x0.
    p = _p("logreg_2d")
    bad = np.array(getattr(p, field))
    bad.flat[3] = math.nan
    q = dataclasses.replace(p, **{field: bad})
    res = numopt.run(method, q)
    _assert_contract(res, q, method, gtol=1e-6)
    assert res.message.startswith("not finite at x0") and "diverged" not in res.message


@settings(max_examples=1000, deadline=None)
@given(
    st.sampled_from(METHODS),
    st.sampled_from(PROBLEMS),
    st.floats(-3.0, 3.0),
    st.integers(1, 60),
    st.integers(1, 8),
    st.sampled_from([1e-6, 1e-3, 1e-1]),
    st.integers(0, 2**32 - 1),
)
def test_stop_reason_matches_its_test(method, pid, log_frac, batch_size, epochs, gtol, seed):
    # η = 10^log_frac / L_max (capped at the ParamSpec maximum 10) covers stable steps and
    # steps 10³× too large. On 1000 such draws about 84% end "completed", 10% "diverged"
    # and 6% converge, so every branch of _assert_contract is exercised. Every run must end
    # for its documented reason, and steps η ≤ 0.01/L_max never trigger the blow-up test.
    p = _p(pid)
    lr = min(10.0, 10.0**log_frac / p.extra["L_max"])
    res = numopt.run(method, p, lr=lr, batch_size=batch_size, epochs=epochs, gtol=gtol,
                     seed=seed, record_every=3)  # fmt: skip
    _assert_contract(res, p, method, gtol=gtol)
    if log_frac <= -2.0:
        assert not res.message.startswith("diverged"), res.message


def test_sag_with_reshuffling_needs_a_small_step():
    # Under per-epoch reshuffling SAG is unstable at η = 0.1 ≈ 1/L_max with single samples,
    # although the same step converges with uniform sampling (not used here).
    p = _p("linreg_2d")
    res = numopt.run("sag", p, lr=0.1, batch_size=1, epochs=50)
    assert not res.converged
    assert res.trace[-1].grad_norm is None or res.trace[-1].grad_norm > 1.0
    ok = numopt.run("sag", p, lr=0.003, batch_size=1, epochs=200, gtol=1e-10)
    assert ok.converged


def test_epoch_budget_is_reported():
    res = numopt.run("sgd", _p("linreg_2d"), epochs=3)
    assert not res.converged
    assert res.message.startswith("completed 3 epochs (60 updates)") and "> gtol" in res.message
    assert_valid_result(res)


def test_converged_at_start():
    p = _p("linreg_2d")
    for method in METHODS:
        res = numopt.run(method, p, x0=p.minima[0], gtol=1e-12)
        assert res.converged and res.n_iter == 0 and len(res.trace) == 1
        assert_valid_result(res)


def test_invalid_inputs():
    p = _p("linreg_2d")
    with pytest.raises(TypeError):
        numopt.run("sgd", problems.get("rosenbrock"))
    with pytest.raises(ValueError):
        numopt.run("sgd", p, lr_schedule="linear")
    with pytest.raises(ValueError):
        numopt.run("sgd", p, lr=0.0)
    with pytest.raises(ValueError):
        numopt.run("sgd", p, batch_size=0)
    with pytest.raises(ValueError):
        numopt.run("sgd", p, x0=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        numopt.run("sgd", p, x0=[math.nan, 0.0])
    with pytest.raises(ValueError):
        numopt.run("sgd_momentum", p, momentum=1.0)
    with pytest.raises(ValueError):
        numopt.run("stochastic_adam", p, eps=0.0)
    with pytest.raises(TypeError):
        numopt.run("sgd", p, momentum=0.9)  # not a parameter of plain SGD
    bad_counts: list[dict[str, Any]] = [
        {"epochs": 2.5},
        {"epochs": 0},
        {"epochs": True},
        {"epochs": math.inf},
        {"epochs": "5"},
        {"batch_size": 10.5},
        {"batch_size": math.nan},
        {"batch_size": 0.0},
        {"record_every": -1},
        {"record_every": 2.5},
        {"record_every": False},
    ]
    for kw in bad_counts:
        with pytest.raises(ValueError, match=next(iter(kw))):
            numopt.run("sgd", p, **kw)


@pytest.mark.parametrize("method", METHODS)
def test_integral_float_counts_are_accepted(method):
    # Regression: the CLI parses `--set epochs=1e2` as the float 100.0, which crashed
    # range() with a TypeError. An integral float must run exactly like the int.
    p = _p("linreg_2d")
    res = numopt.run(method, p, epochs=1e2, batch_size=10.0, record_every=2.0)
    ref = numopt.run(method, p, epochs=100, batch_size=10, record_every=2)
    _assert_contract(res, p, method, gtol=1e-6)
    assert res.message == ref.message
    assert_array_equal(res.x, ref.x)
    assert [s.k for s in res.trace] == [s.k for s in ref.trace]
    for key in ("batch_size", "record_every", "epochs_run", "updates_per_epoch"):
        assert type(res.extra[key]) is int and res.extra[key] == ref.extra[key]


def test_inputs_are_not_mutated():
    p = _p("logreg_2d")
    x0 = np.array([0.5, -0.5])
    X_before, y_before = np.array(p.X), np.array(p.y)
    numopt.run("saga", p, x0=x0, epochs=2)
    assert_array_equal(x0, [0.5, -0.5])
    assert_array_equal(p.X, X_before)
    assert_array_equal(p.y, y_before)


def test_fixture_cases():
    cases = sm.FIXTURE_CASES
    assert {c[0] for c in cases} == set(METHODS)
    assert 3 <= len(cases) <= 12
    for method, pid, params in cases:
        res = numopt.run(method, problems.get(pid), **params)
        assert_valid_result(res)
        assert len(res.trace) < 300
