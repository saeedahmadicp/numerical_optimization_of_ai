# Architecture

`numopt` has three parts that share one catalog of methods and problems:

1. **`src/numopt/`** — a lean, typed Python package: the mathematical reference implementation,
   a CLI, a benchmarking module (`numopt.bench`) and the exporter that produces JSON fixtures.
2. **`web/`** — a static, server-less portal (Vite + React 19 + TypeScript): 16 interactive labs,
   the Methods and Research reference pages, and a TypeScript port of every method. A parity test
   replays every Python fixture with the TypeScript port, so the two implementations cannot drift
   apart.
3. **`research/`** — studies that test new methods against the package's baselines before they
   are promoted into `src/numopt/` (see "Research" below).

The registry holds 168 methods in 16 families and 103 test problems (`numopt list`,
`numopt problems`); every method has at least one parity fixture case.

The pre-2026 implementation was replaced wholesale (its methods had incorrect mathematics and
test-specific special cases); it remains available in the git history before this rewrite.

## Repository layout

```
pyproject.toml                 Python >= 3.11, runtime dep: numpy only (extras: plot = matplotlib, dev)
src/numopt/
  __init__.py                  public API: run, minimize, find_root, list_methods, problems, types
  core/types.py                Step, Result, Problem, Constraint, LinearProgram, LinearSystem, Dataset
  core/registry.py             @register, MethodSpec, ParamSpec, FAMILIES, run(), list_methods()
  core/counting.py             Counted (evaluation counter), problem/start-point resolution
  core/diff.py                 finite-difference fallbacks (gradient, Hessian, Jacobian)
  core/rng.py                  Mulberry32 portable PRNG (bit-identical in TypeScript)
  problems/                    test-problem library, one module per problem kind
    registry.py                add(kind, problem), get(id), list_problems(kind), @factory(kind)
    roots.py systems.py scalar_min.py unconstrained.py least_squares.py stochastic.py
    constrained.py lp.py combinatorial.py linalg.py calculus.py data.py
  roots/                       bracketing.py  open.py  systems.py   (families roots + systems)
  scalar/                      methods.py      (1-D minimization on an interval)
  line_search/                 methods.py      (step-length selection; also used by n-D methods)
  unconstrained/               first_order.py newton.py quasi_newton.py conjugate_gradient.py
                               trust_region.py derivative_free.py
                               accelerated.py         silver_gd, silver_gd_strongly_convex,
                                                      long_step_gd, ogm, fista (adaptive restart)
                               anderson.py            anderson_gd (AA(m) with the RNA term)
                               regularized_newton.py  arc (cubic regularization), reg_newton
                               global_.py             (family global)
                               least_squares.py       (family least_squares)
  stochastic/                  methods.py      (SGD family on finite sums)
  constrained/                 methods.py
  lp/                          simplex.py interior_point.py integer.py
                               pdhg.py         restarted_pdhg (PDLP-style restarted PDHG)
  combinatorial/               knapsack.py tsp.py
  linalg/                      direct.py iterative.py
  interpolation/               methods.py
                               rational.py     aaa, floater_hormann (barycentric rational)
  integration/ differentiation/ regression/   methods.py each
  bench.py                     benchmarking: run_benchmark, performance/data profiles (see below)
  cli.py                       numopt list | problems | run | compare | export | bench
  export.py                    registry.json, problems.json, fixtures/<family>.json
tests/                         pytest + hypothesis; SciPy is a test oracle only (never a runtime dep)
web/                           the portal (see "Web app" below)
research/                      one folder per study (see "Research" below)
docs/                          this file · brand/ (identity, README hero, scripts) · assets/screens/ (README screenshots)
.github/workflows/             ci.yml (every push and pull request) · pages.yml (manual deploy)
```

A family is a registry value (`numopt.FAMILIES`), not a folder: `systems` lives in `roots/`, and
`global` and `least_squares` live in `unconstrained/`.

Each family package `__init__.py` imports its modules and concatenates their `FIXTURE_CASES`.
A module adds methods by `@register(...)` and adds parity fixtures by defining
`FIXTURE_CASES = [(method_id, problem_id, params), ...]` at module level.

## Method contract (Python)

```python
@register(id="bfgs", family="unconstrained", name="BFGS",
          params=(ParamSpec("gtol", 1e-8, min=1e-14, max=1e-2, log=True, help="..."),
                  ParamSpec("max_iter", 500, kind="int", min=1, max=100_000),
                  ParamSpec("line_search", "strong_wolfe", kind="choice", choices=("strong_wolfe", "backtracking"))),
          needs=("f", "grad"), order="superlinear",
          summary="One sentence a student understands.",
          references=("Nocedal & Wright (2006), Algorithm 6.1",))
def bfgs(problem: Problem | Callable, *, x0=None, gtol=1e-8, max_iter=500, line_search="strong_wolfe") -> Result:
```

* Signature: `fn(problem, *, x0=None | bracket=None | seed=None, **params) -> Result`. Every keyword
  except `x0`, `bracket`, `seed` must be declared as a `ParamSpec` with a sensible UI range.
* Methods accept a `Problem` from the library or a bare callable (use `core.counting` helpers).
* **Trace**: one `Step` for `k = 0` (the start) and one per iteration; at most `max_iter + 1` steps.
  `n_iter == trace[-1].k` for iterative methods.
* **Convergence**: `converged=True` only when the documented tolerance test passed. Max-iter,
  non-finite values, singular matrices, lost brackets, unbounded LPs → `converged=False` with a
  clear `message`. Never raise on numerical breakdown. Raise `ValueError` only for invalid input.
* **Counts**: count f, gradient and Hessian evaluations exactly with `Counted`.
* **References**: every docstring names the textbook algorithm/equation it implements and states
  the stopping test. Deviations from the textbook are marked `# NOTE:` with the reason.
* **Determinism**: stochastic methods take `seed` (ParamSpec-free keyword, default 0) and draw ONLY
  from `numopt.core.rng.Rng(seed)` (Mulberry32, bit-identical to `web/src/core/rng.ts`). Never use
  `np.random`. Draw random numbers in a documented order so the TS port can replay it.
* **Purity**: no printing, plotting, global state, or mutation of inputs.
* **Step.info** carries the visual geometry for the web app. Each module docstring has an
  `Info keys:` section that lists every key it emits, its shape, and its meaning, e.g.
  `bracket: [a, b]`, `direction: [n]`, `simplex: [[n]...]`, `radius: float`, `alpha: float`,
  `trials: [[alpha, phi]]`, `tableau: [[...]]`, `basis: [int]`, `entering: int`, `leaving: int`,
  `tour: [int]`, `estimate: float`, `nodes: [x]`. Values must be JSON-serializable.

## Line-search API (used by every n-D descent method)

`numopt.line_search.methods` exposes, besides its registered demo methods:

```python
@dataclass(frozen=True)
class LineSearchResult:
    alpha: float            # accepted step (0.0 if the search failed)
    f_new: float            # f(x + alpha p)
    g_new: ndarray | None   # ∇f(x + alpha p) when computed, else None
    n_fev: int; n_gev: int
    success: bool
    trials: list[tuple[float, float]]   # every (alpha, phi(alpha)) evaluated, in order
    message: str
    n_hev: int = 0          # exact_quadratic evaluates a callable Hessian once

def search(kind, f, grad, x, p, *, f0=None, g0=None, alpha0=1.0, hess=None,
           c1=None, c2=0.9, rho=0.5, max_iter=50, alpha_max=1e3) -> LineSearchResult
# c1=None selects 1e-4 (0.25 for goldstein); hess may be a callable or an (n, n) array.
# kind ∈ {"backtracking", "strong_wolfe", "weak_wolfe", "goldstein", "exact_quadratic"}
#   backtracking: Armijo, N&W Alg. 3.1 (c1=1e-4, rho=0.5)
#   strong_wolfe: strong Wolfe, N&W Alg. 3.5 + zoom Alg. 3.6 with cubic interpolation (c1=1e-4, c2=0.9)
#   weak_wolfe:   bisection/expansion search for the weak Wolfe conditions (Lewis–Overton)
#   goldstein:    Goldstein conditions with bracketing (c=0.25)
#   exact_quadratic: alpha = -gᵀp / pᵀHp using a supplied `hess` (quadratics only)
```

`search()` reports `n_fev`/`n_gev`/`n_hev` for exactly the calls it makes (including f(x) and
∇f(x) when `f0`/`g0` are not given). A caller either adds these counts or reads its own `Counted`
wrappers — never both.

## Benchmarking (`numopt.bench`)

`bench.run_benchmark(methods, cases, budget=, cost=, seeds=)` runs every method on every instance
(a `BenchmarkCase` is a problem id with one or more start points; stochastic methods run once per
seed). A recorder wraps the problem and counts every evaluation, so the best value against the cost
spent is exact and does not depend on how a method fills its trace; a run stops when its budget is
spent. Cost models: `"nfev"` (one unit per f evaluation) and `"nfev+n*ngev"` (a gradient costs n
units, Moré & Wild's simplex-gradient convention); Hessian evaluations are free under both and are
recorded as `n_hev`.

The convergence test is Moré & Wild (2009), eq. 2.2: f(x) ≤ f_L + τ (f(x₀) − f_L), with f_L the
problem's known minimum (`extra["f_min"]`) or the best value any solver found.
`performance_profile(res, tau=)` (Dolan & Moré 2002) and `data_profile(res, tau=)` (Moré & Wild
2009, eq. 2.7) return a `Profile` sampled on a grid that contains every break point;
`plot_performance_profile` / `plot_data_profile` need matplotlib (`pip install -e ".[plot]"`).
`BenchmarkResult.save(path)` writes every run as JSON. The CLI is
`numopt bench PROBLEM … --methods M … --budget B [--cost C] [--tau T …] [--seeds S …] [--json PATH] [--plot PREFIX]`.
Report every comparison at two or more values of τ.

## Problem library

`problems.get(id)` returns a problem; `problems.list_problems(kind)` lists them. Every smooth
problem supplies exact derivatives (verified by tests against finite differences), a plotting
`domain`, a sensible default `x0` (or `bracket`), and its known minima/roots (verified by tests).
2-D problems are preferred for unconstrained/constrained/systems kinds because the web app plots
them; a few n-D problems exist for the CLI and tests.

## Testing rules

* Every method: a convergence test on ≥ 2 problems, an oracle comparison with SciPy/NumPy where
  an equivalent exists, a max-iter/failure-path test, the `assert_valid_result` contract check
  (`tests/conftest.py`), and Hypothesis property tests where an invariant exists (bracket
  contains the root, monotone decrease for descent methods with line search, feasibility for
  projected methods, exactness of quadrature on polynomials of the right degree, ...).
* Test files are flat: `tests/test_<package>_<module>.py`.
* `ruff check`, `ruff format --check` and `pyright` must be clean for `src/` and `tests/`.

## Web app

[web/README.md](../web/README.md) is the full reference (APIs, conventions, the approved home
page design). The layout:

```
web/
  package.json                 vite, react 19, typescript (strict), vitest, katex, marked, motion, three (lazy)
  src/core/types.ts            Step, Result, Problem, ... (camelCase mirror of the Python types)
  src/core/registry.ts         MethodSpec / ParamSpec mirror; registerMethod / getMethod / runMethod
  src/core/rng.ts              Mulberry32, bit-identical to numopt.core.rng
  src/generated/               output of `numopt export` (registry, problems, fixtures) — never hand-edit
  src/methods/<pkg>/<module>.ts   TS port; same ids, params, trace semantics and Step.info keys
                                  (mirrors src/numopt/<pkg>/<module>.py, including accelerated.ts,
                                  anderson.ts, regularized_newton.ts, rational.ts and lp/pdhg.ts)
  src/problems/<kind>.ts       TS port of the problem library (same ids)
  src/viz/                     rendering primitives: HiDPI canvas, axes, Plot1D, Contour2D (worker),
                               Surface3D (three.js, lazy), ConvergenceChart, IterationTable, MatrixView
  src/play/                    the playback engine: useTracePlayer, usePlayerKeyboard, timeline
  src/ui/                      design system: tokens.css, components, PlaybackBar, MethodCard, Formula
  src/labs/index.ts            the lab registry (discovered, see below)
  src/labs/_shell/             LabShell and the shared lab blocks (pickers, slots, presets, runs, status)
  src/labs/<lab-id>/           meta.ts + index.tsx (+ setup.ts, presets, local components and tests)
  src/app/                     App, hash router, URL state, header + search, Home (HeroShowcase,
                               LabCard gallery, home/previews/labs/<lab-id>.ts), catalog
  src/site/                    reference pages (lazy): Methods, Method (live run + parity replay),
                               Research + Study, Python
  vite/catalog.ts              Vite plugin: the build-time virtual:numopt/* modules (counts, index,
                               parity summaries, research)
  vite/research.ts             research READMEs → HTML at build time (marked + KaTeX in Node)
  tests/                       vitest (parity with fixtures, per-lab dumps, unit) · fixtures/ (scripts)
  e2e/                         playwright smoke tests
```

**Per-lab registry.** Every lab is one folder, `src/labs/<id>/`, and adding one changes no shared
file. `meta.ts` default-exports a `LabMeta` (`id` = folder, `title`, `group` from `LAB_GROUPS`,
`problem` in TeX, `pitch`, `families`, optional `problemKinds`, `status`, `order`, `icon`);
`index.tsx` default-exports the lab component. `src/labs/index.ts` discovers both with
`import.meta.glob` — metas eagerly (they are tiny), components lazily, one chunk per lab — and
builds `LABS` (`getLab`, `labsInGroup`, `labForFamily`, `labForProblemKind`). A folder with a
`meta.ts` and no `index.tsx` is listed as *planned*. Method counts come from
`src/generated/registry.json` at build time (`CATALOG.byFamily`), so the home page never loads a
lab to count it. Each lab also has a home-page preview, `src/app/home/previews/labs/<id>.ts`, that
runs the real TS ports on a registered problem; `previews.test.ts` fails when a lab has none.
All 16 families have an open lab.

Routes (hash router, so the build works from any sub-path): `#/` (home), `#/labs`, `#/lab/<id>`,
`#/methods`, `#/method/<id>`, `#/research`, `#/research/<study>`, `#/python`. The raw generated
JSON never reaches the bundle: `vite/catalog.ts` serves counts, a lazy index, per-family parity
summaries and the rendered studies as virtual modules; the fixtures are test-only.

Parity rule: for every fixture, the TS port reproduces the final `x` within 1e-6 (relative), the
first `min(10, n)` trace iterates within 1e-8, and `nIter` exactly (deterministic methods).
Stochastic methods draw from the shared Mulberry32 generator (`numopt.core.rng` /
`src/core/rng.ts`), so they are parity-checked like deterministic methods; libm differences in
`log`/`cos` may cause late divergence, so their parity check covers the first 10 iterates (1e-8)
and the final objective value (1e-6 relative) only.

**Parity fixtures.** The committed fixtures are the canonical output (written on aarch64 Linux
with the NumPy wheel's OpenBLAS); the TS parity tests against them stay as strict as above on
every machine, since the TS ports are deterministic. Another CPU or BLAS rounds the Python export
differently, so a fixture case must not be chaotic with respect to rounding: rerun with x0
changed by 1 to 3 ulp, it must keep its counts and its iterates within the tolerance of
`check_generated.py` (1e-9 of their size in the run), unless its iterates are bit-identical on
every platform (only correctly rounded arithmetic reaches them). Runs in a curved valley such as
BFGS, L-BFGS, Barzilai–Borwein or Anderson acceleration on `rosenbrock`, and PCG on `hilbert_5`,
fail that test and are not fixtures. AAA removes the two rounding-decided choices of its
algorithm: a greedy pick or a sign among values within 1e-8 (relative) of the largest goes to the
first index, and a non-unique minimal singular vector is replaced by the minimum-norm null vector
with Σw = 1 (`numopt.interpolation.rational`).

**Reference data for the web tests.** Two kinds, both written by Python
([web/tests/fixtures/README.md](../web/tests/fixtures/README.md)):

| Data | Written by | In git |
|:--|:--|:--|
| parity fixtures in `web/src/generated/` | `npm run gen` (`numopt export`) | yes |
| per-lab Python reference dumps in `web/tests/**/fixtures/` (~14 MB) | `npm run gen:test-fixtures` | no |

Hygiene scripts in `web/tests/fixtures/`: `check_generated.py` (`npm run gen:check`) exports into a
temporary folder and compares the result with the committed `src/generated/` — text (with rounded
numerals in messages masked), counts, flags and structure exactly, and the state x, f, ‖∇f‖ of
every step within |a − b| ≤ 1e-9·S + 1e-12, S the field's largest magnitude in that run (not
byte identity, since NumPy may round the last bit differently on another CPU) — and exits with
status 1 when they are stale; `gen_test_fixtures.py` writes every ignored dump (`--missing`,
`--list`); `gen_platform.py` records whether the Python that wrote the dumps reproduces the
committed export byte for byte (`platform_python.json`): the dump tests are exact there and use
the tolerances each one states elsewhere (`platform.ts`), e.g. on the x86-64 CI runner;
`ensure.mjs` is the `pretest` hook that generates any absent dump with `../.venv/bin/python` (or
`$NUMOPT_PYTHON`); `gen_rng_fixture.py` writes the Mulberry32 reference streams.

UX: Home gallery of labs → each Lab has a problem picker, method multi-select (compare up to 4),
parameter controls generated from `ParamSpec`, click/drag start point, animated main view,
convergence chart, iteration table synced to the playhead, a MethodCard with the update rule in
LaTeX, a PlaybackBar (Space, ←/→, Home/End, R), shareable URL state, light/dark themes, and
reduced-motion support.

## Research

`research/<slug>/` holds one study each: `README.md` (Question · Background · Method · Setup ·
Results · Discussion · Reproduce · References), `method.py` (the method with the numopt contract
and a `PARAMS` dict of `ParamSpec`s), `test_method.py`, a deterministic `run.py`, `results/*.json`
(every number the README quotes) and `figures/`. Code in `research/` imports `numopt` and never
changes it; the headline comparison uses `numopt.bench` profiles at two or more tolerances.
[research/README.md](../research/README.md) has the protocol and the table of the nine studies.

A study that passes review is promoted: the method moves into its family module with
`@register(..., params=PARAMS[...])`, its tests into `tests/`, it gets `FIXTURE_CASES`, and it is
ported to `web/src/methods/`. The promoted methods so far: `silver_gd`,
`silver_gd_strongly_convex`, `long_step_gd`, `ogm`, `fista` (`unconstrained/accelerated.py`),
`anderson_gd` (`unconstrained/anderson.py`), `arc`, `reg_newton`
(`unconstrained/regularized_newton.py`), `aaa`, `floater_hormann` (`interpolation/rational.py`),
`clenshaw_curtis` (`integration/methods.py`) and `restarted_pdhg` (`lp/pdhg.py`). Each
module's docstring names the study it came from. The studies render on the portal at
`#/research/<slug>`, built from the READMEs by `web/vite/research.ts`.

## Continuous integration

`.github/workflows/ci.yml` runs on every push to `main`, every pull request and by hand:

| Job | Steps |
|:--|:--|
| Python 3.11 and 3.13 | `pip install -e ".[dev,plot]"` into `.venv`; `ruff check src tests`; `ruff format --check src tests`; `pyright`; `pytest` |
| Parity fixtures are current | `pip install -e .`; `python web/tests/fixtures/check_generated.py` |
| Web (Node 24) | `npm ci`; `npm run lint`; `npm run format:check`; `npm run gen:test-fixtures`; `npm test`; `npm run build` (the `dist/` is uploaded as an artifact) |

`.github/workflows/pages.yml` builds `web/` and deploys it to GitHub Pages. It runs by hand
(Actions → Pages → Run workflow) after Settings → Pages → Source is set to "GitHub Actions". The
build reads only committed files (`web/src/generated/{registry,problems}.json` and `research/`),
so it needs no Python.
