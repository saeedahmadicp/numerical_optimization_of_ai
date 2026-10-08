# Contributing to numopt

numopt has three parts that share one catalog of methods and problems: the Python package
([`src/numopt/`](src/numopt/)), the web portal ([`web/`](web/)) and the research studies
([`research/`](research/)). [docs/architecture.md](docs/architecture.md) explains how they fit
together; this page says how to set up, what to run before a pull request, and the recipe for each
kind of change.

## Set up

Python 3.11 or later and, for the portal, Node 24.

```bash
git clone https://github.com/saeedahmadicp/numopt
cd numopt
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
# with uv instead: uv venv && uv pip install -e ".[dev]"
cd web && npm install
```

The virtual environment must be `.venv/` at the repository root: `pyproject.toml` points pyright
at it, and the web scripts call `../.venv/bin/numopt` and `../.venv/bin/python`.

## Before you open a pull request

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs three jobs. Run the same commands
locally.

| Job | Commands (from the repository root unless noted) |
|:--|:--|
| Python 3.11 and 3.13 | `.venv/bin/ruff check src tests` · `.venv/bin/ruff format --check src tests` · `.venv/bin/pyright` · `.venv/bin/pytest` |
| Parity fixtures are current | `.venv/bin/python web/tests/fixtures/check_generated.py` (in `web/`: `npm run gen:check`) |
| Web (in `web/`) | `npm run lint` · `npm run format:check` · `npm run gen:test-fixtures` · `npm test` · `npm run build` |

`npm run e2e` runs the Playwright smoke tests on a preview build (`E2E_PORT=…` picks a private
port). It is not part of CI; run it when you change routes, the home page or a lab's layout.

### The repository hygiene scripts

The web tests compare the TypeScript ports with Python output of two kinds
([`web/tests/fixtures/README.md`](web/tests/fixtures/README.md)):

| Data | Written by | In git |
|:--|:--|:--|
| Parity fixtures: `web/src/generated/{registry,problems}.json`, `fixtures/<family>.json` | `npm run gen` (`numopt export`) | yes |
| Per-lab Python reference dumps under `web/tests/**/fixtures/` (about 14 MB) | `npm run gen:test-fixtures` | no |

* **`npm run gen`** — run it after every change to a method, a problem, a parameter or
  `FIXTURE_CASES`, and commit `web/src/generated/`. Never edit those files by hand.
* **`npm run gen:check`** (`check_generated.py`) — exports into a temporary folder and compares
  the result with the committed files under the parity tolerances. It exits with status 1 and
  lists every difference when the committed fixtures are stale. CI runs it.
* **`npm run gen:test-fixtures`** (`gen_test_fixtures.py`) — writes every dump that git ignores;
  `--missing` writes only the absent ones and `--list` prints the table of generators. A new dump
  goes into `GENERATORS` there and into `web/.gitignore`.
* **`pretest`** (`ensure.mjs`) — `npm test` first generates any absent dump with
  `../.venv/bin/python` (or `$NUMOPT_PYTHON`). `npx vitest run` skips this hook, so run
  `npm run gen:test-fixtures` first when you call vitest directly.

## Add or change a method

1. **Implement it** in its family module under `src/numopt/<family>/`, decorated with
   `@register(...)`: every keyword is a `ParamSpec` with a default, a UI range and help text;
   `needs`, `order`, `summary` and `references` are filled in. Follow the method contract in
   [docs/architecture.md](docs/architecture.md#method-contract-python):
   * the docstring names the algorithm and the equations it implements and states the stopping
     test; every deviation from the source is a `# NOTE:` with the reason;
   * one `Step` for k = 0 and one per iteration, with the geometry in `Step.info` (list every key
     in the module docstring's `Info keys:` section);
   * exact `n_fev` / `n_gev` / `n_hev` counts with `Counted`;
   * `converged=True` only when the documented test passed; never raise on numerical breakdown;
   * randomness only from `numopt.core.rng.Rng(seed)`, never `np.random`.
2. **Test it** in `tests/test_<package>_<module>.py`: convergence on at least two problems, an
   oracle comparison with SciPy or NumPy where an equivalent exists, a failure-path test, the
   `assert_valid_result` contract check from `tests/conftest.py`, and Hypothesis property tests
   where an invariant exists.
3. **Add fixture cases**: append `(method_id, problem_id, params)` to the module's
   `FIXTURE_CASES`, then run `npm run gen` in `web/`.
4. **Port it** to `web/src/methods/<package>/<module>.ts` with the same id, parameters, trace
   semantics and `info` keys ([web/README.md, "Porting a method"](web/README.md#porting-a-method)).
   `npm test` replays every fixture case against the port: the first ten iterates within 10⁻⁸,
   the final iterate within 10⁻⁶ (relative) and the iteration count exactly.
5. The method then appears in `numopt list`, on the Methods page and in its family's lab without
   further changes.

A test problem follows the same path: `src/numopt/problems/<kind>.py` (exact derivatives, a
plotting `domain`, a default `x0` or `bracket`, the known minima or roots, all checked by tests),
then `web/src/problems/<kind>.ts`.

## Add or change a lab

Each lab is one folder, `web/src/labs/<id>/`, and no shared file changes to add one: `meta.ts`
(title, group, pitch, families) and `index.tsx` (the lab, built on the shared `LabShell`). The
registry in `web/src/labs/index.ts` discovers both. The home page needs a preview in
`web/src/app/home/previews/labs/<id>.ts`, and a test fails without one.
[web/README.md](web/README.md#how-to-build-a-lab) has the full recipe.

Every change to the portal keeps these rules:

* **No layout shift.** A component keeps its size while its content changes (playback, a new run,
  a recomputation).
* **Hover never moves an element.** No translate, lift or scale on cards or buttons; a hover
  changes color, border or an inner animation in place.
* **Every text is legible under critical review:** size, weight, contrast in both themes, font
  family, and the same name for the same thing everywhere (method names come from the registry).
* **The home page stays clean:** the hero showcase and the visual lab gallery with live hover
  previews, and no dense index. The approved design is in
  [web/README.md, "Home page — approved design"](web/README.md#home-page--approved-design-do-not-revert);
  propose changes to it before you make them.
* Motion respects `prefers-reduced-motion`; every control works from the keyboard.

Check a visual change with Playwright at 1440 × 900 and 390 × 844, in light and dark, with no
horizontal overflow ([web/README.md, "Visual verification"](web/README.md#visual-verification)).

## Add a research study

A new or improved method starts as a study in `research/<slug>/`: a falsifiable question, a
budget-matched comparison against numopt's baselines with `numopt.bench` profiles at two or more
tolerances, a *Where it loses* section, and a deterministic `run.py`. Code in `research/` imports
`numopt` and never changes it. [research/README.md](research/README.md) has the protocol, the
checklist and the promotion steps into `src/numopt/`.

Run a study's tests one folder at a time (every study has a `test_method.py`):

```bash
for d in research/*/; do .venv/bin/python -m pytest "$d" -q -o addopts=""; done
```

## Writing

Documentation, UI text and messages follow the voice in [docs/brand/brand.md](docs/brand/brand.md):
name the quantity, then the symbol; state outcomes with their evidence ("Converged in 38
iterations · ‖∇f‖∞ = 1.3×10⁻¹¹ ≤ 10⁻⁸"); use the textbook's name for a method; U+2212 for minus and ×
for scientific notation outside code. A count in the README or on the portal comes from the
registry (`numopt list`, `numopt problems`), never from memory.

The README screenshots live in [`docs/assets/screens/`](docs/assets/screens/) (WebP, light theme,
1440 × 900, at most 300 KB each). Recapture them when a pictured lab changes.

## License

By contributing, you agree that your contribution is released under the [MIT License](LICENSE).
