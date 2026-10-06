# `web/tests/fixtures/` — Python reference data for the web tests

The web tests compare the TypeScript ports with the Python package on two kinds of data.

| data | where | written by | in git |
| --- | --- | --- | --- |
| parity fixtures (`registry.json`, `problems.json`, `fixtures/<family>.json`) | `web/src/generated/` | `npm run gen` (= `numopt export`) | yes |
| extra Python reference dumps, one per lab (`*_python.json`, `oracle.json`, ...) | `web/tests/**/fixtures/`, `web/tests/combinatorial/extra_python.json`, `web/src/methods/constrained/__fixtures__/` | `npm run gen:test-fixtures` | no (`web/.gitignore`) |

The extra dumps are about 14 MB of deterministic output, so git does not store them. Generate
them after a clone, and again after a change to the Python package:

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"   # once, from the repo root
cd web
npm run gen:test-fixtures        # runs every generator listed in gen_test_fixtures.py
npm test                         # pretest (ensure.mjs) also generates any absent dump
```

`npx vitest run` does not run `pretest`: run `npm run gen:test-fixtures` first, or the suites
that read a dump fail with `ENOENT` and the path of the absent file.

## Files in this folder

- `gen_test_fixtures.py` — runs every generator (`--missing`: only absent outputs;
  `--list`: the table of generators and their outputs). Add a new dump here and to
  `web/.gitignore`.
- `ensure.mjs` — the `pretest` hook. It finds `../.venv/bin/python` (or `$NUMOPT_PYTHON`) and
  calls `gen_test_fixtures.py --missing`.
- `check_generated.py` — `npm run gen:check`: exports into a temporary folder and compares the
  result with the committed `web/src/generated/` under the parity tolerances. It fails when the
  committed files are stale. CI runs it.
- `gen_rng_fixture.py` → `rng_python.json`: reference streams from `numopt.core.rng.Rng` (seeds
  0, 1, 42, 123456789, 2³²−1, −7): `random`, `uniform(-3, 5)`, `normal(1.5, 2)`, `integers(n)`
  for n = 1..50 twice, `permutation(n)` for n ∈ {1, 2, 5, 10, 31}, and `choice` over `a..g`.

## File format of `web/src/generated/`

`numopt export` writes each file as a JSON array with one element per line; each element is
compact JSON (no indentation). Floats use Python's shortest round-trip `repr`, so the values are
bit-exact. The format cut the parity fixtures from 11.0 MB to 7.1 MB, and a regenerated
fixture still shows a per-case `git diff`.
