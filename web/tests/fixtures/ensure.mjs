// `npm test` runs this first (the "pretest" script). It generates the Python reference dumps
// that git ignores (see gen_test_fixtures.py) when any of them is absent. It needs the numopt
// package: ../.venv/bin/python by default, or the interpreter named by $NUMOPT_PYTHON.
import { existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const here = (rel) => fileURLToPath(new URL(rel, import.meta.url));
const candidates = [
  process.env.NUMOPT_PYTHON,
  here('../../../.venv/bin/python'),
  here('../../../.venv/Scripts/python.exe'),
].filter(Boolean);
const python = candidates.find((p) => p === process.env.NUMOPT_PYTHON || existsSync(p));

if (!python) {
  console.error(
    [
      'pretest: no Python environment for numopt, so the Python reference dumps under',
      'web/tests/**/fixtures cannot be generated (git does not store them). Create it once:',
      '  python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"   # from the repo root',
      'or point NUMOPT_PYTHON at an interpreter that can import numopt, then run',
      '  npm run gen:test-fixtures',
    ].join('\n'),
  );
  process.exit(1);
}

const run = spawnSync(python, [here('./gen_test_fixtures.py'), '--missing'], {
  stdio: 'inherit',
});
process.exit(run.status ?? 1);
