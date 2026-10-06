/**
 * Browser parity of the scalar ports (test tooling, never bundled).
 *
 * Vitest runs the ports in Node, whose Math.exp/Math.log can differ from the browser's by an
 * ulp; on the libm-sensitive problems (drug_concentration, x_log_x) the Node traces may part from
 * Python at the accuracy floor, so tests/scalar/scalar.test.ts relaxes those cases. This script
 * checks the runs users actually see: it replays every Python run (the 8 parity fixtures of
 * src/generated/fixtures/scalar.json and the 104 runs of tests/scalar/fixtures/scalar_python.json,
 * which include the lab's default pairs) inside Chromium and requires nIter equal and every
 * iterate bit-identical.
 *
 *   npx vite --port 5391 --strictPort &                 # any free port
 *   node tests/scalar/browser-parity.mjs http://localhost:5391
 *
 * Exits 1 on any difference.
 */
import fs from 'node:fs';
import { chromium } from 'playwright';

const base = process.argv[2] ?? 'http://localhost:5173';
const read = (rel) => JSON.parse(fs.readFileSync(new URL(rel, import.meta.url), 'utf8'));
const runs = [
  ...read('../../src/generated/fixtures/scalar.json'),
  ...read('./fixtures/scalar_python.json').runs,
];
const cases = runs.map((c) => ({
  method: c.method,
  problem: c.problem,
  params: c.params,
  nIter: c.result.n_iter,
  xs: c.result.trace.map((s) => s.x),
}));

const browser = await chromium.launch();
const page = await (await browser.newContext()).newPage();
await page.goto(`${base}/#/`);
const out = await page.evaluate(async (cases) => {
  const reg = await import('/src/core/registry.ts');
  const problems = await import('/src/problems/registry.ts');
  await import('/src/methods/scalar/methods.ts');
  await import('/src/problems/scalar_min.ts');
  return cases.map((c) => {
    const { spec, fn } = reg.getMethod(c.method);
    const defaults = Object.fromEntries(spec.params.map((p) => [p.name, p.default]));
    const r = fn(problems.getProblem(c.problem), { ...defaults, ...c.params });
    let split = -1;
    const n = Math.min(r.trace.length, c.xs.length);
    for (let i = 0; i < n && split < 0; i++) if (r.trace[i].x !== c.xs[i]) split = i;
    return {
      label: `${c.method} on ${c.problem} ${JSON.stringify(c.params)}`,
      nIter: r.nIter,
      want: c.nIter,
      split,
    };
  });
}, cases);
await browser.close();

const bad = out.filter((o) => o.nIter !== o.want || o.split >= 0);
console.log(`${out.length - bad.length}/${out.length} runs identical to Python in Chromium`);
for (const b of bad)
  console.log(
    `  differs: ${b.label}: nIter ${b.nIter} (Python ${b.want}), first split at k = ${b.split}`,
  );
process.exit(bad.length ? 1 : 0);
