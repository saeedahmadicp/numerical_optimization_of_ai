/**
 * Full-trace agreement of the roots ports with the Python reference, beyond the parity fixtures:
 * every bracketing and open method on every roots problem, start variants, budgets, tolerances,
 * failure paths, input errors and bare callables (finite differences). Data:
 * tests/roots/fixtures/roots_oracle.json.gz (written by gen_roots_oracle.py; test-only).
 *
 * Checked per run: converged, nIter, nFev, nGev, nHev, message and extra exactly; every trace
 * step's x, f(x), step size and info to 1e-12 (relative). Invalid input must throw Python's
 * ValueError message.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { fileURLToPath } from 'node:url';
import { getMethod } from '../../src/core/registry';
import { getProblem } from '../../src/problems/registry';
import type { Result } from '../../src/core/types';
import type { RootProblem } from '../../src/problems/roots';
import '../../src/methods/roots/bracketing';
import '../../src/methods/roots/open';
import '../../src/problems/roots';

interface OracleRun {
  method: string;
  problem: string;
  params: Record<string, unknown>;
  error?: string;
  result?: Record<string, unknown>;
}

const FILE = fileURLToPath(new URL('./fixtures/roots_oracle.json.gz', import.meta.url));
const data = JSON.parse(gunzipSync(readFileSync(FILE)).toString('utf8')) as {
  problems: Record<string, number[][]>;
  runs: OracleRun[];
};

const CUSTOM: Record<string, unknown> = {
  cubic_nograd: {
    id: 'cubic_nograd',
    f: (x: number) => (x * x - 2.0) * x - 5.0,
    x0: 2.0,
    bracket: [2.0, 3.0],
  },
};

/**
 * Runs whose orbit is chaotic, where glibc itself rounds one value incorrectly, so the two
 * implementations part after a step (index → number of leading steps compared). Newton on
 * cos x − x from x₀ = 10 wanders through near-horizontal tangents; at x₅₆ = −1.3022231416374677
 * glibc's cos returns 0.26535604302967475 while the exact value rounds to 0.2653560430296748
 * (checked with a 80-digit Taylor series), and from there the orbits separate.
 */
const LIBM_CHAOS: Record<number, number> = Object.fromEntries(
  data.runs
    .map((c, i) => [c, i] as const)
    .filter(([c]) => c.method === 'newton' && c.problem === 'cos_minus_x' && c.params.x0 === 10)
    .map(([, i]) => [i, 57]),
);

/** Python's `to_jsonable`: NaN → null, ±∞ → "inf"/"-inf"; camelCase result keys → snake_case. */
function jsonable(v: unknown): unknown {
  if (typeof v === 'number') {
    if (Number.isNaN(v)) return null;
    if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf';
    return v;
  }
  if (Array.isArray(v)) return v.map(jsonable);
  if (v && typeof v === 'object')
    return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, jsonable(x)]));
  return v;
}

function pyResult(r: Result) {
  return jsonable({
    method: r.method,
    x: r.x,
    fun: r.fun,
    converged: r.converged,
    message: r.message,
    n_iter: r.nIter,
    n_fev: r.nFev,
    n_gev: r.nGev,
    n_hev: r.nHev,
    extra: r.extra,
    trace: r.trace.map((s) => ({
      k: s.k,
      x: s.x,
      fun: s.fun,
      grad_norm: s.gradNorm,
      step_size: s.stepSize,
      info: s.info,
    })),
  }) as Record<string, unknown>;
}

/** Deep comparison with a relative tolerance on numbers; returns the first difference. */
function diff(a: unknown, b: unknown, path: string, rtol: number): string | null {
  if (typeof a === 'number' && typeof b === 'number') {
    if (a === b) return null;
    return Math.abs(a - b) <= rtol * Math.max(Math.abs(a), Math.abs(b), 1e-300)
      ? null
      : `${path}: ${a} ≠ ${b}`;
  }
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return `${path}: length ${a.length} ≠ ${b.length}`;
    for (let i = 0; i < a.length; i++) {
      const d = diff(a[i], b[i], `${path}[${i}]`, rtol);
      if (d) return d;
    }
    return null;
  }
  if (a && b && typeof a === 'object' && typeof b === 'object') {
    const ka = Object.keys(a).sort(),
      kb = Object.keys(b).sort();
    if (ka.join() !== kb.join()) return `${path}: keys ${ka.join()} ≠ ${kb.join()}`;
    for (const k of ka) {
      const d = diff(
        (a as Record<string, unknown>)[k],
        (b as Record<string, unknown>)[k],
        `${path}.${k}`,
        rtol,
      );
      if (d) return d;
    }
    return null;
  }
  return a === b ? null : `${path}: ${JSON.stringify(a)} ≠ ${JSON.stringify(b)}`;
}

describe('roots problems agree with Python', () => {
  for (const [id, rows] of Object.entries(data.problems)) {
    it(id, () => {
      const p = getProblem<RootProblem>(id);
      for (const [x, f, g, h] of rows) {
        expect(diff([p.f(x), p.grad(x), p.hess(x)], [f, g, h], `${id}(${x})`, 4e-16)).toBeNull();
        // The fast drawing variant stays within a few ulps.
        expect(Math.abs(p.plot(x) - f)).toBeLessThanOrEqual(1e-14 * Math.max(1, Math.abs(f)));
      }
    });
  }
});

describe('roots methods agree with Python run by run', () => {
  data.runs.forEach((c, i) => {
    const name = `${c.method} on ${c.problem} ${JSON.stringify(c.params)} #${i}`;
    it(name, () => {
      const { spec, fn } = getMethod(c.method);
      const problem = CUSTOM[c.problem] ?? getProblem(c.problem);
      const defaults = Object.fromEntries(spec.params.map((p) => [p.name, p.default]));
      const call = () => fn(problem, { ...defaults, ...(c.params as Record<string, never>) });
      if (c.error !== undefined) {
        expect(call).toThrow(c.error);
        return;
      }
      const got = pyResult(call());
      const want = c.result!;
      const libm = LIBM_CHAOS[i];
      if (libm !== undefined) {
        // See LIBM_CHAOS: compare the stored steps before the libm rounding difference only.
        const head = (t: unknown) => (t as unknown[]).slice(0, Math.min(libm, 20));
        expect(diff(head(got.trace), head(want.trace), 'trace', 1e-12)).toBeNull();
        return;
      }
      const cut = want.trace_length as number | undefined;
      if (cut !== undefined) {
        // The fixture keeps the first 20 and the last 3 steps of long runs.
        const tr = got.trace as unknown[];
        expect(tr.length).toBe(cut);
        got.trace = [...tr.slice(0, 20), ...tr.slice(-3)];
        got.trace_length = cut;
      }

      for (const key of ['converged', 'n_iter', 'n_fev', 'n_gev', 'n_hev', 'message', 'extra'])
        expect(got[key], key).toEqual(want[key]);
      expect(diff(got, want, 'result', 1e-12)).toBeNull();
    });
  });
});
