/**
 * Constrained family: TS port vs the Python reference beyond the parity fixtures.
 *
 * `__fixtures__/constrained_reference.json` (test data only; regenerate with
 * `.venv/bin/python web/src/methods/constrained/__fixtures__/gen_constrained_reference.py`) holds
 * every method on every compatible problem, extra starts and parameters, failure paths, direct
 * `solve_qp` calls and the problem values. Checked here: nIter, converged, the message, the
 * evaluation counts and `extra`, the first 12 steps and the last step in full (info included),
 * the QP solutions, and f, ∇f, ∇²f, c, ∇c, ∇²c and the metadata of every problem.
 */
import { describe, expect, it } from 'vitest';
import { CANONICAL } from '../../../tests/fixtures/platform';
import { reviveNumbers, stepFromJson } from '../../core/json';
import { getMethod } from '../../core/registry';
import { getProblem } from '../../problems/registry';
import type { Result } from '../../core/types';
import type { ConstrainedProblem } from '../../problems/constrained';
import { lstsq, pyG, pyRepr, solveQp } from './methods';
import './methods';
import '../../problems/constrained';

interface Summary {
  n_iter: number;
  converged: boolean;
  message: string;
  x: number[];
  fun: number;
  n_fev: number;
  n_gev: number;
  n_hev: number;
  extra: Record<string, unknown>;
  head: Record<string, unknown>[];
  last: Record<string, unknown>;
  n_steps: number;
}
interface Case {
  method: string;
  problem: string;
  params: Record<string, unknown>;
  result?: Summary;
  error?: string;
}
interface QpCase {
  args: (number[][] | number[] | null)[];
  ok: boolean;
  x: number[];
  lam_eq: number[];
  lam_ub: number[];
  active: number[];
  n_iter: number;
  message: string;
}
interface ProblemValues {
  points: {
    x: number[];
    f: number;
    grad: number[];
    hess: number[][];
    c: number[];
    cgrad: number[][];
    chess: number[][][];
  }[];
  extra: Record<string, unknown>;
  meta: Record<string, unknown>;
}

/**
 * Deep comparison with the Python reference (as tests/shared-ports/compare.ts): Python NaN is
 * JSON null; numbers match when |got − want| ≤ atol + rtol·|want|; returns the first mismatch.
 */
interface Tol {
  rtol: number;
  atol: number;
}

function mismatch(got: unknown, want: unknown, tol: Tol, path = '$'): string | null {
  if (want === null || want === undefined) {
    const ok = got === null || got === undefined || (typeof got === 'number' && Number.isNaN(got));
    return ok ? null : `${path}: got ${String(got)}, want null`;
  }
  if (typeof want === 'number') {
    if (typeof got !== 'number') return `${path}: got ${JSON.stringify(got)}, want ${want}`;
    if (!Number.isFinite(want) || !Number.isFinite(got))
      return Object.is(got, want) || got === want ? null : `${path}: got ${got}, want ${want}`;
    return Math.abs(got - want) <= tol.atol + tol.rtol * Math.abs(want)
      ? null
      : `${path}: got ${got}, want ${want} (diff ${Math.abs(got - want)})`;
  }
  if (Array.isArray(want)) {
    if (!Array.isArray(got)) return `${path}: got ${JSON.stringify(got)}, want an array`;
    if (got.length !== want.length) return `${path}: length ${got.length}, want ${want.length}`;
    for (let i = 0; i < want.length; i++) {
      const m = mismatch(got[i], want[i], tol, `${path}[${i}]`);
      if (m) return m;
    }
    return null;
  }
  if (typeof want === 'object') {
    if (got === null || typeof got !== 'object' || Array.isArray(got))
      return `${path}: got ${JSON.stringify(got)}, want an object`;
    const gk = Object.keys(got).sort(),
      wk = Object.keys(want).sort();
    if (gk.join() !== wk.join()) return `${path}: keys [${gk.join(', ')}], want [${wk.join(', ')}]`;
    for (const k of wk) {
      const m = mismatch(
        (got as Record<string, unknown>)[k],
        (want as Record<string, unknown>)[k],
        tol,
        `${path}.${k}`,
      );
      if (m) return m;
    }
    return null;
  }
  return got === want ? null : `${path}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`;
}

// Test data only: loaded through Vite's glob import (never reachable from the app bundle).
const RAW = import.meta.glob('./__fixtures__/constrained_reference.json', {
  eager: true,
  import: 'default',
});
const REF = reviveNumbers<{ cases: Case[]; qp: QpCase[]; problems: Record<string, ProblemValues> }>(
  Object.values(RAW)[0],
);

const STEP_TOL: Tol = { rtol: 1e-8, atol: 1e-12 };
const INFO_TOL: Tol = { rtol: 1e-7, atol: 1e-10 };
const FINAL_TOL: Tol = { rtol: 1e-6, atol: 1e-9 };
/**
 * Info keys that divide rounding noise: SQP's merit penalty μ = (∇fᵀp + ½pᵀBp)/((1 − ρ)‖c⁻‖₁)
 * once ‖c⁻‖₁ ≈ 1e-12, and the merit built from it. The iterates themselves agree to 1e-16.
 */
const NOISY: Record<string, string[]> = { sqp: ['mu', 'merit'] };
const NOISY_TOL: Tol = { rtol: 1e-3, atol: 1e-10 };

const NUM = /[-+]?\d+(?:\.\d+)?(?:e[-+]\d+)?/gi;
/** Same message up to the last digit of its 3-significant-digit numbers (residuals ≈ 1e-16 are noise). */
function sameMessage(got: string, want: string): boolean {
  if (got.replace(NUM, '#') !== want.replace(NUM, '#')) return false;
  const a = got.match(NUM) ?? [],
    b = want.match(NUM) ?? [];
  return a.every((t, i) => {
    const x = Number(t),
      y = Number(b[i]);
    return Math.abs(x - y) <= 1e-12 + 0.05 * Math.abs(y);
  });
}

function run(c: Case): Result {
  const { spec, fn } = getMethod(c.method);
  const defaults = Object.fromEntries(spec.params.map((p) => [p.name, p.default]));
  return fn(getProblem(c.problem), { ...defaults, ...(c.params as Record<string, never>) });
}

/** A step as the exporter writes it (camelCase fields of `stepFromJson`). */
const stepOf = (raw: Record<string, unknown>) => stepFromJson(raw);

describe('constrained methods vs the Python reference', () => {
  REF.cases.forEach((c, idx) => {
    const name = `${c.method} on ${c.problem} ${JSON.stringify(c.params)} #${idx}`;
    if (c.error !== undefined) {
      it(`${name} raises ValueError`, () => {
        expect(() => run(c)).toThrow();
        try {
          run(c);
        } catch (e) {
          // Same message where the method raises it itself (Python's run() may add a prefix).
          expect(c.error).toContain((e as Error).message.slice(0, 40));
        }
      });
      return;
    }
    it(name, () => {
      const want = c.result!;
      const got = run(c);
      expect(got.nIter).toBe(want.n_iter);
      expect(got.converged).toBe(want.converged);
      expect(got.trace.length).toBe(want.n_steps);
      expect([got.nFev, got.nGev, got.nHev]).toEqual([want.n_fev, want.n_gev, want.n_hev]);
      expect(sameMessage(got.message, want.message)).toBe(true);
      want.head.forEach((raw, k) => {
        const ref = stepOf(raw);
        const step = got.trace[k];
        expect(mismatch(step.x, ref.x, STEP_TOL, `trace[${k}].x`)).toBeNull();
        expect(mismatch(step.fun, ref.fun, STEP_TOL, `trace[${k}].fun`)).toBeNull();
        expect(mismatch(step.stepSize, ref.stepSize, STEP_TOL, `trace[${k}].stepSize`)).toBeNull();
        const { info: gi, ...gs } = step;
        const { info: wi, ...ws } = ref;
        expect(Object.keys(gs).sort()).toEqual(Object.keys(ws).sort());
        expect(Object.keys(gi).sort()).toEqual(Object.keys(wi).sort());
        for (const key of Object.keys(wi)) {
          // Quotients of rounding noise near feasibility / stationarity (see the file header).
          // Off the canonical platform (tests/fixtures/platform.ts) every info value is held to
          // NOISY_TOL: the dump's own rounding moves e.g. SQP's θ by 6e-5 relative (sqp on
          // circle_eq from [0.05, 0.05], measured with two OpenBLAS kernels); x and f stay at
          // STEP_TOL.
          const tol = !CANONICAL || NOISY[c.method]?.includes(key) ? NOISY_TOL : INFO_TOL;
          expect(mismatch(gi[key], wi[key], tol, `trace[${k}].info.${key}`)).toBeNull();
        }
      });
      const last = got.trace[got.trace.length - 1];
      const wantLast = stepOf(want.last);
      expect(last.k).toBe(wantLast.k);
      expect(mismatch(last.x, wantLast.x, FINAL_TOL, 'last.x')).toBeNull();
      expect(mismatch(last.fun, wantLast.fun, FINAL_TOL, 'last.fun')).toBeNull();
      for (const key of ['outer', 'inner', 'active'])
        expect(last.info[key], `last.info.${key}`).toEqual(wantLast.info[key]);
      expect(mismatch(got.x, want.x, FINAL_TOL, 'x')).toBeNull();
      const { kkt_residual: kg, violation: vg, multipliers: mg, ...countsG } = got.extra;
      const { kkt_residual: kw, violation: vw, multipliers: mw, ...countsW } = want.extra;
      expect(countsG).toEqual(countsW);
      expect(mismatch(mg, mw, { rtol: 1e-4, atol: 1e-7 }, 'multipliers')).toBeNull();
      expect(mismatch([kg, vg], [kw, vw], { rtol: 1e-1, atol: 1e-12 }, 'kkt')).toBeNull();
    });
  });
});

describe('solveQp (Goldfarb–Idnani) vs numopt.constrained.methods.solve_qp', () => {
  REF.qp.forEach((q, idx) => {
    it(`QP #${idx}: ${q.message}`, () => {
      const [G, a, Ae, be, Ai, bi] = q.args as [
        number[][],
        number[],
        number[][] | null,
        number[] | null,
        number[][] | null,
        number[] | null,
      ];
      const r = solveQp(G, a, Ae, be, Ai, bi);
      expect(r.ok).toBe(q.ok);
      expect(r.message).toBe(q.message);
      expect(r.nIter).toBe(q.n_iter);
      if (!q.ok) return;
      expect(r.active).toEqual(q.active);
      expect(mismatch(r.x, q.x, { rtol: 1e-12, atol: 1e-14 })).toBeNull();
      expect(mismatch(r.lamEq, q.lam_eq, { rtol: 1e-10, atol: 1e-13 })).toBeNull();
      expect(mismatch(r.lamUb, q.lam_ub, { rtol: 1e-10, atol: 1e-13 })).toBeNull();
    });
  });
});

describe('constrained problems vs numopt.problems.constrained', () => {
  for (const [id, ref] of Object.entries(REF.problems)) {
    it(`${id}: values, derivatives and metadata`, () => {
      const p = getProblem<ConstrainedProblem>(id);
      const tol: Tol = { rtol: 1e-13, atol: 1e-13 };
      for (const pt of ref.points) {
        const x = pt.x;
        expect(mismatch(p.f(x), pt.f, tol, 'f')).toBeNull();
        expect(mismatch(p.grad(x), pt.grad, tol, 'grad')).toBeNull();
        expect(mismatch(p.hess(x), pt.hess, tol, 'hess')).toBeNull();
        expect(
          mismatch(
            p.constraints.map((c) => c.fun(x)),
            pt.c,
            tol,
            'c',
          ),
        ).toBeNull();
        expect(
          mismatch(
            p.constraints.map((c) => c.grad(x)),
            pt.cgrad,
            tol,
            'cgrad',
          ),
        ).toBeNull();
        expect(
          mismatch(
            p.extra.constraint_hess.map((h) => h(x)),
            pt.chess,
            tol,
            'chess',
          ),
        ).toBeNull();
      }
      const { constraint_hess: _drop, ...extra } = p.extra;
      void _drop;
      expect(mismatch(extra, ref.extra, { rtol: 0, atol: 0 }, 'extra')).toBeNull();
      const meta = ref.meta;
      expect(p.name).toBe(meta.name);
      expect(p.latex).toBe(meta.latex);
      expect(p.description).toBe(meta.description);
      expect(p.tags).toEqual(meta.tags);
      expect(p.domain).toEqual(meta.domain);
      expect(p.x0).toEqual(meta.x0);
      expect(mismatch(p.minima, meta.minima, { rtol: 0, atol: 0 })).toBeNull();
      expect(p.constraints.map((c) => ({ kind: c.kind, latex: c.latex }))).toEqual(
        meta.constraints,
      );
    });
  }
});

describe('Python formatting helpers', () => {
  it('pyG matches format(v, ".3g")', () => {
    expect(pyG(6.37e-7)).toBe('6.37e-07');
    expect(pyG(0.019)).toBe('0.019');
    expect(pyG(1e-6)).toBe('1e-06');
    expect(pyG(0)).toBe('0');
    expect(pyG(1.02)).toBe('1.02');
    expect(pyG(1e10)).toBe('1e+10');
    expect(pyG(123.4)).toBe('123');
    expect(pyG(1234.5)).toBe('1.23e+03');
    expect(pyG(NaN)).toBe('nan');
  });
  it('pyRepr matches repr(float)', () => {
    expect(pyRepr(1)).toBe('1.0');
    expect(pyRepr(-0.5)).toBe('-0.5');
    expect(pyRepr(1e-5)).toBe('1e-05');
    expect(pyRepr(1.5e-7)).toBe('1.5e-07');
    expect(pyRepr(1e16)).toBe('1e+16');
    expect(pyRepr(0.1 + 0.2)).toBe('0.30000000000000004');
  });
  it('lstsq gives the minimum-norm least-squares solution', () => {
    // Underdetermined: x + y = 2 → (1, 1).
    expect(mismatch(lstsq([[1, 1]], [2]), [1, 1], { rtol: 1e-15, atol: 1e-15 })).toBeNull();
    // Overdetermined: columns (1, 1)ᵀ, b = (1, 3) → 2.
    expect(mismatch(lstsq([[1], [1]], [1, 3]), [2], { rtol: 1e-15, atol: 1e-15 })).toBeNull();
    // Rank-deficient: duplicated rows, minimum norm.
    expect(
      mismatch(
        lstsq(
          [
            [1, 1],
            [2, 2],
          ],
          [2, 4],
        ),
        [1, 1],
        { rtol: 1e-13, atol: 1e-13 },
      ),
    ).toBeNull();
  });
});
