/**
 * Unit tests for the systems port beyond the exported fixtures:
 *   - problems: metadata equals problems.json; F and J equal Python at x0, the roots and seeded
 *     points; J agrees with central differences; F(root) ≈ 0;
 *   - methods: extra Python runs (singular Jacobian, budget, finite-difference B₀, damping,
 *     every problem's default start) from tests/systems/fixtures/systems_extra.json
 *     (regenerate with `.venv/bin/python web/tests/systems/fixtures/gen_systems_extra.py`);
 *   - helpers: cond₂ equals np.linalg.cond, Python's `:.3g`, input errors.
 */
import { describe, expect, it } from 'vitest';
import { CANONICAL, sameTemplate } from '../fixtures/platform';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { getMethod, runMethod } from '../../src/core/registry';
import { getProblem } from '../../src/problems/registry';
import { reviveNumbers } from '../../src/core/json';
import type { Matrix, Vector } from '../../src/core/types';
import { cond2, fma, pyG, scaledNorm } from '../../src/methods/roots/systems';
import { listSystems, type SystemProblem } from '../../src/problems/systems';

const read = <T>(rel: string): T =>
  reviveNumbers<T>(JSON.parse(readFileSync(fileURLToPath(new URL(rel, import.meta.url)), 'utf8')));

interface Extra {
  values: { id: string; x: Vector; F: Vector; J: Matrix }[];
  runs: {
    method: string;
    problem: string;
    params: Record<string, never>;
    x: Vector;
    converged: boolean;
    message: string;
    n_iter: number;
    n_fev: number;
    n_gev: number;
    extra: Record<string, unknown>;
    xs: Vector[];
  }[];
  cond: { A: Matrix; cond: number }[];
  full: {
    method: string;
    problem: string;
    params: Record<string, never>;
    converged: boolean;
    n_iter: number;
    xs: Vector[];
  }[];
}
const EXTRA = read<Extra>('./fixtures/systems_extra.json');
const META = read<Record<string, unknown>[]>('../../src/generated/problems.json').filter(
  (p) => p.kind === 'systems',
);

const rel = (a: number, b: number) => Math.abs(a - b) / Math.max(1, Math.abs(b));

/**
 * Off the canonical platform (tests/fixtures/platform.ts) the Python dump rounds differently, and
 * runs that wander do not keep Python's path: Broyden on rosenbrock_system and freudenstein_roth
 * (no convergence in 100 steps) part after their first 10 iterates (2.5 and 70 apart at the
 * end), Newton on freudenstein_roth after them (0.06), and Broyden from [2, 1.386] at once (4.8
 * within 10 steps). There the comparison is the parity rule: the counts and the flag, the first
 * 10 iterates within 1e-8 (except the runs in PLATFORM_CHAOTIC), the final x within 1e-6 when the
 * run converged.
 */
const PLATFORM_CHAOTIC = new Set(['broyden on freudenstein_roth {"x0":[2,1.386]}']);
const PARITY_STEPS = 10;

describe('systems problems', () => {
  it('match problems.json (ids, order, metadata)', () => {
    const ts = listSystems();
    expect(ts.map((p) => p.id)).toEqual(META.map((m) => m.id));
    for (const m of META) {
      const p = getProblem<SystemProblem>(m.id as string);
      expect(p.name).toBe(m.name);
      expect(p.latex).toBe(m.latex);
      expect(p.dim).toBe(m.dim);
      expect(p.domain).toEqual(m.domain);
      expect(p.x0).toEqual(m.x0);
      expect(p.description).toBe(m.description);
      expect(p.tags).toEqual(m.tags);
      expect(p.bracket).toBe(m.bracket);
      expect(p.minima).toEqual(m.minima);
      (m.roots as Vector[]).forEach((r, i) =>
        r.forEach((v, j) => expect(p.roots[i][j]).toBeCloseTo(v, 15)),
      );
    }
  });

  it('F and J equal Python at x0, the roots and random points', () => {
    expect(EXTRA.values.length).toBeGreaterThan(20);
    for (const c of EXTRA.values) {
      const p = getProblem<SystemProblem>(c.id);
      p.f(c.x).forEach((v, i) => expect(rel(v, c.F[i])).toBeLessThan(1e-14));
      p.jac(c.x).forEach((row, i) =>
        row.forEach((v, j) => expect(rel(v, c.J[i][j])).toBeLessThan(1e-14)),
      );
      // The scalar components the lab draws use the same arithmetic.
      expect(p.components[0](c.x[0], c.x[1])).toBe(p.f(c.x)[0]);
      expect(p.components[1](c.x[0], c.x[1])).toBe(p.f(c.x)[1]);
    }
  });

  it('J agrees with central differences and F vanishes at every root', () => {
    for (const p of listSystems()) {
      const pts = [p.x0, ...p.roots, [0.3, -0.7]];
      for (const x of pts) {
        const J = p.jac(x);
        for (let j = 0; j < 2; j++) {
          const h = 1e-6 * Math.max(1, Math.abs(x[j]));
          const xp = x.slice(),
            xm = x.slice();
          xp[j] += h;
          xm[j] -= h;
          const fp = p.f(xp),
            fm = p.f(xm);
          for (let i = 0; i < 2; i++)
            expect(Math.abs((fp[i] - fm[i]) / (2 * h) - J[i][j])).toBeLessThan(
              1e-5 * Math.max(1, Math.abs(J[i][j])),
            );
        }
      }
      for (const r of p.roots) expect(scaledNorm(p.f(r))).toBeLessThan(1e-12);
    }
  });

  it('IEEE semantics far away: no exceptions, non-finite values instead', () => {
    const p = getProblem<SystemProblem>('trig_system');
    expect(p.f([Infinity, 0]).some(Number.isNaN)).toBe(true);
    const q = getProblem<SystemProblem>('circle_line');
    expect(q.f([1e200, 1e200])[0]).toBe(Infinity);
  });
});

describe('systems methods (every iterate of the long preset runs)', () => {
  // The lab's presets replay these runs step by step, so every iterate must be Python's, not
  // only the first ten: the port reproduces NumPy's fused multiply-adds for that.
  for (const c of EXTRA.full) {
    it(`${c.method} on ${c.problem} ${JSON.stringify(c.params)}: all ${c.n_iter} iterates`, () => {
      const r = runMethod(c.method, getProblem(c.problem), c.params);
      expect(r.converged).toBe(c.converged);
      expect(r.nIter).toBe(c.n_iter);
      expect(r.trace.length).toBe(c.xs.length);
      let worst = 0;
      const xs = CANONICAL ? c.xs : c.xs.slice(0, PARITY_STEPS);
      xs.forEach((x, k) =>
        x.forEach((v, i) => {
          const d = Math.abs((r.trace[k].x as Vector)[i] - v) / (1 + Math.abs(v));
          worst = Math.max(worst, d);
        }),
      );
      // 9 of the 10 runs are bit-identical; trig_system differs in the last bit of sin/cos.
      expect(worst).toBeLessThanOrEqual(CANONICAL ? 1e-12 : 1e-8);
      if (!CANONICAL && c.converged) {
        const end = c.xs[c.xs.length - 1];
        (r.x as Vector).forEach((v, i) => expect(rel(v, end[i])).toBeLessThanOrEqual(1e-6));
      }
    });
  }
});

describe('systems methods (extra Python runs)', () => {
  for (const c of EXTRA.runs) {
    it(`${c.method} on ${c.problem} ${JSON.stringify(c.params)}`, () => {
      const r = runMethod(c.method, getProblem(c.problem), c.params);
      expect(r.converged).toBe(c.converged);
      expect(r.nIter).toBe(c.n_iter);
      expect(r.nFev).toBe(c.n_fev);
      expect(r.nGev).toBe(c.n_gev);
      expect(r.extra).toEqual(c.extra);
      const mask = (m: string) => m.replace(/[-+]?\d+(\.\d+)?(e[-+]\d+)?/g, '#');
      // Another platform may round a singular J to cond₂ = inf instead of 4.8e16.
      if (CANONICAL) expect(mask(r.message)).toBe(mask(c.message));
      else expect(sameTemplate(r.message, c.message), `${r.message} vs ${c.message}`).toBe(true);
      const name = `${c.method} on ${c.problem} ${JSON.stringify(c.params)}`;
      if (!CANONICAL && (PLATFORM_CHAOTIC.has(name) || !c.converged)) {
        if (!PLATFORM_CHAOTIC.has(name))
          c.xs
            .slice(0, PARITY_STEPS)
            .forEach((x, k) =>
              x.forEach((v, i) =>
                expect(Math.abs((r.trace[k].x as Vector)[i] - v)).toBeLessThanOrEqual(
                  1e-8 * (1 + Math.abs(v)),
                ),
              ),
            );
        return;
      }
      // Every run, including those that wander for the whole budget (Broyden on Rosenbrock /
      // Freudenstein–Roth): the port replays NumPy's rounding, so their paths agree too.
      c.xs.forEach((x, k) =>
        x.forEach((v, i) =>
          expect(Math.abs((r.trace[k].x as Vector)[i] - v)).toBeLessThanOrEqual(
            1e-8 * (1 + Math.abs(v)),
          ),
        ),
      );
      (r.x as Vector).forEach((v, i) =>
        expect(Math.abs(v - c.x[i])).toBeLessThanOrEqual(1e-10 + 1e-6 * Math.abs(c.x[i])),
      );
    });
  }

  it('trace contract: one step per iteration, info keys per method', () => {
    const p = getProblem('intersecting_circles');
    const n = runMethod('newton_system', p, { damping: true });
    expect(n.trace.length).toBe(n.nIter + 1);
    expect(Object.keys(n.trace[0].info).sort()).toEqual(['residual', 'residual_norm']);
    expect(Object.keys(n.trace[1].info).sort()).toEqual(
      ['alpha', 'jacobian', 'newton_step', 'residual', 'residual_norm', 'step', 'trials'].sort(),
    );
    // Damped: the first trial is the full step α = 1 with φ = ½‖F(x + p)‖².
    const t = n.trace[1].info.trials as [number, number][];
    expect(t[0][0]).toBe(1);
    const b = runMethod('broyden', p, {});
    expect(Object.keys(b.trace[1].info).sort()).toEqual(
      ['jacobian', 'residual', 'residual_norm', 'secant', 'step'].sort(),
    );
    // The Broyden matrix after step k satisfies the secant equation B_k s = y.
    const B = b.trace[2].info.jacobian as Matrix;
    const { s, y } = b.trace[1].info.secant as { s: Vector; y: Vector };
    const Bs = B.map((row) => row[0] * s[0] + row[1] * s[1]);
    Bs.forEach((v, i) => expect(Math.abs(v - y[i])).toBeLessThan(1e-12 * (1 + Math.abs(y[i]))));
  });

  it('the Newton point is where the two linearizations cross', () => {
    const p = getProblem<SystemProblem>('trig_system');
    const r = runMethod('newton_system', p, {});
    const x0 = r.trace[0].x as Vector;
    const F0 = r.trace[0].info.residual as Vector;
    const J = r.trace[1].info.jacobian as Matrix;
    const x1 = r.trace[1].x as Vector;
    for (let i = 0; i < 2; i++) {
      const lin = F0[i] + J[i][0] * (x1[0] - x0[0]) + J[i][1] * (x1[1] - x0[1]);
      expect(Math.abs(lin)).toBeLessThan(1e-14);
    }
  });

  it('rejects invalid input like Python', () => {
    const p = getProblem('circle_line');
    expect(() => runMethod('newton_system', p, { x0: [1, 2, 3] })).toThrow(/x0 has 3 entries/);
    expect(() => runMethod('broyden', p, { jacobian0: 'nope' })).toThrow(/jacobian0/);
    expect(() => runMethod('newton_system', p, { nope: 1 })).toThrow(/unknown parameter/);
    const nonsquare = { id: 'ns', dim: 2, x0: [1, 1], f: (x: Vector) => [x[0], x[1], 1] };
    expect(() => getMethod('newton_system').fn(nonsquare, { x0: [1, 1] })).toThrow(/square/);
  });

  it('never throws on breakdown: non-finite F(x₀), divergence', () => {
    const p = getProblem<SystemProblem>('trig_system');
    const r = runMethod('newton_system', p, { x0: [NaN, 0] });
    expect(r.converged).toBe(false);
    expect(r.message).toBe('F(x₀) is not finite');
    // Newton on arctan from |x₀| > 1.3917 overshoots further every step.
    const bad = {
      id: 'atan',
      dim: 2,
      x0: [2, 0],
      f: (x: Vector) => [Math.atan(x[0]), x[1]],
      jac: (x: Vector) => [
        [1 / (1 + x[0] * x[0]), 0],
        [0, 1],
      ],
    };
    const d = getMethod('newton_system').fn(bad, { ftol: 1e-10, xtol: 1e-12, max_iter: 100 });
    expect(d.converged).toBe(false);
    expect(d.message).toMatch(/diverged|singular/);
  });
});

describe('numerical helpers', () => {
  it('cond₂ equals np.linalg.cond (2×2 via the LAPACK path, n×n via Jacobi)', () => {
    // Off the canonical platform, κ₂ ≥ 1e14 of a numerically singular matrix is σ_max over a
    // rounding-level σ_min (4.8e16 on aarch64, 2.5e16 on x86-64): only "huge" is shared.
    for (const c of EXTRA.cond)
      if (!CANONICAL && c.cond >= 1e14) expect(cond2(c.A)).toBeGreaterThanOrEqual(1e14);
      else expect(rel(cond2(c.A), c.cond)).toBeLessThan(1e-12);
  });
  it('fma rounds once', () => {
    expect(fma(0.1, 10, -1)).toBe(5.551115123125783e-17);
    expect(fma(2, 3, 4)).toBe(10);
  });
  it("pyG mirrors Python's :.3g", () => {
    expect(pyG(1e12)).toBe('1e+12');
    expect(pyG(0)).toBe('0');
    expect(pyG(1.5e-7)).toBe('1.5e-07');
    expect(pyG(123456)).toBe('1.23e+05');
    expect(pyG(Infinity)).toBe('inf');
    expect(pyG(2.48e-16)).toBe('2.48e-16');
    expect(pyG(0.5)).toBe('0.5');
    expect(pyG(12.0)).toBe('12');
    expect(pyG(4.48e16)).toBe('4.48e+16');
  });
  it('scaledNorm neither overflows nor underflows', () => {
    expect(scaledNorm([3e300, 4e300])).toBeCloseTo(5e300, -290);
    expect(scaledNorm([3e-300, 4e-300])).toBeCloseTo(5e-300, 310);
    expect(scaledNorm([0, 0])).toBe(0);
    expect(Number.isNaN(scaledNorm([NaN, 1]))).toBe(true);
  });
});
