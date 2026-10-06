/**
 * The roots lab's drawn mathematics: for every step of every method on every roots problem,
 * the model drawn for the step has its zero at x_k (tangent, chord, hyperbola, parabola,
 * inverse parabola, fixed-point line), the animated head starts at (x_k, f(x_k)) and lands on
 * (x_{k+1}, f(x_{k+1})), the camera frames the step, and the filled update rule is valid KaTeX.
 */
import { describe, expect, it } from 'vitest';
import katex from 'katex';
import { listMethods } from '../../src/core/registry';
import { listProblems } from '../../src/problems/registry';
import type { Result, Step } from '../../src/core/types';
import type { RootProblem } from '../../src/problems/roots';
import {
  cameraAt,
  clampWindow,
  headAt,
  hyperbola,
  mullerParabola,
  stepGeometry,
  stepWindow,
} from '../../src/labs/roots/geometry';
import { CARD_QUANTITIES, cardStep, filledRule } from '../../src/labs/roots/rules';
import '../../src/labs/roots/setup';

const methods = listMethods('roots');
const problems = listProblems<RootProblem>('roots');

function runAll(): { id: string; p: RootProblem; r: Result }[] {
  const out: { id: string; p: RootProblem; r: Result }[] = [];
  for (const m of methods)
    for (const p of problems) {
      const params = Object.fromEntries(m.spec.params.map((s) => [s.name, s.default]));
      try {
        out.push({ id: m.spec.id, p, r: m.fn(p, params) });
      } catch {
        // invalid default input (none expected)
      }
    }
  return out;
}
const runs = runAll();

const near = (a: number, b: number, scale: number) =>
  expect(Math.abs(a - b)).toBeLessThanOrEqual(1e-6 * Math.max(1, Math.abs(scale)));

describe('step geometry has its zero at the iterate', () => {
  it('runs every method on every problem', () => {
    expect(runs.length).toBe(methods.length * problems.length);
  });
  for (const { id, p, r } of runs) {
    it(`${id} on ${p.id}`, () => {
      r.trace.forEach((s: Step, k) => {
        const x = s.x as number;
        if (!Number.isFinite(x) || k === 0) return;
        const info = s.info;
        const geo = stepGeometry(id, s, p.plot);
        if (id === 'newton') {
          const tg = info.tangent as { point: [number, number]; slope: number };
          near(tg.point[0] - tg.point[1] / tg.slope, x, x);
          const seg = geo.prims.find((q) => q.t === 'seg');
          expect(seg && seg.t === 'seg' && seg.b).toEqual([x, 0]);
        }
        if (id === 'halley') {
          const h = info.hyperbola as {
            center: number;
            alpha: number;
            beta: number;
            gamma: number;
          };
          near(hyperbola(h)(x), 0, 1);
          near(
            hyperbola(h)(h.center),
            (info.derivatives as number[])[0],
            (info.derivatives as number[])[0],
          );
        }
        if (id === 'muller') {
          const par = info.parabola as { center: number; a: number; b: number; c: number };
          const fx = mullerParabola(par)(x);
          expect(Math.abs(fx)).toBeLessThanOrEqual(1e-6 * Math.max(1, Math.abs(par.c)));
        }
        if (id === 'inverse_quadratic_interpolation') {
          const ip = info.inverse_parabola as { a: number; b: number; c: number };
          near(ip.c, x, x);
        }
        if (
          ['regula_falsi', 'illinois', 'pegasus', 'anderson_bjorck'].includes(id) &&
          info.step === 'chord'
        ) {
          const [[a, fa], [b, fb]] = info.chord as number[][];
          near(b - (fb * (b - a)) / (fb - fa), x, x);
        }
        if (geo.bracket) {
          expect(x).toBeGreaterThanOrEqual(geo.bracket[0]);
          expect(x).toBeLessThanOrEqual(geo.bracket[1]);
        }
        // The head starts at (x_{k−1}, f_{k−1}) and lands on (x_k, f_k).
        const prev = r.trace[k - 1];
        expect(headAt(id, prev, s, 0)).toEqual([prev.x, prev.fun]);
        const end = headAt(id, prev, s, 1);
        expect(end[0]).toBe(x);
        near(end[1], s.fun ?? 0, s.fun ?? 1);
        // The filled rule is valid KaTeX.
        const rule = filledRule(id, s);
        if (rule) expect(() => katex.renderToString(rule, { throwOnError: true })).not.toThrow();
        // Every card quantity resolves on the derived step (or is honestly absent).
        const cs = cardStep(id, r.trace, k)!;
        for (const q of CARD_QUANTITIES[id] ?? []) {
          const v = q.key === 'stepSize' ? cs.stepSize : cs.info[q.key.slice(5)];
          expect(
            v === undefined || v === null || typeof v === 'number' || typeof v === 'string',
          ).toBe(true);
        }
      });
    });
  }
});

describe('camera', () => {
  it('frames the whole problem at k = 0 and each later step', () => {
    for (const { p, r } of runs) {
      const w0 = stepWindow(r.trace, 0, p.domain);
      expect(w0[0]).toBeLessThanOrEqual(p.domain[0]);
      expect(w0[1]).toBeGreaterThanOrEqual(p.domain[1]);
      for (let k = 1; k < r.trace.length; k++) {
        const x = r.trace[k].x as number;
        if (!Number.isFinite(x)) continue;
        const [lo, hi] = cameraAt(r.trace, k, p.domain);
        expect(hi).toBeGreaterThan(lo);
        const raw = stepWindow(r.trace, k, p.domain);
        if (raw[0] === lo && raw[1] === hi) {
          expect(x).toBeGreaterThanOrEqual(lo);
          expect(x).toBeLessThanOrEqual(hi);
        }
      }
    }
  });
  it('zooms geometrically between steps and never widens past 12 problem widths', () => {
    const trace = runs.find((q) => q.id === 'bisection' && q.p.id === 'cubic')!.r.trace;
    const w = (t: number) => {
      const [a, b] = cameraAt(trace, t, [0, 3.5]);
      return b - a;
    };
    expect(w(5.5)).toBeLessThan(w(5));
    expect(Math.sqrt(w(5) * w(6))).toBeCloseTo(w(5.5), 10);
    expect(clampWindow([-1e13, 1e13], [-6, 6])[1] - clampWindow([-1e13, 1e13], [-6, 6])[0]).toBe(
      144,
    );
  });
});
