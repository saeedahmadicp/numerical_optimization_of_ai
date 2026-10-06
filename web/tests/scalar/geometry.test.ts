/**
 * The 1-D lab's pure geometry (src/labs/scalar/{geometry,camera,stepNote,columns}.ts): what the
 * stage draws must be read from Step.info exactly, and the derived quantities (evaluation counts,
 * proportions, contraction rates, the follow camera) must hold for every method and problem.
 */
import { describe, expect, it } from 'vitest';
import { getMethod, listMethods } from '../../src/core/registry';
import type { Result } from '../../src/core/types';
import { getProblem, listProblems } from '../../src/problems/registry';
import type { ScalarProblem } from '../../src/problems/scalar_min';
import '../../src/methods/scalar/methods';
import '../../src/problems/scalar_min';
import {
  ERROR_FLOOR,
  GUIDE_RATES,
  kindOf,
  metricValues,
  nearestMinimizer,
  nextView,
  parabolaAt,
  proportions,
  ratePerEvaluation,
  stepViews,
  usesStart,
} from '../../src/labs/scalar/geometry';
import { PAD, followView, fullView } from '../../src/labs/scalar/camera';
import { digitsFor, stepNote, texNum } from '../../src/labs/scalar/stepNote';
import { columnsFor } from '../../src/labs/scalar/columns';
import { PRESETS, DEFAULT_SELECTION, DEFAULT_PROBLEM } from '../../src/labs/scalar/presets';

const METHODS = listMethods('scalar').map((m) => m.spec.id);
const PROBLEMS = listProblems<ScalarProblem>('scalar_min');

function run(id: string, problemId: string, params: Record<string, unknown> = {}): Result {
  const { spec, fn } = getMethod(id);
  const defaults = Object.fromEntries(spec.params.map((p) => [p.name, p.default]));
  return fn(getProblem(problemId), { ...defaults, ...(params as Record<string, never>) });
}

describe('stepViews', () => {
  for (const id of METHODS)
    it(`${id}: one view per step; evaluations add up to the run's counts`, () => {
      for (const p of PROBLEMS) {
        const r = run(id, p.id);
        const v = stepViews(id, r.trace);
        expect(v.length).toBe(r.trace.length);
        v.forEach((s, k) => {
          expect(s.k).toBe(k);
          expect(s.x).toBe(r.trace[k].x);
          if (k > 0) expect(s.nEval).toBeGreaterThan(v[k - 1].nEval);
          expect(s.roi[0]).toBeLessThanOrEqual(s.roi[1]);
        });
        // Runs that end on a step (not on a rejected trial) have counted every evaluation.
        const total = r.nFev + r.nGev + r.nHev;
        if (r.converged || /max_iter/.test(r.message)) expect(v[v.length - 1].nEval).toBe(total);
        // Newton keeps no bracket; every other method reports one at every step.
        expect(v.every((s) => (s.bracket === null) === (id === 'newton_1d'))).toBe(true);
      }
    });

  it('golden section: the probes cut the bracket ρ : 1 − 2ρ : ρ and the reported point is one of them', () => {
    const rho = (3 - Math.sqrt(5)) / 2;
    const r = run('golden_section', 'sin_1d');
    for (const s of stepViews('golden_section', r.trace).slice(0, 30)) {
      const [a, b] = s.bracket!;
      const [p, q] = s.probes;
      const [r1, r2, r3] = proportions(a, b, p.x, q.x);
      expect(r1).toBeCloseTo(rho, 9);
      expect(r2).toBeCloseTo(1 - 2 * rho, 9);
      expect(r3).toBeCloseTo(rho, 9);
      expect([p.x, q.x]).toContain(s.x);
      expect(s.probes.filter((x) => x.best)).toHaveLength(1);
    }
  });

  it('every evaluated point of an elimination step is one of its probes', () => {
    for (const id of [
      'golden_section',
      'fibonacci_search',
      'dichotomous_search',
      'ternary_search',
    ]) {
      const v = stepViews(id, run(id, 'quartic_1d').trace);
      for (const s of v)
        for (const [ex, ey] of s.evaluated) {
          const probe = s.probes.find((p) => p.x === ex);
          expect(probe).toBeDefined();
          expect(ey).toBe(probe!.f);
        }
    }
  });
});

describe('nextView', () => {
  it('the next bracket and trial are those of step k + 1', () => {
    for (const id of METHODS) {
      const r = run(id, 'drug_concentration');
      for (let k = 0; k + 1 < r.trace.length; k++) {
        const n = nextView(id, r.trace, k)!;
        expect(n).not.toBeNull();
        const br = r.trace[k + 1].info.bracket as number[] | undefined;
        if (br) expect(n.nextBracket).toEqual(br);
        if (id === 'brent_minimize' || id === 'parabolic_interpolation')
          expect(n.trials[0][0]).toBe(r.trace[k + 1].info.trial);
        if (id === 'newton_1d') expect(n.nextX).toBe(r.trace[k + 1].x);
      }
      expect(nextView(id, r.trace, r.trace.length - 1)).toBeNull();
    }
  });

  it('an elimination step discards exactly one end, at a probe of step k', () => {
    const r = run('golden_section', 'multimodal_1d');
    const v = stepViews('golden_section', r.trace);
    for (let k = 0; k + 1 < r.trace.length; k++) {
      const n = nextView('golden_section', r.trace, k)!;
      const [a, b] = v[k].bracket!;
      const [na, nb] = n.nextBracket!;
      const [x1, x2] = v[k].probes.map((p) => p.x);
      if (n.stepKind === 'right') expect([na, nb]).toEqual([a, x2]);
      else expect([na, nb]).toEqual([x1, b]);
    }
  });

  it("Newton's model is the Taylor parabola at x_k and its minimizer is x_{k+1}", () => {
    const r = run('newton_1d', 'sin_1d', { x0: 3.0 });
    const p = getProblem<ScalarProblem>('sin_1d');
    for (let k = 0; k + 1 < r.trace.length; k++) {
      const n = nextView('newton_1d', r.trace, k)!;
      const xk = r.trace[k].x as number;
      expect(parabolaAt(n.parabola!, xk)).toBeCloseTo(p.f(xk), 12);
      if (n.stepKind === 'newton') {
        const c = n.parabola!.coef;
        expect(n.parabola!.center - c[1] / (2 * c[2])).toBeCloseTo(n.nextX!, 12);
      }
    }
  });

  it("Brent's and parabolic interpolation's parabolas pass through their three points", () => {
    for (const id of ['brent_minimize', 'parabolic_interpolation']) {
      const r = run(id, 'rational_1d');
      const v = stepViews(id, r.trace);
      for (let k = 0; k + 1 < r.trace.length; k++) {
        const n = nextView(id, r.trace, k)!;
        if (!n.parabola) continue;
        const pts =
          id === 'brent_minimize'
            ? (r.trace[k].info.xwv as number[])
            : (r.trace[k].info.triple as number[]);
        const fs =
          id === 'brent_minimize'
            ? (r.trace[k].info.f_xwv as number[])
            : (r.trace[k].info.f_triple as number[]);
        pts.forEach((x, j) => expect(parabolaAt(n.parabola!, x)).toBeCloseTo(fs[j], 9));
        expect(v[k]).toBeDefined();
      }
    }
  });
});

describe('convergence measures', () => {
  it('measured contraction per evaluation matches the theory of each elimination method', () => {
    for (const [id, want] of [
      ['golden_section', 0.618],
      ['fibonacci_search', 0.618],
      ['ternary_search', 0.816],
      ['dichotomous_search', 0.707],
    ] as const) {
      const rate = ratePerEvaluation(stepViews(id, run(id, 'quadratic_1d').trace))!;
      expect(Math.abs(rate - want)).toBeLessThan(0.012);
      expect(Math.abs(GUIDE_RATES[id]!.rate - want)).toBeLessThan(0.002);
    }
  });

  it('error uses the nearest known minimizer and never drops below the floor', () => {
    expect(nearestMinimizer([5.14, 3.39, 7.0], 3.5)).toBe(3.39);
    expect(nearestMinimizer([], 1)).toBeNull();
    const r = run('brent_minimize', 'quadratic_1d');
    const vals = metricValues(stepViews('brent_minimize', r.trace), 'error', 2);
    expect(vals.every((x) => x !== null && x >= ERROR_FLOOR)).toBe(true);
    const widths = metricValues(
      stepViews('newton_1d', run('newton_1d', 'quadratic_1d').trace),
      'width',
      2,
    );
    expect(widths.every((x) => x === null)).toBe(true);
  });
});

describe('camera', () => {
  it('the whole-interval view holds the domain and the inputs', () => {
    expect(fullView([0, 24], [0, 12], 1)).toEqual([0, 24]);
    const v = fullView([0, 1], [-1, 0.5], 3);
    expect(v[0]).toBeLessThan(-1);
    expect(v[1]).toBeGreaterThan(3);
  });

  it('the follow view frames the bracket with PAD, stays inside the whole view and narrows', () => {
    const p = getProblem<ScalarProblem>('drug_concentration');
    const views = stepViews('golden_section', run('golden_section', p.id).trace);
    const full = fullView(p.domain, p.bracket, null);
    let last = Infinity;
    for (let k = 0; k < views.length; k++) {
      const [lo, hi] = followView(views, k, true, full);
      expect(lo).toBeGreaterThanOrEqual(full[0]);
      expect(hi).toBeLessThanOrEqual(full[1]);
      const [a, b] = views[k].bracket!;
      expect(lo).toBeLessThanOrEqual(a);
      expect(hi).toBeGreaterThanOrEqual(b);
      const w = hi - lo;
      if (k > 0) expect(w).toBeLessThanOrEqual(last * (1 + 1e-12));
      if (k > 2) expect(w / (b - a)).toBeCloseTo(PAD, 6);
      last = w;
    }
    // Between steps the width moves geometrically (eased), never outside the two frames.
    const w3 = followView(views, 3, true, full),
      w35 = followView(views, 3.5, true, full),
      w4 = followView(views, 4, true, full);
    expect(w35[1] - w35[0]).toBeLessThan(w3[1] - w3[0]);
    expect(w35[1] - w35[0]).toBeGreaterThan(w4[1] - w4[0]);
  });

  it("Newton's start (a single point) borrows its neighbor's width", () => {
    const views = stepViews('newton_1d', run('newton_1d', 'quadratic_1d', { x0: 0.5 }).trace);
    const [lo, hi] = followView(views, 0, false, [-1, 5]);
    expect(hi - lo).toBeGreaterThan(1);
  });
});

describe('step notes', () => {
  it('formats numbers in TeX with U+2212-free minus, powers of ten and dashes', () => {
    expect(texNum(1.2345678)).toBe('1.2346');
    expect(texNum(-2.5e-9, 3)).toBe('-2.5 \\times 10^{-9}');
    expect(texNum(1e-8)).toBe('10^{-8}');
    expect(texNum(null)).toBe('\\text{—}');
    expect(texNum(Infinity)).toBe('\\infty');
    expect(digitsFor(1e-8, 2)).toBeGreaterThanOrEqual(11);
    expect(digitsFor(1, 2)).toBe(5);
  });

  it('every method has a note at every step; elimination notes name the dropped end', () => {
    for (const id of METHODS) {
      const r = run(id, 'quartic_1d');
      const v = stepViews(id, r.trace);
      for (let k = 0; k < r.trace.length; k++) {
        const note = stepNote(id, r, v, k)!;
        expect(note).toBeTruthy();
        expect(note).not.toContain('NaN');
        expect(note).not.toContain('undefined');
        if (kindOf(id) === 'interval' && k + 1 < r.trace.length) {
          const cut = r.trace[k + 1].info.cut;
          expect(note).toContain(cut === 'right' ? `b_{${k}}]` : `[a_{${k}}`);
        }
      }
    }
  });
});

describe('lab wiring', () => {
  it('iteration-table columns have unique keys for every method', () => {
    for (const id of METHODS) {
      const keys = columnsFor(id).map((c) => c.key);
      expect(new Set(keys).size).toBe(keys.length);
    }
  });

  it('defaults and presets name registered methods and problems, and every preset sets a bracket', () => {
    expect(PROBLEMS.some((p) => p.id === DEFAULT_PROBLEM)).toBe(true);
    for (const s of DEFAULT_SELECTION) expect(METHODS).toContain(s.id);
    expect(PRESETS.length).toBeGreaterThanOrEqual(2);
    expect(PRESETS.length).toBeLessThanOrEqual(4);
    for (const p of PRESETS) {
      expect(PROBLEMS.some((q) => q.id === p.problem)).toBe(true);
      for (const m of p.methods ?? []) expect(METHODS).toContain(m.id);
      expect(p.extra?.br).toMatch(/^-?[\d.]+,-?[\d.]+$/);
      if (p.methods?.some((m) => usesStart(m.id))) expect(p.start).toBeDefined();
    }
  });

  it('the presets show what their titles claim', () => {
    // Rounding stops dichotomous search while the bracket is still wide.
    const d = run('dichotomous_search', 'drug_concentration', { xtol: 1e-10, delta_ratio: 0.001 });
    expect(d.converged).toBe(false);
    const br = d.extra.bracket as number[];
    expect(br[1] - br[0]).toBeGreaterThan(0.05);
    // A kink defeats Newton: the iteration budget runs out between −1 and 1.
    const n = run('newton_1d', 'abs_shifted', { x0: -0.5 });
    expect(n.converged).toBe(false);
    expect(new Set(n.trace.slice(2).map((s) => s.x))).toEqual(new Set([-1, 1]));
    // A fixed end slows the parabolas: x_r = 2 stays in the triple for 28 steps.
    const pi = run('parabolic_interpolation', 'quartic_1d');
    const bt = run('brent_minimize', 'quartic_1d');
    expect(pi.trace.slice(0, 29).every((s) => (s.info.triple as number[])[2] === 2)).toBe(true);
    expect(pi.nIter).toBe(31);
    expect(bt.nIter).toBe(10);
    expect(pi.nIter).toBeGreaterThan(bt.nIter);
  });
});
