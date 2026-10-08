/**
 * The interpolation lab's mathematics: node layouts, the active dataset, ω, the Lagrange basis
 * and Lebesgue function, the exact step evaluators (they must reproduce each step's info.curve),
 * the error-vs-n sweep, and the filled-in MethodCard rules.
 */
import { describe, expect, it } from 'vitest';
import { getMethod, defaults } from '../../src/core/registry';
import { getProblem } from '../../src/problems/registry';
import type { DataProblem } from '../../src/problems/data';
import { linspace } from '../../src/problems/data';
import '../../src/methods/interpolation/methods';
import '../../src/problems/data';
import {
  activeData,
  basisFn,
  chebyshevNodes,
  chebyshevOmegaMax,
  dataKey,
  detectLayout,
  lebesgueFn,
  maxAbs,
  nodalPoly,
  offViewPeaks,
  stepFunction,
  stepTerm,
  viewRange,
} from '../../src/labs/interpolation/geometry';
import { errorVsN } from '../../src/labs/interpolation/sweep';
import { filledRule, texNum } from '../../src/labs/interpolation/cardRule';

const runge = getProblem<DataProblem>('runge_equispaced');
const run = (id: string, data: unknown, params = {}) => {
  const m = getMethod(id);
  return m.fn(data, { ...defaults(m.spec), ...params });
};

describe('node layouts and the active dataset', () => {
  it('detects the library layouts', () => {
    expect(detectLayout(runge.x, -1, 1)).toBe('equi');
    expect(detectLayout(getProblem<DataProblem>('runge_chebyshev').x, -1, 1)).toBe('cheb');
    expect(detectLayout([-1, 0.1, 1], -1, 1)).toBe('data');
  });

  it('resamples f at n nodes and keys the data by its values', () => {
    const d = activeData(runge, 'cheb', 15, null);
    expect(d.x).toHaveLength(15);
    expect(d.x).toEqual(chebyshevNodes(15, -1, 1));
    d.x.forEach((xi, i) => expect(d.y[i]).toBe(runge.fTrue!(xi)));
    expect(d.name).toBe('Runge function, 15 Chebyshev nodes');
    expect(activeData(runge, 'data', 11, null).id).toBe('runge_equispaced');
    const e1 = activeData(runge, 'data', 11, [0, 1, 0.5, 2, 1, 0]);
    const e2 = activeData(runge, 'data', 11, [0, 1, 0.5, 2.0000001, 1, 0]);
    expect(e1.edited).toBe(true);
    expect(e1.x).toEqual([0, 0.5, 1]);
    expect(e1.id).not.toBe(e2.id);
    expect(dataKey([1], [2])).toBe(dataKey([1], [2]));
  });
});

describe('node polynomial, basis and Lebesgue function', () => {
  it('Chebyshev nodes attain the minimal max|ω| = 2((b − a)/4)ⁿ', () => {
    for (const n of [5, 11, 15]) {
      const m = maxAbs(nodalPoly(chebyshevNodes(n, -1, 1)), -1, 1, 20001).value;
      expect(m / chebyshevOmegaMax(n, -1, 1)).toBeCloseTo(1, 4);
      const equi = maxAbs(nodalPoly(linspace(-1, 1, n)), -1, 1, 20001).value;
      expect(equi).toBeGreaterThan(m);
    }
  });

  it('the basis is cardinal and sums to one; Λₙ grows fast on equispaced nodes', () => {
    const x = linspace(-1, 1, 9);
    for (let j = 0; j < 9; j++)
      x.forEach((xm, m) => expect(basisFn(x, j)(xm)).toBeCloseTo(m === j ? 1 : 0, 12));
    const sum = (t: number) => x.reduce((s, _, j) => s + basisFn(x, j)(t), 0);
    expect(sum(0.37)).toBeCloseTo(1, 12);
    const L = (nodes: number[]) => maxAbs(lebesgueFn(nodes), -1, 1, 4001).value;
    expect(L(linspace(-1, 1, 21))).toBeGreaterThan(1e4);
    expect(L(chebyshevNodes(21, -1, 1))).toBeLessThan(3.1);
  });
});

describe('exact step evaluators reproduce every info.curve', () => {
  const datasets = ['runge_equispaced', 'sine_samples', 'anscombe_1', 'step_data'];
  const ids = [
    'lagrange',
    'barycentric',
    'newton_divided_differences',
    'neville',
    'chebyshev_interpolation',
    'linear_spline',
    'cubic_spline_natural',
    'cubic_spline_not_a_knot',
    'pchip',
  ];
  for (const pid of datasets)
    for (const id of ids)
      it(`${id} on ${pid}`, () => {
        const d = activeData(getProblem<DataProblem>(pid), 'data', 0, null);
        const res = run(id, d);
        const grid = (res.extra.eval as { x: number[] }).x;
        res.trace.forEach((s, k) => {
          const g = stepFunction(id, res, k, d);
          const curve = s.info.curve as number[] | undefined;
          if (!curve) return expect(g).toBeNull();
          const scale = Math.max(1, ...curve.map(Math.abs));
          grid.forEach((t, i) => expect(Math.abs(g!(t) - curve[i])).toBeLessThan(1e-9 * scale));
        });
      });

  it('the term of a step is the difference of consecutive approximants', () => {
    const d = activeData(runge, 'data', 0, null);
    for (const id of [
      'lagrange',
      'newton_divided_differences',
      'barycentric',
      'chebyshev_interpolation',
    ]) {
      const res = run(id, d);
      for (const k of [1, 4, 10]) {
        const term = stepTerm(id, res, k, d)!;
        const cur = stepFunction(id, res, k, d)!,
          prev = stepFunction(id, res, k - 1, d)!;
        for (const t of [-0.93, -0.2, 0.41, 0.88]) expect(term(t)).toBeCloseTo(cur(t) - prev(t), 9);
      }
    }
  });
});

describe('view helpers', () => {
  it('keeps the data in view and caps runaway curves', () => {
    const [lo, hi] = viewRange([0, 1], [[0, 50, -40]]);
    expect(lo).toBeLessThan(0);
    expect(hi).toBeGreaterThan(1);
    expect(hi).toBeLessThan(3);
    expect(lo).toBeGreaterThan(-2);
  });

  it('finds where a sampled curve leaves the view', () => {
    const ys = Array.from({ length: 101 }, (_, i) => Math.sin((i / 100) * 2 * Math.PI) * 3);
    const peaks = offViewPeaks(ys, 0, 1, -1, 1);
    expect(peaks).toHaveLength(2);
    expect(peaks[0].x).toBeCloseTo(0.25, 2);
    expect(peaks[0].y).toBeCloseTo(3, 6);
    expect(peaks[1].y).toBeCloseTo(-3, 6);
  });
});

describe("Runge's phenomenon in the error-vs-n study", () => {
  it('equispaced polynomial error grows; Chebyshev and the spline error decay', () => {
    const sel = [
      { id: 'barycentric', slot: 0, params: {} },
      { id: 'cubic_spline_not_a_knot', slot: 1, params: {} },
    ];
    const ns = [11, 21, 31];
    const [eq, spl] = errorVsN(runge, 'equi', sel, ns);
    const [ch] = errorVsN(runge, 'cheb', sel, ns);
    expect(eq.values[0]!).toBeCloseTo(1.9157, 3);
    expect(eq.values[2]!).toBeGreaterThan(eq.values[1]!);
    expect(eq.values[1]!).toBeGreaterThan(eq.values[0]!);
    expect(ch.values[2]!).toBeLessThan(ch.values[0]!);
    expect(spl.values[2]!).toBeLessThan(spl.values[0]!);
  });
});

describe('MethodCard rules with the step filled in', () => {
  it('typesets numbers for TeX', () => {
    expect(texNum(0.5)).toBe('0.5');
    expect(texNum(-220.94)).toBe('-220.9');
    expect(texNum(1.2e-7)).toBe('1.2\\times10^{-7}');
    expect(texNum(null)).toBe('\\text{—}');
  });

  it('fills the Newton coefficient and the node product', () => {
    const d = activeData(runge, 'data', 0, null);
    const res = run('newton_divided_differences', d);
    const tex = filledRule('newton_divided_differences', res, 10, d.x, d.y);
    expect(tex).toContain('p_{10}(x) = p_{9}(x)');
    expect(tex).toContain('(-220.9)');
    expect(tex).toContain('(x + 1)');
    const lag = filledRule(
      'lagrange',
      run('lagrange', getProblem('sine_samples')),
      6,
      getProblem<DataProblem>('sine_samples').x,
      getProblem<DataProblem>('sine_samples').y,
    );
    expect(lag).toContain('p_{5}(x) - 1');
  });

  it('every method and stage has a rule', () => {
    const d = activeData(runge, 'data', 0, null);
    for (const id of [
      'barycentric',
      'neville',
      'chebyshev_interpolation',
      'linear_spline',
      'cubic_spline_clamped',
      'pchip',
    ]) {
      const res = run(id, d);
      res.trace.forEach((_, k) =>
        expect(filledRule(id, res, k, d.x, d.y).length).toBeGreaterThan(20),
      );
    }
  });
});
