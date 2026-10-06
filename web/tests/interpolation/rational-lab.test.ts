/**
 * The interpolation lab's views of the rational methods: the exact step evaluators reproduce
 * every step's info.curve (AAA from the support points and weights of that step, Floater–Hormann
 * at its last step), AAA's support set and poles per step, Floater–Hormann's window of local
 * polynomials, the filled-in MethodCard rules, and the presets' numbers.
 */
import katex from 'katex';
import { describe, expect, it } from 'vitest';
import { defaults, getMethod } from '../../src/core/registry';
import { getProblem } from '../../src/problems/registry';
import type { DataProblem } from '../../src/problems/data';
import '../../src/methods/interpolation/methods';
import '../../src/methods/interpolation/rational';
import '../../src/problems/data';
import {
  aaaIntervalPoles,
  aaaPoles,
  aaaSupport,
  activeData,
  blendWindow,
  localPoly,
  methodKind,
  stepFunction,
} from '../../src/labs/interpolation/geometry';
import { filledRule } from '../../src/labs/interpolation/cardRule';
import { PRESETS } from '../../src/labs/interpolation/config';
import { errorVsN } from '../../src/labs/interpolation/sweep';

const nums = (v: unknown): number[] => (Array.isArray(v) ? (v as number[]) : []);
const run = (id: string, data: unknown, params = {}) => {
  const m = getMethod(id);
  return m.fn(data, { ...defaults(m.spec), ...params });
};
const runge = getProblem<DataProblem>('runge_equispaced');

describe('step evaluators of the rational methods', () => {
  it('AAA: every step evaluates to its info.curve; the support set grows by one', () => {
    for (const id of ['runge_equispaced', 'sine_samples', 'noisy_linear', 'step_data']) {
      const data = activeData(getProblem<DataProblem>(id), 'data', 0, null);
      const r = run('aaa', data);
      const grid = nums((r.extra.eval as { x: number[] }).x);
      r.trace.forEach((s, k) => {
        const g = stepFunction('aaa', r, k, data)!;
        const curve = nums(s.info.curve);
        grid.forEach((t, i) => {
          if (Number.isFinite(curve[i]) && Math.abs(curve[i]) < 1e6) expect(g(t)).toBeCloseTo(curve[i], 9);
        });
        expect(aaaSupport(r, k)).toHaveLength(k);
      });
      expect(methodKind('aaa')).toBe('aaa');
    }
  });

  it('Floater–Hormann: no curve before the last step, then the final interpolant', () => {
    const data = activeData(runge, 'equi', 15, null);
    const r = run('floater_hormann', data, { d: 3 });
    const last = r.trace.length - 1;
    expect(stepFunction('floater_hormann', r, 0, data)).toBeNull();
    const g = stepFunction('floater_hormann', r, last, data)!;
    data.x.forEach((xi, i) => expect(g(xi)).toBeCloseTo(data.y[i], 12));
    expect(methodKind('floater_hormann')).toBe('blend');
  });

  it("Floater–Hormann's window: the local polynomials through node k interpolate their d + 1 nodes", () => {
    const data = activeData(runge, 'equi', 11, null);
    const r = run('floater_hormann', data, { d: 3 });
    const nodes = nums(r.extra.nodes),
      vals = nums(r.extra.values);
    for (let k = 0; k < nodes.length; k++) {
      const win = blendWindow(r, k)!;
      expect(win.d).toBe(3);
      for (let i = win.lo; i <= win.hi; i++) {
        expect(i).toBeLessThanOrEqual(k);
        expect(i + 3).toBeGreaterThanOrEqual(k);
        const p = localPoly(nodes, vals, i, 3);
        for (let j = i; j <= i + 3; j++) expect(p(nodes[j])).toBeCloseTo(vals[j], 12);
      }
    }
  });

  it("AAA's poles: ±0.2i for Runge, certified real poles for noisy data", () => {
    const r = run('aaa', activeData(runge, 'equi', 15, null));
    const poles = aaaPoles(r, r.trace.length - 1, 1);
    expect(poles).toHaveLength(2);
    for (const p of poles) {
      expect(Math.abs(p.re)).toBeLessThan(1e-12);
      expect(Math.abs(Math.abs(p.im) - 0.2)).toBeLessThan(1e-12);
      expect(p.doublet).toBe(false);
    }
    expect(aaaIntervalPoles(r, r.trace.length - 1)).toEqual([]);
    const noisy = getProblem<DataProblem>('noisy_linear');
    const rn = run('aaa', noisy);
    const real = aaaIntervalPoles(rn, rn.trace.length - 1);
    expect(real).toHaveLength(7);
    for (const p of real) expect(p >= 0 && p <= 10).toBe(true);
  });
});

describe('MethodCard rules and presets', () => {
  it('the filled-in rules are valid TeX at every step', () => {
    for (const id of ['runge_equispaced', 'noisy_linear', 'step_data', 'sine_samples']) {
      const data = activeData(getProblem<DataProblem>(id), 'data', 0, null);
      for (const m of ['aaa', 'floater_hormann']) {
        const r = run(m, data);
        const nodes = m === 'floater_hormann' ? nums(r.extra.nodes) : data.x;
        const values = m === 'floater_hormann' ? nums(r.extra.values) : data.y;
        for (let k = 0; k < r.trace.length; k++) {
          const tex = filledRule(m, r, k, nodes, values);
          expect(tex).toContain('aligned');
          expect(() => katex.renderToString(tex, { throwOnError: true, displayMode: true }), `${id} ${m} k=${k}`).not.toThrow();
        }
      }
    }
  });

  it("the presets' numbers are what the methods compute", () => {
    const at = (n: number) => activeData(runge, 'equi', n, null);
    const ids = PRESETS.map((p) => p.id);
    expect(ids).toEqual(expect.arrayContaining(['aaa', 'floater-hormann', 'aaa-noise']));
    // AAA: three support points, error 6.7e-16; Newton's degree-14 polynomial: 7.2.
    const a = run('aaa', at(15));
    expect(a.nIter).toBe(3);
    expect(a.fun!).toBeLessThan(1e-15);
    expect(run('newton_divided_differences', at(15)).fun!).toBeCloseTo(7.19, 2);
    // Floater–Hormann d = 3 on 21 nodes: 0.0028; the polynomial: 58; the spline: 0.0032.
    expect(run('floater_hormann', at(21), { d: 3 }).fun!).toBeCloseTo(0.00281, 5);
    expect(run('barycentric', at(21)).fun!).toBeCloseTo(58.5, 1);
    expect(run('cubic_spline_not_a_knot', at(21)).fun!).toBeCloseTo(0.00318, 5);
    // Noisy line: AAA uses 11 of the 20 samples and has 7 real poles; FH has none.
    const noisy = getProblem<DataProblem>('noisy_linear');
    const rn = run('aaa', noisy);
    expect(rn.nIter).toBe(11);
    expect(rn.extra.n_interval_poles).toBe(7);
  });

  it('the error-vs-n sweep runs both rational methods for n = 3 … 40', () => {
    const series = errorVsN(
      runge,
      'equi',
      [
        { id: 'aaa', slot: 0, params: {} },
        { id: 'floater_hormann', slot: 1, params: { d: 3 } },
      ],
      [3, 10, 25, 40],
    );
    expect(series).toHaveLength(2);
    // Three samples cannot reveal Runge's poles; from n = 10 AAA recovers f to rounding.
    expect(series[0].values[0]!).toBeGreaterThan(0.1);
    for (const v of series[0].values.slice(1)) expect(v!).toBeLessThan(1e-12);
    // d = 3 needs 4 nodes (n = 3 throws: no value); the error falls as n grows.
    expect(series[1].values[0]).toBeNull();
    expect(series[1].values[3]!).toBeLessThan(series[1].values[1]!);
  });
});
