/**
 * Regression tests for the interpolation lab review: every filled-in MethodCard rule is valid
 * TeX (KaTeX with throwOnError) on every dataset, layout and step; the O(n) Lagrange partial
 * sums equal the direct basis sums; the "Add node" placement; and the error-vs-n sweep marks
 * Chebyshev interpolation that samples f at its own nodes.
 */
import katex from 'katex';
import { describe, expect, it } from 'vitest';
import { defaults, listMethods } from '../../src/core/registry';
import { getProblem, listProblems } from '../../src/problems/registry';
import type { DataProblem } from '../../src/problems/data';
import '../../src/methods/interpolation/methods';
import '../../src/methods/interpolation/rational';
import '../../src/problems/data';
import {
  activeData,
  basisFn,
  lagrangePartial,
  methodKind,
  widestGapNode,
} from '../../src/labs/interpolation/geometry';
import { filledRule, product } from '../../src/labs/interpolation/cardRule';
import { errorVsN } from '../../src/labs/interpolation/sweep';

const nums = (v: unknown): number[] => (Array.isArray(v) ? (v as number[]) : []);

describe('filled-in MethodCard rules', () => {
  const methods = listMethods('interpolation');
  const problems = listProblems<DataProblem>('data');
  const datasets = problems.flatMap((p) => [
    activeData(p, 'data', p.x.length, null),
    ...(p.fTrue
      ? [5, 11, 15].flatMap((n) => [activeData(p, 'equi', n, null), activeData(p, 'cheb', n, null)])
      : []),
  ]);

  it('covers every method on every dataset', () => {
    // 10 polynomial and piecewise methods, AAA and Floater–Hormann.
    expect(methods.length).toBe(12);
    expect(datasets.length).toBeGreaterThan(problems.length);
  });

  it('is valid TeX at every step (no fused control words such as \\cdotsx)', () => {
    let checked = 0;
    const seen = new Set<string>();
    for (const data of datasets)
      for (const m of methods) {
        let r;
        try {
          r = m.fn(data, defaults(m.spec));
        } catch {
          continue;
        }
        const piecewise = methodKind(m.spec.id) === 'piecewise' || methodKind(m.spec.id) === 'blend';
        const nodes = piecewise && nums(r.extra.nodes).length ? nums(r.extra.nodes) : data.x;
        const values = piecewise && nums(r.extra.values).length ? nums(r.extra.values) : data.y;
        for (let k = 0; k < r.trace.length; k++) {
          const tex = filledRule(m.spec.id, r, k, nodes, values);
          expect(tex, `${data.id} ${m.spec.id} k=${k}`).not.toMatch(/\\cdots[a-zA-Z]/);
          expect(tex, `${data.id} ${m.spec.id} k=${k}`).not.toMatch(/\+ -|- -/);
          if (!seen.has(tex)) {
            seen.add(tex);
            expect(() => katex.renderToString(tex, { throwOnError: true, displayMode: true }), `${data.id} ${m.spec.id} k=${k}`).not.toThrow();
          }
          checked++;
        }
      }
    expect(checked).toBeGreaterThan(1000);
  }, 60_000);

  it('sets a node at 0 off from its neighbours', () => {
    expect(product([-1, -0.8, -0.6, 0])).toBe('(x + 1)(x + 0.8)\\,\\cdots\\,x');
    expect(product([-1, 0, 1])).toBe('(x + 1)\\,x\\,(x - 1)');
    expect(product([0, 1])).toBe('x\\,(x - 1)');
    expect(product([])).toBe('1');
  });

  it('signs the linear-spline secant once (noisy_quadratic: m_0 < 0)', () => {
    const p = getProblem<DataProblem>('noisy_quadratic');
    const m = listMethods('interpolation').find((q) => q.spec.id === 'linear_spline')!;
    const r = m.fn(p, defaults(m.spec));
    const tex = filledRule('linear_spline', r, 0, nums(r.extra.nodes), nums(r.extra.values));
    expect(tex).toContain('S_0(x) = 5.057 - 4.485');
  });
});

describe('Lagrange partial sums in O(n)', () => {
  const nodes = [-1, -0.55, -0.1, 0.3, 0.72, 1];
  const ys = [0.2, -1.1, 0.7, 2.4, -0.3, 0.9];
  it('equals Σ_{j≤k} y_j ℓ_j on a grid and at the nodes', () => {
    for (let k = 0; k < nodes.length; k++) {
      const g = lagrangePartial(nodes, ys.slice(0, k + 1));
      const direct = (t: number) => ys.slice(0, k + 1).reduce((s, y, j) => s + y * basisFn(nodes, j)(t), 0);
      for (let i = 0; i <= 50; i++) {
        const t = -1.1 + (2.2 * i) / 50;
        expect(g(t)).toBeCloseTo(direct(t), 12);
      }
      nodes.forEach((xm, m) => expect(g(xm)).toBe(m <= k ? ys[m] : 0));
    }
  });
});

describe('Add node', () => {
  it('bisects the widest gap, with y on f when f is given', () => {
    const p = widestGapNode([0, 1, 3], [0, 1, 9], [0, 3]);
    expect(p).toEqual({ x: 2, y: 5 });
    const q = widestGapNode([0, 1, 3], [0, 1, 9], [0, 3], (t) => t * t);
    expect(q).toEqual({ x: 2, y: 4 });
  });
  it('uses the free ends of the domain', () => {
    expect(widestGapNode([0.5, 0.6], [1, 2], [-1, 1])).toEqual({ x: -0.25, y: 1 });
  });
});

describe('error vs n', () => {
  it('marks Chebyshev interpolation on equispaced nodes as using its own nodes', () => {
    const runge = getProblem<DataProblem>('runge_equispaced');
    const sel = [
      { id: 'newton_divided_differences', slot: 0, params: {} },
      { id: 'chebyshev_interpolation', slot: 1, params: {} },
    ];
    const equi = errorVsN(runge, 'equi', sel, [5, 11]);
    expect(equi.map((s) => s.ownNodes)).toEqual([false, true]);
    const cheb = errorVsN(runge, 'cheb', sel, [5, 11]);
    expect(cheb.map((s) => s.ownNodes)).toEqual([false, false]);
  });
});
