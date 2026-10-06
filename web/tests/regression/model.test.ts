/**
 * The regression lab's mathematics (src/labs/regression/model.ts) and invariants of the methods
 * that the parity fixtures do not state: the hat-matrix leave-one-out errors equal brute-force
 * refits, tr H(λ) runs from p to 1, IRLS objectives decrease, the minimax line equioscillates,
 * an L1 line passes through ≥ 2 points, least-squares residuals are orthogonal to the design.
 */
import { describe, expect, it } from 'vitest';
import { getMethod, defaults } from '../../src/core/registry';
import { getProblem } from '../../src/problems/registry';
import type { Result, Step } from '../../src/core/types';
import { polyval, vandermonde } from '../../src/methods/regression/methods';
import '../../src/methods/regression/methods';
import '../../src/problems/data';
import {
  LAMBDA_GRID,
  betaAt,
  dataKey,
  decodePoints,
  degreeSweep,
  displayWeights,
  encodePoints,
  fitStats,
  floorPoints,
  medianPairs,
  relativeGap,
  residuals,
  ridgePath,
  type Pt,
} from '../../src/labs/regression/model';
import { texNum, texPoly } from '../../src/labs/regression/text';
import { sci2 } from '../../src/labs/regression/columns';
import katex from 'katex';

function run(id: string, data: unknown, params: Record<string, number | string> = {}): Result {
  const { spec, fn } = getMethod(id);
  return fn(data, { ...defaults(spec), ...params });
}
const pts = (id: string): Pt[] => {
  const p = getProblem(id) as unknown as { x: number[]; y: number[] };
  return p.x.map((x, i) => ({ x, y: p.y[i] }));
};
const pair = (p: readonly Pt[]) => [p.map((q) => q.x), p.map((q) => q.y)] as const;
const beta = (r: Result) => r.x as number[];

describe('leave-one-out errors (hat-matrix identity) equal brute-force refits', () => {
  const data = pts('noisy_quadratic');
  it('polynomial least squares, d = 0 … 6', () => {
    const sweep = degreeSweep(data, undefined, [-2, 4]);
    for (let d = 0; d <= 6; d++) {
      let s = 0;
      for (let i = 0; i < data.length; i++) {
        const rest = data.filter((_, j) => j !== i);
        const b = beta(run('polynomial_regression', pair(rest), { degree: d }));
        s += (data[i].y - polyval(b, data[i].x)) ** 2;
      }
      expect(sweep[d].loo).toBeCloseTo(Math.sqrt(s / data.length), 9);
    }
  });
  it('ridge with an unpenalized intercept, d = 4', () => {
    const lams = [1e-3, 1, 100];
    const path = ridgePath(data, 4, undefined, [-2, 4], lams);
    lams.forEach((lam, n) => {
      let s = 0;
      for (let i = 0; i < data.length; i++) {
        const rest = data.filter((_, j) => j !== i);
        const b = beta(run('ridge_regression', pair(rest), { lam, degree: 4 }));
        s += (data[i].y - polyval(b, data[i].x)) ** 2;
      }
      expect(path[n].loo).toBeCloseTo(Math.sqrt(s / data.length), 8);
      expect(path[n].train).toBeCloseTo(
        run('ridge_regression', pair(data), { lam, degree: 4 }).extra.rmse as number,
        12,
      );
    });
  });
  it('tr H(λ) falls from p = d + 1 to 1 (the mean) along the λ grid', () => {
    const path = ridgePath(data, 3, undefined, [-2, 4]);
    expect(path).toHaveLength(LAMBDA_GRID.length);
    expect(path[0].df!).toBeCloseTo(4, 4);
    expect(path[path.length - 1].df!).toBeLessThan(1.6);
    for (let i = 1; i < path.length; i++)
      expect(path[i].df!).toBeLessThanOrEqual(path[i - 1].df! + 1e-9);
  });
  it('the training error never increases with the degree', () => {
    const sweep = degreeSweep(data, undefined, [-2, 4]);
    expect(sweep).toHaveLength(16);
    for (let d = 1; d < sweep.length; d++)
      expect(sweep[d].train!).toBeLessThanOrEqual(sweep[d - 1].train! + 1e-12);
  });
  it('error against the truth is measured on the domain', () => {
    const p = getProblem('noisy_quadratic') as unknown as { fTrue: (x: number) => number };
    const sweep = degreeSweep(data, p.fTrue, [-2, 4]);
    const best = sweep.reduce((a, b) => ((b.truth ?? Infinity) < (a.truth ?? Infinity) ? b : a));
    expect(best.at).toBe(2);
  });
});

describe('playhead geometry', () => {
  const trace: Step[] = [
    { k: 0, x: [0, 1], fun: 2, gradNorm: null, stepSize: null, info: {} },
    { k: 1, x: [2, 3], fun: 1, gradNorm: null, stepSize: null, info: {} },
  ];
  it('β at integer t is the iterate; between steps it moves straight', () => {
    expect(betaAt(trace, 0, false)).toEqual([0, 1]);
    expect(betaAt(trace, 1, false)).toEqual([2, 3]);
    expect(betaAt(trace, 0.5, false)).toEqual([1, 2]);
    expect(betaAt(trace, 0.5, true)).toEqual([1, 2]); // the ease is symmetric
    expect(betaAt(trace, Infinity, true)).toEqual([2, 3]);
    expect(betaAt([], 0, true)).toBeNull();
  });
  it('display weights: Huber passes through, LAD is relative to the median weight', () => {
    const step = (w: number[]): Step => ({
      k: 0,
      x: [0, 0],
      fun: 0,
      gradNorm: null,
      stepSize: null,
      info: { weights: w },
    });
    expect(displayWeights('huber', step([1, 0.5, 0.1]))).toEqual([1, 0.5, 0.1]);
    expect(displayWeights('lad', step([1, 2, 4, 1e6]))).toEqual([1 / 3, 2 / 3, 1, 1]);
    expect(displayWeights('ols', step([1]))).toBeNull();
  });
  it('relative gap: Huber decreases to a gap at the minimum', () => {
    const res = run('huber_regression', getProblem('outliers_linear'));
    const gap = relativeGap(res);
    expect(gap[gap.length - 1]).toBeNull();
    const vals = gap.filter((g): g is number => g !== null);
    for (let i = 1; i < vals.length; i++) expect(vals[i]).toBeLessThan(vals[i - 1]);
  });
  it('Theil–Sen: the median pair has the median slope', () => {
    const data = pts('noisy_linear'); // 190 pairs: two middle pairs
    const res = run('theil_sen', getProblem('noisy_linear'));
    const slopes = res.trace[0].info.slopes as number[];
    const pairs = medianPairs(data, slopes);
    expect(pairs.length).toBeGreaterThanOrEqual(1);
    const s = pairs.map(([i, j]) => (data[j].y - data[i].y) / (data[j].x - data[i].x));
    expect(s).toContain(slopes[94]);
    const odd = data.slice(0, 15); // 105 pairs: one middle pair
    const r2 = run('theil_sen', pair(odd));
    const [[i, j]] = medianPairs(odd, r2.trace[0].info.slopes as number[]);
    expect((odd[j].y - odd[i].y) / (odd[j].x - odd[i].x)).toBe(beta(r2)[1]);
  });
  it('fit statistics match the method’s extra', () => {
    const res = run('linear_regression', getProblem('anscombe_1'));
    const st = fitStats(pts('anscombe_1'), beta(res));
    expect(st.r2).toBeCloseTo(res.extra.r_squared as number, 12);
    expect(st.adjR2).toBeCloseTo(res.extra.adj_r_squared as number, 12);
    expect(st.rmse).toBeCloseTo(res.extra.rmse as number, 12);
  });
  it('edited data round-trip through the URL encoding', () => {
    const data = pts('anscombe_1');
    expect(decodePoints(encodePoints(data))).toEqual(data);
    expect(decodePoints([1, 2, 3])).toBeNull();
    expect(dataKey(data)).not.toBe(dataKey(data.slice(1)));
  });
  it('TeX numbers', () => {
    expect(texNum(0.000123, 3)).toBe('1.23\\times10^{-4}');
    expect(texNum(123456, 3)).toBe('1.23\\times10^{5}');
    expect(texNum(2.5)).toBe('2.5');
    expect(texNum(null)).toBe('\\text{—}');
    expect(texPoly([1, -2, 0.5])).toBe('\\hat y(x) = 1 - 2\\,x + 0.5\\,x^{2}');
  });
});

describe('method invariants', () => {
  it('least squares: residuals are orthogonal to the columns of the design', () => {
    for (const d of [1, 3, 5]) {
      const data = pts('noisy_quadratic');
      const res = run('polynomial_regression', getProblem('noisy_quadratic'), { degree: d });
      const r = residuals(data, beta(res));
      const X = vandermonde(
        data.map((p) => p.x),
        d,
      );
      for (let j = 0; j <= d; j++) {
        const col = X.map((row) => row[j]);
        const dotp = col.reduce((s, v, i) => s + v * r[i], 0);
        const scale = Math.sqrt(
          col.reduce((s, v) => s + v * v, 0) * r.reduce((s, v) => s + v * v, 0),
        );
        expect(Math.abs(dotp)).toBeLessThan(1e-10 * scale);
      }
    }
  });
  it('Huber and LAD: the objective IRLS majorizes decreases at every step', () => {
    for (const id of ['outliers_linear', 'noisy_linear', 'exponential_growth']) {
      const h = run('huber_regression', getProblem(id));
      for (let k = 1; k < h.trace.length; k++)
        expect(h.trace[k].fun!).toBeLessThanOrEqual(h.trace[k - 1].fun! * (1 + 1e-12));
      const l = run('lad_regression', getProblem(id));
      for (let k = 1; k < l.trace.length; k++) {
        const a = l.trace[k].info.smoothed_objective as number;
        const b = l.trace[k - 1].info.smoothed_objective as number;
        expect(a).toBeLessThanOrEqual(b * (1 + 1e-12));
      }
    }
  });
  it('LAD: the converged line passes through at least two data points', () => {
    for (const id of ['outliers_linear', 'noisy_linear', 'anscombe_1']) {
      const res = run('lad_regression', getProblem(id));
      expect(floorPoints(pts(id), beta(res), 1e-6).length).toBeGreaterThanOrEqual(2);
    }
  });
  it('minimax: |h| rises strictly and the optimum equioscillates on its reference', () => {
    for (const id of ['noisy_linear', 'outliers_linear', 'anscombe_1', 'runge_equispaced']) {
      const res = run('chebyshev_minimax_line', getProblem(id));
      expect(res.converged).toBe(true);
      const hs = res.trace.map((s) => Math.abs(s.info.level as number));
      for (let k = 1; k < hs.length; k++) expect(hs[k]).toBeGreaterThan(hs[k - 1]);
      const last = res.trace[res.trace.length - 1];
      const data = pts(id);
      const r = residuals(data, beta(res));
      const ref = last.info.reference as number[];
      const h = last.info.level as number;
      const byX = [...ref].sort((a, b) => data[a].x - data[b].x);
      byX.forEach((i, j) => expect(r[i]).toBeCloseTo((j % 2 === 0 ? 1 : -1) * h, 9));
      expect(Math.max(...r.map(Math.abs))).toBeCloseTo(Math.abs(h), 9);
    }
  });
  it('ridge: λ → ∞ flattens the fit to ȳ; λ = 0 is least squares', () => {
    const data = pts('noisy_quadratic');
    const ybar = data.reduce((s, p) => s + p.y, 0) / data.length;
    const flat = beta(
      run('ridge_regression', getProblem('noisy_quadratic'), { lam: 1e12, degree: 3 }),
    );
    expect(flat[0]).toBeCloseTo(ybar, 4);
    const ols = beta(run('polynomial_regression', getProblem('noisy_quadratic'), { degree: 3 }));
    const r0 = beta(run('ridge_regression', getProblem('noisy_quadratic'), { lam: 0, degree: 3 }));
    r0.forEach((b, j) => expect(b).toBeCloseTo(ols[j], 9));
  });
  it('invalid input raises, numerical breakdown does not', () => {
    expect(() =>
      run('huber_regression', [
        [1, 1, 1],
        [0, 1, 2],
      ]),
    ).toThrow('not identifiable');
    expect(() =>
      run(
        'linear_regression',
        [
          [0, 1],
          [0, 1],
        ],
        { solver: 'lu' },
      ),
    ).toThrow('unknown solver');
    const res = run(
      'polynomial_regression',
      [
        [0, 1, 2],
        [1, 0, 1],
      ],
      { degree: 6 },
    );
    expect(res.converged).toBe(false);
    // Minimum-norm interpolant: passes through the data.
    expect(res.x as number[]).toHaveLength(7);
    [0, 1, 2].forEach((x, i) => expect(polyval(res.x as number[], x)).toBeCloseTo([1, 0, 1][i], 8));
  });
});

describe('review fixes', () => {
  it('ridge tr H_λ is evaluated at the run’s own λ, not the nearest grid point', () => {
    const data = pts('noisy_quadratic');
    const at = ridgePath(data, 12, undefined, [-1, 1], [0.4])[0];
    const grid = ridgePath(data, 12, undefined, [-1, 1], [10 ** -0.5])[0];
    // NumPy: tr H(0.4) = 10.747, tr H(10^-0.5) = 10.872.
    expect(at.df).toBeCloseTo(10.747, 2);
    expect(grid.df).toBeCloseTo(10.872, 2);
  });

  it('the Δθ column keeps a fixed 2-digit mantissa', () => {
    expect(sci2(3.54e-12)).toBe('3.5×10⁻¹²');
    expect(sci2(2e-7)).toBe('2.0×10⁻⁷');
    expect(sci2(-1.234)).toBe('−1.2');
    expect(sci2(null)).toBe('—');
  });

  it('every MethodCard rule typesets without a KaTeX error', () => {
    const ids = [
      'linear_regression',
      'polynomial_regression',
      'ridge_regression',
      'huber_regression',
      'lad_regression',
      'theil_sen',
      'chebyshev_minimax_line',
    ];
    for (const id of ids) {
      const rule = getMethod(id).doc?.rule;
      expect(rule, id).toBeTruthy();
      expect(() =>
        katex.renderToString(rule!, { displayMode: true, throwOnError: true }),
      ).not.toThrow();
    }
  });
});
