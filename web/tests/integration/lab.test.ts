/**
 * Quadrature lab helpers: the drawn geometry has exactly the area of the estimate, the node
 * count of the error chart matches the evaluation count, the reference integrals on
 * sub-intervals are right, and the filled-in rules carry this step's numbers.
 */
import { describe, expect, it } from 'vitest';
import { getMethod } from '../../src/core/registry';
import { getProblem } from '../../src/problems/registry';
import type { Problem, Result } from '../../src/core/types';
import '../../src/labs/integration/setup';
import {
  barycentric,
  baryWeights,
  geometryAt,
  geometryExtent,
  nodesUsed,
  observedOrder,
  nestedInterpolant,
  piecesArea,
} from '../../src/labs/integration/geometry';
import { columnsFor } from '../../src/labs/integration/stepColumns';
import { gaussReference, referenceIntegral } from '../../src/labs/integration/reference';
import { filledRule, estimateSymbol, estimateTex } from '../../src/labs/integration/cardRule';
import { agreeingDigits, texNum } from '../../src/labs/integration/quadFormat';
import type { CalculusProblem } from '../../src/problems/calculus';

/**
 * Boole's rule (exact to degree 5) on the equally spaced samples of each piece when their count
 * allows, else Simpson's rule (exact to degree 3): exact for the drawn parabolas and quartics.
 */
function simpsonArea(pieces: readonly { top: [number, number][] }[]): number {
  return pieces.reduce((acc, piece) => {
    const t = piece.top;
    const n = t.length - 1;
    if (n % 4 === 0 && n > 2) {
      const h = (t[n][0] - t[0][0]) / n;
      let a = 0;
      for (let i = 0; i < n; i += 4)
        a += 7 * t[i][1] + 32 * t[i + 1][1] + 12 * t[i + 2][1] + 32 * t[i + 3][1] + 7 * t[i + 4][1];
      return acc + ((2 * h) / 45) * a;
    }
    if (n % 2) return acc + piecesArea([piece]);
    const h = (t[n][0] - t[0][0]) / n;
    let a = t[0][1] + t[n][1];
    for (let i = 1; i < n; i++) a += (i % 2 ? 4 : 2) * t[i][1];
    return acc + (h / 3) * a;
  }, 0);
}

function run(id: string, problemId: string, params: Record<string, unknown> = {}): Result {
  const { spec, fn } = getMethod(id);
  const defaults = Object.fromEntries(spec.params.map((p) => [p.name, p.default]));
  return fn(getProblem(problemId), { ...defaults, ...(params as Record<string, never>) });
}

describe('the drawn area is the estimate', () => {
  const cases: [string, string, Record<string, unknown>][] = [
    ['left_riemann', 'exp_0_1', { levels: 3 }],
    ['right_riemann', 'sin_0_pi', { levels: 3 }],
    ['midpoint_rule', 'gaussian', { levels: 3 }],
    ['trapezoid', 'runge', { levels: 3 }],
    ['gauss_legendre', 'runge', { n: 9 }],
  ];
  for (const [id, pid, params] of cases) {
    it(`${id} on ${pid}`, () => {
      const r = run(id, pid, params);
      const p = getProblem<CalculusProblem>(pid);
      r.trace.forEach((s, k) => {
        const g = geometryAt(id, r.trace, k, p.domain, p.f, 400)!;
        expect(piecesArea(g.pieces)).toBeCloseTo(s.info.estimate as number, 12);
      });
    });
  }
  it('Simpson arches integrate to S_N (sampled parabolas, so to the sampling accuracy)', () => {
    const r = run('simpson', 'exp_0_1', { levels: 2 });
    const p = getProblem<CalculusProblem>('exp_0_1');
    const g = geometryAt('simpson', r.trace, 2, p.domain, p.f, 20000)!;
    expect(piecesArea(g.pieces)).toBeCloseTo(r.trace[2].info.estimate as number, 4);
  });
  it('Romberg draws the Newton–Cotes rule of column min(k, 2): R(k, k) itself for k ≤ 2', () => {
    for (const pid of ['exp_0_1', 'runge', 'gaussian']) {
      const r = run('romberg', pid, { max_levels: 6 });
      const p = getProblem<CalculusProblem>(pid);
      r.trace.forEach((s, k) => {
        const g = geometryAt('romberg', r.trace, k, p.domain, p.f, 1e5)!;
        const j = Math.min(k, 2);
        expect(g.column).toBe(j);
        expect(g.pieces).toHaveLength(2 ** k / 2 ** j);
        const want = (s.info.row as number[])[j];
        expect(simpsonArea(g.pieces)).toBeCloseTo(want, 9);
        if (k <= 2) expect(want).toBe(s.info.estimate as number);
      });
    }
  });
  it('the y-extent includes a rule that overshoots f (Boole quartic on Runge, k = 2)', () => {
    const r = run('romberg', 'runge');
    const p = getProblem<CalculusProblem>('runge');
    const [lo, hi] = geometryExtent('romberg', r.trace, p.domain, p.f);
    // p(x) = 1 + bx² + cx⁴ through (0, 1), (±½, f(½)), (±1, f(1)): minimum −0.38 near x = ±0.8.
    expect(lo).toBeLessThan(-0.37);
    expect(lo).toBeGreaterThan(-0.4);
    expect(hi).toBeCloseTo(1, 2);
  });
  it('Gauss: the error is measured against the interpolant, whose integral is G_m', () => {
    const r = run('gauss_legendre', 'runge', { n: 9 });
    const p = getProblem<CalculusProblem>('runge');
    for (const k of [0, 3, 8]) {
      const g = geometryAt('gauss_legendre', r.trace, k, p.domain, p.f, 400)!;
      expect(g.errorTop).toBeDefined();
      expect(simpsonArea([{ top: g.errorTop! }])).toBeCloseTo(
        r.trace[k].info.estimate as number,
        7,
      );
    }
  });
  it('Gauss cells contain their nodes (Chebyshev–Markov–Stieltjes)', () => {
    const r = run('gauss_legendre', 'runge', { n: 20 });
    const g = geometryAt('gauss_legendre', r.trace, 19, [-1, 1], (x) => x, 400)!;
    g.pieces.forEach((piece, i) => {
      expect(g.nodes[i].x).toBeGreaterThan(piece.top[0][0]);
      expect(g.nodes[i].x).toBeLessThan(piece.top[1][0]);
    });
    expect(g.boundaries[g.boundaries.length - 1]).toBe(1);
  });
  it('Monte Carlo rectangle has the estimate as its area', () => {
    const r = run('monte_carlo_integration', 'sin_0_pi', { n: 10, levels: 3 });
    const g = geometryAt('monte_carlo_integration', r.trace, 3, [0, Math.PI], Math.sin, 400)!;
    expect(piecesArea(g.pieces)).toBeCloseTo(r.x as number, 12);
    expect(g.samples).toHaveLength(80);
    expect(g.samples!.filter((s) => s.fresh)).toHaveLength(40);
  });
  it('adaptive Simpson: accepted quartics plus pending parabolas give the estimate', () => {
    const r = run('adaptive_simpson', 'runge', { tol: 1e-4 });
    const p = getProblem<CalculusProblem>('runge');
    for (const k of [1, 5, r.trace.length - 1]) {
      const g = geometryAt('adaptive_simpson', r.trace, k, p.domain, p.f, 1e5)!;
      // Simpson's rule on the 48 equal samples of each piece: exact for the parabolas, and
      // to O(h⁴) for the quartics.
      const area = g.pieces.reduce((s, piece) => {
        const t = piece.top;
        const n = t.length - 1;
        const h = (t[n][0] - t[0][0]) / n;
        let a = t[0][1] + t[n][1];
        for (let i = 1; i < n; i++) a += (i % 2 ? 4 : 2) * t[i][1];
        return s + (h / 3) * a;
      }, 0);
      expect(area).toBeCloseTo(r.trace[k].info.estimate as number, 9);
    }
  });
});

describe('error chart abscissa', () => {
  it('adaptive Simpson: 4k + 2P + 1 points so far, = nFev at the end', () => {
    const r = run('adaptive_simpson', 'sqrt_0_1', { tol: 1e-6 });
    expect(nodesUsed('adaptive_simpson', r.trace[r.trace.length - 1])).toBe(r.nFev);
    expect(nodesUsed('adaptive_simpson', r.trace[0])).toBe(3);
  });
  it('closed nested rules: N + 1 nodes = nFev', () => {
    const r = run('trapezoid', 'exp_0_1', { levels: 4 });
    expect(nodesUsed('trapezoid', r.trace[4])).toBe(r.nFev);
    expect(nodesUsed('left_riemann', run('left_riemann', 'exp_0_1').trace[0])).toBe(4);
  });
  it('observed order of a clean h² sequence is 2', () => {
    expect(observedOrder(10, 1e-2, 100, 1e-4)).toBeCloseTo(2, 12);
    expect(observedOrder(10, 0, 100, 1e-4)).toBeNull();
  });
});

describe('reference integrals on sub-intervals', () => {
  const sub = (id: string, a: number, b: number) =>
    referenceIntegral(getProblem<CalculusProblem>(id), a, b);
  it('uses the problem’s exact value on its whole domain', () => {
    expect(sub('gaussian', -2, 2)).toBe(1.764162781524843);
  });
  it('matches a Gauss reference for every problem', () => {
    for (const p of [
      'poly3',
      'exp_0_1',
      'sin_0_pi',
      'runge',
      'gaussian',
      'oscillatory',
      'arctan_deriv',
    ]) {
      const q = getProblem<CalculusProblem>(p);
      const a = q.domain[0] + 0.13 * (q.domain[1] - q.domain[0]);
      const b = q.domain[0] + 0.71 * (q.domain[1] - q.domain[0]);
      expect(sub(p, a, b)).toBeCloseTo(gaussReference(q.f, a, b, 64), 12);
    }
    expect(sub('sqrt_0_1', 0.25, 1)).toBeCloseTo(7 / 12, 15);
    expect(sub('abs_kink', 0, 0.5)).toBeCloseTo(0.045 + 0.02, 15);
  });
  it('the methods report the error against the sub-interval value when the lab passes it', () => {
    const p = getProblem<CalculusProblem>('exp_0_1');
    const exact = referenceIntegral(p, 0.2, 0.9);
    const { fn, spec } = getMethod('simpson');
    const defaults = Object.fromEntries(spec.params.map((q) => [q.name, q.default]));
    const r = fn({ ...p, exact } as Problem<number>, { ...defaults, bracket: [0.2, 0.9] });
    expect(r.extra.error as number).toBeLessThan(1e-12);
  });
});

describe('filled-in rules and number formatting', () => {
  it('writes this step’s h, values and result', () => {
    const r = run('trapezoid', 'exp_0_1', { levels: 2 });
    const tex = filledRule('trapezoid', 'T_N = …', r.trace[1], r.trace[0], [0, 1]);
    expect(tex).toContain('T_{8} = h\\left[');
    expect(tex).toContain('h = 0.125');
    expect(tex).toContain('\\tfrac12(1)');
    expect(tex).toContain(texNum(r.trace[1].info.estimate as number, 7));
    expect(estimateSymbol('romberg', run('romberg', 'exp_0_1').trace[3])).toBe('R_{3,3}');
    expect(estimateSymbol('adaptive_simpson', { k: 4, info: {} } as never)).toBe('I_{4}');
    expect(estimateTex('simpson')).toBe('S_N');
    expect(estimateTex('simpson_38')).toBe('S_N^{3/8}');
    expect(estimateTex('gauss_legendre')).toBe('G_m');
  });
  it('Simpson shows its 1-4-2 weight pattern and puts the result on its own line', () => {
    const r = run('simpson', 'exp_0_1', { levels: 2 });
    const tex = filledRule('simpson', 'S_N = …', r.trace[2], r.trace[1], [0, 1]);
    expect(tex).toContain('\\left[1 + 4(1.06) + 2(1.13) + \\cdots + 2.72\\right]');
    expect(tex).toContain(`\\phantom{S_{16}} = ${texNum(r.trace[2].info.estimate as number, 7)}`);
  });
  it('Gauss line names the scaled weights w̃ᵢ = (b − a)/2 · wᵢ', () => {
    const r = run('gauss_legendre', 'gaussian', { n: 3 });
    const tex = filledRule('gauss_legendre', 'G', r.trace[2], r.trace[1], [-2, 2]);
    expect(tex).toContain('\\tilde w_i = \\tfrac{b-a}{2}');
    // 2·(5/9) on [−2, 2]: the trace stores the scaled weight.
    expect(tex).toContain('1.11\\,f(');
  });
  it('Romberg line uses R(k, k−1) and R(k−1, k−1)', () => {
    const r = run('romberg', 'exp_0_1', { max_levels: 3 });
    const tex = filledRule('romberg', 'R', r.trace[2], r.trace[1], [0, 1]);
    expect(tex).toContain('{15}');
    expect(tex).toContain(texNum((r.trace[1].info.row as number[])[1], 8));
  });
  it('texNum and agreeingDigits', () => {
    expect(texNum(1.2e-8, 3)).toBe('1.2\\times 10^{-8}');
    expect(texNum(-0.5)).toBe('-0.5');
    expect(agreeingDigits(1.7641598, 1.7641628, 8)).toEqual({ good: '1.7641', rest: '598' });
    expect(agreeingDigits(2, null).good).toBe('');
  });
  it('barycentric interpolation reproduces a polynomial', () => {
    const pts: [number, number][] = [-1, -0.3, 0.2, 0.9].map((x) => [x, x ** 3 - x]);
    expect(barycentric(pts)(0.5)).toBeCloseTo(0.125 - 0.5, 14);
  });
});

describe('presets and defaults tell the truth', () => {
  it('kink: the midpoint error is (1/130)² at N = 13, 26, 52 (node 4/13 is 1/130 from 0.3)', () => {
    const r = run('midpoint_rule', 'abs_kink', { n: 13, levels: 2 });
    for (const s of r.trace) expect(s.info.error as number).toBeCloseTo((1 / 130) ** 2, 15);
    expect(r.converged).toBe(true);
    expect(Math.abs(4 / 13 - 0.3)).toBeCloseTo(1 / 130, 15);
  });
  it('the first view converges for all three methods', () => {
    expect(run('trapezoid', 'gaussian', { n: 2, levels: 8, tol: 1e-6 }).converged).toBe(true);
    expect(run('simpson', 'gaussian', { n: 2, levels: 6 }).converged).toBe(true);
    expect(run('gauss_legendre', 'gaussian', { n: 16 }).converged).toBe(true);
  });
});

describe('nested rules in the lab (Clenshaw–Curtis, Gauss–Patterson)', () => {
  it('the area under the drawn interpolant is the estimate (both rules are interpolatory)', () => {
    for (const [id, pid] of [
      ['clenshaw_curtis', 'runge'],
      ['clenshaw_curtis', 'gaussian'],
      ['gauss_patterson', 'gaussian'],
      ['gauss_patterson', 'exp_0_1'],
    ] as const) {
      const r = run(id, pid);
      const p = getProblem<CalculusProblem>(pid);
      r.trace.forEach((s, k) => {
        // 1,200 samples per piece: Boole's rule on them is exact to rounding for p.
        const g = geometryAt(id, r.trace, k, p.domain, p.f, 1e5)!;
        expect(g.pieces).toHaveLength(1);
        expect(g.pieces[0].top).toHaveLength(1201);
        expect(simpsonArea(g.pieces)).toBeCloseTo(s.info.estimate as number, 9);
      });
    }
  });
  it('marks the nodes of the previous step as reused and the rest as new', () => {
    const r = run('clenshaw_curtis', 'runge', { n: 3, max_levels: 4, tol: 1e-15 });
    const p = getProblem<CalculusProblem>('runge');
    r.trace.forEach((s, k) => {
      const g = geometryAt('clenshaw_curtis', r.trace, k, p.domain, p.f, 400)!;
      const fresh = g.nodes.filter((n) => n.fresh).length;
      const reused = g.nodes.filter((n) => n.reused).length;
      expect(fresh).toBe(k === 0 ? 0 : s.info.new_nodes);
      expect(reused).toBe(k === 0 ? 0 : (s.info.n_points as number) - (s.info.new_nodes as number));
      expect(g.chebyshev).toBe(s.info.n);
      // Weight shares sum to 1 (∑ w̃ᵢ = b − a).
      expect(g.nodes.reduce((acc, n) => acc + (n.weight ?? 0), 0)).toBeCloseTo(1, 13);
    });
    const gp = run('gauss_patterson', 'gaussian');
    const g = geometryAt('gauss_patterson', gp.trace, 3, [-2, 2], (x) => x, 400)!;
    expect(g.chebyshev).toBeUndefined();
    // Level 3 (15 nodes) reuses the 7 of level 2: the odd positions.
    g.nodes.forEach((n, i) => expect(n.reused).toBe(i % 2 === 1));
  });
  it('the Chebyshev barycentric weights give the same interpolant as the general ones', () => {
    const r = run('clenshaw_curtis', 'runge', { n: 4, max_levels: 3, tol: 1e-15 });
    const pts = r.trace[3].info.nodes as [number, number][];
    const cheb = nestedInterpolant('clenshaw_curtis', pts, -1, 1);
    const xs = pts.map((q) => q[0]);
    const bw = baryWeights(xs, -1, 1);
    const general = (x: number) => {
      let num = 0,
        den = 0;
      for (let i = 0; i < pts.length; i++) {
        const t = bw[i] / (x - xs[i]);
        num += t * pts[i][1];
        den += t;
      }
      return num / den;
    };
    for (const x of [-0.97, -0.5, 0.013, 0.31, 0.88]) expect(cheb(x)).toBeCloseTo(general(x), 12);
    pts.forEach(([x, y]) => expect(cheb(x)).toBe(y));
    expect(nestedInterpolant('gauss_patterson', [[0, 3]], -1, 1)(0.7)).toBe(3);
  });
  it('omits the geometry above 4,096 nodes and counts nodes for the error chart', () => {
    const r = run('clenshaw_curtis', 'sqrt_0_1', { n: 64, max_levels: 7, tol: 1e-15 });
    const g = geometryAt('clenshaw_curtis', r.trace, 7, [0, 1], Math.sqrt, 400)!;
    expect(g.tooFine).toBe(true);
    expect(nodesUsed('clenshaw_curtis', r.trace[7])).toBe(8193);
    expect(nodesUsed('clenshaw_curtis', r.trace[7])).toBe(r.nFev);
    const gp = run('gauss_patterson', 'gaussian');
    expect(nodesUsed('gauss_patterson', gp.trace[gp.trace.length - 1])).toBe(gp.nFev);
  });
  it('symbols, filled rules and table columns', () => {
    const r = run('clenshaw_curtis', 'exp_0_1', { n: 2 });
    expect(estimateTex('clenshaw_curtis')).toBe('C_n');
    expect(estimateTex('gauss_patterson')).toBe('P_N');
    expect(estimateSymbol('clenshaw_curtis', r.trace[2])).toBe('C_{8}');
    const gp = run('gauss_patterson', 'exp_0_1');
    expect(estimateSymbol('gauss_patterson', gp.trace[3])).toBe('P_{15}');
    const tex = filledRule('clenshaw_curtis', 'C', r.trace[2], r.trace[1], [0, 1]);
    expect(tex).toContain('C_{8} = \\textstyle\\sum_j \\tilde w_j\\, f(x_j) = ');
    // From a to b: the first term is the weight 1/(2·63) at x = 0.
    expect(tex).toContain(`${texNum(1 / 126, 3)}\\,f(0) + \\cdots`);
    expect(tex).toContain('\\text{5 of 9 values reused}');
    expect(filledRule('gauss_patterson', 'P', gp.trace[0], undefined, [0, 1])).not.toContain(
      'reused',
    );
    expect(columnsFor('clenshaw_curtis').map((c) => c.key)).toEqual([
      'k',
      'n',
      'new',
      'est',
      'err',
      'errest',
    ]);
    expect(columnsFor('gauss_patterson')[1].key).toBe('N');
  });
  it('preset “nested”: Patterson 31, Clenshaw–Curtis 65 and Gauss–Legendre (n = 16) 129 values', () => {
    const cc = run('clenshaw_curtis', 'gaussian');
    const gp = run('gauss_patterson', 'gaussian');
    const gl = run('gauss_legendre', 'gaussian', { n: 16 });
    expect([cc.converged, gp.converged, gl.converged]).toEqual([true, true, true]);
    expect([gp.nFev, cc.nFev, gl.nFev]).toEqual([31, 65, 129]);
  });
});
