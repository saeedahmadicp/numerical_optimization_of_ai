/**
 * The review fixes of the unconstrained lab: the lens corner, numbers in narrow table columns,
 * the CG and modified-Newton step text, and the line-search rule each method runs.
 */
import { describe, expect, it } from 'vitest';
import './setup';
import { defaults, getMethod } from '../../core/registry';
import { getProblem } from '../../problems/registry';
import type { Problem2D, Result } from '../../core/types';
import { HARD, LENS_SIZE, cornerRect, pathMarks, pickCorner, type Mark } from './lensPlace';
import { short } from './columns';
import { plural, stepView, tauDigits } from './stepTex';
import { searchRuleOf } from './search';

const run = (id: string, pid: string, extra: Record<string, unknown> = {}) => {
  const m = getMethod<Problem2D>(id);
  const problem = getProblem<Problem2D>(pid);
  const params = { ...defaults(m.spec), ...extra };
  const result = m.fn(problem, params as never) as Result;
  return { problem, params, result };
};

describe('lens corner', () => {
  const W = 700;
  const H = 600;
  it('never covers a hard mark (𝐱₀ or the focused step)', () => {
    const tl = cornerRect('tl', W, H)!;
    const tr = cornerRect('tr', W, H)!;
    const marks: Mark[] = [
      { x: tl.x + 50, y: tl.y + 50, w: HARD },
      // Many path points in the top right: it is still the best corner after top left.
      ...Array.from({ length: 5 }, (_, i) => ({ x: tr.x + 20 * i, y: tr.y + 30, w: 1 })),
    ];
    expect(pickCorner(marks, W, H, null)).toBe('bl');
    // From the top left (now blocked) it moves.
    expect(pickCorner(marks, W, H, 'tl')).toBe('bl');
  });
  it('keeps its corner unless another is clearly better', () => {
    const tr = cornerRect('tr', W, H)!;
    const marks: Mark[] = [{ x: tr.x + 10, y: tr.y + 10, w: 1 }];
    // 1 point in tr vs 0 elsewhere: within the hysteresis margin, so tr stays.
    expect(pickCorner(marks, W, H, 'tr')).toBe('tr');
    const many = Array.from({ length: 20 }, (_, i) => ({ x: tr.x + 5 * i, y: tr.y + 9, w: 4 }));
    expect(pickCorner(many, W, H, 'tr')).toBe('tl');
  });
  it('returns null when every corner holds a hard mark, or the plot is too small', () => {
    const marks: Mark[] = (['tl', 'tr', 'bl', 'br'] as const).map((c) => {
      const r = cornerRect(c, W, H)!;
      return { x: r.x + 90, y: r.y + 90, w: HARD };
    });
    expect(pickCorner(marks, W, H, 'tr')).toBeNull();
    expect(cornerRect('tl', LENS_SIZE, LENS_SIZE)).toBeNull();
  });
  it('samples long segments, so a step that crosses a corner counts', () => {
    const m = pathMarks(
      [
        [0, 0],
        [400, 0],
      ],
      HARD,
    );
    expect(m.length).toBeGreaterThan(10);
    expect(m.some((p) => p.x > 150 && p.x < 250)).toBe(true);
  });
});

describe('narrow numeric columns', () => {
  it('write ×10ⁿ with superscripts (never e-notation) and keep d significant digits', () => {
    expect(short(0.00135)).toBe('1.35×10⁻³');
    expect(short(-1.38e-11)).toBe('−1.38×10⁻¹¹');
    expect(short(0.0135)).toBe('0.0135');
    expect(short(999.7)).toBe('1000');
    expect(short(1234.5)).toBe('1235');
    expect(short(12345)).toBe('1.23×10⁴');
    expect(short(1.9e-6, 2)).toBe('1.9×10⁻⁶');
    expect(short(-0.5)).toBe('−0.500');
    expect(short(0)).toBe('0');
    expect(short(null)).toBe('—');
    for (const x of [1e-300, -9.99e-100, 0.009999, 123.456, -4321.9, 6.02e23, -1e-5])
      expect(short(x)).not.toMatch(/\de/);
  });
});

describe('step text', () => {
  it('CG at k = 1: 𝐝₀ = −∇f(𝐱₀), with no β-term and no 𝐝₋₁', () => {
    const { problem, params, result } = run('cg_polak_ribiere', 'rosenbrock');
    const v = stepView({
      kind: 'cg',
      method: 'cg_polak_ribiere',
      trace: result.trace,
      g: 1,
      params,
      problem,
    });
    expect(v.tex).toContain('\\mathbf{d}_{0} &= -\\nabla f(\\mathbf{x}_{0})');
    expect(v.tex).not.toContain('{-1}');
    expect(v.tex).not.toContain('\\beta_{0}');
  });
  it('CG at k ≥ 2 names the arrow β_{k−1} 𝐝_{k−2}, as the canvas and the key do', () => {
    const { problem, params, result } = run('cg_polak_ribiere', 'rosenbrock');
    const g = result.trace.findIndex(
      (_, i) =>
        i >= 2 &&
        typeof result.trace[i - 1].info.beta === 'number' &&
        result.trace[i - 1].info.beta !== 0,
    );
    expect(g).toBeGreaterThanOrEqual(2);
    const v = stepView({
      kind: 'cg',
      method: 'cg_polak_ribiere',
      trace: result.trace,
      g,
      params,
      problem,
    });
    expect(v.tex).toContain(`\\beta_{${g - 1}}\\,\\mathbf{d}_{${g - 2}}`);
  });
  it('modified Newton prints τ to the resolution of β, so ∇²f + τI is visibly nonsingular', () => {
    const { problem, params, result } = run('modified_newton', 'himmelblau', { x0: [0, 0] });
    const s = result.trace[1];
    expect(s.info.tau).toBeCloseTo(42.001, 6);
    const v = stepView({
      kind: 'newton',
      method: 'modified_newton',
      trace: result.trace,
      g: 1,
      params,
      problem,
    });
    expect(v.tex).toContain('42.001');
    expect(v.note).toContain('42.001');
    expect(v.note).toMatch(/1 Cholesky attempt\)/);
    expect(tauDigits(42.001, 1e-3)).toBe(6);
    expect(plural(2, 'Cholesky attempt')).toBe('2 Cholesky attempts');
  });
  it('damped Newton names the line search it ran', () => {
    const { problem, params, result } = run('damped_newton', 'rosenbrock', {
      line_search: 'strong_wolfe',
    });
    const v = stepView({
      kind: 'newton',
      method: 'damped_newton',
      trace: result.trace,
      g: 1,
      params,
      problem,
    });
    expect(v.note).toContain('strong Wolfe');
    expect(v.note).not.toContain('Backtracking');
  });
});

describe('line-search rule per method', () => {
  it('pure Newton and pure BB run no test; GLL with nonmonotone; the param otherwise', () => {
    expect(searchRuleOf('pure_newton', {})).toBe('newton');
    expect(searchRuleOf('barzilai_borwein', { nonmonotone: false })).toBe('bb');
    expect(searchRuleOf('barzilai_borwein', { nonmonotone: true })).toBe('gll');
    expect(searchRuleOf('gradient_descent', { step_rule: 'fixed' })).toBe('fixed');
    expect(searchRuleOf('modified_newton', { line_search: 'goldstein' })).toBe('goldstein');
    expect(searchRuleOf('bfgs', { line_search: 'strong_wolfe' })).toBe('strong_wolfe');
  });
});
