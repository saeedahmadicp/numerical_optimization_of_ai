import { describe, expect, it } from 'vitest';
import { describeResult, evidence, shortMethodName } from './status';
import type { Result } from '../../core/types';

describe('evidence() states tolerances as numbers, never by their ids', () => {
  it('drops "tol =" before a number', () => {
    expect(evidence('bracket half-width 5.82e-11 <= x ≤ tol = 1e-10')).toBe(
      'bracket half-width 5.82×10⁻¹¹ <= x ≤ 10⁻¹⁰',
    );
    expect(evidence('x ≤ tol = 1e-10')).toBe('x ≤ 10⁻¹⁰');
  });

  it('drops "gtol =" before a number', () => {
    expect(evidence('‖∇f‖ = 0.136 > gtol = 1e-6')).toBe('‖∇f‖ = 0.136 > 10⁻⁶');
  });

  it('keeps an id that is not followed by a number, and fills it from params', () => {
    expect(evidence('‖∇f‖ = 1.3e-11 ≤ gtol', { gtol: 1e-8 })).toBe('‖∇f‖ = 1.3×10⁻¹¹ ≤ 10⁻⁸');
  });

  it('does not touch words that contain an id', () => {
    expect(evidence('xtol = 1e-8; stol is not an id')).toBe('10⁻⁸; stol is not an id');
  });
});

describe('evidence() uses one notation on both sides of a comparison', () => {
  it('sets the threshold in ×10ⁿ when the value needs it', () => {
    expect(
      evidence('temperature T = 0.000998 ≤ T_min = 0.001 (frozen); best f = 0.25', { T_min: 1e-3 }),
    ).toBe('temperature T = 9.98×10⁻⁴ ≤ 10⁻³ (frozen); best f = 0.25');
  });
  it('leaves two plain decimals alone', () => {
    expect(evidence('|f| = 0.25 > 0.125')).toBe('|f| = 0.25 > 0.125');
  });
  it('keeps a mantissa that is not 1', () => {
    expect(evidence('‖∇f‖ = 2e-9 ≤ 5e-8')).toBe('‖∇f‖ = 2×10⁻⁹ ≤ 5×10⁻⁸');
  });
});

describe('describeResult() wording', () => {
  const run: Result = {
    method: 'm',
    x: 0,
    fun: 0,
    converged: true,
    message: 'converged',
    nIter: 47,
    nFev: 0,
    nGev: 0,
    nHev: 0,
    trace: [],
    extra: {},
  };
  it('spells out the unit in the short form', () => {
    expect(describeResult(run).short).toBe('converged · 47 iterations');
    expect(describeResult({ ...run, converged: false, message: 'reached max_iter' }).short).toBe(
      'budget · 47 iterations',
    );
    expect(describeResult(run, undefined, ['trial', 'trials']).short).toBe('converged · 47 trials');
  });
  it('adds a lab qualifier to the badge and the sentence', () => {
    const st = describeResult(run, undefined, {
      noun: ['move', 'moves'],
      converged: { badge: '2-opt', long: '2-optimal' },
    });
    expect(st.long).toBe('Converged in 47 moves — 2-optimal');
    expect(st.qualifier).toBe('2-opt');
    expect(st.count).toBe('47');
  });
});

describe('shortMethodName()', () => {
  it('drops the parenthetical variant only', () => {
    expect(shortMethodName('Brent (zeroin)')).toBe('Brent');
    expect(shortMethodName('Conjugate gradient (Polak–Ribière+)')).toBe('Conjugate gradient');
    expect(shortMethodName('Nelder–Mead')).toBe('Nelder–Mead');
  });
});

describe('describeResult() count', () => {
  const base: Result = {
    method: 'm',
    x: 0,
    fun: 0,
    converged: true,
    message: 'converged',
    nIter: 1234,
    nFev: 0,
    nGev: 0,
    nHev: 0,
    trace: [],
    extra: {},
  };
  it('gives the count alone for compact badges', () => {
    expect(describeResult(base).count).toBe('1,234');
    expect(describeResult({ ...base, converged: false, message: 'reached max_iter' }).count).toBe(
      '1,234',
    );
  });
});
