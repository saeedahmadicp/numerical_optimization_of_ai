import { describe, expect, it } from 'vitest';
import {
  axpy,
  cholesky,
  choleskySolve,
  dot,
  eigSym2,
  matvec,
  norm,
  solve,
} from '../src/core/linalg';
import { MINUS, sci, sig, sigFixed, superscript, tick, vec } from '../src/core/format';
import { camel, parseJson, resultFromJson, reviveNumbers } from '../src/core/json';
import {
  clearRegistry,
  getMethod,
  listMethods,
  param,
  registerMethod,
  runMethod,
} from '../src/core/registry';

describe('linalg', () => {
  it('dot, axpy, norm, matvec', () => {
    expect(dot([1, 2, 3], [4, 5, 6])).toBe(32);
    expect(axpy(2, [1, 1], [0, 1])).toEqual([2, 3]);
    expect(norm([3, 4])).toBe(5);
    expect(
      matvec(
        [
          [1, 2],
          [3, 4],
        ],
        [1, 1],
      ),
    ).toEqual([3, 7]);
  });
  it('solve with partial pivoting (zero leading pivot)', () => {
    const x = solve(
      [
        [0, 2, 1],
        [1, 1, 1],
        [2, 1, 0],
      ],
      [3, 3, 3],
    )!;
    expect(x[0]).toBeCloseTo(1, 12);
    expect(x[1]).toBeCloseTo(1, 12);
    expect(x[2]).toBeCloseTo(1, 12);
    expect(
      solve(
        [
          [1, 2],
          [2, 4],
        ],
        [1, 2],
      ),
    ).toBeNull();
  });
  it('cholesky factor and solve; null for indefinite', () => {
    const A = [
      [4, 2],
      [2, 3],
    ];
    const L = cholesky(A)!;
    expect(L[0][0]).toBeCloseTo(2, 14);
    expect(choleskySolve(L, [6, 5]).map((v) => Number(v.toFixed(12)))).toEqual([1, 1]);
    expect(
      cholesky([
        [1, 2],
        [2, 1],
      ]),
    ).toBeNull();
  });
  it('eigSym2 returns ascending eigenpairs', () => {
    const { values, vectors } = eigSym2([
      [2, 1],
      [1, 2],
    ]);
    expect(values[0]).toBeCloseTo(1, 14);
    expect(values[1]).toBeCloseTo(3, 14);
    const [v] = vectors;
    expect(Math.abs(v[0] + v[1])).toBeLessThan(1e-12);
  });
});

describe('format', () => {
  it('superscripts and scientific notation', () => {
    expect(superscript(-12)).toBe('⁻¹²');
    expect(sci(1.234e-8, 3)).toBe('1.23×10⁻⁸');
    expect(sci(-5e6, 3)).toBe(`${MINUS}5×10⁶`);
    expect(sci(Infinity)).toBe('∞');
    expect(sci(null)).toBe('—');
  });
  it('significant figures', () => {
    expect(sig(3.14159, 3)).toBe('3.14');
    expect(sig(-0.5)).toBe(`${MINUS}0.5`);
    expect(sig(123456, 3)).toBe('1.23×10⁵');
    expect(sigFixed(1.5, 4)).toBe('1.500');
    expect(sigFixed(1e-9, 3)).toBe(`1.00e${MINUS}09`);
    expect(vec([1, -2])).toBe(`(1, ${MINUS}2)`);
    expect(tick(0.30000000000000004, 0.1)).toBe('0.3');
  });
});

describe('json loader', () => {
  it('revives infinities and keeps null', () => {
    expect(reviveNumbers({ a: 'inf', b: ['-inf', null, 1] })).toEqual({
      a: Infinity,
      b: [-Infinity, null, 1],
    });
    expect(parseJson('{"x":"inf"}')).toEqual({ x: Infinity });
  });
  it('camelCases snake_case keys', () => {
    expect(camel('grad_norm')).toBe('gradNorm');
    expect(camel('A_ub')).toBe('aUb');
    expect(camel('n_iter')).toBe('nIter');
  });
  it('maps a Python Result payload', () => {
    const r = resultFromJson({
      method: 'm',
      x: [1, 2],
      fun: 0.5,
      converged: true,
      message: 'ok',
      n_iter: 1,
      n_fev: 2,
      n_gev: 1,
      n_hev: 0,
      extra: {},
      trace: [
        { k: 0, x: [0, 0], fun: 'inf', grad_norm: null, step_size: null, info: { radius: 1 } },
      ],
    });
    expect(r.nIter).toBe(1);
    expect(r.trace[0].fun).toBe(Infinity);
    expect(r.trace[0].gradNorm).toBeNull();
    expect(r.trace[0].info).toEqual({ radius: 1 });
  });
});

describe('registry', () => {
  it('registers, lists by family, rejects unknown params', () => {
    clearRegistry();
    const fn = (_p: unknown, o: Record<string, unknown>) => ({
      method: 'demo',
      x: o.alpha,
      fun: 0,
      converged: true,
      message: '',
      nIter: 0,
      nFev: 0,
      nGev: 0,
      nHev: 0,
      trace: [],
      extra: {},
    });
    registerMethod(
      { id: 'demo', family: 'roots', name: 'Demo', params: [param.float('alpha', 0.5)] },
      fn,
    );
    registerMethod({ id: 'b_demo', family: 'unconstrained', name: 'B' }, fn);
    expect(listMethods('roots').map((m) => m.spec.id)).toEqual(['demo']);
    expect(listMethods().map((m) => m.spec.id)).toEqual(['demo', 'b_demo']);
    expect(getMethod('demo').spec.params[0].default).toBe(0.5);
    expect(runMethod('demo', null).x).toBe(0.5);
    expect(() => runMethod('demo', null, { nope: 1 })).toThrow(/unknown parameter/);
    expect(() => registerMethod({ id: 'x', family: 'nope' as never, name: 'x' }, fn)).toThrow(
      /unknown family/,
    );
    clearRegistry();
  });
});
