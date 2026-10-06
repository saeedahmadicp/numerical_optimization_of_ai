import { describe, expect, it } from 'vitest';
import { param } from '../src/core/registry';
import { vecFixed } from '../src/core/format';
import type { Result } from '../src/core/types';
import { codecs } from '../src/app/useUrlState';
import { coerceParam, sanitizeSelection } from '../src/labs/_shell/slots';
import { selectionCodec } from '../src/labs/_shell/useLabRuns';
import { describeResult } from '../src/labs/_shell/status';
import { overlappingPaths, pathCoverage } from '../src/viz/PathLayer';
import { computeField, levelSpecFor } from '../src/viz/contourField';

const methods = [
  {
    spec: {
      id: 'gradient_descent',
      params: [
        param.float('alpha', 1e-3, { min: 1e-5, max: 1, log: true }),
        param.int('max_iter', 100, { min: 1, max: 5000 }),
      ],
    },
  },
  { spec: { id: 'heavy_ball', params: [param.float('beta', 0.9, { min: 0, max: 0.999 })] } },
  { spec: { id: 'nesterov', params: [param.choice('mode', 'a', ['a', 'b'])] } },
];

describe('URL codecs', () => {
  it('tuple(n) rejects the wrong length', () => {
    const c = codecs.tuple(2);
    expect(c.parse('1')).toBeUndefined();
    expect(c.parse('1,2,3')).toBeUndefined();
    expect(c.parse('-1.2,1')).toEqual([-1.2, 1]);
    expect(c.parse('')).toBeUndefined();
  });
  it('oneOf rejects unknown values', () => {
    const c = codecs.oneOf(['2d', '3d']);
    expect(c.parse('3d')).toBe('3d');
    expect(c.parse('foo')).toBeUndefined();
  });
});

describe('method selection', () => {
  it('drops unknown and duplicate ids and reports the unknown ones', () => {
    const sel = selectionCodec.parse('gradient_descent~0,bfgs~1,gradient_descent~2,heavy_ball~1')!;
    const { value, dropped } = sanitizeSelection(sel, methods);
    expect(value.map((m) => m.id)).toEqual(['gradient_descent', 'heavy_ball']);
    expect(dropped).toEqual(['bfgs']);
  });
  it('gives colliding or invalid slots a free slot, keeping valid later slots', () => {
    const sel = selectionCodec.parse('nesterov~3,heavy_ball~3,gradient_descent~9')!;
    const { value } = sanitizeSelection(sel, methods);
    expect(value.map((m) => [m.id, m.slot])).toEqual([
      ['nesterov', 3],
      ['heavy_ball', 0],
      ['gradient_descent', 1],
    ]);
    expect(new Set(value.map((m) => m.slot)).size).toBe(value.length);
  });
  it('keeps declared params only, coerced and clamped', () => {
    const sel = selectionCodec.parse(
      'gradient_descent~0~alpha=5~max_iter=12.6~bogus=1,nesterov~1~mode=z',
    )!;
    const { value } = sanitizeSelection(sel, methods);
    expect(value[0].params).toEqual({ alpha: 1, max_iter: 13 });
    expect(value[1].params).toEqual({});
  });
  it('coerceParam handles every kind', () => {
    expect(coerceParam(param.bool('b', false), 'true')).toBe(true);
    expect(coerceParam(param.float('x', 0), 'abc')).toBeUndefined();
    expect(coerceParam(param.choice('c', 'a', ['a', 'b']), 'b')).toBe('b');
  });
});

describe('run status', () => {
  const base: Result = {
    method: 'm',
    x: null,
    fun: null,
    converged: false,
    message: '',
    nIter: 1000,
    nFev: 0,
    nGev: 0,
    nHev: 0,
    trace: [],
    extra: {},
  };
  it('speaks plain language', () => {
    expect(describeResult({ ...base, converged: true, nIter: 204 }).long).toBe(
      'Converged in 204 iterations',
    );
    expect(describeResult({ ...base, message: 'max_iter reached' }).long).toBe(
      'Stopped at the 1,000-iteration budget without converging',
    );
    expect(describeResult({ ...base, message: 'diverged (non-finite value)', nIter: 3 }).tone).toBe(
      'bad',
    );
    expect(describeResult(base, 'x0 must have length 2')).toMatchObject({
      tone: 'bad',
      short: 'invalid input',
    });
  });
});

describe('overlapping paths', () => {
  it('reports a path hidden under another, once, and ignores distinct ones', () => {
    const a: [number, number][] = Array.from({ length: 50 }, (_, i) => [i / 49, 0]);
    const b: [number, number][] = Array.from({ length: 30 }, (_, i) => [i / 29, 0.001]);
    const c: [number, number][] = Array.from({ length: 50 }, (_, i) => [i / 49, 0.5]);
    // a and b coincide: only the later one (b) is reported as hidden.
    expect(overlappingPaths([a, b, c], 0.01)).toEqual([[1, 0]]);
  });
  it('is one-sided: a short path along a longer one is hidden, not the reverse', () => {
    const long: [number, number][] = [
      [0, 0],
      [1, 0],
      [1, 1],
    ];
    const short: [number, number][] = [
      [0, 0],
      [0.9, 0],
    ];
    expect(overlappingPaths([long, short], 0.01)).toEqual([[1, 0]]);
    expect(pathCoverage(long, short, 0.01)).toBeLessThan(0.5);
  });
});

describe('fixed contour levels', () => {
  it('a sub-window keeps the full-domain transform and level values', () => {
    const f = (x: number, y: number) => (1 - x) ** 2 + 100 * (y - x * x) ** 2;
    const spec = levelSpecFor(f, [
      [-2, 2],
      [-1, 3],
    ]);
    expect(spec.scale).toBe('log');
    const sample = (x0: number, x1: number, y0: number, y1: number, n = 24) => {
      const v = new Float64Array(n * n);
      for (let j = 0; j < n; j++)
        for (let i = 0; i < n; i++)
          v[j * n + i] = f(x0 + ((x1 - x0) * i) / (n - 1), y0 + ((y1 - y0) * j) / (n - 1));
      return v;
    };
    const lut = new Uint8ClampedArray(256 * 3);
    const common = { nx: 24, ny: 24, width: 8, height: 8, levels: 16, scale: 'auto' as const, lut };
    const full = computeField({ id: 1, values: sample(-2, 2, -1, 3), levelSpec: spec, ...common });
    const zoom = computeField({
      id: 2,
      values: sample(0.5, 1, 0.5, 1),
      levelSpec: spec,
      ...common,
    });
    expect(zoom.scale).toBe(full.scale);
    expect(zoom.levelValues).toEqual(full.levelValues);
  });
});

describe('format', () => {
  it('vecFixed keeps digits and sign columns aligned', () => {
    expect(vecFixed([0.4617, -0.21047], 5)).toBe('( 0.46170,−0.21047)');
    expect(vecFixed([0.46257, 0.21128], 5)).toBe('( 0.46257, 0.21128)');
  });
});
