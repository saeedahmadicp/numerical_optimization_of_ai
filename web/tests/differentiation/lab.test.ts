/**
 * Differentiation lab: the geometry drawn from Step.info (model.ts) and the claims the "Try this"
 * presets make, checked on the real port.
 */
import { describe, expect, it } from 'vitest';
import { getMethod, hasMethod } from '../../src/core/registry';
import { getProblem, hasProblem } from '../../src/problems/registry';
import type { Result } from '../../src/core/types';
import '../../src/labs/differentiation/setup';
import {
  alignedDigits,
  digitsLost,
  errorsOf,
  hAt,
  parabolaThrough,
  stencilGeometry,
  texNum,
  ulp,
  viewY,
  windowHalfWidth,
  zeroNumeratorKind,
} from '../../src/labs/differentiation/model';
import {
  DEFAULT_LEVELS,
  DEFAULT_SELECTION,
  PRESETS,
  SWEEP_SPECS,
} from '../../src/labs/differentiation/config';

function run(id: string, problem: string, params: Record<string, unknown>): Result {
  const { spec, fn } = getMethod(id);
  const defaults = Object.fromEntries(spec.params.map((p) => [p.name, p.default]));
  return fn(getProblem(problem), { ...defaults, ...(params as Record<string, never>) });
}

describe('stencil geometry', () => {
  const exp = getProblem('exp_0_1') as unknown as {
    f: (x: number) => number;
    grad: (x: number) => number;
  };
  const x0 = 1;
  const f0 = exp.f(x0);
  const s = exp.grad(x0);

  it('forward difference: the estimate line passes through the secant point; in f − ℓ its slope is the error', () => {
    const r = run('forward_difference', 'exp_0_1', { h0: 0.1, levels: 3 });
    const step = r.trace[2];
    const g = stencilGeometry('forward_difference', step, f0, s, 'dev');
    const D = step.info.estimate as number;
    expect(g.slope).toBeCloseTo(D - s, 14);
    expect(g.points).toHaveLength(2);
    expect(g.chords).toHaveLength(1);
    const [p0, p1] = g.chords[0];
    expect((p1.y - p0.y) / (p1.u - p0.u)).toBeCloseTo(D - s, 9);
    // f mode: the slope is D itself.
    expect(stencilGeometry('forward_difference', step, f0, s, 'f').slope).toBe(D);
  });

  it('five-point stencil: an inner (±h) and an outer (±2h) chord', () => {
    const r = run('five_point_stencil', 'exp_0_1', { h0: 0.1, levels: 1 });
    const g = stencilGeometry('five_point_stencil', r.trace[1], f0, s, 'f');
    expect(g.chords).toHaveLength(2);
    const h = r.trace[1].info.h as number;
    expect(Math.abs(g.chords[0][0].u)).toBeCloseTo(h, 15);
    expect(Math.abs(g.chords[1][0].u)).toBeCloseTo(2 * h, 15);
    // D = (4/3)·C(h) − (1/3)·C(2h)
    const slope = ([a, b]: (typeof g.chords)[number]) => (b.y - a.y) / (b.u - a.u);
    expect((4 / 3) * slope(g.chords[0]) - (1 / 3) * slope(g.chords[1])).toBeCloseTo(
      r.trace[1].info.estimate as number,
      10,
    );
  });

  it('second derivative: the parabola through the stencil has curvature D₂', () => {
    const r = run('second_derivative_central', 'gaussian', { h0: 0.25, levels: 2 });
    const g = getProblem('gaussian') as unknown as {
      f: (x: number) => number;
      grad: (x: number) => number;
    };
    const step = r.trace[2];
    const geo = stencilGeometry('second_derivative_central', step, g.f(0.5), g.grad(0.5), 'dev');
    expect(geo.slope).toBeNull();
    expect(2 * geo.parabola!.a).toBeCloseTo(step.info.estimate as number, 9);
  });

  it('complex step: no real stencil, only the estimate line', () => {
    const r = run('complex_step', 'exp_0_1', { levels: 2 });
    const g = stencilGeometry('complex_step', r.trace[2], f0, s, 'dev');
    expect(g.points).toHaveLength(0);
    expect(g.slope).toBeCloseTo((r.trace[2].info.estimate as number) - s, 15);
  });

  it('parabolaThrough interpolates three points exactly', () => {
    const p = parabolaThrough([
      { u: -1, y: 3 },
      { u: 0, y: 1 },
      { u: 2, y: 9 },
    ])!;
    for (const [u, y] of [
      [-1, 3],
      [0, 1],
      [2, 9],
    ])
      expect(p.a * u * u + p.b * u + p.c).toBeCloseTo(y, 12);
    expect(
      parabolaThrough([
        { u: 0, y: 0 },
        { u: 0, y: 1 },
        { u: 1, y: 2 },
      ]),
    ).toBeNull();
  });

  it('viewY subtracts f(x0) and, in f − ℓ mode, the tangent', () => {
    expect(viewY(3, 2, 1, 0.5, 'f')).toBe(2);
    expect(viewY(3, 2, 1, 0.5, 'dev')).toBe(1);
  });
});

describe('numbers', () => {
  it('ulp is the spacing of doubles', () => {
    expect(ulp(1)).toBe(Number.EPSILON);
    expect(ulp(0.5)).toBe(Number.EPSILON / 2);
    expect(ulp(Math.E)).toBe(2 * Number.EPSILON);
  });
  it('alignedDigits finds the digits two values share', () => {
    const { text, common } = alignedDigits([2.718281828459045, 2.718281828745]);
    expect(text[0].slice(0, common)).toBe('2.718281828');
    expect(alignedDigits([1, -1]).common).toBe(0);
  });
  it('digitsLost measures cancellation', () => {
    expect(digitsLost([1, 1 + 1e-10], [-1, 1])).toBeCloseTo(10, 4);
    expect(digitsLost([1, 1], [-1, 1])).toBe(Infinity);
    expect(digitsLost([0, 1], [-1, 1])).toBe(0);
  });
  it('zeroNumeratorKind tells equal values from a balanced second difference', () => {
    expect(zeroNumeratorKind([2, 2], false)).toBe('equal');
    expect(zeroNumeratorKind([1, 2, 3], true)).toBe('balanced');
    expect(zeroNumeratorKind([1, 2, 3], false)).toBe('rounded');
    // The preset "f″ loses twice as fast" at level 30: three different values, Σcᵢfᵢ = 0.
    const r = run('second_derivative_central', 'gaussian', { x0: [0.5], h0: 0.5, levels: 30 });
    const step = r.trace[30];
    const fs = (step.info.stencil as [number, number][]).map(([, v]) => v);
    expect(new Set(fs).size).toBe(3);
    expect(digitsLost(fs, step.info.weights as number[])).toBe(Infinity);
    expect(zeroNumeratorKind(fs, true)).toBe('balanced');
    expect(step.info.estimate).toBe(0);
  });
  it('texNum typesets powers of ten and keeps zeros on request', () => {
    expect(texNum(1.25e-8, 3)).toBe('1.25\\times10^{-8}');
    expect(texNum(1e-8)).toBe('10^{-8}');
    expect(texNum(1, 10)).toBe('1');
    expect(texNum(1, 10, true)).toBe('1.000000000');
    expect(texNum(null)).toBe('\\text{—}');
  });
  it('hAt follows h0·2^(−t) and the window keeps the stencil in view', () => {
    const r = run('central_difference', 'exp_0_1', { h0: 0.1, levels: 4 });
    expect(hAt(r.trace, 2)).toBe(0.025);
    expect(hAt(r.trace, 2.5)).toBeCloseTo(0.025 / Math.SQRT2, 15);
    expect(hAt(r.trace, 99)).toBe(0.1 / 16);
    expect(windowHalfWidth(1, 2)).toBeGreaterThan(2);
    expect(errorsOf(r.trace).every((e) => e !== null && e > 0)).toBe(true);
  });
});

describe('presets', () => {
  it('reference registered methods and problems with sweep values inside the controls', () => {
    const [h0Spec, levelsSpec] = SWEEP_SPECS;
    for (const p of PRESETS) {
      expect(hasProblem(p.problem!)).toBe(true);
      for (const m of p.methods ?? []) expect(hasMethod(m.id)).toBe(true);
      const h0 = Number(p.extra!.h0),
        levels = Number(p.extra!.K);
      expect(h0).toBeGreaterThanOrEqual(h0Spec.min!);
      expect(h0).toBeLessThanOrEqual(h0Spec.max!);
      expect(levels).toBeLessThanOrEqual(levelsSpec.max!);
      const slots = (p.methods ?? []).map((m) => m.slot);
      expect(new Set(slots).size).toBe(slots.length);
    }
    for (const m of DEFAULT_SELECTION) expect(hasMethod(m.id)).toBe(true);
    expect(levelsSpec.default).toBe(DEFAULT_LEVELS);
  });

  const sweep = (id: string) => {
    const p = PRESETS.find((q) => q.id === id)!;
    return {
      problem: p.problem!,
      x0: p.start as number,
      h0: Number(p.extra!.h0),
      levels: Number(p.extra!.K),
    };
  };

  it('"the V": each finite difference bottoms out, then round-off climbs; the complex step does not', () => {
    const { problem, x0, h0, levels } = sweep('v');
    for (const id of ['forward_difference', 'central_difference', 'five_point_stencil']) {
      const es = errorsOf(run(id, problem, { x0, h0, levels }).trace) as number[];
      const min = Math.min(...es);
      expect(es[0]).toBeGreaterThan(100 * min);
      expect(es[es.length - 1]).toBeGreaterThan(100 * min);
    }
    const cs = errorsOf(run('complex_step', problem, { x0, h0, levels }).trace) as number[];
    expect(Math.max(...cs.slice(25))).toBeLessThanOrEqual(2 * Number.EPSILON * Math.E);
  });

  it('"complex step": central returns exactly 0 at the end, the complex step stays exact', () => {
    const { problem, x0, h0, levels } = sweep('complex');
    const c = run('central_difference', problem, { x0, h0, levels });
    expect(c.trace[levels].info.estimate).toBe(0);
    const z = run('complex_step', problem, { x0, h0, levels });
    expect(Math.abs(z.trace[levels].info.error as number)).toBeLessThan(1e-15);
  });

  it('"kink": central is exact from h = 0.05; Richardson needs more levels', () => {
    const { problem, x0, h0, levels } = sweep('kink');
    const c = errorsOf(run('central_difference', problem, { x0, h0, levels }).trace) as number[];
    const r = errorsOf(
      run('richardson_extrapolation', problem, { x0, h0, levels }).trace,
    ) as number[];
    expect(c[3]).toBeLessThan(1e-14); // h = 0.05
    expect(r[3]).toBeGreaterThan(1e-2);
    expect(r[5]).toBeGreaterThan(1e-5);
  });

  it('"f″": the second difference loses about twice as many digits to round-off', () => {
    const { problem, x0, h0, levels } = sweep('second');
    const d1 = errorsOf(run('central_difference', problem, { x0, h0, levels }).trace) as number[];
    const d2 = errorsOf(
      run('second_derivative_central', problem, { x0, h0, levels }).trace,
    ) as number[];
    expect(Math.min(...d2)).toBeGreaterThan(10 * Math.min(...d1));
    expect(d2[levels]).toBeGreaterThan(1e4 * d1[levels]);
  });
});
