/**
 * The systems lab draws the right mathematics: the model lines ℓ₁, ℓ₂ cross at the Newton point
 * 𝐱ₖ + 𝐩ₖ (and at 𝐱ₖ₊₁ when α = 1), basin labels classify runs correctly, and marching-squares
 * segments are chained into polylines.
 */
import { describe, expect, it } from 'vitest';
import '../../methods/roots/systems';
import '../../problems/systems';
import { runMethod } from '../../core/registry';
import { getProblem } from '../../problems/registry';
import { implicitSegments } from '../../viz/overlays2d';
import type { SystemProblem } from '../../problems/systems';
import {
  geometryIndex,
  linearizationLine,
  modelError,
  modelRoot,
  nearestRoot,
  rootIndex,
  stepGeometry,
} from './geometry';
import { computeBasins, NO_ROOT, OTHER_ROOT } from './basins';
import { chainSegments } from './curves';
import { texNum } from './tex';

/** Signed distance-like value of 𝐳 against the line {L(𝐳) = 0} through its two drawn endpoints. */
function onLine(z: readonly number[], a: readonly number[], b: readonly number[]) {
  const cross = (b[0] - a[0]) * (z[1] - a[1]) - (b[1] - a[1]) * (z[0] - a[0]);
  return Math.abs(cross) / Math.hypot(b[0] - a[0], b[1] - a[1]);
}

describe('step geometry', () => {
  for (const [pid, method] of [
    ['intersecting_circles', 'newton_system'],
    ['trig_system', 'newton_system'],
    ['circle_line', 'broyden'],
  ] as const) {
    it(`${method} on ${pid}: ℓ₁ and ℓ₂ cross at 𝐱ₖ + 𝐩ₖ = 𝐱ₖ₊₁`, () => {
      const p = getProblem<SystemProblem>(pid);
      const r = runMethod(method, p, {});
      for (let j = 1; j < r.trace.length; j++) {
        const g = stepGeometry(r.trace, j)!;
        expect(g).not.toBeNull();
        const scale = 1 + Math.hypot(...g.target);
        // Exactly where the method went (α = 1 without damping).
        expect(Math.hypot(g.target[0] - g.to[0], g.target[1] - g.to[1])).toBeLessThan(
          1e-12 * scale,
        );
        for (let i = 0; i < 2; i++) {
          const ln = linearizationLine(g.from, g.F[i], g.M[i], 10)!;
          expect(onLine(g.target, ln.from, ln.to)).toBeLessThan(1e-9 * scale);
        }
        const root = modelRoot(g.from, g.F, g.M)!;
        expect(Math.hypot(root[0] - g.target[0], root[1] - g.target[1])).toBeLessThan(1e-9 * scale);
      }
    });
  }

  it('damped Newton: the arrow goes to 𝐱ₖ + 𝐩ₖ, the iterate to 𝐱ₖ + α𝐩ₖ', () => {
    const r = runMethod('newton_system', getProblem('trig_system'), {
      x0: [-1, 1.25],
      damping: true,
    });
    const damped = r.trace.map((_, j) => stepGeometry(r.trace, j)).find((g) => g && g.alpha < 1)!;
    expect(damped).toBeDefined();
    const { from, p, alpha, to, trials } = damped;
    expect(to[0]).toBeCloseTo(from[0] + alpha * p[0], 12);
    expect(trials[trials.length - 1][0]).toBe(alpha);
  });

  it('geometryIndex: the step about to be taken, the last one at the end', () => {
    expect(geometryIndex(0, 6)).toBe(1);
    expect(geometryIndex(2.4, 6)).toBe(3);
    expect(geometryIndex(5, 6)).toBe(5);
    expect(geometryIndex(0, 1)).toBe(0);
  });

  it('modelError is 0 for the exact Jacobian and positive for a Broyden matrix', () => {
    const p = getProblem<SystemProblem>('intersecting_circles');
    const r = runMethod('broyden', p, {});
    const g1 = stepGeometry(r.trace, 1)!;
    expect(modelError(g1.M, p.jac(g1.from))).toBe(0); // B₀ = J(𝐱₀)
    const g3 = stepGeometry(r.trace, 3)!;
    expect(modelError(g3.M, p.jac(g3.from))).toBeGreaterThan(0);
  });

  it('rootIndex / nearestRoot', () => {
    const roots = [
      [0, 0],
      [1, 1],
    ];
    expect(rootIndex([1 + 1e-9, 1], roots)).toBe(1);
    expect(rootIndex([0.5, 0.5], roots)).toBe(-1);
    expect(nearestRoot([0.9, 0.8], roots)).toEqual([1, 1]);
  });
});

describe('basins', () => {
  it('classify every start by the root reached (Newton on two circles: symmetric halves)', () => {
    const p = getProblem<SystemProblem>('intersecting_circles');
    const r = computeBasins({
      id: 1,
      problemId: p.id,
      methodId: 'newton_system',
      params: {},
      domain: p.domain,
      nx: 16,
      ny: 12,
    });
    expect(r.labels.length).toBe(16 * 12);
    // Spot-check against a direct run.
    for (const s of [0, 37, 100, 191]) {
      const i = s % 16,
        j = Math.floor(s / 16);
      const x = p.domain[0][0] + ((i + 0.5) / 16) * (p.domain[0][1] - p.domain[0][0]);
      const y = p.domain[1][0] + ((j + 0.5) / 12) * (p.domain[1][1] - p.domain[1][0]);
      const run = runMethod('newton_system', p, { x0: [x, y] });
      const want = run.converged
        ? rootIndex(run.x as number[], p.roots) >= 0
          ? rootIndex(run.x as number[], p.roots)
          : OTHER_ROOT
        : NO_ROOT;
      expect(r.labels[s]).toBe(want);
      expect(r.iters[s]).toBe(run.nIter);
    }
  });

  it('row ranges tile the grid', () => {
    const base = {
      id: 1,
      problemId: 'trig_system',
      methodId: 'newton_system',
      params: {},
      domain: getProblem<SystemProblem>('trig_system').domain,
      nx: 10,
      ny: 8,
    };
    const all = computeBasins(base);
    const a = computeBasins({ ...base, j0: 0, j1: 3 });
    const b = computeBasins({ ...base, j0: 3, j1: 8 });
    expect([...a.labels, ...b.labels]).toEqual([...all.labels]);
  });
});

describe('zero curves', () => {
  it('chainSegments joins a circle into one closed polyline', () => {
    const segs = implicitSegments((x, y) => x * x + y * y - 1, [-2, 2], [-2, 2], 60, 60);
    const lines = chainSegments(segs, 1e-9);
    expect(lines.length).toBe(1);
    const line = lines[0];
    expect(line.length).toBe(segs.length / 4 + 1);
    for (const [x, y] of line) expect(Math.abs(Math.hypot(x, y) - 1)).toBeLessThan(0.01);
    const [a, b] = [line[0], line[line.length - 1]];
    expect(Math.hypot(a[0] - b[0], a[1] - b[1])).toBeLessThan(1e-9);
  });

  it('texNum', () => {
    expect(texNum(1e-8)).toBe('10^{-8}');
    expect(texNum(-2.5e-10)).toBe('-2.5\\times10^{-10}');
    expect(texNum(0.74162)).toBe('0.7416');
    expect(texNum(0)).toBe('0');
  });
});
