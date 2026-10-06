/** The affine-scaling ellipse drawn on the plane is the method's own metric, projected. */
import { describe, expect, it } from 'vitest';
import '../../problems/lp';
import { getProblem } from '../../problems/registry';
import type { LinearProgram } from '../../core/types';
import { affineScaling, scaledForm } from '../../methods/lp/interior_point';
import { dikinMatrix, halfSpaces } from './geometry';
import { affineScalingMetric } from './affineMetric';

const lp = (id: string) => getProblem<LinearProgram>(id);
const quad = (Q: number[][], d: readonly number[]) =>
  d.reduce((s, a, i) => s + a * d.reduce((t, b, j) => t + Q[i][j] * b, 0), 0);

describe('affineScalingMetric', () => {
  it('with t = 0 it is the polygon’s Dikin ellipse Σ gᵢgᵢᵀ/sᵢ²', () => {
    for (const x of [
      [1, 2],
      [3, 3.5],
      [0.2, 0.3],
    ]) {
      const Q = affineScalingMetric(lp('wyndor'), x, 0)!;
      const D = dikinMatrix(halfSpaces(lp('wyndor')), x)!;
      for (let i = 0; i < 2; i++)
        for (let j = 0; j < 2; j++)
          expect(Math.abs(Q[i][j] - D[i][j])).toBeLessThan(1e-9 * (1 + Math.abs(D[i][j])));
    }
  });

  it('while t > 0 it exists where the polygon has none: the start lies outside the polygon', () => {
    const p = lp('wyndor');
    const s0 = affineScaling(p).trace[0];
    const x = s0.x as number[];
    expect(s0.info.artificial).toBe(1);
    expect(dikinMatrix(halfSpaces(p), x)).toBeNull();
    const Q = affineScalingMetric(p, x, 1)!;
    expect(Q[0][0] * Q[1][1] - Q[0][1] ** 2).toBeGreaterThan(0);
  });

  it('every step of the method, normalized in its metric, lies on or inside the drawn ellipse', () => {
    const p = lp('wyndor');
    const r = affineScaling(p);
    const sf = scaledForm(p);
    let checked = 0;
    for (let k = 0; k + 1 < r.trace.length; k++) {
      const s = r.trace[k];
      const t = s.info.artificial as number;
      const x = s.x as number[];
      const Q = affineScalingMetric(p, x, t);
      if (!Q) continue;
      const dx = (r.trace[k + 1].x as number[]).map((v, j) => v - x[j]);
      // ‖Z⁻¹dz‖ of the step: every component of ẑ moves, so it is at least the x-part's norm,
      // and the x-shadow of the unit-normalized step must satisfy dxᵀQdx ≤ 1.
      const zx = x.map((v) => v / sf.beta);
      const dzx = dx.map((v) => v / sf.beta);
      const lower = Math.hypot(...dzx.map((v, j) => v / zx[j]));
      expect(lower).toBeGreaterThan(0);
      // The full normalized step has ‖Z⁻¹dz‖ = 1, so its x-shadow d satisfies dᵀQd ≤ 1 — and
      // d = dx/‖Z⁻¹dz‖ with ‖Z⁻¹dz‖ ≥ lower, so dxᵀQdx/lower² ≥ dᵀQd; check the exact bound
      // with the full norm reconstructed from the slacks below.
      const hs = halfSpaces(p).filter((h) => h.kind === 'row');
      const tNext = r.trace[k + 1].info.artificial as number;
      const r0 = sf.b.map((v, i) => v - sf.A[i].reduce((a, b) => a + b, 0));
      let norm2 = dzx.reduce((a, v, j) => a + (v / zx[j]) ** 2, 0);
      hs.forEach((h, i) => {
        const slack = (xx: readonly number[], tt: number) =>
          (h.b - h.a[0] * xx[0] - h.a[1] * xx[1]) / (sf.beta * sf.col[2 + i]) - r0[i] * tt;
        const s0 = slack(x, t);
        const s1 = slack(r.trace[k + 1].x as number[], tNext);
        norm2 += ((s1 - s0) / s0) ** 2;
      });
      norm2 += ((tNext - t) / t) ** 2;
      const d = dx.map((v) => v / Math.sqrt(norm2));
      expect(quad(Q, d)).toBeLessThanOrEqual(1 + 1e-9);
      checked++;
    }
    expect(checked).toBeGreaterThan(3);
  });

  it('is null outside the interior', () => {
    expect(affineScalingMetric(lp('wyndor'), [-1, 2], 0)).toBeNull();
    expect(affineScalingMetric(lp('wyndor'), [4, 2], 0)).toBeNull();
  });
});
