/** Geometry and wording helpers of the LP lab. */
import { describe, expect, it } from 'vitest';
import '../../problems/lp';
import { getProblem } from '../../problems/registry';
import type { LinearProgram } from '../../core/types';
import { simplex } from '../../methods/lp/simplex';
import {
  centralPath,
  dikinMatrix,
  edgeDirection,
  feasiblePolygon,
  halfSpaces,
  interiorPoint,
  latticePoints,
  lineInBox,
  minCost,
  polytope3,
  vertices2D,
  type TableauInfo,
} from './geometry';
import { boundaryOf, cutTex, fmt, liveRule, texNum, varTex } from './explain';

const lp = (id: string) => getProblem<LinearProgram>(id);
const BOX: [[number, number], [number, number]] = [
  [-100, 100],
  [-100, 100],
];

describe('2-D feasible set', () => {
  it('Wyndor: the polygon has the five textbook vertices', () => {
    const v = vertices2D(halfSpaces(lp('wyndor')))
      .map((p) => p.map((x) => Math.round(x * 1e9) / 1e9 + 0).join(','))
      .sort();
    expect(v).toEqual(['0,0', '0,6', '2,6', '4,0', '4,3'].sort());
    expect(feasiblePolygon(halfSpaces(lp('wyndor')), BOX)).toHaveLength(5);
  });
  it('an infeasible LP has an empty polygon', () => {
    expect(feasiblePolygon(halfSpaces(lp('infeasible_2d')), BOX)).toHaveLength(0);
  });
  it('lineInBox clips a line to the box', () => {
    const seg = lineInBox([1, 1], 4, [
      [0, 5],
      [0, 5],
    ])!;
    expect(seg.map((p) => p[0] + p[1])).toEqual([4, 4]);
  });
  it('lattice points split by feasibility', () => {
    const { feasible } = latticePoints(halfSpaces(lp('ilp_knapsack_like_2d')), [
      [0, 6.5],
      [0, 6.5],
    ]);
    expect(feasible.some(([a, b]) => a === 5 && b === 0)).toBe(true);
    expect(feasible.some(([a, b]) => a === 4 && b === 2)).toBe(false); // 9·4 + 5·2 = 46 > 45
  });
});

describe('interior geometry', () => {
  it('the central path ends at the optimum and starts at an interior point', () => {
    const hs = halfSpaces(lp('wyndor'));
    const poly = feasiblePolygon(hs, BOX);
    const path = centralPath(hs, minCost(lp('wyndor')), interiorPoint(poly, hs)!);
    expect(path.length).toBeGreaterThan(20);
    const end = path[path.length - 1];
    expect(Math.hypot(end[0] - 2, end[1] - 6)).toBeLessThan(1e-3);
  });
  it('the Dikin ellipsoid lies inside the polygon', () => {
    const hs = halfSpaces(lp('wyndor'));
    const x = [1, 2];
    const Q = dikinMatrix(hs, x)!;
    // Every constraint: max over the ellipse of aᵀd = sqrt(aᵀQ⁻¹a) ≤ slack.
    const det = Q[0][0] * Q[1][1] - Q[0][1] * Q[1][0];
    const inv = [
      [Q[1][1] / det, -Q[0][1] / det],
      [-Q[1][0] / det, Q[0][0] / det],
    ];
    for (const h of hs) {
      const reach = Math.sqrt(
        h.a[0] * (inv[0][0] * h.a[0] + inv[0][1] * h.a[1]) +
          h.a[1] * (inv[1][0] * h.a[0] + inv[1][1] * h.a[1]),
      );
      expect(reach).toBeLessThanOrEqual(h.b - (h.a[0] * x[0] + h.a[1] * x[1]) + 1e-12);
    }
    expect(dikinMatrix(hs, [5, 5])).toBeNull();
  });
});

describe('simplex edge read off the tableau', () => {
  it('the edge of each Wyndor pivot leads to the next vertex', () => {
    const r = simplex(lp('wyndor'));
    for (let k = 0; k < r.trace.length - 1; k++) {
      const info = r.trace[k].info as unknown as TableauInfo;
      const eta = edgeDirection(info, info.entering as number, 2);
      const theta = r.trace[k + 1].stepSize as number;
      const x = r.trace[k].x as number[];
      const next = r.trace[k + 1].x as number[];
      expect(x[0] + theta * eta[0]).toBeCloseTo(next[0], 12);
      expect(x[1] + theta * eta[1]).toBeCloseTo(next[1], 12);
    }
  });
});

describe('3-D polytope', () => {
  it('the Klee–Minty cube has 8 vertices, 12 edges and 6 faces', () => {
    const p = polytope3(halfSpaces(lp('klee_minty_3')));
    expect([p.vertices.length, p.edges.length, p.faces.length]).toEqual([8, 12, 6]);
  });
});

describe('wording', () => {
  it('labels and numbers', () => {
    expect(varTex('x12')).toBe('x_{12}');
    expect(varTex('zM')).toBe('z_{M}');
    expect(texNum(10000)).toBe('10000');
    expect(texNum(1.5e-7)).toBe('1.5 \\times 10^{-7}');
    expect(fmt(10000)).toBe('10000');
    expect(fmt(-2.5)).toBe('−2.5');
    expect(cutTex({ coef: [3, 2], rhs: 15 })).toBe('3x_{1} + 2x_{2} \\le 15');
  });
  it('boundaries of tableau variables', () => {
    expect(boundaryOf('x2', lp('wyndor'))).toEqual({ kind: 'axis', j: 1 });
    expect(boundaryOf('s3', lp('wyndor'))).toMatchObject({ kind: 'row', index: 2, b: 18 });
    expect(boundaryOf('g1', lp('wyndor'))).toEqual({ kind: 'cut', index: 0 });
  });
  it('the live rule names the entering and leaving variables', () => {
    const r = simplex(lp('wyndor'));
    const tex = liveRule('simplex', r.trace[0], undefined, lp('wyndor'))!;
    expect(tex).toContain('x_{2}\\ \\text{enters}');
    expect(tex).toContain('s_{2}\\ \\text{leaves}');
    expect(liveRule('simplex', r.trace[2], r.trace[1], lp('wyndor'), 'optimal')).toContain(
      'optimal',
    );
  });
});
