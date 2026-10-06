/**
 * The step lens's support code: the fast loss evaluator and its level sets (lensField.ts), the
 * label layout (lensLabels.ts), the lens's labels and aria text, and the exact zero of βv₀.
 */
import { describe, expect, it } from 'vitest';
import { getMethod, defaults } from '../../src/core/registry';
import { getProblem } from '../../src/problems/registry';
import type { FiniteSumProblem } from '../../src/problems/stochastic';
import '../../src/methods/stochastic/methods';
import '../../src/problems/stochastic';
import { momentumPush, stepGeometry } from '../../src/labs/stochastic/geometry';
import { stepTex } from '../../src/labs/stochastic/equation';
import { stepLabels } from '../../src/labs/stochastic/overlays';
import { levelSets, lossEvaluator, quadraticModel } from '../../src/labs/stochastic/lensField';
import {
  arrowCandidates,
  boxOf,
  overlaps,
  placeLabels,
  pointCandidates,
  segmentHitsRect,
} from '../../src/labs/stochastic/lensLabels';
import { lensAria, lensBox } from '../../src/labs/stochastic/lensModel';

const P = (id: string) => getProblem<FiniteSumProblem>(id);
function run(method: string, problem: string, params: Record<string, unknown> = {}) {
  const { spec, fn } = getMethod(method);
  return fn(P(problem), { ...defaults(spec), ...(params as Record<string, never>) });
}
const IDS = ['linreg_2d', 'logreg_2d', 'ill_conditioned_ls', 'huber_regression_2d'];

describe('lens level sets', () => {
  for (const id of IDS) {
    it(`${id}: the fast evaluator equals problem.f`, () => {
      const p = P(id);
      const { f, quadratic } = lossEvaluator(p);
      expect(quadratic).toBe(p.loss === 'squared');
      const [[a, b], [c, d]] = p.domain;
      for (let t = 0; t < 50; t++) {
        const x = a + (b - a) * ((t * 0.618) % 1),
          y = c + (d - c) * ((t * 0.382 + 0.1) % 1);
        const want = p.f([x, y]);
        expect(Math.abs(f(x, y) - want)).toBeLessThan(1e-11 * (1 + Math.abs(want)));
      }
    });
  }

  it('the quadratic is centred at the reference minimizer, with f(𝐰_c) = f⋆', () => {
    for (const id of ['linreg_2d', 'ill_conditioned_ls']) {
      const p = P(id);
      const q = quadraticModel(p)!;
      expect(Math.abs(q.center[0] - p.minima[0][0])).toBeLessThan(1e-9);
      expect(Math.abs(q.center[1] - p.minima[0][1])).toBeLessThan(1e-9);
      expect(Math.abs(q.fc - p.extra.f_min)).toBeLessThan(1e-12);
    }
    expect(quadraticModel(P('logreg_2d'))).toBeNull();
  });

  it('level-set segments lie on f = level (one grid, two levels)', () => {
    for (const id of IDS) {
      const p = P(id);
      const { f } = lossEvaluator(p);
      const box = { cx: p.minima[0][0], cy: p.minima[0][1], half: 0.5 };
      const l1 = f(box.cx + 0.2, box.cy),
        l2 = f(box.cx + 0.3, box.cy - 0.1);
      const sets = levelSets(f, box, [l1, l2], 49);
      expect(sets).toHaveLength(2);
      sets.forEach((seg, i) => {
        const level = i === 0 ? l1 : l2;
        expect(seg.length).toBeGreaterThan(8);
        // Linear interpolation on a 49 × 49 grid of a smooth f: tight relative to the range.
        const range = f(box.cx + 0.5, box.cy + 0.5) - p.extra.f_min;
        for (let k = 0; k < seg.length; k += 2)
          expect(Math.abs(f(seg[k], seg[k + 1]) - level)).toBeLessThan(0.01 * range);
      });
    }
  });

  it('a 1 µm step near the minimizer still gets distinct level sets (no cancellation)', () => {
    const p = P('linreg_2d');
    const { f } = lossEvaluator(p);
    const [cx, cy] = p.minima[0];
    const a = f(cx + 1e-6, cy + 2e-6),
      b = f(cx + 0.5e-6, cy + 1e-6);
    expect(a).toBeGreaterThan(b);
    const sets = levelSets(f, { cx, cy, half: 3e-6 }, [a, b], 49);
    expect(sets[0].length).toBeGreaterThan(8);
    expect(sets[1].length).toBeGreaterThan(8);
  });
});

describe('lens label layout', () => {
  it('segment–rectangle intersection', () => {
    const r = { x0: 0, y0: 0, x1: 10, y1: 10 };
    expect(segmentHitsRect([-5, 5, 15, 5], r)).toBe(true);
    expect(segmentHitsRect([-5, -5, -1, 20], r)).toBe(false);
    expect(segmentHitsRect([2, 2, 3, 3], r)).toBe(true);
    expect(segmentHitsRect([11, 0, 20, 10], r)).toBe(false);
  });

  it('labels never overlap, stay inside, avoid reserved corners; crowded ones are dropped', () => {
    const W = 196,
      H = 196;
    // Three arrows from one point (the crowded case of the review: Nesterov at k = 30).
    const o: [number, number] = [90, 110];
    const reqs = [
      { w: 26, h: 13, candidates: arrowCandidates(o, [150, 60]) },
      { w: 30, h: 13, candidates: arrowCandidates(o, [160, 95]) },
      { w: 18, h: 13, candidates: arrowCandidates(o, [120, 70]) },
      { w: 12, h: 13, candidates: pointCandidates([120, 70]) },
    ];
    const reserved = [
      { x0: 0, y0: 0, x1: 70, y1: 24 },
      { x0: 0, y0: H - 26, x1: 60, y1: H },
    ];
    const spots = placeLabels(reqs, {
      width: W,
      height: H,
      reserved,
      segments: [
        [o[0], o[1], 150, 60],
        [o[0], o[1], 160, 95],
        [o[0], o[1], 120, 70],
      ],
    });
    const boxes = spots.flatMap((c, i) => (c ? [boxOf(c, reqs[i].w, reqs[i].h)] : []));
    expect(boxes.length).toBeGreaterThanOrEqual(3);
    for (let i = 0; i < boxes.length; i++) {
      const b = boxes[i];
      expect(b.x0).toBeGreaterThanOrEqual(3);
      expect(b.x1).toBeLessThanOrEqual(W - 3);
      for (const r of reserved) expect(overlaps(b, r)).toBe(false);
      for (let j = i + 1; j < boxes.length; j++) expect(overlaps(b, boxes[j])).toBe(false);
    }
    // No room at all: dropped, not drawn over something else.
    const none = placeLabels([{ w: 300, h: 13, candidates: pointCandidates([50, 50]) }], {
      width: W,
      height: H,
    });
    expect(none).toEqual([null]);
  });

  it('the lens box leaves room around the marks for labels', () => {
    const s = run('sgd_nesterov', 'linreg_2d', { batch_size: 5, epochs: 1 }).trace[30];
    const g = stepGeometry(P('linreg_2d'), 'sgd_nesterov', s, 5)!;
    const box = lensBox(g);
    for (const p of [g.base, g.to, g.pushTo, g.evalAt, g.mean!]) {
      expect(Math.abs(p[0] - box.cx)).toBeLessThan(box.half * 0.75);
      expect(Math.abs(p[1] - box.cy)).toBeLessThan(box.half * 0.75);
    }
  });
});

describe('lens labels and text', () => {
  it('the step is labelled first; Nesterov adds βv and the look-ahead', () => {
    const s = run('sgd_nesterov', 'linreg_2d', { batch_size: 5, epochs: 1 }).trace[5];
    const g = stepGeometry(P('linreg_2d'), 'sgd_nesterov', s, 5)!;
    const labels = stepLabels(g, 0);
    expect(labels[0].kind).toBe('arrow');
    expect(labels.map((l) => l.runs.map((r) => r.t).join(''))).toEqual(['−ηg', '−η∇f', 'βv', 'w̃']);
  });

  it('the aria text names what is drawn: Adam has −∇f at the step length, no expected step', () => {
    const s = run('stochastic_adam', 'linreg_2d', { batch_size: 5, epochs: 1 }).trace[5];
    const g = stepGeometry(P('linreg_2d'), 'stochastic_adam', s, 5)!;
    const t = lensAria(g, 5);
    expect(t).toContain('−∇f drawn at the step’s length');
    expect(t).not.toContain('−η∇f');
    expect(t).not.toContain('ellipse');
    const s2 = run('sgd', 'linreg_2d', { batch_size: 5, epochs: 1 }).trace[5];
    const t2 = lensAria(stepGeometry(P('linreg_2d'), 'sgd', s2, 5)!, 5);
    expect(t2).toContain('full-gradient step −η∇f');
    expect(t2).toContain('2σ ellipse');
  });
});

describe('the momentum push βvₖ₋₁', () => {
  it('is exactly 0 at the first update (v₀ = 0), in the geometry and in the printed rule', () => {
    for (const id of ['sgd_momentum', 'sgd_nesterov']) {
      const r = run(id, 'linreg_2d', { batch_size: 5, epochs: 1 });
      const s = r.trace.find((t) => t.k === 1)!;
      const g = stepGeometry(P('linreg_2d'), id, s, 5)!;
      expect(g.pushTo).toEqual(g.base);
      const tex = stepTex(id, s)!;
      expect(tex).toContain(
        '\\underbrace{\\begin{pmatrix}0\\\\0\\end{pmatrix}}_{\\textstyle\\footnotesize \\beta\\mathbf{v}}',
      );
    }
  });

  it('drops rounding residue but keeps genuine components', () => {
    expect(momentumPush(5, [0.1, 0.2], [-1, -2], 0.1)).toEqual([0, 0]);
    const p = momentumPush(5, [0.1 + 1e-17, 0.25], [-1, -2], 0.1);
    expect(p[0]).toBe(0);
    expect(p[1]).toBeCloseTo(0.05, 15);
  });
});
