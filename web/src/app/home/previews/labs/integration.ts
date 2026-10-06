/**
 * Quadrature: the composite midpoint rule on ∫₋₂² e^(−x²) dx with 4, 8 and 16 panels — each level
 * of the method's own trace, rectangles at the midpoints it evaluated.
 */
import '../../../../methods/integration/methods';
import '../../../../problems/calculus';
import { runMethod } from '../../../../core/registry';
import { sci } from '../../../../core/format';
import { texSci } from '../tex';
import type { Problem } from '../../../../core/types';
import { getProblem } from '../../../../problems/registry';
import type { Preview } from '../types';
import { curve, fillAlpha, fitAxes, line, linClock, S, stepAt } from '../draw';

const LEVELS = 3;

export default function build(): Preview {
  const p = getProblem<Problem<number>>('gaussian');
  const f = (x: number) => p.f(x) as number;
  const r = runMethod('midpoint_rule', p, {});
  const levels = r.trace.slice(0, LEVELS).map((s) => ({
    panels: s.info.panels as [number, number][],
    nodes: s.info.nodes as [number, number][],
    n: s.info.n_panels as number,
    error: s.info.error as number,
  }));
  const [a, b] = p.domain as [number, number];
  const K = levels.length - 1;

  return {
    lab: 'integration',
    title: '$\\int e^{-x^2}\\,dx$ on [−2, 2]',
    caption:
      `The composite midpoint rule with ` +
      levels.map((l) => `${l.n} panels (error $${texSci(Math.abs(l.error), 2)}$)`).join(', ') +
      `. Each doubling of the panels cuts the error by roughly four: the rule is second order.`,
    legend: [{ label: 'Midpoint rule', slot: 0, note: `${levels[K].n} panels` }],
    ariaLabel: `The bell curve e^(−x²) on [−2, 2] filled with midpoint rectangles; with ${levels[K].n} panels the error is ${sci(Math.abs(levels[K].error), 2)}.`,
    duration: 4,
    draw(ctx, size, c, u, hero) {
      const { width: w, height: h } = size;
      const s = S(hero);
      const pad = hero ? 20 : 12;
      const view = fitAxes([a - 0.08, b + 0.08], [0, 1.08], w, h, [pad, pad, pad, pad]);
      const k = linClock(u, K);
      const { i, f: fr } = stepAt(k, K);
      const paint = (lv: (typeof levels)[number], alpha: number) => {
        if (alpha <= 0) return;
        lv.panels.forEach(([x0, x1], j) => {
          const [px0, py] = view.toPx(x0, lv.nodes[j][1]);
          const [px1, base] = view.toPx(x1, 0);
          fillAlpha(ctx, c.series[0], (c.mode === 'dark' ? 0.26 : 0.16) * alpha, () =>
            ctx.fillRect(px0, py, px1 - px0, base - py),
          );
          ctx.save();
          ctx.globalAlpha = alpha;
          ctx.strokeStyle = c.series[0];
          ctx.lineWidth = 1;
          ctx.strokeRect(Math.round(px0) + 0.5, py, Math.round(px1 - px0), base - py);
          ctx.restore();
        });
      };
      paint(levels[i], 1 - fr);
      if (fr > 0) paint(levels[i + 1], fr);
      line(ctx, [view.toPx(a - 0.08, 0), view.toPx(b + 0.08, 0)], c.axis, 1);
      line(ctx, curve(f, a - 0.08, b + 0.08, view), c.text2, 1.7 * s, { halo: c.halo });
    },
  };
}
