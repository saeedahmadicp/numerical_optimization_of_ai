/**
 * 1-D minimization on the tilted double well f(x) = x⁴ − 4x² + x, bracket [0.5, 2]: golden
 * section shrinks the bracket by 0.618 per step; Brent fits parabolas through three points.
 */
import '../../../../methods/scalar/methods';
import '../../../../problems/scalar_min';
import { runMethod } from '../../../../core/registry';
import { int } from '../../../../core/format';
import type { Problem } from '../../../../core/types';
import { getProblem } from '../../../../problems/registry';
import type { Preview } from '../types';
import { cross, curve, dot, fillAlpha, fitAxes, line, linClock, S, stepAt } from '../draw';

const XR: [number, number] = [0.35, 2.1];
const STEPS = 12;

interface Parabola {
  center: number;
  coef: [number, number, number];
}

export default function build(): Preview {
  const p = getProblem<Problem<number>>('quartic_1d');
  const f = (x: number) => p.f(x) as number;
  const gs = runMethod('golden_section', p, {});
  const br = runMethod('brent_minimize', p, {});
  const xmin = gs.x as number;
  const K = STEPS;
  const ys = Array.from({ length: 101 }, (_, i) => f(XR[0] + ((XR[1] - XR[0]) * i) / 100));
  const yr: [number, number] = [Math.min(...ys) - 0.4, Math.max(...ys) + 0.3];

  return {
    lab: 'scalar',
    title: 'Tilted double well $x^4 - 4x^2 + x$',
    caption:
      `Bracket [0.5, 2] around the local minimizer $x \\approx$ 1.347. Golden section keeps 0.618 ` +
      `of the bracket per step (${int(gs.nIter)} steps to a width $\\le 10^{-8}$; the first ` +
      `${STEPS} are drawn). ` +
      `Brent's parabolic steps stop after ${int(br.nIter)}.`,
    legend: [
      { label: 'Golden section', slot: 0, note: `${int(gs.nIter)} steps` },
      { label: 'Brent (localmin)', slot: 1, note: `${int(br.nIter)} steps` },
    ],
    ariaLabel: `A double-well curve. Golden section shrinks the bracket [0.5, 2] onto x ≈ 1.347 in ${int(gs.nIter)} steps; Brent's parabolas reach it in ${int(br.nIter)}.`,
    duration: 5,
    draw(ctx, size, c, u, hero) {
      const { width: w, height: h } = size;
      const s = S(hero);
      const pad = hero ? 20 : 12;
      const view = fitAxes(XR, yr, w, h, [pad, pad, pad, pad]);
      const k = linClock(u, K);
      const { i } = stepAt(k, K);
      const g = gs.trace[Math.min(i, gs.trace.length - 1)];
      const [a, bb] = g.info.bracket as [number, number];
      const [pa] = view.toPx(a, 0);
      const [pb] = view.toPx(bb, 0);
      fillAlpha(ctx, c.series[0], c.mode === 'dark' ? 0.16 : 0.1, () =>
        ctx.fillRect(pa, 0, pb - pa, h),
      );
      line(
        ctx,
        [
          [pa, 0],
          [pa, h],
        ],
        c.series[0],
        1,
        { alpha: 0.7 },
      );
      line(
        ctx,
        [
          [pb, 0],
          [pb, h],
        ],
        c.series[0],
        1,
        { alpha: 0.7 },
      );

      line(ctx, curve(f, XR[0], XR[1], view), c.text2, 1.6 * s, { halo: c.halo });

      // Brent's current parabola.
      const bs = br.trace[Math.min(i, br.trace.length - 1)];
      const par = bs.info.parabola as Parabola | null;
      if (par) {
        const q = (z: number) =>
          par.coef[0] + par.coef[1] * (z - par.center) + par.coef[2] * (z - par.center) ** 2;
        ctx.save();
        ctx.beginPath();
        ctx.rect(0, 0, w, h);
        ctx.clip();
        line(ctx, curve(q, XR[0], XR[1], view), c.series[1], 1.5 * s, { dash: [5, 4] });
        ctx.restore();
      }
      const [mx, my] = view.toPx(xmin, f(xmin));
      cross(ctx, mx, my, hero ? 7 : 6, c);
      // Every point golden section has evaluated so far.
      for (let j = 0; j <= Math.min(i, gs.trace.length - 1); j++) {
        for (const x of gs.trace[j].info.evaluated as number[]) {
          const [px, py] = view.toPx(x, f(x));
          dot(ctx, px, py, 2.4 * s, c.series[0], c.halo, j === i ? 1 : 0.55);
        }
      }
      const bx = bs.x as number;
      const [qx, qy] = view.toPx(bx, f(bx));
      dot(ctx, qx, qy, 3.2 * s, c.series[1], c.halo);
    },
  };
}
