/**
 * Line search: φ(α) = f(𝐱₀ + α𝐩) along the steepest-descent direction of the quadratic bowl, the
 * Goldstein wedge between φ(0) + cαφ′(0) and φ(0) + (1 − c)αφ′(0), and the trial steps the
 * Goldstein search takes (α = 1 overshoots the wedge, α = ½ lands inside it).
 */
import '../../../../methods/line_search/methods';
import '../../../../problems/unconstrained';
import { runMethod } from '../../../../core/registry';
import { int, sig } from '../../../../core/format';
import { texNum } from '../tex';
import type { Problem2D } from '../../../../core/types';
import { getProblem } from '../../../../problems/registry';
import type { Preview } from '../types';
import { curve, dot, fillAlpha, fitAxes, line, linClock, ring, S, stepAt } from '../draw';

export default function build(): Preview {
  const p = getProblem<Problem2D>('quadratic_bowl');
  const r = runMethod('goldstein', p, {});
  const x0 = (p.x0 ?? [0, 0]) as number[];
  const dir = r.extra.direction as number[];
  const phi0 = r.extra.phi0 as number;
  const dphi0 = r.extra.dphi0 as number;
  const cc = r.trace[0].info.c1 as number;
  const phi = (a: number) => p.f([x0[0] + a * dir[0], x0[1] + a * dir[1]]);
  const trials = r.trace.slice(1).map((s) => ({
    a: s.info.alpha as number,
    v: s.info.phi as number,
    ok: Boolean(s.info.accepted),
  }));
  const K = trials.length;
  const aMax = Math.max(1.12, ...trials.map((t) => t.a * 1.08));
  const ys = Array.from({ length: 61 }, (_, i) => phi((aMax * i) / 60));
  const yr: [number, number] = [Math.min(...ys, 0) - 1.5, Math.max(...ys, phi0) + 1.5];
  const accepted = trials.find((t) => t.ok);

  return {
    lab: 'line-search',
    title: '$\\varphi(\\alpha) = f(\\mathbf{x}_0 + \\alpha\\mathbf{p})$ on the quadratic bowl',
    caption:
      `Steepest descent from $\\mathbf{x}_0 =$ (−2, 2). Goldstein accepts a step inside the ` +
      `wedge $${texNum(cc, 2)}\\,\\alpha\\varphi'(0) \\ge \\varphi(\\alpha) - \\varphi(0) \\ge ` +
      `${texNum(1 - cc, 2)}\\,\\alpha\\varphi'(0)$: $\\alpha =$ 1 overshoots it, ` +
      (accepted
        ? `$\\alpha =$ ${sig(accepted.a, 3)} is accepted after ${int(K)} trials.`
        : `no step is accepted in ${int(K)} trials.`),
    legend: [{ label: 'Goldstein', slot: 0, note: `${int(K)} trials` }],
    ariaLabel: `The function φ along the descent direction with the Goldstein wedge; the trial α = 1 is rejected and α = ${accepted ? sig(accepted.a, 3) : '—'} is accepted.`,
    duration: 4,
    draw(ctx, size, c, u, hero) {
      const { width: w, height: h } = size;
      const s = S(hero);
      const pad = hero ? 20 : 12;
      const view = fitAxes([0, aMax], yr, w, h, [pad, pad, pad, pad]);
      const k = linClock(u, K);
      const { i } = stepAt(k, K);
      const upper = (a: number) => phi0 + cc * a * dphi0;
      const lower = (a: number) => phi0 + (1 - cc) * a * dphi0;

      // The wedge of acceptable steps.
      fillAlpha(ctx, c.text, c.mode === 'dark' ? 0.08 : 0.05, () => {
        ctx.beginPath();
        const [ax, ay] = view.toPx(0, phi0);
        ctx.moveTo(ax, ay);
        const [ux, uy] = view.toPx(aMax, upper(aMax));
        const [lx, ly] = view.toPx(aMax, lower(aMax));
        ctx.lineTo(ux, uy);
        ctx.lineTo(lx, ly);
        ctx.closePath();
        ctx.fill();
      });
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, w, h);
      ctx.clip();
      line(ctx, [view.toPx(0, phi0), view.toPx(aMax, upper(aMax))], c.text3, 1.1 * s, {
        dash: [5, 4],
      });
      line(ctx, [view.toPx(0, phi0), view.toPx(aMax, lower(aMax))], c.text3, 1.1 * s, {
        dash: [5, 4],
      });
      ctx.restore();
      line(ctx, [view.toPx(0, 0), view.toPx(aMax, 0)], c.axis, 1);
      line(ctx, curve(phi, 0, aMax, view), c.text2, 1.7 * s, { halo: c.halo });
      const [ox, oy] = view.toPx(0, phi0);
      dot(ctx, ox, oy, 2.6 * s, c.text, c.halo);

      for (let j = 0; j < Math.min(K, Math.floor(k) + (k > 0 ? 1 : 0)); j++) {
        const t = trials[j];
        const alpha = j < i ? 1 : Math.min(1, (k - j) * 1.6);
        if (alpha <= 0) continue;
        const [tx, ty] = view.toPx(t.a, t.v);
        const [, axisY] = view.toPx(t.a, 0);
        line(
          ctx,
          [
            [tx, axisY],
            [tx, ty],
          ],
          c.series[0],
          1,
          { dash: [2, 3], alpha },
        );
        if (t.ok) dot(ctx, tx, ty, 3.6 * s, c.series[0], c.halo, alpha);
        else ring(ctx, tx, ty, 3.6 * s, c.series[0], c.halo, 1.6);
      }
    },
  };
}
