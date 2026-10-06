/**
 * Root finding on Wallis' cubic f(x) = x³ − 2x − 5: bisection halves its bracket (the ladder of
 * brackets under the axis), Newton follows its tangents down to the axis. One clock, linear in k.
 */
import '../../../../methods/roots/bracketing';
import '../../../../methods/roots/open';
import '../../../../problems/roots';
import { getMethod, runMethod } from '../../../../core/registry';
import { int } from '../../../../core/format';
import type { Problem } from '../../../../core/types';
import { getProblem } from '../../../../problems/registry';
import type { Preview } from '../types';
import { cross, curve, dot, fillAlpha, fitAxes, line, linClock, ring, S, stepAt } from '../draw';

const NEWTON_X0 = 3.2;
const RUNGS = 10;
const XR: [number, number] = [1.55, 3.35];

export default function build(): Preview {
  const p = getProblem<Problem<number>>('cubic');
  const f = (x: number) => p.f(x) as number;
  const bis = runMethod('bisection', p, {});
  const newton = runMethod('newton', p, { x0: NEWTON_X0 });
  const brackets = bis.trace.map((s) => s.info.bracket as [number, number]);
  const nx = newton.trace.map((s) => s.x as number);
  const root = (p.roots?.[0] as number) ?? nx[nx.length - 1];
  const K = RUNGS;
  // Registry names: one name per method across the site.
  const [bisName, newtonName] = ['bisection', 'newton'].map((id) => getMethod(id).spec.name);

  return {
    lab: 'roots',
    title: 'Wallis’ cubic $f(x) = x^3 - 2x - 5$',
    caption:
      `Bisection from [2, 3] halves the bracket: ${int(bis.nIter)} steps to a half-width ` +
      `$\\le 10^{-10}$ (the first ${RUNGS} brackets are drawn). ${newtonName} from ` +
      `$x_0 =$ ${NEWTON_X0} converges in ${int(newton.nIter)} steps. Both reach ` +
      `$x^\\star \\approx$ 2.0946.`,
    legend: [
      { label: bisName, slot: 0, note: `${int(bis.nIter)} steps` },
      { label: newtonName, slot: 1, note: `${int(newton.nIter)} steps` },
    ],
    ariaLabel:
      `The cubic x³ − 2x − 5 crossing zero near 2.0946. Bisection's brackets shrink from [2, 3] ` +
      `over ${int(bis.nIter)} steps; the ${newtonName} tangents from x = ${NEWTON_X0} reach the root in ${int(newton.nIter)} steps.`,
    duration: 5,
    draw(ctx, size, c, u, hero) {
      const { width: w, height: h } = size;
      const s = S(hero);
      const ladderH = Math.round(h * 0.3);
      const pad = hero ? 20 : 12;
      const view = fitAxes(XR, [-6, 24], w, h, [pad, pad, pad, ladderH + pad * 0.5]);
      const k = linClock(u, K);
      const { i } = stepAt(k, K);
      const [, y0] = view.toPx(0, 0);
      const [, yTop] = view.toPx(0, 24);
      const [, yBot] = view.toPx(0, -6);

      // Current bracket: a soft band behind the curve.
      const br = brackets[Math.min(i, brackets.length - 1)];
      const [ba] = view.toPx(br[0], 0);
      const [bb] = view.toPx(br[1], 0);
      fillAlpha(ctx, c.series[0], c.mode === 'dark' ? 0.16 : 0.1, () =>
        ctx.fillRect(ba, yTop, bb - ba, yBot - yTop),
      );

      // Axis and curve.
      line(ctx, [view.toPx(XR[0], 0), view.toPx(XR[1], 0)], c.axis, 1);
      line(ctx, curve(f, XR[0], XR[1], view), c.text2, 1.6 * s, { halo: c.halo });

      // Newton: tangent to the axis, then up to the curve.
      const shown = Math.min(nx.length - 1, Math.floor(k));
      for (let j = 1; j <= shown; j++) {
        const a = nx[j - 1],
          b = nx[j];
        const pa = view.toPx(a, f(a));
        const pb = view.toPx(b, 0);
        line(ctx, [pa, pb], c.series[1], 1.5 * s, { halo: c.halo });
        line(ctx, [pb, view.toPx(b, f(b))], c.series[1], 1, { dash: [2, 3], alpha: 0.8 });
        dot(ctx, pa[0], pa[1], 2.6 * s, c.series[1], c.halo);
      }
      const head = nx[Math.min(shown, nx.length - 1)];
      const ph = view.toPx(head, f(head));
      dot(ctx, ph[0], ph[1], 3.2 * s, c.series[1], c.halo);

      // The bracket ladder: one rung per step, narrowing to the root.
      const rungGap = ladderH / (RUNGS + 1.5);
      const top = h - ladderH;
      for (let r = 0; r <= Math.min(i, RUNGS - 1); r++) {
        const [a, b] = brackets[r];
        const [pa] = view.toPx(a, 0);
        const [pb] = view.toPx(b, 0);
        const y = top + rungGap * (r + 0.75);
        line(
          ctx,
          [
            [pa, y],
            [Math.max(pb, pa + 1.5), y],
          ],
          c.series[0],
          2 * s,
        );
        if (r === Math.min(i, RUNGS - 1)) {
          ring(ctx, pa, y, 2.2 * s, c.series[0]);
          ring(ctx, pb, y, 2.2 * s, c.series[0]);
        }
      }
      const [rx] = view.toPx(root, 0);
      cross(ctx, rx, y0, hero ? 6 : 5, c);
    },
  };
}
