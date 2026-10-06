/**
 * Stochastic gradients: least-squares regression over 200 samples, minibatch SGD and stochastic
 * Adam from the same start and seed. The noisy paths are the recorded iterates; the level sets are
 * the full loss f(w).
 */
import '../../../../methods/stochastic/methods';
import '../../../../problems/stochastic';
import { runMethod } from '../../../../core/registry';
import { int } from '../../../../core/format';
import { getProblem } from '../../../../problems/registry';
import type { FiniteSumProblem } from '../../../../problems/stochastic';
import { drawPathLayer } from '../../../../viz/PathLayer';
import type { Preview } from '../types';
import { cross, drawField, fitEqual, linClock, ring, S, type Box } from '../draw';

export default function build(): Preview {
  const p = getProblem<FiniteSumProblem>('linreg_2d');
  const runs = [
    { id: 'sgd', label: 'SGD', slot: 0, width: 1.4 },
    { id: 'stochastic_adam', label: 'Adam', slot: 1, width: 1.8 },
  ].map((m) => {
    const r = runMethod(m.id, p, {});
    return { ...m, r, xs: r.trace.map((s) => s.x as [number, number]) };
  });
  const K = Math.max(...runs.map((r) => r.xs.length)) - 1;
  const BOX: Box = [
    [-2.5, 1.8],
    [-1.4, 2.5],
  ];
  const f = (x: number, y: number) => p.f([x, y]);
  const m = p.minima[0];
  const epochs = runs[0].r.extra.epochs_run as number;

  return {
    lab: 'stochastic',
    title: 'Linear regression, 200 samples',
    caption:
      `Same start $\\mathbf{w}_0 =$ (−2, −1), same seed, ${int(epochs)} epochs of minibatches ` +
      `each. Neither reaches $\\|\\nabla f\\| \\le 10^{-6}$ in the budget; both settle near the ` +
      `least-squares solution $\\mathbf{w}^\\star$, ` +
      `jittering with the minibatch noise.`,
    legend: runs.map((r) => ({
      label: r.label,
      slot: r.slot,
      note: `${int(r.r.nIter)} updates`,
    })),
    ariaLabel: `Two noisy paths over the level sets of a least-squares loss, from (−2, −1) toward the minimizer (${m[0].toFixed(2)}, ${m[1].toFixed(2)}).`,
    duration: 5,
    draw(ctx, size, c, u, hero) {
      const { width: w, height: h } = size;
      const s = S(hero);
      const view = fitEqual(BOX, w, h, hero ? 16 : 8);
      drawField(ctx, 'linreg_2d', f, view, size, c, { levels: 12 });
      const [mx, my] = view.toPx(m[0], m[1]);
      cross(ctx, mx, my, hero ? 6 : 5, c);
      drawPathLayer(
        ctx,
        runs.map((r) => ({
          points: r.xs,
          color: c.series[r.slot],
          dots: false,
          width: r.width * s,
          start: false,
          quiet: u >= 1,
          milestones: false,
          end: r.r.converged ? ('converged' as const) : ('stopped' as const),
        })),
        { t: linClock(u, K), toPx: view.toPx, halo: c.halo, trail: 1e9, ease: false },
      );
      const st = runs[0].xs[0];
      const [sx, sy] = view.toPx(st[0], st[1]);
      ring(ctx, sx, sy, hero ? 5 : 4, c.text, c.halo);
    },
  };
}
