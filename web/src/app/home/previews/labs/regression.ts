/**
 * Regression with two gross outliers: the least-squares line is pulled toward them; Huber's
 * iteratively reweighted fit starts from it and down-weights them step by step (faded points).
 */
import '../../../../methods/regression/methods';
import '../../../../problems/data';
import { runMethod } from '../../../../core/registry';
import { int } from '../../../../core/format';
import type { Dataset } from '../../../../core/types';
import { getProblem } from '../../../../problems/registry';
import type { Preview } from '../types';
import { dot, fitAxes, line, linClock, S, stepAt } from '../draw';

export default function build(): Preview {
  const d = getProblem<Dataset>('outliers_linear');
  const ls = runMethod('linear_regression', d, {});
  const hu = runMethod('huber_regression', d, {});
  const lsBeta = ls.x as number[];
  const betas = hu.trace.map((s) => s.x as number[]);
  const weights = hu.trace.map((s) => (s.info.weights as number[] | null) ?? d.x.map(() => 1));
  const K = betas.length - 1;
  const xr: [number, number] = [Math.min(...d.x) - 0.4, Math.max(...d.x) + 0.4];
  const yr: [number, number] = [Math.min(...d.y) - 1.2, Math.max(...d.y) + 1.2];
  const fin = betas[K];
  // The fitted line as TeX: ŷ = a + bx, with the sign of b as the operator.
  const fitTex = ([a, b]: readonly number[]) =>
    `\\hat{y} = ${a.toFixed(2)} ${b < 0 ? '-' : '+'} ${Math.abs(b).toFixed(2)}\\,x`;

  return {
    lab: 'regression',
    title: 'A line with two gross outliers',
    caption:
      `20 points from $y = 1 + 2x + \\text{noise}$, two of them moved far off. Least squares: ` +
      `$${fitTex(lsBeta)}$. Huber (IRLS, ${int(hu.nIter)} reweighting steps from the ` +
      `least-squares fit): $${fitTex(fin)}$.`,
    legend: [
      { label: 'Least squares', slot: 0, note: 'one QR solve' },
      { label: 'Huber (IRLS)', slot: 1, note: `${int(hu.nIter)} steps` },
    ],
    ariaLabel: `Twenty points with two outliers. The least-squares line is tilted by them; the Huber line down-weights them and follows the other points.`,
    duration: 4,
    draw(ctx, size, c, u, hero) {
      const { width: w, height: h } = size;
      const s = S(hero);
      const pad = hero ? 20 : 12;
      const view = fitAxes(xr, yr, w, h, [pad, pad, pad, pad]);
      const k = linClock(u, K);
      const { i, f } = stepAt(k, K);
      const j1 = Math.min(i + 1, K);
      const lineOf = (b0: number, b1: number) => [
        view.toPx(xr[0], b0 + b1 * xr[0]),
        view.toPx(xr[1], b0 + b1 * xr[1]),
      ];
      line(ctx, lineOf(lsBeta[0], lsBeta[1]), c.series[0], 1.8 * s, { halo: c.halo });
      const b0 = betas[i][0] + (betas[j1][0] - betas[i][0]) * f;
      const b1 = betas[i][1] + (betas[j1][1] - betas[i][1]) * f;
      line(ctx, lineOf(b0, b1), c.series[1], 2.2 * s, { halo: c.halo });
      const wt = weights[i];
      d.x.forEach((x, j) => {
        const [px, py] = view.toPx(x, d.y[j]);
        dot(ctx, px, py, 2.9 * s, c.text, c.halo, 0.25 + 0.75 * (wt[j] ?? 1));
      });
    },
  };
}
