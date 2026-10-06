/**
 * Nonlinear least squares in data space: fit y = a·e^(−bt) to 15 noisy samples. Each frame draws
 * the model at the current Levenberg–Marquardt and Gauss–Newton iterates (earlier fits fade).
 */
import '../../../../methods/unconstrained/least_squares';
import '../../../../problems/least_squares';
import { runMethod } from '../../../../core/registry';
import { int } from '../../../../core/format';
import { getProblem } from '../../../../problems/registry';
import type { LeastSquaresProblem } from '../../../../problems/least_squares';
import type { Preview } from '../types';
import { curve, dot, fitAxes, line, linClock, S, stepAt } from '../draw';

export default function build(): Preview {
  const p = getProblem<LeastSquaresProblem>('exp_decay_fit');
  const t = p.extra.t ?? [];
  const y = p.extra.y ?? [];
  const runs = [
    { id: 'levenberg_marquardt', label: 'Levenberg–Marquardt', slot: 0 },
    { id: 'gauss_newton', label: 'Gauss–Newton', slot: 1 },
  ].map((m) => {
    const r = runMethod(m.id, p, {});
    return { ...m, r, xs: r.trace.map((s) => s.x as [number, number]) };
  });
  const K = Math.max(...runs.map((r) => r.xs.length)) - 1;
  const tr: [number, number] = [Math.min(...t) - 0.1, Math.max(...t) + 0.1];
  const yr: [number, number] = [Math.min(0, ...y) - 0.15, Math.max(...y) + 0.2];
  const fin = runs[0].xs[runs[0].xs.length - 1];

  return {
    lab: 'least-squares',
    title: 'Exponential decay fit $y = a\\,e^{-bt}$',
    caption:
      `15 samples of $2.5\\,e^{-1.3t}$ with noise $\\sigma =$ 0.05, from $(a, b) =$ (1, 0.3). ` +
      `Both methods stop when $\\|J^\\top\\mathbf{r}\\|_\\infty$ is at most the gradient tolerance: ` +
      `Levenberg–Marquardt after ${int(runs[0].r.nIter)} steps, Gauss–Newton after ` +
      `${int(runs[1].r.nIter)}, at $(a, b) =$ (${fin[0].toFixed(3)}, ${fin[1].toFixed(3)}).`,
    legend: runs.map((r) => ({ label: r.label, slot: r.slot, note: `${int(r.r.nIter)} steps` })),
    ariaLabel: `Fifteen decaying data points; the fitted exponential moves from a poor start onto the data in ${int(runs[0].r.nIter)} Levenberg–Marquardt steps.`,
    duration: 4.5,
    draw(ctx, size, c, u, hero) {
      const { width: w, height: h } = size;
      const s = S(hero);
      const pad = hero ? 20 : 12;
      const view = fitAxes(tr, yr, w, h, [pad, pad, pad, pad]);
      const k = linClock(u, K);
      const { i, f } = stepAt(k, K);
      line(ctx, [view.toPx(tr[0], 0), view.toPx(tr[1], 0)], c.axis, 1);
      for (const run of [...runs].reverse()) {
        const last = run.xs.length - 1;
        // Earlier fits, faint.
        for (let j = 0; j < Math.min(i, last); j++) {
          const [a, b] = run.xs[j];
          line(
            ctx,
            curve((x) => a * Math.exp(-b * x), tr[0], tr[1], view, 80),
            c.series[run.slot],
            1,
            {
              alpha: 0.18,
            },
          );
        }
        const j0 = Math.min(i, last),
          j1 = Math.min(i + 1, last);
        const a = run.xs[j0][0] + (run.xs[j1][0] - run.xs[j0][0]) * f;
        const b = run.xs[j0][1] + (run.xs[j1][1] - run.xs[j0][1]) * f;
        line(
          ctx,
          curve((x) => a * Math.exp(-b * x), tr[0], tr[1], view, 120),
          c.series[run.slot],
          2 * s,
          {
            halo: c.halo,
            dash: run.slot === 1 ? [6, 4] : [],
          },
        );
      }
      t.forEach((ti, j) => {
        const [px, py] = view.toPx(ti, y[j]);
        dot(ctx, px, py, 2.8 * s, c.text, c.halo);
      });
    },
  };
}
