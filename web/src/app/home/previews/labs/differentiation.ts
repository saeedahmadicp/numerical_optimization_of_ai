/**
 * Numerical differentiation: the error of forward and central differences for f′(0) of e^x as the
 * step h halves 30 times, on log–log axes. Truncation error falls with h until round-off takes
 * over: the V-shaped trade-off.
 */
import '../../../../methods/differentiation/methods';
import '../../../../problems/calculus';
import { runMethod } from '../../../../core/registry';
import { sci } from '../../../../core/format';
import { texSci } from '../tex';
import type { Problem } from '../../../../core/types';
import { getProblem } from '../../../../problems/registry';
import type { Preview } from '../types';
import { dot, fitAxes, line, linClock, S, type Pt } from '../draw';

const FLOOR = 1e-16;

export default function build(): Preview {
  const p = getProblem<Problem<number>>('exp_0_1');
  const runs = [
    { id: 'forward_difference', label: 'Forward difference', slot: 0 },
    { id: 'central_difference', label: 'Central difference', slot: 1 },
  ].map((m) => {
    const r = runMethod(m.id, p, {});
    const pts = r.trace.map(
      (s) =>
        [
          Math.log10(s.info.h as number),
          Math.log10(Math.max(FLOOR, Math.abs(s.info.error as number))),
        ] as const,
    );
    const best = Math.min(...r.trace.map((s) => Math.abs(s.info.error as number)));
    return { ...m, r, pts, best };
  });
  const K = Math.max(...runs.map((r) => r.pts.length)) - 1;
  const all = runs.flatMap((r) => r.pts);
  const xr: [number, number] = [
    Math.min(...all.map((q) => q[0])) - 0.3,
    Math.max(...all.map((q) => q[0])) + 0.3,
  ];
  const yr: [number, number] = [
    Math.min(...all.map((q) => q[1])) - 0.6,
    Math.max(...all.map((q) => q[1])) + 0.6,
  ];

  return {
    lab: 'differentiation',
    title: "Error of $f'(0)$ for $f = e^x$ against the step $h$",
    caption:
      `$h =$ 0.1 halved 30 times, log–log axes. Truncation error falls as $h$ (forward) and ` +
      `$h^2$ (central) until round-off, about $\\varepsilon/h$, takes over. Smallest errors: ` +
      `forward $${texSci(runs[0].best, 2)}$, central $${texSci(runs[1].best, 2)}$.`,
    legend: runs.map((r) => ({ label: r.label, slot: r.slot, note: `best ${sci(r.best, 2)}` })),
    ariaLabel: `Two V-shaped error curves on log–log axes; the smallest errors are ${sci(runs[0].best, 2)} (forward) and ${sci(runs[1].best, 2)} (central).`,
    duration: 4.5,
    draw(ctx, size, c, u, hero) {
      const { width: w, height: h } = size;
      const s = S(hero);
      const pad = hero ? 22 : 14;
      // Small h on the left: the sweep runs right to left, as h shrinks.
      const view = fitAxes(xr, yr, w, h, [pad, pad, pad, pad]);
      for (let e = Math.ceil(yr[0] / 4) * 4; e <= yr[1]; e += 4)
        line(ctx, [view.toPx(xr[0], e), view.toPx(xr[1], e)], c.grid, 1);
      for (let e = Math.ceil(xr[0] / 3) * 3; e <= xr[1]; e += 3)
        line(ctx, [view.toPx(e, yr[0]), view.toPx(e, yr[1])], c.grid, 1);
      const k = linClock(u, K);
      for (const run of runs) {
        const n = Math.min(run.pts.length - 1, Math.floor(k));
        const fr = Math.min(1, k - n);
        const pts: Pt[] = run.pts.slice(0, n + 1).map((q) => view.toPx(q[0], q[1]));
        if (n + 1 < run.pts.length && fr > 0) {
          const a = run.pts[n],
            b = run.pts[n + 1];
          pts.push(view.toPx(a[0] + (b[0] - a[0]) * fr, a[1] + (b[1] - a[1]) * fr));
        }
        line(ctx, pts, c.series[run.slot], 1.8 * s, { halo: c.halo });
        run.pts.slice(0, n + 1).forEach((q) => {
          const [x, y] = view.toPx(q[0], q[1]);
          dot(ctx, x, y, 2 * s, c.series[run.slot], c.halo);
        });
      }
    },
  };
}
