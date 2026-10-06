/**
 * Runge's phenomenon: the Lagrange interpolant of 1/(1 + 25x²) through 11 equally spaced nodes
 * swings wildly near the ends; through 11 Chebyshev nodes it hugs the function. The nodes are
 * added one at a time, and each frame draws the interpolant the method has built so far.
 */
import '../../../../methods/interpolation/methods';
import '../../../../problems/data';
import { getMethod, runMethod } from '../../../../core/registry';
import type { Dataset, Result } from '../../../../core/types';
import { getProblem } from '../../../../problems/registry';
import type { Preview } from '../types';
import { dot, fitAxes, line, linClock, ring, S, stepAt, type Pt } from '../draw';

interface Eval {
  x: number[];
  y: number[];
  f_true: number[];
}

const fmt = (v: number) => v.toFixed(v < 0.995 ? 3 : 2);

export default function build(): Preview {
  const eq = runMethod('lagrange', getProblem<Dataset>('runge_equispaced'), {});
  const ch = runMethod('lagrange', getProblem<Dataset>('runge_chebyshev'), {});
  const ev = eq.extra.eval as Eval;
  const errEq = eq.extra.max_error as number;
  const errCh = ch.extra.max_error as number;
  const nodes = (r: Result) =>
    (r.extra.nodes as number[]).map((x, i) => [x, (r.extra.values as number[])[i]] as const);
  const runs = [
    { r: ch, slot: 0, nodes: nodes(ch), xs: (ch.extra.eval as Eval).x },
    { r: eq, slot: 1, nodes: nodes(eq), xs: ev.x },
  ];
  const K = Math.max(eq.trace.length, ch.trace.length) - 1;
  const n = (eq.extra.nodes as number[]).length;
  const yMax = Math.max(1.05, ...ev.y.filter(Number.isFinite));
  // The registry name (one name per method across the site), then the node set.
  const name = getMethod('lagrange').spec.name;

  return {
    lab: 'interpolation',
    title: 'Runge’s function $1/(1 + 25x^2)$',
    caption:
      `One polynomial of degree ${n - 1} through ${n} samples, built one node at a time. Equally ` +
      `spaced nodes: the largest error on [−1, 1] is ${fmt(errEq)}. Chebyshev nodes: ${fmt(errCh)}.`,
    legend: [
      { label: `${name}, Chebyshev nodes`, slot: 0, note: `max error ${fmt(errCh)}` },
      { label: `${name}, equally spaced nodes`, slot: 1, note: `max error ${fmt(errEq)}` },
    ],
    ariaLabel:
      `Runge's function with two degree-${n - 1} interpolants: through equally spaced nodes it ` +
      `oscillates near the ends (max error ${fmt(errEq)}); through Chebyshev nodes the max error is ${fmt(errCh)}.`,
    duration: 5,
    draw(ctx, size, c, u, hero) {
      const { width: w, height: h } = size;
      const s = S(hero);
      const pad = hero ? 20 : 12;
      const view = fitAxes([-1.04, 1.04], [-0.42, yMax + 0.05], w, h, [pad, pad, pad, pad]);
      const k = linClock(u, K);
      const { i } = stepAt(k, K);

      line(ctx, [view.toPx(-1.04, 0), view.toPx(1.04, 0)], c.axis, 1);
      const truth: Pt[] = ev.x.map((x, j) => view.toPx(x, ev.f_true[j]));
      line(ctx, truth, c.text3, 1.4 * s, { dash: [4, 4] });

      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, w, h);
      ctx.clip();
      for (const run of [...runs].reverse()) {
        const step = run.r.trace[Math.min(i, run.r.trace.length - 1)];
        const vals = step.info.curve as number[];
        const pts: Pt[] = run.xs.map((x, j) => view.toPx(x, vals[j]));
        line(ctx, pts, c.series[run.slot], 2 * s, { halo: c.halo });
      }
      ctx.restore();
      for (const run of runs) {
        const shown = Math.min(i, run.nodes.length - 1);
        for (let j = 0; j <= shown; j++) {
          const [x, y] = view.toPx(run.nodes[j][0], run.nodes[j][1]);
          if (run.slot === 0) dot(ctx, x, y, 2.8 * s, c.series[0], c.halo);
          else ring(ctx, x, y, 3.2 * s, c.series[1], c.halo, 1.5);
        }
      }
    },
  };
}
