/**
 * Global optimization: a particle swarm on the Rastrigin function (a grid of local minima around
 * the global one at the origin). Every particle of every iteration is drawn from the trace.
 */
import '../../../../methods/unconstrained/global_';
import '../../../../problems/unconstrained';
import { runMethod } from '../../../../core/registry';
import { int } from '../../../../core/format';
import type { Problem2D } from '../../../../core/types';
import { getProblem } from '../../../../problems/registry';
import type { Preview } from '../types';
import {
  cross,
  dot,
  drawField,
  fitEqual,
  line,
  logClock,
  ring,
  S,
  stepAt,
  type Box,
} from '../draw';

const BOX: Box = [
  [-5.12, 5.12],
  [-5.12, 5.12],
];

export default function build(): Preview {
  const p = getProblem<Problem2D>('rastrigin');
  const r = runMethod('particle_swarm', p, {});
  const frames = r.trace.map((s) => s.info.particles as [number, number][]);
  const best = r.trace.map((s) => s.x as [number, number]);
  const K = frames.length - 1;
  const n = frames[0].length;
  const f = (x: number, y: number) => p.f([x, y]);

  return {
    lab: 'global',
    title: 'Rastrigin function',
    caption:
      `A swarm of ${n} particles on $[-5.12, 5.12]^2$. It stops when the swarm has collapsed onto ` +
      `one point, after ${int(r.nIter)} iterations, at the global minimizer (0, 0). ` +
      `Time runs linear in log $k$.`,
    legend: [{ label: 'Particle swarm', slot: 0, note: `${int(r.nIter)} iterations` }],
    ariaLabel: `${n} particles spread over the Rastrigin landscape gather at the global minimum (0, 0) over ${int(r.nIter)} iterations.`,
    duration: 5.5,
    draw(ctx, size, c, u, hero) {
      const { width: w, height: h } = size;
      const s = S(hero);
      const view = fitEqual(BOX, w, h, 0);
      drawField(ctx, 'rastrigin', f, view, size, c, {
        fMin: 0,
        levels: 6,
        scale: 'linear',
        lines: false,
      });
      const k = logClock(u, K);
      const { i, f: fr } = stepAt(k, K);
      const a = frames[i],
        b = frames[Math.min(i + 1, K)];
      const [cx, cy] = view.toPx(0, 0);
      cross(ctx, cx, cy, hero ? 6 : 5, c);
      // Where the swarm started (faint rings) and the path of its best point so far.
      for (const [x, y] of frames[0]) {
        const [px, py] = view.toPx(x, y);
        ring(ctx, px, py, 2.4 * s, c.text3, undefined, 1);
      }
      line(
        ctx,
        best.slice(0, i + 1).map(([x, y]) => view.toPx(x, y)),
        c.series[0],
        1.6 * s,
        { halo: c.halo },
      );
      for (let j = 0; j < a.length; j++) {
        const x = a[j][0] + (b[j][0] - a[j][0]) * fr;
        const y = a[j][1] + (b[j][1] - a[j][1]) * fr;
        const [px, py] = view.toPx(x, y);
        dot(ctx, px, py, 2.4 * s, c.series[0], c.halo);
      }
      const g = best[i];
      const [gx, gy] = view.toPx(g[0], g[1]);
      ring(ctx, gx, gy, 6 * s, c.text, c.halo, 1.4);
    },
  };
}
