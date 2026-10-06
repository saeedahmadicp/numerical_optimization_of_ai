/**
 * Nonlinear systems: Newton and Broyden find where the circle x² + y² = 4 meets the line
 * y = x − 1, from 𝐱₀ = (0.5, −1.5). The two zero sets are drawn from the problem's components.
 */
import '../../../../methods/roots/systems';
import '../../../../problems/systems';
import { runMethod } from '../../../../core/registry';
import { int, vec } from '../../../../core/format';
import { getProblem } from '../../../../problems/registry';
import type { SystemProblem } from '../../../../problems/systems';
import { drawPathLayer } from '../../../../viz/PathLayer';
import type { Preview } from '../types';
import { cross, fitEqual, linClock, ring, S, strokeSegments, zeroSet, type Box } from '../draw';

const X0: [number, number] = [0.5, -1.5];
const BOX: Box = [
  [-2.4, 2.4],
  [-3, 2.3],
];

export default function build(): Preview {
  const p = getProblem<SystemProblem>('circle_line');
  const runs = [
    { id: 'newton_system', label: 'Newton', slot: 0 },
    { id: 'broyden', label: 'Broyden', slot: 1 },
  ].map((m) => {
    const r = runMethod(m.id, p, { x0: [...X0] });
    return { ...m, r, xs: r.trace.map((s) => s.x as [number, number]) };
  });
  const K = Math.max(...runs.map((r) => r.xs.length)) - 1;
  const root = runs[0].xs[runs[0].xs.length - 1];
  const [F1, F2] = p.components;
  let cache: { key: string; a: number[]; b: number[] } | null = null;

  return {
    lab: 'systems',
    title: 'Circle and line',
    caption:
      `$F(x, y) = (x^2 + y^2 - 4,\\ y - x + 1) = \\mathbf{0}$ from ` +
      `$\\mathbf{x}_0 =$ ${vec(X0)}. Newton converges in ${int(runs[0].r.nIter)} steps and ` +
      `Broyden in ${int(runs[1].r.nIter)}, both to ${vec(root, 3)}.`,
    legend: runs.map((r) => ({ label: r.label, slot: r.slot, note: `${int(r.r.nIter)} steps` })),
    ariaLabel: `A circle and a line crossing at two points; Newton (${int(runs[0].r.nIter)} steps) and Broyden (${int(runs[1].r.nIter)} steps) reach the lower one from (0.5, −1.5).`,
    duration: 4.5,
    draw(ctx, size, c, u, hero) {
      const { width: w, height: h } = size;
      const s = S(hero);
      const view = fitEqual(BOX, w, h, hero ? 18 : 10);
      const key = `${w}x${h}`;
      if (!cache || cache.key !== key) cache = { key, a: zeroSet(F1, view), b: zeroSet(F2, view) };
      strokeSegments(ctx, cache.a, c.text2, 1.4 * s);
      strokeSegments(ctx, cache.b, c.text2, 1.4 * s, [5, 4]);
      const [rx, ry] = view.toPx(root[0], root[1]);
      cross(ctx, rx, ry, hero ? 6 : 5, c);
      drawPathLayer(
        ctx,
        [...runs].reverse().map((r) => ({
          points: r.xs,
          color: c.series[r.slot],
          dots: true,
          width: 1.9 * s,
          start: false,
          quiet: u >= 1,
          end: r.r.converged ? ('converged' as const) : ('stopped' as const),
        })),
        { t: linClock(u, K), toPx: view.toPx, halo: c.halo, trail: 1e9 },
      );
      const [x0, y0] = view.toPx(X0[0], X0[1]);
      ring(ctx, x0, y0, hero ? 5 : 4, c.text, c.halo);
    },
  };
}
