/**
 * Combinatorial search: 2-opt on 15 random cities. Each step removes two edges of the tour and
 * reconnects it the shorter way (the new edges are highlighted), until no 2-opt move helps.
 */
import '../../../../methods/combinatorial/tsp';
import '../../../../problems/combinatorial';
import { runMethod } from '../../../../core/registry';
import { int } from '../../../../core/format';
import { getProblem } from '../../../../problems/registry';
import type { TspInstance } from '../../../../problems/combinatorial';
import type { Preview } from '../types';
import { dot, fitEqual, line, linClock, S, stepAt, type Box, type Pt } from '../draw';

interface Move {
  added: [number, number][];
}

export default function build(): Preview {
  const p = getProblem<TspInstance>('tsp_random_15');
  const r = runMethod('tsp_two_opt', p, {});
  const tours = r.trace.map((s) => (s.info.tour ?? s.x) as number[]);
  const lengths = r.trace.map((s) => s.fun ?? 0);
  const moves = r.trace.map((s) => (s.info.move as Move | null | undefined) ?? null);
  const K = tours.length - 1;
  const xs = p.coords.map((q) => q[0]),
    ys = p.coords.map((q) => q[1]);
  const BOX: Box = [
    [Math.min(...xs), Math.max(...xs)],
    [Math.min(...ys), Math.max(...ys)],
  ];
  const opt = p.optimumLength;

  return {
    lab: 'combinatorial',
    title: '15 random cities',
    caption:
      `2-opt from the tour 0 → 1 → … → 14: ${int(r.nIter)} improving moves shorten it from ` +
      `${lengths[0].toFixed(1)} to ${lengths[K].toFixed(1)}, where no 2-opt move helps` +
      (opt ? ` (the optimum is ${opt.toFixed(1)}).` : '.'),
    legend: [{ label: '2-opt', slot: 0, note: `${int(r.nIter)} moves` }],
    ariaLabel: `A tour through 15 cities untangles over ${int(r.nIter)} 2-opt moves, from length ${lengths[0].toFixed(1)} to ${lengths[K].toFixed(1)}.`,
    duration: 5,
    draw(ctx, size, c, u, hero) {
      const { width: w, height: h } = size;
      const s = S(hero);
      const view = fitEqual(BOX, w, h, hero ? 26 : 16);
      const px = (i: number): Pt => view.toPx(p.coords[i][0], p.coords[i][1]);
      const k = linClock(u, K);
      const { i, f } = stepAt(k, K);
      const draw = (tour: number[], alpha: number) => {
        const pts = [...tour, tour[0]].map(px);
        line(ctx, pts, c.series[0], 1.8 * s, { halo: c.halo, alpha });
      };
      draw(tours[i], 1 - f);
      if (f > 0) draw(tours[i + 1], f);
      // The edges the latest move added.
      const mv = moves[f > 0.5 ? i + 1 : i];
      if (mv && u < 1)
        for (const [a, b] of mv.added)
          line(ctx, [px(a), px(b)], c.series[0], 3.4 * s, { halo: c.halo });
      p.coords.forEach((_, j) => {
        const [x, y] = px(j);
        dot(ctx, x, y, 2.8 * s, c.text, c.halo);
      });
    },
  };
}
