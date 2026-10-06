/**
 * Constrained optimization: Rosenbrock restricted to the unit disk x² + y² ≤ 1. Projected gradient
 * follows the boundary, the log barrier approaches it from inside, SQP takes a few long steps.
 */
import '../../../../methods/constrained/methods';
import '../../../../problems/constrained';
import { runMethod } from '../../../../core/registry';
import { int, vec } from '../../../../core/format';
import { getProblem } from '../../../../problems/registry';
import type { ConstrainedProblem } from '../../../../problems/constrained';
import { drawPathLayer } from '../../../../viz/PathLayer';
import type { Preview } from '../types';
import {
  boundsOf,
  cross,
  hatchPattern,
  drawField,
  fitEqual,
  logClock,
  ring,
  S,
  strokeSegments,
  zeroSet,
  type Box,
} from '../draw';

export default function build(): Preview {
  const p = getProblem<ConstrainedProblem>('rosenbrock_unit_disk');
  const runs = [
    { id: 'projected_gradient', label: 'Projected gradient', slot: 0, width: 1.6 },
    { id: 'log_barrier', label: 'Log barrier', slot: 1, width: 1.8 },
    { id: 'sqp', label: 'SQP', slot: 2, width: 2 },
  ].map((m) => {
    const r = runMethod(m.id, p, {});
    return { ...m, r, xs: r.trace.map((s) => s.x as [number, number]) };
  });
  const K = Math.max(...runs.map((r) => r.xs.length)) - 1;
  const BOX: Box = boundsOf(
    [
      ...runs.map((r) => r.xs),
      [
        [-1, -1],
        [1, 1],
      ],
    ],
    0.06,
  );
  const f = (x: number, y: number) => p.f([x, y]);
  const g = (x: number, y: number) => Math.max(...p.constraints.map((cn) => cn.fun([x, y])));
  const star = p.minima[0];
  const x0 = p.x0;
  let cache: { key: string; segs: number[]; mask: HTMLCanvasElement | null } | null = null;

  return {
    lab: 'constrained',
    title: 'Rosenbrock in the unit disk',
    caption:
      `$\\min\\,(1 - x)^2 + 100\\,(y - x^2)^2$ subject to $x^2 + y^2 \\le 1$, ` +
      `from $\\mathbf{x}_0 =$ ${vec(x0, 3)}. ` +
      runs.map((r) => `${r.label} ${int(r.r.nIter)} steps`).join(', ') +
      `, each to its KKT test. The minimizer lies on the boundary. Time runs linear in log $k$.`,
    legend: runs.map((r) => ({ label: r.label, slot: r.slot, note: `${int(r.r.nIter)} steps` })),
    ariaLabel:
      `Level sets of the Rosenbrock function with the infeasible region outside the unit disk hatched; ` +
      runs.map((r) => `${r.label}: ${int(r.r.nIter)} steps`).join('; ') +
      ', all reaching the boundary minimizer.',
    duration: 5,
    draw(ctx, size, c, u, hero) {
      const { width: w, height: h } = size;
      const s = S(hero);
      const view = fitEqual(BOX, w, h, hero ? 16 : 8);
      drawField(ctx, 'rosenbrock-disk', f, view, size, c, { fMin: 0, levels: 14 });
      const key = `${w}x${h}x${size.dpr}|${c.mode}`;
      if (!cache || cache.key !== key) {
        // Infeasible set {g > 0}: a tint, rasterized once.
        const W = Math.round(w * size.dpr),
          H = Math.round(h * size.dpr);
        const mask = document.createElement('canvas');
        mask.width = W;
        mask.height = H;
        const mc = mask.getContext('2d');
        if (mc) {
          const img = mc.createImageData(W, H);
          const step = 2;
          for (let py = 0; py < H; py += step)
            for (let px = 0; px < W; px += step) {
              const [x, y] = view.toData(px / size.dpr, py / size.dpr);
              if (g(x, y) <= 0) continue;
              for (let dy = 0; dy < step && py + dy < H; dy++)
                for (let dx = 0; dx < step && px + dx < W; dx++) {
                  const o = ((py + dy) * W + px + dx) * 4;
                  img.data[o + 3] = 255;
                }
            }
          mc.putImageData(img, 0, 0);
          mc.globalCompositeOperation = 'source-in';
          const pat = hatchPattern(mc, c.region, size.dpr);
          if (pat) {
            mc.fillStyle = pat;
            mc.fillRect(0, 0, W, H);
          }
        }
        cache = { key, segs: zeroSet(g, view, 220), mask };
      }
      if (cache.mask) ctx.drawImage(cache.mask, 0, 0, w, h);
      strokeSegments(ctx, cache.segs, c.text2, 1.4 * s);
      const [sx, sy] = view.toPx(star[0], star[1]);
      cross(ctx, sx, sy, hero ? 6 : 5, c);
      drawPathLayer(
        ctx,
        runs.map((r) => ({
          points: r.xs,
          color: c.series[r.slot],
          dots: r.xs.length < 30,
          width: r.width * s,
          start: false,
          quiet: u >= 1,
          end: r.r.converged ? ('converged' as const) : ('stopped' as const),
        })),
        { t: logClock(u, K), toPx: view.toPx, halo: c.halo, trail: 1e9 },
      );
      const [ox, oy] = view.toPx(x0[0], x0[1]);
      ring(ctx, ox, oy, hero ? 5 : 4, c.text, c.halo);
    },
  };
}
