/**
 * Linear programming on Wyndor Glass (max 3x₁ + 5x₂): the simplex method walks the vertices of
 * the feasible polygon, the primal–dual interior-point method cuts through its interior.
 */
import '../../../../methods/lp/simplex';
import '../../../../methods/lp/interior_point';
import '../../../../problems/lp';
import { getMethod, runMethod } from '../../../../core/registry';
import { int } from '../../../../core/format';
import type { LinearProgram } from '../../../../core/types';
import { getProblem } from '../../../../problems/registry';
import { drawPathLayer } from '../../../../viz/PathLayer';
import type { Preview } from '../types';
import { cross, dot, fillAlpha, fitEqual, line, linClock, S, type Box, type Pt } from '../draw';

type P2 = [number, number];

/** Sutherland–Hodgman: clip a convex polygon to a·x ≤ b. */
function clip(poly: P2[], a: number[], b: number): P2[] {
  const out: P2[] = [];
  const val = (p: P2) => a[0] * p[0] + a[1] * p[1] - b;
  for (let i = 0; i < poly.length; i++) {
    const p = poly[i],
      q = poly[(i + 1) % poly.length];
    const vp = val(p),
      vq = val(q);
    if (vp <= 0) out.push(p);
    if (vp * vq < 0) {
      const t = vp / (vp - vq);
      out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
    }
  }
  return out;
}

export default function build(): Preview {
  const lp = getProblem<LinearProgram>('wyndor');
  const simplex = runMethod('simplex', lp, {});
  const ipm = runMethod('primal_dual_ipm', lp, {});
  const walk = simplex.trace.map((s) => (s.info.vertex as number[]).slice(0, 2) as P2);
  const cut = ipm.trace.map((s) => (s.x as number[]).slice(0, 2) as P2);
  let poly: P2[] = [
    [0, 0],
    [20, 0],
    [20, 20],
    [0, 20],
  ];
  (lp.aUb ?? []).forEach((row, i) => (poly = clip(poly, row, lp.bUb![i])));
  const opt = (lp.optimum ?? walk[walk.length - 1]) as P2;
  const K = Math.max(walk.length, cut.length) - 1;
  const BOX: Box = [
    [-0.6, 5.2],
    [-0.6, 7.2],
  ];
  const cDir = lp.c;
  // Registry names: one name per method across the site.
  const [simplexName, ipmName] = ['simplex', 'primal_dual_ipm'].map(
    (id) => getMethod(id).spec.name,
  );

  return {
    lab: 'lp',
    title: 'Wyndor Glass $\\max\\; 3x_1 + 5x_2$',
    caption:
      `The feasible set of three constraints and $\\mathbf{x} \\ge \\mathbf{0}$. ${simplexName} makes ${int(simplex.nIter)} ` +
      `pivots along the edges, (0, 0) → (0, 6) → (2, 6). ${ipmName} reaches (2, 6) ` +
      `through the interior in ${int(ipm.nIter)} iterations. Optimal value 36.`,
    legend: [
      { label: simplexName, slot: 0, note: `${int(simplex.nIter)} pivots` },
      { label: ipmName, slot: 1, note: `${int(ipm.nIter)} iterations` },
    ],
    ariaLabel:
      'A five-sided feasible polygon. Simplex walks from the origin along the edges to the vertex ' +
      '(2, 6); the interior-point path approaches the same vertex from inside.',
    duration: 4.5,
    draw(ctx, size, c, u, hero) {
      const { width: w, height: h } = size;
      const s = S(hero);
      const view = fitEqual(BOX, w, h, hero ? 22 : 12);
      const px = (p: Pt) => view.toPx(p[0], p[1]);

      // Objective level lines 3x₁ + 5x₂ = v (context, not data).
      const [[bx0, bx1], [by0, by1]] = view.box;
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, w, h);
      ctx.clip();
      for (let v = 6; v <= 66; v += 6) {
        const a: P2 = [bx0, (v - cDir[0] * bx0) / cDir[1]];
        const b: P2 = [bx1, (v - cDir[0] * bx1) / cDir[1]];
        if (Math.max(a[1], b[1]) < by0 || Math.min(a[1], b[1]) > by1) continue;
        line(ctx, [px(a), px(b)], c.iso, 1);
      }
      ctx.restore();

      // Axes x₁, x₂ ≥ 0.
      line(ctx, [px([0, 0]), px([bx1, 0])], c.axis, 1);
      line(ctx, [px([0, 0]), px([0, by1])], c.axis, 1);

      // Feasible polygon.
      const pts = poly.map(px);
      fillAlpha(ctx, c.text, c.mode === 'dark' ? 0.08 : 0.05, () => {
        ctx.beginPath();
        pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        ctx.closePath();
        ctx.fill();
      });
      line(ctx, [...pts, pts[0]], c.text2, 1.4 * s);
      pts.forEach(([x, y]) => dot(ctx, x, y, 2.2 * s, c.text2, c.halo));

      const [ox, oy] = px(opt);
      cross(ctx, ox, oy, hero ? 8 : 7, c);
      const t = linClock(u, K);
      drawPathLayer(
        ctx,
        [
          {
            points: cut,
            color: c.series[1],
            dots: true,
            width: 1.8 * s,
            start: true,
            quiet: u >= 1,
            end: 'converged',
          },
          {
            points: walk,
            color: c.series[0],
            dots: true,
            width: 2.4 * s,
            start: true,
            quiet: u >= 1,
            end: 'converged',
          },
        ],
        { t, toPx: view.toPx, halo: c.halo, ease: true, trail: 1e9 },
      );
    },
  };
}
