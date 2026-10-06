/**
 * Descent race on Himmelblau's function from 𝐱₀ = (−3.75, 2.5): four methods take four clearly
 * different routes to the same minimizer (−2.805, 3.131) — chosen so the paths do not tangle.
 * One stopping test for all (‖∇f(𝐱ₖ)‖₂ ≤ 10⁻⁶, each trace cut at the first iterate that passes
 * it), one clock linear in log(k + 1) so a 5-step Newton run and a 300-step heavy-ball run can be
 * watched together.
 */
import '../../../../methods/unconstrained/first_order';
import '../../../../methods/unconstrained/quasi_newton';
import '../../../../methods/unconstrained/newton';
import '../../../../problems/unconstrained';
import { getMethod, runMethod } from '../../../../core/registry';
import { norm } from '../../../../core/linalg';
import { int, vec } from '../../../../core/format';
import type { Problem2D } from '../../../../core/types';
import { getProblem } from '../../../../problems/registry';
import { drawPathLayer, type PathSpec } from '../../../../viz/PathLayer';
import type { Preview } from '../types';
import { boundsOf, cross, drawField, fitEqual, logClock, ring, S, type Box } from '../draw';

const PROBLEM = 'himmelblau';
const X0: [number, number] = [-3.75, 2.5];
const GTOL = 1e-6;

/** Legend names are the registry names (one name per method across the site). Every method runs
 *  with its registry defaults (gradient descent: Armijo backtracking). */
const RUNS = [
  { id: 'gradient_descent', slot: 0, width: 1.6 },
  { id: 'momentum', slot: 1, width: 1.4 },
  { id: 'bfgs', slot: 2, width: 2, dots: true },
  { id: 'damped_newton', slot: 3, width: 2, dots: true },
];

export default function build(): Preview {
  const p = getProblem<Problem2D>(PROBLEM);
  const runs = RUNS.map((r) => {
    const label = getMethod(r.id).spec.name;
    const res = runMethod(r.id, p, { x0: [...X0] });
    const xs: [number, number][] = [];
    let converged = false;
    for (const s of res.trace) {
      const x = s.x as [number, number];
      xs.push(x);
      if (norm(p.grad(x)) <= GTOL) {
        converged = true;
        break;
      }
    }
    return { ...r, label, xs, converged, n: xs.length - 1 };
  });
  const K = Math.max(...runs.map((r) => r.n));
  const BOX: Box = boundsOf(
    runs.map((r) => r.xs),
    0.08,
  );
  const star = runs[0].xs[runs[0].xs.length - 1];
  const f = (x: number, y: number) => p.f([x, y]);
  // Paint order: the long crawler last and thinnest (brand.md §9).
  const order = [...runs].sort((a, b) => a.n - b.n);

  return {
    lab: 'unconstrained',
    title: 'Himmelblau’s function',
    caption:
      `Four methods from $\\mathbf{x}_0 =$ ${vec(X0)}, one stopping test for all: ` +
      `$\\|\\nabla f(\\mathbf{x}_k)\\|_2 \\le 10^{-6}$. All four reach the same minimizer ` +
      `$\\mathbf{x}^\\star \\approx$ ${vec(star)}, each by its own route. ` +
      `Time runs linear in log $k$.`,
    legend: runs.map((r) => ({
      label: r.label,
      slot: r.slot,
      note: r.converged ? `${int(r.n)} iterations` : `stopped after ${int(r.n)}`,
    })),
    ariaLabel:
      `Level sets of Himmelblau's function with four descent paths from ${vec(X0)} to the minimizer ${vec(star)}: ` +
      runs.map((r) => `${r.label}, ${int(r.n)} iterations`).join('; ') +
      '.',
    duration: 5.5,
    draw(ctx, size, c, u, hero) {
      const { width: w, height: h } = size;
      const view = fitEqual(BOX, w, h, hero ? 18 : 10);
      drawField(ctx, PROBLEM, f, view, size, c, { fMin: 0, levels: 14 });
      const t = logClock(u, K);
      const paths: PathSpec[] = order.map((r) => ({
        points: r.xs,
        color: c.series[r.slot],
        label: r.label,
        dots: r.dots ?? false,
        width: r.width * S(hero),
        milestones: false,
        start: false,
        quiet: u >= 1,
        end: r.converged ? 'converged' : 'stopped',
      }));
      const [sx, sy] = view.toPx(star[0], star[1]);
      cross(ctx, sx, sy, hero ? 6 : 5, c);
      drawPathLayer(ctx, paths, { t, toPx: view.toPx, halo: c.halo, ease: true, trail: 1e9 });
      const [x0, y0] = view.toPx(X0[0], X0[1]);
      ring(ctx, x0, y0, hero ? 5 : 4, c.text, c.halo);
    },
  };
}
