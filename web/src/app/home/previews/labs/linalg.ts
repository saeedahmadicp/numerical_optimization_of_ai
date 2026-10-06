/**
 * Linear systems: Shewchuk's 2×2 SPD system A𝐱 = 𝐛 seen as the minimum of φ(𝐱) = ½𝐱ᵀA𝐱 − 𝐛ᵀ𝐱.
 * The two equations are the two lines; Jacobi zigzags, Gauss–Seidel climbs a staircase, conjugate
 * gradient lands in two steps.
 */
import '../../../../methods/linalg/iterative';
import '../../../../problems/linalg';
import { getMethod, runMethod } from '../../../../core/registry';
import { int, vec } from '../../../../core/format';
import type { LinearSystem } from '../../../../core/types';
import { getProblem } from '../../../../problems/registry';
import { drawPathLayer } from '../../../../viz/PathLayer';
import type { Preview } from '../types';
import { cross, drawField, fitEqual, line, logClock, ring, S, type Box } from '../draw';

const BOX: Box = [
  [-0.5, 2.6],
  [-2.6, 0.5],
];

export default function build(): Preview {
  const sys = getProblem<LinearSystem>('spd_2x2');
  const { A, b } = sys;
  // Legend names are the registry names (one name per method across the site).
  const runs = [
    { id: 'jacobi', slot: 0, width: 1.5 },
    { id: 'gauss_seidel', slot: 1, width: 1.7 },
    { id: 'conjugate_gradient_linear', slot: 2, width: 2.2 },
  ].map((m) => {
    const r = runMethod(m.id, sys, {});
    const label = getMethod(m.id).spec.name;
    return { ...m, label, r, xs: r.trace.map((s) => s.x as [number, number]) };
  });
  const K = Math.max(...runs.map((r) => r.xs.length)) - 1;
  const sol = (sys.solution ?? runs[2].xs[runs[2].xs.length - 1]) as [number, number];
  const phi = (x: number, y: number) =>
    0.5 * (A[0][0] * x * x + 2 * A[0][1] * x * y + A[1][1] * y * y) - b[0] * x - b[1] * y;
  const phiMin = phi(sol[0], sol[1]);

  return {
    lab: 'linalg',
    title: 'A 2×2 SPD system $A\\mathbf{x} = \\mathbf{b}$',
    caption:
      // The matrix in words (an inline matrix would make its line taller): rows (3, 2), (2, 6).
      `$A$ with rows ${A.map((row) => vec(row)).join(' and ')}, $\\mathbf{b} =$ ${vec(b)}, ` +
      `from $\\mathbf{x}_0 = \\mathbf{0}$; stop when ` +
      `$\\|\\mathbf{r}_k\\|_2 \\le 10^{-10}\\,\\|\\mathbf{b}\\|_2$. Iterations to that test: ` +
      runs.map((r) => `${r.label} ${int(r.r.nIter)}`).join(', ') +
      '. The ellipses are level sets of $\\tfrac{1}{2}\\mathbf{x}^\\top A\\mathbf{x} - \\mathbf{b}^\\top\\mathbf{x}$.',
    legend: runs.map((r) => ({
      label: r.label,
      slot: r.slot,
      note: `${int(r.r.nIter)} iterations`,
    })),
    ariaLabel:
      `Two lines crossing at the solution ${vec(sol)} over elliptical level sets. ` +
      runs.map((r) => `${r.label}: ${int(r.r.nIter)} iterations`).join('; ') +
      '.',
    duration: 4.5,
    draw(ctx, size, c, u, hero) {
      const { width: w, height: h } = size;
      const s = S(hero);
      const view = fitEqual(BOX, w, h, hero ? 18 : 10);
      drawField(ctx, 'spd_2x2', phi, view, size, c, { fMin: phiMin, levels: 12, scale: 'linear' });
      const [[x0, x1]] = view.box;
      // Row i of A𝐱 = 𝐛 as a line.
      for (const i of [0, 1]) {
        const y = (x: number) => (b[i] - A[i][0] * x) / A[i][1];
        line(ctx, [view.toPx(x0, y(x0)), view.toPx(x1, y(x1))], c.text2, 1.2 * s, {
          dash: i ? [5, 4] : [],
        });
      }
      const [sx, sy] = view.toPx(sol[0], sol[1]);
      cross(ctx, sx, sy, hero ? 6 : 5, c);
      drawPathLayer(
        ctx,
        runs.map((r) => ({
          points: r.xs,
          color: c.series[r.slot],
          dots: r.xs.length < 20,
          width: r.width * s,
          start: false,
          quiet: u >= 1,
          end: 'converged' as const,
        })),
        { t: logClock(u, K), toPx: view.toPx, halo: c.halo, trail: 1e9 },
      );
      const [ox, oy] = view.toPx(0, 0);
      ring(ctx, ox, oy, hero ? 5 : 4, c.text, c.halo);
    },
  };
}
