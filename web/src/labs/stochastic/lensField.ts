/**
 * Level sets of the full loss for the step lens, cheap enough to rebuild on every frame.
 *
 * `problem.f` reproduces NumPy's summation order (for parity), which costs an allocation and a
 * pairwise sum per call; the lens needs thousands of evaluations per step and only to drawing
 * accuracy. So:
 *
 *   squared loss   f is an exact quadratic. With H = AᵀA/n + λI and 𝐛 = Aᵀ𝐲/n it is
 *                  f(𝐰) = f(𝐰_c) + ½(𝐰 − 𝐰_c)ᵀH(𝐰 − 𝐰_c), 𝐰_c = H⁻¹𝐛: O(1) per point, and
 *                  centered at 𝐰_c, so differences of 10⁻¹² near the minimizer are still exact.
 *   logistic/Huber a plain loop over flat columns (no allocation), on a coarser grid.
 *
 * One grid of values serves every level (marching squares per level on the same grid).
 */
import { isoSegments } from '../../viz/contourField';
import type { FiniteSumProblem } from '../../problems/stochastic';

export type Eval2 = (x: number, y: number) => number;

const cache = new WeakMap<FiniteSumProblem, { f: Eval2; quadratic: boolean }>();

/** Quadratic coefficients of a squared-loss problem: H, 𝐰_c, f(𝐰_c) (null if H is singular). */
export function quadraticModel(
  p: FiniteSumProblem,
): { H: [number, number, number]; center: [number, number]; fc: number } | null {
  if (p.loss !== 'squared' || p.dim !== 2) return null;
  const n = p.nSamples;
  let h00 = 0,
    h01 = 0,
    h11 = 0,
    b0 = 0,
    b1 = 0,
    yy = 0;
  for (let i = 0; i < n; i++) {
    const [a0, a1] = p.X[i];
    const y = p.y[i];
    h00 += a0 * a0;
    h01 += a0 * a1;
    h11 += a1 * a1;
    b0 += a0 * y;
    b1 += a1 * y;
    yy += y * y;
  }
  h00 = h00 / n + p.l2;
  h11 = h11 / n + p.l2;
  h01 /= n;
  b0 /= n;
  b1 /= n;
  const det = h00 * h11 - h01 * h01;
  if (!(det > 0)) return null;
  const c0 = (h11 * b0 - h01 * b1) / det;
  const c1 = (h00 * b1 - h01 * b0) / det;
  // f(𝐰) = ½𝐰ᵀH𝐰 − 𝐛ᵀ𝐰 + ‖𝐲‖²/(2n)  ⇒  f(𝐰_c) = ‖𝐲‖²/(2n) − ½𝐛ᵀ𝐰_c.
  const fc = yy / (2 * n) - 0.5 * (b0 * c0 + b1 * c1);
  return { H: [h00, h01, h11], center: [c0, c1], fc };
}

/** A fast f(w₀, w₁) for drawing (cached per problem). */
export function lossEvaluator(p: FiniteSumProblem): { f: Eval2; quadratic: boolean } {
  const hit = cache.get(p);
  if (hit) return hit;
  let out: { f: Eval2; quadratic: boolean };
  const q = quadraticModel(p);
  if (q) {
    const [h00, h01, h11] = q.H;
    const [c0, c1] = q.center;
    const fc = q.fc;
    out = {
      quadratic: true,
      f: (x, y) => {
        const d0 = x - c0,
          d1 = y - c1;
        return fc + 0.5 * (h00 * d0 * d0 + 2 * h01 * d0 * d1 + h11 * d1 * d1);
      },
    };
  } else {
    const n = p.nSamples;
    const a0 = new Float64Array(n),
      a1 = new Float64Array(n),
      ys = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      a0[i] = p.X[i][0];
      a1[i] = p.X[i][1];
      ys[i] = p.y[i];
    }
    const half = 0.5 * p.l2;
    const delta = p.huberDelta;
    const f: Eval2 =
      p.loss === 'logistic'
        ? (x, y) => {
            let s = 0;
            for (let i = 0; i < n; i++) {
              const z = a0[i] * x + a1[i] * y;
              s += Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z))) - ys[i] * z;
            }
            return s / n + half * (x * x + y * y);
          }
        : p.loss === 'huber'
          ? (x, y) => {
              let s = 0;
              for (let i = 0; i < n; i++) {
                const r = Math.abs(a0[i] * x + a1[i] * y - ys[i]);
                s += r <= delta ? 0.5 * r * r : delta * (r - 0.5 * delta);
              }
              return s / n + half * (x * x + y * y);
            }
          : (x, y) => p.f([x, y]);
    out = { quadratic: false, f };
  }
  cache.set(p, out);
  return out;
}

/**
 * Segments (flat [xa, ya, xb, yb, …], data coordinates) of f = level for each level, over the
 * square box, from one nGrid × nGrid grid of f values.
 */
export function levelSets(
  f: Eval2,
  box: { cx: number; cy: number; half: number },
  levels: readonly number[],
  nGrid: number,
): number[][] {
  const x0 = box.cx - box.half,
    y0 = box.cy - box.half,
    h = (2 * box.half) / (nGrid - 1);
  const g = new Float64Array(nGrid * nGrid);
  for (let j = 0; j < nGrid; j++)
    for (let i = 0; i < nGrid; i++) {
      const v = f(x0 + i * h, y0 + j * h);
      g[j * nGrid + i] = Number.isFinite(v) ? v : NaN;
    }
  return levels.map((level) => {
    const raw: number[] = [];
    if (!Number.isFinite(level)) return raw;
    isoSegments(g, nGrid, nGrid, level, raw);
    for (let k = 0; k < raw.length; k += 2) {
      raw[k] = x0 + raw[k] * h;
      raw[k + 1] = y0 + raw[k + 1] * h;
    }
    return raw;
  });
}
