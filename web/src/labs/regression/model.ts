/**
 * Pure mathematics of the regression lab: the fitted curve under the playhead, the robust
 * weights drawn as point opacity, goodness-of-fit statistics at any iterate, the degree sweep
 * (training error, leave-one-out error, error against the true function) and the ridge λ path.
 *
 * Every quantity is computed from the registered methods' traces (Step.x = β, Step.info) or by
 * calling the registered methods themselves, so the lab draws what Python computes.
 */
import { getMethod, defaults } from '../../core/registry';
import type { Result, Step } from '../../core/types';
import { lerp, easeInOut } from '../../play/timeline';
import {
  huberRho,
  median,
  polyval,
  qrLstsq,
  svd,
  vandermonde,
  type RegressionProblem,
} from '../../methods/regression/methods';

export interface Pt {
  x: number;
  y: number;
}

export type Kind = 'ols' | 'poly' | 'ridge' | 'huber' | 'lad' | 'theil' | 'minimax';

const KINDS: Record<string, Kind> = {
  linear_regression: 'ols',
  polynomial_regression: 'poly',
  ridge_regression: 'ridge',
  huber_regression: 'huber',
  lad_regression: 'lad',
  theil_sen: 'theil',
  chebyshev_minimax_line: 'minimax',
};

export const kindOf = (id: string): Kind => KINDS[id] ?? 'ols';

/** Methods whose trace has more than one step (an iteration to watch). */
export const isIterative = (k: Kind) => k === 'huber' || k === 'lad' || k === 'minimax';
/** Methods with a polynomial degree parameter (the lab's shared degree slider). */
export const hasDegree = (id: string) =>
  id === 'polynomial_regression' || id === 'ridge_regression';
/** Least-squares family: the residual squares are the objective. */
export const isLeastSquares = (k: Kind) => k === 'ols' || k === 'poly' || k === 'ridge';

const finiteVec = (v: unknown): v is number[] =>
  Array.isArray(v) && v.length > 0 && v.every((t) => typeof t === 'number' && Number.isFinite(t));

/**
 * β at continuous time t: β_k for an integer t, otherwise the eased blend of β_k and β_{k+1}
 * (a straight-line motion of the coefficients between two iterates, never a new fit).
 */
export function betaAt(trace: readonly Step[], t: number, ease: boolean): number[] | null {
  if (trace.length === 0) return null;
  const lt = Math.max(0, Math.min(t, trace.length - 1));
  const i = Math.floor(lt + 1e-9);
  const a = trace[i].x;
  if (!finiteVec(a)) return null;
  const frac = lt - i;
  if (frac <= 1e-9 || i + 1 >= trace.length) return a.slice();
  const b = trace[i + 1].x;
  if (!finiteVec(b) || b.length !== a.length) return a.slice();
  const u = ease ? easeInOut(frac) : frac;
  return a.map((v, j) => lerp(v, b[j], u));
}

export const residuals = (pts: readonly Pt[], beta: readonly number[]) =>
  pts.map((p) => p.y - polyval(beta, p.x));

/**
 * Point opacity weight in [0, 1] for the focused method at step k:
 *   Huber — the IRLS weight wᵢ = min(1, δσ̂/|rᵢ|) itself (already in (0, 1]);
 *   LAD   — wᵢ = 1/max(|rᵢ|, ε) relative to the median weight, capped at 1: points whose
 *           residual is below the median |r| are opaque, the others fade like 1/|rᵢ|.
 */
export function displayWeights(kind: Kind, step: Step | undefined): number[] | null {
  const w = step?.info.weights;
  if (!Array.isArray(w)) return null;
  const ws = w as number[];
  if (kind === 'huber') return ws.map((v) => Math.max(0, Math.min(1, v)));
  if (kind === 'lad') {
    const med = median(ws);
    if (!(med > 0) || !Number.isFinite(med)) return ws.map(() => 1);
    return ws.map((v) => Math.min(1, v / med));
  }
  return null;
}

/** Goodness of fit of β on the data (Python `_goodness` formulas). */
export interface FitStats {
  rss: number;
  r2: number | null;
  adjR2: number | null;
  rmse: number;
  maxAbs: number;
  sumAbs: number;
}

export function fitStats(pts: readonly Pt[], beta: readonly number[]): FitStats {
  const m = pts.length;
  const r = residuals(pts, beta);
  const rss = r.reduce((s, v) => s + v * v, 0);
  const ybar = pts.reduce((s, p) => s + p.y, 0) / m;
  const tss = pts.reduce((s, p) => s + (p.y - ybar) ** 2, 0);
  const ymax = Math.max(...pts.map((p) => Math.abs(p.y)));
  const devMax = Math.max(...pts.map((p) => Math.abs(p.y - ybar)));
  const flat = devMax <= m * 2.220446049250313e-16 * ymax;
  const r2 = tss > 0 && !flat ? 1 - rss / tss : null;
  const dof = m - beta.length;
  return {
    rss,
    r2,
    adjR2: r2 !== null && dof > 0 ? 1 - ((1 - r2) * (m - 1)) / dof : null,
    rmse: Math.sqrt(rss / m),
    maxAbs: Math.max(...r.map(Math.abs)),
    sumAbs: r.reduce((s, v) => s + Math.abs(v), 0),
  };
}

/** The method's own objective F(β) at a step: Step.fun (RSS, ridge, Huber, Σ|r|, max|r|). */
export const objectiveTex: Record<Kind, string> = {
  ols: '\\textstyle\\sum r_i^2',
  poly: '\\textstyle\\sum r_i^2',
  ridge: '\\textstyle\\sum r_i^2 + \\lambda\\|\\boldsymbol\\beta_{1:}\\|^2',
  huber: '\\textstyle\\sum \\rho_\\delta(r_i/\\hat\\sigma)',
  lad: '\\textstyle\\sum |r_i|',
  theil: '\\textstyle\\sum r_i^2',
  minimax: '\\max_i |r_i|',
};

/**
 * Relative optimality gap (F(β_k) − F⋆)/F⋆ of an iterative run, F the method's own objective
 * and F⋆ the smallest value on its trace (the final one for a converged run). Exact zeros are
 * gaps (null) on the log axis.
 */
export function relativeGap(result: Result): (number | null)[] {
  const f = result.trace.map((s) => s.fun);
  const fin = f.filter((v): v is number => v !== null && Number.isFinite(v));
  if (fin.length === 0) return f.map(() => null);
  const best = Math.min(...fin);
  const denom = Math.abs(best) > 0 ? Math.abs(best) : 1;
  return f.map((v) =>
    v === null || !Number.isFinite(v) || v - best <= 0 ? null : (v - best) / denom,
  );
}

// ---------------------------------------------------------------------------------------
// Geometry of single methods
// ---------------------------------------------------------------------------------------

/** Theil–Sen: the pair(s) (i, j) whose slope is the median (two pairs for an even count). */
export function medianPairs(pts: readonly Pt[], slopes: readonly number[]): [number, number][] {
  const n = slopes.length;
  if (n === 0) return [];
  const h = n >> 1;
  const targets = n % 2 === 1 ? [slopes[h]] : [slopes[h - 1], slopes[h]];
  const out: [number, number][] = [];
  for (const target of targets) {
    let found: [number, number] | null = null;
    for (let i = 0; i < pts.length && !found; i++)
      for (let j = i + 1; j < pts.length; j++) {
        const dx = pts[j].x - pts[i].x;
        if (dx === 0) continue;
        if ((pts[j].y - pts[i].y) / dx === target) {
          found = [i, j];
          break;
        }
      }
    if (found && !out.some((p) => p[0] === found![0] && p[1] === found![1])) out.push(found);
  }
  return out;
}

/** LAD: points the line passes through (|rᵢ| ≤ ε: their IRLS weight sits at the floor 1/ε). */
export function floorPoints(pts: readonly Pt[], beta: readonly number[], eps: number): number[] {
  return residuals(pts, beta)
    .map((r, i) => (Math.abs(r) <= eps * (1 + 1e-9) ? i : -1))
    .filter((i) => i >= 0);
}

// ---------------------------------------------------------------------------------------
// Model selection: the degree sweep and the ridge path
// ---------------------------------------------------------------------------------------

function colNorms(a: number[][]): number[] {
  const p = a[0].length;
  const s = new Array<number>(p).fill(0);
  for (const row of a) for (let j = 0; j < p; j++) s[j] += row[j] * row[j];
  return s.map((v) => (v > 0 ? Math.sqrt(v) : 1));
}

/** Diagonal of the hat matrix A(AᵀA + P)⁻¹Aᵀ (P = diag(pen)), from a QR of [A/s; √P/s]. */
function hatDiagonal(a: number[][], pen: readonly number[]): number[] | null {
  const p = a[0].length;
  const s = colNorms(a);
  const as = a.map((row) => row.map((v, j) => v / s[j]));
  const aug = [
    ...as,
    ...pen
      .map((lam, j) =>
        lam > 0 ? Array.from({ length: p }, (_, c) => (c === j ? Math.sqrt(lam) / s[j] : 0)) : null,
      )
      .filter((r): r is number[] => r !== null),
  ];
  if (aug.length < p) return null;
  const sv = svd(aug).s;
  if (!(sv[sv.length - 1] > Math.max(aug.length, p) * 2.220446049250313e-16 * sv[0])) return null;
  const [, R] = qrLstsq(
    aug,
    aug.map(() => 0),
  );
  // h_ii = ‖R⁻ᵀ aᵢ‖²: forward substitution with Rᵀ (lower triangular).
  return as.map((ai) => {
    const z = new Array<number>(p);
    let h = 0;
    for (let i = 0; i < p; i++) {
      let t = ai[i];
      for (let j = 0; j < i; j++) t -= R[j][i] * z[j];
      z[i] = t / R[i][i];
      h += z[i] * z[i];
    }
    return h;
  });
}

/** Leave-one-out RMSE of a linear smoother from its residuals and hat diagonal: rᵢ/(1 − hᵢᵢ). */
function looRmse(r: readonly number[], h: readonly number[] | null): number | null {
  if (!h) return null;
  let s = 0;
  for (let i = 0; i < r.length; i++) {
    const d = 1 - h[i];
    if (!(d > 1e-10)) return null;
    s += (r[i] / d) ** 2;
  }
  return Math.sqrt(s / r.length);
}

/** RMS of p − f over 400 equispaced points of the domain (the error against the truth). */
function trueError(
  beta: readonly number[],
  fTrue: ((x: number) => number) | undefined,
  domain: readonly [number, number],
): number | null {
  if (!fTrue) return null;
  const n = 400;
  let s = 0;
  for (let i = 0; i < n; i++) {
    const x = domain[0] + ((domain[1] - domain[0]) * i) / (n - 1);
    s += (polyval(beta, x) - fTrue(x)) ** 2;
  }
  const v = Math.sqrt(s / n);
  return Number.isFinite(v) ? v : null;
}

export interface SweepPoint {
  /** Degree d or penalty λ. */
  at: number;
  train: number | null;
  loo: number | null;
  truth: number | null;
  /** Ridge: effective degrees of freedom tr H(λ). */
  df?: number | null;
  /** Ridge: standardized slopes γⱼ = βⱼ‖xʲ − mean‖₂, j = 1..d. */
  gamma?: number[];
}

const asProblem = (pts: readonly Pt[]): RegressionProblem => [
  pts.map((p) => p.x),
  pts.map((p) => p.y),
];

function runFit(id: string, pts: readonly Pt[], params: Record<string, number>): Result | null {
  try {
    const { spec, fn } = getMethod(id);
    return fn(asProblem(pts), { ...defaults(spec), ...params });
  } catch {
    return null;
  }
}

/** Largest degree the sweep shows: past m − 2 the fit interpolates and the training error is 0. */
export const maxSweepDegree = (m: number) => Math.max(0, Math.min(15, m - 2));

/**
 * Polynomial least squares for d = 0 … min(15, m − 2): training RMSE (from the registered
 * method), leave-one-out RMSE (hat-matrix identity, exact for least squares) and the RMS error
 * against the true regression function when the dataset has one.
 */
export function degreeSweep(
  pts: readonly Pt[],
  fTrue: ((x: number) => number) | undefined,
  domain: readonly [number, number],
): SweepPoint[] {
  const out: SweepPoint[] = [];
  const dMax = maxSweepDegree(pts.length);
  for (let d = 0; d <= dMax; d++) {
    const res = runFit('polynomial_regression', pts, { degree: d });
    if (!res || !finiteVec(res.x)) {
      out.push({ at: d, train: null, loo: null, truth: null });
      continue;
    }
    const beta = res.x;
    const r = residuals(pts, beta);
    const h = res.converged
      ? hatDiagonal(
          vandermonde(
            pts.map((p) => p.x),
            d,
          ),
          new Array<number>(d + 1).fill(0),
        )
      : null;
    out.push({
      at: d,
      train: res.extra.rmse as number,
      loo: looRmse(r, h),
      truth: trueError(beta, fTrue, domain),
    });
  }
  return out;
}

/** λ grid of the ridge path: 10⁻⁸ … 10⁴, four points per decade (the ParamSpec range). */
export const LAMBDA_GRID = Array.from({ length: 49 }, (_, i) => 10 ** (-8 + i / 4));

/**
 * Ridge on the λ grid at a fixed degree: training RMSE (registered method), leave-one-out RMSE
 * (hat matrix of the penalized fit with an unpenalized intercept, exact for fixed λ), error
 * against the truth, tr H(λ) and the standardized slopes.
 */
export function ridgePath(
  pts: readonly Pt[],
  degree: number,
  fTrue: ((x: number) => number) | undefined,
  domain: readonly [number, number],
  lambdas: readonly number[] = LAMBDA_GRID,
): SweepPoint[] {
  if (degree < 1) return [];
  const xs = pts.map((p) => p.x);
  const V = vandermonde(xs, degree);
  const m = pts.length;
  const scale = Array.from({ length: degree }, (_, j) => {
    const col = V.map((row) => row[j + 1]);
    const mean = col.reduce((s, v) => s + v, 0) / m;
    const ss = col.reduce((s, v) => s + (v - mean) ** 2, 0);
    return ss > 0 ? Math.sqrt(ss) : 1;
  });
  return lambdas.map((lam) => {
    const res = runFit('ridge_regression', pts, { lam, degree });
    if (!res || !finiteVec(res.x)) return { at: lam, train: null, loo: null, truth: null };
    const beta = res.x;
    const h = hatDiagonal(V, [0, ...new Array<number>(degree).fill(lam)]);
    return {
      at: lam,
      train: res.extra.rmse as number,
      loo: looRmse(residuals(pts, beta), h),
      truth: trueError(beta, fTrue, domain),
      df: h ? h.reduce((s, v) => s + v, 0) : null,
      gamma: beta.slice(1).map((b, j) => b * scale[j]),
    };
  });
}

/** Ridge fits at one λ per decade (the λ path drawn in data space). */
export function ridgeFan(pts: readonly Pt[], degree: number): { lam: number; beta: number[] }[] {
  const out: { lam: number; beta: number[] }[] = [];
  for (let e = -8; e <= 4; e += 2) {
    const res = runFit('ridge_regression', pts, { lam: 10 ** e, degree });
    if (res && finiteVec(res.x)) out.push({ lam: 10 ** e, beta: res.x });
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// Data editing
// ---------------------------------------------------------------------------------------

/** Round a coordinate to 4 significant digits (what a drag can resolve; keeps URLs short). */
export const snap = (v: number) => Number(v.toPrecision(4));

export function encodePoints(pts: readonly Pt[]): number[] {
  return pts.flatMap((p) => [p.x, p.y]);
}

export function decodePoints(v: readonly number[] | null | undefined): Pt[] | null {
  if (!v || v.length < 2 || v.length % 2 !== 0) return null;
  const out: Pt[] = [];
  for (let i = 0; i < v.length; i += 2) {
    if (!Number.isFinite(v[i]) || !Number.isFinite(v[i + 1])) return null;
    out.push({ x: v[i], y: v[i + 1] });
  }
  return out;
}

/** A short stable hash of the data (problem id of an edited dataset → memoized runs). */
export function dataKey(pts: readonly Pt[]): string {
  let h = 2166136261;
  for (const v of encodePoints(pts)) {
    const s = String(v);
    for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
    h = Math.imul(h ^ 44, 16777619);
  }
  return (h >>> 0).toString(36);
}

/** `x = [...]` and `y = [...]` in Python, for the call that reproduces an edited dataset. */
export function pythonData(pts: readonly Pt[]): string {
  const f = (v: number) => (Number.isInteger(v) ? `${v}.0` : String(v));
  return `x = [${pts.map((p) => f(p.x)).join(', ')}]\ny = [${pts.map((p) => f(p.y)).join(', ')}]`;
}

/** The loss a method puts on one residual r (Huber in units of σ̂, as Python's objective). */
export function lossFunction(
  kind: Kind,
  huber?: { delta: number; scale: number } | null,
): (r: number) => number {
  if (kind === 'huber' && huber && huber.scale > 0)
    return (r) => huberRho(r / huber.scale, huber.delta);
  if (kind === 'lad' || kind === 'minimax') return Math.abs;
  return (r) => r * r;
}
