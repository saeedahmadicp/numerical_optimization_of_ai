/**
 * Pure geometry of the line-search lab: the line function φ(α) = f(𝐱₀ + α𝐩), the acceptance
 * sets of each kind, the α window of the φ panel, the minimizer of φ in view, and the zoom
 * interpolant that produced a strong Wolfe trial — rebuilt from the trace, so what the lab draws
 * is exactly what the method computed (tests: geometry.test.ts).
 *
 * The acceptance tests repeat the comparisons of `numopt.line_search.methods` (difference form,
 * φ(α) − φ(0) ≤ c₁αφ'(0)), so a shaded interval and the method's verdict at a trial agree.
 */
import type { Step, Vector } from '../../core/types';
import {
  cubicMinimizer,
  quadraticMinimizer,
  type LineSearchKind,
} from '../../methods/line_search/methods';

/** The exact-quadratic demo confirms a rise by |φ'(α)| ≤ 0.1|φ'(0)| (Python `_EXACT_C2`). */
export const EXACT_C2 = 0.1;
/** Zoom safeguard δ: an interpolated trial is clamped into [a + δw, b − δw] (Python `_SAFEGUARD`). */
export const SAFEGUARD = 0.1;

export interface LineFn {
  x0: Vector;
  p: Vector;
  point: (alpha: number) => Vector;
  phi: (alpha: number) => number;
  dphi: (alpha: number) => number;
}

interface SmoothF {
  f: (x: Vector) => number;
  grad?: (x: Vector) => Vector;
}

/** φ(α) = f(𝐱₀ + α𝐩) and φ'(α) = ∇f(𝐱₀ + α𝐩)ᵀ𝐩 (a central difference when f has no ∇f). */
export function makeLine(problem: SmoothF, x0: readonly number[], p: readonly number[]): LineFn {
  const xs = [...x0];
  const ps = [...p];
  const point = (alpha: number) => xs.map((xi, i) => xi + alpha * ps[i]);
  const phi = (alpha: number) => Number(problem.f(point(alpha)));
  const dphi = (alpha: number) => {
    if (problem.grad) {
      const g = problem.grad(point(alpha));
      let s = 0;
      for (let i = 0; i < ps.length; i++) s += g[i] * ps[i];
      return s;
    }
    const h = 1e-6 * Math.max(1, Math.abs(alpha));
    return (phi(alpha + h) - phi(alpha - h)) / (2 * h);
  };
  return { x0: xs, p: ps, point, phi, dphi };
}

// ── Reading a trial ────────────────────────────────────────────────────────────────────

export interface Trial {
  k: number;
  alpha: number;
  phi: number;
  /** φ'(α); null when the search did not evaluate ∇f at this trial. */
  dphi: number | null;
  phi0: number;
  dphi0: number;
  c1: number;
  c2: number | null;
  phase: string;
  /** [lo, hi] the trial was chosen from (hi may be ∞); null for start / backtrack / exact. */
  interval: [number, number] | null;
  accepted: boolean;
  conditions: Record<string, boolean | null>;
  alphaLo: number | null;
  alphaHi: number | null;
  interp: string | null;
  rho: number | null;
  pHp: number | null;
}

const num = (v: unknown): number => (typeof v === 'number' ? v : NaN);
const numOrNull = (v: unknown): number | null =>
  typeof v === 'number' && !Number.isNaN(v) ? v : null;

/** A typed view of one Step.info of a line-search demo. */
export function trialOf(step: Step): Trial {
  const i = step.info;
  const interval = Array.isArray(i.interval)
    ? ([num(i.interval[0]), num(i.interval[1])] as [number, number])
    : null;
  return {
    k: step.k,
    alpha: num(i.alpha),
    phi: num(i.phi),
    dphi: numOrNull(i.dphi),
    phi0: num(i.phi0),
    dphi0: num(i.dphi0),
    c1: num(i.c1),
    c2: numOrNull(i.c2),
    phase: typeof i.phase === 'string' ? i.phase : '',
    interval,
    accepted: i.accepted === true,
    conditions: (i.conditions ?? {}) as Record<string, boolean | null>,
    alphaLo: numOrNull(i.alpha_lo),
    alphaHi: numOrNull(i.alpha_hi),
    interp: typeof i.interp === 'string' ? i.interp : null,
    rho: numOrNull(i.rho),
    pHp: numOrNull(i.pHp),
  };
}

// ── Acceptance tests (Python `_armijo`, `_curvature`, …) ──────────────────────────────────

export const armijo = (a: number, phi: number, phi0: number, dphi0: number, c: number) =>
  Number.isFinite(phi) && phi - phi0 <= c * a * dphi0;
export const goldsteinLower = (a: number, phi: number, phi0: number, dphi0: number, c: number) =>
  Number.isFinite(phi) && phi - phi0 >= (1.0 - c) * a * dphi0;
export const curvature = (dphi: number, dphi0: number, c2: number) =>
  Number.isFinite(dphi) && dphi >= c2 * dphi0;
export const strongCurvature = (dphi: number, dphi0: number, c2: number) =>
  Number.isFinite(dphi) && Math.abs(dphi) <= -c2 * dphi0;
export const decrease = (phi: number, phi0: number) => Number.isFinite(phi) && phi <= phi0;

/** Whether `kind` accepts the step α (its full acceptance conditions, not one test). */
export function accepts(
  kind: LineSearchKind,
  alpha: number,
  phi: number,
  dphi: number,
  phi0: number,
  dphi0: number,
  c1: number,
  c2: number,
): boolean {
  switch (kind) {
    case 'backtracking':
      return armijo(alpha, phi, phi0, dphi0, c1);
    case 'strong_wolfe':
      return armijo(alpha, phi, phi0, dphi0, c1) && strongCurvature(dphi, dphi0, c2);
    case 'weak_wolfe':
      return armijo(alpha, phi, phi0, dphi0, c1) && curvature(dphi, dphi0, c2);
    case 'goldstein':
      return armijo(alpha, phi, phi0, dphi0, c1) && goldsteinLower(alpha, phi, phi0, dphi0, c1);
    case 'exact_quadratic':
      return decrease(phi, phi0);
  }
}

/** Whether the kind's acceptance needs φ'(α) (Wolfe kinds). */
export const needsSlope = (kind: LineSearchKind) =>
  kind === 'strong_wolfe' || kind === 'weak_wolfe';

/** A sampled α grid on [lo, hi] with φ and φ' (φ' only when asked). */
export interface Samples {
  alpha: Float64Array;
  phi: Float64Array;
  dphi: Float64Array | null;
}

export function sampleLine(line: LineFn, lo: number, hi: number, n: number, slope = true): Samples {
  const alpha = new Float64Array(n + 1);
  const phi = new Float64Array(n + 1);
  const dphi = slope ? new Float64Array(n + 1) : null;
  for (let i = 0; i <= n; i++) {
    const a = lo + ((hi - lo) * i) / n;
    alpha[i] = a;
    phi[i] = line.phi(a);
    if (dphi) dphi[i] = line.dphi(a);
  }
  return { alpha, phi, dphi };
}

/**
 * The α intervals (within the sampled range) where `kind` accepts the step, as maximal runs of
 * accepted grid points; each end is refined by bisection on the acceptance test.
 */
export function acceptableIntervals(
  kind: LineSearchKind,
  line: LineFn,
  s: Samples,
  phi0: number,
  dphi0: number,
  c1: number,
  c2: number,
): [number, number][] {
  const ok = (a: number, phi: number, dphi: number) =>
    a > 0 && accepts(kind, a, phi, dphi, phi0, dphi0, c1, c2);
  const okAt = (a: number) => ok(a, line.phi(a), needsSlope(kind) ? line.dphi(a) : NaN);
  const n = s.alpha.length;
  const at = (i: number) => ok(s.alpha[i], s.phi[i], s.dphi ? s.dphi[i] : line.dphi(s.alpha[i]));
  const edge = (a: number, b: number, inside: 'a' | 'b') => {
    // Bisection between an accepted and a rejected neighbor.
    let lo = a,
      hi = b;
    for (let it = 0; it < 30; it++) {
      const mid = lo + (hi - lo) / 2;
      if (okAt(mid) === (inside === 'a')) lo = mid;
      else hi = mid;
    }
    return inside === 'a' ? lo : hi;
  };
  const out: [number, number][] = [];
  let start = -1;
  let prev = false;
  for (let i = 0; i < n; i++) {
    const cur = at(i);
    if (cur && !prev) start = i === 0 ? s.alpha[0] : edge(s.alpha[i - 1], s.alpha[i], 'b');
    if (!cur && prev) out.push([start, edge(s.alpha[i - 1], s.alpha[i], 'a')]);
    prev = cur;
  }
  if (prev) out.push([start, s.alpha[n - 1]]);
  return out;
}

// ── The first local minimizer of φ ────────────────────────────────────────────────────

/**
 * The first local minimizer α⋆ > 0 of φ on (0, hi]: the first grid point (a linear and a
 * logarithmic grid, so short and long steps are both resolved) where φ stops decreasing, refined
 * by golden section between its neighbors. This is the step an exact line search along the first
 * descent of φ would take. Null when φ still decreases at α = hi or is not finite there.
 */
export function lineMinimizer(line: LineFn, hi: number): { alpha: number; phi: number } | null {
  if (!(hi > 0) || !Number.isFinite(hi)) return null;
  const grid: number[] = [0];
  const n = 600;
  for (let i = 1; i <= n; i++) grid.push((hi * i) / n);
  for (let i = 0; i < n; i++) grid.push(hi * 10 ** (-8 + (8 * i) / n));
  grid.sort((a, b) => a - b);
  const vals = grid.map((a) => line.phi(a));
  let best = -1;
  for (let i = 1; i < grid.length - 1; i++) {
    if (!Number.isFinite(vals[i])) return null;
    // A real dip: below φ(0) by more than rounding, and φ rises after it.
    const below = vals[i] < vals[0] - 1e-12 * Math.max(1, Math.abs(vals[0]));
    if (below && vals[i] <= vals[i - 1] && vals[i] < vals[i + 1]) {
      best = i;
      break;
    }
  }
  if (best < 0) return null;
  let a = grid[best - 1];
  let b = grid[best + 1];
  const g = (Math.sqrt(5) - 1) / 2;
  let c = b - g * (b - a);
  let d = a + g * (b - a);
  let fc = line.phi(c);
  let fd = line.phi(d);
  for (let it = 0; it < 90 && b - a > 1e-15 * Math.max(1e-300, b); it++) {
    if (fc < fd) {
      b = d;
      d = c;
      fd = fc;
      c = b - g * (b - a);
      fc = line.phi(c);
    } else {
      a = c;
      c = d;
      fc = fd;
      d = a + g * (b - a);
      fd = line.phi(d);
    }
  }
  const alpha = (a + b) / 2;
  const phi = line.phi(alpha);
  return phi <= vals[best] ? { alpha, phi } : { alpha: grid[best], phi: vals[best] };
}

// ── The α window of the φ panel ───────────────────────────────────────────────────────

export type WindowMode = 'near' | 'all';

/**
 * The right end of the φ panel's α axis.
 *
 * - `near`: 1.6 × the largest accepted step (or the minimizer of φ), widened to show trials up to
 *   2.5 × that reference; longer trials are drawn as off-view markers.
 * - `all`: every trial step.
 * `fallback` is used when there is no trial at all (every search failed before its first trial).
 */
export function chooseWindow(
  trials: readonly (readonly number[])[],
  accepted: readonly (number | null)[],
  alphaStar: number | null,
  mode: WindowMode,
  fallback: number,
): number {
  const all = trials.flat().filter((a) => Number.isFinite(a) && a > 0);
  if (mode === 'all') {
    const top = Math.max(...all, alphaStar ?? 0);
    return top > 0 ? top * 1.04 : fallback;
  }
  const refs = accepted.filter((a): a is number => a !== null && a > 0);
  // The minimizer of φ joins the reference unless it lies far beyond every accepted step.
  if (alphaStar !== null && (!refs.length || alphaStar <= 3 * Math.max(...refs)))
    refs.push(alphaStar);
  if (!refs.length && all.length) refs.push(Math.min(...all));
  if (!refs.length) return fallback;
  const ref = Math.max(...refs);
  const near = all.filter((a) => a <= 2.5 * ref);
  return Math.max(1.6 * ref, near.length ? 1.06 * Math.max(...near) : 0);
}

/** A robust φ range for the panel: the well of φ with φ(0) in its upper third (`near`). */
export function phiRange(
  s: Samples,
  phi0: number,
  mode: WindowMode,
  extra: readonly number[] = [],
): [number, number] {
  const vals = [...s.phi].filter(Number.isFinite);
  if (!vals.length) return [phi0 - 1, phi0 + 1];
  const lo = Math.min(...vals, phi0);
  if (mode === 'all') {
    const hi = Math.max(...vals, phi0, ...extra.filter(Number.isFinite));
    const pad = (hi - lo) * 0.06 || 1;
    return [lo - pad, hi + pad];
  }
  const depth = phi0 - lo || Math.max(1e-12, Math.abs(phi0) * 1e-3, 1e-9);
  const top = Math.min(Math.max(...vals), phi0 + 0.6 * depth);
  return [lo - 0.14 * depth, Math.max(top, phi0 + 0.18 * depth)];
}

/** A φ' range that shows φ'(0), zero, the curvature band and most of the curve. */
export function slopeRange(dphi: Float64Array, dphi0: number, c2: number | null): [number, number] {
  const vals = [...dphi].filter(Number.isFinite).sort((a, b) => a - b);
  const m = Math.abs(dphi0) || 1;
  const q = (p: number) => (vals.length ? vals[Math.floor((vals.length - 1) * p)] : 0);
  const lo = Math.min(1.15 * dphi0, Math.max(q(0.02), -4 * m));
  const hi = Math.max(0.45 * m, (c2 ?? 0) * m * 1.3, Math.min(q(0.98), 4 * m));
  return [lo, hi];
}

// ── The zoom interpolant behind a strong Wolfe trial ──────────────────────────────────

export interface Interpolant {
  kind: 'cubic' | 'quadratic';
  /** α_lo and α_hi of Alg. 3.6 (not ordered). */
  aLo: number;
  aHi: number;
  /** The bracket [a, b] (ordered) and the safeguard interval [a + δw, b − δw]. */
  bracket: [number, number];
  safe: [number, number];
  /** The interpolant's minimizer before clamping (null when it has none). */
  tStar: number | null;
  /** How the method chose the trial: "cubic", "quadratic", "…_clamped" or "bisection". */
  how: string;
  /** The interpolating polynomial. */
  value: (alpha: number) => number;
}

/** φ, φ' at a step length the search visited before trial k (α = 0 is the start). */
function visited(trace: readonly Step[], k: number, alpha: number) {
  if (alpha === 0) {
    const t = trialOf(trace[0]);
    return { phi: t.phi0, dphi: t.dphi0 };
  }
  for (let j = k - 1; j >= 1; j--) {
    const t = trialOf(trace[j]);
    if (t.alpha === alpha) return { phi: t.phi, dphi: t.dphi };
  }
  return null;
}

/**
 * The interpolant N&W Alg. 3.6 minimized to choose zoom trial k: the cubic through φ, φ' at
 * α_lo and α_hi (eq. 3.59) when φ'(α_hi) is known, else the quadratic through φ(α_lo),
 * φ'(α_lo), φ(α_hi) (eq. 3.58). Null outside the zoom phase.
 */
export function zoomInterpolant(trace: readonly Step[], k: number): Interpolant | null {
  if (k < 1 || k >= trace.length) return null;
  const t = trialOf(trace[k]);
  if (t.phase !== 'zoom' || t.alphaLo === null || t.alphaHi === null || !t.interval) return null;
  const lo = visited(trace, k, t.alphaLo);
  const hi = visited(trace, k, t.alphaHi);
  if (!lo || !hi || lo.dphi === null) return null;
  const [a, b] = t.interval;
  const w = b - a;
  const safe: [number, number] = [a + SAFEGUARD * w, b - SAFEGUARD * w];
  const how = t.interp ?? 'bisection';
  const fa = lo.phi,
    da = lo.dphi,
    fb = hi.phi;
  const h = t.alphaHi - t.alphaLo;
  // The interpolant kind follows what was known: φ'(α_hi) for the cubic.
  const cubic = how.startsWith('cubic') || (how === 'bisection' && hi.dphi !== null);
  if (cubic && hi.dphi !== null) {
    const db = hi.dphi;
    const a0 = t.alphaLo;
    const value = (x: number) => {
      const s = (x - a0) / h;
      const s2 = s * s,
        s3 = s2 * s;
      return (
        (2 * s3 - 3 * s2 + 1) * fa +
        (s3 - 2 * s2 + s) * h * da +
        (-2 * s3 + 3 * s2) * fb +
        (s3 - s2) * h * db
      );
    };
    const tStar = cubicMinimizer(t.alphaLo, fa, da, t.alphaHi, fb, db);
    return {
      kind: 'cubic',
      aLo: t.alphaLo,
      aHi: t.alphaHi,
      bracket: [a, b],
      safe,
      tStar,
      how,
      value,
    };
  }
  if (!Number.isFinite(fb)) return null;
  const C = (fb - fa - da * h) / (h * h);
  const a0 = t.alphaLo;
  const value = (x: number) => fa + da * (x - a0) + C * (x - a0) * (x - a0);
  let tStar: number | null = null;
  try {
    tStar = quadraticMinimizer(t.alphaLo, fa, da, t.alphaHi, fb);
  } catch {
    // h² underflowed (the method fails the same way); draw the interpolant without a minimizer.
  }
  return {
    kind: 'quadratic',
    aLo: t.alphaLo,
    aHi: t.alphaHi,
    bracket: [a, b],
    safe,
    tStar,
    how,
    value,
  };
}

/** The quadratic model q(α) = φ(0) + αφ'(0) + ½α²𝐩ᵀ∇²f𝐩 of the exact-step demo. */
export function quadraticModel(t: Trial): ((alpha: number) => number) | null {
  const c = t.pHp;
  if (c === null || !Number.isFinite(c)) return null;
  return (a) => t.phi0 + a * t.dphi0 + 0.5 * a * a * c;
}

/** Short display name: "Strong Wolfe (bracket + zoom)" → "Strong Wolfe". */
export const shortName = (name: string) => name.replace(/\s*\(.*\)\s*$/, '');
