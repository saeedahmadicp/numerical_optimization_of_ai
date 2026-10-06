/**
 * Pure geometry and number helpers of the differentiation lab (no DOM, unit-tested).
 *
 * View coordinates of the stencil plot: u = x − x0 (the actual, rounded abscissa minus x0) and
 *   y = f(x) − f(x0)                    (mode 'f'),
 *   y = f(x) − f(x0) − f′(x0)·(x − x0)  (mode 'dev': f minus its tangent ℓ at x0).
 * In 'dev' mode a chord's slope is its error D(h) − f′(x0), and the tangent is the u-axis.
 */
import type { Step } from '../../core/types';

export type CurveMode = 'f' | 'dev';

/** Methods whose estimate is f″(x0) instead of f′(x0). */
export const SECOND_DERIVATIVE = new Set(['second_derivative_central']);

/** Pairs of stencil indices that form the chords a method's formula uses. */
export const CHORDS: Readonly<Record<string, readonly (readonly [number, number])[]>> = {
  forward_difference: [[0, 1]],
  backward_difference: [[0, 1]],
  central_difference: [[0, 1]],
  // Inner chord (x0 ± h) and outer chord (x0 ± 2h): D = (4/3)·C(h) − (1/3)·C(2h).
  five_point_stencil: [
    [1, 2],
    [0, 3],
  ],
  richardson_extrapolation: [[0, 1]],
  second_derivative_central: [],
  complex_step: [],
};

/** Largest |offset| of a method's stencil in units of h (the zoom keeps it in view). */
export const REACH: Readonly<Record<string, number>> = {
  five_point_stencil: 2,
};

export interface ViewPoint {
  u: number;
  y: number;
}

export interface StencilGeometry {
  /** The stencil points f(x0 + oᵢh) in view coordinates (empty for the complex step). */
  points: ViewPoint[];
  /** Chords between stencil points. */
  chords: [ViewPoint, ViewPoint][];
  /** The estimate as a line through (x0, f(x0)): y = slope·u (first-derivative methods). */
  slope: number | null;
  /** Second derivative: y = a·u² + b·u + c through the three stencil points. */
  parabola: { a: number; b: number; c: number } | null;
}

/** y of a point in view coordinates. */
export function viewY(fx: number, u: number, f0: number, s: number, mode: CurveMode): number {
  const y = fx - f0;
  return mode === 'dev' ? y - s * u : y;
}

/**
 * The geometry a method's step draws: stencil dots, chords, the estimate line through x0 (slope
 * D − f′ in 'dev' mode) or the interpolating parabola (second derivative).
 */
export function stencilGeometry(
  methodId: string,
  step: Step,
  f0: number,
  s: number,
  mode: CurveMode,
): StencilGeometry {
  const x0 = step.info.x0 as number;
  const estimate = step.info.estimate as number | null;
  const stencil = (step.info.stencil as [number, number | null][]) ?? [];
  if (methodId === 'complex_step') {
    const d = estimate ?? NaN;
    return {
      points: [],
      chords: [],
      slope: Number.isFinite(d) ? (mode === 'dev' ? d - s : d) : null,
      parabola: null,
    };
  }
  const points = stencil.map(([x, fx]) => {
    const u = x - x0;
    return { u, y: fx === null ? NaN : viewY(fx, u, f0, s, mode) };
  });
  const chords = (CHORDS[methodId] ?? [])
    .filter(([i, j]) => i < points.length && j < points.length)
    .map(([i, j]) => [points[i], points[j]] as [ViewPoint, ViewPoint]);
  if (SECOND_DERIVATIVE.has(methodId)) {
    return { points, chords, slope: null, parabola: parabolaThrough(points) };
  }
  const d = estimate ?? NaN;
  return {
    points,
    chords,
    slope: Number.isFinite(d) ? (mode === 'dev' ? d - s : d) : null,
    parabola: null,
  };
}

/** The parabola through three points (Lagrange), or null when it is undefined. */
export function parabolaThrough(
  p: readonly ViewPoint[],
): { a: number; b: number; c: number } | null {
  if (p.length !== 3 || p.some((q) => !Number.isFinite(q.y))) return null;
  const [{ u: x1, y: y1 }, { u: x2, y: y2 }, { u: x3, y: y3 }] = p;
  const d = (x1 - x2) * (x1 - x3) * (x2 - x3);
  if (d === 0) return null;
  const a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / d;
  const b = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / d;
  const c = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / d;
  return { a, b, c };
}

/** The unit in the last place of x (the spacing of doubles at |x|). */
export function ulp(x: number): number {
  const a = Math.abs(x);
  if (!Number.isFinite(a)) return NaN;
  if (a < Number.MIN_VALUE * 2 ** 52) return Number.MIN_VALUE;
  return 2 ** (Math.floor(Math.log2(a)) - 52);
}

/** Continuous step h(t) = h0·2^(−t) of a sweep (the zoom of the stencil view follows it). */
export function hAt(trace: readonly Step[], t: number): number {
  const h0 = trace[0]?.info.h as number | undefined;
  if (!h0) return NaN;
  const tt = Math.max(0, Math.min(t, trace.length - 1));
  return h0 * 2 ** -tt;
}

/** The error column of a trace (null where unknown or non-finite). */
export function errorsOf(trace: readonly Step[]): (number | null)[] {
  return trace.map((s) => {
    const e = s.info.error;
    return typeof e === 'number' && Number.isFinite(e) ? e : null;
  });
}

/**
 * Format numbers with a shared number of decimals so that their common leading digits line up;
 * returns the strings and the length of the common prefix (the digits that cancel).
 */
export function alignedDigits(
  values: readonly number[],
  significant = 16,
): {
  text: string[];
  common: number;
} {
  const finite = values.filter(Number.isFinite);
  if (!finite.length) return { text: values.map(() => '—'), common: 0 };
  const big = Math.max(...finite.map(Math.abs));
  const mag = big === 0 ? 0 : Math.floor(Math.log10(big));
  const decimals = Math.max(0, Math.min(20, significant - 1 - mag));
  const raw = values.map((v) => (Number.isFinite(v) ? v.toFixed(decimals) : '—'));
  // Right-align on the decimal point (same decimals, so pad the integer part).
  const width = Math.max(...raw.map((s) => s.length));
  const text = raw.map((s) => s.padStart(width, ' ').replace('-', '−'));
  let common = 0;
  if (finite.length === values.length && values.length > 1) {
    const first = text[0];
    outer: for (; common < first.length; common++) {
      for (const s of text) if (s[common] !== first[common]) break outer;
    }
  }
  return { text, common };
}

/**
 * Significant digits lost to cancellation in Σ cᵢ fᵢ: log₁₀(max |fᵢ| / |Σ cᵢ fᵢ|), clamped at 0.
 * NaN when undefined; Infinity when the sum is exactly 0 (every digit cancelled).
 */
export function digitsLost(fs: readonly number[], weights: readonly number[]): number {
  if (fs.length !== weights.length || fs.some((v) => !Number.isFinite(v))) return NaN;
  const big = Math.max(...fs.map(Math.abs));
  if (big === 0) return 0;
  // Integer weights (the weights are cᵢ/h^q; their ratios are the integer stencil weights).
  const wMax = Math.max(...weights.map(Math.abs));
  let num = 0;
  fs.forEach((v, i) => (num += (weights[i] / wMax) * v));
  if (num === 0) return Infinity;
  return Math.max(0, Math.log10(big / Math.abs(num)));
}

/**
 * Why a weighted sum Σ cᵢ fᵢ is exactly 0: 'equal' when every fᵢ is bitwise the same value;
 * 'balanced' for a three-point second difference whose two first differences agree to the last
 * bit (they are exact by Sterbenz's lemma, so the test is exact); 'rounded' otherwise.
 */
export function zeroNumeratorKind(
  fs: readonly number[],
  second: boolean,
): 'equal' | 'balanced' | 'rounded' {
  if (fs.every((v) => Object.is(v, fs[0]))) return 'equal';
  if (second && fs.length === 3 && fs[2] - fs[1] === fs[1] - fs[0]) return 'balanced';
  return 'rounded';
}

/** A number for KaTeX: `2.71828`, `1.25\times10^{-8}`. */
export function texNum(v: number | null | undefined, digits = 6, keepZeros = false): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '\\text{—}';
  if (!Number.isFinite(v)) return v > 0 ? '\\infty' : '-\\infty';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e5) {
    const [mant, exp] = v.toExponential(Math.max(0, digits - 1)).split('e');
    const m = mant.replace(/\.?0+$/, '');
    const e = Number(exp);
    return `${m === '1' ? '' : m === '-1' ? '-' : `${m}\\times`}10^{${e}}`;
  }
  const s = v.toPrecision(digits);
  return keepZeros || !s.includes('.') ? s : s.replace(/\.?0+$/, '');
}

/** Half-width of the stencil view for step h: the outermost stencil point (reach·h) plus 1.6h. */
export const windowHalfWidth = (h: number, reach = 1) => h * (Math.max(1, reach) + 1.6);
