/**
 * Pure mathematics behind the interpolation lab: node layouts, the active dataset (layout,
 * node count, edited nodes), the nodal polynomial ω(x) = ∏(x − xᵢ), the Lagrange basis and the
 * Lebesgue function, and an exact evaluator for the approximant of every step of every method
 * (read from Step.x / Result.extra, so what is drawn is what the method computed). For the
 * rational methods: AAA's support points and poles per step, and the local polynomials that
 * Floater–Hormann blends.
 */
import type { Result, Step } from '../../core/types';
import type { DataProblem } from '../../problems/data';
import { chebyshevNodesFirstKind, linspace } from '../../problems/data';
import {
  barycentricEval,
  clenshaw,
  newtonEval,
  ppEval,
  toUnit,
} from '../../methods/interpolation/methods';

export type Layout = 'data' | 'equi' | 'cheb';
export const MIN_NODES = 2;
/** Plot margins (px) shared by the stage and the strip, so their x-axes line up. */
export const PLOT_MARGIN = { left: 46, right: 16, top: 18, bottom: 28 };
export const MAX_NODES = 40;

export type Fn = (x: number) => number;

/** The dataset the methods run on: a library dataset, resampled, or edited by hand. */
export interface ActiveData {
  /** Unique per (x, y): the run cache keys on it. The library id when nothing was changed. */
  id: string;
  /** The library dataset it comes from. */
  baseId: string;
  name: string;
  latex: string;
  description: string;
  x: number[];
  y: number[];
  fTrue?: Fn;
  domain: [number, number];
  /** Nodes were moved, added or removed by hand. */
  edited: boolean;
  /** Which layout produced the nodes ('data' = the library's own nodes or an edit). */
  layout: Layout;
}

export function equispacedNodes(n: number, a: number, b: number): number[] {
  return n === 1 ? [(a + b) / 2] : linspace(a, b, n);
}

/** Roots of T_n mapped to [a, b], ascending (Burden & Faires, Theorem 8.9). */
export function chebyshevNodes(n: number, a: number, b: number): number[] {
  return chebyshevNodesFirstKind(n, a, b);
}

export function layoutNodes(layout: Exclude<Layout, 'data'>, n: number, a: number, b: number) {
  return layout === 'cheb' ? chebyshevNodes(n, a, b) : equispacedNodes(n, a, b);
}

/** Classify a node set: equispaced or Chebyshev (to 1e-9 of the span), else 'data'. */
export function detectLayout(x: readonly number[], a: number, b: number): Layout {
  const s = [...x].sort((p, q) => p - q);
  const tol = 1e-9 * (b - a);
  const near = (ref: number[]) => ref.every((r, i) => Math.abs(r - s[i]) <= tol);
  if (s.length >= 2 && near(equispacedNodes(s.length, a, b))) return 'equi';
  if (s.length >= 1 && near(chebyshevNodes(s.length, a, b))) return 'cheb';
  return 'data';
}

/** FNV-1a over the bit patterns of the data: a short, stable key for an edited dataset. */
export function dataKey(x: readonly number[], y: readonly number[]): string {
  let h = 0x811c9dc5;
  const buf = new Float64Array(1);
  const bytes = new Uint8Array(buf.buffer);
  for (const v of [...x, ...y]) {
    buf[0] = v;
    for (const byte of bytes) {
      h ^= byte;
      h = Math.imul(h, 0x01000193);
    }
  }
  return (h >>> 0).toString(36);
}

/**
 * The dataset for the current controls. `points` (edited nodes) wins over the layout; a layout
 * other than 'data' resamples f_true at n equispaced or Chebyshev nodes of the domain.
 */
export function activeData(
  base: DataProblem,
  layout: Layout,
  n: number,
  points: readonly number[] | null,
): ActiveData {
  const domain: [number, number] = base.domain ?? [Math.min(...base.x), Math.max(...base.x)];
  const common = {
    baseId: base.id,
    latex: base.latex,
    description: base.description,
    fTrue: base.fTrue,
    domain,
  };
  if (points && points.length >= 2 * MIN_NODES && points.length % 2 === 0) {
    const x: number[] = [],
      y: number[] = [];
    for (let i = 0; i < points.length; i += 2) {
      x.push(points[i]);
      y.push(points[i + 1]);
    }
    return {
      ...common,
      id: `${base.id}~${dataKey(x, y)}`,
      name: `${base.name} (edited)`,
      x,
      y,
      edited: true,
      layout: 'data',
    };
  }
  if (layout !== 'data' && base.fTrue) {
    const f = base.fTrue;
    const x = layoutNodes(layout, n, domain[0], domain[1]);
    const y = x.map((t) => f(t));
    return {
      ...common,
      id: `${base.id}~${layout}${n}`,
      name: `${base.name.split(', ')[0]}, ${n} ${layout === 'cheb' ? 'Chebyshev' : 'equispaced'} nodes`,
      x,
      y,
      edited: false,
      layout,
    };
  }
  return {
    ...common,
    id: base.id,
    name: base.name,
    x: [...base.x],
    y: [...base.y],
    edited: false,
    layout: 'data',
  };
}

// ── Polynomials of the node set ─────────────────────────────────────────────────────────

/** ω(x) = ∏ᵢ (x − xᵢ). */
export function nodalPoly(nodes: readonly number[]): Fn {
  return (x) => {
    let p = 1;
    for (const xi of nodes) p *= x - xi;
    return p;
  };
}

/** ℓⱼ(x) = ∏_{m≠j} (x − x_m)/(x_j − x_m). */
export function basisFn(nodes: readonly number[], j: number): Fn {
  return (x) => {
    let p = 1;
    for (let m = 0; m < nodes.length; m++) if (m !== j) p *= (x - nodes[m]) / (nodes[j] - nodes[m]);
    return p;
  };
}

/** Lebesgue function λ(x) = Σⱼ |ℓⱼ(x)|; its maximum is the Lebesgue constant Λₙ. */
export function lebesgueFn(nodes: readonly number[]): Fn {
  const basis = nodes.map((_, j) => basisFn(nodes, j));
  return (x) => basis.reduce((s, l) => s + Math.abs(l(x)), 0);
}

/** max |g| on [a, b] by dense sampling, with the argmax. */
export function maxAbs(g: Fn, a: number, b: number, samples = 2001): { value: number; at: number } {
  let value = 0,
    at = a;
  for (let i = 0; i < samples; i++) {
    const x = a + ((b - a) * i) / (samples - 1);
    const v = Math.abs(g(x));
    if (v > value || !Number.isFinite(v)) {
      value = v;
      at = x;
      if (!Number.isFinite(v)) break;
    }
  }
  return { value, at };
}

// ── Exact step evaluators ───────────────────────────────────────────────────────────────

/**
 * `aaa`: one support point per step (greedy, not in data order), poles per step.
 * `blend`: Floater–Hormann, one weight per sorted node; the interpolant exists at the last step.
 */
export type MethodKind = 'nodewise' | 'neville' | 'chebyshev' | 'piecewise' | 'aaa' | 'blend';

const NODEWISE = new Set(['lagrange', 'barycentric', 'newton_divided_differences']);
const PIECEWISE = new Set([
  'linear_spline',
  'cubic_spline_natural',
  'cubic_spline_clamped',
  'cubic_spline_not_a_knot',
  'pchip',
]);

export function methodKind(id: string): MethodKind {
  if (NODEWISE.has(id)) return 'nodewise';
  if (id === 'neville') return 'neville';
  if (id === 'chebyshev_interpolation') return 'chebyshev';
  if (PIECEWISE.has(id)) return 'piecewise';
  if (id === 'aaa') return 'aaa';
  if (id === 'floater_hormann') return 'blend';
  return 'nodewise';
}

const asNums = (v: unknown): number[] => (Array.isArray(v) ? (v as number[]) : []);

/** Barycentric weights of a node set (B&T eq. 3.2), for evaluators that only know nodes. */
function weightsOf(nodes: readonly number[]): number[] {
  return nodes.map((xj, j) => {
    let p = 1;
    for (let m = 0; m < nodes.length; m++) if (m !== j) p *= xj - nodes[m];
    return 1 / p;
  });
}

/**
 * The Lagrange partial sum p_k = Σ_{j≤k} y_j ℓ_j over the full node set, in O(n) per point:
 * ℓ_j(t) = ω(t) w_j/(t − x_j) (the first barycentric form, Higham 2004). `ys` holds y_0 … y_k.
 * At a node x_m it is y_m when m ≤ k and 0 otherwise (the cardinal property, exactly).
 */
export function lagrangePartial(nodes: readonly number[], ys: readonly number[]): Fn {
  const w = weightsOf(nodes);
  const n = nodes.length,
    k1 = Math.min(ys.length, n);
  return (t) => {
    let omega = 1,
      sum = 0;
    for (let j = 0; j < n; j++) {
      const d = t - nodes[j];
      if (d === 0) return j < k1 ? ys[j] : 0;
      omega *= d;
      if (j < k1) sum += (w[j] * ys[j]) / d;
    }
    return omega * sum;
  };
}

/** Linear interpolation in a 200-point `info.curve` on [a, b] (fallback). */
function gridFn(curve: readonly number[], a: number, b: number): Fn {
  const n = curve.length;
  return (x) => {
    const u = ((x - a) / (b - a)) * (n - 1);
    const i = Math.min(n - 2, Math.max(0, Math.floor(u)));
    const w = u - i;
    return curve[i] * (1 - w) + curve[i + 1] * w;
  };
}

/**
 * The approximant of step `k` as a function of x, or null when the step has none (spline
 * stages before the coefficients). `data` supplies the nodes in the method's order (the
 * polynomial forms use the data order; extra.nodes is used when present).
 */
export function stepFunction(
  methodId: string,
  result: Result,
  k: number,
  data: Pick<ActiveData, 'x' | 'y' | 'domain'>,
): Fn | null {
  const s: Step | undefined = result.trace[k];
  if (!s) return null;
  const nodes = asNums(result.extra.nodes).length ? asNums(result.extra.nodes) : data.x;
  const [a, b] = (result.extra.domain as [number, number] | undefined) ?? data.domain;
  const x = asNums(s.x);
  switch (methodId) {
    case 'lagrange':
      return lagrangePartial(data.x, x);
    case 'barycentric': {
      const nd = data.x.slice(0, x.length),
        vals = data.y.slice(0, x.length);
      return (t) => barycentricEval(nd, x, vals, [t])[0];
    }
    case 'newton_divided_differences': {
      const nd = data.x.slice(0, x.length);
      return (t) => newtonEval(nd, x, [t])[0];
    }
    case 'neville': {
      // Column k: P_{0..k}, the interpolant through the first k + 1 nodes.
      const nd = data.x.slice(0, k + 1),
        vals = data.y.slice(0, k + 1);
      const w = weightsOf(nd);
      return (t) => barycentricEval(nd, w, vals, [t])[0];
    }
    case 'chebyshev_interpolation':
      return (t) => clenshaw(x, toUnit([t], a, b))[0];
    case 'aaa': {
      // Step 0 is the constant mean f; step m the type (m−1, m−1) barycentric rational r_m.
      const z = asNums(s.info.support);
      if (!z.length) {
        const c = asNums(s.info.curve)[0];
        return c === undefined ? null : () => c;
      }
      const w = asNums(s.info.weights),
        f = asNums(s.info.support_values);
      return (t) => barycentricEval(z, w, f, [t])[0];
    }
    case 'floater_hormann': {
      // The weights are complete only at the last step: no interpolant before it.
      if (k < result.trace.length - 1) return null;
      const w = asNums(result.extra.coefficients),
        vals = asNums(result.extra.values);
      if (!w.length || w.length !== nodes.length) return null;
      return (t) => barycentricEval(nodes, w, vals, [t])[0];
    }
    default: {
      if (s.info.stage !== 'coefficients') return null;
      const coef = s.info.coefficients as number[][] | undefined;
      if (coef && nodes.length === coef.length + 1) return (t) => ppEval(nodes, coef, [t])[0];
      const curve = asNums(s.info.curve);
      return curve.length ? gridFn(curve, a, b) : null;
    }
  }
}

/** The term this step adds: yₖℓₖ (Lagrange), pₖ − pₖ₋₁ (Newton, barycentric), cₖTₖ (Chebyshev). */
export function stepTerm(
  methodId: string,
  result: Result,
  k: number,
  data: Pick<ActiveData, 'x' | 'y' | 'domain'>,
): Fn | null {
  if (k < 1 && methodId !== 'lagrange') return null;
  const s = result.trace[k];
  if (!s) return null;
  const x = asNums(s.x);
  const [a, b] = (result.extra.domain as [number, number] | undefined) ?? data.domain;
  switch (methodId) {
    case 'lagrange': {
      const l = basisFn(data.x, k);
      return (t) => x[k] * l(t);
    }
    case 'newton_divided_differences': {
      const nd = data.x.slice(0, k);
      const c = x[k];
      return (t) => c * nd.reduce((p, xi) => p * (t - xi), 1);
    }
    case 'barycentric': {
      const cur = stepFunction(methodId, result, k, data);
      const prev = stepFunction(methodId, result, k - 1, data);
      return cur && prev ? (t) => cur(t) - prev(t) : null;
    }
    case 'chebyshev_interpolation': {
      const c = x[k];
      return (t) => {
        const u = (2 * t - (a + b)) / (b - a);
        // T_k(u) = cos(k arccos u) on [−1, 1]; the recurrence outside.
        let t0 = 1,
          t1 = u;
        if (k === 0) return c;
        for (let i = 2; i <= k; i++) [t0, t1] = [t1, 2 * u * t1 - t0];
        return c * t1;
      };
    }
    default:
      return null;
  }
}

// ── Rational methods ────────────────────────────────────────────────────────────────────

/** Data indices of AAA's support points after step k, in the order chosen (step 1 first). */
export function aaaSupport(result: Result, k: number): number[] {
  const out: number[] = [];
  for (let i = 1; i <= Math.min(k, result.trace.length - 1); i++) {
    const j = result.trace[i].info.node_index;
    if (typeof j === 'number') out.push(j);
  }
  return out;
}

export interface Pole {
  re: number;
  im: number;
  /** |residue| < 10⁻¹³·max|f|: a numerical Froissart doublet (NST 2018, §5). */
  doublet: boolean;
}

/** The poles of AAA's r after step k, with the doublet flag from the residues. */
export function aaaPoles(result: Result, k: number, fScale: number): Pole[] {
  const s = result.trace[Math.min(k, result.trace.length - 1)];
  if (!s) return [];
  const poles = (s.info.poles as number[][] | undefined) ?? [];
  const res = (s.info.residues as number[][] | undefined) ?? [];
  const tol = 1e-13 * Math.max(fScale, Number.MIN_VALUE);
  return poles
    .map((p, i) => ({
      re: p[0],
      im: p[1],
      doublet: res[i] ? Math.hypot(res[i][0], res[i][1]) < tol : false,
    }))
    .filter((p) => Number.isFinite(p.re) && Number.isFinite(p.im));
}

/** Certified real poles of AAA's r in [a, b] after step k (sign changes of the denominator). */
export function aaaIntervalPoles(result: Result, k: number): number[] {
  const s = result.trace[Math.min(k, result.trace.length - 1)];
  return s ? asNums(s.info.interval_poles) : [];
}

/**
 * Floater–Hormann's window at step k: the local polynomials p_i, i ∈ J_k = [lo, hi], each
 * through the d + 1 sorted nodes x_i … x_{i+d}, that contain node k.
 */
export function blendWindow(
  result: Result,
  k: number,
): { lo: number; hi: number; d: number } | null {
  const s = result.trace[k];
  const win = s ? asNums(s.info.window) : [];
  const d = result.extra.d;
  if (win.length !== 2 || typeof d !== 'number') return null;
  return { lo: win[0], hi: win[1], d };
}

/** The local interpolant p_i through the sorted nodes x_i … x_{i+d} (barycentric form). */
export function localPoly(
  nodes: readonly number[],
  values: readonly number[],
  i: number,
  d: number,
): Fn {
  const nd = nodes.slice(i, i + d + 1),
    vals = values.slice(i, i + d + 1);
  const w = weightsOf(nd);
  return (t) => barycentricEval(nd, w, vals, [t])[0];
}

/**
 * The y-range of the main view: the data and f (padded), widened to show the final curves
 * but never beyond `reach` data spans on either side (a Runge curve leaves the view and is
 * labelled instead of squashing the data).
 */
export function viewRange(
  ys: readonly number[],
  curves: readonly (readonly number[])[],
  reach = 1.25,
): [number, number] {
  const fin = ys.filter(Number.isFinite);
  let lo = Math.min(...fin),
    hi = Math.max(...fin);
  if (!(hi > lo)) {
    const d = Math.abs(lo) > 0 ? Math.abs(lo) * 0.5 : 1;
    lo -= d;
    hi += d;
  }
  const span = hi - lo;
  let cLo = lo,
    cHi = hi;
  for (const c of curves)
    for (const v of c) {
      if (!Number.isFinite(v)) continue;
      if (v < cLo) cLo = v;
      if (v > cHi) cHi = v;
    }
  cLo = Math.max(cLo, lo - reach * span);
  cHi = Math.min(cHi, hi + reach * span);
  const pad = 0.08 * (cHi - cLo);
  return [cLo - pad, cHi + pad];
}

/** Local extrema beyond [lo, hi] of a curve sampled uniformly on [a, b]: where it leaves the view. */
export function offViewPeaks(
  ys: ArrayLike<number>,
  a: number,
  b: number,
  lo: number,
  hi: number,
): { x: number; y: number }[] {
  const out: { x: number; y: number }[] = [];
  let run: { x: number; y: number } | null = null;
  const n = ys.length;
  for (let i = 0; i < n; i++) {
    const x = a + ((b - a) * i) / (n - 1);
    const y = ys[i];
    const outside = Number.isFinite(y) && (y > hi || y < lo);
    if (outside) {
      if (!run || Math.abs(y) > Math.abs(run.y)) run = { x, y };
    } else if (run) {
      out.push(run);
      run = null;
    }
  }
  if (run) out.push(run);
  return out;
}

/** Flatten nodes for the URL: x₀, y₀, x₁, y₁, … */
export function flattenPoints(x: readonly number[], y: readonly number[]): number[] {
  return x.flatMap((xi, i) => [xi, y[i]]);
}

/** Round a coordinate for the URL and the labels (6 significant digits). */
export const round6 = (v: number) => Number(v.toPrecision(6));

// ── Sampling cache ──────────────────────────────────────────────────────────────────────

const CACHE = new Map<string, Float64Array>();
const CACHE_MAX = 256;

/**
 * `g` on `samples` uniform points of [a, b], cached under `key` (a step's curve does not change
 * while the playhead moves through it, so playback redraws read the cache).
 */
export function sampled(key: string, g: Fn, a: number, b: number, samples = 601): Float64Array {
  const full = `${key}|${a}|${b}|${samples}`;
  const hit = CACHE.get(full);
  if (hit) return hit;
  const out = new Float64Array(samples);
  for (let i = 0; i < samples; i++) out[i] = g(a + ((b - a) * i) / (samples - 1));
  if (CACHE.size >= CACHE_MAX) CACHE.delete(CACHE.keys().next().value as string);
  CACHE.set(full, out);
  return out;
}

/** Chebyshev nodes minimize max|ω| over [a, b]: 2·((b − a)/4)ⁿ (Burden & Faires, Thm 8.10). */
export function chebyshevOmegaMax(n: number, a: number, b: number): number {
  return 2 * ((b - a) / 4) ** n;
}

/**
 * A new node for the "Add node" action: the midpoint of the widest gap between consecutive
 * nodes (or between the outermost node and the end of the domain), with y = f(x) when `f` is
 * given and finite, else the linear interpolant of its neighbors (the nearest y at an end).
 */
export function widestGapNode(
  x: readonly number[],
  y: readonly number[],
  domain: [number, number],
  f?: Fn,
): { x: number; y: number } | null {
  if (x.length === 0) return null;
  const order = x.map((_, i) => i).sort((i, j) => x[i] - x[j]);
  const [a, b] = domain;
  let best = { w: -1, x: NaN, y: NaN };
  const consider = (lo: number, hi: number, ylo: number, yhi: number) => {
    const w = hi - lo;
    if (w > best.w) best = { w, x: (lo + hi) / 2, y: (ylo + yhi) / 2 };
  };
  const first = order[0],
    last = order[order.length - 1];
  if (x[first] > a) consider(a, x[first], y[first], y[first]);
  for (let k = 0; k + 1 < order.length; k++) {
    const i = order[k],
      j = order[k + 1];
    consider(x[i], x[j], y[i], y[j]);
  }
  if (x[last] < b) consider(x[last], b, y[last], y[last]);
  if (!(best.w > 1e-4 * (b - a))) return null;
  const nx = round6(best.x);
  const fy = f ? f(nx) : NaN;
  return { x: nx, y: round6(Number.isFinite(fy) ? fy : best.y) };
}
