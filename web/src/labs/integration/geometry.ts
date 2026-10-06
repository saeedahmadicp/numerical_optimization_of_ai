/**
 * Quadrature geometry: the shapes a rule integrates, built from `Step.info` only (plus f for
 * the S₁ parabolas of pending adaptive intervals, which the trace does not store).
 *
 * Every rule is drawn as the area it actually integrates, so the shaded area IS the estimate:
 *
 *   Riemann / midpoint   rectangles of height f(node) on each subinterval
 *   trapezoid            chords between neighboring nodes
 *   Simpson / 3/8 / Boole the interpolating polynomial (degree 2 / 3 / 4) on each panel
 *   Romberg (row k)      the closed Newton–Cotes rule of column j = min(k, 2) on the 2^k panels:
 *                        R(k, 0) trapezoids, R(k, 1) Simpson arches, R(k, 2) Boole quartics.
 *                        For k ≤ 2 that is the estimate R(k, k) itself; from k = 3 the columns
 *                        j ≥ 3 are no longer interpolatory Newton–Cotes rules, so the panel draws
 *                        R(k, 2) and says so. New nodes of the row are marked.
 *   Gauss–Legendre       cells [cᵢ₋₁, cᵢ] with cᵢ = a + w̃₁ + … + w̃ᵢ and height f(xᵢ): their
 *                        total area is Σw̃ᵢf(xᵢ) = Gₘ, and each node lies inside its cell
 *                        (Chebyshev–Markov–Stieltjes separation theorem, Szegő §3.41). The cells
 *                        only add up to Gₘ: the pointwise gap between f and a cell means nothing.
 *                        The error is measured against the interpolant pₘ₋₁ through the nodes,
 *                        which Gₘ integrates exactly, so ∫(f − pₘ₋₁) = I − Gₘ (`errorTop`).
 *   Clenshaw–Curtis /    the interpolant p through the nodes of step k, which the rule integrates
 *   Gauss–Patterson      exactly (both rules are interpolatory), so the area under p IS the
 *                        estimate and ∫(f − p) = I − I_k. Dots have area ∝ weight; the nodes
 *                        that step k − 1 already evaluated are filled, the new ones are rings.
 *                        Clenshaw–Curtis also carries its semicircle: the node xⱼ is the shadow
 *                        of the point at angle jπ/n.
 *   adaptive Simpson     the quartic through the 5 nodes of each accepted interval (its
 *                        contribution S₂ + (S₂ − S₁)/15 is Boole's rule on those nodes)
 *   Monte Carlo          the rectangle of height f̄ = I_N/(b − a) over [a, b], ± one standard
 *                        error, and the samples (Uᵢ, f(Uᵢ))
 */
import type { Step } from '../../core/types';

export type XY = [number, number];

/** A filled piece: its top edge from left to right (closed down to y = 0 when drawn). */
export interface Piece {
  top: XY[];
  /** The piece the method is looking at (adaptive: the interval accepted at this step). */
  current?: boolean;
  /** Not yet accepted (adaptive: pending interval with its Simpson parabola S₁). */
  pending?: boolean;
}

export interface QNode {
  x: number;
  y: number;
  /** Weight share wᵢ/(b − a) (Gauss, nested rules: dot area ∝ weight). */
  weight?: number;
  /** Evaluated for the first time at this step (Romberg, adaptive, Monte Carlo, nested rules). */
  fresh?: boolean;
  /** Nested rules: the value was evaluated at an earlier step and is used again. */
  reused?: boolean;
}

export interface Geometry {
  pieces: Piece[];
  /** Panel boundaries (x), drawn as hairlines. */
  boundaries: number[];
  nodes: QNode[];
  /** Gauss: the interpolant of degree m − 1 through the nodes (integrated exactly by Gₘ). */
  interpolant?: (x: number) => number;
  /**
   * The curve the error hatch is measured against, when it is not the pieces' tops (Gauss: the
   * sampled interpolant, so the hatched area is I − Gₘ).
   */
  errorTop?: XY[];
  /** Romberg: the column j whose Newton–Cotes rule is drawn (min(k, 2)). */
  column?: number;
  /** Monte Carlo: mean height f̄ and its standard error (in f units). */
  mean?: { y: number; se: number | null };
  /** Monte Carlo: samples up to this step (older first). */
  samples?: QNode[];
  /** The geometry is too fine to draw piece by piece (N > MAX_DISPLAY nodes). */
  tooFine?: boolean;
  /** Clenshaw–Curtis: the degree n of the rule; node j is cos(jπ/n) of the semicircle. */
  chebyshev?: number;
}

export type RuleKind =
  | 'rect-left'
  | 'rect-right'
  | 'rect-mid'
  | 'trapezoid'
  | 'newton-cotes'
  | 'romberg'
  | 'gauss'
  | 'adaptive'
  | 'monte-carlo'
  | 'nested';

export const RULE_KIND: Record<string, RuleKind> = {
  left_riemann: 'rect-left',
  right_riemann: 'rect-right',
  midpoint_rule: 'rect-mid',
  trapezoid: 'trapezoid',
  simpson: 'newton-cotes',
  simpson_38: 'newton-cotes',
  boole: 'newton-cotes',
  romberg: 'romberg',
  gauss_legendre: 'gauss',
  adaptive_simpson: 'adaptive',
  monte_carlo_integration: 'monte-carlo',
  clenshaw_curtis: 'nested',
  gauss_patterson: 'nested',
};

/** Subintervals per basic panel of the closed Newton–Cotes rules. */
export const SPAN: Record<string, number> = { simpson: 2, simpson_38: 3, boole: 4 };

/** Order p of the composite rules (error O(h^p) for smooth f), as in the Python `_RULES`. */
export const ORDER: Record<string, number> = {
  left_riemann: 1,
  right_riemann: 1,
  midpoint_rule: 2,
  trapezoid: 2,
  simpson: 4,
  simpson_38: 4,
  boole: 6,
};

/** Lagrange interpolant through a few points (stable for the ≤ 5 nodes of a panel). */
export function lagrange(pts: readonly XY[]): (x: number) => number {
  return (x) => {
    let s = 0;
    for (let i = 0; i < pts.length; i++) {
      let l = 1;
      for (let j = 0; j < pts.length; j++)
        if (j !== i) l *= (x - pts[j][0]) / (pts[i][0] - pts[j][0]);
      s += l * pts[i][1];
    }
    return s;
  };
}

/** Barycentric interpolant (second form) through distinct nodes; stable for m ≤ 64. */
export function barycentric(pts: readonly XY[]): (x: number) => number {
  const n = pts.length;
  const w = pts.map(([xi], i) => {
    let p = 1;
    for (let j = 0; j < n; j++) if (j !== i) p *= xi - pts[j][0];
    return 1 / p;
  });
  // Rescale the weights to avoid overflow for larger m (only ratios matter).
  const big = Math.max(...w.map(Math.abs));
  const ws = w.map((v) => v / big);
  return (x) => {
    let num = 0,
      den = 0;
    for (let i = 0; i < n; i++) {
      const d = x - pts[i][0];
      if (d === 0) return pts[i][1];
      const t = ws[i] / d;
      num += t * pts[i][1];
      den += t;
    }
    return num / den;
  };
}

/**
 * Barycentric interpolant through `pts` with given barycentric weights `bw` (second form,
 * Berrut & Trefethen 2004, (4.2)).
 */
function baryWith(pts: readonly XY[], bw: readonly number[]): (x: number) => number {
  return (x) => {
    let num = 0,
      den = 0;
    for (let i = 0; i < pts.length; i++) {
      const d = x - pts[i][0];
      if (d === 0) return pts[i][1];
      const t = bw[i] / d;
      num += t * pts[i][1];
      den += t;
    }
    return num / den;
  };
}

/**
 * Barycentric weights 1/∏_{j≠i}(xᵢ − xⱼ) for any distinct nodes on [a, b], each factor scaled
 * by 4/(b − a) (the capacity of an interval is (b − a)/4), so the products neither overflow nor
 * underflow for the ≤ 127 Patterson nodes; only the ratios matter.
 */
export function baryWeights(xs: readonly number[], a: number, b: number): number[] {
  const c = 4 / (b - a);
  return xs.map((xi, i) => {
    let p = 1;
    for (let j = 0; j < xs.length; j++) if (j !== i) p *= (xi - xs[j]) * c;
    return 1 / p;
  });
}

/**
 * The interpolant through the nodes of a nested rule. Clenshaw–Curtis nodes are the Chebyshev
 * extreme points in their trace order (j = 0 at b), with the explicit barycentric weights
 * (−1)ʲδⱼ, δ = ½ at both ends (Berrut & Trefethen 2004, (5.2)), which are exact and O(n) for any
 * number of nodes; Patterson nodes use `baryWeights`.
 */
export function nestedInterpolant(
  method: string,
  pts: readonly XY[],
  a: number,
  b: number,
): (x: number) => number {
  if (pts.length === 1) return () => pts[0][1];
  const n = pts.length - 1;
  const bw =
    method === 'clenshaw_curtis'
      ? pts.map((_, j) => (j % 2 ? -1 : 1) * (j === 0 || j === n ? 0.5 : 1))
      : baryWeights(
          pts.map((p) => p[0]),
          a,
          b,
        );
  return baryWith(pts, bw);
}

function sampleTop(g: (x: number) => number, x0: number, x1: number, n: number): XY[] {
  const out: XY[] = [];
  for (let i = 0; i <= n; i++) {
    const x = i === n ? x1 : x0 + ((x1 - x0) * i) / n;
    out.push([x, g(x)]);
  }
  return out;
}

const asPts = (v: unknown): XY[] | null => (Array.isArray(v) ? (v as XY[]) : null);

/** Samples per curved panel, given the panel width in px (curves need ~1 sample per 3 px). */
export function curveSamples(panelPx: number): number {
  return Math.max(4, Math.min(48, Math.ceil(panelPx / 3)));
}

/** Composite rules: rectangles, chords or Newton–Cotes polynomials on each panel. */
export function compositeGeometry(method: string, step: Step, pxPerUnit: number): Geometry {
  const kind = RULE_KIND[method];
  const panels = asPts(step.info.panels);
  const nodes = asPts(step.info.nodes);
  if (!panels || !nodes) return { pieces: [], boundaries: [], nodes: [], tooFine: true };
  const boundaries = [panels[0][0], ...panels.map((p) => p[1])];
  const pieces: Piece[] = [];
  if (kind === 'rect-left' || kind === 'rect-right' || kind === 'rect-mid') {
    panels.forEach(([l, r], p) => {
      const y = nodes[p][1];
      pieces.push({
        top: [
          [l, y],
          [r, y],
        ],
      });
    });
  } else if (kind === 'trapezoid') {
    panels.forEach((_, p) => pieces.push({ top: [nodes[p], nodes[p + 1]] }));
  } else {
    const m = SPAN[method];
    panels.forEach(([l, r], p) => {
      const pts = nodes.slice(p * m, p * m + m + 1);
      pieces.push({ top: sampleTop(lagrange(pts), l, r, curveSamples((r - l) * pxPerUnit)) });
    });
  }
  return { pieces, boundaries, nodes: nodes.map(([x, y]) => ({ x, y })) };
}

/** Romberg column j as a closed Newton–Cotes rule: subintervals per panel (2^j). */
const ROMBERG_SPAN = [1, 2, 4];

/**
 * Romberg row k: the closed Newton–Cotes rule of column j = min(k, 2) on the 2^k subintervals —
 * trapezoids (j = 0), Simpson arches (j = 1), Boole quartics (j = 2). R(k, 1) is composite Simpson
 * and R(k, 2) composite Boole on the same grid (each column is one Richardson step), so the area
 * drawn is exactly `row[j]`, and for k ≤ 2 it is the estimate R(k, k). Nodes from rows 0..k.
 */
export function rombergGeometry(trace: readonly Step[], k: number, pxPerUnit = 400): Geometry {
  const pts = new Map<number, number>();
  const fresh = new Set<number>();
  let tooFine = false;
  for (let j = 0; j <= k && j < trace.length; j++) {
    const nodes = asPts(trace[j].info.nodes);
    if (!nodes) {
      tooFine = true;
      continue;
    }
    for (const [x, y] of nodes) {
      pts.set(x, y);
      if (j === k) fresh.add(x);
    }
  }
  const column = Math.min(k, 2);
  if (tooFine) return { pieces: [], boundaries: [], nodes: [], tooFine: true, column };
  const sorted = [...pts.entries()].sort((p, q) => p[0] - q[0]) as XY[];
  const span = ROMBERG_SPAN[column];
  const pieces: Piece[] = [];
  const boundaries: number[] = [];
  for (let i = 0; i + span < sorted.length; i += span) {
    const panel = sorted.slice(i, i + span + 1);
    const l = panel[0][0],
      r = panel[span][0];
    pieces.push({
      top: span === 1 ? panel : sampleTop(lagrange(panel), l, r, curveSamples((r - l) * pxPerUnit)),
    });
    if (!boundaries.length) boundaries.push(l);
    boundaries.push(r);
  }
  return {
    pieces,
    boundaries,
    nodes: sorted.map(([x, y]) => ({ x, y, fresh: k > 0 && fresh.has(x) })),
    column,
  };
}

/** Gauss–Legendre m-point rule: weight cells, weighted nodes and the interpolant. */
export function gaussGeometry(step: Step, a: number, b: number, pxPerUnit = 400): Geometry {
  const nodes = asPts(step.info.nodes) ?? [];
  const weights = (step.info.weights as number[] | null) ?? [];
  const pieces: Piece[] = [];
  const boundaries = [a];
  let c = a;
  nodes.forEach(([, y], i) => {
    const next = i === nodes.length - 1 ? b : c + weights[i];
    pieces.push({
      top: [
        [c, y],
        [next, y],
      ],
    });
    c = next;
    boundaries.push(c);
  });
  const interpolant = nodes.length > 1 ? barycentric(nodes) : () => nodes[0]?.[1] ?? 0;
  const samples = Math.max(64, Math.min(1200, Math.ceil(((b - a) * pxPerUnit) / 2)));
  return {
    pieces,
    boundaries,
    nodes: nodes.map(([x, y], i) => ({ x, y, weight: weights[i] / (b - a) })),
    interpolant,
    errorTop: nodes.length ? sampleTop(interpolant, a, b, samples) : undefined,
  };
}

/**
 * A nested rule at step k: the area under the interpolant through its nodes (= the estimate),
 * the nodes with weight-sized dots, and which of them step k − 1 already evaluated.
 */
export function nestedGeometry(
  method: string,
  trace: readonly Step[],
  k: number,
  a: number,
  b: number,
  pxPerUnit = 400,
): Geometry {
  const step = trace[k];
  const pts = asPts(step?.info.nodes);
  const weights = step?.info.weights as number[] | null | undefined;
  if (!pts || !weights) return { pieces: [], boundaries: [], nodes: [], tooFine: true };
  const prev = k > 0 ? asPts(trace[k - 1].info.nodes) : null;
  const old = new Set(prev?.map((p) => p[0]) ?? []);
  const nodes: QNode[] = pts.map(([x, y], i) => ({
    x,
    y,
    weight: weights[i] / (b - a),
    fresh: k > 0 && !old.has(x),
    reused: k > 0 && old.has(x),
  }));
  const finite = pts.every((p) => Number.isFinite(p[1]));
  const chebyshev = method === 'clenshaw_curtis' ? (step.info.n as number) : undefined;
  if (!finite) return { pieces: [], boundaries: [a, b], nodes, chebyshev };
  const p = nestedInterpolant(method, pts, a, b);
  const samples = Math.max(64, Math.min(1200, Math.ceil(((b - a) * pxPerUnit) / 2)));
  return {
    pieces: [{ top: sampleTop(p, a, b, samples) }],
    boundaries: [a, b],
    nodes,
    chebyshev,
  };
}

/**
 * Adaptive Simpson after step k: accepted intervals of steps 1..k (quartic through their five
 * nodes), the one accepted at step k marked `current`, and the pending intervals with S₁.
 */
export function adaptiveGeometry(
  trace: readonly Step[],
  k: number,
  f: (x: number) => number,
  pxPerUnit: number,
): Geometry {
  const pieces: Piece[] = [];
  const boundaries = new Set<number>();
  const nodes: QNode[] = [];
  for (let j = 1; j <= k && j < trace.length; j++) {
    const iv = trace[j].info.interval as XY | null;
    const pts = asPts(trace[j].info.nodes);
    if (!iv || !pts) continue;
    const top = sampleTop(lagrange(pts), iv[0], iv[1], curveSamples((iv[1] - iv[0]) * pxPerUnit));
    pieces.push({ top, current: j === k });
    boundaries.add(iv[0]).add(iv[1]);
    if (j === k) nodes.push(...pts.map(([x, y]) => ({ x, y, fresh: true })));
  }
  const step = trace[Math.min(k, trace.length - 1)];
  const pending = (step?.info.pending as XY[] | undefined) ?? [];
  for (const [l, r] of pending) {
    const m = l + 0.5 * (r - l);
    const pts: XY[] = [
      [l, f(l)],
      [m, f(m)],
      [r, f(r)],
    ];
    if (!pts.every((p) => Number.isFinite(p[1]))) continue;
    pieces.push({
      top: sampleTop(lagrange(pts), l, r, curveSamples((r - l) * pxPerUnit)),
      pending: true,
    });
    boundaries.add(l).add(r);
  }
  if (k === 0) {
    const pts = asPts(step?.info.nodes) ?? [];
    nodes.push(...pts.map(([x, y]) => ({ x, y, fresh: true })));
  }
  return { pieces, boundaries: [...boundaries].sort((p, q) => p - q), nodes };
}

/** Monte Carlo after step k: samples of steps 0..k and the mean rectangle ± one s.e. */
export function monteCarloGeometry(
  trace: readonly Step[],
  k: number,
  a: number,
  b: number,
): Geometry {
  const samples: QNode[] = [];
  for (let j = 0; j <= k && j < trace.length; j++)
    for (const [x, y] of asPts(trace[j].info.samples) ?? [])
      samples.push({ x, y, fresh: j === k && k > 0 });
  const s = trace[Math.min(k, trace.length - 1)];
  const est = (s?.info.estimate as number) ?? NaN;
  const se = s?.info.err_est as number | null;
  const y = est / (b - a);
  return {
    pieces: [
      {
        top: [
          [a, y],
          [b, y],
        ],
      },
    ],
    boundaries: [a, b],
    nodes: [],
    samples,
    mean: { y, se: se === null || se === undefined ? null : se / (b - a) },
  };
}

/** The geometry of one run at step k. */
export function geometryAt(
  method: string,
  trace: readonly Step[],
  k: number,
  ab: readonly [number, number],
  f: (x: number) => number,
  pxPerUnit: number,
): Geometry | null {
  const step = trace[Math.min(k, trace.length - 1)];
  if (!step) return null;
  switch (RULE_KIND[method]) {
    case 'romberg':
      return rombergGeometry(trace, Math.min(k, trace.length - 1), pxPerUnit);
    case 'gauss':
      return gaussGeometry(step, ab[0], ab[1], pxPerUnit);
    case 'adaptive':
      return adaptiveGeometry(trace, Math.min(k, trace.length - 1), f, pxPerUnit);
    case 'monte-carlo':
      return monteCarloGeometry(trace, Math.min(k, trace.length - 1), ab[0], ab[1]);
    case 'nested':
      return nestedGeometry(method, trace, Math.min(k, trace.length - 1), ab[0], ab[1], pxPerUnit);
    case undefined:
      return null;
    default:
      return compositeGeometry(method, step, pxPerUnit);
  }
}

/** Signed area under the pieces' top edges (trapezoidal on the sampled tops; tests only). */
export function piecesArea(pieces: readonly Piece[]): number {
  let s = 0;
  for (const p of pieces) {
    if (p.pending) continue;
    for (let i = 0; i + 1 < p.top.length; i++) {
      const [x0, y0] = p.top[i],
        [x1, y1] = p.top[i + 1];
      s += ((x1 - x0) * (y0 + y1)) / 2;
    }
  }
  return s;
}

/**
 * Number of nodes the estimate of step k uses — the x-axis of the error chart.
 * Composite: nodes of the rule; Romberg 2^k + 1; Gauss m; nested rules the nodes of rule k (every
 * earlier node is one of them); Monte Carlo N; adaptive Simpson the
 * points evaluated so far, 4k + 2P + 1 (k accepted, P pending intervals: every processed interval
 * costs two new points and 2k + P − 1 intervals have been processed).
 */
export function nodesUsed(method: string, step: Step): number | null {
  const info = step.info;
  switch (RULE_KIND[method]) {
    case 'romberg':
      return (info.n_panels as number) + 1;
    case 'gauss':
    case 'nested':
      return info.n_points as number;
    case 'monte-carlo':
      return info.n_samples as number;
    case 'adaptive': {
      const pending = (info.pending as unknown[] | undefined)?.length ?? 0;
      return 4 * step.k + 2 * pending + 1;
    }
    case 'rect-left':
    case 'rect-right':
    case 'rect-mid':
      return info.n_panels as number;
    case undefined:
      return null;
    default:
      return (info.n_panels as number) + 1;
  }
}

/** Observed order p̂ between two (n, error) points: error ∝ n^(−p̂). */
export function observedOrder(n0: number, e0: number, n1: number, e1: number): number | null {
  if (!(n0 > 0 && n1 > n0 && e0 > 0 && e1 > 0)) return null;
  return -Math.log(e1 / e0) / Math.log(n1 / n0);
}

/**
 * Lowest and highest value of the areas a run integrates over all its steps (Newton–Cotes arches
 * and Romberg quartics can overshoot f — Boole on Runge with 4 panels dips to −0.38). The stage's
 * y-range includes it, so no part of a rule's area is clipped away. Construction lines are left
 * out: the Gauss interpolant of a low m swings far from f and would flatten every panel, and the
 * pending adaptive parabolas are provisional.
 */
export function geometryExtent(
  method: string,
  trace: readonly Step[],
  ab: readonly [number, number],
  f: (x: number) => number,
): [number, number] {
  let lo = Infinity,
    hi = -Infinity;
  const take = (pieces: readonly Piece[]) => {
    for (const p of pieces)
      if (!p.pending)
        for (const [, v] of p.top)
          if (Number.isFinite(v)) {
            lo = Math.min(lo, v);
            hi = Math.max(hi, v);
          }
  };
  const kind = RULE_KIND[method];
  // 30 px per unit of x is plenty for an extent; curveSamples keeps at least 4 per panel.
  const px = 30 / Math.max(1e-12, ab[1] - ab[0]);
  if (kind === 'adaptive') {
    const last = trace.length - 1;
    // Each step adds one accepted interval: its own piece is the last one of its geometry.
    for (let k = 1; k <= last; k++) {
      const g = adaptiveGeometry(trace.slice(0, k + 1), k, f, px);
      take(g.pieces.filter((p) => p.current));
    }
  } else if (kind !== 'monte-carlo' && kind !== undefined) {
    for (let k = 0; k < trace.length; k++) {
      const g = geometryAt(method, trace, k, ab, f, px);
      if (g && !g.tooFine) take(g.pieces);
    }
  }
  return [lo, hi];
}
