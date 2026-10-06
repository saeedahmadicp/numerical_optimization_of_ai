/**
 * Polynomial and piecewise-polynomial interpolation of data (x_i, y_i) — TS port of
 * `numopt.interpolation.methods` (src/numopt/interpolation/methods.py).
 *
 * Every method builds an interpolant p with p(x_i) = y_i and returns it in a representation a
 * plot can evaluate without recomputation (same contract as Python):
 *
 *   - `Result.x`: the coefficient vector of the representation; piecewise methods flatten the
 *     (n−1) × 4 local coefficients row by row; Neville returns the value p(x*).
 *   - `Result.fun`: max_t |p(t) − f_true(t)| over the 200-point plotting grid (null without f_true).
 *   - `Result.extra` (snake_case keys, as exported): kind, coefficients, nodes, values, domain,
 *     eval {x, y, f_true}, max_error, node_residual, plus method-specific keys.
 *
 * `converged` is true when the construction finished, every coefficient and grid value is
 * finite and max_i |p(x_i) − y_i| ≤ 1e-6·max(S, 2⁻¹⁰²²) (S = max|y_i|; the clamped spline adds
 * its end slopes). Invalid data (NaN, length mismatch, repeated nodes) throws; numerical
 * breakdown returns `converged: false` with a message.
 *
 * Info keys (snake_case, as in the Python docstring): curve, node_index, node, basis (lagrange),
 * new_row (newton), column, x_eval (neville), term_index (chebyshev), stage, h, secants,
 * boundary, lower, diag, upper, rhs, pivots, multipliers, slopes, limited, alpha, beta,
 * coefficients (piecewise).
 */
import { registerMethod, param, type MethodDoc } from '../../core/registry';
import { luFactor, luSolve } from '../../core/linalg';
import type { Dataset, MethodFn, Result, Step, Vector } from '../../core/types';
import { linspace } from '../../problems/data';

/** Number of points on the plotting grid in `extra.eval`. */
export const N_GRID = 200;
/** converged requires max_i |p(x_i) − y_i| ≤ NODE_RTOL · max(max_i |y_i|, TINY). */
export const NODE_RTOL = 1e-6;
/** Smallest normal float64, 2⁻¹⁰²². */
const TINY = 2.2250738585072014e-308;

type F = (x: number) => number;
/** A Dataset, or a bare `[x, y]` pair (Python accepts an `(x, y)` tuple). */
export type InterpProblem = Pick<Dataset, 'x' | 'y'> &
  Partial<Pick<Dataset, 'fTrue' | 'domain'>>;
type ProblemArg = InterpProblem | readonly [readonly number[], readonly number[]];

// ---------------------------------------------------------------------------------------
// Python formatting helpers (messages)
// ---------------------------------------------------------------------------------------

/** Python's `format(v, '.{p}g')` (e.g. `1.15e-15`, `0.0123`, `1`, `inf`). */
export function pyG(v: number, p = 6): string {
  if (Number.isNaN(v)) return 'nan';
  if (v === Infinity) return 'inf';
  if (v === -Infinity) return '-inf';
  if (v === 0) return Object.is(v, -0) ? '-0' : '0';
  const [mant, expStr] = v.toExponential(Math.max(0, p - 1)).split('e');
  const exp = Number(expStr);
  const strip = (s: string) => (s.includes('.') ? s.replace(/0+$/, '').replace(/\.$/, '') : s);
  if (exp < -4 || exp >= p) {
    const e = Math.abs(exp);
    return `${strip(mant)}e${exp < 0 ? '-' : '+'}${e < 10 ? '0' : ''}${e}`;
  }
  return strip(v.toFixed(Math.max(0, p - 1 - exp)));
}

// ---------------------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------------------

interface Data {
  x: number[];
  y: number[];
  fTrue: F | null;
  a: number;
  b: number;
  n: number;
}

const maxAbs = (v: readonly number[]) => v.reduce((m, t) => Math.max(m, Math.abs(t)), 0);
const allFinite = (v: readonly number[]) => v.every(Number.isFinite);

/** NODE_RTOL·max(S, 2⁻¹⁰²²): the scale-invariant node-residual limit. */
function nodeLimit(values: readonly number[], dataScale: number | null = null): number {
  const scale = dataScale === null ? maxAbs(values) : dataScale;
  return NODE_RTOL * Math.max(scale, TINY);
}

/** Accept a Dataset or an `[x, y]` pair; validate the data (Python `_resolve`). */
function resolve(problem: ProblemArg): Data {
  let xRaw: readonly number[], yRaw: readonly number[];
  let fTrue: F | null = null;
  let domain: readonly number[] | null | undefined = null;
  if (Array.isArray(problem)) {
    if (problem.length !== 2) throw new TypeError('problem must be a Dataset or an (x, y) pair');
    [xRaw, yRaw] = problem as [readonly number[], readonly number[]];
  } else {
    const d = problem as InterpProblem;
    xRaw = d.x;
    yRaw = d.y;
    fTrue = d.fTrue ?? null;
    domain = d.domain;
  }
  const x = Array.from(xRaw, Number);
  const y = Array.from(yRaw, Number);
  if (x.length !== y.length)
    throw new Error(`x and y must be 1-D of equal length; got (${x.length},) and (${y.length},)`);
  if (x.length === 0) throw new Error('need at least one data point');
  if (!(allFinite(x) && allFinite(y))) throw new Error('data contain NaN or infinite values');
  let a: number, b: number;
  if (domain) {
    a = Number(domain[0]);
    b = Number(domain[1]);
  } else {
    a = Math.min(...x);
    b = Math.max(...x);
  }
  if (!(a < b)) {
    a -= 1.0;
    b += 1.0;
  }
  return { x, y, fTrue, a, b, n: x.length };
}

function requireDistinct(x: readonly number[]): void {
  const s = [...x].sort((p, q) => p - q);
  for (let i = 1; i < s.length; i++)
    if (s[i] - s[i - 1] === 0.0) throw new Error('interpolation nodes must be distinct');
}

/** Data sorted by x (stable), with distinct nodes and at least `minPoints` points. */
function sortedData(data: Data, minPoints: number): [number[], number[]] {
  if (data.n < minPoints)
    throw new Error(`need at least ${minPoints} data points, got ${data.n}`);
  const order = data.x.map((_, i) => i).sort((i, j) => data.x[i] - data.x[j] || i - j);
  const x = order.map((i) => data.x[i]);
  const y = order.map((i) => data.y[i]);
  for (let i = 1; i < x.length; i++)
    if (x[i] - x[i - 1] === 0.0) throw new Error('interpolation nodes must be distinct');
  return [x, y];
}

/** Evaluate f point by point; a throwing point gives NaN (Python `_sample`). */
function sample(f: F, pts: readonly number[]): number[] {
  return pts.map((t) => {
    try {
      return Number(f(t));
    } catch {
      return NaN;
    }
  });
}

/** The plotting grid, f_true on it, and the max-error functional (Python `_Grid`). */
export class Grid {
  readonly t: number[];
  readonly truth: number[] | null;
  constructor(data: { a: number; b: number; fTrue: F | null }) {
    this.t = linspace(data.a, data.b, N_GRID);
    let truth: number[] | null = null;
    if (data.fTrue) {
      const v = sample(data.fTrue, this.t);
      if (allFinite(v)) truth = v;
    }
    this.truth = truth;
  }
  error(curve: readonly number[]): number | null {
    if (this.truth === null) return null;
    if (!allFinite(curve)) return Infinity;
    let m = 0;
    for (let i = 0; i < curve.length; i++) {
      const d = Math.abs(curve[i] - this.truth[i]);
      if (d > m || Number.isNaN(d)) m = d;
    }
    return m;
  }
}

function step(k: number, x: number | number[], fun: number | null, info: Step['info']): Step {
  return { k, x, fun, gradNorm: null, stepSize: null, info };
}

interface FinishOptions {
  kind: string;
  coefficients: unknown;
  coefFlat: readonly number[];
  nodes: number[];
  values: number[];
  grid: Grid;
  evaluate: (t: readonly number[]) => number[];
  trace: Step[];
  nFev?: number;
  domain: [number, number];
  more?: Record<string, unknown>;
  dataScale?: number | null;
}

function finish(method: string, xResult: number | number[], o: FinishOptions): Result {
  const curve = o.evaluate(o.grid.t);
  const atNodes = o.evaluate(o.nodes);
  const finite = allFinite(o.coefFlat) && allFinite(curve) && allFinite(atNodes);
  let residual = Infinity;
  if (finite) {
    residual = 0;
    for (let i = 0; i < atNodes.length; i++)
      residual = Math.max(residual, Math.abs(atNodes[i] - o.values[i]));
  }
  const limit = nodeLimit(o.values, o.dataScale ?? null);
  const ok = finite && residual <= limit;
  const err = o.grid.error(curve);
  let msg: string;
  if (ok)
    msg = `interpolant through ${o.nodes.length} nodes built; max node residual ${pyG(residual, 3)}`;
  else if (finite)
    msg =
      `the interpolant misses the data by ${pyG(residual, 3)} at a node (> ${pyG(limit, 3)}): ` +
      'round-off has destroyed this representation for these nodes';
  else msg = 'non-finite coefficients or values (overflow): the interpolant is unusable';
  const extra: Record<string, unknown> = {
    kind: o.kind,
    coefficients: o.coefficients,
    nodes: o.nodes,
    values: o.values,
    domain: [o.domain[0], o.domain[1]],
    eval: { x: o.grid.t, y: curve, f_true: o.grid.truth },
    max_error: err,
    node_residual: residual,
    ...(o.more ?? {}),
  };
  return {
    method,
    x: xResult,
    fun: err,
    converged: ok,
    message: msg,
    nIter: o.trace[o.trace.length - 1].k,
    nFev: o.nFev ?? 0,
    nGev: 0,
    nHev: 0,
    trace: o.trace,
    extra,
  };
}

/** converged=false result after a numerical breakdown during construction. */
function broken(method: string, trace: Step[], message: string, nFev = 0): Result {
  const last = trace[trace.length - 1];
  return {
    method,
    x: last.x,
    fun: null,
    converged: false,
    message,
    nIter: last.k,
    nFev,
    nGev: 0,
    nHev: 0,
    trace,
    extra: {},
  };
}

// ---------------------------------------------------------------------------------------
// Lagrange form
// ---------------------------------------------------------------------------------------

/** ℓ_j(t) = Π_{m≠j} (t − x_m)/(x_j − x_m), accumulated factor by factor. */
export function lagrangeBasis(nodes: readonly number[], j: number, t: readonly number[]): number[] {
  const out = t.map(() => 1.0);
  for (let m = 0; m < nodes.length; m++) {
    if (m === j) continue;
    const den = nodes[j] - nodes[m];
    for (let i = 0; i < t.length; i++) out[i] = out[i] * ((t[i] - nodes[m]) / den);
  }
  return out;
}

const lagrange: MethodFn<ProblemArg> = (problem) => {
  const data = resolve(problem);
  requireDistinct(data.x);
  const grid = new Grid(data);
  const { x, y } = data;
  let curve = grid.t.map(() => 0.0);
  const trace: Step[] = [];
  for (let k = 0; k < data.n; k++) {
    const basis = lagrangeBasis(x, k, grid.t);
    curve = curve.map((c, i) => c + y[k] * basis[i]);
    trace.push(
      step(k, y.slice(0, k + 1), grid.error(curve), {
        node_index: k,
        node: [x[k], y[k]],
        basis,
        curve,
      }),
    );
  }
  const evaluate = (t: readonly number[]) => {
    let total = t.map(() => 0.0);
    for (let j = 0; j < data.n; j++) {
      const bj = lagrangeBasis(x, j, t);
      total = total.map((s, i) => s + y[j] * bj[i]);
    }
    return total;
  };
  return finish('lagrange', y.slice(), {
    kind: 'lagrange',
    coefficients: y.slice(),
    coefFlat: y,
    nodes: x,
    values: y,
    grid,
    evaluate,
    trace,
    domain: [data.a, data.b],
  });
};

// ---------------------------------------------------------------------------------------
// Barycentric form (second / "true" form)
// ---------------------------------------------------------------------------------------

/** Second barycentric formula (Berrut & Trefethen 2004, eq. (4.2)); p(x_j) = y_j exactly. */
export function barycentricEval(
  nodes: readonly number[],
  w: readonly number[],
  values: readonly number[],
  t: readonly number[],
): number[] {
  return t.map((ti) => {
    let num = 0;
    let den = 0;
    let hit = -1;
    for (let j = 0; j < nodes.length; j++) {
      const d = ti - nodes[j];
      if (d === 0.0 && hit < 0) hit = j;
      const c = w[j] / (d === 0.0 ? 1.0 : d);
      num += c * values[j];
      den += c;
    }
    return hit >= 0 ? values[hit] : num / den;
  });
}

const barycentric: MethodFn<ProblemArg> = (problem) => {
  const data = resolve(problem);
  requireDistinct(data.x);
  const grid = new Grid(data);
  const { x, y } = data;
  let w: number[] = [];
  const trace: Step[] = [];
  for (let k = 0; k < data.n; k++) {
    w = w.map((wj, j) => wj / (x[j] - x[k]));
    let wk = 1.0;
    if (k) {
      let prod = 1.0;
      for (let m = 0; m < k; m++) prod *= x[k] - x[m];
      wk = 1.0 / prod;
    }
    w.push(wk);
    const curve = barycentricEval(x.slice(0, k + 1), w, y.slice(0, k + 1), grid.t);
    trace.push(
      step(k, w.slice(), grid.error(curve), { node_index: k, node: [x[k], y[k]], curve }),
    );
    if (!allFinite(w) || w.some((v) => v === 0.0))
      return broken(
        'barycentric',
        trace,
        `barycentric weights overflowed/underflowed at node ${k}: rescale the data`,
      );
  }
  const wf = w.slice();
  return finish('barycentric', w.slice(), {
    kind: 'barycentric',
    coefficients: w.slice(),
    coefFlat: w,
    nodes: x,
    values: y,
    grid,
    evaluate: (t) => barycentricEval(x, wf, y, t),
    trace,
    domain: [data.a, data.b],
  });
};

// ---------------------------------------------------------------------------------------
// Newton divided differences
// ---------------------------------------------------------------------------------------

/** Nested (Horner) evaluation of the Newton form. */
export function newtonEval(
  nodes: readonly number[],
  coef: readonly number[],
  t: readonly number[],
): number[] {
  return t.map((ti) => {
    let v = coef[coef.length - 1];
    for (let j = coef.length - 2; j >= 0; j--) v = coef[j] + (ti - nodes[j]) * v;
    return v;
  });
}

const newtonDividedDifferences: MethodFn<ProblemArg> = (problem) => {
  const data = resolve(problem);
  requireDistinct(data.x);
  const grid = new Grid(data);
  const { x, y } = data;
  let prev: number[] = [];
  const coef: number[] = [];
  const trace: Step[] = [];
  for (let k = 0; k < data.n; k++) {
    const row = new Array<number>(k + 1);
    row[0] = y[k];
    for (let j = 1; j <= k; j++) row[j] = (row[j - 1] - prev[j - 1]) / (x[k] - x[k - j]);
    coef.push(row[k]);
    const curve = newtonEval(x.slice(0, k + 1), coef, grid.t);
    trace.push(
      step(k, coef.slice(), grid.error(curve), {
        node_index: k,
        node: [x[k], y[k]],
        new_row: row,
        curve,
      }),
    );
    if (!allFinite(row))
      return broken(
        'newton_divided_differences',
        trace,
        `non-finite divided difference at node ${k} (overflow)`,
      );
    prev = row;
  }
  const cf = coef.slice();
  return finish('newton_divided_differences', coef.slice(), {
    kind: 'newton',
    coefficients: coef.slice(),
    coefFlat: coef,
    nodes: x,
    values: y,
    grid,
    evaluate: (t) => newtonEval(x, cf, t),
    trace,
    domain: [data.a, data.b],
  });
};

// ---------------------------------------------------------------------------------------
// Neville
// ---------------------------------------------------------------------------------------

/**
 * All columns of Neville's tableau over the points `t`: `cols[j][r][p]` = Q_{j+r, j}(t_p), the
 * interpolant through x_{r}, …, x_{j+r} (rows i = j..n−1 of column j).
 */
export function nevilleColumns(
  nodes: readonly number[],
  values: readonly number[],
  t: readonly number[],
): number[][][] {
  const n = nodes.length;
  const q = values.map((v) => t.map(() => v));
  const cols: number[][][] = [q.map((r) => r.slice())];
  for (let j = 1; j < n; j++) {
    for (let i = n - 1; i >= j; i--) {
      const qi = q[i],
        qm = q[i - 1];
      const den = nodes[i] - nodes[i - j];
      for (let p = 0; p < t.length; p++)
        qi[p] = ((t[p] - nodes[i - j]) * qi[p] - (t[p] - nodes[i]) * qm[p]) / den;
    }
    cols.push(q.slice(j).map((r) => r.slice()));
  }
  return cols;
}

const neville: MethodFn<ProblemArg> = (problem, { x_frac = 0.95 }) => {
  const xf = Number(x_frac);
  if (!(xf >= 0.0 && xf <= 1.0)) throw new Error('x_frac must lie in [0, 1]');
  const data = resolve(problem);
  requireDistinct(data.x);
  const grid = new Grid(data);
  const { x, y } = data;
  const xStar = data.a + xf * (data.b - data.a);
  const pts = [xStar, ...grid.t];
  const cols = nevilleColumns(x, y, pts);
  const trace: Step[] = [];
  const diagonal: number[] = [];
  cols.forEach((col, k) => {
    const estimate = col[0][0];
    diagonal.push(estimate);
    const curve = col[0].slice(1);
    trace.push(
      step(k, estimate, grid.error(curve), {
        column: col.map((r) => r[0]),
        x_eval: xStar,
        curve,
      }),
    );
  });
  const value = diagonal[diagonal.length - 1];
  let errAt: number | null = null;
  if (data.fTrue && Number.isFinite(value)) {
    const fStar = sample(data.fTrue, [xStar])[0];
    errAt = Number.isFinite(fStar) ? Math.abs(value - fStar) : null;
  }
  const evaluate = (t: readonly number[]) => {
    const c = nevilleColumns(x, y, t);
    return c[c.length - 1][0];
  };
  return finish('neville', value, {
    kind: 'neville',
    coefficients: diagonal.slice(),
    coefFlat: diagonal,
    nodes: x,
    values: y,
    grid,
    evaluate,
    trace,
    domain: [data.a, data.b],
    more: { x_eval: xStar, value, error_at_x_eval: errAt },
  });
};

// ---------------------------------------------------------------------------------------
// Piecewise polynomials
// ---------------------------------------------------------------------------------------

/** Evaluate a piecewise cubic in local power form; the end pieces extrapolate. */
export function ppEval(
  breaks: readonly number[],
  coef: readonly (readonly number[])[],
  t: readonly number[],
): number[] {
  const last = breaks.length - 2;
  return t.map((ti) => {
    // searchsorted(side="right") − 1, clipped to [0, n − 2].
    let lo = 0,
      hi = breaks.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (breaks[mid] <= ti) lo = mid + 1;
      else hi = mid;
    }
    const idx = Math.min(Math.max(lo - 1, 0), last);
    const dt = ti - breaks[idx];
    const [a, b, c, d] = coef[idx];
    return a + dt * (b + dt * (c + dt * d));
  });
}

const diff = (v: readonly number[]) => v.slice(1).map((t, i) => t - v[i]);

/** Local coefficients of the C¹ piecewise cubic with values y and slopes s (Hermite form). */
export function hermiteCoefficients(
  x: readonly number[],
  y: readonly number[],
  s: readonly number[],
): number[][] {
  const h = diff(x);
  const dy = diff(y);
  return h.map((hi, i) => {
    const m = dy[i] / hi;
    return [
      y[i],
      s[i],
      (3.0 * m - 2.0 * s[i] - s[i + 1]) / hi,
      (s[i] + s[i + 1] - 2.0 * m) / (hi * hi),
    ];
  });
}

function finishPiecewise(
  method: string,
  data: Data,
  xs: number[],
  ys: number[],
  coef: number[][],
  trace: Step[],
  grid: Grid,
  k: number,
  info: Record<string, unknown>,
  more?: Record<string, unknown>,
  dataScale: number | null = null,
): Result {
  const curve = ppEval(xs, coef, grid.t);
  const flat = coef.flat();
  trace.push(
    step(k, flat.slice(), grid.error(curve), {
      stage: 'coefficients',
      ...info,
      coefficients: coef,
      curve,
    }),
  );
  return finish(method, flat.slice(), {
    kind: 'piecewise_cubic',
    coefficients: coef,
    coefFlat: flat,
    nodes: xs,
    values: ys,
    grid,
    evaluate: (t) => ppEval(xs, coef, t),
    trace,
    domain: [data.a, data.b],
    more,
    dataScale,
  });
}

const linearSpline: MethodFn<ProblemArg> = (problem) => {
  const data = resolve(problem);
  const [xs, ys] = sortedData(data, 2);
  const grid = new Grid(data);
  const h = diff(xs);
  const dy = diff(ys);
  const m = h.map((hi, i) => dy[i] / hi);
  const coef = m.map((mi, i) => [ys[i], mi, 0.0, 0.0]);
  return finishPiecewise('linear_spline', data, xs, ys, coef, [], grid, 0, { h, secants: m });
};

/**
 * Thomas algorithm: tridiagonal LU without pivoting. Returns [multipliers, pivots, modified rhs,
 * solution]; the solution is null when a pivot is zero or non-finite.
 */
export function thomas(
  lower: readonly number[],
  diag: readonly number[],
  upper: readonly number[],
  rhs: readonly number[],
): [number[], number[], number[], number[] | null] {
  const n = diag.length;
  const piv = diag.slice();
  const z = rhs.slice();
  const mult = new Array<number>(Math.max(n - 1, 0)).fill(0);
  for (let i = 1; i < n; i++) {
    if (piv[i - 1] === 0.0 || !Number.isFinite(piv[i - 1])) return [mult, piv, z, null];
    mult[i - 1] = lower[i - 1] / piv[i - 1];
    piv[i] = diag[i] - mult[i - 1] * upper[i - 1];
    z[i] = rhs[i] - mult[i - 1] * z[i - 1];
  }
  if (piv[n - 1] === 0.0 || !Number.isFinite(piv[n - 1])) return [mult, piv, z, null];
  const s = new Array<number>(n);
  s[n - 1] = z[n - 1] / piv[n - 1];
  for (let i = n - 2; i >= 0; i--) s[i] = (z[i] - upper[i] * s[i + 1]) / piv[i];
  return [mult, piv, z, s];
}

type Boundary = 'natural' | 'clamped' | 'not-a-knot';

/** Tridiagonal system T s = r for the node slopes s_i = S'(x_i) (Python `_spline_system`). */
export function splineSystem(
  xs: readonly number[],
  ys: readonly number[],
  bc: Boundary,
  fprimeA: number,
  fprimeB: number,
): { lower: number[]; diag: number[]; upper: number[]; rhs: number[]; label: string } {
  const n = xs.length;
  const h = diff(xs);
  const dy = diff(ys);
  const m = h.map((hi, i) => dy[i] / hi);
  const lower = new Array<number>(n - 1).fill(0);
  const diag = new Array<number>(n).fill(0);
  const upper = new Array<number>(n - 1).fill(0);
  const rhs = new Array<number>(n).fill(0);
  for (let i = 1; i < n - 1; i++) {
    lower[i - 1] = h[i];
    diag[i] = 2.0 * (h[i - 1] + h[i]);
    upper[i] = h[i - 1];
    rhs[i] = 3.0 * (h[i] * m[i - 1] + h[i - 1] * m[i]);
  }
  let label: string = bc;
  if (bc === 'natural') {
    [diag[0], upper[0], rhs[0]] = [2.0, 1.0, 3.0 * m[0]];
    [lower[n - 2], diag[n - 1], rhs[n - 1]] = [1.0, 2.0, 3.0 * m[n - 2]];
  } else if (bc === 'clamped') {
    [diag[0], upper[0], rhs[0]] = [1.0, 0.0, fprimeA];
    [lower[n - 2], diag[n - 1], rhs[n - 1]] = [0.0, 1.0, fprimeB];
    label = `clamped (S'(x_0) = ${pyG(fprimeA)}, S'(x_n) = ${pyG(fprimeB)})`;
  } else if (n === 2) {
    [diag[0], upper[0], rhs[0]] = [1.0, 0.0, m[0]];
    [lower[0], diag[1], rhs[1]] = [0.0, 1.0, m[0]];
    label = 'not-a-knot (n = 2: straight line)';
  } else if (n === 3) {
    [diag[0], upper[0], rhs[0]] = [1.0, 1.0, 2.0 * m[0]];
    [lower[1], diag[2], rhs[2]] = [1.0, 1.0, 2.0 * m[1]];
    label = 'not-a-knot (n = 3: parabola)';
  } else {
    const h0 = h[0],
      h1 = h[1];
    [diag[0], upper[0]] = [h1, h0 + h1];
    rhs[0] = (h1 * (3.0 * h0 + 2.0 * h1) * m[0] + h0 * h0 * m[1]) / (h0 + h1);
    const g0 = h[n - 2],
      g1 = h[n - 3];
    [lower[n - 2], diag[n - 1]] = [g0 + g1, g1];
    rhs[n - 1] = (g1 * (3.0 * g0 + 2.0 * g1) * m[n - 2] + g0 * g0 * m[n - 3]) / (g0 + g1);
  }
  return { lower, diag, upper, rhs, label };
}

function cubicSpline(
  method: string,
  problem: ProblemArg,
  bc: Boundary,
  fprimeA = 0.0,
  fprimeB = 0.0,
): Result {
  const data = resolve(problem);
  const [xs, ys] = sortedData(data, 2);
  const grid = new Grid(data);
  const h = diff(xs);
  const dy = diff(ys);
  const m = h.map((hi, i) => dy[i] / hi);
  const { lower, diag, upper, rhs, label } = splineSystem(xs, ys, bc, fprimeA, fprimeB);
  const trace: Step[] = [
    step(0, rhs.slice(), null, {
      stage: 'assemble',
      boundary: label,
      h,
      secants: m,
      lower,
      diag,
      upper,
      rhs,
    }),
  ];
  const [mult, piv, z, s] = thomas(lower, diag, upper, rhs);
  trace.push(
    step(1, z.slice(), null, { stage: 'forward_sweep', pivots: piv, multipliers: mult, rhs: z }),
  );
  if (s === null || !allFinite(s))
    return broken(method, trace, 'zero or non-finite pivot in the tridiagonal solve');
  trace.push(step(2, s.slice(), null, { stage: 'back_substitution', slopes: s }));
  const coef = hermiteCoefficients(xs, ys, s);
  let dataScale: number | null = null;
  if (bc === 'clamped') {
    // The end slopes are interpolation data too: in units of y they are worth |f'|·h.
    const slopeSize = Math.max(Math.abs(fprimeA), Math.abs(fprimeB)) * Math.max(...h);
    dataScale = Math.max(maxAbs(ys), slopeSize);
  }
  return finishPiecewise(
    method,
    data,
    xs,
    ys,
    coef,
    trace,
    grid,
    3,
    { slopes: s },
    { boundary: label, slopes: s },
    dataScale,
  );
}

// ---------------------------------------------------------------------------------------
// PCHIP
// ---------------------------------------------------------------------------------------

/** np.sign: −1, 0, 1 (NaN for NaN). */
const npSign = (v: number) => (Number.isNaN(v) ? NaN : v > 0 ? 1 : v < 0 ? -1 : 0);
/** `np.sign(a) != np.sign(b)` (NaN compares unequal). */
const signDiffers = (a: number, b: number) => !(npSign(a) === npSign(b));

/** One-sided three-point end slope with shape limiting (Moler, NCM Ch. 3, pchiptx). */
export function pchipEndSlope(h0: number, h1: number, m0: number, m1: number): [number, boolean] {
  const d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
  if (signDiffers(d, m0)) return [0.0, true];
  if (signDiffers(m0, m1) && Math.abs(d) > 3.0 * Math.abs(m0)) return [3.0 * m0, true];
  return [d, false];
}

const pchip: MethodFn<ProblemArg> = (problem) => {
  const data = resolve(problem);
  const [xs, ys] = sortedData(data, 2);
  const grid = new Grid(data);
  const n = xs.length;
  const h = diff(xs);
  const dy = diff(ys);
  const m = h.map((hi, i) => dy[i] / hi);
  const trace: Step[] = [step(0, m.slice(), null, { stage: 'secants', h, secants: m })];
  const s = new Array<number>(n).fill(0);
  const limited = new Array<boolean>(n).fill(false);
  if (n === 2) {
    s.fill(m[0]);
  } else {
    for (let k = 1; k < n - 1; k++) {
      if (m[k - 1] === 0.0 || m[k] === 0.0 || signDiffers(m[k - 1], m[k])) {
        s[k] = 0.0;
        limited[k] = true;
      } else {
        const w1 = 2.0 * h[k] + h[k - 1];
        const w2 = h[k] + 2.0 * h[k - 1];
        s[k] = (w1 + w2) / (w1 / m[k - 1] + w2 / m[k]);
      }
    }
    [s[0], limited[0]] = pchipEndSlope(h[0], h[1], m[0], m[1]);
    [s[n - 1], limited[n - 1]] = pchipEndSlope(h[n - 2], h[n - 3], m[n - 2], m[n - 3]);
  }
  const alpha = m.map((mi, i) => (mi !== 0.0 ? s[i] / mi : null));
  const beta = m.map((mi, i) => (mi !== 0.0 ? s[i + 1] / mi : null));
  trace.push(step(1, s.slice(), null, { stage: 'slopes', slopes: s, limited, alpha, beta }));
  const coef = hermiteCoefficients(xs, ys, s);
  return finishPiecewise('pchip', data, xs, ys, coef, trace, grid, 2, { slopes: s }, { slopes: s });
};

// ---------------------------------------------------------------------------------------
// Chebyshev interpolation
// ---------------------------------------------------------------------------------------

/** Σ_k c_k T_k(u) by Clenshaw's recurrence (Numerical Recipes, 3rd ed., §5.4 and §5.8). */
export function clenshaw(coef: readonly number[], u: readonly number[]): number[] {
  return u.map((ui) => {
    let b1 = 0.0,
      b2 = 0.0;
    for (let k = coef.length - 1; k > 0; k--) {
      const next = coef[k] + 2.0 * ui * b1 - b2;
      b2 = b1;
      b1 = next;
    }
    return coef[0] + ui * b1 - b2;
  });
}

/** Affine map [a, b] → [−1, 1]: u = (2t − (a + b))/(b − a). */
export function toUnit(t: readonly number[], a: number, b: number): number[] {
  return t.map((ti) => (2.0 * ti - (a + b)) / (b - a));
}

const chebyshevInterpolation: MethodFn<ProblemArg> = (problem, { n_nodes = 0 }) => {
  const nNodes = Number(n_nodes);
  if (nNodes < 0) throw new Error('n_nodes must be ≥ 0');
  const data = resolve(problem);
  const grid = new Grid(data);
  const { a, b } = data;
  let nFev = 0;
  let deviation: number | null = null;
  const f = data.fTrue
    ? (() => {
        const ft = data.fTrue;
        return (t: number) => {
          nFev += 1;
          return ft(t);
        };
      })()
    : null;
  if (f) {
    const fData = sample(f, data.x);
    const d = fData.map((v, i) => Math.abs(v - data.y[i]));
    deviation = allFinite(d) ? Math.max(...d) : Infinity;
  }
  let source: 'f_true' | 'data';
  let coef: number[];
  let nodes: number[], values: number[];
  let n: number;
  if (f && deviation !== null && deviation <= nodeLimit(data.y)) {
    source = 'f_true';
    n = nNodes > 0 ? Math.trunc(nNodes) : data.n;
    const theta = Array.from({ length: n }, (_, j) => (Math.PI * (j + 0.5)) / n);
    const nodesDesc = theta.map((th) => 0.5 * (a + b) + 0.5 * (b - a) * Math.cos(th));
    const fNodes = sample(f, nodesDesc);
    if (!allFinite(fNodes)) {
      const s0 = step(0, [0.0], null, { term_index: 0, curve: new Array(N_GRID).fill(0) });
      return broken('chebyshev_interpolation', [s0], 'f_true is not finite at a Chebyshev node', nFev);
    }
    coef = new Array<number>(n);
    for (let k = 0; k < n; k++) {
      let sum = 0;
      for (let j = 0; j < n; j++) sum += fNodes[j] * Math.cos(k * theta[j]);
      coef[k] = (2.0 / n) * sum;
    }
    coef[0] *= 0.5;
    nodes = nodesDesc.slice().reverse();
    values = fNodes.slice().reverse();
  } else {
    source = 'data';
    requireDistinct(data.x);
    n = data.n;
    const u = toUnit(data.x, a, b);
    const vander = u.map((uj) => {
      const row = new Array<number>(n).fill(0);
      row[0] = 1.0;
      if (n > 1) row[1] = uj;
      for (let k = 2; k < n; k++) row[k] = 2.0 * uj * row[k - 1] - row[k - 2];
      return row;
    });
    const lu = luFactor(vander);
    if (lu.singular) {
      const s0 = step(0, [0.0], null, { term_index: 0, curve: new Array(N_GRID).fill(0) });
      return broken('chebyshev_interpolation', [s0], 'singular Chebyshev–Vandermonde matrix', nFev);
    }
    coef = luSolve(lu, data.y);
    nodes = data.x.slice();
    values = data.y.slice();
  }
  const uGrid = toUnit(grid.t, a, b);
  const trace: Step[] = [];
  for (let k = 0; k < n; k++) {
    const curve = clenshaw(coef.slice(0, k + 1), uGrid);
    trace.push(step(k, coef.slice(0, k + 1), grid.error(curve), { term_index: k, curve }));
  }
  const cf = coef.slice();
  const result = finish('chebyshev_interpolation', coef.slice(), {
    kind: 'chebyshev',
    coefficients: coef.slice(),
    coefFlat: coef,
    nodes,
    values,
    grid,
    evaluate: (t) => clenshaw(cf, toUnit(t, a, b)),
    trace,
    nFev,
    domain: [a, b],
    more: { source, sample_deviation: deviation },
  });
  if (source === 'data' && deviation !== null)
    result.message +=
      `; the data are not samples of f_true (max |f_true(x_i) - y_i| = ${pyG(deviation, 3)}), ` +
      'so the data points are the nodes';
  if (source === 'data' && nNodes > 0 && Math.trunc(nNodes) !== n) {
    const why = deviation === null ? 'without f_true' : 'in data mode';
    result.message += `; n_nodes=${Math.trunc(nNodes)} ignored: ${why} the ${n} data points are the nodes`;
  }
  return result;
};

// ---------------------------------------------------------------------------------------
// Registration (ids, params and references exactly as in Python)
// ---------------------------------------------------------------------------------------

const REF_BF = 'Burden & Faires, Numerical Analysis (10th ed.)';
const SPLINE_REFS = [
  'de Boor, A Practical Guide to Splines (rev. ed. 2001), Ch. IV (slope form, CUBSPL)',
  `${REF_BF}, §3.5, Theorems 3.11–3.12`,
];

const SPLINE_PROS = ['C² smooth', 'local error O(h⁴) away from the ends', 'one tridiagonal solve'];

/** Interpolation error of an n-node polynomial interpolant (Python `_ERR_ORDER`). */
const ERR_ORDER = 'error O(hⁿ), n nodes at spacing h: f − p = f⁽ⁿ⁾(ξ)·ω(x)/n!, ω(x) = ∏ⱼ(x − xⱼ)';

export const DOCS: Record<string, MethodDoc> = {
  lagrange: {
    order: 'O(n²) per point',
    rule: 'p(x) = \\sum_{j=0}^{n-1} y_j\\,\\ell_j(x),\\qquad \\ell_j(x) = \\prod_{m \\ne j} \\frac{x - x_m}{x_j - x_m}',
    intuition:
      'Each basis polynomial ℓⱼ is 1 at its own node and 0 at every other node, so the weighted sum Σ yⱼℓⱼ passes through every data point. The picture adds one term yₖℓₖ per step.',
    pros: ['explicit formula', 'coefficients are the data'],
    cons: ['O(n²) per evaluation', 'adding a node changes every ℓⱼ'],
  },
  barycentric: {
    order: 'O(n) per point',
    rule: 'p(x) = \\frac{\\sum_j \\frac{w_j}{x - x_j}\\, y_j}{\\sum_j \\frac{w_j}{x - x_j}},\\qquad w_j = \\frac{1}{\\prod_{m \\ne j}(x_j - x_m)}',
    intuition:
      'The same polynomial as Lagrange, written as a weighted average of the data. One weight per node, updated in O(k) when node k joins; evaluation is O(n), and this second (true) form is forward stable whenever the Lebesgue constant is small, e.g. on Chebyshev nodes (Higham 2004).',
    pros: ['O(n) per evaluation', 'stable', 'nodes can be added cheaply'],
    cons: ['weights can overflow for many widely spaced nodes'],
  },
  newton_divided_differences: {
    order: 'O(n) per point',
    rule: 'p_k(x) = p_{k-1}(x) + f[x_0,\\dots,x_k]\\prod_{j<k}(x - x_j)',
    intuition:
      'Each new node adds one row to the divided-difference table and one term to p. The new term vanishes at every earlier node, so the old interpolation conditions stay satisfied.',
    pros: ['incremental: one new row per node', 'O(n) Horner evaluation'],
    cons: ['round-off grows with the degree on bad node orderings'],
  },
  neville: {
    order: 'O(n²) per point',
    rule: 'Q_{i,j} = \\frac{(x^\\ast - x_{i-j})\\,Q_{i,j-1} - (x^\\ast - x_i)\\,Q_{i-1,j-1}}{x_i - x_{i-j}}',
    intuition:
      'Neville combines two interpolants on overlapping node windows into one on their union, evaluated at a single point x⋆. Column j of the tableau holds the estimates of degree j; their spread shows how far the value has settled.',
    pros: ['no coefficients needed', 'the tableau spread is an error indicator'],
    cons: ['O(n²) per evaluation point'],
  },
  linear_spline: {
    order: 'O(h²)',
    rule: 'S_i(x) = y_i + m_i\\,(x - x_i),\\qquad m_i = \\frac{y_{i+1} - y_i}{x_{i+1} - x_i}',
    intuition:
      'Join consecutive data points by straight segments. The result is continuous, never overshoots, and converges like h² for smooth f.',
    pros: ['monotone and bounded by the data', 'no system to solve'],
    cons: ['kinks at every node', 'only O(h²)'],
  },
  cubic_spline_natural: {
    order: 'O(h⁴) interior',
    rule: 'h_i s_{i-1} + 2(h_{i-1}+h_i)\\,s_i + h_{i-1} s_{i+1} = 3(h_i m_{i-1} + h_{i-1} m_i),\\quad S\'\'(x_0) = S\'\'(x_{n-1}) = 0',
    intuition:
      'Choose the slopes sᵢ = S′(xᵢ) so that the second derivative is continuous at every interior node; the natural end conditions set the curvature to zero at both ends. One tridiagonal solve gives every slope.',
    pros: SPLINE_PROS,
    cons: ['O(h²) near the ends unless f″ vanishes there', 'overshoots at jumps'],
  },
  cubic_spline_clamped: {
    order: 'O(h⁴)',
    rule: 'h_i s_{i-1} + 2(h_{i-1}+h_i)\\,s_i + h_{i-1} s_{i+1} = 3(h_i m_{i-1} + h_{i-1} m_i),\\quad s_0 = f\'_a,\\; s_{n-1} = f\'_b',
    intuition:
      'The C² cubic spline with the end slopes prescribed. With the true derivatives at the ends it is O(h⁴) everywhere; with wrong ones the error stays near the ends.',
    pros: ['O(h⁴) with exact end slopes'],
    cons: ['needs f′ at both ends', 'overshoots at jumps'],
  },
  cubic_spline_not_a_knot: {
    order: 'O(h⁴)',
    rule: 'h_i s_{i-1} + 2(h_{i-1}+h_i)\\,s_i + h_{i-1} s_{i+1} = 3(h_i m_{i-1} + h_{i-1} m_i),\\quad S\'\'\' \\text{ continuous at } x_1, x_{n-2}',
    intuition:
      'The default spline of MATLAB and SciPy: the first two and the last two pieces are one cubic each, so no end derivative is assumed. It is O(h⁴) up to the ends.',
    pros: ['O(h⁴) without end data'],
    cons: ['overshoots at jumps', 'needs four points for the general case'],
  },
  pchip: {
    order: 'O(h³)',
    rule: '\\frac{w_1 + w_2}{s_k} = \\frac{w_1}{m_{k-1}} + \\frac{w_2}{m_k},\\qquad s_k = 0 \\text{ if } m_{k-1} m_k \\le 0',
    intuition:
      'A C¹ cubic whose node slopes are a weighted harmonic mean of the neighboring secants, and zero at a local extremum. The slopes stay in the Fritsch–Carlson region 0 ≤ α, β ≤ 3, so monotone data give a monotone curve.',
    pros: ['shape preserving: no overshoot', 'local'],
    cons: ['only C¹', 'O(h³), flattens true extrema'],
  },
  chebyshev_interpolation: {
    order: 'geometric',
    rule: 'p(x) = \\sum_{k=0}^{N-1} c_k\\,T_k(u),\\qquad c_k = \\frac{2}{N}\\sum_{j} f(x_j)\\cos\\frac{\\pi k (j+\\frac12)}{N}',
    intuition:
      'Sample f at the roots of T_N and expand the interpolant in Chebyshev polynomials. For analytic f the coefficients |cₖ| decay geometrically, so the truncated series converges as terms are added.',
    pros: ['near-best polynomial approximation', 'coefficient decay measures smoothness'],
    cons: ['needs f at the Chebyshev nodes (else it falls back to the data)'],
  },
};

const NO_PARAMS: never[] = [];

registerMethod(
  {
    id: 'lagrange',
    family: 'interpolation',
    name: 'Lagrange form',
    params: NO_PARAMS,
    needs: ['data'],
    order: ERR_ORDER,
    summary:
      'Write p as Σ y_j ℓ_j, where the basis polynomial ℓ_j is 1 at x_j and 0 at every other node; each evaluation costs O(n²).',
    references: [`${REF_BF}, §3.1, Theorem 3.2`],
  },
  lagrange,
  DOCS.lagrange,
);

registerMethod(
  {
    id: 'barycentric',
    family: 'interpolation',
    name: 'Barycentric form',
    params: NO_PARAMS,
    needs: ['data'],
    order: ERR_ORDER,
    summary: 'Precompute one weight per node in O(n²); then evaluate p with a stable O(n) weighted average.',
    references: [
      'Berrut & Trefethen (2004), Barycentric Lagrange Interpolation, SIAM Review 46(3), ' +
        'eq. (3.2) weights, eq. (4.2) second form',
      'Higham (2004), The numerical stability of barycentric Lagrange interpolation, ' +
        'IMA J. Numer. Anal. 24',
    ],
  },
  barycentric,
  DOCS.barycentric,
);

registerMethod(
  {
    id: 'newton_divided_differences',
    family: 'interpolation',
    name: 'Newton divided differences',
    params: NO_PARAMS,
    needs: ['data'],
    order: ERR_ORDER,
    summary:
      'Add one node at a time; each new node adds one divided difference and one term (O(n²) table, O(n) per evaluation by nested multiplication).',
    references: [`${REF_BF}, §3.3, Alg. 3.2`],
  },
  newtonDividedDifferences,
  DOCS.newton_divided_differences,
);

registerMethod(
  {
    id: 'neville',
    family: 'interpolation',
    name: "Neville's algorithm",
    params: [
      param.float('x_frac', 0.95, {
        min: 0.0,
        max: 1.0,
        help: 'Evaluation point x* = a + x_frac·(b − a) inside the plotting domain [a, b].',
        label: 'Evaluation point',
        tex: 'x^\\ast',
      }),
    ],
    needs: ['data'],
    order: ERR_ORDER,
    summary: 'Evaluate p(x*) directly by combining interpolants on ever larger sets of nodes (O(n²) per point).',
    references: [`${REF_BF}, §3.2, Alg. 3.1 (Theorem 3.5)`],
  },
  neville,
  DOCS.neville,
);

registerMethod(
  {
    id: 'linear_spline',
    family: 'interpolation',
    name: 'Linear spline',
    params: NO_PARAMS,
    needs: ['data'],
    order: 'error O(h²) for C² data',
    summary: 'Join consecutive data points with straight lines.',
    references: [`${REF_BF}, §3.5 (piecewise-linear interpolation)`],
  },
  linearSpline,
  DOCS.linear_spline,
);

registerMethod(
  {
    id: 'cubic_spline_natural',
    family: 'interpolation',
    name: 'Natural cubic spline',
    params: NO_PARAMS,
    needs: ['data'],
    order: 'error O(h²) at the ends, O(h⁴) inside',
    summary: 'The C² piecewise cubic through the data with zero curvature at both ends.',
    references: [...SPLINE_REFS, 'Burden & Faires (10th ed.), Alg. 3.4 (same spline, c-form)'],
  },
  (problem: ProblemArg) => cubicSpline('cubic_spline_natural', problem, 'natural'),
  DOCS.cubic_spline_natural,
);

registerMethod(
  {
    id: 'cubic_spline_clamped',
    family: 'interpolation',
    name: 'Clamped cubic spline',
    params: [
      param.float('fprime_a', 0.0, {
        min: -50.0,
        max: 50.0,
        help: 'Imposed end slope S′(x₀).',
        label: 'Left end slope',
        tex: "S'(x_0)",
      }),
      param.float('fprime_b', 0.0, {
        min: -50.0,
        max: 50.0,
        help: 'Imposed end slope S′(xₙ₋₁).',
        label: 'Right end slope',
        tex: "S'(x_{n-1})",
      }),
    ],
    needs: ['data'],
    order: 'O(h⁴) with exact end slopes',
    summary: 'The C² piecewise cubic through the data with prescribed slopes at both ends.',
    references: [...SPLINE_REFS, 'Burden & Faires (10th ed.), Alg. 3.5 (same spline, c-form)'],
  },
  (problem: ProblemArg, { fprime_a = 0.0, fprime_b = 0.0 }) => {
    const fa = Number(fprime_a),
      fb = Number(fprime_b);
    if (!(Number.isFinite(fa) && Number.isFinite(fb)))
      throw new Error('end slopes must be finite');
    return cubicSpline('cubic_spline_clamped', problem, 'clamped', fa, fb);
  },
  DOCS.cubic_spline_clamped,
);

registerMethod(
  {
    id: 'cubic_spline_not_a_knot',
    family: 'interpolation',
    name: 'Not-a-knot cubic spline',
    params: NO_PARAMS,
    needs: ['data'],
    order: 'O(h⁴)',
    summary: 'The C² cubic spline whose first two and last two pieces are single cubics.',
    references: SPLINE_REFS,
  },
  (problem: ProblemArg) => cubicSpline('cubic_spline_not_a_knot', problem, 'not-a-knot'),
  DOCS.cubic_spline_not_a_knot,
);

registerMethod(
  {
    id: 'pchip',
    family: 'interpolation',
    name: 'PCHIP (monotone cubic)',
    params: NO_PARAMS,
    needs: ['data'],
    order: 'O(h³) on smooth monotone data',
    summary: 'A C¹ piecewise cubic whose slopes are chosen so monotone data stay monotone.',
    references: [
      'Fritsch & Carlson (1980), Monotone piecewise cubic interpolation, SIAM J. Numer. ' +
        'Anal. 17(2) (monotonicity region)',
      'Fritsch & Butland (1984), A method for constructing local monotone piecewise cubic ' +
        'interpolants, SIAM J. Sci. Stat. Comput. 5(2) (weighted harmonic mean)',
      'Moler, Numerical Computing with MATLAB (2004), Ch. 3 (pchiptx end slopes)',
    ],
  },
  pchip,
  DOCS.pchip,
);

registerMethod(
  {
    id: 'chebyshev_interpolation',
    family: 'interpolation',
    name: 'Chebyshev interpolation',
    params: [
      param.int('n_nodes', 0, {
        min: 0,
        max: 200,
        help:
          'Number of Chebyshev nodes when the data are samples of f_true (0 = the ' +
          'dataset size). Ignored otherwise (no f_true, or noisy data): the data points ' +
          'are the nodes.',
        label: 'Chebyshev nodes (0 = data size)',
        tex: 'N',
      }),
    ],
    needs: ['data'],
    order: 'geometric for analytic f',
    summary:
      'Sample f at the Chebyshev nodes and expand the interpolant in Chebyshev polynomials.',
    references: [
      `${REF_BF}, §8.3 (Chebyshev nodes)`,
      'Press et al., Numerical Recipes (3rd ed.), §5.8 (coefficients by discrete ' +
        'orthogonality) and §5.4 (Clenshaw)',
      'Trefethen, Approximation Theory and Approximation Practice (2013), Ch. 3–4',
    ],
  },
  chebyshevInterpolation,
  DOCS.chebyshev_interpolation,
);

export type { Vector };
