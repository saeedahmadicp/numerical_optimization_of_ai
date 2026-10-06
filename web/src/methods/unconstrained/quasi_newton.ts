/**
 * Quasi-Newton methods — TS port of `numopt.unconstrained.quasi_newton`
 * (src/numopt/unconstrained/quasi_newton.py).
 *
 * Each method keeps an approximation H_k of the inverse Hessian, moves along p_k = −H_k∇f(x_k)
 * with a line search (α₀ = 1, strong Wolfe by default), and updates H_k with
 * s_k = x_{k+1} − x_k, y_k = ∇f(x_{k+1}) − ∇f(x_k) so that H_{k+1} y_k = s_k:
 *
 *   bfgs           N&W Alg. 6.1, eq. 6.17 (expanded, exactly symmetric form)
 *   dfp            N&W eq. 6.15
 *   sr1            N&W eq. 6.25 with the skip rule eq. 6.26 (dual form)
 *   broyden_class  N&W eq. 6.32 in inverse form with the dual parameter Φ
 *   lbfgs          N&W Alg. 7.5 with the two-loop recursion Alg. 7.4, H₀ᵏ = γ_k I (eq. 7.20)
 *
 * Shared rules (exactly as in Python): H₀ = I; initial scaling γ = yᵀs/yᵀy before the first
 * applied update (bfgs, dfp, broyden_class); curvature test yᵀs > 1e-10‖s‖‖y‖ with finite ρ and γ;
 * a non-finite H⁺ is skipped; the descent safeguard cos θ > 1e-8 resets H to I (L-BFGS: clears the
 * memory); first-order stopping test ‖∇f‖∞ ≤ gtol; n_hev = 0.
 *
 * Step.info keys (snake_case, as in the Python docstring): grad, direction, alpha, trials, reset,
 * s, y, curvature, rho, update, gamma, H (n ≤ 2, else null); sr1: denominator; broyden_class:
 * phi_inverse; lbfgs: memory.
 */
import { registerMethod, param, type MethodDoc } from '../../core/registry';
import type { Matrix, MethodFn, Result, Step, Vector } from '../../core/types';
import { formatG, formatRepr, search } from '../line_search/methods';
import {
  EPS,
  MethodInputError,
  cosAngle,
  dotp,
  finite,
  maxAbs,
  norm2,
  resolve,
  slopeFailure,
  validateCommon,
  type SmoothProblem,
} from './newton';

/** Curvature test yᵀs > CURVATURE_EPS·‖s‖‖y‖ for the BFGS-type updates. */
const CURVATURE_EPS = 1e-10;
/** SR1 safeguard constant r of N&W eq. 6.26. */
const SR1_R = 1e-8;
/** Descent test cos θ > η (Zoutendijk, N&W Thm 3.2). */
const DESCENT_COS = 1e-8;

export const LINE_SEARCHES = ['strong_wolfe', 'weak_wolfe', 'backtracking', 'goldstein'] as const;

const g3 = (v: number) => formatG(v, 3);

type Info = Record<string, unknown>;

// ---------------------------------------------------------------------------------------
// Small array helpers (numpy expressions, in the Python order)
// ---------------------------------------------------------------------------------------

function eye(n: number): Matrix {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1.0 : 0.0)),
  );
}

/** `H @ v`. */
function matvec(H: Matrix, v: Vector): Vector {
  return H.map((row) => dotp(row, v));
}

/** `np.outer(a, b)`. */
function outer(a: Vector, b: Vector): Matrix {
  return a.map((ai) => b.map((bj) => ai * bj));
}

function zip2(A: Matrix, B: Matrix, op: (a: number, b: number) => number): Matrix {
  return A.map((row, i) => row.map((v, j) => op(v, B[i][j])));
}

const madd = (A: Matrix, B: Matrix) => zip2(A, B, (a, b) => a + b);
const msub = (A: Matrix, B: Matrix) => zip2(A, B, (a, b) => a - b);
const mscale = (c: number, A: Matrix) => A.map((row) => row.map((v) => c * v));
const mdiv = (A: Matrix, c: number) => A.map((row) => row.map((v) => v / c));

/**
 * ‖v‖₂ as max|v_i|·‖v / max|v_i|‖₂, which does not underflow for a tiny nonzero v
 * (`numpy.linalg.norm` forms √(vᵀv), which is 0 for v = 1e-200).
 */
export function scaledNorm(v: Vector): number {
  const vMax = maxAbs(v);
  if (!Number.isFinite(vMax) || vMax === 0.0) return vMax;
  return vMax * norm2(v.map((e) => e / vMax));
}

/** yᵀs > 1e-10‖s‖‖y‖ with a finite ρ = 1/yᵀs and a finite, positive γ = yᵀs/yᵀy. */
export function curvatureOk(s: Vector, y: Vector, ys: number): boolean {
  if (!(Number.isFinite(ys) && ys > 0.0)) return false;
  const yy = dotp(y, y);
  if (!(Number.isFinite(yy) && yy > 0.0)) return false;
  if (!(Number.isFinite(1.0 / ys) && Number.isFinite(ys / yy))) return false;
  return ys > CURVATURE_EPS * scaledNorm(s) * scaledNorm(y);
}

// ---------------------------------------------------------------------------------------
// Inverse-Hessian approximations
// ---------------------------------------------------------------------------------------

interface Approximation {
  /** H g. */
  apply(g: Vector): Vector;
  reset(): void;
  /** Update with the pair (s, y); `sBs` = sᵀBs for the B = H⁻¹ that produced p. */
  update(s: Vector, y: Vector, sBs: number): Info;
  /** The (explicit or implicit) H as an n×n array (for the n ≤ 2 info overlay). */
  matrix(): Matrix;
  /** The method-specific info keys at k = 0. */
  emptyInfo(): Info;
  /** The incoming info keys when no update was attempted (a non-finite pair). */
  unchangedInfo(): Info;
}

type Formula = (H: Matrix, s: Vector, y: Vector, ys: number, sBs: number) => [Matrix, Info] | null;

/** An explicit symmetric H (BFGS, DFP, Broyden class, SR1 via its own update). */
class Dense implements Approximation {
  readonly n: number;
  H: Matrix;
  scaled = false;
  readonly scaleInitial: boolean;
  private readonly formula: Formula;
  private readonly extraNone: Info;

  constructor(n: number, formula: Formula, extraNone: Info = {}, scaleInitial = true) {
    this.n = n;
    this.H = eye(n);
    this.formula = formula;
    this.extraNone = extraNone;
    this.scaleInitial = scaleInitial;
  }

  apply(g: Vector): Vector {
    return matvec(this.H, g);
  }

  reset(): void {
    this.H = eye(this.n);
    this.scaled = false;
  }

  matrix(): Matrix {
    return this.H.map((row) => row.slice());
  }

  emptyInfo(): Info {
    return { ...this.extraNone };
  }

  unchangedInfo(): Info {
    return { rho: null, update: 'skipped', gamma: null, ...this.emptyInfo() };
  }

  /** N&W eq. 6.20: H₀ ← (yᵀs / yᵀy) I before the first update. Returns γ or null. */
  private maybeScale(y: Vector, ys: number): number | null {
    if (!this.scaleInitial || this.scaled) return null;
    const gamma = ys / dotp(y, y);
    this.H = mscale(gamma, eye(this.n));
    this.scaled = true;
    return gamma;
  }

  update(s: Vector, y: Vector, sBsIn: number): Info {
    let sBs = sBsIn;
    const ys = dotp(y, s);
    if (!curvatureOk(s, y, ys))
      return { rho: null, update: 'skipped', gamma: null, ...this.extraNone };
    const gamma = this.maybeScale(y, ys);
    // sᵀBs for the rescaled B₀ = I/γ.
    if (gamma !== null) sBs = dotp(s, s) / gamma;
    const out = this.formula(this.H, s, y, ys, sBs);
    // A non-finite H⁺ (an overflow in a rank-one term) is rejected: H is kept.
    if (out === null || !finite(out[0]))
      return { rho: null, update: 'skipped', gamma, ...this.extraNone };
    this.H = out[0];
    return { rho: 1.0 / ys, update: 'applied', gamma, ...out[1] };
  }
}

/** N&W eq. 6.17 expanded: H − ρ(u sᵀ + s uᵀ) + (ρ² yᵀu + ρ) s sᵀ with u = H y. */
const bfgsFormula: Formula = (H, s, y, ys) => {
  const rho = 1.0 / ys;
  const u = matvec(H, y);
  const Hn = madd(
    msub(H, mscale(rho, madd(outer(u, s), outer(s, u)))),
    mscale(rho * rho * dotp(y, u) + rho, outer(s, s)),
  );
  return [Hn, {}];
};

/** N&W eq. 6.15: H − (Hy)(Hy)ᵀ / (yᵀHy) + s sᵀ / (yᵀs). */
const dfpFormula: Formula = (H, s, y, ys) => {
  const u = matvec(H, y);
  const yHy = dotp(y, u);
  // Reachable when rounding has made H indefinite, or when yᵀHy underflows to 0.
  if (!(Number.isFinite(yHy) && yHy > 0.0)) return null;
  return [madd(msub(H, mdiv(outer(u, u), yHy)), mdiv(outer(s, s), ys)), {}];
};

/**
 * The restricted Broyden class (N&W eq. 6.32) in inverse form:
 * H⁺ = H − u uᵀ/(yᵀu) + s sᵀ/(yᵀs) + Φ (yᵀu) w wᵀ, u = H y, w = s/(yᵀs) − u/(yᵀu),
 * Φ = (1 − φ) / (1 − φ + φ μ), μ = (yᵀHy)(sᵀBs)/(yᵀs)².
 */
function broydenFormula(phi: number): Formula {
  return (H, s, y, ys, sBs) => {
    const u = matvec(H, y);
    const yHy = dotp(y, u);
    if (!(Number.isFinite(yHy) && yHy > 0.0 && Number.isFinite(sBs) && sBs > 0.0)) return null;
    const mu = (yHy / ys) * (sBs / ys);
    if (!Number.isFinite(mu)) return null;
    const bigPhi = (1.0 - phi) / (1.0 - phi + phi * mu);
    const w = s.map((si, i) => si / ys - u[i] / yHy);
    const Hn = madd(
      madd(msub(H, mdiv(outer(u, u), yHy)), mdiv(outer(s, s), ys)),
      mscale(bigPhi * yHy, outer(w, w)),
    );
    return [Hn, { phi_inverse: bigPhi }];
  };
}

/** SR1 (N&W eq. 6.25, skip rule eq. 6.26 in dual form); H₀ = I with no scaling. */
class SR1 extends Dense {
  constructor(n: number) {
    super(n, () => null, { denominator: null }, false);
  }

  override update(s: Vector, y: Vector): Info {
    const Hy = matvec(this.H, y);
    const v = s.map((si, i) => si - Hy[i]);
    const den = dotp(v, y);
    const vNorm = scaledNorm(v);
    const yNorm = scaledNorm(y);
    const skipped: Info = { rho: null, update: 'skipped', gamma: null, denominator: den };
    const ok =
      vNorm > 0.0 &&
      yNorm > 0.0 &&
      Number.isFinite(den) &&
      den !== 0.0 &&
      Math.abs(den) >= SR1_R * yNorm * vNorm;
    if (!ok) return skipped;
    const H = madd(this.H, mdiv(outer(v, v), den));
    // vᵀy near the underflow threshold: H⁺ or ρ overflows, so H is kept.
    if (!(finite(H) && Number.isFinite(1.0 / den))) return skipped;
    this.H = H;
    return { rho: 1.0 / den, update: 'applied', gamma: null, denominator: den };
  }
}

/** N&W Alg. 7.4: H_k g for H₀ᵏ = γI and the stored pairs (oldest first). */
export function twoLoop(g: Vector, pairs: [Vector, Vector, number][], gamma: number): Vector {
  let q = g.slice();
  const a = new Array<number>(pairs.length).fill(0.0);
  for (let i = pairs.length - 1; i >= 0; i--) {
    const [s, y, rho] = pairs[i];
    a[i] = rho * dotp(s, q);
    const ai = a[i];
    q = q.map((qj, j) => qj - ai * y[j]);
  }
  let r = q.map((qj) => gamma * qj);
  pairs.forEach(([s, y, rho], i) => {
    const beta = rho * dotp(y, r);
    const c = a[i] - beta;
    r = r.map((rj, j) => rj + c * s[j]);
  });
  return r;
}

/** Limited-memory BFGS: the last m pairs and H₀ᵏ = γ_k I (N&W eq. 7.20). */
class LBFGS implements Approximation {
  readonly n: number;
  readonly m: number;
  pairs: [Vector, Vector, number][] = [];
  gamma = 1.0;

  constructor(n: number, m: number) {
    this.n = n;
    this.m = m;
  }

  apply(g: Vector): Vector {
    return twoLoop(g, this.pairs, this.gamma);
  }

  reset(): void {
    this.pairs = [];
    this.gamma = 1.0;
  }

  update(s: Vector, y: Vector): Info {
    const ys = dotp(y, s);
    const applied = curvatureOk(s, y, ys);
    if (applied) {
      if (this.pairs.length === this.m) this.pairs.shift();
      this.pairs.push([s.slice(), y.slice(), 1.0 / ys]);
      // eq. 7.20 with the newest stored pair.
      this.gamma = ys / dotp(y, y);
    }
    return {
      rho: applied ? 1.0 / ys : null,
      update: applied ? 'applied' : 'skipped',
      gamma: this.gamma,
      memory: this.pairs.length,
    };
  }

  matrix(): Matrix {
    const cols = eye(this.n).map((e) => this.apply(e));
    return Array.from({ length: this.n }, (_, i) => cols.map((c) => c[i]));
  }

  emptyInfo(): Info {
    return { memory: 0 };
  }

  unchangedInfo(): Info {
    // The memory and γ_k are kept: they still define the H of the "H" info key.
    return { rho: null, update: 'skipped', gamma: this.gamma, memory: this.pairs.length };
  }
}

// ---------------------------------------------------------------------------------------
// The shared quasi-Newton loop
// ---------------------------------------------------------------------------------------

function lineSearchFailure(k: number, g: Vector, p: Vector, fx: number, why: string): string {
  return (
    `line search failed at iteration ${k} (‖∇f‖∞ = ${g3(maxAbs(g))}): ${why}; ` +
    `predicted decrease |∇fᵀp| = ${g3(Math.abs(dotp(g, p)))} vs rounding level of f ` +
    `ε·max(1, |f|) = ${g3(EPS * Math.max(1.0, Math.abs(fx)))}`
  );
}

export interface QuasiNewtonOptions {
  x0?: unknown;
  gtol?: number;
  max_iter?: number;
  line_search?: string;
}

/** Generic line-search quasi-Newton iteration (N&W Alg. 6.1 pattern). */
function quasiNewton(
  method: string,
  problem: SmoothProblem,
  opts: QuasiNewtonOptions,
  make: (n: number) => Approximation,
): Result {
  const gtol = Number(opts.gtol ?? 1e-8);
  const maxIter = Number(opts.max_iter ?? 500);
  const lineSearch = String(opts.line_search ?? 'strong_wolfe');
  validateCommon(gtol, maxIter, lineSearch, LINE_SEARCHES);
  const { x: x0, oracle } = resolve(problem, opts.x0);
  // Quasi-Newton never evaluates ∇²f: only f and ∇f are counted.
  const { f, grad } = oracle;
  let x = x0;
  const n = x.length;
  const approx = make(n);
  let fx = f(x);
  let g = grad(x);
  if (!finite(fx, g))
    throw new MethodInputError(
      `${method}: f and ∇f must be finite at x0; got f=${formatRepr(fx)}, ∇f=[${g.map(formatRepr).join(' ')}]`,
    );
  const trace: Step[] = [];
  const result = (converged: boolean, message: string, k: number): Result => ({
    method,
    x: x.slice(),
    fun: fx,
    converged,
    message,
    nIter: k,
    nFev: f.n,
    nGev: grad.n,
    nHev: 0,
    trace,
    extra: {},
  });
  const hInfo = (): Matrix | null => (n <= 2 ? approx.matrix() : null);

  let info: Info = {
    grad: g.slice(),
    direction: null,
    alpha: null,
    trials: [],
    reset: false,
    s: null,
    y: null,
    curvature: null,
    rho: null,
    update: null,
    gamma: null,
    ...approx.emptyInfo(),
    H: hInfo(),
  };
  trace.push({ k: 0, x: x.slice(), fun: fx, gradNorm: norm2(g), stepSize: null, info });
  let k = 0;
  for (;;) {
    const gnorm = maxAbs(g);
    if (gnorm <= gtol) return result(true, `‖∇f‖∞ = ${g3(gnorm)} ≤ gtol`, k);
    if (k === maxIter)
      return result(false, `reached max_iter=${maxIter} (‖∇f‖∞ = ${g3(gnorm)})`, k);
    let p = approx.apply(g).map((v) => -v);
    const reset = !(cosAngle(g, p) > DESCENT_COS);
    if (reset) {
      approx.reset();
      p = g.map((v) => -v);
    }
    const slope = dotp(g, p);
    if (!(Number.isFinite(slope) && slope < 0.0)) {
      // NOTE (Python): the line search needs a finite ∇fᵀp < 0 in floating point. Here p passed
      // the descent test (or p = −∇f), so a non-finite ∇fᵀp overflowed (‖∇f‖‖p‖ ≳ 1e308) and
      // ∇fᵀp ≥ 0 underflowed: ∇f is near the underflow threshold, far below any decrease that f
      // can resolve.
      return result(false, lineSearchFailure(k + 1, g, p, fx, slopeFailure(slope)), k);
    }
    const ls = search(lineSearch, f, grad, x, p, { f0: fx, g0: g, alpha0: 1.0 });
    if (!ls.success) return result(false, lineSearchFailure(k + 1, g, p, fx, ls.message), k);
    const xNew = x.map((v, i) => v + ls.alpha * p[i]);
    const gNew = ls.gNew !== null ? ls.gNew.slice() : grad(xNew);
    const s = xNew.map((v, i) => v - x[i]);
    const y = gNew.map((v, i) => v - g[i]);
    const ys = dotp(y, s);
    // B_k s_k = −α_k ∇f(x_k), so sᵀBs = −α ∇fᵀs needs no inverse (Broyden class only).
    const sBs = -ls.alpha * dotp(g, s);
    k += 1;
    x = xNew;
    fx = Number(ls.fNew);
    g = gNew;
    const upd = finite(g, s, y) ? approx.update(s, y, sBs) : approx.unchangedInfo();
    info = {
      grad: g.slice(),
      direction: p.slice(),
      alpha: ls.alpha,
      trials: ls.trials.map(([a, v]) => [a, v]),
      reset,
      s,
      y,
      curvature: ys,
      ...upd,
      H: hInfo(),
    };
    trace.push({ k, x: x.slice(), fun: fx, gradNorm: norm2(g), stepSize: ls.alpha, info });
    if (!finite(g)) return result(false, `∇f is not finite at iteration ${k}`, k);
  }
}

// ---------------------------------------------------------------------------------------
// Public functions (Python: bfgs, dfp, sr1, broyden_class, lbfgs)
// ---------------------------------------------------------------------------------------

export function bfgs(problem: SmoothProblem, opts: QuasiNewtonOptions = {}): Result {
  return quasiNewton('bfgs', problem, opts, (n) => new Dense(n, bfgsFormula));
}

export function dfp(problem: SmoothProblem, opts: QuasiNewtonOptions = {}): Result {
  return quasiNewton('dfp', problem, opts, (n) => new Dense(n, dfpFormula));
}

export function sr1(problem: SmoothProblem, opts: QuasiNewtonOptions = {}): Result {
  return quasiNewton('sr1', problem, opts, (n) => new SR1(n));
}

export function broydenClass(
  problem: SmoothProblem,
  opts: QuasiNewtonOptions & { phi?: number } = {},
): Result {
  const phi = Number(opts.phi ?? 0.5);
  if (!(Number.isFinite(phi) && phi >= 0.0 && phi <= 1.0))
    throw new MethodInputError(
      `phi must lie in [0, 1] (the restricted Broyden class), got ${formatRepr(phi)}`,
    );
  return quasiNewton(
    'broyden_class',
    problem,
    opts,
    (n) => new Dense(n, broydenFormula(phi), { phi_inverse: null }),
  );
}

export function lbfgs(
  problem: SmoothProblem,
  opts: QuasiNewtonOptions & { m?: number } = {},
): Result {
  const m = Number(opts.m ?? 10);
  if (!Number.isInteger(m) || m < 1)
    throw new MethodInputError(`m must be a positive integer, got ${String(opts.m)}`);
  return quasiNewton('lbfgs', problem, opts, (n) => new LBFGS(n, m));
}

// ---------------------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------------------

const P_GTOL = param.float('gtol', 1e-8, {
  min: 1e-14,
  max: 1e-2,
  log: true,
  help: 'Stop when ‖∇f(x)‖∞ ≤ gtol.',
  label: 'Gradient tolerance',
  tex: '\\|\\nabla f\\|_\\infty \\le',
});
const P_MAX_ITER = param.int('max_iter', 500, {
  min: 1,
  max: 100_000,
  help: 'Iteration limit.',
  label: 'Max iterations',
});
const P_LINE_SEARCH = param.choice('line_search', 'strong_wolfe', [...LINE_SEARCHES], {
  help: 'Step-length rule (tries α = 1 first). Wolfe searches guarantee yᵀs > 0.',
  label: 'Line search',
});
const P_PHI = param.float('phi', 0.5, {
  min: 0.0,
  max: 1.0,
  help: 'Broyden parameter φ ∈ [0, 1]: φ = 0 is BFGS, φ = 1 is DFP.',
  label: 'Broyden parameter',
  tex: '\\phi',
});
const P_M = param.int('m', 10, {
  min: 1,
  max: 50,
  help: 'Memory: number of (s, y) pairs kept (N&W suggest 3 ≤ m ≤ 20).',
  label: 'Memory',
  tex: 'm',
});

const COMMON_QUANTITIES = [
  { tex: '\\alpha_k', key: 'stepSize', label: 'Step length' },
  { tex: 'y_k^\\top s_k', key: 'info.curvature', label: 'Curvature' },
  { tex: '\\text{update}', key: 'info.update', label: 'Update' },
];

const SECANT = 'H_{k+1} y_k = s_k';

const DOCS: Record<string, MethodDoc> = {
  bfgs: {
    rule:
      'H_{k+1} = (I - \\rho_k s_k y_k^\\top) H_k (I - \\rho_k y_k s_k^\\top) + \\rho_k s_k s_k^\\top, ' +
      '\\quad \\rho_k = \\frac{1}{y_k^\\top s_k}',
    intuition:
      'Learn the curvature from how the gradient changes along each step. A rank-two correction ' +
      `makes the inverse-Hessian estimate satisfy the secant equation ${SECANT} and stay positive definite.`,
    order: 'superlinear',
    pros: ['Only gradients needed', 'Self-correcting: recovers from a bad H quickly'],
    cons: ['O(n²) memory and work per step'],
    quantities: COMMON_QUANTITIES,
  },
  dfp: {
    rule: 'H_{k+1} = H_k - \\frac{H_k y_k y_k^\\top H_k}{y_k^\\top H_k y_k} + \\frac{s_k s_k^\\top}{y_k^\\top s_k}',
    intuition:
      'The first quasi-Newton update: a rank-two secant correction of the inverse Hessian, the dual ' +
      'of BFGS. It corrects a poor estimate much more slowly than BFGS does.',
    order: 'superlinear',
    pros: ['Only gradients needed', 'Exact on quadratics with exact line searches'],
    cons: ['Weak self-correction: slow on curved valleys'],
    quantities: COMMON_QUANTITIES,
  },
  sr1: {
    rule: 'H_{k+1} = H_k + \\frac{v_k v_k^\\top}{v_k^\\top y_k}, \\quad v_k = s_k - H_k y_k',
    intuition:
      'A symmetric rank-one secant update. It can model negative curvature, so H may become ' +
      'indefinite; then the method resets to steepest descent for that step.',
    order: 'superlinear',
    pros: ['Often a more accurate Hessian estimate than BFGS', 'Models negative curvature'],
    cons: ['Updates must be skipped when vᵀy is tiny', 'H can be indefinite'],
    quantities: [
      { tex: '\\alpha_k', key: 'stepSize', label: 'Step length' },
      { tex: 'v_k^\\top y_k', key: 'info.denominator', label: 'Denominator' },
      { tex: '\\text{reset}', key: 'info.reset', label: 'Reset' },
    ],
  },
  broyden_class: {
    rule: 'B_{k+1} = (1 - \\phi)\\,B^{\\mathrm{BFGS}}_{k+1} + \\phi\\,B^{\\mathrm{DFP}}_{k+1}, \\quad \\phi \\in [0, 1]',
    intuition:
      'A one-parameter blend of the BFGS (φ = 0) and DFP (φ = 1) updates. Every member satisfies the ' +
      'secant equation and keeps the approximation positive definite.',
    order: 'superlinear',
    pros: ['Interpolates between BFGS and DFP'],
    cons: ['No member beats BFGS in practice'],
    quantities: [
      ...COMMON_QUANTITIES,
      { tex: '\\Phi', key: 'info.phi_inverse', label: 'Inverse-form parameter' },
    ],
  },
  lbfgs: {
    rule:
      'p_k = -H_k \\nabla f(x_k), \\quad H_k \\text{ from the last } m \\text{ pairs } (s_i, y_i), ' +
      '\\; H_k^0 = \\gamma_k I',
    intuition:
      'BFGS without the matrix: keep only the last m step and gradient-change pairs and apply the ' +
      'inverse-Hessian estimate with the two-loop recursion.',
    order: 'linear',
    pros: ['O(mn) memory and work: scales to large n'],
    cons: ['Forgets curvature older than m steps'],
    quantities: [
      ...COMMON_QUANTITIES,
      { tex: '\\gamma_k', key: 'info.gamma', label: 'Initial scaling' },
      { tex: '|\\mathcal{M}|', key: 'info.memory', label: 'Stored pairs' },
    ],
  },
};

type Fn = (problem: SmoothProblem, opts: QuasiNewtonOptions) => Result;
const wrap =
  (fn: Fn): MethodFn<SmoothProblem> =>
  (problem, o) =>
    fn(problem, o as QuasiNewtonOptions);

const BASE = { family: 'unconstrained' as const, needs: ['f', 'grad'] };

registerMethod(
  {
    ...BASE,
    id: 'bfgs',
    name: 'BFGS',
    params: [P_GTOL, P_MAX_ITER, P_LINE_SEARCH],
    order: 'superlinear',
    summary:
      'Build an inverse-Hessian estimate from gradient changes; the most popular quasi-Newton.',
    references: [
      'Nocedal & Wright (2006), Algorithm 6.1, eqs. 6.17 and 6.20',
      'Broyden, Fletcher, Goldfarb & Shanno (1970)',
    ],
  },
  wrap(bfgs),
  DOCS.bfgs,
);

registerMethod(
  {
    ...BASE,
    id: 'dfp',
    name: 'DFP',
    params: [P_GTOL, P_MAX_ITER, P_LINE_SEARCH],
    order: 'superlinear',
    summary:
      "The first quasi-Newton method: a rank-two update of the inverse Hessian (BFGS's dual).",
    references: ['Nocedal & Wright (2006), eq. 6.15', 'Davidon (1959); Fletcher & Powell (1963)'],
  },
  wrap(dfp),
  DOCS.dfp,
);

registerMethod(
  {
    ...BASE,
    id: 'sr1',
    name: 'SR1 (symmetric rank-one)',
    params: [P_GTOL, P_MAX_ITER, P_LINE_SEARCH],
    order: 'superlinear (n+1-step superlinear)',
    summary: 'A rank-one secant update that can model negative curvature (H may be indefinite).',
    references: ['Nocedal & Wright (2006), §6.2, eqs. 6.25–6.26'],
  },
  wrap(sr1),
  DOCS.sr1,
);

registerMethod(
  {
    ...BASE,
    id: 'broyden_class',
    name: 'Broyden class (φ)',
    params: [P_GTOL, P_MAX_ITER, P_LINE_SEARCH, P_PHI],
    order: 'superlinear',
    summary: 'Blend BFGS (φ = 0) and DFP (φ = 1): the restricted Broyden family of updates.',
    references: [
      'Nocedal & Wright (2006), §6.3, eq. 6.32',
      'Fletcher (1987), Practical Methods of Optimization, §3.4',
    ],
  },
  wrap(broydenClass),
  DOCS.broyden_class,
);

registerMethod(
  {
    ...BASE,
    id: 'lbfgs',
    name: 'L-BFGS',
    params: [P_GTOL, P_MAX_ITER, P_LINE_SEARCH, P_M],
    order: 'linear (fast; superlinear only in the limit m → ∞)',
    summary: 'BFGS that keeps only the last m step/gradient pairs: O(mn) memory and work.',
    references: [
      'Nocedal & Wright (2006), Algorithms 7.4 (two-loop recursion) and 7.5, eq. 7.20',
      'Liu & Nocedal (1989)',
    ],
  },
  wrap(lbfgs),
  DOCS.lbfgs,
);
