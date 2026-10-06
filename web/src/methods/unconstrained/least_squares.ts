/**
 * Nonlinear least squares — TS port of `numopt.unconstrained.least_squares`
 * (src/numopt/unconstrained/least_squares.py): minimize f(x) = ½‖r(x)‖² with r: ℝⁿ → ℝᵐ and its
 * Jacobian J.
 *
 * Both methods replace r near x by its linearization r(x + h) ≈ r + J h, i.e. f by the model
 *
 *     L(h) = ½‖r + J h‖² = f + hᵀg + ½ hᵀJᵀJ h,      g = ∇f = Jᵀr
 *
 * (Nocedal & Wright (2006), §10.3; Madsen, Nielsen & Tingleff (2004), eq. 3.7b). Gauss–Newton
 * minimizes L exactly; Levenberg–Marquardt adds the damping term ½μ‖h‖².
 *
 * Linear algebra: every subproblem is solved from the thin SVD J = U diag(σ) Vᵀ, computed once
 * per Jacobian, never from the normal equations:
 *
 *     Gauss–Newton:         p = −V diag(1/σ) Uᵀ r              (min ‖r + J p‖)
 *     Levenberg–Marquardt:  h = −V diag(σ/(σ² + μ)) Uᵀ r       ((JᵀJ + μI) h = −g; N&W eq. 10.38)
 *
 * Python uses LAPACK (`np.linalg.svd`); this port uses a one-sided Jacobi SVD (Hestenes 1958;
 * Demmel & Veselić 1992), which is accurate to high relative precision in every singular value,
 * so σ_min of the badly scaled Michaelis–Menten Jacobian (κ₂(J) ≈ 1.3·10³) agrees to rounding.
 * The singular vectors may differ in sign from LAPACK's; every product V diag(·) Uᵀ is
 * sign-invariant.
 *
 * Stopping tests, evaluation counts, messages and Step.info keys are those of the Python module
 * (see its docstring). Info keys (snake_case, as exported): step, lambda, gain_ratio,
 * residual_norm, jtj_cond, accepted; Gauss–Newton also alpha, trials ([[alpha, f]]);
 * Levenberg–Marquardt also nu. Python `None` is `null`.
 */
import { registerMethod, param, type MethodDoc } from '../../core/registry';
import type { Matrix, MethodFn, Problem, Result, Step, StepInfo, Vector } from '../../core/types';

// Armijo backtracking constants (Nocedal & Wright, Alg. 3.1).
const ARMIJO_C1 = 1e-4;
const ARMIJO_RHO = 0.5;
const ARMIJO_MAX_BACKTRACKS = 50;

const EPS = 2.220446049250313e-16;
/** Relative rounding noise of f = ½‖r‖² (Higham (2002), §3.1): 100·ε. */
const F_NOISE_REL = 100.0 * EPS;
/** Default Gauss–Newton ftol ≈ 45ε (above the rounding noise of f; see the Python module). */
export const GN_FTOL = 1e-14;
/** Central-difference step ε^(1/3) of `numopt.core.diff.jacobian`. */
const H_CENTRAL = EPS ** (1.0 / 3.0);

const TOL_PARAMS = [
  param.float('gtol', 1e-8, {
    min: 1e-14,
    max: 1e-2,
    log: true,
    help: 'Stop when the gradient ‖Jᵀr‖∞ ≤ gtol.',
    label: 'Gradient tolerance',
    tex: '\\|J^{\\mathsf T}\\mathbf{r}\\|_\\infty \\le',
  }),
  param.float('xtol', 1e-10, {
    min: 1e-15,
    max: 1e-2,
    log: true,
    help: 'Stop when the step ‖h‖₂ ≤ xtol·(‖x‖₂ + xtol).',
    label: 'Step tolerance',
    tex: '\\|\\mathbf{h}\\| / \\|\\mathbf{x}\\| \\le',
  }),
];

// ---------------------------------------------------------------------------------------
// Thin SVD (one-sided Jacobi)
// ---------------------------------------------------------------------------------------

export interface ThinSvd {
  /** m × p, p = min(m, n); columns are the left singular vectors. */
  U: Matrix;
  /** p singular values, descending. */
  s: Vector;
  /** p × n; rows are the right singular vectors. */
  Vt: Matrix;
}

/**
 * Thin SVD of an m × n matrix by one-sided Jacobi rotations on the columns (m ≥ n; for m < n the
 * transpose is factored). Columns are scaled by ‖A‖max first so the column norms neither
 * overflow nor underflow. Returns NaN factors when the matrix is not finite.
 */
export function svdThin(A: Matrix): ThinSvd {
  const m = A.length;
  const n = m ? A[0].length : 0;
  if (m < n) {
    const t = svdThin(A[0].map((_, j) => A.map((row) => row[j])));
    // Aᵀ = U' Σ V'ᵀ  ⇒  A = V' Σ U'ᵀ.
    return {
      U: Array.from({ length: m }, (_, i) => t.Vt.map((row) => row[i])),
      s: t.s,
      Vt: t.U[0].map((_, k) => t.U.map((row) => row[k])),
    };
  }
  const p = n;
  let amax = 0;
  for (const row of A) for (const v of row) amax = Math.max(amax, Math.abs(v));
  if (!Number.isFinite(amax)) {
    return {
      U: Array.from({ length: m }, () => new Array<number>(p).fill(NaN)),
      s: new Array<number>(p).fill(NaN),
      Vt: Array.from({ length: p }, () => new Array<number>(n).fill(NaN)),
    };
  }
  const sc = amax > 0 ? amax : 1;
  // Work column-major: W[j] is column j of A / sc; V[j] is column j of V.
  const W: number[][] = Array.from({ length: n }, (_, j) => A.map((row) => row[j] / sc));
  const V: number[][] = Array.from({ length: n }, (_, j) =>
    Array.from({ length: n }, (_, i) => (i === j ? 1 : 0)),
  );
  for (let sweep = 0; sweep < 60; sweep++) {
    let rotated = false;
    for (let i = 0; i < n - 1; i++)
      for (let j = i + 1; j < n; j++) {
        const a = W[i],
          b = W[j];
        let alpha = 0,
          beta = 0,
          gamma = 0;
        for (let r = 0; r < m; r++) {
          alpha += a[r] * a[r];
          beta += b[r] * b[r];
          gamma += a[r] * b[r];
        }
        if (gamma === 0 || Math.abs(gamma) <= EPS * Math.sqrt(alpha * beta)) continue;
        rotated = true;
        const zeta = (beta - alpha) / (2 * gamma);
        const t = Math.sign(zeta || 1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = c * t;
        for (let r = 0; r < m; r++) {
          const ar = a[r],
            br = b[r];
          a[r] = c * ar - s * br;
          b[r] = s * ar + c * br;
        }
        const vi = V[i],
          vj = V[j];
        for (let r = 0; r < n; r++) {
          const ar = vi[r],
            br = vj[r];
          vi[r] = c * ar - s * br;
          vj[r] = s * ar + c * br;
        }
      }
    if (!rotated) break;
  }
  const norms = W.map((col) => {
    let ss = 0;
    for (const v of col) ss += v * v;
    return Math.sqrt(ss);
  });
  const order = norms.map((_, j) => j).sort((a, b) => norms[b] - norms[a]);
  const s = order.map((j) => norms[j] * sc);
  const U: Matrix = Array.from({ length: m }, (_, r) =>
    order.map((j) => (norms[j] > 0 ? W[j][r] / norms[j] : 0)),
  );
  const Vt: Matrix = order.map((j) => V[j].slice());
  return { U, s, Vt };
}

// ---------------------------------------------------------------------------------------
// Shared helpers (mirrors of the Python module's private functions)
// ---------------------------------------------------------------------------------------

function dotv(a: readonly number[], b: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

const finite = (a: readonly number[]) => a.every((v) => Number.isFinite(v));
const maxAbs = (a: readonly number[]) => a.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
const addv = (a: readonly number[], b: readonly number[]) => a.map((v, i) => v + b[i]);
const scalev = (alpha: number, a: readonly number[]) => a.map((v) => alpha * v);

/** Uᵀ v (p,). */
function utMul(U: Matrix, v: readonly number[]): Vector {
  const p = U[0]?.length ?? 0;
  const out = new Array<number>(p).fill(0);
  for (let k = 0; k < p; k++) {
    let acc = 0;
    for (let i = 0; i < U.length; i++) acc += U[i][k] * v[i];
    out[k] = acc;
  }
  return out;
}

/** −(Vᵀ)ᵀ w = −V w (n,). */
function negVMul(Vt: Matrix, w: readonly number[], n: number): Vector {
  const out = new Array<number>(n).fill(0);
  for (let j = 0; j < n; j++) {
    let acc = 0;
    for (let k = 0; k < Vt.length; k++) acc += Vt[k][j] * w[k];
    out[j] = -acc;
  }
  return out;
}

/** J v (m,). */
function jMul(J: Matrix, v: readonly number[]): Vector {
  return J.map((row) => dotv(row, v));
}

/** ‖v‖₂ as s·‖v/s‖₂ with s = ‖v‖∞ (Higham (2002), §27.5): no overflow, no false underflow. */
export function norm2(v: readonly number[]): number {
  const scale = v.length ? maxAbs(v) : 0;
  if (scale === 0 || !Number.isFinite(scale)) return scale;
  const w = v.map((x) => x / scale);
  return scale * Math.sqrt(dotv(w, w));
}

const halfSq = (r: readonly number[]) => 0.5 * dotv(r, r);

/**
 * Python's `format(x, '.3g')`: 3 significant digits, exponent form when exp < −4 or ≥ 3,
 * trailing zeros removed, `inf` / `nan`.
 */
export function pyG(x: number, digits = 3): string {
  if (Number.isNaN(x)) return 'nan';
  if (!Number.isFinite(x)) return x > 0 ? 'inf' : '-inf';
  if (x === 0) return Object.is(x, -0) ? '-0' : '0';
  const [mant, expS] = x.toExponential(digits - 1).split('e');
  const e = Number(expS);
  const trim = (s: string) => (s.includes('.') ? s.replace(/\.?0+$/, '') : s);
  if (e < -4 || e >= digits) {
    return `${trim(mant)}e${e < 0 ? '-' : '+'}${String(Math.abs(e)).padStart(2, '0')}`;
  }
  return trim(x.toFixed(Math.max(0, digits - 1 - e)));
}

/** r, J, f, g and the thin SVD of J at one point (`_Linearization`). */
class Linearization {
  readonly r: Vector;
  readonly J: Matrix;
  readonly f: number;
  readonly g: Vector;
  readonly U: Matrix;
  readonly s: Vector;
  readonly Vt: Matrix;
  readonly m: number;
  readonly n: number;

  constructor(r: Vector, J: Matrix) {
    this.r = r;
    this.J = J;
    this.m = J.length;
    this.n = J[0]?.length ?? 0;
    const g = new Array<number>(this.n).fill(0);
    for (let j = 0; j < this.n; j++) {
      let acc = 0;
      for (let i = 0; i < this.m; i++) acc += J[i][j] * r[i];
      g[j] = acc;
    }
    this.g = g;
    const { U, s, Vt } = svdThin(J);
    this.U = U;
    this.s = s;
    this.Vt = Vt;
    this.f = halfSq(r);
  }

  /** NumPy's `matrix_rank` tolerance max(m, n)·ε·σ_max. */
  get rankTol(): number {
    return this.s.length ? Math.max(this.m, this.n) * EPS * this.s[0] : 0.0;
  }

  /** Numerical rank < n: σ_min ≤ max(m, n)·ε·σ_max (or fewer than n singular values). */
  get rankDeficient(): boolean {
    if (this.s.length < this.n || this.s[0] === 0.0) return true;
    return this.s[this.s.length - 1] <= this.rankTol;
  }

  /** κ₂(JᵀJ) = (σ_max/σ_min)², or ∞ when J is numerically rank-deficient. */
  get jtjCond(): number {
    if (this.rankDeficient) return Infinity;
    return (this.s[0] / this.s[this.s.length - 1]) ** 2;
  }

  /** Minimum-norm Gauss–Newton step p = −J⁺r, with σᵢ ≤ rankTol treated as 0. */
  gaussNewtonStep(): Vector {
    const tol = this.rankTol;
    const invS = this.s.map((v) => (v > tol ? 1.0 / v : 0.0));
    const utr = utMul(this.U, this.r);
    return negVMul(
      this.Vt,
      invS.map((v, k) => v * utr[k]),
      this.n,
    );
  }

  /** ‖Jp‖²/‖r‖² for the Gauss–Newton step p (scale-free; cos²∠(r, range J)). */
  gaussNewtonRatio(): number {
    const rMax = this.r.length ? maxAbs(this.r) : 0.0;
    if (rMax === 0.0) return 0.0;
    const rHat = this.r.map((v) => v / rMax);
    const tol = this.rankTol;
    const uHat = utMul(this.U, rHat).filter((_, k) => this.s[k] > tol);
    return dotv(uHat, uHat) / dotv(rHat, rHat);
  }
}

function breakdown(lin: Linearization, where: string): string {
  if (!Number.isFinite(lin.f)) return `f = ½‖r‖² overflows ${where} (‖r‖∞ = ${pyG(maxAbs(lin.r))})`;
  if (!finite(lin.g)) return `the gradient ∇f = Jᵀr overflows ${where}`;
  if (!finite(lin.s)) return `the SVD of the Jacobian failed ${where} (non-finite singular values)`;
  return '';
}

/** Problem-like input: a residual with an optional Jacobian. */
type LsProblem = Pick<Problem<Vector>, 'id' | 'dim' | 'residual' | 'jac' | 'x0'>;

interface Counted<T> {
  (x: Vector): T;
  n: number;
}

function counted<T>(fn: (x: Vector) => T): Counted<T> {
  const c = ((x: Vector) => {
    c.n++;
    return fn(x);
  }) as Counted<T>;
  c.n = 0;
  return c;
}

/** Return (x0, counted residual, counted Jacobian) for a problem or a bare r(x) (`_resolve`). */
function resolve(
  problem: LsProblem | ((x: Vector) => Vector),
  x0: unknown,
): [Vector, Counted<Vector>, Counted<Matrix>] {
  let x: Vector;
  let residual: (x: Vector) => Vector;
  let jac: ((x: Vector) => Matrix) | undefined;
  if (typeof problem === 'function') {
    if (x0 === undefined || x0 === null)
      throw new Error('a starting point x0 is required for a bare residual function');
    x = (Array.isArray(x0) ? x0 : [x0]).map(Number);
    residual = problem;
  } else {
    if (!problem.residual)
      throw new Error(
        `${problem.id}: least-squares methods need a residual r(x) (Problem.residual)`,
      );
    const start = x0 ?? problem.x0;
    if (start === undefined || start === null)
      throw new Error(`${problem.id}: no starting point given and the problem has no default x0`);
    x = (Array.isArray(start) ? start : [start]).map(Number);
    if (problem.dim && x.length !== problem.dim)
      throw new Error(`${problem.id}: x0 has ${x.length} entries, expected ${problem.dim}`);
    residual = problem.residual;
    jac = problem.jac;
  }
  const res = counted((z: Vector) => residual(z).map(Number));
  const jacFn = jac
    ? counted((z: Vector) => jac(z).map((row) => row.slice()))
    : counted((z: Vector) => {
        // Central differences (numopt.core.diff.jacobian); its residual calls count in n_fev.
        const f0 = res(z);
        const J: Matrix = f0.map(() => new Array<number>(z.length).fill(0));
        for (let i = 0; i < z.length; i++) {
          const h = H_CENTRAL * Math.max(1.0, Math.abs(z[i]));
          const zp = z.slice(),
            zm = z.slice();
          zp[i] = z[i] + h;
          zm[i] = z[i] - h;
          const fp = res(zp),
            fm = res(zm);
          for (let r = 0; r < f0.length; r++) J[r][i] = (fp[r] - fm[r]) / (2.0 * h);
        }
        return J;
      });
  return [x, res, jacFn];
}

function info(
  lin: Linearization,
  o: { step: Vector | null; lam: number; gain: number | null; accepted: boolean | null },
  extra: StepInfo,
): StepInfo {
  return {
    step: o.step === null ? null : o.step.slice(),
    lambda: o.lam,
    gain_ratio: o.gain,
    residual_norm: Math.sqrt(2.0 * lin.f),
    jtj_cond: lin.jtjCond,
    accepted: o.accepted,
    ...extra,
  };
}

const gtolMet = (lin: Linearization, gtol: number) => maxAbs(lin.g) <= gtol;
const xtolMet = (h: Vector, x: Vector, xtol: number) => norm2(h) <= xtol * (norm2(x) + xtol);
const msgGtol = (lin: Linearization) => `gradient ‖Jᵀr‖∞ = ${pyG(maxAbs(lin.g))} ≤ gtol`;
const msgXtol = (h: Vector, lin: Linearization) =>
  `step ‖h‖ = ${pyG(norm2(h))} ≤ xtol·(‖x‖ + xtol) (‖Jᵀr‖∞ = ${pyG(maxAbs(lin.g))})`;
const MSG_ZERO_F = 'f = ½‖r‖² = 0 in floating point: x attains the global minimum value 0';

/** Evaluate r and J at x0; return (linearization if J is usable, r, failure message). */
function start(
  x: Vector,
  res: Counted<Vector>,
  jac: Counted<Matrix>,
): [Linearization | null, Vector, string] {
  const r = res(x);
  if (!finite(r)) return [null, r, 'residual is not finite at x0'];
  const J = jac(x);
  if (
    J.length !== r.length ||
    J.some((row) => row.length !== x.length) ||
    !J.every((row) => finite(row))
  )
    return [null, r, 'Jacobian is not finite (or has the wrong shape) at x0'];
  const lin = new Linearization(r, J);
  return [lin, r, breakdown(lin, 'at x0')];
}

function failedStart(
  method: string,
  x: Vector,
  r: Vector,
  lin: Linearization | null,
  message: string,
  res: Counted<Vector>,
  jac: Counted<Matrix>,
  lam: number | null,
  extra: StepInfo,
): Result {
  const f = finite(r) ? halfSq(r) : NaN;
  const stepInfo: StepInfo = {
    step: null,
    lambda: lam,
    gain_ratio: null,
    residual_norm: Number.isNaN(f) ? null : Math.sqrt(2.0 * f),
    jtj_cond: lin !== null && finite(lin.s) ? lin.jtjCond : null,
    accepted: null,
    ...extra,
  };
  const step: Step = { k: 0, x: x.slice(), fun: f, gradNorm: null, stepSize: null, info: stepInfo };
  return {
    method,
    x: x.slice(),
    fun: f,
    converged: false,
    message,
    nIter: 0,
    nFev: res.n,
    nGev: jac.n,
    nHev: 0,
    trace: [step],
    extra: {},
  };
}

// ---------------------------------------------------------------------------------------
// Gauss–Newton
// ---------------------------------------------------------------------------------------

type Options = Record<string, unknown>;

export const gaussNewton: MethodFn<LsProblem | ((x: Vector) => Vector)> = (problem, options) => {
  const o = options as Options;
  const gtol = (o.gtol as number | undefined) ?? 1e-8;
  const xtol = (o.xtol as number | undefined) ?? 1e-10;
  const ftol = (o.ftol as number | undefined) ?? GN_FTOL;
  const maxIter = (o.max_iter as number | undefined) ?? 100;
  const lineSearch = (o.line_search as string | undefined) ?? 'backtracking';
  if (lineSearch !== 'backtracking' && lineSearch !== 'none')
    throw new Error(`line_search must be 'backtracking' or 'none', got '${lineSearch}'`);
  const name = 'gauss_newton';
  const [xStart, res, jac] = resolve(problem, o.x0);
  let x = xStart;
  const [lin0, r0, why0] = start(x, res, jac);
  if (lin0 === null || why0)
    return failedStart(name, x, r0, lin0, why0, res, jac, 0.0, { alpha: null, trials: [] });
  let lin: Linearization = lin0;

  const trace: Step[] = [
    {
      k: 0,
      x: x.slice(),
      fun: lin.f,
      gradNorm: norm2(lin.g),
      stepSize: null,
      info: info(
        lin,
        { step: null, lam: 0.0, gain: null, accepted: null },
        { alpha: null, trials: [] },
      ),
    },
  ];

  const done = (converged: boolean, message: string, nIter: number): Result => ({
    method: name,
    x: x.slice(),
    fun: lin.f,
    converged,
    message,
    nIter,
    nFev: res.n,
    nGev: jac.n,
    nHev: 0,
    trace,
    extra: {},
  });

  // k = max_iter + 1 only runs the stopping tests at the last iterate (no evaluations).
  for (let k = 1; k <= maxIter + 1; k++) {
    if (gtolMet(lin, gtol)) return done(true, msgGtol(lin), k - 1);
    if (lin.f === 0.0) return done(true, MSG_ZERO_F, k - 1);
    if (lin.rankDeficient) {
      const sl = lin.s;
      const ratio = sl.length && sl[0] > 0.0 ? sl[sl.length - 1] / sl[0] : 0.0;
      return done(
        false,
        `Jacobian is numerically rank-deficient (σ_min/σ_max = ${pyG(ratio)} ≤ ` +
          'max(m, n)·ε); the Gauss–Newton step is not unique',
        k - 1,
      );
    }
    const utr = utMul(lin.U, lin.r);
    const p = negVMul(
      lin.Vt,
      utr.map((v, i) => v / lin.s[i]),
      lin.n,
    );
    if (xtolMet(p, x, xtol)) return done(true, msgXtol(p, lin), k - 1);
    // NOTE (Python): the ftol test is not in N&W; it makes the rounding-limited stall of the
    // Armijo test a documented stop (‖Jp‖²/‖r‖² at the rounding noise of f).
    const reduction = lin.gaussNewtonRatio();
    if (reduction <= ftol)
      return done(
        true,
        `predicted relative reduction ‖Jp‖²/‖r‖² = ${pyG(reduction)} ≤ ftol`,
        k - 1,
      );
    if (k > maxIter) return done(false, `reached max_iter=${maxIter}`, maxIter);

    const slope = dotv(lin.g, p); // ∇fᵀp = −‖Jp‖² in exact arithmetic
    const trials: [number, number][] = [];
    let alpha = 1.0;
    let rNew: Vector;
    let fNew: number;
    if (lineSearch === 'backtracking') {
      if (!(slope < 0.0))
        return done(false, 'Gauss–Newton step is not a descent direction (rounding)', k - 1);
      let found = false;
      rNew = [];
      fNew = Infinity;
      for (let it = 0; it < ARMIJO_MAX_BACKTRACKS + 1; it++) {
        rNew = res(addv(x, scalev(alpha, p)));
        fNew = finite(rNew) ? halfSq(rNew) : Infinity;
        trials.push([alpha, fNew]);
        // NOTE (Python): f_new < f is required too, so steps below the resolution of f do not
        // freeze the iteration.
        if (fNew < lin.f && fNew <= lin.f + ARMIJO_C1 * alpha * slope) {
          found = true;
          break;
        }
        if (alpha * -slope <= EPS * lin.f) {
          return done(
            false,
            'no measurable decrease along the Gauss–Newton step (the predicted ' +
              `reduction is below the rounding level of f); ‖Jᵀr‖∞ = ` +
              `${pyG(maxAbs(lin.g))} > gtol`,
            k - 1,
          );
        }
        alpha *= ARMIJO_RHO;
      }
      if (!found)
        return done(
          false,
          `Armijo backtracking failed after ${ARMIJO_MAX_BACKTRACKS} halvings`,
          k - 1,
        );
    } else {
      rNew = res(addv(x, p));
      fNew = finite(rNew) ? halfSq(rNew) : Infinity;
      trials.push([alpha, fNew]);
      if (!Number.isFinite(fNew))
        return done(false, 'residual is not finite at the Gauss–Newton step', k - 1);
    }

    const h = scalev(alpha, p);
    // Predicted reduction L(0) − L(h) = −gᵀh − ½‖Jh‖².
    const Jh = jMul(lin.J, h);
    const predicted = -dotv(lin.g, h) - 0.5 * dotv(Jh, Jh);
    const gain = predicted > 0.0 ? (lin.f - fNew) / predicted : null;
    const xNew = addv(x, h);
    const JNew = jac(xNew);
    if (!JNew.every((row) => finite(row)))
      return done(false, 'Jacobian is not finite at the new iterate', k - 1);
    const linNew = new Linearization(rNew, JNew);
    const why = breakdown(linNew, 'at the new iterate');
    if (why) return done(false, why, k - 1);
    x = xNew;
    lin = linNew;
    trace.push({
      k,
      x: x.slice(),
      fun: lin.f,
      gradNorm: norm2(lin.g),
      stepSize: norm2(h),
      info: info(lin, { step: p, lam: 0.0, gain, accepted: true }, { alpha, trials }),
    });
  }
  throw new Error('unreachable: the loop returns at k = max_iter + 1');
};

// ---------------------------------------------------------------------------------------
// Levenberg–Marquardt
// ---------------------------------------------------------------------------------------

export const levenbergMarquardt: MethodFn<LsProblem | ((x: Vector) => Vector)> = (
  problem,
  options,
) => {
  const o = options as Options;
  const gtol = (o.gtol as number | undefined) ?? 1e-8;
  const xtol = (o.xtol as number | undefined) ?? 1e-10;
  const maxIter = (o.max_iter as number | undefined) ?? 200;
  const tau = (o.tau as number | undefined) ?? 1e-3;
  if (!(tau > 0.0)) throw new Error(`tau must be positive, got ${tau}`);
  const name = 'levenberg_marquardt';
  const [xStart, res, jac] = resolve(problem, o.x0);
  let x = xStart;
  const [lin0, r0, why0] = start(x, res, jac);
  let why = why0;
  let mu = NaN;
  if (lin0 !== null) {
    // NOTE (Python): Levenberg's μI damping (MNT Alg. 3.16), not Marquardt's scaled μ·DᵀD.
    // μ₀ = τ·max diag(JᵀJ); the column sums run row by row like NumPy's axis-0 reduction.
    const J = lin0.J;
    let colMax = -Infinity;
    for (let j = 0; j < lin0.n; j++) {
      let acc = 0;
      for (let i = 0; i < lin0.m; i++) acc += J[i][j] * J[i][j];
      colMax = Number.isNaN(acc) ? NaN : Math.max(colMax, acc);
    }
    mu = tau * colMax;
    if (!why && !Number.isFinite(mu))
      why = 'the initial damping μ₀ = τ·max diag(JᵀJ) overflows at x0';
  }
  if (lin0 === null || why) {
    const lam0 = lin0 !== null ? mu : null;
    return failedStart(name, x, r0, lin0, why, res, jac, lam0, { nu: 2.0 });
  }
  let lin: Linearization = lin0;
  let nu = 2.0;
  let blocked = false; // a trial since the last accepted step gave a non-finite residual
  let lastRejected = false; // the previous trial was rejected
  const trace: Step[] = [
    {
      k: 0,
      x: x.slice(),
      fun: lin.f,
      gradNorm: norm2(lin.g),
      stepSize: null,
      info: info(lin, { step: null, lam: mu, gain: null, accepted: null }, { nu }),
    },
  ];

  const done = (converged: boolean, message: string, nIter: number): Result => ({
    method: name,
    x: x.slice(),
    fun: lin.f,
    converged,
    message,
    nIter,
    nFev: res.n,
    nGev: jac.n,
    nHev: 0,
    trace,
    extra: {},
  });

  for (let k = 1; k <= maxIter + 1; k++) {
    if (gtolMet(lin, gtol)) return done(true, msgGtol(lin), k - 1);
    if (lin.f === 0.0) return done(true, MSG_ZERO_F, k - 1);
    // h = −V diag(σ/(σ² + μ)) Uᵀ r solves (JᵀJ + μI) h = −Jᵀr (N&W eq. 10.38).
    const coef = lin.s.map((sv) => {
      const denom = sv * sv + mu;
      return denom > 0.0 ? sv / denom : 0.0;
    });
    const utr = utMul(lin.U, lin.r);
    const h = negVMul(
      lin.Vt,
      coef.map((c, i) => c * utr[i]),
      lin.n,
    );
    if (xtolMet(h, x, xtol)) {
      if (blocked)
        return done(
          false,
          'trial steps reached points where the residual is not finite; the damped ' +
            'step shrank below xtol without meeting gtol',
          k - 1,
        );
      // NOTE (Python): the step test certifies convergence only when the undamped
      // Gauss–Newton step is small too, or its predicted decrease is at the rounding noise of f.
      const pGn = lin.gaussNewtonStep();
      const gnRatio = lin.gaussNewtonRatio();
      if (xtolMet(scalev(0.5, pGn), x, xtol) || gnRatio <= F_NOISE_REL)
        return done(true, msgXtol(h, lin), k - 1);
      if (lastRejected)
        return done(
          false,
          `the damped step ‖h‖ = ${pyG(norm2(h))} is below xtol only ` +
            `because the damping μ = ${pyG(mu)} is large (σ_min² = ` +
            `${pyG(lin.s[lin.s.length - 1] ** 2)}); the Gauss–Newton model still predicts a ` +
            `relative decrease ‖Jp‖²/‖r‖² = ${pyG(gnRatio)} with ‖p‖ = ` +
            `${pyG(norm2(pGn))}, and ‖Jᵀr‖∞ = ${pyG(maxAbs(lin.g))} > gtol`,
          k - 1,
        );
    }

    if (k > maxIter) return done(false, `reached max_iter=${maxIter}`, maxIter);

    const muUsed = mu;
    const rNew = res(addv(x, h));
    const fNew = finite(rNew) ? halfSq(rNew) : NaN;
    const predicted =
      0.5 *
      dotv(
        h,
        h.map((hi, i) => mu * hi - lin.g[i]),
      ); // L(0) − L(h), MNT eq. 3.14
    const gain = predicted > 0.0 && Number.isFinite(fNew) ? (lin.f - fNew) / predicted : null;
    const accepted = gain !== null && gain > 0.0;
    blocked = (blocked || !Number.isFinite(fNew)) && !accepted;
    lastRejected = !accepted;
    if (accepted) {
      const xNew = addv(x, h);
      const JNew = jac(xNew);
      if (!JNew.every((row) => finite(row)))
        return done(false, 'Jacobian is not finite at the new iterate', k - 1);
      const linNew = new Linearization(rNew, JNew);
      const w = breakdown(linNew, 'at the new iterate');
      if (w) return done(false, w, k - 1);
      x = xNew;
      lin = linNew;
      mu *= Math.max(1.0 / 3.0, 1.0 - (2.0 * gain - 1.0) ** 3);
      nu = 2.0;
    } else {
      mu *= nu;
      nu *= 2.0;
    }
    trace.push({
      k,
      x: x.slice(),
      fun: lin.f,
      gradNorm: norm2(lin.g),
      stepSize: accepted ? norm2(h) : 0.0,
      info: info(lin, { step: h, lam: muUsed, gain, accepted }, { nu }),
    });
    if (!Number.isFinite(mu) || !Number.isFinite(nu))
      return done(false, 'damping parameter μ overflowed (no acceptable step found)', k);
  }
  throw new Error('unreachable: the loop returns at k = max_iter + 1');
};

// ---------------------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------------------

// MethodCard quantities are read from Step k: the iterate 𝐱ₖ and the step that produced it
// (from 𝐱ₖ₋₁), hence the k − 1 subscripts.
const GN_DOC: MethodDoc = {
  rule:
    '\\begin{aligned}\\mathbf{p}_k &= \\arg\\min_{\\mathbf{p}} \\|\\mathbf{r}_k + J_k\\mathbf{p}\\|_2 = -J_k^{+}\\mathbf{r}_k\\\\' +
    '\\mathbf{x}_{k+1} &= \\mathbf{x}_k + \\alpha_k\\mathbf{p}_k\\end{aligned}',
  intuition:
    'Replace every residual by its tangent plane and jump to the least-squares solution of ' +
    'that linear model. Near a zero-residual fit JᵀJ is the whole Hessian, so the jump is ' +
    'Newton’s step; where the residuals stay large, the neglected Σ rᵢ∇²rᵢ slows it to linear.',
  order: 'quadratic if 𝐫⋆ = 𝟎',
  pros: [
    'Needs only J — no second derivatives',
    'Quadratic on zero-residual problems',
    'Each step is a linear least-squares solve (SVD, never the normal equations)',
  ],
  cons: [
    'Undefined when J loses rank',
    'Linear or divergent when the residuals at the solution are large',
    'Full steps can overshoot far from the solution (needs a line search)',
  ],
  quantities: [
    { tex: '\\|\\mathbf{x}_k - \\mathbf{x}_{k-1}\\|', key: 'stepSize' },
    { tex: '\\alpha_{k-1}', key: 'info.alpha' },
    { tex: '\\varrho_{k-1}', key: 'info.gain_ratio' },
    { tex: '\\|\\mathbf{r}(\\mathbf{x}_k)\\|', key: 'info.residual_norm' },
    { tex: '\\kappa_2(J_k^{\\mathsf T}J_k)', key: 'info.jtj_cond' },
  ],
};

const LM_DOC: MethodDoc = {
  rule:
    '\\begin{aligned}(J_k^{\\mathsf T}J_k + \\mu_k I)\\,\\mathbf{h}_k &= -J_k^{\\mathsf T}\\mathbf{r}_k\\\\' +
    '\\varrho_k &= \\frac{f(\\mathbf{x}_k) - f(\\mathbf{x}_k + \\mathbf{h}_k)}{L(\\mathbf{0}) - L(\\mathbf{h}_k)}\\end{aligned}',
  intuition:
    'Gauss–Newton with a brake: the damping μ blends the Gauss–Newton step (μ → 0) with a ' +
    'short steepest-descent step (μ → ∞). The gain ratio ϱ — actual over predicted decrease — ' +
    'decides: a good model lowers μ, a rejected step raises it, so μ acts as an implicit trust radius.',
  order: 'superlinear if 𝐫⋆ = 𝟎',
  pros: [
    'Robust far from the solution (every accepted step decreases f)',
    'Well defined even when J loses rank (JᵀJ + μI is positive definite)',
    'No line search: one residual per trial',
  ],
  cons: [
    'Not scale invariant (μI damps all parameters alike)',
    'Rejected trials cost iterations while μ adapts',
    'Linear on large-residual problems, like Gauss–Newton',
  ],
  quantities: [
    { tex: '\\|\\mathbf{x}_k - \\mathbf{x}_{k-1}\\|', key: 'stepSize' },
    { tex: '\\mu_{k-1}', key: 'info.lambda' },
    { tex: '\\varrho_{k-1}', key: 'info.gain_ratio' },
    { tex: '\\nu_k', key: 'info.nu' },
    { tex: '\\|\\mathbf{r}(\\mathbf{x}_k)\\|', key: 'info.residual_norm' },
    { tex: '\\kappa_2(J_k^{\\mathsf T}J_k)', key: 'info.jtj_cond' },
  ],
};

registerMethod(
  {
    id: 'gauss_newton',
    family: 'least_squares',
    name: 'Gauss–Newton',
    params: [
      ...TOL_PARAMS,
      param.float('ftol', GN_FTOL, {
        min: 1e-18,
        max: 1e-4,
        log: true,
        help: 'Stop when the predicted relative reduction ‖Jp‖²/‖r‖² of the full step ≤ ftol.',
        label: 'Reduction tolerance',
        tex: '\\|J\\mathbf{p}\\|^2/\\|\\mathbf{r}\\|^2 \\le',
      }),
      param.int('max_iter', 100, {
        min: 1,
        max: 10_000,
        help: 'Iteration limit.',
        label: 'Iteration budget',
      }),
      param.choice('line_search', 'backtracking', ['backtracking', 'none'], {
        help: 'Armijo backtracking along the Gauss–Newton step, or the full step.',
        label: 'Line search',
      }),
    ],
    needs: ['residual', 'jac'],
    order: 'quadratic for zero residual, linear otherwise',
    summary: 'Linearize the residuals and jump to the least-squares solution of the linear model.',
    references: [
      'Nocedal & Wright (2006), §10.3, eq. 10.23',
      'Björck (1996), Numerical Methods for Least Squares Problems, §9.2',
    ],
  },
  gaussNewton,
  GN_DOC,
);

registerMethod(
  {
    id: 'levenberg_marquardt',
    family: 'least_squares',
    name: 'Levenberg–Marquardt',
    params: [
      ...TOL_PARAMS,
      param.int('max_iter', 200, {
        min: 1,
        max: 10_000,
        help: 'Iteration limit.',
        label: 'Iteration budget',
      }),
      param.float('tau', 1e-3, {
        min: 1e-8,
        max: 1e2,
        log: true,
        help: 'Initial damping μ₀ = tau·max diag(JᵀJ); small trusts Gauss–Newton at once.',
        label: 'Initial damping',
        tex: '\\tau',
      }),
    ],
    needs: ['residual', 'jac'],
    order:
      'superlinear for zero residual (not quadratic: μ falls at most 3× per step), ' +
      'linear otherwise',
    summary:
      'Gauss–Newton with an adaptive damping term: large μ gives short gradient-like ' +
      'steps, small μ gives Gauss–Newton steps.',
    references: [
      'Madsen, Nielsen & Tingleff (2004), Methods for Non-Linear Least Squares Problems, Alg. 3.16',
      "Nielsen (1999), Damping parameter in Marquardt's method, IMM-REP-1999-05",
      'Nocedal & Wright (2006), §10.3',
      'Moré (1978), The Levenberg–Marquardt algorithm: implementation and theory',
    ],
  },
  levenbergMarquardt,
  LM_DOC,
);
