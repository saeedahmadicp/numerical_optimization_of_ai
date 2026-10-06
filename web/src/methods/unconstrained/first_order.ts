/**
 * First-order methods for unconstrained minimization — TS port of
 * `numopt.unconstrained.first_order` (src/numopt/unconstrained/first_order.py).
 *
 * Every method uses only f and ∇f (coordinate descent also uses the diagonal of ∇²f):
 *
 *   - line-search methods: gradient descent (fixed, Armijo backtracking, strong Wolfe or exact
 *     quadratic step) and Barzilai–Borwein (two-point step length, GLL nonmonotone search);
 *   - momentum methods: Polyak's heavy ball and Nesterov's accelerated gradient;
 *   - adaptive methods (exact full gradients): AdaGrad, RMSprop, AdaDelta, Adam, AdamW, AdaMax,
 *     NAdam, AMSGrad;
 *   - cyclic coordinate descent with a 1-D Newton step.
 *
 * Conventions (identical to Python):
 *
 *   - Trace: step k = 0 holds x₀; step k ≥ 1 holds x_k = x_{k−1} + α_k p_k. `stepSize` = α_k
 *     (null at k = 0), `info.alpha` = α_k, `info.direction` = p_k, `info.grad` = ∇f(x_k).
 *     `nIter == trace[trace.length − 1].k`.
 *   - Stopping test (converged): ‖∇f(x_k)‖₂ ≤ gtol, checked at x₀ and after every iteration.
 *   - Failure (converged = false): max_iter; a non-finite f or ∇f; a rise f(x_k) − f(x₀) >
 *     10¹²·max(1, |f(x₀)|); a failed line search; a stall (the accepted step leaves x unchanged);
 *     an overflow of the slope ∇f(x)ᵀp.
 *   - Counts: every call of f, ∇f, ∇²f is counted like Python's `Counted` (central differences
 *     of a missing gradient count 2n f calls in nFev; of a missing Hessian 2n ∇f calls in nGev).
 *
 * Step.info keys are the Python ones (snake_case), listed in the Python module docstring:
 * grad, direction, alpha, and per method trials / alpha0 / pHp (gradient_descent), alpha_bb / bb1 /
 * bb2 / reset / f_ref / trials (barzilai_borwein), velocity (momentum), velocity / lookahead /
 * grad_lookahead (nesterov), v / lr_eff (adagrad, rmsprop), v / u / lr_eff (adadelta), m / v / m_hat
 * / v_hat / lr_eff (+ decay for adamw), m / u / lr_eff (adamax), m / v / m_bar / v_hat / lr_eff
 * (nadam), m / v / v_max / lr_eff (amsgrad), coordinate / sweep / curvature / newton / trials
 * (coordinate_descent). Python `None` is `null`; a NaN stays NaN (the exporter writes JSON null).
 *
 * Floating-point operations keep the Python (NumPy, elementwise) order, so traces agree with the
 * Python fixtures to rounding. Invalid input throws `FirstOrderInputError` (Python `ValueError`)
 * with the Python message; numerical breakdown never throws.
 */
import { registerMethod, param, type MethodDoc } from '../../core/registry';
import type {
  Matrix,
  MethodFn,
  Params,
  ParamSpec,
  Problem,
  Result,
  RunOptions,
  Step,
  StepInfo,
  Vector,
} from '../../core/types';
import {
  C1_DEFAULT,
  LineSearchZeroDivision,
  formatG,
  formatRepr,
  search,
  type LineSearchResult,
  type SearchOptions,
} from '../line_search/methods';

// ---------------------------------------------------------------------------------------
// Constants (same values as the Python module)
// ---------------------------------------------------------------------------------------

/** A rise f(x_k) − f(x₀) above F_DIVERGE·max(1, |f(x₀)|) counts as divergence. */
export const F_DIVERGE = 1e12;
/** Largest finite float; per-coordinate step sizes (`lr_eff`) that overflow are capped here. */
export const FLOAT_MAX = Number.MAX_VALUE;
/** Gradient descent, strong Wolfe rule: α_max = ALPHA_MAX_RATIO·α₀. */
export const ALPHA_MAX_RATIO = 2.0 ** 20;
/** Barzilai–Borwein: step lengths outside [BB_MIN, BB_MAX] are reset. */
export const BB_MIN = 1e-10;
export const BB_MAX = 1e10;
/** Nonmonotone memory M, sufficient-decrease γ and safeguard [σ₁, σ₂] (Raydan 1997, §3). */
export const BB_MEMORY = 10;
export const BB_GAMMA = 1e-4;
export const BB_SIGMA1 = 0.1;
export const BB_SIGMA2 = 0.5;
/** Maximum trial steps in one Barzilai–Borwein nonmonotone search. */
export const BB_MAX_TRIALS = 50;
/** Smallest accepted gtol (‖g‖ = √(gᵀg) underflows to 0 below ~1e-154). */
export const GTOL_MIN = 1e-150;

export const STEP_RULES = ['fixed', 'backtracking', 'strong_wolfe', 'exact_quadratic'] as const;
export const BB_VARIANTS = ['bb1', 'bb2'] as const;

/** Python `ValueError`: invalid parameters or start point. */
export class FirstOrderInputError extends Error {
  override name = 'ValueError';
}

/** A problem from the library or a bare `f(x) -> number` (then `x0` is required). */
export type FirstOrderProblem = Problem<Vector> | ((x: Vector) => number);

// ---------------------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------------------

const P_GTOL = param.float('gtol', 1e-6, {
  min: 1e-14,
  max: 1e-2,
  log: true,
  help: 'Stop when ‖∇f(x)‖₂ ≤ gtol.',
  label: 'Gradient tolerance',
  tex: '\\|\\nabla f\\|_2 \\le',
});

const pMaxIter = (def: number): ParamSpec =>
  param.int('max_iter', def, {
    min: 1,
    max: 100_000,
    help: 'Iteration limit.',
    label: 'Max iterations',
  });

const pLr = (def: number, help = 'Learning rate (step length) α.'): ParamSpec =>
  param.float('lr', def, {
    min: 1e-6,
    max: 2.0,
    log: true,
    help,
    label: 'Learning rate',
    tex: '\\alpha',
  });

const pBeta = (name: string, def: number, help: string, label: string, tex: string): ParamSpec =>
  param.float(name, def, { min: 0.0, max: 0.9999, help, label, tex });

const pEps = (def: number): ParamSpec =>
  param.float('eps', def, {
    min: 1e-12,
    max: 1e-2,
    log: true,
    help: 'ε in the denominator (> 0).',
    label: 'Epsilon',
    tex: '\\varepsilon',
  });

const ADAM_BETAS: ParamSpec[] = [
  pBeta('beta1', 0.9, 'Decay β₁ of the first moment (momentum).', 'First-moment decay', '\\beta_1'),
  pBeta('beta2', 0.999, 'Decay β₂ of the second moment.', 'Second-moment decay', '\\beta_2'),
];

// ---------------------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------------------

const g3 = (v: number) => formatG(v, 3);

/** Python `repr` of a float parameter (`0.0`, `-1e-08`). */
function reprFloat(v: unknown): string {
  return typeof v === 'number' ? formatRepr(v) : reprValue(v);
}

/** Python `repr` of an int-or-float parameter (`0`, `2.5`) or a string (`'bb3'`). */
function reprValue(v: unknown): string {
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : formatRepr(v);
  if (typeof v === 'string') return `'${v}'`;
  if (typeof v === 'boolean') return v ? 'True' : 'False';
  return String(v);
}

function toVector(v: unknown): Vector {
  if (typeof v === 'number') return [v];
  if (Array.isArray(v)) return (v as unknown[]).flat(Infinity).map(Number);
  if (ArrayBuffer.isView(v)) return Array.from(v as unknown as ArrayLike<number>, Number);
  return [Number(v)];
}

function toMatrix(v: unknown, n: number): Matrix {
  if (typeof v === 'number') return [[v]];
  const rows = v as unknown[];
  if (n === 1 && rows.length === 1 && typeof rows[0] === 'number') return [[rows[0]]];
  return rows.map((row) => toVector(row));
}

/** `float(a @ b)` (a plain left-to-right sum, as NumPy's dot for the small n used here). */
function dot(a: readonly number[], b: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** `_norm`: ‖v‖₂, Infinity when ‖v‖² overflows. */
function norm(v: readonly number[]): number {
  return Math.sqrt(dot(v, v));
}

function allFinite(v: readonly number[]): boolean {
  for (const x of v) if (!Number.isFinite(x)) return false;
  return true;
}

/** `np.array_equal(a, b)` (NaN never equals NaN). */
function arrayEqual(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/** x + alpha·p, elementwise (`run.x + alpha * p`). */
function xPlus(x: readonly number[], alpha: number, p: readonly number[]): Vector {
  const out = new Array<number>(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] + alpha * p[i];
  return out;
}

/** `_safe_div`: num/den elementwise, 0 where den == 0. */
function safeDiv(num: readonly number[], den: readonly number[]): Vector {
  return num.map((v, i) => (den[i] !== 0.0 ? v / den[i] : 0.0));
}

/** `_step_sizes`: scale/den elementwise for `lr_eff`, 0 where den == 0, capped at FLOAT_MAX. */
function stepSizes(scale: number, den: readonly number[]): Vector {
  return safeDiv(
    den.map(() => 1.0),
    den,
  ).map((q) => Math.min(scale * q, FLOAT_MAX));
}

/** ε^{1/3} for float64, the central-difference step factor of `numopt.core.diff`. */
const H_CENTRAL = 6.055454452393343e-6;

/** `numopt.core.diff.gradient`: central differences, 2n calls of f. */
function fdGradient(f: (x: Vector) => number, x: Vector): Vector {
  const g = new Array<number>(x.length);
  for (let i = 0; i < x.length; i++) {
    const h = H_CENTRAL * Math.max(1.0, Math.abs(x[i]));
    const xp = x.map((v, j) => v + (j === i ? h : 0.0));
    const xm = x.map((v, j) => v - (j === i ? h : 0.0));
    g[i] = (f(xp) - f(xm)) / (2.0 * h);
  }
  return g;
}

/** `numopt.core.diff.hessian`: symmetrized central differences of ∇f, 2n gradient calls. */
function fdHessian(grad: (x: Vector) => Vector, x: Vector): Matrix {
  const n = x.length;
  const H: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    const h = H_CENTRAL * Math.max(1.0, Math.abs(x[i]));
    const xp = x.map((v, j) => v + (j === i ? h : 0.0));
    const xm = x.map((v, j) => v - (j === i ? h : 0.0));
    const gp = grad(xp),
      gm = grad(xm);
    for (let r = 0; r < n; r++) H[r][i] = (gp[r] - gm[r]) / (2.0 * h);
  }
  return H.map((row, r) => row.map((v, c) => 0.5 * (v + H[c][r])));
}

// ---------------------------------------------------------------------------------------
// Validation (`_check_common`, `_check_positive`, `_check_unit`)
// ---------------------------------------------------------------------------------------

function fail(message: string): never {
  throw new FirstOrderInputError(message);
}

function checkCommon(gtol: number, maxIter: unknown): number {
  if (!(typeof gtol === 'number' && Number.isFinite(gtol) && gtol >= GTOL_MIN))
    fail(`gtol must be a finite number ≥ 1e-150, got ${reprFloat(gtol)}`);
  if (
    typeof maxIter !== 'number' ||
    !Number.isFinite(maxIter) ||
    !Number.isInteger(maxIter) ||
    maxIter < 1
  )
    fail(`max_iter must be a positive integer, got ${reprValue(maxIter)}`);
  return maxIter;
}

function checkPositive(values: Record<string, number>): void {
  for (const [name, v] of Object.entries(values))
    if (!(Number.isFinite(v) && v > 0.0))
      fail(`${name} must be a finite number > 0, got ${reprFloat(v)}`);
}

function checkUnit(values: Record<string, number>): void {
  for (const [name, v] of Object.entries(values))
    if (!(0.0 <= v && v < 1.0)) fail(`${name} must lie in [0, 1), got ${reprFloat(v)}`);
}

// ---------------------------------------------------------------------------------------
// Shared machinery (`_Run`)
// ---------------------------------------------------------------------------------------

/** A callable with a call counter (`numopt.core.counting.Counted`). */
class Counted<A, R> {
  n = 0;
  private readonly fn: (a: A) => R;
  constructor(fn: (a: A) => R) {
    this.fn = fn;
  }
  call(a: A): R {
    this.n++;
    return this.fn(a);
  }
}

interface VectorProblem {
  id: string;
  dim: number;
  f: (x: Vector) => unknown;
  grad?: (x: Vector) => unknown;
  hess?: (x: Vector) => unknown;
  x0?: unknown;
}

/** `vector_problem`: a library problem as is, or a bare callable wrapped as "custom". */
function vectorProblem(problem: FirstOrderProblem, x0: unknown): VectorProblem {
  if (typeof problem === 'function') {
    const dim = x0 !== undefined && x0 !== null ? toVector(x0).length : 2;
    return { id: 'custom', dim, f: problem };
  }
  if (problem === null || typeof problem !== 'object' || typeof problem.f !== 'function')
    throw new TypeError('problem must be a numopt Problem or a callable f(x)');
  return problem as VectorProblem;
}

/** `start_point`: the explicit x0 or the problem's default. */
function startPoint(prob: VectorProblem, x0: unknown): Vector {
  const start = x0 === undefined || x0 === null ? prob.x0 : x0;
  if (start === undefined || start === null)
    fail(`${prob.id}: no starting point given and the problem has no default x0`);
  const x = toVector(start);
  if (prob.dim && x.length !== prob.dim)
    fail(`${prob.id}: x0 has ${x.length} entries, expected ${prob.dim}`);
  return x;
}

function overflowMessage(k: number): string {
  return (
    `overflow at iteration ${k}: ∇f(x)ᵀp is not finite (‖∇f(x)‖ is too large), so no ` +
    'step length can be computed'
  );
}

function stallMessage(k: number, alpha: number): string {
  return (
    `stalled at iteration ${k}: the accepted step α = ${g3(alpha)} does not change x ` +
    '(f cannot be decreased further at this floating-point precision)'
  );
}

/** Counted oracles, the trace and the shared start/stop logic of one method run. */
class Run {
  readonly name: string;
  readonly gtol: number;
  x: Vector;
  fx = NaN;
  g: Vector;
  fStart = NaN;
  readonly trace: Step[] = [];
  private readonly f: Counted<Vector, unknown>;
  private readonly grad: Counted<Vector, unknown>;
  private readonly hess: Counted<Vector, unknown> | null = null;

  constructor(
    name: string,
    problem: FirstOrderProblem,
    x0: unknown,
    gtol: number,
    needHess = false,
  ) {
    const prob = vectorProblem(problem, x0);
    this.name = name;
    this.gtol = gtol;
    this.x = startPoint(prob, x0);
    const f = new Counted<Vector, unknown>((z) => prob.f(z));
    this.f = f;
    const grad = prob.grad;
    this.grad =
      grad !== undefined
        ? new Counted<Vector, unknown>((z) => grad(z))
        : new Counted<Vector, unknown>((z) => fdGradient((w) => Number(f.call(w)), z));
    const gradCounted = this.grad;
    if (needHess) {
      const hess = prob.hess;
      this.hess =
        hess !== undefined
          ? new Counted<Vector, unknown>((z) => hess(z))
          : new Counted<Vector, unknown>((z) => fdHessian((w) => toVector(gradCounted.call(w)), z));
    }
    this.g = this.x.map(() => NaN);
  }

  // -- evaluation -----------------------------------------------------------------------

  value(x: Vector): number {
    return Number(this.f.call(x));
  }

  gradient(x: Vector): Vector {
    return toVector(this.grad.call(x));
  }

  hessian(x: Vector): Matrix {
    if (this.hess === null) throw new Error('hessian() needs needHess');
    return toMatrix(this.hess.call(x), x.length);
  }

  /** `search` from the current x along p, through the counted oracles (counts read here). */
  lineSearch(kind: string, p: Vector, opts: SearchOptions): LineSearchResult {
    return search(
      kind,
      (z) => Number(this.f.call(z)),
      (z) => this.grad.call(z),
      this.x,
      p,
      { ...opts, f0: this.fx, g0: this.g },
    );
  }

  // -- trace and results ----------------------------------------------------------------

  private record(k: number, alpha: number | null, direction: Vector | null, info: StepInfo) {
    this.trace.push({
      k,
      x: this.x.slice(),
      fun: this.fx,
      gradNorm: norm(this.g),
      stepSize: alpha,
      info: {
        grad: this.g.slice(),
        direction: direction === null ? null : direction.slice(),
        alpha,
        ...info,
      },
    });
  }

  result(converged: boolean, message: string): Result {
    return {
      method: this.name,
      x: this.x.slice(),
      fun: this.fx,
      converged,
      message,
      nIter: this.trace[this.trace.length - 1].k,
      nFev: this.f.n,
      nGev: this.grad.n,
      nHev: this.hess === null ? 0 : this.hess.n,
      trace: this.trace,
      extra: {},
    };
  }

  /** Evaluate f and ∇f at x₀, record step 0; return a finished Result if already done. */
  begin(info: StepInfo): Result | null {
    this.fx = this.value(this.x);
    this.g = this.gradient(this.x);
    this.fStart = this.fx;
    this.record(0, null, null, info);
    if (!(Number.isFinite(this.fx) && allFinite(this.g)))
      return this.result(false, 'f(x0) or ∇f(x0) is not finite');
    const gn = norm(this.g);
    if (gn <= this.gtol) return this.result(true, `‖∇f(x)‖ = ${g3(gn)} ≤ gtol at the start point`);
    return null;
  }

  /** Move to xNew, record step k and apply the stopping/divergence tests. */
  advance(
    k: number,
    xNew: Vector,
    fNew: number,
    gNew: Vector,
    alpha: number,
    direction: Vector,
    info: StepInfo,
  ): Result | null {
    this.x = xNew;
    this.fx = fNew;
    this.g = gNew;
    this.record(k, alpha, direction, info);
    if (!(Number.isFinite(fNew) && allFinite(gNew)))
      return this.result(false, `diverged: f(x) or ∇f(x) is not finite at iteration ${k}`);
    // NOTE (Python): relative to the scale of f (floor 1, so f(x₀) = 0 still has a threshold).
    const riseMax = F_DIVERGE * Math.max(1.0, Math.abs(this.fStart));
    if (fNew - this.fStart > riseMax)
      return this.result(
        false,
        `diverged: f(x) − f(x0) = ${g3(fNew - this.fStart)} > ` +
          `1e+12·max(1, |f(x0)|) = ${g3(riseMax)}`,
      );
    const gn = norm(gNew);
    if (gn <= this.gtol) return this.result(true, `‖∇f(x)‖ = ${g3(gn)} ≤ gtol`);
    return null;
  }

  /** Evaluate f and ∇f at xNew, then `advance`. */
  stepTo(k: number, xNew: Vector, alpha: number, p: Vector, info: StepInfo): Result | null {
    const fNew = this.value(xNew);
    const gNew = this.gradient(xNew);
    return this.advance(k, xNew, fNew, gNew, alpha, p, info);
  }

  maxIter(maxIter: number): Result {
    const gn = norm(this.g);
    return this.result(false, `reached max_iter=${maxIter} (‖∇f(x)‖ = ${g3(gn)} > gtol)`);
  }
}

type Opts = RunOptions & Params;

/** A numeric option with the Python keyword default. */
function numOpt(o: Opts, name: string, def: number): number {
  const v = o[name];
  return v === undefined ? def : (v as number);
}

// ---------------------------------------------------------------------------------------
// Gradient descent
// ---------------------------------------------------------------------------------------

/** First trial step of a gradient-descent line search (`_gd_alpha0`). */
export function gdAlpha0(
  gg: number,
  fCur: number,
  alphaPrev: number | null,
  ggPrev: number,
  fPrev: number,
): number {
  const unit = 1.0 / Math.sqrt(gg); // a trial move of unit length
  if (alphaPrev === null) return unit;
  // NOTE (Python): N&W §3.5 offer eq. 3.60 and eq. 3.61 as alternatives; the larger one is taken.
  const guesses = [
    (2.0 * (fPrev - fCur)) / gg, // N&W eq. 3.61 (φ'(0) = −gg)
    (alphaPrev * ggPrev) / gg, // N&W eq. 3.60
  ].filter((a) => Number.isFinite(a) && a > 0.0);
  return guesses.length ? Math.max(...guesses) : unit;
}

export const gradientDescent: MethodFn<FirstOrderProblem> = (problem, o) => {
  const stepRule = (o.step_rule ?? 'backtracking') as string;
  if (!(STEP_RULES as readonly string[]).includes(stepRule))
    fail(
      `step_rule must be one of (${STEP_RULES.map((s) => `'${s}'`).join(', ')}), got ${reprValue(stepRule)}`,
    );
  const lr = numOpt(o, 'lr', 1e-3);
  const gtol = numOpt(o, 'gtol', 1e-6);
  const maxIter = checkCommon(gtol, o.max_iter ?? 5000);
  checkPositive({ lr });
  const run = new Run('gradient_descent', problem, o.x0, gtol, stepRule === 'exact_quadratic');
  const isFixed = stepRule === 'fixed';
  let done = run.begin({ trials: [], alpha0: null, pHp: null });
  if (done) return done;
  let alphaPrev: number | null = null;
  let ggPrev = NaN;
  let fPrev = NaN;
  for (let k = 1; k <= maxIter; k++) {
    const p = run.g.map((v) => -v);
    const gg = dot(run.g, run.g); // = −∇fᵀp, the slope of the line search
    if (!isFixed && !Number.isFinite(gg)) return run.result(false, overflowMessage(k));
    if (isFixed) {
      const xNew = xPlus(run.x, lr, p);
      done = run.stepTo(k, xNew, lr, p, { trials: [], alpha0: null, pHp: null });
    } else {
      let pHp: number | null = null;
      let alpha0: number | null = null;
      const opts: SearchOptions = {};
      if (stepRule === 'exact_quadratic') {
        const H = run.hessian(run.x);
        pHp = dot(
          p,
          H.map((row) => dot(row, p)),
        );
        if (!(Number.isFinite(pHp) && pHp > 0.0))
          return run.result(
            false,
            `exact_quadratic step undefined: pᵀ∇²f p = ${g3(pHp)} ≤ 0 ` +
              `at iteration ${k} (f is not convex along −∇f)`,
          );
        opts.hess = H;
      } else {
        alpha0 = gdAlpha0(gg, run.fx, alphaPrev, ggPrev, fPrev);
        opts.alpha0 = alpha0;
        if (stepRule === 'strong_wolfe')
          opts.alphaMax = Math.min(ALPHA_MAX_RATIO * alpha0, FLOAT_MAX);
      }
      let ls: LineSearchResult;
      try {
        ls = run.lineSearch(stepRule, p, opts);
      } catch (exc) {
        // NOTE (Python): never throw on numerical breakdown (as in conjugate_gradient). The zoom
        // of the strong Wolfe search divides by h², which underflows for brackets narrower than
        // ≈ 1.5e-162 (step lengths α ≲ 1e-154, f scaled by ≈ 2⁴⁹⁶).
        if (exc instanceof LineSearchZeroDivision)
          return run.result(
            false,
            `line search broke down at iteration ${k} (${exc.name}: ${exc.message}); ` +
              'the step lengths are at the limit of float64 (rescale f)',
          );
        throw exc;
      }
      let alpha = ls.alpha;
      let fNew = ls.fNew;
      let gLs = ls.gNew;
      if (!ls.success) {
        const last = ls.trials.length ? ls.trials[ls.trials.length - 1] : null;
        // NOTE (Python): N&W Alg. 3.5 fails when φ still decreases at α_max. That step satisfies
        // the Armijo condition, so it is taken; ∇f there is evaluated again (one more nGev).
        if (
          stepRule === 'strong_wolfe' &&
          last !== null &&
          last[0] === opts.alphaMax &&
          Number.isFinite(last[1]) &&
          last[1] <= run.fx - C1_DEFAULT * last[0] * gg
        ) {
          alpha = last[0];
          fNew = last[1];
          gLs = null;
        } else {
          const hit = ls.trials.find(([, phi]) => phi === -Infinity);
          if (hit !== undefined)
            return run.result(
              false,
              `diverged: f(x + αp) = −∞ at α = ${g3(hit[0])} in the line ` +
                `search of iteration ${k} (f is unbounded below along −∇f)`,
            );
          return run.result(false, `line search failed at iteration ${k}: ${ls.message}`);
        }
      }
      const xNew = xPlus(run.x, alpha, p);
      if (arrayEqual(xNew, run.x)) return run.result(false, stallMessage(k, alpha));
      const gNew = gLs === null ? run.gradient(xNew) : gLs;
      const trials = ls.trials.map(([a, phi]) => [a, phi]);
      alphaPrev = alpha;
      ggPrev = gg;
      fPrev = run.fx;
      done = run.advance(k, xNew, fNew, gNew, alpha, p, { trials, alpha0, pHp });
    }
    if (done) return done;
  }
  return run.maxIter(maxIter);
};

// ---------------------------------------------------------------------------------------
// Barzilai–Borwein
// ---------------------------------------------------------------------------------------

const F64 = new DataView(new ArrayBuffer(8));

/** The exponent e of `math.frexp(v)` (2^(e−1) ≤ |v| < 2^e); 0 for v = 0, ±inf or NaN. */
export function frexpExp(v: number): number {
  if (v === 0 || !Number.isFinite(v)) return 0;
  F64.setFloat64(0, v);
  const biased = (F64.getUint32(0) >>> 20) & 0x7ff;
  if (biased === 0) return frexpExp(v * 2 ** 64) - 64; // subnormal
  return biased - 1022;
}

/**
 * `np.ldexp(x, n)` = x·2ⁿ, rounded once (musl `scalbn`): the factor is applied in steps that stay
 * exact, and the last step leaves n < −53 so a subnormal result is not rounded twice.
 */
export function ldexp(x: number, n: number): number {
  let y = x;
  let k = n;
  if (k > 1023) {
    y *= 2 ** 1023;
    k -= 1023;
    if (k > 1023) {
      y *= 2 ** 1023;
      k -= 1023;
      if (k > 1023) k = 1023;
    }
  } else if (k < -1022) {
    y *= 2 ** -969; // 2⁻¹⁰²²·2⁵³
    k += 969;
    if (k < -1022) {
      y *= 2 ** -969;
      k += 969;
      if (k < -1022) k = -1022;
    }
  }
  return y * 2 ** k;
}

/** `_pow2_scaled`: (v·2⁻ᵉ, e) with 2^(e−1) ≤ ‖v‖∞ < 2^e; (v, 0) when v = 0. Exact. */
function pow2Scaled(v: readonly number[]): [Vector, number] {
  let m = 0;
  for (const t of v) m = Math.max(m, Math.abs(t));
  const e = frexpExp(m);
  return [v.map((t) => ldexp(t, -e)), e];
}

/**
 * `_bb_steps`: (BB1, BB2) = (sᵀs/sᵀy, sᵀy/yᵀy) for finite s ≠ 0; [null, null] when sᵀy ≤ 0.
 *
 * The inner products are formed from s and y scaled by powers of 2 (Higham (2002), §27.8): the
 * scaling is exact, so the quotients equal the unscaled ones bit for bit whenever no product
 * under- or overflows, and stay defined when yᵀy or sᵀy would underflow to 0.
 */
export function bbSteps(
  s: readonly number[],
  y: readonly number[],
): [number | null, number | null] {
  // NOTE (Python): unscaled, yᵀy underflows to 0 for ‖y‖∞ ≲ 1e-162 while sᵀy > 0 is still
  // representable (‖s‖ ≫ ‖y‖). A quotient that overflows becomes +Infinity, which the caller's
  // range test [BB_MIN, BB_MAX] resets.
  const [sHat, eS] = pow2Scaled(s);
  const [yHat, eY] = pow2Scaled(y);
  const sy = dot(sHat, yHat);
  if (!(sy > 0.0)) return [null, null]; // also y = 0
  const ss = dot(sHat, sHat); // both in [1/4, n)
  const yy = dot(yHat, yHat);
  return [ldexp(ss / sy, eS - eY), ldexp(sy / yy, eS - eY)];
}

/** Backtracking factor σ ∈ [σ₁, σ₂] from the quadratic interpolant of φ(α) = f(x − αg). */
export function bbSigma(alpha: number, fAlpha: number, f0: number, gg: number): number {
  if (!Number.isFinite(fAlpha)) return BB_SIGMA1;
  const alphaSq = alpha * alpha;
  if (alphaSq === 0.0) return BB_SIGMA2; // α < 1.5e-162: c is not representable
  const curv = (fAlpha - f0 + gg * alpha) / alphaSq;
  if (!(Number.isFinite(curv) && curv > 0.0)) return BB_SIGMA2;
  const denom = 2.0 * curv * alpha;
  // Underflow with c > 0: σ = gᵀg/(2cα) exceeds every float, so it is clipped to σ₂.
  if (denom === 0.0) return BB_SIGMA2;
  const sigma = gg / denom;
  return Math.min(BB_SIGMA2, Math.max(BB_SIGMA1, sigma));
}

export const barzilaiBorwein: MethodFn<FirstOrderProblem> = (problem, o) => {
  const variant = (o.variant ?? 'bb1') as string;
  if (!(BB_VARIANTS as readonly string[]).includes(variant))
    fail(`variant must be one of ('bb1', 'bb2'), got ${reprValue(variant)}`);
  const nonmonotone = Boolean(o.nonmonotone ?? true);
  const gtol = numOpt(o, 'gtol', 1e-6);
  const maxIter = checkCommon(gtol, o.max_iter ?? 1000);
  const run = new Run('barzilai_borwein', problem, o.x0, gtol);
  let done = run.begin({
    alpha_bb: null,
    bb1: null,
    bb2: null,
    reset: false,
    f_ref: null,
    trials: [],
  });
  if (done) return done;
  const fHist = [run.fx];
  let xPrev: Vector | null = null;
  let gPrev: Vector | null = null;
  for (let k = 1; k <= maxIter; k++) {
    const gg = dot(run.g, run.g);
    if (!Number.isFinite(gg)) return run.result(false, overflowMessage(k));
    let bb1: number | null = null;
    let bb2: number | null = null;
    if (xPrev !== null && gPrev !== null) {
      const xp = xPrev,
        gp = gPrev;
      const s = run.x.map((v, i) => v - xp[i]);
      const y = run.g.map((v, i) => v - gp[i]);
      [bb1, bb2] = bbSteps(s, y);
    }
    let alphaBb = variant === 'bb1' ? bb1 : bb2;
    const reset = alphaBb === null || !(BB_MIN <= alphaBb && alphaBb <= BB_MAX);
    if (reset || alphaBb === null)
      alphaBb = Math.min(BB_MAX, Math.max(BB_MIN, 1.0 / Math.sqrt(gg)));
    const p = run.g.map((v) => -v);
    let alpha = alphaBb;
    let fNew = run.value(xPlus(run.x, alpha, p));
    const trials: [number, number][] = [[alpha, fNew]];
    let fRef: number | null = null;
    if (nonmonotone) {
      fRef = Math.max(...fHist.slice(-BB_MEMORY));
      while (!(Number.isFinite(fNew) && fNew <= fRef - BB_GAMMA * alpha * gg)) {
        if (trials.length >= BB_MAX_TRIALS)
          return run.result(
            false,
            `nonmonotone line search failed at iteration ${k}: no acceptable step ` +
              `in ${BB_MAX_TRIALS} trials (last α = ${g3(alpha)})`,
          );
        alpha *= bbSigma(alpha, fNew, run.fx, gg);
        fNew = run.value(xPlus(run.x, alpha, p));
        trials.push([alpha, fNew]);
      }
    }
    const xNew = xPlus(run.x, alpha, p);
    if (arrayEqual(xNew, run.x)) return run.result(false, stallMessage(k, alpha));
    xPrev = run.x;
    gPrev = run.g;
    done = run.advance(k, xNew, fNew, run.gradient(xNew), alpha, p, {
      alpha_bb: alphaBb,
      bb1,
      bb2,
      reset,
      f_ref: fRef,
      trials,
    });
    if (done) return done;
    fHist.push(run.fx);
  }
  return run.maxIter(maxIter);
};

// ---------------------------------------------------------------------------------------
// Momentum methods
// ---------------------------------------------------------------------------------------

export const momentum: MethodFn<FirstOrderProblem> = (problem, o) => {
  const lr = numOpt(o, 'lr', 1e-3);
  const beta = numOpt(o, 'beta', 0.9);
  const gtol = numOpt(o, 'gtol', 1e-6);
  const maxIter = checkCommon(gtol, o.max_iter ?? 5000);
  checkPositive({ lr });
  checkUnit({ beta });
  const run = new Run('momentum', problem, o.x0, gtol);
  let v = run.x.map(() => 0.0);
  let done = run.begin({ velocity: v.slice() });
  if (done) return done;
  for (let k = 1; k <= maxIter; k++) {
    const g = run.g;
    v = v.map((vi, i) => beta * vi - lr * g[i]);
    const xNew = run.x.map((xi, i) => xi + v[i]);
    done = run.stepTo(
      k,
      xNew,
      lr,
      v.map((vi) => vi / lr),
      { velocity: v.slice() },
    );
    if (done) return done;
  }
  return run.maxIter(maxIter);
};

export const nesterov: MethodFn<FirstOrderProblem> = (problem, o) => {
  const lr = numOpt(o, 'lr', 1e-3);
  const beta = numOpt(o, 'beta', 0.9);
  const gtol = numOpt(o, 'gtol', 1e-6);
  const maxIter = checkCommon(gtol, o.max_iter ?? 5000);
  checkPositive({ lr });
  checkUnit({ beta });
  const run = new Run('nesterov', problem, o.x0, gtol);
  let v = run.x.map(() => 0.0);
  let done = run.begin({ velocity: v.slice(), lookahead: null, grad_lookahead: null });
  if (done) return done;
  for (let k = 1; k <= maxIter; k++) {
    const vPrev = v;
    const y = run.x.map((xi, i) => xi + beta * vPrev[i]);
    const gy = arrayEqual(y, run.x) ? run.g : run.gradient(y);
    if (!allFinite(gy))
      return run.result(false, `diverged: ∇f is not finite at the look-ahead point (k=${k})`);
    v = vPrev.map((vi, i) => beta * vi - lr * gy[i]);
    const xNew = run.x.map((xi, i) => xi + v[i]);
    done = run.stepTo(
      k,
      xNew,
      lr,
      v.map((vi) => vi / lr),
      { velocity: v.slice(), lookahead: y.slice(), grad_lookahead: gy.slice() },
    );
    if (done) return done;
  }
  return run.maxIter(maxIter);
};

// ---------------------------------------------------------------------------------------
// Adaptive (per-coordinate) methods
// ---------------------------------------------------------------------------------------

export const adagrad: MethodFn<FirstOrderProblem> = (problem, o) => {
  const lr = numOpt(o, 'lr', 1.0);
  const eps = numOpt(o, 'eps', 1e-8);
  const gtol = numOpt(o, 'gtol', 1e-6);
  const maxIter = checkCommon(gtol, o.max_iter ?? 5000);
  checkPositive({ lr, eps });
  const run = new Run('adagrad', problem, o.x0, gtol);
  let G = run.x.map(() => 0.0);
  let done = run.begin({ v: G.slice(), lr_eff: null });
  if (done) return done;
  for (let k = 1; k <= maxIter; k++) {
    const g = run.g;
    G = G.map((Gi, i) => Gi + g[i] * g[i]);
    const denom = G.map((Gi) => Math.sqrt(Gi) + eps);
    const p = g.map((gi, i) => -gi / denom[i]);
    done = run.stepTo(k, xPlus(run.x, lr, p), lr, p, {
      v: G.slice(),
      lr_eff: denom.map((d) => lr / d),
    });
    if (done) return done;
  }
  return run.maxIter(maxIter);
};

export const rmsprop: MethodFn<FirstOrderProblem> = (problem, o) => {
  const lr = numOpt(o, 'lr', 0.1);
  const rho = numOpt(o, 'rho', 0.9);
  const eps = numOpt(o, 'eps', 1e-8);
  const gtol = numOpt(o, 'gtol', 1e-6);
  const maxIter = checkCommon(gtol, o.max_iter ?? 5000);
  checkPositive({ lr, eps });
  checkUnit({ rho });
  const run = new Run('rmsprop', problem, o.x0, gtol);
  let E = run.x.map(() => 0.0);
  let done = run.begin({ v: E.slice(), lr_eff: null });
  if (done) return done;
  for (let k = 1; k <= maxIter; k++) {
    const g = run.g;
    E = E.map((Ei, i) => rho * Ei + (1.0 - rho) * g[i] * g[i]);
    const denom = E.map((Ei) => Math.sqrt(Ei) + eps);
    const p = g.map((gi, i) => -gi / denom[i]);
    done = run.stepTo(k, xPlus(run.x, lr, p), lr, p, {
      v: E.slice(),
      lr_eff: denom.map((d) => lr / d),
    });
    if (done) return done;
  }
  return run.maxIter(maxIter);
};

export const adadelta: MethodFn<FirstOrderProblem> = (problem, o) => {
  const rho = numOpt(o, 'rho', 0.95);
  const eps = numOpt(o, 'eps', 1e-6);
  const gtol = numOpt(o, 'gtol', 1e-6);
  const maxIter = checkCommon(gtol, o.max_iter ?? 5000);
  checkPositive({ eps });
  checkUnit({ rho });
  const run = new Run('adadelta', problem, o.x0, gtol);
  let Eg = run.x.map(() => 0.0);
  let Edx = run.x.map(() => 0.0);
  let done = run.begin({ v: Eg.slice(), u: Edx.slice(), lr_eff: null });
  if (done) return done;
  for (let k = 1; k <= maxIter; k++) {
    const g = run.g;
    Eg = Eg.map((e, i) => rho * e + (1.0 - rho) * g[i] * g[i]);
    const egk = Eg;
    const lrEff = Edx.map((e, i) => Math.sqrt(e + eps) / Math.sqrt(egk[i] + eps));
    const dx = lrEff.map((l, i) => -l * g[i]);
    Edx = Edx.map((e, i) => rho * e + (1.0 - rho) * dx[i] * dx[i]);
    const xNew = run.x.map((xi, i) => xi + dx[i]);
    done = run.stepTo(k, xNew, 1.0, dx, { v: Eg.slice(), u: Edx.slice(), lr_eff: lrEff });
    if (done) return done;
  }
  return run.maxIter(maxIter);
};

/** Adam (Kingma & Ba 2015, Alg. 1) and AdamW (Loshchilov & Hutter 2019, Alg. 2, η_t = 1). */
function adamLike(name: 'adam' | 'adamw', problem: FirstOrderProblem, o: Opts): Result {
  const lr = numOpt(o, 'lr', 0.05);
  const beta1 = numOpt(o, 'beta1', 0.9);
  const beta2 = numOpt(o, 'beta2', 0.999);
  const eps = numOpt(o, 'eps', 1e-8);
  const weightDecay = name === 'adamw' ? numOpt(o, 'weight_decay', 1e-3) : 0.0;
  const gtol = numOpt(o, 'gtol', 1e-6);
  const maxIter = checkCommon(gtol, o.max_iter ?? 5000);
  checkPositive({ lr, eps });
  checkUnit({ beta1, beta2 });
  if (!(Number.isFinite(weightDecay) && weightDecay >= 0.0))
    fail(`weight_decay must be a finite number ≥ 0, got ${reprFloat(weightDecay)}`);
  const isW = name === 'adamw';
  const run = new Run(name, problem, o.x0, gtol);
  let m = run.x.map(() => 0.0);
  let v = run.x.map(() => 0.0);
  const zeros = m.slice();
  const extra0: StepInfo = isW ? { decay: null } : {};
  let done = run.begin({
    m: zeros,
    v: zeros.slice(),
    m_hat: zeros.slice(),
    v_hat: zeros.slice(),
    lr_eff: null,
    ...extra0,
  });
  if (done) return done;
  for (let t = 1; t <= maxIter; t++) {
    const g = run.g;
    m = m.map((mi, i) => beta1 * mi + (1.0 - beta1) * g[i]);
    v = v.map((vi, i) => beta2 * vi + (1.0 - beta2) * g[i] * g[i]);
    const b1 = 1.0 - beta1 ** t,
      b2 = 1.0 - beta2 ** t;
    const mHat = m.map((mi) => mi / b1);
    const vHat = v.map((vi) => vi / b2);
    const denom = vHat.map((vh) => Math.sqrt(vh) + eps);
    let step = mHat.map((mh, i) => (-lr * mh) / denom[i]);
    const extra: StepInfo = {};
    if (isW) {
      const decay = run.x.map((xi) => -weightDecay * xi);
      step = step.map((s, i) => s + decay[i]);
      extra.decay = decay;
    }
    const xNew = run.x.map((xi, i) => xi + step[i]);
    done = run.stepTo(
      t,
      xNew,
      lr,
      step.map((s) => s / lr),
      {
        m: m.slice(),
        v: v.slice(),
        m_hat: mHat,
        v_hat: vHat,
        lr_eff: denom.map((d) => lr / d),
        ...extra,
      },
    );
    if (done) return done;
  }
  return run.maxIter(maxIter);
}

export const adam: MethodFn<FirstOrderProblem> = (problem, o) => adamLike('adam', problem, o);

export const adamw: MethodFn<FirstOrderProblem> = (problem, o) => adamLike('adamw', problem, o);

export const adamax: MethodFn<FirstOrderProblem> = (problem, o) => {
  const lr = numOpt(o, 'lr', 0.2);
  const beta1 = numOpt(o, 'beta1', 0.9);
  const beta2 = numOpt(o, 'beta2', 0.999);
  const gtol = numOpt(o, 'gtol', 1e-6);
  const maxIter = checkCommon(gtol, o.max_iter ?? 5000);
  checkPositive({ lr });
  checkUnit({ beta1, beta2 });
  const run = new Run('adamax', problem, o.x0, gtol);
  let m = run.x.map(() => 0.0);
  let u = run.x.map(() => 0.0);
  let done = run.begin({ m: m.slice(), u: u.slice(), lr_eff: null });
  if (done) return done;
  for (let t = 1; t <= maxIter; t++) {
    const g = run.g;
    m = m.map((mi, i) => beta1 * mi + (1.0 - beta1) * g[i]);
    // np.maximum propagates NaN, like Math.max.
    u = u.map((ui, i) => Math.max(beta2 * ui, Math.abs(g[i])));
    const bias = 1.0 - beta1 ** t;
    const p = safeDiv(m, u).map((q) => -q / bias); // bounded m/u first; 0 where u = 0
    const lrEff = stepSizes(lr / bias, u);
    done = run.stepTo(t, xPlus(run.x, lr, p), lr, p, { m: m.slice(), u: u.slice(), lr_eff: lrEff });
    if (done) return done;
  }
  return run.maxIter(maxIter);
};

export const nadam: MethodFn<FirstOrderProblem> = (problem, o) => {
  const lr = numOpt(o, 'lr', 0.02);
  const beta1 = numOpt(o, 'beta1', 0.9);
  const beta2 = numOpt(o, 'beta2', 0.999);
  const eps = numOpt(o, 'eps', 1e-8);
  const gtol = numOpt(o, 'gtol', 1e-6);
  const maxIter = checkCommon(gtol, o.max_iter ?? 5000);
  checkPositive({ lr, eps });
  checkUnit({ beta1, beta2 });
  const run = new Run('nadam', problem, o.x0, gtol);
  let m = run.x.map(() => 0.0);
  let n = run.x.map(() => 0.0);
  const zeros = m.slice();
  let done = run.begin({
    m: zeros,
    v: zeros.slice(),
    m_bar: zeros.slice(),
    v_hat: zeros.slice(),
    lr_eff: null,
  });
  if (done) return done;
  for (let t = 1; t <= maxIter; t++) {
    const g = run.g;
    const c1t = 1.0 - beta1 ** t;
    const gHat = g.map((gi) => gi / c1t);
    m = m.map((mi, i) => beta1 * mi + (1.0 - beta1) * g[i]);
    const c1t1 = 1.0 - beta1 ** (t + 1);
    const mHat = m.map((mi) => mi / c1t1);
    n = n.map((ni, i) => beta2 * ni + (1.0 - beta2) * g[i] * g[i]);
    const c2t = 1.0 - beta2 ** t;
    const nHat = n.map((ni) => ni / c2t);
    const mBar = gHat.map((gh, i) => (1.0 - beta1) * gh + beta1 * mHat[i]);
    const denom = nHat.map((nh) => Math.sqrt(nh) + eps);
    const p = mBar.map((mb, i) => -mb / denom[i]);
    done = run.stepTo(t, xPlus(run.x, lr, p), lr, p, {
      m: m.slice(),
      v: n.slice(),
      m_bar: mBar,
      v_hat: nHat,
      lr_eff: denom.map((d) => lr / d),
    });
    if (done) return done;
  }
  return run.maxIter(maxIter);
};

export const amsgrad: MethodFn<FirstOrderProblem> = (problem, o) => {
  const lr = numOpt(o, 'lr', 0.1);
  const beta1 = numOpt(o, 'beta1', 0.9);
  const beta2 = numOpt(o, 'beta2', 0.999);
  const gtol = numOpt(o, 'gtol', 1e-6);
  const maxIter = checkCommon(gtol, o.max_iter ?? 5000);
  checkPositive({ lr });
  checkUnit({ beta1, beta2 });
  const run = new Run('amsgrad', problem, o.x0, gtol);
  const sqrtB2 = Math.sqrt(beta2),
    sqrt1mB2 = Math.sqrt(1.0 - beta2);
  let m = run.x.map(() => 0.0);
  let r = run.x.map(() => 0.0); // r_t = √v_t
  let rMax = run.x.map(() => 0.0); // r̂_t = √v̂_t
  const zeros = m.slice();
  let done = run.begin({ m: zeros, v: zeros.slice(), v_max: zeros.slice(), lr_eff: null });
  if (done) return done;
  for (let t = 1; t <= maxIter; t++) {
    const g = run.g;
    m = m.map((mi, i) => beta1 * mi + (1.0 - beta1) * g[i]);
    r = r.map((ri, i) => Math.hypot(sqrtB2 * ri, sqrt1mB2 * g[i])); // √(β₂v + (1 − β₂)g²) without g²
    const rt = r;
    rMax = rMax.map((rm, i) => Math.max(rm, rt[i]));
    const p = safeDiv(m, rMax).map((q) => -q); // 0 where r̂ = 0
    const v = r.map((ri) => Math.min(ri * ri, FLOAT_MAX));
    const vMax = rMax.map((ri) => Math.min(ri * ri, FLOAT_MAX));
    done = run.stepTo(t, xPlus(run.x, lr, p), lr, p, {
      m: m.slice(),
      v,
      v_max: vMax,
      lr_eff: stepSizes(lr, rMax),
    });
    if (done) return done;
  }
  return run.maxIter(maxIter);
};

// ---------------------------------------------------------------------------------------
// Cyclic coordinate descent
// ---------------------------------------------------------------------------------------

export const coordinateDescent: MethodFn<FirstOrderProblem> = (problem, o) => {
  const gtol = numOpt(o, 'gtol', 1e-6);
  const maxIter = checkCommon(gtol, o.max_iter ?? 5000);
  const run = new Run('coordinate_descent', problem, o.x0, gtol, true);
  const n = run.x.length;
  let done = run.begin({
    coordinate: null,
    sweep: null,
    curvature: null,
    newton: null,
    trials: [],
  });
  if (done) return done;
  let unchanged = 0; // consecutive iterations that left x unchanged
  for (let k = 1; k <= maxIter; k++) {
    const xOld = run.x;
    const i = (k - 1) % n;
    const sweep = Math.floor((k - 1) / n);
    const gi = run.g[i];
    const h = run.hessian(run.x)[i][i];
    const p = run.x.map(() => 0.0);
    const newtonStep = Number.isFinite(h) && h > 0.0;
    const pI = newtonStep ? -gi / h : -gi;
    const slope = gi * pI; // φ'(0) = ∇fᵀp of the 1-D search
    if (gi !== 0.0 && !(Number.isFinite(pI) && Number.isFinite(slope)))
      return run.result(false, overflowMessage(k));
    // NOTE (Python): a slope that underflows to 0, or a full step that does not change x_i in
    // floating point, is treated like g_i = 0: the coordinate is skipped.
    if (slope < 0.0 && run.x[i] + pI !== run.x[i]) p[i] = pI;
    if (p[i] === 0.0) {
      done = run.advance(k, run.x.slice(), run.fx, run.g, 0.0, p, {
        coordinate: i,
        sweep,
        curvature: h,
        newton: false,
        trials: [],
      });
    } else {
      const ls = run.lineSearch('backtracking', p, { alpha0: 1.0 });
      if (!ls.success)
        return run.result(false, `line search failed at iteration ${k}: ${ls.message}`);
      const xNew = xPlus(run.x, ls.alpha, p);
      done = run.advance(k, xNew, ls.fNew, run.gradient(xNew), ls.alpha, p, {
        coordinate: i,
        sweep,
        curvature: h,
        newton: newtonStep,
        trials: ls.trials.map(([a, phi]) => [a, phi]),
      });
    }
    if (done) return done;
    unchanged = arrayEqual(run.x, xOld) ? unchanged + 1 : 0;
    if (unchanged >= n)
      return run.result(
        false,
        `stalled at iteration ${k}: ${n} consecutive coordinate steps left x unchanged ` +
          '(f cannot be decreased further at this floating-point precision)',
      );
  }
  return run.maxIter(maxIter);
};

// ---------------------------------------------------------------------------------------
// MethodCard material (TS-only)
// ---------------------------------------------------------------------------------------

const Q_ALPHA = { tex: '\\alpha_k', key: 'stepSize', label: 'step length' };
const Q_GRADNORM = { tex: '\\|\\nabla f(x_k)\\|_2', key: 'gradNorm', label: 'gradient norm' };

const DOCS: Record<string, MethodDoc> = {
  gradient_descent: {
    rule: 'x_k = x_{k-1} - \\alpha_k\\,\\nabla f(x_{k-1})',
    intuition:
      'Step straight downhill, along the negative gradient. A rule sets the step length: a fixed α, a backtracking or Wolfe line search, or the exact minimizer of the quadratic model.',
    order: 'linear; ((κ − 1)/(κ + 1))² per step in f with exact steps on a quadratic',
    pros: ['One gradient per step, no matrices', 'Monotone decrease with a line search'],
    cons: ['Zig-zags across narrow valleys', 'Slow when the Hessian is ill-conditioned (large κ)'],
    quantities: [
      Q_ALPHA,
      Q_GRADNORM,
      { tex: '\\alpha_0', key: 'info.alpha0', label: 'first trial step' },
    ],
  },
  barzilai_borwein: {
    rule: '\\alpha_k = \\frac{s^\\top s}{s^\\top y}\\ \\text{(BB1)}\\quad\\text{or}\\quad \\frac{s^\\top y}{y^\\top y}\\ \\text{(BB2)},\\qquad s = x_{k-1} - x_{k-2},\\; y = \\nabla f(x_{k-1}) - \\nabla f(x_{k-2})',
    intuition:
      'A gradient step whose length comes from the last two iterates: the scalar 1/α that best fits the secant equation Bs = y. Steps are not monotone, so a nonmonotone search compares f with the worst of the last ten values.',
    pros: ['Much faster than gradient descent at the same cost', 'No Hessian, no matrix'],
    cons: ['f does not decrease at every step', 'Pure BB can diverge on non-convex f'],
    quantities: [
      Q_ALPHA,
      { tex: '\\alpha^{BB}', key: 'info.alpha_bb', label: 'BB trial step' },
      { tex: 'f_{ref}', key: 'info.f_ref', label: 'nonmonotone reference' },
      Q_GRADNORM,
    ],
  },
  momentum: {
    rule: 'v_k = \\beta\\,v_{k-1} - \\alpha\\,\\nabla f(x_{k-1}),\\qquad x_k = x_{k-1} + v_k',
    intuition:
      'A heavy ball rolling downhill keeps part of its velocity. Oscillations across a valley cancel, and the motion along the valley builds up.',
    pros: ['Rate (√κ − 1)/(√κ + 1) on quadratics with tuned α, β', 'One gradient per step'],
    cons: ['Overshoots and oscillates', 'Needs α and β tuned to the curvature'],
    quantities: [Q_ALPHA, { tex: 'v_k', key: 'info.velocity', label: 'velocity' }, Q_GRADNORM],
  },
  nesterov: {
    rule: 'v_k = \\mu\\,v_{k-1} - \\alpha\\,\\nabla f(x_{k-1} + \\mu\\,v_{k-1}),\\qquad x_k = x_{k-1} + v_k',
    intuition:
      'Momentum that looks ahead: the gradient is taken at the point where the momentum is about to carry the iterate. This damps the overshoot of the heavy ball.',
    pros: ['Accelerated rate 1 − 1/√κ in f with α = 1/L', 'Less overshoot than heavy ball'],
    cons: ['Two gradients per iteration here (look-ahead and stopping test)'],
    quantities: [
      Q_ALPHA,
      { tex: 'x_{k-1} + \\mu v_{k-1}', key: 'info.lookahead', label: 'look-ahead point' },
      { tex: 'v_k', key: 'info.velocity', label: 'velocity' },
      Q_GRADNORM,
    ],
  },
  adagrad: {
    rule: 'G_k = G_{k-1} + g_k^{2},\\qquad x_k = x_{k-1} - \\alpha\\,\\frac{g_k}{\\sqrt{G_k} + \\varepsilon}',
    intuition:
      'Each coordinate gets its own step: α divided by the root of the sum of all its past squared gradients. Steep coordinates take short steps, and every step only shrinks.',
    pros: ['Scale-free per coordinate', 'No line search'],
    cons: ['Steps shrink like 1/√k and can stall before x*'],
    quantities: [
      Q_ALPHA,
      { tex: 'G_k', key: 'info.v', label: 'sum of squared gradients' },
      {
        tex: '\\alpha/(\\sqrt{G_k}+\\varepsilon)',
        key: 'info.lr_eff',
        label: 'per-coordinate step',
      },
      Q_GRADNORM,
    ],
  },
  rmsprop: {
    rule: 'E_k = \\rho\\,E_{k-1} + (1-\\rho)\\,g_k^{2},\\qquad x_k = x_{k-1} - \\alpha\\,\\frac{g_k}{\\sqrt{E_k} + \\varepsilon}',
    intuition:
      'AdaGrad with a decaying average of the squared gradients, so old gradients are forgotten and the steps do not vanish.',
    pros: ['Per-coordinate scaling that adapts over time'],
    cons: ['Normalized steps do not shrink as ∇f → 0: it can hover near x*'],
    quantities: [
      Q_ALPHA,
      { tex: 'E_k', key: 'info.v', label: 'mean of squared gradients' },
      {
        tex: '\\alpha/(\\sqrt{E_k}+\\varepsilon)',
        key: 'info.lr_eff',
        label: 'per-coordinate step',
      },
      Q_GRADNORM,
    ],
  },
  adadelta: {
    rule: '\\Delta x_k = -\\frac{\\sqrt{E[\\Delta x^2]_{k-1} + \\varepsilon}}{\\sqrt{E[g^2]_k + \\varepsilon}}\\,g_k,\\qquad x_k = x_{k-1} + \\Delta x_k',
    intuition:
      'RMSprop whose learning rate is the RMS of the recent updates, so there is no learning rate to set. The first steps are tiny (about √ε) and grow with the updates.',
    pros: ['No learning rate', 'Units of the update match the units of x'],
    cons: ['Slow start', 'No convergence guarantee'],
    quantities: [
      { tex: 'E[g^2]_k', key: 'info.v', label: 'mean of squared gradients' },
      { tex: 'E[\\Delta x^2]_k', key: 'info.u', label: 'mean of squared updates' },
      { tex: '\\Delta x_k', key: 'info.direction', label: 'update' },
      Q_GRADNORM,
    ],
  },
  adam: {
    rule: 'x_k = x_{k-1} - \\alpha\\,\\frac{\\hat m_k}{\\sqrt{\\hat v_k} + \\varepsilon},\\qquad \\hat m_k = \\frac{m_k}{1-\\beta_1^{k}},\\; \\hat v_k = \\frac{v_k}{1-\\beta_2^{k}}',
    intuition:
      'Momentum on the gradient (m) and RMSprop scaling (v), both as exponential averages with a bias correction for the zero start.',
    pros: ['Robust default in machine learning', 'Per-coordinate scaling with momentum'],
    cons: ['No convergence guarantee with a constant α (Reddi et al. 2018)'],
    quantities: [
      Q_ALPHA,
      { tex: '\\hat m_k', key: 'info.m_hat', label: 'first moment' },
      { tex: '\\hat v_k', key: 'info.v_hat', label: 'second moment' },
      Q_GRADNORM,
    ],
  },
  adamw: {
    rule: 'x_k = x_{k-1} - \\Big(\\alpha\\,\\frac{\\hat m_k}{\\sqrt{\\hat v_k} + \\varepsilon} + \\lambda\\,x_{k-1}\\Big)',
    intuition:
      'Adam with weight decay applied directly to x, not through the gradient, so the decay is not rescaled by the adaptive denominator.',
    pros: ['Decay independent of the gradient scaling'],
    cons: ['λ > 0 pulls the limit toward 0: it converges only where x* = 0'],
    quantities: [
      Q_ALPHA,
      { tex: '-\\lambda x_{k-1}', key: 'info.decay', label: 'weight decay' },
      { tex: '\\hat m_k', key: 'info.m_hat', label: 'first moment' },
      Q_GRADNORM,
    ],
  },
  adamax: {
    rule: 'u_k = \\max(\\beta_2 u_{k-1}, |g_k|),\\qquad x_k = x_{k-1} - \\frac{\\alpha}{1-\\beta_1^{k}}\\,\\frac{m_k}{u_k}',
    intuition:
      'Adam with the second moment replaced by an exponentially weighted maximum of |g|: an infinity norm instead of an RMS.',
    pros: ['No ε', 'Step bounded by about α per coordinate'],
    cons: ['No convergence guarantee with a constant α'],
    quantities: [
      Q_ALPHA,
      { tex: 'm_k', key: 'info.m', label: 'first moment' },
      { tex: 'u_k', key: 'info.u', label: 'infinity norm' },
      Q_GRADNORM,
    ],
  },
  nadam: {
    rule: 'x_k = x_{k-1} - \\alpha\\,\\frac{\\bar m_k}{\\sqrt{\\hat n_k} + \\varepsilon},\\qquad \\bar m_k = (1-\\beta_1)\\,\\hat g_k + \\beta_1\\,\\hat m_k',
    intuition:
      'Adam with Nesterov momentum: the step already uses the momentum of the next iteration, a look-ahead without a second gradient.',
    pros: ['Often fewer oscillations than Adam'],
    cons: ['No convergence guarantee with a constant α'],
    quantities: [
      Q_ALPHA,
      { tex: '\\bar m_k', key: 'info.m_bar', label: 'Nesterov momentum' },
      { tex: '\\hat n_k', key: 'info.v_hat', label: 'second moment' },
      Q_GRADNORM,
    ],
  },
  amsgrad: {
    rule: '\\hat v_k = \\max(\\hat v_{k-1}, v_k),\\qquad x_k = x_{k-1} - \\alpha\\,\\frac{m_k}{\\sqrt{\\hat v_k}}',
    intuition:
      'Adam whose per-coordinate step can only shrink: it divides by the running maximum of the second moment. This repairs the convex counterexamples of Reddi et al.',
    pros: ['Non-increasing per-coordinate steps', 'Linear near a strongly convex minimizer'],
    cons: ['A large early gradient keeps the steps small for good'],
    quantities: [
      Q_ALPHA,
      { tex: 'm_k', key: 'info.m', label: 'first moment' },
      { tex: '\\hat v_k', key: 'info.v_max', label: 'running max of v' },
      Q_GRADNORM,
    ],
  },
  coordinate_descent: {
    rule: 'x_k = x_{k-1} - \\alpha_k\\,\\frac{\\partial_i f(x_{k-1})}{\\partial_{ii}^2 f(x_{k-1})}\\,e_i,\\qquad i = (k-1) \\bmod n',
    intuition:
      'Move along one coordinate axis at a time, with a 1-D Newton step, and cycle through the axes. On a quadratic, n steps are one Gauss–Seidel sweep.',
    pros: ['Each step is cheap and one-dimensional', 'Exact line minimizer on quadratics'],
    cons: ['Axis-parallel steps cannot follow a curved or diagonal valley'],
    quantities: [
      Q_ALPHA,
      { tex: 'i', key: 'info.coordinate', label: 'coordinate' },
      { tex: '\\partial_{ii}^2 f', key: 'info.curvature', label: 'curvature' },
      Q_GRADNORM,
    ],
  },
};

// ---------------------------------------------------------------------------------------
// Registration (same ids, params and metadata as the Python @register calls)
// ---------------------------------------------------------------------------------------

registerMethod<FirstOrderProblem>(
  {
    id: 'gradient_descent',
    family: 'unconstrained',
    name: 'Gradient descent',
    params: [
      param.choice('step_rule', 'backtracking', [...STEP_RULES], {
        help:
          'Fixed step lr, Armijo backtracking, strong Wolfe search, or the exact minimizer of ' +
          'the quadratic model along −∇f.',
        label: 'Step rule',
      }),
      pLr(1e-3, "Fixed step length α (step_rule='fixed' only); stable when α < 2/L."),
      P_GTOL,
      pMaxIter(5000),
    ],
    needs: ['f', 'grad'],
    order: 'linear, rate ((κ − 1)/(κ + 1))² in f with exact steps on quadratics',
    summary: 'Step downhill along the negative gradient, with the step length set by a rule.',
    references: [
      'Nocedal & Wright (2006), §3.1–3.3, Alg. 3.1 (backtracking), Alg. 3.5 (strong Wolfe), ' +
        'eq. 3.26 (exact step), eq. 3.60 (initial trial step), Theorem 3.3',
      'Cauchy (1847), C. R. Acad. Sci. Paris 25, 536–538',
    ],
  },
  gradientDescent,
  DOCS.gradient_descent,
);

registerMethod<FirstOrderProblem>(
  {
    id: 'barzilai_borwein',
    family: 'unconstrained',
    name: 'Barzilai–Borwein',
    params: [
      param.choice('variant', 'bb1', [...BB_VARIANTS], {
        help: 'BB1: α = sᵀs/sᵀy (long step); BB2: α = sᵀy/yᵀy (short step).',
        label: 'Variant',
      }),
      param.bool('nonmonotone', true, {
        help: 'Grippo–Lampariello–Lucidi nonmonotone backtracking (M = 10); off = pure BB.',
        label: 'Nonmonotone search',
      }),
      P_GTOL,
      pMaxIter(1000),
    ],
    needs: ['f', 'grad'],
    order: 'R-superlinear on 2-D quadratics, R-linear on n-D quadratics; not monotone',
    summary:
      'Gradient steps whose length mimics a secant (quasi-Newton) equation from the last step.',
    references: [
      'Barzilai & Borwein (1988), IMA J. Numer. Anal. 8, 141–148',
      'Raydan (1997), SIAM J. Optim. 7(1), 26–33, Algorithm GBB',
      'Grippo, Lampariello & Lucidi (1986), SIAM J. Numer. Anal. 23(4), 707–716',
    ],
  },
  barzilaiBorwein,
  DOCS.barzilai_borwein,
);

registerMethod<FirstOrderProblem>(
  {
    id: 'momentum',
    family: 'unconstrained',
    name: 'Heavy-ball momentum (Polyak)',
    params: [
      pLr(1e-3),
      pBeta('beta', 0.9, 'Momentum coefficient β ∈ [0, 1).', 'Momentum', '\\beta'),
      P_GTOL,
      pMaxIter(5000),
    ],
    needs: ['f', 'grad'],
    order: 'linear; rate (√κ − 1)/(√κ + 1) on quadratics with optimal lr, β',
    summary:
      'Gradient descent plus a fraction β of the previous step (a heavy ball rolling downhill).',
    references: [
      'Polyak (1964), USSR Comput. Math. Math. Phys. 4(5), 1–17',
      'Sutskever, Martens, Dahl & Hinton (2013), ICML, eq. 1–2',
    ],
  },
  momentum,
  DOCS.momentum,
);

registerMethod<FirstOrderProblem>(
  {
    id: 'nesterov',
    family: 'unconstrained',
    name: 'Nesterov accelerated gradient',
    params: [
      pLr(1e-3),
      pBeta('beta', 0.9, 'Momentum coefficient μ ∈ [0, 1).', 'Momentum', '\\mu'),
      P_GTOL,
      pMaxIter(5000),
    ],
    needs: ['f', 'grad'],
    order:
      'linear on strongly convex f for suitable constant lr, μ: rate 1 − 1/√κ in f with ' +
      'lr = 1/L, μ = (√κ − 1)/(√κ + 1); the O(1/k²) rate needs a schedule μ_k → 1 (not implemented)',
    summary: 'Momentum that takes the gradient at the look-ahead point x + μv instead of at x.',
    references: [
      'Sutskever, Martens, Dahl & Hinton (2013), ICML, eq. 3–4',
      'Nesterov (1983), Soviet Math. Dokl. 27(2), 372–376',
      'Nesterov (2004), Introductory Lectures on Convex Optimization, §2.2.1, ' +
        'constant step scheme III',
    ],
  },
  nesterov,
  DOCS.nesterov,
);

registerMethod<FirstOrderProblem>(
  {
    id: 'adagrad',
    family: 'unconstrained',
    name: 'AdaGrad',
    params: [pLr(1.0), pEps(1e-8), P_GTOL, pMaxIter(5000)],
    needs: ['f', 'grad'],
    order: 'sublinear in general (step sizes shrink like 1/√k)',
    summary:
      "Divide each coordinate's step by the root of the sum of all its past squared gradients.",
    references: ['Duchi, Hazan & Singer (2011), JMLR 12, 2121–2159 (diagonal AdaGrad)'],
  },
  adagrad,
  DOCS.adagrad,
);

registerMethod<FirstOrderProblem>(
  {
    id: 'rmsprop',
    family: 'unconstrained',
    name: 'RMSprop',
    params: [
      pLr(0.1),
      pBeta('rho', 0.9, 'Decay ρ of the running mean of squared gradients.', 'Decay', '\\rho'),
      pEps(1e-8),
      P_GTOL,
      pMaxIter(5000),
    ],
    needs: ['f', 'grad'],
    order:
      'no convergence guarantee with a constant lr; depending on lr it converges or hovers near x⋆',
    summary: 'AdaGrad with a decaying average of squared gradients, so step sizes do not vanish.',
    references: ['Tieleman & Hinton (2012), COURSERA Neural Networks for ML, Lecture 6.5'],
  },
  rmsprop,
  DOCS.rmsprop,
);

registerMethod<FirstOrderProblem>(
  {
    id: 'adadelta',
    family: 'unconstrained',
    name: 'AdaDelta',
    params: [
      pBeta('rho', 0.95, 'Decay ρ of both running means.', 'Decay', '\\rho'),
      param.float('eps', 1e-6, {
        min: 1e-12,
        max: 1e-2,
        log: true,
        help: 'ε inside both RMS terms (> 0).',
        label: 'Epsilon',
        tex: '\\varepsilon',
      }),
      P_GTOL,
      pMaxIter(5000),
    ],
    needs: ['f', 'grad'],
    order: 'no convergence guarantee; slow start (first steps ≈ √ε)',
    summary: 'RMSprop whose learning rate is the RMS of recent updates, so no lr is needed.',
    references: [
      'Zeiler (2012), ADADELTA: an adaptive learning rate method, arXiv:1212.5701, Alg. 1',
    ],
  },
  adadelta,
  DOCS.adadelta,
);

registerMethod<FirstOrderProblem>(
  {
    id: 'adam',
    family: 'unconstrained',
    name: 'Adam',
    params: [pLr(0.05), ...ADAM_BETAS, pEps(1e-8), P_GTOL, pMaxIter(5000)],
    needs: ['f', 'grad'],
    order: 'no convergence guarantee with a constant lr (Reddi et al. 2018)',
    summary: 'Momentum on the gradient and RMSprop scaling, both with bias-corrected averages.',
    references: ['Kingma & Ba (2015), Adam: a method for stochastic optimization, ICLR, Alg. 1'],
  },
  adam,
  DOCS.adam,
);

registerMethod<FirstOrderProblem>(
  {
    id: 'adamw',
    family: 'unconstrained',
    name: 'AdamW',
    params: [
      pLr(0.05),
      ...ADAM_BETAS,
      pEps(1e-8),
      param.float('weight_decay', 1e-3, {
        min: 0.0,
        max: 0.1,
        help: 'Decoupled weight decay λ: each step also subtracts λ·x.',
        label: 'Weight decay',
        tex: '\\lambda',
      }),
      P_GTOL,
      pMaxIter(5000),
    ],
    needs: ['f', 'grad'],
    order: 'no convergence guarantee with a constant lr; λ > 0 biases the limit toward 0',
    summary: 'Adam with weight decay applied directly to x instead of through the gradient.',
    references: ['Loshchilov & Hutter (2019), Decoupled weight decay regularization, ICLR, Alg. 2'],
  },
  adamw,
  DOCS.adamw,
);

registerMethod<FirstOrderProblem>(
  {
    id: 'adamax',
    family: 'unconstrained',
    name: 'AdaMax',
    params: [pLr(0.2), ...ADAM_BETAS, P_GTOL, pMaxIter(5000)],
    needs: ['f', 'grad'],
    order: 'no convergence guarantee with a constant lr',
    summary: 'Adam with the second moment replaced by an exponentially weighted max of |g|.',
    references: [
      'Kingma & Ba (2015), Adam: a method for stochastic optimization, ICLR, §7.1, Alg. 2',
    ],
  },
  adamax,
  DOCS.adamax,
);

registerMethod<FirstOrderProblem>(
  {
    id: 'nadam',
    family: 'unconstrained',
    name: 'NAdam',
    params: [pLr(0.02), ...ADAM_BETAS, pEps(1e-8), P_GTOL, pMaxIter(5000)],
    needs: ['f', 'grad'],
    order: 'no convergence guarantee with a constant lr',
    summary: "Adam with Nesterov momentum: the step already uses the next iteration's momentum.",
    references: [
      'Dozat (2016), Incorporating Nesterov momentum into Adam, ICLR Workshop; ' +
        'Stanford CS229 report, Alg. 8',
    ],
  },
  nadam,
  DOCS.nadam,
);

registerMethod<FirstOrderProblem>(
  {
    id: 'amsgrad',
    family: 'unconstrained',
    name: 'AMSGrad',
    params: [pLr(0.1), ...ADAM_BETAS, P_GTOL, pMaxIter(5000)],
    needs: ['f', 'grad'],
    order: 'linear near a strongly convex minimizer once v̂ stops growing',
    summary: 'Adam whose per-coordinate step can only shrink: divide by the running max of v.',
    references: ['Reddi, Kale & Kumar (2018), On the convergence of Adam and beyond, ICLR, Alg. 2'],
  },
  amsgrad,
  DOCS.amsgrad,
);

registerMethod<FirstOrderProblem>(
  {
    id: 'coordinate_descent',
    family: 'unconstrained',
    name: 'Cyclic coordinate descent',
    params: [P_GTOL, pMaxIter(5000)],
    needs: ['f', 'grad', 'hess'],
    order: 'linear (one Gauss–Seidel sweep per n iterations on quadratics)',
    summary: 'Minimize along one coordinate axis at a time, cycling through the axes.',
    references: [
      'Wright (2015), Coordinate descent algorithms, Math. Program. 151, 3–34, Alg. 1',
      'Nocedal & Wright (2006), §9.3',
      'Golub & Van Loan (2013), §11.2 (Gauss–Seidel)',
    ],
  },
  coordinateDescent,
  DOCS.coordinate_descent,
);
