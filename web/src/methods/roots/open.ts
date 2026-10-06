/**
 * Open root finders — TS port of `numopt.roots.open` (src/numopt/roots/open.py).
 *
 * Start from a point, no bracket, no convergence guarantee. Conventions (identical to Python):
 *
 *   - Start: `x0` (or the problem's default). Secant, Müller and inverse quadratic interpolation
 *     use auxiliary points x₀ ± h, h = max(delta, √ε·|x₀|), recorded in `info.auxiliary` of step 0.
 *   - Trace: step k holds x_k and f(x_k); its info describes how x_k was produced (tangent,
 *     chord, hyperbola, parabola, inverse parabola, cobweb). `stepSize = |x_k − x_{k−1}|`.
 *   - Converged: f(x_k) == 0, |f(x_k)| ≤ ftol, a sign change within tol, the a posteriori step
 *     test s_k·max(1, ρ/(1 − ρ)) ≤ tol confirmed by the next secant correction, a zero step with
 *     a local slope placing the root within tol, or a noise-floor sign change within tol.
 *   - Failure (never throws): non-finite values, divergence |x_k| > 10¹²·max(1, |x₀|), a cycle,
 *     a stall, a method-specific breakdown, or max_iter.
 *
 * Step.info keys are the Python ones: previous; newton tangent {point, slope}; secant chord;
 * halley hyperbola {center, alpha, beta, gamma}, derivatives; steffensen chord, slope; muller
 * points, parabola {center, a, b, c}; inverse_quadratic_interpolation points, inverse_parabola
 * {a, b, c}; fixed_point cobweb, lam; step 0 auxiliary; slope_probe [x, y] on a judged last step.
 */
import { registerMethod, param, type MethodDoc } from '../../core/registry';
import type { ParamSpec, Params, Result, RunOptions, Step, StepInfo } from '../../core/types';
import {
  Counted,
  EPS,
  copysign,
  finite,
  fmtG,
  makeResult,
  makeStep,
  nextafter,
  num,
  pyMax,
  pyRepr,
  raised,
  scalarProblem,
  signbit,
  type ScalarProblem,
} from './bracketing';

/** Divergence is declared when |x_k| > DIVERGENCE_FACTOR · max(1, |x₀|). */
export const DIVERGENCE_FACTOR = 1e12;
/** √ε, the relative floor of the auxiliary offset (Nocedal & Wright 2006, §8.1). */
export const SQRT_EPS = Math.sqrt(EPS);
/** ε^{1/3} and ε^{1/4}: central-difference steps of `numopt.core.diff`. */
const H_CENTRAL = 6.055454452393343e-6;
const H_SECOND = 0.0001220703125;

const OPEN_PARAMS: ParamSpec[] = [
  param.float('xtol', 1e-10, {
    min: 1e-15,
    max: 1e-2,
    log: true,
    help: 'Stop when the step |xₖ − xₖ₋₁| ≤ xtol + 2ε|xₖ|.',
    label: 'Step tolerance',
    tex: '|x_k - x_{k-1}| \\le',
  }),
  param.float('ftol', 0.0, {
    min: 0.0,
    max: 1e-2,
    help: 'Also stop when |f(x)| ≤ ftol (0 disables).',
    label: 'Residual tolerance',
    tex: '|f| \\le',
  }),
  param.int('max_iter', 100, {
    min: 1,
    max: 10_000,
    help: 'Iteration limit.',
    label: 'Iteration budget',
  }),
];
const DELTA_PARAM = param.float('delta', 0.1, {
  min: 1e-6,
  max: 10.0,
  log: true,
  help: 'Auxiliary starting points are placed at x₀ ± max(delta, √ε·|x₀|).',
  label: 'Auxiliary offset',
  tex: 'h',
});

const tolOf = (x: number, xtol: number) => xtol + 2.0 * EPS * Math.abs(x);
const sign1 = (v: number) => (signbit(v) ? -1.0 : 1.0);

/** The auxiliary offset h = max(delta, √ε·|x₀|). */
function offset(x0: number, delta: number): number {
  if (!(delta > 0.0 && Number.isFinite(delta)))
    throw new Error(`delta must be finite and > 0, got ${pyRepr(delta)}`);
  return pyMax(delta, SQRT_EPS * Math.abs(x0));
}

function auxMessage(points: [number, number][], h: number, delta: number, note: string): string {
  const bad = points
    .filter(([, fx]) => !finite(fx))
    .map(([x, fx]) => `f(${pyRepr(x)}) = ${pyRepr(fx)}`)
    .join(', ');
  return (
    `f is not finite at an auxiliary starting point x₀ ± h, h = max(delta, √ε|x₀|) = ` +
    `${fmtG(h, 3)} (${bad}${note}); a smaller delta (now ${fmtG(delta)}) keeps it inside the domain of f`
  );
}

/**
 * A posteriori error estimate of x_k from its last three steps: `[estimate, ρ]` with
 * ρ = max(s_k/s_{k−1}, s_{k−1}/s_{k−2}), or null without two consecutive decreases.
 */
export function stepTestEstimate(xs: readonly number[]): [number, number] | null {
  const n = xs.length;
  if (n < 4) return null;
  const sK = Math.abs(xs[n - 1] - xs[n - 2]),
    s1 = Math.abs(xs[n - 2] - xs[n - 3]),
    s2 = Math.abs(xs[n - 3] - xs[n - 4]);
  if (!(sK < s1 && s1 < s2)) return null;
  const rho = pyMax(sK / s1, s1 / s2);
  return [sK * pyMax(1.0, rho / (1.0 - rho)), rho];
}

/** Error estimate c/(1 − c/s) of x_k from its next correction c and last step s (∞ if c ≥ s). */
function correctionError(c: number, s: number): number {
  if (!(c < s)) return Infinity;
  return c / (1.0 - c / s);
}

/** Trace bookkeeping and the shared stopping / divergence / cycle tests (Python `_Run`). */
class Run {
  readonly xs: number[];
  readonly fs: number[];
  readonly trace: Step[] = [];
  readonly counters: { grad?: Counted; hess?: Counted } = {};
  readonly extra: Record<string, unknown> = {};

  readonly method: string;
  readonly f: Counted;
  readonly x0: number;
  readonly xtol: number;
  readonly ftol: number;
  readonly maxIter: number;

  constructor(
    method: string,
    f: Counted,
    x0: number,
    f0: number,
    xtol: number,
    ftol: number,
    maxIter: number,
  ) {
    this.method = method;
    this.f = f;
    this.x0 = x0;
    this.xtol = xtol;
    this.ftol = ftol;
    this.maxIter = maxIter;
    this.xs = [x0];
    this.fs = [f0];
  }

  get k(): number {
    return this.xs.length - 1;
  }

  result(converged: boolean, message: string): Result {
    return makeResult(
      this.method,
      this.xs[this.xs.length - 1],
      this.fs[this.fs.length - 1],
      converged,
      message,
      this.k,
      this.f.n,
      this.trace,
      { nGev: this.counters.grad?.n ?? 0, nHev: this.counters.hess?.n ?? 0, extra: this.extra },
    );
  }

  /** Record step 0; return a finished Result if x₀ already settles the problem. */
  start(info: StepInfo | null = null): Result | null {
    const x = this.xs[0],
      fx = this.fs[0];
    this.trace.push(makeStep(0, x, fx, null, info ?? {}));
    if (!finite(x, fx))
      return this.result(false, `f(x₀) is not finite at x₀ = ${pyRepr(x)}${raised(this.f)}`);
    if (fx === 0.0) return this.result(true, 'f(x) is exactly zero');
    if (Math.abs(fx) <= this.ftol)
      return this.result(true, `|f(x)| = ${fmtG(Math.abs(fx), 3)} ≤ ftol`);
    return null;
  }

  atLimit(): Result | null {
    if (this.k >= this.maxIter) return this.result(false, `reached max_iter=${this.maxIter}`);
    return null;
  }

  /** Record step k + 1 and apply the shared tests; return a Result to stop. */
  step(
    xNew: number,
    fNew: number,
    stepInfo: StepInfo,
    { stepTestOk = true, slope = null }: { stepTestOk?: boolean; slope?: number | null } = {},
  ): Result | null {
    const xOld = this.xs[this.xs.length - 1],
      fOld = this.fs[this.fs.length - 1];
    const info: StepInfo = { previous: xOld, ...stepInfo };
    const step = Math.abs(xNew - xOld);
    this.xs.push(xNew);
    this.fs.push(fNew);
    this.trace.push(makeStep(this.k, xNew, fNew, step, info));
    if (!finite(xNew, fNew))
      return this.result(
        false,
        `diverged: non-finite value (x = ${pyRepr(xNew)}, f = ${pyRepr(fNew)}${raised(this.f)})`,
      );
    if (fNew === 0.0) return this.result(true, 'f(x) is exactly zero');
    if (Math.abs(fNew) <= this.ftol)
      return this.result(true, `|f(x)| = ${fmtG(Math.abs(fNew), 3)} ≤ ftol`);
    const tol = tolOf(xNew, this.xtol);
    if (step <= tol && sign1(fNew) !== sign1(fOld))
      return this.result(
        true,
        `f changes sign between xₖ₋₁ and xₖ, |xₖ − xₖ₋₁| = ${fmtG(step, 3)} ≤ tol ` +
          `= ${fmtG(tol, 3)}: a root lies within tol of xₖ`,
      );
    if (step === 0.0) return this.judgeZeroStep(xNew, fNew, tol, slope, info);
    const test = stepTestEstimate(this.xs);
    if (stepTestOk && test !== null && test[0] <= tol) {
      const [estimate, rho] = test;
      // The next secant correction through the two newest iterates (a local slope).
      const correction = fNew !== fOld ? Math.abs(fNew) * (step / Math.abs(fNew - fOld)) : Infinity;
      const error = correctionError(correction, step);
      if (error <= tol)
        return this.result(
          true,
          `step test: |xₖ − xₖ₋₁|·max(1, ρ/(1−ρ)) = ${fmtG(estimate, 3)} ≤ tol = ` +
            `${fmtG(tol, 3)} (ρ = ${fmtG(rho, 3)}) and the next ` +
            `secant correction c = ${fmtG(correction, 3)} gives ` +
            `c/(1 − c/|xₖ − xₖ₋₁|) = ${fmtG(error, 3)} ≤ tol`,
        );
    }
    const bound = DIVERGENCE_FACTOR * pyMax(1.0, Math.abs(this.x0));
    if (Math.abs(xNew) > bound)
      return this.result(false, `diverged: |x| = ${fmtG(Math.abs(xNew), 3)} > ${fmtG(bound, 3)}`);
    const k = this.k;
    for (let j = 0; j < k - 1; j++) {
      // earlier iterates x_0 … x_{k−2}: period k − j ≥ 2
      if (Math.abs(xNew - this.xs[j]) <= tol && Math.abs(fNew) >= Math.abs(this.fs[j])) {
        const done = this.judgeNoiseFloor(j, tol);
        if (done !== null) return done;
        this.extra.cycle_period = k - j;
        return this.result(
          false,
          `cycle of period ${k - j} detected: x_${k} returned to x_${j} = ${fmtG(this.xs[j], 6)} ` +
            'with no decrease in |f|',
        );
      }
    }
    return null;
  }

  /** x_k returned within tol to x_j with no decrease in |f|: noise floor or cycle? */
  private judgeNoiseFloor(j: number, tol: number): Result | null {
    const k = this.k,
      xK = this.xs[k],
      fK = this.fs[k];
    const signK = sign1(fK);
    let rRoot = Infinity;
    for (let i = 0; i < this.xs.length; i++)
      if (sign1(this.fs[i]) !== signK) rRoot = Math.min(rRoot, Math.abs(this.xs[i] - xK));
    if (rRoot <= tol)
      return this.result(
        true,
        `noise floor: x_${k} returned within tol = ${fmtG(tol, 3)} of x_${j} with no decrease in ` +
          `|f|, and f changes sign between x_k and an iterate ${fmtG(rRoot, 3)} away: a root ` +
          'lies within tol of x_k',
      );
    let spread = -Infinity;
    for (const x of this.xs.slice(j)) spread = pyMax(spread, Math.abs(x - xK));
    if (spread <= tol) {
      if (this.signProbe(this.localSlope()))
        return this.result(
          true,
          `noise floor: x_${j} … x_${k} all lie within tol = ${fmtG(tol, 3)} of x_k with no ` +
            'decrease in |f|, and f changes sign between x_k and x_k ± tol',
        );
      return this.result(
        false,
        `stagnated: x_${j} … x_${k} all lie within tol = ${fmtG(tol, 3)} of x_k but |f| no ` +
          `longer decreases (|f(x_k)| = ${fmtG(Math.abs(fK), 3)}, likely the rounding level of f) ` +
          'and f does not change sign within tol of x_k',
      );
    }
    if (rRoot <= spread)
      return this.result(
        false,
        `no progress: x_${k} returned within tol of x_${j} with no decrease in |f|; f ` +
          `changes sign among x_${j} … x_${k}, so a root lies within ${fmtG(rRoot, 3)} of x_k, ` +
          `but that is > tol = ${fmtG(tol, 3)} (jitter at the noise floor of f, or iterates ` +
          'oscillating around the root)',
      );
    return null;
  }

  /**
   * x_k = x_{k−1}: the correction rounded to zero, so the run ends here. Converged when a
   * local slope places the root within tol, or f changes sign between x_k and x_k ± tol.
   * Without the method's own slope, a probe at x_k + tol gives one (`slope_probe`); when its
   * correction alone would accept, a second probe at x_k − tol (`slope_probe_back`) must show
   * a sign change or a correction ≤ tol too (Python `_judge_zero_step`).
   */
  private judgeZeroStep(
    x: number,
    fx: number,
    tol: number,
    slope: number | null,
    info: StepInfo,
  ): Result {
    const m = slope === null ? NaN : slope;
    if (finite(m) && m !== 0.0) {
      const correction = Math.abs(fx / m);
      if (correction <= tol)
        return this.result(
          true,
          `xₖ = xₖ₋₁ and the Newton correction |f(x)/slope| = ${fmtG(correction, 3)} ≤ tol`,
        );
      return this.stalled(x, fx, correction);
    }
    if (this.signProbe(null))
      return this.result(true, 'xₖ = xₖ₋₁ and f changes sign between xₖ and xₖ + tol');
    let correction = Run.probeCorrection(x, fx, info.slope_probe as [number, number]);
    if (correction > tol) return this.stalled(x, fx, correction);
    if (this.signProbe(null, -1.0, 'slope_probe_back'))
      return this.result(true, 'xₖ = xₖ₋₁ and f changes sign between xₖ and xₖ − tol');
    correction = pyMax(
      correction,
      Run.probeCorrection(x, fx, info.slope_probe_back as [number, number]),
    );
    if (correction <= tol)
      return this.result(
        true,
        `xₖ = xₖ₋₁ and the Newton corrections |f(x)/slope| over [xₖ − tol, xₖ] and ` +
          `[xₖ, xₖ + tol] are both ≤ ${fmtG(correction, 3)} ≤ tol`,
      );
    return this.stalled(x, fx, correction);
  }

  /** |f(x_k)/m| with the difference quotient m over [x_k, x_p]; ∞ without a slope. */
  private static probeCorrection(x: number, fx: number, probe: [number, number]): number {
    const [xP, fP] = probe;
    const m = finite(fP) ? (fP - fx) / (xP - x) : NaN;
    return finite(m) && m !== 0.0 ? Math.abs(fx / m) : Infinity;
  }

  private stalled(x: number, fx: number, correction: number): Result {
    return this.result(
      false,
      `stalled: xₖ = xₖ₋₁ = ${fmtG(x, 6)} but |f(x)| = ${fmtG(Math.abs(fx), 3)} and the Newton ` +
        `correction |f(x)/slope| = ${fmtG(correction, 3)} > tol`,
    );
  }

  /**
   * One extra evaluation at x_k ± tol: true when f changes sign across it. The direction is
   * `toward` (±1) when given, else toward the root `slope` predicts (+tol without one); the
   * pair is stored in the last step's `info[key]`.
   */
  signProbe(slope: number | null, toward: number | null = null, key = 'slope_probe'): boolean {
    const xK = this.xs[this.xs.length - 1],
      fK = this.fs[this.fs.length - 1];
    const tol = tolOf(xK, this.xtol);
    // Python: `slope and finite(slope)` — NaN is truthy there, then not finite.
    const dir =
      toward ?? (slope !== null && slope !== 0 && finite(slope) ? -copysign(1.0, fK * slope) : 1.0);
    let xP = xK + dir * tol;
    if (Math.abs(xP - xK) > tol) xP = nextafter(xP, xK);
    const fP = this.f.call(xP);
    this.trace[this.trace.length - 1].info[key] = [xP, fP];
    return finite(fP) && sign1(fP) !== sign1(fK);
  }

  /** The secant slope through x_(k−1), x_k, or null if it does not exist. */
  localSlope(): number | null {
    const n = this.xs.length;
    if (n < 2 || this.xs[n - 1] === this.xs[n - 2]) return null;
    return (this.fs[n - 1] - this.fs[n - 2]) / (this.xs[n - 1] - this.xs[n - 2]);
  }

  /** Stop with converged = false, unless the last step was ≤ tol and a sign probe proves a root. */
  breakdown(message: string): Result {
    const n = this.xs.length;
    if (n >= 2 && finite(this.xs[n - 1], this.fs[n - 1])) {
      const xK = this.xs[n - 1];
      if (
        Math.abs(xK - this.xs[n - 2]) <= tolOf(xK, this.xtol) &&
        this.signProbe(this.localSlope())
      )
        return this.result(
          true,
          `${message}; but the last step was ≤ tol and f changes sign between x_k ` +
            'and x_k ± tol: a root lies within tol of x_k',
        );
    }
    return this.result(false, message);
  }
}

/** f′ from the problem, or a central difference whose f evaluations count in nFev. */
function firstDerivative(prob: ScalarProblem, f: Counted, run: Run): (x: number) => number {
  if (prob.grad) {
    const g = new Counted(prob.grad);
    run.counters.grad = g;
    return (x) => g.call(x);
  }
  run.extra.derivatives = 'finite_difference';
  return (x) => {
    const h = H_CENTRAL * pyMax(1.0, Math.abs(x));
    return (f.call(x + h) - f.call(x - h)) / (2.0 * h);
  };
}

function secondDerivative(prob: ScalarProblem, f: Counted, run: Run): (x: number) => number {
  if (prob.hess) {
    const h = new Counted(prob.hess);
    run.counters.hess = h;
    return (x) => h.call(x);
  }
  run.extra.derivatives = 'finite_difference';
  return (x) => {
    const h = H_SECOND * pyMax(1.0, Math.abs(x));
    return (f.call(x + h) - 2.0 * f.call(x) + f.call(x - h)) / (h * h);
  };
}

interface Setup {
  prob: ScalarProblem;
  f: Counted;
  run: Run;
}

function setup(method: string, problem: unknown, o: RunOptions & Params): Setup {
  const prob = scalarProblem(problem);
  let x0: unknown = o.x0 ?? prob.x0;
  if (x0 === null || x0 === undefined)
    throw new Error(`${prob.id}: no starting point given and the problem has no default x0`);
  if (Array.isArray(x0)) x0 = x0[0];
  const x = Number(x0);
  const f = new Counted(prob.f);
  const fx = f.call(x);
  const run = new Run(method, f, x, fx, num(o.xtol, 1e-10), num(o.ftol, 0.0), num(o.max_iter, 100));
  return { prob, f, run };
}

// ---------------------------------------------------------------------------------------
// Methods
// ---------------------------------------------------------------------------------------

/** Newton's method, Burden & Faires Alg. 2.3: x_{k+1} = x_k − f(x_k)/f′(x_k). */
export function newton(problem: unknown, o: RunOptions & Params): Result {
  const { prob, f, run } = setup('newton', problem, o);
  const fprime = firstDerivative(prob, f, run);
  let done = run.start();
  if (done) return done;
  for (;;) {
    if ((done = run.atLimit())) return done;
    const x = run.xs[run.xs.length - 1],
      fx = run.fs[run.fs.length - 1];
    const d = fprime(x);
    if (!finite(d)) return run.breakdown(`f'(x) is not finite at x = ${pyRepr(x)}`);
    if (d === 0.0)
      return run.breakdown(`f'(x) = 0 at x = ${fmtG(x, 6)}: the tangent is horizontal`);
    const xNew = x - fx / d;
    const fNew = f.call(xNew);
    const info = { tangent: { point: [x, fx], slope: d } };
    if ((done = run.step(xNew, fNew, info, { slope: d }))) return done;
  }
}

/** Secant method, Burden & Faires Alg. 2.4 (x₋₁ = x₀ + h is auxiliary). */
export function secant(problem: unknown, o: RunOptions & Params): Result {
  const { f, run } = setup('secant', problem, o);
  const delta = num(o.delta, 0.1);
  const h = offset(run.x0, delta);
  const xAux = run.x0 + h;
  const fAux = f.call(xAux);
  const note = raised(f);
  let done = run.start({ auxiliary: [[xAux, fAux]] });
  if (done) return done;
  if (!finite(fAux)) return run.breakdown(auxMessage([[xAux, fAux]], h, delta, note));
  let xPrev = xAux,
    fPrev = fAux;
  for (;;) {
    if ((done = run.atLimit())) return done;
    const x = run.xs[run.xs.length - 1],
      fx = run.fs[run.fs.length - 1];
    if (fx === fPrev) return run.breakdown(`horizontal chord: f(xₖ) = f(xₖ₋₁) = ${fmtG(fx, 6)}`);
    const xNew = x - (fx * (x - xPrev)) / (fx - fPrev);
    const fNew = f.call(xNew);
    const info = {
      chord: [
        [xPrev, fPrev],
        [x, fx],
      ],
    };
    if ((done = run.step(xNew, fNew, info))) return done;
    xPrev = x;
    fPrev = fx;
  }
}

/** Halley's method: x_{k+1} = x_k − 2ff′/(2f′² − ff″), the zero of the osculating hyperbola. */
export function halley(problem: unknown, o: RunOptions & Params): Result {
  const { prob, f, run } = setup('halley', problem, o);
  const fprime = firstDerivative(prob, f, run);
  const fsecond = secondDerivative(prob, f, run);
  let done = run.start();
  if (done) return done;
  for (;;) {
    if ((done = run.atLimit())) return done;
    const x = run.xs[run.xs.length - 1],
      fx = run.fs[run.fs.length - 1];
    const d1 = fprime(x);
    if (!finite(d1)) return run.breakdown(`f'(x) is not finite at x = ${pyRepr(x)}`);
    if (d1 === 0.0)
      return run.breakdown(`f'(x) = 0 at x = ${fmtG(x, 6)}: Halley's step degenerates`);
    const d2 = fsecond(x);
    const den = 2.0 * d1 * d1 - fx * d2;
    if (!finite(d2, den))
      return run.breakdown(`f''(x) or 2f'² − ff'' is not finite at x = ${pyRepr(x)}`);
    if (den === 0.0)
      return run.breakdown(`2f'² − f·f'' = 0 at x = ${fmtG(x, 6)}: the hyperbola has no zero`);
    const alpha = (2.0 * fx * d1) / den;
    const xNew = x - alpha;
    const fNew = f.call(xNew);
    const info = {
      hyperbola: { center: x, alpha, beta: -d2 / den, gamma: (2.0 * d1) / den },
      derivatives: [fx, d1, d2],
    };
    if ((done = run.step(xNew, fNew, info, { slope: d1 }))) return done;
  }
}

/** Steffensen's method: x_{k+1} = x_k − f(x_k)²/(f(x_k + f(x_k)) − f(x_k)). */
export function steffensen(problem: unknown, o: RunOptions & Params): Result {
  const { f, run } = setup('steffensen', problem, o);
  const xtol = run.xtol;
  let done = run.start();
  if (done) return done;
  let lastSlope: number | null = null;
  for (;;) {
    if ((done = run.atLimit())) return done;
    const x = run.xs[run.xs.length - 1],
      fx = run.fs[run.fs.length - 1];
    const z = x + fx;
    if (z === x) return steffensenEnd(run, lastSlope, 'x + f(x) rounds to x');
    const fz = f.call(z);
    if (!finite(z, fz)) return run.breakdown(`f(x + f(x)) is not finite (x + f(x) = ${pyRepr(z)})`);
    const den = fz - fx;
    if (den === 0.0)
      return steffensenEnd(run, lastSlope, 'f(x + f(x)) = f(x) (zero slope estimate)');
    // NOTE: the slope over the probe actually taken, z − x (exact), not f(x).
    const hz = z - x;
    // NOTE: f·(h/den), not f²/den (under/overflow of f²).
    const xNew = x - fx * (hz / den);
    const fNew = f.call(xNew);
    lastSlope = den / hz;
    const info = {
      chord: [
        [x, fx],
        [z, fz],
      ],
      slope: lastSlope,
    };
    const probeOk = Math.abs(hz) <= tolOf(xNew, xtol);
    const slope = probeOk ? lastSlope : null;
    if ((done = run.step(xNew, fNew, info, { stepTestOk: probeOk, slope }))) return done;
  }
}

/** End a Steffensen run that cannot form a new slope at x_k (`why` says why). */
function steffensenEnd(run: Run, lastSlope: number | null, why: string): Result {
  const xs = run.xs,
    fx = run.fs[run.fs.length - 1];
  const x = xs[xs.length - 1];
  const tol = tolOf(x, run.xtol);
  if (run.signProbe(lastSlope))
    return run.result(true, `${why} at x_k, and f changes sign between x_k and x_k ± tol`);
  const sK = xs.length >= 2 ? Math.abs(xs[xs.length - 1] - xs[xs.length - 2]) : null;
  const test = stepTestEstimate(xs);
  const est = test !== null ? test[0] : null;
  const c = lastSlope ? Math.abs(fx / lastSlope) : Infinity;
  const error = sK ? correctionError(c, sK) : Infinity;
  if (est !== null && est <= tol && error <= tol)
    return run.result(
      true,
      `${why} at xₖ; the step test |xₖ − xₖ₋₁|·max(1, ρ/(1−ρ)) = ${fmtG(est, 3)} ≤ tol = ` +
        `${fmtG(tol, 3)} and the correction c = |f(x)/slope| = ${fmtG(c, 3)} gives ` +
        `c/(1 − c/|xₖ − xₖ₋₁|) = ${fmtG(error, 3)} ≤ tol`,
    );
  return run.result(
    false,
    `${why} at x = ${fmtG(x, 6)} (|f(x)| = ${fmtG(Math.abs(fx), 3)}): no new slope; f keeps its sign at ` +
      'x ± tol, and the last steps and slope estimate do not place x within tol',
  );
}

/** Müller's method in real arithmetic, Burden & Faires Alg. 2.8. */
export function muller(problem: unknown, o: RunOptions & Params): Result {
  const { f, run } = setup('muller', problem, o);
  const delta = num(o.delta, 0.1);
  const h = offset(run.x0, delta);
  let p0 = run.x0 - h,
    p1 = run.x0 + h;
  let f0 = f.call(p0);
  let note = raised(f);
  let f1 = f.call(p1);
  note = note || raised(f);
  let done = run.start({
    auxiliary: [
      [p0, f0],
      [p1, f1],
    ],
  });
  if (done) return done;
  if (!finite(f0, f1))
    return run.breakdown(
      auxMessage(
        [
          [p0, f0],
          [p1, f1],
        ],
        h,
        delta,
        note,
      ),
    );
  for (;;) {
    if ((done = run.atLimit())) return done;
    const p2 = run.xs[run.xs.length - 1],
      f2 = run.fs[run.fs.length - 1];
    const h1 = p1 - p0,
      h2 = p2 - p1;
    if (h1 === 0.0 || h2 === 0.0 || h1 + h2 === 0.0)
      return run.breakdown('two interpolation points coincide: the parabola is undefined');
    const d1 = (f1 - f0) / h1,
      d2 = (f2 - f1) / h2;
    const a = (d2 - d1) / (h2 + h1);
    const b = d2 + h2 * a;
    const disc = b * b - 4.0 * f2 * a;
    if (disc < 0.0)
      return run.breakdown(
        'the parabola through the last three points has no real root ' +
          `(b² − 4ac = ${fmtG(disc, 3)} < 0); real-arithmetic Müller stops here`,
      );
    const root = Math.sqrt(disc);
    const e = Math.abs(b - root) < Math.abs(b + root) ? b + root : b - root;
    if (e === 0.0) return run.breakdown('the parabola is flat (b = 0 and b² − 4ac = 0)');
    const xNew = p2 - (2.0 * f2) / e;
    const fNew = f.call(xNew);
    const info = {
      points: [
        [p0, f0],
        [p1, f1],
        [p2, f2],
      ],
      parabola: { center: p2, a, b, c: f2 },
    };
    if ((done = run.step(xNew, fNew, info))) return done;
    p0 = p1;
    f0 = f1;
    p1 = p2;
    f1 = f2;
  }
}

/** Inverse quadratic interpolation (Brent's interpolation step, unguarded). */
export function inverseQuadraticInterpolation(problem: unknown, o: RunOptions & Params): Result {
  const { f, run } = setup('inverse_quadratic_interpolation', problem, o);
  const delta = num(o.delta, 0.1);
  const h = offset(run.x0, delta);
  let xa = run.x0 - h,
    xb = run.x0 + h;
  let fa = f.call(xa);
  let note = raised(f);
  let fb = f.call(xb);
  note = note || raised(f);
  let done = run.start({
    auxiliary: [
      [xa, fa],
      [xb, fb],
    ],
  });
  if (done) return done;
  if (!finite(fa, fb))
    return run.breakdown(
      auxMessage(
        [
          [xa, fa],
          [xb, fb],
        ],
        h,
        delta,
        note,
      ),
    );
  for (;;) {
    if ((done = run.atLimit())) return done;
    const xc = run.xs[run.xs.length - 1],
      fc = run.fs[run.fs.length - 1];
    if (fa === fb || fa === fc || fb === fc)
      return run.breakdown('two interpolation points have equal f values');
    const da = (fa - fb) * (fa - fc),
      db = (fb - fa) * (fb - fc),
      dc = (fc - fa) * (fc - fb);
    if (da === 0.0 || db === 0.0 || dc === 0.0)
      return run.breakdown('the f values are too close: the interpolation weights underflow');
    const xNew = (xa * fb * fc) / da + (xb * fa * fc) / db + (xc * fa * fb) / dc;
    const fNew = f.call(xNew);
    // Coefficients of x(y) = a y² + b y + c from divided differences in y (display only).
    const ddAb = (xb - xa) / (fb - fa);
    const ddBc = (xc - xb) / (fc - fb);
    const a2 = (ddBc - ddAb) / (fc - fa);
    const info = {
      points: [
        [xa, fa],
        [xb, fb],
        [xc, fc],
      ],
      inverse_parabola: {
        a: a2,
        b: ddAb - a2 * (fa + fb),
        c: xa - ddAb * fa + a2 * fa * fb,
      },
    };
    if ((done = run.step(xNew, fNew, info))) return done;
    xa = xb;
    fa = fb;
    xb = xc;
    fb = fc;
  }
}

/** Fixed-point iteration x_{k+1} = g(x_k), g(x) = x − λ f(x) (B&F Alg. 2.2). */
export function fixedPoint(problem: unknown, o: RunOptions & Params): Result {
  const { f, run } = setup('fixed_point', problem, o);
  const lam = num(o.lam, 0.1);
  let done = run.start();
  if (done) return done;
  for (;;) {
    if ((done = run.atLimit())) return done;
    const x = run.xs[run.xs.length - 1],
      fx = run.fs[run.fs.length - 1];
    const xNew = x - lam * fx;
    const fNew = f.call(xNew);
    const info = {
      cobweb: [
        [x, x],
        [x, xNew],
        [xNew, xNew],
      ],
      lam,
    };
    if ((done = run.step(xNew, fNew, info))) return done;
  }
}

// ---------------------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------------------

const DOCS: Record<string, MethodDoc> = {
  newton: {
    rule: "x_{k+1} = x_k - \\frac{f(x_k)}{f'(x_k)}",
    intuition:
      'Replace f by its tangent at x_k and jump to where the tangent crosses zero. Near a simple root the number of correct digits doubles each step; far from it the tangent can send the iterate anywhere.',
    order: 'quadratic (simple root); linear at a multiple root',
    pros: ['Quadratic convergence near a simple root', 'One f and one f′ per step'],
    cons: ['Needs f′', 'Diverges, cycles or stops on f′ = 0 from a poor start'],
    quantities: [{ tex: '|x_k - x_{k-1}|', key: 'stepSize' }],
  },
  secant: {
    rule: 'x_{k+1} = x_k - f(x_k)\\,\\frac{x_k - x_{k-1}}{f(x_k) - f(x_{k-1})}',
    intuition:
      'Newton with the tangent replaced by the chord through the last two iterates. No derivative, and almost as fast: the order is the golden ratio φ ≈ 1.618.',
    order: 'superlinear, φ ≈ 1.618',
    pros: ['No derivative', 'One evaluation per step'],
    cons: ['No bracket: can diverge', 'Breaks down on a horizontal chord'],
  },
  halley: {
    rule: "x_{k+1} = x_k - \\frac{2 f f'}{2 f'^2 - f f''}",
    intuition:
      'Fit the hyperbola that matches f, f′ and f″ at x_k and jump to its zero. One more derivative than Newton buys one more order: the digits triple each step.',
    order: 'cubic (simple root)',
    pros: ['Cubic convergence', 'Larger basin than Newton on many functions'],
    cons: ['Needs f′ and f″'],
    quantities: [{ tex: '|x_k - x_{k-1}|', key: 'stepSize' }],
  },
  steffensen: {
    rule: 'x_{k+1} = x_k - \\frac{f(x_k)^2}{f\\bigl(x_k + f(x_k)\\bigr) - f(x_k)}',
    intuition:
      'Newton with f′ estimated by the slope between x_k and x_k + f(x_k). Quadratic without a derivative — but the probe length is f(x_k), so the method mixes units of f and x.',
    order: 'quadratic (two evaluations per step)',
    pros: ['Quadratic without derivatives'],
    cons: ['Not scale invariant: fails when |f| is large'],
    quantities: [{ tex: "\\hat f'(x_{k-1})", key: 'info.slope' }],
  },
  muller: {
    rule: 'x_{k+1} = x_k - \\frac{2c}{b \\pm \\sqrt{b^2 - 4ac}}',
    intuition:
      'Fit a parabola through the last three points and move to its root nearer to x_k. Real arithmetic here: when the parabola misses the axis the method stops.',
    order: 'superlinear, ≈ 1.84',
    pros: ['No derivative; order ≈ 1.84'],
    cons: ['Needs complex arithmetic when b² − 4ac < 0'],
  },
  inverse_quadratic_interpolation: {
    rule: 'x_{k+1} = \\sum_{i} x_i \\prod_{j \\ne i} \\frac{f_j}{f_j - f_i}',
    intuition:
      'Fit x as a quadratic function of y through the last three points and evaluate it at y = 0. This is the step Brent takes when it is safe; here it is unguarded.',
    order: 'superlinear, ≈ 1.84',
    pros: ['No square root, no derivative'],
    cons: ['Breaks down when two values of f coincide; no safeguard'],
  },
  fixed_point: {
    rule: 'x_{k+1} = g(x_k) = x_k - \\lambda f(x_k)',
    intuition:
      'A root of f is a fixed point of g. The iteration contracts near the root when |g′(x⋆)| = |1 − λ f′(x⋆)| < 1 — monotonically if g′ > 0, alternating if g′ < 0.',
    order: 'linear, rate |1 − λ f′(x⋆)|',
    pros: ['One evaluation per step; the simplest iteration'],
    cons: ['Linear at best; diverges when |1 − λ f′| > 1'],
    quantities: [{ tex: '\\lambda', key: 'info.lam' }],
  },
};

function reg(
  meta: {
    id: string;
    name: string;
    order: string;
    summary: string;
    references: string[];
    needs: string[];
    params: ParamSpec[];
  },
  fn: (p: unknown, o: RunOptions & Params) => Result,
) {
  registerMethod({ ...meta, family: 'roots' }, fn, DOCS[meta.id]);
}

reg(
  {
    id: 'newton',
    name: 'Newton–Raphson',
    params: OPEN_PARAMS,
    needs: ['f', 'grad', 'x0'],
    order: 'quadratic (simple root); linear at a multiple root',
    summary: 'Follow the tangent line at x_k down to where it crosses zero.',
    references: ['Burden & Faires, Numerical Analysis (10th ed.), Alg. 2.3'],
  },
  newton,
);
reg(
  {
    id: 'secant',
    name: 'Secant',
    params: [...OPEN_PARAMS, DELTA_PARAM],
    needs: ['f', 'x0'],
    order: 'superlinear (φ ≈ 1.618)',
    summary: "Newton's method with the tangent replaced by the chord through the last two points.",
    references: ['Burden & Faires, Numerical Analysis (10th ed.), Alg. 2.4'],
  },
  secant,
);
reg(
  {
    id: 'halley',
    name: 'Halley',
    params: OPEN_PARAMS,
    needs: ['f', 'grad', 'hess', 'x0'],
    order: 'cubic (simple root)',
    summary: 'Fit the hyperbola that matches f, f′ and f″ at x_k and jump to its zero.',
    references: [
      'Halley (1694); Scavo & Thoo (1995), Amer. Math. Monthly 102(5), 417–426',
      'Press et al., Numerical Recipes (3rd ed.), §9.4, eq. (9.4.7)',
    ],
  },
  halley,
);
reg(
  {
    id: 'steffensen',
    name: 'Steffensen',
    params: OPEN_PARAMS,
    needs: ['f', 'x0'],
    order: 'quadratic (two f evaluations per step)',
    summary: "Newton's method with f′ estimated from the slope between x and x + f(x).",
    references: [
      'Steffensen (1933), Skand. Aktuarietidskr. 16, 64–72',
      'Burden & Faires, Numerical Analysis (10th ed.), Alg. 2.6 with g(x) = x + f(x)',
    ],
  },
  steffensen,
);
reg(
  {
    id: 'muller',
    name: 'Müller',
    params: [...OPEN_PARAMS, DELTA_PARAM],
    needs: ['f', 'x0'],
    order: 'superlinear (≈ 1.84)',
    summary: 'Fit a parabola through the last three points and move to its nearer root.',
    references: ['Burden & Faires, Numerical Analysis (10th ed.), Alg. 2.8'],
  },
  muller,
);
reg(
  {
    id: 'inverse_quadratic_interpolation',
    name: 'Inverse quadratic interpolation',
    params: [...OPEN_PARAMS, DELTA_PARAM],
    needs: ['f', 'x0'],
    order: 'superlinear (≈ 1.84)',
    summary:
      'Fit x as a quadratic function of y through the last three points; evaluate it at y = 0.',
    references: [
      'Brent (1973), Algorithms for Minimization without Derivatives, §4.3',
      'Süli & Mayers, An Introduction to Numerical Analysis (2003), §1.6',
    ],
  },
  inverseQuadraticInterpolation,
);
reg(
  {
    id: 'fixed_point',
    name: 'Fixed-point iteration',
    params: [
      ...OPEN_PARAMS,
      param.float('lam', 0.1, {
        min: -5.0,
        max: 5.0,
        help: 'λ in g(x) = x − λ f(x); converges near r when 0 < λ f′(r) < 2.',
        label: 'Relaxation',
        tex: '\\lambda',
      }),
    ],
    needs: ['f', 'x0'],
    order: 'linear (rate |1 − λ f′(r)|)',
    summary: 'Iterate x ← g(x) = x − λ f(x); a root of f is a fixed point of g.',
    references: ['Burden & Faires, Numerical Analysis (10th ed.), Alg. 2.2 and Thm. 2.4'],
  },
  fixedPoint,
);

/** Parity fixtures of the Python module (method id, problem id, params) — for tests. */
export const FIXTURE_CASES: [string, string, Record<string, unknown>][] = [
  ['newton', 'cubic', {}],
  ['newton', 'newton_cycle', {}],
  ['secant', 'kepler', {}],
  ['halley', 'wilkinson5', {}],
  ['steffensen', 'sqrt2', {}],
  ['muller', 'cos_minus_x', {}],
  ['inverse_quadratic_interpolation', 'cubic', {}],
  ['fixed_point', 'cos_minus_x', { lam: -1.0 }],
];
