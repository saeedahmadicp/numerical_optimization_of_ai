/**
 * Globalized Newton without a line search — TS port of `numopt.unconstrained.regularized_newton`
 * (src/numopt/unconstrained/regularized_newton.py): adaptive cubic regularization (ARC) and
 * gradient-regularized Newton.
 *
 *   arc         Cartis, Gould & Toint (2011a), Alg. 2.1: s_k is the global minimizer of
 *               m_k(s) = f_k + g_kᵀs + ½sᵀB_k s + (σ_k/3)‖s‖³ (CGT Thm. 3.1: (B + λI)s = −g,
 *               λ = σ‖s‖, B + λI ⪰ 0), ρ_k = actual/predicted decides acceptance (ρ ≥ η₁) and
 *               σ: max(min(σ, ‖g‖), ε) if ρ > η₂, σ if η₁ ≤ ρ ≤ η₂, γσ otherwise.
 *   reg_newton  x_{k+1} = x_k − (B_k + λ_k I)⁻¹g_k with λ_k = √(H‖g_k‖) (Mishchenko 2023, Alg. 1),
 *               H adapted by doubling (AdaN, Alg. 2), or λ = 4ʲH‖g‖^α (Doikov–Mishchenko–Nesterov
 *               2024, Alg. 2).
 *
 * Stopping test (both): ‖∇f(x_k)‖∞ ≤ gtol, converged only if ∇²f(x_k) has no eigenvalue below
 * −tol (tol = n·ε·|λ|_max for an analytic ∇²f). Failures: a saddle point or maximizer, non-finite
 * values, max_iter; arc: a model step predicting no decrease, or a rejected step below the
 * rounding level of x; reg_newton: B + λI singular (fixed) or 100 failed trials. The rounding
 * allowance δ = 10³ε·max(|f(x)|, |f(x₊)|) enters ρ (arc) and AdaN's decrease test.
 *
 * Trace: arc has one Step per iteration, accepted or rejected (info keys center, grad, H, sigma,
 * new_sigma, step, step_norm, lambda, lambda_min, hard_case, lambda_iters, predicted, actual, rho,
 * accepted, iteration, trial_point, cauchy_point, newton_point, note); reg_newton has one Step per
 * accepted iterate (grad, hess, hess_eigs, direction, lambda, H_reg, trials, inner_iters, pd,
 * descent). `Result.extra`: arc {n_rejected, sigma_final, lambda_min}; reg_newton {variant,
 * n_trials, lambda_min}.
 *
 * NOTE (port): `np.linalg.eigh` is the LAPACK-exact 2×2 solver of trust_region.ts (Jacobi for
 * n ≥ 3); `Q.T @ g` is the fused-multiply-add chain of OpenBLAS dgemv_t, `Q @ w` a plain sum, as
 * in trust_region.ts. The Newton point of the picture uses Cholesky and two triangular solves
 * instead of `np.linalg.solve` (it never steers the iteration).
 */
import { registerMethod, param, type MethodDoc } from '../../core/registry';
import type { Matrix, MethodFn, Params, Result, RunOptions, Step, Vector } from '../../core/types';
import { formatG, formatRepr } from '../line_search/methods';
import { ValueError, resolveSmooth, type SmoothProblem } from './conjugate_gradient';
import { eigh, fma } from './trust_region';

const EPS = Number.EPSILON;
/** ε_M of CGT §7: the floor of σ after a very successful iteration. */
const SIGMA_MIN = EPS;
/** Rounding allowance δ = ROUNDING·ε·max(|f(x)|, |f(x₊)|). */
const ROUNDING = 1e3;
/** Relative accuracy of the cubic subproblem's secular equation. */
const SECULAR_RTOL = 1e-12;
/** Iteration cap of the secular-equation solver. */
const SECULAR_MAX_ITER = 200;
/** Cap on the trials of the adaptive searches. */
const MAX_TRIALS = 100;

export const VARIANTS = ['adan', 'fixed', 'super_universal'] as const;

const g3 = (v: number) => formatG(v, 3);

// ---------------------------------------------------------------------------------------
// Small dense linear algebra
// ---------------------------------------------------------------------------------------

function dotv(a: readonly number[], b: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
const norm = (v: readonly number[]) => Math.sqrt(dotv(v, v));
const maxAbs = (v: readonly number[]) => {
  let m = 0;
  for (const t of v) {
    const a = Math.abs(t);
    if (Number.isNaN(a)) return NaN;
    if (a > m) m = a;
  }
  return m;
};
const finiteAll = (...vs: (number | readonly number[] | Matrix | null)[]): boolean =>
  vs.every((v) =>
    v === null
      ? false
      : typeof v === 'number'
        ? Number.isFinite(v)
        : (v as readonly (number | readonly number[])[]).every((e) =>
            typeof e === 'number' ? Number.isFinite(e) : e.every((t) => Number.isFinite(t)),
          ),
  );
const symmetrize = (B: Matrix): Matrix => B.map((row, i) => row.map((v, j) => 0.5 * (v + B[j][i])));
const matvec = (A: Matrix, v: readonly number[]): Vector => A.map((row) => dotv(row, v));

/** `Q.T @ v` as OpenBLAS dgemv_t rounds it: an FMA chain for n ≤ 5, a plain dot beyond. */
function qtTimes(Q: Matrix, v: readonly number[]): Vector {
  const n = v.length;
  const out = new Array<number>(Q[0].length);
  for (let j = 0; j < out.length; j++) {
    if (n <= 5) {
      let s = 0.0;
      for (let i = 0; i < n; i++) s = fma(Q[i][j], v[i], s);
      out[j] = s;
    } else {
      let s = 0;
      for (let i = 0; i < n; i++) s += Q[i][j] * v[i];
      out[j] = s;
    }
  }
  return out;
}

/** Ascending eigenvalues and eigenvectors (columns of Q) of a symmetric matrix. */
function eigSym(B: Matrix): { lam: Vector; Q: Matrix } {
  const { values, Q } = eigh(B);
  return { lam: values, Q };
}

// ---------------------------------------------------------------------------------------
// Second-order test (rules of numopt.unconstrained.newton)
// ---------------------------------------------------------------------------------------

interface Source {
  gradFd: boolean;
  hessFd: boolean;
}

function eigTol(lam: Vector, fx: number, src: Source): number {
  const n = lam.length;
  const lamMax = maxAbs(lam);
  if (!src.hessFd) return n * EPS * lamMax;
  const scale = Math.max(1.0, lamMax, src.gradFd && Number.isFinite(fx) ? Math.abs(fx) : 0.0);
  return n * EPS ** (1.0 / 3.0) * scale;
}

function secondOrder(lam: Vector, fx: number, src: Source): [boolean, string] {
  const tol = eigTol(lam, fx, src);
  const lamMin = lam[0];
  const lamTop = lam[lam.length - 1];
  const source = src.hessFd ? 'the finite-difference ∇²f' : '∇²f';
  if (lamMin < -tol) {
    const kind =
      lamTop < -tol
        ? 'a maximizer'
        : lamTop > tol
          ? 'a saddle point'
          : 'a saddle point or a maximizer (∇²f ⪯ 0 is singular)';
    return [
      false,
      `stopped at ${kind}, not a minimizer: ${source} has the eigenvalue ` +
        `λ_min = ${g3(lamMin)} < −tol = ${g3(-tol)}`,
    ];
  }
  if (lamMin <= tol)
    return [
      true,
      `λ_min(${source}) = ${g3(lamMin)} is within ±${g3(tol)} of 0: positive semidefinite, ` +
        'the second-order sufficient condition is not verified',
    ];
  return [true, `${source} is positive definite (λ_min = ${g3(lamMin)}): a strict local minimizer`];
}

function checkCommon(gtol: number, maxIter: number): number {
  if (!(gtol >= 0.0)) throw new ValueError(`gtol must be ≥ 0, got ${formatRepr(gtol)}`);
  if (!Number.isInteger(maxIter) || maxIter < 1)
    throw new ValueError(`max_iter must be a positive integer, got ${String(maxIter)}`);
  return maxIter;
}

function resolve(problem: SmoothProblem, x0: unknown) {
  const r = resolveSmooth(problem, x0);
  const p = typeof problem === 'function' ? null : problem;
  const src: Source = {
    gradFd: !(p && typeof p.grad === 'function'),
    hessFd: !(p && typeof p.hess === 'function'),
  };
  return { ...r, src };
}

/** Evaluate without letting an exception escape (Python's errstate: a NaN result instead). */
function safe<T>(fn: () => T, fallback: T): T {
  try {
    return fn();
  } catch {
    return fallback;
  }
}

// ---------------------------------------------------------------------------------------
// The cubic subproblem (CGT Thm. 3.1, §6.1)
// ---------------------------------------------------------------------------------------

export interface CubicStep {
  s: Vector;
  lam: number;
  lamMin: number;
  hardCase: boolean;
  iters: number;
  /** f − m(s) = ½sᵀ(B + λI)s + λ‖s‖²/6. */
  predicted: number;
  /** False when the secular iteration missed its tolerance (then s = s(hi)). */
  solved: boolean;
}

/** Global minimizer of m(s) = gᵀs + ½sᵀBs + (σ/3)‖s‖³ over ℝⁿ (see the Python docstring). */
export function cubicSubproblem(g: Vector, B: Matrix, sigma: number): CubicStep {
  const n = g.length;
  const { lam: eigvals, Q } = eigSym(B);
  const gamma = qtTimes(Q, g);
  const lam1 = eigvals[0];
  const gnorm = norm(g);
  const bnorm = maxAbs(eigvals);
  let c: Vector;
  let tOff: number;
  if (lam1 >= 0.0) {
    c = eigvals.slice();
    tOff = 0.0;
  } else {
    c = eigvals.map((v) => v - lam1);
    tOff = -lam1;
  }
  c = c.map((v) => Math.max(v, 0.0));
  const qTimes = (w: readonly number[]) => matvec(Q, w);

  const finish = (
    t: number,
    s: Vector,
    hard: boolean,
    iters: number,
    solved: boolean,
  ): CubicStep => {
    const lam = t + tOff;
    const v = qtTimes(Q, s);
    let sum = 0;
    for (let j = 0; j < n; j++) sum += (c[j] + t) * v[j] * v[j];
    const pred = 0.5 * sum + (lam * dotv(v, v)) / 6.0;
    return { s, lam, lamMin: lam1, hardCase: hard, iters, predicted: pred, solved };
  };

  // 1. Hard case.
  const tol = 10.0 * n * EPS;
  if (lam1 < 0.0) {
    const cluster = c.map((v) => v <= tol * bnorm);
    const gamma1 = norm(gamma.filter((_, j) => cluster[j]));
    const target = -lam1 / sigma;
    if (gamma1 <= tol * Math.max(bnorm * target, gnorm)) {
      const coef = gamma.map((v, j) => (cluster[j] ? 0.0 : v / c[j]));
      const sPerp = qTimes(coef).map((v) => -v);
      const sp = norm(sPerp);
      if (sp <= target) {
        const alpha = Math.sqrt(Math.max(target * target - sp * sp, 0.0));
        let u: Vector;
        if (gamma1 > 0.0) {
          const masked = gamma.map((v, j) => (cluster[j] ? v : 0.0));
          u = qTimes(masked).map((v) => -v / gamma1);
        } else {
          // P₁g = 0: the sign is fixed so that the largest |entry| of u is positive.
          u = Q.map((row) => row[0]);
          let iMax = 0;
          for (let i = 1; i < n; i++) if (Math.abs(u[i]) > Math.abs(u[iMax])) iMax = i;
          if (u[iMax] < 0.0) u = u.map((v) => -v);
        }
        return finish(
          0.0,
          sPerp.map((v, i) => v + alpha * u[i]),
          true,
          0,
          true,
        );
      }
    }
  }

  // 2. Safeguarded Newton on φ₁ in t.
  if (gnorm === 0.0) return finish(0.0, new Array<number>(n).fill(0), false, 0, true);
  const disc = Math.sqrt(lam1 * lam1 + 4.0 * sigma * gnorm);
  let hi = (2.0 * sigma * gnorm) / (disc + Math.abs(lam1));
  let lo = 0.0;
  let t = hi;
  let iters = 0;
  while (iters < SECULAR_MAX_ITER) {
    const tc = t;
    const w = gamma.map((v, j) => v / (c[j] + tc));
    const snorm = norm(w);
    const lam = t + tOff;
    const target = lam / sigma;
    if (Math.abs(snorm - target) <= SECULAR_RTOL * target)
      return finish(
        t,
        qTimes(w).map((v) => -v),
        false,
        iters,
        true,
      );
    let qSq = 0;
    for (let j = 0; j < n; j++) qSq += (w[j] * w[j]) / (c[j] + tc);
    const phi = 1.0 / snorm - sigma / lam;
    const dphi = qSq / snorm ** 3 + sigma / (lam * lam);
    let tNew = t - phi / dphi;
    if (snorm > target) lo = t;
    else hi = t;
    if (!(Number.isFinite(tNew) && lo < tNew && tNew <= hi))
      tNew = Math.max(Math.sqrt(lo) * Math.sqrt(hi), 1e-3 * hi);
    iters++;
    if (!(lo < tNew && tNew <= hi) || tNew === t) break;
    t = tNew;
  }
  const hc = hi;
  const s = qTimes(gamma.map((v, j) => v / (c[j] + hc))).map((v) => -v);
  return finish(hi, s, false, iters, false);
}

/** m(s) − f = gᵀs + ½sᵀBs + (σ/3)‖s‖³ (CGT eq. 1.4). */
export function cubicModel(g: Vector, B: Matrix, sigma: number, s: Vector): number {
  const ns = norm(s);
  return dotv(g, s) + 0.5 * dotv(s, matvec(B, s)) + (sigma * ns ** 3) / 3.0;
}

/** s^C = −t g/‖g‖ minimizing m along −g (CGT eq. 2.3). */
export function cubicCauchy(g: Vector, B: Matrix, sigma: number): Vector {
  const gnorm = norm(g);
  if (gnorm === 0.0) return g.map(() => 0.0);
  const u = g.map((v) => v / gnorm);
  const c = dotv(u, matvec(B, u));
  const root = Math.sqrt(c * c + 4.0 * sigma * gnorm);
  const t = c >= 0.0 ? (2.0 * gnorm) / (c + root) : (root - c) / (2.0 * sigma);
  return u.map((v) => -t * v);
}

/** −B⁻¹g by Cholesky when B ≻ 0 (for the picture only), else null. */
function newtonPoint(B: Matrix, g: Vector): Vector | null {
  const n = g.length;
  const L: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let j = 0; j < n; j++) {
    let s = 0;
    for (let k = 0; k < j; k++) s += L[j][k] * L[j][k];
    const d = B[j][j] - s;
    if (!(d > 0.0)) return null;
    L[j][j] = Math.sqrt(d);
    for (let i = j + 1; i < n; i++) {
      let t = 0;
      for (let k = 0; k < j; k++) t += L[i][k] * L[j][k];
      L[i][j] = (B[i][j] - t) / L[j][j];
    }
  }
  const y = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let k = 0; k < i; k++) s += L[i][k] * y[k];
    y[i] = (g[i] - s) / L[i][i];
  }
  const p = new Array<number>(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = 0;
    for (let k = i + 1; k < n; k++) s += L[k][i] * p[k];
    p[i] = (y[i] - s) / L[i][i];
  }
  const out = p.map((v) => -v);
  return finiteAll(out) ? out : null;
}

// ---------------------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------------------

const P_GTOL = param.float('gtol', 1e-8, {
  min: 1e-14,
  max: 1e-2,
  log: true,
  help: 'Stop when ‖∇f(x)‖∞ ≤ gtol (converged only if ∇²f(x) has no negative eigenvalue).',
  label: 'Gradient tolerance',
  tex: '\\|\\nabla f\\|_\\infty \\le',
});
const P_MAX_ITER = param.int('max_iter', 200, {
  min: 1,
  max: 100_000,
  help: 'Iteration limit.',
  label: 'Max iterations',
});

type Opts = RunOptions & Params;
const num = (v: unknown, def: number): number => (typeof v === 'number' ? v : def);

// ---------------------------------------------------------------------------------------
// ARC (CGT Alg. 2.1)
// ---------------------------------------------------------------------------------------

export const arc: MethodFn<SmoothProblem> = (problem, o: Opts) => {
  const gtol = num(o.gtol, 1e-8);
  const maxIter = checkCommon(gtol, num(o.max_iter, 200));
  const sigma0 = num(o.sigma0, 1.0);
  const eta1 = num(o.eta1, 0.1);
  const eta2 = num(o.eta2, 0.9);
  const gammaInc = num(o.gamma, 2.0);
  if (!(sigma0 > 0.0 && sigma0 < Infinity))
    throw new ValueError(`sigma0 must be positive and finite, got ${formatRepr(sigma0)}`);
  if (!(eta1 > 0.0 && eta1 <= eta2 && eta2 < 1.0))
    throw new ValueError(
      `need 0 < eta1 ≤ eta2 < 1, got eta1=${formatRepr(eta1)}, eta2=${formatRepr(eta2)}`,
    );
  if (!(gammaInc > 1.0 && gammaInc < Infinity))
    throw new ValueError(`gamma must be > 1, got ${formatRepr(gammaInc)}`);

  const { x: xStart, f, grad, hess, src } = resolve(problem, o.x0);
  let x = xStart;
  const n = x.length;
  const nanMat = () => Array.from({ length: n }, () => new Array<number>(n).fill(NaN));
  let fx = safe(() => f(x), NaN);
  let g: Vector = safe(() => grad(x), new Array<number>(n).fill(NaN));
  let B: Matrix = safe(() => hess(x), nanMat());
  let sigma = sigma0;
  const trace: Step[] = [];
  let nRejected = 0;

  const baseInfo = (center: Vector, gc: Vector, Bc: Matrix): Record<string, unknown> => ({
    center: center.slice(),
    grad: finiteAll(gc) ? gc.slice() : null,
    H: n === 2 && Bc.length === 2 && finiteAll(Bc) ? Bc.map((r) => r.slice()) : null,
    sigma,
    new_sigma: sigma,
    step: null,
    step_norm: null,
    lambda: null,
    lambda_min: null,
    hard_case: null,
    lambda_iters: null,
    predicted: null,
    actual: null,
    rho: null,
    accepted: null,
    iteration: null,
    trial_point: null,
    cauchy_point: null,
    newton_point: null,
    note: null,
  });

  const done = (converged: boolean, message: string, k: number): Result => {
    let lamMin: number | null = null;
    let ok = converged;
    let msg = message;
    if (finiteAll(B) && B.length === n) {
      const eig = eigSym(B).lam;
      lamMin = eig[0];
      if (ok) {
        const [ok2, why] = secondOrder(eig, fx, src);
        ok = ok2;
        msg += '; ' + why;
      }
    }
    return {
      method: 'arc',
      x: x.slice(),
      fun: fx,
      converged: ok,
      message: msg,
      nIter: k,
      nFev: f.n,
      nGev: grad.n,
      nHev: hess.n,
      trace,
      extra: { n_rejected: nRejected, sigma_final: sigma, lambda_min: lamMin },
    };
  };

  if (!(Number.isFinite(fx) && finiteAll(g, B)) || B.length !== n || g.length !== n) {
    const gn = finiteAll(g) ? norm(g) : null;
    trace.push({
      k: 0,
      x: x.slice(),
      fun: Number.isFinite(fx) ? fx : null,
      gradNorm: gn,
      stepSize: null,
      info: baseInfo(x, g, B),
    });
    return done(false, 'f, ∇f or ∇²f is not finite (or has the wrong shape) at x0', 0);
  }
  B = symmetrize(B);
  trace.push({
    k: 0,
    x: x.slice(),
    fun: fx,
    gradNorm: norm(g),
    stepSize: null,
    info: baseInfo(x, g, B),
  });

  for (let k = 1; k <= maxIter; k++) {
    const ginf = maxAbs(g);
    if (ginf <= gtol) return done(true, `gradient ‖∇f‖∞ = ${g3(ginf)} ≤ gtol`, k - 1);
    const sub = cubicSubproblem(g, B, sigma);
    let s = sub.s;
    let note: string | null = null;
    let predicted = sub.predicted;
    const sCauchy = cubicCauchy(g, B, sigma);
    let lam: number | null = sub.lam;
    if (!sub.solved || !finiteAll(s)) {
      // Safety net: keep the Cauchy condition (CGT eq. 2.2).
      const mS = finiteAll(s) ? cubicModel(g, B, sigma, s) : Infinity;
      const mC = cubicModel(g, B, sigma, sCauchy);
      if (mC < mS) {
        s = sCauchy;
        lam = null;
        predicted = -mC;
        note = 'secular equation not solved to tolerance; Cauchy point used';
      } else {
        predicted = -mS;
        note = 'secular equation not solved to tolerance; s(λ_hi) used';
      }
    }
    if (!(predicted > 0.0 && Number.isFinite(predicted)))
      return done(
        false,
        `the model step predicts no decrease (f − m(s) = ${g3(predicted)} at iteration ` +
          `${k}): the cubic model is at its rounding level; ‖∇f‖∞ = ${g3(ginf)} > gtol`,
        k - 1,
      );
    const trial = x.map((v, i) => v + s[i]);
    const fTrial = safe(() => f(trial), NaN);
    let actual: number | null;
    let rho: number | null;
    if (Number.isFinite(fTrial)) {
      actual = fx - fTrial;
      const deltaRound = ROUNDING * EPS * Math.max(Math.abs(fx), Math.abs(fTrial));
      rho = (fx - fTrial + deltaRound) / (predicted + deltaRound);
    } else {
      actual = null;
      rho = null;
    }
    const rhoValue = rho === null ? -Infinity : rho;
    let kind: string;
    let newSigma: number;
    if (rhoValue > eta2) {
      kind = 'very_successful';
      const gn = norm(g);
      newSigma = Math.max(gn < sigma ? gn : sigma, SIGMA_MIN);
    } else if (rhoValue >= eta1) {
      kind = 'successful';
      newSigma = sigma;
    } else {
      kind = 'unsuccessful';
      newSigma = gammaInc * sigma;
    }
    const accepted = rhoValue >= eta1;
    const stepNorm = norm(s);

    const info = baseInfo(x, g, B);
    const newton = newtonPoint(B, g);
    const xc = x;
    Object.assign(info, {
      new_sigma: newSigma,
      step: s.slice(),
      step_norm: stepNorm,
      lambda: lam,
      lambda_min: sub.lamMin,
      hard_case: sub.hardCase,
      lambda_iters: sub.iters,
      predicted,
      actual,
      rho,
      accepted,
      iteration: kind,
      trial_point: trial.slice(),
      cauchy_point: xc.map((v, i) => v + sCauchy[i]),
      newton_point: newton === null ? null : xc.map((v, i) => v + newton[i]),
      note,
    });
    if (accepted) {
      const gNew = safe(() => grad(trial), new Array<number>(n).fill(NaN));
      const BNew = safe(() => hess(trial), nanMat());
      x = trial;
      fx = fTrial;
      if (!(finiteAll(gNew, BNew) && BNew.length === n)) {
        const gn = finiteAll(gNew) ? norm(gNew) : null;
        trace.push({ k, x: x.slice(), fun: fx, gradNorm: gn, stepSize: stepNorm, info });
        B = nanMat();
        return done(false, `∇f or ∇²f is not finite at the iterate of step ${k}`, k);
      }
      g = gNew;
      B = symmetrize(BNew);
    } else {
      nRejected++;
    }
    trace.push({ k, x: x.slice(), fun: fx, gradNorm: norm(g), stepSize: stepNorm, info });
    sigma = newSigma;
    if (!accepted && stepNorm <= EPS * Math.max(1.0, norm(x)))
      return done(
        false,
        `σ = ${g3(sigma)}: the step fell below the rounding level of x (no acceptable ` +
          `step); ‖∇f‖∞ = ${g3(maxAbs(g))} > gtol`,
        k,
      );
  }
  const ginf = maxAbs(g);
  if (ginf <= gtol) return done(true, `gradient ‖∇f‖∞ = ${g3(ginf)} ≤ gtol`, maxIter);
  return done(false, `reached max_iter=${maxIter}`, maxIter);
};

// ---------------------------------------------------------------------------------------
// Gradient-regularized Newton
// ---------------------------------------------------------------------------------------

/** s = −(B + λI)⁻¹g from B = QΛQᵀ; [null, pd] when B + λI is numerically singular. */
function regSolve(eigvals: Vector, Q: Matrix, g: Vector, lam: number): [Vector | null, boolean] {
  const shifted = eigvals.map((v) => v + lam);
  const scale = Math.max(maxAbs(eigvals), lam);
  const pd = shifted[0] > 0.0;
  let minAbs = Infinity;
  for (const v of shifted) minAbs = Math.min(minAbs, Math.abs(v));
  if (minAbs <= 10.0 * eigvals.length * EPS * scale) return [null, pd];
  const w = qtTimes(Q, g).map((v, j) => v / shifted[j]);
  const s = matvec(Q, w).map((v) => -v);
  return [finiteAll(s) ? s : null, pd];
}

export const regNewton: MethodFn<SmoothProblem> = (problem, o: Opts) => {
  const gtol = num(o.gtol, 1e-8);
  const maxIter = checkCommon(gtol, num(o.max_iter, 200));
  const variant = String(o.variant ?? 'adan');
  const H = num(o.H, 1.0);
  const alpha = num(o.alpha, 1.0);
  if (!(VARIANTS as readonly string[]).includes(variant))
    throw new ValueError(
      `variant must be one of ('adan', 'fixed', 'super_universal'), got '${variant}'`,
    );
  if (!(H > 0.0 && H < Infinity))
    throw new ValueError(`H must be positive and finite, got ${formatRepr(H)}`);
  if (!(2.0 / 3.0 - 1e-12 <= alpha && alpha <= 1.0))
    throw new ValueError(`alpha must lie in [2/3, 1], got ${formatRepr(alpha)}`);

  const { x: xStart, f, grad, hess, src } = resolve(problem, o.x0);
  let x = xStart;
  const n = x.length;
  const nanMat = () => Array.from({ length: n }, () => new Array<number>(n).fill(NaN));
  let fx = safe(() => f(x), NaN);
  let g: Vector = safe(() => grad(x), new Array<number>(n).fill(NaN));
  let B: Matrix = safe(() => hess(x), nanMat());
  const trace: Step[] = [];
  let Hk = H; // adan: H_{k−1}; super_universal: H_k
  let totalTrials = 0;

  const eigOf = (Bm: Matrix): { lam: Vector; Q: Matrix } | null =>
    finiteAll(Bm) && Bm.length === n ? eigSym(symmetrize(Bm)) : null;

  const stateInfo = (gc: Vector, Bm: Matrix, eig: { lam: Vector } | null) => ({
    grad: finiteAll(gc) ? gc.slice() : null,
    hess: n <= 2 && Bm.length === n && finiteAll(Bm) ? Bm.map((r) => r.slice()) : null,
    hess_eigs: eig === null ? null : eig.lam.slice(),
  });
  const incomingNone = {
    direction: null,
    lambda: null,
    H_reg: null,
    trials: [],
    inner_iters: null,
    pd: null,
    descent: null,
  };

  const done = (
    converged: boolean,
    message: string,
    k: number,
    eig: { lam: Vector } | null,
  ): Result => {
    const lamMin = eig === null ? null : eig.lam[0];
    let ok = converged;
    let msg = message;
    if (ok && eig !== null) {
      const [ok2, why] = secondOrder(eig.lam, fx, src);
      ok = ok2;
      msg += '; ' + why;
    }
    return {
      method: 'reg_newton',
      x: x.slice(),
      fun: fx,
      converged: ok,
      message: msg,
      nIter: k,
      nFev: f.n,
      nGev: grad.n,
      nHev: hess.n,
      trace,
      extra: { variant, n_trials: totalTrials, lambda_min: lamMin },
    };
  };

  let eig = eigOf(B);
  const ok0 = Number.isFinite(fx) && finiteAll(g) && g.length === n && eig !== null;
  trace.push({
    k: 0,
    x: x.slice(),
    fun: Number.isFinite(fx) ? fx : null,
    gradNorm: finiteAll(g) ? norm(g) : null,
    stepSize: null,
    info: { ...stateInfo(g, B, eig), ...incomingNone },
  });
  if (!ok0 || eig === null)
    return done(false, 'f, ∇f or ∇²f is not finite (or has the wrong shape) at x0', 0, eig);

  for (let k = 1; k <= maxIter; k++) {
    const ginf = maxAbs(g);
    if (ginf <= gtol) return done(true, `gradient ‖∇f‖∞ = ${g3(ginf)} ≤ gtol`, k - 1, eig);
    const { lam: eigvals, Q }: { lam: Vector; Q: Matrix } = eig!;
    const gnorm = norm(g);
    const trials: unknown[][] = [];
    let acceptedPoint: [Vector, number, Vector, number, number, Vector, boolean] | null = null;
    let fail: string | null = null;

    if (variant === 'fixed') {
      const lam = Math.sqrt(H * gnorm);
      const [s, pd] = regSolve(eigvals, Q, g, lam);
      if (s === null) {
        fail = `∇²f + λI is numerically singular at iteration ${k} (λ = ${g3(lam)})`;
      } else {
        const xp = x.map((v, i) => v + s[i]);
        const fp = safe(() => f(xp), NaN);
        const gp = safe(() => grad(xp), new Array<number>(n).fill(NaN));
        const gpn = finiteAll(gp) ? norm(gp) : NaN;
        trials.push([
          H,
          lam,
          Number.isFinite(fp) ? fp : null,
          Number.isFinite(gpn) ? gpn : null,
          true,
        ]);
        acceptedPoint = [xp, fp, gp, lam, H, s, pd];
      }
    } else {
      let Htry = variant === 'adan' ? (k === 1 ? Hk : Hk / 4.0) : Hk;
      for (let j = 0; j < MAX_TRIALS; j++) {
        let lam: number;
        let Hused: number;
        if (variant === 'adan') {
          Htry *= 2.0;
          lam = Math.sqrt(Htry * gnorm);
          Hused = Htry;
        } else {
          Hused = 4.0 ** j * Hk;
          lam = Hused * gnorm ** alpha;
        }
        const [s, pd] = regSolve(eigvals, Q, g, lam);
        if (s === null) {
          trials.push([Hused, lam, null, null, false]);
          continue;
        }
        const xp = x.map((v, i) => v + s[i]);
        const r = norm(s);
        const gp = safe(() => grad(xp), new Array<number>(n).fill(NaN));
        const gpn = finiteAll(gp) ? norm(gp) : NaN;
        let fp: number;
        let ok: boolean;
        if (variant === 'adan') {
          fp = safe(() => f(xp), NaN);
          const deltaRound = Number.isFinite(fp)
            ? ROUNDING * EPS * Math.max(Math.abs(fx), Math.abs(fp))
            : 0.0;
          ok =
            Number.isFinite(fp) &&
            Number.isFinite(gpn) &&
            gpn <= 2.0 * lam * r &&
            fp <= fx - (2.0 / 3.0) * lam * r * r + deltaRound;
          trials.push([
            Hused,
            lam,
            Number.isFinite(fp) ? fp : null,
            Number.isFinite(gpn) ? gpn : null,
            ok,
          ]);
        } else {
          fp = NaN;
          ok =
            Number.isFinite(gpn) &&
            dotv(
              gp,
              s.map((v) => -v),
            ) >=
              (gpn * gpn) / (4.0 * lam);
          trials.push([Hused, lam, null, Number.isFinite(gpn) ? gpn : null, ok]);
        }
        if (ok) {
          if (variant === 'super_universal') {
            fp = safe(() => f(xp), NaN);
            Hk = Hused / 4.0; // H_{k+1} = 4^{j_k} H_k / 4
          } else {
            Hk = Hused;
          }
          acceptedPoint = [xp, fp, gp, lam, Hused, s, pd];
          break;
        }
      }
      if (acceptedPoint === null)
        fail =
          `the adaptive search found no acceptable step in ${MAX_TRIALS} trials at ` +
          `iteration ${k}; ‖∇f‖∞ = ${g3(ginf)} > gtol`;
    }
    totalTrials += trials.length;

    if (acceptedPoint === null) return done(false, fail ?? 'no step', k - 1, eig);
    const [xp, fp, gp, lam, Hused, s, pd] = acceptedPoint;
    const descent = dotv(g, s) < 0.0;
    x = xp;
    fx = fp;
    g = gp;
    if (Number.isFinite(fx) && finiteAll(g)) {
      B = safe(() => hess(x), nanMat());
      eig = eigOf(B);
    } else {
      B = nanMat();
      eig = null;
    }
    const info = {
      ...stateInfo(g, B, eig),
      direction: s.slice(),
      lambda: lam,
      H_reg: Hused,
      trials,
      inner_iters: trials.length,
      pd,
      descent,
    };
    trace.push({
      k,
      x: x.slice(),
      fun: Number.isFinite(fx) ? fx : null,
      gradNorm: finiteAll(g) ? norm(g) : null,
      stepSize: norm(s),
      info,
    });
    if (!(Number.isFinite(fx) && finiteAll(g)) || eig === null)
      return done(false, `f, ∇f or ∇²f is not finite at the iterate of step ${k}`, k, eig);
  }
  const ginf = maxAbs(g);
  if (ginf <= gtol) return done(true, `gradient ‖∇f‖∞ = ${g3(ginf)} ≤ gtol`, maxIter, eig);
  return done(false, `reached max_iter=${maxIter}`, maxIter, eig);
};

// ---------------------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------------------

const ARC_QUANTITIES: MethodDoc['quantities'] = [
  { tex: '\\sigma_k', key: 'info.sigma' },
  { tex: '\\rho_k', key: 'info.rho' },
  { tex: '\\|s_k\\|', key: 'stepSize' },
];

registerMethod(
  {
    id: 'arc',
    family: 'unconstrained',
    name: 'ARC (adaptive cubic regularization)',
    params: [
      P_GTOL,
      P_MAX_ITER,
      param.float('sigma0', 1.0, {
        min: 1e-4,
        max: 1e4,
        log: true,
        help: 'Initial cubic regularization σ₀ (like an inverse trust radius).',
        label: 'Initial regularization',
        tex: '\\sigma_0',
      }),
      param.float('eta1', 0.1, {
        min: 1e-4,
        max: 0.5,
        help: 'Accept the step when ρ ≥ η₁ (CGT eq. 2.5).',
        label: 'Acceptance threshold',
        tex: '\\eta_1',
      }),
      param.float('eta2', 0.9, {
        min: 0.5,
        max: 0.999,
        help: 'Very successful step when ρ > η₂: σ decreases (eq. 2.6).',
        label: 'Very successful threshold',
        tex: '\\eta_2',
      }),
      param.float('gamma', 2.0, {
        min: 1.1,
        max: 10.0,
        help: 'σ ← γσ after an unsuccessful step (CGT eq. 2.6).',
        label: 'Increase factor',
        tex: '\\gamma',
      }),
    ],
    needs: ['f', 'grad', 'hess'],
    order: 'superlinear (near a minimizer with ∇²f ≻ 0); O(ε⁻³ᐟ²) worst-case evaluations',
    summary:
      'Step to the global minimizer of the Newton model plus a cubic penalty (σ/3)‖s‖³, ' +
      'and adapt σ from how well the model predicted the decrease.',
    references: [
      'Cartis, Gould & Toint (2011), Math. Program. 127, Part I: Alg. 2.1; eq. 1.4, 2.3–2.6; ' +
        'Thm. 3.1; §6.1 eq. 6.6–6.7; §7 (parameters)',
      'Cartis, Gould & Toint (2011), Math. Program. 130, Part II (O(ε⁻³ᐟ²) complexity)',
      "Nesterov & Polyak (2006), Math. Program. 108 (cubic regularization of Newton's method)",
    ],
  },
  arc,
  {
    rule: 's_k = \\arg\\min_s\\ \\nabla f_k^{\\top}s + \\tfrac12 s^{\\top}\\nabla^2 f_k\\,s + \\tfrac{\\sigma_k}{3}\\|s\\|^3,\\quad x_{k+1} = x_k + s_k \\text{ if } \\rho_k \\ge \\eta_1',
    intuition:
      'Newton’s model plus a cubic penalty on long steps: the step solves (∇²f + λI)s = −∇f ' +
      'with λ = σ‖s‖, and σ grows when the model was wrong and shrinks when it was right.',
    quantities: ARC_QUANTITIES,
  },
);

registerMethod(
  {
    id: 'reg_newton',
    family: 'unconstrained',
    name: 'Gradient-regularized Newton',
    params: [
      P_GTOL,
      P_MAX_ITER,
      param.choice('variant', 'adan', [...VARIANTS], {
        help:
          'fixed: λ = √(H‖g‖) (Mishchenko Alg. 1); adan: H adapted by doubling (Alg. 2); ' +
          'super_universal: λ = 4ʲH‖g‖^α (Doikov–Mishchenko–Nesterov Alg. 2).',
        label: 'Variant',
      }),
      param.float('H', 1.0, {
        min: 1e-6,
        max: 1e6,
        log: true,
        help: 'fixed: the constant H (∇²f is 2H-Lipschitz); adaptive variants: the initial H₀.',
        label: 'Regularization constant',
        tex: 'H',
      }),
      param.float('alpha', 1.0, {
        min: 2.0 / 3.0,
        max: 1.0,
        help: 'super_universal only: the power α ∈ [2/3, 1] of ‖∇f‖ in λ.',
        label: 'Gradient power',
        tex: '\\alpha',
      }),
    ],
    needs: ['f', 'grad', 'hess'],
    order: 'superlinear (strongly convex f, Mishchenko Thm. 2); global O(1/k²) for convex f',
    summary:
      'Newton step with ∇²f + λI, where λ = √(H‖∇f‖) shrinks as the gradient vanishes; ' +
      'H is fixed or adapted, and no line search is needed on convex functions.',
    references: [
      'Mishchenko (2023), SIAM J. Optim. 33(3), arXiv:2112.02089v3: Alg. 1 (fixed), ' +
        'Alg. 2 (AdaN), Assumption 1, Thm. 1–3',
      'Doikov, Mishchenko & Nesterov (2024), SIAM J. Optim. 34(1), arXiv:2208.05888v1: ' +
        'Alg. 2 (super-universal, ψ ≡ 0, B = I)',
    ],
  },
  regNewton,
  {
    rule: 'x_{k+1} = x_k - \\big(\\nabla^2 f(x_k) + \\lambda_k I\\big)^{-1}\\nabla f(x_k),\\quad \\lambda_k = \\sqrt{H_k\\,\\|\\nabla f(x_k)\\|}',
    intuition:
      'Newton’s step with the Hessian shifted by λ, which is large far from the solution (a ' +
      'short, gradient-like step) and vanishes as ∇f → 0 (Newton’s step).',
    quantities: [
      { tex: '\\lambda_k', key: 'info.lambda' },
      { tex: 'H_k', key: 'info.H_reg' },
      { tex: '\\|\\nabla f(x_k)\\|', key: 'gradNorm' },
    ],
  },
);
