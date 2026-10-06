/**
 * Iterative solvers for a square linear system A x = b — TS port of
 * `src/numopt/linalg/iterative.py`.
 *
 * Stationary methods (Jacobi, Gauss–Seidel, SOR) iterate x ← G x + c from the splitting
 * A = D + L + U; they converge for every start iff ρ(G) < 1. Krylov and descent methods (steepest
 * descent, CG, preconditioned CG, GMRES) pick x_k from x₀ + span{r₀, A r₀, …}.
 *
 * Stopping test (every method): ‖r_k‖₂ ≤ tol·d with d = ‖b‖₂ (‖r₀‖₂ when b = 0, then
 * ‖A‖_F‖x₀‖₂, then 1), and the rounding level u·(‖A‖_F‖x_k‖₂ + ‖b‖₂) ≤ tol·d.
 *
 * Trace: one Step for k = 0 (x₀) and one per iteration; `fun` = ‖r_k‖₂, `stepSize` =
 * ‖x_k − x_{k−1}‖₂. Info keys (snake_case, as in Python): residual, residual_norm,
 * relative_residual, spectral_radius, sweep, direction, alpha, beta, phi, condition_number,
 * preconditioned_condition_number, rate_bound, preconditioned_residual, cycle, krylov_dim,
 * hessenberg, givens, basis_vector. Counts: nFev = nGev = 0; `extra.n_matvec` for the Krylov
 * methods.
 */
import { param, registerMethod } from '../../core/registry';
import type {
  Matrix,
  MethodFn,
  Params,
  Result,
  RunOptions,
  Step,
  StepInfo,
  Vector,
} from '../../core/types';
import {
  EPS,
  allFinite,
  asymmetry,
  backSubstitution,
  blasDot,
  dotStrided,
  eigvals,
  eigvalsh,
  gemv,
  gemvStrided,
  hypot,
  isSymmetric,
  maxAbs,
  mv,
  norm2,
  normF,
  pivotTolerance,
  pow2Scale,
  pyG,
  resolveSystem,
  scaledDot,
  solveMatrix,
  spectralRadius,
  zeros,
  zerosM,
  type SystemLike,
} from './numerics';

/** Smallest admissible tol (also the ParamSpec minimum). */
export const TOL_MIN = 1e-15;
/** Unit roundoff u = ε/2 = 2⁻⁵³. */
export const UNIT_ROUNDOFF = EPS / 2.0;

const TOL = param.float('tol', 1e-10, {
  min: TOL_MIN,
  max: 1e-1,
  log: true,
  help: 'Stop when ‖b − Ax‖₂ ≤ tol·‖b‖₂ (tol·‖r₀‖₂ when b = 0).',
  label: 'Relative residual',
  tex: '\\|\\mathbf{r}\\|/\\|\\mathbf{b}\\| \\le',
});
const MAX_ITER = param.int('max_iter', 1000, {
  min: 1,
  max: 100_000,
  help: 'Iteration limit.',
  label: 'Iteration budget',
});

type Opts = RunOptions & Params;

// ── Helpers ───────────────────────────────────────────────────────────────────────────

function start(x0: unknown, n: number): Vector {
  if (x0 === null || x0 === undefined) return zeros(n);
  const x = (Array.isArray(x0) ? (x0 as unknown[]).flat(Infinity) : [x0]).map(Number);
  if (x.length !== n) throw new Error(`x0 has ${x.length} entries, expected ${n}`);
  if (!allFinite(x)) throw new Error('x0 must be finite');
  return x;
}

function checkCount(name: string, value: unknown): number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 1)
    throw new Error(`${name} must be an integer ≥ 1, got ${String(value)}`);
  return value;
}

function checkNorms(A: Matrix, b: Vector): number {
  const aNorm = normF(A);
  for (const [name, value] of [
    ['‖A‖_F', aNorm],
    ['‖b‖₂', norm2(b)],
  ] as const) {
    if (!Number.isFinite(value))
      throw new Error(
        `${name} exceeds the float64 range (≈ 1.8e308): rescale A and b by a power of 2 ` +
          '(this is exact and does not change x)',
      );
  }
  return aNorm;
}

function checkTol(tol: number): void {
  if (!(tol >= TOL_MIN))
    throw new Error(
      `tol must be ≥ ${pyG(TOL_MIN, 6)} (a relative residual below ≈ 9u is rounding noise), ` +
        `got ${tol}`,
    );
}

function roundingLevel(aNorm: number, x: Vector, b: Vector): number {
  return UNIT_ROUNDOFF * (aNorm * norm2(x) + norm2(b));
}

/** The denominator d of the stopping test ‖r_k‖₂ ≤ tol·d and its printed name. */
interface Scale {
  value: number;
  name: string;
}

const ratio = (s: Scale, rnorm: number) => `‖r‖/${s.name} = ${pyG(rnorm / s.value)}`;

function scaleOf(A: Matrix, b: Vector, x0: Vector, r0: Vector): Scale {
  const candidates: [number, string][] = [
    [norm2(b), '‖b‖'],
    [norm2(r0), '‖r₀‖'],
    [normF(A) * norm2(x0), '(‖A‖_F‖x₀‖)'],
  ];
  for (const [value, name] of candidates)
    if (value > 0.0 && Number.isFinite(value)) return { value, name };
  return { value: 1.0, name: '1' };
}

function certify(
  rnorm: number,
  x: Vector,
  aNorm: number,
  b: Vector,
  tol: number,
  scale: Scale,
  message: string,
): [boolean, string] {
  const level = roundingLevel(aNorm, x, b);
  if (level <= tol * scale.value) return [true, message];
  return [
    false,
    `${ratio(scale, rnorm)} ≤ tol, but the rounding level of the computed residual, ` +
      `u(‖A‖_F‖x‖ + ‖b‖)/${scale.name} = ${pyG(level / scale.value)}, exceeds tol ` +
      `(‖x‖ = ${pyG(norm2(x))}): the residual cannot certify x (A is singular, or too ` +
      'ill-conditioned for this tol)',
  ];
}

function residualInfo(r: Vector, rnorm: number, scale: Scale): StepInfo {
  return { residual: r.slice(), residual_norm: rnorm, relative_residual: rnorm / scale.value };
}

function residual(A: Matrix, b: Vector, x: Vector): Vector {
  const Ax = mv(A, x);
  return b.map((v, i) => v - Ax[i]);
}

function step(
  k: number,
  x: Vector,
  fun: number,
  info: StepInfo,
  o: { gradNorm?: number | null; stepSize?: number | null } = {},
): Step {
  return {
    k,
    x: x.slice(),
    fun,
    gradNorm: o.gradNorm ?? null,
    stepSize: o.stepSize ?? null,
    info,
  };
}

function result(
  method: string,
  x: Vector,
  fun: number,
  converged: boolean,
  message: string,
  nIter: number,
  trace: Step[],
  extra: Record<string, unknown> = {},
): Result {
  return {
    method,
    x: x.slice(),
    fun,
    converged,
    message,
    nIter,
    nFev: 0,
    nGev: 0,
    nHev: 0,
    trace,
    extra,
  };
}

/** G_ω = (D + ωL)⁻¹((1 − ω)D − ωU), by a solve (never an inverse). */
export function sorMatrix(A: readonly (readonly number[])[], omega: number): Matrix {
  const n = A.length;
  const lhs = zerosM(n);
  const rhs = zerosM(n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      if (i === j) {
        lhs[i][j] = A[i][i];
        rhs[i][j] = (1.0 - omega) * A[i][i];
      } else if (j < i) lhs[i][j] = omega * A[i][j];
      else rhs[i][j] = -omega * A[i][j];
    }
  return solveMatrix(lhs, rhs);
}

/** G_J = −D⁻¹(L + U). */
export function jacobiMatrix(A: readonly (readonly number[])[]): Matrix {
  return A.map((row, i) => row.map((v, j) => (i === j ? -0 : -v / A[i][i])));
}

function isTridiagonal(A: readonly (readonly number[])[]): boolean {
  for (let i = 0; i < A.length; i++)
    for (let j = 0; j < A.length; j++) if (Math.abs(i - j) > 1 && A[i][j] !== 0.0) return false;
  return true;
}

function zeroDiagonal(
  method: string,
  A: Matrix,
  x: Vector,
  r: Vector,
  scale: Scale,
): Result | null {
  const i = A.findIndex((row, q) => row[q] === 0.0);
  if (i < 0) return null;
  const rnorm = norm2(r);
  const trace = [step(0, x, rnorm, residualInfo(r, rnorm, scale))];
  return result(
    method,
    x,
    rnorm,
    false,
    `a_${i}${i} = 0: the iteration divides by the diagonal; reorder the equations`,
    0,
    trace,
  );
}

const divergenceNote = (rho: number) =>
  rho >= 1.0 ? ` (spectral radius ρ(G) = ${pyG(rho, 4)} ≥ 1: the iteration diverges)` : '';

const convergedMsg = (rnorm: number, scale: Scale) => `${ratio(scale, rnorm)} ≤ tol`;

// ── Young's optimal relaxation factor ─────────────────────────────────────────────────

export interface OptimalOmega {
  omega: number;
  omega_opt: number | null;
  spectral_radius_opt: number | null;
  spectral_radius_jacobi: number;
  omega_opt_note: string;
}

/**
 * Young (1950); Burden & Faires §7.4; Saad (2003) §4.2: if A is consistently ordered (checked by
 * the sufficient condition "A is tridiagonal"), the Jacobi spectrum is real and ρ_J < 1, then
 * ω* = 2/(1 + √(1 − ρ_J²)) and ρ(G_ω*) = ω* − 1.
 */
export function optimalOmega(A: readonly (readonly number[])[], omega: number): OptimalOmega {
  const eig = eigvals(jacobiMatrix(A));
  let rhoJ = 0;
  let maxIm = 0;
  for (const { re, im } of eig) {
    rhoJ = Math.max(rhoJ, Math.hypot(re, im));
    maxIm = Math.max(maxIm, Math.abs(im));
  }
  const out: OptimalOmega = {
    omega,
    omega_opt: null,
    spectral_radius_opt: null,
    spectral_radius_jacobi: rhoJ,
    omega_opt_note: '',
  };
  if (!isTridiagonal(A))
    out.omega_opt_note = 'A is not tridiagonal: consistent ordering is not verified';
  else if (maxIm > 1e-12 * Math.max(1.0, rhoJ))
    out.omega_opt_note = 'the Jacobi matrix has complex eigenvalues';
  else if (rhoJ >= 1.0) out.omega_opt_note = `ρ(G_J) = ${pyG(rhoJ, 4)} ≥ 1`;
  else {
    const w = 2.0 / (1.0 + Math.sqrt(1.0 - rhoJ ** 2));
    out.omega_opt = w;
    out.spectral_radius_opt = w - 1.0;
    out.omega_opt_note = "Young's formula (A tridiagonal, real Jacobi spectrum, ρ_J < 1)";
  }
  return out;
}

// ── Stationary methods: Jacobi, Gauss–Seidel, SOR ─────────────────────────────────────

function stationary(
  method: string,
  problem: SystemLike,
  x0: unknown,
  tol: number,
  maxIterIn: unknown,
  omega: number | null,
): Result {
  checkTol(tol);
  const maxIter = checkCount('max_iter', maxIterIn);
  const { A, b } = resolveSystem(problem);
  const aNorm = checkNorms(A, b);
  const n = b.length;
  let x = start(x0, n);
  let r = residual(A, b, x);
  const scale = scaleOf(A, b, x, r);
  const failed = zeroDiagonal(method, A, x, r, scale);
  if (failed) return failed;

  const d = A.map((row, i) => row[i]);
  const off = A.map((row, i) => row.map((v, j) => (i === j ? 0 : v))); // L + U
  const G = omega === null ? jacobiMatrix(A) : sorMatrix(A, omega);
  const rho = spectralRadius(G);
  const extra: Record<string, unknown> = { spectral_radius: rho };
  if (method === 'sor' && omega !== null) Object.assign(extra, optimalOmega(A, omega));

  let rnorm = norm2(r);
  const trace = [step(0, x, rnorm, { ...residualInfo(r, rnorm, scale), spectral_radius: rho })];
  if (rnorm <= tol * scale.value) {
    const [ok, msg] = certify(rnorm, x, aNorm, b, tol, scale, 'x0 already satisfies the tolerance');
    return result(method, x, rnorm, ok, msg, 0, trace, extra);
  }

  for (let k = 1; k <= maxIter; k++) {
    const xOld = x.slice();
    const more: StepInfo = {};
    if (omega === null) {
      // Jacobi (B&F Alg. 7.1): x_i ← (b_i − Σ_{j≠i} a_ij x_j^old) / a_ii for all i at once.
      const offX = gemv(off, xOld);
      x = b.map((bi, i) => (bi - offX[i]) / d[i]);
    } else {
      // SOR (B&F Alg. 7.3), in place, so x_j for j < i is already new.
      const sweep: Vector[] = [x.slice()];
      for (let i = 0; i < n; i++) {
        const sigma =
          blasDot(A[i].slice(0, i), x.slice(0, i)) + blasDot(A[i].slice(i + 1), x.slice(i + 1));
        x[i] = (1.0 - omega) * x[i] + (omega * (b[i] - sigma)) / d[i];
        sweep.push(x.slice());
      }
      more.sweep = sweep;
    }
    r = residual(A, b, x);
    rnorm = norm2(r);
    const stepSize = norm2(x.map((v, i) => v - xOld[i]));
    trace.push(step(k, x, rnorm, { ...residualInfo(r, rnorm, scale), ...more }, { stepSize }));
    if (!(allFinite(x) && Number.isFinite(rnorm))) {
      const msg = `non-finite iterate at k = ${k}${divergenceNote(rho)}`;
      return result(method, x, rnorm, false, msg, k, trace, extra);
    }
    if (rnorm <= tol * scale.value) {
      const [ok, msg] = certify(rnorm, x, aNorm, b, tol, scale, convergedMsg(rnorm, scale));
      return result(method, x, rnorm, ok, msg, k, trace, extra);
    }
  }
  const msg = `reached max_iter=${maxIter} with ${ratio(scale, rnorm)}${divergenceNote(rho)}`;
  return result(method, x, rnorm, false, msg, trace[trace.length - 1].k, trace, extra);
}

const jacobi: MethodFn<SystemLike> = (problem, o: Opts) =>
  stationary('jacobi', problem, o.x0, Number(o.tol ?? 1e-10), o.max_iter ?? 1000, null);

const gaussSeidel: MethodFn<SystemLike> = (problem, o: Opts) =>
  stationary('gauss_seidel', problem, o.x0, Number(o.tol ?? 1e-10), o.max_iter ?? 1000, 1.0);

const sor: MethodFn<SystemLike> = (problem, o: Opts) => {
  const omega = Number(o.omega ?? 1.5);
  if (!(omega > 0.0 && omega < 2.0))
    throw new Error(`omega must lie in (0, 2) (Kahan's theorem), got ${omega}`);
  return stationary('sor', problem, o.x0, Number(o.tol ?? 1e-10), o.max_iter ?? 1000, omega);
};

// ── Steepest descent, CG, preconditioned CG (SPD A) ───────────────────────────────────

const SCALE_RANGE: [number, number] = [2.0 ** -200, 2.0 ** 200];

function checkScale(A: Matrix, b: Vector): void {
  const [lo, hi] = SCALE_RANGE;
  for (const [name, s] of [
    ['A', maxAbs(A.flat())],
    ['b', maxAbs(b)],
  ] as const) {
    if (s !== 0.0 && !(lo <= s && s <= hi))
      throw new Error(
        `max |${name}| = ${pyG(s)} lies outside [2^-200, 2^200]: rescale the system ` +
          '(the inner products rᵀr and pᵀAp would underflow or overflow)',
      );
  }
}

function overflowedStart(method: string, x: Vector, r: Vector, scale: Scale): Result | null {
  if (allFinite(r)) return null;
  const rnorm = norm2(r);
  const trace = [step(0, x, rnorm, residualInfo(r, rnorm, scale))];
  const msg =
    'the initial residual r₀ = b − A x₀ overflows (‖x₀‖ = ' +
    `${pyG(norm2(x))}): x₀ is too large for the float64 range`;
  return result(method, x, rnorm, false, msg, 0, trace, { n_matvec: 1 });
}

/** κ₂(S) = λ_max/λ_min for a symmetric S; null unless S is positive definite. */
export function kappa2(S: readonly (readonly number[])[]): number | null {
  const lam = eigvalsh(S);
  return lam[0] > 0.0 ? lam[lam.length - 1] / lam[0] : null;
}

function notSymmetric(method: string, A: Matrix, x: Vector, r: Vector, scale: Scale): Result {
  const rnorm = norm2(r);
  const trace = [step(0, x, rnorm, residualInfo(r, rnorm, scale))];
  const msg = `A is not symmetric (max |a_ij − a_ji| = ${pyG(asymmetry(A))}); ${method} needs an SPD matrix`;
  return result(method, x, rnorm, false, msg, 0, trace, { n_matvec: 1 });
}

/** φ(x) = ½xᵀAx − bᵀx = −½xᵀ(b + r). */
function phi(x: Vector, b: Vector, r: Vector): number {
  return (
    -0.5 *
    blasDot(
      x,
      b.map((v, i) => v + r[i]),
    )
  );
}

function finish(
  method: string,
  A: Matrix,
  b: Vector,
  x: Vector,
  trace: Step[],
  tol: number,
  scale: Scale,
  nMatvec: number,
  extra: Record<string, unknown>,
  passed: boolean,
  messageIn: string,
): Result {
  const trueNorm = norm2(residual(A, b, x));
  const out = { ...extra, n_matvec: nMatvec + 1, true_residual_norm: trueNorm };
  let converged = false;
  let message = messageIn;
  if (passed) {
    if (Number.isFinite(trueNorm) && trueNorm <= tol * scale.value) {
      [converged, message] = certify(trueNorm, x, normF(A), b, tol, scale, message);
    } else {
      message =
        `the updated residual met tol, but the true residual ${ratio(scale, trueNorm)} ` +
        'does not (rounding-error residual gap)';
    }
  }
  return result(method, x, trueNorm, converged, message, trace[trace.length - 1].k, trace, out);
}

const steepestDescentLinear: MethodFn<SystemLike> = (problem, o: Opts) => {
  const method = 'steepest_descent_linear';
  const tol = Number(o.tol ?? 1e-10);
  checkTol(tol);
  const maxIter = checkCount('max_iter', o.max_iter ?? 1000);
  const { A, b } = resolveSystem(problem);
  checkScale(A, b);
  let x = start(o.x0, b.length);
  let r = residual(A, b, x);
  let nMatvec = 1;
  const scale = scaleOf(A, b, x, r);
  if (!isSymmetric(A)) return notSymmetric(method, A, x, r, scale);
  const kappa = kappa2(A);
  const rate = kappa !== null ? (kappa - 1.0) / (kappa + 1.0) : null;
  const extra: Record<string, unknown> = { condition_number: kappa, rate_bound: rate };

  const failed = overflowedStart(method, x, r, scale);
  if (failed) return failed;
  let sR = pow2Scale(r);
  let rr = scaledDot(r, r, sR);
  let rnorm = norm2(r);
  const trace = [
    step(
      0,
      x,
      rnorm,
      {
        ...residualInfo(r, rnorm, scale),
        direction: r.slice(),
        alpha: null,
        phi: phi(x, b, r),
        condition_number: kappa,
        rate_bound: rate,
      },
      { gradNorm: rnorm },
    ),
  ];
  if (rnorm <= tol * scale.value)
    return finish(
      method,
      A,
      b,
      x,
      trace,
      tol,
      scale,
      nMatvec,
      extra,
      true,
      'x0 already satisfies the tolerance',
    );

  for (let k = 1; k <= maxIter; k++) {
    const q = mv(A, r);
    nMatvec += 1;
    const rq = scaledDot(r, q, sR);
    if (!(rq > 0.0)) {
      const msg = `rᵀAr = ${pyG(rq * sR ** 2)} ≤ 0 at k = ${k}: A is not positive definite`;
      return finish(method, A, b, x, trace, tol, scale, nMatvec, extra, false, msg);
    }
    const alpha = rr / rq;
    const stepSize = Math.abs(alpha) * rnorm;
    x = x.map((v, i) => v + alpha * r[i]);
    r = r.map((v, i) => v - alpha * q[i]);
    sR = pow2Scale(r);
    rr = scaledDot(r, r, sR);
    rnorm = norm2(r);
    trace.push(
      step(
        k,
        x,
        rnorm,
        { ...residualInfo(r, rnorm, scale), direction: r.slice(), alpha, phi: phi(x, b, r) },
        { gradNorm: rnorm, stepSize },
      ),
    );
    if (!(allFinite(x) && Number.isFinite(rnorm)))
      return finish(
        method,
        A,
        b,
        x,
        trace,
        tol,
        scale,
        nMatvec,
        extra,
        false,
        `non-finite iterate at k = ${k}`,
      );
    if (rnorm <= tol * scale.value)
      return finish(
        method,
        A,
        b,
        x,
        trace,
        tol,
        scale,
        nMatvec,
        extra,
        true,
        convergedMsg(rnorm, scale),
      );
  }
  const msg = `reached max_iter=${maxIter} with ${ratio(scale, rnorm)}`;
  return finish(method, A, b, x, trace, tol, scale, nMatvec, extra, false, msg);
};

function cg(method: string, problem: SystemLike, o: Opts, preconditioned: boolean): Result {
  const tol = Number(o.tol ?? 1e-10);
  checkTol(tol);
  const maxIter = checkCount('max_iter', o.max_iter ?? 1000);
  const { A, b } = resolveSystem(problem);
  checkScale(A, b);
  const n = b.length;
  let x = start(o.x0, n);
  let r = residual(A, b, x);
  let nMatvec = 1;
  const scale = scaleOf(A, b, x, r);
  if (!isSymmetric(A)) return notSymmetric(method, A, x, r, scale);
  const d = A.map((row, i) => row[i]);
  const kappa = kappa2(A);
  const extra: Record<string, unknown> = { condition_number: kappa };
  const info0Tail: StepInfo = { condition_number: kappa };
  let kappaEff: number | null;
  if (preconditioned) {
    const bad = d.findIndex((v) => !(v > 0.0));
    if (bad >= 0) {
      const rnorm = norm2(r);
      const trace = [step(0, x, rnorm, residualInfo(r, rnorm, scale))];
      const msg =
        `a_${bad}${bad} = ${pyG(d[bad])} ≤ 0: the Jacobi preconditioner M = diag(A) is not ` +
        'positive definite (A is not SPD)';
      return result(method, x, rnorm, false, msg, 0, trace, { n_matvec: 1 });
    }
    const s = d.map(Math.sqrt);
    kappaEff = kappa2(A.map((row, i) => row.map((v, j) => v / (s[i] * s[j])))); // D^{−1/2} A D^{−1/2}
    extra.preconditioned_condition_number = kappaEff;
    info0Tail.preconditioned_condition_number = kappaEff;
  } else kappaEff = kappa;
  const rate = kappaEff ? (Math.sqrt(kappaEff) - 1.0) / (Math.sqrt(kappaEff) + 1.0) : null;
  extra.rate_bound = rate;
  info0Tail.rate_bound = rate;

  const failed = overflowedStart(method, x, r, scale);
  if (failed) return failed;
  let z = preconditioned ? r.map((v, i) => v / d[i]) : r;
  let p = z.slice();
  let sR = pow2Scale(r);
  let rz = scaledDot(r, z, sR);
  let rnorm = norm2(r);
  const pre = (zz: Vector): StepInfo =>
    preconditioned ? { preconditioned_residual: zz.slice() } : {};
  const trace = [
    step(
      0,
      x,
      rnorm,
      {
        ...residualInfo(r, rnorm, scale),
        ...pre(z),
        direction: p.slice(),
        alpha: null,
        beta: null,
        phi: phi(x, b, r),
        ...info0Tail,
      },
      { gradNorm: rnorm },
    ),
  ];
  if (rnorm <= tol * scale.value)
    return finish(
      method,
      A,
      b,
      x,
      trace,
      tol,
      scale,
      nMatvec,
      extra,
      true,
      'x0 already satisfies the tolerance',
    );

  for (let k = 1; k <= maxIter; k++) {
    const q = mv(A, p);
    nMatvec += 1;
    const sP = pow2Scale(p);
    const pq = scaledDot(p, q, sP);
    if (!(pq > 0.0)) {
      const msg = `pᵀAp = ${pyG(pq * sP ** 2)} ≤ 0 at k = ${k}: A is not positive definite`;
      return finish(method, A, b, x, trace, tol, scale, nMatvec, extra, false, msg);
    }
    const alpha = (rz / pq) * (sR / sP) ** 2;
    const stepSize = Math.abs(alpha) * norm2(p);
    x = x.map((v, i) => v + alpha * p[i]);
    r = r.map((v, i) => v - alpha * q[i]);
    z = preconditioned ? r.map((v, i) => v / d[i]) : r;
    const sNew = pow2Scale(r);
    const rzNew = scaledDot(r, z, sNew);
    const beta = (rzNew / rz) * (sNew / sR) ** 2;
    p = z.map((v, i) => v + beta * p[i]);
    rz = rzNew;
    sR = sNew;
    rnorm = norm2(r);
    trace.push(
      step(
        k,
        x,
        rnorm,
        {
          ...residualInfo(r, rnorm, scale),
          ...pre(z),
          direction: p.slice(),
          alpha,
          beta,
          phi: phi(x, b, r),
        },
        { gradNorm: rnorm, stepSize },
      ),
    );
    if (!(allFinite(x) && Number.isFinite(rnorm)))
      return finish(
        method,
        A,
        b,
        x,
        trace,
        tol,
        scale,
        nMatvec,
        extra,
        false,
        `non-finite iterate at k = ${k}`,
      );
    if (rnorm <= tol * scale.value)
      return finish(
        method,
        A,
        b,
        x,
        trace,
        tol,
        scale,
        nMatvec,
        extra,
        true,
        convergedMsg(rnorm, scale),
      );
  }
  const msg = `reached max_iter=${maxIter} with ${ratio(scale, rnorm)}`;
  return finish(method, A, b, x, trace, tol, scale, nMatvec, extra, false, msg);
}

const conjugateGradientLinear: MethodFn<SystemLike> = (problem, o: Opts) =>
  cg('conjugate_gradient_linear', problem, o, false);

const preconditionedCg: MethodFn<SystemLike> = (problem, o: Opts) =>
  cg('preconditioned_cg', problem, o, true);

// ── GMRES ─────────────────────────────────────────────────────────────────────────────

const gmres: MethodFn<SystemLike> = (problem, o: Opts) => {
  const method = 'gmres';
  const restart = checkCount('restart', o.restart ?? 20);
  const maxIter = checkCount('max_iter', o.max_iter ?? 1000);
  const tol = Number(o.tol ?? 1e-10);
  checkTol(tol);
  const { A, b } = resolveSystem(problem);
  const aNorm = checkNorms(A, b);
  const n = b.length;
  let x = start(o.x0, n);
  const m = Math.min(restart, n);
  const tau = pivotTolerance(A); // rank threshold for the rotated diagonal r_jj

  let r = residual(A, b, x);
  let nMatvec = 1;
  const scale = scaleOf(A, b, x, r);
  const failed = overflowedStart(method, x, r, scale);
  if (failed) return failed;
  let beta = norm2(r);
  const trace: Step[] = [
    step(0, x, beta, {
      ...residualInfo(r, beta, scale),
      cycle: 0,
      krylov_dim: 0,
      hessenberg: null,
      givens: null,
      basis_vector: beta > 0.0 ? r.map((v) => v / beta) : null,
    }),
  ];

  const done = (converged: boolean, message: string, cycles: number): Result =>
    // Result.fun is the true residual β = ‖b − A x‖₂ (trace[-1].fun is the Givens value).
    result(method, x, beta, converged, message, trace[trace.length - 1].k, trace, {
      n_matvec: nMatvec,
      true_residual_norm: beta,
      restart: m,
      cycles,
    });

  if (beta <= tol * scale.value) {
    const [ok, msg] = certify(beta, x, aNorm, b, tol, scale, 'x0 already satisfies the tolerance');
    return done(ok, msg, 0);
  }

  let k = 0;
  let cycle = 0;
  for (;;) {
    const V: Vector[] = Array.from({ length: m + 1 }, () => zeros(n)); // columns v_1 … v_{m+1}
    V[0] = r.map((v) => v / beta);
    const H = zerosM(m + 1, m); // H̄ as built by Arnoldi
    const R = zerosM(m + 1, m); // H̄ after the Givens rotations
    const g = zeros(m + 1);
    g[0] = beta;
    const cs = zeros(m);
    const sn = zeros(m);
    const xStart = x.slice();
    let xj = x.slice();
    for (let j = 0; j < m; j++) {
      let w = gemvStrided(A, V[j]); // A @ V[:, j]: a strided column
      nMatvec += 1;
      const wNorm0 = norm2(w);
      let wNorm1 = 0.0;
      for (let sweep = 0; sweep < 2; sweep++) {
        // modified Gram–Schmidt, then one reorthogonalization pass
        for (let i = 0; i <= j; i++) {
          const hij = dotStrided(w, V[i]); // w @ V[:, i]: a strided column
          H[i][j] += hij;
          const vi = V[i];
          w = w.map((v, q) => v - hij * vi[q]);
        }
        if (sweep === 0) wNorm1 = norm2(w);
      }
      let hNext = norm2(w);
      if (hNext < wNorm1 / Math.sqrt(2.0)) {
        // Kahan–Parlett: w ∈ span(V_j) numerically
        w = zeros(n);
        hNext = 0.0;
      }
      H[j + 1][j] = hNext;
      for (let i = 0; i <= j + 1; i++) R[i][j] = H[i][j];
      for (let i = 0; i < j; i++) {
        // apply the earlier rotations to the new column
        const t = cs[i] * R[i][j] + sn[i] * R[i + 1][j];
        R[i + 1][j] = -sn[i] * R[i][j] + cs[i] * R[i + 1][j];
        R[i][j] = t;
      }
      const den = hypot(R[j][j], R[j + 1][j]);
      k += 1;
      if (den <= tau) {
        x = xj;
        r = residual(A, b, x);
        nMatvec += 1;
        beta = norm2(r);
        trace.push(
          step(
            k,
            x,
            beta,
            {
              ...residualInfo(r, beta, scale),
              cycle,
              krylov_dim: j + 1,
              hessenberg: H.slice(0, j + 2).map((row) => row.slice(0, j + 1)),
              givens: null,
              basis_vector: null,
            },
            { stepSize: 0.0 },
          ),
        );
        return done(
          false,
          `GMRES breakdown at k = ${k}: the rotated diagonal |r_jj| = ${pyG(den)} ≤ ` +
            `τ = ${pyG(tau)}, so the projected least-squares matrix is singular to ` +
            'working precision (A is singular on the Krylov space)',
          cycle + 1,
        );
      }
      cs[j] = R[j][j] / den;
      sn[j] = R[j + 1][j] / den;
      R[j][j] = den;
      R[j + 1][j] = 0.0;
      g[j + 1] = -sn[j] * g[j];
      g[j] = cs[j] * g[j];
      const resEst = Math.abs(g[j + 1]);
      const y = backSubstitution(
        R.slice(0, j + 1).map((row) => row.slice(0, j + 1)),
        g.slice(0, j + 1),
      );
      const xPrev = xj;
      // Rows of V[:, :j+1] (Python stores V as an n × (m+1) array).
      const Vrows = Array.from({ length: n }, (_, q) => V.slice(0, j + 1).map((col) => col[q]));
      const Vy = gemv(Vrows, y);
      xj = xStart.map((v, q) => v + Vy[q]); // x_j = x_start + V[:, :j+1] @ y
      // c = βe₁ − H_j y_j;  r_j = V_j c − y_j w
      const cvec = gemv(
        H.slice(0, j + 1).map((row) => row.slice(0, j + 1)),
        y,
      ).map((v) => -v);
      cvec[0] += beta;
      const yj = y[j];
      const Vc = gemv(Vrows, cvec);
      const rj = Vc.map((v, q) => v - yj * w[q]);
      // NOTE: K_{j+1} ⊆ ℝⁿ, so for j + 1 = n there is no further Arnoldi vector.
      const breakdown = j + 1 === n || H[j + 1][j] <= EPS * wNorm0;
      const hj = H[j + 1][j];
      const vNext = breakdown ? null : w.map((v) => v / hj);
      const stepSize = norm2(xj.map((v, q) => v - xPrev[q]));
      trace.push(
        step(
          k,
          xj,
          resEst,
          {
            ...residualInfo(rj, resEst, scale),
            cycle,
            krylov_dim: j + 1,
            hessenberg: H.slice(0, j + 2).map((row) => row.slice(0, j + 1)),
            givens: [cs[j], sn[j]],
            basis_vector: vNext,
          },
          { stepSize },
        ),
      );
      if (!allFinite(xj)) {
        x = xj;
        beta = NaN;
        return done(false, `non-finite iterate at k = ${k}`, cycle + 1);
      }
      if (resEst <= tol * scale.value || vNext === null || k >= maxIter) break;
      V[j + 1] = vNext;
    }
    x = xj;
    r = residual(A, b, x);
    nMatvec += 1;
    beta = norm2(r);
    cycle += 1;
    if (beta <= tol * scale.value) {
      const [ok, msg] = certify(
        beta,
        x,
        aNorm,
        b,
        tol,
        scale,
        `true residual ${ratio(scale, beta)} ≤ tol`,
      );
      return done(ok, msg, cycle);
    }
    if (!Number.isFinite(beta))
      return done(false, `non-finite residual after cycle ${cycle}`, cycle);
    if (k >= maxIter)
      return done(false, `reached max_iter=${maxIter} with true ${ratio(scale, beta)}`, cycle);
  }
};

// ── Registration ──────────────────────────────────────────────────────────────────────
// Doc prose (intuition, pros, cons) marks inline math as `$…$`; the linear-systems lab's
// card typesets it with KaTeX (labs/linalg/MathProse.tsx).

const RHO = { tex: '\\rho(G)', key: 'info.spectral_radius' };
const REL = { tex: '\\|\\mathbf{r}_k\\|/\\|\\mathbf{b}\\|', key: 'info.relative_residual' };

registerMethod(
  {
    id: 'jacobi',
    family: 'linalg',
    name: 'Jacobi iteration',
    params: [TOL, MAX_ITER],
    needs: ['A', 'b'],
    order: 'linear (rate ρ(G_J))',
    summary: 'Solve equation i for xᵢ using the previous iterate for every other unknown.',
    references: [
      'Burden & Faires, Numerical Analysis (10th ed.), Alg. 7.1',
      'Saad, Iterative Methods for Sparse Linear Systems (2nd ed., 2003), §4.1',
    ],
  },
  jacobi,
  {
    rule: 'x_i^{(k+1)} = \\frac{1}{a_{ii}}\\Big(b_i - \\sum_{j \\ne i} a_{ij}\\,x_j^{(k)}\\Big)',
    intuition:
      'Every equation is solved for its own unknown while the others are frozen at the old ' +
      'iterate, all at once. In the plane each step jumps horizontally to the first line and ' +
      'vertically to the second at the same time; the error shrinks by $\\rho(G_J)$ per step.',
    order: 'linear · rate $\\rho(G_J)$',
    pros: ['Embarrassingly parallel', 'Converges for strictly diagonally dominant $A$'],
    cons: [
      'Diverges when $\\rho(G_J) \\ge 1$, even where Gauss–Seidel converges',
      'Slow: $\\rho(G_J) \\to 1$ as the grid refines',
    ],
    quantities: [REL],
  },
);

registerMethod(
  {
    id: 'gauss_seidel',
    family: 'linalg',
    name: 'Gauss–Seidel iteration',
    params: [TOL, MAX_ITER],
    needs: ['A', 'b'],
    order: 'linear (rate ρ(G_GS))',
    summary: 'Like Jacobi, but each new component is used as soon as it is computed.',
    references: [
      'Burden & Faires, Numerical Analysis (10th ed.), Alg. 7.2',
      'Saad, Iterative Methods for Sparse Linear Systems (2nd ed., 2003), §4.1',
    ],
  },
  gaussSeidel,
  {
    rule: 'x_i^{(k+1)} = \\frac{1}{a_{ii}}\\Big(b_i - \\sum_{j < i} a_{ij}\\,x_j^{(k+1)} - \\sum_{j > i} a_{ij}\\,x_j^{(k)}\\Big)',
    intuition:
      'Each new component is used the moment it is computed, so one sweep is a staircase of ' +
      'axis-parallel moves, each landing exactly on the line of its equation. For SPD $A$ it ' +
      'converges from any start, and for consistently ordered $A$, ' +
      '$\\rho(G_{GS}) = \\rho(G_J)^2$.',
    order: 'linear · rate $\\rho(G_{GS})$',
    pros: [
      'Converges for every SPD $A$ (Ostrowski–Reich)',
      'Twice as fast as Jacobi on model problems',
    ],
    cons: ['Sequential within a sweep', 'Still slow when $\\rho(G_{GS}) \\approx 1$'],
    quantities: [REL],
  },
);

registerMethod(
  {
    id: 'sor',
    family: 'linalg',
    name: 'Successive over-relaxation (SOR)',
    params: [
      param.float('omega', 1.5, {
        min: 0.05,
        max: 1.95,
        help: 'Relaxation factor ω ∈ (0, 2); ω = 1 is Gauss–Seidel, ω > 1 over-relaxes.',
        label: 'Relaxation',
        tex: '\\omega',
      }),
      TOL,
      MAX_ITER,
    ],
    needs: ['A', 'b'],
    order: 'linear (rate ρ(G_ω))',
    summary: 'Gauss–Seidel with each component step stretched by a relaxation factor ω.',
    references: [
      'Burden & Faires, Numerical Analysis (10th ed.), Alg. 7.3 and §7.4',
      'Young, Iterative Solution of Large Linear Systems (1971)',
      'Saad, Iterative Methods for Sparse Linear Systems (2nd ed., 2003), §4.1–4.2',
    ],
  },
  sor,
  {
    rule: 'x_i^{(k+1)} = (1 - \\omega)\\,x_i^{(k)} + \\omega\\,\\tilde x_i^{\\,\\mathrm{GS}}',
    intuition:
      'Take the Gauss–Seidel move for each component and stretch it by $\\omega$: past the line ' +
      'for $\\omega > 1$. Kahan: $\\rho(G_\\omega) \\ge |\\omega - 1|$, so only $0 < \\omega < 2$ can ' +
      "work; for a tridiagonal $A$ Young's $\\omega^\\star = 2/(1 + \\sqrt{1 - \\rho_J^2})$ turns " +
      "$\\rho_J \\approx 1 - ch^2$ into $\\rho \\approx 1 - c'h$.",
    order: 'linear · rate $\\rho(G_\\omega)$',
    pros: ["With Young's $\\omega^\\star$, an order of magnitude faster than Gauss–Seidel"],
    cons: [
      '$\\omega^\\star$ needs $\\rho(G_J)$, rarely known',
      'A poor $\\omega$ can be slower than Gauss–Seidel',
    ],
    quantities: [RHO, REL],
  },
);

registerMethod(
  {
    id: 'steepest_descent_linear',
    family: 'linalg',
    name: 'Steepest descent (SPD system)',
    params: [TOL, MAX_ITER],
    needs: ['A', 'b'],
    order: 'linear (A-norm rate (κ − 1)/(κ + 1))',
    summary: 'Minimize ½xᵀAx − bᵀx by exact line searches along the residual r = b − Ax.',
    references: [
      'Saad, Iterative Methods for Sparse Linear Systems (2nd ed., 2003), §5.3.1',
      'Nocedal & Wright, Numerical Optimization (2nd ed., 2006), §3.3, eq. (3.29)',
      'Shewchuk (1994), An Introduction to the Conjugate Gradient Method Without the ' +
        'Agonizing Pain, §4',
    ],
  },
  steepestDescentLinear,
  {
    rule: '\\alpha_k = \\frac{\\mathbf{r}_k^{\\mathsf{T}}\\mathbf{r}_k}{\\mathbf{r}_k^{\\mathsf{T}}A\\mathbf{r}_k},\\qquad \\mathbf{x}_{k+1} = \\mathbf{x}_k + \\alpha_k\\mathbf{r}_k',
    intuition:
      'The residual $\\mathbf{r} = \\mathbf{b} - A\\mathbf{x}$ is the downhill direction of ' +
      '$\\varphi(\\mathbf{x}) = \\tfrac12\\mathbf{x}^{\\mathsf{T}}A\\mathbf{x} - \\mathbf{b}^{\\mathsf{T}}\\mathbf{x}$; ' +
      'an exact line search walks along it until the path is tangent to a level ellipse. ' +
      'Consecutive steps are orthogonal, so on an elongated bowl the path zigzags at rate ' +
      '$(\\kappa - 1)/(\\kappa + 1)$.',
    order: 'linear · $A$-norm rate $(\\kappa - 1)/(\\kappa + 1)$',
    pros: ['One product $A\\mathbf{r}$ per step', 'Monotone decrease of $\\varphi$'],
    cons: ['Zigzags when $\\kappa$ is large', 'SPD $A$ only'],
    quantities: [
      { tex: '\\alpha_{k-1}', key: 'info.alpha' },
      { tex: '\\varphi(\\mathbf{x}_k)', key: 'info.phi' },
    ],
  },
);

registerMethod(
  {
    id: 'conjugate_gradient_linear',
    family: 'linalg',
    name: 'Conjugate gradient (SPD system)',
    params: [TOL, MAX_ITER],
    needs: ['A', 'b'],
    order: '≤ n steps in exact arithmetic; A-norm rate (√κ − 1)/(√κ + 1)',
    summary:
      'Search along A-conjugate directions; each step minimizes ½xᵀAx − bᵀx over a growing Krylov space.',
    references: [
      'Hestenes & Stiefel (1952), J. Res. Nat. Bur. Standards 49(6)',
      'Nocedal & Wright, Numerical Optimization (2nd ed., 2006), Alg. 5.2 and eq. (5.36)',
      'Trefethen & Bau, Numerical Linear Algebra (1997), Alg. 38.1',
    ],
  },
  conjugateGradientLinear,
  {
    rule: '\\alpha_k = \\frac{\\mathbf{r}_k^{\\mathsf{T}}\\mathbf{r}_k}{\\mathbf{p}_k^{\\mathsf{T}}A\\mathbf{p}_k},\\quad \\mathbf{p}_{k+1} = \\mathbf{r}_{k+1} + \\beta_{k+1}\\mathbf{p}_k',
    intuition:
      'Each new direction is the residual corrected to be $A$-conjugate to every earlier one, ' +
      'so a minimization along it never undoes the previous ones. $\\mathbf{x}_k$ minimizes ' +
      '$\\varphi$ over $\\mathbf{x}_0 + \\mathcal{K}_k(A, \\mathbf{r}_0)$, and in exact arithmetic CG ' +
      'ends in at most $n$ steps.',
    order: 'at most $n$ steps · $A$-norm rate $(\\sqrt{\\kappa} - 1)/(\\sqrt{\\kappa} + 1)$',
    pros: ['Optimal over the Krylov space', 'Rate depends on $\\sqrt{\\kappa}$, not $\\kappa$'],
    cons: ['SPD $A$ only', 'Rounding destroys conjugacy on hard problems'],
    quantities: [
      { tex: '\\alpha_{k-1}', key: 'info.alpha' },
      { tex: '\\beta_k', key: 'info.beta' },
    ],
  },
);

registerMethod(
  {
    id: 'preconditioned_cg',
    family: 'linalg',
    name: 'Preconditioned CG (Jacobi preconditioner)',
    params: [TOL, MAX_ITER],
    needs: ['A', 'b'],
    order: 'A-norm rate (√κ̃ − 1)/(√κ̃ + 1), κ̃ = κ₂(D^{-1/2} A D^{-1/2})',
    summary:
      'CG on the diagonally scaled system: residuals are divided by diag(A) before each update.',
    references: [
      'Nocedal & Wright, Numerical Optimization (2nd ed., 2006), Alg. 5.3',
      'Saad, Iterative Methods for Sparse Linear Systems (2nd ed., 2003), §9.2',
    ],
  },
  preconditionedCg,
  {
    rule: '\\mathbf{z}_k = D^{-1}\\mathbf{r}_k,\\quad \\alpha_k = \\frac{\\mathbf{r}_k^{\\mathsf{T}}\\mathbf{z}_k}{\\mathbf{p}_k^{\\mathsf{T}}A\\mathbf{p}_k},\\quad \\mathbf{p}_{k+1} = \\mathbf{z}_{k+1} + \\beta_{k+1}\\mathbf{p}_k',
    intuition:
      'CG run on the rescaled system $D^{-1/2} A D^{-1/2}$: dividing the residual by the ' +
      'diagonal equalizes the scales of the unknowns. Its rate depends on $\\tilde\\kappa$ of the ' +
      'scaled matrix, which can be far smaller than $\\kappa(A)$ when the diagonal varies a lot.',
    order: '$A$-norm rate $(\\sqrt{\\tilde\\kappa} - 1)/(\\sqrt{\\tilde\\kappa} + 1)$',
    pros: ['Free to build and apply', 'Helps when $\\operatorname{diag}(A)$ varies in scale'],
    cons: ['No help on a constant diagonal (Poisson)', 'SPD $A$ only'],
    quantities: [
      { tex: '\\alpha_{k-1}', key: 'info.alpha' },
      { tex: '\\beta_k', key: 'info.beta' },
    ],
  },
);

registerMethod(
  {
    id: 'gmres',
    family: 'linalg',
    name: 'GMRES (restarted)',
    params: [
      param.int('restart', 20, {
        min: 1,
        max: 500,
        help: 'Krylov dimension m of a cycle before restarting from the current x (GMRES(m)).',
        label: 'Restart length',
        tex: 'm',
      }),
      TOL,
      MAX_ITER,
    ],
    needs: ['A', 'b'],
    order: 'minimal residual over x₀ + K_j(A, r₀); full GMRES ends in ≤ n steps',
    summary:
      'Minimize ‖b − Ax‖₂ over a growing Krylov space built by Arnoldi; restart every m steps.',
    references: [
      'Saad & Schultz (1986), SIAM J. Sci. Stat. Comput. 7(3)',
      'Saad, Iterative Methods for Sparse Linear Systems (2nd ed., 2003), Alg. 6.2, 6.9, 6.11, §6.5.3',
      'Trefethen & Bau, Numerical Linear Algebra (1997), Alg. 35.1',
      'Brown & Walker (1997), GMRES on (nearly) singular systems, SIAM J. Matrix Anal. Appl. 18(1)',
    ],
  },
  gmres,
  {
    rule: '\\mathbf{x}_j = \\mathbf{x}_0 + V_j\\mathbf{y}_j,\\quad \\mathbf{y}_j = \\arg\\min_{\\mathbf{y}}\\big\\|\\beta\\mathbf{e}_1 - \\bar H_j\\mathbf{y}\\big\\|_2',
    intuition:
      'Arnoldi builds an orthonormal basis of the Krylov space $\\mathcal{K}_j(A, \\mathbf{r}_0)$ ' +
      'and GMRES picks the point of $\\mathbf{x}_0 + \\mathcal{K}_j$ with the smallest residual, ' +
      'so $\\|\\mathbf{r}_j\\|$ never increases. It needs no symmetry; restarting after $m$ steps ' +
      'caps the memory and can stall.',
    order: 'minimal residual · at most $n$ steps without restart',
    pros: ['Any nonsingular $A$', 'Monotone residual'],
    cons: ['Memory and work grow with $j$', 'GMRES($m$) can stagnate'],
    quantities: [
      { tex: 'j', key: 'info.krylov_dim' },
      { tex: '\\text{cycle}', key: 'info.cycle' },
    ],
  },
);
