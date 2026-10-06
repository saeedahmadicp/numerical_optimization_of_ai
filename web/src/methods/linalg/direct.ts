/**
 * Direct solvers for a square linear system A x = b — TS port of `src/numopt/linalg/direct.py`.
 *
 * Trace (one Step per stage, for the elimination animation):
 *   k = 0          the initial state (phase "start"),
 *   k = 1 … n      stage k works on pivot column c = k − 1 (phase "eliminate"),
 *   k = n + 1      the triangular solve that produces x (phase "back_substitution" or "solve");
 *                  Gauss–Jordan has no such step (its last stage already shows [I | x]).
 * `Step.x` and `Step.fun` are null until the solution exists; on the final step `fun` is the
 * residual norm ‖b − Ax‖₂.
 *
 * A pivot p counts as zero when |p| ≤ τ = n·ε·‖A‖_F. A finished solve is certified by its normwise
 * backward error η∞ = ‖b − Ax‖∞/(‖A‖∞‖x‖∞ + ‖b‖∞) ≤ 30·n·ε (× max(1, κ̂₁) for Gauss–Jordan).
 *
 * Info keys (snake_case, as in the Python docstring): phase, matrix, pivot, pivot_value, row_swap,
 * multipliers, zero_pivot, L, perm, column, householder_vector, c_prime, d_prime, y, residual.
 * Result.extra: cond_estimate, residual_norm, backward_error, backward_error_bound,
 * growth_factor, and the factors (L, U, P, perm / L / Q, R, householder_vectors / L, U, c_prime,
 * d_prime).
 */
import { registerMethod } from '../../core/registry';
import type { Matrix, MethodFn, Point, Result, Step, StepInfo, Vector } from '../../core/types';
import {
  EPS,
  allFinite,
  allFiniteM,
  argmaxAbs,
  asymmetry,
  backSubstitution,
  backwardError,
  condEstimate,
  copyM,
  eye,
  forwardSubstitution,
  hypot,
  isSymmetric,
  maxAbs,
  maxAbsM,
  mv,
  norm2,
  pivotTolerance,
  pyG,
  resolveSystem,
  transposeM,
  zeros,
  zerosM,
  vecMat,
  type SystemLike,
} from './numerics';

/** Python's `None` for `Step.x` before the solution exists. */
const NONE = null as unknown as Point;

/** The a-posteriori stability test passes when η∞ ≤ BACKWARD_ERROR_FACTOR·n·ε. */
export const BACKWARD_ERROR_FACTOR = 30.0;

function step(k: number, x: Vector | null, fun: number | null, info: StepInfo): Step {
  return { k, x: x === null ? NONE : x.slice(), fun, gradNorm: null, stepSize: null, info };
}

function result(
  method: string,
  x: Vector | null,
  fun: number | null,
  converged: boolean,
  message: string,
  nIter: number,
  trace: Step[],
  extra: Record<string, unknown>,
): Result {
  return {
    method,
    x,
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

function luSolvers(
  L: Matrix,
  U: Matrix,
  perm: readonly number[],
): [(v: Vector) => Vector, (v: Vector) => Vector] {
  const solve = (v: Vector) =>
    backSubstitution(
      U,
      forwardSubstitution(
        L,
        perm.map((p) => v[p]),
        true,
      ),
    );
  const solveT = (v: Vector) => {
    // A = Pᵀ L U, so Aᵀ = Uᵀ Lᵀ P and Aᵀ z = v ⇔ Uᵀ w = v, then Lᵀ (P z) = w.
    const w = forwardSubstitution(transposeM(U), v, false, true);
    const pz = backSubstitution(transposeM(L), w, true);
    const z = zeros(pz.length);
    perm.forEach((p, i) => (z[p] = pz[i]));
    return z;
  };
  return [solve, solveT];
}

/** max(current, max |m_ij|) over the coefficient columns of M. */
function growth(current: number, M: readonly (readonly number[])[], n: number): number {
  let m = 0;
  for (const r of M) m = Math.max(m, maxAbs(r.slice(0, n)));
  return Math.max(current, m);
}

function fail(
  method: string,
  message: string,
  trace: Step[],
  extra: Record<string, unknown> = {},
): Result {
  const out = { cond_estimate: Infinity, residual_norm: null, ...extra };
  return result(method, null, null, false, message, trace[trace.length - 1].k, trace, out);
}

function certify(
  method: string,
  A: Matrix,
  b: Vector,
  x: Vector,
  r: Vector,
  trace: Step[],
  extra: Record<string, unknown>,
  hint: string,
  kappa = 1.0,
): Result {
  const n = b.length;
  const rnorm = extra.residual_norm as number;
  const eta = backwardError(A, b, x, r);
  const bound = BACKWARD_ERROR_FACTOR * n * EPS * Math.max(1.0, kappa);
  const out = { ...extra, backward_error: eta, backward_error_bound: bound };
  const nIter = trace[trace.length - 1].k;
  if (eta <= bound) {
    const msg =
      `factorization completed; residual ‖b − Ax‖₂ = ${pyG(rnorm)}, ` +
      `backward error η∞ = ${pyG(eta)}`;
    return result(method, x, rnorm, true, msg, nIter, trace, out);
  }
  let msg: string;
  if (!Number.isFinite(eta)) {
    msg = 'the residual b − Ax of the computed x is not finite, so x cannot be certified';
  } else {
    const rho = extra.growth_factor as number | null | undefined;
    const g = rho !== null && rho !== undefined ? ` (growth factor ρ = ${pyG(rho)})` : '';
    msg =
      `unstable: the backward error η∞ = ‖b − Ax‖∞/(‖A‖∞‖x‖∞ + ‖b‖∞) = ${pyG(eta)} of the ` +
      `computed x exceeds ${pyG(bound)}${g}, so x is not the solution of a nearby ` +
      `system${hint}`;
  }
  return result(method, x, rnorm, false, msg, nIter, trace, out);
}

function stageInfo(
  M: Matrix,
  c: number,
  pivotValue: number,
  rowSwap: [number, number] | null,
  multipliers: Vector,
  zeroPivot: boolean,
  more: StepInfo = {},
): StepInfo {
  return {
    phase: 'eliminate',
    matrix: copyM(M),
    pivot: [c, c],
    pivot_value: pivotValue,
    row_swap: rowSwap,
    multipliers: multipliers.slice(),
    zero_pivot: zeroPivot,
    ...more,
  };
}

function startInfo(M: Matrix, more: StepInfo = {}): StepInfo {
  return {
    phase: 'start',
    matrix: copyM(M),
    pivot: null,
    pivot_value: null,
    row_swap: null,
    multipliers: null,
    zero_pivot: false,
    ...more,
  };
}

/** [A | b] as an n × (n + 1) matrix. */
function augmented(A: Matrix, b: Vector): Matrix {
  return A.map((r, i) => [...r, b[i]]);
}

function swapRows<T>(M: T[], i: number, j: number): void {
  [M[i], M[j]] = [M[j], M[i]];
}

/** Swap the first `upto` entries of rows i and j of L. */
function swapLeft(L: Matrix, i: number, j: number, upto: number): void {
  for (let q = 0; q < upto; q++) [L[i][q], L[j][q]] = [L[j][q], L[i][q]];
}

const residualOf = (A: Matrix, b: Vector, x: Vector) => {
  const Ax = mv(A, x);
  return b.map((v, i) => v - Ax[i]);
};

// ── Gaussian elimination (with and without partial pivoting) ──────────────────────────

function gaussian(method: string, problem: SystemLike, pivoting: boolean): Result {
  const { A, b } = resolveSystem(problem);
  const n = b.length;
  const tau = pivotTolerance(A);
  const aMax = maxAbsM(A);
  const M = augmented(A, b);
  const L = eye(n);
  const perm = Array.from({ length: n }, (_, i) => i);
  let grow = aMax;
  const permInfo = (): StepInfo => (pivoting ? { perm: perm.slice() } : {});
  const trace: Step[] = [step(0, null, null, startInfo(M, permInfo()))];

  for (let c = 0; c < n; c++) {
    let rowSwap: [number, number] | null = null;
    if (pivoting) {
      // Partial pivoting: the first row index of max |m_ic|, i ≥ c (LAPACK idamax rule).
      const p = argmaxAbs(
        M.map((r) => r[c]),
        c,
      );
      if (p !== c) {
        swapRows(M, c, p);
        swapLeft(L, c, p, c);
        swapRows(perm, c, p);
        rowSwap = [c, p];
      }
    }
    const pivot = M[c][c];
    const mult = zeros(n);
    if (Math.abs(pivot) <= tau) {
      trace.push(step(c + 1, null, null, stageInfo(M, c, pivot, rowSwap, mult, true, permInfo())));
      const hint = pivoting ? '' : '; partial pivoting (row interchanges) may avoid it';
      const what = pivoting ? 'A is singular to working precision' : 'elimination breaks down';
      return fail(
        method,
        `zero pivot |a_${c}${c}| = ${pyG(Math.abs(pivot))} ≤ τ = ${pyG(tau)} at stage ${c + 1}: ` +
          `${what}${hint}`,
        trace,
        { growth_factor: aMax > 0 ? grow / aMax : null },
      );
    }
    for (let i = c + 1; i < n; i++) mult[i] = M[i][c] / pivot;
    // Outer-product update of the rows below the pivot (GVL Alg. 3.2.1 / 3.4.1).
    for (let i = c + 1; i < n; i++)
      for (let j = c; j <= n; j++) M[i][j] = M[i][j] - mult[i] * M[c][j];
    // NOTE: exact zeros for the eliminated entries, as in Python.
    for (let i = c + 1; i < n; i++) {
      M[i][c] = 0.0;
      L[i][c] = mult[i];
    }
    grow = growth(grow, M, n);
    if (!allFiniteM(M)) {
      trace.push(step(c + 1, null, null, stageInfo(M, c, pivot, rowSwap, mult, false, permInfo())));
      return fail(method, `non-finite value at stage ${c + 1}`, trace);
    }
    trace.push(step(c + 1, null, null, stageInfo(M, c, pivot, rowSwap, mult, false, permInfo())));
  }

  const U = M.map((r, i) => r.slice(0, n).map((v, j) => (j >= i ? v : 0)));
  const x = backSubstitution(
    U,
    M.map((r) => r[n]),
  );
  const r = residualOf(A, b, x);
  const rnorm = norm2(r);
  if (!allFinite(x)) return fail(method, 'non-finite value in back substitution', trace);
  trace.push(
    step(n + 1, x, rnorm, {
      phase: 'back_substitution',
      matrix: copyM(M),
      pivot: null,
      pivot_value: null,
      row_swap: null,
      multipliers: null,
      zero_pivot: false,
      residual: r,
      ...permInfo(),
    }),
  );
  const [solve, solveT] = luSolvers(L, U, perm);
  const extra: Record<string, unknown> = {
    cond_estimate: condEstimate(A, solve, solveT),
    residual_norm: rnorm,
    growth_factor: grow / aMax,
    L,
    U,
  };
  let hint: string;
  if (pivoting) {
    extra.perm = perm;
    hint = ': partial pivoting is unstable on this matrix; use Householder QR';
  } else hint = ': a small pivot made the growth large; use partial pivoting';
  return certify(method, A, b, x, r, trace, extra, hint);
}

const gaussianElimination: MethodFn<SystemLike> = (problem) =>
  gaussian('gaussian_elimination', problem, false);

const gaussianEliminationPivoting: MethodFn<SystemLike> = (problem) =>
  gaussian('gaussian_elimination_pivoting', problem, true);

// ── Gauss–Jordan elimination ──────────────────────────────────────────────────────────

const gaussJordan: MethodFn<SystemLike> = (problem) => {
  const method = 'gauss_jordan';
  const { A, b } = resolveSystem(problem);
  const n = b.length;
  const tau = pivotTolerance(A);
  const aMax = maxAbsM(A);
  const M = augmented(A, b);
  const L = eye(n);
  const U = zerosM(n);
  const perm = Array.from({ length: n }, (_, i) => i);
  let grow = aMax;
  const trace: Step[] = [step(0, null, null, startInfo(M, { perm: perm.slice() }))];

  for (let c = 0; c < n; c++) {
    let rowSwap: [number, number] | null = null;
    const p = argmaxAbs(
      M.map((r) => r[c]),
      c,
    );
    if (p !== c) {
      swapRows(M, c, p);
      swapLeft(L, c, p, c);
      swapRows(perm, c, p);
      rowSwap = [c, p];
    }
    const pivot = M[c][c];
    const mult = zeros(n);
    if (Math.abs(pivot) <= tau) {
      trace.push(
        step(
          c + 1,
          null,
          null,
          stageInfo(M, c, pivot, rowSwap, mult, true, { perm: perm.slice() }),
        ),
      );
      return fail(
        method,
        `zero pivot |a_${c}${c}| = ${pyG(Math.abs(pivot))} ≤ τ = ${pyG(tau)} at stage ${c + 1}: ` +
          'A is singular to working precision',
        trace,
        { growth_factor: aMax > 0 ? grow / aMax : null },
      );
    }
    for (let j = c; j < n; j++) U[c][j] = M[c][j];
    for (let i = c + 1; i < n; i++) L[i][c] = M[i][c] / pivot;
    for (let j = 0; j <= n; j++) M[c][j] = M[c][j] / pivot;
    M[c][c] = 1.0; // NOTE: exact 1 instead of p/p.
    for (let i = 0; i < n; i++) if (i !== c) mult[i] = M[i][c];
    for (let i = 0; i < n; i++) {
      if (i === c) continue;
      for (let j = 0; j <= n; j++) M[i][j] = M[i][j] - mult[i] * M[c][j];
      M[i][c] = 0.0; // NOTE: exact zeros, as in _gaussian.
    }
    // NOTE: the growth factor uses the pivot row before it is normalized (U row c) and the
    // other rows after the update.
    grow = Math.max(
      grow,
      maxAbs(U[c].slice(c)),
      growth(
        0.0,
        M.filter((_, i) => i !== c),
        n,
      ),
    );
    if (!allFiniteM(M)) {
      trace.push(
        step(
          c + 1,
          null,
          null,
          stageInfo(M, c, pivot, rowSwap, mult, false, { perm: perm.slice() }),
        ),
      );
      return fail(method, `non-finite value at stage ${c + 1}`, trace);
    }
    let xc: Vector | null = null;
    let rnormC: number | null = null;
    const more: StepInfo = { perm: perm.slice() };
    if (c === n - 1) {
      // The last stage leaves [I | x]: its Step carries the solution and the residual.
      xc = M.map((r) => r[n]);
      const res = residualOf(A, b, xc);
      more.residual = res;
      rnormC = norm2(res);
    }
    trace.push(step(c + 1, xc, rnormC, stageInfo(M, c, pivot, rowSwap, mult, false, more)));
  }

  const x = M.map((r) => r[n]);
  const r = residualOf(A, b, x);
  const rnorm = norm2(r);
  const [solve, solveT] = luSolvers(L, U, perm);
  const kappa = condEstimate(A, solve, solveT);
  const extra: Record<string, unknown> = {
    cond_estimate: kappa,
    residual_norm: rnorm,
    growth_factor: grow / aMax,
    perm,
  };
  const hint = ': Gauss–Jordan is not backward stable; use Householder QR';
  return certify(method, A, b, x, r, trace, extra, hint, kappa);
};

// ── LU factorization ──────────────────────────────────────────────────────────────────

const luDecomposition: MethodFn<SystemLike> = (problem) => {
  const method = 'lu_decomposition';
  const { A, b } = resolveSystem(problem);
  const n = b.length;
  const tau = pivotTolerance(A);
  const aMax = maxAbsM(A);
  const W = copyM(A);
  const L = eye(n);
  const perm = Array.from({ length: n }, (_, i) => i);
  let grow = aMax;
  const trace: Step[] = [step(0, null, null, startInfo(W, { L: copyM(L), perm: perm.slice() }))];

  for (let c = 0; c < n; c++) {
    let rowSwap: [number, number] | null = null;
    const p = argmaxAbs(
      W.map((r) => r[c]),
      c,
    );
    if (p !== c) {
      swapRows(W, c, p);
      swapLeft(L, c, p, c);
      swapRows(perm, c, p);
      rowSwap = [c, p];
    }
    const pivot = W[c][c];
    const mult = zeros(n);
    if (Math.abs(pivot) <= tau) {
      trace.push(
        step(
          c + 1,
          null,
          null,
          stageInfo(W, c, pivot, rowSwap, mult, true, { L: copyM(L), perm: perm.slice() }),
        ),
      );
      return fail(
        method,
        `zero pivot |u_${c}${c}| = ${pyG(Math.abs(pivot))} ≤ τ = ${pyG(tau)} at stage ${c + 1}: ` +
          'A is singular to working precision',
        trace,
        { growth_factor: aMax > 0 ? grow / aMax : null },
      );
    }
    for (let i = c + 1; i < n; i++) {
      mult[i] = W[i][c] / pivot;
      L[i][c] = mult[i];
    }
    for (let i = c + 1; i < n; i++)
      for (let j = c; j < n; j++) W[i][j] = W[i][j] - mult[i] * W[c][j];
    for (let i = c + 1; i < n; i++) W[i][c] = 0.0; // NOTE: exact zeros, as in _gaussian.
    grow = growth(grow, W, n);
    trace.push(
      step(
        c + 1,
        null,
        null,
        stageInfo(W, c, pivot, rowSwap, mult, false, { L: copyM(L), perm: perm.slice() }),
      ),
    );
    if (!allFiniteM(W)) return fail(method, `non-finite value at stage ${c + 1}`, trace);
  }

  const U = W.map((r, i) => r.map((v, j) => (j >= i ? v : 0)));
  const y = forwardSubstitution(
    L,
    perm.map((p) => b[p]),
    true,
  );
  const x = backSubstitution(U, y);
  if (!allFinite(x)) return fail(method, 'non-finite value in the triangular solves', trace);
  const r = residualOf(A, b, x);
  const rnorm = norm2(r);
  trace.push(
    step(n + 1, x, rnorm, {
      phase: 'solve',
      matrix: copyM(U),
      pivot: null,
      pivot_value: null,
      row_swap: null,
      multipliers: null,
      zero_pivot: false,
      L: copyM(L),
      perm: perm.slice(),
      y,
      residual: r,
    }),
  );
  const [solve, solveT] = luSolvers(L, U, perm);
  const P = perm.map((p) => eye(n)[p]);
  const extra: Record<string, unknown> = {
    cond_estimate: condEstimate(A, solve, solveT),
    residual_norm: rnorm,
    growth_factor: grow / aMax,
    P,
    L,
    U,
    perm,
  };
  const hint = ': partial pivoting is unstable on this matrix; use Householder QR';
  return certify(method, A, b, x, r, trace, extra, hint);
};

// ── Cholesky ──────────────────────────────────────────────────────────────────────────

const cholesky: MethodFn<SystemLike> = (problem) => {
  const method = 'cholesky';
  const { A, b } = resolveSystem(problem);
  const n = b.length;
  const tau = pivotTolerance(A);
  const W = copyM(A);
  const L = zerosM(n);
  const trace: Step[] = [step(0, null, null, { ...startInfo(W), L: copyM(L), column: null })];
  if (!isSymmetric(A)) {
    return fail(
      method,
      `A is not symmetric (max |a_ij − a_ji| = ${pyG(asymmetry(A))}); Cholesky needs an SPD matrix`,
      trace,
    );
  }

  for (let c = 0; c < n; c++) {
    const d = W[c][c];
    if (!(d > tau)) {
      trace.push(
        step(c + 1, null, null, {
          ...stageInfo(W, c, d, null, zeros(n), true),
          L: copyM(L),
          column: null,
        }),
      );
      const pivot = `pivot d_${c} = a_${c}${c} − Σ l_${c}j² = ${pyG(d)}`;
      let why: string;
      if (d > 0.0) {
        // NOTE: 0 < d ≤ τ is the numerical-rank heuristic, not a proof of indefiniteness.
        why =
          `${pivot} lies in (0, τ = ${pyG(tau)}] at stage ${c + 1}: A is not numerically ` +
          'positive definite (singular or too ill-conditioned for working precision)';
      } else why = `A is not positive definite: ${pivot} ≤ 0 at stage ${c + 1}`;
      return fail(method, why, trace);
    }
    const lcc = Math.sqrt(d);
    L[c][c] = lcc;
    for (let i = c + 1; i < n; i++) L[i][c] = W[i][c] / lcc;
    for (let i = c + 1; i < n; i++)
      for (let j = c + 1; j < n; j++) W[i][j] = W[i][j] - L[i][c] * L[j][c];
    for (let j = 0; j < n; j++) W[c][j] = 0.0;
    for (let i = 0; i < n; i++) W[i][c] = 0.0;
    trace.push(
      step(c + 1, null, null, {
        ...stageInfo(W, c, d, null, zeros(n), false),
        multipliers: null,
        L: copyM(L),
        column: L.map((r) => r[c]),
      }),
    );
    if (!allFiniteM(W)) return fail(method, `non-finite value at stage ${c + 1}`, trace);
  }

  const LT = transposeM(L);
  const y = forwardSubstitution(L, b);
  const x = backSubstitution(LT, y, true); // L.T is a strided view in Python
  if (!allFinite(x)) return fail(method, 'non-finite value in the triangular solves', trace);
  const r = residualOf(A, b, x);
  const rnorm = norm2(r);
  trace.push(
    step(n + 1, x, rnorm, {
      ...startInfo(LT),
      phase: 'solve',
      L: copyM(L),
      column: null,
      y,
      residual: r,
    }),
  );
  const solve = (v: Vector) => backSubstitution(LT, forwardSubstitution(L, v), true);
  const extra: Record<string, unknown> = {
    cond_estimate: condEstimate(A, solve, solve), // A = Aᵀ
    residual_norm: rnorm,
    L,
  };
  return certify(method, A, b, x, r, trace, extra, '');
};

// ── Householder QR ────────────────────────────────────────────────────────────────────

const qrHouseholder: MethodFn<SystemLike> = (problem) => {
  const method = 'qr_householder';
  const { A, b } = resolveSystem(problem);
  const n = b.length;
  const tau = pivotTolerance(A);
  const M = augmented(A, b); // [R | Qᵀb] in progress
  const V = zerosM(n); // row c: the unit Householder vector of stage c (zeros above c)
  const trace: Step[] = [step(0, null, null, { ...startInfo(M), householder_vector: null })];

  for (let c = 0; c < n; c++) {
    const xcol = M.slice(c).map((r) => r[c]);
    const sigma = norm2(xcol.slice(1));
    const v = zeros(n);
    if (sigma > 0.0) {
      const alpha = hypot(xcol[0], sigma); // ‖x‖₂ without overflow
      const sign = xcol[0] >= 0.0 ? 1.0 : -1.0;
      const vc = xcol.slice();
      vc[0] += sign * alpha;
      const nv = norm2(vc);
      for (let i = 0; i < vc.length; i++) vc[i] /= nv;
      // M[c:, c:] −= 2 v (vᵀ M[c:, c:])
      const w = vecMat(
        vc,
        M.slice(c).map((r) => r.slice(c)),
      );
      for (let i = c; i < n; i++)
        for (let j = c; j <= n; j++) M[i][j] = M[i][j] - 2.0 * vc[i - c] * w[j - c];
      // NOTE: exact values for the reflected column: H x = −sign(x₁)‖x‖ e₁.
      M[c][c] = -sign * alpha;
      for (let i = c + 1; i < n; i++) M[i][c] = 0.0;
      for (let i = c; i < n; i++) v[i] = vc[i - c];
    }
    V[c] = v;
    const rcc = M[c][c];
    const zero = Math.abs(rcc) <= tau;
    trace.push(
      step(c + 1, null, null, {
        ...stageInfo(M, c, rcc, null, zeros(n), zero),
        multipliers: null,
        householder_vector: v,
      }),
    );
    if (zero)
      return fail(
        method,
        `zero diagonal |r_${c}${c}| = ${pyG(Math.abs(rcc))} ≤ τ = ${pyG(tau)} at stage ${c + 1}: ` +
          'A is singular to working precision',
        trace,
      );
    if (!allFiniteM(M)) return fail(method, `non-finite value at stage ${c + 1}`, trace);
  }

  const R = M.map((r, i) => r.slice(0, n).map((v, j) => (j >= i ? v : 0)));
  const qtb = M.map((r) => r[n]);
  const x = backSubstitution(R, qtb);
  if (!allFinite(x)) return fail(method, 'non-finite value in back substitution', trace);
  const r = residualOf(A, b, x);
  const rnorm = norm2(r);
  trace.push(
    step(n + 1, x, rnorm, {
      ...startInfo(M),
      phase: 'back_substitution',
      householder_vector: null,
      residual: r,
    }),
  );
  // Q = H_0 H_1 ⋯ H_{n−1} I, applied from the last reflector to the first.
  const Q = eye(n);
  for (let c = n - 1; c >= 0; c--) {
    const vc = V[c].slice(c);
    const w = vecMat(vc, Q.slice(c));
    for (let i = c; i < n; i++)
      for (let j = 0; j < n; j++) Q[i][j] = Q[i][j] - 2.0 * vc[i - c] * w[j];
  }
  const RT = transposeM(R);
  const solve = (w: Vector) => backSubstitution(R, vecMat(w, Q)); // A⁻¹w = R⁻¹ Qᵀ w
  const solveT = (w: Vector) => mv(Q, forwardSubstitution(RT, w, false, true)); // A⁻ᵀw = Q R⁻ᵀ w
  const extra: Record<string, unknown> = {
    cond_estimate: condEstimate(A, solve, solveT),
    residual_norm: rnorm,
    Q,
    R,
    householder_vectors: V,
  };
  return certify(method, A, b, x, r, trace, extra, '');
};

// ── Thomas algorithm (tridiagonal) ────────────────────────────────────────────────────

const thomas: MethodFn<SystemLike> = (problem) => {
  const method = 'thomas';
  const { A, b } = resolveSystem(problem);
  const n = b.length;
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      if (Math.abs(i - j) > 1 && A[i][j] !== 0.0)
        throw new Error('thomas: A must be tridiagonal (a_ij = 0 for |i − j| > 1)');
  const tau = pivotTolerance(A);
  const sub = [0.0, ...Array.from({ length: n - 1 }, (_, i) => A[i + 1][i])]; // a_i, a_0 unused
  const diag = A.map((r, i) => r[i]);
  const sup = [...Array.from({ length: n - 1 }, (_, i) => A[i][i + 1]), 0.0]; // c_i, c_{n−1} unused
  const cPrime = zeros(n);
  const dPrime = zeros(n);
  const w = zeros(n);
  const M = augmented(A, b);
  const trace: Step[] = [step(0, null, null, { ...startInfo(M), c_prime: null, d_prime: null })];

  for (let i = 0; i < n; i++) {
    const mult = zeros(n);
    let wi: number, rhs: number;
    if (i === 0) {
      wi = diag[0];
      rhs = b[0];
    } else {
      mult[i] = sub[i];
      wi = diag[i] - sub[i] * cPrime[i - 1];
      rhs = b[i] - sub[i] * dPrime[i - 1];
    }
    if (Math.abs(wi) <= tau) {
      trace.push(
        step(i + 1, null, null, {
          ...stageInfo(M, i, wi, null, mult, true),
          c_prime: cPrime.slice(),
          d_prime: dPrime.slice(),
        }),
      );
      return fail(
        method,
        `zero pivot |w_${i}| = ${pyG(Math.abs(wi))} ≤ τ = ${pyG(tau)} at stage ${i + 1}: the ` +
          'Thomas algorithm does not pivot; use Gaussian elimination with pivoting',
        trace,
      );
    }
    w[i] = wi;
    if (i < n - 1) cPrime[i] = sup[i] / wi;
    dPrime[i] = rhs / wi;
    // Row i of [A | b] after the stage: (0 … 0, 1, c′_i, 0 … 0 | d′_i).
    if (i > 0) M[i][i - 1] = 0.0;
    M[i][i] = 1.0;
    if (i < n - 1) M[i][i + 1] = cPrime[i];
    M[i][n] = dPrime[i];
    trace.push(
      step(i + 1, null, null, {
        ...stageInfo(M, i, wi, null, mult, false),
        c_prime: cPrime.slice(),
        d_prime: dPrime.slice(),
      }),
    );
    if (!(Number.isFinite(cPrime[i]) && Number.isFinite(dPrime[i])))
      return fail(method, `non-finite value at stage ${i + 1}`, trace);
  }

  const x = zeros(n);
  x[n - 1] = dPrime[n - 1];
  for (let i = n - 2; i >= 0; i--) x[i] = dPrime[i] - cPrime[i] * x[i + 1];
  if (!allFinite(x)) return fail(method, 'non-finite value in back substitution', trace);
  const r = residualOf(A, b, x);
  const rnorm = norm2(r);
  trace.push(
    step(n + 1, x, rnorm, {
      ...startInfo(M),
      phase: 'back_substitution',
      c_prime: cPrime.slice(),
      d_prime: dPrime.slice(),
      residual: r,
    }),
  );
  const L = zerosM(n);
  const U = eye(n);
  for (let i = 0; i < n; i++) {
    L[i][i] = w[i];
    if (i > 0) L[i][i - 1] = sub[i];
    if (i < n - 1) U[i][i + 1] = cPrime[i];
  }
  const solve = (v: Vector) => backSubstitution(U, forwardSubstitution(L, v));
  const solveT = (v: Vector) =>
    backSubstitution(transposeM(L), forwardSubstitution(transposeM(U), v, true, true), true); // Aᵀ = Uᵀ Lᵀ
  const extra: Record<string, unknown> = {
    cond_estimate: condEstimate(A, solve, solveT),
    residual_norm: rnorm,
    L,
    U,
    c_prime: cPrime,
    d_prime: dPrime,
  };
  const hint = ': the Thomas algorithm does not pivot; use Gaussian elimination with pivoting';
  return certify(method, A, b, x, r, trace, extra, hint);
};

// ── Registration ──────────────────────────────────────────────────────────────────────
// Doc prose (intuition, pros, cons) marks inline math as `$…$`; the linear-systems lab's
// card typesets it with KaTeX (labs/linalg/MathProse.tsx).

const PIVOT = { tex: 'a_{cc}', key: 'info.pivot_value' };

registerMethod(
  {
    id: 'gaussian_elimination',
    family: 'linalg',
    name: 'Gaussian elimination (no pivoting)',
    needs: ['A', 'b'],
    order: 'direct, 2n³/3 flops',
    summary:
      'Subtract multiples of each pivot row to zero the column below it, then back-substitute.',
    references: [
      'Trefethen & Bau, Numerical Linear Algebra (1997), Alg. 20.1',
      'Golub & Van Loan, Matrix Computations (4th ed., 2013), Alg. 3.2.1 and 3.1.2',
    ],
  },
  gaussianElimination,
  {
    rule: 'm_{ic} = \\frac{a_{ic}}{a_{cc}},\\qquad R_i \\leftarrow R_i - m_{ic}\\,R_c\\quad (i > c)',
    intuition:
      'Each stage uses the pivot $a_{cc}$ to subtract a multiple of row $c$ from every row below ' +
      'it, so the column under the pivot becomes zero. After $n$ stages $[A \\mid \\mathbf{b}]$ is ' +
      'upper triangular and back substitution reads off $\\mathbf{x}$ from the bottom up.',
    order: 'direct · $2n^3/3$ flops',
    pros: [
      'The textbook algorithm: $n$ stages, exact in exact arithmetic',
      'Records A = LU as it goes',
    ],
    cons: [
      'Breaks down on a zero pivot even when A is nonsingular (needs_pivoting)',
      'A small pivot makes the multipliers and the growth factor large: unstable',
    ],
    quantities: [PIVOT],
  },
);

registerMethod(
  {
    id: 'gaussian_elimination_pivoting',
    family: 'linalg',
    name: 'Gaussian elimination (partial pivoting)',
    needs: ['A', 'b'],
    order: 'direct, 2n³/3 flops',
    summary: 'Before each stage, swap up the row with the largest entry in the pivot column.',
    references: [
      'Trefethen & Bau, Numerical Linear Algebra (1997), Alg. 21.1',
      'Golub & Van Loan, Matrix Computations (4th ed., 2013), Alg. 3.4.1',
      'Burden & Faires, Numerical Analysis (10th ed.), Alg. 6.2',
    ],
  },
  gaussianEliminationPivoting,
  {
    rule: 'p = \\arg\\max_{i \\ge c} |a_{ic}|,\\quad R_c \\leftrightarrow R_p,\\quad R_i \\leftarrow R_i - \\frac{a_{ic}}{a_{cc}}\\,R_c',
    intuition:
      'Before each stage the row with the largest entry in the pivot column is swapped up, so ' +
      'every multiplier satisfies $|m_{ic}| \\le 1$. The growth factor is then at most $2^{n-1}$ ' +
      'and, in practice, small: the method is backward stable for almost every matrix met in ' +
      'practice.',
    order: 'direct · $2n^3/3$ flops',
    pros: [
      'Never meets a zero pivot unless A is singular',
      'The default dense solver (LAPACK getrf)',
    ],
    cons: ["Growth $2^{n-1}$ is possible (Wilkinson's matrix), though rare"],
    quantities: [PIVOT],
  },
);

registerMethod(
  {
    id: 'gauss_jordan',
    family: 'linalg',
    name: 'Gauss–Jordan elimination',
    needs: ['A', 'b'],
    order: 'direct, n³ flops',
    summary:
      'Normalize each pivot row and clear its column both below and above: [A | b] → [I | x].',
    references: [
      'Burden & Faires, Numerical Analysis (10th ed.), §6.1 (Gauss–Jordan method)',
      'Higham, Accuracy and Stability of Numerical Algorithms (2nd ed., 2002), Ch. 14',
    ],
  },
  gaussJordan,
  {
    rule: 'R_c \\leftarrow R_c / a_{cc},\\qquad R_i \\leftarrow R_i - a_{ic}\\,R_c\\quad (i \\ne c)',
    intuition:
      'Each stage divides the pivot row by its pivot and clears the pivot column above and ' +
      'below, so after $n$ stages the augmented matrix is $[I \\mid \\mathbf{x}]$ and no back ' +
      'substitution is needed. It costs $n^3$ flops, half again as much as Gaussian elimination.',
    order: 'direct · $n^3$ flops',
    pros: ['No back substitution: $\\mathbf{x}$ is the last column', 'Forward stable'],
    cons: ['50 % more flops than elimination', 'Not backward stable (Peters & Wilkinson 1975)'],
    quantities: [PIVOT],
  },
);

registerMethod(
  {
    id: 'lu_decomposition',
    family: 'linalg',
    name: 'LU factorization (Doolittle, partial pivoting)',
    needs: ['A', 'b'],
    order: 'direct, 2n³/3 flops + 2n² per solve',
    summary: 'Factor P·A = L·U once (unit lower L, upper U), then solve L y = P b and U x = y.',
    references: [
      'Golub & Van Loan, Matrix Computations (4th ed., 2013), Alg. 3.4.1',
      'Burden & Faires, Numerical Analysis (10th ed.), Alg. 6.4',
      'Trefethen & Bau, Numerical Linear Algebra (1997), Alg. 21.1',
    ],
  },
  luDecomposition,
  {
    rule: '\\ell_{ic} = \\frac{w_{ic}}{w_{cc}},\\qquad W_{c+1:,\\,c:} \\leftarrow W_{c+1:,\\,c:} - \\boldsymbol{\\ell}\\,\\mathbf{w}_c^{\\mathsf{T}}',
    intuition:
      'The same eliminations as Gaussian elimination, but the multipliers are kept as the ' +
      'columns of a unit lower-triangular $L$ and the right-hand side is left alone. Once ' +
      '$PA = LU$ is known, every new $\\mathbf{b}$ costs only two triangular solves ($2n^2$ flops).',
    order: 'direct · $2n^3/3$ flops, then $2n^2$ per solve',
    pros: ['Factor once, solve many right-hand sides', 'Each stage is a rank-1 Schur update'],
    cons: ['Same growth behavior as partial pivoting'],
    quantities: [PIVOT],
  },
);

registerMethod(
  {
    id: 'cholesky',
    family: 'linalg',
    name: 'Cholesky factorization',
    needs: ['A', 'b'],
    order: 'direct, n³/3 flops',
    summary: 'Factor a symmetric positive definite A = L·Lᵀ, then solve L y = b and Lᵀ x = y.',
    references: [
      'Trefethen & Bau, Numerical Linear Algebra (1997), Alg. 23.1',
      'Golub & Van Loan, Matrix Computations (4th ed., 2013), §4.2 (outer-product Cholesky)',
      'Burden & Faires, Numerical Analysis (10th ed.), Alg. 6.6',
    ],
  },
  cholesky,
  {
    rule: 'l_{cc} = \\sqrt{w_{cc}},\\quad l_{ic} = \\frac{w_{ic}}{l_{cc}},\\quad W \\leftarrow W - \\boldsymbol{\\ell}_c\\boldsymbol{\\ell}_c^{\\mathsf{T}}',
    intuition:
      'For a symmetric positive definite $A$, each stage takes the square root of the pivot ' +
      '$w_{cc}$ and peels off the rank-1 piece $\\boldsymbol{\\ell}_c\\boldsymbol{\\ell}_c^{\\mathsf{T}}$. $A$ is ' +
      'SPD exactly when every pivot stays positive, so the factorization is itself the SPD ' +
      'test; it needs no pivoting and half the work of LU.',
    order: 'direct · $n^3/3$ flops',
    pros: ['Half the flops of LU', 'Backward stable without pivoting'],
    cons: ['Only for symmetric positive definite $A$'],
    quantities: [{ tex: 'w_{cc}', key: 'info.pivot_value' }],
  },
);

registerMethod(
  {
    id: 'qr_householder',
    family: 'linalg',
    name: 'QR factorization (Householder)',
    needs: ['A', 'b'],
    order: 'direct, 4n³/3 flops',
    summary: 'Reflect each column onto the axis to get A = Q·R, then solve R x = Qᵀ b.',
    references: [
      'Trefethen & Bau, Numerical Linear Algebra (1997), Alg. 10.1 and 10.2',
      'Golub & Van Loan, Matrix Computations (4th ed., 2013), §5.1–5.2',
    ],
  },
  qrHouseholder,
  {
    rule: '\\mathbf{v} = \\frac{\\mathbf{x} + \\operatorname{sign}(x_1)\\|\\mathbf{x}\\|\\mathbf{e}_1}{\\|\\cdot\\|},\\quad H = I - 2\\mathbf{v}\\mathbf{v}^{\\mathsf{T}}',
    intuition:
      'Each stage reflects the active column onto the first axis with a Householder mirror ' +
      '$H = I - 2\\mathbf{v}\\mathbf{v}^{\\mathsf{T}}$, which zeros it below the diagonal without ' +
      'growing any entry. The same reflections applied to $\\mathbf{b}$ give ' +
      '$Q^{\\mathsf{T}}\\mathbf{b}$, and $R\\mathbf{x} = Q^{\\mathsf{T}}\\mathbf{b}$ is solved from the ' +
      'bottom up.',
    order: 'direct · $4n^3/3$ flops',
    pros: [
      'Backward stable for every A, no pivoting',
      'Orthogonal transforms never amplify errors',
    ],
    cons: ['Twice the flops of LU'],
    quantities: [{ tex: 'r_{cc}', key: 'info.pivot_value' }],
  },
);

registerMethod(
  {
    id: 'thomas',
    family: 'linalg',
    name: 'Thomas algorithm (tridiagonal)',
    needs: ['A', 'b'],
    order: 'direct, 8n flops',
    summary:
      'Gaussian elimination specialized to a tridiagonal matrix: one forward sweep, one back sweep.',
    references: [
      'Burden & Faires, Numerical Analysis (10th ed.), Alg. 6.7',
      'Higham, Accuracy and Stability of Numerical Algorithms (2nd ed., 2002), Ch. 9',
    ],
  },
  thomas,
  {
    rule: "w_i = \\delta_i - a_i c'_{i-1},\\quad c'_i = \\frac{c_i}{w_i},\\quad d'_i = \\frac{b_i - a_i d'_{i-1}}{w_i}",
    intuition:
      'On a tridiagonal $A = \\operatorname{tridiag}(a_i, \\delta_i, c_i)$ only one entry sits ' +
      'under each pivot, so elimination shrinks to one forward sweep that normalizes each row ' +
      "and one back sweep $x_i = d'_i - c'_i x_{i+1}$. The cost is $O(n)$, not $O(n^3)$.",
    order: 'direct · $8n$ flops',
    pros: ['Linear cost and storage', 'Stable for diagonally dominant or SPD $A$'],
    cons: ['No pivoting: can meet a zero $w_i$', 'Tridiagonal $A$ only'],
    quantities: [{ tex: 'w_i', key: 'info.pivot_value' }],
  },
);
