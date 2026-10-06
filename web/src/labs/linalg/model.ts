/**
 * Pure helpers of the linear-systems lab: method classes, the structure of A (what predicts
 * which method works), the elimination factors as they build up, and the plane geometry of a
 * 2 × 2 system. No React here; tests/linalg/lab.test.ts covers it.
 */
import type { Matrix, Step, Vector } from '../../core/types';
import { int, sig } from '../../core/format';
import type { MathRun } from '../../viz';
import type { LinalgProblem } from '../../problems/linalg';
import { mathTextToPlain } from '../../ui/mathProse';
import { eigvals, eigvalsh, isSymmetric, spectralRadius } from '../../methods/linalg/numerics';
import { jacobiMatrix, optimalOmega, sorMatrix } from '../../methods/linalg/iterative';

// ── Method classes ────────────────────────────────────────────────────────────────────

export const DIRECT = [
  'gaussian_elimination',
  'gaussian_elimination_pivoting',
  'gauss_jordan',
  'lu_decomposition',
  'cholesky',
  'qr_householder',
  'thomas',
] as const;
export const STATIONARY = ['jacobi', 'gauss_seidel', 'sor'] as const;
export const KRYLOV = [
  'steepest_descent_linear',
  'conjugate_gradient_linear',
  'preconditioned_cg',
  'gmres',
] as const;

export const isDirect = (id: string) => (DIRECT as readonly string[]).includes(id);
export const isStationary = (id: string) => (STATIONARY as readonly string[]).includes(id);
export const isKrylov = (id: string) => (KRYLOV as readonly string[]).includes(id);
/** Methods that minimize φ(x) = ½xᵀAx − bᵀx (their geometry is the level ellipses). */
export const isDescent = (id: string) =>
  id === 'steepest_descent_linear' ||
  id === 'conjugate_gradient_linear' ||
  id === 'preconditioned_cg';

// ── Structure of A ────────────────────────────────────────────────────────────────────

export interface Structure {
  n: number;
  symmetric: boolean;
  spd: boolean;
  /** κ₂(A) = σ_max/σ_min (∞ when A is singular to working precision). */
  kappa: number;
  tridiagonal: boolean;
  /** Strictly row diagonally dominant. */
  diagDominant: boolean;
  zeroDiagonal: boolean;
  /** ρ(G_J), ρ(G_GS); null when a diagonal entry is 0. */
  rhoJ: number | null;
  rhoGS: number | null;
  /** Young's ω* and ρ(G_ω*) when the theory applies. */
  omegaOpt: number | null;
  rhoOpt: number | null;
}

function transposeTimes(A: Matrix): Matrix {
  const n = A.length;
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      let s = 0;
      for (let q = 0; q < n; q++) s += A[q][i] * A[q][j];
      return s;
    }),
  );
}

export function structureOf(A: Matrix): Structure {
  const n = A.length;
  const symmetric = isSymmetric(A);
  let kappa: number;
  let spd = false;
  if (symmetric) {
    const lam = eigvalsh(A);
    spd = lam[0] > 0;
    const abs = lam.map(Math.abs);
    const lo = Math.min(...abs);
    kappa = lo > 0 ? Math.max(...abs) / lo : Infinity;
  } else {
    const s2 = eigvalsh(transposeTimes(A)); // σ² ascending
    const lo = Math.sqrt(Math.max(0, s2[0]));
    const hi = Math.sqrt(s2[n - 1]);
    // σ_min below the rounding level of AᵀA means "singular to working precision".
    kappa = s2[0] > s2[n - 1] * 1e-15 ? hi / lo : Infinity;
  }
  let tridiagonal = true;
  let diagDominant = true;
  for (let i = 0; i < n; i++) {
    let off = 0;
    for (let j = 0; j < n; j++) {
      if (j !== i) off += Math.abs(A[i][j]);
      if (Math.abs(i - j) > 1 && A[i][j] !== 0) tridiagonal = false;
    }
    if (!(Math.abs(A[i][i]) > off)) diagDominant = false;
  }
  const zeroDiagonal = A.some((r, i) => r[i] === 0);
  let rhoJ: number | null = null,
    rhoGS: number | null = null,
    omegaOpt: number | null = null,
    rhoOpt: number | null = null;
  if (!zeroDiagonal) {
    rhoJ = spectralRadius(jacobiMatrix(A));
    rhoGS = spectralRadius(sorMatrix(A, 1));
    const y = optimalOmega(A, 1);
    omegaOpt = y.omega_opt;
    rhoOpt = y.spectral_radius_opt;
  }
  return {
    n,
    symmetric,
    spd,
    kappa,
    tridiagonal,
    diagDominant,
    zeroDiagonal,
    rhoJ,
    rhoGS,
    omegaOpt,
    rhoOpt,
  };
}

/** ρ(G_ω) on a grid of ω ∈ (0, 2): the curve of the relaxation panel. */
export function relaxationCurve(
  A: Matrix,
  extra: readonly number[] = [],
  count = 161,
): [number, number][] {
  if (A.some((r, i) => r[i] === 0)) return [];
  // The minimum is a cusp (a square-root singularity at ω⋆): sample ω⋆ itself.
  const ws = Array.from({ length: count }, (_, q) => 0.02 + (1.96 * q) / (count - 1));
  for (const w of extra) if (w > 0.02 && w < 1.98) ws.push(w);
  ws.sort((a, b) => a - b);
  return ws.map((w) => [w, spectralRadius(sorMatrix(A, w))]);
}

/** The eigenvalues of the Jacobi matrix (real for consistently ordered SPD A). */
export function jacobiSpectrum(A: Matrix) {
  return eigvals(jacobiMatrix(A));
}

// ── Elimination: factors as they build up ─────────────────────────────────────────────

export type Phase = 'start' | 'eliminate' | 'back_substitution' | 'solve';

export interface StageView {
  phase: Phase;
  /** Matrix shown at this step (augmented or n × n working matrix). */
  matrix: Matrix;
  pivot: [number, number] | null;
  pivotValue: number | null;
  rowSwap: [number, number] | null;
  multipliers: Vector | null;
  zeroPivot: boolean;
  /** Original equation index of each displayed row (row i of `matrix` is equation perm[i]). */
  perm: number[];
}

export function stageOf(step: Step, n: number): StageView {
  const info = step.info;
  const perm = (info.perm as number[] | undefined) ?? Array.from({ length: n }, (_, i) => i);
  return {
    phase: (info.phase as Phase) ?? 'start',
    matrix: (info.matrix as Matrix) ?? [],
    pivot: (info.pivot as [number, number] | null) ?? null,
    pivotValue: (info.pivot_value as number | null) ?? null,
    rowSwap: (info.row_swap as [number, number] | null) ?? null,
    multipliers: (info.multipliers as Vector | null) ?? null,
    zeroPivot: Boolean(info.zero_pivot),
    perm,
  };
}

/** Swap two rows of a matrix (a copy). */
export function swapped<T>(rows: readonly T[], swap: [number, number] | null): T[] {
  const out = rows.slice();
  if (swap) [out[swap[0]], out[swap[1]]] = [out[swap[1]], out[swap[0]]];
  return out;
}

/**
 * The unit lower-triangular L after each step of an elimination method, with NaN for entries
 * that are not known yet. LU records L in Step.info; for Gaussian elimination (with or without
 * pivoting) it is rebuilt from the multipliers and row swaps exactly as the Python code does.
 */
export function lowerFactors(method: string, trace: readonly Step[], n: number): (Matrix | null)[] {
  if (method === 'lu_decomposition' || method === 'cholesky')
    return trace.map((s) => {
      const L = s.info.L as Matrix | undefined;
      if (!L) return null;
      const pivot = s.info.pivot as number[] | null | undefined;
      // Last column computed so far: the pivot column; none at the start; all on the solve step.
      const c = pivot ? pivot[1] : s.info.phase === 'start' ? -1 : n;
      const done = s.info.zero_pivot ? c - 1 : c;
      const unitDiag = method === 'lu_decomposition';
      return L.map((row, i) =>
        row.map((v, j) => (j > i ? 0 : j === i && unitDiag ? 1 : j <= done ? v : NaN)),
      );
    });
  if (method !== 'gaussian_elimination' && method !== 'gaussian_elimination_pivoting')
    return trace.map(() => null);
  const L: Matrix = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : j > i ? 0 : NaN)),
  );
  const out: (Matrix | null)[] = [];
  for (const s of trace) {
    const stage = stageOf(s, n);
    if (stage.phase === 'eliminate' && stage.pivot) {
      const c = stage.pivot[1];
      if (stage.rowSwap) {
        const [a, b] = stage.rowSwap;
        for (let q = 0; q < c; q++) [L[a][q], L[b][q]] = [L[b][q], L[a][q]];
      }
      if (!stage.zeroPivot && stage.multipliers)
        for (let i = c + 1; i < n; i++) L[i][c] = stage.multipliers[i];
    }
    out.push(L.map((r) => r.slice()));
  }
  return out;
}

// ── Plane geometry of a 2 × 2 system ──────────────────────────────────────────────────

export type Box = [[number, number], [number, number]];

/**
 * A square view (one data unit is one unit on both axes) around x⋆, x₀ and the first iterates
 * of the runs — an over-relaxed step that overshoots the lines must stay in the picture — with
 * points farther than 4× the x₀–x⋆ distance left to the off-view treatment.
 */
export function planeDomain(center: Vector, x0: Vector, extra: readonly Vector[] = []): Box {
  const d0 = Math.max(Math.abs(x0[0] - center[0]), Math.abs(x0[1] - center[1]), 1);
  const pts = [center, x0];
  for (const p of extra)
    if (
      p.every(Number.isFinite) &&
      Math.abs(p[0] - center[0]) <= 4 * d0 &&
      Math.abs(p[1] - center[1]) <= 4 * d0
    )
      pts.push(p);
  const xs = pts.map((p) => p[0]),
    ys = pts.map((p) => p[1]);
  const lo = [Math.min(...xs), Math.min(...ys)],
    hi = [Math.max(...xs), Math.max(...ys)];
  const h = Math.max(hi[0] - lo[0], hi[1] - lo[1], 2) * 0.5 * 1.55;
  const cx = (lo[0] + hi[0]) / 2,
    cy = (lo[1] + hi[1]) / 2;
  return [
    [cx - h, cx + h],
    [cy - h, cy + h],
  ];
}

/** The segment of the line a·x = c inside the box (null if it misses it). */
export function lineInBox(
  a: readonly number[],
  c: number,
  box: Box,
): [[number, number], [number, number]] | null {
  const [[x0, x1], [y0, y1]] = box;
  const pts: [number, number][] = [];
  const add = (x: number, y: number) => {
    if (x >= x0 - 1e-9 && x <= x1 + 1e-9 && y >= y0 - 1e-9 && y <= y1 + 1e-9) pts.push([x, y]);
  };
  if (a[1] !== 0) {
    add(x0, (c - a[0] * x0) / a[1]);
    add(x1, (c - a[0] * x1) / a[1]);
  }
  if (a[0] !== 0) {
    add((c - a[1] * y0) / a[0], y0);
    add((c - a[1] * y1) / a[0], y1);
  }
  if (pts.length < 2) return null;
  // The two points farthest apart (corners can repeat).
  let best: [[number, number], [number, number]] = [pts[0], pts[1]];
  let d = -1;
  for (let i = 0; i < pts.length; i++)
    for (let j = i + 1; j < pts.length; j++) {
      const e = Math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]);
      if (e > d) {
        d = e;
        best = [pts[i], pts[j]];
      }
    }
  return best;
}

/** φ(x) = ½xᵀAx − bᵀx for SPD A, else ½‖Ax − b‖² (both are minimized at the solution). */
export function fieldOf(A: Matrix, b: Vector, spd: boolean): (x: number, y: number) => number {
  if (spd)
    return (x, y) =>
      0.5 * (A[0][0] * x * x + (A[0][1] + A[1][0]) * x * y + A[1][1] * y * y) - b[0] * x - b[1] * y;
  return (x, y) => {
    const r0 = A[0][0] * x + A[0][1] * y - b[0];
    const r1 = A[1][0] * x + A[1][1] * y - b[1];
    return 0.5 * (r0 * r0 + r1 * r1);
  };
}

/** Jacobi's two simultaneous one-equation solves from x: the points it aims at on each line. */
export function jacobiTargets(
  A: Matrix,
  b: Vector,
  x: Vector,
): [[number, number], [number, number]] {
  return [
    [(b[0] - A[0][1] * x[1]) / A[0][0], x[1]],
    [x[0], (b[1] - A[1][0] * x[0]) / A[1][1]],
  ];
}

/**
 * The Gauss–Seidel target of each SOR sub-step (on the line of its equation): from
 * x_i ← (1 − ω)x_i + ω·x_i^GS, x_i^GS = (x_i^new − (1 − ω)x_i^old)/ω.
 */
export function gsTargets(sweep: readonly Vector[], omega: number): [number, number][] {
  const out: [number, number][] = [];
  for (let i = 0; i + 1 < sweep.length; i++) {
    const before = sweep[i],
      after = sweep[i + 1];
    const p: [number, number] = [after[0], after[1]];
    p[i] = (after[i] - (1 - omega) * before[i]) / omega;
    out.push(p);
  }
  return out;
}

/** (x − c)ᵀA(x − c): the squared A-norm distance (level of the φ ellipse through x). */
export function aNorm2(A: Matrix, x: Vector, c: Vector): number {
  const d = x.map((v, i) => v - c[i]);
  let s = 0;
  for (let i = 0; i < d.length; i++) for (let j = 0; j < d.length; j++) s += d[i] * A[i][j] * d[j];
  return s;
}

/** An equation label `3x + 2y = 2` as typeset math runs (numbers upright, variables italic). */
export function equationRuns(a: readonly number[], c: number, names = ['x', 'y']): MathRun[] {
  // Up to 10 digits: 1 + 2⁻³⁰ must not print as 1 (nearly_singular is about that difference).
  const num = (v: number) =>
    Number.isInteger(v) ? String(Math.abs(v)) : String(+Math.abs(v).toPrecision(10));
  const runs: MathRun[] = [];
  a.forEach((v, i) => {
    if (v === 0) return;
    const coef = Math.abs(v) === 1 ? '' : num(v);
    const sign = runs.length === 0 ? (v < 0 ? '−' : '') : v < 0 ? ' − ' : ' + ';
    runs.push({ t: `${sign}${coef}`, style: 'main' }, { t: names[i], style: 'italic' });
  });
  runs.push({ t: ` = ${c < 0 ? '−' : ''}${num(c)}`, style: 'main' });
  return runs;
}

/** A plain equation label `3x + 2y = 2` with U+2212 minus signs. */
export function equationLabel(a: readonly number[], c: number, names = ['x', 'y']): string {
  const num = (v: number) => {
    const s =
      Math.abs(v) < 1e-12
        ? '0'
        : Number.isInteger(v)
          ? String(Math.abs(v))
          : String(+Math.abs(v).toPrecision(4));
    return s;
  };
  let out = '';
  a.forEach((v, i) => {
    if (v === 0) return;
    const coef = Math.abs(v) === 1 ? '' : num(v);
    if (out === '') out = `${v < 0 ? '−' : ''}${coef}${names[i]}`;
    else out += ` ${v < 0 ? '−' : '+'} ${coef}${names[i]}`;
  });
  return `${out || '0'} = ${c < 0 ? '−' : ''}${num(c)}`;
}

/** Problems with a 2-D plane view: the default start point of the iterative methods. */
export function defaultStart(problem: LinalgProblem): number[] {
  if (problem.n !== 2) return new Array<number>(problem.n).fill(0);
  if (problem.id === 'nearly_singular') return [-1, 2];
  return [-2, 2];
}

// ── Gauss–Seidel / SOR staircase ──────────────────────────────────────────────────────

export const easeInOut = (u: number) => (u < 0.5 ? 4 * u * u * u : 1 - (-2 * u + 2) ** 3 / 2);

/** The first fraction `f` (by length) of a polyline, and its end point. */
export function polylinePrefix(
  points: readonly [number, number][],
  f: number,
): { points: [number, number][]; end: [number, number] } {
  if (points.length === 0) return { points: [], end: [NaN, NaN] };
  const lens: number[] = [];
  let total = 0;
  for (let i = 1; i < points.length; i++) {
    const d = Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
    lens.push(d);
    total += d;
  }
  let left = Math.max(0, Math.min(1, f)) * total;
  const out: [number, number][] = [points[0]];
  for (let i = 1; i < points.length; i++) {
    const d = lens[i - 1];
    if (left >= d) {
      out.push(points[i]);
      left -= d;
      continue;
    }
    const u = d > 0 ? left / d : 0;
    const p: [number, number] = [
      points[i - 1][0] + (points[i][0] - points[i - 1][0]) * u,
      points[i - 1][1] + (points[i][1] - points[i - 1][1]) * u,
    ];
    out.push(p);
    return { points: out, end: p };
  }
  return { points: out, end: out[out.length - 1] };
}

// ── SOR's relaxation factor ───────────────────────────────────────────────────────────

export const OMEGA_MIN = 0.05;
export const OMEGA_MAX = 1.95;
export const clampOmega = (w: number) =>
  Math.round(Math.min(OMEGA_MAX, Math.max(OMEGA_MIN, w)) * 100) / 100;

/**
 * A dragged ω: rounded to 0.01, except within 0.005 of Young's ω⋆, where it snaps to the exact
 * ω⋆ (the arrow keys keep the plain 0.01 grid).
 */
export function snapOmega(w: number, omegaOpt: number | null): number {
  if (
    omegaOpt !== null &&
    omegaOpt >= OMEGA_MIN &&
    omegaOpt <= OMEGA_MAX &&
    Math.abs(w - omegaOpt) <= 0.005
  )
    return omegaOpt;
  return clampOmega(w);
}

/** ω with 2 decimals, or 3 when it is not a multiple of 0.01 (an exact ω⋆ typed or snapped). */
export const omegaText = (w: number) =>
  Math.abs(w * 100 - Math.round(w * 100)) < 1e-6 ? w.toFixed(2) : w.toFixed(3);

// ── Components plot: values outside the view ──────────────────────────────────────────

/** The true value of a component outside the view ("−1937", "4.2×10⁷", "∞"). */
export function offText(v: number): string {
  if (Number.isNaN(v)) return 'NaN';
  if (!Number.isFinite(v)) return v > 0 ? '∞' : '−∞';
  return sig(v, 4);
}

/**
 * The components of `pts` outside [lo, hi] (non-finite ones included), as 0-based indices: the
 * plot draws them as chevrons on the frame edge, never as a value pinned to it.
 */
export function offScaleIndices(pts: readonly number[], lo: number, hi: number): number[] {
  const out: number[] = [];
  pts.forEach((v, i) => {
    if (!(Number.isFinite(v) && v >= lo && v <= hi)) out.push(i);
  });
  return out;
}

// ── Rail copy ──────────────────────────────────────────────────────────────────────────

/**
 * The Python problem descriptions call the iteration matrices T_J and T_GS; every label of this
 * lab writes G_J and G_GS (and diag_dominant_3's "every method applies" overlooks Thomas, which
 * needs a tridiagonal A). The rail shows the lab's notation as prose with inline TeX (`$…$`, set
 * by `MathProse` like the structure table below it); the data stay Python's.
 */
export function displayDescription(p: LinalgProblem): string {
  let d = p.description
    .replace(/ρ\(T_GS\)/g, '$\\rho(G_{GS})$')
    .replace(/ρ\(T_J\)/g, '$\\rho(G_J)$')
    .replace(/\bT_GS\b/g, '$G_{GS}$')
    .replace(/\bT_J\b/g, '$G_J$')
    .replace(/\bH_ij\b/g, '$H_{ij}$')
    .replace(/ω\*/g, '$\\omega^\\star$');
  if (p.id === 'diag_dominant_3')
    d = d.replace(
      'Every method in the family applies',
      'Every method of the family applies except Thomas, which needs a tridiagonal A',
    );
  return d;
}

/** The same description as plain text (the problem menu's option lines, accessible names). */
export const displayDescriptionPlain = (p: LinalgProblem): string =>
  mathTextToPlain(displayDescription(p));

/** The rail's one-paragraph reading of the structure of A (which methods apply, and why). */
export function structureHint(s: Structure): string {
  const out: string[] = [];
  const singular = !Number.isFinite(s.kappa);
  if (singular)
    out.push(
      'A is singular: elimination must meet a zero pivot (at the latest at the last stage). ' +
        'A consistent b has infinitely many solutions, which GMRES can still reach.',
    );
  else if (s.spd) out.push('Cholesky, CG and steepest descent apply.');
  else out.push('CG and Cholesky need SPD A; use LU, QR or GMRES.');
  if (!s.tridiagonal) out.push('Thomas needs a tridiagonal A.');
  if (s.rhoJ !== null && s.rhoJ >= 1) out.push('Jacobi diverges here.');
  if (s.zeroDiagonal) out.push('Jacobi, Gauss–Seidel and SOR divide by a zero diagonal entry.');
  if (!singular && s.kappa > 1e5)
    out.push(`About ${int(Math.round(Math.log10(s.kappa)))} of 16 digits are at risk.`);
  return out.join(' ');
}
