/**
 * Simplex methods for linear programs — TS port of `numopt.lp.simplex`
 * (src/numopt/lp/simplex.py). Read that module docstring for the mathematics; this port mirrors
 * it line by line: the same standard form, tableau layout, pivot rules, scaled pivot
 * tolerances, phase 1 / big-M weights, drive-out pivots, statuses and messages.
 *
 * Every LinearProgram  min/max cᵀx s.t. A_ub x ≤ b_ub, A_eq x = b_eq, x ≥ 0  is solved in the
 * minimization standard form min c̃ᵀz s.t. A z = b, z ≥ 0 (b ≥ 0 after flipping rows).
 *
 * Tableau layout (Bertsimas & Tsitsiklis 1997, §3.3, full tableau, rhs last):
 *   row 0        [ c̄ᵀ    | −z̃  ]
 *   rows 1 … m   [ B⁻¹A  | x_B ]
 * big-M keeps two objective rows (the M coefficients, then the constants).
 *
 * Trace: Step k is the state after k pivots; `entering`, `leaving`, `pivot_row`, `ratio_test`
 * describe the pivot chosen on this tableau (null on the final Step). `stepSize` is the step
 * length θ of the pivot that produced the Step. Info keys stay snake_case as in Python:
 * tableau, row_labels, col_labels, basis, nonbasis, entering, leaving, pivot_row, ratio_test,
 * vertex, objective, phase, infeasibility, drive_out, removed_rows, cycling, ray,
 * primal_infeasibility (dual), duals, reduced_costs, direction (revised).
 */
import { registerMethod, param } from '../../core/registry';
import type { LinearProgram, Matrix, ParamSpec, Result, Step, Vector } from '../../core/types';

// ─────────────────────────────────────────────────────────────────────────────────────────
// Small numeric helpers (NumPy semantics)
// ─────────────────────────────────────────────────────────────────────────────────────────

export const EPS = 2.220446049250313e-16;
/** np.finfo(np.float64).tiny */
export const TINY = 2.2250738585072014e-308;

export const zeros = (n: number): number[] => new Array<number>(n).fill(0);
export const ones = (n: number): number[] => new Array<number>(n).fill(1);
export const zerosM = (m: number, n: number): number[][] =>
  Array.from({ length: m }, () => zeros(n));

export function dotv(a: readonly number[], b: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

const maxAbs = (v: readonly number[]) => v.reduce((m, x) => Math.max(m, Math.abs(x)), 0);

/** Python `f"{v:.{p}g}"` (general format). */
export function pyG(v: number, p = 6): string {
  if (Number.isNaN(v)) return 'nan';
  if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf';
  if (v === 0) return Object.is(v, -0) ? '-0' : '0';
  const exp = Math.floor(Math.log10(Math.abs(Number(v.toPrecision(p)))));
  if (exp < -4 || exp >= p) {
    const [mant, e] = v.toExponential(p - 1).split('e');
    const m = mant.includes('.') ? mant.replace(/\.?0+$/, '') : mant;
    const en = Number(e);
    return `${m}e${en < 0 ? '-' : '+'}${String(Math.abs(en)).padStart(2, '0')}`;
  }
  const s = v.toFixed(Math.max(0, p - 1 - exp));
  return s.includes('.') ? s.replace(/\.?0+$/, '') : s;
}

/** Python repr of a tuple of strings: `('dantzig', 'bland')`. */
const pyTuple = (xs: readonly string[]) => `(${xs.map((x) => `'${x}'`).join(', ')})`;

// ─────────────────────────────────────────────────────────────────────────────────────────
// Parameters and references
// ─────────────────────────────────────────────────────────────────────────────────────────

const PIVOT_RULES = ['dantzig', 'bland', 'steepest_edge'] as const;

const TOL_HELP =
  'Optimality / pivot tolerance on the equilibrated problem: c̄ⱼ < −tol·σⱼ·γ enters; ūᵢ > tol·σⱼ/σ_B(i) takes part in the ratio test.';

export const PIVOT_PARAMS: ParamSpec[] = [
  param.choice('pivot_rule', 'dantzig', [...PIVOT_RULES], {
    help: 'Entering-variable rule: most negative reduced cost, smallest index, or steepest edge.',
    label: 'Pivot rule',
  }),
  param.float('tol', 1e-9, {
    min: 1e-14,
    max: 1e-4,
    log: true,
    help: TOL_HELP,
    label: 'Pivot tolerance',
    tex: '\\tau',
  }),
  param.int('max_iter', 200, {
    min: 1,
    max: 10_000,
    help: 'Maximum number of pivots.',
    label: 'Pivot budget',
  }),
];

const REFERENCES = [
  'Bertsimas & Tsitsiklis, Introduction to Linear Optimization (1997), §3.3 (full tableau) and §3.5',
  'Chvátal, Linear Programming (1983), ch. 2–3',
  'Bland (1977), Math. Oper. Res. 2(2):103–107',
];

// ─────────────────────────────────────────────────────────────────────────────────────────
// Standard form
// ─────────────────────────────────────────────────────────────────────────────────────────

/** `min c̃ᵀz s.t. A z = b, z ≥ 0` built from a LinearProgram. */
export interface StandardForm {
  A: Matrix; // (m, N)
  b: Vector; // (m)
  c: Vector; // (N) min-form costs
  labels: string[];
  n: number;
  /** +1 for min, −1 for max: cᵀx = sign · c̃ᵀz */
  sign: number;
  /** Per row: a column with +1 in this row and 0 elsewhere (null: needs an artificial). */
  unitCol: (number | null)[];
  exprConst: Vector; // (N)
  exprCoef: Matrix; // (N, n)
}

const nCols = (A: Matrix, fallback: number) => (A.length ? A[0].length : fallback);

/** σₖ = maxᵢ |A_ik| / ρᵢ with ρᵢ = max_{j<n} |A_ij| (1 for a row without one); σₖ = 1 for a zero column. */
export function columnScales(A: Matrix, n: number, N = nCols(A, 0)): number[] {
  const m = A.length;
  if (m === 0) return ones(N);
  const rho = A.map((row) => {
    const r = maxAbs(row.slice(0, n));
    return r > 0 ? r : 1.0;
  });
  const sigma = zeros(N);
  for (let k = 0; k < N; k++) {
    let s = 0;
    for (let i = 0; i < m; i++) s = Math.max(s, Math.abs(A[i][k]) / rho[i]);
    sigma[k] = s > 0 ? s : 1.0;
  }
  return sigma;
}

/** ρᵢ per standard-form row: max_{j<n} |A_ij|; |bᵢ| for a row with no entry on x (1 if bᵢ = 0). */
export function rowScales(sf: StandardForm): number[] {
  if (sf.b.length === 0) return [];
  return sf.A.map((row, i) => {
    const r = maxAbs(row.slice(0, sf.n));
    if (r > 0) return r;
    return Math.abs(sf.b[i]) >= TINY ? Math.abs(sf.b[i]) : 1.0;
  });
}

const artRows = (sf: StandardForm) =>
  sf.unitCol.map((u, i) => (u === null ? i : -1)).filter((i) => i >= 0);

/** Phase-1 / M-row cost per column: wᵢ = 1/ρᵢ on the artificial of row i, else 0. */
export function artificialWeights(sf: StandardForm, nTotal: number): number[] {
  const N = sf.A.length ? sf.A[0].length : sf.c.length;
  const rho = rowScales(sf);
  const w = zeros(nTotal);
  artRows(sf).forEach((i, k) => {
    w[N + k] = 1.0 / rho[i];
  });
  return w;
}

/** Accepted level capᵢ = tol·(bᵢ + Σⱼ |A_ij|·|zⱼ|) of the artificial of row i. */
export function artificialCaps(sf: StandardForm, z: Vector, nTotal: number, tol: number) {
  const N = sf.A.length ? sf.A[0].length : sf.c.length;
  const cap = zeros(nTotal);
  artRows(sf).forEach((i, k) => {
    let s = 0;
    for (let j = 0; j < N; j++) s += Math.abs(sf.A[i][j]) * Math.abs(z[j]);
    cap[N + k] = tol * (Math.abs(sf.b[i]) + s);
  });
  return cap;
}

/** First basic artificial whose level exceeds its accepted level, as [column, level]. */
export function artificialViolation(
  basis: readonly number[],
  xB: readonly number[],
  cap: readonly number[],
  artificial: ReadonlySet<number>,
): [number, number] | null {
  for (let i = 0; i < basis.length; i++) {
    const j = basis[i];
    if (artificial.has(j) && xB[i] > cap[j]) return [j, xB[i]];
  }
  return null;
}

/** Validate shapes and finiteness; return the LinearProgram (throw on invalid input). */
export function checkLp(lp: unknown): LinearProgram {
  const p = lp as LinearProgram;
  if (!p || typeof p !== 'object' || !Array.isArray(p.c))
    throw new TypeError('problem must be a numopt LinearProgram');
  const n = p.c.length;
  if (n === 0) throw new Error(`${p.id}: c must be a non-empty vector`);
  for (const [nameA, nameB, a, b] of [
    ['A_ub', 'b_ub', p.aUb, p.bUb],
    ['A_eq', 'b_eq', p.aEq, p.bEq],
  ] as const) {
    if ((a === null || a === undefined) !== (b === null || b === undefined))
      throw new Error(`${p.id}: ${nameA} and ${nameB} must be given together`);
    if (!a || !b) continue;
    if (a.some((row) => row.length !== n) || b.length !== a.length)
      throw new Error(`${p.id}: ${nameA} must be (m, ${n}) and ${nameB} must be (m,)`);
    if (!a.every((row) => row.every(Number.isFinite)) || !b.every(Number.isFinite))
      throw new Error(`${p.id}: ${nameA}/${nameB} contain non-finite values`);
  }
  if (!p.c.every(Number.isFinite)) throw new Error(`${p.id}: c contains non-finite values`);
  if (p.sense !== 'min' && p.sense !== 'max')
    throw new Error(`${p.id}: sense must be 'min' or 'max'`);
  return p;
}

export interface LpArrays {
  c: Vector;
  aUb: Matrix;
  bUb: Vector;
  aEq: Matrix;
  bEq: Vector;
}

/** `c, A_ub, b_ub, A_eq, b_eq` as copies (empty (0, n) when absent). */
export function lpArrays(lp: LinearProgram): LpArrays {
  return {
    c: [...lp.c],
    aUb: (lp.aUb ?? []).map((r) => [...r]),
    bUb: [...(lp.bUb ?? [])],
    aEq: (lp.aEq ?? []).map((r) => [...r]),
    bEq: [...(lp.bEq ?? [])],
  };
}

/** The b ≥ 0 standard form (slack per ≤ row, rows with b < 0 multiplied by −1). */
export function standardForm(lp: LinearProgram): StandardForm {
  const { c, aUb, bUb, aEq, bEq } = lpArrays(lp);
  const n = c.length,
    mUb = bUb.length,
    mEq = bEq.length;
  const m = mUb + mEq,
    bigN = n + mUb;
  const A = zerosM(m, bigN);
  for (let i = 0; i < mUb; i++) {
    for (let j = 0; j < n; j++) A[i][j] = aUb[i][j];
    A[i][n + i] = 1.0;
  }
  for (let i = 0; i < mEq; i++) for (let j = 0; j < n; j++) A[mUb + i][j] = aEq[i][j];
  const b = [...bUb, ...bEq];
  const flip = b.map((v) => v < 0);
  for (let i = 0; i < m; i++) {
    if (!flip[i]) continue;
    A[i] = A[i].map((v) => v * -1.0);
    b[i] *= -1.0;
  }
  const sign = lp.sense === 'min' ? 1.0 : -1.0;
  const cStd = [...c.map((v) => sign * v), ...zeros(mUb)];
  const unitCol = Array.from({ length: m }, (_, i) => (i >= mUb || flip[i] ? null : n + i));
  const exprConst = [...zeros(n), ...bUb];
  const exprCoef = [
    ...Array.from({ length: n }, (_, j) => zeros(n).map((_, k) => (k === j ? 1.0 : 0.0))),
    ...aUb.map((row) => row.map((v) => -v)),
  ];
  const labels = [
    ...Array.from({ length: n }, (_, j) => `x${j + 1}`),
    ...Array.from({ length: mUb }, (_, i) => `s${i + 1}`),
  ];
  return { A, b, c: cStd, labels, n, sign, unitCol, exprConst, exprCoef };
}

// ─────────────────────────────────────────────────────────────────────────────────────────
// Tableau engine
// ─────────────────────────────────────────────────────────────────────────────────────────

/** A simplex tableau: `T` has `nObj` objective rows, then one row per constraint. */
export class Tableau {
  T: number[][];
  basis: number[];
  labels: string[];
  nObj: number;
  artificial: Set<number>;
  /** σ per column; null means σ = 1 (an absolute pivot tolerance). */
  scale: number[] | null;
  /** γ per objective row. */
  costScale: number[] = [1.0, 1.0];
  /** Phase-1 / M-row cost per column. */
  artCost: number[] | null = null;

  constructor(
    T: number[][],
    basis: number[],
    labels: string[],
    nObj = 1,
    artificial: Set<number> = new Set(),
    scale: number[] | null = null,
  ) {
    this.T = T;
    this.basis = basis;
    this.labels = labels;
    this.nObj = nObj;
    this.artificial = artificial;
    this.scale = scale;
  }

  get m(): number {
    return this.T.length - this.nObj;
  }

  get N(): number {
    return this.T[0].length - 1;
  }

  rows(): number[][] {
    return this.T.slice(this.nObj);
  }

  /** Values of all N columns at the current basic solution. */
  values(): number[] {
    const z = zeros(this.N);
    this.basis.forEach((j, i) => {
      z[j] = this.T[this.nObj + i][this.N];
    });
    return z;
  }

  /** Entering threshold tol·σⱼ·γ for every column j of objective `row`. */
  costTol(row: number, tol: number): number[] {
    const g = this.costScale[row];
    return Array.from({ length: this.N }, (_, j) => tol * (this.scale ? this.scale[j] : 1.0) * g);
  }

  /** τᵢⱼ = tol·σⱼ/σ_B(i) for every constraint row i. */
  pivotTol(j: number, tol: number): number[] {
    const scale = this.scale;
    if (!scale) return new Array<number>(this.m).fill(tol);
    return this.basis.map((jb) => (tol * scale[j]) / scale[jb]);
  }

  /** Gauss–Jordan pivot on constraint row `i` (0-based) and column `j`. */
  pivot(i: number, j: number): void {
    const r = this.nObj + i;
    const T = this.T;
    const piv = T[r][j];
    T[r] = T[r].map((v) => v / piv);
    const prow = T[r];
    const col = T.map((row) => row[j]);
    col[r] = 0.0;
    for (let a = 0; a < T.length; a++) {
      const f = col[a];
      const row = T[a];
      for (let k = 0; k < row.length; k++) row[k] -= f * prow[k];
    }
    // NOTE: the pivot column is a unit vector in exact arithmetic; store it exactly.
    for (let a = 0; a < T.length; a++) T[a][j] = 0.0;
    T[r][j] = 1.0;
    this.basis[i] = j;
  }

  snapshot(): Record<string, unknown> {
    const objLabels = this.nObj === 1 ? ['z'] : ['zM', 'z'];
    const basic = new Set(this.basis);
    return {
      tableau: this.T.map((row) => [...row]),
      row_labels: [...objLabels, ...this.basis.map((j) => this.labels[j])],
      col_labels: [...this.labels, 'rhs'],
      basis: [...this.basis],
      nonbasis: Array.from({ length: this.N }, (_, j) => j).filter((j) => !basic.has(j)),
    };
  }
}

/** Entering column by `rule` among `allowed` columns with c̄ⱼ < −tol·σⱼ·γ (null: optimal). */
export function chooseEntering(
  tab: Tableau,
  rule: string,
  tol: number,
  allowed: readonly boolean[],
): number | null {
  const N = tab.N;
  let d: number[];
  let eligible: boolean[];
  if (tab.nObj === 2) {
    const dM = tab.T[0].slice(0, N),
      dC = tab.T[1].slice(0, N);
    const thrM = tab.costTol(0, tol),
      thrC = tab.costTol(1, tol);
    eligible = allowed.map((a, j) => a && dM[j] < -thrM[j]);
    d = dM;
    if (!eligible.some(Boolean)) {
      eligible = allowed.map((a, j) => a && Math.abs(dM[j]) <= thrM[j] && dC[j] < -thrC[j]);
      d = dC;
    }
  } else {
    d = tab.T[0].slice(0, N);
    const thr = tab.costTol(0, tol);
    eligible = allowed.map((a, j) => a && d[j] < -thr[j]);
  }
  const cand = eligible.flatMap((e, j) => (e ? [j] : []));
  if (cand.length === 0) return null;
  if (rule === 'bland') return cand[0];
  let score = cand.map((j) => d[j]);
  if (rule === 'steepest_edge') {
    const rows = tab.rows();
    score = score.map((s, q) => {
      let ss = 0;
      for (const row of rows) ss += row[cand[q]] ** 2;
      return s / Math.sqrt(1.0 + ss);
    });
  }
  const best = Math.min(...score);
  const lim = best + tol * (1.0 + Math.abs(best));
  return cand[score.findIndex((s) => s <= lim)];
}

/** θ = min{x_B,i / uᵢ : uᵢ > τᵢ}; ties (rᵢ ≤ θ·(1 + tol)) go to the smallest basic index. */
export function minRatio(
  xB: readonly number[],
  u: readonly number[],
  tol: number,
  basis: readonly number[],
  pivotTol: readonly number[],
): { row: number | null; theta: number; ratios: (number | null)[] } {
  const mask = u.map((v, i) => v > pivotTol[i]);
  // NOTE: a basic value of −1e-17 (rounding) is treated as 0, so θ ≥ 0.
  const ratios = u.map((v, i) => (mask[i] ? Math.max(xB[i], 0.0) / v : null));
  if (!mask.some(Boolean)) return { row: null, theta: Infinity, ratios };
  const theta = Math.min(...(ratios.filter((r) => r !== null) as number[]));
  let leave = -1;
  for (let i = 0; i < u.length; i++) {
    if (!mask[i] || (ratios[i] as number) > theta * (1.0 + tol)) continue;
    if (leave < 0 || basis[i] < basis[leave]) leave = i;
  }
  return { row: leave, theta, ratios };
}

/** Minimum-ratio test on column `j`. */
export function ratioTest(tab: Tableau, j: number, tol: number) {
  const rows = tab.rows();
  const col = rows.map((r) => r[j]),
    rhs = rows.map((r) => r[tab.N]);
  return minRatio(rhs, col, tol, tab.basis, tab.pivotTol(j, tol));
}

/** Original-variable part of the edge direction η (η_B = −B⁻¹Aⱼ, ηⱼ = 1). */
export function rayDirection(tab: Tableau, j: number, n: number): number[] {
  const eta = zeros(tab.N);
  eta[j] = 1.0;
  tab.rows().forEach((row, i) => {
    eta[tab.basis[i]] = -row[j];
  });
  return eta.slice(0, n);
}

/** Collects Steps; `k` is the number of pivots done so far. */
export class Recorder {
  sf: StandardForm;
  maxIter: number;
  record: boolean;
  k = 0;
  trace: Step[] = [];
  lastTheta: number | null = null;
  unboundedCol: number | null = null;
  ray: number[] | null = null;

  constructor(sf: StandardForm, maxIter: number, record = true) {
    this.sf = sf;
    this.maxIter = maxIter;
    this.record = record;
  }

  objective(x: readonly number[]): number {
    return this.sf.sign * dotv(this.sf.c.slice(0, this.sf.n), x);
  }

  emit(tab: Tableau, phase: number, info: Record<string, unknown> = {}): void {
    if (!this.record) return;
    const z = tab.values();
    const x = z.slice(0, this.sf.n);
    const fun = this.objective(x);
    const art = [...tab.artificial].sort((a, b) => a - b);
    let infeas = 0.0;
    if (art.length && tab.artCost) {
      const ac = tab.artCost;
      infeas = art.reduce((s, j) => s + z[j] * ac[j], 0);
    }
    const payload: Record<string, unknown> = {
      ...tab.snapshot(),
      entering: null,
      leaving: null,
      pivot_row: null,
      ratio_test: null,
      vertex: x,
      objective: fun,
      phase,
      infeasibility: infeas,
      ...info,
    };
    this.trace.push({
      k: this.k,
      x: [...x],
      fun,
      gradNorm: null,
      stepSize: this.lastTheta,
      info: payload,
    });
  }
}

function pivotInfo(tab: Tableau, j: number, i: number | null, ratios: (number | null)[]) {
  return {
    entering: j,
    leaving: i === null ? null : tab.basis[i],
    pivot_row: i === null ? null : tab.nObj + i,
    ratio_test: [...ratios],
  };
}

const basisKey = (basis: readonly number[]) => [...basis].sort((a, b) => a - b).join(',');

/** Run primal simplex pivots until optimal / unbounded / cycling / max_iter. */
export function primalLoop(
  tab: Tableau,
  rec: Recorder,
  o: {
    rule: string;
    tol: number;
    phase: number;
    allowed: boolean[];
    emitOptimal?: boolean;
    firstInfo?: Record<string, unknown>;
    afterPivot?: () => void;
  },
): string {
  const { rule, tol, phase, allowed, emitOptimal = true } = o;
  const seen = new Set<string>();
  let pending: Record<string, unknown> = { ...(o.firstInfo ?? {}) };
  const emit = (info: Record<string, unknown> = {}) => {
    rec.emit(tab, phase, { ...info, ...pending });
    pending = {};
  };
  for (;;) {
    const key = basisKey(tab.basis);
    if (seen.has(key)) {
      emit({ cycling: true });
      return 'cycling';
    }
    seen.add(key);
    const j = chooseEntering(tab, rule, tol, allowed);
    if (j === null) {
      if (emitOptimal) emit();
      return 'optimal';
    }
    const { row: i, theta, ratios } = ratioTest(tab, j, tol);
    const info = pivotInfo(tab, j, i, ratios);
    if (i === null) {
      rec.unboundedCol = j;
      rec.ray = rayDirection(tab, j, rec.sf.n);
      emit({ ...info, ray: rec.ray });
      return 'unbounded';
    }
    emit(info);
    if (rec.k >= rec.maxIter) return 'max_iter';
    tab.pivot(i, j);
    o.afterPivot?.();
    rec.k += 1;
    rec.lastTheta = theta;
  }
}

/** γ = maxⱼ |costⱼ| (1 when every cost is 0). */
export function costScale(cost: readonly number[]): number {
  const g = cost.length ? maxAbs(cost) : 0.0;
  return g > 0.0 ? g : 1.0;
}

/** Row `[c̄ | −z̃]` for costs `cost` (length N) and the current basis. */
export function objectiveRow(tab: Tableau, cost: readonly number[]): number[] {
  const rows = tab.rows();
  const N = tab.N;
  const cB = tab.basis.map((j) => cost[j]);
  const row = zeros(N + 1);
  for (let k = 0; k < N; k++) {
    let s = 0;
    for (let i = 0; i < rows.length; i++) s += cB[i] * rows[i][k];
    row[k] = cost[k] - s;
  }
  let z = 0;
  for (let i = 0; i < rows.length; i++) z += cB[i] * rows[i][N];
  row[N] = -z;
  for (const j of tab.basis) row[j] = 0.0; // NOTE: exact zeros for basic columns
  return row;
}

/** Result of an LP solve by the tableau engine (used by the integer methods too). */
export interface LPOutcome {
  status: string;
  message: string;
  tab: Tableau | null;
  x: number[];
  /** cᵀx in the original sense (null unless optimal). */
  value: number | null;
  pivots: number;
  trace: Step[];
  sf: StandardForm;
  ray: number[] | null;
}

/** Tableau with the slack / artificial starting basis (artificials appended last). */
function initialTableau(sf: StandardForm, nObj = 1): Tableau {
  const m = sf.A.length;
  const N = m ? sf.A[0].length : sf.c.length;
  const ar = artRows(sf);
  const nArt = ar.length;
  const labels = [...sf.labels, ...ar.map((i) => `a${i + 1}`)];
  const A = zerosM(m, N + nArt);
  for (let i = 0; i < m; i++) for (let j = 0; j < N; j++) A[i][j] = sf.A[i][j];
  const basis: number[] = [];
  for (let i = 0; i < m; i++) {
    const uc = sf.unitCol[i];
    basis.push(uc !== null ? uc : N + ar.indexOf(i));
  }
  ar.forEach((i, k) => {
    A[i][N + k] = 1.0;
  });
  const T = zerosM(nObj + m, N + nArt + 1);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < N + nArt; j++) T[nObj + i][j] = A[i][j];
    T[nObj + i][N + nArt] = sf.b[i];
  }
  const art = new Set(Array.from({ length: nArt }, (_, k) => N + k));
  const tab = new Tableau(T, basis, labels, nObj, art, columnScales(A, sf.n, N + nArt));
  tab.artCost = artificialWeights(sf, N + nArt);
  return tab;
}

/** Drive-out pivot column: the largest |ā_ij|/τᵢⱼ, ties (relative, within tol) to the smallest index. */
function driveOutColumn(rel: readonly number[], tol: number): number {
  const best = Math.max(...rel);
  return rel.findIndex((r) => r >= best / (1.0 + tol));
}

function infeasibleMessage(labels: readonly string[], col: number, level: number, cap: number) {
  return (
    `LP is infeasible: the minimum of Σ aᵢ/ρᵢ leaves the artificial ${labels[col]} at ` +
    `${pyG(level, 6)} > tol·(bᵢ + Σⱼ|A_ij|zⱼ) = ${pyG(cap, 3)} (its row is violated)`
  );
}

/** Two-phase tableau simplex (Bertsimas & Tsitsiklis §3.5) on a b ≥ 0 standard form. */
export function solveTwoPhase(
  sf: StandardForm,
  o: { rule: string; tol: number; maxIter: number; record?: boolean },
): LPOutcome {
  const { rule, tol, maxIter, record = true } = o;
  const rec = new Recorder(sf, maxIter, record);
  const tab = initialTableau(sf);
  const N = sf.A.length ? sf.A[0].length : sf.c.length;
  const n = sf.n;

  const done = (status: string, message: string, ray: number[] | null = null): LPOutcome => {
    const x = tab.values().slice(0, n);
    const value = status === 'optimal' ? rec.objective(x) : null;
    return { status, message, tab, x, value, pivots: rec.k, trace: rec.trace, sf, ray };
  };

  let removedInfo: Record<string, unknown>;
  if (tab.artificial.size) {
    // Phase 1: min Σ aᵢ/ρᵢ.
    const wCost = artificialWeights(sf, tab.N);
    tab.T[0] = objectiveRow(tab, wCost);
    tab.costScale = [1.0];
    const allowed = new Array<boolean>(tab.N).fill(true);
    const status = primalLoop(tab, rec, { rule, tol, phase: 1, allowed, emitOptimal: false });
    if (status === 'max_iter')
      return done('max_iter', `reached max_iter=${maxIter} pivots in phase 1`);
    if (status === 'cycling')
      return done('cycling', 'cycling detected in phase 1: a basis repeated');
    // Phase 1 is bounded below by 0, so "unbounded" cannot occur here.
    const cap = artificialCaps(sf, tab.values(), tab.N, tol);
    const bad = artificialViolation(
      tab.basis,
      tab.rows().map((r) => r[tab.N]),
      cap,
      tab.artificial,
    );
    if (bad !== null) {
      rec.emit(tab, 1);
      return done('infeasible', infeasibleMessage(tab.labels, bad[0], bad[1], cap[bad[0]]));
    }
    // Drive zero-level artificials out of the basis; drop redundant rows.
    const removed: number[] = [];
    const ar = artRows(sf);
    let i = 0;
    while (i < tab.m) {
      if (!tab.artificial.has(tab.basis[i])) {
        i += 1;
        continue;
      }
      // NOTE: the level is ≤ capᵢ (accepted above); set it to exactly 0 (degenerate pivot).
      const level = tab.T[tab.nObj + i][tab.N];
      tab.T[tab.nObj + i][tab.N] = 0.0;
      tab.T[0][tab.N] += wCost[tab.basis[i]] * level;
      const row = tab.rows()[i].slice(0, N);
      const scale = tab.scale;
      const rel = row.map(
        (v, j) => Math.abs(v) / (tol * (scale ? scale[j] / scale[tab.basis[i]] : 1.0)),
      );
      const j = driveOutColumn(rel, tol);
      if (rel[j] > 1.0) {
        rec.emit(tab, 1, {
          entering: j,
          leaving: tab.basis[i],
          pivot_row: tab.nObj + i,
          ratio_test: null,
          drive_out: true,
        });
        if (rec.k >= rec.maxIter)
          return done('max_iter', `reached max_iter=${maxIter} pivots in phase 1`);
        tab.pivot(i, j);
        rec.k += 1;
        rec.lastTheta = 0.0;
        i += 1;
      } else {
        // Row i of B⁻¹A is zero on the real columns: the constraint is redundant.
        removed.push(ar[tab.basis[i] - N]);
        tab.T.splice(tab.nObj + i, 1);
        tab.basis.splice(i, 1);
      }
    }
    // Phase 2 tableau: delete the artificial columns, install the true objective row.
    const keep = Array.from({ length: tab.N }, (_, j) => j).filter((j) => !tab.artificial.has(j));
    const Nold = tab.N;
    tab.T = tab.T.map((row) => [...keep.map((j) => row[j]), row[Nold]]);
    tab.labels = keep.map((j) => tab.labels[j]);
    if (tab.scale) {
      const sc = tab.scale;
      tab.scale = keep.map((j) => sc[j]);
    }
    tab.artificial = new Set();
    tab.T[0] = objectiveRow(tab, sf.c);
    tab.costScale = [costScale(sf.c)];
    removedInfo = { removed_rows: removed };
  } else {
    tab.T[0] = objectiveRow(tab, sf.c);
    tab.costScale = [costScale(sf.c)];
    removedInfo = {};
  }

  const allowed = new Array<boolean>(tab.N).fill(true);
  const status = primalLoop(tab, rec, { rule, tol, phase: 2, allowed, firstInfo: removedInfo });
  return finish(status, tab, rec, done, maxIter);
}

type Done = (status: string, message: string, ray?: number[] | null) => LPOutcome;

function finish(status: string, tab: Tableau, rec: Recorder, done: Done, maxIter: number) {
  if (status === 'optimal') return done('optimal', 'optimal: every reduced cost c̄ⱼ ≥ −tol');
  if (status === 'unbounded') {
    const name = rec.unboundedCol !== null ? tab.labels[rec.unboundedCol] : '?';
    return done(
      'unbounded',
      `LP is unbounded: column ${name} has c̄ < 0 and no positive entry, ` +
        'so the objective improves without limit along the ray',
      rec.ray,
    );
  }
  if (status === 'cycling')
    return done('cycling', "cycling detected: a basis repeated (use pivot_rule='bland')");
  return done('max_iter', `reached max_iter=${maxIter} pivots`);
}

function toResult(method: string, out: LPOutcome): Result {
  const extra: Record<string, unknown> = { status: out.status, pivots: out.pivots };
  if (out.tab) extra.basis = [...out.tab.basis];
  if (out.ray) extra.ray = out.ray;
  const fun =
    out.value !== null ? out.value : out.trace.length ? out.trace[out.trace.length - 1].fun : null;
  return {
    method,
    x: out.x,
    fun,
    converged: out.status === 'optimal',
    message: out.message,
    nIter: out.pivots,
    nFev: 0,
    nGev: 0,
    nHev: 0,
    trace: out.trace,
    extra,
  };
}

function checkParams(rule: string, tol: number, maxIter: number, rules: readonly string[]) {
  if (!rules.includes(rule))
    throw new Error(`pivot_rule must be one of ${pyTuple(rules)}, got '${rule}'`);
  if (!(tol > 0)) throw new Error('tol must be positive');
  if (maxIter < 1) throw new Error('max_iter must be ≥ 1');
}

interface PivotParams {
  [key: string]: unknown;
  pivot_rule?: unknown;
  tol?: unknown;
  max_iter?: unknown;
}

function pivotParams(o: PivotParams) {
  return {
    rule: String(o.pivot_rule ?? 'dantzig'),
    tol: Number(o.tol ?? 1e-9),
    maxIter: Number(o.max_iter ?? 200),
  };
}

// ─────────────────────────────────────────────────────────────────────────────────────────
// Registered methods
// ─────────────────────────────────────────────────────────────────────────────────────────

/** Primal simplex, full tableau, for `A_ub x ≤ b_ub` with b ≥ 0 (the origin is feasible). */
export function simplex(problem: LinearProgram, o: PivotParams = {}): Result {
  const lp = checkLp(problem);
  const { rule, tol, maxIter } = pivotParams(o);
  checkParams(rule, tol, maxIter, PIVOT_RULES);
  if (lp.aEq && lp.aEq.length > 0 && lp.aEq[0].length > 0)
    throw new Error(`${lp.id}: simplex needs A_ub x ≤ b_ub only; use two_phase_simplex or big_m`);
  if (lp.bUb && lp.bUb.some((v) => v < 0))
    throw new Error(
      `${lp.id}: simplex needs b_ub ≥ 0 (origin feasible); ` +
        'use two_phase_simplex, big_m or dual_simplex',
    );
  const sf = standardForm(lp);
  return toResult('simplex', solveTwoPhase(sf, { rule, tol, maxIter }));
}

/** Two-phase simplex (Bertsimas & Tsitsiklis 1997, §3.5). */
export function twoPhaseSimplex(problem: LinearProgram, o: PivotParams = {}): Result {
  const lp = checkLp(problem);
  const { rule, tol, maxIter } = pivotParams(o);
  checkParams(rule, tol, maxIter, PIVOT_RULES);
  const sf = standardForm(lp);
  return toResult('two_phase_simplex', solveTwoPhase(sf, { rule, tol, maxIter }));
}

/** Big-M method with M kept symbolic (Hillier & Lieberman §4.6). */
export function bigM(problem: LinearProgram, o: PivotParams = {}): Result {
  const lp = checkLp(problem);
  const { rule, tol, maxIter } = pivotParams(o);
  checkParams(rule, tol, maxIter, PIVOT_RULES);
  const sf = standardForm(lp);
  const rec = new Recorder(sf, maxIter);
  const tab = initialTableau(sf, 2);
  const nTotal = tab.N;
  const costM = artificialWeights(sf, nTotal);
  const costC = zeros(nTotal);
  sf.c.forEach((v, j) => {
    costC[j] = v;
  });
  tab.T[0] = objectiveRow(tab, costM);
  tab.T[1] = objectiveRow(tab, costC);
  tab.costScale = [1.0, costScale(costC)];
  const allowed = new Array<boolean>(nTotal).fill(true);

  const status = primalLoop(tab, rec, {
    rule,
    tol,
    phase: 2,
    allowed,
    // [Mⱼ | −Σ wᵢaᵢ] = cost_M − cost_M[B]ᵀ B⁻¹[A | b]: exact once cost_M[B] = 0.
    afterPivot: () => {
      tab.T[0] = objectiveRow(tab, costM);
    },
  });

  const done: Done = (st, message, ray = null) => {
    const x = tab.values().slice(0, sf.n);
    const value = st === 'optimal' ? rec.objective(x) : null;
    return { status: st, message, tab, x, value, pivots: rec.k, trace: rec.trace, sf, ray };
  };
  const cap = artificialCaps(sf, tab.values(), nTotal, tol);
  const bad =
    status === 'optimal' || status === 'unbounded'
      ? artificialViolation(
          tab.basis,
          tab.rows().map((r) => r[tab.N]),
          cap,
          tab.artificial,
        )
      : null;
  const out =
    bad !== null
      ? done('infeasible', infeasibleMessage(tab.labels, bad[0], bad[1], cap[bad[0]]))
      : finish(status, tab, rec, done, maxIter);
  return toResult('big_m', out);
}

/** A x + s = b with one slack per row, b of any sign, A_eq split into ≤ and ≥ rows. */
export function inequalityForm(lp: LinearProgram): StandardForm {
  const { c, aUb, bUb, aEq, bEq } = lpArrays(lp);
  const a = [...aUb, ...aEq, ...aEq.map((r) => r.map((v) => -v))];
  const b = [...bUb, ...bEq, ...bEq.map((v) => -v)];
  const n = c.length,
    m = b.length;
  const A = a.map((row, i) => [...row, ...zeros(m).map((_, k) => (k === i ? 1.0 : 0.0))]);
  const sign = lp.sense === 'min' ? 1.0 : -1.0;
  return {
    A,
    b,
    c: [...c.map((v) => sign * v), ...zeros(m)],
    labels: [
      ...Array.from({ length: n }, (_, j) => `x${j + 1}`),
      ...Array.from({ length: m }, (_, i) => `s${i + 1}`),
    ],
    n,
    sign,
    unitCol: Array.from({ length: m }, (_, i) => n + i),
    exprConst: [...zeros(n), ...b],
    exprCoef: [
      ...Array.from({ length: n }, (_, j) => zeros(n).map((_, k) => (k === j ? 1.0 : 0.0))),
      ...a.map((row) => row.map((v) => -v)),
    ],
  };
}

function slackTableau(sf: StandardForm): Tableau {
  const m = sf.A.length;
  const N = m ? sf.A[0].length : sf.c.length;
  const T = zerosM(1 + m, N + 1);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < N; j++) T[1 + i][j] = sf.A[i][j];
    T[1 + i][N] = sf.b[i];
  }
  const basis = Array.from({ length: m }, (_, i) => sf.n + i); // slack basis B = I
  const tab = new Tableau(T, basis, [...sf.labels], 1, new Set(), columnScales(sf.A, sf.n, N));
  tab.T[0] = objectiveRow(tab, sf.c);
  tab.costScale = [costScale(sf.c)];
  return tab;
}

/** Entering column for leaving row r: argmin c̄ⱼ/|ā_rⱼ| over ā_rⱼ < −τ_rⱼ. */
export function dualRatioTest(
  tab: Tableau,
  r: number,
  tol: number,
): { col: number | null; ratios: (number | null)[] } {
  const N = tab.N;
  const row = tab.rows()[r].slice(0, N);
  const d = tab.T[0].slice(0, N);
  const scale = tab.scale;
  const mask = row.map((v, j) => (scale ? v < (-tol * scale[j]) / scale[tab.basis[r]] : v < -tol));
  for (const j of tab.basis) mask[j] = false;
  // NOTE: c̄ⱼ = −1e-17 (rounding) is treated as 0, so the ratio is ≥ 0.
  const ratios = row.map((v, j) => (mask[j] ? Math.max(d[j], 0.0) / -v : null));
  if (!mask.some(Boolean)) return { col: null, ratios };
  const best = Math.min(...(ratios.filter((x) => x !== null) as number[]));
  const col = ratios.findIndex((x, j) => mask[j] && (x as number) <= best * (1.0 + tol));
  return { col, ratios };
}

/** Dual simplex pivots until primal feasible / infeasible / cycling / max_iter. */
export function dualLoop(
  tab: Tableau,
  rec: Recorder,
  o: { rule: string; tol: number; b: readonly number[]; phase?: number },
): string {
  const { rule, tol, b, phase = 2 } = o;
  const seen = new Set<string>();
  for (;;) {
    const rows = tab.rows();
    const N = tab.N,
      m = tab.m;
    const rhs = rows.map((r) => r[N]);
    const pInf = rhs.reduce((s, v) => s + Math.max(-v, 0.0), 0);
    const key = basisKey(tab.basis);
    if (seen.has(key)) {
      rec.emit(tab, phase, { cycling: true, primal_infeasibility: pInf });
      return 'cycling';
    }
    seen.add(key);
    // Leave test x_B,i < −tol·(|B⁻¹|·|b|)ᵢ.
    const xScale = rows.map((row) => {
      let s = 0;
      for (let k = 0; k < m; k++) s += Math.abs(row[N - m + k]) * Math.abs(b[k]);
      return s;
    });
    const neg = rhs.flatMap((v, i) => (v < -tol * xScale[i] ? [i] : []));
    if (neg.length === 0) {
      rec.emit(tab, phase, { primal_infeasibility: pInf });
      return 'optimal';
    }
    let r: number;
    const byBasis = (cands: number[]) =>
      cands.reduce((best, i) => (tab.basis[i] < tab.basis[best] ? i : best), cands[0]);
    if (rule === 'bland') r = byBasis(neg);
    else {
      const worst = Math.min(...neg.map((i) => rhs[i]));
      r = byBasis(neg.filter((i) => rhs[i] <= worst + tol * (1.0 + Math.abs(worst))));
    }
    const { col: j, ratios } = dualRatioTest(tab, r, tol);
    rec.emit(tab, phase, {
      entering: j,
      leaving: tab.basis[r],
      pivot_row: tab.nObj + r,
      ratio_test: ratios,
      primal_infeasibility: pInf,
    });
    if (j === null) return 'infeasible';
    if (rec.k >= rec.maxIter) return 'max_iter';
    const theta = tab.T[0][j] / -tab.rows()[r][j];
    tab.pivot(r, j);
    rec.k += 1;
    rec.lastTheta = theta;
  }
}

function dualFinish(status: string, done: Done, maxIter: number) {
  if (status === 'optimal')
    return done('optimal', 'optimal: every basic value x_B,i ≥ −tol·(|B⁻¹||b|)ᵢ and every c̄ⱼ ≥ 0');
  if (status === 'infeasible')
    return done(
      'infeasible',
      'LP is infeasible: a row has a negative basic value and no negative entry ' +
        '(the dual is unbounded)',
    );
  if (status === 'cycling')
    return done('cycling', "cycling detected: a basis repeated (use pivot_rule='bland')");
  return done('max_iter', `reached max_iter=${maxIter} pivots`);
}

/** Dual simplex method, full-tableau implementation (Bertsimas & Tsitsiklis §4.5). */
export function dualSimplex(problem: LinearProgram, o: PivotParams = {}): Result {
  const lp = checkLp(problem);
  const { rule, tol, maxIter } = pivotParams(o);
  checkParams(rule, tol, maxIter, ['dantzig', 'bland']);
  const sf = inequalityForm(lp);
  const tab = slackTableau(sf);
  const thr = tab.costTol(0, tol);
  if (sf.c.slice(0, sf.n).some((v, j) => v < -thr[j]))
    throw new Error(
      `${lp.id}: dual_simplex needs a dual-feasible slack basis (min-form costs c̃ ≥ 0); ` +
        'use two_phase_simplex or big_m',
    );
  const rec = new Recorder(sf, maxIter);
  const status = dualLoop(tab, rec, { rule, tol, b: sf.b });
  const done: Done = (st, message, ray = null) => {
    const x = tab.values().slice(0, sf.n);
    const value = st === 'optimal' ? rec.objective(x) : null;
    return { status: st, message, tab, x, value, pivots: rec.k, trace: rec.trace, sf, ray };
  };
  return toResult('dual_simplex', dualFinish(status, done, maxIter));
}

// ─────────────────────────────────────────────────────────────────────────────────────────
// Revised simplex
// ─────────────────────────────────────────────────────────────────────────────────────────

export interface LUFactor {
  LU: number[][];
  p: number[];
  d: number[];
}

/** LU with partial pivoting of the row-equilibrated basis, (D B)[p] = L U (null: singular). */
export function luFactor(B: Matrix): LUFactor | null {
  const m = B.length;
  const rowMax = B.map((r) => maxAbs(r));
  if (rowMax.some((v) => v === 0.0)) return null;
  const d = rowMax.map((v) => 1.0 / v);
  const LU = B.map((row, i) => row.map((v) => v * d[i]));
  const p = Array.from({ length: m }, (_, i) => i);
  const small = m * EPS;
  for (let k = 0; k < m; k++) {
    let piv = k;
    for (let i = k + 1; i < m; i++) if (Math.abs(LU[i][k]) > Math.abs(LU[piv][k])) piv = i;
    if (Math.abs(LU[piv][k]) <= small) return null;
    if (piv !== k) {
      [LU[k], LU[piv]] = [LU[piv], LU[k]];
      [p[k], p[piv]] = [p[piv], p[k]];
    }
    for (let i = k + 1; i < m; i++) LU[i][k] /= LU[k][k];
    for (let i = k + 1; i < m; i++) for (let j = k + 1; j < m; j++) LU[i][j] -= LU[i][k] * LU[k][j];
  }
  return { LU, p, d };
}

/** Solve B y = rhs (or Bᵀ y = rhs) with the factor of `luFactor`. */
export function luSolve(f: LUFactor, rhs: readonly number[], trans = false): number[] {
  const { LU, p, d } = f;
  const m = LU.length;
  if (!trans) {
    const dr = rhs.map((v, i) => d[i] * v);
    const y = p.map((i) => dr[i]);
    for (let i = 0; i < m; i++) {
      let s = 0;
      for (let k = 0; k < i; k++) s += LU[i][k] * y[k];
      y[i] -= s;
    }
    for (let i = m - 1; i >= 0; i--) {
      let s = 0;
      for (let k = i + 1; k < m; k++) s += LU[i][k] * y[k];
      y[i] = (y[i] - s) / LU[i][i];
    }
    return y;
  }
  const v = [...rhs];
  for (let i = 0; i < m; i++) {
    let s = 0;
    for (let k = 0; k < i; k++) s += LU[k][i] * v[k];
    v[i] = (v[i] - s) / LU[i][i];
  }
  for (let i = m - 1; i >= 0; i--) {
    let s = 0;
    for (let k = i + 1; k < m; k++) s += LU[k][i] * v[k];
    v[i] -= s;
  }
  const y = zeros(m);
  p.forEach((pi, i) => {
    y[pi] = v[i];
  });
  return y.map((yi, i) => d[i] * yi);
}

/** Revised-simplex data: A (with artificial columns), b, the basis. */
class RevisedState {
  artRows: number[];
  A: number[][];
  b: number[];
  labels: string[];
  artificial: Set<number>;
  scale: number[];
  artCost: number[];
  basis: number[];

  constructor(sf: StandardForm) {
    const m = sf.A.length;
    const N = m ? sf.A[0].length : sf.c.length;
    this.artRows = artRows(sf);
    const nArt = this.artRows.length;
    this.A = zerosM(m, N + nArt);
    for (let i = 0; i < m; i++) for (let j = 0; j < N; j++) this.A[i][j] = sf.A[i][j];
    this.artRows.forEach((i, k) => {
      this.A[i][N + k] = 1.0;
    });
    this.b = [...sf.b];
    this.labels = [...sf.labels, ...this.artRows.map((i) => `a${i + 1}`)];
    this.artificial = new Set(Array.from({ length: nArt }, (_, k) => N + k));
    this.scale = columnScales(this.A, sf.n, N + nArt);
    this.artCost = artificialWeights(sf, N + nArt);
    this.basis = sf.unitCol.map((uc, i) => (uc !== null ? uc : N + this.artRows.indexOf(i)));
  }

  get N(): number {
    return this.A.length ? this.A[0].length : this.labels.length;
  }

  get m(): number {
    return this.A.length;
  }

  col(j: number): number[] {
    return this.A.map((r) => r[j]);
  }

  basisMatrix(): number[][] {
    return this.A.map((r) => this.basis.map((j) => r[j]));
  }
}

function revisedEntering(
  d: readonly number[],
  rule: string,
  tol: number,
  allowed: readonly boolean[],
  st: RevisedState,
  factor: LUFactor,
  gamma: number,
): number | null {
  const cand = d.flatMap((v, j) => (allowed[j] && v < -tol * st.scale[j] * gamma ? [j] : []));
  if (cand.length === 0) return null;
  if (rule === 'bland') return cand[0];
  let score = cand.map((j) => d[j]);
  if (rule === 'steepest_edge') {
    score = score.map((s, q) => {
      if (!st.m) return s / 1.0;
      const u = luSolve(factor, st.col(cand[q]));
      let ss = 0;
      for (const v of u) ss += v ** 2;
      return s / Math.sqrt(1.0 + ss);
    });
  }
  const best = Math.min(...score);
  const lim = best + tol * (1.0 + Math.abs(best));
  return cand[score.findIndex((s) => s <= lim)];
}

function revisedMessage(status: string, maxIter: number, phase: number) {
  if (status === 'singular') return `basis matrix became numerically singular in phase ${phase}`;
  if (status === 'cycling')
    return `cycling detected in phase ${phase}: a basis repeated (use pivot_rule='bland')`;
  return `reached max_iter=${maxIter} pivots`;
}

function revised(sf: StandardForm, rule: string, tol: number, maxIter: number): LPOutcome {
  const st = new RevisedState(sf);
  const rec = new Recorder(sf, maxIter);
  const n = sf.n;

  const tableauView = (factor: LUFactor, cost: readonly number[]): Tableau => {
    const N = st.N;
    const cols = Array.from({ length: N }, (_, j) => luSolve(factor, st.col(j)));
    const rhs = luSolve(factor, st.b);
    const rows = Array.from({ length: st.m }, (_, i) => [...cols.map((c) => c[i]), rhs[i]]);
    const T = [zeros(N + 1), ...rows];
    const tab = new Tableau(T, [...st.basis], [...st.labels], 1, new Set(st.artificial));
    tab.artCost = st.artCost.slice(0, N);
    tab.T[0] = objectiveRow(tab, cost);
    return tab;
  };

  const emptyTab = (cost: readonly number[]): Tableau => {
    const T = [[...cost, 0]];
    return new Tableau(T, [], [...st.labels], 1, new Set(st.artificial));
  };

  const done = (
    status: string,
    message: string,
    ray: number[] | null = null,
    xB: number[] | null = null,
  ): LPOutcome => {
    const z = zeros(st.N);
    if (xB) st.basis.forEach((j, i) => (z[j] = xB[i]));
    const x = z.slice(0, n);
    const value = status === 'optimal' ? rec.objective(x) : null;
    return { status, message, tab: null, x, value, pivots: rec.k, trace: rec.trace, sf, ray };
  };

  const runPhase = (
    cost: readonly number[],
    phase: number,
    emitOptimal: boolean,
    firstInfo: Record<string, unknown>,
  ): [string, number[] | null] => {
    const seen = new Set<string>();
    let extraInfo: Record<string, unknown> = { ...firstInfo };
    // γ = 1 for the phase-1 row (its costs are 1 on the equilibrated rows).
    const gamma = phase === 1 ? 1.0 : costScale(cost);
    for (;;) {
      const factor = st.m ? luFactor(st.basisMatrix()) : { LU: [], p: [], d: [] };
      if (factor === null) return ['singular', null];
      const xB = st.m ? luSolve(factor, st.b) : [];
      const y = st.m
        ? luSolve(
            factor,
            st.basis.map((j) => cost[j]),
            true,
          )
        : [];
      const d = cost.map((cj, j) => {
        let s = 0;
        for (let i = 0; i < st.m; i++) s += st.A[i][j] * y[i];
        return cj - s;
      });
      for (const j of st.basis) d[j] = 0.0; // NOTE: exact zeros for basic columns
      const tab = st.m ? tableauView(factor, cost) : emptyTab(cost);
      const key = basisKey(st.basis);
      const common: Record<string, unknown> = {
        duals: y,
        reduced_costs: d,
        direction: null,
        ...extraInfo,
      };
      extraInfo = {};
      if (seen.has(key)) {
        rec.emit(tab, phase, { cycling: true, ...common });
        return ['cycling', xB];
      }
      seen.add(key);
      const allowed = new Array<boolean>(st.N).fill(true);
      const q = revisedEntering(d, rule, tol, allowed, st, factor, gamma);
      if (q === null) {
        if (emitOptimal) rec.emit(tab, phase, common);
        return ['optimal', xB];
      }
      const u = st.m ? luSolve(factor, st.col(q)) : [];
      common.direction = u;
      const {
        row: r,
        theta,
        ratios,
      } = minRatio(
        xB,
        u,
        tol,
        st.basis,
        st.basis.map((jb) => (tol * st.scale[q]) / st.scale[jb]),
      );
      if (r === null) {
        const eta = zeros(st.N);
        eta[q] = 1.0;
        st.basis.forEach((jb, i) => (eta[jb] = -u[i]));
        rec.emit(tab, phase, {
          ...pivotInfo(tab, q, null, ratios),
          ray: eta.slice(0, n),
          ...common,
        });
        return ['unbounded', xB];
      }
      rec.emit(tab, phase, { ...pivotInfo(tab, q, r, ratios), ...common });
      if (rec.k >= rec.maxIter) return ['max_iter', xB];
      st.basis[r] = q;
      rec.k += 1;
      rec.lastTheta = theta;
    }
  };

  let firstInfo: Record<string, unknown> = {};
  if (st.artificial.size) {
    // Phase 1: min Σ aᵢ/ρᵢ.
    const wCost = artificialWeights(sf, st.N);
    const [status, xB0] = runPhase(wCost, 1, false, {});
    if (status === 'singular' || status === 'max_iter' || status === 'cycling')
      return done(status, revisedMessage(status, maxIter, 1), null, xB0);
    let xB = xB0 as number[];
    const z = zeros(st.N);
    st.basis.forEach((j, i) => (z[j] = xB[i]));
    const cap = artificialCaps(sf, z, st.N, tol);
    const bad = artificialViolation(st.basis, xB, cap, st.artificial);
    if (bad !== null) {
      const factor = luFactor(st.basisMatrix()) as LUFactor;
      rec.emit(tableauView(factor, wCost), 1);
      return done(
        'infeasible',
        infeasibleMessage(st.labels, bad[0], bad[1], cap[bad[0]]),
        null,
        xB,
      );
    }
    const removed: number[] = [];
    const origRow = Array.from({ length: st.m }, (_, i) => i);
    const NReal = sf.A.length ? sf.A[0].length : sf.c.length;
    let i = 0;
    while (i < st.m) {
      if (!st.artificial.has(st.basis[i])) {
        i += 1;
        continue;
      }
      // NOTE: set the accepted level to exactly 0 by moving b of the artificial's row.
      const colArt = st.col(st.basis[i]);
      let artRow = 0;
      for (let r = 1; r < colArt.length; r++) if (colArt[r] > colArt[artRow]) artRow = r;
      st.b[artRow] -= xB[i];
      xB[i] = 0.0;
      const factor = luFactor(st.basisMatrix());
      if (factor === null)
        return done('singular', revisedMessage('singular', maxIter, 1), null, xB);
      const e = zeros(st.m);
      e[i] = 1.0;
      const binvRow = luSolve(factor, e, true); // πᵀ = row i of B⁻¹
      const row = Array.from({ length: NReal }, (_, j) => {
        let s = 0;
        for (let r = 0; r < st.m; r++) s += binvRow[r] * st.A[r][j];
        return s;
      });
      const rel = row.map((v, j) => Math.abs(v) / ((tol * st.scale[j]) / st.scale[st.basis[i]]));
      const j = driveOutColumn(rel, tol);
      if (rel[j] > 1.0) {
        const tab = tableauView(factor, wCost);
        rec.emit(tab, 1, {
          entering: j,
          leaving: st.basis[i],
          pivot_row: 1 + i,
          ratio_test: null,
          drive_out: true,
        });
        if (rec.k >= rec.maxIter)
          return done('max_iter', revisedMessage('max_iter', maxIter, 1), null, xB);
        st.basis[i] = j;
        rec.k += 1;
        rec.lastTheta = 0.0;
        i += 1;
      } else {
        // Row i of B⁻¹A is zero on the real columns: constraint artRow is redundant.
        removed.push(origRow.splice(artRow, 1)[0]);
        st.A.splice(artRow, 1);
        st.b.splice(artRow, 1);
        xB = xB.filter((_, k) => k !== i);
        st.basis.splice(i, 1);
      }
    }
    st.A = st.A.map((r) => r.slice(0, NReal));
    st.scale = st.scale.slice(0, NReal);
    st.labels = st.labels.slice(0, NReal);
    st.artificial = new Set();
    firstInfo = { removed_rows: removed };
  }
  const [status, xB] = runPhase([...sf.c], 2, true, firstInfo);
  if (status === 'optimal')
    return done('optimal', 'optimal: every reduced cost c̄ⱼ ≥ −tol', null, xB);
  if (status === 'unbounded') {
    const last = rec.trace[rec.trace.length - 1].info;
    const name = st.labels[last.entering as number];
    return done(
      'unbounded',
      `LP is unbounded: column ${name} has c̄ < 0 and B⁻¹A_q ≤ 0, ` +
        'so the objective improves without limit along the ray',
      [...(last.ray as number[])],
      xB,
    );
  }
  return done(status, revisedMessage(status, maxIter, 2), null, xB);
}

/** Revised simplex method with an LU factorization of the basis matrix, two phases. */
export function revisedSimplex(problem: LinearProgram, o: PivotParams = {}): Result {
  const lp = checkLp(problem);
  const { rule, tol, maxIter } = pivotParams(o);
  checkParams(rule, tol, maxIter, PIVOT_RULES);
  const sf = standardForm(lp);
  return toResult('revised_simplex', revised(sf, rule, tol, maxIter));
}

// ─────────────────────────────────────────────────────────────────────────────────────────
// Registration (same ids, params and references as the Python @register calls)
// ─────────────────────────────────────────────────────────────────────────────────────────

const QUANT_PIVOT = [{ tex: '\\theta_k', key: 'stepSize' }];

registerMethod<LinearProgram>(
  {
    id: 'simplex',
    family: 'lp',
    name: 'Primal simplex (tableau)',
    params: PIVOT_PARAMS,
    needs: ['lp'],
    order: 'finite (vertex to adjacent vertex)',
    summary:
      'Start at the origin and pivot to a better adjacent vertex until no reduced cost is negative.',
    references: [...REFERENCES, 'Dantzig (1947); Hillier & Lieberman, §4.3–4.4'],
  },
  (p, o) => simplex(p, o),
  {
    rule: '\\bar c_j = c_j - \\mathbf{c}_B^{\\top}B^{-1}A_j < 0,\\qquad \\theta = \\min_{\\bar u_i > 0} \\frac{x_{B,i}}{\\bar u_i}',
    intuition:
      'Every basis is a vertex of the polygon. A negative reduced cost names an edge along which the objective improves; the ratio test walks that edge until the first constraint becomes tight.',
    order: 'finite',
    pros: ['Exact vertex optimum', 'Reduced costs certify optimality', 'Warm-starts after changes'],
    cons: ['Exponential worst case (Klee–Minty)', 'Degenerate pivots stall', 'Needs b ≥ 0 here'],
    quantities: QUANT_PIVOT,
  },
);

registerMethod<LinearProgram>(
  {
    id: 'two_phase_simplex',
    family: 'lp',
    name: 'Two-phase simplex',
    params: PIVOT_PARAMS,
    needs: ['lp'],
    order: 'finite (vertex to adjacent vertex)',
    summary:
      'Phase 1 minimizes the sum of artificial variables to find a vertex; phase 2 optimizes from it.',
    references: [...REFERENCES, 'Dantzig, Orden & Wolfe (1955)'],
  },
  (p, o) => twoPhaseSimplex(p, o),
  {
    rule: '\\text{I: } \\min \\textstyle\\sum_i a_i/\\rho_i \\;\\; \\text{s.t.}\\; A\\mathbf{z} + \\mathbf{a} = \\mathbf{b},\\qquad \\text{II: } \\min \\tilde{\\mathbf{c}}^{\\top}\\mathbf{z}',
    intuition:
      'When the origin is infeasible there is no vertex to start from. Phase 1 adds artificial variables and minimizes their sum; a zero minimum lands on a true vertex, and phase 2 runs the ordinary simplex from there.',
    order: 'finite',
    pros: ['Any right-hand side, equality rows', 'Detects infeasibility', 'Removes redundant rows'],
    cons: ['Two passes over the tableau', 'Phase 1 path ignores the objective'],
    quantities: [
      { tex: '\\text{phase}', key: 'info.phase' },
      { tex: '\\textstyle\\sum a_i/\\rho_i', key: 'info.infeasibility' },
      ...QUANT_PIVOT,
    ],
  },
);

registerMethod<LinearProgram>(
  {
    id: 'big_m',
    family: 'lp',
    name: 'Big-M simplex',
    params: PIVOT_PARAMS,
    needs: ['lp'],
    order: 'finite (vertex to adjacent vertex)',
    summary:
      'Give each artificial variable a huge cost M so that the simplex drives them to zero while it optimizes.',
    references: [...REFERENCES, 'Hillier & Lieberman, Introduction to Operations Research, §4.6'],
  },
  (p, o) => bigM(p, o),
  {
    rule: '\\min\\; \\tilde{\\mathbf{c}}^{\\top}\\mathbf{z} + M\\textstyle\\sum_i a_i/\\rho_i,\\qquad \\bar c_j = M_j\\,M + c_j',
    intuition:
      'One phase instead of two: each artificial variable costs M, larger than any number it is compared with, so reduced costs are pairs compared lexicographically. The M part is minimized first, then the true cost.',
    order: 'finite',
    pros: ['Single pass', 'Symbolic M: no cancellation error'],
    cons: ['Two objective rows to maintain', 'Infeasibility shows only at the end'],
    quantities: [{ tex: 'M\\text{-part}', key: 'info.infeasibility' }, ...QUANT_PIVOT],
  },
);

registerMethod<LinearProgram>(
  {
    id: 'dual_simplex',
    family: 'lp',
    name: 'Dual simplex (tableau)',
    params: [
      param.choice('pivot_rule', 'dantzig', ['dantzig', 'bland'], {
        help: 'Leaving row: most negative basic value, or smallest basic index (Bland).',
        label: 'Leaving rule',
      }),
      ...PIVOT_PARAMS.slice(1),
    ],
    needs: ['lp'],
    order: 'finite (dual vertex to adjacent dual vertex)',
    summary:
      'Keep every reduced cost ≥ 0 and pivot out negative basic variables until the basis is also primal feasible.',
    references: [
      'Lemke (1954)',
      'Bertsimas & Tsitsiklis, Introduction to Linear Optimization (1997), §4.5',
      'Chvátal, Linear Programming (1983), ch. 10',
    ],
  },
  (p, o) => dualSimplex(p, o),
  {
    rule: 'x_{B,r} < 0,\\qquad j = \\arg\\min_{\\bar a_{rj} < 0} \\frac{\\bar c_j}{|\\bar a_{rj}|}',
    intuition:
      'It starts at a basic point that is optimal in cost but outside the feasible set, and pivots out a negative basic variable each step. The iterates approach the polygon from outside and stop at the first feasible vertex.',
    order: 'finite',
    pros: ['Starts without phase 1 when c ≥ 0', 'The method of choice after adding a cut'],
    cons: ['Needs a dual-feasible start', 'Iterates are infeasible until the end'],
    quantities: [
      { tex: '\\textstyle\\sum (x_{B,i})_-', key: 'info.primal_infeasibility' },
      ...QUANT_PIVOT,
    ],
  },
);

registerMethod<LinearProgram>(
  {
    id: 'revised_simplex',
    family: 'lp',
    name: 'Revised simplex (LU)',
    params: PIVOT_PARAMS,
    needs: ['lp'],
    order: 'finite (vertex to adjacent vertex)',
    summary:
      'The simplex method without a tableau: solve with the basis matrix B for x_B, the duals and one column.',
    references: [
      'Bertsimas & Tsitsiklis, Introduction to Linear Optimization (1997), §3.3 (revised simplex)',
      'Chvátal, Linear Programming (1983), ch. 7',
      'Golub & Van Loan, Matrix Computations (4th ed.), Alg. 3.4.1',
    ],
  },
  (p, o) => revisedSimplex(p, o),
  {
    rule: 'B\\mathbf{x}_B = \\mathbf{b},\\quad B^{\\top}\\mathbf{y} = \\mathbf{c}_B,\\quad \\bar{\\mathbf{c}} = \\mathbf{c} - A^{\\top}\\mathbf{y},\\quad B\\bar{\\mathbf{u}} = A_q',
    intuition:
      'The same vertex walk as the tableau simplex, but only the basis matrix B is factored. Each iteration solves three small systems: for the vertex, the simplex multipliers and the entering column.',
    order: 'finite',
    pros: ['Memory O(m²) instead of O(mn)', 'Same bases as the tableau method'],
    cons: ['Refactors B every iteration here (O(m³))'],
    quantities: [{ tex: '\\mathbf{y}', key: 'info.duals' }, ...QUANT_PIVOT],
  },
);
