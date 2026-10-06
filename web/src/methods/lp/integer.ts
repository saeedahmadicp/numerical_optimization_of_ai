/**
 * Integer linear programming — TS port of `numopt.lp.integer` (src/numopt/lp/integer.py):
 * LP-based branch and bound and Gomory fractional cuts.
 *
 * `branch_and_bound` solves its relaxations with the floating-point tableau simplex
 * (`solveTwoPhase`, Bland's rule); `gomory_cuts` runs everything, the LP relaxation included,
 * in exact rational arithmetic (`Q` below, BigInt numerator/denominator, the counterpart of
 * Python's `fractions.Fraction`).
 *
 * Info keys (snake_case, as in Python):
 *   branch_and_bound: node, tree [{id, parent, depth, branch, bounds, lp_value, lp_x, status}],
 *     bounds, lp_x, branch_var, incumbent, incumbent_value, best_bound, gap.
 *   gomory_cuts: tableau, row_labels, col_labels, basis, nonbasis, vertex,
 *     cuts [{coef, rhs, active}], cut {coef, rhs, source_row, source_var, f0, f}, dual_pivots.
 */
import { registerMethod, param } from '../../core/registry';
import type { LinearProgram, Result, Step } from '../../core/types';
import {
  checkLp,
  dotv,
  lpArrays,
  solveTwoPhase,
  standardForm,
  zeros,
  type StandardForm,
} from './simplex';

/** Pivot limit for one LP relaxation. */
const LP_MAX_PIVOTS = 5000;
const LP_TOL = 1e-9;
/** gomory_cuts: drop basic cut rows once the tableau holds more cut columns than this. */
const MAX_CUT_ROWS = 50;

function integerMask(lp: LinearProgram): boolean[] {
  const n = lp.c.length;
  if (!lp.integer || lp.integer.length === 0) return new Array<boolean>(n).fill(true);
  if (lp.integer.length !== n)
    throw new Error(`${lp.id}: integer flags must have one entry per variable`);
  return lp.integer.map(Boolean);
}

/** Python `round` (half to even). */
export function roundHalfEven(v: number): number {
  const r = Math.round(v);
  return Math.abs(v - Math.trunc(v)) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** Distance from v to the nearest integer. */
const fracDist = (v: number) => Math.abs(v - roundHalfEven(v));

// ─────────────────────────────────────────────────────────────────────────────────────────
// Branch and bound
// ─────────────────────────────────────────────────────────────────────────────────────────

/** The relaxation with the node bounds lo ≤ x ≤ hi appended as ≤ rows. */
function nodeLp(lp: LinearProgram, lo: readonly number[], hi: readonly number[]): LinearProgram {
  const { c, aUb, bUb } = lpArrays(lp);
  const n = c.length;
  const rows = [...aUb],
    rhs = [...bUb];
  for (let j = 0; j < n; j++) {
    if (Number.isFinite(hi[j])) {
      const e = zeros(n);
      e[j] = 1.0;
      rows.push(e);
      rhs.push(hi[j]);
    }
    if (lo[j] > 0) {
      const e = zeros(n);
      e[j] = -1.0;
      rows.push(e);
      rhs.push(-lo[j]);
    }
  }
  return { ...lp, aUb: rows, bUb: rhs, integer: [] };
}

export interface BBNode {
  id: number;
  parent: number | null;
  depth: number;
  /** [j, '<=' | '>=', v]: the bound added to the parent (null at the root). */
  branch: [number, '<=' | '>=', number] | null;
  /** [lo, hi] per variable (hi null when unbounded). */
  bounds: [number, number | null][];
  lp_value: number | null;
  lp_x: number[] | null;
  status: 'open' | 'branched' | 'integer' | 'infeasible' | 'pruned_bound' | 'unbounded';
}

interface BBParams {
  [key: string]: unknown;
  strategy?: unknown;
  int_tol?: unknown;
  max_iter?: unknown;
}

/** LP-based branch and bound (Land & Doig 1960; Wolsey 1998, §7.3). */
export function branchAndBound(problem: LinearProgram, o: BBParams = {}): Result {
  const lp = checkLp(problem);
  const strategy = String(o.strategy ?? 'best_bound');
  const intTol = Number(o.int_tol ?? 1e-6);
  const maxIter = Number(o.max_iter ?? 200);
  if (strategy !== 'best_bound' && strategy !== 'depth_first')
    throw new Error("strategy must be 'best_bound' or 'depth_first'");
  if (!(0 < intTol && intTol < 0.5) || maxIter < 1)
    throw new Error('int_tol must be in (0, 0.5) and max_iter ≥ 1');
  const isInt = integerMask(lp);
  const c = [...lp.c];
  const n = c.length;
  const sign = lp.sense === 'min' ? 1.0 : -1.0; // min-form value = sign · cᵀx

  const tree: BBNode[] = [];
  const bounds: [number[], number[]][] = [];
  const parentBound: number[] = []; // min-form LP value of the parent (−inf at the root)

  const newNode = (parent: number | null, branch: BBNode['branch'], lo: number[], hi: number[]) => {
    const nid = tree.length;
    tree.push({
      id: nid,
      parent,
      depth: parent === null ? 0 : tree[parent].depth + 1,
      branch,
      bounds: lo.map((l, j) => [l, Number.isFinite(hi[j]) ? hi[j] : null]),
      lp_value: null,
      lp_x: null,
      status: 'open',
    });
    bounds.push([lo, hi]);
    parentBound.push(parent === null ? -Infinity : sign * (tree[parent].lp_value as number));
  };

  newNode(null, null, zeros(n), new Array<number>(n).fill(Infinity));
  let incX: number[] | null = null;
  let incVal = Infinity; // min form
  const trace: Step[] = [];
  let totalPivots = 0;
  let status: string | null = null;
  let lpMessage = '';

  const pruneTol = (v: number) => 1e-9 * (1.0 + Math.abs(v));
  const openNodes = () => tree.filter((t) => t.status === 'open').map((t) => t.id);
  const bestOpenBound = () => Math.min(incVal, ...openNodes().map((i) => parentBound[i]));

  let k = 0;
  for (;;) {
    const candidates = openNodes();
    if (candidates.length === 0) {
      status = incX !== null ? 'optimal' : 'infeasible';
      break;
    }
    if (k > maxIter) {
      status = 'max_iter';
      break;
    }
    let nid = candidates[0];
    for (const i of candidates) {
      const better =
        strategy === 'best_bound'
          ? parentBound[i] < parentBound[nid]
          : tree[i].depth > tree[nid].depth;
      if (better) nid = i;
    }
    const node = tree[nid];
    const [lo, hi] = bounds[nid];
    let lpX: number[] | null = null;
    let branchVar: number | null = null;
    if (parentBound[nid] >= incVal - pruneTol(incVal)) {
      node.status = 'pruned_bound';
    } else {
      const out = solveTwoPhase(standardForm(nodeLp(lp, lo, hi)), {
        rule: 'bland',
        tol: LP_TOL,
        maxIter: LP_MAX_PIVOTS,
        record: false,
      });
      totalPivots += out.pivots;
      if (out.status === 'infeasible') node.status = 'infeasible';
      else if (out.status === 'unbounded') {
        node.status = 'unbounded';
        status = 'unbounded';
      } else if (out.status !== 'optimal') {
        node.status = 'infeasible';
        status = 'lp_failure';
        lpMessage = out.message;
      } else {
        lpX = [...out.x];
        const val = sign * dotv(c, lpX);
        node.lp_value = dotv(c, lpX);
        node.lp_x = [...lpX];
        if (val >= incVal - pruneTol(incVal)) node.status = 'pruned_bound';
        else {
          const x = lpX;
          const frac = x.map((v, j) => (isInt[j] ? fracDist(v) : 0.0));
          if (frac.every((f) => f <= intTol)) {
            node.status = 'integer';
            incX = x.map((v, j) => (isInt[j] ? roundHalfEven(v) : v));
            incVal = sign * dotv(c, incX);
          } else {
            node.status = 'branched';
            // Most fractional: distance to nearest integer closest to ½.
            const score = frac.map((f) => (f > intTol ? f : -1.0));
            const top = Math.max(...score);
            branchVar = score.findIndex((s) => s >= top - 1e-12);
            const v = x[branchVar];
            const hiDn = [...hi],
              loUp = [...lo];
            hiDn[branchVar] = Math.floor(v);
            loUp[branchVar] = Math.ceil(v);
            newNode(nid, [branchVar, '<=', Math.floor(v)], [...lo], hiDn);
            newNode(nid, [branchVar, '>=', Math.ceil(v)], loUp, [...hi]);
          }
        }
      }
    }
    const bb = bestOpenBound();
    const incOrig = incX === null ? null : dotv(c, incX);
    const bbOrig = Number.isFinite(bb) ? sign * bb : null;
    const gap =
      incOrig === null || bbOrig === null
        ? null
        : Math.abs(incOrig - bbOrig) / (1.0 + Math.abs(incOrig));
    trace.push({
      k,
      x: lpX as unknown as number[],
      fun: lpX === null ? null : dotv(c, lpX),
      gradNorm: null,
      stepSize: null,
      info: {
        node: nid,
        tree: tree.map((t) => ({ ...t })),
        bounds: node.bounds,
        lp_x: lpX,
        branch_var: branchVar,
        incumbent: incX === null ? null : [...incX],
        incumbent_value: incOrig,
        best_bound: bbOrig,
        gap,
      },
    });
    if (status === 'unbounded' || status === 'lp_failure') break;
    k += 1;
  }

  const messages: Record<string, string> = {
    optimal: `optimal: search tree exhausted after ${trace.length} nodes`,
    infeasible:
      'ILP is infeasible: every node was infeasible or pruned without an integer solution',
    unbounded: 'LP relaxation is unbounded: the ILP is unbounded or infeasible',
    max_iter:
      `reached max_iter=${maxIter}: processed the root and ${maxIter} more nodes ` +
      'with open nodes left',
  };
  let message = messages[status ?? ''] ?? '';
  if (status === 'lp_failure') message = `an LP relaxation failed: ${lpMessage}`;
  const xOut = incX ?? new Array<number>(n).fill(NaN);
  return {
    method: 'branch_and_bound',
    x: xOut,
    fun: incX === null ? null : dotv(c, incX),
    converged: status === 'optimal',
    message,
    nIter: trace[trace.length - 1].k,
    nFev: 0,
    nGev: 0,
    nHev: 0,
    trace,
    extra: {
      status,
      nodes: tree.length,
      lp_pivots: totalPivots,
      tree: tree.map((t) => ({ ...t })),
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────────────────────
// Exact rationals
// ─────────────────────────────────────────────────────────────────────────────────────────

const gcd = (a: bigint, b: bigint): bigint => {
  a = a < 0n ? -a : a;
  b = b < 0n ? -b : b;
  while (b) [a, b] = [b, a % b];
  return a;
};

/** An exact rational n/d with d > 0 and gcd(n, d) = 1 (Python `fractions.Fraction`). */
export class Q {
  readonly n: bigint;
  readonly d: bigint;

  constructor(n: bigint, d: bigint = 1n) {
    if (d === 0n) throw new RangeError('zero denominator');
    if (d < 0n) {
      n = -n;
      d = -d;
    }
    const g = gcd(n, d) || 1n;
    this.n = n / g;
    this.d = d / g;
  }

  static ZERO = new Q(0n);
  static ONE = new Q(1n);

  static int(v: number): Q {
    return new Q(BigInt(Math.trunc(v)));
  }

  /** The exact binary value of a finite double (`Fraction(float(v))`). */
  static fromFloat(v: number): Q {
    if (!Number.isFinite(v)) throw new RangeError('non-finite value');
    if (Number.isInteger(v)) return new Q(BigInt(v));
    let e = 0n;
    let x = v;
    while (!Number.isInteger(x)) {
      x *= 2;
      e += 1n;
    }
    return new Q(BigInt(x), 1n << e);
  }

  add(o: Q): Q {
    return new Q(this.n * o.d + o.n * this.d, this.d * o.d);
  }
  sub(o: Q): Q {
    return new Q(this.n * o.d - o.n * this.d, this.d * o.d);
  }
  mul(o: Q): Q {
    return new Q(this.n * o.n, this.d * o.d);
  }
  div(o: Q): Q {
    return new Q(this.n * o.d, this.d * o.n);
  }
  neg(): Q {
    return new Q(-this.n, this.d);
  }
  cmp(o: Q): number {
    const a = this.n * o.d,
      b = o.n * this.d;
    return a < b ? -1 : a > b ? 1 : 0;
  }
  isZero(): boolean {
    return this.n === 0n;
  }
  sign(): number {
    return this.n < 0n ? -1 : this.n > 0n ? 1 : 0;
  }
  /** ⌊v⌋ as a rational. */
  floor(): Q {
    let q = this.n / this.d;
    if (this.n < 0n && q * this.d !== this.n) q -= 1n;
    return new Q(q);
  }
  /** Fractional part v − ⌊v⌋ ∈ [0, 1). */
  frac(): Q {
    return this.sub(this.floor());
  }
  /** Nearest double (exact for the small rationals of the library problems). */
  toNumber(): number {
    const n = this.n,
      d = this.d;
    const lim = 1n << 53n;
    if ((n < 0n ? -n : n) < lim && d < lim) return Number(n) / Number(d);
    // Scale both to ~64 significant bits, then divide (relative error ≲ 2⁻⁵²).
    const bits = (v: bigint) => (v < 0n ? -v : v).toString(2).length;
    const shift = BigInt(Math.max(0, Math.max(bits(n), bits(d)) - 64));
    return Number(n >> shift) / Number(d >> shift);
  }
}

const qmin = <T>(xs: readonly T[], key: (x: T) => [Q, number]): T => {
  let best = xs[0];
  let bk = key(best);
  for (const x of xs.slice(1)) {
    const k = key(x);
    const c = k[0].cmp(bk[0]);
    if (c < 0 || (c === 0 && k[1] < bk[1])) {
      best = x;
      bk = k;
    }
  }
  return best;
};

/** A simplex tableau over ℚ, same layout as the float `Tableau` (one objective row). */
class QTableau {
  T: Q[][];
  basis: number[];
  labels: string[];

  constructor(T: Q[][], basis: number[], labels: string[]) {
    this.T = T;
    this.basis = basis;
    this.labels = labels;
  }

  get m() {
    return this.T.length - 1;
  }
  get N() {
    return this.T[0].length - 1;
  }

  pivot(i: number, j: number) {
    const r = 1 + i;
    const piv = this.T[r][j];
    const prow = this.T[r].map((v) => v.div(piv));
    this.T[r] = prow;
    this.T.forEach((row, k) => {
      const f = row[j];
      if (k !== r && !f.isZero()) this.T[k] = row.map((a, q) => a.sub(f.mul(prow[q])));
    });
    this.basis[i] = j;
  }

  setObjective(cost: readonly Q[]) {
    let row = [...cost, Q.ZERO];
    this.basis.forEach((jb, i) => {
      const cb = cost[jb];
      if (!cb.isZero()) row = row.map((a, q) => a.sub(cb.mul(this.T[1 + i][q])));
    });
    this.T[0] = row;
  }

  values(): Q[] {
    const z = new Array<Q>(this.N).fill(Q.ZERO);
    this.basis.forEach((j, i) => {
      z[j] = this.T[1 + i][this.N];
    });
    return z;
  }

  snapshot(): Record<string, unknown> {
    const basic = new Set(this.basis);
    return {
      tableau: this.T.map((row) => row.map((v) => v.toNumber())),
      row_labels: ['z', ...this.basis.map((j) => this.labels[j])],
      col_labels: [...this.labels, 'rhs'],
      basis: [...this.basis],
      nonbasis: Array.from({ length: this.N }, (_, j) => j).filter((j) => !basic.has(j)),
    };
  }
}

/** Primal simplex with Bland's rule (exact) until optimal / unbounded / budget. */
function qPrimal(tab: QTableau, budget: number): [string, number] {
  let pivots = 0;
  for (;;) {
    const obj = tab.T[0];
    let j = -1;
    for (let q = 0; q < tab.N; q++)
      if (obj[q].sign() < 0) {
        j = q;
        break;
      }
    if (j < 0) return ['optimal', pivots];
    const rows = Array.from({ length: tab.m }, (_, i) => i).filter(
      (i) => tab.T[1 + i][j].sign() > 0,
    );
    if (rows.length === 0) return ['unbounded', pivots];
    if (pivots >= budget) return ['max_iter', pivots];
    const col = j;
    const i = qmin(rows, (i) => [tab.T[1 + i][tab.N].div(tab.T[1 + i][col]), tab.basis[i]]);
    tab.pivot(i, col);
    pivots += 1;
  }
}

/** Dual simplex with Bland's rule (exact) until primal feasible. */
function qDual(tab: QTableau, budget: number): [string, number] {
  let pivots = 0;
  for (;;) {
    const neg = Array.from({ length: tab.m }, (_, i) => i).filter(
      (i) => tab.T[1 + i][tab.N].sign() < 0,
    );
    if (neg.length === 0) return ['optimal', pivots];
    const r = neg.reduce((b, i) => (tab.basis[i] < tab.basis[b] ? i : b), neg[0]);
    const row = tab.T[1 + r];
    const cols = Array.from({ length: tab.N }, (_, j) => j).filter((j) => row[j].sign() < 0);
    if (cols.length === 0) return ['infeasible', pivots];
    if (pivots >= budget) return ['max_iter', pivots];
    const j = qmin(cols, (j) => [tab.T[0][j].div(row[j].neg()), j]);
    tab.pivot(r, j);
    pivots += 1;
  }
}

/** Two-phase simplex over ℚ on an integer standard form. */
function qTwoPhase(sf: StandardForm, budget: number): [string, QTableau, number] {
  const m = sf.A.length;
  const N = m ? sf.A[0].length : sf.c.length;
  const art = sf.unitCol.map((u, i) => (u === null ? i : -1)).filter((i) => i >= 0);
  const nArt = art.length;
  const T: Q[][] = [new Array<Q>(N + nArt + 1).fill(Q.ZERO)];
  const basis: number[] = [];
  for (let i = 0; i < m; i++) {
    const row = [
      ...sf.A[i].map((v) => Q.int(v)),
      ...new Array<Q>(nArt).fill(Q.ZERO),
      Q.int(sf.b[i]),
    ];
    const uc = sf.unitCol[i];
    if (uc === null) {
      const k = art.indexOf(i);
      row[N + k] = Q.ONE;
      basis.push(N + k);
    } else basis.push(uc);
    T.push(row);
  }
  const tab = new QTableau(T, basis, [...sf.labels, ...art.map((i) => `a${i + 1}`)]);
  let pivots = 0;
  if (nArt) {
    tab.setObjective([...new Array<Q>(N).fill(Q.ZERO), ...new Array<Q>(nArt).fill(Q.ONE)]);
    const [status, p] = qPrimal(tab, budget);
    pivots += p;
    if (status === 'max_iter') return [status, tab, pivots];
    if (!tab.T[0][tab.N].isZero()) return ['infeasible', tab, pivots];
    let i = 0;
    while (i < tab.m) {
      if (tab.basis[i] < N) {
        i += 1;
        continue;
      }
      let j = -1;
      for (let q = 0; q < N; q++)
        if (!tab.T[1 + i][q].isZero()) {
          j = q;
          break;
        }
      if (j < 0) {
        tab.T.splice(1 + i, 1);
        tab.basis.splice(i, 1);
      } else {
        tab.pivot(i, j);
        pivots += 1;
        i += 1;
      }
    }
    tab.T = tab.T.map((row) => [...row.slice(0, N), row[row.length - 1]]);
    tab.labels = tab.labels.slice(0, N);
  }
  tab.setObjective(sf.c.map((v) => Q.fromFloat(v)));
  const [status, p] = qPrimal(tab, budget - pivots);
  return [status, tab, pivots + p];
}

const isIntegerArray = (arr: readonly number[]) =>
  arr.every((v) => v === roundHalfEven(v) && Math.abs(v) < 2 ** 53);

export interface GomoryCut {
  coef: number[];
  rhs: number;
  source_row: number;
  source_var: string;
  f0: number;
  f: number[];
}

/** Gomory's fractional cutting-plane algorithm for pure integer programs (exact arithmetic). */
export function gomoryCuts(
  problem: LinearProgram,
  o: { [key: string]: unknown; max_iter?: unknown } = {},
): Result {
  const lp = checkLp(problem);
  const maxIter = Number(o.max_iter ?? 50);
  if (maxIter < 1) throw new Error('max_iter must be ≥ 1');
  if (!integerMask(lp).every(Boolean))
    throw new Error(`${lp.id}: gomory_cuts needs every variable integer (pure ILP)`);
  const { aUb, bUb, aEq, bEq } = lpArrays(lp);
  for (const arr of [aUb.flat(), bUb, aEq.flat(), bEq])
    if (arr.length && !isIntegerArray(arr))
      throw new Error(`${lp.id}: gomory_cuts needs integer constraint data A and b`);
  const sf = standardForm(lp);
  const cQ = lp.c.map((v) => Q.fromFloat(v));
  const n = lp.c.length;
  const [lpStatus, tab, lpPivots] = qTwoPhase(sf, LP_MAX_PIVOTS);
  const exprConst = sf.exprConst.map((v) => Q.int(v));
  const exprCoef = sf.exprCoef.map((row) => row.map((v) => Q.int(v)));
  const cuts: GomoryCut[] = [];
  const active: boolean[] = [];
  const trace: Step[] = [];
  let totalDual = 0;
  const nStd = tab.N; // columns ≥ nStd are cut slacks g1, g2, ...

  const dropBasicCuts = () => {
    const rowsDesc = tab.basis
      .map((j, i) => (j >= nStd ? i : -1))
      .filter((i) => i >= 0)
      .sort((a, b) => b - a);
    for (const i of rowsDesc) {
      const j = tab.basis[i];
      active[Number(tab.labels[j].slice(1)) - 1] = false;
      tab.T.splice(1 + i, 1);
      tab.basis.splice(i, 1);
      for (const t of tab.T) t.splice(j, 1);
      tab.basis = tab.basis.map((jb) => (jb > j ? jb - 1 : jb));
      tab.labels.splice(j, 1);
      exprConst.splice(j, 1);
      exprCoef.splice(j, 1);
    }
  };

  const emit = (k: number, cut: GomoryCut | null, dualPivots: number) => {
    const z = tab.values().slice(0, n);
    const x = z.map((v) => v.toNumber());
    const fun = cQ.reduce((s, cj, j) => s.add(cj.mul(z[j])), Q.ZERO).toNumber();
    trace.push({
      k,
      x,
      fun,
      gradNorm: null,
      stepSize: null,
      info: {
        ...tab.snapshot(),
        vertex: [...x],
        cuts: cuts.map((q, i) => ({ coef: q.coef, rhs: q.rhs, active: active[i] })),
        cut,
        dual_pivots: dualPivots,
      },
    });
  };

  const make = (status: string, message: string, extra: Record<string, unknown>): Result => {
    const final = trace[trace.length - 1];
    return {
      method: 'gomory_cuts',
      x: [...(final.x as number[])],
      fun: final.fun,
      converged: status === 'optimal',
      message,
      nIter: final.k,
      nFev: 0,
      nGev: 0,
      nHev: 0,
      trace,
      extra: { status, ...extra },
    };
  };

  emit(0, null, 0);
  if (lpStatus !== 'optimal') {
    const status = lpStatus === 'infeasible' || lpStatus === 'unbounded' ? lpStatus : 'lp_failure';
    const messages: Record<string, string> = {
      infeasible: 'LP relaxation is infeasible: phase-1 minimum of the artificial sum > 0',
      unbounded: 'LP relaxation is unbounded: the ILP is unbounded or infeasible',
    };
    return make(status, messages[status] ?? `LP relaxation not solved: ${lpStatus}`, {
      cuts: [],
      dual_pivots: 0,
      lp_pivots: lpPivots,
    });
  }

  let status = 'max_iter',
    message = `reached max_iter=${maxIter} cuts`;
  let k = 0;
  for (;;) {
    const fRhs = Array.from({ length: tab.m }, (_, i) => tab.T[1 + i][tab.N].frac());
    if (fRhs.every((f) => f.isZero())) {
      status = 'optimal';
      message = 'optimal: every basic value of the LP optimum is integral';
      break;
    }
    if (k >= maxIter) break;
    // Largest f(b̄_r), ties: lowest row.
    let r = 0;
    for (let i = 1; i < tab.m; i++) if (fRhs[i].cmp(fRhs[r]) > 0) r = i;
    const f0 = fRhs[r];
    const row = tab.T[1 + r];
    const basic = new Set(tab.basis);
    const f = Array.from({ length: tab.N }, (_, j) => (basic.has(j) ? Q.ZERO : row[j].frac()));
    // Cut in the original variables: Σ f_j (e0_j + e_jᵀx) ≥ f0  ⇔  coefᵀx ≤ rhs.
    const coefX = Array.from({ length: n }, (_, i) =>
      f.reduce((s, fj, j) => s.add(fj.mul(exprCoef[j][i])), Q.ZERO).neg(),
    );
    const rhsX = f.reduce((s, fj, j) => s.add(fj.mul(exprConst[j])), Q.ZERO).sub(f0);
    const cut: GomoryCut = {
      coef: coefX.map((v) => v.toNumber()),
      rhs: rhsX.toNumber(),
      source_row: 1 + r,
      source_var: tab.labels[tab.basis[r]],
      f0: f0.toNumber(),
      f: f.map((v) => v.toNumber()),
    };
    // The slack g = Σ f_j z_j − f0 = rhs_x − coef_xᵀx in the original variables.
    exprConst.push(rhsX);
    exprCoef.push(coefX.map((v) => v.neg()));
    // Append column g and the row −f·z + g = −f0 (g is basic in it).
    tab.T = tab.T.map((t) => [...t.slice(0, -1), Q.ZERO, t[t.length - 1]]);
    tab.T.push([...f.map((v) => v.neg()), Q.ONE, f0.neg()]);
    tab.labels = [...tab.labels, `g${cuts.length + 1}`];
    tab.basis = [...tab.basis, tab.N - 1];
    cuts.push(cut);
    active.push(true);
    const [st, dualPivots] = qDual(tab, LP_MAX_PIVOTS);
    totalDual += dualPivots;
    if (st === 'optimal' && tab.N - nStd > MAX_CUT_ROWS) dropBasicCuts();
    k += 1;
    emit(k, cut, dualPivots);
    if (st === 'infeasible') {
      status = 'infeasible';
      message =
        'ILP is infeasible: after a valid cut the LP relaxation is infeasible ' +
        '(exact dual simplex)';
      break;
    }
    if (st !== 'optimal') {
      status = 'lp_failure';
      message = `dual simplex stopped (${st}) after a cut`;
      break;
    }
  }
  return make(status, message, {
    cuts: cuts.map((q, i) => ({ ...q, active: active[i] })),
    dual_pivots: totalDual,
    lp_pivots: lpPivots,
  });
}

// ─────────────────────────────────────────────────────────────────────────────────────────
// Registration
// ─────────────────────────────────────────────────────────────────────────────────────────

registerMethod<LinearProgram>(
  {
    id: 'branch_and_bound',
    family: 'lp',
    name: 'Branch and bound',
    params: [
      param.choice('strategy', 'best_bound', ['best_bound', 'depth_first'], {
        help: 'Node selection: best LP bound first, or deepest node first.',
        label: 'Node selection',
      }),
      param.float('int_tol', 1e-6, {
        min: 1e-10,
        max: 1e-2,
        log: true,
        help: 'A value within int_tol of an integer counts as integral.',
        label: 'Integrality tolerance',
      }),
      param.int('max_iter', 200, {
        min: 1,
        max: 10_000,
        help: 'Maximum number of nodes processed after the root (Step 0 is the root).',
        label: 'Node budget',
      }),
    ],
    needs: ['lp', 'integer'],
    order: 'finite (exponential worst case)',
    summary:
      'Solve LP relaxations, split on a fractional variable, and discard sub-problems whose bound cannot beat the best integer solution.',
    references: [
      'Land & Doig (1960), Econometrica 28(3):497–520',
      'Wolsey, Integer Programming (1998), §7.3–7.4',
      'Winston, Operations Research (4th ed.), §9.3',
    ],
  },
  (p, o) => branchAndBound(p, o),
  {
    rule: 'x_j \\le \\lfloor x_j^{\\text{LP}} \\rfloor \\;\\;\\vee\\;\\; x_j \\ge \\lceil x_j^{\\text{LP}} \\rceil,\\qquad \\text{prune if } z^{\\text{LP}} \\le z^{\\text{inc}}',
    intuition:
      'Solve the LP relaxation; if a variable is fractional, split the box in two so that the fractional point lies in neither half. A node whose LP bound cannot beat the best integer point found so far is pruned without being explored.',
    order: 'finite',
    pros: ['Exact optimum with a certified gap', 'Works for mixed-integer programs'],
    cons: ['Exponential worst case', 'Solves one LP per node (no warm start here)'],
    quantities: [
      { tex: '\\text{node}', key: 'info.node' },
      { tex: 'z^{\\text{inc}}', key: 'info.incumbent_value' },
      { tex: '\\bar z', key: 'info.best_bound' },
      { tex: '\\text{gap}', key: 'info.gap' },
    ],
  },
);

registerMethod<LinearProgram>(
  {
    id: 'gomory_cuts',
    family: 'lp',
    name: 'Gomory fractional cuts',
    params: [
      param.int('max_iter', 50, {
        min: 1,
        max: 1000,
        help: 'Maximum number of cuts.',
        label: 'Cut budget',
      }),
    ],
    needs: ['lp', 'integer'],
    order: 'finite with lexicographic rules (Gomory 1958); slow in practice',
    summary:
      'Read a valid inequality off a fractional row of the optimal tableau, add it, and reoptimize with the dual simplex.',
    references: [
      'Gomory (1958), Bull. AMS 64:275–278',
      'Bertsimas & Tsitsiklis, Introduction to Linear Optimization (1997), §11.1',
      'Wolsey, Integer Programming (1998), §8.6',
    ],
  },
  (p, o) => gomoryCuts(p, o),
  {
    rule: '\\sum_{j \\in N} f(\\bar a_{rj})\\, z_j \\;\\ge\\; f(\\bar b_r),\\qquad f(v) = v - \\lfloor v \\rfloor',
    intuition:
      'A row of the optimal tableau with a fractional right-hand side yields an inequality that every integer point satisfies but the current vertex violates. Adding it slices the vertex off the polygon; the dual simplex reoptimizes, in exact rational arithmetic.',
    order: 'finite (lexicographic rules)',
    pros: ['No search tree', 'Each cut is a valid inequality forever'],
    cons: ['Cuts get shallow: slow in practice', 'Needs integer data'],
    quantities: [
      { tex: '\\#\\text{cuts}', key: 'k' },
      { tex: '\\text{dual pivots}', key: 'info.dual_pivots' },
    ],
  },
);
