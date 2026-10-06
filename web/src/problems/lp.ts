/**
 * Linear and integer programming test problems — TS port of `numopt.problems.lp`
 * (src/numopt/problems/lp.py). Every problem is a `LinearProgram`
 *
 *   min/max cᵀx  s.t.  A_ub x ≤ b_ub,  A_eq x = b_eq,  x ≥ 0  (optionally x integer),
 *
 * with the same ids, data, optima and descriptions as Python (checked against
 * src/generated/problems.json in tests/lp/problems.test.ts). `domain` is the plotting box for
 * 2-variable problems and empty otherwise.
 *
 * TS-only display fields: `latex` (the program in KaTeX, generated from the data so it can never
 * disagree with it), `dim` and `tags`.
 */
import type { LinearProgram } from '../core/types';
import { addProblem } from './registry';

export interface LPProblem extends LinearProgram {
  /** The program typeset in KaTeX (generated from c, A, b, sense and integrality). */
  latex: string;
  /** Number of variables. */
  dim: number;
  tags: string[];
}

type Spec = Omit<
  LinearProgram,
  'aUb' | 'bUb' | 'aEq' | 'bEq' | 'integer' | 'optimum' | 'optimalValue' | 'domain'
> & {
  aUb?: number[][];
  bUb?: number[];
  aEq?: number[][];
  bEq?: number[];
  integer?: boolean[];
  optimum?: number[];
  optimalValue?: number;
  domain?: [[number, number], [number, number]];
  tags?: string[];
};

const MINUS = '-';

/** A coefficient times a variable, for KaTeX (`3x_1`, `-x_2`, `\tfrac12 x_1`). */
function term(a: number, j: number, first: boolean): string {
  const v = `x_{${j + 1}}`;
  const mag = Math.abs(a);
  const coef = mag === 1 ? '' : num(mag);
  // A leading minus is unary: braced, so KaTeX does not space it as a binary operator after the
  // empty atom that `aligned` puts at the start of a cell.
  const sign = a < 0 ? (first ? `{${MINUS}}` : ` ${MINUS} `) : first ? '' : ' + ';
  return `${sign}${coef}${v}`;
}

function num(v: number): string {
  if (Number.isInteger(v)) return String(v);
  const s = String(v);
  return s.length > 6 ? v.toPrecision(4) : s;
}

/** `a₁x₁ + a₂x₂ …` (zero coefficients omitted). */
export function linearTex(a: readonly number[]): string {
  let out = '';
  a.forEach((v, j) => {
    if (v === 0) return;
    out += term(v, j, out === '');
  });
  return out || '0';
}

/** The program in KaTeX: objective, then one aligned row per constraint (≤ for the stored rows). */
export function lpLatex(lp: LinearProgram): string {
  const n = lp.c.length;
  const lines: string[] = [];
  const many = n > 4 || (lp.aUb?.length ?? 0) + (lp.aEq?.length ?? 0) > 4;
  if (many) {
    // Matrix form for the larger programs, one line per block.
    const int = lp.integer.length && lp.integer.every(Boolean);
    const rows = [
      `\\${lp.sense} & \\; \\mathbf{c}^{\\top}\\mathbf{x}`,
      ...(lp.aUb ? [`& \\; A_{\\text{ub}}\\mathbf{x} \\le \\mathbf{b}_{\\text{ub}}`] : []),
      ...(lp.aEq ? [`& \\; A_{\\text{eq}}\\mathbf{x} = \\mathbf{b}_{\\text{eq}}`] : []),
      `& \\; ${int ? `\\mathbf{x} \\in \\mathbb{Z}^{${n}}_{\\ge 0}` : `\\mathbf{x} \\in \\mathbb{R}^{${n}}_{\\ge 0}`}`,
    ];
    rows[1] = rows[1].replace('& \\;', '\\text{s.t.} & \\;');
    return `\\begin{aligned} ${rows.join(' \\\\ ')} \\end{aligned}`;
  }
  lines.push(`\\${lp.sense} & \\; ${linearTex(lp.c)}`);
  (lp.aUb ?? []).forEach((row, i) => {
    const b = (lp.bUb as number[])[i];
    lines.push(`& \\; ${linearTex(row)} \\le ${b < 0 ? MINUS + num(-b) : num(b)}`);
  });
  (lp.aEq ?? []).forEach((row, i) => {
    const b = (lp.bEq as number[])[i];
    lines.push(`& \\; ${linearTex(row)} = ${num(b)}`);
  });
  const int = lp.integer.length > 0 && lp.integer.every(Boolean);
  lines.push(`& \\; \\mathbf{x} ${int ? '\\in \\mathbb{Z}^{' + n + '}_{\\ge 0}' : '\\ge 0'}`);
  // The first constraint row carries "s.t.".
  if (lines.length > 1) lines[1] = lines[1].replace('& \\;', '\\text{s.t.} & \\;');
  return `\\begin{aligned} ${lines.join(' \\\\ ')} \\end{aligned}`;
}

function lp(s: Spec): LPProblem {
  const base: LinearProgram = {
    id: s.id,
    name: s.name,
    c: s.c,
    aUb: s.aUb ?? null,
    bUb: s.bUb ?? null,
    aEq: s.aEq ?? null,
    bEq: s.bEq ?? null,
    sense: s.sense,
    integer: s.integer ?? [],
    optimum: s.optimum ?? null,
    optimalValue: s.optimalValue ?? null,
    description: s.description,
    domain: s.domain ?? [],
  };
  return addProblem('lp', {
    ...base,
    latex: lpLatex(base),
    dim: s.c.length,
    tags: s.tags ?? [],
  } satisfies LPProblem);
}

export const wyndor = lp({
  id: 'wyndor',
  name: 'Wyndor Glass Co.',
  c: [3.0, 5.0],
  aUb: [
    [1.0, 0.0],
    [0.0, 2.0],
    [3.0, 2.0],
  ],
  bUb: [4.0, 12.0, 18.0],
  sense: 'max',
  optimum: [2.0, 6.0],
  optimalValue: 36.0,
  description:
    'Hillier & Lieberman, Introduction to Operations Research, §3.1: ' +
    'max 3x₁ + 5x₂ s.t. x₁ ≤ 4, 2x₂ ≤ 12, 3x₁ + 2x₂ ≤ 18. The origin is feasible.',
  domain: [
    [0.0, 7.0],
    [0.0, 10.0],
  ],
  tags: ['2-D', 'textbook'],
});

export const diet2d = lp({
  id: 'diet_2d',
  name: 'Two-food diet',
  c: [2.0, 3.0],
  aUb: [
    [-1.0, -1.0],
    [-1.0, -3.0],
    [-2.0, -1.0],
  ],
  bUb: [-4.0, -6.0, -5.0],
  sense: 'min',
  optimum: [3.0, 1.0],
  optimalValue: 9.0,
  description:
    'min 2x₁ + 3x₂ s.t. x₁ + x₂ ≥ 4, x₁ + 3x₂ ≥ 6, 2x₁ + x₂ ≥ 5 (written as ≤ rows ' +
    'with negative right-hand sides). The origin is infeasible, so the primal simplex ' +
    'needs phase 1 or big-M; the slack basis is dual feasible (c ≥ 0), so the dual ' +
    'simplex starts directly.',
  domain: [
    [0.0, 7.0],
    [0.0, 6.0],
  ],
  tags: ['2-D', 'phase 1'],
});

export const degenerate2d = lp({
  id: 'degenerate_2d',
  name: 'Degenerate vertex',
  c: [2.0, 1.0],
  aUb: [
    [1.0, 1.0],
    [1.0, -1.0],
    [1.0, 0.0],
  ],
  bUb: [4.0, 2.0, 2.0],
  sense: 'max',
  optimum: [2.0, 2.0],
  optimalValue: 6.0,
  description:
    'max 2x₁ + x₂ s.t. x₁ + x₂ ≤ 4, x₁ − x₂ ≤ 2, x₁ ≤ 2. Three constraints meet at ' +
    '(2, 0), so the ratio test ties there; the simplex then makes a degenerate pivot ' +
    '(step length 0, basis changes, vertex does not) before it moves to (2, 2). ' +
    'Cycling needs m ≥ 2 rows and n − m ≥ 3 nonbasic columns in standard form ' +
    '(Marshall & Suurballe 1969); with 2 variables n − m = 2, so a 2-variable LP ' +
    'cannot cycle. See beale_cycling for a problem that does.',
  domain: [
    [0.0, 5.0],
    [0.0, 5.0],
  ],
  tags: ['2-D', 'degenerate'],
});

export const bealeCycling = lp({
  id: 'beale_cycling',
  name: "Beale's cycling example",
  c: [10.0, -57.0, -9.0, -24.0],
  aUb: [
    [0.5, -5.5, -2.5, 9.0],
    [0.5, -1.5, -0.5, 1.0],
    [1.0, 0.0, 0.0, 0.0],
  ],
  bUb: [0.0, 0.0, 1.0],
  sense: 'max',
  optimum: [1.0, 0.0, 1.0, 0.0],
  optimalValue: 1.0,
  description:
    'Chvátal, Linear Programming (1983), p. 31 (after Beale 1955). With the ' +
    'largest-coefficient (Dantzig) entering rule and smallest-subscript tie-breaking ' +
    "in the ratio test the simplex method cycles through 6 degenerate bases; Bland's " +
    'rule terminates.',
  tags: ['4-D', 'cycling'],
});

export const unbounded2d = lp({
  id: 'unbounded_2d',
  name: 'Unbounded LP',
  c: [1.0, 1.0],
  aUb: [
    [-1.0, 1.0],
    [1.0, -2.0],
  ],
  bUb: [1.0, 2.0],
  sense: 'max',
  description:
    'Unbounded: max x₁ + x₂ s.t. −x₁ + x₂ ≤ 1, x₁ − 2x₂ ≤ 2. The feasible region ' +
    'contains the ray (4, 2) + t·(1, 1), t ≥ 0, along which the objective grows ' +
    'without bound.',
  domain: [
    [0.0, 8.0],
    [0.0, 8.0],
  ],
  tags: ['2-D', 'unbounded'],
});

export const infeasible2d = lp({
  id: 'infeasible_2d',
  name: 'Infeasible LP',
  c: [3.0, 2.0],
  aUb: [
    [1.0, 1.0],
    [-1.0, -1.0],
  ],
  bUb: [2.0, -4.0],
  sense: 'max',
  description:
    'Infeasible: x₁ + x₂ ≤ 2 and x₁ + x₂ ≥ 4 cannot both hold. Phase 1 ends with a ' +
    'positive sum of artificial variables (minimum 2).',
  domain: [
    [0.0, 5.0],
    [0.0, 5.0],
  ],
  tags: ['2-D', 'infeasible'],
});

export const kleeMinty3 = lp({
  id: 'klee_minty_3',
  name: 'Klee–Minty cube (n = 3)',
  c: [100.0, 10.0, 1.0],
  aUb: [
    [1.0, 0.0, 0.0],
    [20.0, 1.0, 0.0],
    [200.0, 20.0, 1.0],
  ],
  bUb: [1.0, 100.0, 10000.0],
  sense: 'max',
  optimum: [0.0, 0.0, 10000.0],
  optimalValue: 10000.0,
  description:
    "Klee & Minty (1972) in Chvátal's form (Linear Programming, ch. 4): " +
    'maximize $\\sum_{j=1}^{n} 10^{n-j} x_j$ subject to ' +
    '$2\\sum_{j=1}^{i-1} 10^{i-j} x_j + x_i \\le 100^{i-1}$ for $i = 1, \\dots, n$ and ' +
    '$\\mathbf{x} \\ge 0$. The Dantzig rule visits all $2^n = 8$ vertices (7 pivots) before ' +
    'it reaches the optimum.',
  tags: ['3-D', 'worst case'],
});

export const transportSmall = lp({
  id: 'transport_small',
  name: 'Small transportation problem',
  // Variables x_ij (supplier i → customer j), ordered x11, x12, x13, x21, x22, x23.
  c: [8.0, 6.0, 10.0, 9.0, 12.0, 13.0],
  aUb: [
    [1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1],
  ],
  bUb: [25.0, 30.0],
  aEq: [
    [1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1],
  ],
  bEq: [10.0, 25.0, 15.0],
  sense: 'min',
  optimum: [0.0, 25.0, 0.0, 10.0, 0.0, 15.0],
  optimalValue: 435.0,
  description:
    'Two suppliers (capacities 25, 30) and three customers (demands 10, 25, 15, met ' +
    'exactly) with unit costs [[8, 6, 10], [9, 12, 13]]. Six variables x₁₁…x₂₃, ' +
    'two ≤ rows and three equality rows.',
  tags: ['6-D', 'equality rows'],
});

export const ilpKnapsackLike2d = lp({
  id: 'ilp_knapsack_like_2d',
  name: 'Two-variable integer program',
  c: [8.0, 5.0],
  aUb: [
    [1.0, 1.0],
    [9.0, 5.0],
  ],
  bUb: [6.0, 45.0],
  sense: 'max',
  integer: [true, true],
  optimum: [5.0, 0.0],
  optimalValue: 40.0,
  description:
    'Winston, Operations Research (4th ed.), §9.3: max 8x₁ + 5x₂ s.t. x₁ + x₂ ≤ 6, ' +
    '9x₁ + 5x₂ ≤ 45, x integer. The LP relaxation optimum is (3.75, 2.25) with value ' +
    '41.25; the integer optimum is (5, 0) with value 40.',
  domain: [
    [0.0, 6.5],
    [0.0, 6.5],
  ],
  tags: ['2-D', 'integer'],
});

export const ilp3var = lp({
  id: 'ilp_3var',
  name: 'Three-variable integer program',
  c: [4.0, 3.0, 3.0],
  aUb: [
    [4.0, 2.0, 1.0],
    [3.0, 4.0, 2.0],
    [2.0, 1.0, 3.0],
  ],
  bUb: [10.0, 14.0, 7.0],
  sense: 'max',
  integer: [true, true, true],
  optimum: [1.0, 2.0, 1.0],
  optimalValue: 13.0,
  description:
    'max 4x₁ + 3x₂ + 3x₃ s.t. 4x₁ + 2x₂ + x₃ ≤ 10, 3x₁ + 4x₂ + 2x₃ ≤ 14, ' +
    '2x₁ + x₂ + 3x₃ ≤ 7, x integer. LP relaxation: (1.2, 2.2, 0.8), value 13.8; ' +
    'integer optimum (1, 2, 1), value 13.',
  tags: ['3-D', 'integer'],
});
