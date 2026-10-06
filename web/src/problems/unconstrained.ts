/**
 * Unconstrained test problems — TS port of `numopt.problems.unconstrained`
 * (src/numopt/problems/unconstrained.py): smooth f: ℝⁿ → ℝ with exact gradient and Hessian.
 *
 * Most problems are 2-D (the labs draw their contours); `rosenbrock_nd` (n = 10) and
 * `quadratic_nd` (n = 20) serve the parity tests and the n-D readouts. Labs that plot contours
 * must use `listUnconstrained2D()` (or filter `dim === 2`), because Python registers the n-D
 * problems under the same kind.
 *
 * Every formula keeps the Python operation order (including NumPy's pairwise summation in
 * `np.sum`), so values agree with Python to the last bit or within a few ulps (libm).
 *
 * `extra` mirrors `Problem.extra`: `minima_f`, `n_global`, `f_min`, and for the quadratics `A`, `c`
 * (f = ½(x − c)ᵀA(x − c)); `quadratic_nd` also has `eigenvalues` and `seed`.
 */
import { Rng } from '../core/rng';
import type { Matrix, Problem, Problem2D, Vector } from '../core/types';
import { addProblem, listProblems } from './registry';

export interface UnconstrainedExtra {
  minima_f: number[];
  n_global: number;
  f_min: number;
  A?: Matrix;
  c?: Vector;
  eigenvalues?: number[];
  seed?: number;
  [key: string]: unknown;
}

/** A smooth n-D minimization problem with exact derivatives (any dimension). */
export interface UnconstrainedProblem extends Problem<Vector> {
  f: (x: Vector) => number;
  grad: (x: Vector) => Vector;
  hess: (x: Vector) => Matrix;
  /** `[[lo_1, hi_1], …, [lo_n, hi_n]]` — the plotting box. */
  domain: [number, number][];
  x0: Vector;
  /** Known local minimizers, global ones first. */
  minima: Vector[];
  bracket: null;
  roots: Vector[];
  constraints: [];
  exact: null;
  description: string;
  tags: string[];
  extra: UnconstrainedExtra;
}

const PI = Math.PI;
const TWO_PI = 2.0 * Math.PI;
const E = Math.E;

/**
 * `np.sum` of a contiguous float64 array: NumPy's pairwise summation (8 accumulators for
 * 8 ≤ n ≤ 128, a plain loop from −0.0 below 8 terms, recursive halves above 128).
 */
export function npSum(a: readonly number[], start = 0, n = a.length): number {
  if (n < 8) {
    let res = -0.0;
    for (let i = 0; i < n; i++) res += a[start + i];
    return res;
  }
  if (n <= 128) {
    const r = [0, 1, 2, 3, 4, 5, 6, 7].map((j) => a[start + j]);
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] += a[start + i + j];
    let res = r[0] + r[1] + (r[2] + r[3]) + (r[4] + r[5] + (r[6] + r[7]));
    for (; i < n; i++) res += a[start + i];
    return res;
  }
  let n2 = Math.trunc(n / 2);
  n2 -= n2 % 8;
  return npSum(a, start, n2) + npSum(a, start + n2, n - n2);
}

function dot(a: readonly number[], b: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function diag(d: readonly number[]): Matrix {
  return d.map((v, i) => d.map((_, j) => (i === j ? v : 0)));
}

interface Spec {
  id: string;
  name: string;
  latex: string;
  f: (x: Vector) => number;
  grad: (x: Vector) => Vector;
  hess: (x: Vector) => Matrix;
  domain: [number, number][];
  x0: Vector;
  minima: Vector[];
  minimaF: number[];
  nGlobal: number;
  description: string;
  tags: string[];
  extra?: Record<string, unknown>;
}

/** Assemble and register a problem; `minimaF[0]` is the global minimum value. */
function problem(s: Spec): UnconstrainedProblem {
  if (s.minima.length !== s.minimaF.length || !(1 <= s.nGlobal && s.nGlobal <= s.minima.length))
    throw new Error(`${s.id}: inconsistent minima metadata`);
  return addProblem('unconstrained', {
    id: s.id,
    name: s.name,
    latex: s.latex,
    dim: s.x0.length,
    domain: s.domain,
    f: s.f,
    grad: s.grad,
    hess: s.hess,
    x0: s.x0,
    bracket: null,
    minima: s.minima,
    roots: [],
    constraints: [],
    exact: null,
    description: s.description,
    tags: s.tags,
    extra: {
      minima_f: s.minimaF,
      n_global: s.nGlobal,
      f_min: s.minimaF[0],
      ...(s.extra ?? {}),
    },
  } satisfies UnconstrainedProblem);
}

const box = (lo: number, hi: number, n: number): [number, number][] =>
  Array.from({ length: n }, () => [lo, hi] as [number, number]);

// ---------------------------------------------------------------------------------------
// Quadratics: f(x) = ½ (x − c)ᵀ A (x − c), ∇f = A(x − c), ∇²f = A
// ---------------------------------------------------------------------------------------

/**
 * dᵀAd summed like `np.einsum("i...,ij,j...->...", d, A, d)` (NumPy 2.x): the terms
 * (d_i A_ij) d_j are added row by row and the row sums added for n = 2, and added in one
 * sequence over (i, j) for every other n (measured bit-for-bit against NumPy).
 */
export function quadForm(A: Matrix, d: readonly number[]): number {
  const n = d.length;
  let s = 0;
  for (let i = 0; i < n; i++) {
    if (n === 2) {
      let row = 0;
      for (let j = 0; j < n; j++) row += d[i] * A[i][j] * d[j];
      s += row;
    } else {
      for (let j = 0; j < n; j++) s += d[i] * A[i][j] * d[j];
    }
  }
  return s;
}

function quadratic(A: Matrix, c: Vector) {
  const Ac = A.map((r) => r.slice());
  const cc = c.slice();
  const d = (x: Vector) => x.map((v, i) => v - cc[i]);
  return {
    f: (x: Vector) => 0.5 * quadForm(Ac, d(x)),
    grad: (x: Vector) => {
      const dx = d(x);
      return Ac.map((row) => dot(row, dx));
    },
    hess: () => Ac.map((r) => r.slice()),
  };
}

export const quadraticBowl = (() => {
  // Eigenvalues 2 and 4 (cond 2), eigenvectors (1, 1)/√2 and (1, −1)/√2.
  const A = [
    [3.0, 1.0],
    [1.0, 3.0],
  ];
  const c = [1.0, -0.5];
  return problem({
    id: 'quadratic_bowl',
    name: 'Quadratic bowl',
    latex: 'f(x,y) = \\tfrac12\\left(3(x-1)^2 + 2(x-1)(y+\\tfrac12) + 3(y+\\tfrac12)^2\\right)',
    ...quadratic(A, c),
    domain: box(-3.0, 3.0, 2),
    x0: [-2.0, 2.0],
    minima: [[1.0, -0.5]],
    minimaF: [0.0],
    nGlobal: 1,
    description:
      'A well-conditioned convex quadratic (Hessian eigenvalues 2 and 4, ' +
      'condition number 2). Every descent method should converge quickly.',
    tags: ['quadratic', 'convex', 'well-conditioned', '2d'],
    extra: { A, c },
  });
})();

export const quadraticIll = (() => {
  // A = Q diag(1, 50) Qᵀ with Q = [[0.8, −0.6], [0.6, 0.8]] (a rotation by atan(3/4)), then
  // symmetrized; cond(A) = 50. The literals are the float64 values NumPy computes for this
  // product (tests/problems_unconstrained.test.ts recomputes them).
  const A = [
    [18.64, -23.520000000000003],
    [-23.520000000000003, 32.36],
  ];
  const c = [0.0, 0.0];
  return problem({
    id: 'quadratic_ill',
    name: 'Ill-conditioned quadratic',
    latex:
      'f(x) = \\tfrac12 x^\\top Q\\,\\mathrm{diag}(1, 50)\\,Q^\\top x,\\ ' +
      'Q = \\begin{pmatrix}0.8 & -0.6\\\\ 0.6 & 0.8\\end{pmatrix}',
    ...quadratic(A, c),
    domain: box(-3.0, 3.0, 2),
    x0: [-2.0, 2.0],
    minima: [[0.0, 0.0]],
    minimaF: [0.0],
    nGlobal: 1,
    description:
      'A rotated convex quadratic with condition number 50: long, narrow, ' +
      'tilted elliptic contours. Steepest descent zig-zags; Newton converges in one step.',
    tags: ['quadratic', 'convex', 'ill-conditioned', '2d'],
    extra: { A, c },
  });
})();

/**
 * The `quadratic_nd` construction, replayed exactly from Python (draws from `Rng(seed)` in order):
 *   1. v_i = rng.normal() for i = 0..n−1      → Householder reflector H = I − 2 v vᵀ / vᵀv
 *   2. c_i = rng.uniform(−2, 2) for i = 0..n−1 → the minimizer
 *   λ_i = 10^(2i/(n − 1)) and A = H diag(λ) H, symmetrized.
 */
export function buildQuadraticNd(n = 20, seed = 20): { A: Matrix; c: Vector; lam: Vector } {
  const rng = new Rng(seed);
  const v = Array.from({ length: n }, () => rng.normal());
  const c = Array.from({ length: n }, () => rng.uniform(-2.0, 2.0));
  const lam = Array.from({ length: n }, (_, i) => 10.0 ** ((2.0 * i) / (n - 1)));
  const s = 2.0 / dot(v, v);
  const H = v.map((vi, i) => v.map((vj, j) => (i === j ? 1 : 0) - s * (vi * vj)));
  // (H diag(λ))_ik = H_ik λ_k; A = (H diag(λ)) H
  const B = H.map((row) => row.map((h, k) => h * lam[k]));
  const A0 = B.map((row) => H[0].map((_, j) => row.reduce((acc, b, k) => acc + b * H[k][j], 0)));
  const A = A0.map((row, i) => row.map((a, j) => 0.5 * (a + A0[j][i])));
  return { A, c, lam };
}

export const quadraticNd = (() => {
  const n = 20;
  const seed = 20;
  const { A, c, lam } = buildQuadraticNd(n, seed);
  return problem({
    id: 'quadratic_nd',
    name: 'Random SPD quadratic (n = 20)',
    latex: 'f(x) = \\tfrac12 (x - c)^\\top A (x - c),\\ A \\succ 0,\\ \\kappa(A) = 100',
    ...quadratic(A, c),
    domain: box(-3.0, 3.0, n),
    x0: new Array<number>(n).fill(0.0),
    minima: [c.slice()],
    minimaF: [0.0],
    nGlobal: 1,
    description:
      'A 20-dimensional SPD quadratic A = H diag(λ) H, with H a Householder ' +
      'reflector and λ geometric from 1 to 100 (seeded with numopt.core.rng.Rng(20)).',
    tags: ['quadratic', 'convex', 'n-d'],
    extra: { A, c, eigenvalues: lam, seed },
  });
})();

// ---------------------------------------------------------------------------------------
// Classic 2-D test functions
// ---------------------------------------------------------------------------------------

/** Chained Rosenbrock Σ 100(x_{i+1} − x_i²)² + (1 − x_i)² (the scipy.optimize.rosen form). */
export function rosenF(x: Vector): number {
  const terms = new Array<number>(x.length - 1);
  for (let i = 0; i < x.length - 1; i++) {
    const d = x[i + 1] - x[i] ** 2;
    terms[i] = 100.0 * d ** 2 + (1.0 - x[i]) ** 2;
  }
  return npSum(terms);
}

export function rosenGrad(x: Vector): Vector {
  const n = x.length;
  const g = new Array<number>(n).fill(0);
  for (let i = 0; i < n - 1; i++) {
    const d = x[i + 1] - x[i] ** 2;
    g[i] = -400.0 * x[i] * d - 2.0 * (1.0 - x[i]);
  }
  for (let i = 1; i < n; i++) g[i] += 200.0 * (x[i] - x[i - 1] ** 2);
  return g;
}

export function rosenHess(x: Vector): Matrix {
  const n = x.length;
  const dg = new Array<number>(n).fill(0);
  for (let i = 0; i < n - 1; i++) dg[i] = 1200.0 * x[i] ** 2 - 400.0 * x[i + 1] + 2.0;
  for (let i = 1; i < n; i++) dg[i] += 200.0;
  const H = diag(dg);
  for (let i = 0; i < n - 1; i++) {
    const off = -400.0 * x[i];
    H[i][i + 1] = off;
    H[i + 1][i] = off;
  }
  return H;
}

export const rosenbrock = problem({
  id: 'rosenbrock',
  name: 'Rosenbrock',
  latex: 'f(x,y) = (1-x)^2 + 100\\,(y-x^2)^2',
  f: rosenF,
  grad: rosenGrad,
  hess: rosenHess,
  domain: [
    [-2.0, 2.0],
    [-1.0, 3.0],
  ],
  x0: [-1.2, 1.0],
  minima: [[1.0, 1.0]],
  minimaF: [0.0],
  nGlobal: 1,
  description:
    "Rosenbrock's banana valley (MGH problem 1, standard start (−1.2, 1)). " +
    'Finding the valley is easy; following its curved floor to (1, 1) is hard.',
  tags: ['nonconvex', 'valley', 'classic', '2d'],
});

export const rosenbrockNd = (() => {
  const n = 10;
  const x0 = Array.from({ length: n }, (_, i) => (i % 2 === 0 ? -1.2 : 1.0));
  const local = [
    -0.993263372856369, 0.9966060394434352, 0.9982406113911125, 0.9989884337678007,
    0.999226153439665, 0.9990736481633443, 0.9984541774029088, 0.9970562516630377,
    0.9941793752280661, 0.9883926301288679,
  ];
  return problem({
    id: 'rosenbrock_nd',
    name: 'Rosenbrock (n = 10)',
    latex: 'f(x) = \\sum_{i=1}^{n-1} \\left[100\\,(x_{i+1}-x_i^2)^2 + (1-x_i)^2\\right],\\ n = 10',
    f: rosenF,
    grad: rosenGrad,
    hess: rosenHess,
    domain: box(-2.0, 2.0, n),
    x0,
    minima: [new Array<number>(n).fill(1.0), local],
    minimaF: [0.0, rosenF(local)],
    nGlobal: 1,
    description:
      'The extended Rosenbrock function in 10 dimensions (the chained form used ' +
      'by scipy.optimize.rosen). Besides the global minimizer (1, …, 1) it has a local ' +
      'minimizer near (−1, 1, …, 1) (Shang & Qiu 2006 show this for 4 ≤ n ≤ 30).',
    tags: ['nonconvex', 'valley', 'n-d'],
  });
})();

export const himmelblau = problem({
  id: 'himmelblau',
  name: 'Himmelblau',
  latex: 'f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2',
  f: ([x, y]) => (x ** 2 + y - 11.0) ** 2 + (x + y ** 2 - 7.0) ** 2,
  grad: ([x, y]) => {
    const a = x ** 2 + y - 11.0;
    const b = x + y ** 2 - 7.0;
    return [4.0 * x * a + 2.0 * b, 2.0 * a + 4.0 * y * b];
  },
  hess: ([x, y]) => {
    const h11 = 12.0 * x ** 2 + 4.0 * y - 42.0;
    const h12 = 4.0 * (x + y);
    const h22 = 4.0 * x + 12.0 * y ** 2 - 26.0;
    return [
      [h11, h12],
      [h12, h22],
    ];
  },
  domain: box(-5.0, 5.0, 2),
  x0: [0.0, 0.0],
  minima: [
    [3.0, 2.0],
    [-2.805118086952745, 3.131312518250573],
    [-3.779310253377747, -3.283185991286169],
    [3.5844283403304917, -1.8481265269644036],
  ],
  minimaF: [0.0, 0.0, 0.0, 0.0],
  nGlobal: 4,
  description:
    "Himmelblau's function has four global minimizers, all with f = 0, and a " +
    'local maximum near (−0.2708, −0.9230). Which minimizer a method finds depends on ' +
    'the start point.',
  tags: ['nonconvex', 'multiple-minima', 'classic', '2d'],
});

const BEALE_C = [1.5, 2.25, 2.625];

export const beale = problem({
  id: 'beale',
  name: 'Beale',
  latex: 'f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2',
  // f = Σ_{i=1}^{3} t_i², t_i = c_i − x + x yⁱ (MGH problem 5); Python's builtin sum starts at 0.
  f: ([x, y]) => {
    let s = 0;
    BEALE_C.forEach((c, k) => {
      const i = k + 1;
      s += (c - x + x * y ** i) ** 2;
    });
    return s;
  },
  grad: ([x0, y]) => {
    const g = [0.0, 0.0];
    BEALE_C.forEach((c, k) => {
      const i = k + 1;
      const t = c - x0 + x0 * y ** i;
      const a = [y ** i - 1.0, i * x0 * y ** (i - 1)];
      g[0] += 2.0 * t * a[0];
      g[1] += 2.0 * t * a[1];
    });
    return g;
  },
  hess: ([x0, y]) => {
    const H = [
      [0.0, 0.0],
      [0.0, 0.0],
    ];
    BEALE_C.forEach((c, k) => {
      const i = k + 1;
      const t = c - x0 + x0 * y ** i;
      const dt = [y ** i - 1.0, i * x0 * y ** (i - 1)];
      const d2xy = i * y ** (i - 1);
      const d2yy = i >= 2 ? i * (i - 1) * x0 * y ** (i - 2) : 0.0;
      const S = [
        [0.0, d2xy],
        [d2xy, d2yy],
      ];
      for (let r = 0; r < 2; r++)
        for (let q = 0; q < 2; q++) H[r][q] += 2.0 * (dt[r] * dt[q] + t * S[r][q]);
    });
    return H;
  },
  domain: box(-4.5, 4.5, 2),
  x0: [1.0, 1.0],
  minima: [[3.0, 0.5]],
  minimaF: [0.0],
  nGlobal: 1,
  description:
    "Beale's function (MGH problem 5, standard start (1, 1)): flat plateaus " +
    'and steep ridges near the corners of the box; global minimum f(3, 0.5) = 0.',
  tags: ['nonconvex', 'classic', '2d'],
});

export const booth = problem({
  id: 'booth',
  name: 'Booth',
  latex: 'f(x,y) = (x + 2y - 7)^2 + (2x + y - 5)^2',
  f: ([x, y]) => (x + 2.0 * y - 7.0) ** 2 + (2.0 * x + y - 5.0) ** 2,
  grad: ([x, y]) => {
    const a = x + 2.0 * y - 7.0;
    const b = 2.0 * x + y - 5.0;
    return [2.0 * a + 4.0 * b, 4.0 * a + 2.0 * b];
  },
  hess: () => [
    [10.0, 8.0],
    [8.0, 10.0],
  ],
  domain: box(-10.0, 10.0, 2),
  x0: [-5.0, -5.0],
  minima: [[1.0, 3.0]],
  minimaF: [0.0],
  nGlobal: 1,
  description:
    "Booth's function (J&Y 22): a convex quadratic with Hessian eigenvalues " +
    '2 and 18 (condition number 9).',
  tags: ['quadratic', 'convex', '2d'],
});

export const matyas = problem({
  id: 'matyas',
  name: 'Matyas',
  latex: 'f(x,y) = 0.26\\,(x^2 + y^2) - 0.48\\,xy',
  f: ([x, y]) => 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y,
  grad: ([x, y]) => [0.52 * x - 0.48 * y, 0.52 * y - 0.48 * x],
  hess: () => [
    [0.52, -0.48],
    [-0.48, 0.52],
  ],
  domain: box(-10.0, 10.0, 2),
  x0: [8.0, 2.0],
  minima: [[0.0, 0.0]],
  minimaF: [0.0],
  nGlobal: 1,
  description:
    "Matyas' function (J&Y 71): a convex quadratic whose Hessian eigenvalues " +
    'are 0.04 and 1 (condition number 25), so the valley along y = x is very flat.',
  tags: ['quadratic', 'convex', 'ill-conditioned', '2d'],
});

export const threeHumpCamel = (() => {
  const f = ([u, v]: Vector) => 2.0 * u ** 2 - 1.05 * u ** 4 + u ** 6 / 6.0 + u * v + v ** 2;
  const a = 1.747552345830289,
    b = -0.8737761729151445;
  // f at the two local minima, as Python computes it (pinned: libm pow may differ by an ulp).
  const fLoc = 0.29863844223686054;
  return problem({
    id: 'three_hump_camel',
    name: 'Three-hump camel',
    latex: 'f(x,y) = 2x^2 - 1.05x^4 + \\tfrac{x^6}{6} + xy + y^2',
    f,
    grad: ([u, v]) => [4.0 * u - 4.2 * u ** 3 + u ** 5 + v, u + 2.0 * v],
    hess: ([u]) => [
      [4.0 - 12.6 * u ** 2 + 5.0 * u ** 4, 1.0],
      [1.0, 2.0],
    ],
    domain: box(-3.0, 3.0, 2),
    x0: [-2.0, 1.5],
    minima: [
      [0.0, 0.0],
      [a, b],
      [-a, -b],
    ],
    minimaF: [0.0, fLoc, fLoc],
    nGlobal: 1,
    description:
      'Three-hump camel (J&Y 29): a global minimum at the origin and two ' +
      'symmetric local minima (f ≈ 0.2986) separated by saddle points.',
    tags: ['nonconvex', 'multiple-minima', '2d'],
  });
})();

export const sixHumpCamel = (() => {
  const g1 = [0.08984201310031807, -0.7126564030207396];
  const l1 = [-1.7036067149699814, 0.7960835686726251];
  const l2 = [1.6071047529201974, 0.5686514548841313];
  const minima = [g1, [-g1[0], -g1[1]], l1, [-l1[0], -l1[1]], l2, [-l2[0], -l2[1]]];
  return problem({
    id: 'six_hump_camel',
    name: 'Six-hump camel',
    latex: 'f(x,y) = \\left(4 - 2.1x^2 + \\tfrac{x^4}{3}\\right)x^2 + xy + (-4 + 4y^2)\\,y^2',
    f: ([u, v]) =>
      (4.0 - 2.1 * u ** 2 + u ** 4 / 3.0) * u ** 2 + u * v + (-4.0 + 4.0 * v ** 2) * v ** 2,
    grad: ([u, v]) => [8.0 * u - 8.4 * u ** 3 + 2.0 * u ** 5 + v, u - 8.0 * v + 16.0 * v ** 3],
    hess: ([u, v]) => [
      [8.0 - 25.2 * u ** 2 + 10.0 * u ** 4, 1.0],
      [1.0, -8.0 + 48.0 * v ** 2],
    ],
    domain: [
      [-2.0, 2.0],
      [-1.2, 1.2],
    ],
    x0: [-1.5, -0.5],
    minima,
    // f at each minimum, as Python computes it (pinned: libm pow may differ by an ulp).
    minimaF: [
      -1.0316284534898774, -1.0316284534898774, -0.21546382438372036, -0.21546382438372036,
      2.1042503103112598, 2.1042503103112598,
    ],
    nGlobal: 2,
    description:
      'Six-hump camel (J&Y 30): two global minima f ≈ −1.0316 at ' +
      '±(0.0898, −0.7127) and four local minima, in point-symmetric pairs.',
    tags: ['nonconvex', 'multiple-minima', 'classic', '2d'],
  });
})();

export const goldsteinPrice = (() => {
  // f = A·B with A = 1 + u² P, B = 30 + v² Q (J&Y 58), where
  //   u = x + y + 1,  P = 19 − 14x + 3x² − 14y + 6xy + 3y²,
  //   v = 2x − 3y,    Q = 18 − 32x + 12x² + 48y − 36xy + 27y².
  // ∇f = B∇A + A∇B, ∇²f = B∇²A + A∇²B + ∇A∇Bᵀ + ∇B∇Aᵀ.
  const du = [1.0, 1.0];
  const dv = [2.0, -3.0];
  const d2P = [
    [6.0, 6.0],
    [6.0, 6.0],
  ];
  const d2Q = [
    [24.0, -36.0],
    [-36.0, 54.0],
  ];
  const parts = ([p, q]: Vector) => {
    const u = p + q + 1.0;
    const P = 19.0 - 14.0 * p + 3.0 * p ** 2 - 14.0 * q + 6.0 * p * q + 3.0 * q ** 2;
    const v = 2.0 * p - 3.0 * q;
    const Q = 18.0 - 32.0 * p + 12.0 * p ** 2 + 48.0 * q - 36.0 * p * q + 27.0 * q ** 2;
    return { p, q, u, P, v, Q };
  };
  const derivs = (x: Vector) => {
    const { p, q, u, P, v, Q } = parts(x);
    const dP = [-14.0 + 6.0 * p + 6.0 * q, -14.0 + 6.0 * p + 6.0 * q];
    const dQ = [-32.0 + 24.0 * p - 36.0 * q, 48.0 - 36.0 * p + 54.0 * q];
    const A = 1.0 + u ** 2 * P;
    const B = 30.0 + v ** 2 * Q;
    // dA = 2.0 * u * P * du + u**2 * dP (NumPy: scalar products first, then elementwise)
    const sA = 2.0 * u * P,
      sB = 2.0 * v * Q;
    const dA = du.map((e, i) => sA * e + u ** 2 * dP[i]);
    const dB = dv.map((e, i) => sB * e + v ** 2 * dQ[i]);
    const d2A: Matrix = [0, 1].map((i) =>
      [0, 1].map(
        (j) =>
          2.0 * P * (du[i] * du[j]) +
          2.0 * u * (du[i] * dP[j] + dP[i] * du[j]) +
          u ** 2 * d2P[i][j],
      ),
    );
    const d2B: Matrix = [0, 1].map((i) =>
      [0, 1].map(
        (j) =>
          2.0 * Q * (dv[i] * dv[j]) +
          2.0 * v * (dv[i] * dQ[j] + dQ[i] * dv[j]) +
          v ** 2 * d2Q[i][j],
      ),
    );
    return { A, B, dA, dB, d2A, d2B };
  };
  return problem({
    id: 'goldstein_price',
    name: 'Goldstein–Price',
    latex:
      'f(x,y) = \\left[1 + (x+y+1)^2(19 - 14x + 3x^2 - 14y + 6xy + 3y^2)\\right]' +
      '\\left[30 + (2x-3y)^2(18 - 32x + 12x^2 + 48y - 36xy + 27y^2)\\right]',
    f: (x) => {
      const { u, P, v, Q } = parts(x);
      return (1.0 + u ** 2 * P) * (30.0 + v ** 2 * Q);
    },
    grad: (x) => {
      const { A, B, dA, dB } = derivs(x);
      return [B * dA[0] + A * dB[0], B * dA[1] + A * dB[1]];
    },
    hess: (x) => {
      const { A, B, dA, dB, d2A, d2B } = derivs(x);
      return [0, 1].map((i) =>
        [0, 1].map((j) => B * d2A[i][j] + A * d2B[i][j] + dA[i] * dB[j] + dB[i] * dA[j]),
      );
    },
    domain: box(-2.0, 2.0, 2),
    x0: [-1.0, 1.0],
    minima: [
      [0.0, -1.0],
      [-0.6, -0.4],
      [1.8, 0.2],
      [1.2, 0.8],
    ],
    minimaF: [3.0, 30.0, 84.0, 840.0],
    nGlobal: 1,
    description:
      'Goldstein–Price (J&Y 58): global minimum f(0, −1) = 3 and local minima ' +
      'f = 30, 84, 840; values span six orders of magnitude over the box.',
    tags: ['nonconvex', 'multiple-minima', 'badly-scaled', '2d'],
  });
})();

export const rastrigin = (() => {
  // f = 10 n + Σ (x_i² − 10 cos 2πx_i)
  const f = (x: Vector) =>
    10.0 * x.length + npSum(x.map((v) => v ** 2 - 10.0 * Math.cos(TWO_PI * v)));
  const a = 0.9949586376523347;
  // f at the four nearest local minima, as Python computes it (pinned: libm cos may differ).
  const fNear = 0.9949590570932898;
  return problem({
    id: 'rastrigin',
    name: 'Rastrigin',
    latex: 'f(x) = 10n + \\sum_{i=1}^{n}\\left(x_i^2 - 10\\cos 2\\pi x_i\\right),\\ n = 2',
    f,
    grad: (x) => x.map((v) => 2.0 * v + 20.0 * PI * Math.sin(TWO_PI * v)),
    hess: (x) => diag(x.map((v) => 2.0 + 40.0 * PI ** 2 * Math.cos(TWO_PI * v))),
    domain: box(-5.12, 5.12, 2),
    x0: [3.3, -2.6],
    minima: [
      [0.0, 0.0],
      [a, 0.0],
      [-a, 0.0],
      [0.0, a],
      [0.0, -a],
    ],
    minimaF: [0.0, fNear, fNear, fNear, fNear],
    nGlobal: 1,
    description:
      "Rastrigin's function: a paraboloid covered by a regular grid of local " +
      'minima near the integer points. Only the four local minima nearest the origin are ' +
      'listed; local methods stop in whichever basin they start.',
    tags: ['nonconvex', 'multimodal', '2d'],
  });
})();

export const ackley = (() => {
  // f = −20 exp(−0.2 r) − exp(s) + e + 20, r = √(Σx_i²/n), s = (1/n) Σ cos 2πx_i (J&Y 1),
  // written with expm1 so that f(0) = 0 exactly.
  const n = 2;
  const f = (x: Vector) => {
    const r = Math.sqrt(npSum(x.map((v) => v ** 2)) / n);
    const s = npSum(x.map((v) => Math.cos(TWO_PI * v))) / n;
    return -20.0 * Math.expm1(-0.2 * r) - E * Math.expm1(s - 1.0);
  };
  // Gradient and Hessian of −exp(s): ∂/∂x_i = (2π/n) eˢ sin 2πx_i.
  const smoothPart = (x: Vector) => {
    const es = Math.exp(npSum(x.map((v) => Math.cos(TWO_PI * v))) / n);
    const sn = x.map((v) => Math.sin(TWO_PI * v));
    const cs = x.map((v) => Math.cos(TWO_PI * v));
    const k = (TWO_PI / n) * es;
    const g = sn.map((v) => k * v);
    const H = sn.map((si, i) =>
      sn.map((sj, j) => k * ((i === j ? TWO_PI * cs[i] : 0.0) - (TWO_PI / n) * (si * sj))),
    );
    return { g, H };
  };
  const radius = (x: Vector) => Math.hypot(...x) / Math.sqrt(n);
  return problem({
    id: 'ackley',
    name: 'Ackley',
    latex:
      'f(x) = -20\\,e^{-0.2\\sqrt{\\frac1n\\sum x_i^2}} - e^{\\frac1n\\sum\\cos 2\\pi x_i} + e + 20',
    f,
    grad: (x) => {
      const { g } = smoothPart(x);
      const r = radius(x);
      // The cone term is not differentiable at 0; 0 is in its subdifferential (Python NOTE).
      if (r === 0.0) return g;
      const w = 4.0 * Math.exp(-0.2 * r);
      return g.map((gi, i) => gi + w * (x[i] / (n * r)));
    },
    hess: (x) => {
      const { H } = smoothPart(x);
      const r = radius(x);
      // The cone term has no Hessian at 0: return the smooth part only (Python NOTE).
      if (r === 0.0) return H;
      const dr = x.map((v) => v / (n * r));
      const w = 4.0 * Math.exp(-0.2 * r);
      return H.map((row, i) =>
        row.map((h, j) => {
          const d2r = ((i === j ? 1.0 / n : 0.0) - dr[i] * dr[j]) / r;
          return h + w * (d2r - 0.2 * (dr[i] * dr[j]));
        }),
      );
    },
    domain: box(-5.0, 5.0, 2),
    x0: [2.6, -3.4],
    minima: [[0.0, 0.0]],
    minimaF: [0.0],
    nGlobal: 1,
    description:
      "Ackley's function (J&Y 1): a nearly flat outer region with many local " +
      'minima and a deep, narrow funnel at the origin. f has a cone-shaped kink at the ' +
      'origin; grad(0) returns the zero subgradient.',
    tags: ['nonconvex', 'multimodal', 'nonsmooth-at-minimum', '2d'],
  });
})();

// Stationary points of ½(x⁴ − 16x² + 5x): the roots of 2x³ − 16x + 2.5 = 0.
const ST_GLOBAL = -2.903534027771177;
const ST_LOCAL = 2.746802770990837;

export const styblinskiTang = (() => {
  const a = ST_GLOBAL,
    b = ST_LOCAL;
  return problem({
    id: 'styblinski_tang',
    name: 'Styblinski–Tang',
    latex: 'f(x) = \\tfrac12\\sum_{i=1}^{n}\\left(x_i^4 - 16x_i^2 + 5x_i\\right),\\ n = 2',
    f: (x) => 0.5 * npSum(x.map((v) => v ** 4 - 16.0 * v ** 2 + 5.0 * v)),
    grad: (x) => x.map((v) => 2.0 * v ** 3 - 16.0 * v + 2.5),
    hess: (x) => diag(x.map((v) => 6.0 * v ** 2 - 16.0)),
    domain: box(-5.0, 5.0, 2),
    x0: [0.5, 0.5],
    minima: [
      [a, a],
      [a, b],
      [b, a],
      [b, b],
    ],
    // f at each minimum, as Python computes it (pinned: libm pow may differ by an ulp).
    minimaF: [-78.33233140754282, -64.19561235905536, -64.19561235905536, -50.05889331056788],
    nGlobal: 1,
    description:
      'Styblinski–Tang (J&Y 144): separable; each coordinate has a deep and a ' +
      'shallow well, giving one global minimum (f ≈ −78.332) and three local minima.',
    tags: ['nonconvex', 'multiple-minima', 'separable', '2d'],
  });
})();

export const mccormick = (() => {
  // Minimizers: x + y = σ_k = −2π/3 + 2πk, x = (σ_k + 1)/2, y = (σ_k − 1)/2 (k = 0, 1 in the box).
  const point = (k: number): Vector => {
    const sigma = (-2.0 * PI) / 3.0 + TWO_PI * k;
    return [0.5 * (sigma + 1.0), 0.5 * (sigma - 1.0)];
  };
  return problem({
    id: 'mccormick',
    name: 'McCormick',
    latex: 'f(x,y) = \\sin(x+y) + (x-y)^2 - 1.5x + 2.5y + 1',
    f: ([x, y]) => Math.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1.0,
    grad: ([p, q]) => {
      const c = Math.cos(p + q);
      return [c + 2.0 * (p - q) - 1.5, c - 2.0 * (p - q) + 2.5];
    },
    hess: ([p, q]) => {
      const s = Math.sin(p + q);
      return [
        [2.0 - s, -2.0 - s],
        [-2.0 - s, 2.0 - s],
      ];
    },
    domain: [
      [-1.5, 4.0],
      [-3.0, 4.0],
    ],
    x0: [3.0, -2.0],
    minima: [point(0), point(1)],
    minimaF: [-PI / 3.0 - Math.sqrt(3.0) / 2.0, (2.0 * PI) / 3.0 - Math.sqrt(3.0) / 2.0],
    nGlobal: 1,
    description:
      "McCormick's function (J&Y 80) on its box [−1.5, 4] × [−3, 4]: global " +
      'minimum f(−0.5472, −1.5472) = −π/3 − √3/2 ≈ −1.9132 over the box, and a local ' +
      'minimum at (2.5944, 1.5944). Unconstrained, f is unbounded below along x − y = 1 ' +
      '(f = σ/2 + sin σ with σ = x + y → −∞), so the global label holds on the box only.',
    tags: ['nonconvex', 'bounded-domain', '2d'],
  });
})();

export const bohachevsky = problem({
  id: 'bohachevsky',
  name: 'Bohachevsky',
  latex: 'f(x,y) = x^2 + 2y^2 - 0.3\\cos 3\\pi x - 0.4\\cos 4\\pi y + 0.7',
  f: ([x, y]) =>
    x ** 2 + 2.0 * y ** 2 - 0.3 * Math.cos(3.0 * PI * x) - 0.4 * Math.cos(4.0 * PI * y) + 0.7,
  grad: ([p, q]) => [
    2.0 * p + 0.9 * PI * Math.sin(3.0 * PI * p),
    4.0 * q + 1.6 * PI * Math.sin(4.0 * PI * q),
  ],
  hess: ([p, q]) =>
    diag([
      2.0 + 2.7 * PI ** 2 * Math.cos(3.0 * PI * p),
      4.0 + 6.4 * PI ** 2 * Math.cos(4.0 * PI * q),
    ]),
  domain: box(-1.0, 1.0, 2),
  x0: [0.8, -0.7],
  minima: [[0.0, 0.0]],
  minimaF: [0.0],
  nGlobal: 1,
  description:
    'Bohachevsky function 1 (J&Y 17): a convex bowl with cosine ripples that ' +
    'create many shallow local minima; global minimum f(0, 0) = 0.',
  tags: ['nonconvex', 'multimodal', '2d'],
});

export const levi13 = problem({
  id: 'levi13',
  name: 'Lévi N.13',
  latex: 'f(x,y) = \\sin^2 3\\pi x + (x-1)^2(1 + \\sin^2 3\\pi y) + (y-1)^2(1 + \\sin^2 2\\pi y)',
  f: ([p, q]) =>
    Math.sin(3.0 * PI * p) ** 2 +
    (p - 1.0) ** 2 * (1.0 + Math.sin(3.0 * PI * q) ** 2) +
    (q - 1.0) ** 2 * (1.0 + Math.sin(TWO_PI * q) ** 2),
  grad: ([p, q]) => {
    const s3q = Math.sin(3.0 * PI * q);
    const s2q = Math.sin(TWO_PI * q);
    const gx = 3.0 * PI * Math.sin(6.0 * PI * p) + 2.0 * (p - 1.0) * (1.0 + s3q ** 2);
    const gy =
      3.0 * PI * (p - 1.0) ** 2 * Math.sin(6.0 * PI * q) +
      2.0 * (q - 1.0) * (1.0 + s2q ** 2) +
      TWO_PI * (q - 1.0) ** 2 * Math.sin(4.0 * PI * q);
    return [gx, gy];
  },
  hess: ([p, q]) => {
    const s3q = Math.sin(3.0 * PI * q);
    const s2q = Math.sin(TWO_PI * q);
    const h11 = 18.0 * PI ** 2 * Math.cos(6.0 * PI * p) + 2.0 * (1.0 + s3q ** 2);
    const h12 = 6.0 * PI * (p - 1.0) * Math.sin(6.0 * PI * q);
    const h22 =
      18.0 * PI ** 2 * (p - 1.0) ** 2 * Math.cos(6.0 * PI * q) +
      2.0 * (1.0 + s2q ** 2) +
      8.0 * PI * (q - 1.0) * Math.sin(4.0 * PI * q) +
      8.0 * PI ** 2 * (q - 1.0) ** 2 * Math.cos(4.0 * PI * q);
    return [
      [h11, h12],
      [h12, h22],
    ];
  },
  domain: box(-2.0, 4.0, 2),
  x0: [-1.3, 3.1],
  minima: [[1.0, 1.0]],
  minimaF: [0.0],
  nGlobal: 1,
  description:
    'Lévi function N.13: oscillating terms create rows of local minima ' +
    'around the global minimum f(1, 1) = 0.',
  tags: ['nonconvex', 'multimodal', '2d'],
});

/** Every unconstrained problem, in the Python registration order. */
export const UNCONSTRAINED_PROBLEMS: UnconstrainedProblem[] = [
  quadraticBowl,
  quadraticIll,
  quadraticNd,
  rosenbrock,
  rosenbrockNd,
  himmelblau,
  beale,
  booth,
  matyas,
  threeHumpCamel,
  sixHumpCamel,
  goldsteinPrice,
  rastrigin,
  ackley,
  styblinskiTang,
  mccormick,
  bohachevsky,
  levi13,
];

/** True for the 2-D problems a contour plot can draw. */
export function is2D(p: { dim: number }): p is Problem2D {
  return p.dim === 2;
}

/**
 * The 2-D problems of kind `unconstrained` (registered ones, so temporary demo problems that are
 * also 2-D are included), in registration order.
 */
export function listUnconstrained2D(): Problem2D[] {
  return listProblems<Problem2D>('unconstrained').filter(is2D);
}
