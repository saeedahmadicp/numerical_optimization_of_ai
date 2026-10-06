/**
 * Constrained test problems — TS port of `numopt.problems.constrained`
 * (src/numopt/problems/constrained.py): min f(x) subject to smooth constraints, all in ℝ².
 *
 * Conventions (as in Python):
 *   - `Constraint.kind === 'ineq'` means c_i(x) ≤ 0; `'eq'` means c_i(x) = 0.
 *   - Lagrangian L(x, λ) = f(x) + Σ λ_i c_i(x), λ_i ≥ 0 for inequalities.
 *   - `minima` lists the known local constrained minimizers, global first.
 *
 * `extra` mirrors `Problem.extra`: `minima_f`, `n_global`, `f_min`, `multipliers`, `active`,
 * `affine`, `constraint_hess` (per constraint, x ↦ ∇²c_i(x)), and for the simple convex sets
 * `projection` ('box' | 'disk' | 'polyhedron') with `bounds` / `disk` / `A_ub, b_ub, A_eq, b_eq`,
 * plus `vertices` of bounded polyhedra (Frank–Wolfe's linear-minimization oracle).
 *
 * Every formula keeps the Python operation order (squares are written as products, which is
 * what `x ** 2` computes for float64), so values agree with Python to the last bit, or within a
 * few ulps where libm (`sin`, `cos`, `exp`) differs.
 */
import type { Constraint, Matrix, Problem, Vector } from '../core/types';
import { addProblem, listProblems } from './registry';

export type Projection = 'box' | 'disk' | 'polyhedron';

export interface ConstrainedExtra {
  minima_f: number[];
  n_global: number;
  f_min: number;
  /** For each minimizer, λ_i of every constraint (problem order). */
  multipliers: number[][];
  /** For each minimizer, the indices of the constraints with c_i(x*) = 0. */
  active: number[][];
  /** Per constraint: c_i is affine. */
  affine: boolean[];
  /** Per constraint: x ↦ ∇²c_i(x) (2 × 2). */
  constraint_hess: ((x: Vector) => Matrix)[];
  projection?: Projection;
  bounds?: [number, number][];
  disk?: { center: [number, number]; radius: number };
  A_ub?: number[][];
  b_ub?: number[];
  A_eq?: number[][];
  b_eq?: number[];
  /** Vertices (counter-clockwise) of a bounded polyhedral feasible set. */
  vertices?: [number, number][];
  /** hs21: the published (infeasible) start of Hock & Schittkowski. */
  x0_hs?: [number, number];
  [key: string]: unknown;
}

/** A 2-D constrained problem with exact derivatives of f and of every constraint. */
export interface ConstrainedProblem extends Problem<Vector> {
  f: (x: Vector) => number;
  grad: (x: Vector) => Vector;
  hess: (x: Vector) => Matrix;
  domain: [[number, number], [number, number]];
  x0: Vector;
  minima: Vector[];
  bracket: null;
  roots: Vector[];
  constraints: Constraint[];
  exact: null;
  description: string;
  tags: string[];
  extra: ConstrainedExtra;
}

const SQRT2 = Math.sqrt(2.0);
const SQRT5 = Math.sqrt(5.0);

interface Con {
  kind: 'ineq' | 'eq';
  fun: (x: Vector) => number;
  grad: (x: Vector) => Vector;
  hess: (x: Vector) => Matrix;
  latex: string;
  affine: boolean;
}

const zeros2 = (): Matrix => [
  [0, 0],
  [0, 0],
];

/** The affine constraint c(x) = a₁x + a₂y − b (≤ 0 or = 0). */
function linear(kind: Con['kind'], a: readonly number[], b: number, latex: string): Con {
  const a1 = a[0],
    a2 = a[1];
  return {
    kind,
    fun: (x) => a1 * x[0] + a2 * x[1] - b,
    grad: () => [a1, a2],
    hess: zeros2,
    latex,
    affine: true,
  };
}

/** c(x) = (x − c₁)² + (y − c₂)² − r² (≤ 0: inside the disk; = 0: on the circle). */
function disk(kind: Con['kind'], center: readonly number[], radius: number, latex: string): Con {
  const c1 = center[0],
    c2 = center[1];
  const rSq = radius * radius;
  return {
    kind,
    fun: (x) => {
      const d0 = x[0] - c1,
        d1 = x[1] - c2;
      return d0 * d0 + d1 * d1 - rSq;
    },
    grad: (x) => [2.0 * (x[0] - c1), 2.0 * (x[1] - c2)],
    hess: () => [
      [2, 0],
      [0, 2],
    ],
    latex,
    affine: false,
  };
}

/** Python `str(int(v))` for integral floats, else `repr(float(v))` (the same for these bounds). */
const fmt = (v: number): string => String(v);

/** The four bounds lo_j − x_j ≤ 0, x_j − hi_j ≤ 0 (j = 1, 2, in this order). */
function box(bounds: readonly (readonly [number, number])[], names = ['x', 'y']): Con[] {
  const cons: Con[] = [];
  bounds.forEach(([lo, hi], j) => {
    const e = [0.0, 0.0];
    e[j] = 1.0;
    const ne = e.map((v) => -v); // [-1, -0] — the signed zero is kept, as in Python
    cons.push(linear('ineq', ne, -lo, `${fmt(lo)} - ${names[j]} \\le 0`));
    cons.push(linear('ineq', e, hi, `${names[j]} - ${fmt(hi)} \\le 0`));
  });
  return cons;
}

/** `A_ub, b_ub, A_eq, b_eq` of a problem whose constraints are all affine. */
function polyhedron(cons: readonly Con[]) {
  const origin = [0, 0];
  const A_ub: number[][] = [],
    b_ub: number[] = [],
    A_eq: number[][] = [],
    b_eq: number[] = [];
  for (const c of cons) {
    const a = c.grad(origin).slice();
    const b = -c.fun(origin); // c(x) = aᵀx − b  ⇒  b = −c(0)
    (c.kind === 'ineq' ? A_ub : A_eq).push(a);
    (c.kind === 'ineq' ? b_ub : b_eq).push(b);
  }
  return { A_ub, b_ub, A_eq, b_eq };
}

interface Spec {
  id: string;
  name: string;
  latex: string;
  f: (x: Vector) => number;
  grad: (x: Vector) => Vector;
  hess: (x: Vector) => Matrix;
  constraints: Con[];
  domain: [[number, number], [number, number]];
  x0: [number, number];
  minima: [number, number][];
  minima_f: number[];
  multipliers: number[][];
  active: number[][];
  n_global: number;
  description: string;
  tags: string[];
  extra?: Partial<ConstrainedExtra>;
}

function problem(s: Spec): ConstrainedProblem {
  return addProblem('constrained', {
    id: s.id,
    name: s.name,
    latex: s.latex,
    dim: 2,
    domain: s.domain,
    f: s.f,
    grad: s.grad,
    hess: s.hess,
    x0: [...s.x0],
    bracket: null,
    minima: s.minima.map((m) => [...m]),
    roots: [],
    constraints: s.constraints.map((c) => ({
      kind: c.kind,
      fun: c.fun,
      grad: c.grad,
      latex: c.latex,
    })),
    exact: null,
    description: s.description,
    tags: s.tags,
    extra: {
      minima_f: s.minima_f,
      n_global: s.n_global,
      f_min: s.minima_f[0],
      multipliers: s.multipliers,
      active: s.active,
      affine: s.constraints.map((c) => c.affine),
      constraint_hess: s.constraints.map((c) => c.hess),
      ...(s.extra ?? {}),
    },
  });
}

// ── Quadratic over the unit disk ─────────────────────────────────────────────────────

problem({
  id: 'quadratic_disk',
  name: 'Quadratic over the unit disk',
  latex: '\\min\\ (x-2)^2 + (y-1)^2 \\quad \\text{s.t.}\\ x^2 + y^2 \\le 1',
  f: (x) => {
    const d0 = x[0] - 2.0,
      d1 = x[1] - 1.0;
    return d0 * d0 + d1 * d1;
  },
  grad: (x) => [2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)],
  hess: () => [
    [2, 0],
    [0, 2],
  ],
  constraints: [disk('ineq', [0.0, 0.0], 1.0, 'x^2 + y^2 - 1 \\le 0')],
  domain: [
    [-1.5, 2.5],
    [-1.5, 1.75],
  ],
  x0: [-0.5, 0.5],
  minima: [[2.0 / SQRT5, 1.0 / SQRT5]],
  minima_f: [6.0 - 2.0 * SQRT5],
  multipliers: [[SQRT5 - 1.0]],
  active: [[0]],
  n_global: 1,
  description:
    'The unconstrained minimizer (2, 1) lies outside the unit disk, so the constraint is active: ' +
    'the solution is the boundary point (2, 1)/√5 where the level circle of f touches the disk, ' +
    'with multiplier λ⋆ = √5 − 1.',
  tags: ['convex', 'quadratic', 'active', 'disk', '2d'],
  extra: { projection: 'disk', disk: { center: [0.0, 0.0], radius: 1.0 } },
});

// ── Rosenbrock inside disks ──────────────────────────────────────────────────────────

function rosen(x: Vector): number {
  const a = 1.0 - x[0],
    b = x[1] - x[0] * x[0];
  return a * a + 100.0 * (b * b);
}

function rosenGrad(x: Vector): Vector {
  const x0sq = x[0] * x[0];
  return [-2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x0sq), 200.0 * (x[1] - x0sq)];
}

function rosenHess(x: Vector): Matrix {
  return [
    [2.0 - 400.0 * x[1] + 1200.0 * (x[0] * x[0]), -400.0 * x[0]],
    [-400.0 * x[0], 200.0],
  ];
}

const ROSEN_LATEX = '(1-x)^2 + 100\\,(y-x^2)^2';

problem({
  id: 'rosenbrock_disk',
  name: 'Rosenbrock in the disk x² + y² ≤ 2',
  latex: `\\min\\ ${ROSEN_LATEX} \\quad \\text{s.t.}\\ x^2 + y^2 \\le 2`,
  f: rosen,
  grad: rosenGrad,
  hess: rosenHess,
  constraints: [disk('ineq', [0.0, 0.0], SQRT2, 'x^2 + y^2 - 2 \\le 0')],
  domain: [
    [-1.6, 1.6],
    [-1.6, 1.6],
  ],
  x0: [-1.2, 0.5],
  minima: [[1.0, 1.0]],
  minima_f: [0.0],
  multipliers: [[0.0]],
  active: [[0]],
  n_global: 1,
  description:
    'The Rosenbrock minimizer (1, 1) lies exactly on the circle x² + y² = 2. The constraint is ' +
    'active with multiplier λ⋆ = 0 (weakly active: strict complementarity fails), so penalty and ' +
    'barrier methods approach (1, 1) without the constraint pushing back. Rosenbrock is not ' +
    'convex: det ∇²f = 400 − 80000(y − x²), so ∇²f is indefinite where y > x² + 1/200 and Newton ' +
    'steps there need a Hessian modification.',
  tags: ['nonconvex', 'degenerate', 'weakly-active', 'disk', '2d'],
  extra: { projection: 'disk', disk: { center: [0.0, 0.0], radius: SQRT2 } },
});

problem({
  id: 'rosenbrock_unit_disk',
  name: 'Rosenbrock in the unit disk',
  latex: `\\min\\ ${ROSEN_LATEX} \\quad \\text{s.t.}\\ x^2 + y^2 \\le 1`,
  f: rosen,
  grad: rosenGrad,
  hess: rosenHess,
  constraints: [disk('ineq', [0.0, 0.0], 1.0, 'x^2 + y^2 - 1 \\le 0')],
  domain: [
    [-1.5, 1.5],
    [-1.5, 1.5],
  ],
  x0: [-0.5, 0.0],
  minima: [[0.78641515416842789, 0.61769831252339347]],
  minima_f: [0.045674808719500221],
  multipliers: [[0.12149655699928837]],
  active: [[0]],
  n_global: 1,
  description:
    "Rosenbrock's banana valley cut off by the unit circle (the classic example of MATLAB's " +
    'fmincon documentation). The minimizer (0.7864, 0.6177) lies on the circle where the valley ' +
    'meets it, with multiplier λ⋆ ≈ 0.1215.',
  tags: ['nonconvex', 'active', 'disk', '2d'],
  extra: { projection: 'disk', disk: { center: [0.0, 0.0], radius: 1.0 } },
});

// ── Box-constrained quadratic ────────────────────────────────────────────────────────

{
  // f = ½(x − c)ᵀA(x − c), A = [[2, 1], [1, 2]], c = (2, 0), box [−1, 1]²; x* = (1, ½).
  const bounds: [number, number][] = [
    [-1.0, 1.0],
    [-1.0, 1.0],
  ];
  const cons = box(bounds);
  problem({
    id: 'box_quadratic',
    name: 'Quadratic in a box',
    latex:
      '\\min\\ (x-2)^2 + (x-2)\\,y + y^2 \\quad \\text{s.t.}\\ -1 \\le x \\le 1,\\ -1 \\le y \\le 1',
    f: (x) => {
      const d0 = x[0] - 2.0,
        d1 = x[1];
      return d0 * d0 + d0 * d1 + d1 * d1;
    },
    // A @ (x − c), row by row.
    grad: (x) => {
      const d0 = x[0] - 2.0,
        d1 = x[1] - 0.0;
      return [2.0 * d0 + 1.0 * d1, 1.0 * d0 + 2.0 * d1];
    },
    hess: () => [
      [2.0, 1.0],
      [1.0, 2.0],
    ],
    constraints: cons,
    domain: [
      [-1.5, 2.5],
      [-1.5, 1.5],
    ],
    x0: [-0.5, -0.5],
    minima: [[1.0, 0.5]],
    minima_f: [0.75],
    multipliers: [[0.0, 1.5, 0.0, 0.0]],
    active: [[1]],
    n_global: 1,
    description:
      'A coupled convex quadratic whose unconstrained minimizer (2, 0) lies outside the box ' +
      '[−1, 1]². Only the bound x ≤ 1 is active at the solution (1, ½); clipping (2, 0) to the box ' +
      'would give the wrong point (1, 0).',
    tags: ['convex', 'quadratic', 'box', 'linear-constraints', '2d'],
    extra: {
      projection: 'box',
      bounds,
      vertices: [
        [-1.0, -1.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
      ],
      ...polyhedron(cons),
    },
  });
}

// ── Equality-constrained quadratic ───────────────────────────────────────────────────

{
  const cons = [linear('eq', [1.0, 1.0], 1.0, 'x + y - 1 = 0')];
  problem({
    id: 'linear_eq_quadratic',
    name: 'Quadratic on a line',
    latex: '\\min\\ x^2 + 2y^2 \\quad \\text{s.t.}\\ x + y = 1',
    f: (x) => x[0] * x[0] + 2.0 * (x[1] * x[1]),
    grad: (x) => [2.0 * x[0], 4.0 * x[1]],
    hess: () => [
      [2.0, 0.0],
      [0.0, 4.0],
    ],
    constraints: cons,
    domain: [
      [-1.5, 2.0],
      [-1.5, 2.0],
    ],
    x0: [-0.5, 1.5],
    minima: [[2.0 / 3.0, 1.0 / 3.0]],
    minima_f: [2.0 / 3.0],
    multipliers: [[-4.0 / 3.0]],
    active: [[0]],
    n_global: 1,
    description:
      'The textbook Lagrange-multiplier example: the solution (2/3, 1/3) is where an ellipse ' +
      "x² + 2y² = const touches the line x + y = 1; ∇f = (4/3)(1, 1) is parallel to the line's " +
      'normal, with multiplier ν⋆ = −4/3. The default start lies on the line.',
    tags: ['convex', 'quadratic', 'equality', 'linear-constraints', '2d'],
    extra: { projection: 'polyhedron', ...polyhedron(cons) },
  });
}

// ── Hock–Schittkowski problem 21 ─────────────────────────────────────────────────────

{
  const cons = [
    linear('ineq', [-10.0, 1.0], -10.0, '10 - 10x + y \\le 0'),
    ...box([
      [2.0, 50.0],
      [-50.0, 50.0],
    ]),
  ];
  problem({
    id: 'hs21',
    name: 'Hock–Schittkowski 21',
    latex:
      '\\min\\ 0.01x^2 + y^2 - 100 \\quad \\text{s.t.}\\ 10x - y \\ge 10,\\ ' +
      '2 \\le x \\le 50,\\ -50 \\le y \\le 50',
    f: (x) => 0.01 * (x[0] * x[0]) + x[1] * x[1] - 100.0,
    grad: (x) => [0.02 * x[0], 2.0 * x[1]],
    hess: () => [
      [0.02, 0.0],
      [0.0, 2.0],
    ],
    constraints: cons,
    domain: [
      [0.0, 52.0],
      [-52.0, 52.0],
    ],
    x0: [10.0, 10.0],
    minima: [[2.0, 0.0]],
    minima_f: [-99.96],
    multipliers: [[0.0, 0.04, 0.0, 0.0, 0.0]],
    active: [[1]],
    n_global: 1,
    description:
      'Hock & Schittkowski (1981) test problem 21: a convex quadratic with one general linear ' +
      'inequality and bounds. Only the bound x ≥ 2 is active at (2, 0). The published start ' +
      '(−1, −1) is infeasible; the default start (10, 10) is strictly feasible so that ' +
      'interior-point methods can use it.',
    tags: ['convex', 'quadratic', 'linear-constraints', 'bounds', 'hock-schittkowski', '2d'],
    extra: {
      projection: 'polyhedron',
      ...polyhedron(cons),
      vertices: [
        [2.0, -50.0],
        [50.0, -50.0],
        [50.0, 50.0],
        [6.0, 50.0],
        [2.0, 10.0],
      ],
      x0_hs: [-1.0, -1.0],
    },
  });
}

// ── Two half-planes, both active ─────────────────────────────────────────────────────

{
  const cons = [
    linear('ineq', [1.0, 2.0], 2.0, 'x + 2y - 2 \\le 0'),
    linear('ineq', [2.0, 1.0], 2.0, '2x + y - 2 \\le 0'),
  ];
  problem({
    id: 'halfplanes_quadratic',
    name: 'Quadratic in a wedge of two half-planes',
    latex:
      '\\min\\ (x-2)^2 + 2(y-\\tfrac32)^2 \\quad \\text{s.t.}\\ x + 2y \\le 2,\\ 2x + y \\le 2',
    f: (x) => {
      const d0 = x[0] - 2.0,
        d1 = x[1] - 1.5;
      return d0 * d0 + 2.0 * (d1 * d1);
    },
    grad: (x) => [2.0 * (x[0] - 2.0), 4.0 * (x[1] - 1.5)],
    hess: () => [
      [2.0, 0.0],
      [0.0, 4.0],
    ],
    constraints: cons,
    domain: [
      [-2.0, 3.0],
      [-2.0, 3.0],
    ],
    x0: [-1.0, -1.0],
    minima: [[2.0 / 3.0, 2.0 / 3.0]],
    minima_f: [19.0 / 6.0],
    multipliers: [[4.0 / 3.0, 2.0 / 3.0]],
    active: [[0, 1]],
    n_global: 1,
    description:
      'Two linear inequalities form an unbounded wedge; the unconstrained minimizer (2, 3/2) lies ' +
      'outside it and the solution is the corner (2/3, 2/3) where both constraints are active ' +
      'with multipliers 4/3 and 2/3.',
    tags: ['convex', 'quadratic', 'linear-constraints', 'vertex-solution', '2d'],
    extra: { projection: 'polyhedron', ...polyhedron(cons) },
  });
}

// ── Linear objective on the unit circle (nonlinear equality) ─────────────────────────

problem({
  id: 'circle_eq',
  name: 'Linear function on the unit circle',
  latex: '\\min\\ x + y \\quad \\text{s.t.}\\ x^2 + y^2 = 1',
  f: (x) => x[0] + x[1],
  grad: () => [1.0, 1.0],
  hess: zeros2,
  constraints: [disk('eq', [0.0, 0.0], 1.0, 'x^2 + y^2 - 1 = 0')],
  domain: [
    [-1.6, 1.6],
    [-1.6, 1.6],
  ],
  x0: [0.3, 1.2],
  minima: [[-1.0 / SQRT2, -1.0 / SQRT2]],
  minima_f: [-SQRT2],
  multipliers: [[1.0 / SQRT2]],
  active: [[0]],
  n_global: 1,
  description:
    'A linear objective on a nonlinear equality constraint (a nonconvex feasible set). The ' +
    "Lagrangian's curvature 2ν⋆I comes only from the constraint. The point (1/√2, 1/√2) is also " +
    'a KKT point (ν = −1/√2), but it is the maximizer.',
  tags: ['nonconvex', 'equality', 'nonlinear-constraints', '2d'],
});

// ── Mishra's bird, constrained (multimodal) ──────────────────────────────────────────

/** sin x, cos x, sin y, cos y, A = e^{(1−cos x)²}, B = e^{(1−sin y)²}, a = A'/A, b = B'/B. */
function mishraParts(x: Vector) {
  const sx = Math.sin(x[0]),
    cx = Math.cos(x[0]),
    sy = Math.sin(x[1]),
    cy = Math.cos(x[1]);
  const A = Math.exp((1.0 - cx) * (1.0 - cx));
  const B = Math.exp((1.0 - sy) * (1.0 - sy));
  const a = 2.0 * (1.0 - cx) * sx;
  const b = -2.0 * (1.0 - sy) * cy;
  return { sx, cx, sy, cy, A, B, a, b };
}

problem({
  id: 'mishra_bird_constrained',
  name: "Mishra's bird (constrained)",
  latex:
    '\\min\\ \\sin y\\, e^{(1-\\cos x)^2} + \\cos x\\, e^{(1-\\sin y)^2} + (x-y)^2' +
    ' \\quad \\text{s.t.}\\ (x+5)^2 + (y+5)^2 \\le 25',
  f: (x) => {
    const { cx, sy, A, B } = mishraParts(x);
    const d = x[0] - x[1];
    return sy * A + cx * B + d * d;
  },
  grad: (x) => {
    const { sx, cx, sy, cy, A, B, a, b } = mishraParts(x);
    const d = x[0] - x[1];
    return [sy * A * a - sx * B + 2.0 * d, cy * A + cx * B * b - 2.0 * d];
  },
  hess: (x) => {
    const { sx, cx, sy, cy, A, B, a, b } = mishraParts(x);
    const aPrime = 2.0 * (sx * sx + (1.0 - cx) * cx);
    const bPrime = 2.0 * (cy * cy) + 2.0 * (1.0 - sy) * sy;
    const fxx = sy * A * (a * a + aPrime) - cx * B + 2.0;
    const fxy = cy * A * a - sx * B * b - 2.0;
    const fyy = -sy * A + cx * B * (b * b + bPrime) + 2.0;
    return [
      [fxx, fxy],
      [fxy, fyy],
    ];
  },
  constraints: [disk('ineq', [-5.0, -5.0], 5.0, '(x+5)^2 + (y+5)^2 - 25 \\le 0')],
  domain: [
    [-10.0, 0.0],
    [-10.0, 0.0],
  ],
  x0: [-5.0, -5.0],
  minima: [
    [-3.1302468034546562, -1.5821421769300335],
    [-9.1907144922242008, -7.727253571757136],
    [-3.1757265085378856, -7.8198477790263903],
    [-8.9438004526852257, -1.9264941858848208],
    [-5.3776666083326097, -5.6179076792316662],
  ],
  minima_f: [
    -106.76453674926471, -97.894220784141453, -87.310882733003581, -21.518248963023638,
    1.4870191265420765,
  ],
  multipliers: [[0.0], [6.4201335864045408], [0.0], [8.0114423573423998], [0.0]],
  active: [[], [0], [], [0], []],
  n_global: 1,
  description:
    'A multimodal benchmark (Mishra 2006) restricted to a disk. It has five local minimizers: ' +
    'three interior ones (including the global one) and two on the boundary circle with an ' +
    'active constraint. Local methods find whichever basin they start in.',
  tags: ['nonconvex', 'multimodal', 'disk', '2d'],
  extra: { projection: 'disk', disk: { center: [-5.0, -5.0], radius: 5.0 } },
});

/** The constrained problems in Python registration order. */
export function listConstrained(): ConstrainedProblem[] {
  return listProblems<ConstrainedProblem>('constrained');
}
