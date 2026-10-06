/**
 * Nonlinear systems F(x) = 0 in ℝ² — TS port of `numopt.problems.systems`
 * (src/numopt/problems/systems.py).
 *
 * Every problem is 2-D with
 *   - `f`    — the residual map F, (2,) → (2,);
 *   - `jac`  — the exact Jacobian J(x) = ∂F/∂x, (2,) → (2, 2) (row i = ∇Fᵢ);
 *   - `x0`   — a default starting point;
 *   - `roots` — every real root inside `domain`;
 *   - `domain` — the plotting window `[[x_min, x_max], [y_min, y_max]]`.
 *
 * Formulas keep the Python operation order (squares written as products), so values agree with
 * Python to the last bit (or within an ulp of libm for `sin`/`cos`). Far from the domain they
 * follow IEEE semantics: overflow gives ±∞ and a non-finite argument gives NaN, never an exception.
 *
 * TS-only (not in Python): `components` gives F₁ and F₂ as separate scalar functions so the lab can
 * draw the zero curves F₁ = 0 and F₂ = 0 without allocating a vector per grid point, and
 * `componentTex` names them for labels.
 */
import type { Matrix, Problem, Vector } from '../core/types';
import { addProblem, listProblems } from './registry';

/** A square 2-D nonlinear system with its exact Jacobian. */
export interface SystemProblem extends Problem<Vector> {
  f: (x: Vector) => Vector;
  jac: (x: Vector) => Matrix;
  /** `[[xmin, xmax], [ymin, ymax]]` — the plotting window. */
  domain: [[number, number], [number, number]];
  x0: Vector;
  /** Every real root inside `domain`. */
  roots: Vector[];
  minima: [];
  bracket: null;
  constraints: [];
  exact: null;
  description: string;
  tags: string[];
  extra: Record<string, unknown>;
  /** TS-only: Fᵢ(x, y) as scalar functions (same arithmetic as `f`). */
  components: [(x: number, y: number) => number, (x: number, y: number) => number];
  /** TS-only: LaTeX of F₁ and F₂ (for legends and the method card). */
  componentTex: [string, string];
}

type Fn2 = (x: number, y: number) => number;

function system(
  id: string,
  name: string,
  latex: string,
  F1: Fn2,
  F2: Fn2,
  J: (x: number, y: number) => Matrix,
  o: {
    x0: Vector;
    roots: Vector[];
    domain: [[number, number], [number, number]];
    description: string;
    tags?: string[];
    componentTex: [string, string];
  },
): SystemProblem {
  return addProblem('systems', {
    id,
    name,
    latex,
    dim: 2,
    f: (v: Vector) => [F1(v[0], v[1]), F2(v[0], v[1])],
    jac: (v: Vector) => J(v[0], v[1]),
    domain: o.domain,
    x0: o.x0,
    roots: o.roots,
    minima: [],
    bracket: null,
    constraints: [],
    exact: null,
    description: o.description,
    tags: o.tags ?? [],
    extra: {},
    components: [F1, F2],
    componentTex: o.componentTex,
  });
}

// Python's math.sin/cos raise ValueError for ±inf; numopt's _sin/_cos return nan. JS already does.
const sin = Math.sin;
const cos = Math.cos;

// --- circle and line -------------------------------------------------------------------------
// x² + y² = 4 and y = x − 1  ⇒  2x² − 2x − 3 = 0  ⇒  x = (1 ± √7)/2, y = x − 1.
const S7 = Math.sqrt(7.0);

export const CIRCLE_LINE = system(
  'circle_line',
  'Circle and line',
  'F(x,y) = \\begin{pmatrix} x^2 + y^2 - 4 \\\\ y - x + 1 \\end{pmatrix}',
  (x, y) => x * x + y * y - 4.0,
  (x, y) => y - x + 1.0,
  (x, y) => [
    [2.0 * x, 2.0 * y],
    [-1.0, 1.0],
  ],
  {
    x0: [2.0, 2.0],
    roots: [
      [(1.0 - S7) / 2.0, (-1.0 - S7) / 2.0],
      [(1.0 + S7) / 2.0, (S7 - 1.0) / 2.0],
    ],
    domain: [
      [-3.0, 3.0],
      [-3.0, 3.0],
    ],
    description:
      'The circle of radius 2 meets the line y = x − 1 in two points. J is singular where x + y = 0 (the line is tangent to a circle centered at 0 there).',
    tags: ['polynomial', 'two-roots'],
    componentTex: ['x^2 + y^2 - 4', 'y - x + 1'],
  },
);

// --- gradient of Rosenbrock's function ---------------------------------------------------------
// f(x, y) = (1 − x)² + 100 (y − x²)²;  F = ∇f;  J = ∇²f.
export const ROSENBROCK_SYSTEM = system(
  'rosenbrock_system',
  'Rosenbrock gradient = 0',
  'F(x,y) = \\nabla\\big[(1-x)^2 + 100(y-x^2)^2\\big]',
  (x, y) => -400.0 * x * (y - x * x) - 2.0 * (1.0 - x),
  (x, y) => 200.0 * (y - x * x),
  (x, y) => [
    [1200.0 * x * x - 400.0 * y + 2.0, -400.0 * x],
    [-400.0 * x, 200.0],
  ],
  {
    x0: [-1.2, 1.0],
    roots: [[1.0, 1.0]],
    domain: [
      [-2.0, 2.0],
      [-1.0, 3.0],
    ],
    description:
      "Stationarity conditions of Rosenbrock's function: the only root is the minimizer " +
      '(1, 1). J = ∇²f is singular on the parabola y = x² + 1/200.',
    tags: ['polynomial', 'badly-scaled'],
    componentTex: ['-400x(y - x^2) - 2(1 - x)', '200(y - x^2)'],
  },
);

// --- Freudenstein & Roth (Moré, Garbow & Hillstrom 1981, problem 2) ---------------------------
// F₁ − F₂ = −2(y − 4)(y² + 2y + 2), so y = 4 and x = 5 is the only real root.
export const FREUDENSTEIN_ROTH = system(
  'freudenstein_roth',
  'Freudenstein–Roth',
  'F(x,y) = \\begin{pmatrix} -13 + x + ((5-y)y - 2)y \\\\ -29 + x + ((y+1)y - 14)y \\end{pmatrix}',
  (x, y) => -13.0 + x + ((5.0 - y) * y - 2.0) * y,
  (x, y) => -29.0 + x + ((y + 1.0) * y - 14.0) * y,
  (_x, y) => [
    [1.0, (10.0 - 3.0 * y) * y - 2.0],
    [1.0, (3.0 * y + 2.0) * y - 14.0],
  ],
  {
    x0: [0.5, -2.0],
    roots: [[5.0, 4.0]],
    domain: [
      [-2.0, 14.0],
      [-3.0, 6.0],
    ],
    description:
      'Moré–Garbow–Hillstrom test problem 2. The only root is (5, 4), but ½‖F‖² also has a ' +
      'non-zero local minimum near (11.41, −0.897), close to where J is singular ' +
      '(6y² − 8y − 12 = 0).',
    tags: ['polynomial', 'MGH', 'local-minimum-trap'],
    componentTex: ['-13 + x + ((5-y)y - 2)y', '-29 + x + ((y+1)y - 14)y'],
  },
);

// --- trigonometric system ------------------------------------------------------------------------
// cos x = cos y ⇒ y = ±x (mod 2π); y = x gives 2 sin x = 1 ⇒ x = π/6 or 5π/6; y = −x gives 0 = 1.
export const TRIG_SYSTEM = system(
  'trig_system',
  'Trigonometric system',
  'F(x,y) = \\begin{pmatrix} \\sin x + \\sin y - 1 \\\\ \\cos x - \\cos y \\end{pmatrix}',
  (x, y) => sin(x) + sin(y) - 1.0,
  (x, y) => cos(x) - cos(y),
  (x, y) => [
    [cos(x), cos(y)],
    [-sin(x), sin(y)],
  ],
  {
    x0: [1.0, 0.2],
    roots: [
      [Math.PI / 6.0, Math.PI / 6.0],
      [(5.0 * Math.PI) / 6.0, (5.0 * Math.PI) / 6.0],
    ],
    domain: [
      [-1.0, 3.5],
      [-1.0, 3.5],
    ],
    description:
      'Two roots on the diagonal, at (π/6, π/6) and (5π/6, 5π/6). det J = sin(x + y) ' +
      'vanishes on the line x + y = π that separates them.',
    tags: ['transcendental', 'two-roots'],
    componentTex: ['\\sin x + \\sin y - 1', '\\cos x - \\cos y'],
  },
);

// --- two intersecting circles --------------------------------------------------------------------
// (x−1)² + y² = 4 and (x+1)² + (y−1)² = 4; subtracting gives y = 2x + ½, then 5x² = 11/4.
const XC = Math.sqrt(0.55);

export const INTERSECTING_CIRCLES = system(
  'intersecting_circles',
  'Two intersecting circles',
  'F(x,y) = \\begin{pmatrix} (x-1)^2 + y^2 - 4 \\\\ (x+1)^2 + (y-1)^2 - 4 \\end{pmatrix}',
  (x, y) => (x - 1.0) * (x - 1.0) + y * y - 4.0,
  (x, y) => (x + 1.0) * (x + 1.0) + (y - 1.0) * (y - 1.0) - 4.0,
  (x, y) => [
    [2.0 * (x - 1.0), 2.0 * y],
    [2.0 * (x + 1.0), 2.0 * (y - 1.0)],
  ],
  {
    x0: [2.0, 2.0],
    roots: [
      [-XC, 0.5 - 2.0 * XC],
      [XC, 0.5 + 2.0 * XC],
    ],
    domain: [
      [-3.5, 3.5],
      [-2.5, 3.5],
    ],
    description:
      'Circles of radius 2 centered at (1, 0) and (−1, 1). det J = 4(1 − x − 2y) vanishes on the line through both centers, which separates the two roots.',
    tags: ['polynomial', 'two-roots'],
    componentTex: ['(x-1)^2 + y^2 - 4', '(x+1)^2 + (y-1)^2 - 4'],
  },
);

/** Every registered system, in Python's registration order. */
export function listSystems(): SystemProblem[] {
  return listProblems<SystemProblem>('systems');
}
