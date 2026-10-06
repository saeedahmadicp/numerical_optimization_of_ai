/**
 * Nonlinear least-squares problems — TS port of `numopt.problems.least_squares`
 * (src/numopt/problems/least_squares.py): minimize f(x) = ½‖r(x)‖² over parameters x ∈ ℝⁿ.
 *
 * Every problem supplies the residual r: ℝⁿ → ℝᵐ, its Jacobian J = ∂r/∂x (m × n), and the exact
 * derivatives of f (Nocedal & Wright (2006), eqs. 10.4–10.5):
 *
 *     ∇f(x)  = J(x)ᵀ r(x),
 *     ∇²f(x) = J(x)ᵀ J(x) + Σᵢ rᵢ(x) ∇²rᵢ(x)      (Gauss–Newton term + second-order term).
 *
 * The sign convention is r = model − data, so J is the derivative of the model.
 *
 * The synthetic data (`exp_decay_fit`, `circle_fit`) are drawn from the shared Mulberry32 `Rng`
 * in the Python order, and every formula keeps the Python operation order (NumPy's pairwise
 * `np.sum` in f), so values agree with Python to the last bit or within a few ulps (libm).
 *
 * `extra` mirrors `Problem.extra`: `t`, `y` (curve fits), `points_x`, `points_y`, `radius`
 * (circle fit), `model` (LaTeX), `true_params`, `noise_std`, `seed`, `m`, `minima_f`, `n_global`,
 * `f_min`.
 */
import { Rng } from '../core/rng';
import type { Matrix, Problem, Vector } from '../core/types';
import { addProblem } from './registry';

export interface LeastSquaresExtra {
  m: number;
  minima_f: number[];
  n_global: number;
  f_min: number;
  model: string;
  t?: number[];
  y?: number[];
  points_x?: number[];
  points_y?: number[];
  radius?: number;
  true_params?: number[];
  noise_std?: number;
  seed?: number;
  [key: string]: unknown;
}

/** A 2-D nonlinear least-squares problem with exact residual derivatives. */
export interface LeastSquaresProblem extends Problem<Vector> {
  f: (x: Vector) => number;
  grad: (x: Vector) => Vector;
  hess: (x: Vector) => Matrix;
  jac: (x: Vector) => Matrix;
  residual: (x: Vector) => Vector;
  /** `[[lo_1, hi_1], [lo_2, hi_2]]` — the plotting box in parameter space. */
  domain: [[number, number], [number, number]];
  x0: Vector;
  /** Known local minimizers, global first. */
  minima: Vector[];
  bracket: null;
  roots: Vector[];
  constraints: [];
  exact: null;
  description: string;
  tags: string[];
  extra: LeastSquaresExtra;
}

/**
 * `np.sum` of a contiguous float64 array: NumPy's pairwise summation (a plain loop from −0.0
 * below 8 terms, 8 accumulators for 8 ≤ n ≤ 128, recursive halves above).
 */
function npSum(a: readonly number[], start = 0, n = a.length): number {
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

interface Spec {
  id: string;
  name: string;
  latex: string;
  residual: (x: Vector) => Vector;
  jac: (x: Vector) => Matrix;
  /** Stacked second derivatives ∇²rᵢ, shape (m, n, n). */
  residualHessians: (x: Vector) => Matrix[];
  domain: [[number, number], [number, number]];
  x0: Vector;
  minima: Vector[];
  nGlobal: number;
  m: number;
  description: string;
  tags: string[];
  extra: Record<string, unknown> & { model: string };
}

/** Build f = ½‖r‖², ∇f = Jᵀr and ∇²f = JᵀJ + Σ rᵢ∇²rᵢ from the residual pieces (`_ls_problem`). */
function lsProblem(s: Spec): LeastSquaresProblem {
  const f = (x: Vector): number => {
    const r = s.residual(x);
    return 0.5 * npSum(r.map((v) => v * v));
  };
  const grad = (x: Vector): Vector => {
    const J = s.jac(x);
    const r = s.residual(x);
    const n = x.length;
    const g = new Array<number>(n).fill(0);
    for (let j = 0; j < n; j++) {
      let acc = 0;
      for (let i = 0; i < r.length; i++) acc += J[i][j] * r[i];
      g[j] = acc;
    }
    return g;
  };
  const hess = (x: Vector): Matrix => {
    const r = s.residual(x);
    const J = s.jac(x);
    const R = s.residualHessians(x);
    const n = x.length;
    const H: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    for (let a = 0; a < n; a++)
      for (let b = 0; b < n; b++) {
        let gn = 0;
        for (let i = 0; i < r.length; i++) gn += J[i][a] * J[i][b];
        let so = 0;
        for (let i = 0; i < r.length; i++) so += r[i] * R[i][a][b];
        H[a][b] = gn + so;
      }
    return H.map((row, a) => row.map((v, b) => 0.5 * (v + H[b][a])));
  };
  const minimaF = s.minima.map((p) => f(p));
  return addProblem('least_squares', {
    id: s.id,
    name: s.name,
    latex: s.latex,
    dim: s.x0.length,
    domain: s.domain,
    f,
    grad,
    hess,
    jac: s.jac,
    residual: s.residual,
    x0: s.x0,
    bracket: null,
    minima: s.minima,
    roots: [],
    constraints: [],
    exact: null,
    description: s.description,
    tags: ['least-squares', ...s.tags],
    extra: {
      ...s.extra,
      m: s.m,
      minima_f: minimaF,
      n_global: s.nGlobal,
      f_min: minimaF[0],
    },
  });
}

// ---------------------------------------------------------------------------------------
// Exponential decay  y ≈ a·exp(−b t)
// ---------------------------------------------------------------------------------------

export const expDecayFit = (() => {
  // t_i = 0.3 i for i = 0..14 and y_i = a* exp(−b* t_i) + σ·rng.normal(), drawn in order from Rng(7).
  const seed = 7,
    sigma = 0.05,
    aTrue = 2.5,
    bTrue = 1.3;
  const t = Array.from({ length: 15 }, (_, i) => 0.3 * i);
  const rng = new Rng(seed);
  const y = t.map((ti) => aTrue * Math.exp(-bTrue * ti) + sigma * rng.normal());

  const residual = (x: Vector): Vector => t.map((ti, i) => x[0] * Math.exp(-x[1] * ti) - y[i]);
  const jac = (x: Vector): Matrix =>
    t.map((ti) => {
      const e = Math.exp(-x[1] * ti);
      return [e, -x[0] * ti * e];
    });
  const residualHessians = (x: Vector): Matrix[] =>
    t.map((ti) => {
      // ∂²rᵢ/∂a² = 0, ∂²rᵢ/∂a∂b = −tᵢeᵢ, ∂²rᵢ/∂b² = a tᵢ² eᵢ, with eᵢ = exp(−b tᵢ).
      const e = Math.exp(-x[1] * ti);
      const ab = -ti * e;
      return [
        [0, ab],
        [ab, x[0] * ti ** 2 * e],
      ];
    });

  return lsProblem({
    id: 'exp_decay_fit',
    name: 'Exponential decay fit',
    latex: '\\min_{a,b}\\ \\tfrac12\\sum_{i=1}^{15}\\left(a\\,e^{-b t_i} - y_i\\right)^2',
    residual,
    jac,
    residualHessians,
    domain: [
      [0.0, 4.0],
      [0.0, 3.0],
    ],
    x0: [1.0, 0.3],
    minima: [[2.4935921290442926, 1.3348854533574257]],
    nGlobal: 1,
    m: t.length,
    description:
      'Fit y = a·exp(−b t) to 15 noisy samples of 2.5·exp(−1.3 t) (σ = 0.05). ' +
      'A small-residual problem: Gauss–Newton converges fast near the solution.',
    tags: ['curve-fit', 'small-residual', '2d'],
    extra: {
      t,
      y,
      model: 'y = a\\,e^{-b t}',
      true_params: [aTrue, bTrue],
      noise_std: sigma,
      seed,
    },
  });
})();

// ---------------------------------------------------------------------------------------
// Rosenbrock as a zero-residual least-squares problem
// ---------------------------------------------------------------------------------------

export const rosenbrockLs = lsProblem({
  id: 'rosenbrock_ls',
  name: 'Rosenbrock (least squares)',
  latex:
    'r(x,y) = \\begin{pmatrix} 10\\,(y - x^2) \\\\ 1 - x \\end{pmatrix},\\ ' +
    'f = \\tfrac12\\|r\\|^2',
  residual: (x) => [10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]],
  jac: (x) => [
    [-20.0 * x[0], 10.0],
    [-1.0, 0.0],
  ],
  residualHessians: () => [
    [
      [-20.0, 0.0],
      [0.0, 0.0],
    ],
    [
      [0.0, 0.0],
      [0.0, 0.0],
    ],
  ],
  domain: [
    [-2.0, 2.0],
    [-1.0, 3.0],
  ],
  x0: [-1.2, 1.0],
  minima: [[1.0, 1.0]],
  nGlobal: 1,
  m: 2,
  description:
    "Rosenbrock's function written as residuals (MGH problem 1); f is half the " +
    "usual Rosenbrock value. J is square with det J = 10, so Gauss–Newton is Newton's " +
    'method for r(x) = 0: the full step (no line search) converges in two steps from ' +
    '(−1.2, 1), via (1, −3.84). With Armijo backtracking the first full step is rejected ' +
    '(f rises from 12.1 to 1171), and the damped iteration takes 10 steps.',
  tags: ['zero-residual', 'classic', '2d'],
  extra: { model: 'r = (10(y - x^2),\\ 1 - x)' },
});

// ---------------------------------------------------------------------------------------
// Circle fit with known radius: find the centre (a, b)
// ---------------------------------------------------------------------------------------

export const circleFit = (() => {
  // From Rng(3), for i = 0..11 in order: θᵢ = rng.uniform(π/6, 5π/6), εᵢ = σ·rng.normal(),
  // pᵢ = c* + (R + εᵢ)(cos θᵢ, sin θᵢ).
  const seed = 3,
    sigma = 0.05,
    radius = 2.0;
  const center: [number, number] = [1.0, -0.5];
  const rng = new Rng(seed);
  const px: number[] = [];
  const py: number[] = [];
  for (let i = 0; i < 12; i++) {
    const theta = rng.uniform(Math.PI / 6.0, (5.0 * Math.PI) / 6.0);
    const rho = radius + sigma * rng.normal();
    px.push(center[0] + rho * Math.cos(theta));
    py.push(center[1] + rho * Math.sin(theta));
  }

  const residual = (x: Vector): Vector =>
    px.map((p, i) => Math.hypot(p - x[0], py[i] - x[1]) - radius);
  const jac = (x: Vector): Matrix =>
    px.map((p, i) => {
      // ∂dᵢ/∂(a, b) = −(pᵢ − c)/dᵢ = −eᵢ (unit vector from the centre to the point).
      const dx = p - x[0];
      const dy = py[i] - x[1];
      const d = Math.hypot(dx, dy);
      // NOTE (as in Python): dᵢ is not differentiable when the centre sits on a data point
      // (dᵢ = 0); the zero row is used there (0 is in the subdifferential of ‖·‖ at 0).
      return d > 0.0 ? [-dx / d, -dy / d] : [0.0, 0.0];
    });
  const residualHessians = (x: Vector): Matrix[] =>
    px.map((p, i) => {
      // ∇²dᵢ = (I − eᵢeᵢᵀ)/dᵢ (the curvature of the distance function); 0 where dᵢ = 0.
      const dx = p - x[0];
      const dy = py[i] - x[1];
      const d = Math.hypot(dx, dy);
      if (!(d > 0.0)) {
        return [
          [0, 0],
          [0, 0],
        ];
      }
      const ex = dx / d,
        ey = dy / d;
      const off = (-ex * ey) / d;
      return [
        [(1.0 - ex * ex) / d, off],
        [off, (1.0 - ey * ey) / d],
      ];
    });

  return lsProblem({
    id: 'circle_fit',
    name: 'Circle fit (known radius)',
    latex:
      '\\min_{a,b}\\ \\tfrac12\\sum_{i=1}^{12}\\left(\\sqrt{(x_i-a)^2 + (y_i-b)^2} - R\\right)^2,\\ R = 2',
    residual,
    jac,
    residualHessians,
    domain: [
      [-2.0, 4.0],
      [-3.0, 4.5],
    ],
    x0: [3.0, -2.5],
    minima: [
      [0.9572854075202153, -0.5108428174802644],
      [1.2991121563858266, 2.742363185576751],
    ],
    nGlobal: 1,
    m: px.length,
    description:
      'Find the center of a circle of known radius 2 from 12 noisy points on a ' +
      '120° arc. The arc is short, so the mirror image of the true center across the arc ' +
      'is a second, worse local minimum; which one a method finds depends on x₀.',
    tags: ['geometry', 'multiple-minima', '2d'],
    extra: {
      points_x: px,
      points_y: py,
      radius,
      model: '(x - a)^2 + (y - b)^2 = R^2',
      true_params: [...center],
      noise_std: sigma,
      seed,
    },
  });
})();

// ---------------------------------------------------------------------------------------
// Michaelis–Menten enzyme kinetics: Puromycin (treated) data
// ---------------------------------------------------------------------------------------

// Bates & Watts, "Nonlinear Regression Analysis and Its Applications" (1988), Appendix A1.3
// (also R's `Puromycin` data, state == "treated"): substrate concentration (ppm) and initial
// velocity (counts/min²). The least-squares estimates are θ̂ = (212.68, 0.06412) with residual
// sum of squares 1195 (Bates & Watts, §2.2).
const PURO_CONC = [0.02, 0.02, 0.06, 0.06, 0.11, 0.11, 0.22, 0.22, 0.56, 0.56, 1.1, 1.1];
const PURO_RATE = [76.0, 47.0, 97.0, 107.0, 123.0, 139.0, 159.0, 152.0, 191.0, 201.0, 207.0, 200.0];

export const michaelisMenten = lsProblem({
  id: 'michaelis_menten',
  name: 'Michaelis–Menten (Puromycin)',
  latex: '\\min_{V,K}\\ \\tfrac12\\sum_{i=1}^{12}\\left(\\frac{V S_i}{K + S_i} - v_i\\right)^2',
  residual: (x) => PURO_CONC.map((S, i) => (x[0] * S) / (x[1] + S) - PURO_RATE[i]),
  jac: (x) =>
    PURO_CONC.map((S) => {
      const q = 1.0 / (x[1] + S);
      return [S * q, -x[0] * S * q * q];
    }),
  residualHessians: (x) =>
    PURO_CONC.map((S) => {
      // ∂²rᵢ/∂V² = 0, ∂²rᵢ/∂V∂K = −Sᵢ/(K+Sᵢ)², ∂²rᵢ/∂K² = 2 V Sᵢ/(K+Sᵢ)³.
      const q = 1.0 / (x[1] + S);
      const vk = -S * q * q;
      return [
        [0, vk],
        [vk, 2.0 * x[0] * S * q ** 3],
      ];
    }),
  domain: [
    [150.0, 260.0],
    [0.02, 0.12],
  ],
  x0: [205.0, 0.08],
  minima: [[212.68374314253606, 0.06412128168156707]],
  nGlobal: 1,
  m: PURO_CONC.length,
  description:
    'Fit the Michaelis–Menten law v = V S/(K + S) to the classic Puromycin ' +
    '(treated) enzyme-kinetics data of Bates & Watts (1988); start (205, 0.08) as in the ' +
    'book. V and K differ in scale by ~3000×, so JᵀJ is badly conditioned (κ ≈ 1.7·10⁶ at the solution).',
  tags: ['curve-fit', 'real-data', 'badly-scaled', '2d'],
  extra: {
    t: [...PURO_CONC],
    y: [...PURO_RATE],
    model: 'v = \\frac{V S}{K + S}',
  },
});
