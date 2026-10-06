/**
 * Calculus test problems — TS port of `numopt.problems.calculus`
 * (src/numopt/problems/calculus.py): integrands for quadrature and functions to differentiate.
 *
 * Every problem is 1-D with `f`, the exact first derivative `grad` (f′) and the exact second
 * derivative `hess` (f″), the integration interval `domain = [a, b]`, the exact integral `exact`
 * and the point `x0` at which the differentiation methods estimate f′(x0).
 *
 * Python writes `f` with NumPy ufuncs so it also accepts complex numbers (complex-step
 * derivative). The TS port is real-only: a complex-step port must evaluate f(x + ih) itself.
 */
import type { Problem } from '../core/types';
import { addProblem } from './registry';

export interface CalculusProblem extends Problem<number> {
  f: (x: number) => number;
  /** f′(x), exact. */
  grad: (x: number) => number;
  /** f″(x), exact. */
  hess: (x: number) => number;
  dim: 1;
  /** The integration interval [a, b]. */
  domain: [number, number];
  /** Where the differentiation methods estimate f′. */
  x0: number;
  /** ∫ₐᵇ f(x) dx. */
  exact: number;
  bracket: null;
  minima: number[];
  roots: number[];
  constraints: [];
  description: string;
  tags: string[];
  extra: Record<string, number>;
}

type Spec = Omit<
  CalculusProblem,
  'dim' | 'bracket' | 'minima' | 'roots' | 'constraints' | 'extra'
> & {
  extra?: Record<string, number>;
};

function problem(s: Spec): CalculusProblem {
  return addProblem('calculus', {
    ...s,
    dim: 1,
    bracket: null,
    minima: [],
    roots: [],
    constraints: [],
    extra: s.extra ?? {},
  } satisfies CalculusProblem);
}

/** NumPy `np.sign`: −1, 0 or 1 (NaN for NaN). */
function sign(x: number): number {
  if (Number.isNaN(x)) return NaN;
  return x > 0 ? 1.0 : x < 0 ? -1.0 : 0.0;
}

/** f(x) = x³ − 2x + 1 on [0, 2]; ∫ = 2. */
export const poly3 = problem({
  id: 'poly3',
  name: 'Cubic polynomial',
  latex: 'f(x) = x^3 - 2x + 1',
  f: (x) => x ** 3 - 2.0 * x + 1.0,
  grad: (x) => 3.0 * x ** 2 - 2.0,
  hess: (x) => 6.0 * x,
  domain: [0.0, 2.0],
  x0: 1.0,
  exact: 2.0,
  description: 'A cubic: rules exact for degree 3 (Simpson, Boole, Gauss n ≥ 2) give it exactly.',
  tags: ['smooth', 'polynomial'],
});

/** f(x) = eˣ on [0, 1]; ∫ = e − 1 (Python: `math.expm1(1)`). */
export const exp01 = problem({
  id: 'exp_0_1',
  name: 'Exponential',
  latex: 'f(x) = e^{x}',
  f: (x) => Math.exp(x),
  grad: (x) => Math.exp(x),
  hess: (x) => Math.exp(x),
  domain: [0.0, 1.0],
  x0: 1.0,
  exact: 1.718281828459045,
  description: 'Smooth and positive; every rule shows its textbook convergence order.',
  tags: ['smooth', 'analytic'],
});

/** f(x) = sin x on [0, π]; ∫ = 2. */
export const sin0Pi = problem({
  id: 'sin_0_pi',
  name: 'Sine on [0, π]',
  latex: 'f(x) = \\sin x',
  f: (x) => Math.sin(x),
  grad: (x) => Math.cos(x),
  hess: (x) => -Math.sin(x),
  domain: [0.0, Math.PI],
  x0: Math.PI / 3.0,
  exact: 2.0,
  description: 'One arch of the sine; f′(π/3) = 1/2.',
  tags: ['smooth', 'analytic'],
});

/** Runge's function 1/(1 + 25x²) on [−1, 1]; ∫ = (2/5)·arctan 5. */
export const runge = problem({
  id: 'runge',
  name: 'Runge function',
  latex: 'f(x) = \\frac{1}{1 + 25x^2}',
  f: (x) => 1.0 / (1.0 + 25.0 * x ** 2),
  grad: (x) => {
    const u = 1.0 + 25.0 * x ** 2;
    return (-50.0 * x) / u ** 2;
  },
  hess: (x) => {
    const u = 1.0 + 25.0 * x ** 2;
    return (3750.0 * x ** 2 - 50.0) / u ** 3;
  },
  domain: [-1.0, 1.0],
  x0: 0.2,
  // 0.4 * math.atan(5.0), as Python computes it
  exact: 0.5493603067780064,
  description: 'Complex poles at ±i/5 near the interval slow down polynomial-based rules.',
  tags: ['smooth', 'analytic', 'runge'],
});

/** f(x) = √x on [0, 1]; ∫ = 2/3. f′ and f″ are unbounded at 0; f(x < 0) is NaN. */
export const sqrt01 = problem({
  id: 'sqrt_0_1',
  name: 'Square root',
  latex: 'f(x) = \\sqrt{x}',
  f: (x) => Math.sqrt(x),
  grad: (x) => 0.5 / Math.sqrt(x),
  hess: (x) => -0.25 / (x * Math.sqrt(x)),
  domain: [0.0, 1.0],
  x0: 0.25,
  exact: 2.0 / 3.0,
  description:
    'Singular derivative at x = 0: rules of order ≥ 2 drop to h^{1.5} (Riemann sums keep h).',
  tags: ['singular-derivative'],
  extra: { singular_point: 0.0 },
});

/** f(x) = e^{−x²} on [−2, 2]; ∫ = √π·erf(2). */
export const gaussian = problem({
  id: 'gaussian',
  name: 'Gaussian',
  latex: 'f(x) = e^{-x^2}',
  f: (x) => Math.exp(-(x ** 2)),
  grad: (x) => -2.0 * x * Math.exp(-(x ** 2)),
  hess: (x) => (4.0 * x ** 2 - 2.0) * Math.exp(-(x ** 2)),
  domain: [-2.0, 2.0],
  x0: 0.5,
  // math.sqrt(math.pi) * math.erf(2.0), as Python computes it (JS has no erf)
  exact: 1.764162781524843,
  description: 'The bell curve; the exact integral needs the error function erf.',
  tags: ['smooth', 'analytic'],
});

/** f(x) = sin(10x) on [0, π]; ∫ = 0 (five full periods, odd about π/2). */
export const oscillatory = problem({
  id: 'oscillatory',
  name: 'Oscillatory sine',
  latex: 'f(x) = \\sin(10x)',
  f: (x) => Math.sin(10.0 * x),
  grad: (x) => 10.0 * Math.cos(10.0 * x),
  hess: (x) => -100.0 * Math.sin(10.0 * x),
  domain: [0.0, Math.PI],
  x0: 0.5,
  exact: 0.0,
  description: 'Five periods on [0, π]; the exact integral is 0 by symmetry.',
  tags: ['smooth', 'oscillatory'],
});

/** f(x) = 1/(1 + x²) = (arctan x)′ on [0, 1]; ∫ = π/4. */
export const arctanDeriv = problem({
  id: 'arctan_deriv',
  name: 'Derivative of arctan',
  latex: 'f(x) = \\frac{1}{1 + x^2}',
  f: (x) => 1.0 / (1.0 + x ** 2),
  grad: (x) => (-2.0 * x) / (1.0 + x ** 2) ** 2,
  hess: (x) => (6.0 * x ** 2 - 2.0) / (1.0 + x ** 2) ** 3,
  domain: [0.0, 1.0],
  x0: 0.5,
  exact: Math.PI / 4.0,
  description: 'Integrates to π/4: the classic way to compute π by quadrature.',
  tags: ['smooth', 'analytic'],
});

/**
 * f(x) = |x − 0.3| on [0, 1]; ∫ = 0.29. `grad` is sign(x − 0.3) (0 at the kink, the midpoint of
 * the subdifferential) and `hess` is 0 (f″ is a Dirac delta at the kink).
 */
export const absKink = problem({
  id: 'abs_kink',
  name: 'Absolute value with a kink',
  latex: 'f(x) = |x - 0.3|',
  f: (x) => {
    const d = x - 0.3;
    return d >= 0 ? d : -d;
  },
  grad: (x) => sign(x - 0.3),
  hess: (x) => 0.0 * x,
  domain: [0.0, 1.0],
  x0: 0.35,
  exact: 0.29,
  description: 'A kink at x = 0.3: quadrature loses its order; stencils that straddle it fail.',
  tags: ['nonsmooth'],
  extra: { kink: 0.3 },
});

/** Every calculus problem, in the Python registration order. */
export const CALCULUS_PROBLEMS: CalculusProblem[] = [
  poly3,
  exp01,
  sin0Pi,
  runge,
  sqrt01,
  gaussian,
  oscillatory,
  arctanDeriv,
  absKink,
];
