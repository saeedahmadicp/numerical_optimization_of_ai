/**
 * Scalar root-finding problems f(x) = 0 — TS port of `numopt.problems.roots`
 * (src/numopt/problems/roots.py), kind `"roots"`.
 *
 * Every problem is 1-D with
 *
 *   - `f`       the function;
 *   - `grad`    the exact first derivative f′(x);
 *   - `hess`    the exact second derivative f″(x) (Halley's method uses it);
 *   - `bracket` an interval [a, b] with f(a)·f(b) < 0 for the bracketing methods;
 *   - `x0`      a starting point for the open methods;
 *   - `roots`   every real root of f, in increasing order, to full double precision;
 *   - `domain`  the plotting window [x_min, x_max].
 *
 * Functions are written in the same factored form as Python and evaluated in the same order.
 * `exp`, `sin`, `cos` and `atan` are the correctly rounded versions of `crmath.ts`: glibc (the
 * Python reference) rounds correctly in all but ~0.1 % of arguments, V8's `Math.*` differs from
 * it by one ulp on 3–10 % of them — enough to change the path of a root finder near a root of
 * eˣ − 10. `plot` is the same function on the fast `Math.*` versions, for drawing only. IEEE semantics far from the domain: overflow gives ±Infinity, a non-finite argument
 * gives NaN or ±Infinity — never an exception (Python wraps `math.exp`, `**`, `math.sin` and
 * `math.cos` to the same effect; JavaScript never raises).
 */
import type { Problem } from '../core/types';
import { crAtan, crCos, crExp, crSin } from '../methods/roots/crmath';
import { addProblem } from './registry';

export interface RootProblem extends Problem<number> {
  f: (x: number) => number;
  /** f′(x), exact. */
  grad: (x: number) => number;
  /** f″(x), exact. */
  hess: (x: number) => number;
  dim: 1;
  /** Plotting window [x_min, x_max]. */
  domain: [number, number];
  /** A bracket [a, b] with f(a)·f(b) < 0. */
  bracket: [number, number];
  /** Default start of the open methods. */
  x0: number;
  /** Every real root, increasing. */
  roots: number[];
  minima: [];
  constraints: [];
  exact: null;
  description: string;
  tags: string[];
  extra: Record<string, number>;
  /** f on the platform's fast `Math.*` (for drawing curves; within an ulp or two of `f`). */
  plot: (x: number) => number;
}

type Spec = Omit<
  RootProblem,
  'dim' | 'minima' | 'constraints' | 'exact' | 'extra' | 'tags' | 'plot'
> & {
  plot?: (x: number) => number;
  tags?: string[];
  extra?: Record<string, number>;
};

function scalar(s: Spec): RootProblem {
  return addProblem('roots', {
    ...s,
    dim: 1,
    minima: [],
    constraints: [],
    exact: null,
    tags: s.tags ?? [],
    extra: s.extra ?? {},
    plot: s.plot ?? s.f,
  } satisfies RootProblem);
}

/** xⁿ for an integer n ≥ 0 (Python `_pow`: x**n, ±inf on overflow — JS gives that natively). */
const pow = (x: number, n: number) => x ** n;

// --- x² − 2 ---------------------------------------------------------------------------

export const SQRT2 = scalar({
  id: 'sqrt2',
  name: 'Square root of 2',
  latex: 'f(x) = x^2 - 2',
  f: (x) => x * x - 2.0,
  grad: (x) => 2.0 * x,
  hess: () => 2.0,
  bracket: [1.0, 2.0],
  x0: 1.0,
  roots: [-1.4142135623730951, 1.4142135623730951],
  domain: [-2.0, 2.5],
  description: 'The classic first example: a convex parabola with a simple root at √2.',
  tags: ['polynomial', 'simple-root', 'convex'],
});

// --- Wallis' cubic x³ − 2x − 5 ----------------------------------------------------------

export const CUBIC = scalar({
  id: 'cubic',
  name: "Wallis' cubic",
  latex: 'f(x) = x^3 - 2x - 5',
  f: (x) => (x * x - 2.0) * x - 5.0,
  grad: (x) => 3.0 * x * x - 2.0,
  hess: (x) => 6.0 * x,
  bracket: [2.0, 3.0],
  x0: 2.0,
  roots: [2.0945514815423265],
  domain: [0.0, 3.5],
  description:
    "The equation Wallis used (1685) to present Newton's method; one real root near 2.0946.",
  tags: ['polynomial', 'simple-root', 'historical'],
});

// --- cos x − x ---------------------------------------------------------------------------

export const COS_MINUS_X = scalar({
  id: 'cos_minus_x',
  name: 'cos x = x',
  latex: 'f(x) = \\cos x - x',
  f: (x) => crCos(x) - x,
  grad: (x) => -crSin(x) - 1.0,
  hess: (x) => -crCos(x),
  plot: (x) => Math.cos(x) - x,
  bracket: [0.0, 1.0],
  x0: 1.0,
  roots: [0.7390851332151607],
  domain: [-1.0, 2.0],
  description:
    'The fixed point of cos (the Dottie number). f is decreasing, so the fixed-point ' +
    'iteration x ← x − λ f(x) needs λ < 0.',
  tags: ['transcendental', 'simple-root', 'fixed-point'],
});

// --- x¹⁰ − 1 -----------------------------------------------------------------------------

export const X10_MINUS_1 = scalar({
  id: 'x10_minus_1',
  name: 'x¹⁰ − 1',
  latex: 'f(x) = x^{10} - 1',
  f: (x) => pow(x, 10) - 1.0,
  grad: (x) => 10.0 * pow(x, 9),
  hess: (x) => 90.0 * pow(x, 8),
  bracket: [0.0, 1.3],
  x0: 1.3,
  roots: [-1.0, 1.0],
  domain: [0.0, 1.4],
  description:
    'Flat on the left and steep on the right of x = 1, so plain regula falsi keeps the ' +
    'right end fixed and converges slowly; Illinois-type methods fix this. Newton from ' +
    'x₀ = 0.5 overshoots to x ≈ 51.6.',
  tags: ['polynomial', 'simple-root', 'regula-falsi-hard'],
});

// --- Kepler's equation -------------------------------------------------------------------

export const KEPLER_E = 0.9;
export const KEPLER_M = 0.3;

export const KEPLER = scalar({
  id: 'kepler',
  name: "Kepler's equation (e = 0.9)",
  latex: 'f(E) = E - e\\sin E - M,\\quad e = 0.9,\\ M = 0.3',
  f: (x) => x - KEPLER_E * crSin(x) - KEPLER_M,
  grad: (x) => 1.0 - KEPLER_E * crCos(x),
  hess: (x) => KEPLER_E * crSin(x),
  plot: (x) => x - KEPLER_E * Math.sin(x) - KEPLER_M,
  bracket: [0.0, Math.PI],
  x0: KEPLER_M,
  roots: [1.103517720303087],
  domain: [0.0, Math.PI],
  description:
    "Eccentric anomaly E of an orbit with eccentricity e = 0.9 at mean anomaly M = 0.3. With the naive start E₀ = M, f′(E₀) = 1 − e cos M ≈ 0.14 is small and Newton's first step overshoots to E ≈ 2.2.",
  tags: ['transcendental', 'simple-root', 'astronomy'],
  extra: { e: KEPLER_E, M: KEPLER_M },
});

// --- (x − 1)²(x + 2) ---------------------------------------------------------------------

export const DOUBLE_ROOT = scalar({
  id: 'double_root',
  name: 'Double root',
  latex: 'f(x) = (x-1)^2 (x+2)',
  f: (x) => (x - 1.0) * (x - 1.0) * (x + 2.0),
  grad: (x) => 3.0 * (x - 1.0) * (x + 1.0),
  hess: (x) => 6.0 * x,
  bracket: [-3.0, 0.0],
  x0: 2.0,
  roots: [-2.0, 1.0],
  domain: [-3.0, 2.5],
  description:
    'f touches zero at the double root x = 1 without a sign change, so no bracket can ' +
    'isolate it; Newton converges to it only linearly (rate ½). The default bracket ' +
    'isolates the simple root x = −2.',
  tags: ['polynomial', 'multiple-root'],
  extra: { multiplicity_at_1: 2.0 },
});

// --- arctan x ----------------------------------------------------------------------------

/**
 * Newton on arctan diverges for |x₀| > x_c, where x_c solves (1 + x²)·arctan x = 2x (then the
 * Newton map sends x_c to −x_c: a 2-cycle that separates the two regimes).
 */
export const ATAN_NEWTON_CRITICAL = 1.3917452002707353;

export const ATAN_NEWTON = scalar({
  id: 'atan_newton',
  name: 'arctan x (Newton diverges)',
  latex: 'f(x) = \\arctan x',
  f: crAtan,
  plot: Math.atan,
  grad: (x) => 1.0 / (1.0 + x * x),
  hess: (x) => (-2.0 * x) / ((1.0 + x * x) * (1.0 + x * x)),
  bracket: [-1.5, 2.0],
  x0: 1.5,
  roots: [0.0],
  domain: [-6.0, 6.0],
  description:
    "Newton's method converges to 0 only for |x₀| < 1.3917452; from x₀ = 1.5 the " +
    'iterates alternate in sign and grow without bound.',
  tags: ['transcendental', 'simple-root', 'newton-diverges'],
  extra: { newton_critical_x0: ATAN_NEWTON_CRITICAL },
});

// --- x³ − 2x + 2 -------------------------------------------------------------------------

export const NEWTON_CYCLE = scalar({
  id: 'newton_cycle',
  name: 'Newton 2-cycle',
  latex: 'f(x) = x^3 - 2x + 2',
  f: (x) => (x * x - 2.0) * x + 2.0,
  grad: (x) => 3.0 * x * x - 2.0,
  hess: (x) => 6.0 * x,
  bracket: [-2.0, -1.0],
  x0: 0.0,
  roots: [-1.7692923542386316],
  domain: [-2.5, 2.0],
  description:
    'Newton from x₀ = 0 cycles 0 → 1 → 0 → … forever and never reaches the only real root x ≈ −1.769 (the cycle is superattracting because f″(0) = 0).',
  tags: ['polynomial', 'simple-root', 'newton-cycles'],
});

// --- eˣ − 10 -----------------------------------------------------------------------------

export const STEEP_EXP = scalar({
  id: 'steep_exp',
  name: 'Steep exponential',
  latex: 'f(x) = e^x - 10',
  f: (x) => crExp(x) - 10.0,
  grad: crExp,
  hess: crExp,
  plot: (x) => Math.exp(x) - 10.0,
  bracket: [0.0, 6.0],
  x0: 4.0,
  roots: [2.302585092994046],
  domain: [0.0, 6.0],
  description:
    'Very steep on the right of the bracket (f(6) ≈ 393), so regula falsi keeps the ' +
    'right end and crawls; Newton from the left (x₀ < 0) jumps far to the right first.',
  tags: ['transcendental', 'simple-root', 'steep'],
});

// --- Wilkinson-type polynomial ∏(x − i), i = 1..5 ------------------------------------------

const W_ROOTS = [1.0, 2.0, 3.0, 4.0, 5.0];

function wF(x: number): number {
  let p = 1.0;
  for (const r of W_ROOTS) p *= x - r;
  return p;
}

/** f′(x) = Σᵢ ∏_{j≠i} (x − j)   (product rule; accurate near every root). */
function wDf(x: number): number {
  let total = 0.0;
  for (let i = 0; i < W_ROOTS.length; i++) {
    let p = 1.0;
    for (let j = 0; j < W_ROOTS.length; j++) if (j !== i) p *= x - W_ROOTS[j];
    total += p;
  }
  return total;
}

/** f″(x) = 2 Σ_{i<j} ∏_{l∉{i,j}} (x − l). */
function wD2f(x: number): number {
  let total = 0.0;
  const n = W_ROOTS.length;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let p = 1.0;
      for (let m = 0; m < n; m++) if (m !== i && m !== j) p *= x - W_ROOTS[m];
      total += p;
    }
  }
  return 2.0 * total;
}

export const WILKINSON5 = scalar({
  id: 'wilkinson5',
  name: 'Wilkinson polynomial (degree 5)',
  latex: 'f(x) = \\prod_{i=1}^{5} (x - i)',
  f: wF,
  grad: wDf,
  hess: wD2f,
  bracket: [0.6, 5.3],
  x0: 5.4,
  roots: [...W_ROOTS],
  domain: [0.5, 5.5],
  description:
    'Five simple roots 1, …, 5; the default bracket holds all of them, so each bracketing ' +
    'method shows which root its own steps lead to. Evaluated in product form.',
  tags: ['polynomial', 'multiple-roots-in-bracket'],
});
