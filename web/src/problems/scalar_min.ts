/**
 * One-dimensional minimization problems (kind `scalar_min`) — TS port of
 * `numopt.problems.scalar_min` (src/numopt/problems/scalar_min.py).
 *
 * Every problem has `dim == 1`; `f`, `grad` (= f') and `hess` (= f'') take and return numbers.
 * Each has a `bracket` (an interval that contains a local minimizer: the default input of the
 * interval methods), a default start `x0` (Newton, minimum bracketing), its interior local
 * minimizers `minima` (the global one first) and a plotting `domain`. Outside its mathematical
 * domain a function returns NaN (it never throws), exactly as in Python.
 *
 * The floating-point expressions keep the Python order and form (`x ** 4 - 4.0 * x ** 2 + x`,
 * `-D * (exp(-ke t) - exp(-ka t))`), so values agree with the Python fixtures to rounding.
 */
import type { Problem } from '../core/types';
import { addProblem } from './registry';

/** A 1-D minimization problem with exact derivatives. */
export interface ScalarProblem extends Problem<number> {
  f: (x: number) => number;
  grad: (x: number) => number;
  hess: (x: number) => number;
  domain: [number, number];
  bracket: [number, number];
  x0: number;
  minima: number[];
  description: string;
  tags: string[];
}

/** Python `math.copysign(v, s)`. */
function copysign(v: number, s: number): number {
  const neg = s < 0 || Object.is(s, -0);
  return neg ? -Math.abs(v) : Math.abs(v);
}

const KIND = 'scalar_min';

// ── quadratic_1d ──────────────────────────────────────────────────────────────────────

export const quadratic1d: ScalarProblem = addProblem(KIND, {
  id: 'quadratic_1d',
  name: 'Quadratic',
  latex: 'f(x) = (x - 2)^2 + 1',
  f: (x: number) => (x - 2.0) ** 2 + 1.0,
  grad: (x: number) => 2.0 * (x - 2.0),
  hess: () => 2.0,
  dim: 1,
  domain: [-1.0, 5.0],
  bracket: [0.0, 5.0],
  x0: 0.5,
  minima: [2.0],
  description:
    "A convex parabola with minimizer x⋆ = 2 and f⋆ = 1. Parabolic interpolation and Newton's method are exact on it.",
  tags: ['smooth', 'unimodal', 'convex'],
});

// ── quartic_1d ────────────────────────────────────────────────────────────────────────

// Critical points of f(x) = x⁴ - 4x² + x solve the depressed cubic x³ + p x + q = 0 with
// p = -2, q = 1/4 (Viète's trigonometric formula, k = 0, 1, 2).
const QP = -2.0,
  QQ = 0.25;
const QUARTIC_CRIT = [0, 1, 2].map(
  (k) =>
    2.0 *
    Math.sqrt(-QP / 3.0) *
    Math.cos(
      Math.acos(((3.0 * QQ) / (2.0 * QP)) * Math.sqrt(-3.0 / QP)) / 3.0 - (2.0 * Math.PI * k) / 3.0,
    ),
);
/** Global minimizer ≈ −1.47300. */
export const QUARTIC_GLOBAL_MIN = QUARTIC_CRIT[2];
/** Local minimizer ≈ 1.34700. */
export const QUARTIC_LOCAL_MIN = QUARTIC_CRIT[0];
/** Local maximizer ≈ 0.12600. */
export const QUARTIC_LOCAL_MAX = QUARTIC_CRIT[1];

export const quartic1d: ScalarProblem = addProblem(KIND, {
  id: 'quartic_1d',
  name: 'Tilted double well',
  latex: 'f(x) = x^4 - 4x^2 + x',
  f: (x: number) => x ** 4 - 4.0 * x ** 2 + x,
  grad: (x: number) => 4.0 * x ** 3 - 8.0 * x + 1.0,
  hess: (x: number) => 12.0 * x ** 2 - 8.0,
  dim: 1,
  domain: [-2.2, 2.2],
  // The default bracket holds only the *local* minimizer ≈ 1.347.
  bracket: [0.5, 2.0],
  x0: 1.0,
  minima: [QUARTIC_GLOBAL_MIN, QUARTIC_LOCAL_MIN],
  description:
    "Two wells: the global minimizer x ≈ −1.4730 (f ≈ −5.4442) and a local minimizer x ≈ 1.3470 (f ≈ −2.6186), separated by a local maximum at x ≈ 0.1260. f″ < 0 on |x| < √(2/3), where Newton's step points uphill.",
  tags: ['smooth', 'multimodal'],
});

// ── sin_1d ────────────────────────────────────────────────────────────────────────────

export const sin1d: ScalarProblem = addProblem(KIND, {
  id: 'sin_1d',
  name: 'Sine',
  latex: 'f(x) = \\sin x',
  f: (x: number) => Math.sin(x),
  grad: (x: number) => Math.cos(x),
  hess: (x: number) => -Math.sin(x),
  dim: 1,
  domain: [0.0, 2.0 * Math.PI],
  bracket: [0.0, 2.0 * Math.PI],
  x0: 4.0,
  minima: [1.5 * Math.PI],
  description:
    'One period of the sine. The interior minimizer is x⋆ = 3π/2 with f⋆ = −1; f″ = −sin x < 0 on (0, π), where Newton needs its safeguard.',
  tags: ['smooth'],
});

// ── x_log_x ───────────────────────────────────────────────────────────────────────────

export const xLogX: ScalarProblem = addProblem(KIND, {
  id: 'x_log_x',
  name: 'x log x',
  latex: 'f(x) = x \\ln x',
  f: (x: number) => {
    if (x > 0.0) return x * Math.log(x);
    if (x === 0.0) return 0.0; // the continuous extension lim_{x→0⁺} x ln x = 0
    return NaN; // outside the domain
  },
  grad: (x: number) => {
    if (x > 0.0) return Math.log(x) + 1.0;
    if (x === 0.0) return -Infinity;
    return NaN;
  },
  hess: (x: number) => {
    if (x > 0.0) return 1.0 / x;
    if (x === 0.0) return Infinity;
    return NaN;
  },
  dim: 1,
  domain: [0.0, 2.0],
  bracket: [0.0, 2.0],
  x0: 0.5,
  minima: [Math.exp(-1.0)],
  description:
    'The entropy-type function x ln x, convex on x > 0 with minimizer x⋆ = 1/e and f⋆ = −1/e. f′ → −∞ at 0; f is undefined (nan) for x < 0, so a long Newton step from x₀ ≥ 1 leaves the domain.',
  tags: ['smooth', 'unimodal', 'convex', 'domain'],
});

// ── abs_shifted ───────────────────────────────────────────────────────────────────────

const KINK = 0.3;

export const absShifted: ScalarProblem = addProblem(KIND, {
  id: 'abs_shifted',
  name: 'Shifted absolute value',
  latex: 'f(x) = |x - 0.3| + \\tfrac{1}{2}x^2',
  f: (x: number) => Math.abs(x - KINK) + 0.5 * x * x,
  // The derivative where it exists; at the kink the minimum-norm subgradient, 0.
  grad: (x: number) => (x === KINK ? 0.0 : copysign(1.0, x - KINK) + x),
  hess: () => 1.0,
  dim: 1,
  domain: [-1.0, 1.0],
  bracket: [-1.0, 1.0],
  x0: -0.5,
  minima: [KINK],
  description:
    "Convex but not differentiable at the minimizer x⋆ = 0.3 (f⋆ = 0.045), because 0 ∈ ∂f(0.3) = [−0.7, 1.3]. Interval elimination still works; Newton's method oscillates between −1 and 1 because f′ jumps by 2 at the kink.",
  tags: ['nonsmooth', 'unimodal', 'convex'],
});

// ── multimodal_1d ─────────────────────────────────────────────────────────────────────

const W = 10.0 / 3.0;

export const multimodal1d: ScalarProblem = addProblem(KIND, {
  id: 'multimodal_1d',
  name: 'Sum of sines',
  latex: 'f(x) = \\sin x + \\sin\\tfrac{10x}{3}',
  f: (x: number) => Math.sin(x) + Math.sin(W * x),
  grad: (x: number) => Math.cos(x) + W * Math.cos(W * x),
  hess: (x: number) => -Math.sin(x) - W * W * Math.sin(W * x),
  dim: 1,
  domain: [2.7, 7.5],
  bracket: [2.7, 7.5],
  x0: 5.0,
  // Roots of f' with f'' > 0 (Brent's root finder to |f'| ≤ 1e-15), global minimizer first.
  minima: [5.145735290256129, 3.387251718444631, 7.0001491168622545],
  description:
    'Three local minima on [2.7, 7.5]; the global one is x ≈ 5.14574 with f ≈ −1.89960. f decreases at 2.7 and increases at 7.5, so every interval method ends at an interior local minimizer, but not necessarily the global one.',
  tags: ['smooth', 'multimodal'],
});

// ── flat_valley ───────────────────────────────────────────────────────────────────────

/** −1/x², computed as −(1/x)² so that a tiny x gives −∞ instead of 1/0. */
function negInvSq(x: number): number {
  const s = 1.0 / x;
  return -(s * s);
}

export const flatValley: ScalarProblem = addProblem(KIND, {
  id: 'flat_valley',
  name: 'Flat valley',
  latex: 'f(x) = e^{-1/x^2}',
  f: (x: number) => (x === 0.0 ? 0.0 : Math.exp(negInvSq(x))),
  // Log form exp(−1/x² − m ln|x|): the tiny factor and the huge power never meet as ∞·0.
  grad: (x: number) =>
    x === 0.0 ? 0.0 : copysign(2.0 * Math.exp(negInvSq(x) - 3.0 * Math.log(Math.abs(x))), x),
  hess: (x: number) =>
    x === 0.0 ? 0.0 : (4.0 - 6.0 * x * x) * Math.exp(negInvSq(x) - 6.0 * Math.log(Math.abs(x))),
  dim: 1,
  domain: [-1.0, 2.0],
  bracket: [-1.0, 2.0],
  x0: 0.5,
  minima: [0.0],
  description:
    "The minimizer x⋆ = 0 is infinitely flat: f and all its derivatives vanish there. In double precision f(x) underflows to exactly 0 for |x| < 0.0367, so no f-comparison can locate x⋆ better than that, and f′(x) < 10⁻⁸ already for |x| < 0.2. Newton's step x − 2x³/(4 − 6x²) converges sublinearly.",
  tags: ['smooth', 'unimodal', 'flat'],
});

// ── rational_1d ───────────────────────────────────────────────────────────────────────

export const rational1d: ScalarProblem = addProblem(KIND, {
  id: 'rational_1d',
  name: 'Rational model',
  latex: 'f(x) = \\frac{x^2 - 2x + 3}{x + 1}',
  f: (x: number) => {
    const d = x + 1.0;
    if (d === 0.0) return NaN; // the pole
    return (x * x - 2.0 * x + 3.0) / d;
  },
  // With u = x + 1: f = u + 6/u − 4, f' = 1 − 6/u², f'' = 12/u³.
  grad: (x: number) => {
    const d = x + 1.0;
    if (d === 0.0) return NaN;
    return (x * x + 2.0 * x - 5.0) / (d * d);
  },
  hess: (x: number) => {
    const d = x + 1.0;
    if (d === 0.0) return NaN;
    return 12.0 / (d * d * d);
  },
  dim: 1,
  domain: [0.0, 6.0],
  bracket: [0.0, 6.0],
  x0: 0.5,
  minima: [Math.sqrt(6.0) - 1.0],
  description:
    'A rational function of the kind a Padé or rational least-squares fit produces. On x > −1 it equals (x + 1) + 6/(x + 1) − 4, which is strictly convex with minimizer x⋆ = √6 − 1 ≈ 1.4495 and f⋆ = 2√6 − 4 ≈ 0.8990. It is asymmetric: steep on the left, nearly linear on the right.',
  tags: ['smooth', 'unimodal', 'convex'],
});

// ── drug_concentration ────────────────────────────────────────────────────────────────

/** Bateman function C(t) = D (e^{−k_e t} − e^{−k_a t}): D [mg/L], k_a, k_e [1/h]. */
export const DRUG_D = 10.0;
export const DRUG_KA = 1.0;
export const DRUG_KE = 0.2;
/** Peak time t* = ln(k_a/k_e) / (k_a − k_e). */
export const DRUG_T_MAX = Math.log(DRUG_KA / DRUG_KE) / (DRUG_KA - DRUG_KE);

export const drugConcentration: ScalarProblem = addProblem(KIND, {
  id: 'drug_concentration',
  name: 'Peak drug concentration',
  latex: 'f(t) = -D\\,(e^{-k_e t} - e^{-k_a t}),\\ D = 10,\\ k_a = 1,\\ k_e = 0.2',
  f: (t: number) => -DRUG_D * (Math.exp(-DRUG_KE * t) - Math.exp(-DRUG_KA * t)),
  grad: (t: number) =>
    -DRUG_D * (DRUG_KA * Math.exp(-DRUG_KA * t) - DRUG_KE * Math.exp(-DRUG_KE * t)),
  hess: (t: number) =>
    -DRUG_D *
    (DRUG_KE * DRUG_KE * Math.exp(-DRUG_KE * t) - DRUG_KA * DRUG_KA * Math.exp(-DRUG_KA * t)),
  dim: 1,
  domain: [0.0, 24.0],
  bracket: [0.0, 12.0],
  x0: 1.0,
  minima: [DRUG_T_MAX],
  description:
    "Plasma concentration after an oral dose (one-compartment model with first-order absorption). The peak is at t⋆ = ln(kₐ/kₑ)/(kₐ − kₑ) = ln 5/0.8 ≈ 2.0118 h with C ≈ 5.3499 mg/L. −C is convex only for t < 2t⋆ ≈ 4.02 h; beyond that Newton's step points the wrong way. (A nod to the legacy 'drug_effectiveness' example.)",
  tags: ['smooth', 'unimodal', 'application'],
});

/** Every scalar_min problem in Python registry order. */
export const SCALAR_PROBLEMS: readonly ScalarProblem[] = [
  quadratic1d,
  quartic1d,
  sin1d,
  xLogX,
  absShifted,
  multimodal1d,
  flatValley,
  rational1d,
  drugConcentration,
];
