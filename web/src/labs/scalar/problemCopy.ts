/**
 * How the lab's rail presents the scalar_min problems. The registered problems keep the Python
 * metadata verbatim (tests/scalar/scalar.test.ts checks it against problems.json); this file only
 * re-typesets it for the page: a display formula that fits the 270 px rail (two aligned rows when
 * the constants make it long), the variable written as x (the stage's axis), ⋆ instead of *,
 * true minus signs and primes, and no developer notes.
 */
import type { ScalarProblem } from '../../problems/scalar_min';

interface Copy {
  latex?: string;
  description?: string;
}

const COPY: Record<string, Copy> = {
  quadratic_1d: {
    description:
      'A convex parabola with minimizer x⋆ = 2 and f⋆ = 1. Parabolic interpolation and ' +
      'Newton’s method are exact on it.',
  },
  quartic_1d: {
    description:
      'Two wells: the global minimizer x ≈ −1.4730 (f ≈ −5.4442) and a local minimizer ' +
      'x ≈ 1.3470 (f ≈ −2.6186), separated by a local maximum at x ≈ 0.1260. f″ < 0 on ' +
      '|x| < √(2/3), where Newton’s step points uphill.',
  },
  sin_1d: {
    description:
      'One period of the sine. The interior minimizer is x⋆ = 3π/2 with f⋆ = −1; f″ = −sin x ' +
      '< 0 on (0, π), where Newton needs its safeguard.',
  },
  x_log_x: {
    description:
      'The entropy-type function x ln x, convex on x > 0 with minimizer x⋆ = 1/e and ' +
      'f⋆ = −1/e. f′ → −∞ at 0, and f is undefined for x < 0, so a long Newton step from ' +
      'x₀ ≥ 1 leaves the domain.',
  },
  abs_shifted: {
    description:
      'Convex but not differentiable at the minimizer x⋆ = 0.3 (f⋆ = 0.045): 0 ∈ ∂f(0.3) = ' +
      '[−0.7, 1.3]. Interval elimination still works; Newton’s method oscillates between −1 and ' +
      '1 because f′ jumps by 2 at the kink.',
  },
  multimodal_1d: {
    description:
      'Three local minima on [2.7, 7.5]; the global one is x ≈ 5.14574 with f ≈ −1.89960. f ' +
      'decreases at 2.7 and increases at 7.5, so every interval method ends at an interior ' +
      'local minimizer, but not necessarily the global one.',
  },
  flat_valley: {
    description:
      'The minimizer x⋆ = 0 is infinitely flat: f and all its derivatives vanish there. In ' +
      'double precision f(x) underflows to exactly 0 for |x| < 0.0367, so no f-comparison ' +
      'locates x⋆ better than that, and f′(x) < 10⁻⁸ already for |x| < 0.2. Newton’s step ' +
      'x − 2x³/(4 − 6x²) converges sublinearly.',
  },
  rational_1d: {
    description:
      'A rational function of the kind a Padé or rational least-squares fit produces. On ' +
      'x > −1 it equals (x + 1) + 6/(x + 1) − 4, strictly convex with minimizer x⋆ = √6 − 1 ≈ ' +
      '1.4495 and f⋆ = 2√6 − 4 ≈ 0.8990. It is asymmetric: steep on the left, nearly linear ' +
      'on the right.',
  },
  drug_concentration: {
    latex:
      '\\begin{aligned} f(x) &= -D\\,\\bigl(e^{-k_e x} - e^{-k_a x}\\bigr)\\\\ ' +
      'D &= 10,\\ \\ k_a = 1,\\ \\ k_e = 0.2 \\end{aligned}',
    description:
      'Minus the plasma concentration C(t) after an oral dose (one-compartment model, ' +
      'first-order absorption); x is the time t in hours. The peak is at x⋆ = ln(kₐ/kₑ)/(kₐ − kₑ) ' +
      '= ln 5 / 0.8 ≈ 2.0118 h, where C ≈ 5.3499 mg/L. f is convex only for x < 2x⋆ ≈ 4.02 h; ' +
      'beyond that Newton’s step points the wrong way.',
  },
};

/** The problem as the rail shows it (same id, name, tags; page typesetting). */
export function presentProblem(p: ScalarProblem): ScalarProblem {
  const c = COPY[p.id];
  if (!c) return p;
  return { ...p, latex: c.latex ?? p.latex, description: c.description ?? p.description };
}
