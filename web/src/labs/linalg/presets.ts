/**
 * "Try this": curated views of the linear-systems lab, each revealing one phenomenon.
 *
 * Titles are plain text (vectors in Unicode bold, 𝐱⋆); notes may typeset TeX between `$…$`, but
 * name spectral radii in words rather than with `G_J` subscripts (tests/linalg/lab-review.test.ts).
 * Every number in a note is what the lab itself shows for that preset (tests/linalg/lab.test.ts
 * checks the counts).
 */
import type { LabPreset } from '../_shell';
import type { MethodSelection } from '../_shell';

/** Young's optimal ω for tridiag(−1, 2, −1), n = 10: 2/(1 + sin(π/11)) = 1.56039… (exact). */
export const POISSON_OMEGA_OPT = 2 / (1 + Math.sin(Math.PI / 11));

/** First view: three rates on Shewchuk's 2 × 2 system — ρ_J = 0.471, ρ_GS = 2/9, CG in 2 steps. */
export const DEFAULT_SELECTION: MethodSelection[] = [
  { id: 'jacobi', slot: 0, params: {} },
  { id: 'gauss_seidel', slot: 1, params: {} },
  { id: 'conjugate_gradient_linear', slot: 2, params: {} },
];

export const PRESETS: LabPreset[] = [
  {
    id: 'cg-two-steps',
    title: 'Conjugate gradient reaches 𝐱⋆ in n = 2 steps',
    note: 'Steepest descent zigzags between the same ellipses; the second conjugate-gradient direction is $A$-conjugate to the first and points at the center.',
    problem: 'spd_2x2',
    methods: [
      { id: 'steepest_descent_linear', slot: 0, params: {} },
      { id: 'conjugate_gradient_linear', slot: 2, params: {} },
    ],
    start: [-2, 2],
    extra: { v: 'iterates' },
  },
  {
    id: 'jacobi-diverges',
    title: 'Jacobi diverges, Gauss–Seidel converges',
    note: "Jacobi's iteration matrix has spectral radius √5/2 ≈ 1.118, so every sweep grows the error; Gauss–Seidel's has 1/2 and halves it.",
    problem: 'jacobi_diverges',
    methods: [
      { id: 'jacobi', slot: 0, params: { max_iter: 60 } },
      { id: 'gauss_seidel', slot: 1, params: {} },
    ],
    extra: { v: 'iterates' },
  },
  {
    id: 'young',
    title: "Young's ω⋆ on the Poisson matrix",
    note: 'Gauss–Seidel contracts the error by cos²(π/11) ≈ 0.921 per sweep; SOR at ω⋆ ≈ 1.560 by ω⋆ − 1 ≈ 0.560. SOR needs 49 sweeps, Gauss–Seidel 279, Jacobi 556. Drag ω on the curve.',
    problem: 'poisson_1d_10',
    methods: [
      { id: 'jacobi', slot: 0, params: {} },
      { id: 'gauss_seidel', slot: 1, params: {} },
      { id: 'sor', slot: 3, params: { omega: POISSON_OMEGA_OPT } },
    ],
    extra: { v: 'iterates' },
  },
  {
    id: 'zero-pivot',
    title: 'A zero pivot stops elimination; a row swap fixes it',
    note: 'a₁₁ = 0, although det A = 1. Partial pivoting swaps up the largest entry of the column and goes on.',
    problem: 'needs_pivoting',
    methods: [
      { id: 'gaussian_elimination', slot: 0, params: {} },
      { id: 'gaussian_elimination_pivoting', slot: 1, params: {} },
    ],
    extra: { v: 'matrix' },
  },
];
