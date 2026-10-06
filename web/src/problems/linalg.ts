/**
 * Linear systems A x = b — TS port of `numopt.problems.linalg` (src/numopt/problems/linalg.py).
 *
 * Same ids, order, data, solutions, descriptions and tags as Python (checked against
 * src/generated/problems.json by tests/linalg/problems.test.ts). `latex` and `n` are TS-only
 * display fields (the problem picker shows the system as a formula).
 *
 * Tags: spd, symmetric, nonsymmetric, tridiagonal, diagonally_dominant, ill_conditioned,
 * needs_pivoting, jacobi_diverges, singular, 2d (see the Python module docstring).
 */
import type { LinearSystem, Matrix, Vector } from '../core/types';
import { addProblem } from './registry';

export interface LinalgProblem extends LinearSystem {
  /** The system as display LaTeX (TS-only). */
  latex: string;
  /** Dimension n of the square system. */
  n: number;
}

function frozenM(rows: number[][]): Matrix {
  return Object.freeze(rows.map((r) => Object.freeze(r.slice()) as number[])) as Matrix;
}

function frozenV(v: number[] | null): Vector | null {
  return v === null ? null : (Object.freeze(v.slice()) as Vector);
}

/** `\begin{bmatrix}…\end{bmatrix}` of small integer data, with U+2212-free TeX minus. */
function bmatrix(rows: readonly (readonly (number | string)[])[]): string {
  return `\\begin{bmatrix}${rows.map((r) => r.join(' & ')).join(' \\\\ ')}\\end{bmatrix}`;
}

const col = (v: readonly (number | string)[]) => bmatrix(v.map((x) => [x]));

function system(
  p: Omit<LinalgProblem, 'n' | 'A' | 'b' | 'solution'> & {
    A: number[][];
    b: number[];
    solution: number[] | null;
  },
): LinalgProblem {
  return addProblem('linalg', {
    ...p,
    A: frozenM(p.A),
    b: frozenV(p.b) as Vector,
    solution: frozenV(p.solution),
    tags: Object.freeze(p.tags.slice()) as string[],
    n: p.b.length,
  });
}

// Shewchuk (1994), eq. (4): the quadratic form ½xᵀAx − bᵀx has elliptical contours, λ = 2 and 7.
export const spd2x2 = system({
  id: 'spd_2x2',
  name: '2×2 SPD system (Shewchuk)',
  A: [
    [3.0, 2.0],
    [2.0, 6.0],
  ],
  b: [2.0, -8.0],
  solution: [2.0, -2.0],
  description:
    "Shewchuk's 2×2 example: eigenvalues 2 and 7 (κ₂ = 3.5). Its solution minimizes " +
    'φ(x) = ½xᵀAx − bᵀx, so iterates can be drawn on the elliptical contours of φ. ' +
    'ρ(T_J) = √2/3 ≈ 0.471, ρ(T_GS) = 2/9.',
  tags: ['spd', 'symmetric', 'tridiagonal', 'diagonally_dominant', '2d'],
  latex: `${bmatrix([
    [3, 2],
    [2, 6],
  ])}\\mathbf{x} = ${col([2, -8])}`,
});

export const diagDominant3 = system({
  id: 'diag_dominant_3',
  name: '3×3 diagonally dominant system',
  A: [
    [4.0, -1.0, 1.0],
    [-1.0, 4.0, -2.0],
    [1.0, -2.0, 4.0],
  ],
  b: [12.0, -1.0, 5.0],
  solution: [3.0, 1.0, 1.0],
  description:
    'Strictly diagonally dominant and SPD (κ₂ ≈ 3.37). Every method in the family ' +
    'applies; ρ(T_J) ≈ 0.683 and ρ(T_GS) ≈ 0.177.',
  tags: ['spd', 'symmetric', 'diagonally_dominant'],
  latex: `${bmatrix([
    [4, -1, 1],
    [-1, 4, -2],
    [1, -2, 4],
  ])}\\mathbf{x} = ${col([12, -1, 5])}`,
});

// −u″ = f on (0, 1), u(0) = u(1) = 0, central differences with h = 1/11, scaled by h²:
// tridiag(−1, 2, −1) u = h² f. With h² f ≡ 1 the discrete solution is u_i = i(11 − i)/2.
export const poisson1d10 = (() => {
  const n = 10;
  const A = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 2.0 : Math.abs(i - j) === 1 ? -1.0 : 0.0)),
  );
  return system({
    id: 'poisson_1d_10',
    name: '1-D Poisson matrix, n = 10',
    A,
    b: new Array<number>(n).fill(1.0),
    solution: Array.from({ length: n }, (_, k) => ((k + 1) * (n + 1 - (k + 1))) / 2.0),
    description:
      'The scaled second-difference matrix tridiag(−1, 2, −1) of −u″ = 1 on (0, 1), ' +
      'h = 1/11. Eigenvalues 2 − 2cos(jπ/11), κ₂ ≈ 48.4. ρ(T_J) = cos(π/11) ≈ 0.959, ' +
      'so Jacobi is slow; SOR with ω* = 2/(1 + sin(π/11)) ≈ 1.560 is much faster.',
    tags: ['spd', 'symmetric', 'tridiagonal'],
    latex: '\\operatorname{tridiag}(-1,\\,2,\\,-1)\\,\\mathbf{u} = \\mathbf{1},\\quad n = 10',
  });
})();

export const hilbert5 = (() => {
  const n = 5;
  // H_ij = 1/(i + j + 1) (0-based): float(Fraction(1, k)) is the correctly rounded 1/k, which is
  // what IEEE division returns.
  const A = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => 1.0 / (i + j + 1)),
  );
  // b = H·1 summed in exact rational arithmetic, then rounded once to float64 (Python uses
  // Fraction): 137/60, 29/20, 153/140, 743/840, 1879/2520. These literals are those doubles.
  const b = [2.283333333333333, 1.45, 1.0928571428571427, 0.8845238095238095, 0.7456349206349207];
  return system({
    id: 'hilbert_5',
    name: 'Hilbert matrix, n = 5',
    A,
    b,
    solution: new Array<number>(n).fill(1.0),
    description:
      'H_ij = 1/(i + j − 1): SPD but κ₂ ≈ 4.77e5, so about 5–6 of the 16 significant ' +
      'digits are lost. `solution` is the exact solution of the exact rational system; ' +
      'the exact solution of the rounded float system differs from it by about 2e-12 ' +
      '(max-norm; the bound κ·u ≈ 5e-11).',
    tags: ['spd', 'symmetric', 'ill_conditioned'],
    latex: 'H\\mathbf{x} = H\\mathbf{1},\\quad H_{ij} = \\frac{1}{i + j - 1}',
  });
})();

export const needsPivoting = system({
  id: 'needs_pivoting',
  name: 'Zero leading pivot',
  A: [
    [0.0, 2.0, 1.0],
    [1.0, -2.0, -3.0],
    [-1.0, 1.0, 2.0],
  ],
  b: [-8.0, 0.0, 3.0],
  solution: [-4.0, -5.0, 2.0],
  description:
    'det A = 1, but a₁₁ = 0: Gaussian elimination without row interchanges breaks down ' +
    'at the first stage, while partial pivoting solves it. The zero diagonal entry also ' +
    'makes Jacobi and Gauss–Seidel undefined.',
  tags: ['nonsymmetric', 'needs_pivoting'],
  latex: `${bmatrix([
    [0, 2, 1],
    [1, -2, -3],
    [-1, 1, 2],
  ])}\\mathbf{x} = ${col([-8, 0, 3])}`,
});

// δ = 2⁻³⁰ makes 1 + δ and 2 + δ exactly representable, so the stored float system has exactly
// the solution (1, 1).
const DELTA = 2.0 ** -30;
export const nearlySingular = system({
  id: 'nearly_singular',
  name: 'Two nearly parallel lines',
  A: [
    [1.0, 1.0],
    [1.0, 1.0 + DELTA],
  ],
  b: [2.0, 2.0 + DELTA],
  solution: [1.0, 1.0],
  description:
    'x + y = 2 and x + (1 + δ)y = 2 + δ with δ = 2⁻³⁰: two lines that meet at (1, 1) ' +
    'at an angle of about 4.7e-10 rad (≈ δ/2). κ₂ ≈ 4.3e9, so a relative change of 1e-10 in b ' +
    'can move the solution by O(1). ρ(T_J) ≈ 1 − 4.7e-10.',
  tags: ['spd', 'symmetric', 'tridiagonal', 'ill_conditioned', '2d'],
  latex: `\\begin{gathered}${bmatrix([
    [1, 1],
    [1, '1 + \\delta'],
  ])}\\mathbf{x} = ${col([2, '2 + \\delta'])}\\\\ \\delta = 2^{-30}\\end{gathered}`,
});

export const jacobiDiverges = system({
  id: 'jacobi_diverges',
  name: 'Jacobi diverges, Gauss–Seidel converges',
  A: [
    [2.0, -1.0, 1.0],
    [2.0, 2.0, 2.0],
    [-1.0, -1.0, 2.0],
  ],
  b: [-1.0, 4.0, -5.0],
  solution: [1.0, 2.0, -1.0],
  description:
    'A classic splitting example: ρ(T_J) = √5/2 ≈ 1.118 > 1, so Jacobi ' +
    'diverges, but ρ(T_GS) = 1/2, so Gauss–Seidel converges. Not diagonally dominant.',
  tags: ['nonsymmetric', 'jacobi_diverges'],
  latex: `${bmatrix([
    [2, -1, 1],
    [2, 2, 2],
    [-1, -1, 2],
  ])}\\mathbf{x} = ${col([-1, 4, -5])}`,
});

export const nonsymmetric4 = system({
  id: 'nonsymmetric_4',
  name: '4×4 nonsymmetric system',
  A: [
    [4.0, 1.0, 0.0, 2.0],
    [-1.0, 5.0, 2.0, 0.0],
    [0.0, -2.0, 6.0, 1.0],
    [3.0, 0.0, -1.0, 7.0],
  ],
  b: [8.0, 7.0, -9.0, 11.0],
  solution: [1.0, 2.0, -1.0, 1.0],
  description:
    'Nonsymmetric, strictly row diagonally dominant, κ₂ ≈ 3.15, with a complex pair of ' +
    'eigenvalues 5.5 ± 2.1i. CG does not apply; GMRES, LU and QR do.',
  tags: ['nonsymmetric', 'diagonally_dominant'],
  latex: `${bmatrix([
    [4, 1, 0, 2],
    [-1, 5, 2, 0],
    [0, -2, 6, 1],
    [3, 0, -1, 7],
  ])}\\mathbf{x} = ${col([8, 7, -9, 11])}`,
});

export const singular3 = system({
  id: 'singular_3',
  name: 'Singular 3×3 system',
  A: [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
  ],
  b: [6.0, 15.0, 24.0],
  solution: null,
  description:
    'rank A = 2 with null space span{(1, −2, 1)}. b = A·(1, 1, 1) is consistent, so ' +
    'there are infinitely many solutions (1, 1, 1) + t(1, −2, 1) and no unique one. ' +
    'Direct methods must report a zero pivot; the minimum-norm solution is (1, 1, 1).',
  tags: ['nonsymmetric', 'singular'],
  latex: `${bmatrix([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
  ])}\\mathbf{x} = ${col([6, 15, 24])}`,
});

/** Every linalg problem in Python's registration order. */
export const LINALG_PROBLEMS: readonly LinalgProblem[] = [
  spd2x2,
  diagDominant3,
  poisson1d10,
  hilbert5,
  needsPivoting,
  nearlySingular,
  jacobiDiverges,
  nonsymmetric4,
  singular3,
];
