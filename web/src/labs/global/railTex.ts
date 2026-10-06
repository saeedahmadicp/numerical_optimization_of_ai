/**
 * The rail formula of a problem written in vector form. The registry's `latex` is the Python
 * metadata and writes the argument as a plain x; the brand sets vectors bold (𝐱) and keeps x, y
 * for the coordinates of the plane, so the n-dimensional definitions are restated here.
 */
const VECTOR_TEX: Record<string, string> = {
  rastrigin: String.raw`f(\mathbf{x}) = 10n + \sum_{i=1}^{n}\left(x_i^2 - 10\cos 2\pi x_i\right),\ n = 2`,
  ackley: String.raw`f(\mathbf{x}) = -20\,e^{-0.2\sqrt{\frac1n\sum x_i^2}} - e^{\frac1n\sum\cos 2\pi x_i} + e + 20`,
  styblinski_tang: String.raw`f(\mathbf{x}) = \tfrac12\sum_{i=1}^{n}\left(x_i^4 - 16x_i^2 + 5x_i\right),\ n = 2`,
  quadratic_ill: String.raw`f(\mathbf{x}) = \tfrac12\, \mathbf{x}^\top Q\,\mathrm{diag}(1, 50)\,Q^\top \mathbf{x},\ Q = \begin{pmatrix}0.8 & -0.6\\ 0.6 & 0.8\end{pmatrix}`,
};

/** The picker entry of a problem, with its vector-form formula where the registry's is plain. */
export function railProblem<P extends { id: string; name: string; latex: string }>(p: P) {
  const latex = VECTOR_TEX[p.id];
  return latex ? { ...p, latex } : p;
}
