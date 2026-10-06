/** Numbers, vectors and matrices as KaTeX source (4 significant digits, ×10ⁿ outside [10⁻³, 10⁵)). */
import type { Matrix } from '../../core/types';

export function texNum(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '\\text{—}';
  if (!Number.isFinite(v)) return v > 0 ? '\\infty' : '-\\infty';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e5) {
    const [mant, exp] = v.toExponential(Math.max(0, digits - 2)).split('e');
    const m = mant.replace(/\.?0+$/, '');
    return `${m === '1' ? '' : m === '-1' ? '-' : `${m}\\times`}10^{${Number(exp)}}`;
  }
  const s = v.toPrecision(digits);
  return s.includes('.') ? s.replace(/\.?0+$/, '') : s;
}

/** A column vector `\begin{pmatrix} a \\ b \end{pmatrix}`. */
export function texCol(v: readonly number[], digits = 4): string {
  return `\\begin{pmatrix} ${v.map((x) => texNum(x, digits)).join(' \\\\ ')} \\end{pmatrix}`;
}

export function texMat(M: Matrix, digits = 4): string {
  return `\\begin{pmatrix} ${M.map((r) => r.map((x) => texNum(x, digits)).join(' & ')).join(' \\\\ ')} \\end{pmatrix}`;
}

/** A point `(a,\, b)`. */
export function texPt(v: readonly number[], digits = 4): string {
  return `(${v.map((x) => texNum(x, digits)).join(',\\, ')})`;
}
