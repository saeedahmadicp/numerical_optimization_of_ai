/** Number formatting shared by the lab's views (plain text, mono cells and KaTeX). */
import { sci, sig } from '../../core/format';

/** A matrix entry: exact zeros as "0", integers as such, else 4 significant digits. */
export function cellText(v: number): string {
  if (Number.isNaN(v)) return '·';
  if (v === 0) return '0';
  if (Number.isInteger(v) && Math.abs(v) < 1e5) return String(v).replace('-', '−');
  return sig(v, 4);
}

export const fmt = (v: number | null | undefined) =>
  v === null || v === undefined
    ? '—'
    : Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5
      ? sci(v, 3)
      : sig(v, 4);

/** TeX for a number (U+2212-free: KaTeX typesets the minus). */
export function texNum(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '\\text{—}';
  if (!Number.isFinite(v)) return v > 0 ? '\\infty' : '-\\infty';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e5) {
    const e = Math.floor(Math.log10(a));
    const m = +(v / 10 ** e).toPrecision(Math.max(1, digits - 1));
    return `${m}\\times 10^{${e}}`;
  }
  return String(+v.toPrecision(digits));
}

/**
 * A value within 5×10⁻⁴ of 1 or 2 (but not equal) as "1 − 4.7×10⁻¹⁰": ρ(G_J) on a nearly
 * singular matrix, or Young's ω⋆ near 2, must not round to the very number the problem is about.
 * Returns null for any other value.
 */
function nearRef(v: number): { ref: number; d: number } | null {
  if (!Number.isFinite(v)) return null;
  for (const ref of [1, 2]) {
    const d = v - ref;
    if (d !== 0 && Math.abs(d) < 5e-4) return { ref, d };
  }
  return null;
}

/** Plain text: `1 − 4.7×10⁻¹⁰` near 1 or 2, else `fallback(v)`. */
export function nearText(v: number, fallback: (v: number) => string): string {
  const q = nearRef(v);
  return q ? `${q.ref} ${q.d < 0 ? '−' : '+'} ${sci(Math.abs(q.d), 2)}` : fallback(v);
}

/** TeX: `1 - 4.7\times 10^{-10}` near 1 or 2, else `fallback(v)`. */
export function nearTex(v: number, fallback: (v: number) => string): string {
  const q = nearRef(v);
  return q ? `${q.ref} ${q.d < 0 ? '-' : '+'} ${texNum(Math.abs(q.d), 3)}` : fallback(v);
}

/**
 * The pivot of stage c (0-based) in the notation of each method's rule: the working matrix W of
 * LU and Cholesky, R of Householder QR, Thomas's w_i, and a_cc of the eliminations.
 */
export function pivotSymbol(method: string, c: number): string {
  const ci = c + 1;
  if (method === 'cholesky' || method === 'lu_decomposition') return `w_{${ci}${ci}}`;
  if (method === 'qr_householder') return `r_{${ci}${ci}}`;
  if (method === 'thomas') return `w_{${ci}}`;
  return `a_{${ci}${ci}}`;
}

/**
 * Thomas, stage i, with A = tridiag(a, δ, c) and right-hand side b: the first stage has no
 * a₁, c′₀ or d′₀, and the last has no c′ₙ.
 */
export function thomasStageTex(i: number, n: number, p: string): string {
  const w =
    i === 1
      ? `w_{1} = \\delta_{1} = ${p}`
      : `w_{${i}} = \\delta_{${i}} - a_{${i}}c'_{${i - 1}} = ${p}`;
  const cp = i < n ? `,\\quad c'_{${i}} = c_{${i}}/w_{${i}}` : '';
  const dp =
    i === 1 ? `d'_{1} = b_{1}/w_{1}` : `d'_{${i}} = (b_{${i}} - a_{${i}}d'_{${i - 1}})/w_{${i}}`;
  return `${w}${cp},\\quad ${dp}`;
}

/** Split `text` into alternating prose and TeX parts (odd indices are TeX). */
export function splitMath(text: string): string[] {
  return text.split(/(?<!\\)\$/);
}
