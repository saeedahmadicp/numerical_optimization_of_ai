/** Number and wording helpers of the line-search lab (KaTeX numbers, interpolation names). */
import { sig } from '../../core/format';

/**
 * `digits` significant digits, U+2212 minus, ×10ⁿ outside [10⁻³, 10⁵), plain digits inside.
 * (`toPrecision` alone returns exponent form when the exponent reaches `digits`:
 * 1000 → "1.00e+3"; here 1000 → "1000" and 17042 → "17040".)
 */
export function num(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || !Number.isFinite(v) || v === 0) return sig(v, digits);
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e5) return sig(v, digits);
  const str = v.toPrecision(digits);
  // 99999 rounds to 1.00e+5: past the plain range after rounding.
  if (Math.abs(Number(str)) >= 1e5) return sig(Number(str), digits);
  const plain = str.includes('e')
    ? String(Number(str))
    : str.includes('.')
      ? str.replace(/\.?0+$/, '')
      : str;
  return plain.replace(/^-/, '−');
}

/** 4 significant digits (see `num`). */
export const fmtNum = (v: number | null | undefined) => num(v, 4);

/** A vector as `(1.23, −4.5)` (see `num`). */
export const vecNum = (x: readonly number[], digits = 4) =>
  `(${x.map((q) => num(q, digits)).join(', ')})`;

/** A number as KaTeX: `-1.23\times10^{-8}` (KaTeX sets the minus sign). */
export function texNum(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '\\text{—}';
  if (!Number.isFinite(v)) return v > 0 ? '\\infty' : '-\\infty';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e5) {
    const [mant, e] = v.toExponential(2).split('e');
    const mm = mant.replace(/\.?0+$/, '');
    return `${mm === '1' ? '' : mm === '-1' ? '-' : `${mm}\\times`}10^{${Number(e)}}`;
  }
  return String(Number(v.toPrecision(4)));
}

/** "cubic_clamped" → "the cubic fit, clamped into the safeguard interval,". */
export function interpWords(interp: string | null, kind?: 'cubic' | 'quadratic'): string {
  switch (interp) {
    case 'cubic':
      return 'the minimizer of the cubic fit';
    case 'quadratic':
      return 'the minimizer of the quadratic fit';
    case 'cubic_clamped':
      return 'the cubic fit’s minimizer, clamped into the safeguard interval,';
    case 'quadratic_clamped':
      return 'the quadratic fit’s minimizer, clamped into the safeguard interval,';
    case 'bisection':
      return kind
        ? `bisection (the ${kind} fit gave no usable minimizer, or the bracket stalled)`
        : 'bisection';
    default:
      return 'the zoom';
  }
}
