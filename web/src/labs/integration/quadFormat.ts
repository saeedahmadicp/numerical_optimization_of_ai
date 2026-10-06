/**
 * Number formatting for the quadrature lab: TeX numbers for the filled-in update rules, and the
 * split of an estimate into the digits that agree with the exact integral and those that do not.
 */
import { sig } from '../../core/format';

/** A number as TeX: 4–6 significant digits, ×10ⁿ outside [10⁻³, 10⁵), a real minus sign. */
export function texNum(v: number | null | undefined, digits = 5): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '\\text{—}';
  if (!Number.isFinite(v)) return v > 0 ? '\\infty' : '-\\infty';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e5) {
    const [m, e] = v.toExponential(Math.max(0, digits - 1)).split('e');
    const mant = m.includes('.') ? m.replace(/\.?0+$/, '') : m;
    return `${mant === '1' ? '' : mant === '-1' ? '-' : `${mant}\\times `}10^{${Number(e)}}`;
  }
  return sig(v, digits).replace('−', '-');
}

/**
 * The estimate written to `digits` significant figures, split where it stops agreeing with the
 * exact value (same formatting for both). `good` are the leading characters both share.
 */
export function agreeingDigits(
  estimate: number,
  exact: number | null,
  digits = 12,
): { good: string; rest: string } {
  const fmt = (v: number) => {
    if (!Number.isFinite(v)) return sig(v);
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e5)) return sig(v, digits);
    return v.toFixed(Math.max(0, digits - 1 - Math.floor(Math.log10(a || 1)))).replace('-', '−');
  };
  const e = fmt(estimate);
  if (exact === null || !Number.isFinite(estimate)) return { good: '', rest: e };
  const x = fmt(exact);
  let i = 0;
  while (i < e.length && i < x.length && e[i] === x[i]) i++;
  // Never end the agreeing part on the sign or the decimal point alone.
  const good = e.slice(0, i).replace(/[−.]$/, '');
  return { good, rest: e.slice(good.length) };
}

/** Correct significant digits −log₁₀(|error| / |I|), clamped to [0, 16]; null if unknown. */
export function correctDigits(
  error: number | null | undefined,
  exact: number | null,
): number | null {
  if (error === null || error === undefined || exact === null) return null;
  if (error === 0) return 16;
  const scale = Math.max(Math.abs(exact), 1e-300);
  return Math.max(0, Math.min(16, -Math.log10(error / scale)));
}
