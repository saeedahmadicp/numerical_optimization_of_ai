/**
 * Iteration-table columns of the roots lab: k, xₖ with its correct leading digits in strong
 * ink, f(xₖ), the error of the method's estimate, and the bracket width b − a (bracketing) or
 * the step |xₖ − xₖ₋₁| (open methods).
 *
 * Brent, Chandrupatla and ITP keep a bracket end as their estimate x̂ₖ (`info.best`); for them
 * the error column is |x̂ₖ − x⋆| and a trial point that is not the estimate is marked with a
 * dagger (see estimate.ts).
 */
import { createElement } from 'react';
import type { Step } from '../../core/types';
import type { Column } from '../../viz';
import { Formula } from '../../ui/components/Formula';
import { sci, sig, sigFixed } from '../../core/format';
import { estimateOf, isProbe } from './estimate';
import styles from './RootsLab.module.css';

const texLabel = (t: string) => createElement(Formula, { tex: t });

/** Significant digits shown for xₖ. */
const X_DIGITS = 10;

/**
 * Correct significant digits of x as an approximation of target: ⌊−log₁₀(|x − x⋆| / |x⋆|)⌋,
 * clamped to [0, X_DIGITS]. Counted from the error, not from a shared string prefix, so an
 * iterate just below a round root (0.99999999587 for x⋆ = 1) still has its 8 digits.
 */
export function correctDigits(x: number, target: number | null): number {
  if (target === null || target === 0 || !Number.isFinite(x)) return 0;
  const rel = Math.abs(x - target) / Math.abs(target);
  if (rel === 0) return X_DIGITS;
  if (!Number.isFinite(rel)) return 0;
  return Math.max(0, Math.min(X_DIGITS, Math.floor(-Math.log10(rel))));
}

/** Length of the prefix of a formatted number that holds its first n significant digits. */
export function sigPrefix(s: string, n: number): number {
  if (n <= 0) return 0;
  let seen = 0;
  let started = false;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (c === 'e') return i;
    if (c >= '0' && c <= '9') {
      if (c !== '0') started = true;
      if (started && ++seen === n) return i + 1;
    }
  }
  return s.length;
}

/** xₖ (10 significant digits) with its correct digits in strong ink. */
export function digits(x: number, target: number | null, probe = false) {
  const s = sigFixed(x, X_DIGITS);
  if (target === null || target === 0 || !Number.isFinite(x)) return s;
  const n = correctDigits(x, target);
  const cut = sigPrefix(s, n);
  const title = probe
    ? 'Trial point: the method keeps the other bracket end as its estimate'
    : n > 0
      ? `${n} correct significant digit${n === 1 ? '' : 's'} (strong ink)`
      : undefined;
  return createElement(
    'span',
    { className: styles.digits, title },
    createElement('span', { className: styles.digitsOk }, s.slice(0, cut)),
    s.slice(cut),
    probe
      ? createElement('span', { className: styles.probe, 'aria-label': ' (trial point)' }, '†')
      : null,
  );
}

export function rootColumns(
  target: number | null,
  bracketing: boolean,
  keepsBest = false,
): Column[] {
  return [
    { key: 'k', label: texLabel('k'), value: (s) => s.k, align: 'right', width: '26px' },
    {
      key: 'x',
      label: texLabel('x_k'),
      width: 'minmax(88px, 1.2fr)',
      value: (s) => digits(s.x as number, target, keepsBest && isProbe(s)),
    },
    {
      key: 'fun',
      label: texLabel('f(x_k)'),
      value: (s) => sig(s.fun, 2),
      align: 'right',
      // Fits a 2-digit negative exponent: "−4.1×10⁻¹⁰" (and the others "5.8×10⁻¹¹").
      width: 'minmax(70px, 1.1fr)',
    },
    {
      key: 'err',
      label: texLabel(keepsBest ? '|\\hat x_k - x^\\star|' : '|x_k - x^\\star|'),
      value: (s: Step) => (target === null ? '—' : sci(Math.abs(estimateOf(s) - target), 2)),
      align: 'right',
      width: 'minmax(60px, 1fr)',
    },
    bracketing
      ? {
          key: 'width',
          label: texLabel('b - a'),
          value: (s: Step) => {
            const b = s.info.new_bracket as number[] | undefined;
            return b ? sci(b[1] - b[0], 2) : '—';
          },
          align: 'right',
          width: 'minmax(60px, 0.8fr)',
        }
      : {
          key: 'step',
          label: texLabel('|x_k - x_{k-1}|'),
          value: (s: Step) => (s.stepSize === null ? '—' : sci(s.stepSize, 2)),
          align: 'right',
          width: 'minmax(60px, 0.8fr)',
        },
  ];
}
