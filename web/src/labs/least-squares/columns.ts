/** Iteration-table columns of the least-squares lab (headers in KaTeX, values in tabular mono). */
import { createElement } from 'react';
import { Formula } from '../../ui/components/Formula';
import { MINUS, sig, sigFixed } from '../../core/format';
import { defaultColumns, type Column } from '../../viz';
import type { Step } from '../../core/types';
import type { MethodKind } from './models';

const tex = (t: string) => createElement(Formula, { tex: t });
const [kCol] = defaultColumns;

/**
 * A short number for a narrow cell: `digits` significant figures, plain inside [10⁻², 10^digits),
 * else a 2-digit mantissa and exponent (`4.7e−4`, `−1.2e44`). At most 7 characters, so μₖ and
 * ϱₖ (with its ✓/×) fit a 380 px panel.
 */
export function compactNum(x: number | null | undefined, digits = 2): string {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  if (!Number.isFinite(x)) return x > 0 ? '∞' : `${MINUS}∞`;
  if (x === 0) return '0';
  const a = Math.abs(x);
  if (a >= 1e-2 && a < 10 ** digits) {
    const s = String(Number(x.toPrecision(digits)));
    return s.replace(/^-/, MINUS);
  }
  const [mant, exp] = x.toExponential(1).split('e');
  const e = Number(exp);
  return `${mant.replace(/\.0$/, '').replace(/^-/, MINUS)}e${e < 0 ? MINUS : ''}${Math.abs(e)}`;
}

/** 𝐱ₖ with 4 significant digits (the panel is 380 px wide). */
const xCol: Column = {
  key: 'x',
  label: tex('\\mathbf{x}_k'),
  width: 'minmax(112px, 1.6fr)',
  value: (s) => {
    const x = s.x as number[];
    return `(${x.map((v) => sig(v, 4)).join(', ')})`;
  },
};
const fCol: Column = {
  key: 'fun',
  label: tex('f(\\mathbf{x}_k)'),
  value: (s) => sigFixed(s.fun, 3),
  align: 'right',
  width: 'minmax(56px, 1fr)',
};
/** The step that leaves 𝐱ₖ (the lab passes rows with `info.next` = Step k + 1's info). */
const next = (s: Step, key: string): unknown =>
  (s.info.next as Record<string, unknown> | null | undefined)?.[key] ?? null;
const num = (v: unknown, d = 3) => (typeof v === 'number' ? compactNum(v, d) : '—');

/**
 * k, 𝐱ₖ, f (‖∇f‖ is in the card and the chart), then αₖ (Gauss–Newton) or μₖ (Levenberg–
 * Marquardt), and ϱₖ. For Levenberg–Marquardt the ϱₖ cell carries the decision: ✓ accepted,
 * × rejected — the accept/reject story is the point of the method, so it never scrolls away.
 */
export function columnsFor(kind: MethodKind): Column[] {
  if (kind === 'gauss_newton')
    return [
      kCol,
      xCol,
      fCol,
      {
        key: 'alpha',
        label: tex('\\alpha_k'),
        value: (s) => num(next(s, 'alpha')),
        align: 'right',
        width: 'minmax(44px, 0.7fr)',
      },
      {
        key: 'gain',
        label: tex('\\varrho_k'),
        value: (s) => num(next(s, 'gain_ratio')),
        align: 'right',
        width: 'minmax(48px, 0.8fr)',
      },
    ];
  return [
    kCol,
    xCol,
    fCol,
    {
      key: 'mu',
      label: tex('\\mu_k'),
      value: (s) => num(next(s, 'lambda'), 2),
      align: 'right',
      width: 'minmax(46px, 0.7fr)',
    },
    {
      key: 'gain',
      label: tex('\\varrho_k\\ \\scriptstyle\\checkmark/\\times'),
      value: (s) => {
        const a = next(s, 'accepted');
        if (a === null) return '—';
        const mark = a === true ? '✓' : '×';
        return `${num(next(s, 'gain_ratio'))} ${mark}`;
      },
      align: 'right',
      width: 'minmax(64px, 1fr)',
    },
  ];
}
