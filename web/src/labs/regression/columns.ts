/** Iteration-table columns per regression method (headers in KaTeX, values in tabular mono). */
import { createElement, Fragment } from 'react';
import { Formula } from '../../ui/components/Formula';
import { MINUS, sigFixed, superscript } from '../../core/format';
import type { Column } from '../../viz';
import type { Kind } from './model';

/** A KaTeX header with a spoken name (the typeset math is hidden from assistive tech). */
const tex = (t: string, spoken: string) =>
  createElement(
    Fragment,
    null,
    createElement('span', { 'aria-hidden': true }, createElement(Formula, { tex: t })),
    createElement('span', { className: 'visually-hidden' }, spoken),
  );

/** Scientific with a fixed 2-digit mantissa (2.0×10⁻⁷, 3.5×10⁻¹²): rows align, none drop digits. */
export function sci2(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  if (!Number.isFinite(v)) return v > 0 ? '∞' : `${MINUS}∞`;
  if (v === 0) return '0';
  const [mant, exp] = v.toExponential(1).split('e');
  const e = Number(exp);
  const m = mant.replace('-', MINUS);
  return e === 0 ? m : `${m}×10${superscript(e)}`;
}
const coef = (j: number): Column => ({
  key: `b${j}`,
  label: tex(`\\beta_{${j}}`, `beta ${j}`),
  value: (s) => (Array.isArray(s.x) ? sigFixed((s.x as number[])[j] ?? null, 6) : '—'),
  align: 'right',
  width: 'minmax(66px, 1fr)',
});
const k: Column = {
  key: 'k',
  label: tex('k', 'k'),
  value: (s) => s.k,
  align: 'right',
  width: '30px',
};
const fun = (t: string, spoken: string): Column => ({
  key: 'fun',
  label: tex(t, spoken),
  value: (s) => sigFixed(s.fun, 6),
  align: 'right',
  width: 'minmax(66px, 1fr)',
});
const dTheta: Column = {
  key: 'dtheta',
  label: tex('\\|\\Delta\\boldsymbol\\theta\\|_\\infty', 'step size, max norm'),
  value: (s) => (s.stepSize === null ? '—' : sci2(s.stepSize)),
  align: 'right',
  width: 'minmax(80px, 0.9fr)',
};

export function columnsFor(kind: Kind, degree: number): Column[] {
  switch (kind) {
    case 'huber':
      return [
        k,
        coef(0),
        coef(1),
        fun('\\sum\\rho_\\delta', 'Huber objective'),
        dTheta,
        {
          key: 'down',
          label: tex('w_i\\!<\\!1', 'down-weighted points'),
          value: (s) => ((s.info.weights as number[]) ?? []).filter((v) => v < 1).length,
          align: 'right',
          width: '44px',
        },
      ];
    case 'lad':
      return [k, coef(0), coef(1), fun('\\sum|r_i|', 'sum of absolute residuals'), dTheta];
    case 'minimax':
      return [
        k,
        coef(0),
        coef(1),
        {
          key: 'h',
          label: tex('h_k', 'leveled error h'),
          value: (s) => sigFixed(s.info.level as number, 5),
          align: 'right',
          width: 'minmax(64px, 1fr)',
        },
        fun('\\max|r_i|', 'largest residual'),
        {
          key: 'ref',
          label: tex('\\text{ref.}', 'reference'),
          value: (s) => ((s.info.reference as number[]) ?? []).join(' '),
          align: 'right',
          width: 'minmax(56px, 0.8fr)',
        },
        {
          key: 'enter',
          label: tex('\\text{in}', 'entering point'),
          value: (s) => (typeof s.info.entering === 'number' ? s.info.entering : '—'),
          align: 'right',
          width: '34px',
        },
      ];
    default: {
      const p = kind === 'ols' || kind === 'theil' ? 2 : degree + 1;
      const shown = Math.min(p, 3);
      return [
        k,
        ...Array.from({ length: shown }, (_, j) => coef(j)),
        kind === 'ridge'
          ? fun('\\text{RSS} + \\lambda\\|\\boldsymbol\\beta_{1:}\\|^2', 'penalized objective')
          : fun('\\sum r_i^2', 'residual sum of squares'),
      ];
    }
  }
}
