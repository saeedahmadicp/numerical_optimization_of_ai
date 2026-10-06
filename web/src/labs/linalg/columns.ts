/** Iteration-table columns of the linear-systems lab (headers in KaTeX, values in tabular mono). */
import { createElement } from 'react';
import { Formula } from '../../ui/components/Formula';
import type { Column } from '../../viz';
import { sigFixed, superscript, vecFixed } from '../../core/format';
import type { Step } from '../../core/types';

const tex = (t: string) => createElement(Formula, { tex: t });

/**
 * Compact numbers for narrow columns: 3 significant digits, `5.50×10⁻⁶` in scientific notation
 * (the notation of the card and the tooltips; tabular figures keep the column aligned).
 */
export function num3(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  if (!Number.isFinite(v)) return v > 0 ? '∞' : '−∞';
  const a = Math.abs(v);
  if (v !== 0 && (a < 1e-3 || a >= 1e5)) {
    const [m, e] = v.toExponential(2).split('e');
    return `${m.replace('-', '−')}×10${superscript(Number(e))}`;
  }
  return sigFixed(v, 3);
}

function xCell(s: Step): string {
  if (!Array.isArray(s.x)) return '—';
  const x = s.x as number[];
  if (x.length <= 2) return vecFixed(x, 4);
  return `${vecFixed(x.slice(0, 2), 3).replace(/\)$/, '')}, …)`;
}

export const ITERATIVE_COLUMNS: Column[] = [
  { key: 'k', label: tex('k'), value: (s) => s.k, align: 'right', width: '30px' },
  { key: 'x', label: tex('\\mathbf{x}_k'), value: xCell, width: 'minmax(96px, 1fr)' },
  {
    key: 'r',
    label: tex('\\|\\mathbf{r}_k\\|_2'),
    value: (s) => num3(s.fun),
    align: 'right',
    width: '74px',
  },
  {
    key: 'rel',
    label: tex('\\|\\mathbf{r}_k\\|/d'),
    value: (s) => num3(s.info.relative_residual as number | null),
    align: 'right',
    width: '74px',
  },
  {
    key: 'dx',
    label: tex('\\|\\Delta\\mathbf{x}\\|'),
    value: (s) => num3(s.stepSize),
    align: 'right',
    width: '74px',
  },
];

const PHASE: Record<string, string> = {
  start: 'start',
  eliminate: 'stage',
  back_substitution: 'back subst.',
  solve: 'solve',
};

export const DIRECT_COLUMNS: Column[] = [
  { key: 'k', label: tex('k'), value: (s) => s.k, align: 'right', width: '30px' },
  {
    key: 'phase',
    label: 'phase',
    value: (s) => {
      const p = String(s.info.phase ?? '');
      const piv = s.info.pivot as number[] | null;
      return p === 'eliminate' && piv ? `stage ${piv[1] + 1}` : (PHASE[p] ?? p);
    },
    width: 'minmax(78px, 1fr)',
  },
  {
    key: 'pivot',
    label: 'pivot',
    value: (s) =>
      s.info.zero_pivot
        ? `${sigFixed(s.info.pivot_value as number, 3)} ✗`
        : sigFixed(s.info.pivot_value as number | null, 4),
    align: 'right',
    width: 'minmax(70px, 1fr)',
  },
  {
    key: 'swap',
    label: 'swap',
    value: (s) => {
      const w = s.info.row_swap as number[] | null;
      return w ? `R${w[0] + 1} ↔ R${w[1] + 1}` : '—';
    },
    width: 'minmax(64px, 0.8fr)',
  },
  {
    key: 'r',
    label: tex('\\|\\mathbf{b} - A\\mathbf{x}\\|_2'),
    value: (s) => num3(s.fun),
    align: 'right',
    width: 'minmax(80px, 1fr)',
  },
];
