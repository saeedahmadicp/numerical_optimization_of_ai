/** Iteration-table columns per LP method (headers in KaTeX, values in tabular mono). */
import { createElement } from 'react';
import { Formula } from '../../ui/components/Formula';
import type { Column } from '../../viz/columns';
import type { Step } from '../../core/types';
import { sigFixed } from '../../core/format';
import { fmt, varText } from './explain';

const tex = (t: string) => createElement(Formula, { tex: t });
const num = (v: unknown, d = 4) => sigFixed(typeof v === 'number' ? v : null, d);

const K: Column = { key: 'k', label: tex('k'), value: (s) => s.k, align: 'right', width: '28px' };
const X: Column = {
  key: 'x',
  label: tex('\\mathbf{x}_k'),
  width: 'minmax(112px, 1.8fr)',
  value: (s) =>
    Array.isArray(s.x) ? `(${(s.x as number[]).map((v) => fmt(v, 4)).join(', ')})` : '—',
};
const OBJ: Column = {
  key: 'fun',
  label: tex('\\mathbf{c}^{\\top}\\mathbf{x}_k'),
  value: (s) => num(s.fun, 5),
  align: 'right',
  width: 'minmax(64px, 1fr)',
};

const label = (s: Step, key: 'entering' | 'leaving') => {
  const j = s.info[key] as number | null;
  const labels = s.info.col_labels as string[] | undefined;
  return j === null || j === undefined || !labels ? '—' : varText(labels[j]);
};

export function columnsFor(methodId: string): Column[] {
  switch (methodId) {
    case 'simplex':
    case 'two_phase_simplex':
    case 'big_m':
    case 'revised_simplex':
    case 'dual_simplex':
      return [
        K,
        X,
        OBJ,
        {
          key: 'in',
          label: 'in',
          value: (s) => label(s, 'entering'),
          align: 'right',
          width: '36px',
        },
        {
          key: 'out',
          label: 'out',
          value: (s) => label(s, 'leaving'),
          align: 'right',
          width: '36px',
        },
        {
          key: 'theta',
          label: tex(methodId === 'dual_simplex' ? '\\delta_k' : '\\theta_k'),
          value: (s) => num(s.stepSize, 3),
          align: 'right',
          width: 'minmax(48px, 0.8fr)',
        },
      ];
    case 'primal_dual_ipm':
      return [
        K,
        X,
        OBJ,
        {
          key: 'mu',
          label: tex('\\mu_k'),
          value: (s) => num(s.info.mu, 3),
          align: 'right',
          width: 'minmax(56px, 0.9fr)',
        },
        {
          key: 'alpha',
          label: tex('\\alpha_k'),
          value: (s) => num(s.info.alpha_primal, 3),
          align: 'right',
          width: 'minmax(48px, 0.8fr)',
        },
      ];
    case 'affine_scaling':
      return [
        K,
        X,
        OBJ,
        {
          key: 'alpha',
          label: tex('\\alpha_k'),
          value: (s) => num(s.info.alpha, 3),
          align: 'right',
          width: 'minmax(52px, 0.8fr)',
        },
        {
          key: 't',
          label: tex('t_k'),
          value: (s) => num(s.info.artificial, 3),
          align: 'right',
          width: 'minmax(52px, 0.8fr)',
        },
      ];
    case 'restarted_pdhg':
      return [
        K,
        X,
        OBJ,
        {
          key: 'kkt',
          label: tex('\\text{KKT}_k'),
          value: (s) => num(s.info.kkt, 3),
          align: 'right',
          width: 'minmax(60px, 0.9fr)',
        },
        {
          key: 'epoch',
          label: tex('n'),
          value: (s) => `${String(s.info.epoch)}${s.info.restarted ? ' ↻' : ''}`,
          align: 'right',
          width: '44px',
        },
      ];
    case 'branch_and_bound':
      return [
        K,
        {
          key: 'node',
          label: 'node',
          value: (s) => String(s.info.node),
          align: 'right',
          width: '40px',
        },
        X,
        {
          key: 'inc',
          label: tex('z^{\\text{inc}}'),
          value: (s) => num(s.info.incumbent_value, 4),
          align: 'right',
          width: 'minmax(56px, 0.9fr)',
        },
        {
          key: 'bound',
          label: tex('\\bar z'),
          value: (s) => num(s.info.best_bound, 4),
          align: 'right',
          width: 'minmax(56px, 0.9fr)',
        },
      ];
    case 'gomory_cuts':
      return [
        K,
        X,
        OBJ,
        {
          key: 'cuts',
          label: 'cuts',
          value: (s) => String((s.info.cuts as unknown[] | undefined)?.length ?? 0),
          align: 'right',
          width: '40px',
        },
        {
          key: 'f0',
          label: tex('f(\\bar b_r)'),
          value: (s) => num((s.info.cut as { f0: number } | null)?.f0, 3),
          align: 'right',
          width: 'minmax(52px, 0.8fr)',
        },
      ];
    default:
      return [K, X, OBJ];
  }
}
