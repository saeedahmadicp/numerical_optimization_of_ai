import { createElement, type ReactNode } from 'react';
import { Formula } from '../ui/components/Formula';
import type { Step } from '../core/types';
import { sigFixed, vecFixed } from '../core/format';

export interface Column {
  key: string;
  label: ReactNode;
  /** Cell text from a step. */
  value: (s: Step) => ReactNode;
  align?: 'left' | 'right';
  /** CSS grid track, e.g. `56px` or `minmax(96px, 1fr)`. */
  width?: string;
}

const tex = (t: string) => createElement(Formula, { tex: t });

/** Common columns: k, 𝐱ₖ, f(𝐱ₖ), ‖∇f(𝐱ₖ)‖, αₖ — headers in KaTeX, values in tabular mono. */
export const defaultColumns: Column[] = [
  { key: 'k', label: tex('k'), value: (s) => s.k, align: 'right', width: '28px' },
  {
    key: 'x',
    label: tex('\\mathbf{x}_k'),
    width: 'minmax(118px, 1.6fr)',
    value: (s) => (Array.isArray(s.x) ? vecFixed(s.x, 5) : sigFixed(s.x as number, 8)),
  },
  {
    key: 'fun',
    label: tex('f(\\mathbf{x}_k)'),
    value: (s) => sigFixed(s.fun, 4),
    align: 'right',
    width: 'minmax(72px, 1fr)',
  },
  {
    key: 'gradNorm',
    label: tex('\\|\\nabla f\\|'),
    value: (s) => sigFixed(s.gradNorm, 3),
    align: 'right',
    width: 'minmax(60px, 1fr)',
  },
  {
    key: 'stepSize',
    label: tex('\\alpha_k'),
    value: (s) => sigFixed(s.stepSize, 3),
    align: 'right',
    width: 'minmax(48px, 0.7fr)',
  },
];
