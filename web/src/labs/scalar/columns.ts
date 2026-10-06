/**
 * Iteration-table columns per kind of 1-D method: the bracket and its width for elimination
 * methods, the triple and the step type for parabolic interpolation and Brent, f′ and f″ for
 * Newton, the triple (a, b, c) for minimum bracketing. Headers in KaTeX, values in tabular mono.
 */
import { createElement } from 'react';
import { Formula } from '../../ui/components/Formula';
import type { Column } from '../../viz';
import type { Step } from '../../core/types';
import { sigFixed } from '../../core/format';
import { kindOf } from './geometry';

const tex = (t: string) => createElement(Formula, { tex: t });
const n = (v: unknown, d = 7) => sigFixed(typeof v === 'number' ? v : null, d);
const at = (s: Step, key: string, i: number) => {
  const v = s.info[key];
  return Array.isArray(v) ? (v[i] as number) : null;
};
const width = (s: Step) => {
  const b = s.info.bracket as number[] | undefined;
  return b ? b[1] - b[0] : null;
};

const K: Column = { key: 'k', label: tex('k'), value: (s) => s.k, align: 'right', width: '26px' };
const STEP: Column = {
  key: 'step',
  label: 'step',
  value: (s) => String(s.info.step ?? s.info.cut ?? '—').replace('_', ' '),
  width: 'minmax(58px, 0.8fr)',
};
const WIDTH: Column = {
  key: 'width',
  label: tex('b - a'),
  value: (s) => n(width(s), 3),
  align: 'right',
  width: 'minmax(64px, 0.8fr)',
};

const INTERVAL: Column[] = [
  K,
  {
    key: 'a',
    label: tex('a_k'),
    value: (s) => n(at(s, 'bracket', 0)),
    align: 'right',
    width: 'minmax(78px, 1fr)',
  },
  {
    key: 'b',
    label: tex('b_k'),
    value: (s) => n(at(s, 'bracket', 1)),
    align: 'right',
    width: 'minmax(78px, 1fr)',
  },
  WIDTH,
  {
    key: 'cut',
    label: 'cut',
    value: (s) => (s.info.cut === 'left' ? 'left' : s.info.cut === 'right' ? 'right' : '—'),
    width: '40px',
  },
];

const PARABOLIC: Column[] = [
  K,
  {
    key: 'xm',
    label: tex('x_m'),
    value: (s) => n(at(s, 'triple', 1)),
    align: 'right',
    width: 'minmax(78px, 1fr)',
  },
  {
    key: 'u',
    label: tex('u'),
    value: (s) => n(s.info.trial),
    align: 'right',
    width: 'minmax(78px, 1fr)',
  },
  WIDTH,
  STEP,
];

const BRENT: Column[] = [
  K,
  { key: 'x', label: tex('x'), value: (s) => n(s.x), align: 'right', width: 'minmax(78px, 1fr)' },
  {
    key: 'u',
    label: tex('u'),
    value: (s) => n(s.info.trial),
    align: 'right',
    width: 'minmax(78px, 1fr)',
  },
  WIDTH,
  STEP,
];

const NEWTON: Column[] = [
  K,
  {
    key: 'x',
    label: tex('x_k'),
    value: (s) => n(s.x, 9),
    align: 'right',
    width: 'minmax(92px, 1.2fr)',
  },
  {
    key: 'g',
    label: tex("f'(x_k)"),
    value: (s) => n(s.info.g, 3),
    align: 'right',
    width: 'minmax(62px, 0.8fr)',
  },
  {
    key: 'h',
    label: tex("f''(x_k)"),
    value: (s) => n(s.info.h, 3),
    align: 'right',
    width: 'minmax(62px, 0.8fr)',
  },
  STEP,
];

const BRACKETING: Column[] = [
  K,
  {
    key: 'a',
    label: tex('a'),
    value: (s) => n(at(s, 'triple', 0), 5),
    align: 'right',
    width: 'minmax(60px, 1fr)',
  },
  {
    key: 'b',
    label: tex('b'),
    value: (s) => n(at(s, 'triple', 1), 5),
    align: 'right',
    width: 'minmax(60px, 1fr)',
  },
  {
    key: 'c',
    label: tex('c'),
    value: (s) => n(at(s, 'triple', 2), 5),
    align: 'right',
    width: 'minmax(60px, 1fr)',
  },
  STEP,
];

export function columnsFor(methodId: string): Column[] {
  switch (kindOf(methodId)) {
    case 'interval':
      return INTERVAL;
    case 'parabolic':
      return PARABOLIC;
    case 'brent':
      return BRENT;
    case 'newton':
      return NEWTON;
    default:
      return BRACKETING;
  }
}
