/**
 * Iteration-table columns per kind of method: k, 𝐱ₖ, f(𝐱ₖ), then the quantities that decide
 * the step. Vectors are written as columns (two stacked components between rule parentheses,
 * in rows of ROW_HEIGHT px), so every number keeps 4 significant digits and no cell is cut.
 */
import { createElement, type ReactNode } from 'react';
import { Formula } from '../../ui/components/Formula';
import type { Step } from '../../core/types';
import { sci } from '../../core/format';
import { SciText } from '../../ui/components/Num';
import styles from './UnconstrainedLab.module.css';
import type { Column } from '../../viz';
import type { Kind } from './catalog';

const tex = (t: string) => createElement(Formula, { tex: t });
const num = (x: unknown): number | null => (typeof x === 'number' && Number.isFinite(x) ? x : null);
/**
 * A number for a narrow numeric column: d significant digits in [10⁻², 10⁴), else d-digit
 * scientific notation with a superscript exponent, `1.35×10⁻³` (sci from core/format; render it
 * through SciText so the exponent is a real <sup>). Never e-notation: the brand keeps that for
 * code. The table columns are fixed grid tracks, so the longer ×10ⁿ form never resizes them.
 */
export function short(x: unknown, d = 3): string {
  const v = num(x);
  if (v === null) return typeof x === 'number' && !Number.isNaN(x) ? (x > 0 ? '∞' : '−∞') : '—';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e-2 && a < 1e4) {
    const t = v.toPrecision(d);
    // toPrecision writes 1234 as "1.23e+3": integers from 10ᵈ on are written out.
    return (t.includes('e') ? String(Math.round(v)) : t).replace(/^-/, '−');
  }
  return sci(v, d);
}

const fx = (x: unknown, d = 3) => short(x, d);
const word = (x: unknown) => (typeof x === 'string' ? x.replace(/_/g, ' ') : '—');

/** Row height of the iteration table: two lines of 11 px mono for the column vectors. */
export const ROW_HEIGHT = 36;

/** A 2-vector as a column: its components stacked, right-aligned, between parentheses. */
function colVec(x: unknown, d = 4) {
  if (!Array.isArray(x)) return '—';
  return createElement(
    'span',
    { className: styles.colVec },
    ...(x as unknown[]).map((c, i) =>
      createElement('span', { key: i }, createElement(SciText, { text: short(c, d) })),
    ),
  );
}

const K: Column = { key: 'k', label: tex('k'), value: (s) => s.k, align: 'right', width: '30px' };
const XK: Column = {
  key: 'x',
  label: tex('\\mathbf{x}_k'),
  width: 'minmax(86px, 1.4fr)',
  value: (s) => colVec(s.x),
};
const F: Column = {
  key: 'fun',
  label: tex('f(\\mathbf{x}_k)'),
  value: (s) => short(s.fun, 4),
  align: 'right',
  width: 'minmax(66px, 1fr)',
};
const GN: Column = {
  key: 'gradNorm',
  label: tex('\\|\\nabla f\\|_2'),
  value: (s) => short(s.gradNorm, 3),
  align: 'right',
  width: 'minmax(58px, 0.9fr)',
};

function col(
  key: string,
  label: string,
  value: (s: Step) => ReactNode,
  width = 'minmax(58px, 0.9fr)',
  align: 'left' | 'right' = 'right',
): Column {
  return { key, label: tex(label), value, width, align };
}

const ALPHA = col('alpha', '\\alpha_k', (s) => fx(s.info.alpha));
const vnorm = (x: unknown) =>
  Array.isArray(x) && x.length === 2 ? Math.hypot(x[0] as number, x[1] as number) : null;

/**
 * At most five columns, so the table fits the 380 px insights column without scrolling
 * sideways (‖∇f‖ is always in the convergence chart; it stays here for the first-order methods,
 * whose stopping test it is and whose step it sets).
 */
const EXTRA: Record<Kind, Column[]> = {
  line: [GN, ALPHA],
  coord: [
    GN,
    col(
      'i',
      'i',
      (s) => (num(s.info.coordinate) === null ? '—' : String((s.info.coordinate as number) + 1)),
      '30px',
    ),
  ],
  heavy: [GN, col('v', '\\|\\mathbf{v}_k\\|', (s) => fx(vnorm(s.info.velocity)))],
  nesterov: [GN, col('v', '\\|\\mathbf{v}_k\\|', (s) => fx(vnorm(s.info.velocity)))],
  adaptive: [col('D', '\\mathbf{D}_k', (s) => colVec(s.info.lr_eff, 3), 'minmax(78px, 1fr)')],
  newton: [
    ALPHA,
    col('lmin', '\\lambda_{\\min}', (s) =>
      Array.isArray(s.info.hess_eigs) ? fx(s.info.hess_eigs[0]) : '—',
    ),
  ],
  qn: [ALPHA, col('ys', '\\mathbf{y}^{\\top}\\mathbf{s}', (s) => fx(s.info.curvature))],
  cg: [
    ALPHA,
    // β = 0 on a restart: the cell names the restart instead.
    col(
      'beta',
      '\\beta_k',
      (s) => (typeof s.info.restart === 'string' ? 'restart' : fx(s.info.beta)),
      'minmax(62px, 0.9fr)',
    ),
  ],
  tr: [
    col('radius', '\\Delta_k', (s) => fx(s.info.radius)),
    // ρ of a rejected step is marked ✗ (the iterate repeats in the next row).
    col(
      'rho',
      '\\rho_k',
      (s) => `${fx(s.info.rho)}${s.info.accepted === false ? ' ✗' : ''}`,
      'minmax(62px, 0.9fr)',
    ),
  ],
  nm: [
    col('op', '\\text{operation}', (s) => word(s.info.operation), 'minmax(96px, 1.2fr)', 'left'),
  ],
  powell: [
    col('df', '\\Delta f_{\\max}', (s) => fx(s.info.largest_decrease)),
    col(
      'rep',
      '\\text{repl.}',
      (s) => (num(s.info.replaced) === null ? '—' : `u${(s.info.replaced as number) + 1}`),
      '48px',
      'left',
    ),
  ],
  hj: [col('out', '\\text{outcome}', (s) => word(s.info.outcome), 'minmax(104px, 1.2fr)', 'left')],
  // A certified checkpoint (the theorem's bound holds at this k) is marked ✓.
  schedule: [
    GN,
    col(
      'h',
      'h_{k-1}',
      (s) =>
        `${fx(s.info.h)}${typeof s.info.bound_f === 'number' || typeof s.info.bound_dist === 'number' ? ' ✓' : ''}`,
      'minmax(62px, 0.9fr)',
    ),
  ],
  ogm: [GN, col('theta', '\\theta_k', (s) => fx(s.info.theta))],
  // β of the step that formed 𝐲ₖ; "restart" where the momentum was reset at 𝐱ₖ.
  fista: [
    col('gy', '\\|\\nabla f(\\mathbf{y}_k)\\|', (s) => fx(s.info.grad_norm_y), 'minmax(66px, 1fr)'),
    col(
      'beta',
      '\\beta_k',
      (s) => (s.info.restarted === true ? 'restart' : fx(s.info.beta)),
      'minmax(62px, 0.9fr)',
    ),
  ],
  anderson: [
    GN,
    col('m', 'm_k', (s) => (num(s.info.memory) === null ? '—' : String(s.info.memory)), '40px'),
  ],
  arc: [
    col('sigma', '\\sigma_k', (s) => fx(s.info.sigma)),
    // ρ of a rejected step is marked ✗ (the iterate repeats in the next row).
    col(
      'rho',
      '\\rho_k',
      (s) => `${fx(s.info.rho)}${s.info.accepted === false ? ' ✗' : ''}`,
      'minmax(62px, 0.9fr)',
    ),
  ],
  regnewton: [
    col('lambda', '\\lambda_k', (s) => fx(s.info.lambda)),
    col(
      'trials',
      '\\text{trials}',
      (s) => (num(s.info.inner_iters) === null ? '—' : String(s.info.inner_iters)),
      '48px',
    ),
  ],
  compass: [
    col('d', '\\Delta_k', (s) => fx(s.info.step)),
    col(
      'ok',
      '\\text{decrease}',
      (s) => (s.info.success === true ? 'yes' : s.info.success === false ? 'no' : '—'),
      '56px',
      'left',
    ),
  ],
};

export function columnsFor(kind: Kind): Column[] {
  return [K, XK, F, ...EXTRA[kind]];
}
