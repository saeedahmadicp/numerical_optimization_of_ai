/**
 * The update rule of each constrained method with the numbers of the current step filled in
 * (MethodCard): the general rule first, then this step's instance, one relation per line.
 */
import type { Step } from '../../core/types';
import { sci, sig, superscript } from '../../core/format';
import { ACTIVE_TOL } from '../../methods/constrained/methods';
import { texNum, texVec } from './geometry';

/** Two significant digits; exact powers of ten as 10ⁿ; ×10ⁿ outside [10⁻³, 10⁴). */
export function shortNum(v: unknown): string {
  if (typeof v !== 'number') return '—';
  if (v === 0 || !Number.isFinite(v)) return sig(v, 2);
  const e = Math.round(Math.log10(Math.abs(v)));
  if (v > 0 && Math.abs(e) >= 2 && Math.abs(v - 10 ** e) <= 1e-12 * v) return `10${superscript(e)}`;
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e4) return sci(v, 2);
  // sig(v, 2) would print 250 as 2.5e+2 (toPrecision); keep every integer digit.
  return sig(v, Math.max(2, Math.floor(Math.log10(a)) + 1));
}

/**
 * The tables' compact form (≤ 7 characters, 8 with a sign): 1.3e−5, 1e−12, 1e7; plain decimals in [10⁻³, 10⁴).
 * Superscript exponents need ~9 characters, more than the Iterations columns hold.
 */
export function compactNum(v: unknown): string {
  if (typeof v !== 'number') return '—';
  if (v === 0 || !Number.isFinite(v)) return sig(v, 2);
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e4) {
    // Two-digit exponents (round-off level, or huge) keep one significant digit: 4e−16.
    const big = Math.abs(Math.floor(Math.log10(a))) >= 10;
    const [mant, ex] = v.toExponential(big ? 0 : 1).split('e');
    return `${mant.replace(/\.0$/, '')}e${Number(ex)}`.replace(/-/g, '−');
  }
  return shortNum(v);
}

/**
 * Status of constraint i at x_k with the methods' own tolerance (Python's ACTIVE_TOL = 10⁻⁶):
 * round-off of 10⁻¹⁵ on an active constraint at a converged point is "active", not "violated".
 */
export function constraintStatus(kind: 'eq' | 'ineq', cv: number): string {
  if (kind === 'eq') return Math.abs(cv) <= ACTIVE_TOL ? 'satisfied' : 'violated';
  return cv > ACTIVE_TOL ? 'violated' : cv >= -ACTIVE_TOL ? 'active' : 'inactive';
}

const X = (k: number) => `\\mathbf{x}_{${k}}`;

/** The instance lines of step k (k ≥ 1) for `method`, or null when there are none. */
export function ruleInstance(method: string, step: Step | undefined): string[] | null {
  if (!step || step.k === 0) return null;
  const k = step.k;
  const info = step.info;
  const x = texVec(step.x as number[]);
  if (!info.from) return null;
  switch (method) {
    case 'projected_gradient':
      return [
        `\\mathbf{u} = ${X(k - 1)} - ${texNum(info.s as number)}\\,\\nabla f(${X(k - 1)}) = ${texVec(info.unprojected as number[])}`,
        `${X(k)} = P_C(\\mathbf{u}) = ${x}`,
      ];
    case 'frank_wolfe': {
      const g = texNum(info.gamma as number);
      // Three lines: γ has up to four digits and would push one long line past the card.
      return [
        `\\mathbf{s}_{${k - 1}} = ${texVec(info.vertex as number[], 3)},\\quad \\gamma_{${k - 1}} = ${g}`,
        `${X(k)} = (1 - ${g})\\,${X(k - 1)} + ${g}\\,\\mathbf{s}_{${k - 1}}`,
        `\\phantom{${X(k)}} = ${x}`,
      ];
    }
    case 'quadratic_penalty':
    case 'augmented_lagrangian':
      return [
        `${X(k)} = ${X(k - 1)} + ${texNum(info.alpha as number)}\\,\\mathbf{p}_{${k - 1}} = ${x}`,
      ];
    case 'log_barrier':
      return [
        `${X(k)} = ${X(k - 1)} + ${texNum(info.alpha as number)}\\,\\Delta\\mathbf{x} = ${x},\\quad t = ${texNum(info.t as number)}`,
      ];
    case 'sqp':
      return [
        `\\mathbf{p}_{${k - 1}} = ${texVec(info.direction as number[])}`,
        `${X(k)} = ${X(k - 1)} + ${texNum(info.alpha as number)}\\,\\mathbf{p}_{${k - 1}} = ${x}`,
      ];
    default:
      return null;
  }
}

const END = '\\end{gathered}';

/** The general rule (the method doc) followed by this step's instance lines. */
export function filledRule(method: string, rule: string, step: Step | undefined): string {
  const lines = ruleInstance(method, step);
  if (!lines) return rule;
  const body = rule.startsWith('\\begin{gathered}')
    ? rule.slice('\\begin{gathered}'.length, rule.lastIndexOf(END))
    : rule;
  return `\\begin{gathered}${body} \\\\[6pt] ${lines.join(' \\\\ ')}${END}`;
}

/**
 * The problem formula on two or three lines for the rail: the objective, then "s.t." and the
 * constraints (the first one alone when the list is long).
 */
export function twoLineLatex(latex: string): string {
  const cut = latex.indexOf('\\quad \\text{s.t.}');
  if (cut < 0) return latex;
  let obj = latex.slice(0, cut).trim();
  if (obj.length > 56) {
    // A long objective: break before the top-level " + " nearest its middle.
    let depth = 0,
      at = -1;
    for (let i = 0; i < obj.length; i++) {
      const ch = obj[i];
      if (ch === '{' || ch === '(') depth++;
      else if (ch === '}' || ch === ')') depth--;
      else if (depth === 0 && obj.startsWith(' + ', i))
        if (at < 0 || Math.abs(i - obj.length / 2) < Math.abs(at - obj.length / 2)) at = i;
    }
    if (at > 0) obj = `${obj.slice(0, at)} \\\\ \\quad + ${obj.slice(at + 3)}`;
  }
  const cons = latex
    .slice(cut + '\\quad \\text{s.t.}'.length)
    .replace(/^\\ /, '')
    .trim();
  const items = cons.split(',\\ ');
  const rest = items.length > 2 ? `\\\\ ${items.slice(1).join(',\\ ')}` : '';
  const head = items.length > 2 ? items[0] : items.join(',\\ ');
  return `\\begin{gathered}${obj} \\\\ \\text{s.t.}\\ ${head} ${rest}\\end{gathered}`;
}
