/** Pure helpers for the method catalog: rate classes, what a method needs, filters (tested). */
import type { CatalogMethod } from '../app/catalogTypes';
import { normalize } from '../app/searchIndex';

export type RateClass =
  | 'quadratic'
  | 'superlinear'
  | 'linear'
  | 'sublinear'
  | 'finite'
  | 'truncation'
  | 'approximation'
  | 'none';

export const RATE_LABEL: Record<RateClass, string> = {
  quadratic: 'Quadratic or faster',
  superlinear: 'Superlinear',
  linear: 'Linear',
  sublinear: 'Sublinear',
  finite: 'Finite or direct',
  truncation: 'Error order in h',
  approximation: 'Approximation order',
  none: 'No rate guarantee',
};

export const RATE_ORDER: RateClass[] = [
  'quadratic',
  'superlinear',
  'linear',
  'sublinear',
  'finite',
  'truncation',
  'approximation',
  'none',
];

/** Classify the free-text `order` of the registry by its leading claim. */
export function rateClass(order: string): RateClass {
  const o = order.trim().toLowerCase();
  if (!o) return 'none';
  if (/^(quadratic|cubic)/.test(o)) return 'quadratic';
  if (/^(r-)?superlinear/.test(o)) return 'superlinear';
  if (/^sublinear/.test(o)) return 'sublinear';
  // Accuracy in the number of nodes or samples (quadrature and interpolation), tested before the
  // finite ("exact, …") and linear ("geometric …") rules that would otherwise claim these.
  if (
    /^exact for degree|^the n-point rule|^geometric for analytic|^spectral|^o\(n\^\{?[-−]/.test(o)
  )
    return 'approximation';
  if (/^(r-)?linear|^geometric|^a-norm rate|^≤ n steps|^minimal residual|^outer:/.test(o))
    return 'linear';
  if (/^(finite|direct|exact|heuristic|local search)/.test(o)) return 'finite';
  // Error in the step h. An operation count ("O(n²) per evaluation point") is not a rate.
  if (/^o\(h|^error o\(h/.test(o)) return 'truncation';
  return 'none';
}

/** Derivative requirement, the filter researchers ask first. */
export type NeedsClass = 'f' | 'grad' | 'hess' | 'other';

export const NEEDS_LABEL: Record<NeedsClass, string> = {
  f: 'Values of f only',
  grad: 'Uses ∇f',
  hess: 'Uses ∇²f',
  other: 'Data, matrices or graphs',
};

export function needsClass(needs: readonly string[]): NeedsClass {
  if (needs.includes('hess')) return 'hess';
  if (needs.some((n) => n === 'grad' || n === 'grad_batch' || n === 'jac')) return 'grad';
  if (needs.includes('f') || needs.includes('residual')) return 'f';
  return 'other';
}

/** TeX for one `needs` entry (`grad` → ∇f), or null to print the word. */
export const NEEDS_TEX: Record<string, string> = {
  f: 'f',
  grad: '\\nabla f',
  hess: '\\nabla^2 f',
  jac: 'J_F',
  grad_batch: '\\nabla f_i',
  residual: 'r',
  A: 'A',
  b: '\\mathbf b',
  x0: '\\mathbf{x}_0',
  bracket: '[a, b]',
  interval: '[a, b]',
};

export interface MethodFilter {
  q: string;
  family: string;
  needs: string;
  rate: string;
  det: string;
}

export function matchesFilter(m: CatalogMethod, f: MethodFilter): boolean {
  if (f.family && m.family !== f.family) return false;
  if (f.needs && needsClass(m.needs) !== f.needs) return false;
  if (f.rate && rateClass(m.order) !== f.rate) return false;
  if (f.det === 'deterministic' && !m.deterministic) return false;
  if (f.det === 'stochastic' && m.deterministic) return false;
  const tokens = normalize(f.q).trim().split(/\s+/).filter(Boolean);
  if (!tokens.length) return true;
  const hay = normalize(
    `${m.name} ${m.id} ${m.family} ${m.summary} ${m.order} ${m.references.join(' ')} ${m.tags.join(' ')} ${m.params.map((p) => p.name).join(' ')}`,
  );
  return tokens.every((t) => hay.includes(t));
}

/** Python repr of a registry default (`1e-10`, `True`, `'strong_wolfe'`, `None`). */
export function pyRepr(v: unknown, kind?: string): string {
  if (v === null || v === undefined) return 'None';
  if (typeof v === 'boolean') return v ? 'True' : 'False';
  if (typeof v === 'string') return v === 'inf' ? 'inf' : `"${v}"`;
  if (Array.isArray(v)) return `[${v.map((x) => pyRepr(x)).join(', ')}]`;
  if (typeof v === 'number') {
    if (kind === 'int' || (Number.isInteger(v) && kind !== 'float')) return String(v);
    if (Number.isInteger(v) && Math.abs(v) < 1e16) return `${v}.0`;
    const s = String(v);
    const m = /^(-?[\d.]+)e([+-])(\d+)$/.exec(s);
    if (m) return `${m[1]}e${m[2]}${m[3].padStart(2, '0')}`;
    if (Math.abs(v) < 1e-4 && v !== 0) {
      const [mant, exp] = v.toExponential().split('e');
      const e = Number(exp);
      return `${mant.replace(/\.?0+$/, '')}e${e < 0 ? '-' : '+'}${String(Math.abs(e)).padStart(2, '0')}`;
    }
    return s;
  }
  return String(v);
}
