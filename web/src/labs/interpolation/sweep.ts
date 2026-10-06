/**
 * The error-vs-n study: max |p − f| on the 200-point grid for n = 3 … 40 nodes of one layout,
 * for each selected method with its parameters — the classic figure behind Runge's example
 * (equispaced: the polynomial error grows geometrically; Chebyshev: it decays for analytic f).
 */
import { defaults, getMethod, hasMethod } from '../../core/registry';
import type { DataProblem } from '../../problems/data';
import type { MethodSelection } from '../_shell';
import { activeData } from './geometry';

export interface SweepSeries {
  id: string;
  slot: number;
  label: string;
  x: number[];
  values: (number | null)[];
  /**
   * The method ignored the swept nodes and sampled f at its own Chebyshev nodes (Chebyshev
   * interpolation whenever the data are exact samples of f): the curve does not belong to the
   * x-axis' node layout and is drawn dashed.
   */
  ownNodes: boolean;
}

export function errorVsN(
  base: DataProblem,
  layout: 'equi' | 'cheb',
  selection: readonly MethodSelection[],
  ns: readonly number[],
): SweepSeries[] {
  if (!base.fTrue) return [];
  return selection
    .filter((s) => hasMethod(s.id))
    .map((sel) => {
      const m = getMethod(sel.id);
      let ownNodes = false;
      const values = ns.map((n) => {
        try {
          const r = m.fn(activeData(base, layout, n, null), { ...defaults(m.spec), ...sel.params });
          if (r.extra.source === 'f_true' && layout === 'equi') ownNodes = true;
          return typeof r.fun === 'number' && Number.isFinite(r.fun) ? r.fun : null;
        } catch {
          return null;
        }
      });
      return { id: sel.id, slot: sel.slot, label: m.spec.name, x: [...ns], values, ownNodes };
    });
}
