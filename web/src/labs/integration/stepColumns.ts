/**
 * The iteration table of the focused run, synced to the playhead. Columns follow the method:
 * composite rules and Romberg show N, the estimate, the true error, the method's own error
 * estimate ε̂, the observed ratio ρ and the observed order log₂ρ; Gauss shows m; adaptive Simpson
 * shows the accepted interval, its depth d, ε̂ = |S₂ − S₁|/15 and whether it passed (the local
 * tolerance τ is in the readout); Monte Carlo
 * shows N and the standard error; the nested rules show the degree n (Clenshaw–Curtis) or the
 * node count N (Patterson), the number of new nodes, and ε̂ = |I_k − I_{k−1}|. Headers use the lab's one symbol per estimate (`estimateTex`:
 * T_N, S_N, R_{k,k}, G_m, I_k, I_N, …). The estimate is set to 8 significant digits and the
 * errors to 2 (h = (b − a)/N is left to the readout), so the table fits the details column; the
 * error columns carry the rest.
 */
import type { Step } from '../../core/types';
import { int, sci, sig, sigFixed } from '../../core/format';
import type { TableColumn } from '../../viz';
import { RULE_KIND } from './geometry';
import { estimateTex } from './cardRule';

const num = (v: unknown, d = 10) => (typeof v === 'number' ? sigFixed(v, d) : '—');
const small = (v: unknown) => (typeof v === 'number' ? sci(v, 2) : '—');

const K: TableColumn<Step> = { key: 'k', tex: 'k', value: (s) => s.k, width: '2rem' };
const EST = (tex: string): TableColumn<Step> => ({
  key: 'est',
  tex,
  value: (s) => num(s.info.estimate, 8),
});
const ERR = (tex: string): TableColumn<Step> => ({
  key: 'err',
  tex,
  label: 'true error',
  value: (s) => small(s.info.error),
});
const ERR_EST: TableColumn<Step> = {
  key: 'errest',
  tex: '\\hat\\varepsilon',
  label: 'error estimate',
  value: (s) => small(s.info.err_est),
};
const RATIO: TableColumn<Step> = {
  key: 'ratio',
  tex: '\\log_2 \\rho_k',
  label: 'observed order log2 of the ratio',
  value: (s) => {
    const r = s.info.ratio;
    return typeof r === 'number' && r > 0 ? sigFixed(Math.log2(r), 3) : '—';
  },
};

export function columnsFor(method: string): TableColumn<Step>[] {
  switch (RULE_KIND[method]) {
    case 'romberg':
      return [
        K,
        { key: 'N', tex: 'N', value: (s) => int(s.info.n_panels as number) },
        EST('R_{k,k}'),
        ERR('|R_{k,k} - I|'),
        { ...ERR_EST, tex: '\\hat\\varepsilon_k', label: 'error estimate |R(k,k) − R(k−1,k−1)|' },
        { ...RATIO, label: 'observed order of the trapezoid column' },
      ];
    case 'gauss':
      return [
        K,
        { key: 'm', tex: 'm', value: (s) => s.info.n_points as number },
        EST('G_m'),
        ERR('|G_m - I|'),
        { ...ERR_EST, tex: '\\hat\\varepsilon_m' },
      ];
    case 'adaptive':
      return [
        K,
        {
          key: 'iv',
          tex: '[\\alpha, \\beta]',
          label: 'accepted interval',
          value: (s) => {
            const iv = s.info.interval as [number, number] | null;
            return iv ? `[${sig(iv[0], 3)}, ${sig(iv[1], 3)}]` : '—';
          },
        },
        { key: 'depth', tex: 'd', label: 'depth', value: (s) => s.info.depth as number },
        { ...ERR_EST, tex: '\\hat\\varepsilon', label: 'error estimate |S2 − S1|/15' },
        {
          key: 'ok',
          header: 'Test',
          label:
            'acceptance test: ✓ passed, ✓✓ passed on two levels, or forced at the maximum depth',
          align: 'left',
          value: (s) =>
            s.info.passed === null
              ? '—'
              : s.info.forced
                ? 'forced'
                : s.info.passed && s.info.parent_passed
                  ? '✓✓'
                  : '✓',
        },
        EST('I_k'),
      ];
    case 'nested': {
      const cc = method === 'clenshaw_curtis';
      const sym = estimateTex(method);
      return [
        K,
        cc
          ? { key: 'n', tex: 'n', label: 'degree', value: (s) => int(s.info.n as number) }
          : { key: 'N', tex: 'N', label: 'nodes', value: (s) => int(s.info.n_points as number) },
        {
          key: 'new',
          header: 'New',
          label: 'nodes evaluated for the first time at this step',
          value: (s) => int(s.info.new_nodes as number),
        },
        EST(sym),
        ERR(`|${sym} - I|`),
        {
          ...ERR_EST,
          tex: '\\hat\\varepsilon_k',
          label: 'error estimate |I_k − I_(k−1)|',
        },
      ];
    }
    case 'monte-carlo':
      return [
        K,
        { key: 'N', tex: 'N', value: (s) => int(s.info.n_samples as number) },
        EST('I_N'),
        ERR('|I_N - I|'),
        { ...ERR_EST, tex: '\\text{s.e.}', label: 'standard error' },
      ];
    default: {
      const sym = estimateTex(method);
      return [
        K,
        { key: 'N', tex: 'N', value: (s) => int(s.info.n_panels as number) },
        EST(sym),
        ERR(`|${sym} - I|`),
        { ...ERR_EST, tex: '\\hat\\varepsilon_N' },
        RATIO,
      ];
    }
  }
}
