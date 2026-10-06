/**
 * Table columns of the global lab: the iteration table (one row per step, method-specific
 * columns) and the population table (one row per member of the current generation).
 */
import { createElement } from 'react';
import { Formula } from '../../ui/components/Formula';
import type { Step, Vector } from '../../core/types';
import { int, sigFixed, vecFixed } from '../../core/format';
import type { Column, TableColumn } from '../../viz';
import { acceptedCount, searchScale } from './geometry';

const tex = (t: string) => createElement(Formula, { tex: t });
const yesNo = (v: unknown) => (v === true ? 'yes' : v === false ? 'no' : '—');

const K: Column = { key: 'k', label: tex('k'), value: (s) => s.k, align: 'right', width: '34px' };
const BEST_X: Column = {
  key: 'x',
  label: tex('\\mathbf{x}_{\\mathrm{best}}'),
  width: 'minmax(132px, 1.8fr)',
  value: (s) => vecFixed(s.x as Vector, 4),
};
const BEST_F: Column = {
  key: 'fun',
  label: tex('f(\\mathbf{x}_{\\mathrm{best}})'),
  value: (s) => sigFixed(s.fun, 4),
  align: 'right',
  width: 'minmax(76px, 1fr)',
};

function num(key: string, label: string, read: (s: Step) => number | null, digits = 3): Column {
  return {
    key,
    label: tex(label),
    value: (s) => sigFixed(read(s), digits),
    align: 'right',
    width: 'minmax(58px, 0.9fr)',
  };
}

function word(key: string, label: string, read: (s: Step) => string): Column {
  return { key, label: tex(label), value: read, align: 'right', width: '44px' };
}

/** Iteration-table columns per method. */
export function iterationColumns(method: string, popSize?: number): Column[] {
  switch (method) {
    case 'simulated_annealing':
      return [
        K,
        BEST_X,
        BEST_F,
        num('T', 'T_k', (s) => s.info.temperature as number),
        num('p', 'p_k', (s) => s.info.accept_prob as number | null, 2),
        word('acc', '\\text{acc.}', (s) => yesNo(s.info.accepted)),
      ];
    case 'particle_swarm':
      return [K, BEST_X, BEST_F, num('spread', '\\text{spread}', (s) => searchScale(method, s))];
    case 'differential_evolution':
      return [
        K,
        BEST_X,
        BEST_F,
        {
          key: 'won',
          label: tex('\\text{won}'),
          value: (s) => (s.k === 0 ? '—' : `${acceptedCount(s)}/${popSize ?? '?'}`),
          align: 'right',
          width: '52px',
        },
        num('spread', '\\text{spread}', (s) => searchScale(method, s)),
      ];
    case 'cma_es':
      return [
        K,
        BEST_X,
        BEST_F,
        num('sigma', '\\sigma_{k+1}', (s) => s.info.sigma as number),
        word('h', 'h_\\sigma', (s) => yesNo(s.info.h_sigma)),
      ];
    case 'basin_hopping':
      return [
        K,
        BEST_X,
        BEST_F,
        num('fz', 'f(\\mathbf{z})', (s) => s.info.local_f as number, 4),
        word('acc', '\\text{acc.}', (s) => yesNo(s.info.accepted)),
        {
          key: 'minima',
          label: tex('\\#\\min'),
          value: (s) => int((s.info.minima as unknown[]).length),
          align: 'right',
          width: '44px',
        },
      ];
    default:
      return [K, BEST_X, BEST_F];
  }
}

// ── Population table ──────────────────────────────────────────────────────────────────

export interface MemberRow {
  i: number;
  x: Vector;
  f: number;
  /** Method-specific third and fourth values. */
  a?: number | string | null;
  b?: number | string | null;
}

export interface PopulationView {
  title: string;
  caption: string;
  rows: MemberRow[];
  columns: TableColumn<MemberRow>[];
  highlight: number | null;
}

const IDX: TableColumn<MemberRow> = {
  key: 'i',
  tex: 'i',
  value: (r) => r.i,
  width: '2.5rem',
};
const X: TableColumn<MemberRow> = {
  key: 'x',
  tex: '\\mathbf{x}_i',
  value: (r) => vecFixed(r.x, 4),
  align: 'left',
  mono: true,
};
const F: TableColumn<MemberRow> = {
  key: 'f',
  tex: 'f(\\mathbf{x}_i)',
  value: (r) => sigFixed(r.f, 4),
};
const cell = (v: number | string | null | undefined) =>
  typeof v === 'number' ? sigFixed(v, 4) : (v ?? '—');

/** The current generation of a population method as table rows (null for annealing). */
export function populationView(method: string, step: Step): PopulationView | null {
  const info = step.info;
  switch (method) {
    case 'particle_swarm': {
      const X0 = info.particles as Vector[];
      const fs = info.particles_f as number[];
      const pf = info.personal_best_f as number[];
      const vs = info.velocities as Vector[];
      const rows = X0.map((x, i) => ({ i, x, f: fs[i], a: pf[i], b: Math.hypot(...vs[i]) }));
      return {
        title: 'Swarm',
        caption: `Particles after iteration ${step.k}: position, value, personal best value f(𝐩ᵢ) and speed ‖𝐯ᵢ‖.`,
        rows,
        columns: [
          IDX,
          X,
          F,
          { key: 'a', tex: 'f(\\mathbf{p}_i)', value: (r) => cell(r.a) },
          { key: 'b', tex: '\\|\\mathbf{v}_i\\|', value: (r) => cell(r.b) },
        ],
        highlight: argminIndex(fs),
      };
    }
    case 'differential_evolution': {
      const pop = info.population as Vector[];
      const fs = info.population_f as number[];
      const tf = info.trials_f as number[];
      const acc = info.accepted as boolean[];
      const rows = pop.map((x, i) => ({
        i,
        x,
        f: fs[i],
        a: tf.length ? tf[i] : null,
        b: acc.length ? (acc[i] ? 'won' : 'lost') : '—',
      }));
      return {
        title: 'Population',
        caption: `Members after generation ${step.k}, with the value of this generation’s trial 𝐮ᵢ and whether it replaced 𝐱ᵢ.`,
        rows,
        columns: [
          IDX,
          X,
          F,
          { key: 'a', tex: 'f(\\mathbf{u}_i)', value: (r) => cell(r.a) },
          { key: 'b', header: 'Trial', value: (r) => cell(r.b), mono: false },
        ],
        highlight: info.best_index as number,
      };
    }
    case 'cma_es': {
      const X0 = info.population as Vector[];
      const fs = info.population_f as number[];
      const sel = info.selected as number[];
      if (!X0.length)
        return {
          title: 'Samples',
          caption: 'Step 0 has no samples yet: the distribution starts at 𝐦₀ = 𝐱₀ with C = I.',
          rows: [],
          columns: [IDX, X, F],
          highlight: null,
        };
      const order = X0.map((_, i) => i).sort((a, b) => fs[a] - fs[b]);
      const rank = new Map(sel.map((j, r) => [j, r + 1]));
      const rows = order.map((j) => ({ i: j, x: X0[j], f: fs[j], a: rank.get(j) ?? null }));
      return {
        title: 'Samples',
        caption: `The λ = ${X0.length} samples of generation ${step.k}, best first; the μ = ${sel.length} best (rank 1…μ) move the mean and update C.`,
        rows,
        columns: [
          { ...IDX, tex: 'j' },
          { ...X, tex: '\\mathbf{x}_j' },
          { ...F, tex: 'f(\\mathbf{x}_j)' },
          {
            key: 'a',
            header: 'Rank',
            value: (r) => (typeof r.a === 'number' ? String(r.a) : '—'),
          },
        ],
        highlight: 0,
      };
    }
    case 'basin_hopping': {
      const minima = info.minima as { x: Vector; f: number; hits: number }[];
      const rows = minima
        .map((m, i) => ({ i: i + 1, x: m.x, f: m.f, a: m.hits }))
        .sort((p, q) => p.f - q.f);
      return {
        title: 'Minima',
        caption: `The ${minima.length} distinct local minima found by hop ${step.k}, lowest first, with the number of hops that reached each (order of discovery in #).`,
        rows,
        columns: [
          { ...IDX, tex: '\\#' },
          { ...X, tex: '\\mathbf{z}' },
          { ...F, tex: 'f(\\mathbf{z})' },
          { key: 'a', header: 'Hits', value: (r) => String(r.a) },
        ],
        highlight: 0,
      };
    }
    default:
      return null;
  }
}

function argminIndex(x: readonly number[]): number {
  let b = 0;
  for (let i = 1; i < x.length; i++) if (x[i] < x[b]) b = i;
  return b;
}
