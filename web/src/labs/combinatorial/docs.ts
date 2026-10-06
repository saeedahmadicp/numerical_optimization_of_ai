/**
 * Per-method teaching data of the combinatorial lab: the update rule with the current step's
 * numbers filled in (KaTeX), the live quantities of the card, and the iteration-table columns.
 * Everything is read from `Step.info` (the Python keys).
 */
import type { Step } from '../../core/types';
import type { TableColumn } from '../../viz';
import { int, sig } from '../../core/format';
import type { CombinatorialInstance } from '../../problems/combinatorial';
import type { BbNode } from '../../methods/combinatorial/knapsack';
import { fmtGap, gapPercent, selectedItems, type Kind } from './model';

export interface DocCtx {
  problem: CombinatorialInstance;
  /** L⋆ or z⋆ used for the gap (may be the best found). */
  reference: number | null;
  /** The reference is the best solution found by the runs (the optimum is unknown). */
  bestFound?: boolean;
}

export interface Quantity {
  tex: string;
  value: string;
  /** Plain-language name (tooltip, accessible label). */
  label: string;
}

/** A number for TeX: 4 significant digits, ASCII minus, ×10ⁿ as \times 10^{n}. */
export function tn(x: number | null | undefined, digits = 4): string {
  if (x === null || x === undefined || Number.isNaN(x)) return '\\text{—}';
  if (!Number.isFinite(x)) return x > 0 ? '\\infty' : '-\\infty';
  const s = sig(x, digits).replace(/−/g, '-');
  const m = /^(-?[\d.]+)×10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)$/.exec(s);
  if (!m) return s;
  const sup = '⁰¹²³⁴⁵⁶⁷⁸⁹';
  const e = [...m[2]].map((c) => (c === '⁻' ? '-' : String(sup.indexOf(c)))).join('');
  return `${m[1]} \\times 10^{${e}}`;
}

const num = (v: unknown) => (typeof v === 'number' ? v : null);
const list = (xs: readonly number[], max = 8) =>
  xs.length <= max ? xs.join(', ') : `${xs.slice(0, max).join(', ')}, …`;

function dist(p: CombinatorialInstance, a: number, b: number): number {
  if (p.kind !== 'tsp') return NaN;
  const dx = p.coords[a][0] - p.coords[b][0],
    dy = p.coords[a][1] - p.coords[b][1];
  return Math.sqrt(dx * dx + dy * dy);
}

const d = (a: number, b: number) => `d(${a},${b})`;

// ── Filled-in rules ────────────────────────────────────────────────────────────────────

type Filled = (step: Step, ctx: DocCtx, prev?: Step) => string | null;

const FILLED: Record<string, Filled> = {
  knapsack_dp(step, { problem }) {
    if (problem.kind !== 'knapsack') return null;
    const k = step.k;
    if (k === 0) return 'z_0(d) = 0, \\qquad d = 0, \\dots, C';
    const w = num(step.info.weight)!,
      v = num(step.info.value)!,
      i = num(step.info.item)!;
    const C = problem.capacity;
    return (
      '\\begin{aligned}' +
      `&\\text{item } ${i}: \\; w_{${i}} = ${w},\\ v_{${i}} = ${v} \\\\ ` +
      `z_{${k}}(d) &= \\max\\bigl(z_{${k - 1}}(d),\\; z_{${k - 1}}(d - ${w}) + ${v}\\bigr)` +
      ` \\\\ z_{${k}}(C) &= z_{${k}}(${C}) = ${tn(step.fun)}` +
      '\\end{aligned}'
    );
  },
  knapsack_greedy(step, { problem }) {
    if (problem.kind !== 'knapsack') return null;
    const info = step.info;
    const C = problem.capacity;
    if (info.phase === 'start')
      return `r = C = ${C}, \\qquad \\text{scan by } v_i / w_i \\text{ descending}`;
    const i = num(info.item);
    if (info.phase === 'single_item_fix') {
      const z = num(info.value)!;
      const single = i === null ? 0 : problem.values[i];
      return `z = \\max\\bigl(z_{\\text{greedy}},\\; \\max_{w_i \\le C} v_i\\bigr) = ${z}${
        info.taken ? `\\quad (\\text{item } ${i},\\ v = ${single})` : ''
      }`;
    }
    if (i === null) return null;
    const w = problem.weights[i],
      v = problem.values[i];
    const rBefore = (num(info.residual) ?? 0) + (info.taken ? w : 0);
    const decision = info.fits
      ? `w_{${i}} = ${w} \\le r = ${rBefore} \\;\\Rightarrow\\; x_{${i}} = 1`
      : `w_{${i}} = ${w} > r = ${rBefore} \\;\\Rightarrow\\; x_{${i}} = 0`;
    return `\\begin{aligned} &\\frac{v_{${i}}}{w_{${i}}} = \\frac{${v}}{${w}} = ${tn(v / w)} \\\\ &${decision} \\end{aligned}`;
  },
  knapsack_branch_bound(step) {
    const nodes = (step.info.nodes as BbNode[] | undefined) ?? [];
    const node = nodes[nodes.length - 1];
    if (!node) return null;
    const z = num(step.info.best_value) ?? 0;
    const rest = node.bound - node.value;
    const U = `U = V + \\mathrm{LP} = ${node.value} + ${tn(rest)} = ${tn(node.bound)}`;
    if (node.status === 'leaf')
      return `\\text{all items fixed: } V = ${node.value}${node.incumbent ? ` \\Rightarrow z = ${z}` : ` \\le z = ${z}`}`;
    const floor = Math.floor(node.bound + 1e-9);
    const order = step.info.order as number[];
    return (
      '\\begin{aligned}' +
      `${U.replace('U = V', 'U &= V')} \\\\ ` +
      (node.status === 'pruned'
        ? `\\lfloor U \\rfloor &= ${floor} \\le z = ${z} \\;\\Rightarrow\\; \\text{prune}`
        : `\\lfloor U \\rfloor &= ${floor} > z = ${z} \\;\\Rightarrow\\; \\text{branch on } x_{${order[node.depth]}}`) +
      '\\end{aligned}'
    );
  },
  tsp_nearest_neighbor(step, { problem }) {
    const tour = step.info.tour as number[];
    if (step.k === 0) return `\\pi_0 = ${tour[0]}`;
    if (step.info.closed) {
      const last = tour[tour.length - 1];
      return `\\begin{aligned}L &= \\ell(\\text{path}) + d(\\pi_{n-1}, \\pi_0) \\\\ &= ${tn(step.fun! - dist(problem, last, tour[0]))} + ${tn(dist(problem, last, tour[0]))} = ${tn(step.fun)}\\end{aligned}`;
    }
    const a = tour[tour.length - 2],
      b = tour[tour.length - 1];
    return `\\begin{aligned} \\pi_{${step.k}} &= \\arg\\min_{j \\notin \\{\\pi_0, \\dots, \\pi_{${step.k - 1}}\\}} d(${a}, j) = ${b} \\\\ ${d(a, b)} &= ${tn(dist(problem, a, b))} \\end{aligned}`;
  },
  tsp_two_opt: twoOptFilled,
  tsp_simulated_annealing(step) {
    const move = step.info.move as Record<string, unknown> | null;
    const T = num(step.info.temperature);
    if (!move || T === null) return `T_0 = t_0\\,\\bar d = ${tn(T)}`;
    const delta = move.delta as number;
    const acc = move.accepted as boolean;
    const [[a, b], [c, e]] = move.removed as number[][];
    const line1 = `\\Delta &= ${d(a, c)} + ${d(b, e)} - ${d(a, b)} - ${d(c, e)} \\\\ &= ${tn(delta)}`;
    const line2 =
      delta <= 0
        ? `&\\Delta \\le 0 \\;\\Rightarrow\\; \\text{accepted}`
        : `e^{-\\Delta/T} &= e^{-${tn(delta)}/${tn(T)}} = ${tn(Math.exp(-delta / T), 3)} \\;\\Rightarrow\\; \\text{${acc ? 'accepted' : 'rejected'}}`;
    return `\\begin{aligned}${line1} \\\\ ${line2}\\end{aligned}`;
  },
  tsp_or_opt(step, { problem }) {
    const move = step.info.move as Record<string, unknown> | null;
    if (!move) return `L(\\pi_0) = ${tn(step.fun)}`;
    const seg = move.segment as number[];
    const removed = move.removed as number[][];
    const added = move.added as number[][];
    const sum = (es: number[][]) => es.reduce((s, [a, b]) => s + dist(problem, a, b), 0);
    return (
      '\\begin{aligned}' +
      `&\\text{move } (${seg.join(', ')})${move.reversed ? '\\text{, reversed}' : ''} \\\\ ` +
      `&\\Delta = \\textstyle\\sum_{\\text{added}} d - \\sum_{\\text{removed}} d = ${tn(sum(added))} - ${tn(sum(removed))} = ${tn(move.delta as number)}` +
      '\\end{aligned}'
    );
  },
  tsp_genetic(step) {
    const i = step.info;
    return `\\begin{aligned} L_{\\text{best}} &= ${tn(num(i.length))}, \\quad \\bar L = ${tn(num(i.mean_length))} \\\\ L_{\\text{worst}} &= ${tn(num(i.worst_length))} \\end{aligned}`;
  },
  tsp_ant_colony(step) {
    const i = step.info;
    if (step.k === 0)
      return `\\tau_0 = m / C^{\\text{nn}} = ${tn(num(i.tau_max))}, \\qquad C^{\\text{nn}} = ${tn(step.fun)}`;
    return `\\begin{aligned} \\tau_{ij} &\\in [${tn(num(i.tau_min), 3)},\\ ${tn(num(i.tau_max), 3)}] \\\\ L_{\\text{it}} &= ${tn(step.fun)}, \\quad L_{\\text{best}} = ${tn(num(i.best_length))} \\end{aligned}`;
  },
  tsp_held_karp(step) {
    const i = step.info;
    if (step.k === 0) return 'C(\\{\\,\\}, 0) = 0';
    if (i.closed)
      return `L^\\star = \\min_{j} C(\\{1, \\dots, n-1\\}, j) + d(j, 0) = ${tn(step.fun)}`;
    return `\\begin{aligned} |S| &= ${i.subset_size}: \\; ${int(num(i.states) ?? 0)} \\text{ states} \\\\ \\min_{|S| = ${i.subset_size},\\, j} C(S, j) &= ${tn(step.fun)} \\end{aligned}`;
  },
};

function twoOptFilled(step: Step, { problem }: DocCtx): string | null {
  const move = step.info.move as Record<string, unknown> | null;
  if (!move) return `L(\\pi_0) = ${tn(step.fun)}`;
  const [[a, b], [c, e]] = move.removed as number[][];
  const ac = dist(problem, a, c),
    be = dist(problem, b, e),
    ab = dist(problem, a, b),
    ce = dist(problem, c, e);
  return (
    '\\begin{aligned}' +
    `\\Delta &= ${d(a, c)} + ${d(b, e)} - ${d(a, b)} - ${d(c, e)} \\\\ ` +
    `&= ${tn(ac)} + ${tn(be)} - ${tn(ab)} - ${tn(ce)} = ${tn(move.delta as number)}` +
    '\\end{aligned}'
  );
}

export function filledRule(id: string, step: Step | undefined, ctx: DocCtx): string | null {
  if (!step) return null;
  try {
    return FILLED[id]?.(step, ctx) ?? null;
  } catch {
    return null;
  }
}

// ── Live quantities ────────────────────────────────────────────────────────────────────

const v4 = (x: unknown) => (typeof x === 'number' ? sig(x, 5) : '—');

export function quantities(
  id: string,
  step: Step | undefined,
  ctx: DocCtx,
  kind: Kind,
): Quantity[] {
  if (!step) return [];
  const i = step.info;
  const gap = fmtGap(gapPercent(kind, step, ctx.reference));
  if (kind === 'knapsack') {
    const p = ctx.problem.kind === 'knapsack' ? ctx.problem : null;
    const items = selectedItems(step.x);
    const W = p ? items.reduce((s, j) => s + p.weights[j], 0) : 0;
    const common: Quantity[] = [
      { tex: 'z', value: v4(step.fun), label: 'Value of the selection' },
      {
        tex: 'W',
        value: p ? `${W} / ${p.capacity}` : '—',
        label: 'Weight of the selection / capacity',
      },
      ctx.bestFound
        ? {
            tex: '(z_{\\text{best}} - z)/z_{\\text{best}}',
            value: gap,
            label: 'Relative gap to the best found',
          }
        : { tex: '(z^\\star - z)/z^\\star', value: gap, label: 'Relative gap to the optimum' },
    ];
    if (id === 'knapsack_branch_bound') {
      const nodes = (i.nodes as BbNode[] | undefined) ?? [];
      const n = nodes[nodes.length - 1];
      return [
        { tex: '\\text{node}', value: int(step.k), label: 'Node (pop order)' },
        { tex: 'U', value: n ? sig(n.bound, 5) : '—', label: 'Dantzig bound of the node' },
        {
          tex: '\\text{open}',
          value: int(num(i.open_nodes) ?? 0),
          label: 'Open nodes on the stack',
        },
        ...common,
      ];
    }
    if (id === 'knapsack_greedy')
      return [
        { tex: 'k', value: int(step.k), label: 'Items examined' },
        { tex: 'r', value: v4(i.residual), label: 'Residual capacity' },
        { tex: 'v_i/w_i', value: v4(i.ratio), label: 'Ratio of the item examined' },
        ...common,
      ];
    return [
      { tex: 'k', value: int(step.k), label: 'Rows filled (items considered)' },
      {
        tex: '\\text{item}',
        value: i.item === null ? '—' : String(i.item),
        label: 'Item of this row',
      },
      {
        tex: 'x_k',
        value: items.length ? `{${list(items, 6)}}` : '∅',
        label: 'Selection backtracked from (k, C)',
      },
      ...common,
    ];
  }
  const q: Quantity[] = [
    { tex: 'k', value: int(step.k), label: 'Step' },
    { tex: 'L(\\pi_k)', value: v4(step.fun), label: 'Length of the tour (or open path)' },
    ctx.bestFound
      ? {
          tex: '\\text{gap to } L_{\\text{best}}',
          value: gap,
          label: 'Relative gap (L − L best)/L best to the best tour found',
        }
      : { tex: '(L - L^\\star)/L^\\star', value: gap, label: 'Relative gap to the optimum' },
  ];
  const move = i.move as Record<string, unknown> | null | undefined;
  switch (id) {
    case 'tsp_two_opt':
    case 'tsp_or_opt':
      q.push({
        tex: '\\Delta',
        value: move ? v4(move.delta) : '—',
        label: 'Length change of the move',
      });
      break;
    case 'tsp_simulated_annealing':
      q.push(
        { tex: 'T', value: v4(i.temperature), label: 'Temperature' },
        {
          tex: '\\text{accept}',
          value:
            typeof i.acceptance_rate === 'number' ? `${sig(i.acceptance_rate * 100, 3)} %` : '—',
          label: 'Acceptance rate since the previous recorded step',
        },
        { tex: 'L_{\\text{best}}', value: v4(i.best_length), label: 'Best length so far' },
      );
      break;
    case 'tsp_genetic':
      q.push(
        { tex: '\\bar L', value: v4(i.mean_length), label: 'Mean length of the population' },
        {
          tex: 'L_{\\text{worst}}',
          value: v4(i.worst_length),
          label: 'Worst length of the population',
        },
        { tex: '\\text{stall}', value: String(i.stall), label: 'Generations without improvement' },
      );
      break;
    case 'tsp_ant_colony':
      q.push(
        { tex: 'L_{\\text{best}}', value: v4(i.best_length), label: 'Best ant tour so far' },
        { tex: '\\tau_{\\max}', value: v4(i.tau_max), label: 'Largest pheromone' },
        { tex: '\\text{stall}', value: String(i.stall), label: 'Iterations without improvement' },
      );
      break;
    case 'tsp_held_karp':
      q.push(
        { tex: '|S|', value: String(i.subset_size), label: 'Subset size of the DP layer' },
        {
          tex: '\\text{states}',
          value: int(num(i.states) ?? 0),
          label: 'States C(S, j) in the layer',
        },
      );
      break;
    case 'tsp_nearest_neighbor':
      q.push({ tex: '\\pi_k', value: String(i.current), label: 'City just appended' });
      break;
  }
  return q;
}

// ── Iteration-table columns ────────────────────────────────────────────────────────────

const col = (
  key: string,
  tex: string,
  value: (s: Step) => string,
  label?: string,
  align: 'left' | 'right' = 'right',
): TableColumn<Step> => ({ key, tex, value: (s) => value(s), label, align });

const kCol = (header = 'k') => col('k', header, (s) => int(s.k));
const funCol = (tex: string) => col('fun', tex, (s) => v4(s.fun));
const edge = (e: number[]) => `${e[0]}–${e[1]}`;

export function columns(id: string, ctx: DocCtx, kind: Kind): TableColumn<Step>[] {
  const gapCol = col(
    'gap',
    '\\text{gap}',
    (s) => fmtGap(gapPercent(kind, s, ctx.reference)),
    ctx.bestFound ? 'Relative gap to the best found' : 'Relative gap to the optimum',
  );
  switch (id) {
    case 'knapsack_dp':
      return [
        kCol(),
        col('item', 'i', (s) => (s.info.item === null ? '—' : String(s.info.item)), 'Item'),
        col('w', 'w_i', (s) => (s.info.weight === null ? '—' : String(s.info.weight))),
        col('v', 'v_i', (s) => (s.info.value === null ? '—' : String(s.info.value))),
        funCol('z_k(C)'),
        gapCol,
      ];
    case 'knapsack_greedy':
      return [
        kCol(),
        col('item', 'i', (s) => (s.info.item === null ? '—' : String(s.info.item))),
        col('ratio', 'v_i/w_i', (s) => v4(s.info.ratio)),
        col(
          'take',
          'x_i',
          (s) =>
            s.info.phase === 'start'
              ? '—'
              : s.info.phase === 'single_item_fix'
                ? s.info.taken
                  ? 'single'
                  : 'keep'
                : s.info.taken
                  ? '1'
                  : '0 (no fit)',
          'Decision',
        ),
        col('r', 'r', (s) => String(s.info.residual), 'Residual capacity'),
        funCol('z'),
      ];
    case 'knapsack_branch_bound':
      return [
        kCol('\\text{node}'),
        col('depth', '\\text{depth}', (s) => String(last(s)?.depth ?? '—')),
        col('fix', '\\text{branch}', (s) => {
          const n = last(s);
          return n && n.item !== null ? `x${sub(n.item)} = ${n.decision}` : 'root';
        }),
        col('V', 'V', (s) => String(last(s)?.value ?? '—'), 'Value of the fixed items'),
        col('U', 'U', (s) => (last(s) ? sig(last(s)!.bound, 5) : '—'), 'Dantzig bound'),
        col(
          'status',
          '\\text{status}',
          (s) => {
            const n = last(s);
            return n ? `${n.status}${n.incumbent ? ' ★' : ''}` : '—';
          },
          'Status',
          'left',
        ),
        funCol('z'),
      ];
    case 'tsp_two_opt':
    case 'tsp_or_opt':
      return [
        kCol(),
        col(
          'move',
          '\\text{removed} \\to \\text{added}',
          (s) => {
            const m = s.info.move as Record<string, unknown> | null;
            if (!m) return 'start';
            return `${(m.removed as number[][]).map(edge).join(' ')} → ${(m.added as number[][]).map(edge).join(' ')}`;
          },
          'Edges exchanged',
          'left',
        ),
        col('delta', '\\Delta', (s) => {
          const dlt = (s.info.move as { delta?: number } | null)?.delta;
          return typeof dlt === 'number' ? sig(dlt, 4) : '—';
        }),
        funCol('L'),
      ];
    case 'tsp_simulated_annealing':
      return [
        kCol(),
        col('T', 'T', (s) => v4(s.info.temperature)),
        col('acc', '\\text{accept}', (s) =>
          typeof s.info.acceptance_rate === 'number'
            ? `${sig(s.info.acceptance_rate * 100, 3)} %`
            : '—',
        ),
        funCol('L'),
        col('best', 'L_{\\text{best}}', (s) => v4(s.info.best_length)),
        gapCol,
      ];
    case 'tsp_genetic':
      return [
        kCol(),
        funCol('L_{\\text{best}}'),
        col('mean', '\\bar L', (s) => v4(s.info.mean_length)),
        col('worst', 'L_{\\text{worst}}', (s) => v4(s.info.worst_length)),
        col('stall', '\\text{stall}', (s) => String(s.info.stall)),
        gapCol,
      ];
    case 'tsp_ant_colony':
      return [
        kCol(),
        funCol('L_{\\text{it}}'),
        col('mean', '\\bar L', (s) => v4(s.info.mean_length)),
        col('best', 'L_{\\text{best}}', (s) => v4(s.info.best_length)),
        col('tau', '\\tau_{\\max}', (s) => v4(s.info.tau_max)),
        col('stall', '\\text{stall}', (s) => String(s.info.stall)),
      ];
    case 'tsp_held_karp':
      return [
        kCol(),
        col('S', '|S|', (s) => String(s.info.subset_size)),
        col('states', '\\text{states}', (s) => int(num(s.info.states) ?? 0)),
        funCol('\\min C(S, j)'),
        col(
          'path',
          '\\text{path}',
          (s) => list(s.info.tour as number[], 6),
          'Cheapest path',
          'left',
        ),
      ];
    case 'tsp_nearest_neighbor':
    default:
      return [
        kCol(),
        col('cur', '\\pi_k', (s) => String(s.info.current ?? '—')),
        funCol('L'),
        gapCol,
      ];
  }
}

const SUBS = '₀₁₂₃₄₅₆₇₈₉';
const sub = (n: number) => [...String(n)].map((c) => SUBS[Number(c)]).join('');
function last(s: Step): BbNode | undefined {
  const nodes = s.info.nodes as BbNode[] | undefined;
  return nodes?.[nodes.length - 1];
}
