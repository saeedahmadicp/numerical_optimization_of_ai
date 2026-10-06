/**
 * Pure helpers of the combinatorial lab (no React): problem kinds, edited instances, the
 * reference optimum, the gap series, layouts, DP backtrack paths and branch-and-bound trees.
 */
import type { RegisteredMethod } from '../../core/registry';
import type { Result, Step } from '../../core/types';
import { int, sig } from '../../core/format';
import { describeResult, type RunStatus, type StatusWording } from '../_shell/status';
import type { Codec } from '../../app/useUrlState';
import type {
  CombinatorialInstance,
  KnapsackInstance,
  TspInstance,
} from '../../problems/combinatorial';
import { knapsackDp, type BbNode } from '../../methods/combinatorial/knapsack';
import { HELD_KARP_MAX_N, tspHeldKarp } from '../../methods/combinatorial/tsp';

export type Kind = 'knapsack' | 'tsp';

export const methodKind = (m: RegisteredMethod): Kind =>
  m.spec.needs.includes('knapsack') ? 'knapsack' : 'tsp';

// ── Edited instances (direct manipulation) ─────────────────────────────────────────────

/** A moved city: index and new coordinates. */
export type CityEdit = [index: number, x: number, y: number];

const round1 = (v: number) => Math.round(v * 10) / 10;

/** `?c=3:45.5:60,7:12:80` — moved cities, one decimal. Invalid entries are dropped. */
export const CITY_EDITS_CODEC: Codec<CityEdit[]> = {
  parse(s) {
    const out: CityEdit[] = [];
    for (const part of s.split(',')) {
      const [i, x, y] = part.split(':').map(Number);
      if (Number.isInteger(i) && i >= 0 && Number.isFinite(x) && Number.isFinite(y))
        out.push([i, x, y]);
    }
    return out.length ? out : undefined;
  },
  format: (v) => v.map(([i, x, y]) => `${i}:${round1(x)}:${round1(y)}`).join(','),
};

/** Replace or add the edit of city `i` (sorted by index, so equal edits give equal URLs). */
export function setCityEdit(edits: readonly CityEdit[], i: number, x: number, y: number) {
  return [...edits.filter((e) => e[0] !== i), [i, round1(x), round1(y)] as CityEdit].sort(
    (a, b) => a[0] - b[0],
  );
}

/** The instance with its cities moved; a new id so runs are recomputed, optimum unknown. */
export function applyCityEdits(base: TspInstance, edits: readonly CityEdit[]): TspInstance {
  const valid = edits.filter(([i]) => i < base.coords.length);
  if (!valid.length) return base;
  const coords = base.coords.map((c) => [c[0], c[1]] as [number, number]);
  for (const [i, x, y] of valid) coords[i] = [x, y];
  return {
    ...base,
    id: `${base.id}+${CITY_EDITS_CODEC.format(valid)}`,
    name: `${base.name} (edited)`,
    coords,
    optimumLength: null,
    description: `${valid.length} ${valid.length === 1 ? 'city' : 'cities'} moved from the library instance ${base.id}.`,
  };
}

/** The instance with capacity C; a new id, optimum unknown. */
export function withCapacity(base: KnapsackInstance, C: number | null): KnapsackInstance {
  if (C === null || C === base.capacity) return base;
  return {
    ...base,
    id: `${base.id}@C=${C}`,
    name: `${base.name}, C = ${C}`,
    capacity: C,
    optimumValue: null,
    description: `Capacity changed from ${base.capacity} to ${C}.`,
  };
}

export const baseId = (id: string) => id.split(/[+@]/)[0];

// ── Reference optimum ──────────────────────────────────────────────────────────────────

export interface Reference {
  /** L⋆ (TSP) or z⋆ (knapsack); null when unknown and nothing was found. */
  value: number | null;
  /** Where it comes from: the library, an exact solve in the browser, or the best run. */
  source: 'library' | 'held-karp' | 'dp' | 'best-found' | 'none';
}

/** L⋆ / z⋆: the library value, else an exact solve (Held–Karp for n ≤ 16, DP), else null. */
export function exactReference(p: CombinatorialInstance): Reference {
  if (p.kind === 'knapsack') {
    if (p.optimumValue !== null) return { value: p.optimumValue, source: 'library' };
    try {
      return { value: knapsackDp(p, {}).fun, source: 'dp' };
    } catch {
      return { value: null, source: 'none' };
    }
  }
  if (p.optimumLength !== null) return { value: p.optimumLength, source: 'library' };
  if (p.coords.length <= HELD_KARP_MAX_N) {
    try {
      return { value: tspHeldKarp(p, {}).fun, source: 'held-karp' };
    } catch {
      return { value: null, source: 'none' };
    }
  }
  return { value: null, source: 'none' };
}

/** Fall back to the best complete solution of the runs when the optimum is unknown. */
export function withBestFound(ref: Reference, kind: Kind, results: readonly Result[]): Reference {
  if (ref.value !== null) return ref;
  const vals = results.filter((r) => r.fun !== null && r.trace.length).map((r) => r.fun as number);
  if (!vals.length) return ref;
  return { value: kind === 'tsp' ? Math.min(...vals) : Math.max(...vals), source: 'best-found' };
}

/** Is the step's tour complete (not the open path of a constructive method)? */
export function isCompleteTour(step: Step): boolean {
  return step.info.closed === undefined || step.info.closed === true;
}

/**
 * Length of the tour a panel shows: the best tour so far for the Ant System (its result; the
 * step's `fun` is the iteration-best ant), else the step's own tour.
 */
export function shownLength(step: Step): number {
  const best = step.info.best_length;
  return Array.isArray(step.info.pheromone) || step.info.pheromone === null
    ? typeof best === 'number'
      ? best
      : (step.fun as number)
    : (step.fun as number);
}

/** Relative gap in percent: (L − L⋆)/L⋆ (TSP) or (z⋆ − z)/z⋆ (knapsack); null when undefined. */
export function gapPercent(kind: Kind, step: Step, ref: number | null): number | null {
  if (ref === null || step.fun === null) return null;
  if (kind === 'tsp') {
    if (!isCompleteTour(step)) return null;
    return ref > 0 ? ((shownLength(step) - ref) / ref) * 100 : 0;
  }
  return ref > 0 ? ((ref - step.fun) / ref) * 100 : 0;
}

export function fmtGap(g: number | null): string {
  if (g === null) return '—';
  if (Math.abs(g) < 5e-9) return '0 %';
  return `${sig(g, g < 1 ? 2 : 3)} %`;
}

// ── Step units and run status ──────────────────────────────────────────────────────────

/** What one trace step of a method is (singular, plural). */
export const UNITS: Record<string, [string, string]> = {
  knapsack_dp: ['item row', 'item rows'],
  knapsack_greedy: ['item', 'items'],
  knapsack_branch_bound: ['node', 'nodes'],
  tsp_nearest_neighbor: ['city', 'cities'],
  tsp_two_opt: ['move', 'moves'],
  tsp_or_opt: ['move', 'moves'],
  tsp_simulated_annealing: ['proposal', 'proposals'],
  tsp_genetic: ['generation', 'generations'],
  tsp_ant_colony: ['iteration', 'iterations'],
  tsp_held_karp: ['layer', 'layers'],
};

/**
 * What one recorded trace step (one unit on the chart's x-axis) of a run is: "1 move",
 * "100 proposals", "1 node". Methods differ, so the chart legend states it per method.
 */
export function traceStepUnit(id: string, params: Record<string, unknown>): string {
  if (id === 'knapsack_dp') return '1 item row';
  if (id === 'knapsack_greedy') return '1 item';
  if (id === 'tsp_nearest_neighbor') return '1 city';
  if (id === 'tsp_held_karp') return '1 layer';
  const m = Number(params.record_every ?? (id === 'tsp_simulated_annealing' ? 100 : 1));
  const every = Number.isInteger(m) && m >= 1 ? m : 1;
  return `${int(every)} ${unit(id, every)}`;
}

export const unit = (id: string, n: number) => {
  const u = UNITS[id] ?? ['iteration', 'iterations'];
  return n === 1 ? u[0] : u[1];
};

/** What convergence certifies, per method: the badge word and the sentence's qualifier. */
const CONVERGED: Record<string, { badge: string; long: string } | undefined> = {
  knapsack_dp: { badge: 'optimal', long: 'optimal' },
  knapsack_greedy: { badge: 'pass', long: 'pass complete' },
  knapsack_branch_bound: { badge: 'optimal', long: 'optimal' },
  tsp_nearest_neighbor: { badge: 'tour', long: 'tour complete' },
  tsp_two_opt: { badge: '2-opt', long: '2-optimal' },
  tsp_or_opt: { badge: 'Or-opt', long: 'Or-optimal' },
  tsp_simulated_annealing: { badge: 'frozen', long: 'frozen' },
  tsp_held_karp: { badge: 'optimal', long: 'optimal' },
};

/**
 * The shell's status wording for a combinatorial method: its unit ("moves", "generations") and
 * what convergence certifies ("✓ 2-opt 47", "Converged in 47 moves — 2-optimal"). Pass it to
 * RunSummary items and `describeResult`, so the stage header, the card and the method page say
 * the same thing.
 */
export function statusWording(id: string): StatusWording {
  return { noun: UNITS[id] ?? ['iteration', 'iterations'], converged: CONVERGED[id] };
}

/**
 * The run as the status counts it: branch and bound counts the nodes it bounds (nFev), greedy the
 * items it examined (without the single-item check, which the evidence line names), and a GA or
 * Ant System run that stopped on a stall is reported as a stall (the shell's warning), not as
 * convergence: a stall certifies nothing.
 */
export function statusResult(id: string, result: Result): Result {
  if (id === 'knapsack_branch_bound') return { ...result, nIter: result.nFev };
  if (id === 'knapsack_greedy') {
    const fix = result.trace.some((s) => s.info.phase === 'single_item_fix');
    return fix ? { ...result, nIter: result.nIter - 1 } : result;
  }
  if ((id === 'tsp_genetic' || id === 'tsp_ant_colony') && result.converged)
    return { ...result, converged: false };
  return result;
}

/**
 * The status in words (a table cell, a test): the shell's `describeResult` with this lab's
 * wording, its short form led by what convergence certifies ("2-optimal · 47 moves", "pass
 * complete · 10 items + single-item check"; a stall: "stalled · 40 generations").
 */
export function runStatus(id: string, result: Result, error?: string): RunStatus {
  const wording = statusWording(id);
  const counted = statusResult(id, result);
  const st = describeResult(counted, error, wording);
  if (error || !counted.converged || !wording.converged) return st;
  const n = counted.nIter;
  const noun = wording.noun ?? ['iteration', 'iterations'];
  const fix =
    id === 'knapsack_greedy' && result.trace.some((s) => s.info.phase === 'single_item_fix');
  return {
    ...st,
    short: `${wording.converged.long} · ${int(n)} ${n === 1 ? noun[0] : noun[1]}${fix ? ' + single-item check' : ''}`,
  };
}

// ── Layouts ────────────────────────────────────────────────────────────────────────────

/** Columns × rows for n panels in a W × H box that maximize the side of a square panel. */
export function chooseGrid(n: number, W: number, H: number): { cols: number; rows: number } {
  let best = { cols: 1, rows: Math.max(1, n), side: -1 };
  for (let cols = 1; cols <= Math.max(1, n); cols++) {
    const rows = Math.ceil(n / cols);
    const side = Math.min(W / cols, H / rows);
    if (side > best.side + 0.5) best = { cols, rows, side };
  }
  return { cols: best.cols, rows: best.rows };
}

/** Bounding box of the cities with a margin, as [[xmin, xmax], [ymin, ymax]]. */
export function cityDomain(coords: readonly (readonly number[])[], pad = 0.08) {
  let x0 = Infinity,
    x1 = -Infinity,
    y0 = Infinity,
    y1 = -Infinity;
  for (const [x, y] of coords) {
    x0 = Math.min(x0, x);
    x1 = Math.max(x1, x);
    y0 = Math.min(y0, y);
    y1 = Math.max(y1, y);
  }
  const span = Math.max(x1 - x0, y1 - y0, 1e-9);
  const m = span * pad;
  return [
    [x0 - m, x1 + m],
    [y0 - m, y1 + m],
  ] as [[number, number], [number, number]];
}

// ── Knapsack geometry ──────────────────────────────────────────────────────────────────

/** Cells (row, d) of the DP backtrack from (k, C) up to row 0, using the trace's take rows. */
export function dpBacktrackPath(
  trace: readonly Step[],
  k: number,
  weights: readonly number[],
  C: number,
): [number, number][] {
  const path: [number, number][] = [[k, C]];
  let d = C;
  for (let row = k; row >= 1; row--) {
    const take = trace[row]?.info.take as boolean[] | undefined;
    if (take?.[d]) d -= weights[row - 1];
    path.push([row - 1, d]);
  }
  return path;
}

/** Every branch-and-bound node bounded up to (and including) trace step k. */
export function bbNodesUpTo(trace: readonly Step[], k: number): BbNode[] {
  const out: BbNode[] = [];
  for (let s = 0; s <= Math.min(k, trace.length - 1); s++)
    out.push(...((trace[s].info.nodes as BbNode[] | undefined) ?? []));
  return out;
}

/** Item fixings x_i ∈ {0, 1} on the path from the root to `node`. */
export function bbFixings(all: readonly BbNode[], node: BbNode | undefined): Map<number, number> {
  const byId = new Map(all.map((n) => [n.id, n]));
  const out = new Map<number, number>();
  let cur = node;
  while (cur && cur.item !== null && cur.decision !== null) {
    out.set(cur.item, cur.decision);
    cur = cur.parent === null ? undefined : byId.get(cur.parent);
  }
  return out;
}

/** Selected item indices of a 0/1 vector. */
export const selectedItems = (x: unknown): number[] =>
  Array.isArray(x) ? (x as number[]).flatMap((v, i) => (v ? [i] : [])) : [];

/**
 * The method name without its parenthetical qualifier ("Ant System (ant colony optimization)" →
 * "Ant System"), for panel heads too narrow for the full name (the full name stays in `title`
 * and in the accessible name).
 */
export const shortMethodName = (name: string): string =>
  name.replace(/\s*\([^)]*\)\s*$/, '') || name;
