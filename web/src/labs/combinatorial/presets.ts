/**
 * Defaults and "Try this" presets. Each preset is a claim the viewer can check on screen; the
 * numbers come from the Python reference (the TS ports replay it exactly).
 */
import type { LabPreset, MethodSelection } from '../_shell';
import type { Kind } from './model';

export const DEFAULT_PROBLEM = 'tsp_cities_20';

/**
 * First view: 2-opt untangles the index-order tour on 20 clustered cities and stops at a
 * 2-optimal tour 0.66 % above L⋆ after 47 moves, while the Ant System's pheromone concentrates
 * on the edges of short tours (0.07 % above L⋆).
 */
export const DEFAULTS: Record<Kind, MethodSelection[]> = {
  tsp: [
    { id: 'tsp_two_opt', slot: 0, params: {} },
    { id: 'tsp_ant_colony', slot: 1, params: {} },
  ],
  knapsack: [
    { id: 'knapsack_dp', slot: 0, params: {} },
    { id: 'knapsack_branch_bound', slot: 1, params: {} },
    { id: 'knapsack_greedy', slot: 2, params: {} },
  ],
};

/** Shown by the shell's "Try this" (the note under the active preset only). */
export const PRESETS: LabPreset[] = [
  {
    id: 'same-start',
    title: 'One start, three local optima',
    note: 'From the nearest-neighbor tour (6.4 % above L⋆), 2-opt stops 2.9 % above; Or-opt’s segment moves reach the optimum.',
    problem: 'tsp_random_15',
    methods: [
      { id: 'tsp_nearest_neighbor', slot: 0, params: {} },
      { id: 'tsp_two_opt', slot: 1, params: { init: 'nearest_neighbor' } },
      { id: 'tsp_or_opt', slot: 2, params: { init: 'nearest_neighbor' } },
    ],
  },
  {
    id: 'convex',
    title: 'On a circle, 2-optimal is optimal',
    note: 'Every crossing is removed by a 2-opt move, and the only tour without crossings is the polygon. Held–Karp certifies it.',
    problem: 'tsp_circle_12',
    methods: [
      { id: 'tsp_two_opt', slot: 0, params: {} },
      { id: 'tsp_held_karp', slot: 1, params: {} },
    ],
  },
  {
    id: 'trap',
    title: 'Greedy walks into the trap',
    note: 'The small item has the best ratio, so greedy packs it and only one large item still fits: 53 against z⋆ = 100.',
    problem: 'knapsack_greedy_trap',
    methods: [
      { id: 'knapsack_greedy', slot: 0, params: { single_item_fix: false } },
      { id: 'knapsack_dp', slot: 1, params: {} },
    ],
  },
  {
    id: 'prune',
    title: '35 nodes instead of 2²¹',
    note: 'The Dantzig bound prunes almost the whole include/exclude tree of 20 items; the first dive is the greedy packing.',
    problem: 'knapsack_20',
    methods: [
      { id: 'knapsack_branch_bound', slot: 0, params: {} },
      { id: 'knapsack_greedy', slot: 1, params: {} },
    ],
  },
];
