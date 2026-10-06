/**
 * The 0-1 knapsack problem — TS port of `numopt.combinatorial.knapsack` (read it for the full
 * docstrings): dynamic programming, greedy (Ext-Greedy), and depth-first branch and bound.
 *
 * Problem (Martello & Toth 1990, §2.1): maximize z = Σᵢ vᵢxᵢ subject to Σᵢ wᵢxᵢ ≤ C, xᵢ ∈ {0, 1},
 * with integer vᵢ ≥ 0, wᵢ ≥ 1, C ≥ 0. Every method returns `x` = the 0/1 selection (original
 * item order) and `fun` = its value; `extra` holds `weight`, `items`, `lp_bound` and, when known,
 * `optimum_value` (greedy adds `greedy_value`, `best_single_item`; branch and bound adds `nodes`,
 * `upper_bound`). `info` keys are the Python ones, snake_case (see the module docstring there).
 *
 * Integer arithmetic: values and weights are JS numbers, exact while every total stays below
 * 2⁵³ (Python uses unbounded ints; the library instances are far below that bound).
 */
import { param, registerMethod } from '../../core/registry';
import type { MethodFn, Result, Step } from '../../core/types';
import { isKnapsack, type KnapsackInstance } from '../../problems/combinatorial';

/** Refuse DP tables with more cells than this (memory and trace size). */
export const MAX_DP_CELLS = 5_000_000;

function validate(problem: unknown): KnapsackInstance {
  if (!isKnapsack(problem)) throw new TypeError('problem must be a numopt KnapsackInstance');
  const { values, weights, capacity, id } = problem;
  if (values.length !== weights.length)
    throw new Error(`${id}: ${values.length} values but ${weights.length} weights`);
  if (values.length === 0) throw new Error(`${id}: the instance has no items`);
  const isInt = (a: unknown) => typeof a === 'number' && Number.isInteger(a);
  if (!values.every((v) => isInt(v) && v >= 0))
    throw new Error(`${id}: values must be non-negative integers`);
  if (!weights.every((w) => isInt(w) && w >= 1))
    throw new Error(`${id}: weights must be positive integers`);
  if (!(isInt(capacity) && capacity >= 0))
    throw new Error(`${id}: capacity must be a non-negative integer`);
  if (values.reduce((a, b) => a + b, 0) > Number.MAX_VALUE)
    throw new Error(`${id}: the total value Σ vᵢ exceeds the float range`);
  return problem;
}

/** Item indices sorted by vᵢ/wᵢ descending; exact cross-multiplication, ties by index. */
export function ratioOrder(values: readonly number[], weights: readonly number[]): number[] {
  const idx = values.map((_, i) => i);
  return idx.sort((i, j) => {
    const lhs = values[i] * weights[j],
      rhs = values[j] * weights[i];
    if (lhs !== rhs) return lhs > rhs ? -1 : 1;
    return i - j;
  });
}

/**
 * Dantzig's LP bound on the best value from items `order[start:]` in capacity `residual`:
 * fill in ratio order until the split item s no longer fits, then add `residual · v_s / w_s`.
 * Returns `[exact LP bound, floor U₁]` (the floor in exact integer arithmetic).
 */
export function dantzigBound(
  values: readonly number[],
  weights: readonly number[],
  order: readonly number[],
  start: number,
  residual: number,
): [number, number] {
  let total = 0;
  for (let pos = start; pos < order.length; pos++) {
    const i = order[pos];
    const w = weights[i];
    if (w <= residual) {
      residual -= w;
      total += values[i];
    } else {
      const num = residual * values[i];
      return [total + num / w, total + (num - (num % w)) / w];
    }
  }
  return [total, total];
}

function selectionResult(
  method: string,
  inst: KnapsackInstance,
  x: number[],
  converged: boolean,
  message: string,
  nIter: number,
  nFev: number,
  trace: Step[],
  extra: Record<string, unknown> = {},
): Result {
  let value = 0,
    weight = 0;
  const items: number[] = [];
  x.forEach((xi, i) => {
    if (xi) {
      value += inst.values[i];
      weight += inst.weights[i];
      items.push(i);
    }
  });
  const [lp] = dantzigBound(
    inst.values,
    inst.weights,
    ratioOrder(inst.values, inst.weights),
    0,
    inst.capacity,
  );
  const info: Record<string, unknown> = { weight, items, lp_bound: lp, ...extra };
  if (inst.optimumValue !== null) info.optimum_value = inst.optimumValue;
  return {
    method,
    x,
    fun: value,
    converged,
    message,
    nIter,
    nFev,
    nGev: 0,
    nHev: 0,
    trace,
    extra: info,
  };
}

const step = (k: number, x: number[], fun: number, info: Record<string, unknown>): Step => ({
  k,
  x,
  fun,
  gradNorm: null,
  stepSize: null,
  info,
});

// ── Dynamic programming ────────────────────────────────────────────────────────────────

/**
 * Bellman DP (Kellerer et al. 2004, §2.3): z_j(d) = z_{j−1}(d) if d < w_j, else
 * max(z_{j−1}(d), z_{j−1}(d − w_j) + v_j); take_j(d) records a strict improvement and the
 * optimum is backtracked from (n, C). Step k = j is the table row after item j − 1.
 */
export const knapsackDp: MethodFn<KnapsackInstance> = (problem) => {
  const inst = validate(problem);
  const n = inst.values.length,
    C = inst.capacity;
  if ((n + 1) * (C + 1) > MAX_DP_CELLS)
    throw new Error(`${inst.id}: DP table of ${(n + 1) * (C + 1)} cells exceeds ${MAX_DP_CELLS}`);
  const values = inst.values,
    weights = inst.weights;
  let z = new Array<number>(C + 1).fill(0);
  const take: boolean[][] = Array.from({ length: n }, () => new Array<boolean>(C + 1).fill(false));

  const backtrack = (rows: number): number[] => {
    const x = new Array<number>(n).fill(0);
    let d = C;
    for (let j = rows - 1; j >= 0; j--) {
      if (take[j][d]) {
        x[j] = 1;
        d -= weights[j];
      }
    }
    return x;
  };

  const trace: Step[] = [
    step(0, new Array<number>(n).fill(0), 0, {
      item: null,
      weight: null,
      value: null,
      capacity: C,
      table_row: [...z],
      take: new Array<boolean>(C + 1).fill(false),
    }),
  ];
  for (let j = 0; j < n; j++) {
    const w = weights[j],
      v = values[j];
    if (w <= C) {
      const next = [...z];
      for (let d = w; d <= C; d++) {
        const candidate = z[d - w] + v;
        if (candidate > z[d]) {
          take[j][d] = true;
          next[d] = candidate;
        }
      }
      z = next;
    }
    trace.push(
      step(j + 1, backtrack(j + 1), z[C], {
        item: j,
        weight: w,
        value: v,
        capacity: C,
        table_row: [...z],
        take: [...take[j]],
      }),
    );
  }
  return selectionResult(
    'knapsack_dp',
    inst,
    backtrack(n),
    true,
    `DP table complete (${n} × ${C + 1}): z⋆ = zₙ(C) = ${z[C]}`,
    n,
    n * (C + 1),
    trace,
  );
};

// ── Greedy ─────────────────────────────────────────────────────────────────────────────

/**
 * Greedy by value/weight ratio (Kellerer et al. 2004, §2.1): scan the ratio order once and pack
 * every item that still fits; with `single_item_fix` (Ext-Greedy) keep the better of that and the
 * most valuable single item that fits, which guarantees z ≥ z⋆/2.
 */
export const knapsackGreedy: MethodFn<KnapsackInstance> = (problem, opts) => {
  const singleItemFix = opts.single_item_fix === undefined ? true : Boolean(opts.single_item_fix);
  const inst = validate(problem);
  const n = inst.values.length,
    C = inst.capacity;
  const values = inst.values,
    weights = inst.weights;
  const order = ratioOrder(values, weights);

  let x = new Array<number>(n).fill(0);
  let residual = C,
    value = 0,
    weight = 0;
  const trace: Step[] = [
    step(0, [...x], 0, {
      order,
      phase: 'start',
      item: null,
      ratio: null,
      fits: null,
      taken: null,
      residual,
      value: 0,
      weight: 0,
    }),
  ];
  order.forEach((i, idx) => {
    const fits = weights[i] <= residual;
    if (fits) {
      x[i] = 1;
      residual -= weights[i];
      value += values[i];
      weight += weights[i];
    }
    trace.push(
      step(idx + 1, [...x], value, {
        order,
        phase: 'greedy',
        item: i,
        ratio: values[i] / weights[i],
        fits,
        taken: fits,
        residual,
        value,
        weight,
      }),
    );
  });
  const greedyValue = value;
  let nIter = n,
    nFev = n;
  let bestSingle: number | null = null;
  let msg: string;
  if (singleItemFix) {
    for (let i = 0; i < n; i++) {
      if (weights[i] <= C && (bestSingle === null || values[i] > values[bestSingle]))
        bestSingle = i;
    }
    nFev += n;
    nIter += 1;
    const switched = bestSingle !== null && values[bestSingle] > greedyValue;
    if (switched && bestSingle !== null) {
      x = new Array<number>(n).fill(0);
      x[bestSingle] = 1;
      value = values[bestSingle];
      weight = weights[bestSingle];
      residual = C - weights[bestSingle];
    }
    trace.push(
      step(nIter, [...x], value, {
        order,
        phase: 'single_item_fix',
        item: bestSingle,
        ratio: bestSingle === null ? null : values[bestSingle] / weights[bestSingle],
        fits: bestSingle !== null,
        taken: switched,
        residual,
        value,
        weight,
      }),
    );
    msg =
      `Ext-Greedy complete: greedy value ${greedyValue}, best single item ` +
      `${bestSingle === null ? 0 : values[bestSingle]}; kept ${value} ` +
      '(heuristic: guaranteed ≥ ½ of the optimum)';
  } else {
    msg = `greedy pass complete: value ${value} (heuristic: no approximation guarantee)`;
  }
  return selectionResult('knapsack_greedy', inst, x, true, msg, nIter, nFev, trace, {
    greedy_value: greedyValue,
    best_single_item: bestSingle,
  });
};

// ── Branch and bound ───────────────────────────────────────────────────────────────────

/** One bounded node of the depth-first search (`info.nodes[]`, Python keys). */
export interface BbNode {
  id: number;
  parent: number | null;
  depth: number;
  item: number | null;
  decision: number | null;
  value: number;
  weight: number;
  bound: number;
  status: 'branched' | 'pruned' | 'leaf';
  incumbent: boolean;
}

type StackEntry = [
  depth: number,
  value: number,
  weight: number,
  items: number[],
  parent: number | null,
  item: number | null,
  decision: number | null,
];

function bbStep(
  k: number,
  x: number[],
  bestValue: number,
  order: number[],
  nodes: BbNode[],
  nOpen: number,
): Step {
  return step(k, x, bestValue, { order, nodes, best_value: bestValue, open_nodes: nOpen });
}

/**
 * Depth-first branch and bound with the Dantzig bound (Horowitz & Sahni 1974; Martello & Toth
 * 1990, §2.5.1): items in ratio order, include child explored first, a node is pruned when
 * U₁ = ⌊V + LP bound of the rest⌋ ≤ z. Converged when the stack empties, or when `max_nodes`
 * is reached with every open node dominated (U₁ ≤ z).
 */
export const knapsackBranchBound: MethodFn<KnapsackInstance> = (problem, opts) => {
  const maxNodes = Number(opts.max_nodes ?? 100_000);
  const recordEvery = Number(opts.record_every ?? 1);
  const inst = validate(problem);
  if (Math.trunc(maxNodes) < 1 || Math.trunc(recordEvery) < 1)
    throw new Error('max_nodes and record_every must be ≥ 1');
  const n = inst.values.length,
    C = inst.capacity;
  const values = inst.values,
    weights = inst.weights;
  const order = ratioOrder(values, weights);

  const stack: StackEntry[] = [[0, 0, 0, [], null, null, null]];
  let bestValue = 0;
  let bestItems: number[] = [];
  const trace: Step[] = [];
  let pending: BbNode[] = [];
  let k = -1;

  const selection = (items: readonly number[]) => {
    const x = new Array<number>(n).fill(0);
    for (const i of items) x[i] = 1;
    return x;
  };

  while (stack.length && k + 1 < maxNodes) {
    const [depth, value, weight, items, parent, item, decision] = stack.pop()!;
    k += 1;
    const [rest, restFloor] = dantzigBound(values, weights, order, depth, C - weight);
    const bound = value + rest,
      boundFloor = value + restFloor;
    const incumbent = value > bestValue;
    if (incumbent) {
      bestValue = value;
      bestItems = items;
    }
    let status: BbNode['status'];
    if (depth === n) status = 'leaf';
    else if (boundFloor <= bestValue) status = 'pruned';
    else {
      status = 'branched';
      const j = order[depth];
      stack.push([depth + 1, value, weight, items, k, j, 0]);
      if (weights[j] <= C - weight)
        stack.push([depth + 1, value + values[j], weight + weights[j], [...items, j], k, j, 1]);
    }
    pending.push({ id: k, parent, depth, item, decision, value, weight, bound, status, incumbent });
    if (k % recordEvery === 0 || !stack.length) {
      trace.push(bbStep(k, selection(bestItems), bestValue, order, pending, stack.length));
      pending = [];
    }
  }
  if (pending.length)
    trace.push(bbStep(k, selection(bestItems), bestValue, order, pending, stack.length));

  const x = selection(bestItems);
  if (!stack.length) {
    return selectionResult(
      'knapsack_branch_bound',
      inst,
      x,
      true,
      `search tree exhausted after ${k + 1} nodes: the incumbent z = ${bestValue} is optimal`,
      k,
      k + 1,
      trace,
      { nodes: k + 1, upper_bound: bestValue },
    );
  }
  let openBound = -Infinity;
  for (const [d, v, w] of stack)
    openBound = Math.max(openBound, v + dantzigBound(values, weights, order, d, C - w)[1]);
  const upper = Math.max(bestValue, openBound);
  const proven = upper <= bestValue;
  const message = proven
    ? `reached max_nodes=${maxNodes} with ${stack.length} open nodes, all dominated ` +
      `(U₁ ≤ z): the incumbent z = ${bestValue} is optimal`
    : `reached max_nodes=${maxNodes} with ${stack.length} open nodes: incumbent ` +
      `${bestValue}, upper bound ${upper}`;
  return selectionResult('knapsack_branch_bound', inst, x, proven, message, k, k + 1, trace, {
    nodes: k + 1,
    upper_bound: upper,
  });
};

// ── Registration ───────────────────────────────────────────────────────────────────────

const KPP = 'Kellerer, Pferschy & Pisinger (2004), Knapsack Problems';
const MT = 'Martello & Toth (1990), Knapsack Problems';

registerMethod(
  {
    id: 'knapsack_dp',
    family: 'combinatorial',
    name: 'Knapsack dynamic programming',
    params: [],
    needs: ['knapsack'],
    order: 'exact, O(n·C) time and memory (pseudo-polynomial)',
    summary: 'Fill a table of the best value for every capacity, one item at a time.',
    references: ['Bellman (1957), Dynamic Programming, Ch. 1', `${MT}, §2.6`, `${KPP}, §2.3`],
  },
  knapsackDp,
  {
    // Items are numbered from 0 (as in the data), so row j adds item j − 1.
    rule: 'z_j(d) = \\max\\bigl(z_{j-1}(d),\\; z_{j-1}(d - w_{j-1}) + v_{j-1}\\bigr)',
    intuition:
      'Row j of the table answers every capacity at once: the best value using items 0 to ' +
      'j − 1. Each cell either skips item j − 1 (the cell above) or packs it (the cell ' +
      'wⱼ₋₁ to the left in the row above, plus vⱼ₋₁).',
    order: 'exact · O(n·C)',
    pros: ['Exact, with a certificate: the whole table', 'Simple, no search'],
    cons: ['Pseudo-polynomial: time and memory grow with C, not with log C'],
  },
);

registerMethod(
  {
    id: 'knapsack_greedy',
    family: 'combinatorial',
    name: 'Greedy by value/weight ratio',
    params: [
      param.bool('single_item_fix', true, {
        help:
          'Also try the most valuable single item and keep the better solution ' +
          '(Ext-Greedy, value ≥ ½ z⋆).',
        label: 'Best single-item fix',
      }),
    ],
    needs: ['knapsack'],
    order: 'heuristic, O(n log n); ½-approximation with the single-item fix',
    summary: 'Pack items in order of value per unit weight while they fit.',
    references: [`${KPP}, §2.1 (Greedy, Ext-Greedy)`, `${MT}, §2.4`],
  },
  knapsackGreedy,
  {
    rule: 'x_i = \\bigl[\\, w_i \\le r \\,\\bigr], \\quad r \\leftarrow r - w_i x_i \\quad (i \\text{ by } v_i/w_i \\downarrow)',
    intuition:
      'Sort the items by value per unit weight and pack each one that still fits. One small, ' +
      'dense item can block a large one, so the fix also tries the best single item.',
    order: 'heuristic · ½-approx. with the fix',
    pros: ['O(n log n), one pass', 'Ext-Greedy: z ≥ z⋆/2'],
    cons: ['No guarantee without the fix: arbitrarily bad on the trap instance'],
  },
);

registerMethod(
  {
    id: 'knapsack_branch_bound',
    family: 'combinatorial',
    name: 'Branch and bound (depth-first, Dantzig bound)',
    params: [
      param.int('max_nodes', 100_000, {
        min: 1,
        max: 10_000_000,
        help: 'Stop (not converged) after bounding this many nodes.',
        label: 'Node budget',
      }),
      param.int('record_every', 1, {
        min: 1,
        max: 10_000,
        help: 'Record one trace step every this many nodes (the first and last are always recorded).',
        label: 'Record every',
      }),
    ],
    needs: ['knapsack'],
    order: 'exact, exponential worst case',
    summary:
      'Search the include/exclude tree depth-first and cut subtrees whose LP bound ' +
      'cannot beat the best solution found.',
    references: [
      'Horowitz & Sahni (1974), J. ACM 21(2), 277–292',
      `${MT}, §2.5.1 (Horowitz–Sahni) and §2.2.1 (bound U₁)`,
      `${KPP}, §2.4`,
    ],
  },
  knapsackBranchBound,
  {
    rule: 'U_1 = \\Bigl\\lfloor V + \\textstyle\\sum_{\\text{fits}} v_i + r\\,\\tfrac{v_s}{w_s} \\Bigr\\rfloor \\le z \\;\\Rightarrow\\; \\text{prune}',
    intuition:
      'Decide the items one at a time, packing first. The LP relaxation fills the rest by ' +
      'ratio and cuts the split item; if even that cannot beat the best packing found, the ' +
      'whole subtree is skipped.',
    order: 'exact · exponential worst case',
    pros: ['Exact, usually explores a tiny part of the 2ⁿ tree', 'First dive = greedy'],
    cons: ['Exponential in the worst case (weakly correlated, large n)'],
  },
);
