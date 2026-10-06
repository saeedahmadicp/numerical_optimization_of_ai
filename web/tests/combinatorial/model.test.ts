/** Pure helpers of the combinatorial lab (src/labs/combinatorial/model.ts, docs.ts). */
import { describe, expect, it } from 'vitest';
import '../../src/labs/combinatorial/setup';
import { getProblem } from '../../src/problems/registry';
import { getMethod, runMethod } from '../../src/core/registry';
import type { KnapsackInstance, TspInstance } from '../../src/problems/combinatorial';
import {
  CITY_EDITS_CODEC,
  applyCityEdits,
  bbFixings,
  bbNodesUpTo,
  chooseGrid,
  dpBacktrackPath,
  exactReference,
  gapPercent,
  runStatus,
  setCityEdit,
  traceStepUnit,
  withCapacity,
} from '../../src/labs/combinatorial/model';
import { columns, filledRule, quantities, tn } from '../../src/labs/combinatorial/docs';
import type { BbNode } from '../../src/methods/combinatorial/knapsack';

const ks10 = getProblem<KnapsackInstance>('knapsack_10');
const circle = getProblem<TspInstance>('tsp_circle_12');

describe('edited instances', () => {
  it('city edits round-trip through the URL codec, sorted and rounded', () => {
    const e = setCityEdit(setCityEdit([], 7, 12.345, 80), 3, 45.5, 60.04);
    expect(e).toEqual([
      [3, 45.5, 60],
      [7, 12.3, 80],
    ]);
    expect(CITY_EDITS_CODEC.parse(CITY_EDITS_CODEC.format(e))).toEqual(e);
    expect(CITY_EDITS_CODEC.parse('x:1:2')).toBeUndefined();
  });
  it('applyCityEdits gives a new id and an unknown optimum', () => {
    const p = applyCityEdits(circle, [[0, 50, 50]]);
    expect(p.id).not.toBe(circle.id);
    expect(p.coords[0]).toEqual([50, 50]);
    expect(p.optimumLength).toBeNull();
    expect(applyCityEdits(circle, [[99, 1, 1]])).toBe(circle);
  });
  it('the exact reference of an edited instance is computed in the browser', () => {
    const p = applyCityEdits(circle, [[0, 50, 50]]);
    const ref = exactReference(p);
    expect(ref.source).toBe('held-karp');
    expect(ref.value).toBeCloseTo(runMethod('tsp_held_karp', p).fun!, 12);
    const k = withCapacity(ks10, 100);
    expect(exactReference(k)).toEqual({ value: runMethod('knapsack_dp', k).fun, source: 'dp' });
    expect(withCapacity(ks10, ks10.capacity)).toBe(ks10);
  });
});

describe('geometry', () => {
  it('the DP backtrack path ends at the selection of the run', () => {
    const r = runMethod('knapsack_dp', ks10);
    const path = dpBacktrackPath(r.trace, 10, ks10.weights, ks10.capacity);
    expect(path[0]).toEqual([10, ks10.capacity]);
    expect(path[path.length - 1][0]).toBe(0);
    // Every diagonal step is an item of x; the weight drops by exactly that item's weight.
    const packed: number[] = [];
    for (let s = 0; s < path.length - 1; s++) {
      const [row, d] = path[s],
        [, d2] = path[s + 1];
      if (d !== d2) {
        packed.push(row - 1);
        expect(d - d2).toBe(ks10.weights[row - 1]);
      }
    }
    expect(packed.sort((a, b) => a - b)).toEqual(
      (r.x as number[]).flatMap((v, i) => (v ? [i] : [])),
    );
  });
  it('branch-and-bound fixings follow the path to the root', () => {
    const r = runMethod('knapsack_branch_bound', ks10);
    const all = bbNodesUpTo(r.trace, r.trace.length - 1);
    expect(all).toHaveLength(34);
    const leaf = all.find((n: BbNode) => n.status === 'leaf')!;
    expect(bbFixings(all, leaf).size).toBe(leaf.depth);
  });
  it('chooseGrid picks the layout with the largest square panel', () => {
    expect(chooseGrid(2, 1000, 600)).toEqual({ cols: 2, rows: 1 });
    expect(chooseGrid(2, 380, 700)).toEqual({ cols: 1, rows: 2 });
    expect(chooseGrid(4, 900, 700)).toEqual({ cols: 2, rows: 2 });
  });
  it('gaps: open paths have none, tours and selections do', () => {
    const nn = runMethod('tsp_nearest_neighbor', circle);
    expect(gapPercent('tsp', nn.trace[3], 248.466)).toBeNull();
    expect(gapPercent('tsp', nn.trace[nn.trace.length - 1], nn.fun)).toBe(0);
    const g = runMethod('knapsack_greedy', ks10);
    expect(gapPercent('knapsack', g.trace[g.trace.length - 1], 173)).toBeCloseTo((1 / 173) * 100, 12);
  });
});

describe('teaching data', () => {
  it('every method has a filled rule, quantities and columns at every step', () => {
    for (const [id, p] of [
      ['knapsack_dp', ks10],
      ['knapsack_greedy', ks10],
      ['knapsack_branch_bound', ks10],
      ['tsp_nearest_neighbor', circle],
      ['tsp_two_opt', circle],
      ['tsp_or_opt', circle],
      ['tsp_simulated_annealing', circle],
      ['tsp_genetic', circle],
      ['tsp_ant_colony', circle],
      ['tsp_held_karp', circle],
    ] as const) {
      const r = runMethod(id, p);
      const kind = p.kind;
      const ctx = { problem: p, reference: 100 };
      for (const s of r.trace) {
        expect(filledRule(id, s, ctx), `${id} k=${s.k}`).toBeTruthy();
        expect(quantities(id, s, ctx, kind).length).toBeGreaterThanOrEqual(3);
        for (const c of columns(id, ctx, kind)) expect(c.value(s, 0)).toBeDefined();
      }
      // A stall certifies nothing: the warning style (the GA and the ant colony stop on one).
      const stalls = id === 'tsp_genetic' || id === 'tsp_ant_colony';
      expect(runStatus(id, r).tone).toBe(stalls ? 'warn' : 'good');
      if (stalls) expect(runStatus(id, r).icon).toBe('△');
    }
  });
  it('tn typesets numbers for KaTeX', () => {
    expect(tn(-19.549)).toBe('-19.55');
    expect(tn(1.2e-8)).toBe('1.2 \\times 10^{-8}');
    expect(tn(null)).toBe('\\text{—}');
  });
});

describe('review fixes', () => {
  it('greedy counts examined items apart from the single-item check', () => {
    const fix = runMethod('knapsack_greedy', ks10);
    expect(fix.nIter).toBe(ks10.values.length + 1);
    expect(runStatus('knapsack_greedy', fix).short).toBe(
      `pass complete · ${ks10.values.length} items + single-item check`,
    );
    const plain = runMethod('knapsack_greedy', ks10, { single_item_fix: false });
    expect(runStatus('knapsack_greedy', plain).short).toBe(`pass complete · ${ks10.values.length} items`);
  });
  it('branch and bound reports nodes (nFev), as the pill and the legend unit say', () => {
    const r = runMethod('knapsack_branch_bound', ks10);
    expect(runStatus('knapsack_branch_bound', r).short).toContain(`${r.nFev} node`);
  });
  it('each method states what one recorded trace step is', () => {
    expect(traceStepUnit('tsp_simulated_annealing', {})).toBe('100 proposals');
    expect(traceStepUnit('tsp_simulated_annealing', { record_every: 1 })).toBe('1 proposal');
    expect(traceStepUnit('tsp_two_opt', {})).toBe('1 move');
    expect(traceStepUnit('tsp_genetic', { record_every: 5 })).toBe('5 generations');
    expect(traceStepUnit('knapsack_branch_bound', {})).toBe('1 node');
    expect(traceStepUnit('knapsack_dp', {})).toBe('1 item row');
    expect(traceStepUnit('tsp_held_karp', {})).toBe('1 layer');
  });
  it('the DP rule names the 0-based item of row j (w_{j-1}, v_{j-1})', () => {
    const r = runMethod('knapsack_dp', ks10);
    const s = r.trace[5];
    const tex = filledRule('knapsack_dp', s, { problem: ks10, reference: null })!;
    expect(tex).toContain(`w_{4} = ${ks10.weights[4]}`);
    expect(tex).toContain(`v_{4} = ${ks10.values[4]}`);
    expect(getMethod('knapsack_dp').doc?.rule).toContain('w_{j-1}');
  });
  it('the Ant System deposits only on the edges of each ant tour', () => {
    expect(getMethod('tsp_ant_colony').doc?.rule).toContain('[(i,j) \\in T_a]');
  });
  it('the gap quantity names the best found when the optimum is unknown', () => {
    const r = runMethod('tsp_two_opt', circle);
    const q = quantities('tsp_two_opt', r.trace[0], { problem: circle, reference: 1, bestFound: true }, 'tsp');
    expect(q.some((x) => x.tex.includes('L_{\\text{best}}'))).toBe(true);
  });
});
