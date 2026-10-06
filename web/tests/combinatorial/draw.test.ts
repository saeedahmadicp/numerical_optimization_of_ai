/**
 * Canvas drawing of the combinatorial lab against a recording fake context: stale runs (a tour
 * of another instance) must not throw, and the uniform pheromone of the Ant System at step 0
 * must not be drawn as a full graph.
 */
import { describe, expect, it } from 'vitest';
import '../../src/labs/combinatorial/setup';
import { getProblem } from '../../src/problems/registry';
import { runMethod } from '../../src/core/registry';
import type { TspInstance } from '../../src/problems/combinatorial';
import type { ChartColors } from '../../src/ui/colors';
import { drawTspPanel, type Pt } from '../../src/labs/combinatorial/tspDraw';
import { cityDomain } from '../../src/labs/combinatorial/model';

function fakeContext() {
  const calls: { name: string; style?: unknown; width?: unknown }[] = [];
  const state: Record<string, unknown> = {};
  const ctx = new Proxy(state, {
    get(target, prop: string) {
      if (prop in target) return target[prop];
      if (prop === 'measureText') return (t: string) => ({ width: t.length * 6 });
      return (...args: unknown[]) => {
        void args;
        calls.push({ name: prop, style: target.strokeStyle, width: target.lineWidth });
      };
    },
    set(target, prop: string, v) {
      target[prop] = v;
      return true;
    },
  });
  return { ctx: ctx as unknown as CanvasRenderingContext2D, calls };
}

const colors: ChartColors = {
  mode: 'light',
  surface: '#ffffff',
  grid: '#eeeeee',
  axis: '#cccccc',
  tick: '#888888',
  text: '#111111',
  text2: '#333333',
  text3: '#666666',
  iso: '#999999',
  isoStrong: '#777777',
  halo: '#ffffff',
  crosshair: '#000000',
  playhead: '#000000',
  accent: '#d97757',
  series: ['#1f6feb', '#e8710a', '#12a4a4', '#8e44ad'],
  fontSans: 'Inter',
  fontMono: 'JetBrains Mono',
  fontSerif: 'Newsreader',
  fontMath: 'KaTeX_Math',
  region: '#000000',
};

const random15 = getProblem<TspInstance>('tsp_random_15');
const circle12 = getProblem<TspInstance>('tsp_circle_12');

describe('TSP panel drawing', () => {
  it('a stale 15-city trace drawn on 12 cities does not throw', () => {
    const coords = circle12.coords.map((c) => [c[0], c[1]] as Pt);
    for (const id of ['tsp_two_opt', 'tsp_or_opt', 'tsp_ant_colony', 'tsp_genetic', 'tsp_simulated_annealing']) {
      const r = runMethod(id, random15, { seed: 0 });
      const { ctx } = fakeContext();
      for (const k of [0, Math.floor(r.trace.length / 2), r.trace.length - 1])
        expect(() =>
          drawTspPanel(ctx, 400, 400, {
            coords,
            domain: cityDomain(coords),
            colors,
            slot: 0,
            methodId: id,
            trace: r.trace,
            k,
            frac: 0.3,
            labels: true,
            schedule: { t0: 10, tMin: 0.01 },
          }),
        ).not.toThrow();
    }
  });
  it('Ant System step 0: the uniform pheromone is not drawn edge by edge', () => {
    const r = runMethod('tsp_ant_colony', circle12, { seed: 0 });
    const coords = circle12.coords.map((c) => [c[0], c[1]] as Pt);
    const draw = (k: number) => {
      const { ctx, calls } = fakeContext();
      drawTspPanel(ctx, 400, 400, {
        coords,
        domain: cityDomain(coords),
        colors,
        slot: 1,
        methodId: 'tsp_ant_colony',
        trace: r.trace,
        k,
        frac: 0,
      });
      // Pheromone edges are stroked in the series color with an rgba alpha.
      return calls.filter((c) => c.name === 'stroke' && String(c.style).startsWith('rgba(232, 113, 10')).length;
    };
    expect(draw(0)).toBe(0);
    expect(draw(Math.min(2, r.trace.length - 1))).toBeGreaterThan(0);
  });
});
