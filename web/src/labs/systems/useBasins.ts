/**
 * Basins of attraction, computed off the main thread: a small pool of workers (basins.worker.ts)
 * shares the rows of each grid. The grid is refined in passes (coarse first, so the picture appears
 * at once), and finished grids are cached per problem / method / parameters.
 */
import { useEffect, useState } from 'react';
import type { Params } from '../../core/types';
import { computeBasins, type BasinRequest, type BasinResult } from './basins';

export interface BasinGrid {
  nx: number;
  ny: number;
  labels: Int8Array;
  iters: Uint16Array;
  /** Wall time of the slowest worker for this pass (ms). */
  ms: number;
}

export interface BasinState {
  grid: BasinGrid | null;
  /** True once the finest pass is in. */
  done: boolean;
  /** Cells along the longer axis of the pass being shown. */
  resolution: number;
}

/** Cells along the longer axis of the domain, coarse to fine. */
export const PASSES = [48, 112, 184] as const;

const CACHE = new Map<string, BasinGrid>();

function gridSize(domain: [[number, number], [number, number]], n: number): [number, number] {
  const wx = domain[0][1] - domain[0][0],
    wy = domain[1][1] - domain[1][0];
  return wx >= wy
    ? [n, Math.max(8, Math.round((n * wy) / wx))]
    : [Math.max(8, Math.round((n * wx) / wy)), n];
}

function poolSize(): number {
  const hc = typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 2 : 2;
  return Math.max(1, Math.min(4, hc - 1));
}

function spawn(): Worker | null {
  try {
    return new Worker(new URL('./basins.worker.ts', import.meta.url), { type: 'module' });
  } catch {
    return null;
  }
}

/** Run one pass over the pool; rows are interleaved in blocks so the work is balanced. */
function runPass(
  workers: Worker[],
  base: Omit<BasinRequest, 'id' | 'nx' | 'ny' | 'j0' | 'j1'>,
  nx: number,
  ny: number,
): Promise<BasinGrid> {
  const labels = new Int8Array(nx * ny);
  const iters = new Uint16Array(nx * ny);
  const W = workers.length;
  const block = Math.max(1, Math.ceil(ny / (W * 4)));
  const jobs: { j0: number; j1: number }[] = [];
  for (let j = 0; j < ny; j += block) jobs.push({ j0: j, j1: Math.min(ny, j + block) });
  let ms = 0;
  const put = (r: BasinResult) => {
    labels.set(r.labels, r.j0 * nx);
    iters.set(r.iters, r.j0 * nx);
    ms += r.ms;
  };
  if (W === 0) {
    for (const job of jobs) put(computeBasins({ ...base, id: 0, nx, ny, ...job }));
    return Promise.resolve({ nx, ny, labels, iters, ms });
  }
  return new Promise((resolve, reject) => {
    let next = 0,
      done = 0;
    const feed = (w: Worker) => {
      if (next >= jobs.length) return;
      const job = jobs[next++];
      w.postMessage({ ...base, id: next, nx, ny, ...job } satisfies BasinRequest);
    };
    for (const w of workers) {
      w.onmessage = (e: MessageEvent<BasinResult>) => {
        put(e.data);
        done++;
        if (done === jobs.length) resolve({ nx, ny, labels, iters, ms: ms / W });
        else feed(w);
      };
      w.onerror = (e) => reject(e);
      feed(w);
    }
  });
}

/**
 * The basins of `methodId` (with `params`) on `problemId`, or `{grid: null}` while disabled.
 * Changing any input cancels the running computation (its workers are terminated).
 */
export function useBasins(
  enabled: boolean,
  problemId: string,
  methodId: string | null,
  params: Params,
  domain: [[number, number], [number, number]],
): BasinState {
  const key = JSON.stringify([problemId, methodId, params, domain]);
  const [state, setState] = useState<{ key: string; grid: BasinGrid; resolution: number } | null>(
    null,
  );

  useEffect(() => {
    if (!enabled || !methodId || CACHE.has(key)) return;
    const [pid, mid, prm, dom] = JSON.parse(key) as [
      string,
      string,
      Params,
      [[number, number], [number, number]],
    ];
    const workers: Worker[] = [];
    for (let i = 0; i < poolSize(); i++) {
      const w = spawn();
      if (!w) break;
      workers.push(w);
    }
    let cancelled = false;
    const base = { problemId: pid, methodId: mid, params: prm, domain: dom };
    void (async () => {
      try {
        for (const n of PASSES) {
          const [nx, ny] = gridSize(dom, n);
          // Without workers only the coarse pass runs (on the main thread).
          if (workers.length === 0 && n !== PASSES[0]) break;
          const grid = await runPass(workers, base, nx, ny);
          if (cancelled) return;
          if (n === PASSES[PASSES.length - 1]) CACHE.set(key, grid);
          setState({ key, grid, resolution: n });
        }
      } catch {
        // A failed worker leaves the last finished pass on screen.
      } finally {
        workers.forEach((w) => w.terminate());
      }
    })();
    return () => {
      cancelled = true;
      workers.forEach((w) => w.terminate());
    };
  }, [enabled, key, methodId]);

  if (!enabled || !methodId) return { grid: null, done: false, resolution: 0 };
  const cached = CACHE.get(key);
  if (cached) return { grid: cached, done: true, resolution: PASSES[PASSES.length - 1] };
  if (state && state.key === key)
    return { grid: state.grid, done: false, resolution: state.resolution };
  return { grid: null, done: false, resolution: 0 };
}
