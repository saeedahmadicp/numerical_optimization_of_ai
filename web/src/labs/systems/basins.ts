/**
 * Basins of attraction: run a method from every cell center of an nx × ny grid over the problem's
 * domain and record which root it reached and how many iterations it took. The same function runs
 * in the worker (basins.worker.ts) and, as a fallback, on the main thread.
 *
 * Labels: 0, 1, … = the known root `roots[i]` (within 10⁻⁶ relative); OTHER_ROOT = converged to a
 * root that is not in the list (e.g. another period of the trigonometric system); NO_ROOT = the run
 * stopped without converging (budget, singular Jacobian, line-search failure, divergence).
 */
import { getMethod } from '../../core/registry';
import { getProblem } from '../../problems/registry';
import type { Params, Vector } from '../../core/types';
import { rootIndex } from './geometry';

export const OTHER_ROOT = -2;
export const NO_ROOT = -1;

export interface BasinRequest {
  id: number;
  problemId: string;
  methodId: string;
  params: Params;
  domain: [[number, number], [number, number]];
  nx: number;
  ny: number;
  /** Rows [j0, j1) of the grid (default: all), so a pool of workers can share one grid. */
  j0?: number;
  j1?: number;
}

export interface BasinResult {
  id: number;
  nx: number;
  ny: number;
  j0: number;
  j1: number;
  /** Rows j0 … j1 − 1, row-major from the bottom row (y = y_min) up; one label per cell. */
  labels: Int8Array;
  /** Iterations used per cell. */
  iters: Uint16Array;
  /** Wall time in ms (for the caption). */
  ms: number;
}

export function computeBasins(req: BasinRequest): BasinResult {
  const t0 = performance.now();
  const { nx, ny, domain } = req;
  const problem = getProblem<{ roots?: Vector[] }>(req.problemId);
  const roots = problem.roots ?? [];
  const { spec, fn } = getMethod(req.methodId);
  const params = {
    ...Object.fromEntries(spec.params.map((p) => [p.name, p.default])),
    ...req.params,
  };
  const j0 = req.j0 ?? 0,
    j1 = req.j1 ?? ny;
  const labels = new Int8Array(nx * (j1 - j0));
  const iters = new Uint16Array(nx * (j1 - j0));
  const [[x0, x1], [y0, y1]] = domain;
  for (let j = j0; j < j1; j++) {
    const y = y0 + ((j + 0.5) / ny) * (y1 - y0);
    for (let i = 0; i < nx; i++) {
      const x = x0 + ((i + 0.5) / nx) * (x1 - x0);
      let label = NO_ROOT,
        n = 0;
      try {
        const r = fn(problem, { ...params, x0: [x, y] });
        n = r.nIter;
        if (r.converged) {
          const k = rootIndex(r.x as Vector, roots);
          label = k >= 0 ? k : OTHER_ROOT;
        }
      } catch {
        label = NO_ROOT;
      }
      labels[(j - j0) * nx + i] = label;
      iters[(j - j0) * nx + i] = Math.min(65535, n);
    }
  }
  return { id: req.id, nx, ny, j0, j1, labels, iters, ms: performance.now() - t0 };
}
