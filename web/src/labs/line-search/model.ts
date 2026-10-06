/**
 * Everything the lab derives from one set of runs: the line φ along the shared direction 𝐩,
 * the minimizer of φ, the α window of the φ panel, the samples of φ and φ' on it, and the
 * acceptable set of each method. One pure function, so the lab memoizes it in one place.
 */
import type { Problem2D } from '../../core/types';
import type { LineSearchKind } from '../../methods/line_search/methods';
import type { LabRun } from '../_shell';
import {
  acceptableIntervals,
  chooseWindow,
  lineMinimizer,
  makeLine,
  sampleLine,
  trialOf,
  type LineFn,
  type Samples,
  type WindowMode,
} from './geometry';

export interface LineModel {
  line: LineFn | null;
  /** The search direction recorded by the runs (null when no run has a trace). */
  p: number[] | null;
  phi0: number;
  dphi0: number;
  /** At least one run made a trial step. */
  searched: boolean;
  hi: number;
  /** Longest trial step (the ray is drawn at least that far). */
  rayEnd: number;
  alphaStar: { alpha: number; phi: number } | null;
  /** min φ over the minimizer, φ(0) and every trial: the reference of the gap measure. */
  phiStar: number;
  samples: Samples | null;
  /** Acceptable α intervals in [0, hi], per run (same order as `runs`). */
  intervals: [number, number][][];
}

export function buildModel(
  problem: Problem2D,
  x0: readonly number[],
  runs: readonly LabRun[],
  mode: WindowMode,
): LineModel {
  const first = runs.find((r) => r.result.trace.length)?.result.trace[0];
  const head = first ? trialOf(first) : null;
  const p = (first?.info.direction as number[] | undefined) ?? null;
  const pOk = !!p && p.every((x) => Number.isFinite(x)) && p.some((x) => x !== 0);
  const line = pOk && p ? makeLine(problem, x0, p) : null;
  const phi0 = head?.phi0 ?? NaN;
  const dphi0 = head?.dphi0 ?? NaN;
  const trials = runs.map((r) => r.result.trace.slice(1).map((s) => trialOf(s).alpha));
  const finite = trials.flat().filter((a) => Number.isFinite(a));
  const span = Math.max(
    problem.domain[0][1] - problem.domain[0][0],
    problem.domain[1][1] - problem.domain[1][0],
  );
  const pNorm = p && pOk ? Math.hypot(...p) : 0;
  const fallback = pNorm > 0 ? (0.5 * span) / pNorm : 1;
  const scanHi = Math.max(fallback, ...finite);
  const alphaStar = line && dphi0 < 0 ? lineMinimizer(line, scanHi) : null;
  const accepted = runs.map((r) =>
    r.result.converged && typeof r.result.extra.alpha === 'number' ? r.result.extra.alpha : null,
  );
  const hi = chooseWindow(trials, accepted, alphaStar?.alpha ?? null, mode, fallback);
  const samples = line ? sampleLine(line, 0, hi, 600) : null;
  const intervals = runs.map((r) => {
    if (!line || !samples || !r.result.trace.length || !(dphi0 < 0)) return [];
    const t0 = trialOf(r.result.trace[0]);
    return acceptableIntervals(
      r.sel.id as LineSearchKind,
      line,
      samples,
      phi0,
      dphi0,
      t0.c1,
      t0.c2 ?? 0.9,
    );
  });
  const phis = runs
    .flatMap((r) => r.result.trace.map((s) => trialOf(s).phi))
    .filter((x) => Number.isFinite(x));
  const phiStar = Math.min(
    alphaStar?.phi ?? Infinity,
    Number.isFinite(phi0) ? phi0 : Infinity,
    ...phis,
  );
  return {
    line,
    p,
    phi0,
    dphi0,
    searched: runs.some((r) => r.result.trace.length > 1),
    hi,
    rayEnd: Math.max(hi, ...finite),
    alphaStar,
    phiStar,
    samples,
    intervals,
  };
}
