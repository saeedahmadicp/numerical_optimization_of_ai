/**
 * Shared helpers of the roots lab: the method's estimate after a step, and the timing of one
 * playback step.
 *
 * Estimate. Brent, Chandrupatla and ITP evaluate a trial point x_k but keep as their estimate
 * the bracket end with the smaller |f| (`info.best`, Python's b / x_m; `Result.x` is the last
 * one). Near the end the trial point is a deliberate probe a tolerance away from that end, so
 * |x_k − x⋆| can rise by several decades on the last step while the returned answer is exact to
 * the last ulp. Error measures therefore use `info.best` when the step has one.
 *
 * Timing. Each playback step [k, k+1) first holds step k at rest, then cross-fades the
 * geometry to step k+1, then moves the iterate along the model to x_{k+1}. The MethodCard, the
 * table row and the "k" readout switch to step k+1 halfway through the cross-fade
 * (`shownStep`), so text and figure always describe the same step.
 */
import type { Step } from '../../core/types';

/** The method's estimate of the root after step `s`: `info.best` when present, else x_k. */
export function estimateOf(s: Step): number {
  const b = s.info.best;
  return typeof b === 'number' ? b : (s.x as number);
}

/** True when step `s` keeps an estimate other than its own trial point x_k. */
export function isProbe(s: Step): boolean {
  const b = s.info.best;
  return typeof b === 'number' && b !== s.x;
}

/** Fraction of a step spent at rest on step k. */
export const HOLD = 0.4;
/** Fraction of a step spent cross-fading the geometry from step k to step k+1. */
export const FADE = 0.2;

export const ease = (u: number) => (u <= 0 ? 0 : u >= 1 ? 1 : u * u * (3 - 2 * u));

/**
 * Phases of the fractional part u ∈ [0, 1) of the playhead: `fade` (0 → 1) is the weight of
 * step k+1's geometry, `travel` (0 → 1) the progress of the iterate from x_k to x_{k+1}.
 */
export function phase(u: number): { fade: number; travel: number } {
  return {
    fade: ease((u - HOLD) / FADE),
    travel: ease((u - HOLD - FADE) / (1 - HOLD - FADE)),
  };
}

/** The step the figure shows at playhead t: k until halfway through the cross-fade, then k+1. */
export function shownStep(t: number): number {
  if (!Number.isFinite(t)) return t;
  const k = Math.floor(t + 1e-9);
  return t - k >= HOLD + FADE / 2 ? k + 1 : k;
}
