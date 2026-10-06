/**
 * Pure helpers of the step strip (StepStrip.tsx): the window of steps shown and the bar scale.
 */
import type { BarStrip } from './stepTex';

/** Steps shown at once; longer runs page forward with the playhead. */
export const STRIP_WINDOW = 64;

/** The steps [first, last] of the window that holds step k of a run with n steps. */
export function stripWindow(n: number, k: number): [number, number] {
  if (n <= STRIP_WINDOW) return [1, Math.max(1, n)];
  const first = Math.floor((Math.max(1, k) - 1) / STRIP_WINDOW) * STRIP_WINDOW + 1;
  return [first, Math.min(n, first + STRIP_WINDOW - 1)];
}

/**
 * Bar height in [0, 1], log or linear over the whole run (and the reference level), so the scale
 * does not change when the window pages forward. On a log scale the smallest value keeps a
 * visible stub of 12 %.
 */
export function barScale(spec: BarStrip): (v: number) => number {
  const vals = spec.values.filter((v): v is number => v !== null && Number.isFinite(v));
  if (spec.log) {
    const pos = vals.filter((v) => v > 0);
    if (!pos.length) return () => 0;
    const extra = spec.ref && spec.ref.value > 0 ? [spec.ref.value] : [];
    const lo = Math.log(Math.min(...pos, ...extra));
    const hi = Math.log(Math.max(...pos, ...extra));
    if (!(hi > lo)) return () => 0.6;
    return (v) => (v > 0 ? 0.12 + (0.88 * (Math.log(v) - lo)) / (hi - lo) : 0);
  }
  const hi = Math.max(1, ...vals);
  return (v) => Math.max(0, v) / hi;
}
