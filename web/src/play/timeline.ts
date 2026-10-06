/**
 * Pure timeline math shared by the player and its tests.
 *
 * The playhead `t` is continuous in [0, maxK]. Method i has `lengths[i]` steps (k = 0..len-1) and
 * stops at its own last step: its local time is `min(t, len_i - 1)`.
 */

export const SPEEDS = [0.25, 0.5, 1, 2, 4, 8] as const;

export function maxStep(lengths: readonly number[]): number {
  return Math.max(0, ...lengths.map((n) => n - 1));
}

export function localT(t: number, length: number): number {
  return Math.max(0, Math.min(t, length - 1));
}

/**
 * Steps per second at 1× speed: the whole run takes ~7 s, but never slower than 2.5 steps/s
 * (short traces stay readable) and never faster than 90 steps/s.
 */
export function baseRate(maxK: number): number {
  return Math.min(90, Math.max(2.5, maxK / 7));
}

/** Advance the playhead by `dt` seconds. With `discrete`, t stays on integer steps. */
export function advance(
  t: number,
  dt: number,
  rate: number,
  maxK: number,
  discrete: boolean,
  acc = 0,
): { t: number; acc: number } {
  if (!discrete) return { t: Math.min(maxK, t + dt * rate), acc: 0 };
  // Reduced motion: accumulate time and jump whole steps.
  const total = acc + dt * rate;
  const whole = Math.floor(total);
  return { t: Math.min(maxK, Math.floor(t) + whole), acc: total - whole };
}

/** Smooth ease for inter-step interpolation (cubic in-out). */
export function easeInOut(u: number): number {
  return u < 0.5 ? 4 * u * u * u : 1 - (-2 * u + 2) ** 3 / 2;
}

/**
 * Position along a trace at continuous time `t`: index `i = floor(t)` and eased fraction `u`.
 * Used by PathLayer and every animated view so methods move in lockstep.
 */
export function segmentAt(t: number, length: number, ease = true): { i: number; u: number } {
  const lt = localT(t, length);
  const i = Math.min(Math.floor(lt), Math.max(0, length - 1));
  const frac = lt - i;
  return { i, u: ease ? easeInOut(frac) : frac };
}

export function lerp(a: number, b: number, u: number): number {
  return a + (b - a) * u;
}
