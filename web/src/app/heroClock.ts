/** The hero's shared clock (docs/brand/scripts/make_hero.py, `k_of_tau`). Pure; tested. */
export const T_PRE = 0.5;
export const T_DRAW = 9.0;
export const T_END = T_PRE + T_DRAW;

/** k rises 0 → 1 during T_PRE, then log k is linear in time up to log kMax at T_END. */
export function kOfTau(tau: number, kMax = 20_000): number {
  if (tau < T_PRE) return Math.max(0, tau) / T_PRE;
  return kMax ** Math.min(1, (tau - T_PRE) / T_DRAW);
}
