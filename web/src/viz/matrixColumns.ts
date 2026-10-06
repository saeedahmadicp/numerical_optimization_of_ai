import { sig } from '../core/format';

/** A cell's text: `sig` with `digits`, and 0 for |v| < 10⁻¹². */
export const cellText = (v: number, digits: number) => (Math.abs(v) < 1e-12 ? '0' : sig(v, digits));

/** The widest formatted value of each column over a run's matrices, in characters. */
export function matrixColumnChars(
  matrices: readonly (readonly (readonly number[])[])[],
  digits = 4,
): number[] {
  const out: number[] = [];
  for (const m of matrices)
    for (const row of m)
      row.forEach((v, j) => {
        out[j] = Math.max(out[j] ?? 0, cellText(v, digits).length);
      });
  return out;
}
