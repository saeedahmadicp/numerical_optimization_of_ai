/** Run-stable readout widths (see Reserve.tsx). */

/**
 * The candidates for the widest value of a run: every distinct string within one character of
 * the longest. Glyph widths differ a little (superscripts, "1" in a proportional face), so the
 * Reserve stacks all of them and takes the true maximum.
 */
export function widest(values: Iterable<string>): string[] {
  const all = [...new Set(values)];
  const max = Math.max(0, ...all.map((v) => v.length));
  return all.filter((v) => v.length >= max - 1);
}
