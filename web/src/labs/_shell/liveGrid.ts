/**
 * Column spans for the live-quantities grid (3 columns): a wide cell takes 2, and the cell before
 * a gap (a wide cell that does not fit the rest of its row, or the end of the grid) stretches to
 * the row's end, so the grid never shows an empty cell. The order of the cells is kept.
 */
export function liveSpans(wide: readonly boolean[], cols = 3): number[] {
  const out = wide.map((w) => (w ? Math.min(2, cols) : 1));
  let col = 0;
  let prev = -1;
  for (let i = 0; i < out.length; i++) {
    if (col + out[i] > cols) {
      if (prev >= 0) out[prev] += cols - col;
      col = 0;
    }
    col += out[i];
    prev = i;
    if (col === cols) col = 0;
  }
  if (col > 0 && prev >= 0) out[prev] += cols - col;
  return out;
}
