/** Pure geometry for the charts (tested in tests/platform.test.ts). */
import { niceDomain } from './scales';

/** A slope triangle on log-log axes: a run of one decade and a rise of `slope` decades. */
export interface SlopeGuide {
  slope: number;
  /** Lower-left corner (data coordinates); default: inside the bottom-right of the plot. */
  at?: readonly [number, number];
  /** Run in decades (default 1). */
  decades?: number;
  /** Label next to the rise (default: the slope, e.g. "2"). */
  label?: string;
}

/** Suggest a log-k axis: the longest run is ≥ 30× the shortest (brand.md §9). */
export function suggestLogK(lengths: readonly number[]): boolean {
  const ls = lengths.filter((n) => n > 1);
  if (ls.length < 2) return false;
  return Math.max(...ls) >= 30 * Math.min(...ls);
}

/** Triangle corners [a, b, c] in data space for a slope guide (right angle at b). */
export function slopeTriangle(
  g: SlopeGuide,
  xDomain: [number, number],
  yDomain: [number, number],
): [[number, number], [number, number], [number, number]] {
  const d = g.decades ?? 1;
  const [x0, y0] = g.at ?? [
    10 ** (Math.log10(xDomain[1]) - d - 0.35),
    10 ** (Math.log10(yDomain[0]) + 0.4 + (g.slope < 0 ? Math.abs(g.slope) * d : 0)),
  ];
  const x1 = x0 * 10 ** d;
  const y1 = y0 * 10 ** (g.slope * d);
  return [
    [x0, y0],
    [x1, y0],
    [x1, y1],
  ];
}

/** Robust domain of the points, padded 8 % and rounded to nice ticks. */
export function dataDomain(values: readonly number[]): [number, number] {
  const fin = values.filter(Number.isFinite);
  if (!fin.length) return [0, 1];
  const lo = Math.min(...fin),
    hi = Math.max(...fin);
  const pad = (hi - lo) * 0.08 || Math.max(1, Math.abs(lo) * 0.1);
  return niceDomain(lo - pad, hi + pad);
}
