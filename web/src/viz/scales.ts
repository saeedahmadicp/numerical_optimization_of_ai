/**
 * Scales: map data → pixels, with "nice" tick generation (d3-style 1-2-5 steps).
 * Pure functions; no DOM.
 */

export interface Scale {
  kind: 'linear' | 'log';
  domain: [number, number];
  range: [number, number];
  (v: number): number;
  invert(px: number): number;
  ticks(count?: number): number[];
  /** Step between major ticks (linear) or 1 (log, in decades). */
  tickStep(count?: number): number;
  copy(domain?: [number, number], range?: [number, number]): Scale;
}

/** The 1-2-5 step closest to (hi - lo) / count. */
export function niceStep(lo: number, hi: number, count = 6): number {
  const span = Math.abs(hi - lo);
  if (!(span > 0) || !Number.isFinite(span)) return 1;
  const raw = span / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const err = raw / mag;
  const m = err >= 7.07 ? 10 : err >= 3.16 ? 5 : err >= 1.41 ? 2 : 1;
  return m * mag;
}

export function linearTicks(lo: number, hi: number, count = 6): number[] {
  const a = Math.min(lo, hi),
    b = Math.max(lo, hi);
  const step = niceStep(a, b, count);
  if (!Number.isFinite(step) || step <= 0) return [];
  const start = Math.ceil(a / step - 1e-9),
    end = Math.floor(b / step + 1e-9);
  const out: number[] = [];
  for (let i = start; i <= end && out.length < 200; i++) {
    const v = i * step;
    // Clean floating noise (0.30000000000000004 → 0.3).
    out.push(Number(v.toPrecision(12)) || 0);
  }
  return out;
}

/** Extend [lo, hi] outward to the nearest nice ticks. */
export function niceDomain(lo: number, hi: number, count = 6): [number, number] {
  if (lo === hi) {
    const d = lo === 0 ? 1 : Math.abs(lo) * 0.1;
    return [lo - d, hi + d];
  }
  const step = niceStep(lo, hi, count);
  return [Math.floor(lo / step) * step, Math.ceil(hi / step) * step];
}

export function linearScale(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain,
    [r0, r1] = range;
  const k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
  const s = ((v: number) => r0 + (v - d0) * k) as Scale;
  s.kind = 'linear';
  s.domain = domain;
  s.range = range;
  s.invert = (px) => (k === 0 ? d0 : d0 + (px - r0) / k);
  s.ticks = (count = 6) => linearTicks(d0, d1, count);
  s.tickStep = (count = 6) => niceStep(d0, d1, count);
  s.copy = (d = domain, r = range) => linearScale(d, r);
  return s;
}

/** Base-10 log scale. Non-positive inputs map to the bottom of the range. */
export function logScale(domain: [number, number], range: [number, number]): Scale {
  const lo = Math.max(domain[0], Number.MIN_VALUE),
    hi = Math.max(domain[1], lo * 10);
  const l0 = Math.log10(lo),
    l1 = Math.log10(hi);
  const [r0, r1] = range;
  const k = (r1 - r0) / (l1 - l0);
  const s = ((v: number) => (v > 0 ? r0 + (Math.log10(v) - l0) * k : r0)) as Scale;
  s.kind = 'log';
  s.domain = [lo, hi];
  s.range = range;
  s.invert = (px) => 10 ** (l0 + (px - r0) / k);
  s.ticks = (count = 6) => logTicks(lo, hi, count);
  s.tickStep = () => 1;
  s.copy = (d = [lo, hi], r = range) => logScale(d, r);
  return s;
}

/** Powers of ten inside [lo, hi], thinned to about `count` (every 2nd/3rd decade when dense). */
export function logTicks(lo: number, hi: number, count = 6): number[] {
  const e0 = Math.ceil(Math.log10(lo) - 1e-9),
    e1 = Math.floor(Math.log10(hi) + 1e-9);
  const n = e1 - e0 + 1;
  const every = Math.max(1, Math.ceil(n / Math.max(1, count)));
  const out: number[] = [];
  for (let e = e0; e <= e1; e++) if (e % every === 0 || n <= count) out.push(10 ** e);
  return out;
}

/** Decade bounds that contain the finite positive values of `values`. */
export function logExtent(values: Iterable<number>, floor = 1e-300): [number, number] {
  let lo = Infinity,
    hi = -Infinity;
  for (const v of values) {
    if (!(v > floor) || !Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (lo === Infinity) return [1e-16, 1];
  const e0 = Math.floor(Math.log10(lo)),
    e1 = Math.ceil(Math.log10(hi));
  return [10 ** e0, 10 ** Math.max(e1, e0 + 1)];
}

/** Map a pixel coordinate to the center of a device pixel, so a 1-device-px line is crisp. */
export function crisp(v: number, dpr: number): number {
  return (Math.round(v * dpr - 0.5) + 0.5) / dpr;
}
