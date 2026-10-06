import { describe, expect, it } from 'vitest';
import { linearScale, linearTicks, logScale, logTicks, niceStep } from '../src/viz/scales';
import { chooseTransform, computeField, isoSegments } from '../src/viz/contourField';
import { advance, baseRate, easeInOut, localT, maxStep, segmentAt } from '../src/play/timeline';
import { CONTOUR_MAPS, SEQUENTIAL } from '../src/ui/colors';

describe('scales', () => {
  it('nice 1-2-5 ticks', () => {
    expect(niceStep(0, 10, 5)).toBe(2);
    expect(linearTicks(-1, 1, 4)).toEqual([-1, -0.5, 0, 0.5, 1]);
    expect(linearTicks(0, 0.3, 3)).toEqual([0, 0.1, 0.2, 0.3]);
  });
  it('linear and log scales invert', () => {
    const s = linearScale([0, 10], [0, 100]);
    expect(s(5)).toBe(50);
    expect(s.invert(25)).toBe(2.5);
    const l = logScale([1e-6, 1], [100, 0]);
    expect(l(1e-3)).toBeCloseTo(50, 10);
    expect(l.invert(50)).toBeCloseTo(1e-3, 15);
    // Decades are aligned to multiples of the stride, so 10⁰ is always a tick when in range.
    expect(logTicks(1e-8, 1, 3)).toEqual([1e-6, 1e-3, 1]);
  });
});

describe('contour field (marching squares)', () => {
  it('a circle iso-line has points at the right radius', () => {
    const n = 41;
    const g = new Float64Array(n * n);
    for (let j = 0; j < n; j++)
      for (let i = 0; i < n; i++) g[j * n + i] = Math.hypot(i - 20, j - 20);
    const segs: number[] = [];
    isoSegments(g, n, n, 10, segs);
    expect(segs.length).toBeGreaterThan(40);
    for (let k = 0; k < segs.length; k += 2)
      expect(Math.abs(Math.hypot(segs[k] - 20, segs[k + 1] - 20) - 10)).toBeLessThan(0.08);
  });
  it('chooses log spacing for a steep valley and linear for a bowl', () => {
    const rosen: number[] = [],
      bowl: number[] = [];
    for (let j = 0; j < 50; j++)
      for (let i = 0; i < 50; i++) {
        const x = -2 + (4 * i) / 49,
          y = -1 + (4 * j) / 49;
        rosen.push((1 - x) ** 2 + 100 * (y - x * x) ** 2);
        bowl.push(x * x + y * y);
      }
    expect(chooseTransform(rosen, 'auto').scale).toBe('log');
    expect(chooseTransform(bowl, 'auto').scale).toBe('linear');
  });
  it('computeField produces an opaque raster and level values in range', () => {
    const nx = 20,
      ny = 20;
    const values = new Float64Array(nx * ny).map((_, k) => (k % nx) ** 2 + Math.floor(k / nx) ** 2);
    const r = computeField({
      id: 1,
      values,
      nx,
      ny,
      width: 30,
      height: 24,
      levels: 8,
      scale: 'linear',
      lut: CONTOUR_MAPS.light.lut,
    });
    expect(r.pixels.length).toBe(30 * 24 * 4);
    expect(r.pixels[3]).toBe(255);
    expect(r.levelValues.length).toBe(7);
    expect(r.segments.length / 4).toBe(r.segLevel.length);
  });
});

describe('colormaps', () => {
  const lum = (lut: Uint8ClampedArray, i: number) =>
    0.2126 * lut[i * 3] + 0.7152 * lut[i * 3 + 1] + 0.0722 * lut[i * 3 + 2];
  it('contour maps are monotone in lightness (light: rising, dark: falling)', () => {
    for (let i = 8; i < 256; i += 8) {
      expect(lum(CONTOUR_MAPS.light.lut, i)).toBeGreaterThanOrEqual(
        lum(CONTOUR_MAPS.light.lut, i - 8) - 0.5,
      );
      expect(lum(CONTOUR_MAPS.dark.lut, i)).toBeLessThanOrEqual(
        lum(CONTOUR_MAPS.dark.lut, i - 8) + 0.5,
      );
      expect(lum(SEQUENTIAL.lut, i)).toBeGreaterThan(lum(SEQUENTIAL.lut, i - 8));
    }
  });
});

describe('timeline', () => {
  it('each method stops at its own last step', () => {
    expect(maxStep([5, 12, 3])).toBe(11);
    expect(localT(7.5, 5)).toBe(4);
    expect(localT(2.5, 5)).toBe(2.5);
    expect(segmentAt(2.5, 5, false)).toEqual({ i: 2, u: 0.5 });
    expect(segmentAt(9, 5)).toEqual({ i: 4, u: 0 });
  });
  it('advances continuously, or in whole steps under reduced motion', () => {
    expect(advance(1, 0.5, 4, 10, false).t).toBe(3);
    expect(advance(9.5, 1, 4, 10, false).t).toBe(10);
    const a = advance(1, 0.1, 4, 10, true);
    expect(a.t).toBe(1);
    expect(a.acc).toBeCloseTo(0.4, 12);
    const b = advance(a.t, 0.2, 4, 10, true, a.acc);
    expect(b.t).toBe(2);
    expect(Number.isInteger(b.t)).toBe(true);
  });
  it('ease and base rate are sane', () => {
    expect(easeInOut(0)).toBe(0);
    expect(easeInOut(1)).toBe(1);
    expect(easeInOut(0.5)).toBe(0.5);
    expect(baseRate(5)).toBe(2.5);
    expect(baseRate(10_000)).toBe(90);
  });
});
