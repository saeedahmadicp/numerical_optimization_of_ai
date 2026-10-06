/**
 * Contour field computation — pure, DOM-free, used by the Web Worker (and a synchronous
 * fallback/tests).
 *
 * Input: f sampled on an nx × ny grid over the view (row j = y0 + j·dy, column i = x0 + i·dx).
 * Output: a band raster at pixel resolution (bilinear in the transformed field, then quantized)
 * and iso-line segments from marching squares at every band boundary.
 */

export type LevelScale = 'auto' | 'linear' | 'log';

export interface FieldRequest {
  id: number;
  values: Float64Array; // nx * ny, row-major from y0 (bottom) upward
  nx: number;
  ny: number;
  /** Output raster size in device pixels. */
  width: number;
  height: number;
  levels: number;
  scale: LevelScale;
  /** Known minimum value of f (e.g. at a known minimizer), if any. */
  fMin?: number | null;
  /**
   * Fixed level transform (from `levelSpecFor` over the problem's full domain). When given, the
   * levels do not depend on the visible window: pan/zoom only re-rasterize, and one f value keeps
   * one color in every view. Without it the transform is chosen from `values`.
   */
  levelSpec?: LevelSpec | null;
  /** 256×3 colormap LUT (t = 0 → lowest f). */
  lut: Uint8ClampedArray;
}

export interface FieldResult {
  id: number;
  width: number;
  height: number;
  /** RGBA, row-major from the top pixel row. */
  pixels: Uint8ClampedArray;
  /** Segments [x1, y1, x2, y2, ...] in grid units (i, j); `segLevel[s]` is the level index. */
  segments: Float32Array;
  segLevel: Uint16Array;
  levelValues: number[];
  scale: 'linear' | 'log';
  fMin: number;
  fMax: number;
}

/** Transform used for level spacing: identity (linear) or log10(f − fmin + δ) (log). */
export interface LevelTransform {
  scale: 'linear' | 'log';
  fMin: number;
  fMax: number;
  delta: number;
  g(f: number): number;
  ginv(g: number): number;
}

/** The serializable part of a LevelTransform (it can be posted to the worker). */
export type LevelSpec = Pick<LevelTransform, 'scale' | 'fMin' | 'fMax' | 'delta'>;

export function transformFromSpec({ scale, fMin, fMax, delta }: LevelSpec): LevelTransform {
  if (scale === 'log')
    return {
      scale,
      fMin,
      fMax,
      delta,
      g: (f) => Math.log10(Math.max(f - fMin, 0) + delta),
      ginv: (g) => 10 ** g - delta + fMin,
    };
  return { scale, fMin, fMax, delta: 0, g: (f) => f, ginv: (g) => g };
}

/**
 * Level transform for a whole problem: samples f on an n × n grid over `domain` (the problem's
 * full domain, not the visible window) and picks linear/log spacing once.
 */
export function levelSpecFor(
  f: (x: number, y: number) => number,
  domain: readonly [readonly [number, number], readonly [number, number]],
  scale: LevelScale = 'auto',
  knownMin?: number | null,
  n = 96,
): LevelSpec {
  const [[x0, x1], [y0, y1]] = domain;
  const values = new Float64Array(n * n);
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++) {
      const v = f(x0 + (i / (n - 1)) * (x1 - x0), y0 + (j / (n - 1)) * (y1 - y0));
      values[j * n + i] = Number.isFinite(v) ? v : NaN;
    }
  const { scale: s, fMin, fMax, delta } = chooseTransform(values, scale, knownMin);
  return { scale: s, fMin, fMax, delta };
}

export function chooseTransform(
  values: ArrayLike<number>,
  scale: LevelScale,
  knownMin?: number | null,
): LevelTransform {
  let lo = Infinity,
    hi = -Infinity;
  const finite: number[] = [];
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    finite.push(v);
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (finite.length === 0) {
    lo = 0;
    hi = 1;
  }
  if (knownMin !== null && knownMin !== undefined && Number.isFinite(knownMin))
    lo = Math.min(lo, knownMin);
  if (hi <= lo) hi = lo + 1;
  let resolved: 'linear' | 'log' = scale === 'log' ? 'log' : 'linear';
  if (scale === 'auto') {
    // Ill-conditioned / steep functions: most of the domain sits far above the minimum while
    // the interesting structure is near it. Use log spacing when the median is low in the range.
    finite.sort((a, b) => a - b);
    const med = finite[Math.floor(finite.length / 2)] ?? lo;
    const q90 = finite[Math.floor(finite.length * 0.9)] ?? hi;
    resolved = (med - lo) / (hi - lo) < 0.12 || (q90 - lo) / (hi - lo) < 0.35 ? 'log' : 'linear';
  }
  const delta = (hi - lo) * 2e-4;
  return transformFromSpec({ scale: resolved, fMin: lo, fMax: hi, delta });
}

/** Marching squares on a transformed grid `g` for one threshold; appends segments. */
export function isoSegments(
  g: Float64Array,
  nx: number,
  ny: number,
  level: number,
  out: number[],
): void {
  for (let j = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      const a = g[j * nx + i],
        b = g[j * nx + i + 1],
        c = g[(j + 1) * nx + i + 1],
        d = g[(j + 1) * nx + i];
      if (!(Number.isFinite(a) && Number.isFinite(b) && Number.isFinite(c) && Number.isFinite(d)))
        continue;
      const code =
        (a > level ? 1 : 0) | (b > level ? 2 : 0) | (c > level ? 4 : 0) | (d > level ? 8 : 0);
      if (code === 0 || code === 15) continue;
      // Edge points: bottom (a-b), right (b-c), top (d-c), left (a-d).
      const t = (p: number, q: number) => (p === q ? 0.5 : (level - p) / (q - p));
      const B: [number, number] = [i + t(a, b), j];
      const R: [number, number] = [i + 1, j + t(b, c)];
      const T: [number, number] = [i + t(d, c), j + 1];
      const L: [number, number] = [i, j + t(a, d)];
      const seg = (p: [number, number], q: [number, number]) => out.push(p[0], p[1], q[0], q[1]);
      switch (code) {
        case 1:
        case 14:
          seg(L, B);
          break;
        case 2:
        case 13:
          seg(B, R);
          break;
        case 3:
        case 12:
          seg(L, R);
          break;
        case 4:
        case 11:
          seg(R, T);
          break;
        case 6:
        case 9:
          seg(B, T);
          break;
        case 7:
        case 8:
          seg(L, T);
          break;
        case 5:
        case 10: {
          // Saddle: disambiguate with the cell-center value.
          const center = (a + b + c + d) / 4 > level;
          if ((code === 5) === center) {
            seg(L, T);
            seg(B, R);
          } else {
            seg(L, B);
            seg(R, T);
          }
          break;
        }
      }
    }
  }
}

export function computeField(req: FieldRequest): FieldResult {
  const { values, nx, ny, width, height, levels, lut } = req;
  const tr = req.levelSpec
    ? transformFromSpec(req.levelSpec)
    : chooseTransform(values, req.scale, req.fMin);
  const g = new Float64Array(values.length);
  for (let i = 0; i < values.length; i++) g[i] = Number.isFinite(values[i]) ? tr.g(values[i]) : NaN;
  const g0 = tr.g(tr.fMin);
  const g1 = tr.g(tr.fMax);
  const n = Math.max(2, levels);
  const step = (g1 - g0) / n;
  const thresholds: number[] = [];
  for (let l = 1; l < n; l++) thresholds.push(g0 + l * step);

  // Band raster: bilinear interpolation of g at each pixel center, then quantize.
  const pixels = new Uint8ClampedArray(width * height * 4);
  const colors = new Uint8ClampedArray(n * 3);
  for (let b = 0; b < n; b++) {
    const idx = Math.round(((b + 0.5) / n) * 255) * 3;
    colors[b * 3] = lut[idx];
    colors[b * 3 + 1] = lut[idx + 1];
    colors[b * 3 + 2] = lut[idx + 2];
  }
  const sx = (nx - 1) / width,
    sy = (ny - 1) / height;
  for (let py = 0; py < height; py++) {
    const gy = (height - py - 0.5) * sy; // top pixel row = highest y
    const j = Math.min(ny - 2, Math.max(0, Math.floor(gy)));
    const v = gy - j;
    const row0 = j * nx,
      row1 = (j + 1) * nx;
    for (let px = 0; px < width; px++) {
      const gx = (px + 0.5) * sx;
      const i = Math.min(nx - 2, Math.max(0, Math.floor(gx)));
      const u = gx - i;
      const a = g[row0 + i],
        b = g[row0 + i + 1],
        c = g[row1 + i + 1],
        d = g[row1 + i];
      const val = (a * (1 - u) + b * u) * (1 - v) + (d * (1 - u) + c * u) * v;
      const o = (py * width + px) * 4;
      if (!Number.isFinite(val)) {
        pixels[o + 3] = 0;
        continue;
      }
      let band = Math.floor((val - g0) / step);
      if (band < 0) band = 0;
      else if (band >= n) band = n - 1;
      pixels[o] = colors[band * 3];
      pixels[o + 1] = colors[band * 3 + 1];
      pixels[o + 2] = colors[band * 3 + 2];
      pixels[o + 3] = 255;
    }
  }

  const segs: number[] = [];
  const segLevel: number[] = [];
  thresholds.forEach((th, l) => {
    const before = segs.length;
    isoSegments(g, nx, ny, th, segs);
    for (let s = before; s < segs.length; s += 4) segLevel.push(l + 1);
  });

  return {
    id: req.id,
    width,
    height,
    pixels,
    segments: new Float32Array(segs),
    segLevel: new Uint16Array(segLevel),
    levelValues: thresholds.map((th) => tr.ginv(th)),
    scale: tr.scale,
    fMin: tr.fMin,
    fMax: tr.fMax,
  };
}
