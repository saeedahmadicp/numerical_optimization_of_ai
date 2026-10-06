/**
 * Color in JS: the method palette (mirrors tokens.css) and perceptual colormaps for canvases.
 *
 * Colormaps are defined by OKLCH anchors and sampled into 256-entry RGB lookup tables, so
 * lightness is monotone and steps are perceptually even (no rainbow).
 */

export type ThemeMode = 'light' | 'dark';

/** Method slots, fixed order (slot i → `--series-{i+1}`). Validated all-pairs; see tokens.css. */
export const SERIES = {
  light: ['#2a78d6', '#eb6834', '#1baf7a', '#882892'],
  dark: ['#3987e5', '#d95926', '#199e70', '#a13bab'],
} as const;
export const MAX_SERIES = 4;

export const seriesVar = (slot: number) => `var(--series-${(slot % MAX_SERIES) + 1})`;

// ---------------------------------------------------------------------------------------
// OKLCH → sRGB
// ---------------------------------------------------------------------------------------

export type RGB = [number, number, number];

function oklabToLinear(L: number, a: number, b: number): RGB {
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.291485548 * b;
  const l = l_ ** 3,
    m = m_ ** 3,
    s = s_ ** 3;
  return [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ];
}

const gamma = (x: number) => {
  const c = Math.min(1, Math.max(0, x));
  return c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055;
};

export function oklch(L: number, C: number, hDeg: number): RGB {
  const h = (hDeg * Math.PI) / 180;
  const [r, g, b] = oklabToLinear(L, C * Math.cos(h), C * Math.sin(h));
  return [Math.round(gamma(r) * 255), Math.round(gamma(g) * 255), Math.round(gamma(b) * 255)];
}

export const rgbToHex = ([r, g, b]: RGB) =>
  '#' + [r, g, b].map((v) => v.toString(16).padStart(2, '0')).join('');

// ---------------------------------------------------------------------------------------
// Colormaps
// ---------------------------------------------------------------------------------------

/** OKLCH anchor: position t ∈ [0, 1], lightness, chroma, hue (degrees). */
type Anchor = [t: number, L: number, C: number, h: number];

function interpAnchors(anchors: Anchor[], t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t));
  let i = 0;
  while (i < anchors.length - 2 && x > anchors[i + 1][0]) i++;
  const [t0, L0, C0, h0] = anchors[i];
  const [t1, L1, C1, h1] = anchors[i + 1];
  const u = t1 === t0 ? 0 : (x - t0) / (t1 - t0);
  // Interpolate in OKLab (a, b) rather than hue angle, to avoid hue swings through gray.
  const a0 = C0 * Math.cos((h0 * Math.PI) / 180),
    b0 = C0 * Math.sin((h0 * Math.PI) / 180);
  const a1 = C1 * Math.cos((h1 * Math.PI) / 180),
    b1 = C1 * Math.sin((h1 * Math.PI) / 180);
  return [L0 + (L1 - L0) * u, a0 + (a1 - a0) * u, b0 + (b1 - b0) * u];
}

export interface Colormap {
  name: string;
  /** 256 × RGB, packed. */
  lut: Uint8ClampedArray;
  /** The same LUT as hex strings (handy for SVG/CSS gradients). */
  hex: string[];
}

function buildColormap(name: string, anchors: Anchor[], n = 256): Colormap {
  const lut = new Uint8ClampedArray(n * 3);
  const hex: string[] = [];
  for (let i = 0; i < n; i++) {
    const [L, a, b] = interpAnchors(anchors, i / (n - 1));
    const [r, g, bb] = oklabToLinear(L, a, b);
    const rgb: RGB = [
      Math.round(gamma(r) * 255),
      Math.round(gamma(g) * 255),
      Math.round(gamma(bb) * 255),
    ];
    lut.set(rgb, i * 3);
    hex.push(rgbToHex(rgb));
  }
  return { name, lut, hex };
}

/**
 * Contour fills (t = 0 → lowest f, t = 1 → highest f).
 *
 * Fills sit in a lightness band that method marks never use (light: L 0.80–0.985, marks L ≤ 0.72;
 * dark: L 0.17–0.40, marks L ≥ 0.50), so every path stays legible over any band. Chroma is low:
 * the field is context, the paths are content. The anchor flips with the theme: the basin is the
 * band furthest from the surface (darkest in light mode, brightest in dark mode).
 */
export const CONTOUR_MAPS: Record<ThemeMode, Colormap> = {
  // The basin (t = 0) is a low-chroma teal-gray, not blue: series slot 1 is blue and is the
  // default first method, so a blue basin would hide it exactly where the paths converge.
  light: buildColormap('contour-light', [
    [0, 0.8, 0.032, 212],
    [0.3, 0.868, 0.034, 196],
    [0.62, 0.93, 0.03, 150],
    [1, 0.982, 0.03, 88],
  ]),
  dark: buildColormap('contour-dark', [
    [0, 0.42, 0.034, 212],
    [0.32, 0.33, 0.03, 200],
    [0.66, 0.255, 0.022, 170],
    [1, 0.195, 0.012, 95],
  ]),
};

/**
 * General-purpose perceptual sequential map (cividis-like: deep blue → gray → gold), monotone in
 * lightness. For heatmaps and 3-D surfaces where the field itself is the content.
 */
export const SEQUENTIAL: Colormap = buildColormap('sequential', [
  [0, 0.27, 0.09, 268],
  [0.35, 0.47, 0.07, 245],
  [0.6, 0.62, 0.04, 140],
  [0.82, 0.78, 0.11, 96],
  [1, 0.93, 0.15, 98],
]);

/** RGB of a colormap at t ∈ [0, 1]. */
export function sample(map: Colormap, t: number): RGB {
  const i = Math.round(Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0)) * 255) * 3;
  return [map.lut[i], map.lut[i + 1], map.lut[i + 2]];
}

// ---------------------------------------------------------------------------------------
// Reading tokens for canvas drawing
// ---------------------------------------------------------------------------------------

export interface ChartColors {
  mode: ThemeMode;
  surface: string;
  grid: string;
  axis: string;
  tick: string;
  text: string;
  text2: string;
  text3: string;
  iso: string;
  isoStrong: string;
  halo: string;
  crosshair: string;
  playhead: string;
  accent: string;
  series: string[];
  fontSans: string;
  fontMono: string;
  /** Newsreader (display): rate annotations on charts. */
  fontSerif: string;
  /** KaTeX Math italic: variable names drawn on canvases (x, y, k). */
  fontMath: string;
  /** Feasible-set tint and hatch on contour plots (neutral ink, never a series color). */
  region: string;
}

/** Snapshot the chart tokens from computed styles (call again after a theme change). */
export function readChartColors(el: Element = document.documentElement): ChartColors {
  const cs = getComputedStyle(el);
  const v = (name: string) => cs.getPropertyValue(name).trim();
  const mode: ThemeMode = cs.colorScheme.includes('dark') ? 'dark' : 'light';
  return {
    mode,
    surface: v('--chart-surface'),
    grid: v('--chart-grid'),
    axis: v('--chart-axis'),
    tick: v('--chart-tick'),
    text: v('--color-text'),
    text2: v('--color-text-2'),
    text3: v('--color-text-3'),
    iso: v('--chart-iso'),
    isoStrong: v('--chart-iso-strong'),
    halo: v('--chart-halo'),
    crosshair: v('--chart-crosshair'),
    playhead: v('--chart-playhead'),
    accent: v('--color-accent'),
    series: [1, 2, 3, 4].map((i) => v(`--series-${i}`)),
    fontSans: v('--font-sans'),
    fontMono: v('--font-mono'),
    fontSerif: v('--font-serif'),
    fontMath: v('--font-math'),
    region: v('--chart-region'),
  };
}
