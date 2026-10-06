/**
 * Chart text is never below 11 px (brand.md §9: "keep its smallest type at ≥ 11 px"). A fake 2-D
 * context wraps `fillText` and records the effective font size of every label the shared
 * renderers draw (sub- and superscripts: at least 9 px): axes (linear, log, factored ×10ⁿ), math runs with exponents, overlay labels and
 * the off-view label of a path that leaves the view.
 */
import { describe, expect, it } from 'vitest';
import type { ChartColors } from '../ui/colors';
import { axisExponent, drawAxes } from './axes';
import { drawMath, pow10Runs, iterateRuns, SCRIPT } from './mathText';
import { drawOverlays2D } from './overlays2d';
import { drawPathLayer } from './PathLayer';
import { linearScale, logScale } from './scales';

interface Drawn {
  text: string;
  size: number;
}

/** A recording stand-in for CanvasRenderingContext2D: enough for the shared renderers. */
function fakeContext() {
  const drawn: Drawn[] = [];
  const stack: string[] = [];
  const state = { font: '10px sans-serif' };
  const sizeOf = (font: string) => Number(/(\d+(?:\.\d+)?)px/.exec(font)?.[1] ?? NaN);
  const noop = () => undefined;
  const ctx = {
    get font() {
      return state.font;
    },
    set font(f: string) {
      state.font = f;
    },
    fillText(text: string) {
      if (String(text).trim()) drawn.push({ text: String(text), size: sizeOf(state.font) });
    },
    strokeText: noop,
    measureText: (t: string) => ({ width: String(t).length * sizeOf(state.font) * 0.6 }),
    save: () => stack.push(state.font),
    restore: () => {
      const f = stack.pop();
      if (f) state.font = f;
    },
    beginPath: noop,
    closePath: noop,
    moveTo: noop,
    lineTo: noop,
    arc: noop,
    ellipse: noop,
    rect: noop,
    fill: noop,
    stroke: noop,
    clip: noop,
    fillRect: noop,
    strokeRect: noop,
    setLineDash: noop,
    drawImage: noop,
    translate: noop,
    rotate: noop,
    scale: noop,
    setTransform: noop,
    createLinearGradient: () => ({ addColorStop: noop }),
    createRadialGradient: () => ({ addColorStop: noop }),
  } as unknown as CanvasRenderingContext2D & Record<string, unknown>;
  for (const k of [
    'fillStyle',
    'strokeStyle',
    'lineWidth',
    'lineJoin',
    'lineCap',
    'globalAlpha',
    'textAlign',
    'textBaseline',
    'shadowBlur',
    'shadowColor',
    'globalCompositeOperation',
    'imageSmoothingEnabled',
  ])
    (ctx as Record<string, unknown>)[k] = undefined;
  return { ctx, drawn };
}

const colors = {
  mode: 'light',
  surface: '#fff',
  grid: '#eee',
  axis: '#ccc',
  tick: '#666',
  text: '#111',
  text2: '#444',
  text3: '#666',
  iso: '#ddd',
  isoStrong: '#bbb',
  halo: '#fff',
  crosshair: '#999',
  playhead: '#333',
  accent: '#4b2ace',
  series: ['#2a78d6', '#eb6834', '#1baf7a', '#882892'],
  fontSans: 'Inter',
  fontMono: 'JetBrains Mono',
  fontSerif: 'Newsreader',
  fontMath: 'KaTeX_Math',
  region: '#000',
} satisfies ChartColors;

const frame = { left: 48, top: 26, right: 560, bottom: 300 };
const MIN = 11;

/** A sub- or superscript drawn by `drawMath`: a short index or exponent (k, 12, −8, ⋆). */
const isScript = (d: Drawn) => /^[−\d⋆a-zA-Z+]{1,4}$/.test(d.text);
const SCRIPT_MIN = 9;

function expectLegible(drawn: Drawn[]) {
  expect(drawn.length).toBeGreaterThan(0);
  const small = drawn.filter((d) => !(d.size >= MIN || (isScript(d) && d.size >= SCRIPT_MIN)));
  expect(small, `labels below ${MIN} px: ${JSON.stringify(small)}`).toEqual([]);
}

describe('chart text is at least 11 px', () => {
  it('linear axes (default size)', () => {
    const { ctx, drawn } = fakeContext();
    drawAxes(ctx, {
      x: linearScale([0, 40], [frame.left, frame.right]),
      y: linearScale([-3, 3], [frame.bottom, frame.top]),
      frame,
      colors,
      dpr: 2,
      xLabel: 'k',
      yLabel: 'value',
    });
    expectLegible(drawn);
  });

  it('log axes: powers of ten and their exponents (≥ 9 px scripts)', () => {
    const { ctx, drawn } = fakeContext();
    drawAxes(ctx, {
      x: logScale([1, 1e4], [frame.left, frame.right]),
      y: logScale([1e-12, 1e2], [frame.bottom, frame.top]),
      frame,
      colors,
      dpr: 1,
      inset: true,
    });
    const exps = drawn.filter((d) => /^[−\d]+$/.test(d.text) && d.text !== '10' && d.text !== '1');
    expect(exps.length).toBeGreaterThan(0);
    for (const e of exps) expect(e.size).toBeGreaterThanOrEqual(SCRIPT_MIN);
    // The full-size runs (the "10") are ≥ 11 px.
    expectLegible(drawn.filter((d) => d.text === '10' || d.text === '1'));
  });

  it('a caller cannot shrink tick labels below 11 px', () => {
    const { ctx, drawn } = fakeContext();
    drawAxes(ctx, {
      x: linearScale([0, 10], [frame.left, frame.right]),
      y: linearScale([0, 1], [frame.bottom, frame.top]),
      frame,
      colors,
      dpr: 1,
      fontSize: 9.5,
    });
    expectLegible(drawn);
  });

  it('linear ticks around 10⁻⁵ factor the power of ten out (no Unicode superscripts)', () => {
    const { ctx, drawn } = fakeContext();
    const y = linearScale([-6e-5, 6e-5], [frame.bottom, frame.top]);
    drawAxes(ctx, {
      x: linearScale([0, 10], [frame.left, frame.right]),
      y,
      frame,
      colors,
      dpr: 1,
      yLabel: 'f',
    });
    expect(axisExponent(y.ticks(5))).toBe(-5);
    expect(drawn.some((d) => /[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]/.test(d.text))).toBe(false);
    expect(drawn.some((d) => d.text === '×')).toBe(true);
    expectLegible(drawn);
  });

  it('math runs keep scripts at ≥ 76 %', () => {
    expect(SCRIPT).toBeGreaterThanOrEqual(0.76);
    const { ctx, drawn } = fakeContext();
    drawMath(ctx, pow10Runs(-8), 0, 0, { size: 12.5 });
    drawMath(ctx, iterateRuns('x', 12, '(0.76, −3.18)'), 0, 0, { size: 12 });
    expectLegible(drawn);
  });

  it('scripts never drop below 9.5 px, even on an 11-px label', () => {
    const { ctx, drawn } = fakeContext();
    drawMath(ctx, pow10Runs(-8), 0, 0, { size: 11 });
    drawMath(ctx, iterateRuns('x', 3), 0, 0, { size: 11.5 });
    const scripts = drawn.filter((d) => d.text === '−8' || d.text === '3');
    expect(scripts.length).toBe(2);
    for (const d of scripts) expect(d.size).toBeGreaterThanOrEqual(9.5);
  });

  it('overlay labels and text', () => {
    const { ctx, drawn } = fakeContext();
    drawOverlays2D(
      ctx,
      { toPx: (x, y) => [100 + 40 * x, 200 - 40 * y], width: 600, height: 400, dpr: 1, colors },
      [
        { kind: 'point', at: [1, 1], label: 'x⋆ = (1, 1)' },
        { kind: 'text', at: [0, 0], text: 'note', size: 9 },
        { kind: 'arrow', from: [0, 0], to: [1, 0], label: '−∇f' },
      ],
    );
    expectLegible(drawn);
  });

  it('the off-view label of a path that leaves the view', () => {
    const { ctx, drawn } = fakeContext();
    drawPathLayer(
      ctx,
      [
        {
          points: [
            [0, 0],
            [0.5, 0.2],
            [0.76, -30],
            [1, 1],
          ],
          color: '#eb6834',
          label: 'Newton',
        },
      ],
      {
        t: 3,
        toPx: (x, y) => [100 + 100 * x, 200 - 100 * y],
        halo: '#fff',
        bounds: { left: 0, top: 0, right: 600, bottom: 400 },
        offViewLabels: true,
      },
    );
    expectLegible(drawn);
  });
});
