/**
 * Every lab has a preview; each one builds from real runs, states what it shows, and draws every
 * frame without throwing. The hero scenes only show runs that converged.
 */
import katex from 'katex';
import { describe, expect, it } from 'vitest';
import { splitMath } from '../../../ui/mathProse';
import type { ChartColors } from '../../../ui/colors';
import type { Preview, PreviewModule } from './types';
import { HERO_SCENES } from '../heroScenes';

const modules = import.meta.glob<PreviewModule>('./labs/*.ts', { eager: true });
const labDirs = Object.keys(import.meta.glob('../../../labs/*/meta.ts')).map((path) =>
  path.split('/').at(-2)!,
);

const colors: ChartColors = {
  mode: 'light',
  surface: '#fcfcfb',
  grid: '#eeeeea',
  axis: '#ccc',
  tick: '#6b6963',
  text: '#141413',
  text2: '#52514e',
  text3: '#6b6963',
  iso: 'rgba(0,0,0,0.1)',
  isoStrong: 'rgba(0,0,0,0.2)',
  halo: '#fcfcfb',
  crosshair: '#000',
  playhead: '#000',
  accent: '#4b2ace',
  series: ['#2a78d6', '#eb6834', '#1baf7a', '#882892'],
  fontSans: 'sans-serif',
  fontMono: 'monospace',
  fontSerif: 'serif',
  fontMath: 'serif',
  region: 'rgba(0,0,0,0.05)',
};

/** A canvas context that accepts every call (node has no canvas). */
function fakeContext(): CanvasRenderingContext2D {
  const handler: ProxyHandler<object> = {
    get: (_t, key) => {
      if (key === 'createImageData')
        return (w: number, h: number) => ({
          data: new Uint8ClampedArray(w * h * 4),
          width: w,
          height: h,
        });
      if (key === 'getContext') return () => fakeContext();
      return () => fakeContext();
    },
    set: () => true,
  };
  return new Proxy({}, handler) as CanvasRenderingContext2D;
}

/**
 * Unicode look-alikes of math that must not appear in the prose of a title or caption: those are
 * set in Inter (or a fallback face), not KaTeX (brand.md §3). Bold 𝐱, sub- and superscripts, norms,
 * relations, operators and Greek letters belong in `$…$`.
 */
const UNICODE_MATH = /[\u{1D400}-\u{1D7FF}₀₁₂₃₄₅₆₇₈₉ₖₙᵢⱼ⁰¹²³⁴⁵⁶⁷⁸⁹⁻ᵀ‖∇⋆≤≥≈∫½ŷαβγδεηλμσφψω]/u;

/** Every `$…$` part of `text` typesets without a KaTeX error; the prose has no Unicode math. */
function expectMathProse(text: string) {
  const parts = splitMath(text);
  parts.forEach((part, i) => {
    if (i % 2 === 0) expect(part, `prose in “${text}”`).not.toMatch(UNICODE_MATH);
    else
      expect(() =>
        katex.renderToString(part, { throwOnError: true, strict: 'error' }),
      ).not.toThrow();
  });
}

const previews = new Map<string, Preview>(
  Object.entries(modules).map(([path, m]) => [path.replace(/^.*\/(.+)\.ts$/, '$1'), m.default()]),
);

describe('lab previews', () => {
  it('every lab folder has a preview', () => {
    expect([...previews.keys()].sort()).toEqual([...labDirs].sort());
  });

  for (const [id, p] of previews) {
    it(`${id}: builds from real runs and draws every frame`, () => {
      expect(p.lab).toBe(id);
      expect(p.caption.length).toBeGreaterThan(40);
      expect(p.ariaLabel.length).toBeGreaterThan(30);
      expect(p.legend.length).toBeGreaterThan(0);
      expect(p.legend.length).toBeLessThanOrEqual(4);
      expect(p.caption).not.toMatch(/NaN|undefined|null|e[-+]\d/);
      expectMathProse(p.title);
      expectMathProse(p.caption);
      expect(p.ariaLabel).not.toContain('$');
      for (const l of p.legend) expect(l.note).not.toMatch(/NaN|undefined/);
      // Node has no DOM: stub document.createElement for the cached raster layers.
      const g = globalThis as unknown as { document?: unknown; ImageData?: unknown };
      g.document ??= {
        createElement: () => ({ width: 0, height: 0, getContext: () => fakeContext() }),
      };
      g.ImageData ??= class {};
      const ctx = fakeContext();
      for (const u of [0, 0.13, 0.5, 0.77, 1])
        for (const hero of [false, true])
          p.draw(
            ctx,
            { width: hero ? 600 : 360, height: hero ? 450 : 225, dpr: 1 },
            colors,
            u,
            hero,
          );
    });
  }

  it('hero scenes exist and show only converged runs', () => {
    for (const id of HERO_SCENES) expect(previews.has(id)).toBe(true);
    expect(previews.get('unconstrained')!.legend.every((l) => l.note.endsWith('iterations'))).toBe(
      true,
    );
  });
});
