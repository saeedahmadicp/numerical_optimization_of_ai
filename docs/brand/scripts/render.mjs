#!/usr/bin/env node
/**
 * Render SVG or HTML files to PNG with headless Chromium (Playwright).
 *
 *   node docs/brand/scripts/render.mjs <input.svg|input.html> <output.png> [width height] [--scale 2] [--dark]
 *   node docs/brand/scripts/render.mjs --frames <input.svg> <outdir> <width> <height> <fps> <seconds> [--scale 1]
 *
 * Playwright is resolved from web/node_modules (installed by the web team for e2e tests), so the
 * brand scripts add no dependency of their own. An SVG is rendered at its own size unless a
 * width/height is given. `--dark` emulates `prefers-color-scheme: dark`.
 * `--frames` captures an animated SVG frame by frame by pausing all SMIL/CSS animations at t.
 */
import { createRequire } from 'node:module';
import { mkdirSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(resolve(here, '../../../web/package.json'));
const { chromium } = require('playwright');

const args = process.argv.slice(2);
const flag = (name, dflt) => {
  const i = args.indexOf(name);
  if (i < 0) return dflt;
  const v = args[i + 1];
  args.splice(i, 2);
  return Number(v);
};
const dark = args.includes('--dark');
if (dark) args.splice(args.indexOf('--dark'), 1);
const frames = args.includes('--frames');
if (frames) args.splice(args.indexOf('--frames'), 1);
const scale = flag('--scale', frames ? 1 : 2);

function svgSize(path) {
  const src = readFileSync(path, 'utf8');
  const w = /<svg[^>]*\bwidth="([\d.]+)"/.exec(src);
  const h = /<svg[^>]*\bheight="([\d.]+)"/.exec(src);
  if (w && h) return [Number(w[1]), Number(h[1])];
  const vb = /viewBox="[\d.\-]+ [\d.\-]+ ([\d.]+) ([\d.]+)"/.exec(src);
  return vb ? [Number(vb[1]), Number(vb[2])] : [800, 600];
}

const browser = await chromium.launch();
try {
  if (frames) {
    const [input, outdir, w, h, fps, seconds] = args;
    mkdirSync(outdir, { recursive: true });
    const page = await browser.newPage({
      viewport: { width: Number(w), height: Number(h) },
      deviceScaleFactor: scale,
      colorScheme: dark ? 'dark' : 'light',
    });
    const svg = readFileSync(input, 'utf8');
    await page.setContent(
      `<!doctype html><html><body style="margin:0;background:transparent">${svg}</body></html>`,
    );
    const n = Math.round(Number(fps) * Number(seconds));
    for (let i = 0; i < n; i++) {
      const t = i / Number(fps);
      await page.evaluate((t) => {
        const svgEl = document.querySelector('svg');
        if (svgEl.pauseAnimations) {
          svgEl.pauseAnimations();
          svgEl.setCurrentTime(t);
        }
        for (const a of document.getAnimations()) {
          a.pause();
          a.currentTime = t * 1000;
        }
      }, t);
      await page.screenshot({
        path: `${outdir}/f${String(i).padStart(4, '0')}.png`,
        omitBackground: false,
      });
    }
  } else {
    const [input, output, w, h] = args;
    const abs = resolve(input);
    const [sw, sh] = w ? [Number(w), Number(h)] : svgSize(abs);
    const page = await browser.newPage({
      viewport: { width: sw, height: sh },
      deviceScaleFactor: scale,
      colorScheme: dark ? 'dark' : 'light',
    });
    if (abs.endsWith('.svg')) {
      const svg = readFileSync(abs, 'utf8');
      await page.setContent(
        `<!doctype html><html><body style="margin:0;background:transparent;width:${sw}px;height:${sh}px">` +
          `<img src="data:image/svg+xml;base64,${Buffer.from(svg).toString('base64')}" width="${sw}" height="${sh}" style="display:block"></body></html>`,
      );
      // An <img> renders the SVG exactly as GitHub does (no scripts, no external resources).
      await page.waitForFunction(() => document.images[0]?.complete);
      // Freeze any animation at its end state for the static capture.
      await page.waitForTimeout(150);
    } else {
      await page.goto(pathToFileURL(abs).href);
      await page.evaluate(() => document.fonts.ready);
      await page.waitForTimeout(150);
    }
    await page.screenshot({ path: output, omitBackground: true });
  }
} finally {
  await browser.close();
}
