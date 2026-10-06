/**
 * Lazy KaTeX loader. KaTeX (≈ 250 KB JS + 29 KB CSS + fonts) loads on first use in its own chunk,
 * so pages without math (the home page) never download it. `<Formula>` subscribes to it; labs
 * may call `preloadKatex()` at import time so math is ready sooner.
 *
 * Pages full of math wait for it before their first paint: `withKatex(load)` wraps a lazy
 * page's loader so the route's Suspense fallback covers the ~100 ms until KaTeX, its stylesheet
 * and its main faces are in, and every formula paints once, in its final width (no layout shift
 * from the plain-text fallback to KaTeX). App.tsx does this for the labs, #/methods, #/method/…
 * and #/research/….
 */
import type katexType from 'katex';

export type Katex = typeof katexType;

let katex: Katex | null = null;
let loading: Promise<void> | null = null;
const listeners = new Set<() => void>();

/** Start loading KaTeX and its stylesheet (idempotent). */
export function preloadKatex(): Promise<void> {
  if (!loading) {
    loading = Promise.all([import('katex'), import('katex/dist/katex.min.css')]).then(([m]) => {
      katex = m.default;
      listeners.forEach((l) => l());
    });
  }
  return loading;
}

/** useSyncExternalStore adapter: subscribing starts the load. */
export function subscribeKatex(cb: () => void): () => void {
  listeners.add(cb);
  void preloadKatex();
  return () => listeners.delete(cb);
}

export const getKatex = (): Katex | null => katex;

/** The faces almost every formula uses: upright, bold (vectors) and italic (variables). */
const KATEX_FACES = ['400 1em KaTeX_Main', '700 1em KaTeX_Main', 'italic 400 1em KaTeX_Math'];

/** Longest a page waits for KaTeX before it paints with the plain-text fallback. */
export const KATEX_WAIT_MS = 1500;

let ready: Promise<void> | null = null;

/**
 * Resolves when KaTeX, its stylesheet and its main fonts have loaded, or after `timeout` ms (a
 * slow or failed font must not hold a page back; the plain-text fallback then covers it).
 */
export function katexReady(timeout = KATEX_WAIT_MS): Promise<void> {
  if (!ready) {
    ready = preloadKatex()
      .then(() => {
        const fonts = typeof document !== 'undefined' ? document.fonts : undefined;
        if (!fonts) return;
        return Promise.all(KATEX_FACES.map((f) => fonts.load(f))).then(() => undefined);
      })
      .catch(() => undefined);
  }
  return Promise.race([ready, new Promise<void>((resolve) => setTimeout(resolve, timeout))]);
}

/** A lazy loader that also waits for `katexReady()`: `lazy(withKatex(() => import('./Page')))`. */
export function withKatex<T>(load: () => Promise<T>): () => Promise<T> {
  return () => Promise.all([load(), katexReady()]).then(([m]) => m);
}
