import {
  memo,
  useLayoutEffect,
  useMemo,
  useRef,
  useSyncExternalStore,
  type RefObject,
} from 'react';
import { getKatex, subscribeKatex, type Katex } from '../katex';
import { texToText } from '../texText';

export interface FormulaProps {
  tex: string;
  display?: boolean;
  className?: string;
  /**
   * Accessible name in words ("gradient ∇f"). With a label the formula is one image
   * (`role="img"`); without one, screen readers read KaTeX's MathML.
   */
  label?: string;
  /**
   * While KaTeX loads, show a plain-text rendering (`true`: derived from the TeX) in its place
   * instead of an empty placeholder, so the text is never blank. The plain text is narrower or
   * wider than the typeset formula, so the swap moves the line around it: pages full of math
   * avoid the swap by waiting for KaTeX before their first paint (`withKatex` in ui/katex.ts, used
   * for the lab, method and research routes); the fallback is for surfaces that cannot wait.
   */
  fallback?: boolean | string;
  /**
   * Shrink the type (down to 85 %) so a display formula fits its container without scrolling.
   * When 85 % is still too wide, the formula is typeset in text layout (`\\displaystyle`), which
   * breaks across lines at its top-level relations and operators instead of being clipped. The fit
   * is measured again when the KaTeX web fonts finish loading and when the box or formula resizes.
   */
  fit?: boolean;
  /**
   * Formulas that take turns in one box (a rule filled in step by step) share one fit: the
   * smallest scale any of them needed at this width, so the type does not change size per step.
   */
  fitGroup?: string;
}

const cache = new Map<string, string>();

function render(k: Katex, tex: string, display: boolean): string {
  const key = `${display ? 'D' : 'I'}${tex}`;
  let html = cache.get(key);
  if (html === undefined) {
    html = k.renderToString(tex, {
      displayMode: display,
      throwOnError: false,
      output: 'htmlAndMathml',
      strict: 'ignore',
      // \htmlClass lets MethodCards tag symbols they highlight live.
      trust: (ctx) => ctx.command === '\\htmlClass',
    });
    if (cache.size > 500) cache.clear();
    cache.set(key, html);
  }
  return html;
}

/**
 * KaTeX math, rendered once per (tex, mode) and memoized. KaTeX itself loads lazily; on a page
 * loaded through `withKatex` it is already in, so the formula renders typeset from the first paint.
 */
export const Formula = memo(function Formula({
  tex,
  display = false,
  className,
  label,
  fallback,
  fit = false,
  fitGroup,
}: FormulaProps) {
  const k = useSyncExternalStore(subscribeKatex, getKatex, getKatex);
  const html = useMemo(() => (k ? render(k, tex, display) : null), [k, tex, display]);
  // Wrapping fallback for `fit`: text layout breaks lines at top-level relations and operators.
  const wrapHtml = useMemo(
    () => (k && fit && display ? render(k, `\\displaystyle ${tex}`, false) : null),
    [k, tex, fit, display],
  );
  const Tag = display ? 'div' : 'span';
  if (html === null) {
    // Placeholder while KaTeX loads: keeps the line box, exposes the source to assistive tech.
    const text = fallback ? (typeof fallback === 'string' ? fallback : texToText(tex)) : null;
    return (
      <Tag
        className={className}
        role={label ? 'img' : undefined}
        aria-label={label}
        style={{
          minHeight: display ? '2.4em' : '1.2em',
          color: text ? 'var(--color-text-3)' : undefined,
          fontStyle: text ? 'italic' : undefined,
        }}
      >
        {text ?? (label ? null : <span className="visually-hidden">{texToText(tex)}</span>)}
      </Tag>
    );
  }
  return (
    <FitTag
      Tag={Tag}
      fit={fit}
      className={className}
      label={label}
      html={html}
      wrapHtml={wrapHtml}
      group={fitGroup}
    />
  );
});

/**
 * Smallest scale `fit` applies before it falls back to the wrapping layout. At 85 % a 1.2em
 * display formula in a 13-px box keeps its scripts at about 10 px.
 */
const MIN_FIT = 0.85;

/** Shared fits (`fitGroup`): the smallest scale a group needed at a box width. */
const GROUPS = new Map<string, { width: number; scale: number }>();

/**
 * Renders KaTeX HTML; with `fit`, scales the font down until the formula fits its box, and swaps
 * in the wrapping layout (`wrapHtml`) when even 85 % does not fit. Measured again after web fonts
 * load (KaTeX's fonts arrive after the first paint) and whenever the box width changes.
 */
function FitTag({
  Tag,
  fit,
  className,
  label,
  html,
  wrapHtml,
  group,
}: {
  Tag: 'div' | 'span';
  fit: boolean;
  className?: string;
  label?: string;
  html: string;
  wrapHtml: string | null;
  group?: string;
}) {
  const box = useRef<HTMLElement>(null);
  useLayoutEffect(() => {
    const el = box.current;
    if (!fit || !el) return;
    let wrapped = false;
    let measuring = false;
    const apply = () => {
      if (measuring) return;
      measuring = true;
      if (wrapped) {
        // Measure the one-line layout again (the box may have grown).
        el.innerHTML = html;
        el.removeAttribute('data-wrapped');
        el.style.textAlign = '';
        wrapped = false;
      }
      el.style.fontSize = '';
      const inner = el.querySelector<HTMLElement>('.katex');
      const avail = el.clientWidth;
      const need = inner ? inner.scrollWidth : 0;
      let scale = inner && need > avail && avail > 0 ? avail / need : 1;
      if (group && avail > 0) {
        const g = GROUPS.get(group);
        if (g && g.width === avail) scale = Math.min(scale, g.scale);
        GROUPS.set(group, { width: avail, scale });
        if (GROUPS.size > 200) GROUPS.delete(GROUPS.keys().next().value as string);
      }
      if (inner && scale < 1) {
        if (scale >= MIN_FIT || !wrapHtml) {
          el.style.fontSize = `${Math.max(MIN_FIT, scale) * 100}%`;
        } else {
          el.innerHTML = wrapHtml;
          el.setAttribute('data-wrapped', '');
          el.style.textAlign = 'center';
          wrapped = true;
        }
      }
      measuring = false;
    };
    // Only a change of the box width needs a new fit (wrapping changes its height).
    let width = -1;
    const ro = new ResizeObserver(() => {
      if (el.clientWidth === width) return;
      width = el.clientWidth;
      apply();
    });
    apply();
    width = el.clientWidth;
    ro.observe(el);
    const fonts = typeof document !== 'undefined' ? document.fonts : undefined;
    let alive = true;
    const onFonts = () => alive && apply();
    void fonts?.ready.then(onFonts);
    fonts?.addEventListener?.('loadingdone', onFonts);
    return () => {
      alive = false;
      ro.disconnect();
      fonts?.removeEventListener?.('loadingdone', onFonts);
    };
  }, [fit, html, wrapHtml, group]);
  return (
    <Tag
      ref={box as RefObject<HTMLDivElement & HTMLSpanElement>}
      className={className}
      role={label ? 'img' : undefined}
      aria-label={label}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
