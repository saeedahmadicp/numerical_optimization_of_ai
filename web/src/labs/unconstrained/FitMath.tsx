/**
 * One type size for every step formula of a run. `FitGroup` holds the formulas of "This step"
 * (the current step and the hidden layout samples typeset beside it); every `FitBox` inside it
 * is measured, and the group takes the one scale (down to 85 %) at which the widest of them
 * fits its box. Within a run the scale only shrinks, so stepping never makes the type jump
 * between sizes; a new run, a new box width or new fonts start the measurement again.
 *
 * Measured again whenever the result could change: the box resizes (a tab shown again, a phone
 * rotated), KaTeX's HTML arrives or changes (a new step), or a web font finishes loading (KaTeX
 * loads its large delimiter fonts only when a formula first uses them). A formula that still
 * does not fit at 85 % becomes a focusable horizontal scroll region with a fade at the cut edge.
 */
import { useLayoutEffect, useRef, type ReactNode } from 'react';
import { Formula } from '../../ui/components/Formula';

/** The smallest scale, as the shared `<Formula fit>` (scripts stay legible). */
const MIN_SCALE = 0.85;
const SCALE_VAR = '--step-fit';

export function FitGroup({
  run,
  className,
  children,
}: {
  /** Changes when a new run starts (a new trace): the scale is measured from scratch. */
  run: unknown;
  className?: string;
  children: ReactNode;
}) {
  const root = useRef<HTMLDivElement>(null);
  const kept = useRef<{ run: unknown; width: number; scale: number } | null>(null);

  useLayoutEffect(() => {
    const el = root.current;
    if (!el) return;
    let frame = 0;
    let busy = false;
    const boxes = () => [...el.querySelectorAll<HTMLElement>('[data-fit-box]')];
    const need = () => {
      let s = 1;
      for (const b of boxes()) {
        const inner = b.querySelector<HTMLElement>('.katex');
        // A 2 px margin keeps the last glyph (a closing parenthesis) clear of the edge.
        const room = b.clientWidth - 2;
        if (!inner || room <= 0) continue;
        const w = inner.scrollWidth;
        if (w > room) s = Math.min(s, room / w);
      }
      return s;
    };
    const apply = () => {
      if (busy) return;
      busy = true;
      const width = el.clientWidth;
      const prev = kept.current;
      const same = prev !== null && prev.run === run && prev.width === width;
      // Measure at the scale the run already has (or at full size for a new run). Glyph
      // advances do not scale exactly linearly (hinting, rounding): refine twice.
      let scale = same ? prev.scale : 1;
      el.style.setProperty(SCALE_VAR, String(scale));
      for (let i = 0; i < 3; i++) {
        const s = need();
        if (s >= 1 || scale <= MIN_SCALE) break;
        scale = Math.max(MIN_SCALE, scale * s * (i === 0 ? 1 : 0.99));
        el.style.setProperty(SCALE_VAR, String(scale));
      }
      kept.current = { run, width, scale };
      // A formula still too wide at 85 % scrolls (the visible one only: samples are hidden).
      for (const b of boxes()) {
        const inner = b.querySelector<HTMLElement>('.katex');
        const over = !!inner && inner.scrollWidth > b.clientWidth;
        const ghost = b.closest('[data-ghost]') !== null;
        b.toggleAttribute('data-overflow', over);
        if (over && !ghost) {
          b.tabIndex = 0;
          b.setAttribute('role', 'region');
          b.setAttribute('aria-label', b.dataset.fitLabel ?? 'Formula, scrolls sideways');
        } else {
          b.removeAttribute('tabindex');
          b.removeAttribute('role');
          b.removeAttribute('aria-label');
        }
      }
      // Ignore the mutations and resizes this measurement caused.
      requestAnimationFrame(() => {
        busy = false;
      });
    };
    const later = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(apply);
    };
    apply();
    later();
    let width = el.clientWidth;
    const ro = new ResizeObserver(() => {
      if (el.clientWidth === width) return;
      width = el.clientWidth;
      later();
    });
    ro.observe(el);
    const mo = new MutationObserver(later);
    mo.observe(el, { childList: true, subtree: true, characterData: true });
    const fonts = typeof document !== 'undefined' ? document.fonts : undefined;
    const onFonts = () => {
      // New glyph widths: measure from full size again.
      kept.current = null;
      later();
    };
    fonts?.addEventListener?.('loadingdone', onFonts);
    void fonts?.ready.then(later);
    return () => {
      cancelAnimationFrame(frame);
      ro.disconnect();
      mo.disconnect();
      fonts?.removeEventListener?.('loadingdone', onFonts);
    };
  }, [run]);

  return (
    <div ref={root} className={className}>
      {children}
    </div>
  );
}

/** A display formula measured by the enclosing `FitGroup`. */
export function FitBox({
  tex,
  className,
  label,
}: {
  tex: string;
  className?: string;
  /** Accessible name when the formula has to scroll. */
  label?: string;
}) {
  return (
    <div className={className} data-fit-box="" data-fit-label={label}>
      <Formula tex={tex} display />
    </div>
  );
}
