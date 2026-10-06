/**
 * A horizontally scrollable box that says so: a fade edge on each side where content is cut
 * off (a wide KaTeX display), updated on scroll and resize. Focusable only while it scrolls,
 * so keyboard users can reach the hidden part.
 */
import { useEffect, useRef, useState, type ReactNode } from 'react';
import styles from './StochasticLab.module.css';

export function ScrollFade({ className, children }: { className?: string; children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const [edges, setEdges] = useState({ left: false, right: false });
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let frame = 0;
    // Coalesced to one read per frame (the formula re-renders on every playback frame).
    const update = () => {
      if (frame) return;
      frame = requestAnimationFrame(measure);
    };
    const measure = () => {
      frame = 0;
      const left = el.scrollLeft > 1;
      const right = el.scrollLeft + el.clientWidth < el.scrollWidth - 1;
      setEdges((e) => (e.left === left && e.right === right ? e : { left, right }));
    };
    update();
    el.addEventListener('scroll', update, { passive: true });
    const ro = new ResizeObserver(update);
    ro.observe(el);
    // KaTeX renders into a child: watch it too (the formula changes width with the numbers).
    const mo = new MutationObserver(update);
    mo.observe(el, { childList: true, subtree: true, characterData: true });
    return () => {
      cancelAnimationFrame(frame);
      el.removeEventListener('scroll', update);
      ro.disconnect();
      mo.disconnect();
    };
  }, []);
  const scrolls = edges.left || edges.right;
  return (
    <div
      ref={ref}
      className={`${styles.scrollFade} ${className ?? ''}`}
      data-fade-left={edges.left || undefined}
      data-fade-right={edges.right || undefined}
      tabIndex={scrolls ? 0 : undefined}
      role={scrolls ? 'group' : undefined}
      aria-label={scrolls ? 'Formula (scrolls sideways)' : undefined}
    >
      {children}
    </div>
  );
}
