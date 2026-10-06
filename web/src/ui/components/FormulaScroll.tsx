import { useLayoutEffect, useRef, useState, type ReactNode } from 'react';

/**
 * A box for a formula that may be wider than its column. When the content overflows, the box
 * becomes a keyboard-reachable scroll region (tabindex 0, role region, "Formula (scrolls
 * sideways)") with a fade at the right edge while more is hidden; when it fits, it is a plain
 * box (no tab stop, no landmark). Use it around display math that cannot wrap.
 */
export function FormulaScroll({
  children,
  label = 'Formula (scrolls sideways)',
  className,
}: {
  children: ReactNode;
  label?: string;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [overflow, setOverflow] = useState(false);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => {
      const over = el.scrollWidth > el.clientWidth + 1;
      setOverflow(over);
      const more = over && el.scrollLeft + el.clientWidth < el.scrollWidth - 1;
      if (more) el.setAttribute('data-more', '');
      else el.removeAttribute('data-more');
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    if (el.firstElementChild) ro.observe(el.firstElementChild);
    el.addEventListener('scroll', measure, { passive: true });
    return () => {
      ro.disconnect();
      el.removeEventListener('scroll', measure);
    };
  }, []);
  return (
    <div
      ref={ref}
      className={`formula-scroll ${className ?? ''}`}
      tabIndex={overflow ? 0 : undefined}
      role={overflow ? 'region' : undefined}
      aria-label={overflow ? label : undefined}
    >
      {children}
    </div>
  );
}
