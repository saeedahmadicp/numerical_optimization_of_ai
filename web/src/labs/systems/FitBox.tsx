/**
 * Scales its content (a KaTeX display) down until it fits the box in both width and height, so
 * the last row of an aligned formula is never cut off in a short panel. Fits again when the box
 * resizes and once the web fonts have loaded (KaTeX metrics change then).
 *
 * `sizers` are the largest contents of the run (rendered hidden in the same grid cell): the box
 * then measures the maximum over the run, so its content keeps one size and one scale while the
 * playback steps through it (no layout shift between steps).
 */
import { useLayoutEffect, useRef, type ReactNode } from 'react';

export function FitBox({
  className,
  fitKey,
  min = 0.6,
  sizers,
  children,
}: {
  className?: string;
  /** Changes when the measured content changes (the run, or the formula's TeX without sizers). */
  fitKey: string;
  /** Smallest scale (the content scrolls below it). */
  min?: number;
  /** The run's largest contents, rendered invisibly so the box reserves their size. */
  sizers?: readonly ReactNode[];
  children: ReactNode;
}) {
  const box = useRef<HTMLDivElement>(null);
  const inner = useRef<HTMLDivElement>(null);
  const fitRef = useRef<() => void>(() => {});
  // Every render: the inner box holds the maximum of the run (sizers), so the scale changes only
  // when the run, the box or a step larger than every sizer changes it.
  useLayoutEffect(() => {
    const fit = () => {
      const el = box.current,
        ic = inner.current;
      if (!el || !ic) return;
      ic.style.fontSize = '';
      const availW = el.clientWidth,
        availH = el.clientHeight;
      const needW = ic.scrollWidth,
        needH = ic.scrollHeight;
      if (availW <= 0 || availH <= 0 || needW <= 0 || needH <= 0) return;
      const s = Math.min(1, availW / needW, availH / needH);
      if (s < 1) ic.style.fontSize = `${Math.max(min, s * 0.98) * 100}%`;
    };
    fitRef.current = fit;
    fit();
  });
  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    let alive = true;
    const apply = () => {
      if (alive) fitRef.current();
    };
    const ro = new ResizeObserver(apply);
    ro.observe(el);
    void document.fonts?.ready.then(apply);
    return () => {
      alive = false;
      ro.disconnect();
    };
  }, [fitKey]);
  return (
    <div ref={box} className={className}>
      <div
        ref={inner}
        style={{ display: 'grid', width: 'max-content', maxWidth: 'none', alignItems: 'start' }}
      >
        <div style={{ gridArea: '1 / 1', display: 'flow-root' }}>{children}</div>
        {sizers?.map((s, i) => (
          <div
            key={i}
            aria-hidden="true"
            style={{ gridArea: '1 / 1', display: 'flow-root', visibility: 'hidden' }}
          >
            {s}
          </div>
        ))}
      </div>
    </div>
  );
}
