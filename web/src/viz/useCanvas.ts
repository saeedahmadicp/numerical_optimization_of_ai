import { useCallback, useEffect, useLayoutEffect, useRef, useState, type RefObject } from 'react';
import { resetLabelRects } from './labelRects';

export interface CanvasSize {
  /** CSS pixels. */
  width: number;
  height: number;
  /** devicePixelRatio used for the backing store. */
  dpr: number;
}

export type DrawFn = (ctx: CanvasRenderingContext2D, size: CanvasSize) => void;

/**
 * A HiDPI canvas that tracks its CSS box (ResizeObserver) and devicePixelRatio, and redraws at
 * most once per animation frame. `draw` runs after every render (coalesced) and on resize, so
 * callers just re-render with new props; the context arrives pre-scaled to CSS pixels.
 */
export function useCanvas(
  draw: DrawFn,
  options: {
    maxDpr?: number;
    /**
     * A static layer (a contour raster, a landscape): redraw only when one of these values
     * changes (compared with `Object.is`), or the size, DPR or fonts change, instead of after
     * every render. A playing player re-renders 60 times a second; repainting a full-size,
     * high-DPI layer each time is the main cost of playback on a 2× or 3× screen.
     */
    deps?: readonly unknown[];
  } = {},
) {
  const { maxDpr = 3, deps } = options;
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [size, setSize] = useState<CanvasSize>({ width: 0, height: 0, dpr: 1 });
  const drawRef = useRef(draw);
  const sizeRef = useRef(size);
  const frame = useRef(0);

  useLayoutEffect(() => {
    drawRef.current = draw;
  });

  const paint = useCallback(() => {
    frame.current = 0;
    const canvas = canvasRef.current;
    const s = sizeRef.current;
    if (!canvas || s.width === 0 || s.height === 0) return;
    const w = Math.round(s.width * s.dpr),
      h = Math.round(s.height * s.dpr);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(s.dpr, 0, 0, s.dpr, 0, 0);
    ctx.clearRect(0, 0, s.width, s.height);
    resetLabelRects(ctx);
    drawRef.current(ctx, s);
  }, []);

  const redraw = useCallback(() => {
    if (frame.current) return;
    frame.current = requestAnimationFrame(paint);
  }, [paint]);

  // Size + DPR tracking.
  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const measure = () => {
      const r = canvas.getBoundingClientRect();
      const dpr = Math.min(maxDpr, window.devicePixelRatio || 1);
      const next = { width: Math.round(r.width), height: Math.round(r.height), dpr };
      const prev = sizeRef.current;
      if (prev.width !== next.width || prev.height !== next.height || prev.dpr !== next.dpr) {
        sizeRef.current = next;
        setSize(next);
      }
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(canvas);
    let mq: MediaQueryList | null = null;
    const onDpr = () => {
      measure();
      mq?.removeEventListener('change', onDpr);
      mq = window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
      mq.addEventListener('change', onDpr);
    };
    onDpr();
    return () => {
      ro.disconnect();
      mq?.removeEventListener('change', onDpr);
    };
  }, [maxDpr]);

  // Redraw after every render (or when `deps` or the size change) and when web fonts arrive.
  const lastDeps = useRef<readonly unknown[] | null>(null);
  useEffect(() => {
    if (deps) {
      const now = [...deps, size];
      const prev = lastDeps.current;
      if (prev && prev.length === now.length && prev.every((d, i) => Object.is(d, now[i]))) return;
      lastDeps.current = now;
    }
    redraw();
  });
  // Late fonts (KaTeX faces requested by canvas math labels) also trigger a redraw.
  useEffect(() => {
    let alive = true;
    const fonts = document.fonts;
    fonts?.ready.then(() => alive && redraw());
    const onLoaded = () => alive && redraw();
    fonts?.addEventListener?.('loadingdone', onLoaded);
    return () => {
      alive = false;
      fonts?.removeEventListener?.('loadingdone', onLoaded);
      cancelAnimationFrame(frame.current);
      frame.current = 0;
    };
  }, [redraw]);

  return { canvasRef, size, redraw };
}

/** CSS size + devicePixelRatio of an element, tracked with ResizeObserver. */
export function useElementSize(ref: RefObject<HTMLElement | null>, maxDpr = 3): CanvasSize {
  const [size, setSize] = useState<CanvasSize>({ width: 0, height: 0, dpr: 1 });
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => {
      const r = el.getBoundingClientRect();
      const dpr = Math.min(maxDpr, window.devicePixelRatio || 1);
      setSize((prev) =>
        prev.width === Math.round(r.width) &&
        prev.height === Math.round(r.height) &&
        prev.dpr === dpr
          ? prev
          : { width: Math.round(r.width), height: Math.round(r.height), dpr },
      );
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref, maxDpr]);
  return size;
}
