import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type FocusEvent as RFocusEvent,
  type KeyboardEvent as RKeyboardEvent,
  type PointerEvent as RPointerEvent,
} from 'react';
import { CONTOUR_MAPS, type ChartColors } from '../ui/colors';
import { useChartColors } from '../ui/theme';
import { SciText } from '../ui/components/Num';
import { ViewToolbar } from './ViewToolbar';
import { sci, sig } from '../core/format';
import { useCanvas, useElementSize } from './useCanvas';
import { linearScale, type Scale } from './scales';
import { drawAxes } from './axes';
import { drawPathLayer, type PathSpec } from './PathLayer';
import { drawOverlayLabels, drawOverlays2D, type Overlay2D } from './overlays2d';
import { addLabelRect } from './labelRects';
import { drawMath, iterateRuns, starRuns } from './mathText';
import { computeField, levelSpecFor, type FieldResult, type LevelScale } from './contourField';
import styles from './viz.module.css';

export type Domain2D = [[number, number], [number, number]];

/** What overlays receive: scales from data to CSS pixels, and the canvas size. */
export interface View2D {
  x: Scale;
  y: Scale;
  width: number;
  height: number;
  dpr: number;
  colors: ChartColors;
  toPx: (x: number, y: number) => [number, number];
}

export interface Contour2DProps {
  /** f(x, y) — evaluated on the main thread on a coarse grid; banding runs in a worker. */
  f: (x: number, y: number) => number;
  domain: Domain2D;
  /** Cache key for the field (usually the problem id + anything that changes f). */
  cacheKey: string;
  levels?: number;
  levelScale?: LevelScale;
  /** Known minimum value (improves log spacing near the minimizer). */
  fMin?: number | null;
  paths?: readonly PathSpec[];
  /** Playhead for the paths. */
  t?: number;
  ease?: boolean;
  /** Known minimizers 𝐱⋆, drawn as + crosses. */
  minima?: readonly (readonly [number, number])[];
  /** Label minimizers with their coordinates ("𝐱⋆ = (1, 1)"); 'auto' = when there is one. */
  minimaLabels?: boolean | 'auto';
  /** A shared start point 𝐱₀: a hollow ring in ink with its label (set `start: false` on paths). */
  start?: readonly [number, number] | null;
  /** Declarative geometry drawn over the field, under the paths (see overlays2d.ts). */
  overlays?: readonly Overlay2D[];
  /** Dash steps that leave the view, with chevrons and a label at the exit (default true). */
  offViewLabels?: boolean;
  /**
   * Click (without dragging), or Enter on the keyboard crosshair → data coordinates. Enables the
   * crosshair cursor.
   */
  onPick?: (p: [number, number]) => void;
  /** Custom drawing after the field and before the paths (trust regions, simplices, ...). */
  overlay?: (ctx: CanvasRenderingContext2D, view: View2D) => void;
  /** Allow wheel/drag/pinch pan & zoom (default true). */
  interactive?: boolean;
  showAxes?: boolean;
  showReadout?: boolean;
  /** Keep one data unit the same length on both axes (default true). */
  equalAspect?: boolean;
  ariaLabel: string;
  className?: string;
  /** Hide the zoom buttons (hero, thumbnails). */
  chrome?: boolean;
  /** Axis names drawn inside the plot (default `['x', 'y']`; `null` hides them). */
  axisLabels?: [string, string] | null;
  /** Name of the plotted value in the hover readout and the keyboard announcement (default `f`; e.g. `‖F‖`). */
  valueLabel?: string;
  /** Custom drawing after the paths (labels that must sit on top of them, e.g. off-view chips). */
  overlayAfter?: (ctx: CanvasRenderingContext2D, view: View2D) => void;
  /**
   * The window shown first and restored by Reset (default: the whole `domain`). Levels and
   * colors still come from `domain`, so a framed view keeps the whole-domain color scale.
   */
  initialView?: Domain2D;
  /** Called once per field (view/size/theme) when its raster is first shown: the landscape is visible. */
  onFieldReady?: () => void;
}

// ── Worker + cache ───────────────────────────────────────────────────────────────────

interface CachedField {
  view: Domain2D;
  canvas: HTMLCanvasElement;
  result: FieldResult;
  nx: number;
  ny: number;
}

const FIELD_CACHE = new Map<string, CachedField>();
const CACHE_LIMIT = 16;

let worker: Worker | null | undefined;
let nextId = 1;
const pending = new Map<number, (r: FieldResult) => void>();

function getWorker(): Worker | null {
  if (worker !== undefined) return worker;
  try {
    worker = new Worker(new URL('./contour.worker.ts', import.meta.url), { type: 'module' });
    worker.onmessage = (e: MessageEvent<FieldResult>) => {
      const cb = pending.get(e.data.id);
      pending.delete(e.data.id);
      cb?.(e.data);
    };
    worker.onerror = () => {
      worker = null;
    };
  } catch {
    worker = null;
  }
  return worker;
}

function requestField(req: Omit<Parameters<typeof computeField>[0], 'id'>): Promise<FieldResult> {
  const id = nextId++;
  const w = getWorker();
  if (!w) return Promise.resolve(computeField({ ...req, id }));
  return new Promise((resolve) => {
    pending.set(id, resolve);
    w.postMessage({ ...req, id }, [req.values.buffer]);
  });
}

function rememberField(key: string, value: CachedField) {
  FIELD_CACHE.delete(key);
  FIELD_CACHE.set(key, value);
  while (FIELD_CACHE.size > CACHE_LIMIT)
    FIELD_CACHE.delete(FIELD_CACHE.keys().next().value as string);
}

// ── Helpers ──────────────────────────────────────────────────────────────────────────

function fitAspect(d: Domain2D, w: number, h: number): Domain2D {
  if (w <= 0 || h <= 0) return d;
  const [[x0, x1], [y0, y1]] = d;
  const dx = x1 - x0,
    dy = y1 - y0;
  const target = w / h;
  if (dx / dy > target) {
    const ny = dx / target,
      cy = (y0 + y1) / 2;
    return [
      [x0, x1],
      [cy - ny / 2, cy + ny / 2],
    ];
  }
  const nx = dy * target,
    cx = (x0 + x1) / 2;
  return [
    [cx - nx / 2, cx + nx / 2],
    [y0, y1],
  ];
}

const viewKey = (v: Domain2D) =>
  v
    .flat()
    .map((n) => n.toPrecision(6))
    .join(',');

const fmtPt = (v: number) => sig(v, 3);

const fmtF = (v: number) => (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5 ? sci(v, 3) : sig(v, 4));

/**
 * Filled-band contour plot with iso-lines, pan & zoom, click-to-pick and animated paths.
 *
 * Levels are fixed per problem (`levelSpecFor` over the full `domain`), so colors mean the same
 * f in every view. Keyboard: the plot is focusable; arrows move a crosshair (Shift = larger
 * steps; the view follows), + / − zoom about it, 0 resets the view, Enter sets the start point.
 * Touch: one finger scrolls the page and a tap picks; two fingers pan and zoom.
 */
export function Contour2D({
  f,
  domain,
  cacheKey,
  levels = 16,
  levelScale = 'auto',
  fMin = null,
  paths = [],
  t = 0,
  ease = true,
  minima = [],
  minimaLabels = 'auto',
  start = null,
  overlays,
  offViewLabels = true,
  onPick,
  overlay,
  interactive = true,
  showAxes = true,
  showReadout = true,
  equalAspect = true,
  ariaLabel,
  className,
  chrome = true,
  axisLabels = ['x', 'y'],
  valueLabel = 'f',
  overlayAfter,
  initialView,
  onFieldReady,
}: Contour2DProps) {
  const colors = useChartColors();
  const [userView, setUserView] = useState<Domain2D | null>(null);
  /** Pointer position in CSS px inside the plot (data coordinates follow the current view). */
  const [hoverPx, setHoverPx] = useState<[number, number] | null>(null);
  /** Keyboard crosshair in data coordinates (shown while the plot has keyboard focus). */
  const [kb, setKb] = useState<[number, number] | null>(null);
  const descId = useId();
  /** The most recently computed field (shown, remapped, while a newer one is computed). */
  const [latest, setLatest] = useState<CachedField | null>(null);
  const [panning, setPanning] = useState(false);

  // Reset the user's pan/zoom when the problem, its domain or the initial view changes.
  const domainKey = viewKey(domain) + cacheKey;
  const frameKey = domainKey + (initialView ? viewKey(initialView) : '');
  const [lastDomainKey, setLastDomainKey] = useState(frameKey);
  if (lastDomainKey !== frameKey) {
    setLastDomainKey(frameKey);
    setUserView(null);
    setLatest(null);
    setKb(null);
  }

  // Fixed levels for the whole problem: computed once per problem/domain, not per view.
  const levelSpec = useMemo(
    () => levelSpecFor(f, domain, levelScale, fMin),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [domainKey, levelScale, fMin],
  );

  /** The visible data window for a canvas of w × h CSS pixels. */
  const home = initialView ?? domain;
  const viewFor = (w: number, h: number): Domain2D =>
    userView ?? (equalAspect ? fitAspect(home, w, h) : home);

  // Field key for the current size (known after the first layout).
  const wrap = useRef<HTMLDivElement>(null);
  const size = useElementSize(wrap);
  const view = viewFor(size.width, size.height);
  const xsNow = linearScale(view[0], [0, size.width]);
  const ysNow = linearScale(view[1], [size.height, 0]);
  const hover: [number, number] | null = hoverPx
    ? [xsNow.invert(hoverPx[0]), ysNow.invert(hoverPx[1])]
    : kb;
  const key = `${cacheKey}|${viewKey(view)}|${size.width}x${size.height}@${size.dpr}|${colors.mode}|${levels}|${levelScale}|${fMin ?? ''}`;
  const field = FIELD_CACHE.get(key) ?? latest;

  const { canvasRef: fieldRef } = useCanvas(
    (ctx, s) => {
      const v = viewFor(s.width, s.height);
      const xs = linearScale(v[0], [0, s.width]);
      const ys = linearScale(v[1], [s.height, 0]);
      ctx.fillStyle = colors.surface;
      ctx.fillRect(0, 0, s.width, s.height);
      if (!field) return;
      // Draw the cached raster mapped from its own view to the current one (instant pan/zoom).
      const [[cx0, cx1], [cy0, cy1]] = field.view;
      const left = xs(cx0),
        right = xs(cx1),
        top = ys(cy1),
        bottom = ys(cy0);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(field.canvas, left, top, right - left, bottom - top);
      // Iso-lines in data space → crisp at any zoom.
      const { segments, segLevel } = field.result;
      const gx = (i: number) => xs(cx0 + (i / (field.nx - 1)) * (cx1 - cx0));
      const gy = (j: number) => ys(cy0 + (j / (field.ny - 1)) * (cy1 - cy0));
      ctx.lineCap = 'round';
      for (const strong of [false, true]) {
        ctx.strokeStyle = strong ? colors.isoStrong : colors.iso;
        ctx.lineWidth = strong ? 0.9 : 0.7;
        ctx.beginPath();
        for (let k = 0; k < segLevel.length; k++) {
          if ((segLevel[k] % 4 === 0) !== strong) continue;
          const o = k * 4;
          ctx.moveTo(gx(segments[o]), gy(segments[o + 1]));
          ctx.lineTo(gx(segments[o + 2]), gy(segments[o + 3]));
        }
        ctx.stroke();
      }
      // The field is static during playback: repaint it only when its inputs change.
    },
    { deps: [field, userView, equalAspect, viewKey(home), colors] },
  );

  const viewRef = useRef(view);
  const fRef = useRef(f);
  const onFieldReadyRef = useRef(onFieldReady);
  useLayoutEffect(() => {
    viewRef.current = view;
    fRef.current = f;
    onFieldReadyRef.current = onFieldReady;
  });
  // A cached field is shown at once: report it too (once per field key).
  const reported = useRef('');
  useEffect(() => {
    if (!FIELD_CACHE.has(key) || reported.current === key) return;
    reported.current = key;
    onFieldReadyRef.current?.();
  }, [key]);

  // (Re)compute the field when the view, size, theme or function changes. Debounced while
  // interacting; cached per problem/view/size/theme.
  const { width, height, dpr } = size;
  const hasField = latest !== null;
  useEffect(() => {
    if (width === 0 || height === 0 || FIELD_CACHE.has(key)) return;
    let cancelled = false;
    const run = async () => {
      const v = viewRef.current;
      const rasterScale = Math.min(dpr, 1.5);
      const W = Math.max(1, Math.round(width * rasterScale));
      const H = Math.max(1, Math.round(height * rasterScale));
      const nx = Math.max(48, Math.min(360, Math.round(width / 2.4)));
      const ny = Math.max(48, Math.min(360, Math.round(height / 2.4)));
      const values = new Float64Array(nx * ny);
      const fn = fRef.current;
      const [[x0, x1], [y0, y1]] = v;
      for (let j = 0; j < ny; j++) {
        const y = y0 + (j / (ny - 1)) * (y1 - y0);
        for (let i = 0; i < nx; i++) {
          const val = fn(x0 + (i / (nx - 1)) * (x1 - x0), y);
          values[j * nx + i] = Number.isFinite(val) ? val : NaN;
        }
      }
      const lut = CONTOUR_MAPS[colors.mode].lut;
      const result = await requestField({
        values,
        nx,
        ny,
        width: W,
        height: H,
        levels,
        scale: levelScale,
        fMin,
        levelSpec,
        lut: lut.slice(),
      });
      if (cancelled) return;
      const canvas = document.createElement('canvas');
      canvas.width = result.width;
      canvas.height = result.height;
      canvas
        .getContext('2d')
        ?.putImageData(
          new ImageData(new Uint8ClampedArray(result.pixels), result.width, result.height),
          0,
          0,
        );
      const entry: CachedField = { view: v, canvas, result, nx, ny };
      rememberField(key, entry);
      setLatest(entry);
      reported.current = key;
      onFieldReadyRef.current?.();
    };
    const timer = window.setTimeout(run, hasField ? (panning ? 220 : 90) : 0);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [
    key,
    width,
    height,
    dpr,
    colors.mode,
    levels,
    levelScale,
    fMin,
    panning,
    hasField,
    levelSpec,
  ]);

  // ── Overlay: axes, custom overlay, paths, minima, crosshair ───────────────────────────
  const { canvasRef: overlayRef } = useCanvas((ctx, size) => {
    const v = viewFor(size.width, size.height);
    const xs = linearScale(v[0], [0, size.width]);
    const ys = linearScale(v[1], [size.height, 0]);
    const toPx = (x: number, y: number): [number, number] => [xs(x), ys(y)];
    const view2d: View2D = {
      x: xs,
      y: ys,
      width: size.width,
      height: size.height,
      dpr: size.dpr,
      colors,
      toPx,
    };
    if (showAxes) {
      drawAxes(ctx, {
        x: xs,
        y: ys,
        frame: { left: 0, top: 0, right: size.width, bottom: size.height },
        colors,
        dpr: size.dpr,
        grid: false,
        baseline: false,
        inset: true,
        xLabel: axisLabels?.[0],
        yLabel: axisLabels?.[1],
      });
    }
    // Overlay labels wait for the label pass after the paths (no path is drawn over a label).
    const overlayView = {
      toPx,
      toData: (px: number, py: number): [number, number] => [xs.invert(px), ys.invert(py)],
      width: size.width,
      height: size.height,
      dpr: size.dpr,
      colors,
    };
    const overlayLabels = overlays?.length
      ? drawOverlays2D(ctx, overlayView, overlays, { labels: 'defer' })
      : [];
    overlay?.(ctx, view2d);
    // Minimizers 𝐱⋆: + crosses with a halo, labelled with their coordinates.
    const labelMin = minimaLabels === 'auto' ? minima.length === 1 : minimaLabels;
    for (const [mx, my] of minima) {
      const [px, py] = toPx(mx, my);
      ctx.save();
      ctx.lineCap = 'round';
      for (const [w, c] of [
        [3.5, colors.halo],
        [1.5, colors.text],
      ] as const) {
        ctx.strokeStyle = c;
        ctx.lineWidth = w;
        ctx.beginPath();
        ctx.moveTo(px - 5.5, py);
        ctx.lineTo(px + 5.5, py);
        ctx.moveTo(px, py - 5.5);
        ctx.lineTo(px, py + 5.5);
        ctx.stroke();
      }
      ctx.restore();
      if (labelMin && px > 0 && px < size.width && py > 0 && py < size.height) {
        const right = px < size.width - 110;
        const w = drawMath(
          ctx,
          starRuns('x', `(${fmtPt(mx)}, ${fmtPt(my)})`),
          right ? px + 10 : px - 10,
          py - 9,
          {
            size: 12,
            align: right ? 'left' : 'right',
            color: colors.text2,
            halo: colors.halo,
          },
        );
        addLabelRect(ctx, { x: right ? px + 10 : px - 10 - w, y: py - 18, w, h: 14 });
      }
    }
    if (start) {
      const [sx, sy] = toPx(start[0], start[1]);
      ctx.save();
      ctx.lineWidth = 4;
      ctx.strokeStyle = colors.halo;
      ctx.beginPath();
      ctx.arc(sx, sy, 5, 0, Math.PI * 2);
      ctx.stroke();
      ctx.lineWidth = 1.75;
      ctx.strokeStyle = colors.text;
      ctx.stroke();
      ctx.restore();
      const w0 = drawMath(ctx, iterateRuns('x', 0), sx - 9, sy - 8, {
        size: 12,
        align: 'right',
        color: colors.text2,
        halo: colors.halo,
      });
      addLabelRect(ctx, { x: sx - 9 - w0, y: sy - 18, w: w0, h: 16 });
    }
    drawPathLayer(ctx, paths, {
      t,
      toPx,
      halo: colors.halo,
      ease,
      bounds: { left: 0, top: 0, right: size.width, bottom: size.height },
      offViewLabels,
      textColor: colors.text2,
    });
    overlayAfter?.(ctx, view2d);
    drawOverlayLabels(ctx, overlayView, overlayLabels);
    if (hover && !panning) {
      const [px, py] = toPx(hover[0], hover[1]);
      ctx.save();
      if (!hoverPx && kb) {
        // Keyboard crosshair: a ringed target so it reads as "the point Enter will pick".
        ctx.strokeStyle = colors.halo;
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.arc(px, py, 7, 0, Math.PI * 2);
        ctx.stroke();
        ctx.strokeStyle = colors.accent;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      ctx.strokeStyle = colors.crosshair;
      ctx.lineWidth = 1 / size.dpr;
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(Math.round(px * size.dpr) / size.dpr + 0.5 / size.dpr, 0);
      ctx.lineTo(Math.round(px * size.dpr) / size.dpr + 0.5 / size.dpr, size.height);
      ctx.moveTo(0, Math.round(py * size.dpr) / size.dpr + 0.5 / size.dpr);
      ctx.lineTo(size.width, Math.round(py * size.dpr) / size.dpr + 0.5 / size.dpr);
      ctx.stroke();
      ctx.restore();
    }
  });

  // ── Interaction ───────────────────────────────────────────────────────────────────────
  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const gesture = useRef<{
    startX: number;
    startY: number;
    moved: boolean;
    view: Domain2D;
    dist?: number;
    mid?: [number, number];
  } | null>(null);

  const toData = useCallback((clientX: number, clientY: number): [number, number] | null => {
    const r = wrap.current?.getBoundingClientRect();
    if (!r) return null;
    const v = viewRef.current;
    const xs = linearScale(v[0], [0, r.width]);
    const ys = linearScale(v[1], [r.height, 0]);
    return [xs.invert(clientX - r.left), ys.invert(clientY - r.top)];
  }, []);

  const zoomAt = useCallback((cx: number, cy: number, factor: number, from?: Domain2D) => {
    const r = wrap.current?.getBoundingClientRect();
    if (!r) return;
    const v = from ?? viewRef.current;
    const ax = v[0][0] + ((cx - r.left) / r.width) * (v[0][1] - v[0][0]);
    const ay = v[1][1] - ((cy - r.top) / r.height) * (v[1][1] - v[1][0]);
    const nv: Domain2D = [
      [ax + (v[0][0] - ax) * factor, ax + (v[0][1] - ax) * factor],
      [ay + (v[1][0] - ay) * factor, ay + (v[1][1] - ay) * factor],
    ];
    const span = nv[0][1] - nv[0][0];
    if (span < 1e-9 || span > 1e7) return;
    setUserView(nv);
  }, []);

  /** Zoom about a point given in data coordinates. */
  const zoomAbout = useCallback((ax: number, ay: number, factor: number) => {
    const v = viewRef.current;
    const nv: Domain2D = [
      [ax + (v[0][0] - ax) * factor, ax + (v[0][1] - ax) * factor],
      [ay + (v[1][0] - ay) * factor, ay + (v[1][1] - ay) * factor],
    ];
    const span = nv[0][1] - nv[0][0];
    if (span < 1e-9 || span > 1e7) return;
    setUserView(nv);
  }, []);

  useEffect(() => {
    const el = wrap.current;
    if (!el || !interactive) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
      zoomAt(e.clientX, e.clientY, Math.exp(Math.max(-1, Math.min(1, dy * 0.0016))));
      setPanning(true);
      window.clearTimeout(wheelTimer.current);
      wheelTimer.current = window.setTimeout(() => setPanning(false), 160);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [interactive, zoomAt]);
  const wheelTimer = useRef(0);

  const onPointerDown = (e: RPointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    const pts = [...pointers.current.values()];
    gesture.current = {
      startX: e.clientX,
      startY: e.clientY,
      moved: pts.length > 1,
      view: viewRef.current,
      dist: pts.length === 2 ? Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y) : undefined,
      mid: pts.length === 2 ? [(pts[0].x + pts[1].x) / 2, (pts[0].y + pts[1].y) / 2] : undefined,
    };
  };

  const onPointerMove = (e: RPointerEvent<HTMLDivElement>) => {
    if (e.pointerType === 'mouse' || !pointers.current.size) {
      const r = wrap.current?.getBoundingClientRect();
      if (r) setHoverPx([e.clientX - r.left, e.clientY - r.top]);
    }
    const g = gesture.current;
    if (!g || !pointers.current.has(e.pointerId)) return;
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (!interactive) return;
    const pts = [...pointers.current.values()];
    if (pts.length === 2 && g.dist && g.mid) {
      // Two fingers: pinch zooms about the start midpoint, moving the midpoint pans.
      const d = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
      const r = wrap.current!.getBoundingClientRect();
      const factor = g.dist / Math.max(1, d);
      const v = g.view;
      const ax = v[0][0] + ((g.mid[0] - r.left) / r.width) * (v[0][1] - v[0][0]);
      const ay = v[1][1] - ((g.mid[1] - r.top) / r.height) * (v[1][1] - v[1][0]);
      const zx: [number, number] = [ax + (v[0][0] - ax) * factor, ax + (v[0][1] - ax) * factor];
      const zy: [number, number] = [ay + (v[1][0] - ay) * factor, ay + (v[1][1] - ay) * factor];
      const mx = (pts[0].x + pts[1].x) / 2 - g.mid[0],
        my = (pts[0].y + pts[1].y) / 2 - g.mid[1];
      const sx = (zx[1] - zx[0]) / r.width,
        sy = (zy[1] - zy[0]) / r.height;
      if (zx[1] - zx[0] > 1e-9 && zx[1] - zx[0] < 1e7)
        setUserView([
          [zx[0] - mx * sx, zx[1] - mx * sx],
          [zy[0] + my * sy, zy[1] + my * sy],
        ]);
      g.moved = true;
      setPanning(true);
      return;
    }
    const dx = e.clientX - g.startX,
      dy = e.clientY - g.startY;
    if (!g.moved && Math.hypot(dx, dy) < 4) return;
    g.moved = true;
    // One finger on a touch screen never pans (the page scrolls instead; see touch-action).
    if (e.pointerType === 'touch') return;
    setPanning(true);
    const r = wrap.current!.getBoundingClientRect();
    const v = g.view;
    const sx = (v[0][1] - v[0][0]) / r.width,
      sy = (v[1][1] - v[1][0]) / r.height;
    setUserView([
      [v[0][0] - dx * sx, v[0][1] - dx * sx],
      [v[1][0] + dy * sy, v[1][1] + dy * sy],
    ]);
  };

  const onPointerUp = (e: RPointerEvent<HTMLDivElement>) => {
    const g = gesture.current;
    pointers.current.delete(e.pointerId);
    if (pointers.current.size > 0) return;
    gesture.current = null;
    setPanning(false);
    if (g && !g.moved && onPick) {
      const p = toData(e.clientX, e.clientY);
      if (p) onPick([Number(p[0].toPrecision(4)), Number(p[1].toPrecision(4))]);
    }
  };

  // The browser took the gesture (e.g. a one-finger page scroll): end it without picking.
  const onPointerCancel = (e: RPointerEvent<HTMLDivElement>) => {
    pointers.current.delete(e.pointerId);
    if (pointers.current.size > 0) return;
    gesture.current = null;
    setPanning(false);
  };

  const hoverF = hover ? f(hover[0], hover[1]) : null;
  const zoomed = userView !== null;

  // ── Keyboard exploration ──────────────────────────────────────────────────────────────
  const keyboardFocus = useRef(false);
  const onFocus = (e: RFocusEvent<HTMLDivElement>) => {
    if (!interactive || !e.currentTarget.matches(':focus-visible')) return;
    keyboardFocus.current = true;
    if (!kb) setKb([(view[0][0] + view[0][1]) / 2, (view[1][0] + view[1][1]) / 2]);
  };
  const onBlur = () => {
    keyboardFocus.current = false;
    setKb(null);
  };
  const onKeyDown = (e: RKeyboardEvent<HTMLDivElement>) => {
    if (!interactive || e.metaKey || e.ctrlKey || e.altKey) return;
    const cur: [number, number] = kb ?? [
      (view[0][0] + view[0][1]) / 2,
      (view[1][0] + view[1][1]) / 2,
    ];
    const wx = view[0][1] - view[0][0],
      wy = view[1][1] - view[1][0];
    const stepFrac = e.shiftKey ? 0.2 : 0.04;
    let handled = true;
    let next: [number, number] | null = null;
    switch (e.key) {
      case 'ArrowLeft':
        next = [cur[0] - wx * stepFrac, cur[1]];
        break;
      case 'ArrowRight':
        next = [cur[0] + wx * stepFrac, cur[1]];
        break;
      case 'ArrowUp':
        next = [cur[0], cur[1] + wy * stepFrac];
        break;
      case 'ArrowDown':
        next = [cur[0], cur[1] - wy * stepFrac];
        break;
      case '+':
      case '=':
        zoomAbout(cur[0], cur[1], 0.7);
        break;
      case '-':
      case '_':
        zoomAbout(cur[0], cur[1], 1 / 0.7);
        break;
      case '0':
        setUserView(null);
        break;
      case 'Enter':
      case ' ':
        if (onPick) onPick([Number(cur[0].toPrecision(4)), Number(cur[1].toPrecision(4))]);
        else handled = false;
        break;
      default:
        handled = false;
    }
    if (!handled) return;
    e.preventDefault();
    e.stopPropagation(); // keep the global playback shortcuts out of it
    if (next) {
      setKb(next);
      // Pan so the crosshair stays inside the view (with a 5% margin).
      const mx = wx * 0.05,
        my = wy * 0.05;
      const dx =
        next[0] < view[0][0] + mx
          ? next[0] - (view[0][0] + mx)
          : next[0] > view[0][1] - mx
            ? next[0] - (view[0][1] - mx)
            : 0;
      const dy =
        next[1] < view[1][0] + my
          ? next[1] - (view[1][0] + my)
          : next[1] > view[1][1] - my
            ? next[1] - (view[1][1] - my)
            : 0;
      if (dx || dy)
        setUserView([
          [view[0][0] + dx, view[0][1] + dx],
          [view[1][0] + dy, view[1][1] + dy],
        ]);
    }
  };
  const kbF = kb ? f(kb[0], kb[1]) : null;
  const kbText =
    kb && kbF !== null
      ? `${axisLabels?.[0] ?? 'x'} ${sig(kb[0], 4)}, ${axisLabels?.[1] ?? 'y'} ${sig(kb[1], 4)}, ${valueLabel} ${Number.isFinite(kbF) ? fmtF(kbF) : 'undefined'}`
      : '';

  return (
    <div
      ref={wrap}
      className={`${styles.stack} ${className ?? ''}`}
      data-panning={panning || undefined}
      data-pickable={onPick ? true : undefined}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerCancel}
      onPointerLeave={() => setHoverPx(null)}
      role="group"
      aria-label="Contour plot"
    >
      <div
        className={styles.layer}
        role="img"
        aria-label={ariaLabel}
        aria-describedby={interactive ? descId : undefined}
        tabIndex={interactive ? 0 : undefined}
        data-plot-focus={interactive || undefined}
        onFocus={onFocus}
        onBlur={onBlur}
        onKeyDown={onKeyDown}
      >
        <canvas ref={fieldRef} className={styles.layer} />
        <canvas ref={overlayRef} className={styles.layer} />
      </div>
      {interactive && (
        <>
          <span id={descId} className="visually-hidden">
            Arrow keys move a crosshair, Shift for larger steps. Plus and minus zoom, 0 resets the
            view.{onPick ? ' Enter sets the start point at the crosshair.' : ''}
          </span>
          <span className="visually-hidden" aria-live="polite">
            {kbText}
          </span>
        </>
      )}
      {showReadout && hover && hoverF !== null && (
        <div
          className={styles.readout}
          aria-hidden="true"
          data-beside-toolbar={(chrome && interactive) || undefined}
        >
          <span>
            <ReadoutName name={axisLabels?.[0] ?? 'x'} /> <SciText text={sig(hover[0], 4)} />
          </span>
          <span>
            <ReadoutName name={axisLabels?.[1] ?? 'y'} /> <SciText text={sig(hover[1], 4)} />
          </span>
          <span>
            <ReadoutName name={valueLabel} /> <SciText text={fmtF(hoverF)} />
          </span>
        </div>
      )}
      {chrome && interactive && (
        <ViewToolbar
          onZoomIn={() => zoomAtCenter(0.7)}
          onZoomOut={() => zoomAtCenter(1 / 0.7)}
          onReset={() => setUserView(null)}
          canReset={zoomed}
        />
      )}
    </div>
  );

  function zoomAtCenter(factor: number) {
    const r = wrap.current?.getBoundingClientRect();
    if (r) zoomAt(r.left + r.width / 2, r.top + r.height / 2, factor);
  }
}

/** A readout name: one letter in math italic, anything longer (‖F‖, φ) as written. */
function ReadoutName({ name }: { name: string }) {
  return /^[A-Za-z]$/.test(name) ? <i>{name}</i> : <span>{name}</span>;
}
