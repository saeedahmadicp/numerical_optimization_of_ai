/**
 * Renders step geometry (geometry.ts) over the contour field. Each method is drawn into a scratch
 * layer and composited with one opacity, so a method that is not focused steps back as a whole
 * (marks, halos and labels alike) and a new step fades in over the first moments of its segment.
 */
import type { View2D } from '../../viz';
import { drawOverlays2D } from '../../viz/overlays2d';
import type { Curve, Geometry } from './geometry';
import { thinRingLabels } from './ringLabels';

export interface GeometryLayer {
  geometry: Geometry;
  /** 0–1: focus weight × fade-in. */
  opacity: number;
}

let scratch: HTMLCanvasElement | null = null;

function layerCanvas(w: number, h: number): HTMLCanvasElement | null {
  if (typeof document === 'undefined') return null;
  scratch ??= document.createElement('canvas');
  if (scratch.width !== w || scratch.height !== h) {
    scratch.width = w;
    scratch.height = h;
  }
  return scratch;
}

function strokeCurve(ctx: CanvasRenderingContext2D, c: Curve, view: View2D) {
  if (c.segs.length < 4) return;
  ctx.beginPath();
  for (let k = 0; k < c.segs.length; k += 4) {
    const a = view.toPx(c.segs[k], c.segs[k + 1]);
    const b = view.toPx(c.segs[k + 2], c.segs[k + 3]);
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
  }
  ctx.lineCap = 'round';
  ctx.strokeStyle = view.colors.halo;
  ctx.lineWidth = (c.width ?? 1.4) + 2;
  ctx.globalAlpha = 0.6 * (c.alpha ?? 1);
  ctx.setLineDash([]);
  ctx.stroke();
  ctx.strokeStyle = c.slot === undefined ? view.colors.text : view.colors.series[c.slot];
  ctx.lineWidth = c.width ?? 1.4;
  ctx.globalAlpha = c.alpha ?? 1;
  ctx.setLineDash(c.dashed ? [5, 4] : []);
  ctx.stroke();
}

function drawGeometry(ctx: CanvasRenderingContext2D, view: View2D, g: Geometry) {
  for (const c of g.curves) {
    ctx.save();
    strokeCurve(ctx, c, view);
    ctx.restore();
  }
  drawOverlays2D(
    ctx,
    {
      toPx: view.toPx,
      toData: (px, py) => [view.x.invert(px), view.y.invert(py)],
      width: view.width,
      height: view.height,
      dpr: view.dpr,
      colors: view.colors,
    },
    thinRingLabels(ctx, g.overlays, view.toPx, view.colors),
  );
}

/** Draw the layers in order (the focused method last, so it sits on top). */
export function drawGeometryLayers(
  ctx: CanvasRenderingContext2D,
  view: View2D,
  layers: readonly GeometryLayer[],
): void {
  for (const l of layers) {
    if (l.opacity <= 0.01 || (l.geometry.curves.length === 0 && l.geometry.overlays.length === 0))
      continue;
    if (l.opacity >= 0.99) {
      drawGeometry(ctx, view, l.geometry);
      continue;
    }
    const w = Math.round(view.width * view.dpr);
    const h = Math.round(view.height * view.dpr);
    const canvas = layerCanvas(w, h);
    const lc = canvas?.getContext('2d');
    if (!canvas || !lc) {
      drawGeometry(ctx, view, l.geometry);
      continue;
    }
    lc.setTransform(1, 0, 0, 1, 0, 0);
    lc.clearRect(0, 0, w, h);
    lc.setTransform(view.dpr, 0, 0, view.dpr, 0, 0);
    drawGeometry(lc, view, l.geometry);
    ctx.save();
    ctx.globalAlpha = l.opacity;
    ctx.drawImage(canvas, 0, 0, view.width, view.height);
    ctx.restore();
  }
}
