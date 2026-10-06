/**
 * The step lens: the focused method's current step, magnified to fill a small square in the
 * corner of the landscape that covers the least of the drawing (lensPlace.ts), whenever the step
 * is too small to read there. It draws the level sets
 * of f through 𝐱ₖ₋₁ and 𝐱ₖ (so tangency and overshoot are visible), the step's geometry from
 * geometry.ts at the lens's scale, the step itself, and a scale bar. Equal scale on both axes.
 */
import { useMemo, type CSSProperties } from 'react';
import type { Problem2D, Step } from '../../core/types';
import { useChartColors } from '../../ui/theme';
import { useCanvas } from '../../viz/useCanvas';
import { drawOverlays2D, implicitSegments } from '../../viz/overlays2d';
import { drawMath, m as mathMain, v as mathVar } from '../../viz/mathText';
import { easeInOut, lerp } from '../../play/timeline';
import { sig } from '../../core/format';
import type { Geometry } from './geometry';
import { thinRingLabels } from './ringLabels';
import { niceLength, type LensBox } from './lens';
import styles from './UnconstrainedLab.module.css';

export interface StepLensProps {
  problem: Problem2D;
  geometry: Geometry;
  box: LensBox;
  trace: readonly Step[];
  /** The step shown (𝐱ₖ₋₁ → 𝐱ₖ) and how far the head has travelled along it. */
  k: number;
  u: number;
  ease: boolean;
  slot: number;
  name: string;
  /** Position inside the plot (the corner chosen by lensPlace.ts). */
  style?: CSSProperties;
}

const GRID = 64;
const LEVELS_CACHE = new WeakMap<object, { key: string; segs: number[][] }>();

/** Level sets of f through 𝐱ₖ₋₁ and 𝐱ₖ inside the lens square (cached per step and box). */
function levelSets(problem: Problem2D, step: Step, prev: Step, box: LensBox): number[][] {
  const key = `${problem.id}|${box.cx},${box.cy},${box.half}`;
  const hit = LEVELS_CACHE.get(step);
  if (hit && hit.key === key) return hit.segs;
  const xr: [number, number] = [box.cx - box.half, box.cx + box.half];
  const yr: [number, number] = [box.cy - box.half, box.cy + box.half];
  const levels = [prev.fun, step.fun].filter(
    (v): v is number => typeof v === 'number' && Number.isFinite(v),
  );
  const segs = levels.map((c) =>
    implicitSegments((x, y) => problem.f([x, y]) - c, xr, yr, GRID, GRID, 0),
  );
  LEVELS_CACHE.set(step, { key, segs });
  return segs;
}

export function StepLens({
  problem,
  geometry,
  box,
  trace,
  k,
  u,
  ease,
  slot,
  name,
  style,
}: StepLensProps) {
  const colors = useChartColors();
  const prev = trace[k - 1];
  const cur = trace[k];
  const segs = useMemo(() => levelSets(problem, cur, prev, box), [problem, cur, prev, box]);

  const { canvasRef } = useCanvas((ctx, { width, height, dpr }) => {
    const size = Math.min(width, height);
    const s = size / (2 * box.half);
    const toPx = (x: number, y: number): [number, number] => [
      width / 2 + (x - box.cx) * s,
      height / 2 - (y - box.cy) * s,
    ];
    const toData = (px: number, py: number): [number, number] => [
      box.cx + (px - width / 2) / s,
      box.cy - (py - height / 2) / s,
    ];
    ctx.fillStyle = colors.surface;
    ctx.fillRect(0, 0, width, height);
    // Level sets through 𝐱ₖ₋₁ (dashed) and 𝐱ₖ (solid).
    segs.forEach((sg, i) => {
      ctx.save();
      ctx.beginPath();
      for (let j = 0; j < sg.length; j += 4) {
        const a = toPx(sg[j], sg[j + 1]);
        const b = toPx(sg[j + 2], sg[j + 3]);
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
      }
      ctx.strokeStyle = colors.text2;
      ctx.globalAlpha = 0.5;
      ctx.lineWidth = 1.1;
      ctx.setLineDash(i === 0 && segs.length > 1 ? [4, 3] : []);
      ctx.stroke();
      ctx.restore();
    });
    // The step's geometry at the lens's scale.
    for (const c of geometry.curves) {
      ctx.save();
      ctx.beginPath();
      for (let j = 0; j < c.segs.length; j += 4) {
        const a = toPx(c.segs[j], c.segs[j + 1]);
        const b = toPx(c.segs[j + 2], c.segs[j + 3]);
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
      }
      ctx.strokeStyle = c.slot === undefined ? colors.text : colors.series[c.slot];
      ctx.globalAlpha = c.alpha ?? 1;
      ctx.lineWidth = c.width ?? 1.3;
      ctx.stroke();
      ctx.restore();
    }
    drawOverlays2D(
      ctx,
      { toPx, toData, width, height, dpr, colors },
      thinRingLabels(
        ctx,
        geometry.caption ? geometry.overlays.filter((o) => o.kind !== 'text') : geometry.overlays,
        toPx,
        colors,
      ),
    );
    // The step 𝐱ₖ₋₁ → head.
    const a = prev.x as number[];
    const b = cur.x as number[];
    const t = ease ? easeInOut(u) : u;
    const head: [number, number] = [lerp(a[0], b[0], t), lerp(a[1], b[1], t)];
    const pa = toPx(a[0], a[1]);
    const ph = toPx(head[0], head[1]);
    ctx.save();
    ctx.lineCap = 'round';
    ctx.strokeStyle = colors.halo;
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.moveTo(pa[0], pa[1]);
    ctx.lineTo(ph[0], ph[1]);
    ctx.stroke();
    ctx.strokeStyle = colors.series[slot];
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = colors.halo;
    ctx.beginPath();
    ctx.arc(pa[0], pa[1], 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = colors.series[slot];
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.arc(pa[0], pa[1], 3.4, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = colors.halo;
    ctx.beginPath();
    ctx.arc(ph[0], ph[1], 5.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = colors.series[slot];
    ctx.beginPath();
    ctx.arc(ph[0], ph[1], 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    // Scale bar (bottom left) and the step index (top left).
    const len = niceLength((size * 0.32) / s);
    if (len > 0) {
      const px = len * s;
      const y = height - 12;
      ctx.save();
      ctx.strokeStyle = colors.text2;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(10, y - 3);
      ctx.lineTo(10, y);
      ctx.lineTo(10 + px, y);
      ctx.lineTo(10 + px, y - 3);
      ctx.stroke();
      ctx.restore();
      drawMath(ctx, [mathMain(sig(len, 1).replace(/^-/, '−'))], 14 + px, y + 1, {
        size: 11,
        color: colors.text2,
        halo: colors.surface,
      });
    }
    drawMath(
      ctx,
      [
        mathMain('step '),
        mathVar('k'),
        mathMain(` = ${k}${geometry.caption ? ` · ${geometry.caption}` : ''}`),
      ],
      10,
      16,
      {
        size: 11.5,
        color: colors.text2,
        halo: colors.surface,
      },
    );
  });

  return (
    <div
      className={styles.lens}
      style={style}
      role="img"
      aria-label={`${name}, step ${k} magnified, with the level sets of f through the old and the new iterate.`}
    >
      <canvas ref={canvasRef} className={styles.lensCanvas} />
    </div>
  );
}
