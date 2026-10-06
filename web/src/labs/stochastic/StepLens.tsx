/**
 * The step lens: the focused method's current step, magnified so it fills a small square
 * whatever its length. Near the minimizer a step is a few pixels on the landscape; here its
 * geometry is legible: the level sets of f through 𝐰ₖ₋₁ and 𝐰ₖ, the full-gradient step −η∇f
 * (or, for the adaptive methods, −∇f at the step's length), the 2σ ellipse of the mini-batch
 * step, and the step as the head-to-tail sum of its per-sample parts. Equal scale on both axes;
 * a scale bar gives the length.
 *
 * Cost: the level sets are rebuilt on every step during playback, so they come from
 * lensField.ts (an exact O(1) quadratic for the squared loss, a flat loop on a coarser grid
 * otherwise), never from the parity-exact `problem.f`.
 *
 * Labels are placed by lensLabels.ts (no collisions, clear of the corner texts); below
 * `LABEL_MIN` px the lens draws no labels and the stage caption names the marks.
 */
import { useMemo } from 'react';
import { useChartColors } from '../../ui/theme';
import { useCanvas } from '../../viz/useCanvas';
import { drawOverlays2D, niceStep, drawMath, measureMath, mathMain, mathVar } from '../../viz';
import { tick as fmtTick } from '../../core/format';
import { tickFont } from '../../viz/axes';
import type { FiniteSumProblem } from '../../problems/stochastic';
import type { StepGeometry } from './geometry';
import { lensAria, lensBox } from './lensModel';
import { stepLabels, stepOverlays } from './overlays';
import { levelSets, lossEvaluator } from './lensField';
import { arrowCandidates, placeLabels, pointCandidates, type LabelRequest } from './lensLabels';
import styles from './StochasticLab.module.css';

/** Narrower lenses (phones) draw no labels. */
const LABEL_MIN = 150;
const LABEL_SIZE = 11.5;

export interface StepLensProps {
  problem: FiniteSumProblem;
  geometry: StepGeometry;
  slot: number;
  k: number;
}

export function StepLens({ problem, geometry, slot, k }: StepLensProps) {
  const colors = useChartColors();
  const box = useMemo(() => lensBox(geometry), [geometry]);
  const minLength = box.half * 0.04;
  const overlays = useMemo(
    () => stepOverlays(geometry, slot, { labels: false, chain: true, minLength }),
    [geometry, slot, minLength],
  );
  const labels = useMemo(() => stepLabels(geometry, minLength), [geometry, minLength]);
  // Level sets of f through 𝐰ₖ₋₁ and 𝐰ₖ, from one grid over the lens.
  const levels = useMemo(() => {
    const ev = lossEvaluator(problem);
    return levelSets(
      ev.f,
      box,
      [ev.f(geometry.base[0], geometry.base[1]), ev.f(geometry.to[0], geometry.to[1])],
      ev.quadratic ? 49 : 33,
    );
  }, [problem, geometry, box]);

  const { canvasRef } = useCanvas((ctx, s) => {
    const W = s.width,
      H = s.height;
    const scale = Math.min(W, H) / (2 * box.half);
    const toPx = (x: number, y: number): [number, number] => [
      W / 2 + (x - box.cx) * scale,
      H / 2 - (y - box.cy) * scale,
    ];
    const toData = (px: number, py: number): [number, number] => [
      box.cx + (px - W / 2) / scale,
      box.cy - (py - H / 2) / scale,
    ];
    ctx.fillStyle = colors.surface;
    ctx.fillRect(0, 0, W, H);
    // Level sets: f(𝐰ₖ₋₁) solid hairline, f(𝐰ₖ) dotted.
    levels.forEach((seg, li) => {
      ctx.save();
      ctx.strokeStyle = colors.isoStrong || colors.text3;
      ctx.globalAlpha = li === 0 ? 0.9 : 0.7;
      ctx.lineWidth = 1;
      ctx.setLineDash(li === 0 ? [] : [2, 3]);
      ctx.beginPath();
      for (let i = 0; i < seg.length; i += 4) {
        const [ax, ay] = toPx(seg[i], seg[i + 1]);
        const [bx, by] = toPx(seg[i + 2], seg[i + 3]);
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
      }
      ctx.stroke();
      ctx.restore();
    });
    drawOverlays2D(ctx, { toPx, toData, width: W, height: H, dpr: s.dpr, colors }, overlays);
    // The new iterate, as on the landscape: a filled head in the method color.
    const [hx, hy] = toPx(geometry.to[0], geometry.to[1]);
    ctx.fillStyle = colors.halo;
    ctx.beginPath();
    ctx.arc(hx, hy, 5.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = colors.series[slot % colors.series.length];
    ctx.beginPath();
    ctx.arc(hx, hy, 3.8, 0, Math.PI * 2);
    ctx.fill();
    // Scale bar (bottom left): a 1-2-5 length near a quarter of the lens.
    const len = niceStep(0, (W / scale) * 0.25, 1);
    const px = len * scale;
    const y = H - 12;
    ctx.strokeStyle = colors.text2;
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    ctx.moveTo(10, y - 3);
    ctx.lineTo(10, y);
    ctx.lineTo(10 + px, y);
    ctx.lineTo(10 + px, y - 3);
    ctx.stroke();
    ctx.font = tickFont(colors);
    ctx.fillStyle = colors.text2;
    ctx.textBaseline = 'bottom';
    const scaleText = fmtTick(len, len);
    ctx.fillText(scaleText, 14 + px, y + 2);
    const scaleW = 14 + px + ctx.measureText(scaleText).width;
    const kRuns = [mathMain('update '), mathVar('k'), mathMain(` = ${k}`)];
    const kW = drawMath(ctx, kRuns, 10, 16, {
      size: 11.5,
      color: colors.text2,
      halo: colors.halo,
    });

    if (W < LABEL_MIN) return;
    // Labels: laid out clear of each other, of the shafts and of the two corner texts.
    const reqs: LabelRequest[] = labels.map((l) => {
      const w = measureMath(ctx, l.runs, LABEL_SIZE);
      if (l.kind === 'arrow') {
        const a = toPx(l.from[0], l.from[1]),
          b = toPx(l.to[0], l.to[1]);
        return { w, h: 13, candidates: arrowCandidates(a, b) };
      }
      return { w, h: 13, candidates: pointCandidates(toPx(l.at[0], l.at[1])) };
    });
    const segments = labels.flatMap((l) =>
      l.kind === 'arrow'
        ? [[...toPx(l.from[0], l.from[1]), ...toPx(l.to[0], l.to[1])] as const]
        : [],
    );
    const spots = placeLabels(reqs, {
      width: W,
      height: H,
      reserved: [
        { x0: 0, y0: 0, x1: 14 + kW, y1: 24 },
        { x0: 0, y0: H - 26, x1: scaleW + 6, y1: H },
        // The dots: 𝐰ₖ₋₁, 𝐰ₖ, the expected position and the look-ahead.
        ...[geometry.base, geometry.to, geometry.mean, geometry.evalAt].flatMap((p) => {
          if (!p) return [];
          const [px, py] = toPx(p[0], p[1]);
          return [{ x0: px - 5, y0: py - 5, x1: px + 5, y1: py + 5 }];
        }),
      ],
      segments,
    });
    spots.forEach((c, i) => {
      if (!c) return;
      drawMath(ctx, labels[i].runs, c.x, c.y, {
        size: LABEL_SIZE,
        align: c.align,
        baseline: 'middle',
        color: colors.text2,
        halo: colors.halo,
      });
    });
  });

  return (
    <div className={styles.lens}>
      <canvas
        ref={canvasRef}
        className={styles.lensCanvas}
        role="img"
        aria-label={lensAria(geometry, k)}
      />
    </div>
  );
}
