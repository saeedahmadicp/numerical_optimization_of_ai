/**
 * The strip under the stage, on the same x-axis as the plot above:
 *
 *  - error:  |p(x) − f(x)| of each method's current step on a log axis; the ticks mark the
 *            focused method's interpolation nodes, where its error vanishes;
 *  - omega:  the nodal polynomial ω(x) = ∏(x − xᵢ) of these nodes against the equispaced and
 *            Chebyshev node sets of the same size — f − p = f⁽ⁿ⁾(ξ)/n! · ω(x) (B&F, Thm 3.3);
 *  - basis:  the Lagrange basis ℓⱼ (the current one in the method's color) and the Lebesgue
 *            function λ = Σ|ℓⱼ|, whose maximum Λₙ bounds how much p can amplify data errors;
 *  - poles:  the complex plane under the same x-axis (Re z): AAA's support points on the real
 *            axis and the poles of r_k (× a pole, ○ a Froissart doublet), the certified real
 *            poles in [a, b] ringed.
 */
import { memo } from 'react';
import { sci, sig } from '../../core/format';
import { useChartColors } from '../../ui/theme';
import type { ChartColors } from '../../ui/colors';
import {
  drawAxes,
  drawMath,
  linearScale,
  logScale,
  mathMain as mm,
  mathSub as msub,
  mathVar as mv,
  useCanvas,
  type MathRun,
} from '../../viz';
import {
  aaaIntervalPoles,
  aaaPoles,
  aaaSupport,
  basisFn,
  chebyshevNodes,
  equispacedNodes,
  lebesgueFn,
  nodalPoly,
  PLOT_MARGIN,
  sampled,
  stepFunction,
  type ActiveData,
} from './geometry';
import type { RunView } from './InterpPlot';
import styles from './InterpolationLab.module.css';

export type StripMode = 'error' | 'omega' | 'basis' | 'poles';

export interface StripPlotProps {
  mode: StripMode;
  data: ActiveData;
  runs: readonly RunView[];
  /** Index of the basis polynomial to highlight (the node the focused method added). */
  basisIndex: number | null;
  ariaLabel: string;
}

const M = { ...PLOT_MARGIN, top: 22, bottom: 22 };
const asNums = (v: unknown): number[] => (Array.isArray(v) ? (v as number[]) : []);
const FLOOR = 1e-16;

function polyline(
  ctx: CanvasRenderingContext2D,
  ys: ArrayLike<number>,
  a: number,
  b: number,
  P: (x: number, y: number) => [number, number],
) {
  ctx.beginPath();
  let pen = false;
  for (let i = 0; i < ys.length; i++) {
    const [px, py0] = P(a + ((b - a) * i) / (ys.length - 1), ys[i]);
    if (!Number.isFinite(py0)) {
      pen = false;
      continue;
    }
    const py = Math.max(-4000, Math.min(4000, py0));
    if (pen) ctx.lineTo(px, py);
    else ctx.moveTo(px, py);
    pen = true;
  }
  ctx.stroke();
}

function stroke(
  ctx: CanvasRenderingContext2D,
  colors: ChartColors,
  draw: () => void,
  color: string,
  width: number,
  dash: number[] = [],
  alpha = 1,
) {
  ctx.save();
  ctx.lineJoin = 'round';
  ctx.strokeStyle = colors.halo;
  ctx.lineWidth = width + 2;
  ctx.globalAlpha = 0.7 * alpha;
  draw();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash);
  draw();
  ctx.restore();
}

/** A legend entry drawn on the canvas: a short line sample, then math/word runs. */
function legend(
  ctx: CanvasRenderingContext2D,
  colors: ChartColors,
  x: number,
  y: number,
  color: string,
  dash: number[],
  runs: MathRun[],
): number {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6;
  ctx.setLineDash(dash);
  ctx.beginPath();
  ctx.moveTo(x, y - 4);
  ctx.lineTo(x + 16, y - 4);
  ctx.stroke();
  ctx.restore();
  return 22 + drawMath(ctx, runs, x + 22, y, { size: 12, color: colors.text2, halo: colors.halo });
}

const words = (t: string): MathRun => ({ t, style: 'sans' });

function StripPlotImpl({ mode, data, runs, basisIndex, ariaLabel }: StripPlotProps) {
  const colors = useChartColors();
  const [a, b] = data.domain;
  const xPad = 0.03 * (b - a);
  const xd: [number, number] = [a - xPad, b + xPad];

  const { canvasRef } = useCanvas((ctx, s) => {
    const frame = {
      left: M.left,
      top: M.top,
      right: s.width - M.right,
      bottom: s.height - M.bottom,
    };
    const x = linearScale(xd, [frame.left, frame.right]);
    const series = (slot: number) => colors.series[slot % colors.series.length];

    if (mode === 'error') {
      if (!data.fTrue) {
        drawMath(
          ctx,
          [words('No f is known for these data, so there is no error to plot.')],
          frame.left + 8,
          (frame.top + frame.bottom) / 2,
          { size: 13, color: colors.text3 },
        );
        return;
      }
      const f = data.fTrue;
      const truth = sampled(`f|${data.baseId}`, f, a, b);
      const curves = runs
        .map((run) => {
          const k = Math.min(run.k, run.result.trace.length - 1);
          const g = k >= 0 ? stepFunction(run.id, run.result, k, data) : null;
          if (!g) return null;
          const ys = sampled(`${run.key}|${k}`, g, a, b);
          return { run, err: Array.from(ys, (v, i) => Math.max(FLOOR, Math.abs(v - truth[i]))) };
        })
        .filter((c): c is { run: RunView; err: number[] } => c !== null);
      // A fixed decade range per dataset: the final errors set the top, 10⁻¹⁶ the floor.
      let hi = 1e-2;
      for (const run of runs) {
        const last = run.result.trace.length - 1;
        const g = last >= 0 ? stepFunction(run.id, run.result, last, data) : null;
        if (!g) continue;
        const ys = sampled(`${run.key}|${last}`, g, a, b);
        for (let i = 0; i < ys.length; i++) hi = Math.max(hi, Math.abs(ys[i] - truth[i]));
      }
      // Fixed while the playhead moves (partial sums that leave it are clipped); the floor sits
      // ten decades under the top so the error levels, not the zeros at the nodes, get the height.
      const yTop = 10 ** Math.ceil(Math.log10(Math.min(hi, 1e8)));
      const floor = Math.max(FLOOR, yTop * 1e-10);
      const y = logScale([floor, yTop], [frame.bottom, frame.top]);
      drawAxes(ctx, {
        x,
        y,
        frame,
        colors,
        dpr: s.dpr,
        yName: [mm('|'), mv('p'), mm(' − '), mv('f'), mm('|')],
        yTicks: 3,
      });
      const P = (u: number, v: number): [number, number] => [x(u), y(Math.max(floor, v))];
      ctx.save();
      ctx.beginPath();
      ctx.rect(frame.left, frame.top, frame.right - frame.left, frame.bottom - frame.top);
      ctx.clip();
      for (const c of [...curves].sort((p, q) => Number(p.run.focus) - Number(q.run.focus)))
        stroke(
          ctx,
          colors,
          () => polyline(ctx, c.err, a, b, P),
          series(c.run.slot),
          c.run.focus ? 1.8 : 1.3,
        );
      ctx.restore();
      if (curves.length === 0)
        drawMath(
          ctx,
          [words('No approximant yet: the splines have no pieces before their last stage.')],
          (frame.left + frame.right) / 2,
          (frame.top + frame.bottom) / 2,
          { size: 12.5, color: colors.text3, align: 'center' },
        );
      // Ticks where the focused method's error vanishes by construction: its interpolation
      // nodes. Chebyshev interpolation that resamples f uses its own nodes (the roots of T_N);
      // a node-adding method at step k interpolates only x_0 … x_k (later nodes: faint).
      const focus = runs.find((r) => r.focus) ?? runs[0];
      const ticks: { x: number; on: boolean }[] = [];
      if (focus?.kind === 'chebyshev' && focus.result.extra.source === 'f_true') {
        for (const xi of asNums(focus.result.extra.nodes)) ticks.push({ x: xi, on: true });
      } else if (focus && (focus.kind === 'nodewise' || focus.kind === 'neville')) {
        data.x.forEach((xi, i) => ticks.push({ x: xi, on: i <= focus.k }));
      } else if (focus?.kind === 'aaa') {
        // AAA interpolates its support points only; the other samples are fitted, not hit.
        const used = new Set(aaaSupport(focus.result, focus.k));
        data.x.forEach((xi, i) => ticks.push({ x: xi, on: used.has(i) }));
      } else {
        for (const xi of data.x) ticks.push({ x: xi, on: true });
      }
      ctx.save();
      ctx.lineWidth = 1.5;
      for (const t of ticks) {
        ctx.strokeStyle = t.on && focus ? series(focus.slot) : colors.text3;
        ctx.globalAlpha = t.on ? 0.9 : 0.4;
        ctx.beginPath();
        // A rug just under the frame: the curves dive to the floor at these points themselves.
        ctx.moveTo(x(t.x), frame.bottom + 1);
        ctx.lineTo(x(t.x), frame.bottom + (t.on ? 5 : 3.5));
        ctx.stroke();
      }
      ctx.restore();
      return;
    }

    if (mode === 'poles') {
      drawPoles(ctx, colors, s.dpr, frame, x, xd, data, runs);
      return;
    }

    if (mode === 'omega') {
      const n = data.x.length;
      const sets: {
        label: MathRun[];
        nodes: number[];
        color: string;
        dash: number[];
        width: number;
      }[] = [
        { label: [words('these nodes')], nodes: data.x, color: colors.text, dash: [], width: 1.8 },
      ];
      if (data.layout !== 'equi')
        sets.push({
          label: [words('equispaced')],
          nodes: equispacedNodes(n, a, b),
          color: colors.text3,
          dash: [5, 4],
          width: 1.3,
        });
      if (data.layout !== 'cheb')
        sets.push({
          label: [words('Chebyshev')],
          nodes: chebyshevNodes(n, a, b),
          color: colors.text3,
          dash: [1.5, 3],
          width: 1.5,
        });
      const curves = sets.map((set) => {
        const ys = sampled(`omega|${set.nodes.join(',')}`, nodalPoly(set.nodes), a, b, 801);
        let m = 0;
        for (const v of ys) if (Number.isFinite(v)) m = Math.max(m, Math.abs(v));
        return { ...set, ys, max: m };
      });
      const top = Math.max(...curves.map((c) => c.max)) * 1.12 || 1;
      const y = linearScale([-top, top], [frame.bottom, frame.top]);
      drawAxes(ctx, {
        x,
        y,
        frame,
        colors,
        dpr: s.dpr,
        yName: [mv('ω'), mm('('), mv('x'), mm(')')],
        yTicks: 3,
      });
      const P = (u: number, v: number): [number, number] => [x(u), y(v)];
      ctx.save();
      ctx.beginPath();
      ctx.rect(frame.left, frame.top, frame.right - frame.left, frame.bottom - frame.top);
      ctx.clip();
      for (const c of [...curves].reverse())
        stroke(ctx, colors, () => polyline(ctx, c.ys, a, b, P), c.color, c.width, c.dash);
      for (const xi of data.x) {
        ctx.beginPath();
        ctx.arc(x(xi), y(0), 2.4, 0, Math.PI * 2);
        ctx.fillStyle = colors.text;
        ctx.fill();
      }
      ctx.restore();
      // Legend with max|ω| for each node set.
      let lx = frame.left + 8;
      const ly = frame.bottom - 6;
      for (const c of curves) {
        lx +=
          legend(ctx, colors, lx, ly, c.color, c.dash, [
            ...c.label,
            mm('  max|'),
            mv('ω'),
            mm('| = ' + sci(c.max, 2)),
          ]) + 14;
      }
      return;
    }

    // basis
    const n = data.x.length;
    const lam = lebesgueFn(data.x);
    const lamYs = sampled(`lebesgue|${data.id}`, lam, a, b, 801);
    let L = 0,
      Lx = a;
    lamYs.forEach((v, i) => {
      if (v > L) {
        L = v;
        Lx = a + ((b - a) * i) / (lamYs.length - 1);
      }
    });
    const yHi = Math.min(Math.max(1.35, L * 1.08), 3.2);
    const y = linearScale([-0.9, yHi], [frame.bottom, frame.top]);
    drawAxes(ctx, {
      x,
      y,
      frame,
      colors,
      dpr: s.dpr,
      yName: [mv('ℓ'), msub('j', 'italic'), mm('('), mv('x'), mm(')')],
      yTicks: 3,
    });
    const P = (u: number, v: number): [number, number] => [x(u), y(v)];
    ctx.save();
    ctx.beginPath();
    ctx.rect(frame.left, frame.top, frame.right - frame.left, frame.bottom - frame.top);
    ctx.clip();
    const hl = basisIndex !== null && basisIndex < n ? basisIndex : null;
    const focus = runs.find((r) => r.focus);
    for (let j = 0; j < n; j++) {
      if (j === hl) continue;
      const ys = sampled(`basis|${data.id}|${j}`, basisFn(data.x, j), a, b, 401);
      stroke(ctx, colors, () => polyline(ctx, ys, a, b, P), colors.text3, 1, [], 0.45);
    }
    stroke(ctx, colors, () => polyline(ctx, lamYs, a, b, P), colors.text2, 1.3, [5, 3]);
    if (hl !== null) {
      const ys = sampled(`basis|${data.id}|${hl}`, basisFn(data.x, hl), a, b, 401);
      const c = focus ? series(focus.slot) : colors.text;
      stroke(ctx, colors, () => polyline(ctx, ys, a, b, P), c, 2);
      // Cardinal property: ℓⱼ(xⱼ) = 1, ℓⱼ(xₘ) = 0.
      data.x.forEach((xi, m) => {
        ctx.beginPath();
        ctx.arc(x(xi), y(m === hl ? 1 : 0), m === hl ? 3.6 : 2.2, 0, Math.PI * 2);
        ctx.fillStyle = m === hl ? c : colors.text2;
        ctx.fill();
      });
    }
    ctx.restore();
    // Λₙ = max λ, labelled where it is attained (above the view when it does not fit).
    const off = L * 1.0 > yHi;
    const px = x(Lx);
    const label = [mv('Λ'), msub(String(n), 'main'), mm(` = max λ = ${sig(L, 3)}`)];
    drawMath(
      ctx,
      off ? [mm('↑ '), ...label] : label,
      Math.min(px, frame.right - 150),
      off ? frame.top + 12 : y(L) - 6,
      {
        size: 12,
        color: colors.text2,
        halo: colors.halo,
      },
    );
  });

  return (
    <canvas ref={canvasRef} className={styles.stripCanvas} role="img" aria-label={ariaLabel} />
  );
}

/** A pole glyph: × (pole) or ○ (Froissart doublet), with a halo. */
function poleGlyph(
  ctx: CanvasRenderingContext2D,
  colors: ChartColors,
  px: number,
  py: number,
  doublet: boolean,
  color: string,
  r = 4,
) {
  ctx.save();
  ctx.lineCap = 'round';
  for (const [c, w] of [
    [colors.halo, 4],
    [color, 1.8],
  ] as const) {
    ctx.strokeStyle = c;
    ctx.lineWidth = w;
    ctx.beginPath();
    if (doublet) ctx.arc(px, py, r, 0, Math.PI * 2);
    else {
      ctx.moveTo(px - r, py - r);
      ctx.lineTo(px + r, py + r);
      ctx.moveTo(px + r, py - r);
      ctx.lineTo(px - r, py + r);
    }
    ctx.stroke();
  }
  ctx.restore();
}

/**
 * The complex plane under the stage: Re z on the stage's x-axis, Im z on a symmetric axis that
 * is fixed per dataset (it fits the final poles of every AAA run that lie under the view), so
 * the poles move, not the axis, while the playhead moves.
 */
function drawPoles(
  ctx: CanvasRenderingContext2D,
  colors: ChartColors,
  dpr: number,
  frame: { left: number; top: number; right: number; bottom: number },
  x: ReturnType<typeof linearScale>,
  xd: [number, number],
  data: ActiveData,
  runs: readonly RunView[],
) {
  const [a, b] = data.domain;
  const span = b - a;
  const aaaRuns = runs.filter((r) => r.kind === 'aaa' && r.result.trace.length > 0);
  if (aaaRuns.length === 0) {
    const msg = runs.some((r) => r.kind === 'blend')
      ? 'Floater–Hormann has no real poles for any d; add AAA to see the poles of a rational approximant.'
      : 'Polynomials and splines have no poles. Add AAA to see the poles of a rational approximant.';
    drawMath(ctx, [words(msg)], frame.left + 8, (frame.top + frame.bottom) / 2, {
      size: 13,
      color: colors.text3,
    });
    return;
  }
  const fScale = Math.max(...data.y.map(Math.abs), 0);
  const inX = (re: number) => re >= xd[0] && re <= xd[1];
  let H = 0;
  for (const run of aaaRuns)
    for (const p of aaaPoles(run.result, run.result.trace.length - 1, fScale))
      if (inX(p.re)) H = Math.max(H, Math.abs(p.im));
  H = Math.min(Math.max(H * 1.2, 0.15 * span), 0.75 * span);
  const y = linearScale([-H, H], [frame.bottom, frame.top]);
  drawAxes(ctx, { x, y, frame, colors, dpr, yName: [mm('Im '), mv('z')], yTicks: 3 });
  ctx.save();
  ctx.beginPath();
  ctx.rect(frame.left, frame.top, frame.right - frame.left, frame.bottom - frame.top);
  ctx.clip();
  // The real axis and the interval [a, b] on it.
  const y0 = y(0);
  ctx.strokeStyle = colors.text3;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(frame.left, y0);
  ctx.lineTo(frame.right, y0);
  ctx.stroke();
  ctx.strokeStyle = colors.text2;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(x(a), y0);
  ctx.lineTo(x(b), y0);
  ctx.stroke();
  // The samples: short ticks on the axis.
  ctx.lineWidth = 1.2;
  for (const xi of data.x) {
    ctx.beginPath();
    ctx.moveTo(x(xi), y0 - 3.5);
    ctx.lineTo(x(xi), y0 + 3.5);
    ctx.stroke();
  }
  let outside = 0;
  const ordered = [...aaaRuns].sort((p, q) => Number(p.focus) - Number(q.focus));
  for (const run of ordered) {
    const color = colors.series[run.slot % colors.series.length];
    const k = Math.min(run.k, run.result.trace.length - 1);
    // Support points: filled dots on the real axis.
    for (const i of aaaSupport(run.result, k)) {
      ctx.beginPath();
      ctx.arc(x(data.x[i]), y0, run.focus ? 3.4 : 2.8, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    }
    for (const p of aaaPoles(run.result, k, fScale)) {
      if (!inX(p.re) || Math.abs(p.im) > H) {
        if (run.focus || aaaRuns.length === 1) outside++;
        continue;
      }
      poleGlyph(ctx, colors, x(p.re), y(p.im), p.doublet, color, run.focus ? 4.2 : 3.4);
    }
    // Certified real poles in [a, b]: ringed, so a pole on the axis cannot hide among the ticks.
    for (const p of aaaIntervalPoles(run.result, k)) {
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.arc(x(p), y0, 8, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
  }
  ctx.restore();
  // Legend (bottom left) and the count of poles beyond the view (top right).
  const ly = frame.bottom - 7;
  let lx = frame.left + 10;
  const ink = colors.text2;
  poleGlyph(ctx, colors, lx, ly - 4, false, ink, 3.6);
  lx +=
    9 +
    drawMath(ctx, [words('pole')], lx + 9, ly, { size: 12, color: ink, halo: colors.halo }) +
    14;
  poleGlyph(ctx, colors, lx, ly - 4, true, ink, 3.6);
  lx +=
    9 +
    drawMath(ctx, [words('Froissart doublet')], lx + 9, ly, {
      size: 12,
      color: ink,
      halo: colors.halo,
    }) +
    14;
  ctx.beginPath();
  ctx.arc(lx, ly - 4, 3, 0, Math.PI * 2);
  ctx.fillStyle = ink;
  ctx.fill();
  drawMath(ctx, [words('support point')], lx + 9, ly, { size: 12, color: ink, halo: colors.halo });
  if (outside > 0)
    drawMath(
      ctx,
      [words(`${outside} more pole${outside === 1 ? '' : 's'} outside this view`)],
      frame.right - 8,
      frame.top + 14,
      {
        size: 12,
        color: ink,
        halo: colors.halo,
        align: 'right',
      },
    );
}

/** Redraws only when its inputs change (not on every playback frame of the lab). */
export const StripPlot = memo(StripPlotImpl);
