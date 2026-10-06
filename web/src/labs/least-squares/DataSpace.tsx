/**
 * The data-space pane: what the parameters 𝐱ₖ mean for the data.
 *
 *   curve fits   the data (tᵢ, yᵢ), the model curve m(t; 𝐱) of every method morphing with the
 *                playhead, and the residual sticks rᵢ = m(tᵢ; 𝐱) − yᵢ of the focused method;
 *   circle fit   the data points, the circle of radius R about each method's center (a, b), and
 *                the residual sticks along the radii (rᵢ = ‖𝐩ᵢ − 𝐜‖ − R);
 *   Rosenbrock   the residual plane 𝐫 = (r₁, r₂): the image of each path, the target 𝐫 = 𝟎,
 *                and the focused method's linear prediction 𝐫ₖ + Jₖ𝐡ₖ next to where the step
 *                actually lands — the gain ratio as a picture.
 */
import { useMemo, useRef, type PointerEvent as RPointerEvent } from 'react';
import { useChartColors } from '../../ui/theme';
import type { ChartColors } from '../../ui/colors';
import {
  drawAxes,
  drawMath,
  linearScale,
  mathBold as b,
  mathMain as mm,
  mathSub as sub,
  mathVar as mv,
  niceDomain,
  pathPosition,
  useCanvas,
  type Scale,
} from '../../viz';
import { circleFoot, curveRange, type DataSpace } from './models';
import styles from './LeastSquaresLab.module.css';

export interface DataRun {
  /** Iterates 𝐱ₖ (parameters). */
  points: readonly (readonly [number, number])[];
  slot: number;
  name: string;
  focused: boolean;
}

export interface DataSpaceProps {
  space: DataSpace;
  runs: readonly DataRun[];
  t: number;
  ease: boolean;
  /**
   * Focused method's linear prediction in the residual plane: 𝐫ₖ → 𝐫ₖ + Jₖ(step), with the
   * step named as the method names it (Gauss–Newton αₖ𝐩ₖ, Levenberg–Marquardt 𝐡ₖ).
   */
  prediction?: {
    from: readonly [number, number];
    to: readonly [number, number];
    k: number;
    step: 'p' | 'alpha_p' | 'h';
  } | null;
  /** Circle fit: a click sets the start center (the data plane is the parameter plane there). */
  onPick?: (p: [number, number]) => void;
  ariaLabel: string;
}

const M = { left: 42, right: 14, top: 16, bottom: 26 };

interface Frame {
  x: Scale;
  y: Scale;
  left: number;
  right: number;
  top: number;
  bottom: number;
}

function pad([a, c]: [number, number], f: number): [number, number] {
  const s = c - a || Math.abs(a) || 1;
  return [a - s * f, c + s * f];
}

interface Domains {
  x: [number, number];
  y: [number, number];
  equal: boolean;
  /** Curve fits: the t-range where the model is sampled. */
  curve?: [number, number];
}

/** Fixed per run set (the axes never chase the animation). */
function domains(space: DataSpace, runs: readonly DataRun[]): Domains {
  const all = runs.flatMap((r) => r.points);
  if (space.kind === 'curve') {
    const tLo = Math.min(...space.t),
      tHi = Math.max(...space.t);
    const tDom = pad(niceDomain(Math.min(0, tLo), tHi), 0.025);
    // The model is drawn (and sizes the y-axis) only where it means something: t ≥ 0, up to
    // the axis end. The padding below 0 is for the axis alone — Michaelis–Menten has a pole
    // at S = −K, inside that strip when K is small.
    const curve = curveRange(tLo, tDom);
    const yLo = Math.min(...space.y),
      yHi = Math.max(...space.y);
    const span = yHi - yLo || 1;
    let lo = yLo,
      hi = yHi;
    for (const p of all)
      for (let i = 0; i <= 24; i++) {
        const v = space.model(curve[0] + ((curve[1] - curve[0]) * i) / 24, p);
        if (Number.isFinite(v)) {
          lo = Math.min(lo, v);
          hi = Math.max(hi, v);
        }
      }
    // A wild iterate must not squash the data: at most 0.6 spans beyond it.
    lo = Math.max(lo, yLo - 0.6 * span);
    hi = Math.min(hi, yHi + 0.6 * span);
    return { x: tDom, y: pad(niceDomain(Math.min(0, lo), hi), 0.03), equal: false, curve };
  }
  if (space.kind === 'circle') {
    const xs = [...space.px],
      ys = [...space.py];
    for (const p of all) {
      xs.push(p[0] - space.radius, p[0] + space.radius);
      ys.push(p[1] - space.radius, p[1] + space.radius);
    }
    return {
      x: pad([Math.min(...xs), Math.max(...xs)], 0.05),
      y: pad([Math.min(...ys), Math.max(...ys)], 0.05),
      equal: true,
    };
  }
  const rs = all.map((p) => space.residual(p)).filter((r) => r.every(Number.isFinite));
  const r1 = [0, ...rs.map((r) => r[0])],
    r2 = [0, ...rs.map((r) => r[1])];
  return {
    x: niceDomain(...pad([Math.min(...r1), Math.max(...r1)], 0.08)),
    y: niceDomain(...pad([Math.min(...r2), Math.max(...r2)], 0.08)),
    equal: false,
  };
}

function frameFor(d: Domains, w: number, h: number): Frame {
  const left = M.left,
    right = w - M.right,
    top = M.top,
    bottom = h - M.bottom;
  let xd = d.x,
    yd = d.y;
  if (d.equal) {
    const pw = right - left,
      ph = bottom - top;
    const sx = (xd[1] - xd[0]) / pw,
      sy = (yd[1] - yd[0]) / ph;
    const s = Math.max(sx, sy);
    const cx = (xd[0] + xd[1]) / 2,
      cy = (yd[0] + yd[1]) / 2;
    xd = [cx - (s * pw) / 2, cx + (s * pw) / 2];
    yd = [cy - (s * ph) / 2, cy + (s * ph) / 2];
  }
  return {
    x: linearScale(xd, [left, right]),
    y: linearScale(yd, [bottom, top]),
    left,
    right,
    top,
    bottom,
  };
}

function dot(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  r: number,
  fill: string,
  halo: string,
) {
  ctx.fillStyle = halo;
  ctx.beginPath();
  ctx.arc(x, y, r + 2, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = fill;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
}

function ring(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  r: number,
  color: string,
  halo: string,
) {
  ctx.fillStyle = halo;
  ctx.beginPath();
  ctx.arc(x, y, r + 2, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.stroke();
}

/** A polyline with a surface halo underneath. */
function stroke(
  ctx: CanvasRenderingContext2D,
  pts: readonly (readonly [number, number])[],
  color: string,
  c: ChartColors,
  width = 2,
  alpha = 1,
  dash: number[] = [],
) {
  const path = () => {
    ctx.beginPath();
    let pen = false;
    for (const [x, y] of pts) {
      if (!Number.isFinite(x) || !Number.isFinite(y) || Math.abs(y) > 1e6 || Math.abs(x) > 1e6) {
        pen = false;
        continue;
      }
      if (pen) ctx.lineTo(x, y);
      else ctx.moveTo(x, y);
      pen = true;
    }
  };
  ctx.save();
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.globalAlpha = 0.75 * alpha;
  ctx.strokeStyle = c.halo;
  ctx.lineWidth = width + 2.5;
  ctx.setLineDash(dash);
  path();
  ctx.stroke();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  path();
  ctx.stroke();
  ctx.restore();
}

export function DataSpacePane({
  space,
  runs,
  t,
  ease,
  prediction,
  onPick,
  ariaLabel,
}: DataSpaceProps) {
  const colors = useChartColors();
  const dom = useMemo(() => domains(space, runs), [space, runs]);
  const frameRef = useRef<Frame | null>(null);

  const { canvasRef } = useCanvas((ctx, s) => {
    const f = frameFor(dom, s.width, s.height);
    frameRef.current = f;
    const c = colors;
    const names =
      space.kind === 'curve'
        ? { xName: [mv(space.tName)], yName: [mv(space.yName)] }
        : space.kind === 'circle'
          ? { xName: [mv('x')], yName: [mv('y')] }
          : { xName: [mv('r'), sub('1')], yName: [mv('r'), sub('2')] };
    drawAxes(ctx, { x: f.x, y: f.y, frame: f, colors: c, dpr: s.dpr, ...names });
    ctx.save();
    ctx.beginPath();
    ctx.rect(f.left, f.top, f.right - f.left, f.bottom - f.top);
    ctx.clip();
    const P = (x: number, y: number): [number, number] => [f.x(x), f.y(y)];
    const heads = runs.map((r) => pathPosition(r.points, t, ease));
    // Focused method last (on top).
    const order = runs
      .map((_, i) => i)
      .sort((i, j) => Number(runs[i].focused) - Number(runs[j].focused));

    if (space.kind === 'curve') {
      const [t0, t1] = dom.curve ?? f.x.domain;
      for (const i of order) {
        const th = heads[i];
        if (!th) continue;
        const r = runs[i];
        const col = c.series[r.slot % c.series.length];
        if (r.focused) {
          // Residual sticks rᵢ = m(tᵢ; 𝐱) − yᵢ.
          ctx.save();
          ctx.strokeStyle = col;
          ctx.globalAlpha = 0.65;
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          space.t.forEach((ti, k) => {
            const mi = space.model(ti, th);
            if (!Number.isFinite(mi)) return;
            ctx.moveTo(...P(ti, space.y[k]));
            ctx.lineTo(...P(ti, mi));
          });
          ctx.stroke();
          ctx.restore();
        }
        const pts: [number, number][] = [];
        for (let k = 0; k <= 200; k++) {
          const tt = t0 + ((t1 - t0) * k) / 200;
          pts.push(P(tt, space.model(tt, th)));
        }
        stroke(ctx, pts, col, c, r.focused ? 2.2 : 1.6, r.focused ? 1 : 0.7);
      }
      space.t.forEach((ti, k) => {
        const [x, y] = P(ti, space.y[k]);
        dot(ctx, x, y, 3.4, c.text, c.halo);
      });
    } else if (space.kind === 'circle') {
      for (const i of order) {
        const th = heads[i];
        if (!th) continue;
        const r = runs[i];
        const col = c.series[r.slot % c.series.length];
        const center: [number, number] = [th[0], th[1]];
        if (r.focused) {
          ctx.save();
          ctx.strokeStyle = col;
          ctx.globalAlpha = 0.65;
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          space.px.forEach((x, k) => {
            const foot = circleFoot(center, space.radius, [x, space.py[k]]);
            ctx.moveTo(...P(x, space.py[k]));
            ctx.lineTo(...P(foot[0], foot[1]));
          });
          ctx.stroke();
          ctx.restore();
        }
        const pts: [number, number][] = [];
        for (let k = 0; k <= 180; k++) {
          const a = (2 * Math.PI * k) / 180;
          pts.push(
            P(center[0] + space.radius * Math.cos(a), center[1] + space.radius * Math.sin(a)),
          );
        }
        stroke(ctx, pts, col, c, r.focused ? 2.2 : 1.6, r.focused ? 1 : 0.7);
        const [cx, cy] = P(center[0], center[1]);
        dot(ctx, cx, cy, 3, col, c.halo);
      }
      space.px.forEach((x, k) => {
        const [px, py] = P(x, space.py[k]);
        dot(ctx, px, py, 3.4, c.text, c.halo);
      });
    } else {
      const rOf = (p: readonly number[]) => space.residual(p);
      // The target: the zero residual.
      const [ox, oy] = P(0, 0);
      for (const [w, col] of [
        [3.5, c.halo],
        [1.5, c.text],
      ] as const) {
        ctx.strokeStyle = col;
        ctx.lineWidth = w;
        ctx.beginPath();
        ctx.moveTo(ox - 6, oy);
        ctx.lineTo(ox + 6, oy);
        ctx.moveTo(ox, oy - 6);
        ctx.lineTo(ox, oy + 6);
        ctx.stroke();
      }
      drawMath(ctx, [b('r'), mm(' = '), b('0')], ox + 9, oy - 8, {
        size: 13,
        color: c.text2,
        halo: c.halo,
      });
      for (const i of order) {
        const r = runs[i];
        const n = r.points.length;
        if (n === 0) continue;
        const col = c.series[r.slot % c.series.length];
        const lt = Math.max(0, Math.min(t, n - 1));
        const kk = Math.floor(lt);
        // The image of each straight step 𝐱ⱼ → 𝐱ⱼ₊₁ under 𝐫 is a curve: sample it.
        const trail: [number, number][] = [];
        for (let j = 0; j < kk; j++) {
          const a = r.points[j],
            e = r.points[j + 1];
          for (let q = 0; q <= 16; q++) {
            const u = q / 16;
            const rv = rOf([a[0] + (e[0] - a[0]) * u, a[1] + (e[1] - a[1]) * u]);
            trail.push(P(rv[0], rv[1]));
          }
        }
        const head = heads[i];
        if (head && kk < n - 1) {
          const a = r.points[kk];
          for (let q = 0; q <= 16; q++) {
            const u = q / 16;
            const rv = rOf([a[0] + (head[0] - a[0]) * u, a[1] + (head[1] - a[1]) * u]);
            trail.push(P(rv[0], rv[1]));
          }
        }
        if (trail.length === 0 && head) {
          const rv = rOf(head);
          trail.push(P(rv[0], rv[1]));
        }
        stroke(ctx, trail, col, c, r.focused ? 2 : 1.5, r.focused ? 1 : 0.7);
        for (let j = 0; j <= kk; j++) {
          const rv = rOf(r.points[j]);
          const [x, y] = P(rv[0], rv[1]);
          dot(ctx, x, y, 2.6, col, c.halo);
        }
        {
          // 𝐫(𝐱₀): hollow ring, as 𝐱₀ in the parameter plane.
          const rv = rOf(r.points[0]);
          const [x, y] = P(rv[0], rv[1]);
          ring(ctx, x, y, 5, c.text, c.halo);
        }
        if (head) {
          const rv = rOf(head);
          const [x, y] = P(rv[0], rv[1]);
          dot(ctx, x, y, 4.5, col, c.halo);
        }
      }
      if (prediction) {
        const focused = runs.find((r) => r.focused);
        const col = focused ? c.series[focused.slot % c.series.length] : c.text;
        const a = P(prediction.from[0], prediction.from[1]);
        const e = P(prediction.to[0], prediction.to[1]);
        stroke(ctx, [a, e], col, c, 1.3, 0.9, [5, 4]);
        ring(ctx, e[0], e[1], 4.5, col, c.halo);
        const k = String(prediction.k);
        // 𝐫ₖ + Jₖ𝐡ₖ (LM), 𝐫ₖ + Jₖ𝐩ₖ (GN, full step) or 𝐫ₖ + αₖJₖ𝐩ₖ (GN, backtracked).
        const alpha = prediction.step === 'alpha_p' ? [mv('α'), sub(k)] : [];
        const dir = prediction.step === 'h' ? b('h') : b('p');
        drawMath(
          ctx,
          [b('r'), sub(k), mm(' + '), ...alpha, mv('J'), sub(k), dir, sub(k)],
          e[0] + 9,
          e[1] + 4,
          {
            size: 12.5,
            color: c.text2,
            halo: c.halo,
          },
        );
      }
    }
    ctx.restore();
  });

  const down = useRef<{ x: number; y: number } | null>(null);
  const onDown = (e: RPointerEvent<HTMLCanvasElement>) => {
    down.current = { x: e.clientX, y: e.clientY };
  };
  const onUp = (e: RPointerEvent<HTMLCanvasElement>) => {
    const d = down.current;
    down.current = null;
    const f = frameRef.current;
    if (!onPick || !d || !f || Math.hypot(e.clientX - d.x, e.clientY - d.y) > 5) return;
    const r = e.currentTarget.getBoundingClientRect();
    const x = f.x.invert(e.clientX - r.left),
      y = f.y.invert(e.clientY - r.top);
    onPick([Number(x.toPrecision(4)), Number(y.toPrecision(4))]);
  };

  return (
    <canvas
      ref={canvasRef}
      className={styles.canvas}
      data-pick={onPick ? '' : undefined}
      role="img"
      aria-label={ariaLabel}
      onPointerDown={onDown}
      onPointerUp={onUp}
    />
  );
}
