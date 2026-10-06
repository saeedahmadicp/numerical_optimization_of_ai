/**
 * The 1-D minimization stage: f(x) with the geometry of the focused method's current step drawn
 * from Step.info, and below it the "funnel" of nested brackets of every method, one lane each.
 *
 *   - the bracket [a_k, b_k] (tinted) and, while the playhead moves to k + 1, the part the next
 *     comparison discards (hatched, fading in);
 *   - the probes (x₁, x₂ / λ, μ / x_l, x_m, x_r / x, w, v / a, b, c) on the curve, with their
 *     drop lines, and a ruler at the top that states the proportions they cut (golden section:
 *     0.382 : 0.236 : 0.382);
 *   - the parabola that produces the next trial (parabolic interpolation, Brent) or the Taylor
 *     model whose minimizer is x_{k+1} (Newton), its vertex and the next trial point;
 *   - every evaluation so far (small dots), the other methods' current points, x⋆;
 *   - the input bracket handles a, b and the start x₀, draggable in the whole-interval view.
 *
 * "Zoom to bracket" moves the camera with the focused method's region (eased in log-width
 * between steps); x and f are then shown relative to a printed reference when the view is
 * narrower than 10⁻⁴ of its position, and a locator inset shows where the window sits.
 */
import {
  useMemo,
  useRef,
  useState,
  type PointerEvent as RPointerEvent,
  type ReactNode,
} from 'react';
import { useChartColors } from '../../ui/theme';
import type { ChartColors } from '../../ui/colors';
import { sci, sig, tick } from '../../core/format';
import { easeInOut } from '../../play/timeline';

import {
  adaptiveSample,
  crisp,
  drawAxes,
  drawMath,
  linearScale,
  linearTicks,
  niceStep,
  mathMain,
  mathSub,
  mathSup,
  mathVar,
  useCanvas,
  type MathRun,
} from '../../viz';
import { LABEL_SIZE, TICK_SIZE, labelFont, tickFont } from '../../viz/axes';
import { kindOf, nextView, parabolaAt, proportions, type StepView } from './geometry';
import { PAD, followView, fullView } from './camera';
import type { Result } from '../../core/types';
import styles from './ScalarLab.module.css';

export interface StageRun {
  id: string;
  name: string;
  slot: number;
  views: StepView[];
  result: Result;
}

export type ViewMode = 'full' | 'follow';
export type Handle = 'a' | 'b' | 'x0';

export interface ScalarStageProps {
  f: (x: number) => number;
  domain: [number, number];
  minima: readonly number[];
  runs: readonly StageRun[];
  /** Index (into `runs`) of the method whose geometry is drawn. */
  focus: number;
  /** Continuous local time of each run (player.localT(i)). */
  times: readonly number[];
  ease: boolean;
  mode: ViewMode;
  /** The input bracket (handles a, b) — shown when a bracket method is selected. */
  bracket: [number, number] | null;
  /** The start point (handle x₀) — shown when a start-point method is selected. */
  x0: number | null;
  /** Live drag (`final` on release). */
  onDrag?: (handle: Handle, value: number, final: boolean) => void;
  ariaLabel: string;
  /** A note about the inputs, shown at the top of the plot (above the accuracy-floor note). */
  notice?: ReactNode;
}

/** The top strip (38 px) holds the DOM step badge; the y-axis name sits under it. */
const M = { left: 50, right: 16, top: 62 };
const TICK_ROW = 26;
const EPS = Number.EPSILON;

interface Layout {
  plot: { left: number; right: number; top: number; bottom: number };
  funnel: { left: number; right: number; top: number; bottom: number } | null;
}

/** Height of the strip under the funnel that names its interval (zoomed view only). */
const FUNNEL_AXIS = 17;

function layoutFor(
  w: number,
  h: number,
  lanes: number,
  zoomed: boolean,
  left: number = M.left,
): Layout {
  // Short stages (phones) keep the whole height for the plot; the convergence chart below
  // carries the widths.
  if (h < 400) lanes = 0;
  const funnelH = lanes ? Math.max(56, Math.min(180, Math.round(h * 0.27), 34 + lanes * 40)) : 0;
  const foot = zoomed && lanes ? FUNNEL_AXIS : 4;
  const plotBottom = h - (lanes ? funnelH + TICK_ROW + 8 + foot : TICK_ROW);
  return {
    plot: { left, right: w - M.right, top: M.top, bottom: plotBottom },
    funnel: lanes
      ? { left, right: w - M.right, top: plotBottom + TICK_ROW + 8, bottom: h - foot }
      : null,
  };
}

/** Left margin that fits the widest y tick label (mono: ≈ 0.61 em per character). */
function marginFor(lo: number, hi: number): number {
  const step = niceStep(lo, hi, 8);
  let widest = 0;
  for (const v of linearTicks(lo, hi, 8)) widest = Math.max(widest, tick(v, step).length);
  return Math.max(M.left, Math.min(96, Math.round(widest * TICK_SIZE * 0.61 + 14)));
}

/** Round `c` to a multiple of the decade below `width`: a short reference for offset axes. */
function reference(c: number, width: number): number {
  const step = 10 ** Math.floor(Math.log10(width));
  return Math.round(c / step) * step;
}

function refText(v: number, width: number): string {
  const digits = Math.min(17, Math.max(1, Math.ceil(Math.log10(Math.abs(v) / width)) + 2));
  return Number(v.toPrecision(digits)).toString().replace('-', '−');
}

/** Robust y-range of f on [lo, hi]. */
function yRange(
  f: (x: number) => number,
  lo: number,
  hi: number,
  robust: boolean,
): [number, number] {
  const ys: number[] = [];
  for (let i = 0; i <= 320; i++) {
    const v = f(lo + ((hi - lo) * i) / 320);
    if (Number.isFinite(v)) ys.push(v);
  }
  if (!ys.length) return [-1, 1];
  ys.sort((p, q) => p - q);
  let a = robust ? ys[Math.floor(ys.length * 0.01)] : ys[0];
  let b = robust ? ys[Math.floor(ys.length * 0.99)] : ys[ys.length - 1];
  if (!(b > a)) {
    const pad = Math.max(Math.abs(a) * 8 * EPS, 1e-300);
    a -= pad;
    b += pad;
  }
  // A quantized range (a few ulps) gets wide margins so its rounding levels sit clear of the frame.
  const fewUlps = b - a < 4096 * EPS * Math.max(Math.abs(a), Math.abs(b));
  const pad = (b - a) * (fewUlps ? 0.6 : 0.1);
  return [a - pad, b + pad * (fewUlps ? 1 : 1.6)];
}

const label = (name: [string, string], extra?: string): MathRun[] => {
  const runs: MathRun[] = [/[a-zA-Z]/.test(name[0]) ? mathVar(name[0]) : mathMain(name[0])];
  if (name[1]) runs.push(mathSub(name[1], /[a-z]/.test(name[1]) ? 'italic' : 'main'));
  if (extra) runs.push(mathMain(extra));
  return runs;
};

export function ScalarStage({
  f,
  domain,
  minima,
  runs,
  focus,
  times,
  ease,
  mode,
  bracket,
  x0,
  onDrag,
  ariaLabel,
  notice,
}: ScalarStageProps) {
  const colors = useChartColors();
  const box = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<Handle | null>(null);
  const [hover, setHover] = useState<Handle | null>(null);

  const full = useMemo(() => fullView(domain, bracket, x0), [domain, bracket, x0]);
  const fullY = useMemo(() => yRange(f, full[0], full[1], true), [f, full]);
  const focusRun = runs[focus];
  const tFocus = times[focus] ?? 0;
  const view: [number, number] =
    mode === 'follow' && focusRun ? followView(focusRun.views, tFocus, ease, full) : full;
  const viewW = view[1] - view[0];
  const yDom = mode === 'follow' ? yRange(f, view[0], view[1], false) : fullY;
  const xRef =
    mode === 'follow' && viewW < 1e-4 * Math.max(Math.abs(view[0]), Math.abs(view[1]))
      ? reference((view[0] + view[1]) / 2, viewW)
      : 0;
  const yW = yDom[1] - yDom[0];
  const yRef =
    mode === 'follow' && yW < 1e-4 * Math.max(Math.abs(yDom[0]), Math.abs(yDom[1]))
      ? reference((yDom[0] + yDom[1]) / 2, yW)
      : 0;
  const zoom = (full[1] - full[0]) / viewW;
  const zoomed = mode === 'follow' && zoom > 1.0001;
  /**
   * The values of f in view span fewer than ~4,000 ulps: f is quantized at this scale, so the
   * curve is drawn as its computed values (one dot per pixel) instead of a joined line, and the
   * comparisons that drive the methods are decided by the last bits.
   */
  const quantized = yW < 4096 * EPS * Math.max(Math.abs(yDom[0]), Math.abs(yDom[1]));

  const handlesOn = mode === 'full' && !!onDrag;

  const left = marginFor(yDom[0] - yRef, yDom[1] - yRef);
  const layout = (w: number, h: number) => {
    const L = layoutFor(w, h, runs.length, zoomed, left);
    const sx = linearScale([view[0] - xRef, view[1] - xRef], [L.plot.left, L.plot.right]);
    const sy = linearScale([yDom[0] - yRef, yDom[1] - yRef], [L.plot.bottom, L.plot.top]);
    const X = (x: number) => sx(x - xRef);
    const Y = (y: number) => sy(y - yRef);
    return { L, sx, sy, X, Y };
  };

  const { canvasRef } = useCanvas((ctx, s) => {
    const { L, sx, sy, X, Y } = layout(s.width, s.height);
    const P = L.plot;
    const xName: MathRun[] = xRef
      ? [mathVar('x'), mathMain(` − ${refText(xRef, viewW)}`)]
      : [mathVar('x')];
    const yName: MathRun[] = yRef
      ? [mathVar('f'), mathMain('('), mathVar('x'), mathMain(`) − (${refText(yRef, yW)})`)]
      : [mathVar('f'), mathMain('('), mathVar('x'), mathMain(')')];
    drawAxes(ctx, { x: sx, y: sy, frame: P, colors, dpr: s.dpr, xName, yName });

    ctx.save();
    ctx.beginPath();
    ctx.rect(P.left, P.top - 0.5, P.right - P.left, P.bottom - P.top + 1);
    ctx.clip();

    // A run that threw (invalid input) has no steps: nothing of it to draw.
    const fr = focusRun?.views.length ? focusRun : undefined;
    const lt = fr ? Math.max(0, Math.min(tFocus, fr.views.length - 1)) : 0;
    const k = Math.floor(lt);
    const u = ease ? easeInOut(lt - k) : 0;
    const cur = fr?.views[k];
    const nxt = fr ? nextView(fr.id, fr.result.trace, k) : null;
    const ink = fr ? colors.series[fr.slot % colors.series.length] : colors.text;

    // 1. Bracket tint and the part the next comparison discards.
    if (cur?.bracket) {
      const [a, b] = cur.bracket;
      ctx.fillStyle = ink;
      ctx.globalAlpha = 0.075;
      ctx.fillRect(X(a), P.top, X(b) - X(a), P.bottom - P.top);
      ctx.globalAlpha = 0.6;
      ctx.strokeStyle = ink;
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      for (const e of [a, b]) {
        ctx.moveTo(crisp(X(e), s.dpr), P.top);
        ctx.lineTo(crisp(X(e), s.dpr), P.bottom);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
      if (nxt?.nextBracket && fr && fr.id !== 'bracket_minimum') {
        const [na, nb] = nxt.nextBracket;
        const alpha = 0.16 + 0.5 * u;
        if (na > a) hatch(ctx, X(a), X(na), P.top, P.bottom, colors, alpha);
        if (nb < b) hatch(ctx, X(nb), X(b), P.top, P.bottom, colors, alpha);
      }
    }

    // 2. Known minimizers (+ crosses), the curve on top of the tints.
    if (quantized) dotCurve(ctx, f, view[0], view[1], P, X, Y, colors.text);
    else strokeCurve(ctx, f, view[0], view[1], X, Y, colors.text);
    minima.forEach((m, i) => {
      if (m < view[0] || m > view[1]) return;
      const fm = f(m);
      cross(ctx, X(m), Y(fm), colors);
      if (i === 0 || minima.length === 1)
        drawMath(ctx, [mathVar('x'), mathSup('⋆')], X(m), Y(fm) + 18, {
          size: 12.5,
          align: 'center',
          color: colors.text2,
          halo: colors.halo,
        });
    });

    // 3. The model of the next step: parabola / Taylor model, its vertex, the next trial.
    if (fr && cur && nxt) {
      if (nxt.parabola) {
        const pts = [
          ...cur.probes.map((p) => p.x),
          ...(nxt.vertex !== null ? [nxt.vertex] : []),
          ...nxt.trials.map((t) => t[0]),
        ].filter(Number.isFinite);
        let lo = Math.min(...pts),
          hi = Math.max(...pts);
        const ext = Math.max((hi - lo) * 0.35, viewW * 0.04);
        lo -= ext;
        hi += ext;
        const par = nxt.parabola;
        ctx.save();
        ctx.setLineDash([6, 4]);
        strokeCurve(ctx, (z) => parabolaAt(par, z), lo, hi, X, Y, ink, 1.6, 0.95);
        ctx.restore();
        if (nxt.vertex !== null && Number.isFinite(nxt.vertex)) {
          const vx = nxt.vertex,
            vy = parabolaAt(par, vx);
          diamond(ctx, X(vx), Y(vy), 4.2, colors.halo, ink);
        }
      }
      // Newton: the step from x_k to x_{k+1} along the model, then down to the curve.
      if (fr.id === 'newton_1d' && nxt.nextX !== null && Number.isFinite(nxt.nextX)) {
        const x1 = nxt.nextX;
        // A Newton step lands on the model's minimizer; a gradient step (f'' ≤ 0: the model has
        // no minimizer) lands where backtracking accepted it, on the curve.
        const y1 = nxt.parabola && nxt.stepKind === 'newton' ? parabolaAt(nxt.parabola, x1) : f(x1);
        const py1 = Y(y1);
        if (py1 > P.bottom || py1 < P.top) {
          // The model's minimum lies outside the view: the arrow stops at the frame, a chevron
          // points on, and the landing value is printed (portal-direction §2.4).
          const below = py1 > P.bottom;
          const edge = below ? P.bottom - 2 : P.top + 2;
          const t = (edge - Y(cur.f)) / (py1 - Y(cur.f));
          const ex = X(cur.x) + (X(x1) - X(cur.x)) * Math.max(0, Math.min(1, t));
          arrow(ctx, X(cur.x), Y(cur.f), ex, edge, ink, u);
          const right = ex > (P.left + P.right) / 2;
          drawMath(
            ctx,
            [
              mathVar('m'),
              mathMain('('),
              mathVar('x'),
              mathSub(String(k + 1)),
              mathMain(`) = ${sig(y1, 3)} ${below ? '↓' : '↑'}`),
            ],
            ex + (right ? -10 : 10),
            below ? edge - 8 : edge + 16,
            { size: 12, align: right ? 'right' : 'left', color: colors.text2, halo: colors.halo },
          );
        } else arrow(ctx, X(cur.x), Y(cur.f), X(x1), py1, ink, u);
        ctx.save();
        ctx.setLineDash([2, 3]);
        ctx.strokeStyle = colors.text3;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(X(x1), Y(y1));
        ctx.lineTo(X(x1), Y(f(x1)));
        ctx.stroke();
        ctx.restore();
      }
    }

    // 4. Every evaluation so far (the record), then the other methods' current points.
    if (fr) {
      ctx.fillStyle = ink;
      ctx.globalAlpha = 0.45;
      for (let i = 0; i <= k; i++)
        for (const [ex, ey] of fr.views[i].evaluated)
          if (Number.isFinite(ey) && ex >= view[0] && ex <= view[1]) {
            ctx.beginPath();
            ctx.arc(X(ex), Y(ey), 2.1, 0, Math.PI * 2);
            ctx.fill();
          }
      ctx.globalAlpha = 1;
    }
    runs.forEach((r, i) => {
      if (i === focus) return;
      const t = Math.max(0, Math.min(times[i] ?? 0, r.views.length - 1));
      const v = r.views[Math.floor(t)];
      if (!v || !Number.isFinite(v.f)) return;
      dot(ctx, X(v.x), Y(v.f), 3.6, colors.series[r.slot % colors.series.length], colors.halo);
      if (v.bracket) {
        ctx.strokeStyle = colors.series[r.slot % colors.series.length];
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (const e of v.bracket) {
          ctx.moveTo(X(e), P.bottom);
          ctx.lineTo(X(e), P.bottom - 9);
        }
        ctx.stroke();
      }
    });

    // 5. The focused method's points, labelled in its own notation.
    if (fr && cur) {
      if (kindOf(fr.id) === 'interval' && cur.probes.length === 2) {
        const [p, q] = cur.probes;
        if (Number.isFinite(p.f) && Number.isFinite(q.f)) {
          // The comparison: a level line at the lower probe, across to the other one.
          const lo = p.f <= q.f ? p : q;
          ctx.save();
          ctx.setLineDash([2, 3]);
          ctx.strokeStyle = colors.text2;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(X(p.x), Y(lo.f));
          ctx.lineTo(X(q.x), Y(lo.f));
          ctx.stroke();
          ctx.restore();
        }
      }
      for (const p of cur.probes) {
        if (!Number.isFinite(p.f)) continue;
        const px = X(p.x),
          py = Y(p.f);
        ctx.save();
        ctx.setLineDash([2, 3]);
        ctx.strokeStyle = colors.text3;
        ctx.globalAlpha = 0.8;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(crisp(px, s.dpr), py + 5);
        ctx.lineTo(crisp(px, s.dpr), P.bottom);
        ctx.stroke();
        ctx.restore();
      }
      // Labels: alternate above/below when probes crowd each other.
      const sorted = [...cur.probes].filter((p) => Number.isFinite(p.f)).sort((a, b) => a.x - b.x);
      let lastX = -Infinity;
      let flip = false;
      for (const p of sorted) {
        const px = X(p.x),
          py = Y(p.f);
        dot(ctx, px, py, p.best ? 4.6 : 4, ink, colors.halo);
        flip = px - lastX < 26 ? !flip : false;
        lastX = px;
        const above = py - 16 > P.top + 8 && !flip;
        const lx = Math.max(P.left + 10, Math.min(P.right - 10, px));
        drawMath(ctx, label(p.name), lx, above ? py - 11 : py + 21, {
          size: 13,
          align: 'center',
          color: colors.text,
          halo: colors.halo,
        });
      }
      // The next trial(s): rings that appear while the playhead moves to k + 1. The trial
      // points are named as in "This step": u, and u′ when a step evaluates a second point
      // (minimum bracketing: the vertex, then the φ-step from it). Elimination methods' new
      // probes get their own names at k + 1.
      if (nxt && fr.id !== 'newton_1d') {
        ctx.globalAlpha = 0.35 + 0.65 * u;
        const named = kindOf(fr.id) !== 'interval';
        nxt.trials.forEach(([tx, ty], i) => {
          if (!Number.isFinite(ty)) return;
          ring(ctx, X(tx), Y(ty), 4.4, ink, colors.halo);
          if (named && i < 2)
            drawMath(
              ctx,
              [mathVar('u'), ...(i === 1 ? [mathMain('′')] : [])],
              X(tx) + 9,
              Y(ty) + 4,
              {
                size: 13,
                color: colors.text2,
                halo: colors.halo,
              },
            );
        });
        ctx.globalAlpha = 1;
      }
      if (fr.id === 'newton_1d' && nxt && nxt.trials.length > 1) {
        // Backtracking trials of a gradient step (rejected: hollow; the accepted one is x_{k+1}).
        nxt.trials.slice(0, -1).forEach(([tx, ty]) => {
          if (Number.isFinite(ty)) ring(ctx, X(tx), Y(ty), 3.4, ink, colors.halo);
        });
      }
    }

    // 6. Proportion ruler: how the probes cut the bracket (golden section: 0.382 : 0.236 : 0.382).
    if (fr && cur?.bracket && cur.probes.length === 2 && kindOf(fr.id) === 'interval') {
      const [a, b] = cur.bracket;
      const [p, q] = cur.probes;
      const ry = P.top + 16;
      const xs = [a, p.x, q.x, b].map(X);
      ctx.strokeStyle = colors.text2;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(xs[0], ry);
      ctx.lineTo(xs[3], ry);
      for (const px of xs) {
        ctx.moveTo(crisp(px, s.dpr), ry - 4);
        ctx.lineTo(crisp(px, s.dpr), ry + 4);
      }
      ctx.stroke();
      const parts = proportions(a, b, p.x, q.x);
      ctx.font = tickFont(colors);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      parts.forEach((r, i) => {
        // Centre each number on the visible part of its segment; skip it when that part is
        // narrower than the text (a segment cut by the frame never shows a clipped number).
        const lo = Math.max(xs[i], P.left + 2),
          hi = Math.min(xs[i + 1], P.right - 2);
        const txt = r.toFixed(3);
        if (hi - lo < ctx.measureText(txt).width + 6) return;
        const mid = (lo + hi) / 2;
        ctx.strokeStyle = colors.halo;
        ctx.lineWidth = 3;
        ctx.strokeText(txt, mid, ry - 3);
        ctx.fillStyle = colors.text2;
        ctx.fillText(txt, mid, ry - 3);
      });
      // a_k and b_k at the ruler's ends; an end outside the view is pinned to the frame with an
      // outward chevron.
      for (const [e, nm, al] of [
        [a, 'a', 'right'],
        [b, 'b', 'left'],
      ] as const) {
        const px = X(e);
        // No room outside the end (it sits at the frame): the label goes under the ruler,
        // inside the bracket; an end beyond the frame gets an outward chevron as well.
        const room = nm === 'a' ? px - P.left > 22 : P.right - px > 22;
        const off = nm === 'a' ? px < P.left : px > P.right;
        const name: MathRun[] = [mathVar(nm), mathSub(String(k))];
        let at = px + (al === 'right' ? -5 : 5);
        let y = ry + 4;
        let align: 'left' | 'right' = al;
        let runs = name;
        if (!room) {
          y = ry + 18;
          align = nm === 'a' ? 'left' : 'right';
          const edge = nm === 'a' ? Math.max(px, P.left) + 4 : Math.min(px, P.right) - 4;
          at = edge;
          if (off) runs = nm === 'a' ? [mathMain('‹ '), ...name] : [...name, mathMain(' ›')];
        }
        drawMath(ctx, runs, at, y, {
          size: 12,
          align,
          color: colors.text2,
          halo: colors.halo,
        });
      }
    }
    ctx.restore();

    // 7. Input handles (whole-interval view): the bracket a, b and the start x₀.
    if (handlesOn) {
      if (bracket) {
        for (const [e, nm] of [
          [bracket[0], 'a'],
          [bracket[1], 'b'],
        ] as const) {
          const px = X(e);
          if (px < P.left - 1 || px > P.right + 1) continue;
          const active = drag === nm || hover === nm;
          handle(ctx, px, P.bottom, active, colors);
          drawMath(ctx, [mathVar(nm), mathSub('0')], px, P.bottom - 15, {
            size: 12.5,
            align: 'center',
            color: colors.text,
            halo: colors.halo,
          });
        }
      }
      if (x0 !== null && Number.isFinite(x0)) {
        const fx = f(x0);
        if (Number.isFinite(fx)) {
          const px = X(x0),
            py = Math.max(P.top + 6, Math.min(P.bottom - 6, Y(fx)));
          ring(
            ctx,
            px,
            py,
            drag === 'x0' || hover === 'x0' ? 7 : 5.5,
            colors.text,
            colors.halo,
            1.75,
          );
          drawMath(ctx, [mathVar('x'), mathSub('0')], px + 10, py - 9, {
            size: 13,
            color: colors.text,
            halo: colors.halo,
          });
        }
      }
    }

    // 9. The funnel: every method's bracket at every step so far, one lane per method.
    if (L.funnel) {
      // Zoomed: the funnel keeps the whole interval, with the window of the plot marked on it
      // and joined to the plot's corners, like a magnifier.
      const fx = zoomed ? linearScale(full, [L.funnel.left, L.funnel.right]) : X;
      if (zoomed) {
        const F = L.funnel;
        const w0 = fx(view[0]),
          w1 = Math.max(fx(view[1]), w0 + 1.5);
        ctx.save();
        ctx.fillStyle = ink;
        ctx.globalAlpha = 0.07;
        ctx.fillRect(w0, F.top, w1 - w0, F.bottom - F.top);
        ctx.globalAlpha = 0.7;
        ctx.strokeStyle = ink;
        ctx.lineWidth = 1;
        ctx.strokeRect(crisp(w0, s.dpr), crisp(F.top, s.dpr), w1 - w0, F.bottom - F.top);
        ctx.globalAlpha = 0.55;
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = colors.text3;
        // Toward the plot's corners, stopping under the tick labels.
        const yStop = P.bottom + TICK_ROW - 2;
        const along = (fx0: number, px: number) =>
          fx0 + ((px - fx0) * (F.top - yStop)) / (F.top - P.bottom);
        ctx.beginPath();
        ctx.moveTo(w0, F.top);
        ctx.lineTo(along(w0, P.left), yStop);
        ctx.moveTo(w1, F.top);
        ctx.lineTo(along(w1, P.right), yStop);
        ctx.stroke();
        ctx.restore();
      }
      drawFunnel(ctx, L.funnel, runs, times, fx, colors, s.dpr);
      if (zoomed) {
        const F = L.funnel;
        // The funnel's interval and the zoom factor.
        ctx.save();
        ctx.font = tickFont(colors);
        ctx.fillStyle = colors.text3;
        ctx.textBaseline = 'bottom';
        ctx.textAlign = 'left';
        ctx.fillText(sig(full[0], 4), F.left, s.height - 1);
        ctx.textAlign = 'right';
        ctx.fillText(sig(full[1], 4), F.right, s.height - 1);
        ctx.restore();
        drawMath(
          ctx,
          [mathMain('window × '), ...zoomRuns(zoom)],
          (F.left + F.right) / 2,
          s.height - 3,
          {
            size: TICK_SIZE + 1,
            align: 'center',
            color: colors.text2,
            halo: colors.halo,
          },
        );
      }
    }
  });

  // ── Pointer: drag the bracket ends and the start point (whole-interval view) ──────────
  const hit = (e: { clientX: number; clientY: number }): Handle | null => {
    const r = box.current?.getBoundingClientRect();
    if (!r || !handlesOn) return null;
    const { L, X, Y } = layout(r.width, r.height);
    const px = e.clientX - r.left,
      py = e.clientY - r.top;
    let best: Handle | null = null;
    let bestD = 14;
    if (bracket && py > L.plot.bottom - 34 && py < L.plot.bottom + 14) {
      for (const [h, v] of [
        ['a', bracket[0]],
        ['b', bracket[1]],
      ] as const) {
        const d = Math.abs(X(v) - px);
        if (d < bestD) [best, bestD] = [h, d];
      }
    }
    if (x0 !== null) {
      const fy = f(x0);
      const yy = Number.isFinite(fy)
        ? Math.max(L.plot.top + 6, Math.min(L.plot.bottom - 6, Y(fy)))
        : NaN;
      const d = Math.hypot(X(x0) - px, (yy - py) * 0.6);
      if (d < Math.min(bestD, 16)) best = 'x0';
    }
    return best;
  };
  const valueAt = (clientX: number): number => {
    const r = box.current!.getBoundingClientRect();
    const { L, sx } = layout(r.width, r.height);
    const px = Math.max(L.plot.left, Math.min(L.plot.right, clientX - r.left));
    const v = sx.invert(px) + xRef;
    // Snap to 4 significant digits of the view width (readable URLs, exact for the methods).
    const step = 10 ** (Math.floor(Math.log10(viewW)) - 3);
    return Number((Math.round(v / step) * step).toPrecision(12));
  };
  const clampFor = (h: Handle, v: number): number => {
    const lo = domain[0],
      hi = domain[1];
    if (h === 'x0') return Math.max(lo, Math.min(hi, v));
    if (!bracket) return v;
    const gap = viewW * 0.01;
    if (h === 'a') return Math.max(lo, Math.min(v, bracket[1] - gap));
    return Math.min(hi, Math.max(v, bracket[0] + gap));
  };

  const onPointerDown = (e: RPointerEvent<HTMLDivElement>) => {
    const h = hit(e);
    if (!h) {
      // A click on the plot sets x₀ when a start-point method is selected.
      if (handlesOn && x0 !== null) {
        const r = box.current!.getBoundingClientRect();
        const { L } = layout(r.width, r.height);
        const py = e.clientY - r.top;
        if (py >= L.plot.top && py <= L.plot.bottom)
          onDrag?.('x0', clampFor('x0', valueAt(e.clientX)), true);
      }
      return;
    }
    e.currentTarget.setPointerCapture(e.pointerId);
    setDrag(h);
  };
  const onPointerMove = (e: RPointerEvent<HTMLDivElement>) => {
    if (drag) {
      onDrag?.(drag, clampFor(drag, valueAt(e.clientX)), false);
      return;
    }
    if (e.pointerType === 'mouse') {
      const h = hit(e);
      if (h !== hover) setHover(h);
    }
  };
  const onPointerUp = (e: RPointerEvent<HTMLDivElement>) => {
    if (!drag) return;
    onDrag?.(drag, clampFor(drag, valueAt(e.clientX)), true);
    setDrag(null);
  };

  return (
    <div
      ref={box}
      className={styles.stageCanvas}
      data-drag={drag ? 'true' : undefined}
      data-handle={hover && !drag ? 'true' : undefined}
      data-pick={handlesOn && x0 !== null ? 'true' : undefined}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={() => setDrag(null)}
      onPointerLeave={() => setHover(null)}
    >
      <canvas ref={canvasRef} role="img" aria-label={ariaLabel} />
      {(quantized || notice) && (
        <div className={styles.plotNotes} style={{ top: M.top + 30, left: left + 8 }}>
          {notice}
          {quantized && (
            <p className={styles.floorNote} role="note">
              At this scale f takes only a few rounding levels: each dot is one computed value.
              Probes {sci(viewW / PAD, 2)} apart are compared by their last bits, so this is the
              accuracy floor.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

// ── Drawing helpers ─────────────────────────────────────────────────────────────────────

function zoomRuns(z: number): MathRun[] {
  if (z < 1000) return [mathMain(z < 10 ? z.toFixed(1) : Math.round(z).toString())];
  const e = Math.floor(Math.log10(z));
  const m = z / 10 ** e;
  return [mathMain(`${m.toFixed(1)} × 10`), mathSup(String(e))];
}

/** f at one point per pixel, as dots: at the floor the computed values sit on rounding levels. */
function dotCurve(
  ctx: CanvasRenderingContext2D,
  g: (x: number) => number,
  from: number,
  to: number,
  P: { left: number; right: number },
  X: (x: number) => number,
  Y: (y: number) => number,
  color: string,
) {
  const n = Math.max(2, Math.round(P.right - P.left));
  ctx.save();
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.8;
  for (let i = 0; i <= n; i++) {
    const x = from + ((to - from) * i) / n;
    const y = g(x);
    if (!Number.isFinite(y)) continue;
    ctx.fillRect(X(x) - 0.8, Y(y) - 0.8, 1.6, 1.6);
  }
  ctx.restore();
}

function strokeCurve(
  ctx: CanvasRenderingContext2D,
  g: (x: number) => number,
  from: number,
  to: number,
  X: (x: number) => number,
  Y: (y: number) => number,
  color: string,
  width = 1.75,
  alpha = 0.88,
) {
  if (!(to > from)) return;
  const pts = adaptiveSample(g, from, to, (a, b) => [X(a), Y(b)], 0.3, 160, 8);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.globalAlpha = alpha;
  ctx.lineWidth = width;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  let pen = false;
  for (const [a, b] of pts) {
    const px = X(a),
      py = Y(b);
    if (!Number.isFinite(py) || Math.abs(py) > 1e6) {
      pen = false;
      continue;
    }
    if (pen) ctx.lineTo(px, py);
    else ctx.moveTo(px, py);
    pen = true;
  }
  ctx.stroke();
  ctx.restore();
}

function hatch(
  ctx: CanvasRenderingContext2D,
  x0: number,
  x1: number,
  top: number,
  bottom: number,
  colors: ChartColors,
  alpha: number,
) {
  if (x1 - x0 < 0.5) return;
  ctx.save();
  ctx.beginPath();
  ctx.rect(x0, top, x1 - x0, bottom - top);
  ctx.clip();
  ctx.fillStyle = colors.region;
  ctx.globalAlpha = alpha * 0.16;
  ctx.fillRect(x0, top, x1 - x0, bottom - top);
  ctx.globalAlpha = alpha * 0.55;
  ctx.strokeStyle = colors.region;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = x0 - (bottom - top); x < x1; x += 7) {
    ctx.moveTo(x, bottom);
    ctx.lineTo(x + (bottom - top), top);
  }
  ctx.stroke();
  ctx.restore();
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
  ctx.arc(x, y, r + 1.8, 0, Math.PI * 2);
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
  stroke: string,
  halo: string,
  width = 1.6,
) {
  ctx.save();
  ctx.fillStyle = halo;
  ctx.beginPath();
  ctx.arc(x, y, r + 1.8, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = stroke;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function diamond(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  r: number,
  fill: string,
  stroke: string,
) {
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(x, y - r);
  ctx.lineTo(x + r, y);
  ctx.lineTo(x, y + r);
  ctx.lineTo(x - r, y);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.restore();
}

function cross(ctx: CanvasRenderingContext2D, x: number, y: number, colors: ChartColors) {
  for (const [w, c] of [
    [3.5, colors.halo],
    [1.5, colors.text],
  ] as const) {
    ctx.strokeStyle = c;
    ctx.lineWidth = w;
    ctx.beginPath();
    ctx.moveTo(x - 6, y);
    ctx.lineTo(x + 6, y);
    ctx.moveTo(x, y - 6);
    ctx.lineTo(x, y + 6);
    ctx.stroke();
  }
}

function arrow(
  ctx: CanvasRenderingContext2D,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  color: string,
  u: number,
) {
  const len = Math.hypot(x1 - x0, y1 - y0);
  if (!(len > 2) || !Number.isFinite(len)) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.55 + 0.45 * u;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  const ux = (x1 - x0) / len,
    uy = (y1 - y0) / len,
    s = Math.min(8, len * 0.4);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - ux * s - uy * s * 0.5, y1 - uy * s + ux * s * 0.5);
  ctx.lineTo(x1 - ux * s + uy * s * 0.5, y1 - uy * s - ux * s * 0.5);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function handle(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  active: boolean,
  colors: ChartColors,
) {
  const s = active ? 7 : 6;
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(x, y - s * 1.2);
  ctx.lineTo(x + s, y + 1);
  ctx.lineTo(x - s, y + 1);
  ctx.closePath();
  ctx.fillStyle = colors.halo;
  ctx.lineWidth = 3;
  ctx.strokeStyle = colors.halo;
  ctx.stroke();
  ctx.fillStyle = active ? colors.accent : colors.text;
  ctx.fill();
  ctx.restore();
}

/** Height of a funnel lane's header row (swatch, name, width readout). */
const LANE_HEAD = Math.ceil(LABEL_SIZE * 1.5);

/** A surface-colored backing for a label drawn over other marks. */
function plate(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  fill: string,
) {
  ctx.save();
  ctx.fillStyle = fill;
  ctx.fillRect(x, y, w, h);
  ctx.restore();
}

function drawFunnel(
  ctx: CanvasRenderingContext2D,
  F: { left: number; right: number; top: number; bottom: number },
  runs: readonly StageRun[],
  times: readonly number[],
  X: (x: number) => number,
  colors: ChartColors,
  dpr: number,
) {
  const n = runs.length;
  const laneGap = 6;
  const laneH = (F.bottom - F.top - laneGap * (n - 1)) / n;
  ctx.save();
  ctx.font = labelFont(colors);
  runs.forEach((r, i) => {
    const top = F.top + i * (laneH + laneGap);
    const color = colors.series[r.slot % colors.series.length];
    const t = Math.max(0, Math.min(times[i] ?? 0, r.views.length - 1));
    const kNow = Math.floor(t);
    const head = LANE_HEAD;
    const rows = Math.max(1, r.views.length);
    const rowH = Math.min(6, (laneH - head - 2) / rows);
    const cy = top + head / 2 - 0.5;
    // Lane frame (a hairline under the header).
    ctx.strokeStyle = colors.grid;
    ctx.lineWidth = 1 / dpr;
    ctx.beginPath();
    ctx.moveTo(F.left, crisp(top + head - 1, dpr));
    ctx.lineTo(F.right, crisp(top + head - 1, dpr));
    ctx.stroke();
    // Header: swatch, name, current width. Each label sits on a surface-colored plate, so the
    // zoom window and its connectors (drawn before the funnel) never cross the text.
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    ctx.font = labelFont(colors);
    const nameW = ctx.measureText(r.name).width;
    plate(ctx, F.left - 2, top, 13 + nameW + 6, head - 2, colors.surface);
    ctx.fillStyle = color;
    ctx.fillRect(F.left, cy - 4, 8, 8);
    ctx.fillStyle = colors.text2;
    ctx.fillText(r.name, F.left + 13, cy);
    const v = r.views[kNow];
    if (v) {
      const w = v.bracket ? v.bracket[1] - v.bracket[0] : null;
      // Minimum bracketing names its triple (a, b, c): its width is |c − a|.
      const wName = r.id === 'bracket_minimum' ? '|c − a|' : 'b − a';
      const txt = w !== null ? `k = ${kNow}   ${wName} = ${sci(w, 2)}` : `k = ${kNow}`;
      ctx.font = tickFont(colors);
      const txtW = ctx.measureText(txt).width;
      // Keep clear of the name on a narrow stage (drop the readout instead of overlapping).
      if (F.right - txtW > F.left + 13 + nameW + 12) {
        plate(ctx, F.right - txtW - 4, top, txtW + 6, head - 2, colors.surface);
        ctx.textAlign = 'right';
        ctx.fillStyle = colors.text3;
        ctx.fillText(txt, F.right, cy);
      }
      ctx.font = labelFont(colors);
    }
    // Rows k = 0 … kNow: nested brackets (or Newton's iterates).
    ctx.save();
    ctx.beginPath();
    ctx.rect(F.left, top + head, F.right - F.left, laneH - head);
    ctx.clip();
    const barH = Math.max(1.25, rowH * 0.78);
    let prev: [number, number] | null = null;
    for (let k = 0; k <= Math.min(kNow, r.views.length - 1); k++) {
      const s = r.views[k];
      const y = top + head + 1 + k * rowH;
      const current = k === kNow;
      ctx.globalAlpha = current ? 1 : 0.55;
      ctx.fillStyle = color;
      if (s.bracket) {
        const x0 = X(s.bracket[0]),
          x1 = Math.max(X(s.bracket[1]), x0 + 1);
        ctx.fillRect(x0, y, x1 - x0, barH);
        if (current && Number.isFinite(s.x)) {
          ctx.fillStyle = colors.halo;
          ctx.fillRect(X(s.x) - 0.75, y - 1, 1.5, barH + 2);
        }
      } else if (Number.isFinite(s.x)) {
        const px = X(s.x),
          py = y + barH / 2;
        if (prev) {
          ctx.globalAlpha = current ? 1 : 0.3;
          ctx.strokeStyle = color;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(prev[0], prev[1]);
          ctx.lineTo(px, py);
          ctx.stroke();
        }
        ctx.beginPath();
        ctx.arc(px, py, Math.max(1.2, Math.min(2.6, rowH / 2)), 0, Math.PI * 2);
        ctx.fill();
        prev = [px, py];
      }
    }
    ctx.restore();
    ctx.globalAlpha = 1;
  });
  ctx.restore();
}
