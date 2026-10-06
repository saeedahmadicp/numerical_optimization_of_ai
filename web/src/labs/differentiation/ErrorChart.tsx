/**
 * Error |D(h) − f⁽ᵠ⁾(x0)| against the step h on log–log axes, one V per method. The h axis runs
 * toward 0 from left to right, on the same clock as the playback bar. For the focused method it
 * also draws the method's own error model (the Richardson truncation estimate, ∝ hᵖ, and the
 * round-off model, ∝ ε/hᵠ) whose crossing the level selection looks for, the selected step h⋆
 * (ring), the textbook h_opt (axis tick) and the floor ε·|f⁽ᵠ⁾(x0)|. Exact zeros sit on the
 * bottom edge as hollow triangles.
 */
import { useRef, useState, type PointerEvent as RPointerEvent } from 'react';
import { useChartColors } from '../../ui/theme';
import { sci } from '../../core/format';
import type { Step } from '../../core/types';
import { useCanvas } from '../../viz/useCanvas';
import { crisp, logScale } from '../../viz/scales';
import {
  drawMath,
  measureMath,
  m as mm,
  pow10Runs,
  sans,
  sub,
  sup,
  v as mv,
  type MathRun,
} from '../../viz/mathText';
import { easeInOut } from '../../play/timeline';
import { errorsOf } from './model';
import styles from './DifferentiationLab.module.css';

export interface ErrorRun {
  id: string;
  name: string;
  slot: number;
  trace: readonly Step[];
  /** Local continuous playhead. */
  t: number;
  kBest: number | null;
  hOpt: number | null;
  /** Truncation order p and derivative order q. */
  order: number;
  derivative: number;
}

export interface ErrorChartProps {
  runs: readonly ErrorRun[];
  focus: number;
  /** ε·|f′(x0)| or ε·|f″(x0)| for the focused method's derivative (null if unknown). */
  floor: number | null;
  showModel: boolean;
  ease: boolean;
  onSeek?: (k: number) => void;
  ariaLabel: string;
}

const M = { left: 50, right: 16, top: 30, bottom: 34 };

interface Hover {
  k: number;
  px: number;
  /** Width of the chart box (read in the event handler, not during render). */
  width: number;
}

export function ErrorChart({
  runs,
  focus,
  floor,
  showModel,
  ease,
  onSeek,
  ariaLabel,
}: ErrorChartProps) {
  const colors = useChartColors();
  const box = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<Hover | null>(null);
  const focusRun = runs[focus] ?? runs[0];

  // Domains (data only; model lines are clipped).
  const hs = runs.flatMap((r) => r.trace.map((s) => s.info.h as number)).filter((h) => h > 0);
  const hMax = hs.length ? Math.max(...hs) : 1;
  const hMin = hs.length ? Math.min(...hs) : 1e-3;
  const errs = runs
    .flatMap((r) => errorsOf(r.trace))
    .filter((e): e is number => e !== null && e > 0);
  if (floor !== null && floor > 0) errs.push(floor);
  const eLo = errs.length ? 10 ** Math.floor(Math.log10(Math.min(...errs)) - 0.35) : 1e-16;
  const eHi = errs.length ? 10 ** Math.ceil(Math.log10(Math.max(...errs)) + 0.15) : 1;
  const xDom: [number, number] = [hMin / 1.6, hMax * 1.6];

  const scales = (w: number, h: number) => ({
    // Reversed: large h on the left, h → 0 to the right.
    x: logScale(xDom, [w - M.right, M.left]),
    y: logScale([eLo, eHi], [h - M.bottom, M.top]),
  });

  const { canvasRef } = useCanvas((ctx, size) => {
    const { width, height, dpr } = size;
    const frame = { left: M.left, top: M.top, right: width - M.right, bottom: height - M.bottom };
    if (frame.right - frame.left < 60 || frame.bottom - frame.top < 40) return;
    const { x, y } = scales(width, height);
    const hair = 1 / dpr;

    // ── Grid and axes (decades) ───────────────────────────────────────────────────────────
    const decades = (lo: number, hi: number) => {
      const out: number[] = [];
      for (let e = Math.ceil(Math.log10(lo)); e <= Math.floor(Math.log10(hi)); e++) out.push(e);
      return out;
    };
    const xd = decades(xDom[0], xDom[1]);
    const yd = decades(eLo, eHi);
    const xEvery = Math.max(
      1,
      Math.ceil(xd.length / Math.max(2, Math.floor((frame.right - frame.left) / 64))),
    );
    const yEvery = Math.max(
      1,
      Math.ceil(yd.length / Math.max(2, Math.floor((frame.bottom - frame.top) / 34))),
    );
    ctx.save();
    ctx.strokeStyle = colors.grid;
    ctx.lineWidth = hair;
    ctx.beginPath();
    for (const e of xd) {
      if (e % xEvery) continue;
      const px = crisp(x(10 ** e), dpr);
      ctx.moveTo(px, frame.top);
      ctx.lineTo(px, frame.bottom);
    }
    for (const e of yd) {
      if (e % yEvery) continue;
      const py = crisp(y(10 ** e), dpr);
      ctx.moveTo(frame.left, py);
      ctx.lineTo(frame.right, py);
    }
    ctx.stroke();
    ctx.strokeStyle = colors.axis;
    ctx.beginPath();
    ctx.moveTo(crisp(frame.left, dpr), frame.top);
    ctx.lineTo(crisp(frame.left, dpr), crisp(frame.bottom, dpr));
    ctx.lineTo(frame.right, crisp(frame.bottom, dpr));
    ctx.stroke();
    // x-axis name at the right end of the tick row; ticks that would collide are skipped.
    const xName: MathRun[] = [sans('step '), mv('h'), mm(' → 0')];
    const nameW = drawMath(ctx, xName, frame.right, frame.bottom + 27, {
      size: 12.5,
      align: 'right',
      color: colors.text2,
    });
    // The focused method's textbook h_opt (for the complex step: h_ε, below which truncation is
    // under ε) is a special tick of the h axis: a triangle under the axis, its label in the tick
    // row, where it cannot cross the data, the floor or the zero marks.
    const hOptPx =
      focusRun &&
      showModel &&
      focusRun.hOpt !== null &&
      focusRun.hOpt > xDom[0] &&
      focusRun.hOpt < xDom[1]
        ? x(focusRun.hOpt)
        : null;
    if (hOptPx !== null) {
      ctx.fillStyle = colors.text2;
      ctx.beginPath();
      ctx.moveTo(hOptPx, frame.bottom + 1);
      ctx.lineTo(hOptPx - 4, frame.bottom + 6.5);
      ctx.lineTo(hOptPx + 4, frame.bottom + 6.5);
      ctx.closePath();
      ctx.fill();
      drawMath(
        ctx,
        [mv('h'), focusRun.id === 'complex_step' ? sub('ε', 'italic') : sub('opt')],
        hOptPx,
        frame.bottom + 17,
        { size: 12, align: 'center', color: colors.text, halo: colors.halo },
      );
    }
    for (const e of xd) {
      if (e % xEvery) continue;
      const px = x(10 ** e);
      if (px > frame.right - nameW - 22 && px < frame.right + 4) continue;
      if (hOptPx !== null && Math.abs(px - hOptPx) < 30) continue;
      drawMath(ctx, pow10Runs(e), px, frame.bottom + 15, {
        size: 11.5,
        align: 'center',
        color: colors.tick,
      });
    }
    for (const e of yd) {
      if (e % yEvery) continue;
      drawMath(ctx, pow10Runs(e), frame.left - 6, y(10 ** e) + 4, {
        size: 11.5,
        align: 'right',
        color: colors.tick,
      });
    }
    const q = focusRun?.derivative ?? 1;
    // KaTeX sets f′ with the prime raised: draw it as a superscript, not on the baseline.
    const deriv = sup(q === 2 ? '\u200a′′' : '\u200a′');
    const mixed = new Set(runs.map((r) => r.derivative)).size > 1;
    const hasZero = runs.some((r) => errorsOf(r.trace).some((e) => e === 0));
    const yNameW = drawMath(
      ctx,
      [
        mm('|'),
        mv('D'),
        mm('('),
        mv('h'),
        mm(') − '),
        mv('f'),
        ...(mixed ? [sup('(q)')] : [deriv]),
        mm('('),
        mv('x'),
        sub('0'),
        mm(')|'),
        ...(mixed ? [sans('   each against its own derivative')] : []),
      ],
      frame.left + 2,
      frame.top - 11,
      { size: 12.5, color: colors.text2 },
    );
    // Key for the exact zeros (they have no place on a log axis).
    if (hasZero) {
      const key: MathRun[] = [sans('error exactly 0')];
      const kw = measureMath(ctx, key, 11.5);
      const left = frame.right - kw;
      if (left - 14 > frame.left + 2 + yNameW + 16) {
        drawMath(ctx, key, frame.right, frame.top - 11, {
          size: 11.5,
          align: 'right',
          color: colors.text3,
        });
        ctx.strokeStyle = colors.text2;
        ctx.lineWidth = 1.25;
        ctx.beginPath();
        ctx.moveTo(left - 9, frame.top - 11);
        ctx.lineTo(left - 12.5, frame.top - 17);
        ctx.lineTo(left - 5.5, frame.top - 17);
        ctx.closePath();
        ctx.stroke();
      }
    }
    ctx.restore();

    ctx.save();
    ctx.beginPath();
    ctx.rect(
      frame.left - 7,
      frame.top - 4,
      frame.right - frame.left + 14,
      frame.bottom - frame.top + 4 + 7,
    );
    ctx.clip();

    // ── ε floor ───────────────────────────────────────────────────────────────────────────
    if (floor !== null && floor > 0) {
      const py = crisp(y(floor), dpr);
      ctx.strokeStyle = colors.text3;
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(frame.left, py);
      ctx.lineTo(frame.right, py);
      ctx.stroke();
      ctx.setLineDash([]);
      drawMath(
        ctx,
        [mv('ε'), mm('|'), mv('f'), deriv, mm('('), mv('x'), sub('0'), mm(')|')],
        frame.left + 8,
        py - 6,
        { size: 12, color: colors.text3, halo: colors.halo },
      );
    }

    // ── Playhead: h(t) of the focused run ─────────────────────────────────────────────────
    if (focusRun) {
      const t = Math.max(0, Math.min(focusRun.t, focusRun.trace.length - 1));
      const k = Math.floor(t);
      const e = ease ? easeInOut(t - k) : t - k;
      const h0 = focusRun.trace[0]?.info.h as number;
      const px = crisp(x(h0 * 2 ** -(k + e)), dpr);
      ctx.strokeStyle = colors.playhead;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(px, frame.top);
      ctx.lineTo(px, frame.bottom);
      ctx.stroke();
    }

    // Screen points along every series polyline (labels keep clear of them).
    const dataPts: [number, number][] = [];
    for (const r of runs) {
      const es = errorsOf(r.trace);
      let prev: [number, number] | null = null;
      r.trace.forEach((st, k) => {
        const e = es[k];
        if (e === null || !(e > 0)) {
          prev = null;
          return;
        }
        const cur: [number, number] = [x(st.info.h as number), y(e)];
        if (prev)
          for (let j = 1; j < 4; j++)
            dataPts.push([
              prev[0] + ((cur[0] - prev[0]) * j) / 4,
              prev[1] + ((cur[1] - prev[1]) * j) / 4,
            ]);
        dataPts.push(cur);
        prev = cur;
      });
    }

    if (floor !== null && floor > 0)
      for (let px = frame.left; px <= frame.right; px += 12) dataPts.push([px, y(floor)]);

    // ── The focused method's error model ──────────────────────────────────────────────────
    if (focusRun && showModel) {
      const color = colors.series[focusRun.slot % colors.series.length];
      const model = (key: 'err_est' | 'roundoff', dash: number[], label: MathRun[]) => {
        const pts = focusRun.trace
          .map((s) => [s.info.h as number, s.info[key] as number | null] as const)
          .filter(([, v]) => v !== null && Number.isFinite(v) && (v as number) > 0) as [
          number,
          number,
        ][];
        if (pts.length < 2) return;
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.75;
        ctx.lineWidth = 1.25;
        ctx.setLineDash(dash);
        ctx.beginPath();
        pts.forEach(([h, v], i) => (i ? ctx.lineTo(x(h), y(v)) : ctx.moveTo(x(h), y(v))));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
        // Label at a point of the line inside the frame whose label box keeps the most room
        // from the data (ties: the far end for round-off, the start for truncation).
        const lw = measureMath(ctx, label, 11.5);
        const right = key === 'roundoff';
        let atX = NaN,
          atY = NaN;
        let room = -Infinity;
        // The label names the regime the line models: truncation for h ≥ h⋆, round-off for
        // h ≤ h⋆ (elsewhere the estimate is noise and follows no power of h).
        const hStar =
          focusRun.kBest !== null ? (focusRun.trace[focusRun.kBest]?.info.h as number) : null;
        const inRegime = pts.filter(([h]) =>
          hStar === null ? true : right ? h <= hStar * 1.0001 : h >= hStar * 0.9999,
        );
        const cand = inRegime.length >= 2 ? inRegime : pts;
        // Candidates: above the line (preferred) or below it, at every level.
        for (let c = 0; c < 2 * cand.length; c++) {
          const i = c % cand.length;
          const below = c >= cand.length;
          const [h, v] = cand[i];
          const lx = x(h) + (right ? -6 : 6);
          const ly = y(v) + (below ? 19 : -8);
          const x0b = right ? lx - lw : lx;
          const x1b = x0b + lw;
          if (x0b < frame.left + 4 || x1b > frame.right - 4) continue;
          if (ly - 10 < frame.top + 2 || ly + 3 > frame.bottom - 4) continue;
          let d = 60;
          for (const [px, py] of dataPts) {
            const dx = px < x0b ? x0b - px : px > x1b ? px - x1b : 0;
            const dy = py < ly - 10 ? ly - 10 - py : py > ly + 3 ? py - ly - 3 : 0;
            d = Math.min(d, Math.hypot(dx, dy));
          }
          // Prefer the usual end of the line when the room is about the same.
          const bias = (right ? i : cand.length - i) / cand.length - (below ? 1 : 0);
          if (d + bias > room) {
            room = d + bias;
            atX = lx;
            atY = ly;
          }
        }
        if (Number.isFinite(atX))
          drawMath(ctx, label, atX, atY, {
            size: 11.5,
            align: right ? 'right' : 'left',
            color: colors.text2,
            halo: colors.halo,
          });
        // The next label keeps clear of this line too.
        for (const [h, v] of pts) dataPts.push([x(h), y(v)]);
      };
      const p = focusRun.order;
      // Richardson's estimate is the change of the diagonal, |D(k, k) − D(k−1, k−1)|, which
      // falls like h^{2k+2} on smooth f: no single power describes it.
      model(
        'err_est',
        [1.5, 3],
        focusRun.id === 'richardson_extrapolation'
          ? [mm('est. truncation |Δ'), mv('D'), sub('kk'), mm('|')]
          : [mm('est. truncation ∝ '), mv('h'), ...(p > 1 ? [sup(String(p))] : [])],
      );
      if (focusRun.id !== 'complex_step')
        model(
          'roundoff',
          [7, 4],
          [mm('round-off model ∝ '), mv('ε'), mm('/'), mv('h'), ...(q === 2 ? [sup('2')] : [])],
        );
    }

    // ── The V of every method (focus last, on top) ────────────────────────────────────────
    const order = runs
      .map((_, i) => i)
      .sort((a, b) => (a === focus ? 1 : b === focus ? -1 : a - b));
    for (const i of order) {
      const r = runs[i];
      const color = colors.series[r.slot % colors.series.length];
      const strong = i === focus;
      const es = errorsOf(r.trace);
      const pt = (k: number): [number, number] | null => {
        const e = es[k];
        if (e === null || e === undefined) return null;
        return [x(r.trace[k].info.h as number), e > 0 ? y(e) : frame.bottom];
      };
      const t = Math.max(0, Math.min(r.t, r.trace.length - 1));
      const kNow = Math.floor(t);
      const u = ease ? easeInOut(t - kNow) : t - kNow;
      // Segments: past solid, future faint.
      ctx.lineJoin = 'round';
      for (let k = 1; k < r.trace.length; k++) {
        const a = pt(k - 1),
          b = pt(k);
        // An exact zero has no place on a log axis: no segment leads to its mark.
        if (!a || !b || es[k - 1] === 0 || es[k] === 0) continue;
        ctx.strokeStyle = color;
        ctx.lineWidth = strong ? 2 : 1.6;
        const future = k > kNow;
        ctx.globalAlpha = future ? 0.2 : strong ? 1 : 0.85;
        ctx.beginPath();
        ctx.moveTo(...a);
        ctx.lineTo(...b);
        ctx.stroke();
        if (future && k === kNow + 1 && u > 0) {
          ctx.globalAlpha = strong ? 1 : 0.85;
          ctx.beginPath();
          ctx.moveTo(...a);
          ctx.lineTo(a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u);
          ctx.stroke();
        }
      }
      // Level dots (exact zeros: hollow triangles on the bottom edge).
      for (let k = 0; k < r.trace.length; k++) {
        const p = pt(k);
        if (!p) continue;
        ctx.globalAlpha = k > kNow ? 0.25 : 1;
        if (es[k] === 0) {
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.25;
          ctx.beginPath();
          ctx.moveTo(p[0], p[1] - 1);
          ctx.lineTo(p[0] - 3.5, p[1] - 7);
          ctx.lineTo(p[0] + 3.5, p[1] - 7);
          ctx.closePath();
          ctx.stroke();
          continue;
        }
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(p[0], p[1], strong ? 2.4 : 2, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
      // Selected h⋆: ring (label for the focused method).
      if (r.kBest !== null) {
        const p = pt(r.kBest);
        if (p) {
          ctx.strokeStyle = colors.halo;
          ctx.lineWidth = 4;
          ctx.beginPath();
          ctx.arc(p[0], p[1], 7, 0, Math.PI * 2);
          ctx.stroke();
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.6;
          ctx.beginPath();
          ctx.arc(p[0], p[1], 7, 0, Math.PI * 2);
          ctx.stroke();
          if (strong)
            drawMath(ctx, [mv('h'), sup('⋆')], p[0], p[1] + 21, {
              size: 13,
              align: 'center',
              color: colors.text,
              halo: colors.halo,
            });
        }
      }
      // Head.
      const a = pt(kNow);
      if (a) {
        const b =
          kNow + 1 < r.trace.length && es[kNow] !== 0 && es[kNow + 1] !== 0 ? pt(kNow + 1) : null;
        const hx = b ? a[0] + (b[0] - a[0]) * u : a[0];
        const hy = b ? a[1] + (b[1] - a[1]) * u : a[1];
        ctx.fillStyle = colors.halo;
        ctx.beginPath();
        ctx.arc(hx, hy, 6.2, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(hx, hy, 4.4, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.restore();

    if (hover) {
      ctx.strokeStyle = colors.crosshair || colors.text3;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      const px = crisp(hover.px, dpr);
      ctx.moveTo(px, frame.top);
      ctx.lineTo(px, frame.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  });

  const levelAt = (clientX: number): Hover | null => {
    const el = box.current;
    if (!el || !focusRun) return null;
    const r = el.getBoundingClientRect();
    const { x } = scales(r.width, r.height);
    let best = -1,
      dist = Infinity;
    focusRun.trace.forEach((s, k) => {
      const d = Math.abs(x(s.info.h as number) - (clientX - r.left));
      if (d < dist) {
        dist = d;
        best = k;
      }
    });
    if (best < 0 || dist > 24) return null;
    return { k: best, px: x(focusRun.trace[best].info.h as number), width: r.width };
  };

  const onMove = (e: RPointerEvent<HTMLDivElement>) => setHover(levelAt(e.clientX));
  const onUp = (e: RPointerEvent<HTMLDivElement>) => {
    const hv = levelAt(e.clientX);
    if (hv && onSeek) onSeek(hv.k);
  };

  const tip = hover && focusRun?.trace[hover.k];
  return (
    <div
      ref={box}
      className={styles.canvasBox}
      role="img"
      aria-label={ariaLabel}
      style={onSeek ? { cursor: 'pointer' } : undefined}
      onPointerMove={onMove}
      onPointerLeave={() => setHover(null)}
      onPointerUp={onUp}
    >
      <canvas ref={canvasRef} />
      {tip && hover && (
        <div
          className={styles.tip}
          style={
            hover.px > hover.width / 2
              ? { right: hover.width - hover.px + 12 }
              : { left: hover.px + 12 }
          }
          aria-hidden="true"
        >
          <div className={styles.tipHead}>
            <i>h</i> = {sci(tip.info.h as number, 3)} · level {hover.k}
          </div>
          {runs.map((r) => {
            const s = r.trace[hover.k];
            const err = s ? (s.info.error as number | null) : null;
            return (
              <div key={r.id} className={styles.tipRow}>
                <span
                  className={styles.tipSwatch}
                  style={{ background: `var(--series-${(r.slot % 4) + 1})` }}
                />
                <span className={styles.tipName}>{r.name}</span>
                <span className={styles.tipValue}>
                  {err === null ? '—' : err === 0 ? '0' : sci(err, 2)}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
