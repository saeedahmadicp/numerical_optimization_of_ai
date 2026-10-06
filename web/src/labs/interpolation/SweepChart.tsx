/**
 * The Runge figure: max |p − f| against the number of nodes n = 3 … 40, one curve per selected
 * method, on a log axis. A rule marks the n on the stage; hover reads the values at an n and a
 * click resamples f at that n. Chebyshev interpolation that resamples f at its own Chebyshev
 * nodes is drawn dashed and labelled so, since it does not use the equispaced nodes of the axis.
 */
import { memo, useMemo, useRef, useState } from 'react';
import { sci } from '../../core/format';
import { Swatch } from '../../ui/components';
import { useChartColors } from '../../ui/theme';
import {
  crisp,
  drawAxes,
  drawMath,
  linearScale,
  logExtent,
  logScale,
  mathMain as mm,
  mathSans,
  mathVar as mv,
  useCanvas,
  type Scale,
} from '../../viz';
import styles from './InterpolationLab.module.css';

export interface SweepLine {
  id: string;
  slot: number;
  label: string;
  x: readonly number[];
  values: readonly (number | null)[];
  /** The method samples f at its own Chebyshev nodes, not at the nodes on the x-axis. */
  ownNodes?: boolean;
}

export interface SweepChartProps {
  series: readonly SweepLine[];
  layout: 'equi' | 'cheb';
  /** The n on the stage (marked by a rule), or null when the stage shows other nodes. */
  current: number | null;
  onPick: (n: number) => void;
  ariaLabel: string;
}

const M = { left: 48, right: 14, top: 24, bottom: 40 };
const DASH = [5, 3];

/** A copy of `s` whose ticks are fixed values (the integer node counts the study uses). */
function withTicks(s: Scale, ticks: number[]): Scale {
  const out = Object.assign((v: number) => s(v), s) as Scale;
  out.ticks = () => ticks;
  return out;
}

function SweepChartImpl({ series, layout, current, onPick, ariaLabel }: SweepChartProps) {
  const colors = useChartColors();
  const box = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  const ns = series[0]?.x ?? [];
  const nLo = ns.length ? ns[0] : 3,
    nHi = ns.length ? ns[ns.length - 1] : 40;
  const ticks = useMemo(() => {
    const t = [nLo];
    for (let v = Math.ceil((nLo + 1) / 10) * 10; v <= nHi; v += 10) t.push(v);
    return t;
  }, [nLo, nHi]);
  const yDom = useMemo(
    () =>
      logExtent(series.flatMap((s) => s.values.filter((v): v is number => v !== null && v > 0))),
    [series],
  );

  const scales = (w: number, h: number) => ({
    x: withTicks(linearScale([nLo - 1, nHi + 1], [M.left, w - M.right]), ticks),
    y: logScale(yDom, [h - M.bottom, M.top]),
  });

  const { canvasRef, size } = useCanvas((ctx, s) => {
    const { x, y } = scales(s.width, s.height);
    const frame = {
      left: M.left,
      top: M.top,
      right: s.width - M.right,
      bottom: s.height - M.bottom,
    };
    drawAxes(ctx, {
      x,
      y,
      frame,
      colors,
      dpr: s.dpr,
      yName: [mm('max |'), mv('p'), mm(' − '), mv('f'), mm('|')],
    });
    // The axis name under the tick row, centred (the ticks keep the full width), at the size
    // and color drawAxes gives an axis name.
    drawMath(
      ctx,
      [mathSans(layout === 'cheb' ? 'Chebyshev nodes ' : 'equispaced nodes '), mv('n')],
      (frame.left + frame.right) / 2,
      frame.bottom + 33,
      { size: 13, align: 'center', color: colors.text2 },
    );
    const ok = (v: number | null | undefined): v is number =>
      v !== null && v !== undefined && v > 0 && Number.isFinite(v);
    const clampY = (v: number) => Math.max(frame.top - 4, Math.min(frame.bottom, y(v)));

    // The n on the stage.
    if (current !== null && current >= nLo && current <= nHi) {
      const px = crisp(x(current), s.dpr);
      ctx.save();
      ctx.strokeStyle = colors.playhead;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(px, frame.top);
      ctx.lineTo(px, frame.bottom);
      ctx.stroke();
      ctx.restore();
      const right = px > frame.right - 60;
      drawMath(
        ctx,
        [mv('n'), { t: ` = ${current}`, style: 'main' }],
        right ? px - 5 : px + 5,
        frame.top + 12,
        {
          size: 12,
          align: right ? 'right' : 'left',
          color: colors.text2,
          halo: colors.halo,
        },
      );
    }

    ctx.save();
    ctx.beginPath();
    ctx.rect(frame.left, frame.top - 4, frame.right - frame.left, frame.bottom - frame.top + 4);
    ctx.clip();
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    for (const sr of series) {
      const color = colors.series[sr.slot % colors.series.length];
      const trace = () => {
        ctx.beginPath();
        let pen = false;
        sr.values.forEach((v, i) => {
          if (!ok(v)) {
            pen = false;
            return;
          }
          const [px, py] = [x(sr.x[i]), clampY(v)];
          if (pen) ctx.lineTo(px, py);
          else ctx.moveTo(px, py);
          pen = true;
        });
      };
      trace();
      ctx.setLineDash([]);
      ctx.strokeStyle = colors.halo;
      ctx.lineWidth = 4.2;
      ctx.globalAlpha = 0.85;
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = color;
      ctx.lineWidth = sr.ownNodes ? 1.6 : 2;
      ctx.setLineDash(sr.ownNodes ? DASH : []);
      ctx.stroke();
      ctx.setLineDash([]);
      // Values at the n on the stage and under the pointer.
      for (const at of [current, hover]) {
        if (at === null) continue;
        const i = sr.x.indexOf(at);
        const v = sr.values[i];
        if (i < 0 || !ok(v)) continue;
        ctx.beginPath();
        ctx.arc(x(at), clampY(v), 4.4, 0, Math.PI * 2);
        ctx.fillStyle = colors.halo;
        ctx.fill();
        ctx.beginPath();
        ctx.arc(x(at), clampY(v), 3, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      }
    }
    ctx.restore();

    if (hover !== null && hover !== current) {
      const hx = crisp(x(hover), s.dpr);
      ctx.save();
      ctx.strokeStyle = colors.crosshair;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(hx, frame.top);
      ctx.lineTo(hx, frame.bottom);
      ctx.stroke();
      ctx.restore();
    }
  });

  const nAt = (clientX: number): number | null => {
    const r = box.current?.getBoundingClientRect();
    if (!r || !ns.length) return null;
    const { x } = scales(r.width, r.height);
    const v = Math.round(x.invert(clientX - r.left));
    return Math.max(nLo, Math.min(nHi, v));
  };

  const tipLeft = hover !== null && size.width ? scales(size.width, size.height).x(hover) : 0;
  const tipRight = tipLeft > size.width / 2;
  return (
    <div className={styles.sweep}>
      <div className={styles.legend}>
        {series.map((s) => (
          <span key={s.id} className={styles.legendItem}>
            <span
              className={styles.legendLine}
              data-dashed={s.ownNodes || undefined}
              style={{ ['--_c' as string]: `var(--series-${(s.slot % 4) + 1})` }}
            />
            {s.label}
            {s.ownNodes && <span className={styles.legendNote}>own Chebyshev nodes</span>}
          </span>
        ))}
      </div>
      <div
        ref={box}
        className={styles.sweepChart}
        role="img"
        aria-label={ariaLabel}
        onPointerMove={(e) => setHover(nAt(e.clientX))}
        onPointerLeave={() => setHover(null)}
        onClick={(e) => {
          const n = nAt(e.clientX);
          if (n !== null) onPick(n);
        }}
      >
        <canvas ref={canvasRef} />
        {hover !== null && (
          <div
            className={styles.tooltip}
            // In the half of the chart away from the pointer, so it never covers the n it reads.
            style={tipRight ? { left: M.left + 4, top: 4 } : { right: M.right + 2, top: 4 }}
          >
            <div className={styles.tooltipTitle}>
              n = {hover} {layout === 'cheb' ? 'Chebyshev' : 'equispaced'} nodes
            </div>
            {series.map((s) => {
              const i = s.x.indexOf(hover);
              return (
                <div key={s.id} className={styles.tooltipRow}>
                  <span className={styles.tooltipKey}>
                    <Swatch slot={s.slot} size={8} />
                    {s.label}
                  </span>
                  <b>{i >= 0 ? sci(s.values[i] ?? null, 3) : '—'}</b>
                </div>
              );
            })}
            {hover !== current && (
              <div className={styles.tooltipHint}>Click to resample f at n = {hover}</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/** Redraws only when its inputs change (not on every playback frame of the lab). */
export const SweepChart = memo(SweepChartImpl);
