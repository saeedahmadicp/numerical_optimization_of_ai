/**
 * The three terms of PDHG's relative KKT error against the iteration k, on a log axis: the primal
 * residual, the dual residual and the duality gap of the iterate (one method, so one color; the
 * terms differ by dash, named in the legend). The stopping tolerance ε is a horizontal rule, the
 * method stops when all three are below it (for the iterate or the average). Every restart is a
 * vertical hairline with the same square that marks it on the plane.
 */
import { useMemo, useRef, useState, type ReactNode } from 'react';
import type { Step } from '../../core/types';
import { useChartColors } from '../../ui/theme';
import { Formula } from '../../ui/components/Formula';
import { int, sci } from '../../core/format';
import { useCanvas } from '../../viz/useCanvas';
import { linearScale, logExtent, logScale, crisp } from '../../viz/scales';
import { drawAxes } from '../../viz/axes';
import { drawMath, m as mm, sans, sup, v as mv, type MathRun } from '../../viz/mathText';
import viz from '../../viz/viz.module.css';
import { drawRestartMark, restartKs } from './pdhgView';
import styles from './LPLab.module.css';

export interface ResidualRun {
  name: string;
  slot: number;
  trace: Step[];
  converged: boolean;
  tol: number;
}

interface Term {
  key: 'primal_residual' | 'dual_residual' | 'gap';
  name: string;
  tex: string;
  dash: number[];
}

const TERMS: Term[] = [
  { key: 'primal_residual', name: 'primal', tex: '\\|A\\mathbf z_k - \\mathbf b\\|', dash: [] },
  {
    key: 'dual_residual',
    name: 'dual',
    tex: '\\|(\\mathbf c - A^{\\top}\\mathbf y_k)_-\\|',
    dash: [7, 4],
  },
  {
    key: 'gap',
    name: 'gap',
    tex: '|\\mathbf c^{\\top}\\mathbf z_k - \\mathbf b^{\\top}\\mathbf y_k|',
    dash: [1.5, 3.5],
  },
];

const M = { left: 48, right: 14, top: 18, bottom: 32 };
const X_NAME: MathRun[] = [sans('iteration '), mv('k')];
const Y_NAME: MathRun[] = [sans('relative residual')];

/** A short line sample in the legend, dashed like its curve. */
function DashSample({ dash, slot }: { dash: number[]; slot: number }) {
  return (
    <svg className={styles.dashSample} width="22" height="8" aria-hidden="true">
      <line
        x1="1"
        y1="4"
        x2="21"
        y2="4"
        stroke={`var(--series-${(slot % 4) + 1})`}
        strokeWidth="2"
        strokeLinecap={dash.length && dash[0] < 2 ? 'round' : 'butt'}
        strokeDasharray={dash.length ? dash.join(' ') : undefined}
      />
    </svg>
  );
}

export function ResidualChart({
  run,
  t,
  onSeek,
}: {
  run: ResidualRun;
  /** Global playhead (continuous step index). */
  t: number;
  onSeek?: (k: number) => void;
}) {
  const colors = useChartColors();
  const box = useRef<HTMLDivElement>(null);
  const [hoverK, setHoverK] = useState<number | null>(null);
  const n = run.trace.length;
  const maxK = Math.max(1, n - 1);
  const restarts = useMemo(() => restartKs(run.trace), [run.trace]);
  const values = useMemo(
    () =>
      TERMS.map((term) =>
        run.trace.map((s) => {
          const v = s.info[term.key];
          return typeof v === 'number' && Number.isFinite(v) && v > 0 ? v : null;
        }),
      ),
    [run.trace],
  );
  const yDom = useMemo<[number, number]>(
    () => logExtent([...values.flat().filter((v): v is number => v !== null), run.tol]),
    [values, run.tol],
  );
  const scales = (w: number, h: number) => ({
    x: linearScale([0, maxK], [M.left, w - M.right]),
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
      xInteger: true,
      xName: X_NAME,
      yName: Y_NAME,
    });
    const color = colors.series[run.slot % colors.series.length];
    const kEnd = Math.min(t, n - 1);

    // Restarts: a hairline at each k; the reached ones carry the square.
    for (const k of restarts) {
      const px = crisp(x(k), s.dpr);
      ctx.save();
      ctx.strokeStyle = colors.text3;
      ctx.globalAlpha = k <= kEnd ? 0.75 : 0.3;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      ctx.moveTo(px, frame.top);
      ctx.lineTo(px, frame.bottom);
      ctx.stroke();
      ctx.restore();
      if (k <= kEnd) drawRestartMark(ctx, px, frame.top, color, colors.halo, 3.25);
    }

    // Tolerance ε.
    const ty = crisp(y(run.tol), s.dpr);
    ctx.save();
    ctx.strokeStyle = colors.text2;
    ctx.globalAlpha = 0.8;
    ctx.lineWidth = 1;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(frame.left, ty);
    ctx.lineTo(frame.right, ty);
    ctx.stroke();
    ctx.restore();
    const e = Math.round(Math.log10(run.tol));
    const epsRuns: MathRun[] =
      Math.abs(run.tol - 10 ** e) <= 1e-9 * run.tol
        ? [mv('ε'), mm(' = 10'), sup(String(e).replace('-', '−'))]
        : [mv('ε'), mm(` = ${sci(run.tol, 2)}`)];
    // Left end: the residuals start far above ε, so the label is clear of the curves there.
    drawMath(ctx, epsRuns, frame.left + 6, ty - 5, {
      size: 12,
      align: 'left',
      color: colors.text2,
      halo: colors.halo,
    });

    // The three terms: faint after the playhead, full before it.
    ctx.save();
    ctx.beginPath();
    ctx.rect(
      frame.left - 4,
      frame.top - 4,
      frame.right - frame.left + 8,
      frame.bottom - frame.top + 8,
    );
    ctx.clip();
    ctx.lineJoin = 'round';
    for (const pass of ['future', 'past'] as const) {
      TERMS.forEach((term, q) => {
        const vals = values[q];
        const from = pass === 'past' ? 0 : Math.floor(kEnd);
        const to = pass === 'past' ? Math.floor(kEnd) : n - 1;
        const path = () => {
          ctx.beginPath();
          let pen = false;
          let lastPx = -Infinity;
          for (let k = from; k <= to; k++) {
            const v = vals[k];
            if (v === null) {
              pen = false;
              continue;
            }
            const px = x(k),
              py = y(v);
            if (pen && px - lastPx < 0.75 && k !== to) continue;
            if (pen) ctx.lineTo(px, py);
            else ctx.moveTo(px, py);
            lastPx = px;
            pen = true;
          }
        };
        ctx.setLineDash(term.dash);
        ctx.lineCap = term.dash.length && term.dash[0] < 2 ? 'round' : 'butt';
        if (pass === 'past') {
          path();
          ctx.strokeStyle = colors.halo;
          ctx.globalAlpha = 0.85;
          ctx.lineWidth = 4;
          ctx.setLineDash([]);
          ctx.stroke();
          ctx.setLineDash(term.dash);
        }
        path();
        ctx.strokeStyle = color;
        ctx.globalAlpha = pass === 'future' ? 0.18 : 1;
        ctx.lineWidth = pass === 'future' ? 1.25 : 1.75;
        ctx.stroke();
      });
    }
    ctx.restore();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;

    // Playhead.
    if (Number.isFinite(t)) {
      const px = crisp(x(Math.min(t, maxK)), s.dpr);
      ctx.strokeStyle = colors.playhead;
      ctx.lineWidth = 1 / s.dpr;
      ctx.beginPath();
      ctx.moveTo(px, frame.top - 4);
      ctx.lineTo(px, frame.bottom);
      ctx.stroke();
    }
    if (hoverK !== null) {
      const hx = crisp(x(hoverK), s.dpr);
      ctx.strokeStyle = colors.crosshair;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(hx, frame.top);
      ctx.lineTo(hx, frame.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  });

  const kFromEvent = (e: { clientX: number }) => {
    const r = box.current?.getBoundingClientRect();
    if (!r) return null;
    const k = Math.round(scales(r.width, r.height).x.invert(e.clientX - r.left));
    return Math.max(0, Math.min(maxK, k));
  };
  const tipLeft = hoverK !== null && size.width ? scales(size.width, size.height).x(hoverK) : 0;
  const last = run.trace[n - 1]?.info ?? {};
  const fmtV = (v: unknown) => sci(typeof v === 'number' ? v : null, 2);
  const legend: ReactNode = (
    <div className={`${viz.legend} ${styles.residualLegend}`}>
      {TERMS.map((term) => (
        <span key={term.key} className={viz.legendItem}>
          <DashSample dash={term.dash} slot={run.slot} />
          {term.name} <Formula tex={term.tex} />
        </span>
      ))}
      <span className={viz.legendItem}>
        <span
          className={styles.restartSwatch}
          style={{ ['--_c' as string]: `var(--series-${(run.slot % 4) + 1})` }}
          aria-hidden="true"
        />
        restart
      </span>
    </div>
  );

  return (
    <div className={styles.residualChart}>
      {legend}
      <div
        ref={box}
        className={viz.chart}
        style={{ flex: 1, cursor: onSeek ? 'pointer' : undefined }}
        role="img"
        aria-label={
          `Relative primal residual, dual residual and duality gap of ${run.name} by iteration, ` +
          `with ${int(restarts.length)} restarts and the tolerance ${sci(run.tol, 2)}. ` +
          `At k = ${int(n - 1)}: primal ${fmtV(last.primal_residual)}, dual ${fmtV(last.dual_residual)}, gap ${fmtV(last.gap)}.`
        }
        onPointerMove={(e) => setHoverK(kFromEvent(e))}
        onPointerLeave={() => setHoverK(null)}
        onClick={(e) => {
          const k = kFromEvent(e);
          if (k !== null) onSeek?.(k);
        }}
      >
        <canvas ref={canvasRef} />
        {hoverK !== null && (
          <div
            className={viz.tooltip}
            style={{
              left: Math.min(Math.max(8, tipLeft + 12), Math.max(8, size.width - 170)),
              top: 8,
            }}
          >
            <div className={viz.tooltipTitle}>
              k = {int(hoverK)}
              {restarts.includes(hoverK) ? ' · restart' : ''}
            </div>
            {TERMS.map((term) => (
              <div key={term.key} className={viz.tooltipRow}>
                <span className={viz.tooltipKey}>
                  <DashSample dash={term.dash} slot={run.slot} />
                  {term.name}
                </span>
                <b>{fmtV(run.trace[hoverK]?.info[term.key])}</b>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
