/**
 * The iterates on a number line of the signed error xₖ − x⋆, on a logarithmic scale that is
 * symmetric about the root: one tick per decade, so every tick crossed is one more correct
 * digit. Bisection walks in equal strides (one bit per step: 0.3 decade), Newton's strides
 * double, Halley's triple. Bracketing methods also show their bracket, which straddles x⋆.
 * Brent, Chandrupatla and ITP are placed at their estimate (`info.best`, see estimate.ts), not
 * at the trial point, which ends as a probe a tolerance away from the answer.
 */
import { useMemo } from 'react';
import type { Step } from '../../core/types';
import { useChartColors } from '../../ui/theme';
import {
  drawMath,
  mathMain,
  mathSub,
  mathSup,
  mathVar,
  measureMath,
  pow10Runs,
  useCanvas,
} from '../../viz';
import { LABEL_SIZE, TICK_SIZE, labelFont, tickFont } from '../../viz/axes';
import { estimateOf, phase } from './estimate';
import styles from './RootsLab.module.css';

export interface LineRun {
  name: string;
  slot: number;
  trace: readonly Step[];
  t: number;
  target: number | null;
  focused: boolean;
}

const LANE = 22;
const AXIS = 24;
/** The lane-name gutter grows with the longest name, from MIN_LEFT up to MAX_LEFT. */
const MIN_LEFT = 72;
const MAX_LEFT = 140;
const RIGHT = 14;
/** Room left of a name: the swatch dot and its gap (17 px) plus a gap before the axis. */
const NAME_PAD = 25;
/** The least gap between two decade labels, and between a label and x⋆. */
const LABEL_GAP = 8;

/** The display name of a lane: without its parenthetical ("Brent (zeroin)" → "Brent"). */
const laneName = (name: string) => name.replace(/\s*\(.*\)\s*$/, '');
/** The short form when the gutter cannot hold the name: its first word ("Newton–Raphson" → "Newton"). */
const shortName = (name: string) => laneName(name).split(/[\s–-]/)[0];

/** Decade range [lo, hi] (log10 |xₖ − x⋆|) shown by the line. */
function decadeRange(runs: readonly LineRun[]): [number, number] {
  let lo = Infinity,
    hi = -Infinity;
  for (const r of runs) {
    if (r.target === null) continue;
    for (const s of r.trace) {
      const d = Math.abs(estimateOf(s) - r.target);
      if (!(d > 0) || !Number.isFinite(d)) continue;
      const e = Math.log10(d);
      lo = Math.min(lo, e);
      hi = Math.max(hi, e);
    }
    const br = r.trace[0]?.info.bracket as number[] | undefined;
    if (br) for (const e of br) hi = Math.max(hi, Math.log10(Math.abs(e - r.target) || 1e-300));
  }
  if (!Number.isFinite(lo)) return [-16, 1];
  const top = Math.min(Math.ceil(hi), 4);
  const bottom = Math.max(Math.floor(lo), top - 17, -17);
  return [Math.min(bottom, top - 2), top];
}

export function NumberLine({ runs }: { runs: readonly LineRun[] }) {
  const colors = useChartColors();
  const [lo, hi] = useMemo(() => decadeRange(runs), [runs]);
  const height = AXIS + runs.length * LANE + 6;

  const { canvasRef } = useCanvas((ctx, s) => {
    // The gutter fits the longest name (semibold, as when focused), within [MIN_LEFT, MAX_LEFT]
    // and at most 30 % of the width; a name that still does not fit falls back to its first word.
    ctx.font = labelFont(colors).replace(/^500/, '600');
    const longest = Math.max(0, ...runs.map((r) => ctx.measureText(laneName(r.name)).width));
    const left = Math.min(
      Math.max(MIN_LEFT, Math.ceil(longest) + NAME_PAD),
      MAX_LEFT,
      s.width * 0.3,
    );
    const x0 = left,
      x1 = s.width - RIGHT;
    const mid = (x0 + x1) / 2;
    const half = (x1 - x0) / 2;
    /** Signed error → px. Errors below 10^lo collapse onto the root. */
    const pos = (d: number) => {
      if (!Number.isFinite(d)) return d > 0 ? x1 + 6 : x0 - 6;
      const a = Math.abs(d);
      const u = a <= 10 ** lo ? 0 : Math.min(1.04, (Math.log10(a) - lo) / (hi - lo));
      return mid + Math.sign(d) * u * half;
    };
    // Axis: decades on both sides, labelled on the right; x⋆ at the center. The label stride
    // (1, 2, 4, … decades, so 10⁰ stays labelled) is the smallest that leaves LABEL_GAP between
    // the widest labels at this width: 10⁻¹² and 10⁻⁸ never touch at phone width.
    const tickSize = TICK_SIZE + 1;
    const starRuns = [mathVar('x'), mathSup('⋆')];
    let labelW = 0;
    for (let e = Math.ceil(lo); e <= hi; e++)
      labelW = Math.max(labelW, measureMath(ctx, pow10Runs(e), tickSize));
    const pxPerDecade = half / (hi - lo);
    let every = 1;
    while (pxPerDecade * every < labelW + LABEL_GAP && every < 64) every *= 2;
    /** The x⋆ label's right edge: a decade label left of it would overprint it. */
    const starRight = mid + measureMath(ctx, starRuns, 12) / 2 + LABEL_GAP;
    ctx.save();
    ctx.font = tickFont(colors);
    ctx.strokeStyle = colors.grid;
    ctx.lineWidth = 1 / s.dpr;
    const top = AXIS - 4,
      bottom = height - 4;
    for (let e = Math.ceil(lo); e <= hi; e++) {
      for (const sgn of [-1, 1]) {
        const px = Math.round(pos(sgn * 10 ** e) * s.dpr) / s.dpr + 0.5 / s.dpr;
        ctx.globalAlpha = e % every === 0 ? 1 : 0.45;
        ctx.beginPath();
        ctx.moveTo(px, top);
        ctx.lineTo(px, bottom);
        ctx.stroke();
      }
      const runsE = pow10Runs(e);
      if (
        e % every === 0 &&
        e > lo &&
        pos(10 ** e) - measureMath(ctx, runsE, tickSize) / 2 >= starRight
      )
        drawMath(ctx, runsE, pos(10 ** e), AXIS - 9, {
          size: tickSize,
          align: 'center',
          color: colors.tick,
        });
    }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = colors.axis;
    ctx.beginPath();
    ctx.moveTo(Math.round(mid) + 0.5, top - 2);
    ctx.lineTo(Math.round(mid) + 0.5, bottom);
    ctx.stroke();
    drawMath(ctx, starRuns, mid, AXIS - 9, {
      size: 12,
      align: 'center',
      color: colors.text2,
    });
    drawMath(
      ctx,
      [mathVar('x'), mathSub('k'), mathMain(' < '), mathVar('x'), mathSup('⋆')],
      x0,
      AXIS - 9,
      {
        size: LABEL_SIZE,
        color: colors.text3,
      },
    );
    ctx.restore();

    runs.forEach((r, i) => {
      const cy = AXIS + i * LANE + LANE / 2;
      const color = colors.series[r.slot % colors.series.length];
      // Lane name (identity is never color alone).
      ctx.save();
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(8, cy, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.font = r.focused ? labelFont(colors).replace(/^500/, '600') : labelFont(colors);
      ctx.fillStyle = r.focused ? colors.text : colors.text2;
      ctx.textBaseline = 'middle';
      const room = left - NAME_PAD;
      let name = laneName(r.name);
      if (ctx.measureText(name).width > room) name = shortName(r.name);
      while (ctx.measureText(name).width > room && name.length > 4) name = name.slice(0, -2) + '…';
      ctx.fillText(name, 17, cy);
      ctx.strokeStyle = colors.grid;
      ctx.lineWidth = 1 / s.dpr;
      ctx.beginPath();
      ctx.moveTo(x0, cy);
      ctx.lineTo(x1, cy);
      ctx.stroke();
      ctx.restore();
      if (r.target === null || r.trace.length === 0) return;
      const n = r.trace.length;
      const t = Math.min(r.t, n - 1);
      const k = Math.floor(t);
      const u = t - k;
      const err = (st: Step) => estimateOf(st) - (r.target as number);
      const { fade, travel } = k < n - 1 ? phase(u) : { fade: 0, travel: 0 };
      ctx.save();
      // Bracket band (it straddles x⋆ while the root is inside).
      const br = (fade >= 0.5 ? r.trace[k + 1] : r.trace[k]).info.bracket as number[] | undefined;
      if (br) {
        const a = pos(Math.min(...br) - r.target),
          b = pos(Math.max(...br) - r.target);
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.16;
        ctx.fillRect(Math.min(a, b), cy - 6, Math.max(2, Math.abs(b - a)), 12);
        ctx.globalAlpha = 0.6;
        ctx.fillRect(Math.min(a, b), cy - 6, 1.5, 12);
        ctx.fillRect(Math.max(a, b) - 1.5, cy - 6, 1.5, 12);
      }
      for (let j = 0; j <= k; j++) {
        ctx.globalAlpha = Math.max(0.2, 0.85 - (k - j) * 0.1);
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(pos(err(r.trace[j])), cy, 2.2, 0, Math.PI * 2);
        ctx.fill();
      }
      let hx = pos(err(r.trace[k]));
      if (travel > 0) hx = hx + (pos(err(r.trace[k + 1])) - hx) * travel;
      ctx.globalAlpha = 1;
      ctx.fillStyle = colors.halo;
      ctx.beginPath();
      ctx.arc(hx, cy, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(hx, cy, 4.25, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    });
  });

  const label = runs
    .map((r) => {
      const st = r.trace[Math.min(r.trace.length - 1, Math.floor(r.t))];
      if (!st || r.target === null) return `${r.name}: no root`;
      const e = Math.abs(estimateOf(st) - r.target);
      return `${r.name}: |x${st.k} − x⋆| = ${e === 0 ? '0' : e.toExponential(1)}`;
    })
    .join('; ');

  return (
    <div className={styles.line} style={{ height }}>
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={`Signed distance of each iterate to its root on a log scale (one tick per decade). ${label}.`}
      />
    </div>
  );
}
