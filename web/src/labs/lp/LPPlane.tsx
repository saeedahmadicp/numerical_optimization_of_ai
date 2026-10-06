/**
 * The 2-D picture of a linear program: the feasible polygon filled with the level sets of the
 * objective, the constraint lines, the vertices, the objective level line through each current
 * iterate, and — for the focused method — the geometry of the step: the simplex edge with its
 * ratio-test points, the predictor and the central path, the Dikin ellipse, the branch-and-bound
 * box and branching lines, the Gomory cuts, the restarts of PDHG (squares) with its epoch average.
 * The objective direction is a draggable handle.
 */
import { useMemo, useRef, type KeyboardEvent, type PointerEvent } from 'react';
import type { LinearProgram, Step } from '../../core/types';
import { useChartColors } from '../../ui/theme';
import { CONTOUR_MAPS, sample, type ChartColors } from '../../ui/colors';
import { useCanvas } from '../../viz/useCanvas';
import { linearScale } from '../../viz/scales';
import { drawAxes } from '../../viz/axes';
import { drawPathLayer, overlappingPaths, pathPosition, type PathSpec } from '../../viz/PathLayer';
import { drawOverlays2D } from '../../viz/overlays2d';
import {
  drawMath,
  measureMath,
  m as mm,
  v as mv,
  b as mb,
  sub,
  sup,
  type MathRun,
} from '../../viz/mathText';
import {
  boundSpaces,
  centralPath,
  clipPolygon,
  edgeDirection,
  feasiblePolygon,
  halfSpaces,
  interiorPoint,
  latticePoints,
  lineInBox,
  minCost,
  objectiveRows,
  vertices2D,
  type HalfSpace,
  type P2,
  type TableauInfo,
} from './geometry';
import { boundaryOf, fmt, INTERIOR_METHODS } from './explain';
import { affineScalingMetric } from './affineMetric';
import { PDHG, drawRestartMark } from './pdhgView';
import styles from './LPLab.module.css';

export interface PlaneRun {
  id: string;
  slot: number;
  name: string;
  trace: Step[];
  converged: boolean;
}

export interface LPPlaneProps {
  lp: LinearProgram;
  domain: [[number, number], [number, number]];
  runs: PlaneRun[];
  /** Global playhead. */
  t: number;
  /** Current step of run i. */
  localK: (i: number) => number;
  ease: boolean;
  /** Index (into `runs`) of the method whose step geometry is drawn. */
  focus: number;
  optimum: number[] | null;
  optimalValue: number | null;
  integer: boolean;
  onObjectiveChange?: (c: number[]) => void;
  ariaLabel: string;
}

type Box = [[number, number], [number, number]];

/** Equal-aspect view of `d` in a w × h canvas, with a margin. */
function fitView(d: Box, w: number, h: number): Box {
  const pad = 0.07;
  const dx = d[0][1] - d[0][0],
    dy = d[1][1] - d[1][0];
  let x0 = d[0][0] - pad * dx,
    x1 = d[0][1] + pad * dx * 0.6,
    y0 = d[1][0] - pad * dy,
    y1 = d[1][1] + pad * dy * 0.6;
  const target = w / h;
  // Equal scale on both axes; the extra room is shared around the problem's box.
  if ((x1 - x0) / (y1 - y0) > target) {
    const extra = (x1 - x0) / target - (y1 - y0);
    y0 -= extra * 0.35;
    y1 += extra * 0.65;
  } else {
    const extra = (y1 - y0) * target - (x1 - x0);
    x0 -= extra * 0.3;
    x1 += extra * 0.7;
  }
  return [
    [x0, x1],
    [y0, y1],
  ];
}

/** `3x₁ + 2x₂ ≤ 18` as math runs. */
function linearRuns(a: readonly number[], rel: string, b: number): MathRun[] {
  const runs: MathRun[] = [];
  a.forEach((v, j) => {
    if (Math.abs(v) < 1e-12) return;
    const mag = Math.abs(v);
    const first = runs.length === 0;
    const sign = v < 0 ? (first ? '−' : ' − ') : first ? '' : ' + ';
    runs.push(
      mm(sign + (Math.abs(mag - 1) < 1e-12 ? '' : fmt(mag, 4))),
      mv('x'),
      sub(String(j + 1)),
    );
  });
  runs.push(mm(` ${rel} ${fmt(b, 4)}`));
  return runs;
}

const ROUND = (v: number) => Math.round(v * 100) / 100;

function arrow(ctx: CanvasRenderingContext2D, a: P2, b: P2, head = 7) {
  const ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
  ctx.beginPath();
  ctx.moveTo(a[0], a[1]);
  ctx.lineTo(b[0], b[1]);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(b[0], b[1]);
  ctx.lineTo(b[0] - head * Math.cos(ang - 0.42), b[1] - head * Math.sin(ang - 0.42));
  ctx.lineTo(b[0] - head * Math.cos(ang + 0.42), b[1] - head * Math.sin(ang + 0.42));
  ctx.closePath();
  ctx.fill();
}

function polyPath(
  ctx: CanvasRenderingContext2D,
  pts: readonly P2[],
  toPx: (x: number, y: number) => P2,
) {
  pts.forEach((p, i) => {
    const [px, py] = toPx(p[0], p[1]);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.closePath();
}

/** Diagonal hatch over the current clip. */
function hatch(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  color: string,
  gap = 7,
  alpha = 0.16,
) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.globalAlpha = alpha;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let s = -h; s < w + h; s += gap) {
    ctx.moveTo(s, h);
    ctx.lineTo(s + h, 0);
  }
  ctx.stroke();
  ctx.restore();
}

/** Improving direction of the objective (c for max, −c for min), unit length. */
function improving(lp: LinearProgram): P2 {
  const d = minCost(lp).map((v) => -v);
  const n = Math.hypot(d[0], d[1]) || 1;
  return [d[0] / n, d[1] / n];
}

export function LPPlane({
  lp,
  domain,
  runs,
  t,
  localK,
  ease,
  focus,
  optimum,
  optimalValue,
  integer,
  onObjectiveChange,
  ariaLabel,
}: LPPlaneProps) {
  const colors = useChartColors();
  const hs = useMemo(() => halfSpaces(lp), [lp]);
  const span = Math.max(domain[0][1] - domain[0][0], domain[1][1] - domain[1][0]);
  // The feasible set inside a box far larger than the view, so a polygon edge in view is a constraint.
  const bigBox: Box = useMemo(
    () => [
      [domain[0][0] - 60 * span, domain[0][1] + 60 * span],
      [domain[1][0] - 60 * span, domain[1][1] + 60 * span],
    ],
    [domain, span],
  );
  const poly = useMemo(() => feasiblePolygon(hs, bigBox), [hs, bigBox]);
  const bounded = useMemo(
    () =>
      poly.length > 0 &&
      poly.every((p) => Math.abs(p[0]) < 30 * span && Math.abs(p[1]) < 30 * span),
    [poly, span],
  );
  const verts = useMemo(() => vertices2D(hs), [hs]);
  const lattice = useMemo(
    () => (integer ? latticePoints(hs, domain) : null),
    [hs, domain, integer],
  );
  const showCentral = runs.some((r) => INTERIOR_METHODS.has(r.id));
  const central = useMemo(() => {
    if (!showCentral || !bounded) return [];
    const ip = interiorPoint(poly, hs);
    return ip ? centralPath(hs, minCost(lp), ip) : [];
  }, [showCentral, bounded, poly, hs, lp]);
  const dir = improving(lp);
  // A path that runs along another one is drawn on top, dashed, and the stage says so.
  const overlaps = useMemo(
    () =>
      overlappingPaths(
        runs.map((r) =>
          r.id === 'branch_and_bound'
            ? []
            : r.trace.map((st) => st.x as unknown as [number, number]),
        ),
        span * 0.01,
      ),
    [runs, span],
  );
  const dashed = useMemo(() => new Set(overlaps.map(([i]) => i)), [overlaps]);

  const { canvasRef, size } = useCanvas((ctx, { width: W, height: H, dpr }) => {
    if (W < 10 || H < 10) return;
    const view = fitView(domain, W, H);
    const X = linearScale(view[0], [0, W]);
    const Y = linearScale(view[1], [H, 0]);
    const toPx = (x: number, y: number): P2 => [X(x), Y(y)];
    const viewBox: Box = view;
    drawScene(ctx, {
      W,
      H,
      dpr,
      colors,
      lp,
      hs,
      poly,
      verts,
      lattice,
      central,
      runs,
      t,
      localK,
      ease,
      focus,
      optimum,
      optimalValue,
      toPx,
      X,
      Y,
      view: viewBox,
      dashed,
    });
    // Objective direction handle (drawn; the button over it takes input).
    const base = handleBase(W);
    const tip: P2 = [base[0] + dir[0] * HANDLE_LEN, base[1] - dir[1] * HANDLE_LEN];
    ctx.save();
    ctx.strokeStyle = colors.halo;
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.moveTo(base[0], base[1]);
    ctx.lineTo(tip[0], tip[1]);
    ctx.stroke();
    ctx.strokeStyle = colors.text;
    ctx.fillStyle = colors.text;
    ctx.lineWidth = 1.75;
    arrow(ctx, base, tip, 8);
    ctx.beginPath();
    ctx.arc(base[0], base[1], 2.5, 0, Math.PI * 2);
    ctx.fill();
    const lab: MathRun[] = lp.sense === 'max' ? [mb('c')] : [mm('−'), mb('c')];
    const off = 14;
    drawMath(ctx, lab, tip[0] + dir[0] * off, tip[1] - dir[1] * off + 4, {
      size: 14,
      align: 'center',
      color: colors.text,
      halo: colors.halo,
    });
    ctx.restore();
  });

  // ── Objective handle: drag or arrow keys rotate c (|c| kept). ──
  const dragging = useRef(false);
  const W = size.width;
  const base = handleBase(W);
  const tip: P2 = [base[0] + dir[0] * HANDLE_LEN, base[1] - dir[1] * HANDLE_LEN];
  const setAngle = (ang: number) => {
    if (!onObjectiveChange) return;
    const r = Math.hypot(lp.c[0], lp.c[1]) || 1;
    const d: P2 = [Math.cos(ang), Math.sin(ang)];
    const s = lp.sense === 'max' ? 1 : -1;
    onObjectiveChange([ROUND(s * r * d[0]), ROUND(s * r * d[1])]);
  };
  const angle = Math.atan2(dir[1], dir[0]);
  const onPointerDown = (e: PointerEvent<HTMLButtonElement>) => {
    dragging.current = true;
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: PointerEvent<HTMLButtonElement>) => {
    if (!dragging.current) return;
    const rect = (e.currentTarget.parentElement as HTMLElement).getBoundingClientRect();
    const px = e.clientX - rect.left - base[0],
      py = -(e.clientY - rect.top - base[1]);
    if (Math.hypot(px, py) < 6) return;
    const deg = Math.round((Math.atan2(py, px) * 180) / Math.PI);
    setAngle((deg * Math.PI) / 180);
  };
  const onPointerUp = () => {
    dragging.current = false;
  };
  const onKey = (e: KeyboardEvent<HTMLButtonElement>) => {
    const stepDeg = e.shiftKey ? 15 : 3;
    let deg: number | null = null;
    const cur = (angle * 180) / Math.PI;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') deg = cur + stepDeg;
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') deg = cur - stepDeg;
    if (deg === null) return;
    e.preventDefault();
    e.stopPropagation();
    setAngle((Math.round(deg) * Math.PI) / 180);
  };
  const cText = `(${lp.c.map((v) => fmt(v, 3)).join(', ')})`;

  // Which paths run along which: one line, grouped by the path they follow.
  const note = useMemo(() => {
    if (!overlaps.length) return null;
    const byTarget = new Map<number, number[]>();
    for (const [i, j] of overlaps) byTarget.set(j, [...(byTarget.get(j) ?? []), i]);
    return [...byTarget.entries()]
      .map(([j, is]) => {
        const who = is.map((i) => runs[i].name);
        const list =
          who.length > 1 ? `${who.slice(0, -1).join(', ')} and ${who[who.length - 1]}` : who[0];
        return `${list} follow${who.length > 1 ? '' : 's'} the ${runs[j].name} path`;
      })
      .join('; ');
  }, [overlaps, runs]);

  return (
    <div className={styles.planeWrap}>
      {note && (
        <p className={styles.note} role="note" title={note}>
          {note} <span className={styles.noteDash}>(dashed, on top)</span>
        </p>
      )}
      <div className={styles.canvasBox}>
        <canvas ref={canvasRef} className={styles.canvas} role="img" aria-label={ariaLabel} />
        {onObjectiveChange && W > 0 && (
          <button
            type="button"
            className={styles.cHandle}
            style={{ left: tip[0], top: tip[1] }}
            data-own-keys=""
            aria-label={`Objective vector c = ${cText}. Drag, or press the arrow keys, to rotate it.`}
            title="Drag to rotate the objective"
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerCancel={onPointerUp}
            onKeyDown={onKey}
          />
        )}
      </div>
    </div>
  );
}

const HANDLE_LEN = 44;
const handleBase = (W: number): P2 => [Math.max(70, W - 74), 70];

// ── Drawing ─────────────────────────────────────────────────────────────────────────────

interface Scene {
  W: number;
  H: number;
  dpr: number;
  colors: ChartColors;
  lp: LinearProgram;
  hs: HalfSpace[];
  poly: P2[];
  verts: P2[];
  lattice: { feasible: P2[]; infeasible: P2[] } | null;
  central: P2[];
  runs: PlaneRun[];
  t: number;
  localK: (i: number) => number;
  ease: boolean;
  focus: number;
  optimum: number[] | null;
  optimalValue: number | null;
  toPx: (x: number, y: number) => P2;
  X: ReturnType<typeof linearScale>;
  Y: ReturnType<typeof linearScale>;
  view: Box;
  dashed: Set<number>;
}

function drawScene(ctx: CanvasRenderingContext2D, s: Scene) {
  const { W, H, colors: c, lp, hs, poly, toPx, view } = s;
  ctx.fillStyle = c.surface;
  ctx.fillRect(0, 0, W, H);
  const cMin = minCost(lp);
  const fmin = (p: readonly number[]) => cMin[0] * p[0] + cMin[1] * p[1];

  // 1. Feasible polygon filled with the objective's level sets (basin = the optimal side).
  const inView = poly.length ? clipToBox(poly, view) : [];
  if (inView.length >= 3) {
    const vals = inView.map(fmin);
    const lo = Math.min(...vals),
      hi = Math.max(...vals);
    const nb = 9;
    const map = CONTOUR_MAPS[c.mode];
    for (let i = 0; i < nb; i++) {
      const l0 = lo + ((hi - lo) * i) / nb,
        l1 = lo + ((hi - lo) * (i + 1)) / nb;
      let band = clipPolygon(inView, cMin, l1);
      band = clipPolygon(
        band,
        cMin.map((v) => -v),
        -l0,
      );
      if (band.length < 3) continue;
      const [r, g, b] = sample(map, i / (nb - 1));
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.beginPath();
      polyPath(ctx, band, toPx);
      ctx.fill();
    }
    // Iso-lines at the band edges.
    ctx.save();
    ctx.beginPath();
    polyPath(ctx, inView, toPx);
    ctx.clip();
    ctx.strokeStyle = c.iso;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 1; i < nb; i++) {
      const seg = lineInBox(cMin, lo + ((hi - lo) * i) / nb, view);
      if (!seg) continue;
      const [a, b] = [toPx(...seg[0]), toPx(...seg[1])];
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
    }
    ctx.stroke();
    ctx.restore();
  }
  // 2. Hatch the infeasible side.
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, 0, W, H);
  if (poly.length >= 3) {
    // Reverse orientation so even-odd leaves the polygon out.
    polyPath(ctx, [...poly].reverse(), toPx);
  }
  ctx.clip('evenodd');
  ctx.fillStyle = c.region;
  ctx.globalAlpha = 0.045;
  ctx.fillRect(0, 0, W, H);
  ctx.globalAlpha = 1;
  hatch(ctx, W, H, c.region, 8, 0.09);
  ctx.restore();

  // 3. Grid and inset axes.
  drawAxes(ctx, {
    x: s.X,
    y: s.Y,
    frame: { left: 0, top: 0, right: W, bottom: H },
    colors: c,
    dpr: s.dpr,
    inset: true,
    baseline: false,
    xName: [mv('x'), sub('1')],
    yName: [mv('x'), sub('2')],
  });
  // The axes x₁ = 0 and x₂ = 0 themselves.
  ctx.save();
  ctx.strokeStyle = c.axis;
  ctx.lineWidth = 1;
  ctx.beginPath();
  const o = toPx(0, 0);
  ctx.moveTo(o[0], 0);
  ctx.lineTo(o[0], H);
  ctx.moveTo(0, o[1]);
  ctx.lineTo(W, o[1]);
  ctx.stroke();
  ctx.restore();

  // 4. Lattice points (integer programs).
  if (s.lattice) {
    ctx.save();
    ctx.fillStyle = c.text;
    for (const p of s.lattice.infeasible) {
      const [px, py] = toPx(...p);
      ctx.globalAlpha = 0.22;
      ctx.beginPath();
      ctx.arc(px, py, 1.4, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 0.85;
    for (const p of s.lattice.feasible) {
      const [px, py] = toPx(...p);
      ctx.beginPath();
      ctx.arc(px, py, 2.4, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  // 5. Constraint lines with their inequalities.
  ctx.save();
  hs.forEach((h) => {
    if (h.kind === 'bound') return;
    const seg = lineInBox(h.a, h.b, view);
    if (!seg) return;
    const [a, b] = [toPx(...seg[0]), toPx(...seg[1])];
    ctx.strokeStyle = c.text3;
    ctx.globalAlpha = h.kind === 'eq' ? 0.9 : 0.5;
    ctx.lineWidth = 1;
    ctx.setLineDash(h.kind === 'eq' ? [] : [2, 3]);
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
  });
  ctx.restore();
  // Polygon boundary (only real constraints are in view).
  if (poly.length >= 3) {
    ctx.save();
    ctx.strokeStyle = c.text;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    polyPath(ctx, poly, toPx);
    ctx.stroke();
    ctx.restore();
  }
  // Constraint labels at the end of each line that lies inside the view (right / top first).
  labelConstraints(ctx, s);
  // Vertices.
  ctx.save();
  for (const p of s.verts) {
    const [px, py] = toPx(...p);
    if (px < -5 || px > W + 5 || py < -5 || py > H + 5) continue;
    ctx.fillStyle = c.surface;
    ctx.strokeStyle = c.text;
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    ctx.arc(px, py, 2.6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();

  if (poly.length < 3 && lp.c.length === 2) {
    drawMath(
      ctx,
      [{ t: 'No feasible point: the constraints exclude each other', style: 'sans' }],
      W / 2,
      H / 2,
      {
        size: 12.5,
        align: 'center',
        color: c.text2,
        halo: c.halo,
      },
    );
  }

  // 6. Central path.
  if (s.central.length > 1) {
    ctx.save();
    ctx.strokeStyle = c.text2;
    ctx.globalAlpha = 0.75;
    ctx.lineWidth = 1.25;
    ctx.setLineDash([1.5, 3]);
    ctx.lineCap = 'round';
    ctx.beginPath();
    s.central.forEach((p, i) => {
      const [px, py] = toPx(...p);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
    ctx.restore();
    // The label goes where no method's path runs (the paths are drawn later, on top).
    const samples = pathSamples(s);
    const label = [serif('central path')];
    const w = measureMath(ctx, label, 13);
    let at: P2 | null = null;
    for (const f of [0.12, 0.22, 0.32, 0.45, 0.6, 0.75, 0.05]) {
      const [mx, my] = toPx(...s.central[Math.floor((s.central.length - 1) * f)]);
      for (const [dx, dy] of [
        [8, -6],
        [-8 - w, -6],
        [8, 16],
        [-8 - w, 16],
      ]) {
        const box: Rect = [mx + dx, my + dy - 12, mx + dx + w, my + dy + 3];
        if (box[0] < 4 || box[2] > W - 4 || box[1] < 4 || box[3] > H - 4) continue;
        if (samples.some((p) => inside(p, pad(box, 4)))) continue;
        if (hitsAxes(box, toPx(0, 0))) continue;
        at = [mx + dx, my + dy];
        break;
      }
      if (at) break;
    }
    if (at)
      drawMath(ctx, label, at[0], at[1], {
        size: 13,
        color: c.text2,
        halo: c.halo,
      });
  }

  // 7. Optimal level line cᵀx = z⋆.
  if (s.optimalValue !== null && s.optimum) {
    const seg = lineInBox(lp.c, s.optimalValue, view);
    if (seg) {
      const [a, b] = [toPx(...seg[0]), toPx(...seg[1])];
      ctx.save();
      ctx.strokeStyle = c.text;
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = 1;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.stroke();
      ctx.restore();
    }
  }

  // 8. Level line through each current iterate (the sweep).
  s.runs.forEach((r, i) => {
    if (r.id === 'branch_and_bound') return;
    const pts = r.trace.map((st) => st.x as unknown as P2 | null).filter((p): p is P2 => !!p);
    const p = pathPosition(pts, s.t, s.ease);
    if (!p) return;
    const seg = lineInBox(lp.c, lp.c[0] * p[0] + lp.c[1] * p[1], view);
    if (!seg) return;
    const [a, b] = [toPx(...seg[0]), toPx(...seg[1])];
    ctx.save();
    ctx.strokeStyle = c.series[r.slot];
    ctx.globalAlpha = i === s.focus ? 0.8 : 0.45;
    ctx.lineWidth = i === s.focus ? 1.4 : 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
    ctx.restore();
  });

  // 9. The focused method's step geometry; Gomory cuts of every run.
  s.runs.forEach((r, i) => {
    if (r.id === 'gomory_cuts') drawCuts(ctx, s, r, s.localK(i), i === s.focus);
  });
  const fr = s.runs[s.focus];
  if (fr) drawStepGeometry(ctx, s, fr);

  // 10. Paths.
  const paths: PathSpec[] = s.runs
    .map((r, i) => ({ r, i }))
    .filter(({ r }) => r.id !== 'branch_and_bound')
    .sort((a, b) => Number(s.dashed.has(a.i)) - Number(s.dashed.has(b.i)))
    .map(({ r, i }) => ({
      points: r.trace.map((st) => st.x as unknown as [number, number]).filter(Boolean),
      color: c.series[r.slot],
      label: r.name,
      // PDHG takes hundreds of short steps: a line without a dot per iterate.
      dots: r.id !== PDHG,
      muted: false,
      dash: s.dashed.has(i) ? [7, 6] : undefined,
      end: r.converged ? ('converged' as const) : ('stopped' as const),
      width: r.id === PDHG ? 1.5 : INTERIOR_METHODS.has(r.id) ? 1.75 : 2.25,
    }));
  drawPathLayer(ctx, paths, {
    t: s.t,
    toPx,
    halo: c.halo,
    ease: s.ease,
    bounds: { left: 0, top: 0, right: W, bottom: H },
    offViewLabels: true,
    textColor: c.text2,
  });
  // PDHG restarts reached so far: a square at each restart point (the epoch average).
  s.runs.forEach((r, i) => {
    if (r.id !== PDHG) return;
    const kNow = s.localK(i);
    for (const st of r.trace) {
      if (st.k > kNow) break;
      if (st.info.restarted !== true) continue;
      const x = st.x as number[];
      const [px, py] = toPx(x[0], x[1]);
      drawRestartMark(ctx, px, py, c.series[r.slot], c.halo, i === s.focus ? 4 : 3.25);
    }
  });

  // 11. Optimum.
  if (s.optimum) {
    drawOverlays2D(ctx, { toPx, width: W, height: H, dpr: s.dpr, colors: c }, [
      {
        kind: 'point',
        at: [s.optimum[0], s.optimum[1]],
        shape: 'cross',
        radius: 6,
      },
    ]);
    placeLabel(
      ctx,
      s,
      [mb('x'), sup('⋆'), mm(` = (${s.optimum.map((v) => fmt(v, 4)).join(', ')})`)],
      toPx(s.optimum[0], s.optimum[1]),
      10,
      { size: 12.5, color: c.text, halo: c.halo, prefer: 'outward' },
    );
  }
}

function clipToBox(poly: readonly P2[], box: Box): P2[] {
  let p = [...poly];
  p = clipPolygon(p, [1, 0], box[0][1]);
  p = clipPolygon(p, [-1, 0], -box[0][0]);
  p = clipPolygon(p, [0, 1], box[1][1]);
  p = clipPolygon(p, [0, -1], -box[1][0]);
  return p;
}

type Rect = [number, number, number, number];
const rectsOverlap = (a: Rect, b: Rect) =>
  !(a[2] < b[0] || a[0] > b[2] || a[3] < b[1] || a[1] > b[3]);

const serif = (t: string): MathRun => ({ t, style: 'serif-italic' });

/** Centroid of the visible feasible polygon in pixels (the canvas center when it is empty). */
function polygonCenterPx(s: Scene): P2 {
  const p = s.poly.length ? clipToBox(s.poly, s.view) : [];
  if (p.length < 3) return [s.W / 2, s.H / 2];
  const px = p.map((v) => s.toPx(...v));
  return [
    px.reduce((a, v) => a + v[0], 0) / px.length,
    px.reduce((a, v) => a + v[1], 0) / px.length,
  ];
}

/** Path samples in pixels, at most ~6 px apart, of every run's polyline. */
function pathSamples(s: Scene): P2[] {
  const out: P2[] = [];
  for (const r of s.runs) {
    if (r.id === 'branch_and_bound') continue;
    const pts = r.trace
      .map((st) => st.x as number[] | null)
      .filter((x): x is number[] => !!x)
      .map((x) => s.toPx(x[0], x[1]));
    for (let i = 0; i < pts.length; i++) {
      out.push(pts[i]);
      if (i === 0) continue;
      const [a, b] = [pts[i - 1], pts[i]];
      const n = Math.min(200, Math.floor(Math.hypot(b[0] - a[0], b[1] - a[1]) / 6));
      for (let q = 1; q < n; q++)
        out.push([a[0] + ((b[0] - a[0]) * q) / n, a[1] + ((b[1] - a[1]) * q) / n]);
    }
  }
  return out;
}

const pad = (r: Rect, g: number): Rect => [r[0] - g, r[1] - g, r[2] + g, r[3] + g];
const inside = (p: P2, r: Rect) => p[0] >= r[0] && p[0] <= r[2] && p[1] >= r[1] && p[1] <= r[3];

/**
 * Draw `runs` next to the pixel point `at`, on the first of eight sides (ordered by their angle
 * to the preferred direction: away from the polygon, or into it) whose box stays in the canvas
 * and clears the axis lines, the other vertices, the objective handle and, optionally, the paths.
 */
function placeLabel(
  ctx: CanvasRenderingContext2D,
  s: Scene,
  runs: MathRun[],
  at: P2,
  gap: number,
  o: { size: number; color: string; halo: string; prefer?: 'outward' | 'inward'; paths?: P2[] },
) {
  const w = measureMath(ctx, runs, o.size);
  const h = o.size + 2;
  const cen = polygonCenterPx(s);
  let d: P2 = [at[0] - cen[0], at[1] - cen[1]];
  if (o.prefer === 'inward') d = [-d[0], -d[1]];
  const dl = Math.hypot(d[0], d[1]);
  d = dl > 1 ? [d[0] / dl, d[1] / dl] : [0.7, -0.7];
  const dirs = Array.from({ length: 8 }, (_, i): P2 => {
    const a = (i * Math.PI) / 4;
    return [Math.cos(a), -Math.sin(a)];
  }).sort((p, q) => q[0] * d[0] + q[1] * d[1] - (p[0] * d[0] + p[1] * d[1]));
  const { verts, fixed, axes } = obstacles(s);
  // The labelled point's own vertex is not an obstacle; the tick bands and the handle always are.
  const others = [...verts.filter((b) => !inside(at, pad(b, 2))), ...fixed];
  const onAxis = Math.abs(at[0] - axes[0]) < 2 || Math.abs(at[1] - axes[1]) < 2;
  let chosen: Rect | null = null;
  for (const dir of dirs) {
    const ax = at[0] + dir[0] * gap,
      ay = at[1] + dir[1] * gap;
    const x0 = dir[0] > 0.3 ? ax : dir[0] < -0.3 ? ax - w : ax - w / 2;
    const y0 = dir[1] > 0.3 ? ay : dir[1] < -0.3 ? ay - h : ay - h / 2;
    const box: Rect = [x0, y0, x0 + w, y0 + h];
    if (!chosen) chosen = box;
    if (box[0] < 4 || box[2] > s.W - 4 || box[1] < 4 || box[3] > s.H - 4) continue;
    if (others.some((b) => rectsOverlap(box, b))) continue;
    if (!onAxis && hitsAxes(box, axes)) continue;
    if (onAxis && hitsAxes(pad(box, -1), axes)) continue;
    if (o.paths?.some((p) => inside(p, pad(box, 3)))) continue;
    chosen = box;
    break;
  }
  if (!chosen) return;
  drawMath(ctx, runs, chosen[0], chosen[3] - 3, {
    size: o.size,
    align: 'left',
    color: o.color,
    halo: o.halo,
  });
}

/** What a canvas label must not cover: the vertices, the axis lines and the objective handle. */
function obstacles(s: Scene): { boxes: Rect[]; verts: Rect[]; fixed: Rect[]; axes: P2 } {
  const verts: Rect[] = s.verts.map((v) => {
    const [px, py] = s.toPx(...v);
    return [px - 6, py - 6, px + 6, py + 6];
  });
  const hb = handleBase(s.W);
  const fixed: Rect[] = [
    [hb[0] - 58, hb[1] - 58, hb[0] + 58, hb[1] + 58],
    // The inset tick labels: a band along the bottom (x₁) and one along the left (x₂).
    [0, s.H - 30, s.W, s.H],
    [0, 0, 38, s.H],
  ];
  return { boxes: [...verts, ...fixed], verts, fixed, axes: s.toPx(0, 0) };
}

const hitsAxes = (r: Rect, o: P2) => (r[0] < o[0] && o[0] < r[2]) || (r[1] < o[1] && o[1] < r[3]);

function labelConstraints(ctx: CanvasRenderingContext2D, s: Scene) {
  const { hs, view, toPx, colors: c, W, H } = s;
  const { boxes, axes } = obstacles(s);
  const placed: Rect[] = [];
  const size = 12;
  hs.forEach((h) => {
    if (h.kind === 'bound' || h.kind === 'extra') return;
    if (h.kind === 'eq' && h.a.some((v, j) => v !== (s.lp.aEq?.[h.index]?.[j] ?? NaN))) return; // one label per equality
    const seg = lineInBox(h.a, h.b, view);
    if (!seg) return;
    // Start from the end nearer the top-right, which reads best.
    const [p, q] = seg;
    const end = p[1] + p[0] * 0.3 > q[1] + q[0] * 0.3 ? p : q;
    const other = end === p ? q : p;
    const [ex, ey] = toPx(...end);
    const [ox, oy] = toPx(...other);
    const len = Math.hypot(ox - ex, oy - ey) || 1;
    // A ≥ row stored as −aᵀx ≤ −b reads better in its own direction.
    const flip = h.kind === 'row' && h.a.every((v) => v <= 0) && h.b <= 0;
    const runs = flip
      ? linearRuns(
          h.a.map((v) => -v),
          '≥',
          -h.b,
        )
      : linearRuns(h.a, h.kind === 'eq' ? '=' : '≤', h.b);
    const w = measureMath(ctx, runs, size);
    // The label sits beside its line on the infeasible side (where aᵀx grows), off the polygon edge.
    const nl = Math.hypot(h.a[0], h.a[1]) || 1;
    const n: P2 = [h.a[0] / nl, -h.a[1] / nl];
    const half = Math.abs(n[0]) * (w / 2) + Math.abs(n[1]) * 7;
    const u0 = Math.min(0.45, Math.max(34 / len, 0.16));
    let best: Rect | null = null;
    for (const u of [u0, u0 + 0.12, u0 + 0.24, u0 + 0.36, 0.5, 0.08]) {
      if (u > 0.9) continue;
      let cx = ex + (ox - ex) * u + n[0] * (5 + half),
        cy = ey + (oy - ey) * u + n[1] * (5 + half);
      cx = Math.min(W - 6 - w / 2, Math.max(6 + w / 2, cx));
      cy = Math.min(H - 26, Math.max(24, cy));
      const box: Rect = [cx - w / 2, cy - 7, cx + w / 2, cy + 7];
      if (!best) best = box;
      const clear =
        !placed.some((b) => rectsOverlap(box, b)) &&
        !boxes.some((b) => rectsOverlap(box, b)) &&
        !hitsAxes(box, axes);
      if (clear) {
        best = box;
        break;
      }
    }
    if (!best) return;
    placed.push(best);
    drawMath(ctx, runs, (best[0] + best[2]) / 2, best[3] - 3, {
      size,
      align: 'center',
      color: c.text2,
      halo: c.halo,
    });
  });
}

/** Geometry of the focused method's current step. */
function drawStepGeometry(ctx: CanvasRenderingContext2D, s: Scene, r: PlaneRun) {
  const { colors: c, toPx, view, lp, W, H } = s;
  const k = s.localK(s.focus);
  const step = r.trace[k];
  if (!step) return;
  const color = c.series[r.slot];
  const info = step.info as Record<string, unknown>;
  const ov = { toPx, width: W, height: H, dpr: s.dpr, colors: c };

  const strokeBoundary = (label: string, inequalityForm: boolean, width = 1.25, alpha = 0.7) => {
    const bd = boundaryOf(label, lp, inequalityForm);
    if (!bd) return;
    let seg: [P2, P2] | null = null;
    if (bd.kind === 'axis') seg = lineInBox(bd.j === 0 ? [1, 0] : [0, 1], 0, view);
    else if (bd.kind === 'row') seg = lineInBox(bd.a, bd.b, view);
    if (!seg) return;
    const [a, b] = [toPx(...seg[0]), toPx(...seg[1])];
    ctx.save();
    ctx.strokeStyle = color;
    ctx.globalAlpha = alpha;
    ctx.lineWidth = width;
    ctx.setLineDash([10, 3, 2, 3]);
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
    ctx.restore();
  };

  if (Array.isArray(info.tableau) && r.id !== 'gomory_cuts') {
    const ti = info as unknown as TableauInfo;
    const x = step.x as number[];
    const q = ti.entering;
    if (r.id === 'dual_simplex') {
      if (ti.leaving !== null) strokeBoundary(ti.col_labels[ti.leaving], true);
      return;
    }
    if (q === null || q === undefined) return;
    if (ti.leaving !== null && ti.leaving !== undefined)
      strokeBoundary(ti.col_labels[ti.leaving], false);
    const eta = edgeDirection(ti, q, 2);
    if (Math.hypot(eta[0], eta[1]) < 1e-14) return;
    const ratios = (ti.ratio_test ?? []).map((v, i) => ({ v, i })).filter((o) => o.v !== null) as {
      v: number;
      i: number;
    }[];
    const nObj = objectiveRows(ti);
    const x0 = toPx(x[0], x[1]);
    const len = Math.hypot(eta[0], eta[1]);
    const unit: P2 = [eta[0] / len, eta[1] / len];
    const reach = 3 * Math.max(view[0][1] - view[0][0], view[1][1] - view[1][0]);
    // The edge: to the winning ratio, or off the view for a ray.
    const theta = ratios.length ? Math.min(...ratios.map((o) => o.v)) : Infinity;
    const far = Number.isFinite(theta)
      ? [x[0] + theta * eta[0], x[1] + theta * eta[1]]
      : [x[0] + reach * unit[0], x[1] + reach * unit[1]];
    const x1 = toPx(far[0], far[1]);
    // The edge being walked: a wide translucent band under the arrow.
    ctx.save();
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.22;
    ctx.lineWidth = 9;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(x0[0], x0[1]);
    ctx.lineTo(x1[0], x1[1]);
    ctx.stroke();
    ctx.restore();
    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.95;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    if (Number.isFinite(theta) && theta > 1e-12) arrow(ctx, x0, x1, 8);
    else if (!Number.isFinite(theta)) arrow(ctx, x0, x1, 9);
    ctx.restore();
    // Ratio-test points: where each blocking constraint becomes tight along the edge. Only the
    // pivot row is named as the leaving variable; rows that tie with it are marked as rings, and
    // the label lists them, so the canvas agrees with the tableau.
    const tol = 1e-12 * (1 + (Number.isFinite(theta) ? theta : 0));
    const tied = ratios.filter((o) => Math.abs(o.v - theta) <= tol).map((o) => o.i);
    const pivotI =
      ti.pivot_row !== null && ti.pivot_row !== undefined ? ti.pivot_row - nObj : (tied[0] ?? -1);
    const nameRuns = (i: number): MathRun[] => {
      const name = ti.row_labels[nObj + i];
      return [mv(name[0]), sub(name.slice(1))];
    };
    const degenerate = Number.isFinite(theta) && theta <= 1e-12;
    const pivotLabel = (): MathRun[] => {
      const others = tied.filter((i) => i !== pivotI);
      const runs: MathRun[] = [...nameRuns(pivotI)];
      for (const i of others) runs.push(mm(' = '), ...nameRuns(i));
      runs.push(mm(' = 0'));
      if (others.length) runs.push(serif('  tie: '), ...nameRuns(pivotI), serif(' leaves'));
      if (degenerate) runs.push(serif('  · '), mv('θ'), mm(' = 0'), serif(', degenerate pivot'));
      return runs;
    };
    ratios.forEach(({ v, i }) => {
      const p: P2 = [x[0] + v * eta[0], x[1] + v * eta[1]];
      const isPivot = i === pivotI;
      const isTied = tied.includes(i);
      drawOverlays2D(ctx, ov, [
        {
          kind: 'point',
          at: p,
          slot: r.slot,
          shape: isPivot || isTied ? 'ring' : 'cross',
          radius: isPivot ? 6 : isTied ? 4.5 : 3.5,
        },
      ]);
    });
    if (pivotI >= 0 && Number.isFinite(theta)) {
      const at = toPx(x[0] + theta * eta[0], x[1] + theta * eta[1]);
      placeLabel(ctx, s, pivotLabel(), at, 10, {
        size: 12.5,
        color: c.text,
        halo: c.halo,
        // A degenerate pivot is told inside the region, clear of the axes it sits on.
        prefer: degenerate ? 'inward' : 'outward',
      });
    }
    if (!Number.isFinite(theta)) {
      const tip = toPx(x[0] + 0.12 * reach * unit[0], x[1] + 0.12 * reach * unit[1]);
      drawMath(ctx, [serif('unbounded ray')], tip[0] + 8, tip[1] - 6, {
        size: 13,
        color: c.text2,
        halo: c.halo,
      });
    }
    return;
  }

  if (r.id === 'primal_dual_ipm') {
    const next = r.trace[k + 1];
    const xa = next?.info.x_affine as number[] | undefined;
    if (xa) {
      const x = step.x as number[];
      const a = toPx(x[0], x[1]),
        b = toPx(xa[0], xa[1]);
      ctx.save();
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.6;
      ctx.lineWidth = 1.25;
      ctx.setLineDash([3, 3]);
      arrow(ctx, a, b, 6);
      ctx.restore();
      drawMath(ctx, [serif('predictor')], b[0] + 7, b[1] + 4, {
        size: 12,
        color: c.text2,
        halo: c.halo,
      });
    }
    return;
  }

  if (r.id === 'affine_scaling') {
    // The method's own metric ‖Z⁻¹d‖ ≤ 1 on z = (x, s, t), seen in the plane: while the
    // artificial t > 0 it is not the polygon's Dikin ellipse, and may reach outside the polygon.
    const x = step.x as number[];
    const Q = affineScalingMetric(lp, x, Number(info.artificial ?? 0));
    if (Q)
      drawOverlays2D(ctx, ov, [
        {
          kind: 'ellipse',
          center: [x[0], x[1]],
          matrix: Q,
          radius: 1,
          slot: r.slot,
          fill: true,
          width: 1.25,
          alpha: 0.9,
        },
      ]);
    return;
  }

  if (r.id === 'branch_and_bound') {
    drawBranchAndBound(ctx, s, r, k);
    return;
  }

  if (r.id === PDHG) drawPdhgStep(ctx, s, r, k);
}

/**
 * PDHG's step: the running average x̄ of the epoch (a ring, tied to the iterate by a hairline);
 * at a restart, where the PDHG step went (a cross) and the jump from there to the average. The
 * strip under the plot names these marks; the canvas carries no text here, because PDHG spends
 * most of its steps next to x⋆, whose label would collide with any other.
 */
function drawPdhgStep(ctx: CanvasRenderingContext2D, s: Scene, r: PlaneRun, k: number) {
  const { colors: c, toPx, W, H } = s;
  const step = r.trace[k];
  const info = step.info as Record<string, unknown>;
  if (k === 0) return;
  const color = c.series[r.slot];
  const ov = { toPx, width: W, height: H, dpr: s.dpr, colors: c };
  const x = step.x as number[];
  const at = toPx(x[0], x[1]);
  const other = (info.restarted === true ? info.x_pdhg : info.x_avg) as number[];
  const to = toPx(other[0], other[1]);
  // Nothing to show when the two points share a pixel neighborhood.
  if (Math.hypot(to[0] - at[0], to[1] - at[1]) < 6) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  if (info.restarted === true) {
    ctx.globalAlpha = 0.85;
    ctx.lineWidth = 1.25;
    ctx.setLineDash([3, 3]);
    arrow(ctx, to, at, 6);
  } else {
    ctx.globalAlpha = 0.6;
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(at[0], at[1]);
    ctx.lineTo(to[0], to[1]);
    ctx.stroke();
  }
  ctx.restore();
  drawOverlays2D(ctx, ov, [
    {
      kind: 'point',
      at: [other[0], other[1]],
      slot: r.slot,
      shape: info.restarted === true ? 'cross' : 'ring',
      radius: info.restarted === true ? 4 : 5,
    },
  ]);
}

/** Gomory cuts of a run at its current step: the lines, and for the focused run the slivers they remove. */
function drawCuts(ctx: CanvasRenderingContext2D, s: Scene, r: PlaneRun, k: number, full: boolean) {
  const { colors: c, toPx, view, W, H } = s;
  const step = r.trace[k];
  if (!step) return;
  const color = c.series[r.slot];
  const cuts = (step.info.cuts as { coef: number[]; rhs: number; active: boolean }[]) ?? [];
  cuts.forEach((cut, q) => {
    const removed = full
      ? clipPolygon(
          clipToBox(s.poly, view),
          cut.coef.map((v) => -v),
          -cut.rhs,
        )
      : [];
    if (removed.length >= 3) {
      ctx.save();
      ctx.beginPath();
      polyPath(ctx, removed, toPx);
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.16;
      ctx.fill();
      ctx.clip();
      hatch(ctx, W, H, color, 5, 0.35);
      ctx.restore();
    }
    const seg = lineInBox(cut.coef, cut.rhs, view);
    if (!seg) return;
    const [a, b] = [toPx(...seg[0]), toPx(...seg[1])];
    ctx.save();
    ctx.strokeStyle = color;
    ctx.globalAlpha = full ? 1 : 0.7;
    ctx.lineWidth = full && q === cuts.length - 1 ? 2.25 : 1.5;
    ctx.setLineDash(cut.active ? [] : [5, 4]);
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
    ctx.restore();
    if (full) {
      const mid: P2 = [a[0] * 0.3 + b[0] * 0.7, a[1] * 0.3 + b[1] * 0.7];
      drawMath(ctx, linearRuns(cut.coef, '≤', cut.rhs), mid[0] + 6, mid[1] - 6, {
        size: 12,
        color: c.text,
        halo: c.halo,
      });
    }
  });
}

interface BBNode {
  id: number;
  parent: number | null;
  branch: [number, string, number] | null;
  bounds: [number, number | null][];
  lp_value: number | null;
  lp_x: number[] | null;
  status: string;
}

function drawBranchAndBound(ctx: CanvasRenderingContext2D, s: Scene, r: PlaneRun, k: number) {
  const { colors: c, toPx, view, W, H, poly } = s;
  const step = r.trace[k];
  const info = step.info as Record<string, unknown>;
  const tree = info.tree as BBNode[];
  const node = tree[info.node as number];
  const color = c.series[r.slot];
  const sub2 = (n: BBNode) => {
    let p = clipToBox(poly, view);
    for (const h of boundSpaces(n.bounds)) p = clipPolygon(p, h.a, h.b);
    return p;
  };
  // Open nodes: their sub-boxes, outlined.
  for (const n of tree) {
    if (n.status !== 'open') continue;
    const p = sub2(n);
    if (p.length < 3) continue;
    ctx.save();
    ctx.strokeStyle = c.text3;
    ctx.globalAlpha = 0.8;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    polyPath(ctx, p, toPx);
    ctx.stroke();
    ctx.restore();
  }
  // The processed node's region.
  const region = sub2(node);
  if (region.length >= 3) {
    ctx.save();
    ctx.beginPath();
    polyPath(ctx, region, toPx);
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.14;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.75;
    ctx.stroke();
    ctx.restore();
  }
  // Branching: the strip ⌊v⌋ < xⱼ < ⌈v⌉ leaves both children.
  const j = info.branch_var as number | null;
  if (j !== null && node.lp_x) {
    const v = node.lp_x[j];
    const lo = Math.floor(v),
      hi = Math.ceil(v);
    const strip = clipPolygon(
      clipPolygon(clipToBox(poly, view), j === 0 ? [1, 0] : [0, 1], hi),
      j === 0 ? [-1, 0] : [0, -1],
      -lo,
    );
    ctx.save();
    if (strip.length >= 3) {
      ctx.beginPath();
      polyPath(ctx, strip, toPx);
      ctx.clip();
      hatch(ctx, W, H, color, 5, 0.45);
    }
    ctx.restore();
    for (const [val, rel] of [
      [lo, '≤'],
      [hi, '≥'],
    ] as const) {
      const seg = lineInBox(j === 0 ? [1, 0] : [0, 1], val, view);
      if (!seg) continue;
      const [a, b] = [toPx(...seg[0]), toPx(...seg[1])];
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.stroke();
      ctx.restore();
      const at = j === 0 ? toPx(val, view[1][1]) : toPx(view[0][1], val);
      drawMath(
        ctx,
        [mv('x'), sub(String(j + 1)), mm(` ${rel} ${val}`)],
        at[0] + (j === 0 ? (rel === '≤' ? -6 : 6) : -8),
        at[1] + (j === 0 ? 34 : rel === '≤' ? 14 : -6),
        {
          size: 12.5,
          align: j === 0 ? (rel === '≤' ? 'right' : 'left') : 'right',
          color: c.text,
          halo: c.halo,
        },
      );
    }
  }
  // LP optima of the nodes solved so far.
  const ov = { toPx, width: W, height: H, dpr: s.dpr, colors: c };
  for (const n of tree) {
    if (!n.lp_x || n.id === node.id) continue;
    drawOverlays2D(ctx, ov, [
      { kind: 'point', at: [n.lp_x[0], n.lp_x[1]], slot: r.slot, shape: 'ring', radius: 3.5 },
    ]);
  }
  const inc = info.incumbent as number[] | null;
  const near = (p: readonly number[] | null | undefined, q: readonly number[] | null | undefined) =>
    !!p && !!q && Math.hypot(p[0] - q[0], p[1] - q[1]) < 1e-6 * (1 + Math.hypot(q[0], q[1]));
  if (node.lp_x)
    drawOverlays2D(ctx, ov, [
      {
        kind: 'point',
        at: [node.lp_x[0], node.lp_x[1]],
        slot: r.slot,
        shape: 'dot',
        radius: 4.5,
        label: near(node.lp_x, s.optimum)
          ? undefined
          : [mv('z'), sup('LP'), mm(` = ${fmt(node.lp_value as number, 4)}`)],
        labelSide: 'right',
      },
    ]);
  if (inc) {
    const [px, py] = toPx(inc[0], inc[1]);
    ctx.save();
    ctx.fillStyle = color;
    ctx.strokeStyle = c.halo;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(px, py - 7);
    ctx.lineTo(px + 7, py);
    ctx.lineTo(px, py + 7);
    ctx.lineTo(px - 7, py);
    ctx.closePath();
    ctx.stroke();
    ctx.fill();
    ctx.restore();
    if (!near(inc, s.optimum))
      drawMath(ctx, [serif('incumbent')], px + 10, py + 16, {
        size: 12.5,
        color: c.text2,
        halo: c.halo,
      });
  }
}
