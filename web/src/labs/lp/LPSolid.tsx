/**
 * The 3-D picture of a linear program: its polytope (vertices, edges, faces by enumeration),
 * projected orthographically after a rotation the viewer controls (drag, or the arrow keys).
 * Each axis is scaled to the polytope's extent — the Klee–Minty cube spans 1 × 100 × 10⁴ — and
 * the caption says so. Paths, the simplex edge of the focused method, branch-and-bound
 * sub-polytopes and Gomory-cut polytopes are drawn in the method's color.
 */
import { useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react';
import type { LinearProgram, Step } from '../../core/types';
import { useChartColors } from '../../ui/theme';
import { useCanvas } from '../../viz/useCanvas';
import { drawPathLayer, type PathSpec } from '../../viz/PathLayer';
import { drawMath, m as mm, v as mv, b as mb, sub, sup } from '../../viz/mathText';
import { Formula } from '../../ui/components/Formula';
import {
  boundSpaces,
  edgeDirection,
  halfSpaces,
  lattice3,
  polytope3,
  type HalfSpace,
  type Polytope,
  type TableauInfo,
} from './geometry';
import { fmt, texNum } from './explain';
import type { PlaneRun } from './LPPlane';
import { PDHG, drawRestartMark } from './pdhgView';
import styles from './LPLab.module.css';

export interface LPSolidProps {
  lp: LinearProgram;
  runs: PlaneRun[];
  t: number;
  localK: (i: number) => number;
  ease: boolean;
  focus: number;
  optimum: number[] | null;
  integer: boolean;
  ariaLabel: string;
}

type P2 = [number, number];

const YAW0 = -0.72,
  PITCH0 = 0.42;
/** Axis length as a multiple of the polytope's extent. */
const AXIS_LEN = 1.18;

export function LPSolid({
  lp,
  runs,
  t,
  localK,
  ease,
  focus,
  optimum,
  integer,
  ariaLabel,
}: LPSolidProps) {
  const colors = useChartColors();
  const hs = useMemo(() => halfSpaces(lp), [lp]);
  const poly = useMemo(() => polytope3(hs), [hs]);
  const hi = useMemo(() => {
    const h = [0, 1, 2].map((d) => Math.max(0, ...poly.vertices.map((v) => v[d])));
    return h.map((v) => (v > 0 ? v : 1));
  }, [poly]);
  const lattice = useMemo(() => (integer ? lattice3(hs, hi) : []), [integer, hs, hi]);
  const [rot, setRot] = useState<[number, number]>([YAW0, PITCH0]);
  const drag = useRef<{ x: number; y: number; r: [number, number] } | null>(null);

  const { canvasRef } = useCanvas((ctx, { width: W, height: H }) => {
    if (W < 10 || H < 10) return;
    ctx.fillStyle = colors.surface;
    ctx.fillRect(0, 0, W, H);
    const [yaw, pitch] = rot;
    const cy = Math.cos(yaw),
      sy = Math.sin(yaw),
      cp = Math.cos(pitch),
      sp = Math.sin(pitch);
    /** World (normalized, centered) → [screen x (unscaled), screen up (unscaled), depth]. */
    const rotate = (x: readonly number[]): [number, number, number] => {
      const X = x[0] / hi[0] - 0.5,
        Y = x[1] / hi[1] - 0.5,
        Z = x[2] / hi[2] - 0.5;
      const x1 = X * cy - Y * sy,
        y1 = X * sy + Y * cy;
      const y2 = y1 * cp - Z * sp,
        z2 = y1 * sp + Z * cp;
      return [x1, z2, y2];
    };
    // Fit this rotation's projection — the vertices, the axis ends, the optimum — into the
    // figure, with room for the axis names and the caption under it.
    const fitPts: number[][] = [...poly.vertices, [0, 0, 0]];
    for (let d = 0; d < 3; d++) {
      const e = [0, 0, 0];
      e[d] = hi[d] * AXIS_LEN;
      fitPts.push(e);
    }
    if (optimum) fitPts.push([...optimum]);
    let x0 = Infinity,
      x1 = -Infinity,
      z0 = Infinity,
      z1 = -Infinity;
    for (const p of fitPts) {
      const [u, w] = rotate(p);
      x0 = Math.min(x0, u);
      x1 = Math.max(x1, u);
      z0 = Math.min(z0, w);
      z1 = Math.max(z1, w);
    }
    const side = 30,
      top = 24,
      bottom = 46;
    const availW = Math.max(40, W - 2 * side),
      availH = Math.max(40, H - top - bottom);
    const S = Math.min(availW / Math.max(1e-9, x1 - x0), availH / Math.max(1e-9, z1 - z0));
    const cx0 = side + (availW - S * (x1 - x0)) / 2 - S * x0,
      cy0 = top + (availH - S * (z1 - z0)) / 2 + S * z1;
    const proj = (x: readonly number[]): [number, number, number] => {
      const [u, w, depth] = rotate(x);
      return [cx0 + S * u, cy0 - S * w, depth];
    };
    const toPx = (x: readonly number[]): P2 => {
      const p = proj(x);
      return [p[0], p[1]];
    };
    /** Is a face with outward normal a (original units) turned toward the viewer? */
    const facing = (a: readonly number[]) => {
      const n = [a[0] * hi[0], a[1] * hi[1], a[2] * hi[2]];
      const y1 = n[0] * sy + n[1] * cy;
      return y1 * cp - n[2] * sp < 0;
    };

    // Axes from the origin.
    ctx.save();
    ctx.strokeStyle = colors.axis;
    ctx.lineWidth = 1;
    for (let d = 0; d < 3; d++) {
      const end = [0, 0, 0];
      end[d] = hi[d] * AXIS_LEN;
      const a = toPx([0, 0, 0]),
        b = toPx(end);
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.stroke();
      drawMath(ctx, [mv('x'), sub(String(d + 1))], b[0] + 4, b[1] + 4, {
        size: 14,
        color: colors.text2,
        halo: colors.halo,
      });
    }
    ctx.restore();

    drawPolytope(ctx, poly, hs, toPx, facing, colors.text, colors.text3, colors.region, 1);

    // Lattice points (integer programs).
    if (lattice.length) {
      ctx.save();
      ctx.fillStyle = colors.text;
      ctx.globalAlpha = 0.75;
      for (const p of lattice) {
        const [px, py] = toPx(p);
        ctx.beginPath();
        ctx.arc(px, py, 2.2, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }

    // Focused method's step geometry.
    const fr = runs[focus];
    if (fr) {
      const k = localK(focus);
      const step = fr.trace[k];
      const color = colors.series[fr.slot];
      if (step) drawStep3(ctx, { step, run: fr, hs, toPx, facing, color, colors, lp });
    }

    // Paths (projected; the path layer then works in screen coordinates).
    const paths: PathSpec[] = runs
      .filter((r) => r.id !== 'branch_and_bound')
      .map((r) => ({
        points: r.trace.map((s) => toPx(s.x as number[])),
        color: colors.series[r.slot],
        label: r.name,
        dots: r.id !== PDHG,
        end: r.converged ? 'converged' : 'stopped',
        width:
          r.id === PDHG
            ? 1.5
            : r.id === 'primal_dual_ipm' || r.id === 'affine_scaling'
              ? 1.75
              : 2.25,
      }));
    drawPathLayer(ctx, paths, {
      t,
      toPx: (x, y) => [x, y],
      halo: colors.halo,
      ease,
      textColor: colors.text2,
    });
    // PDHG restarts reached so far (squares at the epoch averages).
    runs.forEach((r, i) => {
      if (r.id !== PDHG) return;
      const kNow = localK(i);
      for (const st of r.trace) {
        if (st.k > kNow) break;
        if (st.info.restarted !== true) continue;
        const [px, py] = toPx(st.x as number[]);
        drawRestartMark(ctx, px, py, colors.series[r.slot], colors.halo, i === focus ? 4 : 3.25);
      }
    });

    if (optimum) {
      const [px, py] = toPx(optimum);
      ctx.save();
      ctx.strokeStyle = colors.halo;
      ctx.lineWidth = 4;
      const cross = () => {
        ctx.beginPath();
        ctx.moveTo(px - 6, py);
        ctx.lineTo(px + 6, py);
        ctx.moveTo(px, py - 6);
        ctx.lineTo(px, py + 6);
        ctx.stroke();
      };
      cross();
      ctx.strokeStyle = colors.text;
      ctx.lineWidth = 1.75;
      cross();
      ctx.restore();
      // The label points away from the polytope's projected center, off its edges.
      const pv = poly.vertices.map(toPx);
      const cen: P2 = pv.length
        ? [
            pv.reduce((a, v) => a + v[0], 0) / pv.length,
            pv.reduce((a, v) => a + v[1], 0) / pv.length,
          ]
        : [W / 2, H / 2];
      const right = px >= cen[0] - 1;
      const below = py > cen[1] + 1;
      drawMath(
        ctx,
        [mb('x'), sup('⋆'), mm(` = (${optimum.map((v) => fmt(v, 4)).join(', ')})`)],
        px + (right ? 10 : -10),
        py + (below ? 20 : -10),
        {
          size: 13,
          align: right ? 'left' : 'right',
          color: colors.text,
          halo: colors.halo,
        },
      );
    }
  });

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    drag.current = { x: e.clientX, y: e.clientY, r: rot };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d) return;
    setRot([d.r[0] + (e.clientX - d.x) * 0.008, clampPitch(d.r[1] + (e.clientY - d.y) * 0.008)]);
  };
  const onPointerUp = () => {
    drag.current = null;
  };
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const s = e.shiftKey ? 0.26 : 0.09;
    const moves: Record<string, [number, number]> = {
      ArrowLeft: [-s, 0],
      ArrowRight: [s, 0],
      ArrowUp: [0, -s],
      ArrowDown: [0, s],
    };
    if (e.key === '0') {
      setRot([YAW0, PITCH0]);
      e.preventDefault();
      e.stopPropagation();
      return;
    }
    const mv2 = moves[e.key];
    if (!mv2) return;
    e.preventDefault();
    e.stopPropagation();
    setRot(([a, b]) => [a + mv2[0], clampPitch(b + mv2[1])]);
  };

  return (
    <div
      className={styles.planeWrap}
      tabIndex={0}
      data-own-keys=""
      data-plot-focus=""
      aria-label={`${ariaLabel} Drag or use the arrow keys to rotate; 0 resets the view.`}
      role="group"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onDoubleClick={() => setRot([YAW0, PITCH0])}
      onKeyDown={onKey}
      style={{ cursor: 'grab', touchAction: 'pan-y' }}
    >
      <canvas ref={canvasRef} className={styles.canvas} role="img" aria-label={ariaLabel} />
      <p className={styles.solidCaption}>
        <Formula
          tex={`x_1 \\in [0, ${texNum(hi[0])}],\\ x_2 \\in [0, ${texNum(hi[1])}],\\ x_3 \\in [0, ${texNum(hi[2])}]`}
        />{' '}
        <span>— each axis scaled to the polytope · drag to rotate</span>
      </p>
    </div>
  );
}

const clampPitch = (p: number) => Math.max(-1.2, Math.min(1.35, p));

function drawPolytope(
  ctx: CanvasRenderingContext2D,
  poly: Polytope,
  hs: readonly HalfSpace[],
  toPx: (x: readonly number[]) => P2,
  facing: (a: readonly number[]) => boolean,
  ink: string,
  faint: string,
  tint: string,
  alpha: number,
  fill = true,
) {
  const visibleFace = poly.faces.map((f) => facing(hs[f.plane].a));
  if (fill) {
    ctx.save();
    poly.faces.forEach((f, i) => {
      if (!visibleFace[i]) return;
      ctx.beginPath();
      f.cycle.forEach((vi, q) => {
        const [px, py] = toPx(poly.vertices[vi]);
        if (q === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.closePath();
      ctx.fillStyle = tint;
      ctx.globalAlpha = 0.05 * alpha;
      ctx.fill();
    });
    ctx.restore();
  }
  // An edge is visible when one of its faces is.
  const faceOf = (i: number, j: number) =>
    poly.faces
      .filter((f) => f.cycle.includes(i) && f.cycle.includes(j))
      .map((f) => poly.faces.indexOf(f));
  ctx.save();
  ctx.lineJoin = 'round';
  for (const [i, j] of poly.edges) {
    const vis = faceOf(i, j).some((q) => visibleFace[q]);
    const a = toPx(poly.vertices[i]),
      b = toPx(poly.vertices[j]);
    ctx.strokeStyle = vis ? ink : faint;
    ctx.globalAlpha = (vis ? 1 : 0.55) * alpha;
    ctx.lineWidth = vis ? 1.5 : 1;
    ctx.setLineDash(vis ? [] : [3, 3]);
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
  }
  ctx.setLineDash([]);
  for (const v of poly.vertices) {
    const [px, py] = toPx(v);
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = ink;
    ctx.lineWidth = 1.1;
    ctx.beginPath();
    ctx.arc(px, py, 2.4, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();
}

interface Step3 {
  step: Step;
  run: PlaneRun;
  hs: HalfSpace[];
  toPx: (x: readonly number[]) => P2;
  facing: (a: readonly number[]) => boolean;
  color: string;
  colors: ReturnType<typeof useChartColors>;
  lp: LinearProgram;
}

function drawStep3(ctx: CanvasRenderingContext2D, o: Step3) {
  const { step, run, hs, toPx, facing, color, colors } = o;
  const info = step.info as Record<string, unknown>;
  if (Array.isArray(info.tableau) && run.id !== 'gomory_cuts' && run.id !== 'dual_simplex') {
    const ti = info as unknown as TableauInfo;
    const q = ti.entering;
    if (q === null || q === undefined) return;
    const x = step.x as number[];
    const eta = edgeDirection(ti, q, 3);
    const ratios = (ti.ratio_test ?? []).filter((v): v is number => v !== null);
    const theta = ratios.length ? Math.min(...ratios) : Infinity;
    if (!Number.isFinite(theta) || theta <= 1e-12) return;
    const end = x.map((v, j) => v + theta * eta[j]);
    const a = toPx(x),
      b = toPx(end);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.9;
    ctx.lineWidth = 5;
    ctx.globalAlpha = 0.22;
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
    ctx.restore();
    return;
  }
  if (run.id === 'branch_and_bound') {
    const tree = info.tree as {
      id: number;
      bounds: [number, number | null][];
      lp_x: number[] | null;
      status: string;
    }[];
    const node = tree[info.node as number];
    const sub3 = polytope3([...hs, ...boundSpaces(node.bounds)]);
    drawPolytope(
      ctx,
      sub3,
      [...hs, ...boundSpaces(node.bounds)],
      toPx,
      facing,
      color,
      color,
      color,
      1,
      false,
    );
    for (const n of tree)
      if (n.lp_x) {
        const [px, py] = toPx(n.lp_x);
        ctx.save();
        ctx.strokeStyle = color;
        ctx.fillStyle = n.id === node.id ? color : colors.surface;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(px, py, n.id === node.id ? 4.5 : 3.2, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.restore();
      }
    const inc = info.incumbent as number[] | null;
    if (inc) {
      const [px, py] = toPx(inc);
      ctx.save();
      ctx.fillStyle = color;
      ctx.strokeStyle = colors.halo;
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
    }
    return;
  }
  if (run.id === 'gomory_cuts') {
    const cuts = (info.cuts as { coef: number[]; rhs: number; active: boolean }[]) ?? [];
    if (!cuts.length) return;
    const extra: HalfSpace[] = cuts
      .filter((c) => c.active)
      .map((c, i) => ({ a: c.coef, b: c.rhs, kind: 'extra' as const, index: i }));
    const cutPoly = polytope3([...hs, ...extra]);
    drawPolytope(ctx, cutPoly, [...hs, ...extra], toPx, facing, color, color, color, 1, true);
  }
}
