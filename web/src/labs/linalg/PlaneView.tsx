/**
 * A 2 × 2 system in the plane: each equation is a line and the solution is where they cross.
 * The field is φ(x) = ½xᵀAx − bᵀx (SPD A; its level sets are ellipses centered at x⋆), else
 * ½‖Ax − b‖². Every method draws the geometry of the step in progress:
 *   Jacobi        the two simultaneous one-equation solves (to line 1 horizontally, to line 2
 *                 vertically) that together give the new iterate;
 *   Gauss–Seidel  the staircase of every sweep, each move landing on its line: the path itself
 *                 is the staircase (solid), drawn up to the playhead; the chord x_k → x_{k+1}
 *                 between the iterates is only a faint dashed guide;
 *   SOR           the same staircase, overshooting each line by ω (the GS target is ringed);
 *   SD, CG, PCG   the level ellipse the step is tangent to and the next search direction;
 *   elimination   the rows of the current augmented matrix: row operations turn the lines
 *                 about x⋆ until the second one is horizontal.
 */
import { useMemo } from 'react';
import type { Matrix, Step, Vector } from '../../core/types';
import { vec } from '../../core/format';
import { Contour2D, mathBold, mathSub, type Overlay2D, type PathSpec } from '../../viz';
import type { LinalgProblem } from '../../problems/linalg';
import type { LabRun } from '../_shell';
import {
  aNorm2,
  easeInOut,
  equationRuns,
  fieldOf,
  gsTargets,
  isDescent,
  isDirect,
  jacobiTargets,
  lineInBox,
  planeDomain,
  polylinePrefix,
  type Box,
} from './model';

type Pt = [number, number];

export interface PlaneViewProps {
  problem: LinalgProblem;
  spd: boolean;
  solution: Vector;
  runs: readonly LabRun[];
  /** Local playhead of each run (player.localT(i)). */
  localT: (i: number) => number;
  t: number;
  ease: boolean;
  x0: number[];
  onPick: (p: [number, number]) => void;
  /** Methods other than this one step back (null: none). */
  focusId: string | null;
  seriesColors: string[];
}

const pt = (x: unknown): Pt => [(x as number[])[0], (x as number[])[1]];

/** Methods whose path is drawn as the staircase of their sweeps. */
const STAIRS = new Set(['gauss_seidel', 'sor']);
/** At most this many past sweeps are drawn as a staircase (the rest are below the pixel). */
const MAX_STAIRS = 80;

export function PlaneView({
  problem,
  spd,
  solution,
  runs,
  localT,
  t,
  ease,
  x0,
  onPick,
  focusId,
  seriesColors,
}: PlaneViewProps) {
  const A = problem.A as Matrix;
  const b = problem.b;
  // The first iterates of the iterative runs (SOR with ω near 2 overshoots far past the lines).
  const early = runs
    .filter((r) => !isDirect(r.sel.id))
    .flatMap((r) => r.result.trace.slice(1, 5).map((s) => s.x as number[]))
    .filter(Array.isArray);
  const earlyKey = JSON.stringify(early.map((p) => p.map((v) => +v.toPrecision(3))));
  const domain = useMemo<Box>(
    () => planeDomain(solution, x0, JSON.parse(earlyKey) as number[][]),
    [solution, x0, earlyKey],
  );
  const f = useMemo(() => fieldOf(A, b, spd), [A, b, spd]);
  const fMin = useMemo(() => f(solution[0], solution[1]), [f, solution]);
  const span = domain[0][1] - domain[0][0];

  // The two equations, in ink, labelled at their upper end.
  const lines = useMemo<Overlay2D[]>(() => {
    const out: Overlay2D[] = [];
    A.forEach((row, i) => {
      const seg = lineInBox(row, b[i], domain);
      if (!seg) return;
      const [p, q] = seg[0][1] >= seg[1][1] ? seg : [seg[1], seg[0]];
      out.push({ kind: 'segment', from: p, to: q, width: 1.4, alpha: 0.85 });
      const u = 0.1;
      const at: Pt = [p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u];
      out.push({
        kind: 'text',
        at: [at[0] + span * 0.012, at[1]],
        text: equationRuns(row, b[i]),
        align: 'left',
        size: 13,
      });
    });
    return out;
  }, [A, b, domain, span]);

  const live = runs.map((r, i) => {
    const trace = r.result.trace;
    const last = trace.length - 1;
    const lt = Math.min(localT(i), last);
    return {
      r,
      trace,
      last,
      lt,
      s: Math.max(1, Math.min(last, Math.ceil(lt - 1e-9))),
      k: Math.floor(lt + 1e-9),
    };
  });

  const geometry: Overlay2D[] = [];
  for (const { r, trace, last, s, k, lt } of live) {
    const slot = r.sel.slot;
    const id = r.sel.id;
    const quiet = focusId !== null && focusId !== id;
    const alpha = quiet ? 0.3 : 1;
    if (isDirect(id)) {
      const step = trace[k];
      const M = step?.info.matrix as Matrix | undefined;
      if (M && M[0]?.length === 3)
        M.forEach((row) => {
          const seg = lineInBox([row[0], row[1]], row[2], domain);
          if (seg)
            geometry.push({
              kind: 'segment',
              from: seg[0],
              to: seg[1],
              slot,
              dashed: true,
              width: 1.6,
              alpha: 0.85 * alpha,
            });
        });
      if (Array.isArray(step?.x))
        geometry.push({ kind: 'point', at: pt(step.x), slot, shape: 'dot', radius: 4.5 });
      continue;
    }
    if (last < 1) continue;
    const from = trace[s - 1]?.x;
    const to = trace[s]?.x;
    if (!Array.isArray(from) || !Array.isArray(to)) continue;
    if (id === 'jacobi') {
      const [p1, p2] = jacobiTargets(A, b, from as number[]);
      geometry.push(
        {
          kind: 'segment',
          from: pt(from),
          to: p1,
          slot,
          dashed: true,
          width: 1.2,
          alpha: 0.9 * alpha,
        },
        {
          kind: 'segment',
          from: pt(from),
          to: p2,
          slot,
          dashed: true,
          width: 1.2,
          alpha: 0.9 * alpha,
        },
        {
          kind: 'segment',
          from: p1,
          to: pt(to),
          slot,
          dashed: true,
          width: 0.8,
          alpha: 0.45 * alpha,
        },
        {
          kind: 'segment',
          from: p2,
          to: pt(to),
          slot,
          dashed: true,
          width: 0.8,
          alpha: 0.45 * alpha,
        },
        { kind: 'point', at: p1, slot, shape: 'ring', radius: 3.5 },
        { kind: 'point', at: p2, slot, shape: 'ring', radius: 3.5 },
      );
    } else if (STAIRS.has(id)) {
      // The path: the staircase of every finished sweep, then the sweep in progress up to the
      // playhead, with the head where the playhead is.
      const stairs: Pt[] = [];
      for (let j = Math.max(1, k - MAX_STAIRS + 1); j <= Math.min(k, last); j++) {
        const sw = trace[j].info.sweep as Vector[] | undefined;
        if (!sw) continue;
        sw.forEach((p, q) => {
          if (q > 0 || stairs.length === 0) stairs.push(pt(p));
        });
      }
      const u = lt - k;
      const inProgress = k < last && u > 1e-6;
      let head: Pt | null = null;
      if (inProgress) {
        const sw = trace[k + 1].info.sweep as Vector[] | undefined;
        if (sw) {
          const pre = polylinePrefix(sw.map(pt), ease ? easeInOut(u) : u);
          if (stairs.length === 0) stairs.push(...pre.points);
          else stairs.push(...pre.points.slice(1));
          head = pre.end;
        }
      }
      if (stairs.length > 1)
        geometry.push({
          kind: 'polyline',
          points: stairs,
          slot,
          width: quiet ? 1.2 : 1.9,
          alpha: (quiet ? 0.45 : 1) * alpha,
        });
      if (head) geometry.push({ kind: 'point', at: head, slot, shape: 'dot', radius: 4.5 });
      const sweep = trace[s].info.sweep as Vector[] | undefined;
      if (sweep) {
        // The whole sweep in progress as a dashed preview (none at rest: it is drawn solid).
        if (inProgress)
          geometry.push({
            kind: 'polyline',
            points: sweep.map(pt),
            slot,
            dashed: true,
            width: 1.1,
            alpha: 0.6 * alpha,
          });
        if (id === 'sor') {
          const omega = Number(r.result.extra.omega ?? r.sel.params.omega ?? 1.5);
          for (const g of gsTargets(sweep, omega))
            geometry.push({ kind: 'point', at: g, slot, shape: 'ring', radius: 3.5 });
        }
      }
    } else if (isDescent(id)) {
      const lev = (x: number[]) => Math.sqrt(Math.max(0, aNorm2(A, x, solution)));
      if (spd) {
        geometry.push(
          {
            kind: 'ellipse',
            center: pt(solution),
            matrix: A,
            radius: lev(from as number[]),
            slot,
            width: 1,
            alpha: 0.35 * alpha,
          },
          {
            kind: 'ellipse',
            center: pt(solution),
            matrix: A,
            radius: lev(to as number[]),
            slot,
            width: 1.4,
            alpha: 0.9 * alpha,
          },
        );
      }
      const d = trace[s].info.direction as number[] | undefined;
      const nrm = d ? Math.hypot(d[0], d[1]) : 0;
      if (d && nrm > 0 && s < last) {
        const len = span * 0.16;
        const tip: Pt = [to[0] + (d[0] / nrm) * len, to[1] + (d[1] / nrm) * len];
        geometry.push({
          kind: 'arrow',
          from: pt(to),
          to: tip,
          slot,
          width: 1.5,
          alpha,
          label: [mathBold(id === 'steepest_descent_linear' ? 'r' : 'p'), mathSub(String(s))],
        });
      }
    }
  }

  const paths: PathSpec[] = live
    .filter(({ r, trace }) => !isDirect(r.sel.id) && trace.length > 0)
    .map(({ r, trace }) => {
      const stairs = STAIRS.has(r.sel.id);
      return {
        points: trace.map((st: Step) => pt(st.x)),
        color: seriesColors[r.sel.slot],
        label: r.method.spec.name,
        muted: focusId !== null && focusId !== r.sel.id,
        end: r.result.converged ? ('converged' as const) : ('stopped' as const),
        start: false,
        dots: trace.length <= 80,
        // Gauss–Seidel / SOR: the staircase overlay is the path; the chords between the
        // iterates are a faint guide and the overlay draws the head.
        ...(stairs ? { dash: [2, 4], width: 1, quiet: true } : {}),
      };
    });

  const names = runs.map((r) => r.method.spec.name).join(', ');
  return (
    <Contour2D
      f={f}
      domain={domain}
      cacheKey={`linalg:${problem.id}:${spd ? 'phi' : 'res'}`}
      fMin={fMin}
      paths={paths}
      t={t}
      ease={ease}
      minima={[pt(solution)]}
      minimaLabels
      start={[x0[0], x0[1]]}
      overlays={[...lines, ...geometry]}
      onPick={onPick}
      ariaLabel={`The two equations of ${problem.name} as lines meeting at x⋆ = ${vec(solution, 4)}, over the level sets of ${spd ? 'φ(x) = ½xᵀAx − bᵀx' : '½‖Ax − b‖²'}, with the iterates of ${names || 'no method'} from x₀ = ${vec(x0, 4)}. Click to move x₀.`}
    />
  );
}
