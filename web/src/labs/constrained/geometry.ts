/**
 * Geometry of the constrained lab, as declarative overlays (`Overlay2D`, data coordinates).
 *
 *   feasibleSet(problem)         infeasible side hatched, every boundary c_i = 0 labelled
 *   stepGeometry(…)              what the method looked at to take step k (from Step.info):
 *       projected_gradient       gradient step, projection (dashed), the projection arc and its trials
 *       frank_wolfe              the vertex atom s_{k−1}, the segment toward it, all atoms so far
 *       quadratic_penalty / AL   the quasi-Newton step p and the backtracking trials
 *       log_barrier              the Newton step Δx and the central path x⋆(t) of the stages so far
 *       sqp                      the QP step p and the linearized constraints at x_{k−1}
 *   kktArrows(…)                 −∇f(x_k) and the chain λ_i ∇c_i(x_k), on one scale
 *   meritLandscape(…)            the function the method minimizes at step k (penalty Q, L_A,
 *                                barrier f − t⁻¹Σ log(−c_i), ℓ1 merit φ₁) as f(x, y)
 *
 * Pure functions (no React), unit-tested in geometry.test.ts.
 */
import type { Step, Vector } from '../../core/types';
import type { ConstrainedProblem } from '../../problems/constrained';
import { FeasibleSet } from '../../methods/constrained/methods';
import {
  implicitSegments,
  mathBold as b,
  mathMain as m,
  mathSub as sub,
  mathVar as v,
  type MathRun,
  type Overlay2D,
} from '../../viz';

type Pt = [number, number];
const pt = (x: unknown): Pt => [(x as number[])[0], (x as number[])[1]];
const add = (a: Pt, d: readonly number[], s = 1): Pt => [a[0] + s * d[0], a[1] + s * d[1]];

/** Methods whose landscape toggle can show a modified objective. */
export const MERIT_METHODS = new Set([
  'quadratic_penalty',
  'augmented_lagrangian',
  'log_barrier',
  'sqp',
]);

/** Constraint label runs: c₁, c₂, … */
export const cLabel = (i: number): MathRun[] => [v('c'), sub(String(i + 1))];

/** The feasible set: hatch where some inequality is violated, every boundary labelled. */
export function feasibleSet(
  p: ConstrainedProblem,
  domain: readonly (readonly [number, number])[] = p.domain,
): Overlay2D[] {
  const ineq = p.constraints
    .filter((c) => c.kind === 'ineq')
    .map((c) => (x: number, y: number) => c.fun([x, y]));
  const out: Overlay2D[] = [];
  if (ineq.length)
    out.push({ kind: 'constraints', g: ineq, cacheKey: `${p.id}|feasible`, boundary: false });
  const avoid = [...p.minima, p.x0].map((q) => [q[0], q[1]] as Pt);
  const ineqIdx = p.constraints.flatMap((c, i) => (c.kind === 'ineq' ? [i] : []));
  p.constraints.forEach((c, i) => {
    const g = (x: number, y: number) => c.fun([x, y]);
    // On the boundary of the feasible set: the other inequalities hold there.
    const others = (x: number, y: number) =>
      ineqIdx.every((j) => j === i || p.constraints[j].fun([x, y]) <= 1e-9);
    out.push({
      kind: 'implicit',
      g,
      cacheKey: `${p.id}|c${i}`,
      width: c.kind === 'eq' ? 2.2 : 1.4,
    });
    const at = labelAnchor(g, domain, avoid, others) ?? labelAnchor(g, domain, avoid);
    if (at) {
      out.push({ kind: 'text', at, text: cLabel(i), align: 'center' });
      avoid.push(at); // spread the labels: the next one avoids this one
    }
  });
  return out;
}

/**
 * Where to label the curve g = 0: among the points of the curve inside the
 * domain (1.5 % margins) that lie on the boundary of the feasible set (`others(x) ≤ 0`), the one farthest from
 * the points to avoid (minimizers, the default start), nudged to the feasible side.
 */
export function labelAnchor(
  g: (x: number, y: number) => number,
  domain: readonly (readonly [number, number])[],
  avoid: readonly Pt[],
  others: (x: number, y: number) => boolean = () => true,
): Pt | null {
  const [[x0, x1], [y0, y1]] = domain;
  const W = x1 - x0,
    H = y1 - y0;
  const segs = implicitSegments(g, [x0, x1], [y0, y1], 97, 97);
  const inside = (q: Pt) =>
    q[0] > x0 + 0.015 * W &&
    q[0] < x1 - 0.015 * W &&
    q[1] > y0 + 0.015 * H &&
    q[1] < y1 - 0.015 * H;
  let best: Pt | null = null,
    score = -Infinity;
  for (let k = 0; k + 3 < segs.length; k += 4) {
    const q: Pt = [(segs[k] + segs[k + 2]) / 2, (segs[k + 1] + segs[k + 3]) / 2];
    if (!inside(q) || !others(q[0], q[1])) continue;
    const d = Math.min(...avoid.map((a) => Math.hypot((q[0] - a[0]) / W, (q[1] - a[1]) / H)), 1);
    // Prefer the upper half of the view a little (labels read better above a curve).
    const sc = d + 0.05 * ((q[1] - y0) / H);
    if (sc > score) {
      score = sc;
      best = q;
    }
  }
  if (!best) return null;
  // Move the label 3.5 % of the view to the feasible side (g < 0) of the curve.
  const h = 1e-6 * Math.max(W, H);
  const gx = (g(best[0] + h, best[1]) - g(best[0] - h, best[1])) / (2 * h);
  const gy = (g(best[0], best[1] + h) - g(best[0], best[1] - h)) / (2 * h);
  const n = Math.hypot(gx * W, gy * H) || 1;
  return [best[0] - (0.035 * gx * W * W) / n, best[1] - (0.035 * gy * H * H) / n];
}

/** The feasible-set oracle of a problem with an exact projection, else null. */
export function feasibleOracle(p: ConstrainedProblem): FeasibleSet | null {
  try {
    return new FeasibleSet(p, 'lab');
  } catch {
    return null;
  }
}

/** The step index whose geometry is shown at playhead t: the move into x_k shows while the head travels. */
export function geometryIndex(localT: number, length: number): number {
  if (length <= 0) return 0;
  return Math.max(0, Math.min(length - 1, Math.ceil(localT - 1e-6)));
}

/** End points of the centering stages up to step k (the central path x⋆(t) as sampled). */
export function centralPath(trace: readonly Step[], k: number): Pt[] {
  const out: Pt[] = [];
  for (let j = 1; j <= k && j < trace.length; j++) {
    const next = trace[j + 1];
    const stageEnds = !next || (next.info.outer as number) !== (trace[j].info.outer as number);
    if (stageEnds) out.push(pt(trace[j].x));
  }
  return out;
}

/** Vertex atoms s_0, …, s_{k−1} used by Frank–Wolfe up to step k (unique, in order). */
export function atoms(trace: readonly Step[], k: number): Pt[] {
  const seen = new Set<string>();
  const out: Pt[] = [];
  for (let j = 1; j <= k && j < trace.length; j++) {
    const s = trace[j].info.vertex as number[] | undefined;
    if (!s) continue;
    const key = `${s[0]},${s[1]}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(pt(s));
  }
  return out;
}

const iterLabel = (sym: string, k: number): MathRun[] => [b(sym), sub(String(k))];

/**
 * The line {x : c0 + aᵀ(x − x̄) = 0} clipped to the domain grown by `grow` spans on every side (so
 * it still crosses the view after a pan), or null when a = 0.
 */
export function linearization(
  a: readonly number[],
  c0: number,
  xbar: readonly number[],
  domain: readonly (readonly [number, number])[],
  grow = 1,
): [Pt, Pt] | null {
  const aa = a[0] * a[0] + a[1] * a[1];
  if (!(aa > 0) || !Number.isFinite(c0)) return null;
  const p0: Pt = [xbar[0] - (c0 * a[0]) / aa, xbar[1] - (c0 * a[1]) / aa];
  const n = Math.sqrt(aa);
  const d: Pt = [-a[1] / n, a[0] / n];
  const [[x0, x1], [y0, y1]] = domain;
  const span = grow * Math.max(x1 - x0, y1 - y0);
  const box = [
    [x0 - span, x1 + span],
    [y0 - span, y1 + span],
  ];
  let lo = -Infinity,
    hi = Infinity;
  for (let j = 0; j < 2; j++) {
    if (d[j] === 0) {
      if (p0[j] < box[j][0] || p0[j] > box[j][1]) return null;
      continue;
    }
    const t0 = (box[j][0] - p0[j]) / d[j],
      t1 = (box[j][1] - p0[j]) / d[j];
    lo = Math.max(lo, Math.min(t0, t1));
    hi = Math.min(hi, Math.max(t0, t1));
  }
  if (!(hi > lo)) return null;
  return [
    [p0[0] + lo * d[0], p0[1] + lo * d[1]],
    [p0[0] + hi * d[0], p0[1] + hi * d[1]],
  ];
}

/**
 * The part of the segment a → b inside the box (Liang–Barsky): [a', b'] or null when it misses
 * the box. `clipped` says whether b itself lies outside.
 */
export function clipSegment(
  a: Pt,
  b: Pt,
  box: readonly (readonly [number, number])[],
): { from: Pt; to: Pt; clipped: boolean } | null {
  const d = [b[0] - a[0], b[1] - a[1]];
  let lo = 0,
    hi = 1;
  for (let j = 0; j < 2; j++) {
    if (d[j] === 0) {
      if (a[j] < box[j][0] || a[j] > box[j][1]) return null;
      continue;
    }
    const t0 = (box[j][0] - a[j]) / d[j],
      t1 = (box[j][1] - a[j]) / d[j];
    lo = Math.max(lo, Math.min(t0, t1));
    hi = Math.min(hi, Math.max(t0, t1));
  }
  if (!(hi > lo)) return null;
  return {
    from: [a[0] + lo * d[0], a[1] + lo * d[1]],
    to: [a[0] + hi * d[0], a[1] + hi * d[1]],
    clipped: hi < 1,
  };
}

/** The domain shrunk by `f` of its width and height on every side. */
export function insetBox(
  domain: readonly (readonly [number, number])[],
  f: number,
): [[number, number], [number, number]] {
  const [[x0, x1], [y0, y1]] = domain;
  const w = x1 - x0,
    h = y1 - y0;
  return [
    [x0 + f * w, x1 - f * w],
    [y0 + f * h, y1 - f * h],
  ];
}

const inBox = (q: Pt, box: readonly (readonly [number, number])[]) =>
  q[0] >= box[0][0] && q[0] <= box[0][1] && q[1] >= box[1][0] && q[1] <= box[1][1];

/** Index of the listed minimizer nearest to x (the run's own x⋆ for references λ⋆, f⋆). */
export function nearestMinimizer(p: ConstrainedProblem, x: readonly number[] | undefined): number {
  if (!x || !p.minima.length) return 0;
  let best = 0,
    dist = Infinity;
  p.minima.forEach((m, i) => {
    const d = Math.hypot(m[0] - x[0], m[1] - x[1]);
    if (d < dist) {
      dist = d;
      best = i;
    }
  });
  return best;
}

export type LabelSide = 'right' | 'left' | 'above' | 'below';

/**
 * The box (data units) the "𝐱⋆ = (…)" label covers on each side of 𝐱⋆: about 115 × 16 px at a
 * gap of 10 px, converted with `pxPerUnit` (default: the domain spans ≈ 550 px).
 */
export function labelBox(
  xs: Pt,
  side: LabelSide,
  domain: readonly (readonly [number, number])[],
  pxPerUnit?: number,
): [[number, number], [number, number]] {
  const [[x0, x1], [y0, y1]] = domain;
  const ppu = pxPerUnit && pxPerUnit > 0 ? pxPerUnit : 550 / Math.max(x1 - x0, y1 - y0);
  const L = 115 / ppu,
    h = 18 / ppu,
    g = 10 / ppu;
  switch (side) {
    case 'right':
      return [
        [xs[0] + g, xs[0] + g + L],
        [xs[1] - h / 2, xs[1] + h / 2],
      ];
    case 'left':
      return [
        [xs[0] - g - L, xs[0] - g],
        [xs[1] - h / 2, xs[1] + h / 2],
      ];
    case 'above':
      return [
        [xs[0] - L / 2, xs[0] + L / 2],
        [xs[1] + g, xs[1] + g + h],
      ];
    default:
      return [
        [xs[0] - L / 2, xs[0] + L / 2],
        [xs[1] - g - h, xs[1] - g],
      ];
  }
}

/**
 * Where the "𝐱⋆ = (…)" label goes: the side whose label box crosses the fewest marks (path
 * segments up to the playhead, the KKT arrows, the step's arrows; each with a weight) and stays
 * inside the frame. Ties keep the reading order right, left, above, below.
 */
export function minimizerLabelSide(
  xs: Pt,
  domain: readonly (readonly [number, number])[],
  marks: readonly { from: Pt; to: Pt; weight: number }[],
  pxPerUnit?: number,
): LabelSide {
  const [[x0, x1], [y0, y1]] = domain;
  const order: LabelSide[] = ['right', 'left', 'above', 'below'];
  let best: LabelSide = 'right',
    cost = Infinity;
  order.forEach((side, rank) => {
    const box = labelBox(xs, side, domain, pxPerUnit);
    let c = 0.01 * rank;
    if (box[0][0] < x0 || box[0][1] > x1 || box[1][0] < y0 || box[1][1] > y1) c += 10;
    for (const mk of marks) if (clipSegment(mk.from, mk.to, box)) c += mk.weight;
    if (c < cost) {
      cost = c;
      best = side;
    }
  });
  return best;
}

export interface LinearizedConstraint {
  i: number;
  from: Pt;
  to: Pt;
  /** In the QP's working set at step k (drawn stronger, labelled). */
  working: boolean;
  /** Where its label goes, or null. */
  label: Pt | null;
}

/**
 * SQP step k: the linearized constraints c_i(x_{k−1}) + ∇c_i(x_{k−1})ᵀ(x − x_{k−1}) = 0, clipped
 * to the domain grown by a quarter span, with a label point beside the working-set lines.
 */
export function linearizedConstraints(
  trace: readonly Step[],
  k: number,
  p: ConstrainedProblem,
  domain: readonly (readonly [number, number])[] = p.domain,
): LinearizedConstraint[] {
  const s = trace[k];
  if (!s || k === 0 || !s.info.from) return [];
  const from = pt(s.info.from);
  const cPrev = trace[k - 1].info.constraints as number[];
  const working = new Set((s.info.working_set as number[]) ?? []);
  const [[x0, x1], [y0, y1]] = domain;
  const span = Math.max(x1 - x0, y1 - y0);
  const labelBox = insetBox(domain, 0.08);
  const out: LinearizedConstraint[] = [];
  p.constraints.forEach((c, i) => {
    const a = c.grad([from[0], from[1]]);
    const seg = linearization(a, cPrev[i], from, domain, 0.25);
    if (!seg) return;
    const on = working.has(i);
    let label: Pt | null = null;
    if (on) {
      // Beside the foot of x_{k−1} on the line, on the side where the constraint is violated.
      const aa = a[0] * a[0] + a[1] * a[1];
      const foot: Pt = [from[0] - (cPrev[i] * a[0]) / aa, from[1] - (cPrev[i] * a[1]) / aa];
      const n = Math.sqrt(aa);
      const t: Pt = [-a[1] / n, a[0] / n];
      for (const sgn of [1, -1]) {
        const at = add(foot, t, sgn * 0.2 * span);
        if (!inBox(at, labelBox)) continue;
        label = add(at, [a[0] / n, a[1] / n], 0.03 * span);
        break;
      }
    }
    out.push({ i, from: seg[0], to: seg[1], working: on, label });
  });
  return out;
}

/** Draw linearized constraints as dotted ink lines (no halo: they must not read as a path). */
export function drawLinearized(
  ctx: CanvasRenderingContext2D,
  lines: readonly LinearizedConstraint[],
  toPx: (x: number, y: number) => [number, number],
  ink: string,
): void {
  ctx.save();
  ctx.strokeStyle = ink;
  ctx.lineCap = 'round';
  for (const l of lines) {
    const a = toPx(l.from[0], l.from[1]),
      b = toPx(l.to[0], l.to[1]);
    ctx.globalAlpha = l.working ? 0.8 : 0.4;
    ctx.lineWidth = l.working ? 1.6 : 1.2;
    ctx.setLineDash([0.1, l.working ? 4.5 : 5.5]);
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
  }
  ctx.restore();
}

/** Geometry of step k of `method` (the move from x_{k−1} into x_k), in the method's color slot. */
export function stepGeometry(
  method: string,
  trace: readonly Step[],
  k: number,
  slot: number,
  p: ConstrainedProblem,
  C: FeasibleSet | null,
  domain: readonly (readonly [number, number])[] = p.domain,
): Overlay2D[] {
  const s = trace[k];
  if (!s) return [];
  const info = s.info;
  const out: Overlay2D[] = [];
  const x = pt(s.x);
  if (k === 0) {
    // An infeasible x0 is projected first (projected gradient, Frank–Wolfe, log barrier).
    if (info.projected_from) {
      out.push({
        kind: 'segment',
        from: pt(info.projected_from),
        to: x,
        slot,
        dashed: true,
        width: 1.3,
      });
    }
    return out;
  }
  const from = pt(info.from);
  switch (method) {
    case 'projected_gradient': {
      const g = info.gradient as number[];
      const u = pt(info.unprojected);
      const trials = (info.trials as number[][]) ?? [];
      const sBar = trials[0]?.[0] ?? (info.s as number);
      if (C) {
        // The projection arc s ↦ P_C(x_{k−1} − s∇f(x_{k−1})), s ∈ [0, s̄].
        const arc: Pt[] = [];
        for (let j = 0; j <= 48; j++) {
          try {
            arc.push(
              pt(
                C.project([from[0] - ((sBar * j) / 48) * g[0], from[1] - ((sBar * j) / 48) * g[1]]),
              ),
            );
          } catch {
            break;
          }
        }
        out.push({ kind: 'polyline', points: arc, width: 1.2, alpha: 0.55 });
      }
      out.push({
        kind: 'arrow',
        from,
        to: u,
        width: 1.2,
        alpha: 0.75,
        label: [m('−'), v('s'), m('∇'), v('f')],
      });
      out.push({ kind: 'segment', from: u, to: x, slot, dashed: true, width: 1.5 });
      out.push({ kind: 'point', at: u, shape: 'ring', radius: 3 });
      const arcPts = (info.arc as number[][]) ?? [];
      arcPts
        .slice(0, -1)
        .forEach((a) => out.push({ kind: 'point', at: pt(a), slot, shape: 'ring', radius: 2.5 }));
      break;
    }
    case 'frank_wolfe': {
      const vtx = pt(info.vertex);
      atoms(trace, k).forEach((a) => out.push({ kind: 'point', at: a, slot, radius: 2.5 }));
      out.push({ kind: 'segment', from, to: vtx, slot, dashed: true, width: 1.4 });
      // Label the atom only while it stands apart from x_k and the minimizers (late iterates crowd x⋆).
      const [[dx0, dx1], [dy0, dy1]] = domain;
      const near = (q: readonly number[]) =>
        Math.hypot(q[0] - vtx[0], q[1] - vtx[1]) < 0.06 * Math.max(dx1 - dx0, dy1 - dy0);
      const crowded = near(x) || p.minima.some(near);
      out.push({
        kind: 'point',
        at: vtx,
        slot,
        radius: 4.5,
        ...(crowded ? {} : { label: iterLabel('s', k - 1), labelSide: 'above' as const }),
      });
      break;
    }
    case 'quadratic_penalty':
    case 'augmented_lagrangian':
    case 'log_barrier':
    case 'sqp': {
      const d = info.direction as number[];
      const alpha = (info.alpha as number) ?? 1;
      const full = add(from, d);
      if (method === 'sqp')
        // The lines themselves are drawn dotted by the lab (`linearizedConstraints`); only the
        // labels of the working-set lines are overlays.
        for (const l of linearizedConstraints(trace, k, p, domain))
          if (l.label)
            out.push({
              kind: 'text',
              at: l.label,
              text: [...cLabel(l.i), m(' linearized')],
              align: 'center',
              size: 11,
            });
      if (method !== 'sqp') {
        // Outer iterates: the end point of every stage so far (for the barrier, the central
        // path x⋆(t) as sampled; for the penalty methods, the minimizers x(μ, λ)).
        const path = centralPath(trace, k);
        if (path.length > 1)
          out.push({ kind: 'polyline', points: path, dashed: true, width: 1.1, alpha: 0.7 });
        path.forEach((q) => out.push({ kind: 'point', at: q, shape: 'ring', radius: 3 }));
      }
      const name = method === 'log_barrier' ? [m('Δ'), b('x')] : [b('p'), sub(String(k - 1))];
      // The full step can leave the view by orders of magnitude (SQP far from x⋆): clip it at
      // the frame, where the arrowhead then reads as "continues", and say so in its label.
      const clip = clipSegment(from, full, insetBox(domain, 0.02));
      const to = clip && clip.from[0] === from[0] && clip.from[1] === from[1] ? clip.to : full;
      const clipped = !!clip && clip.clipped && to !== full;
      const label = clipped ? [...name, m(' (clipped)')] : name;
      if (alpha < 1) {
        // The full step (dashed) and the backtracking trials before the accepted one.
        out.push({
          kind: 'arrow',
          from,
          to,
          slot,
          dashed: true,
          width: 1.2,
          alpha: 0.8,
          label,
        });
        const view = insetBox(domain, 0);
        ((info.trials as number[][]) ?? []).slice(0, -1).forEach(([a]) => {
          const q = add(from, d, a);
          if (inBox(q, view)) out.push({ kind: 'point', at: q, slot, shape: 'ring', radius: 2.5 });
        });
      } else {
        out.push({ kind: 'arrow', from, to, slot, width: 1.2, alpha: 0.8, label });
      }
      break;
    }
  }
  return out;
}

export interface KktPicture {
  overlays: Overlay2D[];
  /** Data units per unit of gradient (arrows share one scale). */
  scale: number;
  /** Indices of the constraints whose term λ_i∇c_i is drawn. */
  terms: number[];
}

/**
 * −∇f(x_k) from x_k (ink) and the chain λ_i∇c_i(x_k) tip to tail (method color): at a KKT
 * point the chain ends at the tip of −∇f, because ∇f + Σ λ_i ∇c_i = 0.
 */
export function kktArrows(
  step: Step,
  p: ConstrainedProblem,
  slot: number,
  span: number,
): KktPicture | null {
  const x = step.x as number[];
  if (!Array.isArray(x)) return null;
  const g = p.grad(x);
  if (!g.every(Number.isFinite)) return null;
  const lam = step.info.multipliers as number[] | undefined;
  const gn = Math.hypot(g[0], g[1]);
  const vecs: { i: number; w: Vector }[] = [];
  if (lam) {
    lam.forEach((l, i) => {
      if (!Number.isFinite(l) || Math.abs(l) <= 1e-12 * (1 + gn)) return;
      const a = p.constraints[i].grad(x);
      vecs.push({ i, w: [l * a[0], l * a[1]] });
    });
  }
  const sum = vecs.reduce((s, t) => s + Math.hypot(t.w[0], t.w[1]), 0);
  const M = Math.max(gn, sum);
  if (!(M > 0)) return null;
  // Terms below 3 % of the picture would be invisible stubs with piled-up labels: dropped.
  const shown = vecs.filter((t) => Math.hypot(t.w[0], t.w[1]) >= 0.03 * M);
  const scale = (0.16 * span) / M;
  const origin = pt(x);
  const tip = add(origin, g, -scale);
  // The chain goes first and wider, −∇f on top in ink: where they agree (a KKT point) the
  // arrow reads as one colored arrow with an ink core.
  const chain: Overlay2D[] = [];
  const out: Overlay2D[] = [
    { kind: 'arrow', from: origin, to: tip, width: 1.5 },
    {
      // −∇f is labelled beyond its tip: at a KKT point the chain below covers its shaft.
      kind: 'text',
      at: add(tip, g, (-0.045 * span) / gn),
      text: [m('−∇'), v('f')],
      align: 'center',
    },
  ];
  let at = origin;
  for (const { i, w } of shown) {
    const to = add(at, w, scale);
    chain.push({
      kind: 'arrow',
      from: at,
      to,
      slot,
      width: 3,
      label: [v('λ'), sub(String(i + 1)), m('∇'), v('c'), sub(String(i + 1))],
    });
    at = to;
  }
  return { overlays: [...chain, ...out], scale, terms: shown.map((t) => t.i) };
}

export interface Landscape {
  f: (x: number, y: number) => number;
  /** Cache key of the field (changes with the outer parameters). */
  key: string;
  /** What is drawn, in KaTeX. */
  tex: string;
}

/** A number in TeX: 4 significant digits, 10^{n} for exact powers of ten, ×10ⁿ outside [10⁻³, 10⁵). */
export function texNum(x: number, digits = 4): string {
  if (!Number.isFinite(x)) return Number.isNaN(x) ? '\\text{NaN}' : x > 0 ? '\\infty' : '-\\infty';
  if (x === 0) return '0';
  const e = Math.round(Math.log10(Math.abs(x)));
  if (Math.abs(x - 10 ** e) <= 1e-12 * Math.abs(x) && Math.abs(e) >= 2) return `10^{${e}}`;
  if (Math.abs(x) < 1e-3 || Math.abs(x) >= 1e5) {
    const [mant, ex] = x.toExponential(digits - 1).split('e');
    const mm = mant.replace(/\.?0+$/, '');
    return `${mm === '1' ? '' : mm === '-1' ? '-' : `${mm} \\times `}10^{${Number(ex)}}`;
  }
  return String(Number(x.toPrecision(digits)));
}

/** A vector in TeX: (a,\, b). */
export const texVec = (x: readonly number[], digits = 4) =>
  `(${x.map((t) => texNum(t, digits)).join(',\\, ')})`;

/**
 * The function `method` minimizes to produce step k, as f(x, y), from that step's info. The
 * barrier is shown divided by t (same minimizer, the scale of f): f − (1/t) Σ_I log(−c_i).
 */
export function meritLandscape(
  method: string,
  step: Step | undefined,
  p: ConstrainedProblem,
): Landscape | null {
  if (!step) return null;
  const info = step.info;
  const cons = p.constraints;
  const isEq = cons.map((c) => c.kind === 'eq');
  switch (method) {
    case 'quadratic_penalty': {
      const mu = info.mu as number;
      return {
        key: `Q|${mu}`,
        tex: `Q(\\mathbf{x};\\,\\mu),\\ \\mu = ${texNum(mu)}`,
        f: (x, y) => {
          const z = [x, y];
          let s = 0;
          cons.forEach((c, i) => {
            const val = c.fun(z);
            const r = isEq[i] ? val : Math.max(val, 0);
            s += r * r;
          });
          return p.f(z) + 0.5 * mu * s;
        },
      };
    }
    case 'augmented_lagrangian': {
      const mu = info.mu as number;
      const lam = info.lambda as number[];
      return {
        key: `LA|${mu}|${lam.join(',')}`,
        tex: `L_A(\\mathbf{x};\\boldsymbol\\lambda,\\mu),\\ \\mu = ${texNum(mu)},\\ \\boldsymbol\\lambda = ${texVec(lam, 3)}`,
        f: (x, y) => {
          const z = [x, y];
          let val = p.f(z);
          let ineq = 0;
          cons.forEach((c, i) => {
            const ci = c.fun(z);
            if (isEq[i]) val += lam[i] * ci + 0.5 * mu * ci * ci;
            else {
              const plus = Math.max(lam[i] + mu * ci, 0);
              ineq += plus * plus - lam[i] * lam[i];
            }
          });
          return val + ineq / (2 * mu);
        },
      };
    }
    case 'log_barrier': {
      const t = info.t as number;
      return {
        key: `B|${t}`,
        tex: `B_t = f - \\tfrac{1}{t}\\textstyle\\sum_{I}\\log(-c_i),\\ t = ${texNum(t)}`,
        f: (x, y) => {
          const z = [x, y];
          let s = 0;
          for (let i = 0; i < cons.length; i++) {
            if (isEq[i]) continue;
            const ci = cons[i].fun(z);
            if (!(ci < 0)) return NaN;
            s += Math.log(-ci);
          }
          return p.f(z) - s / t;
        },
      };
    }
    case 'sqp': {
      const mu = info.mu as number;
      return {
        key: `L1|${mu}`,
        tex: `\\phi_1(\\mathbf{x};\\mu) = f + \\mu\\|c^{+}\\|_1,\\ \\mu = ${texNum(mu)}`,
        f: (x, y) => {
          const z = [x, y];
          let s = 0;
          cons.forEach((c, i) => {
            const ci = c.fun(z);
            s += isEq[i] ? Math.abs(ci) : Math.max(ci, 0);
          });
          return p.f(z) + mu * s;
        },
      };
    }
    default:
      return null;
  }
}
