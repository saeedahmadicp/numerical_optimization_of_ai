/**
 * Step geometry of the unconstrained lab, as data: what each method looked at to take the step
 * 𝐱ₖ₋₁ → 𝐱ₖ, read from Step.info (keys as documented in src/numopt/unconstrained/*.py). Pure
 * functions, no canvas: draw.ts renders the result, geometry.test.ts checks it.
 *
 * Step index. While the playhead t moves from k − 1 to k, the head travels along the step that
 * produced 𝐱ₖ, so the geometry of step g = ⌈t⌉ is shown (centered at 𝐱ₖ₋₁, the head moving inside
 * it); paused on an integer k, g = k is the step that led to the head.
 */
import type { Matrix, Problem2D, Step } from '../../core/types';
import { implicitSegments, type Overlay2D } from '../../viz/overlays2d';
import { b, m, sub, sup, v, type MathRun } from '../../viz/mathText';
import { sci, sig } from '../../core/format';
import type { Kind } from './catalog';

export type Pt = readonly [number, number];

/** A set of disjoint segments `[xa, ya, xb, yb, …]` in data coordinates (model level sets). */
export interface Curve {
  segs: number[];
  slot?: number;
  dashed?: boolean;
  width?: number;
  alpha?: number;
}

type EllipseOverlay = Extract<Overlay2D, { kind: 'ellipse' }>;

export interface Geometry {
  /** Drawn first (under the overlays): model and level curves. */
  curves: Curve[];
  overlays: Overlay2D[];
  /** A word for the step (Nelder–Mead's operation), shown in the lens's corner. */
  caption?: string;
}

export interface GeometryInput {
  kind: Kind;
  method: string;
  trace: readonly Step[];
  /** The step shown (see the module comment); 0 shows the start state. */
  g: number;
  params: Readonly<Record<string, unknown>>;
  problem: Problem2D;
  slot: number;
  /** Labels and secondary marks only for the focused method. */
  labels: boolean;
  /** Visible span of the view (data units), for local boxes and minimum sizes. */
  span: number;
}

// ── small vector helpers (2-D) ───────────────────────────────────────────────────────

const isNum = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x);
export const isVec = (x: unknown): x is [number, number] =>
  Array.isArray(x) && x.length === 2 && isNum(x[0]) && isNum(x[1]);
const isMat = (x: unknown): x is Matrix =>
  Array.isArray(x) && x.length === 2 && isVec(x[0]) && isVec(x[1]);

const add = (a: Pt, c: Pt): Pt => [a[0] + c[0], a[1] + c[1]];
const sub2 = (a: Pt, c: Pt): Pt => [a[0] - c[0], a[1] - c[1]];
const mul = (s: number, a: Pt): Pt => [s * a[0], s * a[1]];
const dot = (a: Pt, c: Pt) => a[0] * c[0] + a[1] * c[1];
export const len = (a: Pt) => Math.hypot(a[0], a[1]);
const mv = (M: Matrix, a: Pt): Pt => [
  M[0][0] * a[0] + M[0][1] * a[1],
  M[1][0] * a[0] + M[1][1] * a[1],
];

/** Inverse of a symmetric 2×2 matrix (null when singular or not finite). */
export function inv2(M: Matrix): Matrix | null {
  const det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
  if (!Number.isFinite(det) || Math.abs(det) < 1e-300) return null;
  const s = (M[0][1] + M[1][0]) / 2;
  const out = [
    [M[1][1] / det, -s / det],
    [-s / det, M[0][0] / det],
  ];
  return out.flat().every(Number.isFinite) ? out : null;
}

/** Symmetric positive definite (Sylvester's criterion on the symmetrized matrix). */
export function isSpd(M: Matrix): boolean {
  const s = (M[0][1] + M[1][0]) / 2;
  return M[0][0] > 0 && M[0][0] * M[1][1] - s * s > 0;
}

const pt = (x: unknown): Pt | null => (isVec(x) ? [x[0], x[1]] : null);
const xOf = (s: Step | undefined): Pt | null => (s ? pt(s.x) : null);

// ── step index ───────────────────────────────────────────────────────────────────────

/**
 * The step whose geometry is shown at local time `lt` of a trace with `n` steps, and how far the
 * head has travelled along it (u ∈ (0, 1]; 1 when paused on the step).
 */
export function stepAt(lt: number, n: number): { g: number; u: number } {
  if (n <= 1 || lt <= 1e-9) return { g: 0, u: 1 };
  const g = Math.min(n - 1, Math.ceil(lt - 1e-9));
  return { g, u: Math.min(1, Math.max(0, lt - (g - 1))) };
}

// ── the line search of a step (φ panel and canvas) ───────────────────────────────────

export interface Ray {
  origin: Pt;
  dir: Pt;
  /** Accepted step length (null when the search failed). */
  alpha: number | null;
  /** Every (α, φ(α)) the search evaluated, in order. */
  trials: readonly (readonly [number, number])[];
}

/** The line search behind step g, for methods that step along a direction (null otherwise). */
export function rayOf(kind: Kind, trace: readonly Step[], g: number): Ray | null {
  if (g < 1 || g >= trace.length) return null;
  const prev = trace[g - 1];
  const cur = trace[g];
  const origin = xOf(prev);
  if (!origin) return null;
  let dir: Pt | null = null;
  if (kind === 'fista') {
    // FISTA steps from the extrapolated point 𝐲ₖ along −∇f(𝐲ₖ); its backtracking tries
    // α = 1/L̄ (info.trials holds [L̄, f]).
    const y = pt(cur.info.y);
    const gy = pt(cur.info.grad_y);
    const a = cur.info.alpha;
    if (!y || !gy || len(gy) === 0) return null;
    const raw = Array.isArray(cur.info.trials) ? (cur.info.trials as unknown[]) : [];
    const trials = raw
      .filter(
        (t): t is [number, number] =>
          Array.isArray(t) && isNum(t[0]) && t[0] > 0 && typeof t[1] === 'number',
      )
      .map(([L, f]) => [1 / L, f] as const);
    return { origin: y, dir: [-gy[0], -gy[1]], alpha: isNum(a) ? a : null, trials };
  }
  if (kind === 'cg') dir = pt(prev.info.direction);
  else if (
    kind === 'line' ||
    kind === 'coord' ||
    kind === 'newton' ||
    kind === 'qn' ||
    kind === 'schedule'
  )
    dir = pt(cur.info.direction);
  if (!dir || len(dir) === 0) return null;
  const a = cur.info.alpha;
  const raw = Array.isArray(cur.info.trials) ? (cur.info.trials as unknown[]) : [];
  const trials = raw.filter(
    (t): t is [number, number] => Array.isArray(t) && isNum(t[0]) && typeof t[1] === 'number',
  );
  return { origin, dir, alpha: isNum(a) ? a : null, trials };
}

// ── local level sets ─────────────────────────────────────────────────────────────────

const SEG_CACHE = new WeakMap<object, Map<string, number[]>>();

/** Level set g = 0 in the square of half-width `h` around `c`, cached per step object. */
function localCurve(
  owner: object,
  key: string,
  gfun: (x: number, y: number) => number,
  c: Pt,
  h: number,
  n = 84,
): number[] {
  let map = SEG_CACHE.get(owner);
  if (!map) SEG_CACHE.set(owner, (map = new Map()));
  const k = `${key}|${c[0]},${c[1]},${h}`;
  const hit = map.get(k);
  if (hit) return hit;
  const segs =
    h > 0 && Number.isFinite(h)
      ? implicitSegments(gfun, [c[0] - h, c[0] + h], [c[1] - h, c[1] + h], n, n, 0)
      : [];
  map.set(k, segs);
  return segs;
}

/** The quadratic model m(𝐱) = gᵀ(𝐱 − 𝐜) + ½(𝐱 − 𝐜)ᵀB(𝐱 − 𝐜) (without the constant f(𝐜)). */
function model(c: Pt, gr: Pt, B: Matrix) {
  return (x: number, y: number) => {
    const d: Pt = [x - c[0], y - c[1]];
    return dot(gr, d) + 0.5 * dot(d, mv(B, d));
  };
}

/**
 * The level set of the quadratic model through `through`. With B ≻ 0 it is the ellipse centered
 * at the model's minimizer (exact, as an overlay); otherwise a hyperbola or lines, traced in a
 * square of half-width `h` around 𝐜.
 */
function modelLevel(
  owner: object,
  key: string,
  c: Pt,
  gr: Pt,
  B: Matrix,
  through: Pt,
  h: number,
  slot: number,
  dashed = false,
  width = 1.4,
): { overlay?: EllipseOverlay; curve?: Curve; center?: Pt } {
  const Bi = isSpd(B) ? inv2(B) : null;
  if (Bi) {
    const center = sub2(c, mv(Bi, gr));
    const d = sub2(through, center);
    const r2 = dot(d, mv(B, d));
    if (!(r2 > 0) || !Number.isFinite(r2)) return { center };
    return {
      center,
      overlay: {
        kind: 'ellipse',
        center,
        matrix: B,
        radius: Math.sqrt(r2),
        slot,
        dashed,
        width,
      },
    };
  }
  const mf = model(c, gr, B);
  const level = mf(through[0], through[1]);
  const segs = localCurve(owner, key, (x, y) => mf(x, y) - level, c, h);
  return { curve: { segs, slot, dashed, width } };
}

// ── labels ───────────────────────────────────────────────────────────────────────────

const vecSup = (name: string, s: string): MathRun[] => [b(name), sup(s)];

// ── builders per kind ────────────────────────────────────────────────────────────────

const EMPTY: Geometry = { curves: [], overlays: [] };

/** Ray from the origin through every trial; rejected trials as small rings. */
function rayMarks(ray: Ray, slot: number, labels: boolean): Overlay2D[] {
  const out: Overlay2D[] = [];
  const alphas = ray.trials.map((t) => t[0]).concat(ray.alpha !== null ? [ray.alpha] : []);
  const far = Math.max(0, ...alphas);
  if (far > 0)
    out.push({
      kind: 'segment',
      from: ray.origin,
      to: add(ray.origin, mul(far, ray.dir)),
      slot,
      dashed: true,
      width: 1,
      alpha: 0.85,
    });
  if (labels)
    ray.trials.forEach(([a]) => {
      if (ray.alpha !== null && a === ray.alpha) return;
      out.push({
        kind: 'point',
        at: add(ray.origin, mul(a, ray.dir)),
        slot,
        shape: 'ring',
        radius: 2.6,
      });
    });
  return out;
}

function lineGeometry(i: GeometryInput, owner: Step): Geometry {
  const ray = rayOf(i.kind, i.trace, i.g);
  if (!ray) return EMPTY;
  const curves: Curve[] = [];
  const overlays = rayMarks(ray, i.slot, i.labels);
  // Exact steps on a quadratic: the step ends where the ray touches a level set of f.
  if (i.method === 'gradient_descent' && i.params.step_rule === 'exact_quadratic' && i.labels) {
    const x1 = xOf(i.trace[i.g]);
    if (x1) {
      const f1 = i.problem.f([x1[0], x1[1]]);
      const h = Math.max(len(sub2(x1, ray.origin)) * 1.25, i.span * 0.05);
      curves.push({
        segs: localCurve(owner, 'level', (x, y) => i.problem.f([x, y]) - f1, x1, h),
        width: 1.1,
        alpha: 0.75,
      });
    }
  }
  return { curves, overlays };
}

function coordGeometry(i: GeometryInput): Geometry {
  const cur = i.trace[i.g];
  const x0 = xOf(i.trace[i.g - 1]);
  const c = cur.info.coordinate;
  if (!x0 || typeof c !== 'number') return EMPTY;
  const e: Pt = c === 0 ? [1, 0] : [0, 1];
  const h = i.span * 0.18;
  // The coordinate axis through 𝐱ₖ₋₁ (a curve, so the lens frames the step, not the axis).
  const a0 = add(x0, mul(-h, e));
  const a1 = add(x0, mul(h, e));
  const curves: Curve[] = [
    { segs: [a0[0], a0[1], a1[0], a1[1]], slot: i.slot, dashed: true, width: 1, alpha: 0.7 },
  ];
  const overlays: Overlay2D[] = [];
  const ray = rayOf('coord', i.trace, i.g);
  if (ray) overlays.push(...rayMarks(ray, i.slot, i.labels));
  if (i.labels)
    overlays.push({
      kind: 'text',
      at: add(x0, mul(h, e)),
      text: [b('e'), sub(String(c + 1))],
      align: 'left',
    });
  return { curves, overlays };
}

function heavyGeometry(i: GeometryInput): Geometry {
  const x0 = xOf(i.trace[i.g - 1]);
  const x1 = xOf(i.trace[i.g]);
  const vPrev = pt(i.trace[i.g - 1].info.velocity);
  const beta = Number(i.params.beta);
  if (!x0 || !x1 || !vPrev || !Number.isFinite(beta)) return EMPTY;
  const mid = add(x0, mul(beta, vPrev));
  return {
    curves: [],
    overlays: [
      {
        kind: 'arrow',
        from: x0,
        to: mid,
        slot: i.slot,
        dashed: true,
        width: 1.3,
        label: i.labels ? [v('β'), b('v'), sub('k−1', 'italic')] : undefined,
      },
      {
        kind: 'arrow',
        from: mid,
        to: x1,
        width: 1.3,
        label: i.labels ? [m('−'), v('α'), m('∇'), v('f')] : undefined,
      },
    ],
  };
}

function nesterovGeometry(i: GeometryInput): Geometry {
  const x0 = xOf(i.trace[i.g - 1]);
  const x1 = xOf(i.trace[i.g]);
  const la = pt(i.trace[i.g].info.lookahead);
  if (!x0 || !x1 || !la) return EMPTY;
  const overlays: Overlay2D[] = [
    {
      kind: 'arrow',
      from: x0,
      to: la,
      slot: i.slot,
      dashed: true,
      width: 1.3,
      label: i.labels ? [v('μ'), b('v'), sub('k−1', 'italic')] : undefined,
    },
    { kind: 'arrow', from: la, to: x1, width: 1.3 },
  ];
  if (i.labels)
    overlays.push({
      kind: 'point',
      at: la,
      slot: i.slot,
      shape: 'ring',
      radius: 3.2,
      label: [m('look-ahead')],
      labelSide: 'right',
    });
  return { curves: [], overlays };
}

const U_NAME: Record<string, string> = {
  adam: 'm̂',
  adamw: 'm̂',
  nadam: 'm̄',
  adamax: 'm',
  amsgrad: 'm',
};

/** The vector the per-coordinate step multiplies (see first_order.py, "Info keys"). */
export function adaptiveMultiplicand(method: string, trace: readonly Step[], g: number): Pt | null {
  const cur = trace[g].info;
  switch (method) {
    case 'adagrad':
    case 'rmsprop':
    case 'adadelta':
      return pt(trace[g - 1].info.grad);
    case 'adam':
    case 'adamw':
      return pt(cur.m_hat);
    case 'nadam':
      return pt(cur.m_bar);
    case 'adamax':
    case 'amsgrad':
      return pt(cur.m);
    default:
      return null;
  }
}

/**
 * Adaptive methods move by −D𝐮 with the diagonal D = diag(lr_eff) and the multiplicand 𝐮 (the
 * gradient or a moment). The ellipse {𝐱ₖ₋₁ − D𝐰 : ‖𝐰‖ = ‖𝐮‖} is where the step could land for
 * every 𝐮 of the same length: its axes show the per-coordinate step sizes, and the dashed ink
 * arrow (−𝐮, scaled to the step's length) shows how far D turns the step away from −𝐮.
 */
function adaptiveGeometry(i: GeometryInput): Geometry {
  const x0 = xOf(i.trace[i.g - 1]);
  const lr = pt(i.trace[i.g].info.lr_eff);
  const u = adaptiveMultiplicand(i.method, i.trace, i.g);
  if (!x0 || !lr || !u) return EMPTY;
  const nu = len(u);
  const a: Pt = [Math.abs(lr[0]) * nu, Math.abs(lr[1]) * nu];
  const step = mul(-1, [lr[0] * u[0], lr[1] * u[1]]);
  const overlays: Overlay2D[] = [];
  if (a[0] > 0 && a[1] > 0 && Math.max(a[0], a[1]) < i.span * 40)
    overlays.push({
      kind: 'ellipse',
      center: x0,
      matrix: [
        [1 / (a[0] * a[0]), 0],
        [0, 1 / (a[1] * a[1])],
      ],
      slot: i.slot,
      fill: true,
      width: 1.2,
    });
  const sl = len(step);
  if (i.labels && sl > 0 && nu > 0) {
    overlays.push({
      kind: 'arrow',
      from: x0,
      to: add(x0, mul(-sl / nu, u)),
      dashed: true,
      width: 1.1,
      label: [m('−'), b(U_NAME[i.method] ?? 'g')],
    });
    // AdamW: the decoupled decay is the last leg of the step.
    const decay = pt(i.trace[i.g].info.decay);
    const x1 = xOf(i.trace[i.g]);
    if (decay && x1 && len(decay) > 0)
      overlays.push({ kind: 'arrow', from: add(x0, step), to: x1, width: 1.1 });
  }
  return { curves: [], overlays };
}

function newtonGeometry(i: GeometryInput): Geometry {
  const prev = i.trace[i.g - 1];
  const cur = i.trace[i.g];
  const x0 = xOf(prev);
  const gr = pt(prev.info.grad);
  const H0 = prev.info.hess;
  const p = pt(cur.info.direction);
  if (!x0 || !gr || !isMat(H0) || !p) return EMPTY;
  const tau = isNum(cur.info.tau) ? cur.info.tau : 0;
  const B: Matrix = [
    [H0[0][0] + tau, H0[0][1]],
    [H0[1][0], H0[1][1] + tau],
  ];
  const steepest = cur.info.direction_type === 'steepest';
  const overlays: Overlay2D[] = [];
  const curves: Curve[] = [];
  if (!steepest) {
    const h = Math.max(2.2 * len(p), i.span * 0.12);
    const lvl = modelLevel(cur, 'model', x0, gr, B, x0, h, i.slot, false, 1.4);
    if (lvl.overlay) overlays.push(lvl.overlay);
    if (lvl.curve) curves.push(lvl.curve);
    const np = add(x0, p);
    if (i.labels)
      overlays.push({
        kind: 'point',
        at: np,
        slot: i.slot,
        shape: 'cross',
        radius: 4,
        label: vecSup('x', 'N'),
        labelSide: 'right',
      });
  }
  const ray = rayOf('newton', i.trace, i.g);
  if (ray && i.method !== 'pure_newton') overlays.push(...rayMarks(ray, i.slot, i.labels));
  return { curves, overlays };
}

function qnGeometry(i: GeometryInput): Geometry {
  const prev = i.trace[i.g - 1];
  const cur = i.trace[i.g];
  const x0 = xOf(prev);
  const gr = pt(prev.info.grad);
  const p = pt(cur.info.direction);
  if (!x0 || !gr || !p) return EMPTY;
  const Hraw =
    cur.info.reset === true
      ? [
          [1, 0],
          [0, 1],
        ]
      : prev.info.H;
  const overlays: Overlay2D[] = [];
  const curves: Curve[] = [];
  const h = Math.max(2.2 * len(p), i.span * 0.12);
  // The true Hessian's model, dashed ink: how far the estimate is from ∇²f.
  if (i.labels && i.problem.hess) {
    const Ht = i.problem.hess([x0[0], x0[1]]);
    if (isMat(Ht) && isSpd(Ht)) {
      const t = modelLevel(cur, 'true', x0, gr, Ht, x0, h, i.slot, true, 1);
      if (t.overlay) overlays.push({ ...t.overlay, slot: undefined, alpha: 0.6 });
    }
  }
  if (isMat(Hraw)) {
    const B = inv2(Hraw);
    if (B) {
      const lvl = modelLevel(cur, 'qn', x0, gr, B, x0, h, i.slot, false, 1.5);
      if (lvl.overlay) overlays.push({ ...lvl.overlay, fill: true });
      if (lvl.curve) curves.push(lvl.curve);
    }
  }
  const ray = rayOf('qn', i.trace, i.g);
  if (ray) overlays.push(...rayMarks(ray, i.slot, i.labels));
  return { curves, overlays };
}

function cgGeometry(i: GeometryInput, owner: Step): Geometry {
  const prev = i.trace[i.g - 1];
  const x0 = xOf(prev);
  const ray = rayOf('cg', i.trace, i.g);
  if (!x0 || !ray || ray.alpha === null) return EMPTY;
  const overlays: Overlay2D[] = [];
  const curves: Curve[] = [];
  const a = ray.alpha;
  const beta = prev.info.beta;
  const dPrev = i.g >= 2 ? pt(i.trace[i.g - 2].info.direction) : null;
  const gr = i.problem.grad([x0[0], x0[1]]);
  // d = −∇f + β d_prev, drawn tip to tail at the scale α of the accepted step.
  if (isNum(beta) && beta !== 0 && dPrev && isVec(gr)) {
    const mid = add(x0, mul(-a, gr as Pt));
    overlays.push(
      {
        kind: 'arrow',
        from: x0,
        to: mid,
        dashed: true,
        width: 1.1,
        label: i.labels ? [m('−'), v('α'), m('∇'), v('f')] : undefined,
      },
      {
        kind: 'arrow',
        from: mid,
        to: add(mid, mul(a * beta, dPrev)),
        slot: i.slot,
        dashed: true,
        width: 1.3,
        label: i.labels
          ? [v('α'), v('β'), sub('k−1', 'italic'), b('d'), sub('k−2', 'italic')]
          : undefined,
      },
    );
  }
  // The level set the previous line search ended on: the previous direction 𝐝ₖ₋₂ is (nearly)
  // tangent to it at 𝐱ₖ₋₁ (an exact search ends where ∇f(𝐱ₖ₋₁)ᵀ𝐝ₖ₋₂ = 0).
  if (i.labels && i.g >= 2) {
    const f0 = i.problem.f([x0[0], x0[1]]);
    const hgt = Math.max(len(mul(a, ray.dir)) * 1.2, i.span * 0.06);
    curves.push({
      segs: localCurve(owner, 'level', (x, y) => i.problem.f([x, y]) - f0, x0, hgt),
      width: 1.1,
      alpha: 0.75,
    });
  }
  overlays.push(...rayMarks(ray, i.slot, i.labels));
  return { curves, overlays };
}

function trGeometry(i: GeometryInput): Geometry {
  const s = i.trace[i.g];
  const info = s.info;
  const c = pt(info.center);
  const r = info.radius;
  if (!c || !isNum(r)) return EMPTY;
  const overlays: Overlay2D[] = [
    { kind: 'disk', center: c, radius: r, slot: i.slot, fill: true, width: 1.4 },
  ];
  const curves: Curve[] = [];
  if (i.g === 0) return { curves, overlays };
  const gr = pt(info.grad);
  const H = info.H;
  const trial = pt(info.trial_point);
  if (gr && isMat(H) && trial) {
    const lvl = modelLevel(s, 'tr', c, gr, H, trial, r * 1.9, i.slot, false, 1.1);
    if (lvl.overlay) overlays.push({ ...lvl.overlay, alpha: 0.75 });
    if (lvl.curve) curves.push({ ...lvl.curve, alpha: 0.75 });
  }
  const nr = info.new_radius;
  if (isNum(nr) && Math.abs(nr - r) > 1e-12 * Math.max(1, r))
    overlays.push({
      kind: 'disk',
      center: c,
      radius: nr,
      slot: i.slot,
      dashed: true,
      width: 1,
      alpha: 0.7,
    });
  if (i.labels) {
    const path = Array.isArray(info.dogleg_path) ? (info.dogleg_path as unknown[]).map(pt) : null;
    if (path && path.every(Boolean)) {
      overlays.push({
        kind: 'polyline',
        points: path as Pt[],
        slot: i.slot,
        dashed: true,
        width: 1.2,
      });
      overlays.push({
        kind: 'point',
        at: path[1]!,
        slot: i.slot,
        radius: 2.6,
        label: vecSup('p', 'U'),
        labelSide: 'left',
      });
    }
    const cg = Array.isArray(info.cg_path) ? (info.cg_path as unknown[]).map(pt) : null;
    if (cg && cg.length > 1 && cg.every(Boolean)) {
      overlays.push({ kind: 'polyline', points: cg as Pt[], slot: i.slot, width: 1.2 });
      for (const q of (cg as Pt[]).slice(1, -1))
        overlays.push({ kind: 'point', at: q, slot: i.slot, radius: 2.2 });
    }
    const cp = pt(info.cauchy_point);
    // Label the Cauchy point only where it stands clear of the center and of 𝐩ᵁ.
    const clear = (q: Pt | null) => !q || !cp || len(sub2(q, cp)) > i.span * 0.03;
    const pu = path && path.every(Boolean) ? (path[1] as Pt) : null;
    if (cp)
      overlays.push({
        kind: 'point',
        at: cp,
        radius: 3,
        shape: 'ring',
        label: clear(c) && clear(pu) ? vecSup('p', 'C') : undefined,
        labelSide: 'below',
      });
    const np = pt(info.newton_point);
    if (np)
      overlays.push({
        kind: 'point',
        at: np,
        slot: i.slot,
        shape: 'cross',
        radius: 4,
        label: vecSup('p', 'B'),
        labelSide: 'right',
      });
  }
  if (info.accepted === false && trial) {
    overlays.push({ kind: 'segment', from: c, to: trial, slot: i.slot, dashed: true, width: 1.2 });
    overlays.push({ kind: 'point', at: trial, slot: i.slot, shape: 'ring', radius: 3.5 });
    if (i.labels)
      overlays.push({
        kind: 'text',
        at: trial,
        text: [m('  '), v('ρ'), m(` = ${fmt(info.rho)} ≤ `), v('η'), m(', rejected')],
        align: 'left',
      });
  }
  return { curves, overlays };
}

// ── the promoted methods: schedules, OGM, FISTA, Anderson, ARC, regularized Newton ────

/**
 * Marks on the path up to step g: the certified checkpoints of a schedule (filled dots) or the
 * restarts of FISTA (ink rings), so the reader sees where the theorem speaks or the momentum
 * was reset while the path is drawn.
 */
function pathMarksUpTo(
  i: GeometryInput,
  test: (s: Step) => boolean,
  mark: (at: Pt, j: number, last: boolean) => Overlay2D,
): Overlay2D[] {
  const out: Overlay2D[] = [];
  let lastJ = -1;
  for (let j = 1; j <= i.g; j++) if (test(i.trace[j])) lastJ = j;
  for (let j = 1; j <= i.g; j++) {
    if (!test(i.trace[j])) continue;
    const at = xOf(i.trace[j]);
    if (at) out.push(mark(at, j, j === lastJ));
  }
  return out;
}

const certified = (s: Step) => isNum(s.info.bound_f) || isNum(s.info.bound_dist);

function checkpointMarks(i: GeometryInput): Overlay2D[] {
  return pathMarksUpTo(i, certified, (at) => ({
    kind: 'point',
    at,
    slot: i.slot,
    shape: 'dot',
    radius: 3.2,
  }));
}

/** A schedule step x_{k−1} → x_k = x_{k−1} − (h/L)∇f, and where the classical step 1/L ends. */
function scheduleGeometry(i: GeometryInput): Geometry {
  const cur = i.trace[i.g];
  const x0 = xOf(i.trace[i.g - 1]);
  const x1 = xOf(cur);
  const p = pt(cur.info.direction);
  const a = cur.info.alpha;
  const h = cur.info.h;
  if (!x0 || !x1 || !p || !isNum(a) || !isNum(h) || h <= 0) return EMPTY;
  const overlays: Overlay2D[] = [...checkpointMarks(i)];
  const unit = add(x0, mul(a / h, p)); // x_{k−1} − ∇f/L
  overlays.push({
    kind: 'arrow',
    from: x0,
    to: x1,
    slot: i.slot,
    width: 1.5,
    label: i.labels ? [v('h'), m(` = ${fmt(h)}`)] : undefined,
  });
  // Where the classical step 1/L would end; labelled only when it stands clear of x_k (h ≥ 1.5),
  // so the two labels never collide.
  if (i.labels && h >= 1.1)
    overlays.push({
      kind: 'point',
      at: unit,
      shape: 'ring',
      radius: 3,
      label: h >= 1.5 ? [m('1/'), v('L')] : undefined,
      labelSide: 'below',
    });
  return { curves: [], overlays };
}

/** OGM: the gradient step x_{k−1} → y_k (length 1/L), then the momentum terms y_k → x_k. */
function ogmGeometry(i: GeometryInput): Geometry {
  const cur = i.trace[i.g];
  const x0 = xOf(i.trace[i.g - 1]);
  const x1 = xOf(cur);
  const y = pt(cur.info.y);
  if (!x0 || !x1 || !y) return EMPTY;
  const overlays: Overlay2D[] = [...checkpointMarks(i)];
  overlays.push(
    {
      kind: 'arrow',
      from: x0,
      to: y,
      width: 1.3,
      label: i.labels ? [m('−∇'), v('f'), m('/'), v('L')] : undefined,
    },
    { kind: 'arrow', from: y, to: x1, slot: i.slot, dashed: true, width: 1.3 },
  );
  if (i.labels)
    overlays.push({
      kind: 'point',
      at: y,
      slot: i.slot,
      shape: 'ring',
      radius: 3.2,
      label: [b('y'), sub(String(i.g), 'italic')],
      labelSide: 'left',
    });
  return { curves: [], overlays };
}

/**
 * FISTA: the momentum β(x_{k−1} − x_{k−2}) to y_k, the backtracked gradient step from y_k, and
 * every restart so far as an ink ring on the path.
 */
function fistaGeometry(i: GeometryInput): Geometry {
  const cur = i.trace[i.g];
  const x0 = xOf(i.trace[i.g - 1]);
  const x1 = xOf(cur);
  const y = pt(cur.info.y);
  if (!x0 || !x1 || !y) return EMPTY;
  // A restart at x_{k−1} makes y_k = x_{k−1}: its label then joins y_k's.
  const restartedHere = i.trace[i.g - 1].info.restarted === true;
  const overlays: Overlay2D[] = pathMarksUpTo(
    i,
    (s) => s.info.restarted === true,
    (at, j, last) => ({
      kind: 'point',
      at,
      shape: 'ring',
      radius: 5,
      label: i.labels && last && j !== i.g - 1 ? [m('restart')] : undefined,
      labelSide: 'above',
    }),
  );
  const beta = cur.info.beta;
  if (isNum(beta) && beta > 0 && len(sub2(y, x0)) > 0)
    overlays.push({
      kind: 'arrow',
      from: x0,
      to: y,
      slot: i.slot,
      dashed: true,
      width: 1.3,
    });
  const ray = rayOf('fista', i.trace, i.g);
  if (ray && i.labels)
    for (const [a] of ray.trials) {
      if (ray.alpha !== null && a === ray.alpha) continue;
      overlays.push({
        kind: 'point',
        at: add(ray.origin, mul(a, ray.dir)),
        slot: i.slot,
        shape: 'ring',
        radius: 2.6,
      });
    }
  overlays.push({
    kind: 'arrow',
    from: y,
    to: x1,
    width: 1.3,
    label: i.labels ? [m('−∇'), v('f'), m('('), b('y'), m(')/'), v('L')] : undefined,
  });
  if (i.labels)
    overlays.push({
      kind: 'point',
      at: y,
      slot: i.slot,
      shape: 'ring',
      radius: 3.2,
      label: restartedHere
        ? [b('y'), sub(String(i.g), 'italic'), m(' (restart)')]
        : [b('y'), sub(String(i.g), 'italic')],
      // The longer label goes right, into the lens, not past its left edge.
      labelSide: restartedHere ? 'right' : 'left',
    });
  return { curves: [], overlays };
}

/**
 * AA(m): the history x_{k−1−m}, …, x_{k−1} (ring size = |weight|), the mix x̄ = Σcᵢxᵢ
 * (where the weights put the iterate before the gradient correction) and the step.
 */
function andersonGeometry(i: GeometryInput): Geometry {
  const cur = i.trace[i.g];
  const x0 = xOf(i.trace[i.g - 1]);
  const x1 = xOf(cur);
  const hist = Array.isArray(cur.info.history) ? (cur.info.history as unknown[]).map(pt) : [];
  const c = Array.isArray(cur.info.coefficients) ? (cur.info.coefficients as unknown[]) : [];
  if (!x0 || !x1 || !hist.every(Boolean)) return EMPTY;
  const pts = hist as Pt[];
  const overlays: Overlay2D[] = [];
  // The history points are the path's own last iterates: rings, no extra line.
  const few = pts.length <= 6;
  pts.forEach((q, j) => {
    const w = isNum(c[j]) ? c[j] : 0;
    overlays.push({
      kind: 'point',
      at: q,
      slot: i.slot,
      shape: 'ring',
      radius: 2.2 + 2.8 * Math.min(1, Math.abs(w)),
      label: i.labels && few && pts.length > 1 ? [m(fmt(w))] : undefined,
      labelSide: 'right',
    });
  });
  const xb = pt(cur.info.x_bar);
  if (xb && pts.length > 1) {
    overlays.push({
      kind: 'arrow',
      from: xb,
      to: x1,
      dashed: true,
      width: 1.1,
    });
    if (i.labels)
      overlays.push({
        kind: 'point',
        at: xb,
        shape: 'cross',
        radius: 4,
        label: [b('x̄')],
        labelSide: 'left',
      });
  }
  overlays.push({ kind: 'arrow', from: x0, to: x1, slot: i.slot, width: 1.5 });
  return { curves: [], overlays };
}

/** m(d) − f = gᵀd + ½dᵀHd + (σ/3)‖d‖³ around the center c, as a function of the point. */
function cubicModel(c: Pt, gr: Pt, H: Matrix, sigma: number) {
  return (x: number, y: number) => {
    const d: Pt = [x - c[0], y - c[1]];
    return dot(gr, d) + 0.5 * dot(d, mv(H, d)) + (sigma / 3) * len(d) ** 3;
  };
}

/**
 * ARC: the cubic model's level set through x_{k−1}, and the ball ‖s‖ = λ/σ that the global model
 * minimizer lies on (the implicit trust region), with the Cauchy and Newton points.
 */
function arcGeometry(i: GeometryInput): Geometry {
  const s = i.trace[i.g];
  const info = s.info;
  const c = pt(info.center);
  const r = info.step_norm;
  const sigma = info.sigma;
  if (!c || !isNum(r) || !isNum(sigma)) return EMPTY;
  // A ball much larger than the view (a rejected long step) is outlined only: filled, it would
  // tint the whole landscape.
  const overlays: Overlay2D[] = [
    { kind: 'disk', center: c, radius: r, slot: i.slot, fill: r < 0.5 * i.span, width: 1.2 },
  ];
  const curves: Curve[] = [];
  const gr = pt(info.grad);
  const H = info.H;
  const trial = pt(info.trial_point);
  // The model's level set through x_{k−1} (m = f(x_{k−1})): it encloses every step that the model
  // predicts to decrease f, and the trial point, the model's global minimizer, lies inside it.
  if (gr && isMat(H) && r > 0) {
    const mf = cubicModel(c, gr, H, sigma);
    curves.push({
      segs: localCurve(s, 'arc', mf, c, Math.max(r * 2.5, i.span * 0.04)),
      slot: i.slot,
      width: 1.1,
      alpha: 0.75,
    });
  }
  if (i.labels) {
    const cp = pt(info.cauchy_point);
    if (cp && (!trial || len(sub2(cp, trial)) > i.span * 0.01))
      overlays.push({
        kind: 'point',
        at: cp,
        radius: 3,
        shape: 'ring',
        label: [b('s'), sup('C')],
        labelSide: 'below',
      });
    const np = pt(info.newton_point);
    if (np && len(sub2(np, c)) < Math.max(4 * r, i.span * 0.05))
      overlays.push({
        kind: 'point',
        at: np,
        slot: i.slot,
        shape: 'cross',
        radius: 4,
        label: vecSup('x', 'N'),
        labelSide: 'right',
      });
  }
  if (trial) {
    if (info.accepted === false) {
      overlays.push({
        kind: 'segment',
        from: c,
        to: trial,
        slot: i.slot,
        dashed: true,
        width: 1.2,
      });
      overlays.push({ kind: 'point', at: trial, slot: i.slot, shape: 'ring', radius: 3.5 });
      // The verdict sits at the trial point, or on the way to it when the trial is far off view.
      const far = len(sub2(trial, c)) > 0.45 * i.span;
      const at = far ? add(c, mul((0.3 * i.span) / len(sub2(trial, c)), sub2(trial, c))) : trial;
      if (i.labels)
        overlays.push({
          kind: 'text',
          at,
          text: [m('  '), v('ρ'), m(` = ${fmt(info.rho)} < `), v('η'), sub('1'), m(', rejected')],
          align: 'left',
        });
    } else overlays.push({ kind: 'arrow', from: c, to: trial, slot: i.slot, width: 1.5 });
  }
  return { curves, overlays };
}

/** x − (H + μI)⁻¹g for a symmetric 2×2 H (null when H + μI is singular). */
function shifted(x: Pt, gr: Pt, H: Matrix, mu: number): Pt | null {
  const Bi = inv2([
    [H[0][0] + mu, H[0][1]],
    [H[1][0], H[1][1] + mu],
  ]);
  return Bi ? sub2(x, mv(Bi, gr)) : null;
}

/**
 * Regularized Newton: the model with ∇²f + λI (its level set through x_{k−1} is centered at
 * x_k), the Newton point (λ = 0), the path λ ↦ x_{k−1} − (∇²f + λI)⁻¹∇f from it toward x_{k−1},
 * and the rejected trials of the adaptive search on that path.
 */
function regNewtonGeometry(i: GeometryInput): Geometry {
  const prev = i.trace[i.g - 1];
  const cur = i.trace[i.g];
  const x0 = xOf(prev);
  const x1 = xOf(cur);
  const gr = pt(prev.info.grad);
  const H0 = prev.info.hess;
  const lam = cur.info.lambda;
  const s = pt(cur.info.direction);
  if (!x0 || !x1 || !gr || !isMat(H0) || !isNum(lam) || !s) return EMPTY;
  const hs = (H0[0][1] + H0[1][0]) / 2;
  const H: Matrix = [
    [H0[0][0], hs],
    [hs, H0[1][1]],
  ];
  const B: Matrix = [
    [H[0][0] + lam, hs],
    [hs, H[1][1] + lam],
  ];
  const overlays: Overlay2D[] = [];
  const curves: Curve[] = [];
  const reach = len(s);
  const lvl = modelLevel(cur, 'reg', x0, gr, B, x0, Math.max(2.2 * reach, i.span * 0.04), i.slot);
  if (lvl.overlay) overlays.push({ ...lvl.overlay, fill: true });
  if (lvl.curve) curves.push(lvl.curve);
  if (i.labels) {
    // λ ↦ x_{k−1} − (H + λI)⁻¹g on λ ∈ [λ_k/30, 30λ_k] (where H + λI ≻ 0).
    const tr = H[0][0] + H[1][1];
    const det = H[0][0] * H[1][1] - hs * hs;
    const lamMin = tr / 2 - Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
    const lo = Math.max(lam / 30, -lamMin + 1e-9 * Math.max(1, Math.abs(lamMin)));
    const hi = lam * 30;
    const path: Pt[] = [];
    if (lo < hi)
      for (let t = 0; t <= 48; t++) {
        const q = shifted(x0, gr, H, lo * (hi / lo) ** (t / 48));
        if (q && len(sub2(q, x0)) < Math.max(6 * reach, i.span * 0.1)) path.push(q);
      }
    if (path.length > 1)
      overlays.push({ kind: 'polyline', points: path, dashed: true, width: 1, alpha: 0.8 });
    const trials = Array.isArray(cur.info.trials) ? (cur.info.trials as unknown[][]) : [];
    for (const t of trials) {
      if (t[4] === true || !isNum(t[1])) continue;
      const q = shifted(x0, gr, H, t[1]);
      if (q && len(sub2(q, x0)) < Math.max(6 * reach, i.span * 0.1))
        overlays.push({ kind: 'point', at: q, slot: i.slot, shape: 'ring', radius: 2.6 });
    }
    if (isSpd(H)) {
      const np = shifted(x0, gr, H, 0);
      if (np && len(sub2(np, x0)) < Math.max(4 * reach, i.span * 0.05))
        overlays.push({
          kind: 'point',
          at: np,
          slot: i.slot,
          shape: 'cross',
          radius: 4,
          label: vecSup('x', 'N'),
          labelSide: 'right',
        });
    }
  }
  overlays.push({ kind: 'arrow', from: x0, to: x1, slot: i.slot, width: 1.5 });
  return { curves, overlays };
}

/** 3 significant digits; ×10ⁿ below 10⁻³ and above 10⁴ (U+2212 minus). */
export const fmt = (x: unknown): string => {
  if (!isNum(x)) return '—';
  return x !== 0 && (Math.abs(x) < 1e-3 || Math.abs(x) >= 1e4) ? sci(x, 2) : sig(x, 3);
};

const OP_LABEL: Record<string, string> = {
  reflect: 'reflect',
  expand: 'expand',
  contract_outside: 'contract (outside)',
  contract_inside: 'contract (inside)',
  shrink: 'shrink',
};
const OP_SYM: Record<string, string> = {
  reflect: 'r',
  expand: 'e',
  contract_outside: 'oc',
  contract_inside: 'ic',
};

function nmGeometry(i: GeometryInput): Geometry {
  const s = i.trace[i.g];
  const simplex = Array.isArray(s.info.simplex) ? (s.info.simplex as unknown[]).map(pt) : [];
  if (simplex.length < 3 || !simplex.every(Boolean)) return EMPTY;
  const pts = simplex as Pt[];
  const op = typeof s.info.operation === 'string' ? s.info.operation : null;
  const overlays: Overlay2D[] = [];
  if (i.g > 0) {
    const old = Array.isArray(i.trace[i.g - 1].info.simplex)
      ? (i.trace[i.g - 1].info.simplex as unknown[]).map(pt)
      : [];
    if (old.length === 3 && old.every(Boolean))
      overlays.push({
        kind: 'polygon',
        points: old as Pt[],
        slot: i.slot,
        dashed: true,
        width: 1,
        alpha: 0.7,
      });
  }
  overlays.push({ kind: 'polygon', points: pts, slot: i.slot, fill: true, width: 1.5 });
  if (!i.labels || i.g === 0 || !op) return { curves: [], overlays };
  const cen = pt(s.info.centroid);
  const worst = pt(s.info.worst);
  const trials = Array.isArray(s.info.trials) ? (s.info.trials as Record<string, unknown>[]) : [];
  if (cen && worst && op !== 'shrink') {
    const far = trials.map((t) => pt(t.x)).filter((q): q is Pt => q !== null);
    const end = far.reduce(
      (acc, q) => (len(sub2(q, worst)) > len(sub2(acc, worst)) ? q : acc),
      cen,
    );
    overlays.push({ kind: 'segment', from: worst, to: end, dashed: true, width: 1, alpha: 0.8 });
    overlays.push({ kind: 'point', at: cen, radius: 2.4, label: [b('x̄')], labelSide: 'left' });
  }
  for (const t of trials) {
    const q = pt(t.x);
    const o = typeof t.op === 'string' ? t.op : '';
    if (!q) continue;
    overlays.push({
      kind: 'point',
      at: q,
      slot: i.slot,
      shape: 'ring',
      radius: 2.8,
      label: [b('x'), sub(OP_SYM[o] ?? o, 'italic')],
      labelSide: 'right',
    });
  }
  // The operation's name, left of the simplex's leftmost vertex (the lens shows it in its corner).
  const left = pts.reduce((r, q) => (q[0] < r[0] ? q : r));
  overlays.push({
    kind: 'text',
    at: [left[0] - i.span * 0.02, left[1]],
    text: OP_LABEL[op] ?? op,
    align: 'right',
  });
  return { curves: [], overlays, caption: OP_LABEL[op] ?? op };
}

function powellGeometry(i: GeometryInput): Geometry {
  const s = i.trace[i.g];
  const lines = Array.isArray(s.info.lines) ? (s.info.lines as Record<string, unknown>[]) : [];
  const overlays: Overlay2D[] = [];
  lines.forEach((l, j) => {
    const o = pt(l.origin);
    const q = pt(l.point);
    if (!o || !q) return;
    overlays.push({
      kind: 'arrow',
      from: o,
      to: q,
      slot: i.slot,
      width: 1.3,
      dashed: j === lines.length - 1 && lines.length > 2,
      label: i.labels && len(sub2(q, o)) > i.span * 0.04 ? [b('u'), sub(String(j + 1))] : undefined,
    });
  });
  const xe = pt(s.info.extrapolated);
  const start = lines.length ? pt(lines[0].origin) : null;
  if (i.labels && xe && start) {
    overlays.push({ kind: 'segment', from: start, to: xe, dashed: true, width: 1, alpha: 0.75 });
    overlays.push({
      kind: 'point',
      at: xe,
      shape: 'ring',
      radius: 3,
      label: [b('x'), sub('E', 'italic')],
      labelSide: 'right',
    });
  }
  return { curves: [], overlays };
}

function hjGeometry(i: GeometryInput): Geometry {
  const s = i.trace[i.g];
  const probes = Array.isArray(s.info.probes) ? (s.info.probes as Record<string, unknown>[]) : [];
  const overlays: Overlay2D[] = [];
  const base = pt(s.info.base);
  const prevBase = pt(s.info.previous_base);
  const pat = pt(s.info.pattern_point);
  if (s.info.move === 'pattern' && prevBase && pat && base) {
    const from = xOf(i.trace[i.g - 1]) ?? prevBase;
    overlays.push({ kind: 'arrow', from, to: pat, slot: i.slot, dashed: true, width: 1.3 });
    if (i.labels)
      overlays.push({
        kind: 'point',
        at: pat,
        slot: i.slot,
        shape: 'ring',
        radius: 3.4,
        label: [b('p')],
        labelSide: 'right',
      });
  }
  const h = s.info.step;
  const center =
    s.info.move === 'pattern' && pat ? pat : (xOf(i.trace[Math.max(0, i.g - 1)]) ?? base);
  if (center && isNum(h))
    overlays.push(
      {
        kind: 'segment',
        from: [center[0] - h, center[1]],
        to: [center[0] + h, center[1]],
        slot: i.slot,
        width: 1,
        alpha: 0.6,
      },
      {
        kind: 'segment',
        from: [center[0], center[1] - h],
        to: [center[0], center[1] + h],
        slot: i.slot,
        width: 1,
        alpha: 0.6,
      },
    );
  if (i.labels)
    for (const p of probes) {
      const q = pt(p.x);
      if (q) overlays.push({ kind: 'point', at: q, slot: i.slot, shape: 'ring', radius: 2.5 });
    }
  return { curves: [], overlays };
}

function compassGeometry(i: GeometryInput): Geometry {
  const s = i.trace[i.g];
  const c = xOf(i.trace[Math.max(0, i.g - 1)]);
  const d = s.info.step;
  if (!c || !isNum(d)) return EMPTY;
  const dirs: Pt[] = [
    [1, 0],
    [0, 1],
    [-1, 0],
    [0, -1],
  ];
  const overlays: Overlay2D[] = dirs.map((e) => ({
    kind: 'segment' as const,
    from: c,
    to: add(c, mul(d, e)),
    slot: i.slot,
    width: 1.1,
    alpha: 0.7,
  }));
  const polls = Array.isArray(s.info.polls) ? (s.info.polls as Record<string, unknown>[]) : [];
  if (i.labels) {
    for (const p of polls) {
      const q = pt(p.x);
      if (q) overlays.push({ kind: 'point', at: q, slot: i.slot, shape: 'ring', radius: 2.8 });
    }
    const nd = s.info.new_step;
    if (i.g > 0 && s.info.success === false && isNum(nd))
      overlays.push({
        kind: 'polygon',
        points: [
          [c[0] + nd, c[1]],
          [c[0], c[1] + nd],
          [c[0] - nd, c[1]],
          [c[0], c[1] - nd],
        ],
        slot: i.slot,
        dashed: true,
        width: 1,
        alpha: 0.7,
      });
  }
  return { curves: [], overlays };
}

/** The geometry of step `g` for one method (empty when the step has nothing to draw). */
export function buildGeometry(i: GeometryInput): Geometry {
  const s = i.trace[i.g];
  if (!s) return EMPTY;
  if (i.g === 0) {
    if (i.kind === 'tr') return trGeometry(i);
    if (i.kind === 'nm') return nmGeometry(i);
    if (i.kind === 'compass') return compassGeometry(i);
    return EMPTY;
  }
  switch (i.kind) {
    case 'line':
      return lineGeometry(i, s);
    case 'coord':
      return coordGeometry(i);
    case 'heavy':
      return heavyGeometry(i);
    case 'nesterov':
      return nesterovGeometry(i);
    case 'adaptive':
      return adaptiveGeometry(i);
    case 'newton':
      return newtonGeometry(i);
    case 'qn':
      return qnGeometry(i);
    case 'cg':
      return cgGeometry(i, s);
    case 'tr':
      return trGeometry(i);
    case 'nm':
      return nmGeometry(i);
    case 'powell':
      return powellGeometry(i);
    case 'hj':
      return hjGeometry(i);
    case 'compass':
      return compassGeometry(i);
    case 'schedule':
      return scheduleGeometry(i);
    case 'ogm':
      return ogmGeometry(i);
    case 'fista':
      return fistaGeometry(i);
    case 'anderson':
      return andersonGeometry(i);
    case 'arc':
      return arcGeometry(i);
    case 'regnewton':
      return regNewtonGeometry(i);
  }
}

// ── convergence measures ─────────────────────────────────────────────────────────────

/** The known minimizer nearest to `x` (null when the problem lists none). */
export function nearestMinimizer(problem: Problem2D, x: unknown): Pt | null {
  const mins = (problem.minima ?? []).filter(isVec) as Pt[];
  const p = pt(x);
  if (!mins.length || !p) return mins[0] ?? null;
  return mins.reduce((best, q) => (len(sub2(q, p)) < len(sub2(best, p)) ? q : best));
}

export type Metric = 'gap' | 'grad' | 'dist';

/**
 * One convergence series. 𝐱⋆ is the known minimizer nearest the run's last iterate (local
 * methods converge to the minimizer of their basin), f⋆ = f(𝐱⋆). Values below the rounding
 * level (10⁻¹⁶ relative) are drawn at it, so an exact zero does not cut the curve.
 */
export function seriesValues(
  metric: Metric,
  trace: readonly Step[],
  problem: Problem2D,
): (number | null)[] {
  const last = trace[trace.length - 1];
  const star = last ? nearestMinimizer(problem, last.x) : null;
  if (metric === 'grad')
    return trace.map((s) => {
      let gn = s.gradNorm;
      if (gn === null || gn === undefined) {
        const x = pt(s.x);
        const gr = x ? problem.grad([x[0], x[1]]) : null;
        gn = isVec(gr) ? len(gr) : null;
      }
      return gn === null || !Number.isFinite(gn) ? null : Math.max(gn, 1e-16);
    });
  if (!star) return trace.map(() => null);
  if (metric === 'dist') {
    const floor = 1e-16 * Math.max(1, len(star));
    return trace.map((s) => {
      const x = pt(s.x);
      return x ? Math.max(len(sub2(x, star)), floor) : null;
    });
  }
  const fStar = problem.f([star[0], star[1]]);
  const floor = 1e-16 * Math.max(1, Math.abs(fStar));
  return trace.map((s) =>
    s.fun === null || !Number.isFinite(s.fun) ? null : Math.max(s.fun - fStar, floor),
  );
}
