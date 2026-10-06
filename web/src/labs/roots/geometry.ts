/**
 * Step geometry of the root finders, in data coordinates — the picture of how x_k was produced.
 *
 * `stepGeometry(method, step, f)` reads `Step.info` (the Python keys, see
 * src/numopt/roots/bracketing.py and open.py) and returns drawable primitives: the bracket and
 * the part the sign test discards, chords (with Illinois-type scaled ordinates), tangents,
 * Halley's osculating hyperbola, Müller's parabola, the inverse parabola x(y), Ridders'
 * exponentially transformed points, ITP's interpolation/truncation/projection, Steffensen's
 * probe and the fixed-point line of slope 1/λ. `headPath` is the route the animated iterate
 * takes from (x_k, f(x_k)) to (x_{k+1}, f(x_{k+1})): along the model whose zero is x_{k+1},
 * then up or down to the curve.
 *
 * Pure functions (no DOM): tested in tests/roots/geometry.test.ts.
 */
import type { Step } from '../../core/types';

export type Pt = readonly [number, number];

/**
 * Label codes (the plot typesets them): 'm' midpoint, 'tol' the probe x_{k−1} ± tol, 'x_half',
 * 'x_f' ITP's points, 'r' its projection radius, 'z' Steffensen's probe point, 'flen' its probe
 * length f(x_{k−1}), 'aux' an auxiliary start, `scale:<m>` a scaled ordinate m·f(a).
 */
export type Label = string;

export type Prim =
  /** A segment; `ext` extends it by that fraction of its length past both ends. */
  | { t: 'seg'; a: Pt; b: Pt; dash?: boolean; faint?: boolean; ext?: number }
  /** y = g(x) on [from, to]; samples are split where g jumps (poles). */
  | {
      t: 'curve';
      g: (x: number) => number;
      from: number;
      to: number;
      dash?: boolean;
      faint?: boolean;
    }
  /** A sampled path (inverse parabolas x(y)). */
  | { t: 'path'; pts: Pt[]; dash?: boolean; faint?: boolean }
  | { t: 'pt'; p: Pt; shape: 'dot' | 'ring' | 'x'; label?: Label; faint?: boolean }
  /** Dashed vertical from (x, 0) to (x, y). */
  | { t: 'drop'; x: number; y: number }
  /** A short mark on the x axis with a label under or over it. */
  | { t: 'tick'; x: number; label?: Label }
  /** A horizontal dimension line from a to b at height y (probe length, projection radius). */
  | { t: 'span'; a: number; b: number; y: number; label?: Label }
  /** A dotted connector (scaled ordinate ↔ true value). */
  | { t: 'link'; a: Pt; b: Pt; label?: Label };

export interface StepGeometry {
  prims: Prim[];
  /** The sign-change interval x_k was computed from. */
  bracket?: [number, number];
  /** The interval kept by the sign test (`new_bracket`). */
  kept?: [number, number];
  /** How x_k was produced, in words ("chord", "bisection", "rejected inverse quadratic"…). */
  kind?: string;
}

export const BRACKETING = new Set([
  'bisection',
  'regula_falsi',
  'illinois',
  'pegasus',
  'anderson_bjorck',
  'ridders',
  'brent',
  'chandrupatla',
  'itp',
]);

const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const pair = (v: unknown): [number, number] | undefined =>
  Array.isArray(v) && v.length === 2 && isNum(v[0]) && isNum(v[1]) ? [v[0], v[1]] : undefined;
const pts = (v: unknown): Pt[] => (Array.isArray(v) ? (v.map(pair).filter(Boolean) as Pt[]) : []);

/** x(y) = a y² + b y + c sampled over y ∈ [lo, hi] (the inverse parabola is a function of y). */
export function inverseParabolaPath(a: number, b: number, c: number, lo: number, hi: number): Pt[] {
  const out: Pt[] = [];
  for (let i = 0; i <= 64; i++) {
    const y = lo + ((hi - lo) * i) / 64;
    out.push([a * y * y + b * y + c, y]);
  }
  return out;
}

/** The inverse quadratic x(y) through three points, as a callable. */
export function lagrangeInverse(p: readonly Pt[]): (y: number) => number {
  const [[x0, y0], [x1, y1], [x2, y2]] = p;
  return (y) =>
    (x0 * (y - y1) * (y - y2)) / ((y0 - y1) * (y0 - y2)) +
    (x1 * (y - y0) * (y - y2)) / ((y1 - y0) * (y1 - y2)) +
    (x2 * (y - y0) * (y - y1)) / ((y2 - y0) * (y2 - y1));
}

/** y range that covers the points and 0, padded by 12 %. */
function yRange(p: readonly Pt[]): [number, number] {
  const ys = [0, ...p.map((q) => q[1])];
  const lo = Math.min(...ys),
    hi = Math.max(...ys);
  const pad = (hi - lo) * 0.12;
  return [lo - pad, hi + pad];
}

function xRange(xs: number[], pad = 0.18): [number, number] {
  const lo = Math.min(...xs),
    hi = Math.max(...xs);
  const w = hi - lo || Math.abs(lo) * 0.1 || 1;
  return [lo - w * pad, hi + w * pad];
}

/** Halley's osculating hyperbola y = (t + α)/(βt + γ), t = x − center. */
export function hyperbola(h: { center: number; alpha: number; beta: number; gamma: number }) {
  return (x: number) => {
    const t = x - h.center;
    return (t + h.alpha) / (h.beta * t + h.gamma);
  };
}

/** Müller's parabola p(x) = a t² + b t + c, t = x − center. */
export function mullerParabola(p: { center: number; a: number; b: number; c: number }) {
  return (x: number) => {
    const t = x - p.center;
    return p.a * t * t + p.b * t + p.c;
  };
}

/**
 * The geometry of step `step` of method `method` on f (f is used only to place points on the
 * curve, e.g. the true value at a bracket end next to its scaled ordinate).
 */
export function stepGeometry(method: string, step: Step, f: (x: number) => number): StepGeometry {
  const info = step.info;
  const x = step.x as number;
  const fx = step.fun ?? NaN;
  const prims: Prim[] = [];
  const g: StepGeometry = { prims };
  const bracket = pair(info.bracket);
  const kept = pair(info.new_bracket);
  if (bracket) g.bracket = [Math.min(...bracket), Math.max(...bracket)];
  if (kept) g.kept = [Math.min(...kept), Math.max(...kept)];
  const drop = () => isNum(fx) && prims.push({ t: 'drop', x, y: fx });

  if (BRACKETING.has(method) && bracket) {
    // The two ends and their values: one above, one below the axis.
    for (const e of bracket) prims.push({ t: 'pt', p: [e, f(e)], shape: 'dot', faint: true });
  }

  switch (method) {
    case 'bisection':
      g.kind = 'midpoint';
      prims.push({ t: 'tick', x, label: 'm' });
      drop();
      break;
    case 'regula_falsi':
    case 'illinois':
    case 'pegasus':
    case 'anderson_bjorck': {
      const kind = info.step as string;
      g.kind = kind === 'verify' ? 'verifying probe' : kind;
      const chord = pts(info.chord);
      if (kind === 'chord' && chord.length === 2) {
        const [a, b] = chord;
        prims.push({ t: 'seg', a, b });
        // A scaled ordinate (Illinois-type): show it next to the true value of f.
        const trueA = f(a[0]);
        if (
          Number.isFinite(trueA) &&
          Math.abs(a[1] - trueA) > 1e-9 * Math.max(1, Math.abs(trueA))
        ) {
          const m = a[1] / trueA;
          prims.push({ t: 'link', a: [a[0], trueA], b: a, label: `scale:${fmtFactor(m)}` });
          prims.push({ t: 'pt', p: a, shape: 'ring' });
        }
      } else if (kind === 'verify') {
        prims.push({ t: 'tick', x, label: 'tol' });
      } else {
        prims.push({ t: 'tick', x, label: 'm' });
      }
      drop();
      break;
    }
    case 'ridders': {
      g.kind = info.step === 'verify' ? 'verifying probe' : 'Ridders';
      const tr = pts(info.transformed);
      if (isNum(info.midpoint) && isNum(info.f_mid))
        prims.push({ t: 'pt', p: [info.midpoint, info.f_mid], shape: 'dot', label: 'm' });
      if (tr.length === 3 && bracket && isNum(info.exp_factor) && isNum(info.midpoint)) {
        const lo = bracket[0];
        const d = info.midpoint - lo;
        const lam = d !== 0 ? Math.log(info.exp_factor) / d : 0;
        // h(x) = f(x)·e^{λ(x − lo)}: the three transformed values lie on one line.
        prims.push({
          t: 'curve',
          g: (s) => f(s) * Math.exp(lam * (s - lo)),
          from: bracket[0],
          to: bracket[1],
          dash: true,
          faint: true,
        });
        prims.push({ t: 'seg', a: tr[0], b: tr[2] });
        for (const p of tr) prims.push({ t: 'pt', p, shape: 'ring' });
      } else if (info.step === 'verify') {
        prims.push({ t: 'tick', x, label: 'tol' });
      }
      drop();
      break;
    }
    case 'brent':
    case 'chandrupatla': {
      const kind = info.step as string;
      const attempted = (method === 'brent' ? info.attempted : kind) as string | null;
      const p = pts(info.points);
      const rejected = method === 'brent' && attempted !== null && attempted !== kind;
      g.kind = rejected
        ? `bisection (${attempted === 'secant' ? 'secant' : 'inverse quadratic'} rejected)`
        : kind === 'inverse_quadratic'
          ? 'inverse quadratic'
          : kind;
      if (attempted === 'secant' && p.length === 2) {
        prims.push({ t: 'seg', a: p[0], b: p[1], ext: 0.6, dash: rejected, faint: rejected });
      } else if (attempted === 'inverse_quadratic' && p.length === 3) {
        const xy = lagrangeInverse(p);
        const [lo, hi] = yRange(p);
        const path: Pt[] = [];
        for (let i = 0; i <= 64; i++) {
          const y = lo + ((hi - lo) * i) / 64;
          path.push([xy(y), y]);
        }
        prims.push({ t: 'path', pts: path, dash: rejected, faint: rejected });
      }
      for (const q of p) prims.push({ t: 'pt', p: q, shape: 'ring', faint: rejected });
      if (kind === 'bisection') prims.push({ t: 'tick', x, label: 'm' });
      drop();
      break;
    }
    case 'itp': {
      g.kind = info.projected ? 'projected' : 'truncated';
      if (bracket) {
        const fa = f(bracket[0]),
          fb = f(bracket[1]);
        prims.push({ t: 'seg', a: [bracket[0], fa], b: [bracket[1], fb], faint: true });
      }
      if (isNum(info.x_half)) prims.push({ t: 'tick', x: info.x_half, label: 'x_half' });
      if (isNum(info.x_f)) prims.push({ t: 'tick', x: info.x_f, label: 'x_f' });
      if (isNum(info.r) && isNum(info.x_half) && info.r > 0)
        prims.push({
          t: 'span',
          a: info.x_half - info.r,
          b: info.x_half + info.r,
          y: 0,
          label: 'r',
        });
      drop();
      break;
    }
    // ── open methods ──────────────────────────────────────────────────────────────────
    case 'newton': {
      const tg = info.tangent as { point: [number, number]; slope: number } | undefined;
      if (tg) {
        prims.push({ t: 'pt', p: tg.point, shape: 'dot', faint: true });
        prims.push({ t: 'seg', a: tg.point, b: [x, 0], ext: 0.25 });
      }
      drop();
      break;
    }
    case 'secant': {
      const ch = pts(info.chord);
      if (ch.length === 2) {
        for (const p of ch) prims.push({ t: 'pt', p, shape: 'ring' });
        const far = Math.abs(ch[0][0] - x) > Math.abs(ch[1][0] - x) ? ch[0] : ch[1];
        prims.push({ t: 'seg', a: far, b: [x, 0], ext: 0.12 });
      }
      drop();
      break;
    }
    case 'halley': {
      const h = info.hyperbola as
        { center: number; alpha: number; beta: number; gamma: number } | undefined;
      if (h) {
        const [lo, hi] = xRange([h.center, x], 0.35);
        prims.push({ t: 'curve', g: hyperbola(h), from: lo, to: hi });
        prims.push({ t: 'pt', p: [h.center, f(h.center)], shape: 'dot', faint: true });
      }
      drop();
      break;
    }
    case 'steffensen': {
      const ch = pts(info.chord);
      if (ch.length === 2) {
        const [[x0, f0], [z, fz]] = ch;
        prims.push({ t: 'pt', p: [z, fz], shape: 'ring', label: 'z' });
        const far = Math.abs(z - x) > Math.abs(x0 - x) ? ([z, fz] as Pt) : ([x0, f0] as Pt);
        prims.push({ t: 'seg', a: far, b: [x, 0], ext: 0.12 });
        // The probe length is f(x_{k−1}) itself — a value of f used as a distance in x.
        prims.push({ t: 'span', a: x0, b: z, y: 0, label: 'flen' });
      }
      drop();
      break;
    }
    case 'muller': {
      const p = pts(info.points);
      const par = info.parabola as { center: number; a: number; b: number; c: number } | undefined;
      if (par && p.length === 3) {
        const [lo, hi] = xRange([...p.map((q) => q[0]), x], 0.2);
        prims.push({ t: 'curve', g: mullerParabola(par), from: lo, to: hi });
      }
      for (const q of p) prims.push({ t: 'pt', p: q, shape: 'ring' });
      drop();
      break;
    }
    case 'inverse_quadratic_interpolation': {
      const p = pts(info.points);
      const ip = info.inverse_parabola as { a: number; b: number; c: number } | undefined;
      if (ip && p.length === 3) {
        const [lo, hi] = yRange(p);
        prims.push({ t: 'path', pts: inverseParabolaPath(ip.a, ip.b, ip.c, lo, hi) });
      }
      for (const q of p) prims.push({ t: 'pt', p: q, shape: 'ring' });
      drop();
      break;
    }
    case 'fixed_point': {
      const prev = info.previous;
      if (isNum(prev) && isNum(info.lam)) {
        const fp = f(prev);
        // x_{k} = x_{k−1} − λ f(x_{k−1}): the line of slope 1/λ through (x_{k−1}, f(x_{k−1})).
        prims.push({ t: 'pt', p: [prev, fp], shape: 'dot', faint: true });
        prims.push({ t: 'seg', a: [prev, fp], b: [x, 0], ext: 0.2, dash: true });
      }
      drop();
      break;
    }
    default:
      break;
  }
  const aux = pts(info.auxiliary);
  for (const q of aux) prims.push({ t: 'pt', p: q, shape: 'ring', label: 'aux' });
  const probe = pair(info.slope_probe);
  if (probe) prims.push({ t: 'pt', p: probe, shape: 'x' });
  return g;
}

function fmtFactor(m: number): string {
  if (Math.abs(m - 0.5) < 1e-12) return '½';
  const r = Math.abs(m) >= 0.01 ? m.toFixed(2) : m.toExponential(1);
  return r.replace('-', '−');
}

/**
 * Where the animated iterate is at fraction u ∈ [0, 1] of the move from step `cur` to step
 * `next`: along the model whose zero is x_{k+1} (straight for chords, tangents and bisection;
 * the hyperbola, parabola or inverse parabola otherwise), then vertically onto the curve.
 */
export function headAt(method: string, cur: Step, next: Step, u: number): Pt {
  const x0 = cur.x as number,
    y0 = cur.fun ?? 0;
  const x1 = next.x as number,
    y1 = next.fun ?? 0;
  const split = 0.62;
  if (u >= split) {
    const v = (u - split) / (1 - split);
    return [x1, v * y1];
  }
  const v = u / split;
  if (v <= 0) return [x0, y0];
  const info = next.info;
  if (method === 'halley' && info.hyperbola) {
    const h = hyperbola(
      info.hyperbola as { center: number; alpha: number; beta: number; gamma: number },
    );
    const xs = x0 + (x1 - x0) * v;
    const ys = h(xs);
    if (Number.isFinite(ys) && Math.abs(ys) <= 4 * Math.max(Math.abs(y0), 1e-300)) return [xs, ys];
  }
  if (method === 'muller' && info.parabola) {
    const p = mullerParabola(info.parabola as { center: number; a: number; b: number; c: number });
    const xs = x0 + (x1 - x0) * v;
    return [xs, p(xs)];
  }
  if (method === 'inverse_quadratic_interpolation' && info.inverse_parabola) {
    const ip = info.inverse_parabola as { a: number; b: number; c: number };
    const ys = y0 * (1 - v);
    return [ip.a * ys * ys + ip.b * ys + ip.c, ys];
  }
  return [x0 + (x1 - x0) * v, y0 + (0 - y0) * v];
}

// ── Camera ──────────────────────────────────────────────────────────────────────────────

const smooth = (u: number) => (u <= 0 ? 0 : u >= 1 ? 1 : u * u * (3 - 2 * u));

/** Narrowest window the camera frames: 10⁻⁶ relative (axis labels stay readable). */
export const MIN_REL_WIDTH = 2e-6;

/**
 * The x-window that frames step k of a run: the whole problem at k = 0, then the bracket
 * (bracketing methods) or the points the step was built from (open methods), padded so the
 * geometry sits in the middle 60 % of the view.
 */
export function stepWindow(
  trace: readonly Step[],
  k: number,
  domain: readonly [number, number],
): [number, number] {
  const st = trace[k];
  const xs: number[] = [st.x as number];
  const info = st.info;
  const br = pair(info.bracket);
  if (br) xs.push(...br);
  if (isNum(info.previous)) xs.push(info.previous);
  for (const key of ['chord', 'points']) for (const p of pts(info[key])) xs.push(p[0]);
  if (k === 0) xs.push(...domain, ...pts(info.auxiliary).map((p) => p[0]));
  const finite = xs.filter(Number.isFinite);
  if (!finite.length) return [domain[0], domain[1]];
  const lo = Math.min(...finite),
    hi = Math.max(...finite);
  const c = (lo + hi) / 2;
  const w = Math.max(hi - lo, MIN_REL_WIDTH * Math.max(1, Math.abs(c)));
  const half = k === 0 ? w * 0.54 : w * 0.8;
  return [c - half, c + half];
}

/** The camera at continuous step t: centers move linearly, widths geometrically (a zoom). */
export function cameraAt(
  trace: readonly Step[],
  t: number,
  domain: readonly [number, number],
): [number, number] {
  const n = trace.length;
  if (n === 0) return [domain[0], domain[1]];
  const tt = Math.max(0, Math.min(t, n - 1));
  const k = Math.floor(tt);
  const a = stepWindow(trace, k, domain);
  if (k >= n - 1) return clampWindow(a, domain);
  const b = stepWindow(trace, k + 1, domain);
  const v = smooth(tt - k);
  const ca = (a[0] + a[1]) / 2,
    cb = (b[0] + b[1]) / 2;
  const wa = a[1] - a[0],
    wb = b[1] - b[0];
  const c = ca + (cb - ca) * v;
  const w = wa ** (1 - v) * wb ** v;
  return clampWindow([c - w / 2, c + w / 2], domain);
}

/**
 * Keep the camera near the problem: at most 12 problem widths wide and centered within 4 widths
 * of it. A diverging run then leaves the view (and is drawn at the edge with its value) instead
 * of zooming the picture out to 10¹³.
 */
export function clampWindow(
  w: readonly [number, number],
  domain: readonly [number, number],
): [number, number] {
  const dw = domain[1] - domain[0];
  const width = Math.min(w[1] - w[0], 12 * dw);
  const c = Math.max(domain[0] - 4 * dw, Math.min(domain[1] + 4 * dw, (w[0] + w[1]) / 2));
  return [c - width / 2, c + width / 2];
}

/**
 * The window a handle is dragged in: the problem window, widened to hold the handle. The
 * followed camera can be 12 problem widths wide after a divergent run, or 10⁻¹⁰ wide at
 * convergence; neither is a usable scale for placing a start point or a bracket end.
 */
export function dragWindow(
  view: readonly [number, number],
  domain: readonly [number, number],
  value: number,
): [number, number] {
  const dw = domain[1] - domain[0];
  const vw = view[1] - view[0];
  const inView = value >= view[0] && value <= view[1];
  if (inView && vw <= 2 * dw && vw >= dw / 8) return [view[0], view[1]];
  const lo = Math.min(domain[0], value - 0.1 * dw);
  const hi = Math.max(domain[1], value + 0.1 * dw);
  return [lo, hi];
}
