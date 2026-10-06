/**
 * Pure geometry of the least-squares lab (no React, unit-tested in tests/least-squares).
 *
 *   - `dataSpaceFor(problem)`: what the data-space pane draws for a problem — a curve fit
 *     (t, y, model m(t; 𝐱)), a circle fit (points, R) or, for Rosenbrock, the residual plane.
 *   - `linearization(problem, x)`: r, J, JᵀJ and the Gauss–Newton step at 𝐱 (from the same thin
 *     SVD as the method port, so the drawn GN point is the method's).
 *   - `stepGeometry(...)`: the parameter-space overlays of one step, read from Step.info: the
 *     Gauss–Newton model ellipse, the step and its backtracking trials, the LM trust disk
 *     ‖𝐡‖ ≤ Δ implied by μ, the Levenberg–Marquardt curve 𝐡(μ) and the tangency point.
 */
import type { Matrix, Step, Vector } from '../../core/types';
import type { Overlay2D } from '../../viz';
import { svdThin } from '../../methods/unconstrained/least_squares';
import type { LeastSquaresProblem } from '../../problems/least_squares';
import { mathBold as b, mathMain as m, mathSub as sub, mathVar as v } from '../../viz';
import { sig } from '../../core/format';

// ── Data space ───────────────────────────────────────────────────────────────────────

export type DataSpace =
  | {
      kind: 'curve';
      t: number[];
      y: number[];
      /** The model m(t; 𝐱) whose residuals are rᵢ = m(tᵢ; 𝐱) − yᵢ. */
      model: (t: number, x: readonly number[]) => number;
      /** Axis names (t, y) and the model in TeX. */
      tName: string;
      yName: string;
      tex: string;
    }
  | {
      kind: 'circle';
      px: number[];
      py: number[];
      radius: number;
      tex: string;
    }
  | {
      kind: 'residual';
      /** r(𝐱) ∈ ℝ²: the plane where the solution is the origin. */
      residual: (x: readonly number[]) => Vector;
      tex: string;
    };

/** Names of the two parameters (axis names of the parameter plane). */
export function paramNames(problem: Pick<LeastSquaresProblem, 'id'>): [string, string] {
  switch (problem.id) {
    case 'michaelis_menten':
      return ['V', 'K'];
    case 'rosenbrock_ls':
      return ['x', 'y'];
    default:
      return ['a', 'b'];
  }
}

export function dataSpaceFor(problem: LeastSquaresProblem): DataSpace {
  const ex = problem.extra;
  switch (problem.id) {
    case 'exp_decay_fit':
      return {
        kind: 'curve',
        t: ex.t ?? [],
        y: ex.y ?? [],
        model: (t, x) => x[0] * Math.exp(-x[1] * t),
        tName: 't',
        yName: 'y',
        tex: 'y = a\\,e^{-b t}',
      };
    case 'michaelis_menten':
      return {
        kind: 'curve',
        t: ex.t ?? [],
        y: ex.y ?? [],
        model: (S, x) => (x[0] * S) / (x[1] + S),
        tName: 'S',
        yName: 'v',
        tex: 'v = \\dfrac{V S}{K + S}',
      };
    case 'circle_fit':
      return {
        kind: 'circle',
        px: ex.points_x ?? [],
        py: ex.points_y ?? [],
        radius: ex.radius ?? 1,
        tex: '(x - a)^2 + (y - b)^2 = R^2',
      };
    default:
      return {
        kind: 'residual',
        residual: (x) => problem.residual([x[0], x[1]]),
        tex: '\\mathbf{r}(x, y) \\in \\mathbb{R}^2',
      };
  }
}

/**
 * The t-range of the model curve: [max(0, t_min), axis end]. The axis is padded below 0, but
 * the model means nothing at t < 0 (Michaelis–Menten has its pole S = −K there).
 */
export function curveRange(tLo: number, tDom: readonly [number, number]): [number, number] {
  return [Math.max(0, Math.min(tLo, tDom[1])), tDom[1]];
}

/** The point of the circle of center c and radius R nearest to p (the foot of the residual). */
export function circleFoot(
  c: readonly [number, number],
  R: number,
  p: readonly [number, number],
): [number, number] {
  const dx = p[0] - c[0],
    dy = p[1] - c[1];
  const d = Math.hypot(dx, dy);
  if (!(d > 0)) return [c[0] + R, c[1]];
  return [c[0] + (R * dx) / d, c[1] + (R * dy) / d];
}

// ── Linearization ────────────────────────────────────────────────────────────────────

export interface Linear {
  r: Vector;
  J: Matrix;
  /** JᵀJ (2 × 2). */
  M: Matrix;
  g: Vector;
  /** Minimum-norm Gauss–Newton step −J⁺r (null when J is numerically rank-deficient). */
  p: Vector | null;
  /** Singular values of J (descending) and the factors, for 𝐡(μ). */
  s: Vector;
  U: Matrix;
  Vt: Matrix;
}

export function linearization(problem: LeastSquaresProblem, x: readonly number[]): Linear | null {
  const xv = [x[0], x[1]];
  const r = problem.residual(xv);
  const J = problem.jac(xv);
  if (!r.every(Number.isFinite) || !J.every((row) => row.every(Number.isFinite))) return null;
  const n = 2;
  const M: Matrix = [0, 1].map((a) =>
    [0, 1].map((c) => J.reduce((acc, row) => acc + row[a] * row[c], 0)),
  );
  const g = [0, 1].map((a) => J.reduce((acc, row, i) => acc + row[a] * r[i], 0));
  const { U, s, Vt } = svdThin(J);
  if (!s.every(Number.isFinite)) return null;
  const tol = Math.max(J.length, n) * 2.220446049250313e-16 * (s[0] ?? 0);
  const deficient = s.length < n || !(s[0] > 0) || s[s.length - 1] <= tol;
  const p = deficient ? null : hOfMu({ U, s, Vt, r }, 0);
  return { r, J, M, g, p, s, U, Vt };
}

/** 𝐡(μ) = −V diag(σ/(σ² + μ)) Uᵀ𝐫 — the Levenberg–Marquardt step for damping μ (μ = 0: GN). */
export function hOfMu(lin: Pick<Linear, 'U' | 's' | 'Vt' | 'r'>, mu: number): Vector {
  const { U, s, Vt, r } = lin;
  const n = Vt[0]?.length ?? 0;
  const w = s.map((sk, k) => {
    const d = sk * sk + mu;
    const c = mu === 0 ? (sk > 0 ? 1 / sk : 0) : d > 0 ? sk / d : 0;
    let utr = 0;
    for (let i = 0; i < U.length; i++) utr += U[i][k] * r[i];
    return c * utr;
  });
  const out = new Array<number>(n).fill(0);
  for (let j = 0; j < n; j++) {
    let acc = 0;
    for (let k = 0; k < Vt.length; k++) acc += Vt[k][j] * w[k];
    out[j] = -acc;
  }
  return out;
}

/** (u − w)ᵀM(u − w). */
function quad(M: Matrix, u: readonly number[], w: readonly number[] = [0, 0]): number {
  const d0 = u[0] - w[0],
    d1 = u[1] - w[1];
  return M[0][0] * d0 * d0 + 2 * M[0][1] * d0 * d1 + M[1][1] * d1 * d1;
}

// ── Step geometry ────────────────────────────────────────────────────────────────────

export type MethodKind = 'gauss_newton' | 'levenberg_marquardt';

/**
 * A point of the step that the reader must be able to find even when it leaves the view: the
 * Gauss–Newton point 𝐱ₖ + 𝐩ₖ (with its backtracking trials on the same ray), the LM trial
 * 𝐱ₖ + 𝐡ₖ, the end μ = 0 of the curve 𝐡(μ). `path` runs from 𝐱ₖ to `at` along what is drawn.
 */
export interface StepMark {
  role: 'gn' | 'lm' | 'mu0';
  at: [number, number];
  path: [number, number][];
  /** Gauss–Newton: the failed Armijo trials 𝐱ₖ + αⱼ𝐩ₖ. */
  trials?: [number, number][];
  /** LM: the trial was rejected (ϱₖ ≤ 0). */
  rejected?: boolean;
}

/** Whether a label may be drawn for the mark `from → to`, anchored at `at` (data units). */
export type LabelTest = (
  from: [number, number],
  to: [number, number],
  at: [number, number],
) => boolean;

export interface StepGeometry {
  overlays: Overlay2D[];
  marks: StepMark[];
  /** The step being drawn: x_k → x_k + h (null at the last iterate). */
  from: [number, number] | null;
  to: [number, number] | null;
  accepted: boolean | null;
  /** Radius of the trust disk implied by μ (LM), ‖𝐡ₖ‖. */
  radius: number | null;
}

const EMPTY: StepGeometry = {
  overlays: [],
  marks: [],
  from: null,
  to: null,
  accepted: null,
  radius: null,
};

const pt = (x: readonly number[]): [number, number] => [x[0], x[1]];
const plus = (a: readonly number[], c: readonly number[], s = 1): [number, number] => [
  a[0] + s * c[0],
  a[1] + s * c[1],
];

/**
 * The geometry of the step taken from 𝐱ₖ (trace[k]) to 𝐱ₖ₊₁ (trace[k + 1]), drawn while the
 * playhead is in [k, k + 1). Everything except the model ellipses and 𝐡(μ) comes from
 * trace[k + 1].info; the model is recomputed at 𝐱ₖ (the same r and J the method used).
 */
export function stepGeometry(
  problem: LeastSquaresProblem,
  trace: readonly Step[],
  k: number,
  slot: number,
  kind: MethodKind,
  /**
   * Which text labels to draw. The lab decides in pixels (a step a few pixels long, or a label
   * on top of 𝐱⋆'s, gets none); without it every label is drawn.
   */
  showLabel: LabelTest = () => true,
): StepGeometry {
  const cur = trace[k];
  const next = trace[k + 1];
  if (!cur || !next) return EMPTY;
  const xk = cur.x as number[];
  const info = next.info;
  const step = info.step as number[] | null;
  if (!step) return EMPTY;
  const lin = linearization(problem, xk);
  const overlays: Overlay2D[] = [];
  const marks: StepMark[] = [];
  const kk = String(k);
  const from = pt(xk);
  const mid = (a: readonly number[], c: readonly number[]): [number, number] => [
    (a[0] + c[0]) / 2,
    (a[1] + c[1]) / 2,
  ];

  if (kind === 'gauss_newton') {
    const p = step;
    const alpha = (info.alpha as number | null) ?? 1;
    const trials = (info.trials as [number, number][] | undefined) ?? [];
    const gnPoint = plus(xk, p);
    if (lin) {
      const q = quad(lin.M, p);
      // The model's level set through 𝐱ₖ: {𝐱ₖ + 𝐡 : L(𝐡) = L(𝟎)}, centered on the GN point.
      overlays.push({
        kind: 'ellipse',
        center: gnPoint,
        matrix: lin.M,
        radius: Math.sqrt(q),
        slot,
        width: 1.1,
        alpha: 0.75,
        dashed: true,
      });
      // …and through the accepted point 𝐱ₖ + α𝐩 (radius (1 − α)·√(𝐩ᵀM𝐩)).
      if (alpha < 1)
        overlays.push({
          kind: 'ellipse',
          center: gnPoint,
          matrix: lin.M,
          radius: (1 - alpha) * Math.sqrt(q),
          slot,
          width: 1,
          alpha: 0.55,
        });
    }
    overlays.push({
      kind: 'arrow',
      from: pt(xk),
      to: gnPoint,
      slot,
      width: 1.4,
      dashed: alpha < 1,
      alpha: alpha < 1 ? 0.8 : 1,
    });
    // Backtracking trials 𝐱ₖ + αⱼ𝐩: rejected ones hollow, the accepted one solid.
    trials.forEach(([a], j) => {
      const accepted = j === trials.length - 1;
      if (a === 1 && accepted) return;
      overlays.push({
        kind: 'point',
        at: plus(xk, p, a),
        slot,
        shape: accepted ? 'dot' : 'ring',
        radius: accepted ? 3.5 : 3,
      });
    });
    overlays.push({
      kind: 'point',
      at: gnPoint,
      slot,
      shape: 'ring',
      radius: 4.5,
      label: showLabel(from, gnPoint, gnPoint)
        ? [b('x'), sub(kk), m(' + '), b('p'), sub(kk)]
        : undefined,
      labelSide: 'right',
    });
    marks.push({
      role: 'gn',
      at: gnPoint,
      path: [from, gnPoint],
      // The failed trials other than α = 1 (that one is the Gauss–Newton point itself).
      trials: trials
        .slice(0, -1)
        .filter(([a]) => a !== 1)
        .map(([a]) => plus(xk, p, a)),
    });
    return {
      overlays,
      marks,
      from,
      to: plus(xk, p, alpha),
      accepted: true,
      radius: null,
    };
  }

  // Levenberg–Marquardt.
  const h = step;
  const accepted = info.accepted === true;
  const gain = info.gain_ratio as number | null;
  const radius = Math.hypot(h[0], h[1]);
  const end = plus(xk, h);
  overlays.push({ kind: 'disk', center: pt(xk), radius, slot, fill: true, width: 1, alpha: 0.9 });
  if (lin) {
    // The curve 𝐡(μ), μ ∈ [0, ∞): from the Gauss–Newton point (μ = 0) into 𝐱ₖ along −∇f.
    const s0 = lin.s[0] ?? 1;
    const sl = lin.s[lin.s.length - 1] ?? 1;
    const lo = Math.log10(Math.max(1e-300, sl * sl * 1e-3));
    const hi = Math.log10(Math.max(1e-300, s0 * s0 * 1e4));
    const curve: [number, number][] = [];
    if (lin.p) curve.push(plus(xk, lin.p));
    for (let i = 0; i <= 96; i++)
      curve.push(plus(xk, hOfMu(lin, 10 ** (lo + ((hi - lo) * i) / 96))));
    curve.push(pt(xk));
    overlays.push({ kind: 'polyline', points: curve, slot, width: 1, alpha: 0.6, dashed: true });
    if (lin.p) marks.push({ role: 'mu0', at: plus(xk, lin.p), path: [...curve].reverse() });
    if (lin.p) {
      // The model level set tangent to the disk at 𝐱ₖ + 𝐡ₖ (𝐡ₖ minimizes L on ‖𝐡‖ ≤ ‖𝐡ₖ‖).
      const gn = plus(xk, lin.p);
      overlays.push({
        kind: 'ellipse',
        center: gn,
        matrix: lin.M,
        radius: Math.sqrt(quad(lin.M, h, lin.p)),
        slot,
        width: 1,
        alpha: 0.7,
      });
      overlays.push({
        kind: 'point',
        at: gn,
        slot,
        shape: 'ring',
        radius: 3.5,
        label: showLabel(from, gn, gn) ? [v('μ'), m(' = 0')] : undefined,
        labelSide: 'right',
      });
    }
  }
  overlays.push({
    kind: 'arrow',
    from: pt(xk),
    to: end,
    slot,
    width: 1.6,
    dashed: !accepted,
    label:
      gain === null || gain === undefined || !showLabel(from, end, mid(xk, end))
        ? undefined
        : [v('ϱ'), sub(kk), m(` = ${sig(gain, 3)}`)],
  });
  if (!accepted) overlays.push({ kind: 'point', at: end, slot, shape: 'ring', radius: 3.5 });
  marks.push({ role: 'lm', at: end, path: [from, end], rejected: !accepted });
  return { overlays, marks, from, to: end, accepted, radius };
}
