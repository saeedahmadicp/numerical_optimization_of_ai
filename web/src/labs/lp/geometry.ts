/**
 * Pure geometry of linear programs for the LP lab: half-spaces, the feasible polygon (2-D) and
 * polytope (3-D), vertex enumeration, the simplex edge direction read off a tableau, the
 * central path and the Dikin ellipsoid. No DOM; unit-tested in geometry.test.ts.
 *
 * Every constraint is a half-space aᵀx ≤ b. `x ≥ 0` is written −xⱼ ≤ 0; an equality row becomes
 * two half-spaces (its feasible set is the line itself).
 */
import type { LinearProgram } from '../../core/types';

export type Vec = number[];

export interface HalfSpace {
  a: Vec;
  b: number;
  /** `row`: a stored inequality (A_ub row `index`); `eq`: one side of an equality row;
   *  `bound`: xⱼ ≥ 0 (j = `index`); `extra`: a cut or a branching bound. */
  kind: 'row' | 'eq' | 'bound' | 'extra';
  index: number;
}

/** Every constraint of the LP as half-spaces aᵀx ≤ b (bounds last). */
export function halfSpaces(lp: LinearProgram): HalfSpace[] {
  const n = lp.c.length;
  const out: HalfSpace[] = [];
  (lp.aUb ?? []).forEach((a, i) => out.push({ a: [...a], b: lp.bUb![i], kind: 'row', index: i }));
  (lp.aEq ?? []).forEach((a, i) => {
    out.push({ a: [...a], b: lp.bEq![i], kind: 'eq', index: i });
    out.push({ a: a.map((v) => -v), b: -lp.bEq![i], kind: 'eq', index: i });
  });
  for (let j = 0; j < n; j++) {
    const a = new Array<number>(n).fill(0);
    a[j] = -1;
    out.push({ a, b: 0, kind: 'bound', index: j });
  }
  return out;
}

const dot = (a: readonly number[], b: readonly number[]) => a.reduce((s, v, i) => s + v * b[i], 0);

/** Scale of a constraint for relative feasibility tests. */
const hsScale = (h: HalfSpace) =>
  1 + Math.abs(h.b) + h.a.reduce((m, v) => Math.max(m, Math.abs(v)), 0);

/** Slack bᵢ − aᵢᵀx of every half-space (negative = violated). */
export function slacks(hs: readonly HalfSpace[], x: readonly number[]): number[] {
  return hs.map((h) => h.b - dot(h.a, x));
}

export function isFeasible(hs: readonly HalfSpace[], x: readonly number[], tol = 1e-9): boolean {
  return hs.every(
    (h) => h.b - dot(h.a, x) >= -tol * hsScale(h) * (1 + Math.max(...x.map(Math.abs))),
  );
}

// ── 2-D ──────────────────────────────────────────────────────────────────────────────────

export type P2 = [number, number];

/** Sutherland–Hodgman: the part of convex polygon `poly` with aᵀx ≤ b. */
export function clipPolygon(poly: readonly P2[], a: readonly number[], b: number): P2[] {
  const out: P2[] = [];
  const n = poly.length;
  if (n === 0) return out;
  const f = (p: P2) => a[0] * p[0] + a[1] * p[1] - b;
  for (let i = 0; i < n; i++) {
    const p = poly[i],
      q = poly[(i + 1) % n];
    const fp = f(p),
      fq = f(q);
    if (fp <= 0) out.push(p);
    if ((fp < 0 && fq > 0) || (fp > 0 && fq < 0)) {
      const t = fp / (fp - fq);
      out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
    }
  }
  return out;
}

export const boxPolygon = (
  box: readonly [readonly [number, number], readonly [number, number]],
): P2[] => [
  [box[0][0], box[1][0]],
  [box[0][1], box[1][0]],
  [box[0][1], box[1][1]],
  [box[0][0], box[1][1]],
];

/** The feasible polygon inside `box` (empty when infeasible inside it). */
export function feasiblePolygon(
  hs: readonly HalfSpace[],
  box: readonly [readonly [number, number], readonly [number, number]],
): P2[] {
  let poly = boxPolygon(box);
  for (const h of hs) {
    poly = clipPolygon(poly, h.a, h.b);
    if (poly.length === 0) break;
  }
  return poly;
}

/** Vertices of a 2-D feasible set: feasible intersections of pairs of constraint lines. */
export function vertices2D(hs: readonly HalfSpace[]): P2[] {
  const out: P2[] = [];
  for (let i = 0; i < hs.length; i++)
    for (let j = i + 1; j < hs.length; j++) {
      const [a, b] = [hs[i], hs[j]];
      const det = a.a[0] * b.a[1] - a.a[1] * b.a[0];
      if (Math.abs(det) < 1e-12) continue;
      const x: P2 = [(a.b * b.a[1] - a.a[1] * b.b) / det, (a.a[0] * b.b - a.b * b.a[0]) / det];
      if (!isFeasible(hs, x, 1e-9)) continue;
      if (out.some((p) => Math.hypot(p[0] - x[0], p[1] - x[1]) < 1e-9 * (1 + Math.hypot(...x))))
        continue;
      out.push(x);
    }
  return out;
}

/**
 * Where the line aᵀx = b crosses the box: the segment inside it (null if it misses). Used for
 * constraint lines, objective level lines and cuts.
 */
export function lineInBox(
  a: readonly number[],
  b: number,
  box: readonly [readonly [number, number], readonly [number, number]],
): [P2, P2] | null {
  const [[x0, x1], [y0, y1]] = box;
  const pts: P2[] = [];
  const push = (p: P2) => {
    if (
      p[0] >= x0 - 1e-9 &&
      p[0] <= x1 + 1e-9 &&
      p[1] >= y0 - 1e-9 &&
      p[1] <= y1 + 1e-9 &&
      !pts.some((q) => Math.abs(q[0] - p[0]) < 1e-12 && Math.abs(q[1] - p[1]) < 1e-12)
    )
      pts.push(p);
  };
  if (Math.abs(a[1]) > 1e-15) {
    push([x0, (b - a[0] * x0) / a[1]]);
    push([x1, (b - a[0] * x1) / a[1]]);
  }
  if (Math.abs(a[0]) > 1e-15) {
    push([(b - a[1] * y0) / a[0], y0]);
    push([(b - a[1] * y1) / a[0], y1]);
  }
  if (pts.length < 2) return null;
  // The two points farthest apart.
  let best: [P2, P2] = [pts[0], pts[1]];
  let d = -1;
  for (let i = 0; i < pts.length; i++)
    for (let j = i + 1; j < pts.length; j++) {
      const dd = Math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]);
      if (dd > d) {
        d = dd;
        best = [pts[i], pts[j]];
      }
    }
  return best;
}

/** Integer points of the box, split by feasibility. */
export function latticePoints(
  hs: readonly HalfSpace[],
  box: readonly [readonly [number, number], readonly [number, number]],
  limit = 2500,
): { feasible: P2[]; infeasible: P2[] } {
  const feasible: P2[] = [],
    infeasible: P2[] = [];
  const [[x0, x1], [y0, y1]] = box;
  const count = (Math.floor(x1) - Math.ceil(x0) + 1) * (Math.floor(y1) - Math.ceil(y0) + 1);
  if (count > limit) return { feasible, infeasible };
  for (let i = Math.ceil(x0); i <= Math.floor(x1); i++)
    for (let j = Math.ceil(y0); j <= Math.floor(y1); j++)
      (isFeasible(hs, [i, j], 1e-9) ? feasible : infeasible).push([i, j]);
  return { feasible, infeasible };
}

// ── Central path and Dikin ellipsoid (inequality form Gx ≤ h) ─────────────────────────────

function solve2(H: number[][], g: number[]): number[] | null {
  const det = H[0][0] * H[1][1] - H[0][1] * H[1][0];
  if (!(Math.abs(det) > 1e-300)) return null;
  return [(H[1][1] * g[0] - H[0][1] * g[1]) / det, (H[0][0] * g[1] - H[1][0] * g[0]) / det];
}

/**
 * Minimize t·cᵀx − Σ log(hᵢ − gᵢᵀx) by damped Newton from a strictly feasible x (2-D).
 * Returns the minimizer or null when Newton fails (unbounded barrier, no interior).
 */
function barrierMin(hs: readonly HalfSpace[], c: readonly number[], t: number, x0: P2): P2 | null {
  let x: P2 = [x0[0], x0[1]];
  const phi = (p: P2) => {
    let v = t * (c[0] * p[0] + c[1] * p[1]);
    for (const h of hs) {
      const s = h.b - (h.a[0] * p[0] + h.a[1] * p[1]);
      if (!(s > 0)) return Infinity;
      v -= Math.log(s);
    }
    return v;
  };
  for (let it = 0; it < 80; it++) {
    const g = [t * c[0], t * c[1]];
    const H = [
      [0, 0],
      [0, 0],
    ];
    for (const h of hs) {
      const s = h.b - (h.a[0] * x[0] + h.a[1] * x[1]);
      if (!(s > 0)) return null;
      g[0] += h.a[0] / s;
      g[1] += h.a[1] / s;
      for (let i = 0; i < 2; i++)
        for (let j = 0; j < 2; j++) H[i][j] += (h.a[i] * h.a[j]) / (s * s);
    }
    const d = solve2(H, g);
    if (!d) return null;
    const dec = d[0] * g[0] + d[1] * g[1]; // Newton decrement²
    if (dec < 1e-18) return x;
    let alpha = 1;
    const f0 = phi(x);
    for (let ls = 0; ls < 60; ls++) {
      const xn: P2 = [x[0] - alpha * d[0], x[1] - alpha * d[1]];
      if (phi(xn) <= f0 - 0.25 * alpha * dec) {
        x = xn;
        break;
      }
      alpha *= 0.5;
      if (ls === 59) return x;
    }
    if (dec < 1e-14) return x;
  }
  return x;
}

/**
 * The central path {x(μ) = argmin c̃ᵀx − μ Σ log sᵢ(x)} of a bounded 2-D LP without equality
 * rows, from the analytic center toward the optimum (`cMin` = min-form costs). Scale-free:
 * the same curve as the path of the scaled problem the interior-point methods iterate on.
 */
export function centralPath(hs: readonly HalfSpace[], cMin: readonly number[], interior: P2): P2[] {
  if (hs.some((h) => h.kind === 'eq')) return [];
  const center = barrierMin(hs, [0, 0], 0, interior);
  if (!center) return [];
  const out: P2[] = [center];
  const scale = Math.max(1e-12, Math.hypot(cMin[0], cMin[1]));
  const m = hs.length;
  // Duality gap along the path is m/t: sweep t until the gap is ~1e-7 of the objective scale.
  let x = center;
  for (let i = 0; i <= 90; i++) {
    const t = (1e-3 / scale) * Math.pow(10, (i / 90) * 10);
    const next = barrierMin(hs, cMin, t, x);
    if (!next) break;
    const last = out[out.length - 1];
    if (Math.hypot(next[0] - last[0], next[1] - last[1]) > 1e-9) out.push(next);
    x = next;
    if (m / t < 1e-7 * scale) break;
  }
  return out;
}

/** A strictly interior point of a 2-D polygon (its vertex centroid), or null. */
export function interiorPoint(poly: readonly P2[], hs: readonly HalfSpace[]): P2 | null {
  if (poly.length < 3) return null;
  const c: P2 = [
    poly.reduce((s, p) => s + p[0], 0) / poly.length,
    poly.reduce((s, p) => s + p[1], 0) / poly.length,
  ];
  return slacks(hs, c).every((s) => s > 1e-12) ? c : null;
}

/**
 * Dikin ellipsoid matrix Q = Σ gᵢgᵢᵀ/sᵢ² at a strictly feasible x: {x + d : dᵀQd ≤ 1} is the
 * unit ball of the affine-scaling metric, inscribed in the polygon. Null when x is not
 * strictly feasible.
 */
export function dikinMatrix(hs: readonly HalfSpace[], x: readonly number[]): number[][] | null {
  const n = x.length;
  const Q = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (const h of hs) {
    const s = h.b - dot(h.a, x);
    if (!(s > 1e-12)) return null;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Q[i][j] += (h.a[i] * h.a[j]) / (s * s);
  }
  return Q;
}

// ── Simplex geometry read off a tableau ────────────────────────────────────────────────────

export interface TableauInfo {
  tableau: number[][];
  row_labels: string[];
  col_labels: string[];
  basis: number[];
  entering: number | null;
  leaving: number | null;
  pivot_row: number | null;
  ratio_test: (number | null)[] | null;
}

/** Number of objective rows (`z`, or `zM` and `z` for big-M). */
export const objectiveRows = (info: Pick<TableauInfo, 'row_labels'>) =>
  info.row_labels.filter((l) => l === 'z' || l === 'zM').length;

/**
 * The edge direction η restricted to the original variables when column `q` enters:
 * ηⱼ = 1 for j = q, −ūᵢ for the basic xⱼ of row i, 0 otherwise (Bertsimas & Tsitsiklis §3.2).
 */
export function edgeDirection(info: TableauInfo, q: number, n: number): Vec {
  const eta = new Array<number>(n).fill(0);
  if (q < n) eta[q] = 1;
  const nObj = objectiveRows(info);
  info.basis.forEach((j, i) => {
    if (j < n) eta[j] = -info.tableau[nObj + i][q];
  });
  return eta;
}

// ── 3-D polytope ───────────────────────────────────────────────────────────────────────────

export interface Polytope {
  vertices: Vec[];
  /** Indices into `hs` of the half-spaces tight at each vertex. */
  active: number[][];
  edges: [number, number][];
  /** Per half-space with ≥ 3 tight vertices: the vertex cycle of that face. */
  faces: { plane: number; cycle: number[] }[];
}

function solve3(A: number[][], b: number[]): number[] | null {
  const det = (M: number[][]) =>
    M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
    M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
    M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
  const D = det(A);
  if (!(Math.abs(D) > 1e-12)) return null;
  return [0, 1, 2].map((k) => det(A.map((row, i) => row.map((v, j) => (j === k ? b[i] : v)))) / D);
}

/** Vertices, edges and faces of {x ∈ ℝ³ : aᵢᵀx ≤ bᵢ} by enumeration of plane triples. */
export function polytope3(hs: readonly HalfSpace[]): Polytope {
  const vertices: Vec[] = [];
  const active: number[][] = [];
  const m = hs.length;
  for (let i = 0; i < m; i++)
    for (let j = i + 1; j < m; j++)
      for (let k = j + 1; k < m; k++) {
        const x = solve3([hs[i].a, hs[j].a, hs[k].a], [hs[i].b, hs[j].b, hs[k].b]);
        if (!x || !isFeasible(hs, x, 1e-9)) continue;
        const scale = 1 + Math.max(...x.map(Math.abs));
        if (vertices.some((v) => v.every((vi, q) => Math.abs(vi - x[q]) < 1e-9 * scale))) continue;
        vertices.push(x);
        active.push(
          hs.flatMap((h, q) =>
            Math.abs(h.b - dot(h.a, x)) <= 1e-9 * hsScale(h) * scale ? [q] : [],
          ),
        );
      }
  const edges: [number, number][] = [];
  for (let i = 0; i < vertices.length; i++)
    for (let j = i + 1; j < vertices.length; j++) {
      const common = active[i].filter((q) => active[j].includes(q));
      // An edge: two tight planes in common whose normals are independent.
      let ok = false;
      for (let a = 0; a < common.length && !ok; a++)
        for (let b = a + 1; b < common.length && !ok; b++) {
          const u = hs[common[a]].a,
            v = hs[common[b]].a;
          const cr = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
          ];
          ok = Math.hypot(...cr) > 1e-9;
        }
      if (ok) edges.push([i, j]);
    }
  const faces: Polytope['faces'] = [];
  hs.forEach((h, q) => {
    const vs = vertices.flatMap((_, i) => (active[i].includes(q) ? [i] : []));
    if (vs.length < 3) return;
    // Order the cycle by angle around the centroid, in the plane's own basis.
    const cen = [0, 1, 2].map((d) => vs.reduce((s, i) => s + vertices[i][d], 0) / vs.length);
    const nrm = h.a;
    const ref = Math.abs(nrm[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    const e1 = [
      nrm[1] * ref[2] - nrm[2] * ref[1],
      nrm[2] * ref[0] - nrm[0] * ref[2],
      nrm[0] * ref[1] - nrm[1] * ref[0],
    ];
    const e2 = [
      nrm[1] * e1[2] - nrm[2] * e1[1],
      nrm[2] * e1[0] - nrm[0] * e1[2],
      nrm[0] * e1[1] - nrm[1] * e1[0],
    ];
    const ang = (i: number) => {
      const d = vertices[i].map((v, k) => v - cen[k]);
      return Math.atan2(dot(d, e2), dot(d, e1));
    };
    faces.push({ plane: q, cycle: [...vs].sort((a, b) => ang(a) - ang(b)) });
  });
  return { vertices, active, edges, faces };
}

/** Integer points of the polytope's bounding box that are feasible (3-D lattice). */
export function lattice3(hs: readonly HalfSpace[], hi: readonly number[], limit = 4000): Vec[] {
  const out: Vec[] = [];
  const nx = Math.floor(hi[0]),
    ny = Math.floor(hi[1]),
    nz = Math.floor(hi[2]);
  if ((nx + 1) * (ny + 1) * (nz + 1) > limit) return out;
  for (let i = 0; i <= nx; i++)
    for (let j = 0; j <= ny; j++)
      for (let k = 0; k <= nz; k++) if (isFeasible(hs, [i, j, k], 1e-9)) out.push([i, j, k]);
  return out;
}

/** Box bounds lo ≤ x ≤ hi (hi null = none) as extra half-spaces. */
export function boundSpaces(bounds: readonly (readonly [number, number | null])[]): HalfSpace[] {
  const n = bounds.length;
  const out: HalfSpace[] = [];
  bounds.forEach(([lo, hi], j) => {
    const e = new Array<number>(n).fill(0);
    if (hi !== null) {
      e[j] = 1;
      out.push({ a: [...e], b: hi, kind: 'extra', index: j });
    }
    if (lo > 0) {
      e[j] = -1;
      out.push({ a: [...e], b: -lo, kind: 'extra', index: j });
    }
  });
  return out;
}

/** The min-form cost vector c̃ (c for min, −c for max). */
export const minCost = (lp: LinearProgram) => lp.c.map((v) => (lp.sense === 'min' ? v : -v));
