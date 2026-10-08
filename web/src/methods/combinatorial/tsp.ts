/**
 * The symmetric Euclidean travelling-salesman problem — TS port of `numopt.combinatorial.tsp`
 * (read it for the full docstrings): nearest neighbour, 2-opt, Or-opt, simulated annealing,
 * a genetic algorithm (OX), Ant System and Held–Karp.
 *
 * L(π) = Σₖ d(π_k, π_{k+1 mod n}), d(i, j) = ‖cᵢ − cⱼ‖₂. Methods return `x` = the tour (n city
 * indices; the closing edge is implicit) and `fun` = its length. Numerical conventions shared
 * with Python: d = sqrt(dx·dx + dy·dy); tour lengths summed in tour order, closing edge last;
 * 2-opt deltas as (d(a,c) + d(b,d)) − (d(a,b) + d(c,d)); a move improves only when
 * Δ < −tol with tol = 1e−10 · max d (ties within tol go to the smaller index).
 *
 * Random numbers (SA, GA, ACO) come only from `Rng` (Mulberry32) in the Python draw order.
 * `info` keys are the Python ones (snake_case).
 */
import { param, registerMethod } from '../../core/registry';
import { Rng } from '../../core/rng';
import type { MethodFn, Result, Step } from '../../core/types';
import { isTsp, type TspInstance } from '../../problems/combinatorial';

/** Relative tolerance for "improving" moves and nearest-city ties. */
export const REL_TOL = 1e-10;
/** Held–Karp needs O(2ⁿ·n) memory and O(2ⁿ·n²) time; larger n is refused. */
export const HELD_KARP_MAX_N = 16;
/** Largest n for which the ant-colony trace carries the full pheromone matrix. */
export const PHEROMONE_MAX_N = 40;
/** 2⁻⁴⁸⁵ ≈ 1.0e−146: smallest admissible largest distance when squares underflow. */
export const MIN_DISTANCE_SCALE = 2 ** -485;
/** Smallest positive normal double, 2⁻¹⁰²². */
const TINY = 2 ** -1022;
const NN_START_UI_MAX = 19;

type Matrix = number[][];
type Move = Record<string, unknown>;

export type TspInput = TspInstance | readonly (readonly number[])[];

// ── Shared helpers ─────────────────────────────────────────────────────────────────────

function instance(problem: TspInput): TspInstance {
  let inst: TspInstance;
  if (isTsp(problem)) inst = problem;
  else if (Array.isArray(problem)) {
    const arr = problem as readonly (readonly number[])[];
    if (!arr.every((r) => Array.isArray(r) && r.length === 2))
      throw new Error(`coordinates must have shape (n, 2)`);
    inst = {
      kind: 'tsp',
      id: 'custom',
      name: 'custom',
      coords: arr.map((r) => [Number(r[0]), Number(r[1])]),
      optimumLength: null,
      description: '',
      latex: '',
      tags: [],
    };
  } else throw new TypeError('problem must be a TspInstance or an (n, 2) coordinate array');
  if (inst.coords.length === 0) throw new Error(`${inst.id}: the instance has no cities`);
  if (!inst.coords.every((c) => c.length === 2))
    throw new Error(`${inst.id}: coordinates must have shape (n, 2)`);
  if (!inst.coords.every((c) => Number.isFinite(c[0]) && Number.isFinite(c[1])))
    throw new Error(`${inst.id}: coordinates must be finite`);
  return inst;
}

/** d(i, j) = sqrt(dx·dx + dy·dy) for every pair of cities (n × n, symmetric, zero diagonal). */
export function distanceMatrix(coords: readonly (readonly number[])[]): Matrix {
  const n = coords.length;
  const D: Matrix = [];
  for (let i = 0; i < n; i++) {
    const row = new Array<number>(n);
    for (let j = 0; j < n; j++) {
      const dx = coords[i][0] - coords[j][0];
      const dy = coords[i][1] - coords[j][1];
      row[j] = Math.sqrt(dx * dx + dy * dy);
    }
    D.push(row);
  }
  return D;
}

function maxOf(D: Matrix): number {
  let m = -Infinity;
  for (const row of D) for (const v of row) if (v > m || Number.isNaN(v)) m = v;
  return m;
}

/** The distance matrix of a validated instance; throws when it over- or underflows. */
function distances(inst: TspInstance): Matrix {
  const D = distanceMatrix(inst.coords);
  if (!D.every((row) => row.every(Number.isFinite)))
    throw new Error(
      `${inst.id}: coordinate differences overflow the distance computation ` +
        '(|dx|, |dy| must stay below ~1e154); rescale the instance',
    );
  const dMax = maxOf(D);
  if (dMax < MIN_DISTANCE_SCALE) {
    const c = inst.coords;
    let underflowed = false;
    for (let i = 0; i < c.length && !underflowed; i++)
      for (let j = 0; j < c.length && !underflowed; j++)
        for (let a = 0; a < 2; a++) {
          const diff = c[i][a] - c[j][a];
          if (diff !== 0 && diff * diff < TINY) underflowed = true;
        }
    if (underflowed)
      throw new Error(
        `${inst.id}: inter-city distances underflow the distance computation ` +
          `(largest distance ${fmtG(dMax, 3)} < 2^-485 ≈ ${fmtG(MIN_DISTANCE_SCALE, 3)}); ` +
          'rescale the instance',
      );
  }
  return D;
}

/** Closed-tour length, summed sequentially in tour order (closing edge last). */
export function tourLength(D: Matrix, tour: readonly number[]): number {
  const n = tour.length;
  let total = 0;
  for (let k = 0; k < n - 1; k++) total += D[tour[k]][tour[k + 1]];
  if (n > 1) total += D[tour[n - 1]][tour[0]];
  return total;
}

function pathLength(D: Matrix, path: readonly number[]): number {
  let total = 0;
  for (let k = 0; k < path.length - 1; k++) total += D[path[k]][path[k + 1]];
  return total;
}

function tolerance(D: Matrix): number {
  return D.length ? REL_TOL * maxOf(D) : 0;
}

/** Python's built-in `sum` of floats (CPython ≥ 3.12: Neumaier-compensated). */
export function pySum(xs: readonly number[]): number {
  let s = 0,
    c = 0;
  for (const x of xs) {
    const t = s + x;
    if (Math.abs(s) >= Math.abs(x)) c += s - t + x;
    else c += x - t + s;
    s = t;
  }
  return c !== 0 && Number.isFinite(c) ? s + c : s;
}

/** Python's `'%.{p}g' % x` (enough of it for messages). */
export function fmtG(x: number, p = 6): string {
  if (!Number.isFinite(x)) return Number.isNaN(x) ? 'nan' : x > 0 ? 'inf' : '-inf';
  if (x === 0) return Object.is(x, -0) ? '-0' : '0';
  const [mant, expStr] = x.toExponential(p - 1).split('e');
  const exp = Number(expStr);
  if (exp < -4 || exp >= p) {
    const m = mant.includes('.') ? mant.replace(/\.?0+$/, '') : mant;
    return `${m}e${exp < 0 ? '-' : '+'}${String(Math.abs(exp)).padStart(2, '0')}`;
  }
  const s = x.toFixed(Math.max(0, p - 1 - exp));
  return s.includes('.') ? s.replace(/\.?0+$/, '') : s;
}

function nearestNeighborTour(D: Matrix, start: number, tol: number): number[] {
  const n = D.length;
  const visited = new Array<boolean>(n).fill(false);
  visited[start] = true;
  const tour = [start];
  for (let s = 0; s < n - 1; s++) {
    const cur = tour[tour.length - 1];
    let bestJ = -1,
      bestD = Infinity;
    for (let j = 0; j < n; j++) {
      if (!visited[j] && D[cur][j] < bestD - tol) {
        bestJ = j;
        bestD = D[cur][j];
      }
    }
    visited[bestJ] = true;
    tour.push(bestJ);
  }
  return tour;
}

function initialTour(D: Matrix, init: string, tol: number): number[] {
  if (init === 'identity') return Array.from({ length: D.length }, (_, i) => i);
  if (init === 'nearest_neighbor') return nearestNeighborTour(D, 0, tol);
  throw new Error(`unknown init '${init}'; expected 'identity' or 'nearest_neighbor'`);
}

function checkInt(name: string, value: unknown, low: number): number {
  const v = Number(value);
  if (!Number.isInteger(v) || v < low)
    throw new Error(`${name} must be an integer ≥ ${low}, got ${String(value)}`);
  return v;
}

function checkUnit(name: string, value: unknown, closedLow = true, closedHigh = true): number {
  const v = Number(value);
  const okLow = closedLow ? v >= 0 : v > 0;
  const okHigh = closedHigh ? v <= 1 : v < 1;
  if (!(Number.isFinite(v) && okLow && okHigh))
    throw new Error(`${name} must lie in the unit interval, got ${String(value)}`);
  return v;
}

function twoOptDelta(D: Matrix, t: readonly number[], i: number, j: number): number {
  const n = t.length;
  const a = t[i],
    b = t[i + 1],
    c = t[j],
    d = t[(j + 1) % n];
  return D[a][c] + D[b][d] - (D[a][b] + D[c][d]);
}

function twoOptMove(t: readonly number[], i: number, j: number, delta: number): Move {
  const n = t.length;
  const a = t[i],
    b = t[i + 1],
    c = t[j],
    d = t[(j + 1) % n];
  return {
    i,
    j,
    removed: [
      [a, b],
      [c, d],
    ],
    added: [
      [a, c],
      [b, d],
    ],
    delta,
  };
}

const edgeKey = (e: readonly number[]) => (e[0] < e[1] ? `${e[0]},${e[1]}` : `${e[1]},${e[0]}`);

/** Drop every edge (as an unordered pair) that occurs in both lists; keep the order. */
function netExchange(removed: number[][], added: number[][]): [number[][], number[][]] {
  const r = new Set(removed.map(edgeKey));
  const a = new Set(added.map(edgeKey));
  const common = new Set([...r].filter((k) => a.has(k)));
  return [
    removed.filter((e) => !common.has(edgeKey(e))),
    added.filter((e) => !common.has(edgeKey(e))),
  ];
}

/** Reverse t[i+1..j] in place. */
function reverse(t: number[], i: number, j: number): void {
  for (let lo = i + 1, hi = j; lo < hi; lo++, hi--) {
    const tmp = t[lo];
    t[lo] = t[hi];
    t[hi] = tmp;
  }
}

const mkStep = (k: number, x: number[], fun: number, info: Record<string, unknown>): Step => ({
  k,
  x,
  fun,
  gradNorm: null,
  stepSize: null,
  info,
});

function finish(
  method: string,
  inst: TspInstance,
  tour: readonly number[],
  length: number,
  converged: boolean,
  message: string,
  nIter: number,
  nFev: number,
  trace: Step[],
  extra: Record<string, unknown> = {},
): Result {
  const info: Record<string, unknown> = { ...extra };
  if (inst.optimumLength !== null) {
    info.optimum_length = inst.optimumLength;
    info.gap = inst.optimumLength > 0 ? (length - inst.optimumLength) / inst.optimumLength : 0.0;
  }
  return {
    method,
    x: [...tour],
    fun: length,
    converged,
    message,
    nIter,
    nFev,
    nGev: 0,
    nHev: 0,
    trace,
    extra: info,
  };
}

/** Every tour has the same length (n ≤ 3, or all cities coincide): return the start tour. */
function trivial(
  method: string,
  inst: TspInstance,
  D: Matrix,
  tour: number[],
  info: Record<string, unknown>,
  reason = '',
  extra: Record<string, unknown> = {},
): Result {
  const length = tourLength(D, tour);
  const s = mkStep(0, [...tour], length, { tour: [...tour], length, ...info });
  const message = reason || `n = ${inst.coords.length} ≤ 3: every tour has the same length`;
  return finish(method, inst, tour, length, true, message, 0, 0, [s], extra);
}

const record = (k: number, every: number) => k % every === 0;

// ── Nearest neighbour ──────────────────────────────────────────────────────────────────

export const tspNearestNeighbor: MethodFn<TspInput> = (problem, opts) => {
  const inst = instance(problem);
  const n = inst.coords.length;
  const start = checkInt('start', opts.start ?? 0, 0);
  if (start >= n) throw new Error(`start must be a city index in [0, ${n - 1}], got ${start}`);
  const D = distances(inst);
  const tol = tolerance(D);

  const visited = new Array<boolean>(n).fill(false);
  visited[start] = true;
  const tour = [start];
  let length = 0;
  const trace: Step[] = [
    mkStep(0, [start], 0, { tour: [start], length: 0, current: start, closed: n === 1 }),
  ];
  for (let k = 1; k < n; k++) {
    const cur = tour[tour.length - 1];
    let bestJ = -1,
      bestD = Infinity;
    for (let j = 0; j < n; j++) {
      if (!visited[j] && D[cur][j] < bestD - tol) {
        bestJ = j;
        bestD = D[cur][j];
      }
    }
    visited[bestJ] = true;
    tour.push(bestJ);
    length += bestD;
    trace.push(
      mkStep(k, [...tour], length, { tour: [...tour], length, current: bestJ, closed: false }),
    );
  }
  let nIter = 0;
  if (n > 1) {
    length = tourLength(D, tour);
    nIter = n;
    trace.push(
      mkStep(n, [...tour], length, { tour: [...tour], length, current: start, closed: true }),
    );
  }
  return finish(
    'tsp_nearest_neighbor',
    inst,
    tour,
    length,
    true,
    `tour complete from city ${start} (heuristic: no optimality guarantee)`,
    nIter,
    0,
    trace,
    { distance_evaluations: (n * (n - 1)) / 2 },
  );
};

// ── 2-opt ──────────────────────────────────────────────────────────────────────────────

export const tspTwoOpt: MethodFn<TspInput> = (problem, opts) => {
  const strategy = String(opts.strategy ?? 'first');
  if (strategy !== 'first' && strategy !== 'best')
    throw new Error(`unknown strategy '${strategy}'; expected 'first' or 'best'`);
  const inst = instance(problem);
  const maxIter = checkInt('max_iter', opts.max_iter ?? 300, 1);
  const recordEvery = checkInt('record_every', opts.record_every ?? 1, 1);
  const D = distances(inst);
  const tol = tolerance(D);
  const t = initialTour(D, String(opts.init ?? 'identity'), tol);
  const n = t.length;
  if (n <= 3) {
    const length = tourLength(D, t);
    return trivial('tsp_two_opt', inst, D, t, { best_length: length, move: null }, '', {
      initial_length: length,
    });
  }

  let length = tourLength(D, t);
  const initialLength = length;
  const trace: Step[] = [
    mkStep(0, [...t], length, { tour: [...t], length, best_length: length, move: null }),
  ];
  let nFev = 0,
    k = 0,
    converged = false;
  let lastMove: Move | null = null;
  for (;;) {
    let best: [number, number, number] | null = null;
    outer: for (let i = 0; i < n - 2; i++) {
      const jEnd = i > 0 ? n : n - 1;
      for (let j = i + 2; j < jEnd; j++) {
        const delta = twoOptDelta(D, t, i, j);
        nFev += 1;
        if (delta < -tol && (best === null || delta < best[0])) {
          best = [delta, i, j];
          if (strategy === 'first') break outer;
        }
      }
    }
    if (best === null) {
      converged = true;
      break;
    }
    if (k === maxIter) break;
    const [delta, i, j] = best;
    const move = twoOptMove(t, i, j, delta);
    reverse(t, i, j);
    length = tourLength(D, t);
    k += 1;
    if (record(k, recordEvery))
      trace.push(mkStep(k, [...t], length, { tour: [...t], length, best_length: length, move }));
    lastMove = move;
  }
  if (trace[trace.length - 1].k !== k)
    trace.push(
      mkStep(k, [...t], length, { tour: [...t], length, best_length: length, move: lastMove }),
    );
  const msg = converged
    ? `2-optimal: no 2-opt move shortens the tour (after ${k} moves)`
    : `reached max_iter=${maxIter} improving moves before a 2-optimal tour`;
  return finish('tsp_two_opt', inst, t, length, converged, msg, k, nFev, trace, {
    initial_length: initialLength,
  });
};

// ── Or-opt ─────────────────────────────────────────────────────────────────────────────

function orOptMoves(n: number): [number, number, number][] {
  const moves: [number, number, number][] = [];
  for (const L of [3, 2, 1]) {
    if (L > n - 3) continue;
    for (let i = 0; i < n; i++) for (let m = 0; m < n - L - 1; m++) moves.push([L, i, m]);
  }
  return moves;
}

const mod = (a: number, n: number) => ((a % n) + n) % n;

export const tspOrOpt: MethodFn<TspInput> = (problem, opts) => {
  const inst = instance(problem);
  const maxIter = checkInt('max_iter', opts.max_iter ?? 300, 1);
  const recordEvery = checkInt('record_every', opts.record_every ?? 1, 1);
  const D = distances(inst);
  const tol = tolerance(D);
  let t = initialTour(D, String(opts.init ?? 'identity'), tol);
  const n = t.length;
  if (n <= 3) {
    const length = tourLength(D, t);
    return trivial('tsp_or_opt', inst, D, t, { best_length: length, move: null }, '', {
      initial_length: length,
    });
  }
  const dd = (a: number, b: number) => D[a][b];

  let length = tourLength(D, t);
  const initialLength = length;
  const trace: Step[] = [
    mkStep(0, [...t], length, { tour: [...t], length, best_length: length, move: null }),
  ];
  const moves = orOptMoves(n);
  let nFev = 0,
    k = 0,
    converged = false;
  let lastMove: Move | null = null;
  const parts = (L: number, i: number) => {
    const seg = Array.from({ length: L }, (_, s) => t[mod(i + s, n)]);
    const p = t[mod(i - 1, n)],
      q = t[mod(i + L, n)];
    const r = Array.from({ length: n - L }, (_, s) => t[mod(i + L + s, n)]);
    return { seg, p, q, r };
  };
  for (;;) {
    let found: [number, number, number, number, boolean] | null = null;
    for (const [L, i, m] of moves) {
      const { seg, p, q, r } = parts(L, i);
      const x = r[m],
        y = r[m + 1];
      const removalGain = dd(p, seg[0]) + dd(seg[L - 1], q) - dd(p, q);
      const orientations = L === 1 ? [false] : [false, true];
      for (const rev of orientations) {
        const first = rev ? seg[L - 1] : seg[0];
        const last = rev ? seg[0] : seg[L - 1];
        const delta = dd(x, first) + dd(last, y) - dd(x, y) - removalGain;
        nFev += 1;
        if (delta < -tol) {
          found = [delta, L, i, m, rev];
          break;
        }
      }
      if (found !== null) break;
    }
    if (found === null) {
      converged = true;
      break;
    }
    if (k === maxIter) break;
    const [delta, L, i, m, rev] = found;
    const { seg, p, q, r } = parts(L, i);
    const x = r[m],
      y = r[m + 1];
    const placed = rev ? [...seg].reverse() : seg;
    t = [...r.slice(0, m + 1), ...placed, ...r.slice(m + 1)];
    length = tourLength(D, t);
    k += 1;
    const [removed, added] = netExchange(
      [
        [p, seg[0]],
        [seg[L - 1], q],
        [x, y],
      ],
      [
        [p, q],
        [x, placed[0]],
        [placed[placed.length - 1], y],
      ],
    );
    lastMove = { segment: seg, removed, added, reversed: rev, delta };
    if (record(k, recordEvery))
      trace.push(
        mkStep(k, [...t], length, { tour: [...t], length, best_length: length, move: lastMove }),
      );
  }
  if (trace[trace.length - 1].k !== k)
    trace.push(
      mkStep(k, [...t], length, { tour: [...t], length, best_length: length, move: lastMove }),
    );
  const msg = converged
    ? `Or-optimal: no segment move shortens the tour (after ${k} moves)`
    : `reached max_iter=${maxIter} improving moves before an Or-optimal tour`;
  return finish('tsp_or_opt', inst, t, length, converged, msg, k, nFev, trace, {
    initial_length: initialLength,
  });
};

// ── Simulated annealing ────────────────────────────────────────────────────────────────

/** Mean of d(i, j) over i < j, summed sequentially in row-major order. */
function meanDistance(D: Matrix): number {
  const n = D.length;
  let total = 0;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) total += D[i][j];
  return total / ((n * (n - 1)) / 2);
}

export const tspSimulatedAnnealing: MethodFn<TspInput> = (problem, opts) => {
  const seed = Number(opts.seed ?? 0);
  const t0 = Number(opts.t0 ?? 1.0);
  const alpha = opts.alpha ?? 0.9995;
  const tMin = Number(opts.t_min ?? 1e-3);
  const inst = instance(problem);
  const maxIter = checkInt('max_iter', opts.max_iter ?? 20_000, 1);
  const recordEvery = checkInt('record_every', opts.record_every ?? 100, 1);
  if (!(Number.isFinite(t0) && t0 > 0 && Number.isFinite(tMin) && tMin > 0))
    throw new Error('t0 and t_min must be positive');
  if (tMin >= t0)
    throw new Error(
      `t_min = ${fmtG(tMin)} must be below t0 = ${fmtG(t0)}: the annealing schedule would be empty`,
    );
  const a = checkUnit('alpha', alpha, false, false);
  const D = distances(inst);
  const tol = tolerance(D);
  const t = initialTour(D, String(opts.init ?? 'identity'), tol);
  const n = t.length;
  const info0 = {
    best_length: tourLength(D, t),
    best_tour: [...t],
    temperature: null,
    acceptance_rate: null,
    move: null,
  };
  if (n <= 3)
    return trivial('tsp_simulated_annealing', inst, D, t, info0, '', { final_temperature: null });
  const dBar = meanDistance(D);
  if (dBar === 0)
    return trivial(
      'tsp_simulated_annealing',
      inst,
      D,
      t,
      info0,
      'all cities coincide: every tour has length 0',
      { final_temperature: null },
    );

  const rng = new Rng(seed);
  let T = t0 * dBar;
  const T_MIN = tMin * dBar;
  let length = tourLength(D, t);
  let bestT = [...t],
    bestLen = length;

  const step = (k: number, temp: number | null, rate: number | null, move: Move | null) =>
    mkStep(k, [...t], length, {
      tour: [...t],
      length,
      best_length: bestLen,
      best_tour: [...bestT],
      temperature: temp,
      acceptance_rate: rate,
      move,
    });

  const trace: Step[] = [step(0, T, null, null)];
  let acceptedWindow = 0,
    proposedWindow = 0;
  let converged = false;
  let k = 0;
  while (k < maxIter) {
    k += 1;
    let i = rng.integers(n);
    let j = (i + 2 + rng.integers(n - 3)) % n;
    if (i > j) [i, j] = [j, i];
    const delta = twoOptDelta(D, t, i, j);
    const accept = delta <= 0 || rng.random() < Math.exp(-delta / T);
    const move = { ...twoOptMove(t, i, j, delta), accepted: accept };
    proposedWindow += 1;
    if (accept) {
      acceptedWindow += 1;
      reverse(t, i, j);
      length = tourLength(D, t);
      if (length < bestLen - tol) {
        bestT = [...t];
        bestLen = length;
      }
    }
    const TUsed = T;
    T = a * T;
    const frozen = T < T_MIN;
    if (record(k, recordEvery) || frozen || k === maxIter) {
      trace.push(step(k, TUsed, acceptedWindow / proposedWindow, move));
      acceptedWindow = proposedWindow = 0;
    }
    if (frozen) {
      converged = true;
      break;
    }
  }
  const msg = converged
    ? `frozen: T = ${fmtG(T)} < t_min·d̄ = ${fmtG(T_MIN)} after ${k} proposals (heuristic stop)`
    : `reached max_iter=${maxIter} proposals before freezing (T = ${fmtG(T)})`;
  return finish('tsp_simulated_annealing', inst, bestT, bestLen, converged, msg, k, k, trace, {
    final_temperature: T,
  });
};

// ── Genetic algorithm ──────────────────────────────────────────────────────────────────

/** Davis' order crossover OX with cut points a ≤ b (inclusive). */
export function orderCrossover(
  p1: readonly number[],
  p2: readonly number[],
  a: number,
  b: number,
): number[] {
  const n = p1.length;
  const child = new Array<number>(n).fill(-1);
  const used = new Array<boolean>(n).fill(false);
  for (let pos = a; pos <= b; pos++) {
    child[pos] = p1[pos];
    used[p1[pos]] = true;
  }
  let write = (b + 1) % n;
  for (let s = 0; s < n; s++) {
    const city = p2[(b + 1 + s) % n];
    if (!used[city]) {
      child[write] = city;
      used[city] = true;
      write = (write + 1) % n;
    }
  }
  return child;
}

export const tspGenetic: MethodFn<TspInput> = (problem, opts) => {
  const seed = Number(opts.seed ?? 0);
  const inst = instance(problem);
  const popSize = checkInt('pop_size', opts.pop_size ?? 40, 2);
  const tournamentSize = checkInt('tournament_size', opts.tournament_size ?? 3, 1);
  const patience = checkInt('patience', opts.patience ?? 60, 1);
  const maxIter = checkInt('max_iter', opts.max_iter ?? 300, 1);
  const recordEvery = checkInt('record_every', opts.record_every ?? 1, 1);
  const crossoverRate = checkUnit('crossover_rate', opts.crossover_rate ?? 0.9);
  const mutationRate = checkUnit('mutation_rate', opts.mutation_rate ?? 0.2);
  const D = distances(inst);
  const tol = tolerance(D);
  const n = inst.coords.length;
  const rng = new Rng(seed);

  let pop = Array.from({ length: popSize }, () => rng.permutation(n));
  let lengths = pop.map((p) => tourLength(D, p));
  let nFev = popSize;

  const argmin = (vals: readonly number[]) => {
    let best = 0;
    for (let idx = 1; idx < vals.length; idx++) if (vals[idx] < vals[best]) best = idx;
    return best;
  };
  const tournament = () => {
    let winner = rng.integers(popSize);
    for (let s = 0; s < tournamentSize - 1; s++) {
      const c = rng.integers(popSize);
      if (lengths[c] < lengths[winner]) winner = c;
    }
    return winner;
  };

  let bestLen = lengths[argmin(lengths)];
  const step = (k: number, stall: number) => {
    const b = argmin(lengths);
    return mkStep(k, [...pop[b]], lengths[b], {
      tour: [...pop[b]],
      length: lengths[b],
      best_length: bestLen,
      mean_length: pySum(lengths) / popSize,
      worst_length: Math.max(...lengths),
      stall,
    });
  };

  let stallRef = bestLen;
  const trace: Step[] = [step(0, 0)];
  let stall = 0;
  let converged = false;
  let k = 0;
  while (k < maxIter) {
    k += 1;
    const elite = argmin(lengths);
    const newPop = [[...pop[elite]]],
      newLen = [lengths[elite]];
    while (newPop.length < popSize) {
      const p1 = tournament(),
        p2 = tournament();
      let child: number[];
      if (rng.random() < crossoverRate) {
        let a = rng.integers(n),
          b = rng.integers(n);
        if (a > b) [a, b] = [b, a];
        child = orderCrossover(pop[p1], pop[p2], a, b);
      } else child = [...pop[p1]];
      if (rng.random() < mutationRate) {
        const i = rng.integers(n),
          j = rng.integers(n);
        [child[i], child[j]] = [child[j], child[i]];
      }
      newPop.push(child);
      newLen.push(tourLength(D, child));
      nFev += 1;
    }
    pop = newPop;
    lengths = newLen;
    const genBest = lengths[argmin(lengths)];
    bestLen = Math.min(bestLen, genBest);
    if (genBest < stallRef - tol) {
      stallRef = genBest;
      stall = 0;
    } else stall += 1;
    const done = stall >= patience;
    if (record(k, recordEvery) || done || k === maxIter) trace.push(step(k, stall));
    if (done) {
      converged = true;
      break;
    }
  }
  const b = argmin(lengths);
  const msg = converged
    ? `stalled: best length unchanged for ${patience} generations (after ${k}; heuristic stop)`
    : `reached max_iter=${maxIter} generations (best still improving within patience)`;
  return finish('tsp_genetic', inst, pop[b], lengths[b], converged, msg, k, nFev, trace);
};

// ── Ant System ─────────────────────────────────────────────────────────────────────────

/** numpy's `logaddexp` (npymath): log(eˣ + eʸ) without overflow. */
export function logaddexp(x: number, y: number): number {
  if (x === y) return x + Math.LN2;
  const tmp = x - y;
  if (tmp > 0) return x + Math.log1p(Math.exp(-tmp));
  if (tmp <= 0) return y + Math.log1p(Math.exp(tmp));
  return tmp; // NaN
}

export const tspAntColony: MethodFn<TspInput> = (problem, opts) => {
  const seed = Number(opts.seed ?? 0);
  const alpha = Number(opts.alpha ?? 1.0);
  const beta = Number(opts.beta ?? 5.0);
  const inst = instance(problem);
  const nAnts = checkInt('n_ants', opts.n_ants ?? 20, 1);
  const patience = checkInt('patience', opts.patience ?? 30, 1);
  const maxIter = checkInt('max_iter', opts.max_iter ?? 100, 1);
  const recordEvery = checkInt('record_every', opts.record_every ?? 1, 1);
  if (!(Number.isFinite(alpha) && alpha >= 0 && Number.isFinite(beta) && beta >= 0))
    throw new Error('alpha and beta must be non-negative');
  const rho = checkUnit('rho', opts.rho ?? 0.5, false, true);
  const D = distances(inst);
  const tol = tolerance(D);
  const n = inst.coords.length;
  const rng = new Rng(seed);

  const nnTour = nearestNeighborTour(D, 0, tol);
  const nnLen = tourLength(D, nnTour);
  const tau0 = nnLen > 0 ? nAnts / nnLen : 1.0;
  const dRef = nnLen > 0 ? nnLen / n : 1.0;
  // ℓ = log(τ/τ₀): 0 off the diagonal, −inf on it.
  let logTauHat: Matrix = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? -Infinity : 0)),
  );
  let positiveMin = Infinity;
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      if (i !== j && D[i][j] > 0) positiveMin = Math.min(positiveMin, D[i][j]);
  const dZero = Number.isFinite(positiveMin) ? Math.min(tol, positiveMin) : 1.0;
  const betaLogEta: Matrix = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      if (!(beta > 0)) return 0;
      const safe = i !== j ? (D[i][j] > 0 ? D[i][j] : dZero) : dRef;
      const ratio = dRef / safe;
      const logEta = Number.isFinite(ratio) ? Math.log(ratio) : Math.log(dRef) - Math.log(safe);
      return beta * logEta;
    }),
  );
  const logKeep = rho < 1 ? Math.log1p(-rho) : -Infinity;
  let nnFallbackSteps = 0;

  const antTour = (logW: Matrix): number[] => {
    const start = rng.integers(n);
    const visited = new Array<boolean>(n).fill(false);
    visited[start] = true;
    const tour = [start];
    for (let s = 0; s < n - 1; s++) {
      const i = tour[tour.length - 1];
      const row = logW[i];
      const cand: number[] = [];
      for (let j = 0; j < n; j++) if (!visited[j]) cand.push(j);
      let top = -Infinity;
      for (const j of cand) if (row[j] > top) top = row[j];
      const u = rng.random();
      let chosen = -1;
      if (Number.isFinite(top)) {
        const w = cand.map((j) => Math.exp(row[j] - top));
        let total = 0;
        for (const wj of w) total += wj;
        const target = u * total;
        let run = 0;
        for (let c = 0; c < cand.length; c++) {
          run += w[c];
          if (target < run) {
            chosen = cand[c];
            break;
          }
        }
        if (chosen < 0) {
          for (let c = cand.length - 1; c >= 0; c--)
            if (w[c] > 0) {
              chosen = cand[c];
              break;
            }
        }
      } else {
        nnFallbackSteps += 1;
        chosen = cand[0];
        for (const j of cand)
          if (D[i][j] < D[i][chosen] || (D[i][j] === D[i][chosen] && j < chosen)) chosen = j;
      }
      visited[chosen] = true;
      tour.push(chosen);
    }
    return tour;
  };

  const pheromone = (): Matrix => logTauHat.map((row) => row.map((l) => tau0 * Math.exp(l)));
  const tauRange = (tau: Matrix): [number, number] => {
    if (n < 2) return [0, 0];
    let lo = Infinity,
      hi = -Infinity;
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        if (i !== j) {
          lo = Math.min(lo, tau[i][j]);
          hi = Math.max(hi, tau[i][j]);
        }
    return [lo, hi];
  };

  let bestT: number[] = [];
  let bestLen = Infinity;
  let tau = pheromone();
  let [lo, hi] = tauRange(tau);
  const trace: Step[] = [
    mkStep(0, [...nnTour], nnLen, {
      tour: [...nnTour],
      length: nnLen,
      best_tour: [],
      best_length: null,
      mean_length: null,
      pheromone: n <= PHEROMONE_MAX_N ? tau : null,
      tau_min: lo,
      tau_max: hi,
      stall: 0,
    }),
  ];
  let nFev = 0,
    stall = 0,
    converged = false,
    k = 0;
  while (k < maxIter) {
    k += 1;
    const logW: Matrix = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) =>
        i === j
          ? -Infinity
          : alpha > 0
            ? alpha * logTauHat[i][j] + betaLogEta[i][j]
            : betaLogEta[i][j],
      ),
    );
    const tours = Array.from({ length: nAnts }, () => antTour(logW));
    const lens = tours.map((tr) => tourLength(D, tr));
    nFev += nAnts;
    const deltaHat: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    tours.forEach((tr, a) => {
      const L = lens[a];
      if (L <= 0) return;
      const depHat = 1.0 / (L * tau0);
      for (let s = 0; s < n; s++) {
        const p = tr[s],
          q = tr[(s + 1) % n];
        if (p !== q) {
          deltaHat[p][q] += depHat;
          deltaHat[q][p] += depHat;
        }
      }
    });
    logTauHat = logTauHat.map((row, i) =>
      row.map((l, j) => logaddexp(logKeep + l, Math.log(deltaHat[i][j]))),
    );
    let itBest = 0;
    for (let idx = 1; idx < nAnts; idx++) if (lens[idx] < lens[itBest]) itBest = idx;
    if (lens[itBest] < bestLen - tol) {
      bestT = [...tours[itBest]];
      bestLen = lens[itBest];
      stall = 0;
    } else stall += 1;
    const done = stall >= patience;
    if (record(k, recordEvery) || done || k === maxIter) {
      tau = pheromone();
      [lo, hi] = tauRange(tau);
      trace.push(
        mkStep(k, [...tours[itBest]], lens[itBest], {
          tour: [...tours[itBest]],
          length: lens[itBest],
          best_tour: [...bestT],
          best_length: bestLen,
          mean_length: pySum(lens) / nAnts,
          pheromone: n <= PHEROMONE_MAX_N ? tau : null,
          tau_min: lo,
          tau_max: hi,
          stall,
        }),
      );
    }
    if (done) {
      converged = true;
      break;
    }
  }
  let msg = converged
    ? `stalled: best length unchanged for ${patience} iterations (after ${k}; heuristic stop)`
    : `reached max_iter=${maxIter} iterations (best still improving within patience)`;
  if (nnFallbackSteps)
    msg +=
      `; ${nnFallbackSteps} construction steps had zero pheromone on every unvisited ` +
      'edge and moved to the nearest city instead';
  return finish('tsp_ant_colony', inst, bestT, bestLen, converged, msg, k, nFev, trace, {
    tau0,
    nn_fallback_steps: nnFallbackSteps,
  });
};

// ── Held–Karp ──────────────────────────────────────────────────────────────────────────

const popcount = (x: number) => {
  let c = 0;
  while (x) {
    x &= x - 1;
    c++;
  }
  return c;
};

export const tspHeldKarp: MethodFn<TspInput> = (problem) => {
  const inst = instance(problem);
  const n = inst.coords.length;
  if (n > HELD_KARP_MAX_N)
    throw new Error(
      `${inst.id}: Held–Karp is limited to n ≤ ${HELD_KARP_MAX_N} cities (got n = ${n})`,
    );
  const D = distances(inst);
  const trace: Step[] = [
    mkStep(0, [0], 0, { subset_size: 0, states: 1, tour: [0], length: 0, closed: n === 1 }),
  ];
  if (n === 1)
    return finish(
      'tsp_held_karp',
      inst,
      [0],
      0,
      true,
      'n = 1: the tour is the single city',
      0,
      0,
      trace,
      {
        states: 1,
        transitions: 0,
      },
    );

  const m = n - 1;
  const size = 1 << m;
  const cost = new Float64Array(size * m).fill(Infinity); // C(S, j) at [S·m + j]
  const parent = new Int32Array(size * m).fill(-1);
  const layers: number[][] = Array.from({ length: m + 1 }, () => []);
  for (let mask = 1; mask < size; mask++) layers[popcount(mask)].push(mask);

  const pathOf = (maskIn: number, jIn: number): number[] => {
    const path: number[] = [];
    let mask = maskIn,
      j = jIn;
    while (j >= 0) {
      path.push(j + 1);
      const prev = parent[mask * m + j];
      mask ^= 1 << j;
      j = prev;
    }
    return [0, ...path.reverse()];
  };

  let transitions = 0,
    states = 0;
  for (let s = 1; s <= m; s++) {
    let layerStates = 0;
    for (const mask of layers[s]) {
      for (let j = 0; j < m; j++) {
        if (!((mask >> j) & 1)) continue;
        layerStates += 1;
        const prev = mask ^ (1 << j);
        if (prev === 0) {
          cost[mask * m + j] = D[0][j + 1];
          continue;
        }
        let bi = 0,
          bc = Infinity;
        for (let i = 0; i < m; i++) {
          const c = cost[prev * m + i] + D[i + 1][j + 1];
          if (c < bc) {
            bc = c;
            bi = i;
          }
        }
        cost[mask * m + j] = bc;
        parent[mask * m + j] = bi;
        transitions += s - 1;
      }
    }
    states += layerStates;
    let bestMask = -1,
      bestJ = -1,
      bestC = Infinity;
    for (const mask of layers[s]) {
      let j = 0;
      for (let jj = 1; jj < m; jj++) if (cost[mask * m + jj] < cost[mask * m + j]) j = jj;
      if (cost[mask * m + j] < bestC) {
        bestMask = mask;
        bestJ = j;
        bestC = cost[mask * m + j];
      }
    }
    const path = pathOf(bestMask, bestJ);
    const length = pathLength(D, path);
    trace.push(
      mkStep(s, path, length, {
        subset_size: s,
        states: layerStates,
        tour: path,
        length,
        closed: false,
      }),
    );
  }
  const full = size - 1;
  let jLast = 0;
  for (let j = 1; j < m; j++)
    if (cost[full * m + j] + D[j + 1][0] < cost[full * m + jLast] + D[jLast + 1][0]) jLast = j;
  const tour = pathOf(full, jLast);
  const length = tourLength(D, tour);
  trace.push(mkStep(n, tour, length, { subset_size: m, states: m, tour, length, closed: true }));
  return finish(
    'tsp_held_karp',
    inst,
    tour,
    length,
    true,
    `exact optimum by Held–Karp DP over ${states} states`,
    n,
    0,
    trace,
    { states, transitions: transitions + m },
  );
};

// ── Registration ───────────────────────────────────────────────────────────────────────

const JM = 'Johnson & McGeoch (1997), The TSP: a case study in local optimization';

const INIT_PARAM = param.choice('init', 'identity', ['identity', 'nearest_neighbor'], {
  help: 'Starting tour: cities in index order, or the nearest-neighbor tour from city 0.',
  label: 'Starting tour',
});
const recordEveryParam = (def: number, max: number, help: string) =>
  param.int('record_every', def, { min: 1, max, help, label: 'Record every' });

registerMethod(
  {
    id: 'tsp_nearest_neighbor',
    family: 'combinatorial',
    name: 'Nearest neighbor',
    params: [
      param.int('start', 0, {
        min: 0,
        max: NN_START_UI_MAX,
        help: 'City where the tour starts (an index 0, …, n−1).',
        label: 'Start city',
      }),
    ],
    needs: ['tsp'],
    order: 'heuristic, O(n²); length ≤ ½(⌈log₂ n⌉ + 1)·L⋆ on metric instances',
    summary: 'From the current city always travel to the closest city not yet visited.',
    references: ['Rosenkrantz, Stearns & Lewis (1977), SIAM J. Comput. 6(3), 563–581', JM],
  },
  tspNearestNeighbor,
  {
    rule: '\\pi_{k+1} = \\arg\\min_{j \\notin \\{\\pi_0, \\dots, \\pi_k\\}} d(\\pi_k, j)',
    intuition:
      'Always walk to the closest city not yet visited, then return home. Early choices are ' +
      'cheap; the last edges pay for them, often with long jumps back across the map.',
    order: 'heuristic · O(n²)',
    pros: ['Fast, deterministic', 'A good start for local search'],
    cons: ['The closing edges can be long: ≤ ½(⌈log₂ n⌉ + 1)·L⋆ only'],
  },
);

registerMethod(
  {
    id: 'tsp_two_opt',
    family: 'combinatorial',
    name: '2-opt local search',
    params: [
      param.choice('strategy', 'first', ['first', 'best'], {
        help: 'Apply the first improving move found, or the best move of the full neighborhood.',
        label: 'Move choice',
      }),
      INIT_PARAM,
      param.int('max_iter', 300, {
        min: 1,
        max: 100_000,
        help: 'Maximum number of improving moves.',
        label: 'Move budget',
      }),
      recordEveryParam(1, 10_000, 'Record every m-th move in the trace.'),
    ],
    needs: ['tsp'],
    order: 'local search; stops at a 2-optimal tour',
    summary: 'Remove two edges and reconnect the tour the other way whenever that shortens it.',
    references: ['Croes (1958), Oper. Res. 6(6), 791–812', JM],
  },
  tspTwoOpt,
  {
    // Cities a, b, c, e (d is the distance): edges (a,b), (c,e) are replaced by (a,c), (b,e).
    rule: '\\Delta = d(a,c) + d(b,e) - d(a,b) - d(c,e) < 0',
    intuition:
      'Two crossing edges can always be uncrossed: remove them, reverse the path between, and ' +
      'reconnect. Repeat until no pair of edges can be exchanged for a shorter pair.',
    order: 'local search · 2-optimal',
    pros: ['No crossing edges at the end (Euclidean)', 'Simple and effective'],
    cons: ['Stops at the first 2-optimal tour, often a few percent above L⋆'],
  },
);

registerMethod(
  {
    id: 'tsp_or_opt',
    family: 'combinatorial',
    name: 'Or-opt local search',
    params: [
      INIT_PARAM,
      param.int('max_iter', 300, {
        min: 1,
        max: 100_000,
        help: 'Maximum number of improving moves.',
        label: 'Move budget',
      }),
      recordEveryParam(1, 10_000, 'Record every m-th move in the trace.'),
    ],
    needs: ['tsp'],
    order: 'local search; stops at an Or-optimal tour',
    summary:
      'Move a chain of 1–3 consecutive cities to a better place in the tour (possibly reversed).',
    references: ['Or (1976), PhD thesis, Northwestern University', JM],
  },
  tspOrOpt,
  {
    rule: '\\Delta = \\bigl[d(x,s_0) + d(s_L,y) - d(x,y)\\bigr] - \\bigl[d(p,s_0) + d(s_L,q) - d(p,q)\\bigr]',
    intuition:
      'Lift a chain of up to three cities out of the tour, close the gap, and insert the chain ' +
      'between two other neighbours, forward or reversed, whenever that is shorter.',
    order: 'local search · Or-optimal',
    pros: ['Repairs misplaced cities that 2-opt cannot reach in one move'],
    cons: ['A local optimum; segment moves cannot uncross every pair of edges'],
  },
);

registerMethod(
  {
    id: 'tsp_simulated_annealing',
    family: 'combinatorial',
    name: 'Simulated annealing (2-opt moves)',
    params: [
      param.float('t0', 1.0, {
        min: 1e-2,
        max: 100.0,
        log: true,
        help: 'Initial temperature, in units of the mean inter-city distance (must exceed t_min).',
        label: 'Initial temperature',
        tex: 'T_0/\\bar d',
      }),
      param.float('alpha', 0.9995, {
        min: 0.9,
        max: 0.99999,
        help: 'Geometric cooling factor: T ← αT after every proposal.',
        label: 'Cooling factor',
        tex: '\\alpha',
      }),
      param.float('t_min', 1e-3, {
        min: 1e-8,
        max: 5e-3,
        log: true,
        help: 'Freezing temperature, in units of the mean inter-city distance (must be below t0).',
        label: 'Freezing temperature',
        tex: 'T_{\\min}/\\bar d',
      }),
      INIT_PARAM,
      param.int('max_iter', 20_000, {
        min: 1,
        max: 1_000_000,
        help: 'Maximum number of proposals.',
        label: 'Proposal budget',
      }),
      recordEveryParam(100, 100_000, 'Record every m-th proposal in the trace.'),
    ],
    needs: ['tsp'],
    order: 'stochastic metaheuristic',
    summary:
      'Propose random 2-opt moves; always accept improvements and accept worse tours with ' +
      'probability exp(−Δ/T) while the temperature T slowly falls.',
    references: [
      'Kirkpatrick, Gelatt & Vecchi (1983), Science 220(4598), 671–680',
      'Černý (1985), J. Optim. Theory Appl. 45(1), 41–51',
      'Aarts & Korst (1989), Simulated Annealing and Boltzmann Machines',
    ],
    deterministic: false,
  },
  tspSimulatedAnnealing,
  {
    rule: 'P(\\text{accept}) = \\min\\bigl(1,\\; e^{-\\Delta / T_k}\\bigr), \\quad T_{k+1} = \\alpha T_k',
    intuition:
      'A random 2-opt move is always kept when it helps and sometimes when it hurts, with a ' +
      'probability that shrinks as the temperature falls. Hot, it wanders; cold, it is 2-opt.',
    order: 'stochastic metaheuristic',
    pros: ['Escapes local optima early on', 'One parameter schedule'],
    cons: ['No certificate; quality depends on the cooling schedule'],
  },
);

registerMethod(
  {
    id: 'tsp_genetic',
    family: 'combinatorial',
    name: 'Genetic algorithm (OX crossover)',
    params: [
      param.int('pop_size', 40, {
        min: 4,
        max: 1000,
        help: 'Number of tours in the population.',
        label: 'Population size',
      }),
      param.float('crossover_rate', 0.9, {
        min: 0.0,
        max: 1.0,
        help: 'Probability that a child is made by order crossover.',
        label: 'Crossover rate',
        tex: 'p_c',
      }),
      param.float('mutation_rate', 0.2, {
        min: 0.0,
        max: 1.0,
        help: 'Probability of a swap mutation per child.',
        label: 'Mutation rate',
        tex: 'p_m',
      }),
      param.int('tournament_size', 3, {
        min: 1,
        max: 20,
        help: 'Individuals drawn per tournament selection.',
        label: 'Tournament size',
      }),
      param.int('patience', 60, {
        min: 1,
        max: 10_000,
        help: 'Stop after this many generations without improvement.',
        label: 'Patience',
      }),
      param.int('max_iter', 300, {
        min: 1,
        max: 100_000,
        help: 'Maximum number of generations.',
        label: 'Generation budget',
      }),
      recordEveryParam(1, 10_000, 'Record every m-th generation in the trace.'),
    ],
    needs: ['tsp'],
    order: 'stochastic metaheuristic',
    summary:
      'Evolve a population of tours: tournament selection, order crossover, swap mutation, elitism.',
    references: [
      'Davis (1985), Applying adaptive algorithms to epistatic domains, IJCAI-85, 162–164 (OX)',
      'Goldberg (1989), Genetic Algorithms in Search, Optimization and Machine Learning, Ch. 5',
      'Larrañaga et al. (1999), Artif. Intell. Rev. 13, 129–170 (TSP operators survey)',
    ],
    deterministic: false,
  },
  tspGenetic,
  {
    rule: '\\text{child} = \\mathrm{OX}(\\pi^{(1)}, \\pi^{(2)}, a, b), \\quad \\pi^{(1)}, \\pi^{(2)} \\text{ by tournament}',
    intuition:
      'Short tours win tournaments and become parents; order crossover keeps a slice of one ' +
      'parent and fills the rest in the order of the other. The best tour always survives.',
    order: 'stochastic metaheuristic',
    pros: ['Population explores many tours at once', 'Elitism: best length never increases'],
    cons: ['Slow to polish a tour without local search', 'Many parameters'],
  },
);

registerMethod(
  {
    id: 'tsp_ant_colony',
    family: 'combinatorial',
    name: 'Ant System (ant colony optimization)',
    params: [
      param.int('n_ants', 20, {
        min: 1,
        max: 500,
        help: 'Ants per iteration (Dorigo et al. suggest m = n).',
        label: 'Ants',
        tex: 'm',
      }),
      param.float('alpha', 1.0, {
        min: 0.0,
        max: 5.0,
        help: 'Pheromone exponent α.',
        label: 'Pheromone exponent',
        tex: '\\alpha',
      }),
      param.float('beta', 5.0, {
        min: 0.0,
        max: 10.0,
        help: 'Visibility exponent β (η = 1/d).',
        label: 'Visibility exponent',
        tex: '\\beta',
      }),
      param.float('rho', 0.5, {
        min: 0.01,
        max: 1.0,
        help: 'Evaporation rate ρ: τ ← (1 − ρ)τ + Σ Δτ.',
        label: 'Evaporation rate',
        tex: '\\rho',
      }),
      param.int('patience', 30, {
        min: 1,
        max: 10_000,
        help: 'Stop after this many iterations without improvement.',
        label: 'Patience',
      }),
      param.int('max_iter', 100, {
        min: 1,
        max: 100_000,
        help: 'Maximum number of colony iterations.',
        label: 'Iteration budget',
      }),
      recordEveryParam(1, 10_000, 'Record every m-th iteration in the trace.'),
    ],
    needs: ['tsp'],
    order: 'stochastic metaheuristic',
    summary:
      'Ants build tours city by city, preferring short edges with much pheromone; ' +
      'pheromone evaporates and short tours deposit more.',
    references: [
      'Dorigo, Maniezzo & Colorni (1996), IEEE Trans. SMC-B 26(1), 29–41 (Ant System)',
      'Dorigo & Stützle (2004), Ant Colony Optimization, Ch. 3 (§3.3.1 Ant System)',
    ],
    deterministic: false,
  },
  tspAntColony,
  {
    // Ant a deposits 1/L_a only on the edges of its own tour T_a (Dorigo & Stützle 2004, eq. 3.3).
    rule:
      '\\begin{aligned} p_{ij} &= \\frac{\\tau_{ij}^{\\alpha}\\, \\eta_{ij}^{\\beta}}{\\sum_{l \\notin \\text{visited}} \\tau_{il}^{\\alpha}\\, \\eta_{il}^{\\beta}} \\\\ ' +
      '\\tau_{ij} &\\leftarrow (1-\\rho)\\,\\tau_{ij} + \\textstyle\\sum_{a} [(i,j) \\in T_a] \\,/\\, L_a \\end{aligned}',
    intuition:
      'Each ant walks a tour, choosing short edges that carry much pheromone. Pheromone ' +
      'evaporates everywhere and is laid back on the edges of the tours, more on shorter ones.',
    order: 'stochastic metaheuristic',
    pros: ['Learns which edges belong to good tours', 'Visible memory: the pheromone map'],
    cons: ['O(m·n²) per iteration', 'Can stagnate on one tour (ρ, α large)'],
  },
);

registerMethod(
  {
    id: 'tsp_held_karp',
    family: 'combinatorial',
    name: 'Held–Karp dynamic programming (exact)',
    params: [],
    needs: ['tsp'],
    order: `exact, O(2ⁿ·n²) time, O(2ⁿ·n) memory (n ≤ ${HELD_KARP_MAX_N})`,
    summary:
      'For every subset of cities and every last city, remember the shortest path from city 0.',
    references: [
      'Held & Karp (1962), J. SIAM 10(1), 196–210',
      'Bellman (1962), J. ACM 9(1), 61–63',
    ],
  },
  tspHeldKarp,
  {
    rule: 'C(S, j) = \\min_{i \\in S \\setminus \\{j\\}} C(S \\setminus \\{j\\}, i) + d(i, j)',
    intuition:
      'The shortest path from city 0 through a set S ending at j only depends on S and j, ' +
      'not on the order inside S. Building the table by |S| gives the optimum exactly.',
    order: 'exact · O(2ⁿ·n²)',
    pros: ['Exact optimum with a certificate', 'Far fewer states than n! tours'],
    cons: [`Exponential memory: limited to n ≤ ${HELD_KARP_MAX_N} here`],
  },
);
