/**
 * Pure geometry for the global lab: what each method looked at in a step, read from Step.info
 * (the keys documented in src/numopt/unconstrained/global_.py), plus the playback phase.
 *
 * Playback convention: at local time lt, the step "being built" is K = ⌈lt⌉ with progress
 * p = lt − (K − 1) ∈ (0, 1]. On whole steps (lt = K, p = 1) the frame is the complete state after
 * step K, so a paused or reduced-motion frame is always an exact iterate.
 */
import type { Step, Vector } from '../../core/types';

export type Pt = [number, number];

export const GLOBAL_METHODS = [
  'simulated_annealing',
  'particle_swarm',
  'differential_evolution',
  'cma_es',
  'basin_hopping',
] as const;
export type GlobalMethodId = (typeof GLOBAL_METHODS)[number];

export function isGlobalMethod(id: string): id is GlobalMethodId {
  return (GLOBAL_METHODS as readonly string[]).includes(id);
}

/** The step being built at local time `lt` and its progress p ∈ (0, 1] (p = 1 at k = 0). */
export function phaseAt(lt: number, length: number): { K: number; p: number } {
  const last = Math.max(0, length - 1);
  const t = Math.max(0, Math.min(lt, last));
  if (t <= 0) return { K: 0, p: 1 };
  const K = Math.min(last, Math.ceil(t - 1e-9));
  return { K, p: Math.min(1, Math.max(0, t - (K - 1))) };
}

/** Rescale p to the sub-interval [a, b] of a step (clamped to [0, 1]). */
export function sub(p: number, a: number, b: number): number {
  return Math.min(1, Math.max(0, (p - a) / (b - a)));
}

export const lerp = (a: number, b: number, u: number) => a + (b - a) * u;
export const lerpPt = (a: readonly number[], b: readonly number[], u: number): Pt => [
  lerp(a[0], b[0], u),
  lerp(a[1], b[1], u),
];

/** The point a method's path follows (its "track"): what the line on the landscape means. */
export function trackPoint(method: string, s: Step): Pt {
  const i = s.info;
  const v =
    method === 'simulated_annealing' || method === 'basin_hopping'
      ? ((i.current as Vector | undefined) ?? (s.x as Vector))
      : method === 'cma_es'
        ? ((i.mean as Vector | undefined) ?? (s.x as Vector))
        : (s.x as Vector);
  return [v[0], v[1]];
}

/** What the drawn path is, per method (caption and aria text). */
export const TRACK_NAME: Record<GlobalMethodId, string> = {
  simulated_annealing: 'the Markov chain 𝐱ₖ',
  particle_swarm: 'the swarm’s best point 𝐠',
  differential_evolution: 'the best member',
  cma_es: 'the mean 𝐦ₖ',
  basin_hopping: 'the current local minimum',
};

/**
 * The search scale of a step, in the ∞-norm of x (one measure for the "spread" chart):
 * annealing max_j σ_k,j; swarm and DE max_i ‖𝐱ᵢ − 𝐱_best‖∞ (the spread test); CMA-ES
 * σ·max_i √C_ii (TolX); basin hopping has a fixed hop size, so null.
 */
export function searchScale(method: string, s: Step): number | null {
  const i = s.info;
  switch (method) {
    case 'simulated_annealing': {
      const sd = i.proposal_sd as Vector | null;
      return sd ? Math.max(...sd) : null;
    }
    case 'particle_swarm':
      return spread(i.particles as Vector[] | undefined, s.x as Vector);
    case 'differential_evolution':
      return spread(i.population as Vector[] | undefined, s.x as Vector);
    case 'cma_es': {
      const C = i.covariance as number[][] | undefined;
      const sigma = i.sigma as number | undefined;
      if (!C || sigma === undefined) return null;
      return sigma * Math.sqrt(Math.max(...C.map((r, j) => r[j])));
    }
    default:
      return null;
  }
}

function spread(points: Vector[] | undefined, best: Vector): number | null {
  if (!points?.length) return null;
  let m = 0;
  for (const p of points)
    for (let j = 0; j < p.length; j++) m = Math.max(m, Math.abs(p[j] - best[j]));
  return m;
}

/** Temperature of a step: T_k for annealing, the fixed T for basin hopping (from params). */
export function temperatureOf(method: string, s: Step, T?: number): number | null {
  if (method === 'simulated_annealing') return (s.info.temperature as number | undefined) ?? null;
  if (method === 'basin_hopping' && s.k > 0) return T ?? null;
  return null;
}

/** M = (σ²C)⁻¹ for the ellipse {x : (x − m)ᵀ M (x − m) = r²} of N(m, σ²C) (2-D). */
export function gaussianEllipseMatrix(
  sigma: number,
  C: readonly (readonly number[])[],
): number[][] | null {
  const s2 = sigma * sigma;
  const a = s2 * C[0][0],
    b = s2 * C[0][1],
    d = s2 * C[1][1];
  const det = a * d - b * b;
  if (!(det > 0) || !Number.isFinite(det)) return null;
  return [
    [d / det, -b / det],
    [-b / det, a / det],
  ];
}

/** Linear interpolation of two covariance matrices σ²C (SPD stays SPD). */
export function lerpCov(
  sa: number,
  Ca: readonly (readonly number[])[],
  sb: number,
  Cb: readonly (readonly number[])[],
  u: number,
): number[][] {
  const A = Ca.map((r) => r.map((v) => v * sa * sa));
  const B = Cb.map((r) => r.map((v) => v * sb * sb));
  return A.map((r, i) => r.map((v, j) => lerp(v, B[i][j], u)));
}

/**
 * The DE target whose construction is drawn in full: the first target from `start` (cyclic)
 * whose trial mixes coordinates of the target and the mutant (so the crossover rectangle has a
 * visible corner), else `start`.
 */
export function deHighlight(
  population: readonly Vector[],
  mutants: readonly Vector[],
  trials: readonly Vector[],
  start: number,
): number {
  const NP = trials.length;
  if (NP === 0) return 0;
  for (let o = 0; o < NP; o++) {
    const i = (start + o) % NP;
    const u = trials[i],
      v = mutants[i],
      x = population[i];
    const fromV = u.some((uj, j) => uj === v[j]);
    const fromX = u.some((uj, j) => uj === x[j] && uj !== v[j]);
    if (fromV && fromX) return i;
  }
  return start % NP;
}

/** Number of accepted trials of a DE generation. */
export function acceptedCount(s: Step): number {
  const a = s.info.accepted as boolean[] | boolean | null | undefined;
  return Array.isArray(a) ? a.filter(Boolean).length : 0;
}

/** Clamp a picked point into the search box. */
export function clampToBox(p: readonly number[], domain: readonly (readonly number[])[]): Pt {
  return [
    Math.min(domain[0][1], Math.max(domain[0][0], p[0])),
    Math.min(domain[1][1], Math.max(domain[1][0], p[1])),
  ];
}

/** The global minimum value f⋆ of a problem (extra.f_min, else the least known minimum). */
export function globalMin(problem: {
  extra?: Record<string, unknown>;
  minima?: unknown[];
  f: (x: Vector) => unknown;
}): number | null {
  const fm = problem.extra?.f_min;
  if (typeof fm === 'number' && Number.isFinite(fm)) return fm;
  const vals = (problem.minima ?? []).map((m) => Number(problem.f(m as Vector)));
  return vals.length ? Math.min(...vals) : null;
}

/** The donors of a DE mutant: 𝐯 = base + F·(plus − minus). */
export interface DeDonors {
  /** 𝐱_{r₁} (rand/1) or 𝐱_best (best/1). */
  base: Vector;
  plus: Vector;
  minus: Vector;
  /** Indices into the old population: [r₁, r₂, r₃] (rand/1) or [r₁, r₂] (best/1). */
  r: number[];
}

const donorCache = new WeakMap<object, DeDonors | null>();

/**
 * Recover the donors of target i's mutant from the old population. Step.info records the
 * mutants but not r₁, r₂, r₃ (global_.py); the mutant is recomputed with the same floating-point
 * expression for every admissible index tuple (distinct, ≠ i) and the first exact match wins
 * (else the first within a relative 10⁻¹²). Null when none matches.
 */
export function deDonors(
  old: readonly Vector[],
  bestOld: Vector,
  v: Vector,
  i: number,
  F: number,
  strategy: string,
): DeDonors | null {
  if (donorCache.has(v)) return donorCache.get(v)!;
  const NP = old.length;
  const n = v.length;
  // Exact match first (the port evaluates the same expression); then with a relative slack.
  // A collapsed population has near-duplicate members, so the exact pass must come first.
  const find = (slack: number): DeDonors | null => {
    const near = (a: number, b: number) => Math.abs(a - b) <= slack * (1 + Math.abs(b));
    const match = (base: Vector, p: Vector, m: Vector) => {
      for (let j = 0; j < n; j++) if (!near(base[j] + F * (p[j] - m[j]), v[j])) return false;
      return true;
    };
    for (let r1 = 0; r1 < NP; r1++) {
      if (r1 === i) continue;
      for (let r2 = 0; r2 < NP; r2++) {
        if (r2 === i || r2 === r1) continue;
        if (strategy === 'best/1/bin') {
          if (match(bestOld, old[r1], old[r2]))
            return { base: bestOld, plus: old[r1], minus: old[r2], r: [r1, r2] };
          continue;
        }
        for (let r3 = 0; r3 < NP; r3++) {
          if (r3 === i || r3 === r1 || r3 === r2) continue;
          if (match(old[r1], old[r2], old[r3]))
            return { base: old[r1], plus: old[r2], minus: old[r3], r: [r1, r2, r3] };
        }
      }
    }
    return null;
  };
  const out = find(0) ?? find(1e-12);
  donorCache.set(v, out);
  return out;
}
