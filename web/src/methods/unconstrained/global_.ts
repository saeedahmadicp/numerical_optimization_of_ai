/**
 * Stochastic global minimization over a search box — TS port of `numopt.unconstrained.global_`
 * (src/numopt/unconstrained/global_.py), family `global`.
 *
 * Methods: simulated annealing, particle swarm, differential evolution, CMA-ES and basin hopping.
 * The conventions are the Python module's:
 *
 * - Search box [lo, hi] = `problem.domain` (an Error, Python's ValueError, without one).
 *   Annealing, swarm and DE keep every iterate inside it; CMA-ES and basin hopping use it to
 *   scale their initial step sizes only.
 * - `x0` (or the problem's default) starts annealing, CMA-ES (the mean) and basin hopping, and is
 *   the first member of the swarm / DE population.
 * - Every random number comes from `new Rng(seed)` (Mulberry32, bit-identical to Python), drawn
 *   in the order each method documents ("Draw order" in the Python docstrings).
 * - Extreme barrier: NaN and +∞ count as +∞; −∞ stops with `converged: false`.
 * - `Step.x`, `Step.fun` are the best point found so far and its value.
 *
 * Step.info keys are the Python ones (snake_case), listed in the Python module docstring.
 * Floating-point operations keep the Python (NumPy) order so the first iterates agree to rounding.
 */
import { registerMethod, param, type MethodDoc } from '../../core/registry';
import { Rng } from '../../core/rng';
import type {
  Matrix,
  MethodFn,
  Params,
  Problem,
  Result,
  RunOptions,
  Step,
  StepInfo,
  Vector,
} from '../../core/types';

/** Machine epsilon of float64. */
const EPS = Number.EPSILON;

type VectorFn = (x: Vector) => number;
export type GlobalProblem = Problem<Vector> | VectorFn;
type Opts = RunOptions & Params;

// ---------------------------------------------------------------------------------------
// Python-style formatting for messages
// ---------------------------------------------------------------------------------------

/** `format(v, ".{p}g")` (Python); ties at the last digit may differ from Python in rare cases. */
export function formatG(v: number, p = 6): string {
  if (Number.isNaN(v)) return 'nan';
  if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf';
  if (v === 0) return Object.is(v, -0) ? '-0' : '0';
  const [mant, expStr] = v.toExponential(p - 1).split('e');
  const exp = Number(expStr);
  if (exp >= -4 && exp < p) {
    const s = v.toFixed(Math.max(0, p - 1 - exp));
    return s.includes('.') ? s.replace(/\.?0+$/, '') : s;
  }
  const m = mant.includes('.') ? mant.replace(/\.?0+$/, '') : mant;
  const e = Math.abs(exp);
  return `${m}e${exp < 0 ? '-' : '+'}${e < 10 ? '0' : ''}${e}`;
}

/** `repr(float)` (Python): `1.0`, `1e-05`, `inf`. */
export function formatRepr(v: number): string {
  if (Number.isNaN(v)) return 'nan';
  if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf';
  const [mant, expStr] = v.toExponential().split('e');
  const exp = Number(expStr);
  if (exp < -4 || exp >= 16) {
    const e = Math.abs(exp);
    return `${mant}e${exp < 0 ? '-' : '+'}${e < 10 ? '0' : ''}${e}`;
  }
  const s = Object.is(v, -0) ? '-0' : String(v);
  return s.includes('.') ? s : `${s}.0`;
}

const g6 = (v: number) => formatG(v, 6);
const g3 = (v: number) => formatG(v, 3);
const list = (x: readonly number[]) => `[${x.map(formatRepr).join(', ')}]`;

// ---------------------------------------------------------------------------------------
// Small vector helpers (NumPy order of operations)
// ---------------------------------------------------------------------------------------

const copy = (x: readonly number[]): Vector => x.slice();
const rows = (points: readonly Vector[]): Vector[] => points.map(copy);
const maxAbs = (x: readonly number[]) => x.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
const maxAbsDiff = (a: readonly number[], b: readonly number[]) =>
  a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);
/** `np.linalg.norm` of a 1-D vector: √(Σ xᵢ²) summed in order. */
function norm2(x: readonly number[]): number {
  let s = 0;
  for (const v of x) s += v * v;
  return Math.sqrt(s);
}
/** `np.mean` of a short 1-D vector (sequential sum, then divide). */
function mean(x: readonly number[]): number {
  let s = 0;
  for (const v of x) s += v;
  return s / x.length;
}
/** Index of the first minimum (`np.argmin`). */
function argmin(x: readonly number[]): number {
  let bi = 0;
  for (let i = 1; i < x.length; i++) if (x[i] < x[bi]) bi = i;
  return bi;
}
const maxOf = (x: readonly number[]) => x.reduce((m, v) => (v > m ? v : m), -Infinity);

// ---------------------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------------------

class InputError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ValueError';
  }
}

/** f with exact counting and the extreme-barrier convention (NaN → +∞). */
class Objective {
  n = 0;
  private readonly fn: VectorFn;
  constructor(fn: VectorFn) {
    this.fn = fn;
  }
  call(x: readonly number[]): number {
    this.n++;
    const v = Number(this.fn(x.slice()));
    return Number.isNaN(v) ? Infinity : v;
  }
}

interface Resolved {
  prob: Problem<Vector>;
  f: VectorFn;
  lo: Vector;
  hi: Vector;
  start: Vector | null;
}

function asVector(x: unknown): Vector {
  if (typeof x === 'number') return [x];
  if (Array.isArray(x)) return (x as unknown[]).flat(Infinity).map(Number);
  throw new InputError(`x0 must be a number or a vector, got ${String(x)}`);
}

function resolve(problem: GlobalProblem, x0: unknown): Resolved {
  const prob: Problem<Vector> =
    typeof problem === 'function'
      ? { id: 'custom', name: 'custom', latex: 'f(x)', dim: 0, domain: [], f: problem }
      : problem;
  const domain = prob.domain ?? [];
  if (!domain.length)
    throw new InputError(
      `${prob.id}: global methods need a search box (problem.domain); for a bare callable ` +
        'pass a problem object with a domain [[lo, hi], ...]',
    );
  // A 1-D scalar problem stores its domain as [lo, hi]; n-D problems as [[lo, hi], ...].
  const pairs = (
    domain.length === 2 && typeof domain[0] === 'number' ? [domain] : domain
  ) as number[][];
  const lo = pairs.map((p) => Number(p[0]));
  const hi = pairs.map((p) => Number(p[1]));
  if (!lo.every((v, i) => Number.isFinite(v) && Number.isFinite(hi[i]) && v < hi[i]))
    throw new InputError(`${prob.id}: invalid search box lo=${list(lo)}, hi=${list(hi)}`);
  const s = x0 !== undefined && x0 !== null ? x0 : (prob.x0 ?? null);
  const start = s === null ? null : asVector(s);
  if (start !== null && start.length !== lo.length)
    throw new InputError(`x0 has ${start.length} entries, the search box has ${lo.length}`);
  const f = prob.f as (x: Vector) => unknown;
  return { prob, f: (x) => Number(f(x)), lo, hi, start };
}

function requireInside(x: Vector, lo: Vector, hi: Vector): void {
  if (!x.every((v, i) => v >= lo[i] && v <= hi[i]))
    throw new InputError(`x0 = ${list(x)} lies outside the search box [${list(lo)}, ${list(hi)}]`);
}

function requireStart(x: Vector | null, prob: Problem<Vector>): Vector {
  if (x === null)
    throw new InputError(`${prob.id}: no starting point given and the problem has no default x0`);
  return x;
}

function checkMaxIter(maxIter: number): void {
  if (!Number.isInteger(maxIter) || maxIter < 1)
    throw new InputError(`max_iter must be a positive integer, got ${maxIter}`);
}

function checkPositive(values: Record<string, number>): void {
  for (const [name, v] of Object.entries(values))
    if (!(v > 0)) throw new InputError(`${name} must be > 0, got ${v}`);
}

function step(
  k: number,
  x: Vector,
  fun: number,
  stepSize: number | null,
  info: StepInfo = {},
): Step {
  return { k, x, fun, gradNorm: null, stepSize, info };
}

function result(
  method: string,
  x: Vector,
  fun: number,
  converged: boolean,
  message: string,
  nIter: number,
  nFev: number,
  trace: Step[],
): Result {
  return { method, x, fun, converged, message, nIter, nFev, nGev: 0, nHev: 0, trace, extra: {} };
}

function unbounded(method: string, x: Vector, trace: Step[], k: number, nfev: number): Result {
  const msg = `f = -inf at x = ${list(x)}: f is unbounded below`;
  return result(method, x, -Infinity, false, msg, k, nfev, trace);
}

function maxIterResult(
  method: string,
  x: Vector,
  fx: number,
  maxIter: number,
  nfev: number,
  trace: Step[],
): Result {
  const msg = `reached max_iter=${maxIter} (best f = ${g6(fx)})`;
  return result(method, x, fx, false, msg, maxIter, nfev, trace);
}

/** One uniform point of the box: n draws, coordinate j = lo_j + (hi_j − lo_j)·u_j. */
function uniformInBox(rng: Rng, lo: Vector, hi: Vector): Vector {
  return lo.map((l, j) => l + (hi[j] - l) * rng.random());
}

/** Population test of the Python `_collapsed`: returns [passed, x-spread, f-spread]. */
function collapsed(
  points: readonly Vector[],
  values: readonly number[],
  best: Vector,
  fBest: number,
  xtol: number,
  ftol: number,
): [boolean, number, number] {
  const xSpread = points.reduce((m, p) => Math.max(m, maxAbsDiff(p, best)), -Infinity);
  const fSpread = maxOf(values) - fBest;
  const tolX = xtol + 2.0 * EPS * maxAbs(best);
  const tolF = ftol + 2.0 * EPS * Math.abs(fBest);
  return [xSpread <= tolX && fSpread <= tolF, xSpread, fSpread];
}

/** Rounding-plateau exit of particle swarm and DE (consecutive generations). */
export const PLATEAU_GENERATIONS = 10;

function onPlateau(fMaxHistory: readonly number[], fBest: number): [boolean, number] {
  if (fMaxHistory.length < PLATEAU_GENERATIONS) return [false, Infinity];
  const fRange = maxOf(fMaxHistory.slice(-PLATEAU_GENERATIONS)) - fBest;
  return [fRange <= 2.0 * EPS * Math.abs(fBest), fRange];
}

function plateauMessage(what: string, xSpread: number, fRange: number, fBest: number): string {
  return (
    `rounding plateau: for ${PLATEAU_GENERATIONS} generations every f of the ${what} was ` +
    `within 2ε|f_best| of f_best (f-range ${g3(fRange)}), so f cannot rank the members; ` +
    `x-spread ${g3(xSpread)}; best f = ${g6(fBest)}`
  );
}

const TOL_PARAMS = [
  param.float('xtol', 1e-6, {
    min: 1e-12,
    max: 1e-1,
    log: true,
    label: 'Spread tolerance',
    tex: '\\max_i \\|\\mathbf{x}_i - \\mathbf{x}_{\\mathrm{best}}\\|_\\infty \\le',
    help: 'Population spread test: every member within xtol (∞-norm) of the best point.',
  }),
  param.float('ftol', 1e-8, {
    min: 1e-14,
    max: 1e-1,
    log: true,
    label: 'Value-spread tolerance',
    tex: '\\max_i f_i - f_{\\mathrm{best}} \\le',
    help: "Population spread test: every member's f within ftol of the best f.",
  }),
];

// ---------------------------------------------------------------------------------------
// Simulated annealing
// ---------------------------------------------------------------------------------------

/**
 * Simulated annealing with Gaussian (Boltzmann) proposals and the Metropolis rule.
 * T_k = T₀·α^{k−1} (geometric) or T₀·ln 2/ln(k + 1) (logarithmic);
 * y = x + σ_k⊙z, σ_k = step·(hi − lo)·√(T_k/T₀); accept with probability min(1, e^{−Δf/T_k}).
 * A proposal outside the box is rejected without evaluating f. Stops (converged) after the
 * first step with T_k ≤ T_min. Draw order per step: z_1..z_n (normal), then the acceptance
 * uniform (always drawn).
 */
export const simulatedAnnealing: MethodFn<GlobalProblem> = (problem, opts: Opts) => {
  const T0 = Number(opts.T0 ?? 10.0);
  const cooling = String(opts.cooling ?? 'geometric');
  const alpha = Number(opts.alpha ?? 0.99);
  const Tmin = Number(opts.T_min ?? 1e-3);
  const stepFrac = Number(opts.step ?? 0.1);
  const maxIter = Number(opts.max_iter ?? 2000);
  const seed = Number(opts.seed ?? 0);
  checkMaxIter(maxIter);
  checkPositive({ T0, T_min: Tmin, step: stepFrac });
  if (cooling !== 'geometric' && cooling !== 'logarithmic')
    throw new InputError(`cooling must be 'geometric' or 'logarithmic', got '${cooling}'`);
  if (!(alpha > 0.0 && alpha < 1.0)) throw new InputError(`alpha must lie in (0, 1), got ${alpha}`);
  const { prob, f, lo, hi, start } = resolve(problem, opts.x0);
  let x = copy(requireStart(start, prob));
  requireInside(x, lo, hi);
  const n = x.length;
  const fobj = new Objective(f);
  const rng = new Rng(seed);
  const sigma0 = lo.map((l, j) => stepFrac * (hi[j] - l));
  let fx = fobj.call(x);
  if (!Number.isFinite(fx)) {
    const msg = `f(x0) is not finite (f = ${formatRepr(fx)}); cannot start`;
    return result('simulated_annealing', x, fx, false, msg, 0, fobj.n, [
      step(0, copy(x), fx, null),
    ]);
  }
  let best = copy(x);
  let fBest = fx;
  const temperature = (k: number) =>
    cooling === 'geometric' ? T0 * alpha ** (k - 1) : (T0 * Math.log(2.0)) / Math.log(k + 1.0);

  const trace: Step[] = [
    step(0, copy(best), fBest, null, {
      best: copy(best),
      best_f: fBest,
      current: copy(x),
      current_f: fx,
      candidate: null,
      proposal_sd: null,
      candidate_f: null,
      inside: null,
      accept_prob: null,
      accepted: null,
      temperature: T0,
    }),
  ];
  let k = 0;
  for (;;) {
    k += 1;
    const T = temperature(k);
    const z: Vector = [];
    for (let j = 0; j < n; j++) z.push(rng.normal());
    const u = rng.random();
    const root = Math.sqrt(T / T0);
    const sigma = sigma0.map((s) => s * root); // Boltzmann annealing: proposal variance ∝ T
    const y = x.map((v, j) => v + sigma[j] * z[j]);
    const inside = y.every((v, j) => v >= lo[j] && v <= hi[j]);
    let fy: number | null = null;
    let probAcc: number;
    if (inside) {
      fy = fobj.call(y);
      const delta = fy - fx;
      probAcc = delta <= 0.0 ? 1.0 : Math.exp(-delta / T);
    } else {
      probAcc = 0.0;
    }
    const accepted = inside && u < probAcc;
    if (accepted) {
      x = y;
      fx = fy as number;
      if (fx < fBest) {
        best = copy(x);
        fBest = fx;
      }
    }
    trace.push(
      step(k, copy(best), fBest, norm2(sigma.map((s, j) => s * z[j])), {
        best: copy(best),
        best_f: fBest,
        current: copy(x),
        current_f: fx,
        candidate: y,
        proposal_sd: sigma,
        candidate_f: fy,
        inside,
        accept_prob: probAcc,
        accepted,
        temperature: T,
      }),
    );
    if (fBest === -Infinity) return unbounded('simulated_annealing', best, trace, k, fobj.n);
    if (T <= Tmin) {
      const msg = `temperature T = ${g3(T)} ≤ Tₘᵢₙ = ${g3(Tmin)} (frozen); best f = ${g6(fBest)}`;
      return result('simulated_annealing', best, fBest, true, msg, k, fobj.n, trace);
    }
    if (k === maxIter)
      return maxIterResult('simulated_annealing', best, fBest, maxIter, fobj.n, trace);
  }
};

// ---------------------------------------------------------------------------------------
// Particle swarm optimization
// ---------------------------------------------------------------------------------------

/** Clerc–Kennedy constriction for φ₁ = φ₂ = 2.05: χ = 2/|2 − φ − √(φ² − 4φ)|, φ = 4.1. */
const PHI = 4.1;
export const CHI = 2.0 / Math.abs(2.0 - PHI - Math.sqrt(PHI * PHI - 4.0 * PHI));
export const C_CONSTRICTION = CHI * 2.05;

/**
 * Global-best particle swarm with inertia weight (synchronous update):
 * v_i ← w v_i + c₁ r₁ ⊙ (p_i − x_i) + c₂ r₂ ⊙ (g − x_i),  x_i ← x_i + v_i.
 * Absorbing walls: a coordinate that leaves the box is set to the bound and its velocity to 0.
 * Stops on a collapsed swarm or a 10-generation rounding plateau. Draw order: initial positions
 * (particle 0 is x0), initial velocities, then per iteration and particle r₁ (n) then r₂ (n).
 */
export const particleSwarm: MethodFn<GlobalProblem> = (problem, opts: Opts) => {
  const N = Number(opts.n_particles ?? 20);
  const w = Number(opts.w ?? CHI);
  const c1 = Number(opts.c1 ?? C_CONSTRICTION);
  const c2 = Number(opts.c2 ?? C_CONSTRICTION);
  const xtol = Number(opts.xtol ?? 1e-6);
  const ftol = Number(opts.ftol ?? 1e-8);
  const maxIter = Number(opts.max_iter ?? 1000);
  const seed = Number(opts.seed ?? 0);
  checkMaxIter(maxIter);
  if (N < 2) throw new InputError(`n_particles must be ≥ 2, got ${N}`);
  for (const [name, v] of [
    ['w', w],
    ['c1', c1],
    ['c2', c2],
  ] as const)
    if (!(v >= 0.0)) throw new InputError(`${name} must be ≥ 0, got ${v}`);
  checkPositive({ xtol, ftol });
  const { f, lo, hi, start } = resolve(problem, opts.x0);
  const n = lo.length;
  const fobj = new Objective(f);
  const rng = new Rng(seed);

  const xs: Vector[] = [];
  if (start !== null) {
    requireInside(start, lo, hi);
    xs.push(copy(start));
  }
  while (xs.length < N) xs.push(uniformInBox(rng, lo, hi));
  let vs: Vector[] = xs.map((x) => {
    const r = uniformInBox(rng, lo, hi);
    return r.map((v, j) => 0.5 * (v - x[j]));
  });
  let fs = xs.map((x) => fobj.call(x));
  const pbest = rows(xs);
  const pbestF = fs.slice();
  let gi = argmin(pbestF);
  let g = copy(pbest[gi]);
  let gF = pbestF[gi];

  const info = (): StepInfo => ({
    best: copy(g),
    best_f: gF,
    particles: rows(xs),
    particles_f: fs.slice(),
    velocities: rows(vs),
    personal_best: rows(pbest),
    personal_best_f: pbestF.slice(),
    global_best: copy(g),
    global_best_f: gF,
  });

  const trace: Step[] = [step(0, copy(g), gF, null, info())];
  if (gF === -Infinity) return unbounded('particle_swarm', copy(g), trace, 0, fobj.n);
  const fMaxHistory: number[] = [];
  let k = 0;
  for (;;) {
    k += 1;
    const nextV: Vector[] = vs.slice();
    for (let i = 0; i < N; i++) {
      const r1: number[] = [];
      for (let j = 0; j < n; j++) r1.push(rng.random());
      const r2: number[] = [];
      for (let j = 0; j < n; j++) r2.push(rng.random());
      const xi = xs[i];
      let v = vs[i].map(
        (vj, j) => w * vj + c1 * r1[j] * (pbest[i][j] - xi[j]) + c2 * r2[j] * (g[j] - xi[j]),
      );
      let x = xi.map((xj, j) => xj + v[j]);
      const out = x.map((xj, j) => xj < lo[j] || xj > hi[j]);
      x = x.map((xj, j) => (xj < lo[j] ? lo[j] : xj > hi[j] ? hi[j] : xj));
      v = v.map((vj, j) => (out[j] ? 0.0 : vj));
      xs[i] = x;
      nextV[i] = v;
    }
    vs = nextV;
    fs = xs.map((x) => fobj.call(x));
    for (let i = 0; i < N; i++) {
      if (fs[i] < pbestF[i]) {
        pbest[i] = copy(xs[i]);
        pbestF[i] = fs[i];
      }
    }
    gi = argmin(pbestF);
    g = copy(pbest[gi]);
    gF = pbestF[gi];
    const stepSize = maxOf(vs.map(norm2));
    trace.push(step(k, copy(g), gF, stepSize, info()));
    if (gF === -Infinity) return unbounded('particle_swarm', copy(g), trace, k, fobj.n);
    const [done, xSpread, fSpread] = collapsed(xs, fs, g, gF, xtol, ftol);
    if (done) {
      const msg = `swarm collapsed: x-spread ${g3(xSpread)}, f-spread ${g3(fSpread)}; best f = ${g6(gF)}`;
      return result('particle_swarm', copy(g), gF, true, msg, k, fobj.n, trace);
    }
    fMaxHistory.push(maxOf(fs));
    const [flat, fRange] = onPlateau(fMaxHistory, gF);
    if (flat) {
      const msg = plateauMessage('swarm', xSpread, fRange, gF);
      return result('particle_swarm', copy(g), gF, true, msg, k, fobj.n, trace);
    }
    if (k === maxIter) return maxIterResult('particle_swarm', copy(g), gF, maxIter, fobj.n, trace);
  }
};

// ---------------------------------------------------------------------------------------
// Differential evolution
// ---------------------------------------------------------------------------------------

/** Uniform index in [0, size) not in `exclude`, by rejection (one uniform per draw). */
function distinctIndex(rng: Rng, size: number, exclude: readonly number[]): number {
  for (;;) {
    const r = rng.integers(size);
    if (!exclude.includes(r)) return r;
  }
}

/**
 * DE/rand/1/bin or DE/best/1/bin (Storn & Price 1997), synchronous generations.
 * Mutant v_i = x_{r₁} + F(x_{r₂} − x_{r₃}) (rand) or x_best + F(x_{r₁} − x_{r₂}) (best);
 * binomial crossover u_ij = v_ij if U_j ≤ CR or j = j_rand; a component outside the box is reset
 * halfway between the bound and x_ij; select u_i if f(u_i) ≤ f(x_i). Draw order per target:
 * r₁, r₂, (r₃) by rejection, j_rand, then U_1..U_n (all drawn).
 */
export const differentialEvolution: MethodFn<GlobalProblem> = (problem, opts: Opts) => {
  const NP = Number(opts.pop_size ?? 20);
  const F = Number(opts.F ?? 0.8);
  const CR = Number(opts.CR ?? 0.9);
  const strategy = String(opts.strategy ?? 'rand/1/bin');
  const xtol = Number(opts.xtol ?? 1e-6);
  const ftol = Number(opts.ftol ?? 1e-8);
  const maxIter = Number(opts.max_iter ?? 300);
  const seed = Number(opts.seed ?? 0);
  checkMaxIter(maxIter);
  if (strategy !== 'rand/1/bin' && strategy !== 'best/1/bin')
    throw new InputError(`strategy must be 'rand/1/bin' or 'best/1/bin', got '${strategy}'`);
  const need = strategy === 'rand/1/bin' ? 4 : 3;
  if (NP < need) throw new InputError(`${strategy} needs pop_size ≥ ${need}, got ${NP}`);
  if (!(F >= 0.0)) throw new InputError(`F must be ≥ 0, got ${F}`);
  if (!(CR >= 0.0 && CR <= 1.0)) throw new InputError(`CR must lie in [0, 1], got ${CR}`);
  checkPositive({ xtol, ftol });
  const { f, lo, hi, start } = resolve(problem, opts.x0);
  const n = lo.length;
  const fobj = new Objective(f);
  const rng = new Rng(seed);

  let pop: Vector[] = [];
  if (start !== null) {
    requireInside(start, lo, hi);
    pop.push(copy(start));
  }
  while (pop.length < NP) pop.push(uniformInBox(rng, lo, hi));
  let popF = pop.map((x) => fobj.call(x));
  let bi = argmin(popF);

  const info = (
    mutants: Vector[],
    trials: Vector[],
    trialsF: number[],
    accepted: boolean[],
  ): StepInfo => ({
    best: copy(pop[bi]),
    best_f: popF[bi],
    population: rows(pop),
    population_f: popF.slice(),
    mutants,
    trials,
    trials_f: trialsF,
    accepted,
    best_index: bi,
  });

  const trace: Step[] = [step(0, copy(pop[bi]), popF[bi], null, info([], [], [], []))];
  if (popF[bi] === -Infinity)
    return unbounded('differential_evolution', copy(pop[bi]), trace, 0, fobj.n);
  const fMaxHistory: number[] = [];
  let k = 0;
  for (;;) {
    k += 1;
    const mutants: Vector[] = [];
    const trials: Vector[] = [];
    const trialsF: number[] = [];
    const accepted: boolean[] = [];
    const newPop = rows(pop);
    const newF = popF.slice();
    const xBest = pop[bi];
    for (let i = 0; i < NP; i++) {
      const r1 = distinctIndex(rng, NP, [i]);
      const r2 = distinctIndex(rng, NP, [i, r1]);
      let v: Vector;
      if (strategy === 'rand/1/bin') {
        const r3 = distinctIndex(rng, NP, [i, r1, r2]);
        v = pop[r1].map((a, j) => a + F * (pop[r2][j] - pop[r3][j]));
      } else {
        v = xBest.map((a, j) => a + F * (pop[r1][j] - pop[r2][j]));
      }
      const jRand = rng.integers(n);
      const cross: boolean[] = [];
      for (let j = 0; j < n; j++) cross.push(rng.random() <= CR || j === jRand);
      const xi = pop[i];
      let u = v.map((vj, j) => (cross[j] ? vj : xi[j]));
      u = u.map((uj, j) => (uj < lo[j] ? 0.5 * (lo[j] + xi[j]) : uj));
      u = u.map((uj, j) => (uj > hi[j] ? 0.5 * (hi[j] + xi[j]) : uj));
      const fu = fobj.call(u);
      const take = fu <= popF[i];
      if (take) {
        newPop[i] = copy(u);
        newF[i] = fu;
      }
      mutants.push(v);
      trials.push(u);
      trialsF.push(fu);
      accepted.push(take);
    }
    pop = newPop;
    popF = newF;
    bi = argmin(popF);
    trace.push(step(k, copy(pop[bi]), popF[bi], null, info(mutants, trials, trialsF, accepted)));
    if (popF[bi] === -Infinity)
      return unbounded('differential_evolution', copy(pop[bi]), trace, k, fobj.n);
    const [done, xSpread, fSpread] = collapsed(pop, popF, pop[bi], popF[bi], xtol, ftol);
    if (done) {
      const msg =
        `population collapsed: x-spread ${g3(xSpread)}, f-spread ${g3(fSpread)}; ` +
        `best f = ${g6(popF[bi])}`;
      return result('differential_evolution', copy(pop[bi]), popF[bi], true, msg, k, fobj.n, trace);
    }
    fMaxHistory.push(maxOf(popF));
    const [flat, fRange] = onPlateau(fMaxHistory, popF[bi]);
    if (flat) {
      const msg = plateauMessage('population', xSpread, fRange, popF[bi]);
      return result('differential_evolution', copy(pop[bi]), popF[bi], true, msg, k, fobj.n, trace);
    }
    if (k === maxIter)
      return maxIterResult(
        'differential_evolution',
        copy(pop[bi]),
        popF[bi],
        maxIter,
        fobj.n,
        trace,
      );
  }
};

// ---------------------------------------------------------------------------------------
// CMA-ES
// ---------------------------------------------------------------------------------------

/** Hansen (2016), App. B.3 ConditionCov: stop when κ(C) exceeds this. */
const CMA_MAX_COND = 1e14;

/**
 * Symmetric eigendecomposition A = B diag(λ) Bᵀ by cyclic Jacobi rotations (LAPACK's `eigh`
 * in NumPy). Eigenvalues ascending; the columns of B are the eigenvectors.
 */
export function eighSym(A: Matrix): { values: number[]; vectors: Matrix } {
  const n = A.length;
  const a = A.map((r) => r.slice());
  const v: Matrix = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  );
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
    if (off === 0 || !Number.isFinite(off)) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p][q];
        if (apq === 0) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * apq);
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let r = 0; r < n; r++) {
          const arp = a[r][p];
          const arq = a[r][q];
          a[r][p] = c * arp - s * arq;
          a[r][q] = s * arp + c * arq;
        }
        for (let r = 0; r < n; r++) {
          const apr = a[p][r];
          const aqr = a[q][r];
          a[p][r] = c * apr - s * aqr;
          a[q][r] = s * apr + c * aqr;
        }
        for (let r = 0; r < n; r++) {
          const vrp = v[r][p];
          const vrq = v[r][q];
          v[r][p] = c * vrp - s * vrq;
          v[r][q] = s * vrp + c * vrq;
        }
      }
    }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => a[i][i] - a[j][j]);
  return {
    values: order.map((i) => a[i][i]),
    vectors: Array.from({ length: n }, (_, r) => order.map((i) => v[r][i])),
  };
}

/** C^{1/2} = B D Bᵀ from C = B D² Bᵀ (Python `_sym_sqrt`). */
function symSqrt(eigval: readonly number[], B: Matrix): Matrix {
  const n = B.length;
  const BD = B.map((row) => row.map((b, l) => b * Math.sqrt(eigval[l])));
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      let s = 0;
      for (let l = 0; l < n; l++) s += BD[i][l] * B[j][l];
      return s;
    }),
  );
}

/** Strategy constants of Hansen's Table 1 (positive weights only), exported for the lab. */
export function cmaConstants(n: number, popSize = 0) {
  const lam = popSize >= 2 ? popSize : 4 + Math.floor(3.0 * Math.log(n));
  const mu = Math.floor(lam / 2);
  const wRaw: number[] = [];
  for (let i = 1; i <= mu; i++) wRaw.push(Math.log((lam + 1) / 2.0) - Math.log(i));
  let wSum = 0;
  for (const v of wRaw) wSum += v;
  const weights = wRaw.map((v) => v / wSum);
  let w2 = 0;
  for (const v of weights) w2 += v ** 2;
  const mueff = 1.0 / w2;
  const cSigma = (mueff + 2.0) / (n + mueff + 5.0);
  const dSigma = 1.0 + 2.0 * Math.max(0.0, Math.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cSigma;
  const cC = (4.0 + mueff / n) / (n + 4.0 + (2.0 * mueff) / n);
  const c1 = 2.0 / ((n + 1.3) ** 2 + mueff);
  const cMu = Math.min(
    1.0 - c1,
    (2.0 * (0.25 + mueff + 1.0 / mueff - 2.0)) / ((n + 2.0) ** 2 + mueff),
  );
  const chiN = Math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n));
  const histLen = 10 + Math.ceil((30.0 * n) / lam);
  return { lam, mu, weights, mueff, cSigma, dSigma, cC, c1, cMu, chiN, histLen };
}

/**
 * The (μ/μ_w, λ)-CMA-ES of Hansen's tutorial (2016), Fig. 6, Table 1 defaults (positive weights
 * only). Samples y_k = C^{1/2} z_k with the symmetric root. Stops on TolX or TolFun (converged),
 * ConditionCov, non-finite σ/C or a lost positive definiteness (not converged).
 * Draw order: per generation z_1..z_λ, each with n normals.
 */
export const cmaEs: MethodFn<GlobalProblem> = (problem, opts: Opts) => {
  const sigma0 = Number(opts.sigma0 ?? 0.3);
  const popSize = Number(opts.pop_size ?? 0);
  const xtol = Number(opts.xtol ?? 1e-10);
  const ftol = Number(opts.ftol ?? 1e-12);
  const maxIter = Number(opts.max_iter ?? 1000);
  const seed = Number(opts.seed ?? 0);
  checkMaxIter(maxIter);
  checkPositive({ sigma0, xtol, ftol });
  if (popSize < 0)
    throw new InputError(`pop_size must be ≥ 0 (0 or 1: the default λ), got ${popSize}`);
  const { prob, f, lo, hi, start } = resolve(problem, opts.x0);
  let m = copy(requireStart(start, prob));
  const n = m.length;
  const fobj = new Objective(f);
  const rng = new Rng(seed);
  const { lam, mu, weights, mueff, cSigma, dSigma, cC, c1, cMu, chiN, histLen } = cmaConstants(
    n,
    popSize,
  );

  let sigma = sigma0 * mean(hi.map((h, j) => h - lo[j]));
  let C: Matrix = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1.0 : 0.0)),
  );
  let { values: eigval, vectors: B } = eighSym(C);
  let pSigma: Vector = new Array<number>(n).fill(0);
  let pC: Vector = new Array<number>(n).fill(0);
  const fM = fobj.call(m);
  if (fM === -Infinity)
    return unbounded('cma_es', copy(m), [step(0, copy(m), fM, null)], 0, fobj.n);
  let best = copy(m);
  let fBest = fM;
  const bestHistory: number[] = [];
  const copyC = (M: Matrix) => M.map((r) => r.slice());

  const info = (
    sample: [Vector, number, Matrix],
    pop: Vector[],
    popF: number[],
    selected: number[],
    hSigma: boolean | null,
  ): StepInfo => ({
    best: copy(best),
    best_f: fBest,
    mean: copy(m),
    sigma,
    covariance: copyC(C),
    sample_mean: sample[0],
    sample_sigma: sample[1],
    sample_covariance: sample[2],
    population: pop,
    population_f: popF,
    selected,
    p_sigma: copy(pSigma),
    p_c: copy(pC),
    h_sigma: hSigma,
  });

  const trace: Step[] = [
    step(0, copy(best), fBest, sigma, info([copy(m), sigma, copyC(C)], [], [], [], null)),
  ];
  let k = 0;
  for (;;) {
    k += 1;
    const sqrtC = symSqrt(eigval, B);
    const sample: [Vector, number, Matrix] = [copy(m), sigma, copyC(C)];
    const Z: Matrix = [];
    for (let i = 0; i < lam; i++) {
      const z: Vector = [];
      for (let j = 0; j < n; j++) z.push(rng.normal());
      Z.push(z);
    }
    // Rows y_k = C^{1/2} z_k (C^{1/2} symmetric): Y = Z @ sqrtC.
    const Y: Matrix = Z.map((z) =>
      Array.from({ length: n }, (_, j) => {
        let s = 0;
        for (let l = 0; l < n; l++) s += z[l] * sqrtC[l][j];
        return s;
      }),
    );
    const X: Matrix = Y.map((y) => m.map((mj, j) => mj + sigma * y[j]));
    const fs = X.map((x) => fobj.call(x));
    // Stable sort: ties keep the sampling order.
    const order = Array.from({ length: lam }, (_, i) => i).sort((a, b) =>
      fs[a] < fs[b] ? -1 : fs[a] > fs[b] ? 1 : 0,
    );
    const sel = order.slice(0, mu);
    if (fs[order[0]] < fBest) {
      best = copy(X[order[0]]);
      fBest = fs[order[0]];
    }
    if (fBest === -Infinity) {
      trace.push(step(k, copy(best), fBest, sigma, info(sample, X.map(copy), fs, sel, null)));
      return unbounded('cma_es', copy(best), trace, k, fobj.n);
    }

    const yW = new Array<number>(n).fill(0);
    const zW = new Array<number>(n).fill(0);
    for (let j = 0; j < n; j++) {
      let sy = 0;
      let sz = 0;
      for (let i = 0; i < mu; i++) {
        sy += weights[i] * Y[sel[i]][j];
        sz += weights[i] * Z[sel[i]][j];
      }
      yW[j] = sy;
      zW[j] = sz;
    }
    m = m.map((mj, j) => mj + sigma * yW[j]);
    const csFac = Math.sqrt(cSigma * (2.0 - cSigma) * mueff);
    pSigma = pSigma.map((p, j) => (1.0 - cSigma) * p + csFac * zW[j]);
    const normPs = norm2(pSigma);
    const hSigma =
      normPs / Math.sqrt(1.0 - (1.0 - cSigma) ** (2 * k)) < (1.4 + 2.0 / (n + 1.0)) * chiN;
    const ccFac = (hSigma ? 1.0 : 0.0) * Math.sqrt(cC * (2.0 - cC) * mueff);
    pC = pC.map((p, j) => (1.0 - cC) * p + ccFac * yW[j]);
    const deltaH = (hSigma ? 0.0 : 1.0) * cC * (2.0 - cC);
    // Σ w_i y_i y_iᵀ = (Y[sel]ᵀ · diag(w)) @ Y[sel]
    const rankMu: Matrix = Array.from({ length: n }, (_, a) =>
      Array.from({ length: n }, (_, b) => {
        let s = 0;
        for (let i = 0; i < mu; i++) s += Y[sel[i]][a] * weights[i] * Y[sel[i]][b];
        return s;
      }),
    );
    const cFac = 1.0 + c1 * deltaH - c1 - cMu;
    const Cn: Matrix = C.map((row, a) =>
      row.map((cab, b) => cFac * cab + c1 * (pC[a] * pC[b]) + cMu * rankMu[a][b]),
    );
    C = Cn.map((row, a) => row.map((v, b) => 0.5 * (v + Cn[b][a]))); // remove rounding asymmetry
    sigma = sigma * Math.exp((cSigma / dSigma) * (normPs / chiN - 1.0));

    trace.push(step(k, copy(best), fBest, sigma, info(sample, X.map(copy), fs, sel, hSigma)));
    bestHistory.push(fs[order[0]]);

    if (!(Number.isFinite(sigma) && sigma > 0.0 && C.every((r) => r.every(Number.isFinite)))) {
      const msg = `step size or covariance became non-finite or zero (σ = ${formatRepr(sigma)})`;
      return result('cma_es', best, fBest, false, msg, k, fobj.n, trace);
    }
    ({ values: eigval, vectors: B } = eighSym(C));
    if (!(eigval[0] > 0.0)) {
      const msg = `covariance matrix C lost positive definiteness (λ_min(C) = ${g3(eigval[0])})`;
      return result('cma_es', best, fBest, false, msg, k, fobj.n, trace);
    }
    if (eigval[n - 1] > CMA_MAX_COND * eigval[0]) {
      const msg = `ConditionCov: κ(C) = ${g3(eigval[n - 1] / eigval[0])} > 1e14`;
      return result('cma_es', best, fBest, false, msg, k, fobj.n, trace);
    }
    const sd = C.map((row, i) => sigma * Math.sqrt(row[i]));
    if (maxOf(sd) <= xtol && maxOf(pC.map((p) => Math.abs(sigma * p))) <= xtol) {
      const msg = `TolX: σ·max√C_ii = ${g3(maxOf(sd))} ≤ xtol; best f = ${g6(fBest)}`;
      return result('cma_es', best, fBest, true, msg, k, fobj.n, trace);
    }
    if (bestHistory.length >= histLen) {
      const recent = [...bestHistory.slice(-histLen), ...fs];
      const fRange = maxOf(recent) - Math.min(...recent);
      if (fRange <= ftol) {
        const msg = `TolFun: f range ${g3(fRange)} ≤ ftol over the last ${histLen} generations`;
        return result('cma_es', best, fBest, true, msg, k, fobj.n, trace);
      }
    }
    if (k === maxIter) return maxIterResult('cma_es', best, fBest, maxIter, fobj.n, trace);
  }
};

// ---------------------------------------------------------------------------------------
// Basin hopping
// ---------------------------------------------------------------------------------------

/** Local Nelder–Mead searches: tolerances and iteration limit. */
const LOCAL_XTOL = 1e-8;
const LOCAL_FTOL = 1e-10;
export const LOCAL_MAX_ITER = 2000;
/** Two local minima are the same if ‖z − m‖∞ ≤ SAME_MIN_TOL·(1 + ‖m‖∞). */
const SAME_MIN_TOL = 1e-4;
/** A hop improves the best minimum only if it lowers f_best by more than this·(1 + |f_best|). */
const IMPROVE_TOL = 1e-8;

interface LocalResult {
  x: Vector;
  fun: number;
  path: Vector[];
  converged: boolean;
  nfev: number;
}

/**
 * The Nelder–Mead method of `numopt.unconstrained.derivative_free.nelder_mead` (LRWW 1998, §2,
 * coefficients 1, 2, ½, ½), reduced to what basin hopping reads: the best vertex, its value,
 * the best vertex of every iteration, the convergence flag and the evaluation count.
 */
export function nelderMeadLocal(
  f: VectorFn,
  x0: Vector,
  xtol: number,
  ftol: number,
  initialStep: number,
  maxIter: number,
): LocalResult {
  const fobj = new Objective(f);
  const n = x0.length;
  const [rho, chi, gamma, sig] = [1.0, 2.0, 0.5, 0.5];
  const fStart = fobj.call(x0);
  if (!Number.isFinite(fStart))
    return { x: copy(x0), fun: fStart, path: [copy(x0)], converged: false, nfev: fobj.n };
  let simplex: Vector[] = [copy(x0)];
  for (let i = 0; i < n; i++) {
    const v = copy(x0);
    v[i] += initialStep;
    simplex.push(v);
  }
  let fvals = [fStart, ...simplex.slice(1).map((v) => fobj.call(v))];
  const sort = () => {
    const order = Array.from({ length: n + 1 }, (_, i) => i).sort((a, b) =>
      fvals[a] < fvals[b] ? -1 : fvals[a] > fvals[b] ? 1 : 0,
    );
    simplex = order.map((i) => simplex[i]);
    fvals = order.map((i) => fvals[i]);
  };
  const size = () =>
    simplex.slice(1).reduce((m, v) => Math.max(m, maxAbsDiff(v, simplex[0])), -Infinity);
  sort();
  const path: Vector[] = [copy(simplex[0])];
  const done = (converged: boolean): LocalResult => ({
    x: copy(simplex[0]),
    fun: fvals[0],
    path,
    converged,
    nfev: fobj.n,
  });
  if (fvals[0] === -Infinity) return done(false);
  let k = 0;
  for (;;) {
    const tolX = xtol + 2.0 * EPS * maxAbs(simplex[0]);
    const tolF = ftol + 2.0 * EPS * Math.abs(fvals[0]);
    if (size() <= tolX && fvals[n] - fvals[0] <= tolF) return done(true);
    if (k === maxIter) return done(false);
    k += 1;
    const worst = copy(simplex[n]);
    // np.mean(np.stack(simplex[:n]), axis=0): rows added in order, then divided by n.
    const centroid = worst.map((_, j) => {
      let s = 0;
      for (let i = 0; i < n; i++) s += simplex[i][j];
      return s / n;
    });
    const along = (c: number) => centroid.map((cj, j) => cj + c * (cj - worst[j]));
    const xr = along(rho);
    const fr = fobj.call(xr);
    let next: [Vector, number] | null;
    if (fvals[0] <= fr && fr < fvals[n - 1]) {
      next = [xr, fr];
    } else if (fr < fvals[0]) {
      const xe = along(rho * chi);
      const fe = fobj.call(xe);
      next = fe < fr ? [xe, fe] : [xr, fr];
    } else if (fr < fvals[n]) {
      const xc = along(rho * gamma);
      const fc = fobj.call(xc);
      next = fc <= fr ? [xc, fc] : null;
    } else {
      const xcc = centroid.map((cj, j) => cj - gamma * (cj - worst[j]));
      const fcc = fobj.call(xcc);
      next = fcc < fvals[n] ? [xcc, fcc] : null;
    }
    if (next !== null) {
      simplex[n] = next[0];
      fvals[n] = next[1];
    } else {
      for (let i = 1; i <= n; i++) {
        simplex[i] = simplex[0].map((a, j) => a + sig * (simplex[i][j] - a));
        fvals[i] = fobj.call(simplex[i]);
      }
    }
    sort();
    path.push(copy(simplex[0]));
    if (fvals[0] === -Infinity) return done(false);
  }
}

interface MinimumRecord {
  x: Vector;
  f: number;
  hits: number;
}

/**
 * Basin hopping (Wales & Doye 1997): y = x + δ, δ_j ~ U[−s_j, s_j]; z = Nelder–Mead minimum from
 * y; x ← z with probability min(1, e^{−(f(z) − f(x))/T}). Stops after `patience` hops without
 * an improvement of the best minimum (converged only when that minimum's local search
 * converged). Draw order per hop: U_1..U_n, then the acceptance uniform (always drawn).
 */
export const basinHopping: MethodFn<GlobalProblem> = (problem, opts: Opts) => {
  const T = Number(opts.T ?? 1.0);
  const stepFrac = Number(opts.step ?? 0.1);
  const patience = Number(opts.patience ?? 20);
  const maxIter = Number(opts.max_iter ?? 100);
  const seed = Number(opts.seed ?? 0);
  checkMaxIter(maxIter);
  checkPositive({ T, step: stepFrac });
  if (patience < 1) throw new InputError(`patience must be ≥ 1, got ${patience}`);
  const { prob, f, lo, hi, start } = resolve(problem, opts.x0);
  const xStart = requireStart(start, prob);
  const n = xStart.length;
  const rng = new Rng(seed);
  const s = lo.map((l, j) => stepFrac * (hi[j] - l));
  const localStep = 0.5 * mean(s);
  let nfev = 0;
  const local = (y: Vector) => {
    const r = nelderMeadLocal(f, y, LOCAL_XTOL, LOCAL_FTOL, localStep, LOCAL_MAX_ITER);
    nfev += r.nfev;
    return r;
  };

  const first = local(xStart);
  let x = first.x;
  let fx = first.fun;
  if (!Number.isFinite(fx)) {
    let msg = `f is not finite at the first local minimum (f = ${formatRepr(fx)}); cannot start`;
    if (fx === -Infinity) msg = 'f = -inf on the first local search: f is unbounded below';
    return result('basin_hopping', x, fx, false, msg, 0, nfev, [step(0, copy(x), fx, null)]);
  }
  let best = copy(x);
  let fBest = fx;
  let bestOk = first.converged;
  const minima: MinimumRecord[] = [{ x: copy(x), f: fx, hits: 1 }];
  let stall = 0;

  const record = (z: Vector, fz: number) => {
    for (const rec of minima) {
      if (maxAbsDiff(z, rec.x) <= SAME_MIN_TOL * (1.0 + maxAbs(rec.x))) {
        rec.hits += 1;
        if (fz < rec.f) {
          rec.x = copy(z);
          rec.f = fz;
        }
        return;
      }
    }
    minima.push({ x: copy(z), f: fz, hits: 1 });
  };

  const info = (y: Vector, lr: LocalResult, p: number | null, acc: boolean | null): StepInfo => ({
    best: copy(best),
    best_f: fBest,
    current: copy(x),
    current_f: fx,
    start: copy(y),
    local_min: copy(lr.x),
    local_f: lr.fun,
    local_path: lr.path,
    local_converged: lr.converged,
    accept_prob: p,
    accepted: acc,
    minima: minima.map((r) => ({ x: copy(r.x), f: r.f, hits: r.hits })),
    stall,
  });

  const trace: Step[] = [step(0, copy(best), fBest, null, info(xStart, first, null, null))];
  let k = 0;
  for (;;) {
    k += 1;
    const u: number[] = [];
    for (let j = 0; j < n; j++) u.push(rng.random());
    const uAcc = rng.random();
    const y = x.map((xj, j) => xj + s[j] * (2.0 * u[j] - 1.0));
    const lr = local(y);
    const z = lr.x;
    const fz = lr.fun;
    if (fz === -Infinity) {
      trace.push(step(k, copy(z), fz, null, info(y, lr, null, null)));
      return unbounded('basin_hopping', z, trace, k, nfev);
    }
    const delta = fz - fx;
    const pAcc = delta <= 0.0 ? 1.0 : Math.exp(-delta / T);
    const accepted = uAcc < pAcc;
    if (Number.isFinite(fz)) record(z, fz);
    if (fz < fBest - IMPROVE_TOL * (1.0 + Math.abs(fBest))) stall = 0;
    else stall += 1;
    if (fz < fBest) {
      best = copy(z);
      fBest = fz;
      bestOk = lr.converged;
    }
    if (accepted) {
      x = z;
      fx = fz;
    }
    trace.push(
      step(k, copy(best), fBest, norm2(y.map((yj, j) => yj - z[j])), info(y, lr, pAcc, accepted)),
    );
    if (stall >= patience) {
      let msg = `best minimum unchanged for ${patience} hops; best f = ${g6(fBest)}`;
      if (!bestOk)
        msg +=
          `, but the local Nelder–Mead search that found it reached its limit of ` +
          `${LOCAL_MAX_ITER} iterations, so the best point is not a converged local minimum`;
      return result('basin_hopping', best, fBest, bestOk, msg, k, nfev, trace);
    }
    if (k === maxIter) return maxIterResult('basin_hopping', best, fBest, maxIter, nfev, trace);
  }
};

// ---------------------------------------------------------------------------------------
// Registration (same ids, params and metadata as the Python @register calls)
// ---------------------------------------------------------------------------------------

/**
 * The DE update rule: the mutation of the selected strategy (or both, for the catalog), the
 * binomial crossover, the bound repair and the greedy selection.
 */
export function deRule(strategy: 'rand/1/bin' | 'best/1/bin' | 'both'): string {
  const mut =
    strategy === 'rand/1/bin'
      ? '\\mathbf{v}_i &= \\mathbf{x}_{r_1} + F\\,(\\mathbf{x}_{r_2} - \\mathbf{x}_{r_3})\\quad (\\text{DE/rand/1})'
      : strategy === 'best/1/bin'
        ? '\\mathbf{v}_i &= \\mathbf{x}_{\\mathrm{best}} + F\\,(\\mathbf{x}_{r_1} - \\mathbf{x}_{r_2})\\quad (\\text{DE/best/1})'
        : '\\mathbf{v}_i &= \\begin{cases} \\mathbf{x}_{r_1} + F\\,(\\mathbf{x}_{r_2} - \\mathbf{x}_{r_3}) & \\text{rand/1}\\\\ ' +
          '\\mathbf{x}_{\\mathrm{best}} + F\\,(\\mathbf{x}_{r_1} - \\mathbf{x}_{r_2}) & \\text{best/1} \\end{cases}';
  return (
    `\\begin{aligned}${mut}\\\\ ` +
    'u_{ij} &= \\begin{cases} v_{ij} & U_j \\le \\mathrm{CR} \\text{ or } j = j_{\\mathrm{rand}}\\\\ x_{ij} & \\text{else}\\end{cases}\\\\ ' +
    'u_{ij} &\\leftarrow \\tfrac12 (\\ell_j + x_{ij}) \\ \\text{ if } u_{ij} < \\ell_j\\\\ ' +
    'u_{ij} &\\leftarrow \\tfrac12 (h_j + x_{ij}) \\ \\text{ if } u_{ij} > h_j\\\\ ' +
    '\\mathbf{x}_i &\\leftarrow \\mathbf{u}_i \\ \\text{ if } f(\\mathbf{u}_i) \\le f(\\mathbf{x}_i)\\end{aligned}'
  );
}

const DOCS: Record<string, MethodDoc> = {
  simulated_annealing: {
    rule:
      '\\begin{aligned}\\mathbf{y} &= \\mathbf{x}_{k-1} + \\boldsymbol\\sigma_k \\odot \\mathbf{z},\\quad ' +
      '\\mathbf{z} \\sim \\mathcal{N}(0, I)\\\\ ' +
      'p_k &= \\min\\!\\bigl(1,\\ e^{-(f(\\mathbf{y}) - f(\\mathbf{x}_{k-1}))/T_k}\\bigr)\\\\ ' +
      '\\mathbf{x}_k &= \\mathbf{y} \\text{ with probability } p_k, \\text{ else } \\mathbf{x}_{k-1}\\end{aligned}',
    intuition:
      'A random walker that always accepts a downhill proposal and accepts an uphill one with ' +
      'probability e^(−Δf/T). Hot, it crosses ridges between basins; as T cools, it settles into ' +
      'one basin. Only the slow logarithmic schedule guarantees convergence in probability to a ' +
      'global minimizer (Hajek 1988).',
    order: 'no rate',
    pros: [
      'Needs only f values, and any f, smooth or not',
      'Can climb out of a local basin while T is large',
    ],
    cons: [
      'The fast (geometric) schedule can freeze in a side basin',
      'The logarithmic schedule that guarantees convergence is impractically slow',
    ],
    quantities: [
      { tex: 'T_k', key: 'info.temperature' },
      { tex: '\\|\\mathbf{y} - \\mathbf{x}_{k-1}\\|', key: 'stepSize' },
      { tex: 'f(\\mathbf{y})', key: 'info.candidate_f' },
      { tex: 'p_k', key: 'info.accept_prob' },
      { tex: '\\text{accepted}', key: 'info.accepted' },
    ],
  },
  particle_swarm: {
    rule:
      '\\begin{aligned}\\mathbf{v}_i &\\leftarrow w\\,\\mathbf{v}_i + c_1 \\mathbf{r}_1 \\odot (\\mathbf{p}_i - \\mathbf{x}_i) ' +
      '+ c_2 \\mathbf{r}_2 \\odot (\\mathbf{g} - \\mathbf{x}_i)\\\\ ' +
      '\\mathbf{x}_i &\\leftarrow \\mathbf{x}_i + \\mathbf{v}_i\\end{aligned}',
    intuition:
      'Each particle keeps its momentum and is pulled, with random strength, toward the best ' +
      'point it has seen and the best point the swarm has seen. The swarm spreads over the box, ' +
      'then contracts on the best basin it found.',
    order: 'no rate',
    pros: ['Needs only f values', 'Simple, and the swarm explores in parallel'],
    cons: [
      'No convergence guarantee to a minimizer; the swarm can collapse on a local minimum',
      'Sensitive to w, c₁, c₂ outside the constriction region',
    ],
    quantities: [{ tex: '\\max_i \\|\\mathbf{v}_i\\|', key: 'stepSize' }],
  },
  differential_evolution: {
    rule: deRule('both'),
    intuition:
      'Differences between members of the population set the step: wide while the population is ' +
      'spread, small once it clusters. Crossover mixes the mutant with the target coordinate by ' +
      'coordinate, and a trial replaces its target only if it is no worse.',
    order: 'no rate',
    pros: [
      'Step sizes adapt to the population without a schedule',
      'Robust on multimodal and non-smooth f',
    ],
    cons: [
      'best/1 is greedy and can converge prematurely on a side minimum',
      'Many evaluations: NP per generation',
    ],
    quantities: [{ tex: 'i_{\\mathrm{best}}', key: 'info.best_index' }],
  },
  cma_es: {
    rule:
      '\\begin{aligned}\\mathbf{x}_j &= \\mathbf{m} + \\sigma\\,\\mathbf{y}_j,\\quad \\mathbf{y}_j \\sim \\mathcal{N}(0, C)\\\\ ' +
      '\\mathbf{m} &\\leftarrow \\mathbf{m} + \\sigma \\langle \\mathbf{y} \\rangle_w,\\quad \\langle \\mathbf{y} \\rangle_w = \\textstyle\\sum_{i\\le\\mu} w_i \\mathbf{y}_{i:\\lambda}\\\\ ' +
      '\\mathbf{p}_\\sigma &\\leftarrow (1 - c_\\sigma)\\,\\mathbf{p}_\\sigma\\\\ ' +
      '&\\qquad + \\sqrt{c_\\sigma (2 - c_\\sigma)\\,\\mu_{\\mathrm{eff}}}\\; C^{-1/2} \\langle \\mathbf{y} \\rangle_w\\\\ ' +
      '\\mathbf{p}_c &\\leftarrow (1 - c_c)\\,\\mathbf{p}_c\\\\ ' +
      '&\\qquad + h_\\sigma \\sqrt{c_c (2 - c_c)\\,\\mu_{\\mathrm{eff}}}\\; \\langle \\mathbf{y} \\rangle_w\\\\ ' +
      'C &\\leftarrow \\bigl(1 - c_1 - c_\\mu + c_1 \\delta(h_\\sigma)\\bigr)\\, C\\\\ ' +
      '&\\qquad + c_1 \\mathbf{p}_c \\mathbf{p}_c^{\\top} + c_\\mu \\textstyle\\sum_{i\\le\\mu} w_i \\mathbf{y}_{i:\\lambda} \\mathbf{y}_{i:\\lambda}^{\\top}\\\\ ' +
      '\\delta(h_\\sigma) &= (1 - h_\\sigma)\\, c_c (2 - c_c),\\quad h_\\sigma \\in \\{0, 1\\}\\\\ ' +
      '\\sigma &\\leftarrow \\sigma \\exp\\!\\Bigl(\\tfrac{c_\\sigma}{d_\\sigma} \\Bigl(\\tfrac{\\|\\mathbf{p}_\\sigma\\|}{\\mathbb{E}\\|\\mathcal{N}(0, I)\\|} - 1\\Bigr)\\Bigr)\\end{aligned}',
    intuition:
      'Sample a Gaussian cloud, move its mean to a weighted average of the best samples, and ' +
      'stretch its covariance along the steps that worked. The ellipse learns the local shape of ' +
      'f, so the method is invariant to rotations and scaling of the coordinates.',
    order: 'linear on quadratics',
    pros: [
      'Learns the scale and orientation of the landscape',
      'Few parameters; the defaults of Hansen’s Table 1 rarely need tuning',
    ],
    cons: [
      'A small population is local: it can settle in the basin nearest its start',
      'Leaves the box (no bound handling)',
    ],
    quantities: [
      { tex: '\\sigma_{k+1}', key: 'stepSize' },
      { tex: '\\mathbf{m}_{k+1}', key: 'info.mean' },
      { tex: 'h_\\sigma', key: 'info.h_sigma' },
    ],
  },
  basin_hopping: {
    rule:
      '\\begin{aligned}\\mathbf{y} &= \\mathbf{x}_{k-1} + \\boldsymbol\\delta,\\quad \\delta_j \\sim U[-s_j, s_j]\\\\ ' +
      '\\mathbf{z} &= \\operatorname{localmin}(\\mathbf{y})\\\\ ' +
      '\\mathbf{x}_k &= \\mathbf{z} \\text{ w.p. } \\min\\!\\bigl(1,\\ e^{-(f(\\mathbf{z}) - f(\\mathbf{x}_{k-1}))/T}\\bigr), \\text{ else } \\mathbf{x}_{k-1}\\end{aligned}',
    intuition:
      'A Metropolis walk on local minima: kick the current minimum, slide downhill with ' +
      'Nelder–Mead, then accept the new minimum by the Metropolis test. The landscape becomes a ' +
      'staircase of basins, and the walk steps between them.',
    order: 'no rate',
    pros: [
      'Each step lands on a local minimum, so f values are directly comparable',
      'Finds and catalogs many minima',
    ],
    cons: [
      'Each hop costs a full local search',
      'The hop size must match the distance between basins',
    ],
    quantities: [
      { tex: 'f(\\mathbf{z})', key: 'info.local_f' },
      { tex: '\\|\\mathbf{y} - \\mathbf{z}\\|', key: 'stepSize' },
      { tex: 'p_k', key: 'info.accept_prob' },
      { tex: '\\text{accepted}', key: 'info.accepted' },
      { tex: '\\text{stall}', key: 'info.stall' },
    ],
  },
};

registerMethod(
  {
    id: 'simulated_annealing',
    family: 'global',
    name: 'Simulated annealing',
    params: [
      param.float('T0', 10.0, {
        min: 1e-3,
        max: 1e3,
        log: true,
        label: 'Initial temperature',
        tex: 'T_0',
        help: 'Initial temperature T₀.',
      }),
      param.choice('cooling', 'geometric', ['geometric', 'logarithmic'], {
        label: 'Cooling schedule',
        help: 'Schedule: Tₜ = T₀·αᵗ (geometric) or T₀·ln 2/ln(t + 2) (logarithmic).',
      }),
      param.float('alpha', 0.99, {
        min: 0.8,
        max: 0.9999,
        label: 'Cooling factor',
        tex: '\\alpha',
        help: 'Geometric cooling factor α (geometric only).',
      }),
      param.float('T_min', 1e-3, {
        min: 1e-8,
        max: 1.0,
        log: true,
        label: 'Freezing temperature',
        tex: 'T_{\\min}',
        help: 'Stop (frozen) when the temperature of a step is ≤ Tₘᵢₙ (T_min).',
      }),
      param.float('step', 0.1, {
        min: 1e-3,
        max: 1.0,
        log: true,
        label: 'Proposal width',
        tex: '\\sigma_0 / (\\mathrm{hi} - \\mathrm{lo})',
        help: 'Proposal standard deviation at T = T₀, as a fraction of the box width (∝ √T after).',
      }),
      param.int('max_iter', 2000, {
        min: 1,
        max: 100_000,
        label: 'Step budget',
        help: 'Step limit.',
      }),
    ],
    needs: ['f'],
    order: 'global convergence in probability only for logarithmic cooling (Hajek 1988)',
    summary: 'Random-walk proposals, accepted uphill with probability exp(−Δf/T) as T cools.',
    references: [
      'Kirkpatrick, Gelatt & Vecchi (1983), Science 220(4598), 671–680',
      'Ingber (1993), Math. Comput. Modelling 18(11), 29–57 (Boltzmann annealing)',
      'Metropolis et al. (1953), J. Chem. Phys. 21(6), 1087–1092',
      'Geman & Geman (1984), IEEE TPAMI 6(6), 721–741 (logarithmic schedule)',
      'Hajek (1988), Math. Oper. Res. 13(2), 311–329 (convergence of logarithmic cooling)',
    ],
    deterministic: false,
  },
  simulatedAnnealing,
  DOCS.simulated_annealing,
);

registerMethod(
  {
    id: 'particle_swarm',
    family: 'global',
    name: 'Particle swarm',
    params: [
      param.int('n_particles', 20, {
        min: 2,
        max: 200,
        label: 'Swarm size',
        tex: 'N',
        help: 'Swarm size N.',
      }),
      param.float('w', CHI, {
        min: 0.0,
        max: 1.2,
        label: 'Inertia',
        tex: 'w',
        help: 'Inertia weight w (default: Clerc–Kennedy constriction χ ≈ 0.7298).',
      }),
      param.float('c1', C_CONSTRICTION, {
        min: 0.0,
        max: 4.0,
        label: 'Cognitive pull',
        tex: 'c_1',
        help: "Cognitive coefficient: pull toward the particle's own best (default χ·2.05).",
      }),
      param.float('c2', C_CONSTRICTION, {
        min: 0.0,
        max: 4.0,
        label: 'Social pull',
        tex: 'c_2',
        help: "Social coefficient: pull toward the swarm's best (default χ·2.05).",
      }),
      ...TOL_PARAMS,
      param.int('max_iter', 1000, {
        min: 1,
        max: 100_000,
        label: 'Iteration budget',
        help: 'Iteration limit.',
      }),
    ],
    needs: ['f'],
    order: 'no rate; the swarm contracts geometrically when w, c₁, c₂ are in the stable region',
    summary:
      "Particles fly through the box, pulled toward their own best and the swarm's best point.",
    references: [
      'Kennedy & Eberhart (1995), Proc. IEEE ICNN, 1942–1948',
      'Shi & Eberhart (1998), Proc. IEEE CEC, 69–73 (inertia weight)',
      'Clerc & Kennedy (2002), IEEE Trans. Evol. Comput. 6(1), 58–73 (constriction)',
    ],
    deterministic: false,
  },
  particleSwarm,
  DOCS.particle_swarm,
);

registerMethod(
  {
    id: 'differential_evolution',
    family: 'global',
    name: 'Differential evolution',
    params: [
      param.int('pop_size', 20, {
        min: 4,
        max: 200,
        label: 'Population size',
        tex: 'N_P',
        help: 'Population size NP.',
      }),
      param.float('F', 0.8, {
        min: 0.0,
        max: 2.0,
        label: 'Differential weight',
        tex: 'F',
        help: 'Differential weight F (mutation scale).',
      }),
      param.float('CR', 0.9, {
        min: 0.0,
        max: 1.0,
        label: 'Crossover probability',
        tex: '\\mathrm{CR}',
        help: 'Crossover probability CR.',
      }),
      param.choice('strategy', 'rand/1/bin', ['rand/1/bin', 'best/1/bin'], {
        label: 'Strategy',
        help: 'Base vector: a random member (rand) or the best member (best).',
      }),
      ...TOL_PARAMS,
      param.int('max_iter', 300, {
        min: 1,
        max: 100_000,
        label: 'Generation budget',
        help: 'Generation limit.',
      }),
    ],
    needs: ['f'],
    order: 'no rate; a population method',
    summary: 'Mutate with scaled differences of population members, cross over, keep the better.',
    references: [
      'Storn & Price (1997), J. Global Optim. 11(4), 341–359',
      'Price, Storn & Lampinen (2005), Differential Evolution: A Practical Approach, Springer',
    ],
    deterministic: false,
  },
  differentialEvolution,
  DOCS.differential_evolution,
);

registerMethod(
  {
    id: 'cma_es',
    family: 'global',
    name: 'CMA-ES',
    params: [
      param.float('sigma0', 0.3, {
        min: 1e-3,
        max: 1.0,
        log: true,
        label: 'Initial step size',
        tex: '\\sigma_0',
        help: 'Initial step size σ₀ as a fraction of the mean box width (Hansen: ≈ 0.3).',
      }),
      param.int('pop_size', 0, {
        min: 0,
        max: 200,
        label: 'Population size',
        tex: '\\lambda',
        help:
          'Population size λ (0 or 1: the default 4 + ⌊3 ln n⌋, since λ ≥ 2 is needed; ' +
          'larger is more global).',
      }),
      param.float('xtol', 1e-10, {
        min: 1e-14,
        max: 1e-2,
        log: true,
        label: 'TolX',
        tex: '\\sigma\\sqrt{C_{ii}} \\le',
        help: 'TolX: stop when σ·√C_ii and σ·|p_c,i| are ≤ xtol for every i.',
      }),
      param.float('ftol', 1e-12, {
        min: 1e-15,
        max: 1e-2,
        log: true,
        label: 'TolFun',
        tex: '\\Delta f \\le',
        help: "TolFun: stop when the recent best f values and the generation's f span ≤ ftol.",
      }),
      param.int('max_iter', 1000, {
        min: 1,
        max: 100_000,
        label: 'Generation budget',
        help: 'Generation limit.',
      }),
    ],
    needs: ['f'],
    order: 'linear (log-linear in f) on convex quadratics, invariant to rotations and scaling',
    summary:
      'Sample a Gaussian, move its mean to the best samples and learn its covariance from them.',
    references: [
      'Hansen (2016), The CMA Evolution Strategy: A Tutorial, arXiv:1604.00772, Fig. 6, Table 1',
      'Hansen & Ostermeier (2001), Evol. Comput. 9(2), 159–195',
    ],
    deterministic: false,
  },
  cmaEs,
  DOCS.cma_es,
);

registerMethod(
  {
    id: 'basin_hopping',
    family: 'global',
    name: 'Basin hopping',
    params: [
      param.float('T', 1.0, {
        min: 1e-3,
        max: 1e3,
        log: true,
        label: 'Temperature',
        tex: 'T',
        help: 'Temperature of the Metropolis test between local minima.',
      }),
      param.float('step', 0.1, {
        min: 1e-3,
        max: 1.0,
        log: true,
        label: 'Hop half-width',
        tex: 's / (\\mathrm{hi} - \\mathrm{lo})',
        help: 'Hop half-width per coordinate as a fraction of the box width.',
      }),
      param.int('patience', 20, {
        min: 1,
        max: 1000,
        label: 'Patience',
        help: 'Stop when the best minimum has not improved for this many hops.',
      }),
      param.int('max_iter', 100, {
        min: 1,
        max: 10_000,
        label: 'Hop budget',
        help: 'Hop limit.',
      }),
    ],
    needs: ['f'],
    order: 'no rate; a Monte Carlo walk on local minima',
    summary:
      'Jump randomly, slide downhill with Nelder–Mead, accept the new minimum by Metropolis.',
    references: [
      'Wales & Doye (1997), J. Phys. Chem. A 101(28), 5111–5116',
      'Li & Scheraga (1987), PNAS 84(19), 6611–6615 (Monte Carlo minimization)',
    ],
    deterministic: false,
  },
  basinHopping,
  DOCS.basin_hopping,
);
