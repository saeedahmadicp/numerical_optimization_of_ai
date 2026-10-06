/**
 * Pure geometry of the 1-D minimization lab: what each step of each method looked at, read from
 * Step.info exactly as the Python module documents it (src/numopt/scalar/methods.py, "Info keys").
 * No DOM, no React — unit-tested in tests/scalar/geometry.test.ts.
 *
 * Convention (forward-looking, like a tangent in a root finder): at the playhead k the stage shows
 * the state after step k AND what the method does next — the comparison that decides the next
 * cut, the parabola whose vertex is the next trial, the Taylor model whose minimizer is x_{k+1}.
 */
import type { Result, Step } from '../../core/types';
import type { ParabolaInfo } from '../../methods/scalar/methods';
import { m, sup, type MathRun } from '../../viz/mathText';

export type MethodKind = 'interval' | 'parabolic' | 'brent' | 'newton' | 'bracketing';

export const INTERVAL_METHODS = [
  'golden_section',
  'fibonacci_search',
  'dichotomous_search',
  'ternary_search',
] as const;

export function kindOf(methodId: string): MethodKind {
  if ((INTERVAL_METHODS as readonly string[]).includes(methodId)) return 'interval';
  if (methodId === 'parabolic_interpolation') return 'parabolic';
  if (methodId === 'brent_minimize') return 'brent';
  if (methodId === 'newton_1d') return 'newton';
  return 'bracketing';
}

/** Methods that start from x₀ (the others start from the bracket [a, b]). */
export const usesStart = (methodId: string) =>
  methodId === 'newton_1d' || methodId === 'bracket_minimum';

/** A point the method holds after a step, with its name in the method's own notation. */
export interface Probe {
  x: number;
  f: number;
  /** Math label: base letter and subscript, e.g. ['x', '1'], ['λ', ''], ['x', 'm']. */
  name: [string, string];
  /** The point the method reports (lowest f). */
  best?: boolean;
}

export interface StepView {
  k: number;
  /** The interval known (or claimed) to hold a minimizer after this step; null for Newton. */
  bracket: [number, number] | null;
  /** Reported point x̂ₖ and f(x̂ₖ). */
  x: number;
  f: number;
  probes: Probe[];
  /** Points evaluated in this step, [x, f(x)] (k = 0: the setup evaluations). */
  evaluated: [number, number][];
  /** Cumulative evaluations after this step (f, f′ and f″ each count one). */
  nEval: number;
  /** Plain-language kind of the step that produced this state (`golden`, `parabolic`, …). */
  stepKind: string | null;
  /** Interval methods: which end this step cut off. */
  cut: 'left' | 'right' | null;
  /** The region worth looking at for this state (follow camera). */
  roi: [number, number];
}

/** The transition k → k+1: what the method looks at to take the next step. */
export interface NextView {
  /** The parabola that produced the next trial (fit on this step's points / Taylor model). */
  parabola: ParabolaInfo | null;
  /** Its vertex (also when the method rejected it). */
  vertex: number | null;
  /** The point(s) the next step evaluates. */
  trials: [number, number][];
  /** Kind of the next step. */
  stepKind: string | null;
  /** The bracket after the next step (its complement inside this bracket is discarded). */
  nextBracket: [number, number] | null;
  /** Newton: the next iterate. */
  nextX: number | null;
}

const num = (v: unknown): number => (typeof v === 'number' ? v : NaN);
const pair = (v: unknown): [number, number] | null =>
  Array.isArray(v) && v.length >= 2 ? [num(v[0]), num(v[1])] : null;
const triple = (v: unknown): [number, number, number] | null =>
  Array.isArray(v) && v.length >= 3 ? [num(v[0]), num(v[1]), num(v[2])] : null;

function hull(xs: readonly number[]): [number, number] {
  const fin = xs.filter(Number.isFinite);
  if (fin.length === 0) return [0, 1];
  return [Math.min(...fin), Math.max(...fin)];
}

/** Probe names per method (interval methods use the textbook letters). */
function intervalNames(methodId: string): [[string, string], [string, string]] {
  return methodId === 'fibonacci_search'
    ? [
        ['λ', ''],
        ['μ', ''],
      ]
    : [
        ['x', '1'],
        ['x', '2'],
      ];
}

/** Evaluations made by step `s` of a method (Python counts; Newton counts f, f′ and f″). */
function evaluationsOf(kind: MethodKind, s: Step): number {
  const info = s.info;
  switch (kind) {
    case 'interval':
      return Array.isArray(info.evaluated) ? info.evaluated.length : 0;
    case 'parabolic':
      return s.k === 0 ? 3 : 1;
    case 'brent':
      return 1;
    case 'newton': {
      const trials = Array.isArray(info.trials) ? info.trials.length : 0;
      // k = 0: f, f′, f″ at x₀. A Newton step: f, f′, f″ at x_{k+1}. A gradient step: one f per
      // backtracking trial, then f′, f″ at the accepted point.
      return s.k === 0 ? 3 : (trials > 0 ? trials : 1) + 2;
    }
    case 'bracketing':
      return Array.isArray(info.trials) ? info.trials.length : 0;
  }
}

/** Views of every step of a run (one per Step, same indices). */
export function stepViews(methodId: string, trace: readonly Step[]): StepView[] {
  const kind = kindOf(methodId);
  let n = 0;
  return trace.map((s, i) => {
    const info = s.info;
    n += evaluationsOf(kind, s);
    const x = num(s.x);
    const f = s.fun ?? NaN;
    const bracket = pair(info.bracket);
    let probes: Probe[] = [];
    let evaluated: [number, number][] = [];
    let cut: StepView['cut'] = null;
    let roiPts: number[] = bracket ? [...bracket] : [];
    switch (kind) {
      case 'interval': {
        const xs = pair(info.interior);
        const fs = pair(info.f_interior);
        const [n1, n2] = intervalNames(methodId);
        if (xs && fs) {
          probes = [
            { x: xs[0], f: fs[0], name: n1, best: fs[0] <= fs[1] },
            { x: xs[1], f: fs[1], name: n2, best: !(fs[0] <= fs[1]) },
          ];
        }
        const ev = Array.isArray(info.evaluated) ? (info.evaluated as number[]) : [];
        evaluated = ev.map((e) => [e, probes.find((p) => p.x === e)?.f ?? NaN]);
        cut = info.cut === 'left' || info.cut === 'right' ? info.cut : null;
        break;
      }
      case 'parabolic': {
        const t = triple(info.triple);
        const ft = triple(info.f_triple);
        if (t && ft) {
          const names: [string, string][] = [
            ['x', 'l'],
            ['x', 'm'],
            ['x', 'r'],
          ];
          probes = t.map((px, j) => ({ x: px, f: ft[j], name: names[j], best: px === x }));
        }
        if (i === 0 && t && ft) evaluated = t.map((px, j) => [px, ft[j]]);
        else if (typeof info.trial === 'number') evaluated = [[info.trial, num(info.f_trial)]];
        break;
      }
      case 'brent': {
        const t = triple(info.xwv);
        const ft = triple(info.f_xwv);
        if (t && ft) {
          const names: [string, string][] = [
            ['x', ''],
            ['w', ''],
            ['v', ''],
          ];
          // x, w, v coincide at the start; show each distinct point once (x first).
          const seen = new Set<number>();
          probes = [];
          t.forEach((px, j) => {
            if (seen.has(px)) return;
            seen.add(px);
            probes.push({ x: px, f: ft[j], name: names[j], best: j === 0 });
          });
        }
        if (i === 0) evaluated = [[x, f]];
        else if (typeof info.trial === 'number') evaluated = [[info.trial, num(info.f_trial)]];
        break;
      }
      case 'newton': {
        probes = [{ x, f, name: ['x', String(s.k)], best: true }];
        const trials = Array.isArray(info.trials) ? (info.trials as [number, number][]) : [];
        evaluated = trials.length ? [] : [[x, f]];
        roiPts = [x];
        const prev = i > 0 ? num(trace[i - 1].x) : x;
        roiPts.push(prev);
        break;
      }
      case 'bracketing': {
        const t = triple(info.triple);
        const ft = triple(info.f_triple);
        if (t && ft) {
          const names: [string, string][] = [
            ['a', ''],
            ['b', ''],
            ['c', ''],
          ];
          probes = t.map((px, j) => ({ x: px, f: ft[j], name: names[j], best: j === 1 }));
          roiPts = [...t];
        }
        const trials = Array.isArray(info.trials) ? (info.trials as [number, number][]) : [];
        evaluated = trials.map((p) => [num(p[0]), num(p[1])]);
        break;
      }
    }
    const stepKind =
      kind === 'interval'
        ? i === 0
          ? 'init'
          : cut
        : typeof info.step === 'string'
          ? info.step
          : null;
    return {
      k: s.k,
      bracket,
      x,
      f,
      probes,
      evaluated,
      nEval: n,
      stepKind,
      cut,
      roi: hull(roiPts.length ? roiPts : [x]),
    };
  });
}

/** What the method does between step k and k + 1 (null at the last step). */
export function nextView(methodId: string, trace: readonly Step[], k: number): NextView | null {
  const next = trace[k + 1];
  if (!next) return null;
  const kind = kindOf(methodId);
  const info = next.info;
  const parabola = (info.parabola as ParabolaInfo | null | undefined) ?? null;
  const vertex = typeof info.vertex === 'number' ? info.vertex : null;
  let trials: [number, number][] = [];
  if (kind === 'interval') {
    const ev = Array.isArray(info.evaluated) ? (info.evaluated as number[]) : [];
    const xs = pair(info.interior),
      fs = pair(info.f_interior);
    trials = ev.map((e) => [e, xs && fs ? (e === xs[0] ? fs[0] : fs[1]) : NaN]);
  } else if (kind === 'parabolic' || kind === 'brent') {
    if (typeof info.trial === 'number') trials = [[info.trial, num(info.f_trial)]];
  } else if (kind === 'newton') {
    const tr = Array.isArray(info.trials) ? (info.trials as [number, number][]) : [];
    const xk = num(trace[k].x);
    const g = num(trace[k].info.g);
    // Backtracking trials x_k − α f′(x_k) (the last one is accepted).
    trials = tr.length
      ? tr.map(([alpha, fa]) => [xk - alpha * g, fa])
      : [[num(next.x), next.fun ?? NaN]];
  } else {
    const tr = Array.isArray(info.trials) ? (info.trials as [number, number][]) : [];
    trials = tr.map((p) => [num(p[0]), num(p[1])]);
  }
  const stepKind =
    kind === 'interval'
      ? typeof info.cut === 'string'
        ? info.cut
        : null
      : typeof info.step === 'string'
        ? info.step
        : null;
  return {
    parabola: kind === 'interval' ? null : parabola,
    vertex,
    trials,
    stepKind,
    nextBracket: pair(info.bracket),
    nextX: kind === 'newton' ? num(next.x) : null,
  };
}

/** Evaluate p(z) = c0 + c1 (z − c) + c2 (z − c)². */
export function parabolaAt(p: ParabolaInfo, z: number): number {
  const d = z - p.center;
  return p.coef[0] + p.coef[1] * d + p.coef[2] * d * d;
}

/**
 * Proportions of the bracket cut by the two interior probes: (x₁ − a)/L, (x₂ − x₁)/L, (b − x₂)/L.
 * Golden section: ρ, 1 − 2ρ, ρ = 0.382, 0.236, 0.382.
 */
export function proportions(
  a: number,
  b: number,
  x1: number,
  x2: number,
): [number, number, number] {
  const L = b - a;
  return [(x1 - a) / L, (x2 - x1) / L, (b - x2) / L];
}

// ── Convergence measures ──────────────────────────────────────────────────────────────

export type Metric = 'width' | 'error';

/** The known local minimizer nearest to x (the one the run approaches). */
export function nearestMinimizer(minima: readonly number[], x: number): number | null {
  let best: number | null = null;
  for (const m of minima) if (best === null || Math.abs(m - x) < Math.abs(best - x)) best = m;
  return best;
}

/** Values below this are drawn at it on the log chart (|x − x⋆| can be exactly 0). */
export const ERROR_FLOOR = 1e-16;

export function metricValues(
  views: readonly StepView[],
  metric: Metric,
  xStar: number | null,
): (number | null)[] {
  return views.map((v) => {
    if (metric === 'width') return v.bracket ? v.bracket[1] - v.bracket[0] : null;
    if (xStar === null || !Number.isFinite(v.x)) return null;
    return Math.max(Math.abs(v.x - xStar), ERROR_FLOOR);
  });
}

/**
 * Measured contraction per evaluation: (w_end / w_0)^{1/(n_end − n_0)} over the steps with a
 * finite positive width (golden section: 1/φ ≈ 0.618). null when it cannot be measured.
 */
export function ratePerEvaluation(views: readonly StepView[]): number | null {
  const pts = views.filter((v) => v.bracket && v.bracket[1] - v.bracket[0] > 0);
  if (pts.length < 3) return null;
  const first = pts[0],
    last = pts[pts.length - 1];
  const w0 = first.bracket![1] - first.bracket![0];
  const w1 = last.bracket![1] - last.bracket![0];
  const dn = last.nEval - first.nEval;
  if (!(dn > 0) || !(w1 > 0)) return null;
  return (w1 / w0) ** (1 / dn);
}

/** The theoretical per-evaluation rate of an interval method, for the guide lines. */
export const GUIDE_RATES: Record<
  string,
  { rate: number; key: 'golden' | 'dichotomous' | 'ternary' } | undefined
> = {
  golden_section: { rate: 2 / (1 + Math.sqrt(5)), key: 'golden' },
  fibonacci_search: { rate: 2 / (1 + Math.sqrt(5)), key: 'golden' },
  dichotomous_search: { rate: Math.SQRT1_2, key: 'dichotomous' },
  ternary_search: { rate: Math.sqrt(2 / 3), key: 'ternary' },
};

/** Guide labels: ∝ (1/φ)ⁿ, ∝ 2^{−n/2}, ∝ (2/3)^{n/2}. */
export const GUIDE_LABELS: Record<'golden' | 'dichotomous' | 'ternary', MathRun[]> = {
  golden: [m('∝ (1/φ)'), sup('n', 'italic')],
  dichotomous: [m('∝ 2'), sup('−'), sup('n', 'italic'), sup('/2')],
  ternary: [m('∝ (2/3)'), sup('n', 'italic'), sup('/2')],
};

/** Largest result index whose trace is non-empty, for safe indexing. */
export function lastIndex(result: Result): number {
  return Math.max(0, result.trace.length - 1);
}
