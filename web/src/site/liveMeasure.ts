/** What the method page can draw for a run, and which error it plots against k (pure). */
import type { Problem, Result, Step } from '../core/types';
import { b, m, sans, sub, v, type MathRun } from '../viz/mathText';

export type AnyProblem = Problem<unknown> & { domain: unknown[] };

const isPair = (v: unknown): v is [number, number] =>
  Array.isArray(v) && v.length === 2 && v.every((x) => typeof x === 'number');

/** What can be drawn for this run. */
export function figureKind(problem: AnyProblem, result: Result): 'contour' | 'curve' | 'chart' {
  const xs = result.trace.map((s) => s.x);
  if (
    problem.dim === 2 &&
    typeof problem.f === 'function' &&
    Array.isArray(problem.domain) &&
    problem.domain.length === 2 &&
    problem.domain.every(isPair) &&
    xs.length > 0 &&
    xs.every(isPair)
  ) {
    try {
      const v = (problem.f as (x: number[]) => unknown)(xs[0] as number[]);
      if (typeof v === 'number') return 'contour';
    } catch {
      /* not a scalar objective */
    }
  }
  if (
    problem.dim === 1 &&
    typeof problem.f === 'function' &&
    isPair(problem.domain) &&
    xs.length > 0 &&
    xs.every((x) => typeof x === 'number')
  )
    return 'curve';
  return 'chart';
}

export interface ErrorMeasure {
  /** Plain text for the chart's accessible name ("‖∇f(xₖ)‖₂"). */
  label: string;
  /** The y-axis name as typeset math runs (bold 𝐱ₖ for vectors, norm subscript 2). */
  yName: MathRun[];
  values: (number | null)[];
  logY: boolean;
}

/** `x_k` as runs: bold upright for a vector iterate, italic for a scalar one. */
const iterate = (vector: boolean, index = 'k'): MathRun[] => [
  vector ? b('x') : v('x'),
  sub(index, 'italic'),
];
const call = (name: string, vector: boolean, index = 'k'): MathRun[] => [
  v(name),
  m('('),
  ...iterate(vector, index),
  m(')'),
];

/** The error to plot against k: ‖∇f‖, |f| (roots), |estimate − exact|, or the gap to the last f. */
export function errorMeasure(
  family: string,
  problem: AnyProblem,
  trace: readonly Step[],
  methodId = '',
): ErrorMeasure {
  const pos = (x: number | null) => (x !== null && Number.isFinite(x) && x > 0 ? x : null);
  const vector = Array.isArray(trace[0]?.x);
  const grads = trace.filter((s) => typeof s.gradNorm === 'number').length;
  if (grads >= Math.max(2, trace.length / 2))
    return {
      label: vector ? '‖∇f(xₖ)‖₂' : '|f′(xₖ)|',
      yName: vector
        ? [m('‖∇'), ...call('f', true), m('‖'), sub('2')]
        : [m('|'), v('f'), m('′('), ...iterate(false), m(')|')],
      values: trace.map((s) => pos(s.gradNorm)),
      logY: true,
    };
  if (family === 'roots' || family === 'systems')
    return family === 'systems'
      ? {
          label: '‖F(xₖ)‖₂',
          yName: [m('‖'), ...call('F', true), m('‖'), sub('2')],
          values: trace.map((s) => pos(s.fun === null ? null : Math.abs(s.fun))),
          logY: true,
        }
      : {
          label: '|f(xₖ)|',
          yName: [m('|'), ...call('f', vector), m('|')],
          values: trace.map((s) => pos(s.fun === null ? null : Math.abs(s.fun))),
          logY: true,
        };
  const exact = typeof problem.exact === 'number' ? problem.exact : null;
  if (exact !== null)
    return {
      label: '|estimate − exact|',
      yName: [sans('|estimate − exact|')],
      values: trace.map((s) => pos(s.fun === null ? null : Math.abs(s.fun - exact))),
      logY: true,
    };
  // The gap to the last iterate 𝐱_K (K = the iteration count), not a code-style "x_final". A tour
  // is not a point: its value is the tour length L(πₖ) (a knapsack's, the total value zₖ).
  const last = [...trace].reverse().find((s) => typeof s.fun === 'number')?.fun ?? null;
  if (last !== null) {
    const gap = trace.map((s) => pos(s.fun === null ? null : Math.abs(s.fun - last)));
    if (gap.filter((x) => x !== null).length >= 2) {
      if (family === 'combinatorial') {
        const tour = methodId.startsWith('tsp');
        const at = (i: string): MathRun[] =>
          tour ? [v('L'), m('('), v('π'), sub(i, 'italic'), m(')')] : [v('z'), sub(i, 'italic')];
        return {
          label: tour
            ? '|L(πₖ) − L(π_K)|, K the last iteration'
            : '|zₖ − z_K|, K the last iteration',
          yName: [m('|'), ...at('k'), m(' − '), ...at('K'), m('|')],
          values: gap,
          logY: true,
        };
      }
      return {
        label: '|f(xₖ) − f(x_K)|, K the last iteration',
        yName: [m('|'), ...call('f', vector), m(' − '), ...call('f', vector, 'K'), m('|')],
        values: gap,
        logY: true,
      };
    }
  }
  return {
    label: 'f(xₖ)',
    yName: call('f', vector),
    values: trace.map((s) => s.fun),
    logY: false,
  };
}
