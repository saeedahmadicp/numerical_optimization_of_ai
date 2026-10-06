/**
 * The method page's live demo: the method's first parity case, run by the TS port in the browser
 * and replayed on the shared player. A 2-D objective is drawn as a contour landscape with the
 * path, a scalar function as a curve with the iterate (and the bracket when the method keeps
 * one); every run gets a convergence chart. The MethodCard beside it follows the playhead.
 */
import { useMemo } from 'react';
import type { RegisteredMethod } from '../core/registry';
import type { Problem, Result } from '../core/types';
import { MethodCard, type MethodCardQuantity } from '../labs/_shell/blocks';
import { useTracePlayer } from '../play/useTracePlayer';
import { PlaybackBar } from '../ui/components/PlaybackBar';
import { useChartColors } from '../ui/theme';
import { Contour2D } from '../viz/Contour2D';
import { ConvergenceChart } from '../viz/ConvergenceChart';
import { Plot1D, type Overlay1D } from '../viz/Plot1D';
import type { PathSpec } from '../viz/PathLayer';
import { errorMeasure, figureKind } from './liveMeasure';
import styles from './MethodPage.module.css';

type AnyProblem = Problem<unknown> & { domain: unknown[] };

/** Families whose iterate `x` is an estimate (a number), not a location. */
const ESTIMATES = new Set(['integration', 'differentiation']);

const isPair = (v: unknown): v is [number, number] =>
  Array.isArray(v) && v.length === 2 && v.every((x) => typeof x === 'number');

export function LiveRun({
  method,
  problem,
  result,
  family,
  call,
  order,
}: {
  method: RegisteredMethod;
  problem: AnyProblem;
  result: Result;
  family: string;
  call: { problem: string; options: Record<string, unknown>; params: Record<string, unknown> };
  /** The registry's rate string, so the card and the page's facts say it the same way. */
  order?: string;
}) {
  const colors = useChartColors();
  const traces = useMemo(() => [result.trace], [result]);
  const player = useTracePlayer(traces, { autoplay: true });
  const step = result.trace[Math.min(player.k, result.trace.length - 1)];
  // An integral or a derivative estimate is not a point on the x axis: those runs get the
  // convergence chart alone (no iterate drawn on the curve of f).
  const kind = ESTIMATES.has(family) ? 'chart' : figureKind(problem, result);
  const err = useMemo(
    () => errorMeasure(family, problem, result.trace, method.spec.id),
    [family, problem, result, method],
  );
  const end = result.converged ? ('converged' as const) : ('stopped' as const);
  const quantities = useMemo(
    () => familyQuantities(family, method, result),
    [family, method, result],
  );
  // One rate string on the page: the registry's, as the facts above show it.
  const shown = useMemo(
    () => (order && method.doc ? { ...method, doc: { ...method.doc, order } } : method),
    [method, order],
  );

  const paths: PathSpec[] = useMemo(
    () =>
      kind === 'contour'
        ? [
            {
              points: result.trace.map((s) => s.x as [number, number]),
              color: colors.series[0],
              label: method.spec.name,
              end,
            },
          ]
        : [],
    [kind, result, colors, method, end],
  );

  let figure = null;
  if (kind === 'contour') {
    const f = problem.f as (x: number[]) => number;
    const minima = (problem.minima ?? []).filter(isPair);
    figure = (
      <Contour2D
        f={(x, y) => f([x, y])}
        domain={problem.domain as [[number, number], [number, number]]}
        cacheKey={`method-page:${problem.id}`}
        paths={paths}
        t={player.t}
        minima={minima}
        ariaLabel={`Contour plot of ${problem.name} with the ${method.spec.name} path from the start point, ${result.trace.length - 1} iterations.`}
      />
    );
  } else if (kind === 'curve') {
    const f = problem.f as (x: number) => number;
    const x = step.x as number;
    const overlays: Overlay1D[] = [];
    const br = step.info.bracket;
    if (isPair(br)) overlays.push({ kind: 'interval', a: br[0], b: br[1], slot: 0 });
    overlays.push({ kind: 'vline', x, slot: 0, dashed: true });
    let fx: number;
    try {
      fx = f(x);
    } catch {
      fx = 0;
    }
    overlays.push({
      kind: 'point',
      x,
      y: Number.isFinite(fx) ? fx : 0,
      slot: 0,
      label: `x${sub(step.k)}`,
    });
    figure = (
      <Plot1D
        f={f}
        domain={problem.domain as [number, number]}
        overlays={overlays}
        zeroLine={family === 'roots'}
        ariaLabel={`${problem.name}: the curve of f with ${method.spec.name}'s iterate x${sub(step.k)} = ${x}.`}
      />
    );
  }

  const chart = (
    <ConvergenceChart
      series={[
        {
          label: method.spec.name,
          slot: 0,
          values: err.values,
          end,
          count: result.nIter,
        },
      ]}
      t={player.t}
      yLabel={err.label}
      yName={err.yName}
      xInteger
      logY={err.logY}
      onSeek={(k) => player.seek(k)}
      legend={false}
      ariaLabel={`${err.label} against the iteration k for ${method.spec.name} on ${problem.name}.`}
    />
  );

  return (
    <div className={styles.live}>
      <div className={styles.liveFigure} data-kind={kind}>
        {figure ? (
          <>
            <div className={styles.landscape}>{figure}</div>
            <div className={styles.chartSmall}>{chart}</div>
          </>
        ) : (
          <div className={styles.chartBig}>{chart}</div>
        )}
        <div className={styles.liveBar}>
          <PlaybackBar player={player} slots={[0]} />
        </div>
      </div>
      <div className={styles.liveCard}>
        <MethodCard
          method={shown}
          slot={0}
          step={step}
          result={result}
          quantities={quantities}
          call={{
            problem: call.problem,
            options: call.options as never,
            params: call.params as never,
          }}
        />
      </div>
    </div>
  );
}

const SUB = '₀₁₂₃₄₅₆₇₈₉';
function sub(k: number): string {
  return String(k)
    .split('')
    .map((d) => SUB[Number(d)] ?? d)
    .join('');
}

/**
 * The card's live cells for families whose iterates are not points of a landscape: the f(𝐱ₖ) and
 * 𝐱ₖ cells are relabelled (an integral estimate, a tour length, coefficients) or dropped (a
 * permutation has no meaningful norm). Cells whose key no step records are left out. Undefined
 * (the default cells) for optimization families.
 */
function familyQuantities(
  family: string,
  method: RegisteredMethod,
  result: Result,
): MethodCardQuantity[] | undefined {
  const doc = method.doc?.quantities ?? [];
  const has = (key: string) =>
    result.trace.some((s) => {
      const v = key.startsWith('info.')
        ? s.info[key.slice(5)]
        : (s as unknown as Record<string, unknown>)[key];
      return v !== null && v !== undefined;
    });
  const keep = (qs: MethodCardQuantity[]) =>
    qs.filter((q) => q.omit || q.key === 'x' || q.key === 'fun' || has(q.key));
  const id = method.spec.id;
  switch (family) {
    case 'integration':
      return keep([
        { key: 'fun', tex: '\\hat I_k' },
        { key: 'x', tex: '', omit: true },
        { key: 'info.n_panels', tex: 'N' },
        { key: 'stepSize', tex: 'h' },
        { key: 'info.error', tex: '|\\hat I_k - I|' },
        { key: 'info.err_est', tex: '\\hat e_k' },
      ]);
    case 'differentiation':
      return keep([
        { key: 'fun', tex: '', omit: true },
        { key: 'x', tex: '', omit: true },
        { key: 'stepSize', tex: '', omit: true },
        ...doc,
      ]);
    case 'combinatorial':
      return keep([
        id.startsWith('tsp') ? { key: 'fun', tex: 'L(\\pi_k)' } : { key: 'fun', tex: 'z_k' },
        { key: 'x', tex: '', omit: true },
        ...doc,
      ]);
    case 'regression':
      return keep([
        { key: 'fun', tex: 'S(\\hat{\\boldsymbol\\beta}_k)' },
        { key: 'x', tex: '\\hat{\\boldsymbol\\beta}_k' },
        ...doc,
      ]);
    case 'interpolation':
      return keep([
        { key: 'fun', tex: '\\max|p_k - f|' },
        { key: 'x', tex: '', omit: true },
        ...doc,
      ]);
    case 'lp':
      return keep([{ key: 'fun', tex: '\\mathbf{c}^\\top\\mathbf{x}_k' }, ...doc]);
    default:
      return undefined;
  }
}
