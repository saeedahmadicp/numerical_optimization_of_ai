/**
 * The runs of the combinatorial lab, computed in a deferred render, together with the problem
 * they were computed on. While a new run is pending, `runs` still belong to the previous problem
 * (possibly of the other kind, or with another number of cities), so the stage must draw them on
 * `problem` from this hook, never on the problem of the URL.
 */
import { useDeferredValue, useMemo } from 'react';
import { defaults, getMethod, hasMethod, type RegisteredMethod } from '../../core/registry';
import type { Result, RunOptions } from '../../core/types';
import type { LabRun, MethodSelection } from '../_shell';

function runAll<P>(
  problem: P,
  selections: readonly MethodSelection[],
  options: RunOptions,
): LabRun[] {
  return selections
    .filter((s) => hasMethod(s.id))
    .map((sel) => {
      const method = getMethod<P>(sel.id) as unknown as RegisteredMethod;
      try {
        const result = method.fn(problem, { ...defaults(method.spec), ...sel.params, ...options });
        return { sel, method, result };
      } catch (e) {
        const message = e instanceof Error ? e.message : String(e);
        const result: Result = {
          method: sel.id,
          x: null,
          fun: null,
          converged: false,
          message,
          nIter: 0,
          nFev: 0,
          nGev: 0,
          nHev: 0,
          trace: [],
          extra: {},
        };
        return { sel, method, result, error: message };
      }
    });
}

export function useCombinatorialRuns<P extends { id: string }>(
  problem: P,
  selections: readonly MethodSelection[],
  options: RunOptions,
): { runs: LabRun[]; problem: P; pending: boolean } {
  const key = JSON.stringify([problem.id, selections, options]);
  // One stable input object per key; React computes the new one in the background.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const input = useMemo(() => ({ problem, selections, options }), [key]);
  const used = useDeferredValue(input);
  const runs = useMemo(() => runAll(used.problem, used.selections, used.options), [used]);
  return { runs, problem: used.problem, pending: used !== input };
}
