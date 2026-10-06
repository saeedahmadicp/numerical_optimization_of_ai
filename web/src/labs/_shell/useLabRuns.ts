import { useEffect, useMemo, useState } from 'react';
import { defaults, getMethod, hasMethod, type RegisteredMethod } from '../../core/registry';
import type { Result, RunOptions } from '../../core/types';
import { useUrlState, type Codec } from '../../app/useUrlState';
import { useToast } from '../../ui/components/Toast';
import { sanitizeSelection, type MethodSelection } from './slots';

export interface LabRun {
  sel: MethodSelection;
  method: RegisteredMethod;
  result: Result;
  /** Set when the method threw (invalid input); `result` is then an empty failed run. */
  error?: string;
}

export interface LabRunsOptions {
  /**
   * Compute new runs in a task after the render that asked for them: while a slow family
   * re-runs, the previous runs stay on screen (`pending` is true), and a burst of changes (a
   * dragged slider) costs one run. Use it for heavy methods. (Unlike a deferred React render,
   * the task is not restarted by the player's per-frame renders, so it always finishes.)
   */
  defer?: boolean;
}

function runAll<P>(
  problem: P | undefined,
  selections: readonly MethodSelection[],
  options: RunOptions,
): LabRun[] {
  if (!problem) return [];
  return selections
    .filter((s) => hasMethod(s.id))
    .map((sel) => {
      const method = getMethod<P>(sel.id) as unknown as RegisteredMethod;
      try {
        const result = method.fn(problem, {
          ...defaults(method.spec),
          ...sel.params,
          ...options,
        });
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

/**
 * Run every selected method on the problem (memoized on problem id, selections and options).
 * Methods are synchronous; they are fast for the didactic sizes the labs use. Pass
 * `{ defer: true }` for heavy families.
 */
export function useLabRuns<P extends { id: string }>(
  problem: P | undefined,
  selections: readonly MethodSelection[],
  options: RunOptions = {},
  opts: LabRunsOptions = {},
): LabRun[] {
  return useLabRunsState(problem, selections, options, opts).runs;
}

/**
 * Like `useLabRuns`, plus `pending`: true while a deferred run is being computed (the previous
 * runs stay on screen). Pass `pending` to `<LabShell pending>`, which dims the stage and says
 * "Recomputing…" instead of blanking the canvas.
 */
export function useLabRunsState<P extends { id: string }>(
  problem: P | undefined,
  selections: readonly MethodSelection[],
  options: RunOptions = {},
  { defer = false }: LabRunsOptions = {},
): { runs: LabRun[]; pending: boolean } {
  const key = JSON.stringify([problem?.id, selections, options]);
  // One stable input object per key.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const input = useMemo(() => ({ problem, selections, options }), [key]);
  const sync = useMemo(
    () => (defer ? null : runAll(input.problem, input.selections, input.options)),
    [input, defer],
  );
  // `defer`: the first runs are computed at once (the stage is never blank); later ones in a
  // macrotask, so per-frame renders of a playing player cannot starve them.
  const [done, setDone] = useState<{ input: typeof input; runs: LabRun[] } | null>(() =>
    defer ? { input, runs: runAll(input.problem, input.selections, input.options) } : null,
  );
  useEffect(() => {
    if (!defer || done?.input === input) return;
    const id = window.setTimeout(
      () => setDone({ input, runs: runAll(input.problem, input.selections, input.options) }),
      0,
    );
    return () => window.clearTimeout(id);
  }, [defer, input, done]);
  if (sync) {
    // Forget the deferred runs while running synchronously: when `defer` switches on again,
    // the stale runs of an earlier problem must not be shown.
    if (done !== null) setDone(null);
    return { runs: sync, pending: false };
  }
  if (done === null) {
    // `defer` switched on after mount (e.g. the lab moved to a larger problem): compute the
    // first deferred run at once, as on mount, so the stage is never blank. (Returning a fresh
    // `[]` here would hand the player new traces on every render: an infinite render loop.)
    const first = { input, runs: runAll(input.problem, input.selections, input.options) };
    setDone(first);
    return { runs: first.runs, pending: false };
  }
  return { runs: done.runs, pending: done.input !== input };
}

/**
 * URL codec for method selections: `gradient_descent~0~alpha=0.002,heavy_ball~1`
 * (id ~ color slot ~ non-default params). Parsing is lenient; `sanitizeSelection` (used by
 * `useMethodSelection`) validates the result against the registry.
 */
export const selectionCodec: Codec<MethodSelection[]> = {
  parse(s) {
    const out: MethodSelection[] = [];
    for (const part of s.split(',')) {
      const [id, slot, ...kv] = part.split('~');
      if (!id || !/^[a-z0-9_]+$/.test(id)) continue;
      const params: MethodSelection['params'] = {};
      for (const pair of kv) {
        const [k, v] = pair.split('=');
        if (!k || v === undefined) continue;
        params[k] =
          v === 'true'
            ? true
            : v === 'false'
              ? false
              : v.trim() !== '' && Number.isFinite(Number(v))
                ? Number(v)
                : decodeURIComponent(v);
      }
      const n = Number(slot);
      out.push({ id, slot: Number.isInteger(n) ? n : -1, params });
    }
    return out.length ? out : undefined;
  },
  format(v) {
    return v
      .map((m) =>
        [
          m.id,
          m.slot,
          ...Object.entries(m.params).map(
            ([k, x]) => `${k}=${typeof x === 'string' ? encodeURIComponent(x) : String(x)}`,
          ),
        ].join('~'),
      )
      .join(',');
  },
};

/**
 * The lab's method selection, stored in the URL (`?m=`) and validated against `available`:
 * unknown ids are dropped, and so are duplicate ids unless `allowDuplicates` (the same method
 * twice with different params, e.g. damped and undamped Newton, keyed by color slot), colliding color slots are reassigned and params are
 * coerced to their specs (see `sanitizeSelection`). When a shared link names methods this lab
 * does not have, a toast says so once. Returns `[selection, setSelection]`.
 */
export function useMethodSelection(
  available: readonly RegisteredMethod[],
  defaultValue: MethodSelection[],
  key = 'm',
  opts: { allowDuplicates?: boolean } = {},
): [
  MethodSelection[],
  (v: MethodSelection[] | ((prev: MethodSelection[]) => MethodSelection[])) => void,
] {
  const [raw, setRaw] = useUrlState<MethodSelection[]>(key, defaultValue, selectionCodec);
  const { value, dropped } = useMemo(() => {
    const r = sanitizeSelection(raw, available, opts);
    return r.value.length
      ? r
      : { value: sanitizeSelection(defaultValue, available, opts).value, dropped: r.dropped };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [raw, available]);
  const toast = useToast();
  const droppedKey = dropped.join(', ');
  useEffect(() => {
    if (droppedKey)
      toast(
        `This link asked for ${droppedKey}, which this lab does not have. It shows the other methods.`,
        'info',
      );
  }, [droppedKey, toast]);
  return [value, setRaw];
}

/** LabRuns as the `focus.runs` of LabShell (`{ value: id, slot, name }`). */
export function focusRuns(
  runs: readonly LabRun[],
): { value: string; slot: number; name: string }[] {
  return runs.map((r) => ({ value: r.sel.id, slot: r.sel.slot, name: r.method.spec.name }));
}
