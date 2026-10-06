/**
 * Per-lab URL state. Every lab keeps its view in the hash query so a link reproduces it:
 *
 *   p   problem id                       useProblemState(problems, 'rosenbrock')
 *   m   methods (id ~ slot ~ params)     useMethodSelection(methods, DEFAULT)
 *   x0  start point                      useStartPoint(problem.x0, 2)
 *   …   lab-specific keys                useUrlState('v', '2d', codecs.oneOf([...]))
 *
 * `labQuery` / `labHref` build the same query from a typed state (presets, search, deep links).
 */
import { useCallback, useMemo, type ReactNode } from 'react';
import { clearUrlKeys, codecs, useUrlState, type Codec } from '../../app/useUrlState';
import { replaceQuery, splitHash } from '../../app/router';
import { selectionCodec } from './useLabRuns';
import { suggestLogK } from '../../viz/chartMath';
import type { MethodSelection } from './slots';

export const URL_KEYS = { problem: 'p', methods: 'm', start: 'x0' } as const;

/** A lab view as data: what a preset, a search result or a deep link sets. */
export interface LabState {
  problem?: string;
  methods?: readonly MethodSelection[];
  /** Start point (n-D) or a scalar start. */
  start?: readonly number[] | number;
  /** Lab-specific keys, already formatted (`{ v: '3d' }`). */
  extra?: Readonly<Record<string, string>>;
}

export function labQuery(state: LabState): URLSearchParams {
  const q = new URLSearchParams();
  if (state.problem) q.set(URL_KEYS.problem, state.problem);
  if (state.methods?.length) q.set(URL_KEYS.methods, selectionCodec.format([...state.methods]));
  if (state.start !== undefined)
    q.set(
      URL_KEYS.start,
      typeof state.start === 'number'
        ? String(state.start)
        : codecs.numbers.format([...state.start]),
    );
  for (const [k, v] of Object.entries(state.extra ?? {})) q.set(k, v);
  return q;
}

/** `#/lab/<id>?p=…&m=…` */
export function labHref(labId: string, state: LabState = {}): string {
  const q = labQuery(state).toString();
  return `#/lab/${labId}${q ? `?${q}` : ''}`;
}

const tupleCodecs = new Map<number, Codec<number[]>>();
/** `codecs.tuple(n)` with a stable identity per n (codecs must not change between renders). */
export function tupleCodec(n: number): Codec<number[]> {
  let c = tupleCodecs.get(n);
  if (!c) tupleCodecs.set(n, (c = codecs.tuple(n)));
  return c;
}

/**
 * The selected problem (`?p=`), falling back to `defaultId` and then to the first problem for
 * unknown ids. Changing the problem drops keys that belong to the old one (default: `x0`).
 */
export function useProblemState<P extends { id: string }>(
  problems: readonly P[],
  defaultId?: string,
  resetKeys: readonly string[] = [URL_KEYS.start],
): [P, (id: string) => void] {
  const fallback = defaultId ?? problems[0]?.id ?? '';
  const [id, setId] = useUrlState(URL_KEYS.problem, fallback);
  const problem =
    problems.find((p) => p.id === id) ?? problems.find((p) => p.id === fallback) ?? problems[0];
  const resetKey = resetKeys.join(',');
  const set = useCallback(
    (next: string) => {
      if (resetKey) clearUrlKeys(resetKey.split(','));
      setId(next);
    },
    [setId, resetKey],
  );
  return [problem, set];
}

/**
 * The start point (`?x0=`): exactly `n` finite numbers, else the problem's default. Pass the
 * problem's `x0` (or a fallback) as `defaultX0`.
 */
export function useStartPoint(
  defaultX0: readonly number[] | number | null | undefined,
  n: number,
): [number[], (x: number[]) => void] {
  const key = JSON.stringify(defaultX0 ?? null);
  const def = useMemo(() => {
    const d = JSON.parse(key) as number[] | number | null;
    if (Array.isArray(d) && d.length === n) return d;
    if (typeof d === 'number' && n === 1) return [d];
    return new Array<number>(n).fill(0);
  }, [key, n]);
  return useUrlState<number[]>(URL_KEYS.start, def, tupleCodec(n));
}

// ── Presets ("Try this") ──────────────────────────────────────────────────────────────

export interface LabPreset extends LabState {
  id: string;
  /** What the viewer will see, as a short claim ("Newton converges in 6 steps"). */
  title: string;
  /**
   * One sentence: why it happens (shown under the title). A string may contain inline TeX
   * between dollar signs (`'Each step is orthogonal: $\\nabla f_{k+1}^\\top \\nabla f_k = 0$.'`),
   * typeset with KaTeX; any other ReactNode is rendered as given.
   */
  note?: ReactNode;
  /**
   * Lab URL keys this preset removes (edits the viewer made that the preset must not inherit,
   * e.g. moved cities `c` and a capacity `C`). A lab that clears the same keys for every preset
   * passes them once as LabShell's `presetClearKeys`.
   */
  clearKeys?: readonly string[];
}

/**
 * Keys a preset fully determines: the standard keys, its `extra` keys, and the keys it clears
 * (its own and the lab's `clearKeys`). Other keys are kept, e.g. a lab's view toggle.
 */
function presetKeys(p: LabPreset, clearKeys: readonly string[] = []): string[] {
  return [
    ...new Set([
      URL_KEYS.problem,
      URL_KEYS.methods,
      URL_KEYS.start,
      ...Object.keys(p.extra ?? {}),
      ...(p.clearKeys ?? []),
      ...clearKeys,
    ]),
  ];
}

/**
 * Apply a preset: set its keys, drop the standard keys it does not set and the keys it clears
 * (`p.clearKeys` and the lab-wide `clearKeys`).
 */
export function applyPreset(p: LabPreset, clearKeys: readonly string[] = []): void {
  const cur = new URLSearchParams(splitHash(window.location.hash).query);
  const next = labQuery(p);
  for (const k of presetKeys(p, clearKeys)) cur.delete(k);
  next.forEach((v, k) => cur.set(k, v));
  replaceQuery(cur);
}

/** True when the current URL holds exactly what the preset sets (and none of the keys it clears). */
export function presetActive(
  p: LabPreset,
  hash: string,
  clearKeys: readonly string[] = [],
): boolean {
  const cur = new URLSearchParams(splitHash(hash).query);
  const want = labQuery(p);
  return presetKeys(p, clearKeys).every((k) => (cur.get(k) ?? null) === (want.get(k) ?? null));
}

// ── Iteration axis ────────────────────────────────────────────────────────────────────

export type KAxisMode = 'auto' | 'lin' | 'log';
const KAXIS_CODEC = codecs.oneOf(['auto', 'lin', 'log'] as const);

/**
 * The convergence chart's iteration axis in the URL (`?kx=lin|log`, one key for every lab).
 * "auto" (the default) chooses log k when the runs differ ≥ 30× in length (`suggestLogK`).
 * Returns the resolved axis and a setter for `<KAxisControl logX onChange>`.
 */
export function useKAxis(lengths: readonly number[]): [boolean, (mode: KAxisMode) => void] {
  const [mode, setMode] = useUrlState<KAxisMode>('kx', 'auto', KAXIS_CODEC);
  const logX = mode === 'auto' ? suggestLogK(lengths) : mode === 'log';
  return [logX, setMode];
}
