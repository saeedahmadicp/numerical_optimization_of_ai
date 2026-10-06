/**
 * The parity check of tests/parity.test.ts, as a pure function the method page runs live in the
 * browser: replay a fixture case with the registered TS port and compare it with the Python record
 * (first min(10, n) iterates within 1e-8, then for deterministic methods n_iter exactly, the final
 * x within 1e-6 relative and the same `converged`; for stochastic methods the final objective
 * within 1e-6 relative).
 */
import { reviveNumbers } from '../core/json';
import type { Params, Result } from '../core/types';
import type { ParityCase } from '../app/catalogTypes';

export interface CheckLine {
  name: string;
  ok: boolean;
  /** Largest scaled error (|a − b| / (atol + rtol |b|)); ≤ 1 passes. Null when not numeric. */
  ratio?: number | null;
  /** What was compared, in words ("33 = 33"). */
  detail: string;
}

export interface ParityOutcome {
  ok: boolean;
  checks: CheckLine[];
  result: Result | null;
  error?: string;
  ms: number;
}

export function flat(x: unknown): number[] {
  if (typeof x === 'number') return [x];
  if (Array.isArray(x)) return (x as unknown[]).flatMap(flat);
  return [];
}

/**
 * Worst scaled error between two (nested) numeric values: max |a − b| / (atol + rtol·|b|).
 * Non-finite entries must be identical. Returns Infinity on a shape mismatch.
 */
export function scaledError(a: unknown, b: unknown, rtol: number, atol = rtol): number {
  const xa = flat(a);
  const xb = flat(b);
  if (xa.length !== xb.length) return Infinity;
  let worst = 0;
  for (let i = 0; i < xa.length; i++) {
    const v = xa[i];
    const w = xb[i];
    if (!Number.isFinite(w) || !Number.isFinite(v)) {
      if (!Object.is(v, w) && !(Number.isNaN(v) && Number.isNaN(w))) return Infinity;
      continue;
    }
    worst = Math.max(worst, Math.abs(v - w) / (atol + rtol * Math.abs(w)));
  }
  return worst;
}

/** Compare a TS result with the reduced Python record. */
export function compareCase(
  result: Result,
  want: ParityCase,
  deterministic: boolean,
): { ok: boolean; checks: CheckLine[] } {
  const w = reviveNumbers<ParityCase>(want);
  const n = Math.min(10, w.head.length);
  const checks: CheckLine[] = [];
  let headWorst = 0;
  let lengthOk = result.trace.length >= n;
  for (let k = 0; k < n && lengthOk; k++) {
    const r = scaledError(result.trace[k].x, w.head[k], 1e-8);
    if (!Number.isFinite(r)) lengthOk = false;
    headWorst = Math.max(headWorst, r);
  }
  checks.push({
    name: `first ${n} iterates`,
    ok: lengthOk && headWorst <= 1,
    ratio: lengthOk ? headWorst : null,
    detail: lengthOk ? 'within 10⁻⁸' : 'shape or length differs',
  });
  if (deterministic) {
    checks.push({
      name: 'iterations',
      ok: result.nIter === w.nIter,
      detail: `${result.nIter} = ${w.nIter}`,
    });
    const r = scaledError(result.x, w.x, 1e-6, 1e-10);
    checks.push({
      name: 'final x',
      ok: r <= 1,
      ratio: Number.isFinite(r) ? r : null,
      detail: 'within 10⁻⁶ (relative)',
    });
    checks.push({
      name: 'converged',
      ok: result.converged === w.converged,
      detail: `${result.converged} = ${w.converged}`,
    });
  } else if (typeof w.fun === 'number' && result.fun !== null) {
    const r = scaledError(result.fun, w.fun, 1e-6, 1e-10);
    checks.push({
      name: 'final f',
      ok: r <= 1,
      ratio: Number.isFinite(r) ? r : null,
      detail: 'within 10⁻⁶ (relative)',
    });
  }
  return { ok: checks.every((c) => c.ok), checks };
}

/** Run options of a case, as the harness passes them: spec defaults, then the case's params. */
export function caseParams(defaults: Params, c: ParityCase): Params {
  return { ...defaults, ...(reviveNumbers(c.params) as Params) };
}
