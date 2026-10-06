/**
 * Deep comparison of TS values against Python reference data (test helper).
 *
 * Python NaN is exported as JSON null, so a TS NaN matches a reference `null`. Numbers match when
 * |got − want| ≤ atol + rtol·|want|; ±Infinity must match exactly. The first mismatch is reported
 * with its path (e.g. `trace[3].info.conditions.armijo`).
 */
import { readFileSync } from 'node:fs';
import { reviveNumbers } from '../../src/core/json';

export interface Tol {
  rtol: number;
  atol: number;
}

export const TIGHT: Tol = { rtol: 1e-12, atol: 1e-300 };

/** The first mismatch between `got` and `want`, or null when they agree. */
export function mismatch(got: unknown, want: unknown, tol: Tol = TIGHT, path = '$'): string | null {
  if (want === null || want === undefined) {
    const ok = got === null || got === undefined || (typeof got === 'number' && Number.isNaN(got));
    return ok ? null : `${path}: got ${String(got)}, want null`;
  }
  if (typeof want === 'number') {
    if (typeof got !== 'number') return `${path}: got ${JSON.stringify(got)}, want ${want}`;
    if (!Number.isFinite(want) || !Number.isFinite(got))
      return Object.is(got, want) || got === want ? null : `${path}: got ${got}, want ${want}`;
    return Math.abs(got - want) <= tol.atol + tol.rtol * Math.abs(want)
      ? null
      : `${path}: got ${got}, want ${want} (diff ${Math.abs(got - want)})`;
  }
  if (Array.isArray(want)) {
    if (!Array.isArray(got)) return `${path}: got ${JSON.stringify(got)}, want an array`;
    if (got.length !== want.length) return `${path}: length ${got.length}, want ${want.length}`;
    for (let i = 0; i < want.length; i++) {
      const m = mismatch(got[i], want[i], tol, `${path}[${i}]`);
      if (m) return m;
    }
    return null;
  }
  if (typeof want === 'object') {
    if (got === null || typeof got !== 'object' || Array.isArray(got))
      return `${path}: got ${JSON.stringify(got)}, want an object`;
    const gk = Object.keys(got).sort(),
      wk = Object.keys(want).sort();
    if (gk.join() !== wk.join()) return `${path}: keys [${gk.join(', ')}], want [${wk.join(', ')}]`;
    for (const k of wk) {
      const m = mismatch(
        (got as Record<string, unknown>)[k],
        (want as Record<string, unknown>)[k],
        tol,
        `${path}.${k}`,
      );
      if (m) return m;
    }
    return null;
  }
  return got === want ? null : `${path}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`;
}

/** Read a JSON file relative to this folder with "inf"/"-inf" revived. */
export function readJson<T>(relative: string): T {
  return reviveNumbers<T>(
    JSON.parse(readFileSync(new URL(relative, import.meta.url), 'utf8')) as unknown,
  );
}

/** Largest relative difference |a − b| / max(|b|, floor) over two flat number lists. */
export function maxRelDiff(a: readonly number[], b: readonly number[], floor = 1): number {
  let m = 0;
  a.forEach((v, i) => {
    const w = b[i];
    if (!Number.isFinite(v) || !Number.isFinite(w)) {
      if (!Object.is(v, w) && !(Number.isNaN(v) && Number.isNaN(w))) m = Infinity;
      return;
    }
    m = Math.max(m, Math.abs(v - w) / Math.max(Math.abs(w), floor));
  });
  return m;
}

export function flat(x: unknown): number[] {
  if (typeof x === 'number') return [x];
  if (x === null) return [NaN];
  if (Array.isArray(x)) return (x as unknown[]).flatMap(flat);
  return [];
}
