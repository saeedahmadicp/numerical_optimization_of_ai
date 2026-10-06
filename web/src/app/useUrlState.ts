/**
 * Shareable view state in the hash query: `#/lab/unconstrained?p=rosenbrock&x0=-1.2,1`.
 *
 *   const [problem, setProblem] = useUrlState('p', 'rosenbrock');
 *   const [x0, setX0] = useUrlState('x0', [-1.2, 1], codecs.numbers);
 *
 * Values equal to the default are removed from the URL, so links stay short.
 */
import { useCallback, useMemo } from 'react';
import { replaceQuery, splitHash, useHash } from './router';

export interface Codec<T> {
  parse(s: string): T | undefined;
  format(v: T): string;
}

export const codecs = {
  string: { parse: (s: string) => s, format: (v: string) => v } as Codec<string>,
  number: {
    parse: (s: string) => (s.trim() !== '' && Number.isFinite(Number(s)) ? Number(s) : undefined),
    format: (v: number) => String(v),
  } as Codec<number>,
  bool: {
    parse: (s: string) => s === '1',
    format: (v: boolean) => (v ? '1' : '0'),
  } as Codec<boolean>,
  numbers: {
    parse: (s: string) => {
      if (s.trim() === '') return undefined;
      const xs = s.split(',').map(Number);
      return xs.length && xs.every(Number.isFinite) ? xs : undefined;
    },
    format: (v: number[]) => v.map((x) => String(Number(x.toPrecision(6)))).join(','),
  } as Codec<number[]>,
  /** Exactly `n` finite numbers (a start point `x0=-1.2,1`); anything else falls back to the default. */
  tuple(n: number): Codec<number[]> {
    return {
      parse: (s) => {
        const xs = codecs.numbers.parse(s);
        return xs && xs.length === n ? xs : undefined;
      },
      format: codecs.numbers.format,
    };
  },
  /** One of a fixed set of strings (`?v=3d`); unknown values fall back to the default. */
  oneOf<T extends string>(values: readonly T[]): Codec<T> {
    return {
      parse: (s) => ((values as readonly string[]).includes(s) ? (s as T) : undefined),
      format: (v) => v,
    };
  },
  strings: {
    parse: (s: string) => (s ? s.split(',').filter(Boolean) : []),
    format: (v: string[]) => v.join(','),
  } as Codec<string[]>,
  json<T>(): Codec<T> {
    return {
      parse: (s) => {
        try {
          return JSON.parse(s) as T;
        } catch {
          return undefined;
        }
      },
      format: (v) => JSON.stringify(v),
    };
  },
};

export function readQuery(): URLSearchParams {
  return new URLSearchParams(splitHash(window.location.hash).query);
}

export function useUrlState<T>(
  key: string,
  defaultValue: T,
  codec: Codec<T> = codecs.string as unknown as Codec<T>,
): [T, (v: T | ((prev: T) => T)) => void] {
  const hash = useHash();
  const raw = useMemo(() => new URLSearchParams(splitHash(hash).query).get(key), [hash, key]);
  const defaultKey = JSON.stringify(defaultValue);
  const value = useMemo(() => {
    if (raw === null) return defaultValue;
    return codec.parse(raw) ?? defaultValue;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [raw, defaultKey, codec]);

  const set = useCallback(
    (next: T | ((prev: T) => T)) => {
      const params = readQuery();
      const cur = params.get(key);
      const prev = cur === null ? defaultValue : (codec.parse(cur) ?? defaultValue);
      const v = typeof next === 'function' ? (next as (p: T) => T)(prev) : next;
      if (JSON.stringify(v) === defaultKey) params.delete(key);
      else params.set(key, codec.format(v));
      replaceQuery(params);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [key, defaultKey, codec],
  );
  return [value, set];
}

/** Remove several keys at once (e.g. when the problem changes, drop the old start point). */
export function clearUrlKeys(keys: string[]): void {
  const params = readQuery();
  keys.forEach((k) => params.delete(k));
  replaceQuery(params);
}
