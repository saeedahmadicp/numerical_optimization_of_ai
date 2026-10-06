/**
 * Loader for `numopt export` JSON.
 *
 * Python's `to_jsonable` writes NaN as `null` and ±∞ as the strings `"inf"` / `"-inf"`. These
 * helpers map them back, and convert snake_case payloads into the camelCase types of `types.ts`.
 */
import type {
  Family,
  LinearProgram,
  LinearSystem,
  MethodSpec,
  ParamSpec,
  ParamValue,
  ProblemMeta,
  Result,
  Step,
} from './types';

type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

/** Recursively map `"inf"` → Infinity, `"-inf"` → -Infinity. `null` stays `null`. */
export function reviveNumbers<T = unknown>(value: unknown): T {
  if (value === 'inf') return Infinity as T;
  if (value === '-inf') return -Infinity as T;
  if (Array.isArray(value)) return value.map((v) => reviveNumbers(v)) as T;
  if (value !== null && typeof value === 'object') {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value)) out[k] = reviveNumbers(v);
    return out as T;
  }
  return value as T;
}

/** A numeric field. `null` (Python NaN or None) stays `null`; the strings map to ±Infinity. */
export function num(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  if (value === 'inf') return Infinity;
  if (value === '-inf') return -Infinity;
  return typeof value === 'number' ? value : Number(value);
}

/** Parse JSON text and revive infinities. */
export function parseJson<T = unknown>(text: string): T {
  return reviveNumbers<T>(JSON.parse(text) as Json);
}

/** snake_case → camelCase (`grad_norm` → `gradNorm`, `A_ub` → `aUb`). */
export function camel(key: string): string {
  const parts = key.split('_').filter(Boolean);
  if (parts.length === 0) return key;
  const [first, ...rest] = parts;
  const head = /^[A-Z]$/.test(first) && rest.length > 0 ? first.toLowerCase() : first;
  return head + rest.map((p) => p[0].toUpperCase() + p.slice(1)).join('');
}

type Raw = Record<string, unknown>;

export function stepFromJson(raw: Raw): Step {
  return {
    k: raw.k as number,
    x: reviveNumbers(raw.x),
    fun: num(raw.fun),
    gradNorm: num(raw.grad_norm),
    stepSize: num(raw.step_size),
    // `info` keys stay as exported (they are documented per module in snake_case).
    info: reviveNumbers((raw.info as Raw | undefined) ?? {}),
  };
}

export function resultFromJson(raw: Raw): Result {
  return {
    method: raw.method as string,
    x: reviveNumbers(raw.x),
    fun: num(raw.fun),
    converged: Boolean(raw.converged),
    message: (raw.message as string) ?? '',
    nIter: raw.n_iter as number,
    nFev: (raw.n_fev as number) ?? 0,
    nGev: (raw.n_gev as number) ?? 0,
    nHev: (raw.n_hev as number) ?? 0,
    trace: ((raw.trace as Raw[] | undefined) ?? []).map(stepFromJson),
    extra: reviveNumbers((raw.extra as Raw | undefined) ?? {}),
  };
}

export function paramSpecFromJson(raw: Raw): ParamSpec {
  return {
    name: raw.name as string,
    default: reviveNumbers<ParamValue>(raw.default),
    kind: (raw.kind as ParamSpec['kind']) ?? 'float',
    min: num(raw.min),
    max: num(raw.max),
    choices: (raw.choices as string[]) ?? [],
    log: Boolean(raw.log),
    help: (raw.help as string) ?? '',
  };
}

export function methodSpecFromJson(raw: Raw): MethodSpec {
  return {
    id: raw.id as string,
    family: raw.family as Family,
    name: raw.name as string,
    params: ((raw.params as Raw[]) ?? []).map(paramSpecFromJson),
    needs: (raw.needs as string[]) ?? [],
    order: (raw.order as string) ?? '',
    summary: (raw.summary as string) ?? '',
    references: (raw.references as string[]) ?? [],
    deterministic: raw.deterministic !== false,
    tags: (raw.tags as string[]) ?? [],
  };
}

export function problemMetaFromJson(raw: Raw): ProblemMeta {
  const r = reviveNumbers<Raw>(raw);
  return {
    kind: r.kind as ProblemMeta['kind'],
    id: r.id as string,
    name: r.name as string,
    latex: (r.latex as string) ?? '',
    dim: (r.dim as number) ?? 1,
    domain: (r.domain as unknown[]) ?? [],
    x0: (r.x0 as ProblemMeta['x0']) ?? null,
    bracket: (r.bracket as [number, number] | null) ?? null,
    minima: (r.minima as ProblemMeta['minima']) ?? [],
    roots: (r.roots as ProblemMeta['roots']) ?? [],
    constraints: (r.constraints as ProblemMeta['constraints']) ?? [],
    exact: num(r.exact),
    description: (r.description as string) ?? '',
    tags: (r.tags as string[]) ?? [],
  };
}

export function linearProgramFromJson(raw: Raw): LinearProgram {
  const r = reviveNumbers<Raw>(raw);
  return {
    id: r.id as string,
    name: r.name as string,
    c: r.c as number[],
    aUb: (r.A_ub as number[][] | null) ?? null,
    bUb: (r.b_ub as number[] | null) ?? null,
    aEq: (r.A_eq as number[][] | null) ?? null,
    bEq: (r.b_eq as number[] | null) ?? null,
    sense: (r.sense as 'min' | 'max') ?? 'min',
    integer: (r.integer as boolean[]) ?? [],
    optimum: (r.optimum as number[] | null) ?? null,
    optimalValue: num(r.optimal_value),
    description: (r.description as string) ?? '',
    domain: (r.domain as unknown[]) ?? [],
  };
}

export function linearSystemFromJson(raw: Raw): LinearSystem {
  const r = reviveNumbers<Raw>(raw);
  return {
    id: r.id as string,
    name: r.name as string,
    A: r.A as number[][],
    b: r.b as number[],
    solution: (r.solution as number[] | null) ?? null,
    description: (r.description as string) ?? '',
    tags: (r.tags as string[]) ?? [],
  };
}

/** One parity case from `fixtures/<family>.json`. */
export interface FixtureCase {
  method: string;
  problem: string;
  params: Record<string, unknown>;
  result: Result;
}

export function fixtureCaseFromJson(raw: Raw): FixtureCase {
  return {
    method: raw.method as string,
    problem: raw.problem as string,
    params: reviveNumbers((raw.params as Raw) ?? {}),
    result: resultFromJson(raw.result as Raw),
  };
}
