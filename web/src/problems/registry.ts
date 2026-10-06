/**
 * TS mirror of `numopt.problems.registry`: problems keyed by id, grouped by kind.
 * Ports live in `src/problems/<kind>.ts` and call `addProblem(kind, problem)` at module level.
 */
import type { ProblemKind } from '../core/types';

export interface HasId {
  id: string;
}

const PROBLEMS = new Map<string, { kind: ProblemKind; problem: HasId }>();

export function addProblem<T extends HasId>(kind: ProblemKind, problem: T): T {
  const existing = PROBLEMS.get(problem.id);
  if (existing && existing.problem !== problem)
    throw new Error(`problem id ${problem.id} is already registered`);
  PROBLEMS.set(problem.id, { kind, problem });
  return problem;
}

export function getProblem<T = HasId>(id: string): T {
  const p = PROBLEMS.get(id);
  if (!p) throw new Error(`unknown problem ${id}`);
  return p.problem as T;
}

export function hasProblem(id: string): boolean {
  return PROBLEMS.has(id);
}

export function kindOf(id: string): ProblemKind | undefined {
  return PROBLEMS.get(id)?.kind;
}

export function listProblems<T = HasId>(kind?: ProblemKind): T[] {
  return [...PROBLEMS.values()]
    .filter((p) => kind === undefined || p.kind === kind)
    .map((p) => p.problem as T);
}
