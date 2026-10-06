/** The Python call that reproduces a combinatorial run (library instance or edited data). */
import type { Params } from '../../core/types';
import type { LabRun } from '../_shell';
import { pythonCall, pyFloat } from '../_shell';
import type { CombinatorialInstance } from '../../problems/combinatorial';
import { baseId } from './model';

/** The Python call that reproduces the run (library instance, or the edited data inline). */
export function pythonCallFor(
  id: string,
  problem: CombinatorialInstance,
  params: Params,
  seed: number | undefined,
  specs: LabRun['method']['spec']['params'],
): string {
  const lib = baseId(problem.id) === problem.id;
  if (lib) return pythonCall(id, { problem: problem.id, params, seed, specs });
  const base = pythonCall(id, { params, seed, specs });
  const data =
    problem.kind === 'tsp'
      ? `[${problem.coords.map(([x, y]) => `[${pyFloat(x)}, ${pyFloat(y)}]`).join(', ')}]`
      : `KnapsackInstance("custom", "custom", values=(${problem.values.join(', ')},), ` +
        `weights=(${problem.weights.join(', ')},), capacity=${problem.capacity})`;
  const call = base.replace(/, f(?=[,)])/, `, ${data}`);
  return problem.kind === 'knapsack'
    ? `from numopt.problems.combinatorial import KnapsackInstance\n${call}`
    : call;
}
