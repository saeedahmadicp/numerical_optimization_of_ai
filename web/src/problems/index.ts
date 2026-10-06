/** Importing this module registers every ported problem (`src/problems/<kind>.ts`). */
import.meta.glob(['./*.ts', '!./index.ts', '!./registry.ts'], { eager: true });

export { addProblem, getProblem, hasProblem, kindOf, listProblems } from './registry';
