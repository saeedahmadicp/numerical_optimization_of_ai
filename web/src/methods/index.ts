/**
 * Importing this module registers every ported method.
 *
 * Ports live in `src/methods/<pkg>/<module>.ts` (mirroring `src/numopt/<pkg>/<module>.py`) and call
 * `registerMethod(...)` at module level. They are picked up automatically by this glob — there is
 * no list to maintain.
 */
import.meta.glob(['./**/*.ts', '!./index.ts', '!./**/*.test.ts'], { eager: true });

export {};
