/**
 * Loads the TS ports a method page needs, on demand: the method modules of one family's folder
 * (`src/methods/<pkg>/*.ts`) and the problem library. Each folder is its own chunk, so a method
 * page never downloads the other 15 families.
 */
import type { Family } from '../core/types';

const METHOD_MODULES = import.meta.glob(['../methods/*/*.ts', '!../methods/**/*.test.ts']);

/** Python package folder of each family (src/numopt/<pkg>/). */
const FOLDER: Partial<Record<Family, string>> = {
  systems: 'roots',
  global: 'unconstrained',
  least_squares: 'unconstrained',
};

export const folderOf = (family: string): string => FOLDER[family as Family] ?? family;

/** The Python source file of a family, relative to the repository root. */
export function pythonSource(family: string): string {
  const special: Record<string, string> = {
    systems: 'roots/systems.py',
    global: 'unconstrained/global_.py',
    least_squares: 'unconstrained/least_squares.py',
  };
  return `src/numopt/${special[family] ?? `${family}/`}`;
}

const loaded = new Map<string, Promise<void>>();

/** Register every port of the family's folder and every problem (idempotent). */
export function loadFamily(family: string): Promise<void> {
  const folder = folderOf(family);
  let p = loaded.get(folder);
  if (!p) {
    const mods = Object.entries(METHOD_MODULES)
      .filter(([path]) => path.split('/')[2] === folder)
      .map(([, load]) => load());
    p = Promise.all([import('../problems'), ...mods]).then(() => undefined);
    loaded.set(folder, p);
  }
  return p;
}
